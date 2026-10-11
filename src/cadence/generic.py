"""A generic brain for simple tasks: senses, cortex, a motor choice, dopamine and memory.

``Brain`` composes the standard regions into one connectome that settles as a whole:

* a sensory region: a blank population for a vector observation, or a ``visual_cortex`` for
  an image;
* an association ``cortex`` that every sensory region projects to;
* a ``motor_cortex`` with one neuron per action, connected to the association cortex by
  reciprocal synapse pairs so that a nudge on the choice reaches the cortex;
* a basal-ganglia analogue, an external ``ActorCritic`` whose linear critic reads the
  association cortex and whose
  dopamine prediction error moves every plastic synapse through its eligibility trace;
* optionally a ``prefrontal_cortex`` driven by a ``Trace`` of the association cortex, a
  working memory of the moments before;
* a default hippocampal-memory analogue, ``SynapticMemory`` from sensory to motor
  neurons, recording outcomes quickly and consolidating them through repetition/salience.

``step`` runs one ongoing perceive/feedback/act loop; demonstrations and rewards are
signals in that loop, without a train/eval mode. The lower-level ``fit``, ``act``
and ``learn`` operations expose individual mechanisms for controlled experiments.
The learned equilibrium is its world model in operation; this composition does
not integrate a learned environmental transition predictor, hierarchical goals or language.
Its connectome comes from a ``Genome``, so ``evolve`` can select its region sizes/densities, and a
designed region can replace any of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._configuration import arousal_config, configured, plain, real, split_genes
from .arousal import Arousal, ArousalConfig
from .brain import Backend, BrainState, Equilibrium
from .brain import Brain as NeuralGraph
from .connectome import Connectome
from .genome import Genome, Projection, develop
from .learning import Learner, LearnerConfig, learning_neuron_model
from .memory import SynapticMemory
from .plasticity import ActorCritic, ActorCriticConfig
from .regions import Region, motor_cortex, prefrontal_cortex, slot_sizes, visual_cortex
from .stream import Efference, FastSynapses, PatternSeparator, Trace

__all__ = ["Brain"]

_REWARD_ARRAYS = (
    "w_critic",
    "trace",
    "trace_bias",
    "trace_critic",
    "second_moment",
    "second_moment_bias",
    "velocity",
    "velocity_bias",
    "salience",
    "_drive",
)


# The default motor lateral inhibition, and the widest readout that keeps it.
# Measured on composed brains (issue 124): up to 8 actions, -0.5 settles in the
# same 32 sweeps as 0.0; from 12 actions the free solve needs damping and about
# nine times the sweeps, and the undamped teaching free phase does not settle.
_LATERAL_DEFAULT = -0.5
_LATERAL_ACTION_LIMIT = 8


def _default_lateral(actions: int) -> float:
    return _LATERAL_DEFAULT if actions <= _LATERAL_ACTION_LIMIT else 0.0


def _learning() -> LearnerConfig:
    # Retain the recovered finite teaching law; numerical damping belongs to
    # qualified free settlement, where it does not change these learning phases.
    return LearnerConfig(
        beta=0.1,
        eta=0.5,
        temperature=0.2,
        tolerance=3e-3,
        free_steps=1024,
        nudged_steps=12,
        momentum=0.9,
    )


def _reward() -> ActorCriticConfig:
    # The composed actor keeps its measured bias rate, 0.05 at eta 1.0: the release matrices
    # and the application evidence were taken with it. A bare ActorCriticConfig derives
    # eta / 10 (issue 143).
    return ActorCriticConfig(
        gamma=0.9, lam=0.8, eta=1.0, eta_bias=0.05, eta_critic=0.3, eligibility_steps=12
    )


def _validate_resting_bias(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("resting_bias must be a finite nonnegative real scalar")
    try:
        bias = float(value)
    except OverflowError as exc:
        raise ValueError("resting_bias must be a finite nonnegative real scalar") from exc
    if not np.isfinite(bias) or bias < 0:
        raise ValueError("resting_bias must be a finite nonnegative real scalar")
    return bias


def _validate_efference_amplitude(value: Any) -> float:
    """The read gain of the efference copy: a finite nonnegative real scalar; zero is the
    released brain without one."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("efference_amplitude must be a finite nonnegative real scalar")
    return float(value)


def _region_widths(populations: Mapping[str, Any], prefix: str) -> list[int]:
    """Numbered regions in topology order, independent of serialized mapping order."""
    names = [name for name in populations
             if name.startswith(prefix) and name[len(prefix):].isdigit()]
    return [len(populations[name]) for name in sorted(names, key=lambda n: int(n[len(prefix):]))]


def _load_composition(value: Any, learner: Learner) -> dict[str, Any] | None:
    """Validate optional construction provenance without rebuilding learned wiring."""
    if value is None:
        return None
    expected = {"inputs", "actions", "modules", "observers", "slots", "lateral",
                "sensory_scale", "seed"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid saved composition provenance")
    for name in ("inputs", "actions", "seed"):
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, int) or number < (name != "seed"):
            raise ValueError(f"invalid saved composition {name}")
    for name in ("modules", "observers", "slots"):
        numbers = value[name]
        if not isinstance(numbers, list) or any(
            isinstance(number, bool) or not isinstance(number, int) or number < 1
            for number in numbers
        ):
            raise ValueError(f"invalid saved composition {name}")
        if not numbers and name != "observers":
            raise ValueError(f"invalid saved composition {name}")
    real("saved sensory_scale", value["sensory_scale"], low=0)
    real("saved lateral", value["lateral"])
    populations = learner.brain.connectome.populations
    modules = _region_widths(populations, "module_") + [len(populations.get("association", ()))]
    observers = _region_widths(populations, "observer_")
    if (
        value["inputs"] != len(populations.get("sensory", ()))
        or value["actions"] != len(learner.output_index)
        or value["modules"] != modules
        or value["observers"] != observers
        or value["slots"] != list(learner.slot_sizes)
    ):
        raise ValueError("saved composition provenance disagrees with the actual layout")
    return dict(plain(value))


def _load_memory(metadata: Any, data: Mapping[str, Any], neurons: int) -> FastSynapses | None:
    """Validate and restore a detached memory before constructing the resumed composition."""
    prefix = "episodic/"

    def array(name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        key = prefix + name
        if key not in data:
            raise ValueError(f"missing saved memory state: {name}")
        value: np.ndarray = data[key]
        if (
            value.dtype.kind not in "iuf"
            or not np.isfinite(value).all()
            or (shape is not None and value.shape != shape)
        ):
            raise ValueError(f"invalid saved memory state: {name}")
        return value.copy()

    if metadata is None:
        if any(key.startswith(prefix) for key in data):
            raise ValueError("saved memory arrays require hippocampus metadata")
        return None
    if not isinstance(metadata, dict):
        raise ValueError("invalid hippocampus metadata")
    try:
        kind = metadata.get("kind")
        if kind not in (None, "consolidating"):
            raise ValueError("unsupported saved memory kind")
        for name in ("normalize", "replace"):
            if not isinstance(metadata[name], bool):
                raise ValueError(f"saved memory {name} must be boolean")
        for name in ("pre", "post", "writes"):
            value = metadata[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"saved memory {name} must be a nonnegative integer")
        pre, post = array("pre", (metadata["pre"],)), array("post", (metadata["post"],))
        if np.any(pre >= neurons) or np.any(post >= neurons):
            raise ValueError("saved memory indices exceed the connectome")
        separator = None
        spec = metadata.get("separator")  # Older unseparated checkpoints omitted this field.
        if spec is None:
            if any(key.startswith(prefix + "separator/") for key in data):
                raise ValueError("saved separator arrays require separator metadata")
        else:
            if not isinstance(spec, dict):
                raise ValueError("invalid separator metadata")
            if set(spec) != {"inputs", "expansion", "winners", "seed", "center"}:
                raise ValueError("incomplete or unsupported separator metadata")
            for name in ("inputs", "expansion", "winners", "seed"):
                value = spec[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"saved separator {name} must be a nonnegative integer")
            if spec["inputs"] != len(pre):
                raise ValueError("saved separator inputs must match memory pre neurons")
            # Do not regenerate these arrays from the seed: projections can be customized,
            # and the running center of FastSynapses can have changed after observations.
            projection = array("separator/projection", (spec["inputs"], spec["expansion"]))
            mean = array("separator/mean", (spec["inputs"],))
            separator = PatternSeparator(**spec)
            separator.projection, separator.mean = projection, mean
        options = {
            name: metadata[name]
            for name in ("decay", "rate", "amplitude", "normalize", "replace", "rule", "writes")
        }
        memory: FastSynapses
        if kind == "consolidating":
            memory = SynapticMemory(
                pre, post, separator=separator, consolidation=metadata["consolidation"], **options
            )
        else:
            if prefix + "consolidated" in data:
                raise ValueError("consolidated weights require consolidating memory metadata")
            memory = FastSynapses(pre, post, separator=separator, **options)
        strength = array("strength")
        if strength.ndim != 3 or strength.shape[1:] != (memory.key_width, len(post)):
            raise ValueError("invalid saved memory strength dimensions")
        mass = array("mass", (len(strength),))
        if np.any(mass < 0):
            raise ValueError("saved memory mass must be nonnegative")
        if isinstance(memory, SynapticMemory):
            memory.consolidated = array("consolidated", (memory.key_width, len(post)))
            if (
                metadata.get("persistent_parameters", memory.consolidated.size)
                != memory.consolidated.size
            ):
                raise ValueError("inconsistent saved persistent parameter count")
        memory.strength, memory.mass = strength, mass
        return memory
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid or incomplete saved memory metadata") from exc


def _validate_life_state(meta: dict[str, Any], data: Mapping[str, Any], learner: Learner) -> None:
    """Reject corrupt continuation state before exposing a resumed agent."""
    n = learner.brain.connectome.n
    populations = learner.brain.connectome.populations
    association = len(populations["association"])
    sensory = len(populations.get("sensory", populations.get("visual/input", ())))

    def array(name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        if name not in data:
            raise ValueError(f"missing saved state: {name}")
        value: np.ndarray = data[name]
        if (
            value.dtype.kind not in "biuf"
            or not np.isfinite(value).all()
            or (shape is not None and value.shape != shape)
        ):
            raise ValueError(f"invalid saved state: {name}")
        return value

    def count(name: str) -> int:
        value = meta.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid saved count: {name}")
        return value

    count("updates")
    bias = meta.get("b_critic")
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not np.isfinite(bias):
        raise ValueError("invalid saved critic bias")
    for name in ("pending", "free_current"):
        if name in meta and not isinstance(meta[name], bool):
            raise ValueError(f"invalid saved flag: {name}")
    batch = 0
    if meta["free_steps"] is not None:
        count("free_steps")
        v = array("free/v")
        if v.ndim != 2 or v.shape[1] != n or not len(v):
            raise ValueError("invalid saved free-state dimensions")
        batch = len(v)
        for name in ("activation", "adaptation"):
            array("free/" + name, (batch, n))
    elif meta.get("pending", False) or meta.get("free_current", False):
        raise ValueError("saved pending action or cache needs a free state")
    if batch:
        array("actor/_drive", (batch, n))
    if ("actor/trace" in data) != ("actor/trace_bias" in data):
        raise ValueError("incomplete saved eligibility traces")
    array("actor/w_critic", (association,))
    for name in ("second_moment", "second_moment_bias", "velocity", "velocity_bias"):
        size = n if name.endswith("bias") else learner.brain.connectome.synapses
        value = array("actor/" + name, (size,))
        if name.startswith("second_moment") and (value < 0).any():
            raise ValueError(f"negative saved second moment: {name}")
    for name, width in (
        ("trace", learner.brain.connectome.synapses),
        ("trace_bias", n),
        ("trace_critic", association + 1),
        ("salience", n),
        ("_drive", n),
    ):
        if "actor/" + name in data:
            array("actor/" + name, (batch, width))
    for name in ("mean", "var"):
        value = array("valence/" + name)
        if value.shape not in ((), (batch,)) or (name == "var" and (value < 0).any()):
            raise ValueError(f"invalid saved valence: {name}")
    if "prepared" in data:
        array("prepared", (batch, sensory))
    if meta.get("pending", False):
        array("pending/value", (batch,))
        for phase in ("plus", "minus"):
            count("pending_" + phase + "_steps")
            for name in ("v", "activation", "adaptation"):
                array(f"pending/{phase}/{name}", (batch, n))
    if "moment/observations" in data or "moment/action" in data:
        if not meta.get("pending", False):
            raise ValueError("saved action observations need pending credit")
        array("moment/observations", (batch, sensory))
        slotted = learner.slot_count > 1
        action = array("moment/action", (batch, learner.slot_count) if slotted else (batch,))
        limit = learner.slot_sizes[None, :] if slotted else len(learner.output_index)
        if action.dtype.kind not in "iu" or (action < 0).any() or (action >= limit).any():
            raise ValueError("invalid saved action indices")
    lived = meta.get("lived")
    if "lived" not in meta:
        if "lived/observations" in data or "lived/action" in data:
            raise ValueError("saved lived arrays require lived metadata")
    else:
        if "arousal" not in meta or not isinstance(lived, dict):
            raise ValueError("a saved lived action needs arousal state")
        forecast, was_sampled = lived.get("forecast"), lived.get("sampled")
        if (
            isinstance(forecast, bool)
            or not isinstance(forecast, (int, float))
            or not np.isfinite(forecast)
            or not isinstance(was_sampled, bool)
            or not isinstance(lived.get("own"), bool)
        ):
            raise ValueError("invalid saved lived forecast")
        recorded = lived.get("recorded")
        if recorded is not None and (
            isinstance(recorded, bool)
            or not isinstance(recorded, (int, float))
            or not np.isfinite(recorded)
            or meta["hippocampus"] is None
        ):
            raise ValueError("invalid saved lived record forecast")
        if was_sampled != bool(meta.get("pending", False)) or batch != 1:
            raise ValueError("a saved lived action must match the pending action of one stream")
        observations = array("lived/observations", (1, sensory))
        slotted = learner.slot_count > 1
        action = array("lived/action", (1, learner.slot_count) if slotted else (1,))
        limit = learner.slot_sizes[None, :] if slotted else len(learner.output_index)
        if action.dtype.kind not in "iu" or (action < 0).any() or (action >= limit).any():
            raise ValueError("invalid saved lived action indices")
        if was_sampled:
            if (
                not np.array_equal(observations, array("moment/observations", (1, sensory)))
                or not np.array_equal(action, array("moment/action", action.shape))
                or float(forecast) != float(array("pending/value", (1,))[0])
            ):
                raise ValueError("saved lived action must match its pending feedback")
    awaiting = meta.get("awaiting")
    if (meta.get("format") == "cadence-generic/5") != ("awaiting" in meta):
        raise ValueError("a sensed state awaiting an outcome belongs to format cadence-generic/5")
    if "awaiting" not in meta:
        if any(name.startswith("awaiting/") for name in data):
            raise ValueError("saved awaiting arrays require awaiting metadata")
    else:
        if not isinstance(awaiting, dict) or "arousal" not in meta:
            raise ValueError("a saved sensed state needs a live stream's arousal")
        if batch != 1 or not (meta.get("pending", False) or "lived" in meta):
            raise ValueError("a saved sensed state must belong to the awaited action of one stream")
        steps = awaiting.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("invalid saved count: awaiting steps")
        for name in ("v", "activation", "adaptation"):
            array("awaiting/" + name, (1, n))
    working = meta["working_memory"]
    if working is not None:
        source, target = working["source"], working["target"]
        if source not in populations or target not in populations:
            raise ValueError("invalid saved working-memory ports")
        width = len(populations[source])
        array("working/trace", (batch, width))
        array("working/last", (batch, width))
        if array("working/cold", (batch,)).dtype != np.bool_:
            raise ValueError("invalid saved working-memory cold flags")
    efference = meta.get("efference")
    if efference is None and (
        "efference" in populations or any(name.startswith("efference/") for name in data)
    ):
        raise ValueError("saved efference neurons and state require efference metadata")
    if efference is not None:
        if meta.get("format") not in ("cadence-generic/4", "cadence-generic/5"):
            raise ValueError(
                "an efference copy belongs to checkpoint formats cadence-generic/4 and /5"
            )
        source, target = efference["source"], efference["target"]
        if (
            source != "motor"
            or target != "efference"
            or target not in populations
            or len(populations[target]) != len(populations[source])
        ):
            raise ValueError("invalid saved efference ports")
        _validate_efference_amplitude(efference["amplitude"])
        width = len(populations[source])
        array("efference/trace", (batch, width))
        array("efference/last", (batch, width))
        if array("efference/cold", (batch,)).dtype != np.bool_:
            raise ValueError("invalid saved efference cold flags")


class Brain:
    """Senses, an association cortex, a motor cortex, basal ganglia and consolidating memory.

    ``connectome`` needs populations ``sensory`` (or ``visual/input`` for an image),
    ``association`` and ``motor``, ``prefrontal`` for a working memory and ``efference``
    (one neuron per motor neuron) for a copy of the issued command; ``genome`` builds
    one. Use ``Brain.compose`` for the default brain with working memory.
    """

    def __init__(
        self,
        connectome: Connectome,
        *,
        episodic: bool = True,
        consolidation: float = 0.05,
        memory_decay: float = 0.9,
        memory_rate: float = 1.0,
        memory_amplitude: float = 1.0,
        working_memory_decay: float = 0.2,
        working_memory_amplitude: float = 3.0,
        working_memory_focus: float = 0.0,
        efference_decay: float = 0.2,
        efference_amplitude: float = 0.0,
        learning: LearnerConfig | None = None,
        reward: ActorCriticConfig | None = None,
        resting_bias: float = 0.0,
        slots: int | Sequence[int] = 1,
        arousal: ArousalConfig | Mapping[str, Any] | bool | None = None,
        seed: int = 0,
        backend: Backend = "cpu",
        device: str | None = None,
        **genes: Any,
    ) -> None:
        learning_genes, actor_genes, arousal_genes = split_genes(genes)
        learning = configured(_learning() if learning is None else learning, learning_genes,
                              "learning")
        reward = configured(_reward() if reward is None else reward, actor_genes, "actor")
        arousal = arousal_config(arousal, arousal_genes)
        if not isinstance(episodic, bool):
            raise ValueError("episodic must be boolean")
        consolidation = real("consolidation", consolidation, low=0, high=1)
        memory_decay = real("memory_decay", memory_decay, low=0, high=1)
        memory_rate = real("memory_rate", memory_rate, low=0, high=1)
        memory_amplitude = real("memory_amplitude", memory_amplitude)
        working_memory_decay = real("working_memory_decay", working_memory_decay, low=0)
        efference_decay = real("efference_decay", efference_decay, low=0)
        if working_memory_decay >= 1 or efference_decay >= 1:
            raise ValueError("working_memory_decay and efference_decay must lie in [0, 1)")
        working_memory_amplitude = real("working_memory_amplitude", working_memory_amplitude)
        working_memory_focus = real("working_memory_focus", working_memory_focus, low=0)
        populations = connectome.populations
        resting_bias = _validate_resting_bias(resting_bias)
        efference_amplitude = _validate_efference_amplitude(efference_amplitude)
        for name in ("association", "motor"):
            if name not in populations:
                raise ValueError(f"a generic brain needs a population named {name!r}")
        sensory = "sensory" if "sensory" in populations else "visual/input"
        if sensory not in populations:
            raise ValueError("a generic brain needs a 'sensory' or 'visual/input' population")
        self.connectome = connectome
        self.sensory_index = np.asarray(populations[sensory], dtype=np.int64)
        self.motor_index = np.asarray(populations["motor"], dtype=np.int64)
        self.association_index = np.asarray(populations["association"], dtype=np.int64)
        bias = None
        if resting_bias:
            # Select named processing populations; boundary exclusions take precedence
            # over overlapping aliases in custom connectomes. This is only an initial
            # operating-point candidate: learning may subsequently change every bias.
            bias = np.zeros(connectome.n)
            selected = np.zeros(connectome.n, dtype=bool)
            excluded = np.zeros(connectome.n, dtype=bool)
            for name, members in populations.items():
                head = name.split("/", 1)[0]
                boundary = head in ("sensory", "visual", "prefrontal", "motor", "efference")
                target = excluded if boundary else selected
                target[np.asarray(members, dtype=np.int64)] = True
            bias[selected & ~excluded] = resting_bias
        brain = NeuralGraph(
            connectome, learning_neuron_model(dt=1.0), backend=backend, device=device, bias=bias
        )
        self.resting_bias = resting_bias
        self._composition: dict[str, Any] | None = None  # construction provenance, not parameters
        self.learner = Learner(brain, list(self.motor_index), learning, slots=slots)
        self.basal_ganglia = ActorCritic(
            self.learner, list(self.association_index), reward, seed=seed
        )
        self.working_memory: Trace | None = None
        if "prefrontal" in populations:
            self.working_memory = Trace(
                connectome,
                decay=working_memory_decay,
                amplitude=working_memory_amplitude,
                focus=working_memory_focus,
                source="association",
                target="prefrontal",
            )
        self.efference: Efference | None = None
        if "efference" in populations:
            if len(populations["efference"]) != len(self.motor_index):
                raise ValueError("an efference population needs one neuron per motor neuron")
            self.efference = Efference(
                connectome,
                decay=efference_decay,
                amplitude=efference_amplitude,
                source="motor",
                target="efference",
            )
        self.hippocampus: FastSynapses | None = None
        if episodic:
            self.hippocampus = SynapticMemory(
                self.sensory_index, self.motor_index, consolidation=consolidation,
                decay=memory_decay, rate=memory_rate, amplitude=memory_amplitude,
            )
        self.rng = np.random.default_rng(seed)
        self._moment: tuple[np.ndarray, np.ndarray] | None = None
        self._prepared: np.ndarray | None = None  # the observations ``learn`` settled
        self.last_learning: dict[str, float] = {}
        self._last_settlement: Mapping[str, Any] | None = None
        # ``live``: the arousal of the stream, and the action it issued that awaits an outcome
        # (observations, action, the value forecast made before the outcome, sampled?, its
        # free state, the brain's own best guess?, the record's forecast of the outcome)
        self.arousal: Arousal | None = None if arousal is None else Arousal(arousal)
        self._lived: (
            tuple[np.ndarray, np.ndarray, float, bool, BrainState, bool, float | None] | None
        ) = None
        self._last_arousal: Mapping[str, Any] | None = None
        # ``wait``: the free state of the action whose outcome is awaited, and the state the
        # stream has sensed since; any operation that replaces that action's state ends it
        self._awaiting: tuple[BrainState, BrainState] | None = None

    @property
    def brain(self) -> NeuralGraph:
        """The current learned dynamics; learning can replace this NeuralGraph instance."""
        return self.learner.brain

    @property
    def pending_feedback(self) -> bool:
        """Whether an issued action still owns the next real outcome.

        Includes greedy routine actions issued by ``live``, which have no
        sampled eligibility. ``wait`` keeps it. A later operation that replaces
        that action also replaces its feedback ownership. Inspect this after a
        refusal to decide whether to retry feedback or only the following action.
        """
        return self.basal_ganglia._pending is not None or (
            self._lived is not None and not self._lived[3]
            and self._lived[4] is self.basal_ganglia.state
        )

    @property
    def decision_id(self) -> int | None:
        """The identity of the ``live`` action that owns the next outcome, or None.

        It is the stream's ``arousal.age`` once ``live`` issued that action (its first
        action is 1), so every action ``live`` issues in a life has its own number;
        ``wait`` does not change it, ``reset`` keeps the age, and a saved brain keeps it.
        Actions that ``step`` or ``act`` issued have none. Pass it to
        ``live(..., decision_id=...)`` so that an outcome reported twice, or late for an
        action the stream has moved past, is refused instead of being credited to the
        action now awaiting one.
        """
        lived, arousal = self._lived, self.arousal
        if (
            arousal is None
            or lived is None
            or lived[4] is not self.basal_ganglia.state
            or not self.pending_feedback
        ):
            return None
        return int(arousal.age)

    @property
    def last_settlement(self) -> Mapping[str, Any] | None:
        """Immutable diagnostics for the latest completed free-answer solve, including refusal.

        ``operation`` is ``act``, ``wait`` or ``predict``; ``step`` uses ``act``, and
        ``accuracy``/``fit`` scores expose their last ``predict`` batch.
        ``qualified`` covers the whole batch; ``row_qualified`` and ``residual``
        are tuples in observation order. ``steps`` counts all free-solve sweeps,
        including damping, within ``budget``. ``residual_checks`` counts extra
        equation transports; ``stagnation_checks`` counts complete-state
        comparisons. ``damping_halvings`` reports the final numerical halving.

        The scope is ``free_answer``: reward eligibility, feedback, teaching,
        imagination and memory work are excluded. These counts are neither
        total computation nor energy. Use ``record_settlements`` to inspect
        actual trajectories, including the separate finite eligibility phases.
        Invalid calls before a solve retain the previous report. Construction,
        ``reset`` and ``load`` start with None; diagnostics are not checkpointed.
        """
        return self._last_settlement

    @property
    def last_arousal(self) -> Mapping[str, Any] | None:
        """Immutable readings of the latest ``live`` moment, or None before the first.

        ``mode`` is ``routine`` or ``aroused``, the mode this moment's action was
        chosen in. ``error`` is the unsigned temporal-difference error of the
        preceding action against the forecast made before its outcome, and
        ``surprise`` and ``want`` are what that outcome added to the arousal
        ``level``. ``temperature`` is the exploration temperature; ``temperatures``
        gives the actual temperature per motor slot. Both are None in routine.
        ``heated_slot`` names the one slot sampled hotter than the base policy,
        or None if there was no extra heating. ``learned`` says a feedback update
        ran for the preceding outcome; ``recorded`` that the outcome which woke
        the brain was written to its memory. ``sweeps`` counts the free-solve
        sweeps of this moment's forecast and answer; ``learning_sweeps`` the
        eligibility and feedback sweeps. These are counts of numerical work,
        not energy. Diagnostics are not checkpointed.
        """
        return self._last_arousal

    # -- construction

    @staticmethod
    def genome(
        inputs: int | Sequence[int],
        actions: int,
        *,
        hidden: int = 64,
        density: float = 1.0,
        lateral: float | None = None,
        working_memory: bool = False,
        memory_scale: float = 12.0,
        features: int = 8,
        field: int = 3,
        seed: int = 0,
        slots: int | Sequence[int] = 1,
    ) -> Genome:
        """The default layout: ``inputs`` is a vector length or an image shape
        ``(height, width)`` or ``(height, width, channels)``.

        ``lateral`` left unset keeps -0.5 motor inhibition up to 8 actions and
        removes it for wider readouts, where it stops the free solve from
        settling; an explicit value is used as given. ``slots`` groups the motor
        neurons into several readouts, one softmax each, with lateral inhibition
        within a slot and the unset ``lateral`` following the largest slot."""
        if lateral is None:
            lateral = (
                _default_lateral(max(slot_sizes(int(actions), slots)))
                if isinstance(actions, (int, np.integer))
                else 0.0
            )
        if isinstance(inputs, (int, np.integer)):
            sensory = Region("sensory", inputs)
            forward = Projection("sensory", "association", density=density, reciprocal=False)
        else:
            shape = tuple(inputs)
            if len(shape) not in (2, 3):
                raise ValueError(
                    "image inputs must be (height, width) or (height, width, channels)"
                )
            height, width, *rest = shape
            sensory = visual_cortex(
                height,
                width,
                channels=rest[0] if rest else 1,
                features=features,
                field=field,
                seed=seed,
            )
            forward = Projection("visual", "association", density=density)
        regions = [
            sensory,
            Region("association", hidden),
            motor_cortex(actions, lateral=lateral, slots=slots),
        ]
        projections = [forward, Projection("association", "motor")]
        if working_memory:
            regions.insert(1, prefrontal_cortex(hidden))
            # the trace of a settled cortex is small; a strong projection lets it steer the next
            # moment
            projections.append(
                Projection("prefrontal", "association", scale=memory_scale, reciprocal=False)
            )
        return Genome(tuple(regions), tuple(projections), label="generic-brain")

    @classmethod
    def build(
        cls,
        inputs: int | Sequence[int],
        actions: int,
        *,
        hidden: int = 64,
        density: float = 1.0,
        lateral: float | None = None,
        working_memory: bool = False,
        memory_scale: float = 12.0,
        episodic: bool = True,
        features: int = 8,
        field: int = 3,
        seed: int = 0,
        slots: int | Sequence[int] = 1,
        **options: Any,
    ) -> Brain:
        """Develop the default genome and wrap it; ``options`` go to the constructor."""
        genome = cls.genome(
            inputs,
            actions,
            hidden=hidden,
            density=density,
            lateral=lateral,
            working_memory=working_memory,
            memory_scale=memory_scale,
            features=features,
            field=field,
            seed=seed,
            slots=slots,
        )
        return cls(develop(genome, seed=seed), episodic=episodic, seed=seed, slots=slots, **options)

    @classmethod
    def compose(
        cls,
        inputs: int,
        actions: int,
        *,
        modules: Sequence[int] = (64,),
        observers: Sequence[int] = (),
        lateral: float | None = None,
        sensory_scale: float = 1.0,
        seed: int = 0,
        slots: int | Sequence[int] = 1,
        **options: Any,
    ) -> Brain:
        """Compose a modular brain with working memory and consolidation.

        ``modules`` specifies a chain of reciprocally connected processing
        regions. The final region is the association cortex. ``observers``
        optionally adds regions with reciprocal access to all processing and
        motor regions, including earlier observers. Every region participates
        in the same neural settlement; observer regions never run as a separate
        critic or override an already completed answer.

        ``lateral`` is the signed weight between each pair of motor neurons.
        Zero removes those lateral synapses, preserving motor/association feedback.
        Left unset, it is -0.5 up to 8 actions and 0.0 above: lateral inhibition
        sharpens a small competing action menu, while on a wider readout it stops
        the undamped free solve from settling and slows damped answers about
        ninefold (issue 124). Select alternatives against the default on
        development tasks; changing inhibition does not itself establish a
        responsive or useful learner.

        ``slots`` groups the motor neurons into several readouts that settle
        together, one softmax each: a count of equal groups, or one size per group
        covering ``actions``. ``act`` and ``step`` then return one index per slot,
        lateral inhibition stays within a slot, and the unset ``lateral`` follows
        the largest slot. One slot is one choice over every action.

        This reuses the existing trace, synaptic memory and learning mechanisms.
        Observer wiring is an experiment, not evidence of learned self-reflection.
        Inputs are fixed external drives; their neural representations can vary.
        ``sensory_scale`` sets the initial sensory projection's magnitude;
        its founder value 1.0 preserves the existing wiring. It is a construction
        setting, not a gain that can be retuned after learning.
        ``options`` accepts the constructor's memory settings and every existing
        configuration field as ``learning_<field>``, ``actor_<field>`` or
        ``arousal_<field>``. ``temperature`` aliases ``learning_temperature``.
        ``learning_homeostasis_rate`` and ``learning_homeostasis_target`` select the
        readout's intrinsic plasticity, a bias step toward a target activation at every
        teaching or reward update; the founder rate 0 leaves the readout as composed.
        ``arousal=True`` enables the founder arousal config; a mapping patches
        those founders. Supplied config objects remain full replacements.
        ``resting_bias`` initializes processing-region biases to a selected nonnegative
        value; it does not guarantee responsive activity or successful acquisition.
        A positive ``efference_amplitude`` adds the efference copy: one ``efference``
        neuron per motor neuron, driven by the fading one-hot of the command the
        stream issued (``efference_decay``) and read by the association region
        through a plastic projection of the working trace's scale. The founder
        value zero builds the brain without it, byte-identical to the released
        composition; the copy is a gene to select against that control.
        Use ``genome``/``Genome`` for custom ports, sparsity and named wiring.
        """

        def size(value: Any) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("population sizes must be positive integers")
            if value < 1:
                raise ValueError("population sizes must be positive integers")
            return int(value)

        inputs, actions = size(inputs), size(actions)
        sensory_scale = real("sensory_scale", sensory_scale, low=0)
        if lateral is None:
            lateral = _default_lateral(max(slot_sizes(actions, slots)))
        if (
            isinstance(lateral, (bool, np.bool_))
            or not isinstance(lateral, (int, float, np.integer, np.floating))
            or not np.isfinite(lateral)
        ):
            raise ValueError("lateral must be a finite signed weight")
        lateral = float(lateral)
        if not np.isfinite(lateral):
            raise ValueError("lateral must be a finite signed weight")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if "episodic" in options and not isinstance(options["episodic"], bool):
            raise ValueError("episodic must be boolean")
        if isinstance(modules, (str, bytes)) or isinstance(observers, (str, bytes)):
            raise ValueError("modules and observers must be sequences of population sizes")
        try:
            widths = tuple(size(value) for value in modules)
            observer_widths = tuple(size(value) for value in observers)
        except TypeError as exc:
            raise ValueError("modules and observers must be sequences of population sizes") from exc
        if not widths:
            raise ValueError("at least one processing module is required")
        names = [f"module_{index}" for index in range(len(widths) - 1)] + ["association"]
        regions = [Region("sensory", inputs)]
        regions.extend(Region(name, width) for name, width in zip(names, widths, strict=True))
        regions.extend(
            (prefrontal_cortex(widths[-1]), motor_cortex(actions, lateral=lateral, slots=slots))
        )
        projections = [Projection("sensory", names[0], scale=sensory_scale, reciprocal=False)]
        projections.extend(
            Projection(left, right) for left, right in zip(names, names[1:], strict=False)
        )
        projections.extend(
            (
                Projection("association", "motor"),
                Projection("prefrontal", "association", scale=12.0, reciprocal=False),
            )
        )
        observed = [*names, "motor"]
        for index, width in enumerate(observer_widths):
            name = f"observer_{index}"
            regions.append(Region(name, width))
            projections.extend(Projection(source, name) for source in observed)
            observed.append(name)
        if _validate_efference_amplitude(options.get("efference_amplitude", 0.0)):
            # Appended last: every earlier region and projection is developed from the same
            # random draws, so the brain with the copy is the brain without it plus the
            # efference neurons and their one projection.
            regions.append(Region("efference", actions))
            projections.append(Projection("efference", "association", scale=12.0, reciprocal=False))
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        result = cls(develop(genome, seed=seed), seed=seed, slots=slots, **options)
        result._composition = {
            "inputs": inputs,
            "actions": actions,
            "modules": list(widths),
            "observers": list(observer_widths),
            "slots": [int(width) for width in result.learner.slot_sizes],
            "lateral": lateral,
            "sensory_scale": sensory_scale,
            "seed": int(seed),
        }
        return result

    def retune(self, *, reset_arousal: bool = False, **genes: Any) -> Brain:
        """Change named settings together, preserving the acquired continuing brain.

        Uses the same ``learning_``, ``actor_`` and ``arousal_`` names as
        ``compose``, its ``temperature`` alias and the existing memory/trace
        settings. ``arousal={...}`` patches the current arousal config; config
        objects supplied as ``learning``, ``reward`` or ``arousal`` replace their
        whole config before named overrides. Unspecified settings are retained,
        including bias rates; pass a bias rate of None to derive it from eta.

        Every value is validated before installation. Parameters, optimizer
        history, memories, traces, random state and pending feedback are kept.
        New settings apply to subsequent operations, including the next reward
        for an already issued action. Changing ``learning_beta`` with sampled
        feedback pending is refused: its saved contrast belongs to the old beta.
        Submit that outcome before changing beta; do not discard it to retune.

        Wiring, initialization and enabling/disabling components are construction
        choices. ``reset_arousal=True`` explicitly resets the arousal level and
        reward references, retaining its age and work counters; no other state
        is reset. Returns this same brain.
        """
        if not isinstance(reset_arousal, bool):
            raise ValueError("reset_arousal must be boolean")
        options = dict(genes)
        construction = {
            "inputs", "actions", "modules", "observers", "lateral", "sensory_scale",
            "resting_bias", "slots", "seed", "episodic", "backend", "device", "connectome",
        }
        for name in options:
            if name in construction:
                raise ValueError(f"{name} is a construction setting; retune preserves the brain")
        learning = options.pop("learning", self.learner.config)
        reward = options.pop("reward", self.basal_ganglia.config)
        current_arousal = None if self.arousal is None else self.arousal.config
        arousal = options.pop("arousal", current_arousal)
        if "arousal" in genes and (self.arousal is None or arousal is None):
            raise ValueError("retune cannot enable or disable arousal; compose it at construction")
        if reset_arousal and self.arousal is None:
            raise ValueError("reset_arousal requires an existing arousal component")

        memory_fields = {
            "working_memory_decay": (self.working_memory, "decay"),
            "working_memory_amplitude": (self.working_memory, "amplitude"),
            "working_memory_focus": (self.working_memory, "focus"),
            "efference_decay": (self.efference, "decay"),
            "efference_amplitude": (self.efference, "amplitude"),
            "memory_decay": (self.hippocampus, "decay"),
            "memory_rate": (self.hippocampus, "rate"),
            "memory_amplitude": (self.hippocampus, "amplitude"),
            "consolidation": (self.hippocampus, "consolidation"),
        }
        memory_changes: list[tuple[Any, str, float]] = []
        for name, (component, field) in memory_fields.items():
            if name not in options:
                continue
            if component is None or (field == "consolidation" and not isinstance(
                component, SynapticMemory
            )):
                raise ValueError(f"{name} requires its existing memory component")
            value = options.pop(name)
            if field == "decay":
                value = real(name, value, low=0, high=1)
                if isinstance(component, Trace) and value == 1:
                    raise ValueError(f"{name} must lie in [0, 1)")
            elif field == "rate":
                assert isinstance(component, FastSynapses)
                value = real(name, value, low=0, high=1 if component.rule == "delta" else np.inf)
            elif field == "consolidation":
                value = real(name, value, low=0, high=1)
            elif field == "focus" or name == "efference_amplitude":
                value = real(name, value, low=0)
            else:
                value = real(name, value)
            memory_changes.append((component, field, value))

        learning_genes, actor_genes, arousal_genes = split_genes(options)
        learning = configured(learning, learning_genes, "learning")
        reward = configured(reward, actor_genes, "actor")
        arousal = arousal_config(
            arousal, arousal_genes, current=current_arousal, retuning=True
        )
        if self.basal_ganglia._pending is not None and learning.beta != self.learner.config.beta:
            raise ValueError("learning_beta cannot change while sampled feedback is pending")

        # Publication begins only after all validation. Never reconstruct a live
        # memory or optimizer to change its scalar settings.
        self.learner.config = learning
        agent = self.basal_ganglia
        agent.config = reward
        if agent._valence is not None:
            agent._valence.level = reward.dopamine_center
            agent._valence.floor = reward.dopamine_floor
            agent._valence.units = not reward.center_scale
        if self.arousal is not None:
            assert arousal is not None
            self.arousal.config = arousal
            if reset_arousal:
                self.arousal.reset()
        for component, field, value in memory_changes:
            setattr(component, field, value)
        return self

    def describe(self) -> dict[str, Any]:
        """Describe effective settings and layout without changing live state.

        ``genes`` contains the named runtime settings of existing components.
        ``initialization`` records the original resting bias and, for composed
        brains, wiring choices including sensory scale. These initial choices
        are provenance, not a description of subsequently learned weights.
        Config and memory sections report the objects currently in use; the
        returned mapping is detached and JSON serializable. No lazy state is
        initialized, no random values are drawn and no settlement is performed.
        """
        learning = self.learner.config.to_dict()
        actor = self.basal_ganglia.config.to_dict()
        arousal = None if self.arousal is None else self.arousal.config.to_dict()
        genes = {f"learning_{name}": value for name, value in learning.items()}
        genes.update({f"actor_{name}": value for name, value in actor.items()})
        if arousal is not None:
            genes.update({f"arousal_{name}": value for name, value in arousal.items()})
        for prefix, component, fields in (
            ("working_memory", self.working_memory, ("decay", "amplitude", "focus")),
            ("efference", self.efference, ("decay", "amplitude")),
            ("memory", self.hippocampus, ("decay", "rate", "amplitude")),
        ):
            if component is not None:
                genes.update({f"{prefix}_{field}": getattr(component, field) for field in fields})
        if isinstance(self.hippocampus, SynapticMemory):
            genes["consolidation"] = self.hippocampus.consolidation
        populations = self.connectome.populations
        result = {
            "layout": {
                "inputs": len(self.sensory_index),
                "actions": len(self.motor_index),
                "modules": _region_widths(populations, "module_") + [len(self.association_index)],
                "observers": _region_widths(populations, "observer_"),
                "slots": self.learner.slot_sizes,
                "neurons": self.connectome.n,
                "synapses": self.connectome.synapses,
                "populations": {name: len(members) for name, members in populations.items()},
            },
            "genes": genes,
            "learning": learning,
            "actor": actor,
            "arousal": arousal,
            "working_memory": None if self.working_memory is None
            else self.working_memory.to_dict(),
            "efference": None if self.efference is None else self.efference.to_dict(),
            "memory": None if self.hippocampus is None else self.hippocampus.to_dict(),
            "initialization": {"resting_bias": self.resting_bias, "composition": self._composition},
            "pending_feedback": self.pending_feedback,
        }
        return dict(plain(result))

    # -- stimulus

    def _observations(self, observations: Any) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim < 2 or not x.shape[0] or not np.isfinite(x).all():
            raise ValueError(
                "observations must be a nonempty finite batch; use [observation] for one"
            )
        x = x.reshape(len(x), -1)
        if x.shape[1] != len(self.sensory_index):
            raise ValueError(
                f"observations need {len(self.sensory_index)} values per row, got {x.shape[1]}"
            )
        return x

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        """The drive of a batch of observations: the sensory neurons stimulated, plus the
        working memory and the hippocampal recall when ``memory`` is set."""
        x = self._observations(observations)
        drive = np.zeros((len(x), self.connectome.n))
        drive[:, self.sensory_index] = x * self.brain.neuron_model.stimulus_amplitude
        if memory and self.working_memory is not None:
            drive = self.working_memory.stimulate(drive)
        if memory and self.efference is not None:
            drive = self.efference.stimulate(drive)
        if memory and self.hippocampus is not None:
            drive = self.hippocampus.stimulate(drive)
        return drive

    # -- independent-sample convenience operations (no mode switch)

    def fit(
        self,
        observations: Any,
        labels: Any,
        *,
        epochs: int = 30,
        batch: int = 32,
    ) -> list[float]:
        """Teach independent samples; return qualified training accuracy after each epoch."""
        drive = self.stimulus(observations, memory=False)
        labels = self.learner._labels(np.asarray(labels))
        if len(labels) != len(drive):
            raise ValueError("observations and labels must have the same batch size")
        for name, value, minimum in (("epochs", epochs, 0), ("batch", batch, 1)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        # Supervision changes the parameters under any previously cached decision.
        self.reset()
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(labels))
            for start in range(0, len(labels), batch):
                rows = order[start : start + batch]
                self.learner.step(drive[rows], labels[rows])
            history.append(self.accuracy(observations, labels))
        return history

    def _equilibrate(
        self, drive: np.ndarray, state: BrainState | None, *, budget: int, tolerance: float
    ) -> Equilibrium:
        """One bounded free solve, with underrelaxation if fast repair does not qualify.

        Finite learners retain the one half-step fallback. Qualified learners
        use their declared damping count, with identical model equations and
        parameters, within the caller's exact total sweep budget.
        This numerical fallback is independent of optional observer wiring.
        """
        from .patch import _copy_state

        cfg = self.learner.config
        brain = self.brain
        if cfg.qualified:
            return brain.equilibrate(
                drive,
                state=_copy_state(state),
                budget=budget,
                tolerance=tolerance,
                damping=cfg.damping,
            )
        first_budget = budget if budget < 2 else (budget + 1) // 2
        phase = brain.equilibrate(
            drive, state=_copy_state(state), budget=first_budget, tolerance=tolerance
        )
        if np.all(phase.qualified) or phase.steps >= budget:
            return phase
        if not all(
            np.isfinite(value).all()
            for value in (phase.state.v, phase.state.activation, phase.state.adaptation)
        ):
            return phase
        damped = NeuralGraph(
            brain.connectome,
            brain.neuron_model.replace(dt=brain.neuron_model.dt / 2),
            backend=brain.backend,
            efficacy=brain.efficacy,
            log_gain=brain.log_gain,
            bias=brain.bias,
            device=str(brain._torch.device) if brain._torch is not None else None,
            dense_limit=brain.dense_limit,
            layout=brain.layout,
            precision=brain.precision,
        )
        following = damped.equilibrate(
            drive, state=_copy_state(phase.state), budget=budget - phase.steps, tolerance=tolerance
        )
        residual = brain.residual(drive, following.state)
        return Equilibrium(
            following.state,
            residual,
            phase.steps + following.steps,
            tolerance,
            residual_checks=phase.residual_checks + following.residual_checks + 1,
            damping_halvings=1,
        )

    def _qualified(
        self, drive: np.ndarray, state: BrainState | None = None, *, operation: str = "answer"
    ) -> BrainState:
        """Check the complete current graph before publishing an answer."""
        cfg = self.learner.config
        if cfg.tolerance is None:
            raise ValueError("Brain answers require a finite residual tolerance")
        phase = self._equilibrate(drive, state, budget=cfg.free_steps, tolerance=cfg.tolerance)
        rows = tuple(bool(value) for value in phase.qualified)
        residual = tuple(float(value) for value in phase.residual)
        self._last_settlement = MappingProxyType(
            {
                "operation": operation,
                "scope": "free_answer",
                "qualified": all(rows),
                "row_qualified": rows,
                "residual": residual,
                "max_residual": max(residual),
                "steps": int(phase.steps),
                "budget": int(cfg.free_steps),
                "tolerance": float(phase.tolerance),
                "residual_checks": int(phase.residual_checks),
                "damping_halvings": int(phase.damping_halvings),
                "stagnation_checks": int(phase.stagnation_checks),
            }
        )
        if not all(rows):
            raise RuntimeError(
                f"brain did not settle within {cfg.free_steps} steps: "
                f"residual={float(np.max(phase.residual)):.6g}, tolerance={cfg.tolerance:g}; "
                "no action issued"
            )
        return phase.state

    def _choices(self, state: BrainState) -> np.ndarray:
        """The most active motor neuron per output slot: ``(batch,)`` for one slot,
        ``(batch, slots)`` when the motor neurons split into several readouts."""
        out = np.asarray(state.activation)[:, self.motor_index]
        learner = self.learner
        if learner.slot_count > 1 and learner.slot_size == 0:  # unequal slots: one argmax each
            choice = [
                np.argmax(out[:, offset : offset + size], axis=1)
                for offset, size in zip(learner.slot_offsets, learner.slot_sizes, strict=True)
            ]
            return np.asarray(np.stack(choice, axis=1), dtype=np.int64)
        if learner.slot_count > 1:
            out = out.reshape(len(out), learner.slot_count, learner.slot_size)
        return np.asarray(np.argmax(out, axis=-1), dtype=np.int64)

    def predict(self, observations: Any) -> np.ndarray:
        """Qualified independent observations, without reading or changing live memory.

        One choice per output slot: ``(batch,)`` for one slot, ``(batch, slots)``
        when ``slots`` split the motor neurons, matching ``act`` and ``step``."""
        state = self._qualified(self.stimulus(observations, memory=False), operation="predict")
        return self._choices(state)

    def accuracy(self, observations: Any, labels: Any) -> float:
        x = self._observations(observations)
        expected = self.learner._labels(np.asarray(labels))
        if len(expected) != len(x):
            raise ValueError("observations and labels must have the same batch size")
        hits = 0
        for start in range(0, len(x), 256):
            predicted = self.predict(x[start : start + 256])
            wanted = expected[start : start + 256]
            hits += int(np.sum(predicted == wanted.reshape(predicted.shape)))
        return hits / expected.size

    # -- ongoing interaction

    def step(
        self,
        observations: Any,
        *,
        reward: Any = None,
        done: Any = None,
        teacher: Any = None,
        salience: Any = None,
        bootstrap: Any = None,
    ) -> np.ndarray:
        """One operating mode: observe, learn from the last outcome, and act again.

        Reward/done describe the preceding action; an omitted reward means no
        reward event on a real transition (zero), not an unknown outcome. Wait for
        the issued action to finish before the next call. ``teacher`` labels the
        current observation. The first
        call has no previous action to reward. Later calls retain the batch's row
        identities; call ``reset`` before starting different streams. Salience is
        nonnegative, per row, and defaults to absolute reward for consolidation.
        There is no training/inference switch. For a frozen measurement use a
        separate instance's ``predict``/greedy ``act``. If the next action fails
        to settle, already accepted feedback remains learned. Retry with ``act``
        after adjusting the solver; do not submit that outcome a second time.
        """
        x = self._observations(observations)
        batch = len(x)
        current = self.basal_ganglia.state
        if current is not None and len(np.atleast_2d(current.v)) != batch:
            raise ValueError("action batch must match the live streams; reset for new streams")
        r = np.zeros(batch) if reward is None else np.asarray(reward, dtype=float)
        ended = np.zeros(batch, bool) if done is None else np.asarray(done)
        if r.shape != (batch,) or not np.isfinite(r).all():
            raise ValueError("reward must be a finite vector matching the observation batch")
        if ended.shape != (batch,) or ended.dtype != np.bool_:
            raise ValueError("done must be a boolean vector matching the observation batch")
        importance = SynapticMemory.salience_vector(
            np.abs(r) if salience is None else salience, batch
        )
        labels = None if teacher is None else self.learner._labels(np.asarray(teacher))
        if labels is not None and len(labels) != batch:
            raise ValueError("teacher must match the observation batch")
        if bootstrap is not None:
            bootstrap = np.asarray(bootstrap, dtype=float)
            if bootstrap.shape != (batch,) or not np.isfinite(bootstrap).all():
                raise ValueError("bootstrap must be a finite vector matching the batch")
        pending = self.basal_ganglia._pending is not None
        if not pending and (reward is not None or done is not None or bootstrap is not None):
            raise RuntimeError("feedback needs a preceding action; start with step(observations)")
        # All feedback and demonstrations are checked before changing any state.
        if pending:
            self.last_learning = self.learn(r, ended, x, bootstrap=bootstrap, salience=importance)
        else:
            self.last_learning = {}
        if labels is not None:
            from .learning import LearningPhaseError

            drive = self.stimulus(x)
            try:
                _, teaching = self.learner.step(drive, labels)
            except LearningPhaseError as error:
                self.last_learning.update(
                    {"demonstration_" + name: value for name, value in error.report.items()}
                )
                self.last_learning["demonstrations"] = 0.0
                raise
            self.last_learning.update(
                {"demonstration_" + name: value for name, value in teaching.items()}
            )
            # A demonstration teaches the slow policy, not an invented reward value.
            self._prepared = None
            self.basal_ganglia._drive = None
            self.last_learning["demonstrations"] = float(batch)
        return self.act(x)

    def live(
        self, observations: Any, *, reward: Any = None, done: Any = None, decision_id: Any = None
    ) -> np.ndarray:
        """One moment of a continuing life: routine while outcomes match, repair when not.

        ``reward`` and ``done`` describe the preceding action, as in ``step``. The
        brain's ``arousal`` decides what this moment costs. Calm, it answers with
        the greedy choice of one qualified settle: no eligibility phases, no
        learning, no memory write, no parameter changes; the eligibility of earlier
        sampled actions fades by one step, as time passes. Its forecasts for the
        preceding action were the critic's value when it acted and, when it has an
        associative memory, the record it held for that action; the outcome is
        measured against the value with the value of the present state and against
        the record as it is, and a surprising outcome or a reward below what life
        usually pays raises the arousal (``Arousal`` gives the law). Aroused, the brain samples its
        action from its policy. Want raises the temperature of one uniformly chosen
        motor slot; the other slots keep the base temperature. It keeps eligibility and
        learns from the outcome as ``step`` does. The outcome that woke a calm
        brain is written to its memory for the situation it was chosen in; the
        actor and critic learn from the outcomes that follow, while it is awake.

        This follows one stream: pass one observation row. An action that
        ``step`` or ``act`` sampled is adopted, so a bootstrapped brain continues
        here without a reset. The mood follows the outcomes ``live`` receives.
        If the forecast settle refuses, nothing has changed and the same call can
        be retried. If the answer refuses after the outcome was taken, the
        outcome stays learned and counted: retry with ``live(observations)``
        without that reward. Readings of the moment are in ``last_arousal``.

        An omitted reward is a zero outcome, not a missing one. When the outcome
        comes later than the next observations, sense them with ``wait`` and report
        the outcome here once it is observed: it is measured against the forecasts
        made when the action was chosen, and credited to that action's eligibility and
        situation as an immediate outcome would be. ``decision_id`` names the action
        the outcome belongs to (``Brain.decision_id``); any other value is refused
        with ``ValueError`` before anything changes.
        """
        arousal = self.arousal
        if arousal is None:
            raise ValueError(
                "live needs arousal genes; construct the brain with arousal=True"
            )
        x = self._observations(observations)
        agent = self.basal_ganglia
        current = agent.state
        if len(x) != 1 or (current is not None and len(np.atleast_2d(current.v)) != 1):
            raise ValueError("live follows one continuing stream; reset before changing streams")
        lived = self._lived
        if lived is not None and current is not lived[4]:
            lived = None  # another operation acted since; the brain's own pending action governs
        sampled = agent._pending is not None
        routine = lived is not None and not lived[3] and not sampled
        if decision_id is not None:
            self._owned(decision_id)
        if not sampled and not routine and (reward is not None or done is not None):
            raise RuntimeError("feedback needs a preceding action; start with live(observations)")
        r = np.zeros(1) if reward is None else np.asarray(reward, dtype=float)
        ended = np.zeros(1, bool) if done is None else np.asarray(done)
        if r.shape != (1,) or not np.isfinite(r).all():
            raise ValueError("reward must be one finite value for the stream")
        if ended.shape != (1,) or ended.dtype != np.bool_:
            raise ValueError("done must be one boolean for the stream")
        answer: tuple[np.ndarray, BrainState] | None = None
        error = surprise = want = 0.0
        record_error = None if lived is None or lived[6] is None else abs(float(r[0]) - lived[6])
        sweeps = learning_sweeps = 0
        learned = recorded = False
        if sampled:
            report = self.learn(r, ended, x)
            self._lived = None  # accepted feedback must never become pending again
            self.last_learning = report
            error = float(report["td_error"])
            learning_sweeps += int(report["free_steps"])
            learned = True
        elif routine:
            assert lived is not None
            answer = self._forecast(x, bool(ended[0]))
            assert self._last_settlement is not None
            sweeps += int(self._last_settlement["steps"])
            following = 0.0 if ended[0] else float(agent.value(answer[1])[0])
            error = abs(float(r[0]) + agent.config.gamma * following - lived[2])
            self.last_learning = {}
        if sampled or routine:
            # Only the brain's own best guess can surprise it; every actual reward
            # enters its income. An adopted action has no record and counts as its own.
            own = True if lived is None else lived[5]
            try:
                surprise, want = arousal.outcome(
                    error, float(r[0]), own=own, learned=sampled, record_error=record_error
                )
            except ValueError as exc:
                if sampled:
                    raise ValueError(
                        "feedback was accepted but arousal could not update; "
                        "retry live(observations) without reward or done"
                    ) from exc
                raise
            self._lived = None  # this outcome is taken, exactly once
            self._awaiting = None  # and the stream continues from where it settled
            if routine:
                if ended[0] and self.working_memory is not None:
                    self.working_memory.reset(1, rows=np.array([0]))
                if ended[0] and self.efference is not None:
                    self.efference.reset(1, rows=np.array([0]))
                # a moment without eligibility has passed: the credit of earlier sampled
                # actions fades as it does between two outcomes that are learned from
                agent.fade(ended)
            if routine and arousal.aroused and self.hippocampus is not None:
                assert lived is not None
                self._record(lived[0], lived[1], r, SynapticMemory.salience_vector(np.abs(r), 1))
                recorded = True
                assert answer is not None
                # the record can change this moment's recall: settle again under it, warm
                agent._free, agent._free_brain, agent._drive = answer[1], self.brain, answer[0]
                answer = None
        aroused = arousal.aroused
        temperature = self.learner.config.temperature * arousal.heat if aroused else None
        # Validate before qualification; invalid heating must draw nothing.
        if temperature is not None:
            agent._temperature(temperature)
        if answer is None:
            answer = self._settled(x)
            assert self._last_settlement is not None
            sweeps += int(self._last_settlement["steps"])
        # Extra exploration perturbs one motor slot at a time. The other slots
        # still sample their learned policy; one-slot sampling is unchanged.
        temperatures: float | np.ndarray | None = temperature
        heated_slot = None
        if temperature is not None and temperature > self.learner.config.temperature:
            slot_count = self.learner.slot_count
            heated_slot = int(agent.rng.integers(slot_count)) if slot_count > 1 else 0
            if slot_count > 1:
                temperatures = np.full(slot_count, self.learner.config.temperature)
                temperatures[heated_slot] = temperature
        action = self._choose(x, *answer, greedy=not aroused, temperature=temperatures)
        state = agent.state
        assert state is not None
        if agent._pending is not None:
            learning_sweeps += int(agent._pending[1].steps) + int(agent._pending[2].steps)
        # the brain's own best guess: the most active motor neuron of every slot
        motor = np.asarray(state.activation)[0, self.motor_index]
        slots = zip(self.learner.slot_offsets, self.learner.slot_sizes, strict=True)
        best = [int(np.argmax(motor[start : start + size])) for start, size in slots]
        chosen = [int(choice) for choice in np.atleast_1d(action[0])]
        self._lived = (
            x.copy(),
            action.copy(),
            float(agent.value(state)[0]),
            aroused,
            state,
            best == chosen,
            self._recorded(x, chosen),
        )
        self._last_arousal = MappingProxyType(
            {
                "mode": arousal.mode,
                "level": float(arousal.level),
                "error": float(error),
                "record_error": None if record_error is None else float(record_error),
                "surprise": float(surprise),
                "want": float(want),
                "temperature": None if temperature is None else float(temperature),
                "temperatures": None
                if temperatures is None
                else tuple(
                    float(value)
                    for value in np.broadcast_to(temperatures, (self.learner.slot_count,))
                ),
                "heated_slot": heated_slot,
                "learned": learned,
                "recorded": recorded,
                "sweeps": int(sweeps),
                "learning_sweeps": int(learning_sweeps),
            }
        )
        arousal.lived(sweeps, learning_sweeps)
        return action

    def wait(self, observations: Any) -> None:
        """One moment of the stream while its issued action's outcome is still to come.

        The brain settles the present observation from its current activity, reading its
        working trace, the copy of its last command and its associative memory, and its
        working trace advances to that state. It issues no action and takes no outcome:
        the awaited action keeps the forecasts made before it, its eligibility and the
        situation it was chosen in, and ``live`` later credits its actual outcome to it
        once. Parameters, the critic, eligibility traces, associative memory, random
        state, the command copy and the arousal state stay as they were: eligibility,
        arousal, its age and youth advance with outcomes and live moments, and one
        outcome is one temporal-difference step however many moments were waited. The
        settle is reported by ``last_settlement`` with ``operation`` ``wait``; like the
        work of a refused attempt, it is not part of ``arousal``'s counts.

        This follows the stream of ``live``: one observation row and arousal genes
        (``ValueError`` otherwise), and an action awaiting its outcome (``RuntimeError``
        otherwise). The caller decides which moments are waited; the brain does not
        choose to wait. There is no deadline. An outcome that will never come is not a
        zero reward: ``act`` replaces the action without learning from it (``step``
        would take a sampled action's omitted reward as a zero outcome), and ``reset``
        begins a new stream. A refused settle raises ``RuntimeError`` and changes
        nothing but ``last_settlement``.
        """
        if self.arousal is None:
            raise ValueError(
                "wait needs arousal genes; construct the brain with arousal=True"
            )
        x = self._observations(observations)
        current = self.basal_ganglia.state
        if len(x) != 1 or (current is not None and len(np.atleast_2d(current.v)) != 1):
            raise ValueError("wait follows one continuing stream; reset before changing streams")
        if current is None or not self.pending_feedback:
            raise RuntimeError(
                "wait needs an issued action awaiting its outcome; start with live(observations)"
            )
        drive = self.stimulus(x)
        state = self._qualified(drive, self._activity(), operation="wait")
        if self.working_memory is not None:
            self.working_memory.update(state)
        self._awaiting = (current, state)

    def _owned(self, decision_id: Any) -> None:
        """Refuse an outcome reported for an action other than the one awaiting it."""
        if (
            isinstance(decision_id, (bool, np.bool_))
            or not isinstance(decision_id, (int, np.integer))
            or decision_id < 1
        ):
            raise ValueError(
                "decision_id must be a positive integer read from Brain.decision_id"
            )
        awaited = self.decision_id
        if awaited is None:
            raise ValueError(
                f"decision_id {int(decision_id)} does not own an outcome: no action that "
                "live issued awaits one; after an outcome was taken and the answer "
                "refused, retry with live(observations) alone"
            )
        if int(decision_id) != awaited:
            raise ValueError(
                f"decision_id {int(decision_id)} does not own the next outcome; "
                f"the awaited action is decision_id {awaited}"
            )

    def _awaited(self) -> BrainState | None:
        """The state sensed by ``wait`` while the current action's outcome is awaited."""
        awaiting = self._awaiting
        if (
            awaiting is None
            or awaiting[0] is not self.basal_ganglia.state
            or not self.pending_feedback
        ):
            return None
        return awaiting[1]

    def _activity(self) -> BrainState | None:
        """The stream's current activity, warm start of its next settle."""
        awaited = self._awaited()
        return self.basal_ganglia.state if awaited is None else awaited

    def _recorded(self, x: np.ndarray, chosen: list[int]) -> float | None:
        """The outcome the associative memory forecasts for the chosen action in this
        situation: the mean of the records it holds for the chosen motor neurons, in reward
        units. None without a memory, or when its reads are silent."""
        memory = self.hippocampus
        if memory is None or not np.isfinite(memory.amplitude) or memory.amplitude == 0.0:
            return None
        units = np.asarray(chosen) + np.asarray(self.learner.slot_offsets)[: len(chosen)]
        values = np.asarray(memory.recall(x))[0, units] / memory.amplitude
        return float(np.mean(values)) if np.isfinite(values).all() else None

    def _forecast(self, x: np.ndarray, ended: bool) -> tuple[np.ndarray, BrainState]:
        """Read the next value without changing the stream. A finished episode's
        forecast starts from rest and fresh traces; accepting its outcome commits
        that reset, so a refused forecast or arousal update preserves feedback."""
        agent = self.basal_ganglia
        if not ended:
            return self._settled(x)
        kept = (agent._free, agent._free_brain, agent._drive)
        traces = [
            (trace, {name: getattr(trace, name).copy() for name in ("trace", "last", "cold")})
            for trace in (self.working_memory, self.efference)
            if trace is not None
        ]
        try:
            for trace, _ in traces:
                trace.reset(1, rows=np.array([0]))
            agent._free = None
            return self._settled(x)
        finally:
            agent._free, agent._free_brain, agent._drive = kept
            for trace, values in traces:
                for name, value in values.items():
                    setattr(trace, name, value)

    # -- lower-level interaction operations

    def imagine(
        self,
        observations: Sequence[Any],
        *,
        budget: int = 1024,
        tolerance: float = 1e-6,
    ) -> tuple[Equilibrium, ...]:
        """Privately settle responses to a supplied sequence of possible observations.

        Each item is a batch, with the same stream identities throughout. A
        private trace advances along the imagined branch; durable memory is read
        without writes. Parameters, live activity, random generators and pending
        real outcomes are unchanged. Motor responses are at ``motor_index`` in
        each returned state's activation. A refused phase ends the branch and
        remains in the result; inspect ``qualified`` before using a response.

        This imagines the brain's responses, not an unobserved environment's
        dynamics. ``TemporalPatchNet.plan`` supplies the separate learned
        consequence-model interface for continuous action planning.
        """
        from copy import copy

        from .patch import _copy_state

        if (
            isinstance(budget, (bool, np.bool_))
            or not isinstance(budget, (int, np.integer))
            or budget < 0
        ):
            raise ValueError("budget must be a nonnegative integer")
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("tolerance must be finite and nonnegative")
        sequence = tuple(self._observations(value) for value in observations)
        if not sequence:
            return ()
        batch = len(sequence[0])
        if any(len(value) != batch for value in sequence):
            raise ValueError("imagined observations must preserve the batch's stream identities")
        current = _copy_state(self._activity())
        if current is not None and len(np.atleast_2d(current.v)) != batch:
            raise ValueError("imagined batch must match the live streams; reset for new streams")
        trace = copy(self.working_memory)
        if trace is not None:
            for name in ("trace", "last", "cold"):
                setattr(trace, name, getattr(trace, name).copy())
        echo = copy(self.efference)
        if echo is not None:
            for name in ("trace", "last", "cold"):
                setattr(echo, name, getattr(echo, name).copy())
        phases = []
        for value in sequence:
            drive = self.stimulus(value, memory=False)
            if trace is not None:
                drive = trace.stimulate(drive)
            if echo is not None:
                drive = echo.stimulate(drive)
            if self.hippocampus is not None:
                drive = self.hippocampus.stimulate(drive)
            phase = self._equilibrate(drive, current, budget=budget, tolerance=tolerance)
            phases.append(phase)
            if not np.all(phase.qualified):
                break
            current = _copy_state(phase.state)
            if trace is not None:
                trace.update(phase.state)
            if echo is not None:
                # the imagined branch issues the response's own best guess, privately
                echo.issue(self._command(self._choices(phase.state)))
        return tuple(phases)

    def act(
        self,
        observations: Any,
        *,
        greedy: bool = False,
        temperature: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Qualify the whole graph and choose an action for each continuing stream.

        ``learning.free_steps`` bounds repair; ``learning.tolerance`` checks the
        full potential/adaptation equations, including optional observers.
        Exhaustion raises ``RuntimeError`` before changing activity, memory,
        random state or pending feedback. A cached state is always checked anew.
        Reward eligibility retains its separate finite nudged-phase contract.
        ``temperature`` samples the settled motor state at another softmax
        temperature than the learner's, or one per motor slot. Eligibility credits
        the policy that actually sampled the action at those temperatures.
        """
        x = self._observations(observations)
        current = self.basal_ganglia.state
        if current is not None and len(np.atleast_2d(current.v)) != len(x):
            raise ValueError("action batch must match the live streams; reset for new streams")
        if temperature is not None:
            temperature = self.basal_ganglia._temperature(temperature)
        drive, free = self._settled(x)
        return self._choose(x, drive, free, greedy=greedy, temperature=temperature)

    def _settled(self, x: np.ndarray) -> tuple[np.ndarray, BrainState]:
        """The drive of validated observations and its qualified free state, warm from the
        stream's last state. Refusal raises before anything changes."""
        drive = self.stimulus(x)
        return drive, self._qualified(drive, self._activity(), operation="act")

    def _choose(
        self,
        x: np.ndarray,
        drive: np.ndarray,
        free: BrainState,
        *,
        greedy: bool,
        temperature: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Issue the action of a qualified free state: the stream advances to it."""
        self.basal_ganglia._free = free
        self.basal_ganglia._free_brain = self.brain
        self.basal_ganglia._drive = drive.copy()
        self._prepared = None
        self._awaiting = None
        action = self.basal_ganglia.act(drive, greedy=greedy, temperature=temperature)
        state = self.basal_ganglia.state
        if self.working_memory is not None and state is not None:
            self.working_memory.update(state)
        if self.efference is not None:
            self.efference.issue(self._command(action))
        self._moment = None if greedy else (x.copy(), action.copy())
        return action

    def _command(self, action: np.ndarray) -> np.ndarray:
        """The ``(batch, motor)`` one-hot of an issued action: one per row, or one per slot."""
        action = np.asarray(action, dtype=np.int64)
        command = np.zeros((len(action), len(self.motor_index)))
        rows = np.arange(len(action))
        if action.ndim == 2:
            command[rows[:, None], action + self.learner.slot_offsets[None, :]] = 1.0
        else:
            command[rows, action] = 1.0
        return command

    def learn(
        self,
        reward: Any,
        done: Any,
        next_observations: Any,
        *,
        bootstrap: Any = None,
        salience: Any = None,
    ) -> dict[str, float]:
        """Dopamine from the reward of the last action; ``done`` rows start a new episode.

        The hippocampus records the reward of the chosen action for the situation it was
        chosen in, and the next choice in that situation reads the record. A refused
        feedback update preserves that pending action and both memories; retry this
        outcome after adjusting its numerical configuration. After ``wait`` the next
        state settles from the state the stream sensed last; the critic's eligibility
        still belongs to the state the action was chosen in.
        """
        following = self._observations(next_observations)
        # Preflight without reading or resetting memories: invalid input must not write an episode.
        reward, done, _, bootstrap = self.basal_ganglia._validated_transition(
            reward, done, self.stimulus(following, memory=False), bootstrap
        )
        importance = SynapticMemory.salience_vector(
            np.abs(reward) if salience is None else salience, len(following)
        )
        memory = self.hippocampus
        # Synaptic writes replace arrays, so retain their old references rather
        # than copying a potentially large persistent matrix for every outcome.
        memory_values = (
            {}
            if memory is None
            else {name: getattr(memory, name) for name in ("strength", "mass", "writes")}
        )
        if isinstance(memory, SynapticMemory):
            memory_values["consolidated"] = memory.consolidated
        separator_mean = (
            None if memory is None or memory.separator is None else memory.separator.mean
        )
        trace = self.working_memory
        trace_values = (
            {}
            if trace is None or not done.any()
            else {name: getattr(trace, name).copy() for name in ("trace", "last", "cold")}
        )
        echo = self.efference
        echo_values = (
            {}
            if echo is None or not done.any()
            else {name: getattr(echo, name).copy() for name in ("trace", "last", "cold")}
        )
        try:
            if memory is not None and self._moment is not None:
                keys, action = self._moment
                self._record(keys, action, reward, importance)
            if trace is not None and done.any():
                trace.reset(len(done), rows=np.flatnonzero(done))
            if echo is not None and done.any():
                echo.reset(len(done), rows=np.flatnonzero(done))
            report = self.basal_ganglia.learn(
                reward, done, self.stimulus(following), bootstrap, warm=self._awaited()
            )
        except Exception:
            if memory is not None:
                for name, value in memory_values.items():
                    setattr(memory, name, value)
                if memory.separator is not None and separator_mean is not None:
                    memory.separator.mean = separator_mean
            if trace is not None:
                for name, value in trace_values.items():
                    setattr(trace, name, value)
            if echo is not None:
                for name, value in echo_values.items():
                    setattr(echo, name, value)
            raise
        self._moment = None
        self._prepared = following.copy()
        self._awaiting = None
        return report

    def _record(
        self, keys: np.ndarray, action: np.ndarray, reward: np.ndarray, importance: np.ndarray
    ) -> None:
        """Write the outcome of a chosen action for the situation it was chosen in."""
        memory = self.hippocampus
        assert memory is not None
        # The chosen motor neurons in motor order: one per row, or one per slot.
        rows: np.ndarray = np.arange(len(action))
        value: np.ndarray = reward
        chosen: np.ndarray = action
        if action.ndim == 2:
            chosen = action + self.learner.slot_offsets[None, :]
            rows, value = rows[:, None], reward[:, None]
        if isinstance(memory, SynapticMemory):
            target = np.zeros((len(action), len(self.motor_index)))
            target[rows, chosen] = value
            observed = np.zeros(target.shape, bool)
            observed[rows, chosen] = True
            memory.observe(keys, target, salience=importance, value_mask=observed)
        else:
            target = memory.recall(keys)
            target[rows, chosen] = value
            memory.observe(keys, target)

    def reset(self) -> None:
        """Start every stream afresh; hippocampal records are kept. A brain with arousal
        begins the new stream calm, with what it was used to forgotten and its age kept."""
        self.basal_ganglia.reset()
        if self.working_memory is not None:
            self.working_memory.reset(0)
        if self.efference is not None:
            self.efference.reset(0)
        self._moment = None
        self._prepared = None
        self.last_learning = {}
        self._last_settlement = None
        self._lived = None
        self._last_arousal = None
        self._awaiting = None
        if self.arousal is not None:
            self.arousal.reset()

    def parameters(self) -> int:
        """Actor/critic and consolidated weights; per-stream transient storage is additional."""
        return self.basal_ganglia.parameters() + (
            self.hippocampus.consolidated.size
            if isinstance(self.hippocampus, SynapticMemory)
            else 0
        )

    def save(self, path: str | Path) -> Path:
        """Save the complete composition, including an action awaiting its real outcome.

        Includes critic, optimizer, random generators, working and episodic memories,
        eligibility, the current free phase, any pending action's nudged states and the
        state ``wait`` sensed since that action was chosen.
        ``Learner.save`` remains available for the slow learned response alone.
        """
        from .checkpoint import _learner_data, _write
        from .receipts import canonical_json

        agent = self.basal_ganglia
        data = _learner_data(self.learner)
        metadata: dict[str, Any] = {
            "format": "cadence-generic/2",
            "resting_bias": _validate_resting_bias(self.resting_bias),
            "reward": agent.config.to_dict(),
            "rng": self.rng.bit_generator.state,
            "actor_rng": agent.rng.bit_generator.state,
            "updates": agent.updates,
            "b_critic": agent.b_critic,
            "working_memory": None
            if self.working_memory is None
            else self.working_memory.to_dict(),
            "hippocampus": None if self.hippocampus is None else self.hippocampus.to_dict(),
            "free_steps": None if agent.state is None else agent.state.steps,
            "free_current": agent._free_brain is self.brain,
            "pending": agent._pending is not None,
            "last_learning": self.last_learning,
        }
        if self._composition is not None:
            metadata["composition"] = plain(self._composition)
        if self.arousal is not None:
            # Arousal is part of the continuation; earlier formats cannot carry it.
            metadata["format"] = "cadence-generic/3"
            metadata["arousal"] = self.arousal.to_dict()
            lived = self._lived
            if lived is not None and agent.state is not lived[4]:
                lived = None  # another operation acted since live issued its action
            if lived is not None:
                data["lived/observations"], data["lived/action"] = lived[0], lived[1]
                metadata["lived"] = {
                    "forecast": float(lived[2]),
                    "sampled": bool(lived[3]),
                    "own": bool(lived[5]),
                    "recorded": None if lived[6] is None else float(lived[6]),
                }
        if self.efference is not None:
            # The copy is part of the continuation; earlier formats cannot carry it. A brain
            # without one writes the metadata of the earlier formats unchanged.
            metadata["format"] = "cadence-generic/4"
            metadata["efference"] = self.efference.to_dict()
        awaited = self._awaited()
        if awaited is not None:
            # What the stream sensed while its outcome is awaited is part of the
            # continuation, which earlier formats cannot carry; a stream that is not
            # waiting writes the metadata of the earlier formats unchanged.
            metadata["format"] = "cadence-generic/5"
            metadata["awaiting"] = {"steps": int(awaited.steps)}
            for name in ("v", "activation", "adaptation"):
                data["awaiting/" + name] = np.asarray(getattr(awaited, name))
        if agent._pending is not None:
            kind, plus, minus, value = agent._pending
            if kind != "states":
                raise ValueError("Brain checkpoints require its standard action states")
            data["pending/value"] = value
            for name, phase in (("plus", plus), ("minus", minus)):
                metadata["pending_" + name + "_steps"] = phase.steps
                for field in ("v", "activation", "adaptation"):
                    data[f"pending/{name}/{field}"] = np.asarray(getattr(phase, field))
            if self._moment is not None:
                data["moment/observations"], data["moment/action"] = self._moment
        for name in _REWARD_ARRAYS:
            value = getattr(agent, name)
            if value is not None:
                data["actor/" + name] = np.asarray(value)
        if agent._trace_device is not None:
            for name, tensor in zip(("trace", "trace_bias"), agent._trace_device, strict=True):
                data["actor/" + name] = tensor.detach().cpu().double().numpy()
        for name in ("mean", "var"):
            data["valence/" + name] = np.asarray(getattr(agent.valence, name))
        if agent.state is not None:
            for name in ("v", "activation", "adaptation"):
                data["free/" + name] = np.asarray(getattr(agent.state, name))
        if self._prepared is not None:
            data["prepared"] = self._prepared
        if self.working_memory is not None:
            for name in ("trace", "last", "cold"):
                data["working/" + name] = getattr(self.working_memory, name)
        if self.efference is not None:
            for name in ("trace", "last", "cold"):
                data["efference/" + name] = getattr(self.efference, name)
        if self.hippocampus is not None:
            for name in ("pre", "post", "strength", "mass"):
                data["episodic/" + name] = getattr(self.hippocampus, name)
            if self.hippocampus.separator is not None:
                for name in ("projection", "mean"):
                    data["episodic/separator/" + name] = getattr(self.hippocampus.separator, name)
            if isinstance(self.hippocampus, SynapticMemory):
                data["episodic/consolidated"] = self.hippocampus.consolidated
        data["generic"] = np.array(canonical_json(metadata))
        return _write(data, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Backend = "cpu",
        device: str | None = None,
        precision: str | None = None,
    ) -> Brain:
        """Resume a complete saved brain, defaulting to portable CPU inference and learning.

        Keep the saved batch row identities, or call ``reset`` for new episodes.
        Cross-backend results are subject to floating-point differences.
        """
        import json

        with np.load(path, allow_pickle=False) as data:
            if "generic" not in data:
                raise ValueError("not a Brain checkpoint; use Learner.load for a learner")
            meta = json.loads(str(data["generic"]))
            if not isinstance(meta, dict) or meta.get("format") not in (
                "cadence-generic/1",
                "cadence-generic/2",
                "cadence-generic/3",
                "cadence-generic/4",
                "cadence-generic/5",
            ):
                raise ValueError("unsupported Brain checkpoint format")
            if meta["format"] == "cadence-generic/5":
                if "arousal" not in meta:
                    raise ValueError("format cadence-generic/5 carries a live stream's arousal")
            elif meta["format"] == "cadence-generic/4":
                if meta.get("efference") is None:
                    raise ValueError("format cadence-generic/4 carries an efference copy")
            elif ("arousal" in meta) != (meta["format"] == "cadence-generic/3"):
                raise ValueError("arousal state belongs to checkpoint format cadence-generic/3")
            arousal = Arousal.from_dict(meta["arousal"]) if "arousal" in meta else None
            if "hippocampus" not in meta:
                raise ValueError("missing hippocampus metadata")
            resting_bias = _validate_resting_bias(meta.get("resting_bias", 0.0))
            learner = Learner.load(path, backend=backend, device=device, precision=precision)
            composition = _load_composition(meta.get("composition"), learner)
            try:
                _validate_life_state(meta, data, learner)
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid or incomplete saved continuation state") from exc
            memory = _load_memory(meta["hippocampus"], data, learner.brain.connectome.n)
            result = cls(
                learner.brain.connectome,
                episodic=False,
                reward=ActorCriticConfig(**meta["reward"]),
                slots=[int(k) for k in learner.slot_sizes],
            )
            result.learner = learner
            # Initialization provenance is separate from the saved, possibly learned bias.
            result.resting_bias = resting_bias
            result._composition = composition
            agent = result.basal_ganglia
            agent.learner = learner
            for name in _REWARD_ARRAYS:
                if "actor/" + name in data:
                    value = data["actor/" + name].copy()
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite saved actor state: {name}")
                    setattr(agent, name, value)
            agent.updates, agent.b_critic = int(meta["updates"]), float(meta["b_critic"])
            result.rng.bit_generator.state = meta["rng"]
            agent.rng.bit_generator.state = meta["actor_rng"]
            agent.valence.mean = data["valence/mean"].copy()
            agent.valence.var = data["valence/var"].copy()
            if meta["free_steps"] is not None:
                agent._free = BrainState(
                    data["free/v"].copy(),
                    data["free/activation"].copy(),
                    data["free/adaptation"].copy(),
                    int(meta["free_steps"]),
                )
                agent._free_brain = learner.brain if meta.get("free_current", False) else None
            result._prepared = data["prepared"].copy() if "prepared" in data else None
            result.last_learning = meta.get("last_learning", {})
            if meta.get("pending", False):
                phases = []
                for name in ("plus", "minus"):
                    phases.append(
                        BrainState(
                            data[f"pending/{name}/v"].copy(),
                            data[f"pending/{name}/activation"].copy(),
                            data[f"pending/{name}/adaptation"].copy(),
                            int(meta["pending_" + name + "_steps"]),
                        )
                    )
                agent._pending = ("states", phases[0], phases[1], data["pending/value"].copy())
                if "moment/observations" in data:
                    result._moment = (
                        data["moment/observations"].copy(),
                        data["moment/action"].copy(),
                    )
            working = meta["working_memory"]
            result.working_memory = None
            if working is not None:
                result.working_memory = Trace(
                    result.connectome,
                    **{
                        name: working[name]
                        for name in ("decay", "amplitude", "focus", "source", "target")
                    },
                )
                for name in ("trace", "last", "cold"):
                    setattr(result.working_memory, name, data["working/" + name].copy())
            echo = meta.get("efference")
            result.efference = None
            if echo is not None:
                result.efference = Efference(
                    result.connectome,
                    **{name: echo[name] for name in ("decay", "amplitude", "source", "target")},
                )
                for name in ("trace", "last", "cold"):
                    setattr(result.efference, name, data["efference/" + name].copy())
            result.hippocampus = memory
            result.arousal = arousal
            if "lived" in meta:
                state = agent._free  # validated above: a lived action has its free state
                assert state is not None
                recorded = meta["lived"].get("recorded")
                result._lived = (
                    data["lived/observations"].copy(),
                    data["lived/action"].copy(),
                    float(meta["lived"]["forecast"]),
                    bool(meta["lived"]["sampled"]),
                    state,
                    bool(meta["lived"]["own"]),
                    None if recorded is None else float(recorded),
                )
            if "awaiting" in meta:
                acted = agent._free  # validated above: the awaited action has its free state
                assert acted is not None
                result._awaiting = (
                    acted,
                    BrainState(
                        data["awaiting/v"].copy(),
                        data["awaiting/activation"].copy(),
                        data["awaiting/adaptation"].copy(),
                        int(meta["awaiting"]["steps"]),
                    ),
                )
        return result
