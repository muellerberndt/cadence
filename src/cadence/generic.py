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
* optionally a ``history`` region driven by a ``History`` of the observations as heard, at
  one or more timescales;
* optionally ``recall`` and ``next`` regions driven by a ``Recall``, the index memory of the
  stream's own recent moments, each moment addressed before its settle;
* a default hippocampal-memory analogue, ``SynapticMemory`` from sensory to motor
  neurons, recording outcomes quickly and consolidating them through repetition/salience.
  With ``memory_choice=True`` (opt-in) it is an episodic memory: its
  records of outcomes and of demonstrated actions are kept twice, keyed by the observation
  (place) and by the temporal context of what the stream saw and did (context), the two
  reads vote in the choice, and an attempt that ended in reward is replayed backwards along
  the path that reached it.

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
from typing import Any, cast

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
from .stream import Efference, FastSynapses, History, PatternSeparator, Recall, Trace

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

# The brain's sparse code, opt-in (``code=units``): each input is read through two rectified
# channels, ON max(x, 0) and OFF max(-x, 0), every channel by _CODE_COVER units at least, with
# positive weights, each unit reading _CODE_FAN_IN channels (more where that many cannot read
# every channel _CODE_COVER times); one unit in _CODE_WINNERS_SHARE is active per observation
# and at least _CODE_WINNERS_FLOOR (too sparse a code leaves the cortex and the memory too
# little overlap between a question and the moment it asks for); the memories key on the code
# as it is.
_CODE_FAN_IN = 32
_CODE_COVER = 2
_CODE_WINNERS_SHARE = 10
_CODE_WINNERS_FLOOR = 24
# The code's own stream of the brain's seed. A brain develops, acts and draws its episodic
# separators from its seed and the next one, so a code drawn from an offset of the seed would
# share its stream with a brain of a nearby seed.
_CODE_STREAM = 7103


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


def _history_decays(value: Any) -> tuple[float, ...]:
    """The decays of a heard history, one block of the region each: reals in [0, 1). The empty
    sequence is the brain without a history."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (Sequence, np.ndarray)):
        raise ValueError("history_decays must be a sequence of reals in [0, 1)")
    decays = tuple(real("history_decays", decay, low=0) for decay in value)
    if any(decay >= 1 for decay in decays):
        raise ValueError("history_decays must lie in [0, 1)")
    return decays


def _recall_genes(recall: Any, genes: dict[str, Any]) -> dict[str, Any]:
    """The recall memory's settings, validated: its founders with every ``recall_<setting>``
    of ``genes`` over them (those names are removed from ``genes``), an option only where it
    is given, or an empty mapping for a brain without it."""
    if not isinstance(recall, bool):
        raise ValueError("recall must be boolean")
    founders = Recall.founders()
    settings: dict[str, Any] = {}
    for name in [name for name in genes if name.startswith("recall_")]:
        setting = name[len("recall_"):]
        if setting not in founders and setting not in Recall.OPTIONS:
            raise TypeError(f"unknown brain setting: {name!r}")
        settings[setting] = Recall.setting(setting, genes.pop(name))
    if not recall:
        if settings:
            raise ValueError("recall settings need recall=True")
        return {}
    return {**founders, **settings}


def _image_shape(inputs: Any) -> tuple[int, int, int] | None:
    """``(height, width, channels)`` of an image shape, or None for a vector length."""
    if isinstance(inputs, (bool, np.bool_, int, np.integer)):
        return None
    if isinstance(inputs, (str, bytes, Mapping)) or not isinstance(inputs, (Sequence, np.ndarray)):
        raise ValueError("inputs is a vector length or an image shape (height, width[, channels])")
    shape = tuple(inputs)
    if len(shape) not in (2, 3):
        raise ValueError("image inputs must be (height, width) or (height, width, channels)")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1
        for value in shape
    ):
        raise ValueError("image inputs must be (height, width) or (height, width, channels)")
    height, width, *rest = (int(value) for value in shape)
    return height, width, rest[0] if rest else 1


def _count(name: str, value: Any, low: int = 0) -> int:
    """A Python or NumPy integer of at least ``low``; booleans are refused."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < low
    ):
        raise ValueError(f"{name} must be an integer of at least {low}")
    return int(value)


def _code_units(code: Any) -> int:
    """The units of the brain's sparse code: ``code`` as given, 0 or None for none."""
    return 0 if code is None else _count("code", code)


def _code_winners(units: int) -> int:
    """The active units of a code of ``units``: one in ten, at least 24. Too sparse a code
    leaves the cortex and the memory too little overlap between a question and the moment it
    asks for."""
    return max(_CODE_WINNERS_FLOOR, units // _CODE_WINNERS_SHARE)


def _code_fan_in(inputs: int, units: int) -> int:
    """The channels each unit of a code of ``units`` reads: 32, or as many as it takes for
    ``units`` units to read each of the ``2 * inputs`` channels (an ON and an OFF channel per
    input) twice."""
    return max(_CODE_FAN_IN, -(-_CODE_COVER * 2 * inputs // units))


def _code_seed(seed: int) -> int:
    """The seed of the code of a brain of ``seed``: drawn from it on a stream of its own,
    ``SeedSequence([seed, 7103])``."""
    return int(np.random.SeedSequence([seed, _CODE_STREAM]).generate_state(1)[0])


def _sparse_code(inputs: int, units: int, seed: int) -> PatternSeparator:
    """The brain's sparse code: ``units`` code neurons with positive weights reading each input
    through two rectified channels, ON ``max(x, 0)`` and OFF ``max(-x, 0)``, every channel read
    by two units at least, each unit reading 32 channels (more where 32 cannot read every
    channel twice), the readers dealt first and the rest of each fan-in drawn at random from
    the seed; the largest sums of one unit in ten kept, at least 24 (``observation_code``
    scales the code to unit length)."""
    if inputs < _CODE_FAN_IN:
        raise ValueError(
            f"a code reads observations of at least {_CODE_FAN_IN} values; the observation "
            f"has {inputs} values"
        )
    if units < _CODE_WINNERS_FLOOR:
        raise ValueError(f"a code holds at least {_CODE_WINNERS_FLOOR} units")
    return PatternSeparator(
        inputs, units, _code_winners(units), seed=seed, fan_in=_code_fan_in(inputs, units),
        cover=_CODE_COVER,
    )


def _unit_rows(code: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length; a row without a positive unit stays zero."""
    norm = np.linalg.norm(code, axis=1, keepdims=True)
    return np.asarray(code / np.where(norm > 0.0, norm, 1.0))


# The episodic memory that ``Brain.compose(..., memory_choice=True)`` builds: place and context
# records vote in the choice, remember demonstrations and replay an attempt that ended in reward.
_EPISODIC_FOUNDERS: dict[str, Any] = {
    "memory_choice": True,
    "memory_context": 0.5,
    # a known outcome moves a record halfway: remembered at once, revised by repetition
    "memory_rate": 0.5,
    "memory_replay": 8,
    "memory_demonstrations": 1.0,
    "memory_expansion": 1024,
    "memory_winners": 24,
}
_EPISODIC_SETTINGS = (
    "memory_choice", "memory_context", "memory_replay", "memory_demonstrations",
    "memory_expansion", "memory_winners", "memory_fan_in",
)


def _load_code(
    meta: dict[str, Any], data: Mapping[str, Any], sensory: int
) -> tuple[int, PatternSeparator, bool] | None:
    """The brain's sparse code a checkpoint names: the observation width, the code with its
    saved tables, and whether the sensory region is the code; None for a brain without one."""
    spec = meta.get("code")
    if spec is None:
        if any(name.startswith("code/") for name in data):
            raise ValueError("saved code tables require code metadata")
        return None
    fields = {"inputs", "expansion", "winners", "seed", "center", "fan_in", "cover", "cortex"}
    if isinstance(spec, dict) and set(spec) == fields - {"cover"}:
        # a code saved before the covering construction drew signed weights independently;
        # this version neither builds such a code nor reads one
        raise ValueError(
            "the saved code lacks the covering construction's field, cover: it was saved "
            "before the covering code and cannot be loaded by this version"
        )
    if not isinstance(spec, dict) or set(spec) != fields or not isinstance(spec["cortex"], bool):
        raise ValueError("invalid saved code metadata")
    for name in ("inputs", "expansion", "winners", "seed", "fan_in", "cover"):
        value = spec[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"saved code {name} must be a nonnegative integer")
    if (
        spec["center"] != 0
        or spec["cover"] != _CODE_COVER
        or not spec["expansion"]
        or spec["fan_in"] != _code_fan_in(spec["inputs"], spec["expansion"])
        or spec["winners"] != _code_winners(spec["expansion"])
    ):
        raise ValueError(
            f"a saved code reads each input through an ON and an OFF channel, {_CODE_FAN_IN} "
            f"channels per unit, and keeps one unit in {_CODE_WINNERS_SHARE} (at least "
            f"{_CODE_WINNERS_FLOOR}), uncentred, every channel read by {_CODE_COVER} units "
            f"(more channels per unit where {_CODE_FAN_IN} cannot read each that often)"
        )
    cortex = spec["cortex"]
    try:
        code = _sparse_code(spec["inputs"], spec["expansion"], spec["seed"])
    except ValueError as exc:
        raise ValueError("invalid saved code metadata") from exc
    if cortex and code.expansion != sensory:
        raise ValueError(
            f"the sensory region of {sensory} neurons is not the saved code of "
            f"{code.expansion} units"
        )
    if not cortex and code.inputs != sensory:
        raise ValueError(
            f"the saved code reads observations of {code.inputs} values where the sensory "
            f"region has {sensory} neurons"
        )
    for name in ("index", "weight"):
        if "code/" + name not in data:
            raise ValueError(f"missing saved code table: {name}")
    index, weight = data["code/index"], data["code/weight"]
    shape = (code.expansion, code.fan_in)
    if (
        index.shape != shape
        or weight.shape != shape
        or index.dtype.kind not in "iu"
        or weight.dtype.kind not in "iuf"
        or not np.isfinite(weight).all()
        or (index < 0).any()
        or (index >= code.channels).any()
        or any(len(np.unique(row)) != len(row) for row in index)
    ):
        raise ValueError(
            "a saved code reads distinct channels of the observation, an ON and an OFF "
            "channel per input"
        )
    if (
        (weight <= 0).any()
        or np.bincount(index.ravel(), minlength=code.channels).min() < code.cover
    ):
        raise ValueError(
            "a saved code reads every channel of the observation, the ON and the OFF channel "
            f"of each input, with {code.cover} units at least, its weights positive"
        )
    # the saved tables, not ones drawn anew: the code is part of every key
    code.index, code.weight = index.astype(np.int64), weight.astype(float)
    return code.inputs, code, cortex


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
    if isinstance(value, dict) and set(value) == expected:
        # saved before 0.82.0, whose composition read the trace at the fixed scale 12
        value = {**value, "working_memory_scale": 12.0}
    if not isinstance(value, dict) or set(value) - {
        "history_scale", "recall_scale", "image", "code"
    } != (expected | {"working_memory_scale"}):
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
    real("saved working_memory_scale", value["working_memory_scale"], low=0)
    real("saved lateral", value["lateral"])
    populations = learner.brain.connectome.populations
    if ("history_scale" in value) != ("history" in populations):
        raise ValueError("a composed history's scale is saved with its population, and only then")
    if "history_scale" in value:
        real("saved history_scale", value["history_scale"], low=0)
    if ("recall_scale" in value) != ("recall" in populations):
        raise ValueError("a composed recall's scale is saved with its population, and only then")
    if "recall_scale" in value:
        real("saved recall_scale", value["recall_scale"], low=0)
    sensory = "sensory" if "sensory" in populations else "visual/input"
    width = len(populations.get(sensory, ()))
    if "image" in value:
        image = value["image"]
        if (
            not isinstance(image, list)
            or len(image) not in (2, 3)
            or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in image)
            or int(np.prod(image)) != value["inputs"]
            or sensory != "visual/input"
        ):
            raise ValueError("invalid saved composition image")
    if "code" in value:
        # the sensory region is the code of the observations, of that many units
        units = value["code"]
        if (
            isinstance(units, bool)
            or not isinstance(units, int)
            or units != width
            or sensory != "sensory"
            or "image" in value
        ):
            raise ValueError("invalid saved composition code")
    modules = _region_widths(populations, "module_") + [len(populations.get("association", ()))]
    observers = _region_widths(populations, "observer_")
    if (
        ("code" not in value and value["inputs"] != width)
        or value["actions"] != len(learner.output_index)
        or value["modules"] != modules
        or value["observers"] != observers
        or value["slots"] != list(learner.slot_sizes)
    ):
        raise ValueError("saved composition provenance disagrees with the actual layout")
    return dict(plain(value))


def _load_memory(
    metadata: Any,
    data: Mapping[str, Any],
    neurons: int,
    prefix: str = "episodic/",
    keys: int | None = None,
) -> FastSynapses | None:
    """Validate and restore a detached memory before constructing the resumed composition.

    The memory's outputs (``post``) are neurons of the graph, below ``neurons``. Its key
    coordinates (``pre``) are neurons too, unless ``keys`` names the width of the code they
    number from zero (the observation code, with the motor neurons after it for context
    records), which can exceed the graph."""

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
        if np.any(pre < 0) or np.any(pre >= (neurons if keys is None else keys)):
            raise ValueError(
                "saved memory keys exceed the connectome"
                if keys is None
                else f"saved memory keys exceed their {keys} coordinates"
            )
        if np.any(post < 0) or np.any(post >= neurons):
            raise ValueError("saved memory outputs exceed the connectome")
        separator = None
        spec = metadata.get("separator")  # Older unseparated checkpoints omitted this field.
        if spec is None:
            if any(key.startswith(prefix + "separator/") for key in data):
                raise ValueError("saved separator arrays require separator metadata")
        else:
            if not isinstance(spec, dict):
                raise ValueError("invalid separator metadata")
            # a sparse separator names its fan-in; the dense one's metadata is as it was
            if set(spec) - {"fan_in"} != {"inputs", "expansion", "winners", "seed", "center"}:
                raise ValueError("incomplete or unsupported separator metadata")
            for name in ("inputs", "expansion", "winners", "seed", "fan_in"):
                value = spec.get(name, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"saved separator {name} must be a nonnegative integer")
            if spec["inputs"] != len(pre):
                raise ValueError("saved separator inputs must match memory pre neurons")
            # Do not regenerate these arrays from the seed: projections can be customized,
            # and the running center of FastSynapses can have changed after observations.
            mean = array("separator/mean", (spec["inputs"],))
            separator = PatternSeparator(**spec)
            if separator.fan_in:
                shape = (spec["expansion"], spec["fan_in"])
                index, weight = array("separator/index", shape), array("separator/weight", shape)
                if (
                    index.dtype.kind not in "iu"
                    or (index < 0).any()
                    or (index >= spec["inputs"]).any()
                    or any(len(np.unique(row)) != len(row) for row in index)
                ):
                    raise ValueError("a sparse separator reads distinct inputs of its key")
                separator.index, separator.weight = index.astype(np.int64), weight
            else:
                projection = array("separator/projection", (spec["inputs"], spec["expansion"]))
                separator.projection = projection
            separator.mean = mean
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


def _validate_life_state(
    meta: dict[str, Any],
    data: Mapping[str, Any],
    learner: Learner,
    code: tuple[int, PatternSeparator, bool] | None,
) -> None:
    """Reject corrupt continuation state before exposing a resumed agent. ``code`` is the
    brain's sparse code the checkpoint names, validated: the observations are saved at their
    width, and the memories' keys have the code's."""
    n = learner.brain.connectome.n
    populations = learner.brain.connectome.populations
    association = len(populations["association"])
    sensory = len(populations.get("sensory", populations.get("visual/input", ())))
    observed = sensory if code is None else code[0]
    code_width = sensory if code is None else code[1].expansion

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
        array("prepared", (batch, observed))
    if meta.get("pending", False):
        array("pending/value", (batch,))
        for phase in ("plus", "minus"):
            count("pending_" + phase + "_steps")
            for name in ("v", "activation", "adaptation"):
                array(f"pending/{phase}/{name}", (batch, n))
    if "moment/observations" in data or "moment/action" in data:
        if not meta.get("pending", False):
            raise ValueError("saved action observations need pending credit")
        array("moment/observations", (batch, observed))
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
        observations = array("lived/observations", (1, observed))
        slotted = learner.slot_count > 1
        action = array("lived/action", (1, learner.slot_count) if slotted else (1,))
        limit = learner.slot_sizes[None, :] if slotted else len(learner.output_index)
        if action.dtype.kind not in "iu" or (action < 0).any() or (action >= limit).any():
            raise ValueError("invalid saved lived action indices")
        if was_sampled:
            if (
                not np.array_equal(observations, array("moment/observations", (1, observed)))
                or not np.array_equal(action, array("moment/action", action.shape))
                or float(forecast) != float(array("pending/value", (1,))[0])
            ):
                raise ValueError("saved lived action must match its pending feedback")
    awaiting = meta.get("awaiting")
    if meta.get("format") == "cadence-generic/5" and "awaiting" not in meta:
        raise ValueError("format cadence-generic/5 carries a sensed state awaiting an outcome")
    if "awaiting" in meta and meta.get("format") not in (
        "cadence-generic/5", "cadence-generic/6", "cadence-generic/7", "cadence-generic/8",
        "cadence-generic/9",
    ):
        raise ValueError(
            "a sensed state awaiting an outcome belongs to formats cadence-generic/5 to /9"
        )
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
        if meta.get("format") not in (
            "cadence-generic/4",
            "cadence-generic/5",
            "cadence-generic/6",
            "cadence-generic/7",
            "cadence-generic/8",
            "cadence-generic/9",
        ):
            raise ValueError(
                "an efference copy belongs to checkpoint formats cadence-generic/4 to /9"
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
    heard = meta.get("history")
    if heard is None and (
        "history" in populations or any(name.startswith("history/") for name in data)
    ):
        raise ValueError("saved history neurons and state require history metadata")
    if heard is not None:
        if (
            not isinstance(heard, dict)
            or set(heard) != {"decays", "amplitude", "source", "target", "neurons"}
            or not isinstance(heard["decays"], list)
            or heard["source"] not in ("sensory", "visual/input")
            or heard["source"] not in populations
            or heard["target"] != "history"
            or "history" not in populations
        ):
            raise ValueError("invalid saved history ports")
        decays = _history_decays(heard["decays"])
        real("saved history_amplitude", heard["amplitude"], low=0)
        width = len(populations["history"])
        if not decays or width != len(decays) * sensory or heard["neurons"] != width:
            raise ValueError("a saved history needs one block of sensory width per decay")
        array("history/trace", (batch, width))
    _validate_recall_state(meta, data, learner, batch, sensory, code_width)
    _validate_episodic_state(meta, data, learner, code_width)


def _validate_recall_state(
    meta: dict[str, Any],
    data: Mapping[str, Any],
    learner: Learner,
    batch: int,
    sensory: int,
    width: int,
) -> None:
    """The settings, slots and address map of a recall memory (formats /8 and /9); ``width``
    is the observation code's, the observations the memory is given."""
    populations = learner.brain.connectome.populations
    recalled = meta.get("recall")
    if recalled is None:
        if any(name.startswith("recall/") for name in data):
            raise ValueError("saved recall state requires recall metadata")
        return
    names = {*Recall.RUNTIME, *Recall.CONSTRUCTION}
    if (
        not isinstance(recalled, dict)
        or set(recalled) - {"inputs", *Recall.OPTIONS}
        != names | {"source", "target", "neurons", "writes"}
        or recalled["source"] not in ("sensory", "visual/input")
        or recalled["source"] not in populations
        or recalled["target"] != "recall"
        or "recall" not in populations
    ):
        raise ValueError("invalid saved recall ports")
    settings = {name: Recall.setting(name, recalled[name]) for name in names}
    # an option is saved where it differs from its founder, and takes its founder where absent
    settings.update({name: Recall.setting(name, recalled[name]) for name in Recall.OPTIONS
                     if name in recalled})
    writes = recalled["writes"]
    if isinstance(writes, bool) or not isinstance(writes, int) or writes < 0:
        raise ValueError("invalid saved recall write count")
    inputs = recalled.get("inputs", sensory)
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs != width:
        raise ValueError(
            f"a saved recall memory keys on observations of {inputs} values where the "
            f"observation code has {width}"
        )
    association = len(populations["association"])
    key = width
    value = key if settings["value"] == "event" else association
    regions = ("recall", "next") if settings["next"] else ("recall",)
    if (
        any(len(populations.get(name, ())) != value for name in regions)
        or recalled["neurons"] != value * len(regions)
    ):
        raise ValueError("a saved recall memory needs recall and next regions of its value width")

    def array(name: str, shape: tuple[int, ...]) -> np.ndarray:
        stored: np.ndarray = data.get("recall/" + name, None)
        if (
            stored is None
            or stored.dtype.kind not in "biuf"
            or not np.isfinite(stored).all()
            or stored.shape != shape
        ):
            raise ValueError(f"invalid saved recall state: {name}")
        return stored

    keys = data.get("recall/keys")
    if keys is None or keys.ndim != 3:
        raise ValueError("invalid saved recall state: keys")
    slots = keys.shape[1]
    if keys.shape[0] != batch or slots > settings["capacity"]:
        raise ValueError("saved recall slots must belong to the saved streams, within capacity")
    array("keys", (batch, slots, key))
    for name in ("values", "nexts"):
        array(name, (batch, slots, value))
    if array("held", (batch, slots)).dtype != np.bool_:
        raise ValueError("invalid saved recall state: held")
    array("previous", (batch, association))
    history = data.get("recall/history")
    array("history", (0 if history is None or not len(history) else batch, key))
    array("w_x", (key, key))
    array("w_c", (association, key))
    array("b", (key,))
    # the actor's eligibility of the address map, and the contrast of a pending action
    eligibility = (batch, key + association + 1, key)
    if "recall/trace" in data:
        array("trace", eligibility)
    if "recall/pending" in data:
        if not meta.get("pending", False):
            raise ValueError("a saved address contrast belongs to a pending action")
        array("pending", eligibility)
    expected = {"keys", "values", "nexts", "held", "previous", "history", "w_x", "w_c", "b"}
    if {name[len("recall/"):] for name in data if name.startswith("recall/")} - {
        "trace", "pending"
    } != expected:
        raise ValueError("unexpected saved recall arrays")


def _validate_episodic_state(
    meta: dict[str, Any], data: Mapping[str, Any], learner: Learner, sensory: int
) -> None:
    """The settings and stream state of an episodic memory's choice (formats /6 to /9);
    ``sensory`` is the width of the observation code the records are keyed on."""
    names = ("episode/context", "episode/decision") + tuple(
        "episode/path/" + name
        for name in (
            "keys", "actions", "observations", "lengths", "rewards", "received", "guess"
        )
    )
    episodic = meta.get("episodic")
    if episodic is None:
        if any(name in data for name in names) or any(n.startswith("temporal/") for n in data):
            raise ValueError("saved episodic stream state requires episodic metadata")
        return
    if (
        not isinstance(episodic, dict)
        or set(episodic) != {"choice", "context", "replay", "demonstrations", "context_memory"}
        or episodic["choice"] is not True
    ):
        raise ValueError("invalid episodic metadata")
    context = real("saved memory_context", episodic["context"], low=0)
    if context >= 1:
        raise ValueError("invalid saved memory_context")
    replay = _count("saved memory_replay", episodic["replay"])
    real("saved memory_demonstrations", episodic["demonstrations"], low=0)
    memory = meta.get("hippocampus")
    if not isinstance(memory, dict) or memory.get("pre") != sensory:
        raise ValueError(
            "an episodic memory's place records are keyed by the observation code: "
            f"{sensory} coordinates, where the saved records have "
            f"{None if not isinstance(memory, dict) else memory.get('pre')}"
        )
    width = sensory + len(learner.output_index)
    temporal = episodic["context_memory"]
    if (temporal is None) != (context == 0):
        raise ValueError("a temporal context keys a second store of records, and only then")
    if temporal is not None and (not isinstance(temporal, dict) or temporal.get("pre") != width):
        raise ValueError(
            "an episodic memory's context records are keyed by the observation code and the "
            f"motor neurons: {width} coordinates"
        )
    if temporal is None and any(name in data for name in names[:2] + (names[2],)):
        raise ValueError("saved temporal context state requires context records")

    def array(name: str) -> np.ndarray:
        value: np.ndarray = data[name]
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"invalid saved state: {name}")
        return value

    for name in ("episode/context", "episode/decision"):
        if name in data:
            value = array(name)
            if value.ndim != 2 or not len(value) or value.shape[1] != width:
                raise ValueError(f"invalid saved state: {name}")
    path = [name for name in names[3:] if name in data]
    if path and len(path) != 6:
        raise ValueError("incomplete saved replay paths")
    if path and (temporal is not None) != ("episode/path/keys" in data):
        raise ValueError("saved replay paths carry context keys exactly with context records")
    if path:
        lengths = array("episode/path/lengths")
        actions = array("episode/path/actions")
        observations = array("episode/path/observations")
        streams = len(lengths)
        if (
            lengths.ndim != 1
            or lengths.dtype.kind not in "iu"
            or not streams
            or (lengths < 0).any()
            or (lengths > replay + 1).any()
        ):
            raise ValueError("invalid saved replay path lengths")
        longest = int(lengths.max())
        slotted = learner.slot_count > 1
        shape = (streams, longest, learner.slot_count) if slotted else (streams, longest)
        if (
            (
                temporal is not None
                and array("episode/path/keys").shape != (streams, longest, width)
            )
            or observations.shape != (streams, longest, sensory)
            or actions.shape != shape
            or actions.dtype.kind not in "iu"
        ):
            raise ValueError("invalid saved replay path dimensions")
        limit = learner.slot_sizes[None, None, :] if slotted else len(learner.output_index)
        if (actions < 0).any() or (actions >= limit).any():
            raise ValueError("invalid saved replay path actions")
        rewards, received = array("episode/path/rewards"), array("episode/path/received")
        guess = array("episode/path/guess")
        if (
            rewards.shape != (streams, longest)
            or received.shape != (streams, longest)
            or received.dtype != np.bool_
            or guess.shape != (streams, longest)
            or not np.isfinite(rewards).all()
            or not np.isfinite(guess).all()
            or (guess < 0).any()
            or (guess > 1).any()
        ):
            raise ValueError("invalid saved replay path outcomes")


class Brain:
    """Senses, an association cortex, a motor cortex, basal ganglia and consolidating memory.

    ``connectome`` needs populations ``sensory`` (or ``visual/input`` for an image),
    ``association`` and ``motor``, ``prefrontal`` for a working memory, ``efference``
    (one neuron per motor neuron) for a copy of the issued command, ``history`` (one
    block of sensory width per entry of ``history_decays``) for the observations as heard
    and, with ``recall=True``, ``recall`` and ``next`` (one neuron per value coordinate each)
    for the index memory of the stream's moments; ``genome`` builds one. Without
    ``recall=True`` populations of those names are ordinary ones. Use ``Brain.compose`` for
    the default brain with working memory.

    The memories key on the observation code, ``observation_code``: the observation itself,
    or with ``code=units`` the brain's own sparse code of it, ``units`` code neurons with
    positive weights that read each input through two rectified channels, ON ``max(x, 0)``
    and OFF ``max(-x, 0)``, and every channel twice at least (each channel is dealt two
    readers first, and each unit's 32 channels, more where 32 cannot read every channel
    twice, are filled at random from the seed after), the largest sums of one unit in ten
    kept (at least 24: too sparse a code leaves the cortex and the memory too little overlap
    between a question and the moment it asks for) and the code scaled to unit length. A
    signed observation reaches the code in both signs; a nonnegative one leaves the OFF
    channels silent. The code's seed is drawn from ``seed`` on a stream of its own
    (``SeedSequence([seed, 7103])``). Left unset, the memories key on the observation
    whatever its width; a code of 1,024 units is the recommended setting above 1,024 inputs,
    with the episodic memory's ``memory_fan_in`` 32 above 256 uncoded key values. With
    ``inputs`` (the width of the observations) the sensory region is the code: every
    observation is coded before the drive, and ``inputs`` must then be named beside a
    ``code`` of the region's width.
    ``sparse_code`` holds the ``PatternSeparator``, ``code_width`` the key width
    (``observation_width`` gives it before the brain is built, to size the ``recall`` and
    ``next`` regions of a custom anatomy). A memory keyed on the code numbers its key
    coordinates from zero (``pre``) and is never read from the drive; the code is its
    separation, so the episodic stores take no separator of their own.
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
        memory_choice: bool = False,
        memory_context: float = 0.0,
        memory_replay: int = 0,
        memory_demonstrations: float = 0.0,
        memory_expansion: int = 0,
        memory_winners: int = 32,
        memory_fan_in: int | None = None,
        working_memory_decay: float = 0.0,
        working_memory_amplitude: float = 3.0,
        working_memory_focus: float = 0.0,
        efference_decay: float = 0.2,
        efference_amplitude: float = 0.0,
        history_decays: Sequence[float] = (),
        history_amplitude: float = 3.0,
        recall: bool = False,
        learning: LearnerConfig | None = None,
        reward: ActorCriticConfig | None = None,
        resting_bias: float = 0.0,
        slots: int | Sequence[int] = 1,
        arousal: ArousalConfig | Mapping[str, Any] | bool | None = None,
        seed: int = 0,
        backend: Backend = "cpu",
        device: str | None = None,
        inputs: int | None = None,
        code: int | None = None,
        **genes: Any,
    ) -> None:
        populations = connectome.populations
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
        # the observation code the memories key on: the observation itself, or the brain's
        # sparse code of it, which with ``inputs`` the sensory region is
        if inputs is not None:
            inputs = _count("inputs", inputs, low=1)
        self.inputs = len(self.sensory_index) if inputs is None else inputs
        self.coded_cortex = inputs is not None
        units = _code_units(code)
        if self.coded_cortex and units != len(self.sensory_index):
            raise ValueError(
                f"with inputs the sensory region is the code: {len(self.sensory_index)} "
                f"neurons for a code of {units} units"
            )
        self.sparse_code: PatternSeparator | None = (
            _sparse_code(self.inputs, units, _code_seed(seed)) if units else None
        )
        self.code_width = units or self.inputs
        recalled = _recall_genes(recall, genes)  # the recall_* genes leave ``genes`` here
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
        if not isinstance(memory_choice, bool):
            raise ValueError("memory_choice must be boolean")
        memory_context = real("memory_context", memory_context, low=0)
        if memory_context >= 1:
            raise ValueError("memory_context must lie in [0, 1)")
        memory_replay = _count("memory_replay", memory_replay)
        memory_demonstrations = real("memory_demonstrations", memory_demonstrations, low=0)
        memory_expansion = _count("memory_expansion", memory_expansion)
        memory_winners = _count("memory_winners", memory_winners, low=1)
        memory_fan_in = _count("memory_fan_in", 0 if memory_fan_in is None else memory_fan_in)
        if self.sparse_code is not None and (memory_expansion or memory_fan_in):
            # a coded brain's memories key on the code as it is: the code is their separation
            raise ValueError(
                "a coded brain's memories key on its code as it is; memory_expansion and "
                "memory_fan_in need a brain without a code"
            )
        separated = bool(memory_expansion)
        if memory_choice and not episodic:
            raise ValueError("memory_choice needs the episodic memory")
        if not memory_choice and (
            memory_context or memory_replay or memory_demonstrations or memory_expansion
        ):
            raise ValueError(
                "memory_context, memory_replay, memory_demonstrations and memory_expansion "
                "need memory_choice"
            )
        if memory_fan_in and not memory_expansion:
            raise ValueError("memory_fan_in needs memory_expansion")
        working_memory_decay = real("working_memory_decay", working_memory_decay, low=0)
        efference_decay = real("efference_decay", efference_decay, low=0)
        if working_memory_decay >= 1 or efference_decay >= 1:
            raise ValueError("working_memory_decay and efference_decay must lie in [0, 1)")
        working_memory_amplitude = real("working_memory_amplitude", working_memory_amplitude)
        working_memory_focus = real("working_memory_focus", working_memory_focus, low=0)
        resting_bias = _validate_resting_bias(resting_bias)
        efference_amplitude = _validate_efference_amplitude(efference_amplitude)
        history_decays = _history_decays(history_decays)
        history_amplitude = real("history_amplitude", history_amplitude, low=0)
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
                boundary = head in (
                    "sensory", "visual", "prefrontal", "motor", "efference", "history"
                ) or (bool(recalled) and head in ("recall", "next"))
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
        self.history: History | None = None
        if "history" in populations:
            if not history_decays or len(populations["history"]) != len(history_decays) * len(
                self.sensory_index
            ):
                raise ValueError(
                    "a history population needs one block of sensory width per history decay"
                )
            self.history = History(
                connectome,
                decays=history_decays,
                amplitude=history_amplitude,
                source=sensory,
                target="history",
            )
        elif history_decays:
            raise ValueError("history_decays need a history population; Brain.compose adds it")
        # The index memory of the stream's moments. Populations named ``recall`` and ``next``
        # without ``recall=True`` are ordinary ones: an instrument may name and drive its own.
        self.recall: Recall | None = None
        if recalled:
            for name in ("recall", "next") if recalled["next"] else ("recall",):
                if name not in populations:
                    raise ValueError(
                        f"recall=True needs a {name!r} population; Brain.compose adds it"
                    )
            # a memory on a code of the observation is given the code, not the observation;
            # with a coded cortex the sensory range is the code
            coded_memory = self.sparse_code is not None and not self.coded_cortex
            self.recall = Recall(
                connectome, **recalled, source=sensory,
                inputs=self.code_width if coded_memory else 0,
            )
            # with ``recall_address_reward`` the actor keeps the address map's eligibility
            self.basal_ganglia.memory = self.recall
        self.hippocampus: FastSynapses | None = None
        # ``memory_choice`` with a temporal context: the same records keyed by the stream's
        # temporal context, the fading activity of the sensory and the motor neurons
        self.context_memory: SynapticMemory | None = None
        if memory_fan_in > self.code_width:
            raise ValueError(
                f"memory_fan_in must not exceed the {self.code_width} coordinates of the "
                "place key"
            )
        if episodic:

            def store(keys: np.ndarray, offset: int) -> SynapticMemory:
                return SynapticMemory(
                    keys, self.motor_index, consolidation=consolidation,
                    decay=memory_decay, rate=memory_rate, amplitude=memory_amplitude,
                    separator=PatternSeparator(
                        len(keys), memory_expansion, memory_winners, seed=seed + offset,
                        fan_in=memory_fan_in,
                    )
                    if separated
                    else None,
                )

            # keyed on the sensory neurons, or on the coordinates of the observation code
            coded = self.sparse_code is not None
            self.hippocampus = store(np.arange(self.code_width) if coded else self.sensory_index, 0)
            if memory_choice and memory_context:
                self.context_memory = store(
                    np.arange(self.code_width + len(self.motor_index))
                    if coded
                    else np.concatenate([self.sensory_index, self.motor_index]),
                    1,
                )
        self.memory_choice = memory_choice
        self.memory_context = memory_context
        self.memory_replay = memory_replay
        self.memory_demonstrations = memory_demonstrations
        # ``memory_choice``, per stream: the temporal context after its last decision, that
        # decision's key, and the recent path, one decision per situation it left: (key,
        # action, observation, outcome, outcome received, how far it was the best guess)
        self._context: np.ndarray | None = None
        self._decision: np.ndarray | None = None
        self._paths: list[
            list[tuple[np.ndarray | None, np.ndarray, np.ndarray, float, bool, float]]
        ] = []
        self.rng = np.random.default_rng(seed)
        # the validated observations a settle is being prepared for and their code, computed
        # once per moment: ``stimulus`` given that same array reads it, ``_choose`` takes it
        self._coded_moment: tuple[np.ndarray, np.ndarray] | None = None
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
    def observation_width(inputs: int | Sequence[int], code: int | None = None) -> int:
        """The width of the observation code of a brain with these ``inputs`` (a vector
        length or an image shape) and this ``code`` (as the constructor takes it), before
        the brain is built: with ``recall=True`` and event values a custom anatomy's
        ``recall`` and ``next`` regions hold that many neurons (the association width with
        ``recall_value="cortex"``)."""
        image = _image_shape(inputs)
        width = int(np.prod(image)) if image is not None else _count("inputs", inputs, low=1)
        return _code_units(code) or width

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
        image = _image_shape(inputs)
        if image is None:
            sensory = Region("sensory", cast(int, inputs))  # a vector length; Region checks it
            forward = Projection("sensory", "association", density=density, reciprocal=False)
        else:
            height, width, channels = image
            sensory = visual_cortex(
                height, width, channels=channels, features=features, field=field, seed=seed
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
        """Develop the default genome and wrap it; ``options`` go to the constructor. The
        memories key on the observation itself (an image's pixels), or with
        ``options["code"]`` on the brain's sparse code of it."""
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
        inputs: int | Sequence[int],
        actions: int,
        *,
        modules: Sequence[int] = (64,),
        observers: Sequence[int] = (),
        lateral: float | None = None,
        sensory_scale: float = 1.0,
        working_memory_scale: float = 1.2,
        history_scale: float = 1.2,
        recall_scale: float = 1.2,
        features: int = 8,
        field: int = 3,
        code: int | None = None,
        seed: int = 0,
        slots: int | Sequence[int] = 1,
        **options: Any,
    ) -> Brain:
        """Compose a modular brain with working memory and consolidation.

        ``inputs`` is a vector length, or an image shape ``(height, width)`` or ``(height,
        width, channels)`` as in ``genome``: the sensory region is then a ``visual_cortex``
        of the image (``features`` maps of ``field`` by ``field`` receptive fields), whose
        feature maps project to the first module as the sensory region does.
        ``compose(inputs=int)`` is as before. The memories key on the observation code
        (``observation_code``): the observation itself, or with ``code=units`` the brain's
        sparse code of it (each input read through an ON and an OFF channel, every channel by
        two units at least, with positive weights, 32 channels per unit or more, one unit in
        ten active, unit length).
        ``code=units`` makes a vector brain's sensory region that code: every observation is
        coded before the drive, so 12,824 inputs reach the cortex as 1,024 units through
        1,024 x 64 sensory synapses in place of 12,824 x 64, and the memories key on the same
        code; the founder leaves the cortex and the memories on the observation. A code of
        1,024 units is the recommended setting above 1,024 inputs. For an image brain
        ``code`` names the memories' code alone, the cortex reading the pixels through the
        visual cortex.

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
        ``working_memory_scale`` sets the initial magnitude of the plastic projection
        through which the prefrontal region feeds the carried interpretation back to the
        association region. The prefrontal region holds the copy at
        ``working_memory_amplitude`` (3.0); the founder scale 1.2 starts its influence on
        the next settle where amplitude 0.3 at the earlier fixed scale 12 did, and
        learning strengthens it where the past predicts the outcome. Releases before
        0.82.0 used 12; pass it to rebuild their composition. It is a construction
        setting, like ``sensory_scale``.
        ``options`` accepts the constructor's memory settings and every existing
        configuration field as ``learning_<field>``, ``actor_<field>`` or
        ``arousal_<field>``. ``temperature`` aliases ``learning_temperature``.
        ``arousal=True`` enables the founder arousal config; a mapping patches
        those founders. Supplied config objects remain full replacements.
        ``resting_bias`` initializes processing-region biases to a selected nonnegative
        value; it does not guarantee responsive activity or successful acquisition.
        A positive ``efference_amplitude`` adds the efference copy: one ``efference``
        neuron per motor neuron, driven by the fading one-hot of the command the
        stream issued (``efference_decay``) and read by the association region
        through a plastic projection of founder scale 12. The founder
        value zero builds the brain without it, byte-identical to the released
        composition; the copy is a gene to select against that control.
        A nonempty ``history_decays`` adds the heard history: a ``history`` region of one
        block of ``inputs`` neurons per decay, each block a fading trace of the observations
        as heard (the values the sensory neurons were stimulated with, before settling) at
        its own decay, held at ``history_amplitude`` (3.0) and read by the association
        region through a plastic projection of initial magnitude ``history_scale`` (1.2),
        initialized like the working trace's. After each act the blocks take in the
        observation the brain acted on, so at a decision the history holds the past and the
        sensory drive the present. The founder ``history_decays=()`` builds the brain
        without it, byte-identical to the composition before the option; the history is a
        gene to select against that control. ``history_scale`` is a construction setting.
        ``recall=True`` adds the recall memory, an index of the stream's own recent moments:
        a ``recall`` and a ``next`` region (one neuron per value coordinate each), read by
        the association region through plastic projections of initial magnitude
        ``recall_scale`` (1.2), appended after any history. After each act, and after each
        moment ``wait`` senses, the moment is written as one slot per stream (``Recall``):
        its key the event heard (``recall_heard`` 1.0 times the observation code, the
        observation itself or the brain's sparse code of it, plus ``recall_context`` (0.0)
        times a fading history of the earlier events, kept at ``recall_context_decay``
        (0.5) per moment), its value the event (``recall_value="event"``) or the settled
        association activity (``"cortex"``), and the slot before learns it as its next.
        Before each settle the moment's address, the key plus a learned map of the event and
        of the association activity of the moment before, weighs the slots by a softmax
        (``recall_temperature`` 0.25, ``recall_recency`` 0.25 per moment of age), and the
        recalled value and next drive the two regions at ``recall_amplitude`` (3.0), a drive
        held through the settle, which answers on it and does not revise the choice. After
        each teacher lesson the address lesson scores each slot's value and next against the
        association's movement under the nudge, read back through the recall and next
        projections, weighs the scores with the address's own attention
        (``recall_address_temperature`` 0.1), and steps the map toward the weighted keys
        (``recall_address`` 0.01, each row's rate bounded by its own input energy).
        ``recall_capacity`` (64) bounds the slots per stream. Options beside the founder
        memory, each a runtime gene described and saved only where it differs from its
        founder: ``recall_address_step="toward"`` steps the address by the named keys less
        the mixture it attended to, in place of the founder's ``"lms"`` step toward the named
        keys; ``recall_recency=None`` charges a slot's age ``r * log(1 + age)``, ``r =
        recall_heard^2 / recall_temperature / log(1 + recall_capacity)``, so an unmatched
        old slot's weight falls as a power of its age; ``recall_write="aroused"`` writes a
        moment only when its act was sampled (and a moment ``wait`` senses while a sampled
        action awaits its outcome); a number ``recall_null`` adds a null slot of logit
        ``recall_null / recall_temperature`` that reads nothing; and
        ``recall_address_reward`` above zero lets reward train the address map, each sampled
        act keeping the step its lesson would take as the map's eligibility and each outcome
        stepping the map by that rate times the dopamine times the trace: the eligibility
        carries the lesson's rate, so at 1.0 a unit of dopamine moves the map by one lesson's
        step, and ``recall_address=0`` leaves the map as it is under reward too.
        The founder ``recall=False`` builds the brain without it, byte-identical to the
        composition before the option; the memory is a gene to select against that control.
        ``recall_scale``, ``recall_capacity``, ``recall_value`` and ``recall_next`` are
        construction settings; the keys, the event values and the address map follow the
        observation code's width, a brain's ``code`` among the ways to bound it.
        ``memory_choice=True`` makes the hippocampus an episodic memory: its records vote
        in every choice, greedy or sampled, added to the motor readout before the softmax.
        Each record is kept twice: keyed by the observation (place), and keyed by the
        stream's temporal context, the observation with the fading trace of what the stream
        saw and did before it (``memory_context``, the share the past keeps per moment, 0.5;
        zero keeps the place records alone). The vote is the mean of the two reads: the
        place records answer a known situation reached along a new path, the context records
        tell apart situations that look the same. It records the outcome of the chosen action
        when the outcome is known: a reward or a punishment, the end of an attempt
        (``done``), or a surprise that woke a calm ``live`` brain; a zero outcome in the
        middle of an attempt leaves the record as it was. A demonstrated action is recorded
        at ``memory_demonstrations`` (1.0) for the situation it was shown in. A write corrects
        the stream's record halfway at once (``memory_rate`` 0.5), a trace that fades by
        ``memory_decay`` (0.9) per later write, and consolidates into the record every stream
        shares and resets keep at ``consolidation`` (0.05). A reward is replayed backwards
        along the stream's recent path (``memory_replay`` steps, 8; one decision per
        situation, the one that left it), through the decisions whose own outcome was
        nothing, back to the last known outcome, discounted per step and up to the discounted
        reward. A reward that ends an attempt (``done``) is replayed whole into both kinds of
        record. In a continuing life a reward is replayed in proportion to its surprise, the
        reward less what the memory expected of the paid decision: into each decision's
        context record, and into its place record as far as it was an exploration rather than
        the brain's best guess. Nothing has to mark the end of an attempt. Keys are separated
        (``memory_expansion`` 1,024 cells, ``memory_winners`` 24; a coded brain keys on its
        code as it is, with no separation). ``memory_fan_in`` above zero makes the separation
        sparse: each of its cells reads that many of the key's values, chosen at random, so
        its size follows the cells and the fan-in alone; the founder 0 keeps it dense, and 32
        is the recommended setting above 256 key values. Each setting can be passed
        beside ``memory_choice=True``. The founder ``memory_choice=False`` keeps the
        hippocampus of earlier releases, keyed by the senses and read into the motor drive;
        ``episodic=False`` builds none. On fresh seeds the episodic memory scored below the
        earlier hippocampus on a continuing bandit whose history does not matter.
        Use ``genome``/``Genome`` for custom ports, sparsity and named wiring.
        """

        def size(value: Any) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("population sizes must be positive integers")
            if value < 1:
                raise ValueError("population sizes must be positive integers")
            return int(value)

        image = _image_shape(inputs)
        inputs = size(inputs) if image is None else int(np.prod(image))
        actions = size(actions)
        units = _code_units(code)
        # the sensory region is the code when a vector brain asks for one
        coded_cortex = bool(units) and image is None
        sensory_width = units if coded_cortex else inputs
        code_width = units or inputs
        sensory_scale = real("sensory_scale", sensory_scale, low=0)
        working_memory_scale = real("working_memory_scale", working_memory_scale, low=0)
        history_scale = real("history_scale", history_scale, low=0)
        heard = _history_decays(options.get("history_decays", ()))
        recall_scale = real("recall_scale", recall_scale, low=0)
        recalled = _recall_genes(
            options.get("recall", False),
            {name: value for name, value in options.items() if name.startswith("recall_")},
        )
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
        if options.get("episodic", True) and options.get("memory_choice") is True:
            for name, value in _EPISODIC_FOUNDERS.items():
                # a coded brain's memories key on the code as it is: no separation of their own
                options.setdefault(name, 0 if units and name == "memory_expansion" else value)
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
        if image is None:
            regions = [Region("sensory", sensory_width)]
        else:
            height, width, channels = image
            regions = [
                visual_cortex(
                    height, width, channels=channels, features=features, field=field, seed=seed
                )
            ]
        regions.extend(Region(name, width) for name, width in zip(names, widths, strict=True))
        regions.extend(
            (prefrontal_cortex(widths[-1]), motor_cortex(actions, lateral=lateral, slots=slots))
        )
        projections = [
            Projection(regions[0].name, names[0], scale=sensory_scale, reciprocal=False)
        ]
        projections.extend(
            Projection(left, right) for left, right in zip(names, names[1:], strict=False)
        )
        projections.extend(
            (
                Projection("association", "motor"),
                Projection(
                    "prefrontal", "association", scale=working_memory_scale, reciprocal=False
                ),
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
        if heard:
            # Appended last, after any efference copy: the brain with the history is the brain
            # without it plus one block of sensory width per decay and their one projection.
            regions.append(Region("history", len(heard) * sensory_width))
            projections.append(
                Projection("history", "association", scale=history_scale, reciprocal=False)
            )
        if recalled:
            # Appended last, after any history: the brain with the recall memory is the brain
            # without it plus the recall and next regions and one projection from each.
            event = recalled["value"] == "event"
            width = code_width if event else widths[-1]
            for name in ("recall", "next") if recalled["next"] else ("recall",):
                regions.append(Region(name, width))
                projections.append(
                    Projection(name, "association", scale=recall_scale, reciprocal=False)
                )
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        result = cls(
            develop(genome, seed=seed), seed=seed, slots=slots, code=units,
            inputs=inputs if coded_cortex else None, **options,
        )
        result._composition = {
            "inputs": inputs,
            "actions": actions,
            "modules": list(widths),
            "observers": list(observer_widths),
            "slots": [int(width) for width in result.learner.slot_sizes],
            "lateral": lateral,
            "sensory_scale": sensory_scale,
            "working_memory_scale": working_memory_scale,
            "seed": int(seed),
        }
        if heard:
            result._composition["history_scale"] = history_scale
        if recalled:
            result._composition["recall_scale"] = recall_scale
        if image is not None:
            result._composition["image"] = list(image)
        if coded_cortex:
            result._composition["code"] = units
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
        Submit that outcome before changing beta; do not discard it to retune. So is
        changing ``recall_address_step``, ``recall_address`` or ``recall_address_reward``
        while that feedback holds the recall memory's address contrast, which belongs to
        the lesson's form and rates it was taken at. At ``recall_address_reward=0`` the
        actor drops the address map's trace, and the brain reports and saves as one
        composed at 0.

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
            "working_memory_scale", "resting_bias", "slots", "seed", "episodic", "backend",
            "device", "connectome", "history_decays", "history_scale", *_EPISODIC_SETTINGS,
            "recall", "recall_scale", *(f"recall_{name}" for name in Recall.CONSTRUCTION),
            "code",
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
            "history_amplitude": (self.history, "amplitude"),
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
            elif field == "focus" or name in ("efference_amplitude", "history_amplitude"):
                value = real(name, value, low=0)
            else:
                value = real(name, value)
            memory_changes.append((component, field, value))
            if component is self.hippocampus and self.context_memory is not None:
                memory_changes.append((self.context_memory, field, value))
        for name in [name for name in options if name.startswith("recall_")]:
            setting = name[len("recall_"):]
            if setting not in Recall.RUNTIME and setting not in Recall.OPTIONS:
                raise TypeError(f"unknown brain setting: {name!r}")
            if self.recall is None:
                raise ValueError(f"{name} requires its existing memory component")
            memory_changes.append(
                (self.recall, setting, Recall.setting(setting, options.pop(name)))
            )
        if self.recall is not None:
            # the null slot's logit, null / temperature, must stay finite under the change
            retuned = {
                field: value for component, field, value in memory_changes
                if component is self.recall
            }
            Recall.null_logit(
                retuned.get("null", self.recall.null),
                retuned.get("temperature", self.recall.temperature),
            )

        learning_genes, actor_genes, arousal_genes = split_genes(options)
        learning = configured(learning, learning_genes, "learning")
        reward = configured(reward, actor_genes, "actor")
        arousal = arousal_config(
            arousal, arousal_genes, current=current_arousal, retuning=True
        )
        if self.basal_ganglia._pending is not None and learning.beta != self.learner.config.beta:
            raise ValueError("learning_beta cannot change while sampled feedback is pending")
        if self.basal_ganglia._pending_memory is not None:
            # held with sampled feedback alone: the pending address contrast belongs to the
            # lesson's form and rates
            for component, field, value in memory_changes:
                if (
                    component is self.recall
                    and field in ("address_step", "address", "address_reward")
                    and value != getattr(component, field)
                ):
                    raise ValueError(
                        f"recall_{field} cannot change while sampled feedback with an address "
                        "contrast is pending"
                    )

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
        if self.recall is not None and not self.recall.address_reward:
            # reward leaves the address map as it is: the actor keeps no eligibility of it, and
            # the brain reports and saves as one composed without it
            agent.trace_memory = agent._pending_memory = None
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
            ("history", self.history, ("decays", "amplitude")),
            ("memory", self.hippocampus, ("decay", "rate", "amplitude")),
        ):
            if component is not None:
                genes.update({f"{prefix}_{field}": getattr(component, field) for field in fields})
        if isinstance(self.hippocampus, SynapticMemory):
            genes["consolidation"] = self.hippocampus.consolidation
        if self.recall is not None:
            genes.update(
                {
                    f"recall_{name}": getattr(self.recall, name)
                    for name in (*Recall.RUNTIME, *Recall.CONSTRUCTION)
                }
            )
            # an option is described where it differs from its founder
            recalled = self.recall.to_dict()
            genes.update(
                {f"recall_{name}": recalled[name] for name in Recall.OPTIONS if name in recalled}
            )
        populations = self.connectome.populations
        result = {
            "layout": {
                "inputs": int(self.inputs),
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
            "history": None if self.history is None else self.history.to_dict(),
            "recall": None if self.recall is None else self.recall.to_dict(),
            "memory": None if self.hippocampus is None else self.hippocampus.to_dict(),
            # what the memories key on: the observation or the brain's sparse code of it, what
            # the cortex reads, and the episodic separation of an uncoded key
            "observation_code": {
                "kind": "observation" if self.sparse_code is None else "code",
                "inputs": int(self.inputs),
                "width": int(self.code_width),
                "units": 0 if self.sparse_code is None else int(self.sparse_code.expansion),
                "fan_in": 0 if self.sparse_code is None else int(self.sparse_code.fan_in),
                "winners": 0 if self.sparse_code is None else int(self.sparse_code.winners),
                "cortex": "code" if self.coded_cortex else "observation",
                "separator": None
                if self.hippocampus is None or self.hippocampus.separator is None
                else ("sparse" if self.hippocampus.separator.fan_in else "dense"),
            },
            # construction settings of an episodic memory whose records vote in the choice
            "episodic": None
            if not self.memory_choice
            else {
                "choice": True,
                "context": self.memory_context,
                "replay": self.memory_replay,
                "demonstrations": self.memory_demonstrations,
                "context_memory": None
                if self.context_memory is None
                else self.context_memory.to_dict(),
            },
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
        if x.shape[1] != self.inputs:
            raise ValueError(f"observations need {self.inputs} values per row, got {x.shape[1]}")
        return x

    def observation_code(self, observations: Any) -> np.ndarray:
        """The code the memories key on, ``(batch, code_width)``: the observations themselves
        for a brain without a code, else their sparse code, the largest positive sums of one
        unit in ten kept and the row scaled to unit length."""
        return self._code(self._observations(observations))

    def _code(self, x: np.ndarray) -> np.ndarray:
        """The observation code of validated observations; the observations themselves,
        the same array, without a code."""
        return x if self.sparse_code is None else _unit_rows(self.sparse_code.code(x))

    def _sensed(self, x: np.ndarray, code: np.ndarray) -> np.ndarray:
        """What the sensory neurons are given: the observations, or with a coded cortex their
        code."""
        return code if self.coded_cortex else x

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        """The drive of a batch of observations: the sensory neurons stimulated, plus the
        working memory, the efference copy, the heard history, the recall memory and the
        hippocampal recall when ``memory`` is set. An episodic memory that votes in the
        choice (``memory_choice``) is not read into the drive. The recall memory addresses
        the moment here and keeps the address for the moment's write and lesson. Both
        memories read the observation code."""
        coded = self._coded_moment  # the moment's code, when this is its settle
        if coded is not None and coded[0] is observations:
            return self._stimulus(coded[0], coded[1], memory=memory)
        x = self._observations(observations)
        return self._stimulus(x, self._code(x), memory=memory)

    def _stimulus(self, x: np.ndarray, code: np.ndarray, *, memory: bool = True) -> np.ndarray:
        """The drive of validated observations and their code."""
        drive = np.zeros((len(x), self.connectome.n))
        drive[:, self.sensory_index] = (
            self._sensed(x, code) * self.brain.neuron_model.stimulus_amplitude
        )
        if memory and self.working_memory is not None:
            drive = self.working_memory.stimulate(drive)
        if memory and self.efference is not None:
            drive = self.efference.stimulate(drive)
        if memory and self.history is not None:
            drive = self.history.stimulate(drive)
        if memory and self.recall is not None:
            # the moment's address, before its settle, reads the stream's slots
            drive = self.recall.stimulate(drive, code)
        if memory and self.hippocampus is not None and not self.memory_choice:
            # the released call for a brain without a code (an instrument may wrap it)
            drive = (
                self.hippocampus.stimulate(drive)
                if self.sparse_code is None
                else self.hippocampus.stimulate(drive, key=code)
            )
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
            supplied = None if salience is None else importance
            self.last_learning = self.learn(r, ended, x, bootstrap=bootstrap, salience=supplied)
        else:
            self.last_learning = {}
        if labels is not None:
            self._lesson(x, labels)
        return self.act(x)

    def lesson(self, observations: Any, labels: Any) -> dict[str, float]:
        """Teach the labels of the present observations without acting on them.

        This is the lesson ``step(..., teacher=labels)`` gives before its act: the drive reads
        every memory the act reads, under ``learning_surprise`` the rows are judged by the
        settle the act reads, and with a recall memory the address lesson follows at the
        taught rows. An episodic memory that remembers demonstrations records the taught
        rows. Returns the teaching report (with a recall memory, ``address_rows``,
        ``address_error`` and ``address_refused`` too), which ``last_learning`` holds with the
        ``demonstration_`` prefix; a refused phase raises ``LearningPhaseError`` and leaves the
        parameters and memories as they were. The next ``act`` settles afresh under the
        changed parameters and writes the moment. The streams are the live ones; an action
        whose sampled outcome is pending is refused, since its eligibility belongs to the
        parameters before the lesson: report the outcome first (``step`` or ``learn``).
        """
        x = self._observations(observations)
        labels = self.learner._labels(np.asarray(labels))
        if len(labels) != len(x):
            raise ValueError("labels must match the observation batch")
        current = self.basal_ganglia.state
        if current is not None and len(np.atleast_2d(current.v)) != len(x):
            raise ValueError("lesson batch must match the live streams; reset for new streams")
        if self.basal_ganglia._pending is not None:
            raise RuntimeError(
                "a sampled action awaits its outcome; report it with step or learn first"
            )
        self.last_learning = {}
        return dict(self._lesson(x, labels))

    def _lesson(self, x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """The lesson of validated observations and labels; ``last_learning`` takes its report
        with the ``demonstration_`` prefix."""
        from .learning import LearningPhaseError

        batch = len(x)
        self._coded_moment = (x, self._code(x))
        drive = self.stimulus(x)
        code = self._coded(x)
        answer = None
        if self.learner.config.surprise:
            # the brain's own answer: the settle its act reads, warm from the stream
            answer = self._choices(self._qualified(drive, self._activity(), operation="act"))
        try:
            lesson, teaching = self.learner.step(drive, labels, answer=answer)
        except LearningPhaseError as error:
            self.last_learning.update(
                {"demonstration_" + name: value for name, value in error.report.items()}
            )
            self.last_learning["demonstrations"] = 0.0
            raise
        if self.recall is not None:
            # the address lesson: the slots scored against the lesson's movement step the map
            teaching.update(self.recall.learn(self.learner.brain, lesson))
        self.last_learning.update(
            {"demonstration_" + name: value for name, value in teaching.items()}
        )
        # A demonstration teaches the slow policy, not an invented reward value. An
        # episodic memory also records the shown action for the situation it was shown in,
        # under the observation code; under the learner's surprise rule, for the rows the
        # lesson taught.
        shown = np.ones(batch, bool) if lesson.taught is None else lesson.taught
        if (self.memory_choice and self.memory_demonstrations
                and self.hippocampus is not None and shown.any()):
            value = np.full(batch, self.memory_demonstrations)
            self._remember(
                code, self._episode_key(code) if self.context_memory is not None else None,
                labels, value, shown,
            )
        self._prepared = None
        self.basal_ganglia._drive = None
        self.last_learning["demonstrations"] = float(batch)
        return teaching

    def live(
        self, observations: Any, *, reward: Any = None, done: Any = None, decision_id: Any = None
    ) -> np.ndarray:
        """One moment of a continuing life: routine while outcomes match, repair when not.

        ``reward`` and ``done`` describe the preceding action, as in ``step``. The
        brain's ``arousal`` decides what this moment costs. Calm, it answers with
        the greedy choice of one qualified settle: no eligibility phases, no
        learning, no write to the associative memory, no parameter changes; the
        eligibility of earlier sampled actions fades by one step, as time passes. The
        working trace, the heard history and the recall memory's slots advance at every
        accepted moment, calm or aroused (with ``recall_write="aroused"`` the slots at the
        aroused moments alone); the recall memory's address map changes in lessons and, with
        ``recall_address_reward``, by the outcomes of sampled actions. Its forecasts for the
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
                if ended[0] and self.history is not None:
                    self.history.reset(1, rows=np.array([0]))
                if ended[0] and self.recall is not None:
                    self.recall.reset(1, rows=np.array([0]))
                # a moment without eligibility has passed: the credit of earlier sampled
                # actions fades as it does between two outcomes that are learned from
                agent.fade(ended)
            if routine and arousal.aroused and self.hippocampus is not None:
                assert lived is not None
                if not self.memory_choice:
                    self._record(
                        self._code(lived[0]), lived[1], r,
                        SynapticMemory.salience_vector(np.abs(r), 1),
                    )
                    recorded = True
                else:
                    # the outcome that woke the brain surprised it: known at any reward
                    self._record_outcome(
                        lived[0], lived[1], r, ended, np.ones(1, bool), woken=True
                    )
                    recorded = True
                if recorded and not self.memory_choice:
                    assert answer is not None
                    # the record can change this moment's recall: settle again under it, warm
                    agent._free, agent._free_brain, agent._drive = (
                        answer[1], self.brain, answer[0]
                    )
                    answer = None
            if routine and self.memory_choice:
                self._received(r)  # a calm outcome joins the path; replay may value it later
            if routine and ended[0]:
                self._end_attempt(ended)
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
        code = self._coded(x)  # computed by the settle of this moment
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
            self._recorded(code, chosen),
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
        working trace, the copy of its last command, its heard history, its recall memory and
        its associative memory; its working trace advances to that state, its heard history
        takes in the observation and its recall memory writes the moment (with
        ``recall_write="aroused"`` when the awaited action was sampled). It issues no action
        and takes no outcome:
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
        self._coded_moment = (x, self._code(x))
        drive = self.stimulus(x)
        state = self._qualified(drive, self._activity(), operation="wait")
        code = self._coded(x)
        if self.working_memory is not None:
            self.working_memory.update(state)
        if self.history is not None:
            self.history.hear(self._sensed(x, code))  # what was sensed joins the heard past
        if self.recall is not None:
            # the sensed moment joins the stream's slots; with ``recall_write="aroused"`` only
            # while the action it awaits was sampled
            self.recall.witness(code, state, aroused=self._moment is not None)
        if self.context_memory is not None:
            self._context = self._episode_key(code)  # what was sensed enters the temporal context
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

    def _recorded(self, code: np.ndarray, chosen: list[int]) -> float | None:
        """The outcome the associative memory forecasts for the chosen action in this
        situation (``code`` its observation code): the mean of the records it holds for the
        chosen motor neurons, in reward units. None without a memory, or when its reads are
        silent."""
        memory = self.hippocampus
        if memory is None or not np.isfinite(memory.amplitude) or memory.amplitude == 0.0:
            return None
        units = np.asarray(chosen) + np.asarray(self.learner.slot_offsets)[: len(chosen)]
        read = self._vote(code, self._decided(1)) if self.memory_choice else memory.recall(code)
        values = np.asarray(read)[0, units] / memory.amplitude
        return float(np.mean(values)) if np.isfinite(values).all() else None

    def _forecast(self, x: np.ndarray, ended: bool) -> tuple[np.ndarray, BrainState]:
        """Read the next value without changing the stream. A finished episode's
        forecast starts from rest and fresh traces; accepting its outcome commits
        that reset, so a refused forecast or arousal update preserves feedback."""
        agent = self.basal_ganglia
        if not ended:
            return self._settled(x)
        kept = (agent._free, agent._free_brain, agent._drive)
        traces: list[tuple[Any, dict[str, np.ndarray]]] = [
            (trace, {name: getattr(trace, name).copy() for name in ("trace", "last", "cold")})
            for trace in (self.working_memory, self.efference)
            if trace is not None
        ]
        if self.history is not None:
            traces.append((self.history, {"trace": self.history.trace.copy()}))
        recalled = None if self.recall is None else self.recall.snapshot()
        try:
            for trace, _ in traces:
                trace.reset(1, rows=np.array([0]))
            if self.recall is not None:
                self.recall.reset(1, rows=np.array([0]))
            agent._free = None
            return self._settled(x)
        finally:
            agent._free, agent._free_brain, agent._drive = kept
            for trace, values in traces:
                for name, value in values.items():
                    setattr(trace, name, value)
            if self.recall is not None and recalled is not None:
                self.recall.restore(recalled)

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
        without writes. The recall memory's live slots are read and none is written;
        the cortex of the moment before and the history follow the branch privately.
        Parameters, live activity, random generators and pending
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
        heard = copy(self.history)
        if heard is not None:
            heard.trace = heard.trace.copy()
        recalled = None if self.recall is None else self.recall.imagined()
        phases = []
        for value in sequence:
            self._coded_moment = (value, self._code(value))
            drive = self.stimulus(value, memory=False)
            code = self._coded(value)
            if trace is not None:
                drive = trace.stimulate(drive)
            if echo is not None:
                drive = echo.stimulate(drive)
            if heard is not None:
                drive = heard.stimulate(drive)
            if recalled is not None:
                # the imagined moment addresses the live slots; nothing is written
                drive = recalled.stimulate(drive, code)
            if self.hippocampus is not None and not self.memory_choice:
                drive = (
                    self.hippocampus.stimulate(drive)
                    if self.sparse_code is None
                    else self.hippocampus.stimulate(drive, key=code)
                )
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
            if heard is not None:
                # the imagined observation, as the sensory neurons were given it, joins the
                # private history
                heard.hear(self._sensed(value, code))
            if recalled is not None:
                recalled.witness(code, phase.state)  # the branch's cortex moves on
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
        stream's last state; the observations' code is computed here once for the moment.
        Refusal raises before anything changes."""
        self._coded_moment = (x, self._code(x))
        try:
            drive = self.stimulus(x)
            return drive, self._qualified(drive, self._activity(), operation="act")
        except BaseException:
            self._coded_moment = None
            raise

    def _coded(self, x: np.ndarray) -> np.ndarray:
        """The observation code of the moment being acted on, computed by ``_settled``; of any
        other observations, computed here."""
        coded = self._coded_moment
        return coded[1] if coded is not None and coded[0] is x else self._code(x)

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
        code = self._coded(x)  # the memories key on the observation code
        key = bias = None
        if self.memory_choice and self.hippocampus is not None:
            key = self._episode_key(code) if self.context_memory is not None else None
            bias = self._vote(code, key)
        action = self.basal_ganglia.act(drive, greedy=greedy, temperature=temperature, bias=bias)
        state = self.basal_ganglia.state
        if self.working_memory is not None and state is not None:
            self.working_memory.update(state)
        if self.efference is not None:
            self.efference.issue(self._command(action))
        if self.history is not None:
            # the observation acted on, as the sensory neurons were given it, joins the heard
            # past; the next drive holds the present
            self.history.hear(self._sensed(x, code))
        if self.recall is not None:
            # the moment acted on joins the stream's slots (with ``recall_write="aroused"`` a
            # sampled act's alone); its settled cortex enters the next moment's address
            self.recall.witness(code, free, aroused=not greedy)
        if bias is not None:
            if key is not None:
                # what the stream did joins its temporal context after what it saw
                self._decision = key
                self._context = key.copy()
                self._context[:, self.code_width:] += (1.0 - self.memory_context) * (
                    self._command(action)
                )
            if self.memory_replay:
                assert state is not None
                # how far the choice was the brain's best guess: the chosen action's
                # probability under its own policy, without the heat of exploration, over the
                # most probable action's (1 for the best guess), averaged over the slots
                p = self.basal_ganglia.probabilities(state, bias=bias)
                rows = np.arange(len(action))
                if action.ndim == 2:
                    slots = np.arange(action.shape[1])
                    picked = p[rows[:, None], slots[None, :], action]
                    guess = (picked / np.maximum(p.max(axis=-1), 1e-12)).mean(axis=1)
                else:
                    guess = p[rows, action] / np.maximum(p.max(axis=-1), 1e-12)
                self._extend_paths(code, key, action, np.minimum(guess, 1.0))
        self._moment = None if greedy else (x.copy(), action.copy())
        return action

    def _vote(self, code: np.ndarray, key: np.ndarray | None) -> np.ndarray:
        """The records' vote for every motor neuron: the mean of the place read of the
        observation code and the context read of the decision's key."""
        memory = self.hippocampus
        assert memory is not None
        place = np.asarray(memory.recall(np.asarray(code, dtype=float)), dtype=float)
        if self.context_memory is None or key is None:
            return place
        context = np.asarray(self.context_memory.recall(key), dtype=float)
        return np.asarray(0.5 * (place + context))

    def _decided(self, batch: int) -> np.ndarray | None:
        """The context key of each stream's last decision, when it belongs to this batch."""
        decision = self._decision
        return decision if decision is not None and len(decision) == batch else None

    def _episode_key(self, code: np.ndarray) -> np.ndarray:
        """The key of a decision on this observation code: the stream's temporal context with
        the code entered, ``(batch, code_width + motor)``. The code takes the share
        ``1 - memory_context``; the context before it keeps the rest."""
        x = np.asarray(code, dtype=float)
        keep = self.memory_context
        fresh = np.zeros((len(x), self.code_width + len(self.motor_index)))
        fresh[:, : self.code_width] = x
        context = self._context
        if context is None or len(context) != len(x) or not keep:
            return (1.0 - keep) * fresh if keep else fresh
        return keep * context + (1.0 - keep) * fresh

    def _extend_paths(
        self, x: np.ndarray, key: np.ndarray | None, action: np.ndarray, guess: np.ndarray
    ) -> None:
        """Add each stream's decision to its recent path; ``x`` is the observation code. A
        decision taken on the code of the stream's previous decision replaces it: that
        decision left the situation as it was, and only the decision that leaves a situation
        lies on the path onwards."""
        if len(self._paths) != len(x):
            self._paths = [[] for _ in range(len(x))]
        for row, path in enumerate(self._paths):
            entry = (
                None if key is None else key[row].copy(),
                np.array(action[row], copy=True),
                x[row].copy(),
                0.0,
                False,
                float(guess[row]),
            )
            if path and np.array_equal(path[-1][2], entry[2]):
                path[-1] = entry
            else:
                path.append(entry)
                del path[: -(self.memory_replay + 1)]

    def _received(self, reward: np.ndarray) -> None:
        """The outcome of each stream's last decision joins its path."""
        if len(self._paths) != len(reward):
            return
        for row, path in enumerate(self._paths):
            if path:
                key, action, observation, _, _, guess = path[-1]
                path[-1] = (key, action, observation, float(reward[row]), True, guess)

    def _record_outcome(
        self,
        observations: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        learned: np.ndarray,
        woken: bool = False,
        salience: np.ndarray | None = None,
    ) -> None:
        """Record the outcome of the last decision for the streams in ``learned``, where it is
        known: a reward or a punishment, the end of an attempt, a surprise that woke a calm
        brain, or an outcome the caller marked with a positive ``salience``. A zero outcome
        in the middle of an attempt is not yet one. A supplied salience also sets how fast
        the outcome consolidates; without one every write moves a record equally. A reward
        replays the decisions that led to it. The records key on the code of the
        ``observations`` the decision was taken on."""
        observations = self._code(observations)
        marked = np.zeros(len(reward), bool) if salience is None else salience > 0
        known = learned & ((reward != 0) | done | woken | marked)
        keys = self._decided(len(action))
        # what the memory expected of the decision, read before its outcome is written; an
        # attempt that ends is a closed episode, whose reward is replayed whole
        surprise = np.where(done, reward, reward - self._held(observations, keys, action))
        if known.any():
            self._remember(observations, keys, action, reward, known, salience)
        self._replay(reward, surprise, done, learned)

    def _held(
        self,
        observations: np.ndarray,
        keys: np.ndarray | None,
        action: np.ndarray,
        memory: FastSynapses | None = None,
    ) -> np.ndarray:
        """Each stream's record of its chosen action in reward units: the records' vote for
        the chosen motor neuron, or the read of ``memory`` alone, averaged over the slots of a
        slotted readout."""
        hippocampus = self.hippocampus
        assert hippocampus is not None
        if not hippocampus.amplitude:
            return np.zeros(len(action))
        if memory is None:
            read = self._vote(observations, keys)
        else:
            assert keys is not None  # one store is read under the keys it is keyed by
            read = memory.recall(keys)
        vote = np.asarray(read) / hippocampus.amplitude
        rows = np.arange(len(action))
        if action.ndim == 2:
            chosen = action + self.learner.slot_offsets[None, :]
            return np.asarray(vote[rows[:, None], chosen].mean(axis=1))
        return np.asarray(vote[rows, action])

    def _replay(
        self, reward: np.ndarray, surprise: np.ndarray, done: np.ndarray, learned: np.ndarray
    ) -> None:
        """Reverse replay: a reward credits, backwards along the stream's recent path (one
        decision per situation, the one that left it), the decisions whose own outcome was
        nothing, back to the last decision whose outcome was known (a reward or a
        punishment), each discounted per step of the path and up to the discounted reward.
        A reward that ends an attempt (``done``) closes an episode: it is replayed whole, into
        both kinds of record. In a continuing life, where it is not known where the paid
        effort began, a reward is replayed in proportion to its surprise (the reward less
        what the memory expected of the paid decision): into each decision's context record,
        that decision in the history it was taken in, and into its place record only as far
        as it was an exploration rather than the brain's best guess. There an expected reward
        replays nothing. Replay only adds credit."""
        if not self.memory_replay or len(self._paths) != len(reward):
            return
        gamma = float(self.basal_ganglia.config.gamma)
        temporal = self.context_memory
        paid = (reward > 0) & (surprise > 0) & learned
        if temporal is None:
            paid &= done  # without context records, a continuing life replays nothing
        rows = [row for row in np.flatnonzero(paid) if self._paths[row]]
        width = len(reward)
        for k in range(1, self.memory_replay + 1):
            discount = gamma**k
            if not discount:
                break  # with gamma zero no credit flows back
            # the decision k steps before the paid one (each path's last entry), while the
            # outcomes on the way back were nothing
            rows = [
                row
                for row in rows
                if len(self._paths[row]) > k
                and self._paths[row][-1 - k][4]
                and self._paths[row][-1 - k][3] == 0.0
            ]
            if not rows:
                break
            earlier = [self._paths[row][-1 - k] for row in rows]
            observations = np.zeros((width, self.code_width))  # the path keeps the codes
            observations[rows] = [entry[2] for entry in earlier]
            keys = None
            if earlier[0][0] is not None:
                keys = np.zeros((width, len(earlier[0][0])))
                keys[rows] = [entry[0] for entry in earlier]
            action = np.zeros((width, *np.shape(earlier[0][1])), dtype=np.int64)
            action[rows] = [entry[1] for entry in earlier]
            ended = np.zeros(width, bool)
            ended[rows] = done[rows]
            guess = np.ones(width)
            guess[rows] = [entry[5] for entry in earlier]
            # what the written records hold: both kinds at the end of an attempt, the context
            # records in a continuing life
            held = self._held(observations, keys, action)
            if temporal is not None and keys is not None and not ended[rows].all():
                held = np.where(ended, held, self._held(keys, keys, action, temporal))
            credit = np.zeros(width)
            credit[rows] = np.minimum(
                discount * surprise[rows], discount * reward[rows] - held[rows]
            )
            write = credit > 0
            if (write & ended).any():
                self._remember(observations, keys, action, held + credit, write & ended)
            if (write & ~ended).any():
                assert temporal is not None and keys is not None
                steady = np.zeros(width)  # every write moves a record equally
                self._record(keys, action, held + credit, steady, write & ~ended, temporal)
                # the place record only as far as the decision was an exploration: the best
                # guess tells nothing about what paid apart from its history
                place = self._held(observations, observations, action, self.hippocampus)
                surprising = credit * (1.0 - guess)
                gain = write & ~ended & (surprising > 0)
                if gain.any():
                    target = place + np.minimum(surprising, discount * reward - place)
                    gain &= target > place
                    if gain.any():
                        self._record(observations, action, target, steady, gain)

    def _end_attempt(self, done: np.ndarray) -> None:
        """The streams whose attempt ended begin the next with no context and no path."""
        rows = np.flatnonzero(done)
        if self._context is not None and len(self._context) == len(done):
            self._context[rows] = 0.0
        if len(self._paths) == len(done):
            for row in rows:
                self._paths[row] = []

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
        still belongs to the state the action was chosen in. With a positive
        ``recall_address_reward`` the recall memory's address map steps on its own trace by
        the same dopamine, and the report names the step (``address_reward_step``).
        """
        following = self._observations(next_observations)
        self._coded_moment = (following, self._code(following))
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
        temporal = self.context_memory
        temporal_values = (
            {}
            if temporal is None
            else {
                name: getattr(temporal, name)
                for name in ("strength", "mass", "writes", "consolidated")
            }
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
        heard = self.history
        heard_trace = None if heard is None or not done.any() else heard.trace.copy()
        recalled = self.recall
        recalled_state = None if recalled is None or not done.any() else recalled.snapshot()
        episode_values = (
            None
            if not self.memory_choice
            else (
                None if self._context is None else self._context.copy(),
                [list(path) for path in self._paths],
            )
        )
        try:
            if memory is not None and self._moment is not None:
                keys, action = self._moment
                if not self.memory_choice:
                    self._record(self._code(keys), action, reward, importance)
                else:
                    self._received(reward)
                    self._record_outcome(
                        keys, action, reward, done, np.ones(len(action), bool),
                        salience=None if salience is None else importance,
                    )
            if self.memory_choice and done.any():
                self._end_attempt(done)
            if trace is not None and done.any():
                trace.reset(len(done), rows=np.flatnonzero(done))
            if echo is not None and done.any():
                echo.reset(len(done), rows=np.flatnonzero(done))
            if heard is not None and done.any():
                heard.reset(len(done), rows=np.flatnonzero(done))
            if recalled is not None and done.any():
                recalled.reset(len(done), rows=np.flatnonzero(done))
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
            if heard is not None and heard_trace is not None:
                heard.trace = heard_trace
            if recalled is not None and recalled_state is not None:
                recalled.restore(recalled_state)
            if temporal is not None:
                for name, value in temporal_values.items():
                    setattr(temporal, name, value)
            if episode_values is not None:
                self._context, self._paths = episode_values
            raise
        self._moment = None
        self._prepared = following.copy()
        self._awaiting = None
        return report

    def _remember(
        self,
        observations: np.ndarray,
        keys: np.ndarray | None,
        action: np.ndarray,
        value: np.ndarray,
        streams: np.ndarray,
        salience: np.ndarray | None = None,
    ) -> None:
        """The episodic memory's write for the streams in ``streams``: the place records under
        the observations and, with a temporal context, the context records under the
        decision's keys. The stream's record moves ``memory_rate`` toward ``value`` at once,
        and the shared record ``consolidation`` toward it, faster with a supplied
        ``salience``; without one every write moves a record equally."""
        sized = np.zeros(len(action)) if salience is None else salience
        self._record(observations, action, value, sized, streams)
        if self.context_memory is not None and keys is not None:
            self._record(keys, action, value, sized, streams, self.context_memory)

    def _record(
        self,
        keys: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        importance: np.ndarray,
        known: np.ndarray | None = None,
        memory: FastSynapses | None = None,
    ) -> None:
        """Write the outcome of a chosen action for the situation it was chosen in; with
        ``known``, only for the streams where it is set. The hippocampus unless ``memory``."""
        memory = self.hippocampus if memory is None else memory
        assert memory is not None
        # Every stream writes when ``known`` is unset: the released write, without a mask.
        write = None if known is None else np.asarray(known, bool)
        if write is not None and not write.any():
            return
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
            memory.observe(keys, target, write, salience=importance, value_mask=observed)
        else:
            target = memory.recall(keys)
            target[rows, chosen] = value
            memory.observe(keys, target, write)

    def reset(self) -> None:
        """Start every stream afresh; hippocampal records are kept. A brain with arousal
        begins the new stream calm, with what it was used to forgotten and its age kept."""
        self.basal_ganglia.reset()
        if self.working_memory is not None:
            self.working_memory.reset(0)
        if self.efference is not None:
            self.efference.reset(0)
        if self.history is not None:
            self.history.reset(0)
        if self.recall is not None:
            self.recall.reset(0)  # the slots are emptied; the address map is learned and stays
        self._moment = None
        self._coded_moment = None
        self._prepared = None
        self.last_learning = {}
        self._last_settlement = None
        self._lived = None
        self._last_arousal = None
        self._awaiting = None
        self._context = None
        self._decision = None
        self._paths = []
        if self.arousal is not None:
            self.arousal.reset()

    def parameters(self) -> int:
        """Actor/critic and consolidated weights and the recall memory's address map; per-stream
        transient storage is additional."""
        recall = self.recall
        return (
            self.basal_ganglia.parameters()
            + (
                self.hippocampus.consolidated.size
                if isinstance(self.hippocampus, SynapticMemory)
                else 0
            )
            + (0 if self.context_memory is None else self.context_memory.consolidated.size)
            + (0 if recall is None else recall.w_x.size + recall.w_c.size + recall.b.size)
        )

    def save(self, path: str | Path) -> Path:
        """Save the complete composition, including an action awaiting its real outcome.

        Includes critic, optimizer, random generators, working and episodic memories, the
        heard history, the recall memory (its slots, address map and code), eligibility, the
        current free phase, any pending action's nudged states and the state ``wait`` sensed
        since that action was chosen.
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
        if self.memory_choice:
            # An episodic memory's choice and its stream state are part of the continuation;
            # earlier formats would read its records into the drive instead.
            metadata["format"] = "cadence-generic/6"
            temporal = self.context_memory
            metadata["episodic"] = {
                "choice": True,
                "context": float(self.memory_context),
                "replay": int(self.memory_replay),
                "demonstrations": float(self.memory_demonstrations),
                "context_memory": None if temporal is None else temporal.to_dict(),
            }
            if temporal is not None:
                for name in ("pre", "post", "strength", "mass", "consolidated"):
                    data["temporal/" + name] = getattr(temporal, name)
                if temporal.separator is not None:
                    for name, value in temporal.separator.arrays().items():
                        data["temporal/separator/" + name] = value
                if self._context is not None:
                    data["episode/context"] = self._context
                if self._decision is not None:
                    data["episode/decision"] = self._decision
            if self._paths and any(self._paths):
                data.update(self._saved_paths())
        if self.history is not None:
            # The heard history is part of the continuation; earlier formats cannot carry it.
            # A brain without one writes the metadata of the earlier formats unchanged.
            metadata["format"] = "cadence-generic/7"
            metadata["history"] = self.history.to_dict()
            data["history/trace"] = self.history.trace
        recall = self.recall
        if recall is not None:
            # The recall memory is part of the continuation; earlier formats cannot carry it.
            # A brain without one writes the metadata of the earlier formats unchanged.
            metadata["format"] = "cadence-generic/8"
            metadata["recall"] = recall.to_dict()
            for name in ("keys", "values", "nexts", "held", "previous", "history", "w_x", "w_c",
                         "b"):
                data["recall/" + name] = getattr(recall, name)
            # the actor's eligibility of the address map, and the pending action's contrast
            if agent.trace_memory is not None:
                data["recall/trace"] = agent.trace_memory
            if agent._pending is not None and agent._pending_memory is not None:
                data["recall/pending"] = agent._pending_memory
        if self.sparse_code is not None:
            # The brain's code is part of the continuation: the records have its width and the
            # keys read its tables. A brain without one writes the earlier formats unchanged.
            metadata["format"] = "cadence-generic/9"
            metadata["code"] = {**self.sparse_code.to_dict(), "cortex": self.coded_cortex}
            tables = self.sparse_code.arrays()
            data["code/index"], data["code/weight"] = tables["index"], tables["weight"]
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
                for name, value in self.hippocampus.separator.arrays().items():
                    data["episodic/separator/" + name] = value
            if isinstance(self.hippocampus, SynapticMemory):
                data["episodic/consolidated"] = self.hippocampus.consolidated
        data["generic"] = np.array(canonical_json(metadata))
        return _write(data, path)

    def _saved_paths(self) -> dict[str, np.ndarray]:
        """The replay paths as padded arrays, with each stream's length."""
        streams, longest = len(self._paths), max(len(path) for path in self._paths)
        width = self.code_width + len(self.motor_index)
        slots = self.learner.slot_count
        keys = np.zeros((streams, longest, width))
        observations = np.zeros((streams, longest, self.code_width))
        actions = np.zeros((streams, longest, slots) if slots > 1 else (streams, longest), int)
        rewards = np.zeros((streams, longest))
        received = np.zeros((streams, longest), bool)
        guesses = np.zeros((streams, longest))
        for row, path in enumerate(self._paths):
            for index, (key, action, observation, reward, known, guess) in enumerate(path):
                if key is not None:
                    keys[row, index] = key
                actions[row, index], observations[row, index] = action, observation
                rewards[row, index], received[row, index] = reward, known
                guesses[row, index] = guess
        saved = {
            "episode/path/actions": actions,
            "episode/path/observations": observations,
            "episode/path/rewards": rewards,
            "episode/path/received": received,
            "episode/path/guess": guesses,
            "episode/path/lengths": np.array([len(path) for path in self._paths]),
        }
        if self.context_memory is not None:
            saved["episode/path/keys"] = keys
        return saved

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
            formats = tuple(f"cadence-generic/{level}" for level in range(1, 10))
            if not isinstance(meta, dict) or meta.get("format") not in formats:
                raise ValueError("unsupported Brain checkpoint format")
            # each format carries the component it was introduced for, and may carry those
            # of the earlier formats; a component under an earlier format name is refused
            level = formats.index(meta["format"]) + 1
            if (level == 9) != ("code" in meta):
                raise ValueError("a brain's code belongs to checkpoint format cadence-generic/9")
            if level == 8 and meta.get("recall") is None:
                raise ValueError("format cadence-generic/8 carries a recall memory")
            if meta.get("recall") is not None and level < 8:
                raise ValueError(
                    "a recall memory belongs to checkpoint formats cadence-generic/8 and /9"
                )
            if level == 7 and meta.get("history") is None:
                raise ValueError("format cadence-generic/7 carries a heard history")
            if meta.get("history") is not None and level < 7:
                raise ValueError(
                    "a heard history belongs to checkpoint formats cadence-generic/7 to /9"
                )
            if level == 6 and "episodic" not in meta:
                raise ValueError("format cadence-generic/6 carries an episodic memory's choice")
            if "episodic" in meta and level < 6:
                raise ValueError(
                    "an episodic memory's choice belongs to checkpoint formats cadence-generic/6 "
                    "to /9"
                )
            if level == 5 and "arousal" not in meta:
                raise ValueError("format cadence-generic/5 carries a live stream's arousal")
            if level == 4 and meta.get("efference") is None:
                raise ValueError("format cadence-generic/4 carries an efference copy")
            if level < 4 and ("arousal" in meta) != (level == 3):
                raise ValueError("arousal state belongs to checkpoint format cadence-generic/3")
            arousal = Arousal.from_dict(meta["arousal"]) if "arousal" in meta else None
            if "hippocampus" not in meta:
                raise ValueError("missing hippocampus metadata")
            resting_bias = _validate_resting_bias(meta.get("resting_bias", 0.0))
            learner = Learner.load(path, backend=backend, device=device, precision=precision)
            composition = _load_composition(meta.get("composition"), learner)
            populations = learner.brain.connectome.populations
            code = _load_code(
                meta, data, len(populations.get("sensory", populations.get("visual/input", ())))
            )
            try:
                _validate_life_state(meta, data, learner, code)
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid or incomplete saved continuation state") from exc
            # a memory keyed on the code numbers its key coordinates from zero: the code's
            # width bounds them, with the motor neurons after it for the context records
            neurons = learner.brain.connectome.n
            coded = None if code is None else code[1].expansion
            memory = _load_memory(meta["hippocampus"], data, neurons, keys=coded)
            heard = meta.get("history")
            recalled = meta.get("recall")
            result = cls(
                learner.brain.connectome,
                episodic=False,
                reward=ActorCriticConfig(**meta["reward"]),
                slots=[int(k) for k in learner.slot_sizes],
                code=0 if code is None else code[1].expansion,
                inputs=code[0] if code is not None and code[2] else None,
                **(
                    {}
                    if heard is None
                    else {
                        "history_decays": heard["decays"],
                        "history_amplitude": heard["amplitude"],
                    }
                ),
                **(
                    {}
                    if recalled is None
                    else {
                        "recall": True,
                        **{
                            f"recall_{name}": recalled[name]
                            for name in (*Recall.RUNTIME, *Recall.CONSTRUCTION, *Recall.OPTIONS)
                            if name in recalled
                        },
                    }
                ),
            )
            if code is not None:
                # the saved code as validated, its settings, seed and tables: the construction
                # above drew a code from the default seed
                result.sparse_code = code[1]
            if heard is not None:
                assert result.history is not None  # validated above: the population is there
                result.history.trace = data["history/trace"].astype(float)
            if recalled is not None:
                recall = result.recall
                assert recall is not None  # validated above: the populations are there
                for name in ("keys", "values", "nexts", "previous", "history", "w_x", "w_c", "b"):
                    setattr(recall, name, data["recall/" + name].astype(float))
                recall.held = data["recall/held"].astype(bool)
                recall.writes = int(recalled["writes"])
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
                if "recall/pending" in data:  # validated above: a pending action's contrast
                    agent._pending_memory = data["recall/pending"].astype(float)
                if "moment/observations" in data:
                    result._moment = (
                        data["moment/observations"].copy(),
                        data["moment/action"].copy(),
                    )
            if "recall/trace" in data:
                agent.trace_memory = data["recall/trace"].astype(float)
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
            if memory is not None:
                # the place records are keyed on the brain's observation code: its coordinates
                # from zero with a code, the sensory neurons without
                expected = (
                    np.arange(result.code_width)
                    if result.sparse_code is not None
                    else result.sensory_index
                )
                if len(memory.pre) != result.code_width:
                    raise ValueError(
                        f"the saved place records are keyed on {len(memory.pre)} coordinates "
                        f"where the brain's observation code has {result.code_width}"
                    )
                if not np.array_equal(memory.pre, expected):
                    raise ValueError(
                        "the saved place records are keyed on other coordinates than the "
                        f"brain's {result.code_width}: a coded brain numbers them from zero "
                        "and an uncoded one keys on its sensory neurons"
                    )
            result.hippocampus = memory
            episodic = meta.get("episodic")
            if episodic is not None:
                result.memory_choice = True
                result.memory_context = float(episodic["context"])
                result.memory_replay = int(episodic["replay"])
                result.memory_demonstrations = float(episodic["demonstrations"])
                temporal = _load_memory(
                    episodic["context_memory"], data, neurons, "temporal/",
                    keys=None if coded is None else coded + len(result.motor_index),
                )
                if temporal is not None and not isinstance(temporal, SynapticMemory):
                    raise ValueError("an episodic memory's context records consolidate")
                if temporal is not None and not np.array_equal(
                    temporal.pre,
                    np.arange(result.code_width + len(result.motor_index))
                    if result.sparse_code is not None
                    else np.concatenate([result.sensory_index, result.motor_index]),
                ):
                    raise ValueError(
                        "the saved context records are keyed on other coordinates than the "
                        "brain's observation code and its motor neurons"
                    )
                result.context_memory = temporal
                for name in ("context", "decision"):
                    if "episode/" + name in data:
                        setattr(result, "_" + name, data["episode/" + name].astype(float))
                if "episode/path/lengths" in data:
                    lengths = data["episode/path/lengths"]
                    keys = data.get("episode/path/keys")
                    actions = data["episode/path/actions"]
                    observations = data["episode/path/observations"]
                    rewards, received = data["episode/path/rewards"], data["episode/path/received"]
                    guesses = data["episode/path/guess"]
                    result._paths = [
                        [
                            (
                                None if keys is None else keys[row, index].astype(float),
                                np.array(actions[row, index], dtype=np.int64),
                                observations[row, index].astype(float),
                                float(rewards[row, index]),
                                bool(received[row, index]),
                                float(guesses[row, index]),
                            )
                            for index in range(int(lengths[row]))
                        ]
                        for row in range(len(lengths))
                    ]
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
