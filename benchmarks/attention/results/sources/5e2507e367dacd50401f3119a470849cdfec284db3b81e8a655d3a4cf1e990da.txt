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
  stream's own recent moments, each moment addressed before its settle (with several heads,
  one pair of regions per head);
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

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast

import numpy as np

from ._configuration import (
    TEACHING,
    arousal_config,
    configured,
    plain,
    real,
    rescaled,
    scales_of,
    split_genes,
    taught,
    teaching_fields,
)
from .arousal import Arousal, ArousalConfig
from .brain import Apical, Backend, BrainState, Equilibrium
from .brain import Brain as NeuralGraph
from .code import LearnedCode
from .connectome import Connectome
from .genome import Genome, Projection, develop
from .learning import Learner, LearnerConfig, learning_neuron_model
from .memories import EpisodicRecords, Memories, Moment, command, record_action
from .memory import SynapticMemory
from .plasticity import ActorCritic, ActorCriticConfig
from .regions import Region, cortex, motor_cortex, prefrontal_cortex, visual_cortex
from .stream import Efference, FastSynapses, History, PatternSeparator, Recall, Trace

__all__ = ["Brain"]

# The checkpoint format of a brain: one group per memory (``Brain.save``). ``Brain.load`` reads
# it and, through 0.85.x, the earlier formats: ``cadence-generic/1`` to ``/9``, which 0.84.3
# reads, and ``/10``, which development builds of the learned code wrote.
BRAIN_FORMAT = "cadence-brain/1"
# A memory's group in BRAIN_FORMAT and the prefix its arrays had in the formats of 0.84.3; the
# episodic memory's arrays are the context store's and the stream's (``_earlier_layout``).
_MEMORY_PREFIXES = {
    "working_memory": "working/",
    "efference": "efference/",
    "history": "history/",
    "recall": "recall/",
    "hippocampus": "episodic/",
}
_REWARD_ARRAYS = (
    "w_critic",
    "trace",
    "trace_bias",
    "trace_critic",
    "salience",
    "_drive",
)


# The motor lateral weight an unset ``lateral`` resolves to: no motor-to-motor synapses, at
# every menu size. At -0.5 the undamped free solve oscillates from 4 actions on (at 8 actions
# a fifth of one-row acts and every act of 16 rows settle only under the damping fallback, at
# 544 sweeps against 32, and a teaching free phase of 16 rows runs out its budget on most
# lessons), so a competing readout is a choice made against 0.0 on development tasks.
_LATERAL_UNSET = 0.0

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

_Operation = TypeVar("_Operation", bound=Callable[..., Any])


def _operation(method: _Operation) -> _Operation:
    """A public operation: the recall memory binds the content products it reuses to its
    bank for the operation's length (``Recall._scope``). A warning the method issues for its
    caller counts this wrapper's frame."""

    @wraps(method)
    def operated(self: Brain, *args: Any, **kwargs: Any) -> Any:
        if self.recall is None:
            return method(self, *args, **kwargs)
        with self.recall._scope():
            return method(self, *args, **kwargs)

    return cast(_Operation, operated)


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


# The settle's step in a brain with an apical compartment (``apical=fraction``). A normalized
# top-down input is a loop of bounded magnitude, not of bounded gain: at the full step its
# fast mode rings and a teaching free phase can run out its budget; the half step damps it.
# The fixed point, and so every answer and contrast the settle reaches, is the same.
_APICAL_STEP = 0.5


def _apical(connectome: Connectome, fraction: float) -> Apical:
    """The top-down half of every reciprocal pair between two regions, as an apical
    compartment of ``fraction``.

    The regions are the populations named without a slash, ordered by their first neuron as
    ``develop`` lays a genome out. A synapse lands apically when its reverse synapse exists
    and its presynaptic region comes after its postsynaptic one: a module's input from the
    module above it, the association's from the motor region, any region's from an observer.
    The one-way projections (the sensory input, the working trace, the efference copy, the
    history and the recall memory) and the synapses within a region stay basal. Each region
    that receives apical synapses is one pool."""
    n = connectome.n
    region = np.full(n, -1, dtype=np.int64)
    named = [
        (min(members), name) for name, members in connectome.populations.items()
        if "/" not in name and len(members)
    ]
    for rank, (_, name) in enumerate(sorted(named)):
        region[np.asarray(connectome.populations[name], dtype=np.int64)] = rank
    pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
    key = pre * n + post
    reverse = post * n + pre
    order = np.sort(key)
    hit = np.minimum(np.searchsorted(order, reverse), max(len(order) - 1, 0))
    paired = (order[hit] == reverse) if len(order) else np.zeros(0, dtype=bool)
    synapses = paired & (region[pre] > region[post]) & (region[post] >= 0)
    groups = np.full(n, -1, dtype=np.int64)
    for rank in np.unique(region[post[synapses]]):
        groups[region == rank] = rank
    return Apical(synapses, groups, fraction)


def _reward() -> ActorCriticConfig:
    # The composed actor's bias rate is 0.05 at eta 1.0; a bare ActorCriticConfig derives
    # eta / 10.
    return ActorCriticConfig(
        gamma=0.9, lam=0.8, eta=1.0, eta_bias=0.05, eta_critic=0.3, eligibility_steps=12
    )


def _validate_efference_amplitude(value: Any) -> float:
    """The read gain of the efference copy: a finite nonnegative real scalar; zero is the
    brain without one."""
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
    is given, or an empty mapping for a brain without it. A setting of 0.84.3 that this
    version removed is refused by name (``Recall.RETIRED``)."""
    if not isinstance(recall, bool):
        raise ValueError("recall must be boolean")
    founders = Recall.founders()
    settings: dict[str, Any] = {}
    for name in [name for name in genes if name.startswith("recall_")]:
        setting = name[len("recall_"):]
        if setting in Recall.RETIRED:
            raise ValueError(Recall.removed(setting))
        if setting not in founders and setting not in Recall.OPTIONS:
            raise TypeError(f"unknown brain setting: {name!r}")
        settings[setting] = Recall.setting(setting, genes.pop(name))
    if not recall:
        if settings:
            raise ValueError("recall settings need recall=True")
        return {}
    return {**founders, **settings}


def _copy_identity(
    connectome: Connectome, recalled: Mapping[str, Any], offset: int
) -> Connectome:
    """The connectome whose copy pathway starts from the declared tokens (``recall_tokens``).

    For motor neuron ``j`` of token ``tokens[j]``: the synapse from every head's ``recall``
    port at value coordinate ``offset + tokens[j]`` (the observation's own coordinates in the
    value) starts at ``recall_copy``, so the token the read delivers drives its own action; the
    synapse from every ``next`` port at that coordinate starts at zero; and the synapse from
    sensory neuron ``tokens[j]`` starts at ``-recall_copy``, so a token the present
    observation already holds, the read's echo of the moment that addressed it, is not copied
    back. Every other synapse into the motor region from a read port, or from the sensory
    region, starts at zero. The same connectome when it already starts so (a saved brain).
    Every one of these synapses learns on its own, as every synapse does."""
    populations = connectome.populations
    n = connectome.n
    value = np.full(n, -1, dtype=np.int64)  # each port neuron's value coordinate
    names = [name for port in Recall.ports(recalled.get("heads", 1), recalled["next"])
             for name in port]
    for name in names:
        neurons = np.asarray(populations[name], dtype=np.int64)
        value[neurons] = np.arange(len(neurons))
        if recalled.get("action", False):  # the command's part: no token, starts at zero
            value[np.asarray(populations[f"{name}_action"], dtype=np.int64)] = -2
    sensory = np.asarray(populations["sensory"], dtype=np.int64)
    heard = np.full(n, -1, dtype=np.int64)
    heard[sensory] = np.arange(len(sensory))
    motor = np.asarray(populations["motor"], dtype=np.int64)
    local_motor = np.full(n, -1, dtype=np.int64)
    local_motor[motor] = np.arange(len(motor))
    tokens = np.asarray(recalled["tokens"], dtype=np.int64)
    pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
    token = np.where(local_motor[post] >= 0, tokens[np.maximum(local_motor[post], 0)], -1)
    pathway = (value[pre] != -1) & (local_motor[post] >= 0)
    identity = pathway & (token >= 0) & (value[pre] == offset + token)
    echoes = (heard[pre] >= 0) & (token >= 0) & (heard[pre] == token)
    declared = int((tokens >= 0).sum())
    if int(identity.sum()) != declared * len(names) or int(echoes.sum()) != declared:
        raise ValueError(
            "recall_tokens need a synapse from every read port's token coordinate and from the "
            "token's sensory neuron into its motor neuron; Brain.compose adds them"
        )
    reading = np.isin(pre, np.concatenate([
        np.asarray(populations[name], dtype=np.int64) for name in names
        if name.startswith("recall")
    ]))
    copy = float(recalled["copy"])
    sign = np.where(pathway | ((heard[pre] >= 0) & (local_motor[post] >= 0)), 0.0, connectome.sign)
    sign[identity & reading] = copy
    sign[echoes] = -copy
    if np.array_equal(sign, connectome.sign):
        return connectome
    return Connectome(
        n, connectome.pre, connectome.post, connectome.count, sign,
        dict(populations), connectome.label,
    )


def _learned_code_units(value: Any) -> int:
    """The units of the brain's learned code: ``learned_code`` as given, 0 or None for none."""
    if value is None:
        return 0
    units = _count("learned_code", value)
    if units == 1:
        raise ValueError("a learned code holds at least 2 units")
    return units


def _learned_code_genes(learned_code: Any, genes: dict[str, Any]) -> dict[str, Any]:
    """The learned code's settings, validated: its units and founders with every
    ``learned_code_<setting>`` of ``genes`` over them (those names are removed from
    ``genes``), or an empty mapping for a brain without it."""
    units = _learned_code_units(learned_code)
    settings: dict[str, Any] = {}
    for name in [name for name in genes if name.startswith("learned_code_")]:
        setting = name[len("learned_code_"):]
        if setting not in LearnedCode.GENES and setting not in LearnedCode.REMOVED:
            raise TypeError(f"unknown brain setting: {name!r}")
        settings[setting] = LearnedCode.setting(setting, genes.pop(name))
    if not units:
        if settings:
            raise ValueError("learned_code settings need learned_code=units")
        return {}
    return {"units": units, **LearnedCode.founders(), **settings}


_LEARNED_CODE_TOGETHER = (
    "code and learned_code together are not built: the learned code's key joins the "
    "observation itself as its identity, at the identity weight 0.9; name one of them"
)


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
        # provenance without a working_memory_scale was composed at the fixed scale 12
        value = {**value, "working_memory_scale": 12.0}
    if not isinstance(value, dict) or set(value) - {
        "history_scale", "recall_scale", "image", "code", "learned_code", "attention", "apical"
    } != (expected | {"working_memory_scale"}):
        raise ValueError("invalid saved composition provenance")
    compartment = learner.brain.apical
    if ("apical" in value) != (compartment is not None) or (
        compartment is not None and real("saved apical", value["apical"], low=0)
        != compartment.fraction
    ):
        raise ValueError("a composed apical compartment is saved with its graph, and only then")
    if "attention" in value and value["attention"] is not True:
        raise ValueError("invalid saved composition attention")
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
    if "learned_code" in value:
        # the code region of that many units, between the sensory region and the first module
        units = value["learned_code"]
        if (
            isinstance(units, bool)
            or not isinstance(units, int)
            or units != len(populations.get("code", ()))
            or sensory != "sensory"
            or "image" in value
            or "code" in value
        ):
            raise ValueError("invalid saved composition learned_code")
    if "attention" in value and (
        "learned_code" not in value or "recall" not in populations or "next" not in populations
    ):
        raise ValueError("saved attention needs a learned code and recall/next populations")
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
        spec = metadata.get("separator")  # An unseparated memory may omit this field.
        if spec is None:
            if any(key.startswith(prefix + "separator/") for key in data):
                raise ValueError("saved separator arrays require separator metadata")
        else:
            if not isinstance(spec, dict):
                raise ValueError("invalid separator metadata")
            # a sparse separator names its fan-in; a dense one's metadata may omit it
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


def _validate_learned_code(meta: dict[str, Any], data: Mapping[str, Any], learner: Learner) -> int:
    """The learned code a checkpoint names, validated against the connectome and its saved
    tables: its units, or 0 for a brain without one."""
    spec = meta.get("learned_code")
    if spec is None:
        if any(name.startswith("learned_code/") for name in data) or any(
            name in data for name in ("moment/code", "lived/code")
        ):
            raise ValueError("saved learned code tables require learned_code metadata")
        return 0
    populations = learner.brain.connectome.populations
    fields = {"inputs", "units", "context", "seed", "stepping", "learning", "top_down", "work",
              *LearnedCode.GENES}
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError("invalid saved learned code metadata")
    if not isinstance(spec["learning"], bool):
        raise ValueError("saved learned code learning must be boolean")
    real("saved learned code top_down", spec["top_down"], low=0)
    for name in ("inputs", "units", "context", "seed", "stepping"):
        value = spec[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"saved learned code {name} must be a nonnegative integer")
    # the held code steps its own phase alone: the brain keeps the model's step, the half
    # step with an apical compartment
    dt = learner.brain.neuron_model.dt
    expected = 1.0 if learner.brain.apical is None else _APICAL_STEP
    if dt != expected:
        raise ValueError("the saved learned code's stepping disagrees with the saved step")
    work = spec["work"]
    if not isinstance(work, dict) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in work.values()
    ):
        raise ValueError("saved learned code work must count nonnegative integers")
    for name in LearnedCode.GENES:
        try:
            LearnedCode.setting(name, spec[name])
        except ValueError as exc:
            raise ValueError(f"invalid saved learned code {name}") from exc
    if "code" not in populations or "prefrontal" not in populations or "sensory" not in populations:
        raise ValueError("a learned code needs sensory, code and prefrontal populations")
    units = int(spec["units"])
    if units != len(populations["code"]):
        raise ValueError(
            f"the code population of {len(populations['code'])} neurons is not the saved "
            f"learned code of {units} units"
        )
    if spec["inputs"] != len(populations["sensory"]) or spec["context"] != len(
        populations["prefrontal"]
    ):
        raise ValueError("the saved learned code reads other widths than the connectome has")
    shapes = {
        "dictionary": (spec["inputs"], units),
        "weights": (units, spec["context"]),
        "thresholds": (units,),
    }
    for name, shape in shapes.items():
        key = "learned_code/" + name
        if key not in data:
            raise ValueError(f"missing saved learned code table: {name}")
        table: np.ndarray = data[key]
        if table.shape != shape or table.dtype.kind not in "iuf" or not np.isfinite(table).all():
            raise ValueError(f"invalid saved learned code table: {name}")
    if (data["learned_code/thresholds"] < 0).any():
        raise ValueError("a learned code's thresholds are nonnegative")
    if not np.allclose(
        np.linalg.norm(data["learned_code/dictionary"], axis=0), 1.0, rtol=0, atol=1e-9
    ):
        raise ValueError("a learned code's dictionary has unit-length columns")
    return units


def _validate_life_state(
    meta: dict[str, Any],
    data: Mapping[str, Any],
    learner: Learner,
    code: tuple[int, PatternSeparator, bool] | None,
    learned: int = 0,
) -> None:
    """Reject corrupt continuation state before exposing a resumed agent. ``code`` is the
    brain's sparse code the checkpoint names, validated: the observations are saved at their
    width, and the memories' keys have the code's. ``learned`` is the units of the learned
    code the checkpoint names, the memories' key width then."""
    n = learner.brain.connectome.n
    populations = learner.brain.connectome.populations
    association = len(populations["association"])
    sensory = len(populations.get("sensory", populations.get("visual/input", ())))
    observed = sensory if code is None else code[0]
    code_width = sensory if code is None else code[1].expansion
    if learned:  # the memories' key: the code, joined to an observation no wider than it
        code_width = learned + (sensory if LearnedCode.joins(sensory, learned) else 0)

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
    if meta.get("pending", False):
        # a sampled action awaiting its outcome was chosen on the drive of its free state
        array("actor/_drive", (batch, n))
    if ("actor/trace" in data) != ("actor/trace_bias" in data):
        raise ValueError("incomplete saved eligibility traces")
    array("actor/w_critic", (association,))
    for name, width in (
        ("trace", learner.brain.connectome.synapses),
        ("trace_bias", n),
        ("trace_critic", association + 1),
        ("salience", n),
        # the drive the free state was settled for, required above with a pending action; a
        # lesson clears it so that the next act settles afresh, and a brain saved before that
        # act keeps its free state without one
        ("_drive", n),
    ):
        if "actor/" + name in data:
            array("actor/" + name, (batch, width))
    if "prepared" in data:
        array("prepared", (batch, observed))
    if meta.get("pending", False):
        array("pending/value", (batch,))
        for phase in ("plus", "minus"):
            count("pending_" + phase + "_steps")
            for name in ("v", "activation", "adaptation"):
                array(f"pending/{phase}/{name}", (batch, n))
    if meta.get("pending", False) or "moment/observations" in data or "moment/action" in data:
        if not meta.get("pending", False):
            raise ValueError("saved action observations need pending credit")
        array("moment/observations", (batch, observed))
        slotted = learner.slot_count > 1
        action = array("moment/action", (batch, learner.slot_count) if slotted else (batch,))
        limit = learner.slot_sizes[None, :] if slotted else len(learner.output_index)
        if action.dtype.kind not in "iu" or (action < 0).any() or (action >= limit).any():
            raise ValueError("invalid saved action indices")
        if learned:  # the code the decision was keyed on, settled at its moment
            array("moment/code", (batch, code_width))
    if "moment/code" in data and (not learned or "moment/observations" not in data):
        raise ValueError("a saved moment's code belongs to a pending action of a learned code")
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
        recorded = lived.get("recorded")  # the record channel's forecast, which 0.84.3 saved
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
        if learned:
            array("lived/code", (1, code_width))
        if was_sampled:
            if (
                not np.array_equal(observations, array("moment/observations", (1, observed)))
                or not np.array_equal(action, array("moment/action", action.shape))
                or float(forecast) != float(array("pending/value", (1,))[0])
            ):
                raise ValueError("saved lived action must match its pending feedback")
    if "lived/code" in data and (not learned or "lived" not in meta):
        raise ValueError("a saved lived code belongs to a lived action of a learned code")
    awaiting = meta.get("awaiting")
    if meta.get("format") == "cadence-generic/5" and "awaiting" not in meta:
        raise ValueError("format cadence-generic/5 carries a sensed state awaiting an outcome")
    if "awaiting" in meta and meta.get("format") not in (
        "cadence-generic/5", "cadence-generic/6", "cadence-generic/7", "cadence-generic/8",
        "cadence-generic/9", "cadence-generic/10",
    ):
        raise ValueError(
            "a sensed state awaiting an outcome belongs to formats cadence-generic/5 to /10"
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
        # the trace's focus of 0.84.3, a weighting of each neuron by its movement, is read
        # at its founder zero
        if working.get("focus", 0.0) != 0.0:
            raise ValueError(
                f"this checkpoint was saved with working_memory_focus={working['focus']!r}, "
                "which this version removed; load it with cadence-net 0.84.3"
            )
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
            "cadence-generic/10",
        ):
            raise ValueError(
                "an efference copy belongs to checkpoint formats cadence-generic/4 to /10"
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
        # with a learned code the history hears the code region, what the cortex reads
        sources = ("code",) if learned else ("sensory", "visual/input")
        if (
            not isinstance(heard, dict)
            or set(heard) != {"decays", "amplitude", "source", "target", "neurons"}
            or not isinstance(heard["decays"], list)
            or heard["source"] not in sources
            or heard["source"] not in populations
            or heard["target"] != "history"
            or "history" not in populations
        ):
            raise ValueError("invalid saved history ports")
        decays = _history_decays(heard["decays"])
        real("saved history_amplitude", heard["amplitude"], low=0)
        width = len(populations["history"])
        heard_width = learned or sensory
        if not decays or width != len(decays) * heard_width or heard["neurons"] != width:
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
    """The settings, slots and address map of a recall memory (formats /8 to /10); ``width``
    is the observation code's, the observations the memory is given. A file saved under a
    setting this version removed is refused with a message naming 0.84.3."""
    populations = learner.brain.connectome.populations
    recalled = meta.get("recall")
    if recalled is None:
        if any(name.startswith("recall/") for name in data):
            raise ValueError("saved recall state requires recall metadata")
        return
    names = {*Recall.RUNTIME, *Recall.CONSTRUCTION}
    # the format of 0.84.3 always wrote the keys of ``FORMAT``; the other removed settings,
    # like the options, were saved only where they differed from their founders
    optional = {"inputs", *Recall.OPTIONS, *Recall.RETIRED}
    if (
        not isinstance(recalled, dict)
        or set(recalled) - optional
        != names | {"source", "target", "neurons", "writes"}
        or recalled["source"] not in ("sensory", "visual/input")
        or recalled["source"] not in populations
        or recalled["target"] != "recall"
        or "recall" not in populations
    ):
        raise ValueError("invalid saved recall ports")
    for name in Recall.RETIRED:
        if name in recalled and recalled[name] != Recall.FORMAT.get(name):
            raise ValueError(_removed_recall(name, recalled[name]))
    if recalled["recency"] is None:
        raise ValueError(_removed_recall("recency", None))
    settings = {name: Recall.setting(name, recalled[name]) for name in names}
    # an option is saved where it differs from its founder, and takes its founder where absent
    settings.update({name: Recall.setting(name, recalled[name]) for name in Recall.OPTIONS
                     if name in recalled})
    settings["position"] = _saved_position(recalled, data)
    settings["compete"] = _saved_compete(recalled)
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
    # with ``action`` a slot also carries the command, one value per motor neuron, in its key
    # and value, and the maps read the command of the moment before
    command = len(learner.output_index) if settings.get("action", False) else 0
    value = key + command  # the values are the events
    # the window's k events in order
    window = settings.get("window", 1)
    address = window * key + command
    heads = settings.get("heads", 1)
    regions = tuple(name for port in Recall.ports(heads, settings["next"]) for name in port)
    if (
        any(len(populations.get(name, ())) != key for name in regions)
        or any(len(populations.get(f"{name}_action", ())) != command
               for name in (regions if command else ()))
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
    if meta["free_steps"] is None:
        # A reset that keeps slots clears the free state before the next streams act.
        # The retained slots still own their rows independently of that cleared state.
        batch = keys.shape[0]
    slots = keys.shape[1]
    if keys.shape[0] != batch or slots > settings["capacity"]:
        raise ValueError("saved recall slots must belong to the saved streams, within capacity")
    array("keys", (batch, slots, address))
    for name in ("values", "nexts"):
        array(name, (batch, slots, value))
    if array("held", (batch, slots)).dtype != np.bool_:
        raise ValueError("invalid saved recall state: held")
    array("previous", (batch, association))
    if command:
        array("acted", (batch, command))
    # the format of 0.84.3 wrote the context history, empty at the one value of ``FORMAT``
    history = data.get("recall/history")
    if history is not None:
        array("history", (0 if not len(history) else batch, key))
    shapes = {"w_x": (key + command, address), "w_c": (association, address), "b": (address,)}
    maps = dict(shapes)
    for h in range(1, heads):  # every further head's address map
        maps.update({f"{name}_{h}": shape for name, shape in shapes.items()})
    count = Recall.position_count(settings.get("position", 0), settings["capacity"])
    for h in range(heads if count else 0):  # every head's position maps, with a position term
        shapes_h = ((key + command, count), (association, count), (count,))
        maps.update(dict(zip(Recall.position_names(h), shapes_h, strict=True)))
    for name, shape in maps.items():
        array(name, shape)
    if window > 1:  # the window's previous events per stream: a window of another length
        past = data.get("recall/past")  # is refused by the shape
        array("past", (0 if past is None or not len(past) else batch, window - 1, key))
    # the actor's eligibility of the address map, and the contrast of a pending action
    eligibility = (batch, key + command + association + 1, heads * address)
    if "recall/trace" in data:
        array("trace", eligibility)
    if "recall/pending" in data:
        if not meta.get("pending", False):
            raise ValueError("a saved address contrast belongs to a pending action")
        array("pending", eligibility)
    # the streams whose latest slot closed an episode, saved while one does
    if "recall/ended" in data and array("ended", (batch,)).dtype != np.bool_:
        raise ValueError("invalid saved recall state: ended")
    expected = {"keys", "values", "nexts", "held", "previous", *maps}
    if command:
        expected |= {"acted"}
    if history is not None:
        expected |= {"history"}
    if window > 1:
        expected |= {"past"}
    if {name[len("recall/"):] for name in data if name.startswith("recall/")} - {
        "trace", "pending", "ended"
    } != expected:
        raise ValueError("unexpected saved recall arrays")


def _saved_position(recalled: dict[str, Any], data: Any) -> int:
    """A saved memory's position term: the option as saved, else its founder where the file
    holds the maps (the option left out at its founder); a file saved before the position term
    holds neither and loads without one."""
    if "position" in recalled:
        return int(Recall.setting("position", recalled["position"]))
    if any(name.startswith("recall/position_") for name in data):
        return int(Recall.options()["position"])
    return 0


def _saved_compete(recalled: dict[str, Any]) -> bool:
    """A saved memory's competing heads: the option as saved (a memory of several heads always
    keeps it); a file of several heads saved before heads competed holds none and loads with
    heads that read alike, as it was saved."""
    if "compete" in recalled:
        return bool(Recall.setting("compete", recalled["compete"]))
    return bool(int(recalled.get("heads", 1)) == 1)


def _removed_recall(name: str, value: Any) -> str:
    """The refusal of a checkpoint saved under a recall setting this version removed."""
    setting = "recency=None" if name == "recency" else f"{name}={value!r}"
    return (
        f"this checkpoint was saved with the recall memory's {setting}, which this version "
        "removed; load it with cadence-net 0.84.3"
    )


def _actor_config(values: Any) -> ActorCriticConfig:
    """A saved actor's config. One saved by 0.84.3 names genes this version removed; it loads
    where they leave the law of this version (the dopamine uncentred, so its floor and scale
    are never read, the critic's step divided by its trace's energy, learning from the
    actor's bounded dopamine, and every synapse stepping on its own contrast, without momentum
    or RMS normalization), and is refused with a message naming 0.84.3 otherwise. The running
    level of the centred dopamine and the optimizer moments that 0.84.3 saved beside it are
    not read."""
    if not isinstance(values, dict):
        raise ValueError("invalid saved actor config")
    values = dict(values)
    for name in ("dopamine_floor", "center_scale"):  # read only by the centred dopamine
        values.pop(name, None)
    for name, kept in (
        ("dopamine_center", (0, 0.0)),
        ("critic_normalize", (True,)),
        ("critic_signal", ("auto", "modulated")),
        ("normalize", (0, 0.0)),
        ("momentum", (0, 0.0)),
    ):
        if name in values:
            value = values.pop(name)
            if isinstance(value, bool) != isinstance(kept[0], bool) or value not in kept:
                raise ValueError(
                    f"this checkpoint was saved with the actor's {name}={value!r}, which this "
                    "version removed; load it with cadence-net 0.84.3"
                )
    return ActorCriticConfig(**values)


def _earlier_layout(
    saved: dict[str, Any], data: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A ``cadence-brain/1`` file in the layout of the formats of 0.84.3, which the reader
    validates: each memory's settings and arrays under the names those formats gave them,
    and the format number of the latest component the file carries."""
    memories = saved.get("memories")
    if not isinstance(memories, dict) or set(memories) - {*_MEMORY_PREFIXES, "episodic"}:
        raise ValueError("invalid saved memories")
    if "actor" not in saved:
        raise ValueError("missing saved actor config")
    meta = {k: v for k, v in saved.items() if k not in ("memories", "actor")}
    meta["reward"] = saved["actor"]
    meta["working_memory"] = memories.get("working_memory")
    meta["hippocampus"] = memories.get("hippocampus")
    for name in ("efference", "history", "recall", "episodic"):
        if name in memories:
            meta[name] = memories[name]
    level = max(
        (number for number, present in (
            (2, True), (3, "arousal" in meta), (4, "efference" in meta),
            (5, "awaiting" in meta), (6, "episodic" in meta), (7, "history" in meta),
            (8, "recall" in meta), (9, "code" in meta), (10, "learned_code" in meta),
        ) if present),
    )
    meta["format"] = f"cadence-generic/{level}"
    arrays: dict[str, Any] = {}
    for name in data:
        if name == "brain":
            continue
        if not name.startswith("memories/"):
            arrays[name] = data[name]
            continue
        group, _, rest = name[len("memories/"):].partition("/")
        if group not in memories or not rest:
            raise ValueError("saved memory arrays require their memory's settings")
        if group != "episodic":
            arrays[_MEMORY_PREFIXES[group] + rest] = data[name]
        elif rest.startswith("context_memory/"):
            arrays["temporal/" + rest[len("context_memory/"):]] = data[name]
        elif rest in ("context", "decision") or rest.startswith("path/"):
            arrays["episode/" + rest] = data[name]
        else:
            raise ValueError("unexpected saved episodic array")
    return meta, arrays


def _validate_episodic_state(
    meta: dict[str, Any], data: Mapping[str, Any], learner: Learner, sensory: int
) -> None:
    """The settings and stream state of an episodic memory's choice (formats /6 to /10);
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


def _observation_width(
    inputs: int | Sequence[int], code: int | None = None, learned_code: int | None = None,
) -> int:
    """``Genome.observation_width``: the key width of the memories of a brain with these
    settings."""
    image = _image_shape(inputs)
    width = int(np.prod(image)) if image is not None else _count("inputs", inputs, low=1)
    units = _learned_code_units(learned_code)
    if units:  # the key: the code, joined to an observation no wider than it
        return units + (width if LearnedCode.joins(width, units) else 0)
    return _code_units(code) or width


def _layout(
    inputs: int | Sequence[int],
    actions: int,
    options: Mapping[str, Any],
    *,
    modules: Sequence[int] = (64,),
    observers: Sequence[int] = (),
    lateral: float | None = None,
    scales: Mapping[str, Any] | None = None,
    features: int = 8,
    field: int = 3,
    code: int | None = None,
    learned_code: int | None = None,
    seed: int = 0,
    slots: int | Sequence[int] = 1,
) -> tuple[Genome, dict[str, Any]]:
    """The layout ``Brain.compose`` develops (``Genome.composed``) and the settings it resolved.
    ``options`` are compose's other settings; the efference copy, the heard history, the recall
    memory and the learned code's ``cortex`` and ``identity`` add regions and projections, and
    the rest do not change the layout."""
    for name in options:
        refusal = rescaled(name)
        if refusal is not None:
            raise ValueError(refusal)

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
    learned = _learned_code_units(learned_code)
    if learned and units:
        raise ValueError(_LEARNED_CODE_TOGETHER)
    if learned and image is not None:
        raise ValueError(
            "a learned code needs a vector brain: an image brain's sensory region is its "
            "visual cortex"
        )
    # the sensory region is the code when a vector brain asks for one
    coded_cortex = bool(units) and image is None
    sensory_width = units if coded_cortex else inputs
    if learned:  # the key: the code, joined to an observation no wider than it
        code_width = learned + (inputs if LearnedCode.joins(inputs, learned) else 0)
    else:
        code_width = units or inputs
    heard_width = learned or sensory_width  # what the history hears: the code it reads
    scale = scales_of(scales)
    sensory_scale, working_memory_scale = scale["sensory"], scale["trace"]
    history_scale, recall_scale = scale["history"], scale["recall"]
    heard = _history_decays(options.get("history_decays", ()))
    recalled = _recall_genes(
        options.get("recall", False),
        {name: value for name, value in options.items() if name.startswith("recall_")},
    )
    if lateral is None:
        lateral = _LATERAL_UNSET
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
    if learned:
        # the code region, every ordered pair of its units a lateral synapse; the
        # dictionary, the lateral weights and the predictor take the region's own tables
        # once the brain is built, in place of the developed draw
        regions.append(cortex(learned, lateral=-1.0, name="code"))
    regions.extend(Region(name, width) for name, width in zip(names, widths, strict=True))
    regions.extend(
        (prefrontal_cortex(widths[-1]), motor_cortex(actions, lateral=lateral, slots=slots))
    )
    if learned:
        projections = [
            Projection("sensory", "code", scale=1.0, reciprocal=False),
            Projection("code", names[0], scale=sensory_scale, reciprocal=False),
        ]
    else:
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
    if learned:
        projections.append(Projection("prefrontal", "code", scale=0.0, reciprocal=False))
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
        regions.append(Region("history", len(heard) * heard_width))
        projections.append(
            Projection("history", "association", scale=history_scale, reciprocal=False)
        )
    if recalled:
        # Appended last, after any history: the brain with the recall memory is the brain
        # without it plus the recall and next regions, one neuron per value coordinate
        # (the values are the events, of the code's width), and one projection from each.
        for name in ("recall", "next") if recalled["next"] else ("recall",):
            regions.append(Region(name, code_width))
            projections.append(
                Projection(name, "association", scale=recall_scale, reciprocal=False)
            )
        # every further head's read port, wired as the first and appended last: the brain of
        # H heads is the brain of H - 1 heads plus one port and its projections
        for port in Recall.ports(recalled.get("heads", 1), recalled["next"])[1:]:
            for name in port:
                regions.append(Region(name, code_width))
                projections.append(
                    Projection(name, "association", scale=recall_scale, reciprocal=False)
                )
        ports = [
            name for port in Recall.ports(recalled.get("heads", 1), recalled["next"])
            for name in port
        ]
        if recalled.get("action", False):
            # every port's read of the executed command, one neuron per motor neuron, appended
            # last: the brain that keeps the command is the brain without it plus these regions,
            # whose synapses into the association start at zero, so a newborn settles as one
            # without them until learning gives them weight
            for name in ports:
                regions.append(Region(f"{name}_action", actions))
                projections.append(
                    Projection(f"{name}_action", "association", scale=0.0, reciprocal=False)
                )
        if recalled.get("copy", 0.0):
            # the copy pathway from every read port into the motor region, appended last: the
            # brain with it is the brain without it plus these synapses, drawn at the initial
            # magnitude ``recall_copy`` (the command's part at zero); declared tokens start it
            # as their identity once developed (``_copy_identity``)
            for name in ports:
                projections.append(
                    Projection(name, "motor", scale=recalled["copy"], reciprocal=False)
                )
            if recalled.get("action", False):
                for name in ports:
                    projections.append(
                        Projection(f"{name}_action", "motor", scale=0.0, reciprocal=False)
                    )
            if recalled.get("tokens") is not None:
                # the present observation's tokens into their actions, the copy's echo
                # suppression, drawn at zero and started at ``-recall_copy`` once developed
                projections.append(
                    Projection(regions[0].name, "motor", scale=0.0, reciprocal=False)
                )
    genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
    return genome, {
        "inputs": inputs,
        "actions": actions,
        "units": units,
        "learned": learned,
        "coded_cortex": coded_cortex,
        "image": image,
        "lateral": lateral,
        "widths": widths,
        "observer_widths": observer_widths,
        "sensory_scale": sensory_scale,
        "working_memory_scale": working_memory_scale,
        "history_scale": history_scale,
        "heard": heard,
        "recall_scale": recall_scale,
        "recalled": recalled,
    }


def _checkpoint_learning_report(
    report: Mapping[str, float], *, restore: bool = False
) -> dict[str, float]:
    """Omit only redundant zero-start diagnostics; restore their public report view."""
    result = dict(report)
    for prefix in ("", "demonstration_"):
        for phase in ("free", "nudged", "opposite"):
            name = prefix + phase
            absolute = result.get(name + "_damping_halvings")
            if isinstance(absolute, bool) or not isinstance(absolute, (int, float)):
                continue
            try:
                if not np.isfinite(float(absolute)) or absolute < 0:
                    continue
            except OverflowError:
                continue
            initial, relative = name + "_initial_damping", name + "_relative_damping_halvings"
            if restore:
                if initial not in result and relative not in result:
                    result[initial], result[relative] = 0.0, absolute
            elif (type(result.get(initial)) in (int, float) and result[initial] == 0
                  and type(result.get(relative)) in (int, float)
                  and result[relative] == absolute):
                result.pop(initial)
                result.pop(relative)
    return result


class Brain:
    """Senses, an association cortex, a motor cortex, basal ganglia and consolidating memory.

    ``connectome`` needs populations ``sensory`` (or ``visual/input`` for an image),
    ``association`` and ``motor``, ``prefrontal`` for a working memory, ``efference``
    (one neuron per motor neuron) for a copy of the issued command, ``history`` (one
    block of sensory width per entry of ``history_decays``) for the observations as heard
    and, with ``recall=True``, ``recall`` and ``next`` (one neuron per value coordinate each)
    for the index memory of the stream's moments, and with ``recall_heads=k`` also
    ``recall_h`` and ``next_h`` for each further head (with ``recall_action`` each of them
    with a ``<region>_action`` region of one neuron per motor neuron, and with
    ``recall_copy`` synapses from them into ``motor``); ``Genome.composed`` lays one out. Without
    ``recall=True`` populations of those names are ordinary ones. ``Brain.compose`` builds
    the default brain with working memory, and with ``connectome`` gives a custom anatomy
    its founders.

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
    whatever its width. The code is a fixed random sparse code and learns nothing; it bounds
    the size of keys on wide observations, and ``memory_fan_in`` bounds an uncoded episodic
    separation the same way. ``learned_code=units`` (opt-in, without ``code``) learns the
    code instead: a region of the brain's own neurons (``LearnedCode``) whose code the cortex
    reads and every memory keys on. With
    ``inputs`` (the width of the observations) the sensory region is the code: every
    observation is coded before the drive, and ``inputs`` must then be named beside a
    ``code`` of the region's width.
    ``sparse_code`` holds the ``PatternSeparator``, ``code_width`` the key width
    (``Genome.observation_width`` gives it before the brain is built, to size the ``recall`` and
    ``next`` regions of a custom anatomy). A memory keyed on the code numbers its key
    coordinates from zero (``pre``) and is never read from the drive; the code is its
    separation, so the episodic stores take no separator of their own.
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> Brain:
        """An independent copy: every array, store, state and random generator copied, the
        immutable connectomes shared, the read-only reports copied as read-only reports."""
        twin = type(self).__new__(type(self))
        memo[id(self)] = twin
        for name, value in vars(self).items():
            if isinstance(value, MappingProxyType):
                value = MappingProxyType(deepcopy(dict(value), memo))
            else:
                value = deepcopy(value, memo)
            setattr(twin, name, value)
        return twin

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
        efference_decay: float = 0.2,
        efference_amplitude: float = 0.0,
        history_decays: Sequence[float] = (),
        history_amplitude: float = 3.0,
        recall: bool = False,
        learning: LearnerConfig | None = None,
        reward: ActorCriticConfig | None = None,
        slots: int | Sequence[int] = 1,
        arousal: ArousalConfig | Mapping[str, Any] | bool | None = None,
        seed: int = 0,
        backend: Backend = "cpu",
        device: str | None = None,
        inputs: int | None = None,
        code: int | None = None,
        learned_code: int | None = None,
        apical: float = 0.0,
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
        # the learned code: a region of the brain's own neurons that settles to a sparse
        # code of the observation; explicit acts and aroused live moments learn it
        learned = _learned_code_genes(learned_code, genes)  # its genes leave ``genes`` here
        if learned and self.sparse_code is not None:
            raise ValueError(_LEARNED_CODE_TOGETHER)
        if learned and sensory != "sensory":
            raise ValueError("a learned code needs a vector brain with a 'sensory' population")
        if learned and backend != "cpu":
            raise ValueError("a learned code settles on the cpu backend in this version")
        for name in ("code", "prefrontal") if learned else ():
            if name not in populations:
                raise ValueError(
                    f"learned_code needs a population named {name!r}; Brain.compose adds it"
                )
        if learned and len(populations["code"]) != learned["units"]:
            raise ValueError(
                f"the code population of {len(populations['code'])} neurons is not the "
                f"learned code of {learned['units']} units"
            )
        self.code_index = (
            np.asarray(populations["code"], dtype=np.int64) if learned else np.zeros(0, np.int64)
        )
        self.prefrontal_index = (
            np.asarray(populations["prefrontal"], dtype=np.int64)
            if "prefrontal" in populations
            else np.zeros(0, np.int64)
        )
        if learned:  # the key: the code, joined to an observation no wider than it
            joined = LearnedCode.joins(self.inputs, learned["units"])
            self.code_width = learned["units"] + (self.inputs if joined else 0)
        # whether the memories key on a code of the observation (fixed or learned) rather
        # than on the observation itself
        self._coded_keys = bool(units) or bool(learned)
        recalled = _recall_genes(recall, genes)  # the recall_* genes leave ``genes`` here
        # declared tokens start the copy pathway as their identity, the echo included
        tokened = recalled.get("tokens") is not None and bool(recalled.get("copy", 0.0))
        if tokened:
            # the observation's own coordinates in the recall values: the observation itself,
            # or a learned code joined to it, after the code's units
            joined = bool(learned) and LearnedCode.joins(self.inputs, learned["units"])
            if sensory != "sensory":
                raise ValueError("recall_tokens need a vector brain with a 'sensory' population")
            if self.sparse_code is not None or (learned and not joined):
                raise ValueError(
                    "recall_tokens need the observation's own coordinates in the recall values: "
                    "a brain without a code, or a learned code joined to an observation no "
                    "wider than it"
                )
            if len(recalled["tokens"]) != len(self.motor_index):
                raise ValueError(
                    "recall_tokens name one observation coordinate per motor neuron "
                    f"({len(self.motor_index)})"
                )
            if max(recalled["tokens"]) >= self.inputs:
                raise ValueError(
                    f"recall_tokens name observation coordinates below {self.inputs}"
                )
            ports = [
                name for port in Recall.ports(recalled.get("heads", 1), recalled["next"])
                for name in port
            ]
            if recalled.get("action", False):
                ports += [f"{name}_action" for name in ports]
            if all(name in populations for name in (*ports, "motor")):  # else the memory names it
                connectome = _copy_identity(
                    connectome, recalled, learned["units"] if learned else 0
                )
                self.connectome = connectome
        learning_genes, actor_genes, arousal_genes = split_genes(genes)
        learning = configured(_learning() if learning is None else learning, learning_genes,
                              "learning")
        if learning.initial_damping and backend != "cpu":
            raise ValueError("nonzero initial_damping requires CPU float64")
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
        if units and (memory_expansion or memory_fan_in):
            # a brain with a fixed code keys its memories on that code as it is: the code is
            # their separation; the learned code's stores separate their keys as without it
            raise ValueError(
                "a fixed code's memories key on the code as it is; memory_expansion and "
                "memory_fan_in need a brain without a fixed code"
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
        efference_amplitude = _validate_efference_amplitude(efference_amplitude)
        history_decays = _history_decays(history_decays)
        history_amplitude = real("history_amplitude", history_amplitude, low=0)
        # the apical compartment (opt-in): the top-down half of every reciprocal pair between
        # regions, normalized to the bottom-up input of the region it lands on
        fraction = real("apical", apical, low=0)
        compartment = _apical(connectome, fraction) if fraction else None
        if compartment is not None and not compartment.synapses.any():
            raise ValueError("apical needs reciprocal pairs between regions; this anatomy has none")
        brain = NeuralGraph(
            connectome,
            learning_neuron_model(dt=1.0 if compartment is None else _APICAL_STEP),
            backend=backend,
            device=device,
            apical=compartment,
        )
        self._composition: dict[str, Any] | None = None  # construction provenance, not parameters
        # ``attention`` is a compose-only recipe marker. Its behavior is fully represented by
        # the learned-code and Recall settings, but the marker keeps the user's construction
        # choice inspectable and round-trippable.
        self.attention = False
        # The learned code's tables are the brain's synapses and biases: the dictionary from
        # the sensory neurons, the lateral weights among the code units and the predictor
        # from the prefrontal region, with the thresholds as the units' biases. The region
        # installs them after its own local learning step, so the lesson's contrast and the
        # dopamine leave them alone.
        self.learned_code: LearnedCode | None = None
        self._code_synapses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        plastic_synapses = plastic_neurons = None
        if learned:
            self.learned_code = LearnedCode(
                len(self.sensory_index), learned["units"], len(self.prefrontal_index),
                # the code's own phase keeps the full step whatever the brain's settle takes
                brain.neuron_model if compartment is None else learning_neuron_model(dt=1.0),
                seed=seed,
                **{name: learned[name] for name in LearnedCode.GENES},
            )
            self._code_synapses = self._learned_code_synapses()
            self._code_forward = self._learned_code_forward()
            # the first module reads the code alone: the sensory neurons project to the code
            # region and nowhere else
            pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
            # the copy pathway's echo suppression (``recall_tokens``) is the one exception:
            # the present observation's tokens reach their own actions
            echo = (
                np.isin(post, self.motor_index) if tokened
                else np.zeros(len(post), dtype=bool)
            )
            if (
                np.isin(pre, self.sensory_index) & ~np.isin(post, self.code_index) & ~echo
            ).any():
                raise ValueError(
                    "a learned code's sensory neurons project to the code region alone"
                )
            brain = self._installed(brain)
            plastic_synapses = np.ones(connectome.synapses, dtype=bool)
            for synapses, _, _ in self._code_synapses.values():
                plastic_synapses[synapses] = False
            plastic_neurons = np.ones(connectome.n, dtype=bool)
            plastic_neurons[self.code_index] = False
        self.learner = Learner(
            brain, list(self.motor_index), learning, slots=slots,
            plastic_synapses=plastic_synapses, plastic_neurons=plastic_neurons,
        )
        self.basal_ganglia = ActorCritic(
            self.learner, list(self.association_index), reward, seed=seed
        )
        self.working_memory: Trace | None = None
        if "prefrontal" in populations:
            self.working_memory = Trace(
                connectome,
                decay=working_memory_decay,
                amplitude=working_memory_amplitude,
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
        # the history hears what the cortex reads: the code region's code with a learned code
        heard_width = len(self.code_index) if learned else len(self.sensory_index)
        if "history" in populations:
            if not history_decays or len(populations["history"]) != len(history_decays) * (
                heard_width
            ):
                raise ValueError(
                    "a history population needs one block of sensory width per history decay"
                )
            self.history = History(
                connectome,
                decays=history_decays,
                amplitude=history_amplitude,
                source="code" if learned else sensory,
                target="history",
            )
        elif history_decays:
            raise ValueError("history_decays need a history population; Brain.compose adds it")
        # The index memory of the stream's moments. Populations named ``recall`` and ``next``
        # without ``recall=True`` are ordinary ones: an instrument may name and drive its own.
        self.recall: Recall | None = None
        if recalled:
            for port in Recall.ports(recalled.get("heads", 1), recalled["next"]):
                for name in port:
                    if name not in populations:
                        raise ValueError(
                            f"recall=True needs a {name!r} population; Brain.compose adds it"
                        )
            # a memory on a code of the observation is given the code, not the observation;
            # with a coded cortex the sensory range is the code
            coded_memory = (self.sparse_code is not None and not self.coded_cortex) or bool(
                learned
            )
            self.recall = Recall(
                connectome, **recalled, source=sensory,
                inputs=self.code_width if coded_memory else 0,
                actions=len(self.motor_index),
            )
            # with ``recall_address_reward`` the actor keeps the address map's eligibility
            self.basal_ganglia.memory = self.recall
        self.hippocampus: FastSynapses | None = None
        # ``memory_choice``: the episodic records that vote in the choice, the store's place
        # records and, with a temporal context, the same records keyed by the stream's
        # temporal context, the fading code of what it sensed and did
        self._episodic: EpisodicRecords | None = None
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
            coded = self._coded_keys
            self.hippocampus = store(np.arange(self.code_width) if coded else self.sensory_index, 0)
            if memory_choice:
                self._episodic = EpisodicRecords(
                    self.hippocampus,
                    store(
                        np.arange(self.code_width + len(self.motor_index))
                        if coded
                        else np.concatenate([self.sensory_index, self.motor_index]),
                        1,
                    )
                    if memory_context
                    else None,
                    width=self.code_width,
                    offsets=self.learner.slot_offsets,
                    share=memory_context,
                    steps=memory_replay,
                    demonstrations=memory_demonstrations,
                )
        self.rng = np.random.default_rng(seed)
        # the validated observations a settle is being prepared for and their code, computed
        # once per moment: ``stimulus`` given that same array reads it, ``_choose`` takes it;
        # with a learned code also the code region's settled state, the whole graph's warm start
        self._coded_moment: (
            tuple[np.ndarray, np.ndarray, BrainState | None, np.ndarray, np.ndarray | None]
            | None
        ) = None
        self._moment: tuple[np.ndarray, np.ndarray] | None = None
        # with a learned code, the code the pending action was keyed on, settled at its moment
        self._moment_code: np.ndarray | None = None
        self._code_settlement: dict[str, Any] | None = None  # the latest code phase's report
        self._prepared: np.ndarray | None = None  # the observations ``learn`` settled
        self.last_learning: dict[str, float] = {}
        self._last_settlement: Mapping[str, Any] | None = None
        # ``live``: the arousal of the stream, and the action it issued that awaits an outcome
        # (observations, action, the value forecast made before the outcome, sampled?, its
        # free state, the brain's own best guess?, the code it was keyed on with a learned
        # code)
        self.arousal: Arousal | None = None if arousal is None else Arousal(arousal)
        self._lived: (
            tuple[np.ndarray, np.ndarray, float, bool, BrainState, bool, np.ndarray | None]
            | None
        ) = None
        self._last_arousal: Mapping[str, Any] | None = None
        # ``live`` with ``hindsight``: the calm transitions of the stream an outcome that wakes
        # the brain credits (drive, free state, action, reward), and the calm action issued
        # last (drive, free state, action) until its outcome arrives
        self._calm: list[tuple[np.ndarray, BrainState, np.ndarray, float]] = []
        # ``act``: its observations and the stream state it left, for the lesson-order warning
        self._acted: tuple[np.ndarray, BrainState | None] | None = None
        self._calm_issued: tuple[np.ndarray, BrainState, np.ndarray] | None = None
        # ``wait``: the free state of the action whose outcome is awaited, and the state the
        # stream has sensed since; any operation that replaces that action's state ends it
        self._awaiting: tuple[BrainState, BrainState] | None = None
        # a terminal forecast's read of the recall memory, held once its end is committed
        self._forecast_read: tuple[Any, ...] | None = None

    @property
    def brain(self) -> NeuralGraph:
        """The current learned dynamics; learning can replace this NeuralGraph instance."""
        return self.learner.brain

    @property
    def memories(self) -> Memories:
        """The present memories by name, in the order the brain reads and writes them, each
        keeping one contract (``cadence.memories``): ``brain.memories["recall"]`` is
        ``brain.recall``; with ``memory_choice`` the episodic records are
        ``brain.memories["episodic"]``."""
        return Memories(
            {
                "working_memory": self.working_memory,
                "efference": self.efference,
                "history": self.history,
                "recall": self.recall,
                "hippocampus": self.hippocampus,
                "episodic": self._episodic,
            }
        )

    @property
    def context_memory(self) -> SynapticMemory | None:
        """The episodic memory's context records (``memory_choice`` with a temporal
        context): ``brain.memories["episodic"].temporal``."""
        return None if self._episodic is None else self._episodic.temporal

    @property
    def memory_choice(self) -> bool:
        """Whether the episodic records vote in the choice."""
        return self._episodic is not None

    @property
    def memory_context(self) -> float:
        """The share of the temporal context the past keeps per moment; 0.0 without
        ``memory_choice``."""
        return 0.0 if self._episodic is None else self._episodic.share

    @property
    def memory_replay(self) -> int:
        """The decisions a reward replays; 0 without ``memory_choice``."""
        return 0 if self._episodic is None else self._episodic.steps

    @property
    def memory_demonstrations(self) -> float:
        """The value a demonstration is recorded at; 0.0 without ``memory_choice``."""
        return 0.0 if self._episodic is None else self._episodic.demonstrations

    def _read_moment(self, x: np.ndarray, code: np.ndarray, place: Any = None) -> Moment:
        """What the memories read of a moment: the store reads under the place key with a
        code (computed here unless given, only where the store reads into the drive), the
        drive itself without one, and nothing under an episodic vote."""
        reads = self.hippocampus is not None and not self.memory_choice
        if reads and self._coded_keys and place is None:
            place = self._placed(x)
        return Moment(x, code, place if self._coded_keys else None, not self.memory_choice)

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

    # -- the learned code's synapses

    def _learned_code_synapses(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The three synapse groups of the learned code in the connectome, each as the
        synapse indices and the local coordinates of their ends: ``dictionary`` (sensory
        ``i`` to code ``j``), ``lateral`` (code ``j`` to code ``k``) and ``predictor``
        (prefrontal ``i`` to code ``j``). Every group must be complete, nothing else may
        lead into the code region, and the sensory and prefrontal neurons must receive no
        synapses, so that the region's fixed point is its own."""
        code = self.learned_code
        assert code is not None
        connectome = self.connectome
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)

        def local(index: np.ndarray) -> np.ndarray:
            out = -np.ones(connectome.n, dtype=np.int64)
            out[index] = np.arange(len(index))
            return out

        sensory, codes, prefrontal = (
            local(self.sensory_index), local(self.code_index), local(self.prefrontal_index)
        )
        into = codes[post] >= 0
        groups = {
            "dictionary": (sensory[pre] >= 0) & into,
            "lateral": (codes[pre] >= 0) & into,
            "predictor": (prefrontal[pre] >= 0) & into,
        }
        expected = {
            "dictionary": code.inputs * code.units,
            "lateral": code.units * (code.units - 1),
            "predictor": code.context * code.units,
        }
        for name, mask in groups.items():
            if int(mask.sum()) != expected[name]:
                raise ValueError(
                    f"a learned code needs its complete {name} synapses: {expected[name]} "
                    f"where the connectome has {int(mask.sum())}; Brain.compose builds them"
                )
        if int(into.sum()) != sum(expected.values()):
            raise ValueError("only the sensory, code and prefrontal neurons may lead into the code")
        boundary = (sensory[post] >= 0) | (prefrontal[post] >= 0)
        if boundary.any():
            raise ValueError(
                "a learned code needs sensory and prefrontal neurons that receive no synapses"
            )
        out = {}
        for name, mask in groups.items():
            synapses = np.flatnonzero(mask)
            source = {"dictionary": sensory, "lateral": codes, "predictor": prefrontal}[name]
            out[name] = (synapses, source[pre[synapses]], codes[post[synapses]])
        return out

    def _learned_code_forward(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The synapses from the code units to the association neurons: the synapse
        indices and the local coordinates of their ends (code ``j``, association ``m``)."""
        connectome = self.connectome
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        codes = -np.ones(connectome.n, dtype=np.int64)
        codes[self.code_index] = np.arange(len(self.code_index))
        targets = -np.ones(connectome.n, dtype=np.int64)
        targets[self.association_index] = np.arange(len(self.association_index))
        synapses = np.flatnonzero((codes[pre] >= 0) & (targets[post] >= 0))
        return synapses, codes[pre[synapses]], targets[post[synapses]]

    def _unit_read(self, x: np.ndarray, drive: np.ndarray, *, memory: bool) -> np.ndarray:
        """The drive with the code's terms from the moment's phase. The code units' drive is
        the phase's potential less their bias: nothing leads into them in the whole graph,
        so they sit at the phase's fixed
        point through the settle, a source to the cortex as the sensory neurons are. And
        the association neurons' term that makes them read the code as
        ``LearnedCode.code_of`` gives it, at unit length, in place of the code units'
        activations: the region's output gain per stream."""
        code = self.learned_code
        assert code is not None
        coded = self._coded_moment
        if coded is not None and coded[0] is x and coded[2] is not None:
            phase = coded[2]
        else:
            phase = self._settle_code(x, context=False, report=False).state
        s = np.atleast_2d(np.asarray(phase.activation, dtype=float))
        if len(s) != len(drive):
            return drive
        out = drive.copy()
        v = np.atleast_2d(np.asarray(phase.v, dtype=float))
        out[:, self.code_index] = v - self.brain.bias[self.code_index]
        unit = code.code_of(s)
        synapses, j, m = self._code_forward
        weights = np.zeros((len(self.association_index), code.units))
        np.add.at(weights, (m, j), self.brain.weights[synapses])
        out[:, self.association_index] += (unit - s) @ weights.T
        return out

    def _installed(self, graph: NeuralGraph) -> NeuralGraph:
        """``graph`` with the learned code's thresholds as the code units' biases and the
        region's synapses at zero: the code is held as a source, and its dictionary, lateral
        weights and predictor act in the region's own phase."""
        code = self.learned_code
        assert code is not None
        efficacy, bias = graph.efficacy.copy(), graph.bias.copy()
        for synapses, _, _ in self._code_synapses.values():
            efficacy[synapses] = 0.0
        bias[self.code_index] = -code.thresholds
        return graph.with_parameters(efficacy=efficacy, bias=bias)

    def _learn_code(self, state: BrainState) -> None:
        """The learned code's step from an accepted moment's settled ``state``: the sensory,
        code and prefrontal activations the settle holds, then the brain's graph takes the
        new tables."""
        code = self.learned_code
        assert code is not None
        s = np.atleast_2d(np.asarray(state.activation, dtype=float))
        coded = self._coded_moment  # the moment's context-free code: the share and the repair
        free = None if coded is None else coded[4]
        activation = s[:, self.code_index]
        last = code.last
        if last is not None and len(np.atleast_2d(last["activation"])) == len(s):
            # the code in context, with the prediction's pull, is what the region learns
            # from, whichever code the cortex reads
            activation = np.atleast_2d(np.asarray(last["activation"], dtype=float))
        if code.learn(
            s[:, self.sensory_index], activation, s[:, self.prefrontal_index], free=free,
        ):
            self.learner.brain = self._installed(self.brain)

    def _step_region(self, halvings: int) -> None:
        """The step of the code's phase: the region's sub-network alone (the rest of the
        brain keeps the model's step and its budgets)."""
        code = self.learned_code
        assert code is not None
        code.restep(halvings)

    @contextmanager
    def _code_attempt(self) -> Iterator[None]:
        """Publish a code phase's numerical policy only with its enclosing operation.

        Attempted settle work and refusal diagnostics remain observable. The previous
        moment, step and live phase are restored when preparation or its consumer refuses.
        """
        code, moment = self.learned_code, self._coded_moment
        held = None if code is None else (code.halvings, code.graph, code.last)
        try:
            yield
        except BaseException:
            self._coded_moment = moment
            if code is not None and held is not None:
                code.halvings, code.graph, code.last = held
            raise

    # -- construction

    @classmethod
    def compose(
        cls,
        inputs: int | Sequence[int],
        actions: int,
        *,
        connectome: Connectome | None = None,
        modules: Sequence[int] = (64,),
        observers: Sequence[int] = (),
        lateral: float | None = None,
        teaching: str | None = None,
        scales: Mapping[str, float] | None = None,
        features: int = 8,
        field: int = 3,
        code: int | None = None,
        learned_code: int | None = None,
        attention: bool = False,
        seed: int = 0,
        slots: int | Sequence[int] = 1,
        **options: Any,
    ) -> Brain:
        """Compose a modular brain with working memory and consolidation.

        ``inputs`` is a vector length, or an image shape ``(height, width)`` or ``(height,
        width, channels)``: the sensory region is then a ``visual_cortex``
        of the image (``features`` maps of ``field`` by ``field`` receptive fields), whose
        feature maps project to the first module as the sensory region does.
        The memories key on the observation code
        (``observation_code``): the observation itself, or with ``code=units`` the brain's
        sparse code of it (each input read through an ON and an OFF channel, every channel by
        two units at least, with positive weights, 32 channels per unit or more, one unit in
        ten active, unit length).
        ``code=units`` makes a vector brain's sensory region that code: every observation is
        coded before the drive, so 12,824 inputs reach the cortex as 1,024 units through
        1,024 x 64 sensory synapses in place of 12,824 x 64, and the memories key on the same
        code; the founder leaves the cortex and the memories on the observation. The code is
        fixed and random and learns nothing. For an image brain
        ``code`` names the memories' code alone, the cortex reading the pixels through the
        visual cortex.
        ``learned_code=units`` (opt-in; a vector brain, without ``code``) places a ``code``
        region of that many neurons between the sensory region and the first module: a
        learned sparse code of the observation, settled in the region's own phase before
        the rest of the brain, under lateral competition through its dictionary and drawn
        toward the prediction the working trace makes of it, then held as a source through
        the whole settle; explicit acts and aroused ``live`` moments learn it by local rules
        (``LearnedCode``). The cortex reads the context-free code, the observation settled
        alone, the heard history holds it, and every memory keys on it at unit length,
        joined to the observation's own identity. ``learned_code_fields`` declares the
        observation's fields; every other part of the region is a constant of it; the
        founder ``learned_code=None`` builds the brain without the region.

        ``attention=True`` is the opt-in single-head contextual-memory recipe for a learned
        code. It requires ``learned_code=units`` and enables Recall with a two-moment ordered
        key, successor values, no age preference and temperature 0.1. The resulting address
        is selected from learned meaning codes before the neural settle and its recalled
        value is held as a source through that same settle. Explicit ``recall_*`` settings
        may tune the recipe, but its window must remain at least two and its successor read
        must remain enabled. This option does not enable episodic action voting and does not
        change the default brain.

        ``modules`` specifies a chain of reciprocally connected processing
        regions. The final region is the association cortex. ``observers``
        optionally adds regions with reciprocal access to all processing and
        motor regions, including earlier observers. Every region participates
        in the same neural settlement; observer regions never run as a separate
        critic or override an already completed answer.

        ``lateral`` is the signed weight between each pair of motor neurons in a
        slot. Left unset it is 0.0 at every menu size: no synapses between the
        motor neurons, the motor and association feedback kept. A negative weight
        makes the actions compete; at -0.5 the undamped free solve oscillates from
        4 actions on, so an act may settle only under the damping fallback and a
        teaching free phase may run out its budget. Select a nonzero weight against
        0.0 on development tasks.

        ``slots`` groups the motor neurons into several readouts that settle
        together, one softmax each: a count of equal groups, or one size per group
        covering ``actions``. ``act`` and ``step`` then return one index per slot,
        and a nonzero ``lateral`` is wired within a slot. One slot is one choice
        over every action.

        Observer wiring is experimental.
        Inputs are fixed external drives; their neural representations can vary.
        ``scales`` maps ``"sensory"``, ``"trace"``, ``"history"`` and ``"recall"`` to the
        initial magnitudes of the projections from the sensory region (founder 1.0), from the
        prefrontal region that feeds the carried interpretation back to the association
        region (1.2: the prefrontal region holds the copy at ``working_memory_amplitude``
        3.0, and learning strengthens the projection where the past predicts the outcome),
        from the heard history (1.2) and from the recall and next regions (1.2); a name left
        out keeps its founder. It is a construction setting; ``retune`` refuses it.
        ``teaching`` is the teaching point of the lesson's learner: ``"founder"`` (steps 0.5
        and 0.05 on the raw contrast, momentum 0.9) or ``"normalized"`` (steps 0.003 and
        0.0003 under a running RMS of forgetting 0.99 and floor 1e-4, momentum 0.9), the
        point for many lessons, as in imitation; ``retune(teaching=...)`` moves a living
        brain to the other. ``learning=LearnerConfig(...)`` replaces the whole
        configuration instead, the expert's way to other steps.
        ``options`` accepts the constructor's memory settings and every other
        configuration field as ``learning_<field>``, ``actor_<field>`` or
        ``arousal_<field>``. ``temperature`` aliases ``learning_temperature``.
        ``arousal=True`` enables the founder arousal config; a mapping patches
        those founders. Supplied config objects remain full replacements.
        A positive ``efference_amplitude`` adds the efference copy: one ``efference``
        neuron per motor neuron, driven by the fading one-hot of the command the
        stream issued (``efference_decay``) and read by the association region
        through a plastic projection of founder scale 12. The founder
        value zero builds the brain without it, the control to select the copy against.
        A nonempty ``history_decays`` adds the heard history: a ``history`` region of one
        block of ``inputs`` neurons per decay, each block a fading trace of the observations
        as heard (the values the sensory neurons were stimulated with, before settling) at
        its own decay, held at ``history_amplitude`` (3.0) and read by the association
        region through a plastic projection of initial magnitude ``scales["history"]``
        (1.2), initialized like the working trace's. After each act the blocks take in the
        observation the brain acted on, so at a decision the history holds the past and the
        sensory drive the present. The founder ``history_decays=()`` builds the brain
        without it.
        ``recall=True`` adds the recall memory, an index of the stream's own recent moments:
        a ``recall`` and a ``next`` region (one neuron per value coordinate each), read by
        the association region through plastic projections of initial magnitude
        ``scales["recall"]`` (1.2), appended after any history. After each act, and after each
        moment ``wait`` senses, the moment is written as one slot per stream (``Recall``):
        its key the event heard (``recall_heard`` 1.0 times the observation code, the
        observation itself or the brain's sparse code of it), its value the event, and the
        slot before learns it as its next.
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
        memory, each a gene described and saved only where it differs from its founder and a
        runtime one but the key and the window: ``recall_address_step="toward"`` steps the
        address by the named keys less the mixture it attended to, in place of the founder's
        ``"lms"`` step toward the named keys; a number ``recall_null`` adds a null slot of logit
        ``recall_null / recall_temperature`` that reads nothing; and
        ``recall_address_reward`` above zero lets reward train the address map, each sampled
        act keeping the step its lesson would take as the map's eligibility and each outcome
        stepping the map by that rate times the dopamine times the trace: the eligibility
        carries the lesson's rate, so at 1.0 a unit of dopamine moves the map by one lesson's
        step, and ``recall_address=0`` leaves the map as it is under reward too.
        When an episode ends the stream's slots are emptied with its working trace;
        ``Brain.reset(keep=("slots",))`` keeps the slots across a reset.
        ``recall_window`` k above 1 (construction, with ``recall_window_decay`` d in (0, 1])
        makes every key an ordered window of the stream's last k events, ``[heard * x_t, d *
        heard * x_(t-1), ..., d^(k-1) * heard * x_(t-k+1)]``, each position on coordinates of
        its own, zeros before a stream's first moment and after an episode ends: a key of
        ``k * code`` values, the address and the map's outputs as wide.
        The founder ``recall=False`` builds the brain without it.
        Every head of a composed recall memory also reads each slot's age: ``recall_position``
        P (founder 32, construction) gives ages below P a bucket each and older ones one
        bucket per doubling, and each head adds a term per bucket read from the moment as its
        address is, so what a moment hears can choose how far back the head looks. The terms
        start at zero and lessons teach them at ``recall_position_rate`` (runtime, founder
        0.1), whatever ``recall_address`` is; ``recall_position=0`` builds none. A brain
        saved without a position term loads without one.
        ``recall_boundary=True`` (runtime, founder off) gives each stream that begins a
        boundary slot first, a silent moment whose next the stream's first event teaches, so
        the start of an episode has a predecessor as every later event has.
        ``recall_copy`` above 0 (construction, founder 0.0) adds the copy pathway: plastic
        synapses from every head's read ports into the motor region, appended last and drawn
        at that magnitude, learned by the settle's contrast. ``recall_tokens`` (construction,
        founder None) declares per action the observation coordinate that is the same token
        (``-1`` for none; a brain without a code, or a learned code joined to the
        observation): the pathway then starts from each recall port as that identity at
        ``recall_copy``, from each next port at zero, and from the sensory region as the
        negative identity, so a token the read delivers drives its own action unless the
        present observation already holds it.
        ``recall_action=True`` (construction, founder off) writes the executed command, one
        value per motor neuron, into every slot's key and value, read on ``<port>_action``
        regions whose synapses into the association start at zero; the address and position
        maps also read the command of the moment before, so a newborn reads and settles as
        one without it.
        ``recall_capacity``, ``recall_next``, ``recall_window``,
        ``recall_window_decay``, ``recall_heads`` (several held heads over the one bank,
        each with its own port and address map; ``recall_compete``, a runtime option with
        founder True, makes them read in turn, each taking of a slot at most the share the
        heads before it left), ``recall_position``, ``recall_copy``,
        ``recall_tokens`` and ``recall_action`` are construction settings; the keys, the
        values and the address map follow the observation code's width, a brain's ``code``
        among the ways to bound it. The settings of 0.84.3 that this version removed
        (``recall_write``, ``recall_key``, ``recall_episodes``, ``recall_value``,
        ``recall_context``, ``recall_context_decay`` and ``recall_recency=None``) are refused
        by name; the CHANGELOG gives each one's reading.
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
        situation, the one that left it, while a decision whose own outcome was known stays
        on the path), through the decisions whose own outcome was nothing, back to the last
        known outcome, discounted per step and up to the discounted reward. A reward that
        ends an attempt (``done``) is replayed whole into both kinds of record. In a
        continuing life a reward is replayed in proportion to its surprise, the reward less
        what the memory expected of the paid decision: into each decision's context record,
        and into its place record as far as it was an exploration rather than the brain's
        best guess. Nothing has to mark the end of an attempt. Keys are separated
        (``memory_expansion`` 1,024 cells, ``memory_winners`` 24; a brain with a fixed code keys
        on that code as it is, with no separation, and a learned code's keys are separated
        as an observation's are). ``memory_fan_in`` above zero makes the separation
        sparse: each of its cells reads that many of the key's values, chosen at random, so
        its size follows the cells and the fan-in alone; the founder 0 keeps it dense, one
        weight per cell and key value. Dense or sparse, the separation is a fixed random
        projection. Each setting can be passed beside ``memory_choice=True``. The founder
        ``memory_choice=False`` keeps the consolidating hippocampus, keyed by the observation
        code and read into the motor drive; ``episodic=False`` builds none. Where the history
        does not matter, compare the episodic memory against the founder on the task.
        ``connectome`` wraps a custom anatomy with these founders, the episodic founders
        filled under ``memory_choice=True`` as here: its ``sensory`` (or ``visual/input``)
        population holds the inputs (the code's units with ``code=units``), its ``motor``
        population the actions, and it brings its own layout, so ``modules``, ``observers``,
        ``lateral``, the projection scales, ``features`` and ``field`` stay at their founders.
        ``Genome.composed`` returns the layout developed here, to edit or evolve;
        ``Genome``, ``Region`` and ``Projection`` give custom ports, sparsity and named wiring.
        """

        if not isinstance(attention, bool):
            raise ValueError("attention must be boolean")
        if attention:
            if not _learned_code_units(learned_code):
                raise ValueError("attention=True needs learned_code=units")
            if options.get("recall", True) is not True:
                raise ValueError("attention=True needs recall enabled")
            options.setdefault("recall", True)
            options.setdefault("recall_window", 2)
            options.setdefault("recall_next", True)
            # one head reads with its key alone; several heads learn their addresses, at a
            # temperature at which the address lesson does not lock onto a first selection
            single = options.get("recall_heads", 1) == 1
            options.setdefault("recall_address", 0.0 if single else 0.01)
            options.setdefault("recall_recency", 0.0)
            options.setdefault("recall_temperature", 0.1 if single else 0.25)

        genome, layout = _layout(
            inputs, actions, options, modules=modules, observers=observers, lateral=lateral,
            scales=scales, features=features, field=field, code=code,
            learned_code=learned_code, seed=seed, slots=slots,
        )
        units, learned = layout["units"], layout["learned"]
        if attention and (
            layout["recalled"]["window"] < 2 or not layout["recalled"]["next"]
        ):
            raise ValueError(
                "attention=True needs recall_window at least 2 and recall_next=True"
            )
        if teaching is not None:
            if "learning" in options:
                raise ValueError(
                    "teaching sets the teaching point of the composed founders and learning= "
                    "replaces the whole configuration: pass one of them"
                )
            options["learning"] = taught(teaching, _learning())
        if "episodic" in options and not isinstance(options["episodic"], bool):
            raise ValueError("episodic must be boolean")
        if options.get("episodic", True) and options.get("memory_choice") is True:
            for name, value in _EPISODIC_FOUNDERS.items():
                # a fixed code's memories key on the code as it is: no separation of their own;
                # the hippocampus separates the patterns of a learned code as of an observation
                options.setdefault(name, 0 if units and name == "memory_expansion" else value)
        custom = connectome is not None
        if connectome is None:
            connectome = develop(genome, seed=seed)
        else:
            # a custom anatomy brings its own layout; compose gives it the founders
            if (
                tuple(modules) != (64,)
                or tuple(observers) != ()
                or lateral is not None
                or scales is not None
                or (features, field) != (8, 3)
            ):
                raise ValueError(
                    "a connectome brings its own layout: modules, observers, lateral, "
                    "scales, features and field are compose's layout settings"
                )
            populations = connectome.populations
            sensory = populations.get("sensory", populations.get("visual/input", ()))
            needed = units if layout["coded_cortex"] else layout["inputs"]
            motor = len(populations.get("motor", ()))
            if len(sensory) != needed or motor != layout["actions"]:
                raise ValueError(
                    f"the connectome's sensory and motor populations hold {len(sensory)} and "
                    f"{motor} neurons, where these inputs need {needed} and these actions "
                    f"{layout['actions']}"
                )
        result = cls(
            connectome, seed=seed, slots=slots, code=units,
            inputs=layout["inputs"] if layout["coded_cortex"] else None,
            learned_code=learned or None, **options,
        )
        result.attention = attention
        if custom:
            return result  # a custom anatomy records no composition
        result._composition = {
            "inputs": layout["inputs"],
            "actions": layout["actions"],
            "modules": list(layout["widths"]),
            "observers": list(layout["observer_widths"]),
            "slots": [int(width) for width in result.learner.slot_sizes],
            "lateral": layout["lateral"],
            "sensory_scale": layout["sensory_scale"],
            "working_memory_scale": layout["working_memory_scale"],
            "seed": int(seed),
        }
        if layout["heard"]:
            result._composition["history_scale"] = layout["history_scale"]
        if layout["recalled"]:
            result._composition["recall_scale"] = layout["recall_scale"]
        if layout["image"] is not None:
            result._composition["image"] = list(layout["image"])
        if layout["coded_cortex"]:
            result._composition["code"] = units
        if learned:
            result._composition["learned_code"] = learned
        if attention:
            result._composition["attention"] = True
        if result.brain.apical is not None:
            result._composition["apical"] = result.brain.apical.fraction
        return result

    def retune(self, **genes: Any) -> Brain:
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
        choices. ``retune`` resets nothing; ``brain.arousal.reset()`` resets the arousal level
        and reward references and keeps its age and work counters. Returns this same brain.
        """
        if "reset_arousal" in genes:
            raise ValueError(
                "reset_arousal was removed in this version: call brain.arousal.reset()"
            )
        options = dict(genes)
        construction = {
            "inputs", "actions", "modules", "observers", "lateral", "scales", "slots", "seed",
            "episodic", "backend", "device", "connectome", "history_decays",
            *_EPISODIC_SETTINGS, "recall", *(f"recall_{name}" for name in Recall.CONSTRUCTION),
            *(f"recall_{name}" for name in Recall.CONSTRUCTION_OPTIONS), "code",
            "learned_code", *(f"learned_code_{name}" for name in LearnedCode.CONSTRUCTION),
            "attention", "apical",
        }
        for name in options:
            refusal = rescaled(name)
            if refusal is not None:
                raise ValueError(refusal)
            if name in construction:
                raise ValueError(f"{name} is a construction setting; retune preserves the brain")
        point = options.pop("teaching", None)
        if point is not None and "learning" in options:
            raise ValueError(
                "teaching sets the teaching point and learning= replaces the whole "
                "configuration: pass one of them"
            )
        learning = options.pop("learning", self.learner.config)
        # a teaching point's fields join the learning genes, so the configuration is checked
        # once with all of them (teaching="founder" with learning_normalize_relative=0 is valid)
        pointed = {} if point is None else teaching_fields(point)
        reward = options.pop("reward", self.basal_ganglia.config)
        current_arousal = None if self.arousal is None else self.arousal.config
        arousal = options.pop("arousal", current_arousal)
        if "arousal" in genes and (self.arousal is None or arousal is None):
            raise ValueError("retune cannot enable or disable arousal; compose it at construction")

        memory_fields = {
            "working_memory_decay": (self.working_memory, "decay"),
            "working_memory_amplitude": (self.working_memory, "amplitude"),
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
            elif name in ("efference_amplitude", "history_amplitude"):
                value = real(name, value, low=0)
            else:
                value = real(name, value)
            memory_changes.append((component, field, value))
            if component is self.hippocampus and self.context_memory is not None:
                memory_changes.append((self.context_memory, field, value))
        for name in [name for name in options if name.startswith("recall_")]:
            setting = name[len("recall_"):]
            if setting in Recall.RETIRED:
                raise ValueError(Recall.removed(setting))
            if setting not in Recall.RUNTIME and setting not in Recall.OPTIONS:
                raise TypeError(f"unknown brain setting: {name!r}")
            if self.recall is None:
                raise ValueError(f"{name} requires its existing memory component")
            memory_changes.append(
                (self.recall, setting, Recall.setting(setting, options.pop(name)))
            )
        for name in [name for name in options if name.startswith("learned_code_")]:
            setting = name[len("learned_code_"):]
            if setting in LearnedCode.REMOVED:
                LearnedCode.setting(setting, options[name])  # refused by name
            if setting not in LearnedCode.RUNTIME:
                raise TypeError(f"unknown brain setting: {name!r}")
            if self.learned_code is None:
                raise ValueError(f"{name} requires its existing learned code")
            value = LearnedCode.setting(setting, options.pop(name))
            memory_changes.append((self.learned_code, setting, value))
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
        learning = configured(learning, {**pointed, **learning_genes}, "learning")
        if learning.initial_damping and (
            self.brain.backend != "cpu" or self.brain.precision not in (None, "float64")
        ):
            raise ValueError("nonzero initial_damping requires CPU float64")
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
        if self.arousal is not None:
            assert arousal is not None
            if arousal.hindsight != self.arousal.config.hindsight:
                # held calm moments belong to the hindsight they were held under
                self._calm, self._calm_issued = [], None
            self.arousal.config = arousal
        for component, field, value in memory_changes:
            setattr(component, field, value)
        if self.learned_code is not None and any(
            component is self.learned_code for component, _, _ in memory_changes
        ):
            # the region's sub-network and the graph take the tables again
            self.learned_code.install()
            self.learner.brain = self._installed(self.brain)
        if self.recall is not None and not self.recall.address_reward:
            # reward leaves the address map as it is: the actor keeps no eligibility of it, and
            # the brain reports and saves as one composed without it
            agent.trace_memory = agent._pending_memory = None
        return self

    def describe(self) -> dict[str, Any]:
        """Describe the brain without changing live state: four sections.

        - ``layout``: the inputs, actions, regions, slots, neurons, synapses and populations;
          ``code``, what the memories key on; ``learned_code``, a learned code's settings and
          counts; and ``composition``, the construction choices of a composed brain (None for
          a custom anatomy), provenance rather than a description of learned weights.
        - ``genes``: every setting by the name ``compose`` and ``retune`` take, the teaching
          point as ``teaching`` (``"custom"`` for steps that ``learning=LearnerConfig(...)``
          set outside both points, whose four effective settings are then under ``learning``).
        - ``memories``: one entry per member of ``brain.memories``, its ``to_dict()``.
        - ``pending``: whether an issued action owns the next real outcome.

        The returned mapping is detached and JSON serializable. No lazy state is initialized,
        no random values are drawn and no settlement is performed.
        """
        learning = self.learner.config.to_dict()
        actor = self.basal_ganglia.config.to_dict()
        arousal = None if self.arousal is None else self.arousal.config.to_dict()
        steps = {
            name: learning[name] for name in ("eta", "eta_bias", "normalize", "normalize_floor")
        }
        teaching = next(
            (point for point, values in TEACHING.items() if values == steps), "custom"
        )
        # the nudge is a standalone Learner's choice, a composed brain's the cross-entropy; the
        # steps and their normalization are the teaching point's
        genes: dict[str, Any] = {"teaching": teaching}
        if teaching == "custom":
            genes["learning"] = steps
        genes.update(
            {
                f"learning_{name}": value
                for name, value in learning.items()
                if name != "nudge" and name not in steps
            }
        )
        if self.brain.apical is not None:  # described where it differs from its founder, 0
            genes["apical"] = self.brain.apical.fraction
        genes.update({f"actor_{name}": value for name, value in actor.items()})
        if arousal is not None:
            genes.update({f"arousal_{name}": value for name, value in arousal.items()})
        for prefix, component, fields in (
            ("working_memory", self.working_memory, ("decay", "amplitude")),
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
        learned = self.learned_code
        if learned is not None:
            genes.update({f"learned_code_{name}": value for name, value in learned.genes().items()})
        if self.attention:
            genes["attention"] = True
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
                # what the memories key on: the observation, the brain's sparse code of it or
                # its learned code, what the cortex reads, and the episodic separation of an
                # uncoded key
                "code": {
                    "kind": "learned" if learned is not None
                    else ("observation" if self.sparse_code is None else "code"),
                    "inputs": int(self.inputs),
                    "width": int(self.code_width),
                    "units": int(learned.units) if learned is not None
                    else (0 if self.sparse_code is None else int(self.sparse_code.expansion)),
                    "fan_in": 0 if self.sparse_code is None else int(self.sparse_code.fan_in),
                    "winners": 0 if self.sparse_code is None else int(self.sparse_code.winners),
                    "cortex": "code" if self.coded_cortex or learned is not None
                    else "observation",
                    "separator": None
                    if self.hippocampus is None or self.hippocampus.separator is None
                    else ("sparse" if self.hippocampus.separator.fan_in else "dense"),
                },
                **({} if learned is None else {"learned_code": learned.to_dict()}),
                "composition": self._composition,
            },
            "genes": genes,
            "memories": {name: memory.to_dict() for name, memory in self.memories.items()},
            "pending": self.pending_feedback,
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
        unit in ten kept and the row scaled to unit length; with a learned code the
        rectified code the region settles to on the observations alone, without a context,
        at unit length, the key of every memory."""
        return self._code(self._observations(observations), context=False)

    def _code(
        self, x: np.ndarray, *, context: bool = True, trace: Trace | None = None
    ) -> np.ndarray:
        """The observation code of validated observations; the observations themselves,
        the same array, without a code. A learned code settles here on the observations
        alone, whatever ``context`` and ``trace`` say: the cortex and the memories read the
        context-free code."""
        if self.learned_code is None:
            return x if self.sparse_code is None else _unit_rows(self.sparse_code.code(x))
        phase = self._settle_code(x, context=False, trace=trace)
        return self._moment_key(x, self.learned_code.code_of(phase.state.activation))

    def _prepare(
        self, x: np.ndarray, *, context: bool = True, trace: Trace | None = None
    ) -> np.ndarray:
        """The code of the moment being settled, computed once and kept with the
        observations (``_coded_moment``): ``stimulus`` given that same array reads it, and
        with a learned code the region's settled state is the whole graph's warm start. A
        moment the region learns from settles twice: in the stream's context (the working
        trace, or ``trace``), the code the region learns from, and on the observations alone
        (reported as ``context_free_steps``), the code the cortex reads and every memory keys
        on; the record holds that key twice, as the moment's key and as its place key."""
        if self.learned_code is None:
            key = self._code(x)
            self._coded_moment = (x, key, None, key, None)
            return key
        code = self.learned_code
        phase = self._settle_code(x, context=context, trace=trace)
        key = self._moment_key(x, code.code_of(phase.state.activation))
        place, free_code, held = key, None, phase.state
        # the context-free code: what the cortex reads and every memory keys on, and what the
        # thresholds' share and the repair read at a moment the region learns from
        if context:
            free = self._settle_code(x, context=False, report=False)
            free_code = np.maximum(np.asarray(free.state.activation, dtype=float), 0.0)
            assert code.last is not None
            code.last["free"] = free_code
            place = self._moment_key(x, code.code_of(free_code))
            key, held = place, free.state
            assert self._code_settlement is not None
            self._code_settlement["context_free_steps"] = int(free.steps)
        self._coded_moment = (x, key, held, place, free_code)
        return key

    def _moment_key(self, x: np.ndarray, code: np.ndarray) -> np.ndarray:
        """The key the memories use for observations ``x`` whose learned code is ``code``
        (the rectified settled code at unit length): the code joined to the observation's
        own identity, ``unit([sqrt(1 - w) code, sqrt(w) unit(x)])`` with ``w`` the
        identity weight 0.9 (``LearnedCode.IDENTITY``), so that a binding stays readable by
        the identity part while the code part moves as the region learns; the code alone
        for an observation wider than the code (``LearnedCode.joins``). The key takes the
        norm the observation itself would have as a key: the recall's temperature reads the
        keys' dot products at that scale."""
        region = self.learned_code
        if region is None:
            return code
        key = np.atleast_2d(code)
        if region.identity:
            weight = region.identity
            identity = _unit_rows(np.atleast_2d(np.asarray(x, dtype=float)))
            key = _unit_rows(
                np.concatenate([np.sqrt(1.0 - weight) * key, np.sqrt(weight) * identity], axis=1)
            )
        key = key * np.linalg.norm(np.atleast_2d(np.asarray(x, dtype=float)), axis=1)[:, None]
        return np.asarray(key)

    def _settle_code(
        self, x: np.ndarray, *, context: bool = True, trace: Trace | None = None,
        report: bool = True,
    ) -> Equilibrium:
        """The code phase of a moment's settle: the learned code's region settled under the
        dictionary's read of the sensory activation the observations give and, with
        ``context``, the predictor's read of the prefrontal activation the working trace
        (or ``trace``, an imagined branch's) gives, warm from the stream's code, by the
        brain's solver policy to the learner's tolerance within its free budget. A phase
        that does not qualify is settled once more at the step the dictionary's coupling
        allows over every state (``LearnedCode.rest_stability``), and refuses the moment before
        anything changes when it does not qualify there either. With ``report``
        the phase is the moment's: it sets the brain's step and the settlement report."""
        code = self.learned_code
        assert code is not None
        cfg = self.learner.config
        if cfg.tolerance is None:
            raise ValueError("Brain answers require a finite residual tolerance")
        tolerance = float(cfg.tolerance)
        graph = self.brain
        model, bias = graph.neuron_model, graph.bias
        sensory = model.activation(x * model.stimulus_amplitude + bias[self.sensory_index])
        z = None
        if context:
            held = self.working_memory if trace is None else trace
            drive = np.zeros((len(x), len(self.prefrontal_index)))
            if held is not None and len(held.trace) == len(x):
                drive = held.amplitude * held.trace
            z = model.activation(drive + bias[self.prefrontal_index])
        warm = None
        current = self._activity()
        if current is not None and len(np.atleast_2d(current.v)) == len(x):
            warm = BrainState(
                np.atleast_2d(current.v)[:, self.code_index].copy(),
                np.atleast_2d(current.activation)[:, self.code_index].copy(),
                np.atleast_2d(current.adaptation)[:, self.code_index].copy(),
                current.steps,
            )
        stimulus = code.stimulus(sensory, z)

        def budget() -> int:
            # the free budget at the phase's step: the region's phase scales its own
            return self.learner.config.free_steps << code.halvings

        phase = code.settle(stimulus, warm, budget=budget(), tolerance=tolerance)
        if not np.all(phase.qualified):
            # once more at the step the dictionary's coupling allows over every state the
            # region can reach; the step is left as it was when this fails too
            previous, rest = code.halvings, code.rest_stability()
            if rest > previous:
                self._step_region(rest)
                fallback = code.settle(stimulus, warm, budget=budget(), tolerance=tolerance)
                # Both attempts are work of this phase, even though the fallback starts
                # from the original warm state. Keep its actual final state and residual.
                phase = Equilibrium(
                    fallback.state, fallback.residual, phase.steps + fallback.steps,
                    fallback.tolerance,
                    residual_checks=phase.residual_checks + fallback.residual_checks,
                    damping_halvings=fallback.damping_halvings,
                    stagnation_checks=phase.stagnation_checks + fallback.stagnation_checks,
                )
                if not np.all(phase.qualified):
                    self._step_region(previous)
        if not np.all(phase.qualified):
            self._code_settlement = {
                "code_steps": int(phase.steps),
                "code_residual": float(np.max(phase.residual)),
                "code_halvings": int(code.halvings),
            }
            raise RuntimeError(
                f"the code region did not settle within {budget()} steps: "
                f"residual={float(np.max(phase.residual)):.6g}, tolerance={tolerance:g}; "
                "no action issued"
            )
        if not report:
            return phase
        # the step the code's phase integrates at: the one the settled code's competition is
        # stable under (the code is held as a source, so the rest of the brain keeps its step)
        self._step_region(code.stability(phase.state.v))
        self._code_settlement = {
            "code_steps": int(phase.steps),
            "code_residual": float(np.max(phase.residual)),
            "code_halvings": int(code.halvings),
        }
        code.last = {
            "sensory": sensory, "context": z, "activation": np.asarray(phase.state.activation),
            "prediction": None if z is None else code.prediction(z),
        }
        return phase

    def _warm(self, drive: np.ndarray, state: BrainState | None) -> BrainState | None:
        """The warm start of a settle under ``drive`` with the learned code's phase in it:
        ``state`` (rest without one) with the sensory and prefrontal potentials at the fixed
        points their drives give them and the code units at the phase's, so the code units
        receive the input they settled under from the first sweep; ``state`` itself without
        the region, or without a phase for this batch."""
        coded = self._coded_moment
        if (
            self.learned_code is None
            or coded is None
            or coded[2] is None
            or len(np.atleast_2d(coded[2].v)) != len(drive)
        ):
            return state
        batch, n = len(drive), self.connectome.n
        if state is None or len(np.atleast_2d(state.v)) != batch:
            v, a, steps = np.zeros((batch, n)), np.zeros((batch, n)), 0
        else:
            v = np.array(np.atleast_2d(state.v), dtype=float, copy=True)
            a = np.array(np.atleast_2d(state.adaptation), dtype=float, copy=True)
            steps = state.steps
        bias = self.brain.bias
        for index in (self.sensory_index, self.prefrontal_index):
            v[:, index] = drive[:, index] + bias[index]
        v[:, self.code_index] = np.atleast_2d(coded[2].v)
        a[:, self.code_index] = np.atleast_2d(coded[2].adaptation)
        return BrainState(v, self.brain.neuron_model.activation(v), a, steps)

    def _key_of(self, observations: np.ndarray, code: np.ndarray | None) -> np.ndarray:
        """The key a decision was taken under: its code as settled at its moment, or the
        observations' code where a fixed code gives the same at any moment."""
        return self._code(observations) if code is None else code

    def _sensed(self, x: np.ndarray, code: np.ndarray) -> np.ndarray:
        """What the sensory neurons are given: the observations, or with a coded cortex their
        code."""
        return code if self.coded_cortex else x

    def _heard(self, x: np.ndarray, code: np.ndarray) -> np.ndarray:
        """What the heard history takes in: what the cortex reads, the observations or their
        code; with a learned code whose key joins the observation's identity, the key's
        code part at unit length (the identity part is the observation the history would
        hear without a code)."""
        region = self.learned_code
        if region is not None:
            part = np.atleast_2d(code)[:, : region.units] / np.sqrt(1.0 - region.identity)
            norm = np.linalg.norm(np.atleast_2d(np.asarray(x, dtype=float)), axis=1)
            return np.asarray(part / np.where(norm > 0.0, norm, 1.0)[:, None])
        return code if self.coded_cortex else x

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        """The drive of a batch of observations: the sensory neurons stimulated, plus the
        working memory, the efference copy, the heard history, the recall memory and the
        hippocampal recall when ``memory`` is set. An episodic memory that votes in the
        choice (``memory_choice``) is not read into the drive. The recall memory addresses
        the moment here and keeps the address for the moment's write and lesson. Both
        memories read the observation code; a learned code settles in the stream's context
        with ``memory``, and on the observations alone without it."""
        coded = self._coded_moment  # the moment's code, when this is its settle
        if coded is not None and coded[0] is observations:
            return self._stimulus(coded[0], coded[1], memory=memory)
        x = self._observations(observations)
        return self._stimulus(x, self._code(x, context=memory), memory=memory)

    def _stimulus(self, x: np.ndarray, code: np.ndarray, *, memory: bool = True) -> np.ndarray:
        """The drive of validated observations and their code."""
        drive = np.zeros((len(x), self.connectome.n))
        drive[:, self.sensory_index] = (
            self._sensed(x, code) * self.brain.neuron_model.stimulus_amplitude
        )
        region = self.learned_code
        if region is not None:
            drive = self._unit_read(x, drive, memory=memory)
        if memory:
            # every memory in order; the recall memory's address, before the settle, reads the
            # stream's slots
            drive = self.memories.stimulus(drive, self._read_moment(x, code))
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
        """Teach independent samples; return qualified training accuracy after each epoch.

        ``fit`` begins with ``reset(keep=("arousal",))``: every stream starts afresh and the
        arousal state stays as it is, since a control on independent samples does not change
        a life."""
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
        self.reset(keep=("arousal",))
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
        from .brain import _copy_state

        cfg = self.learner.config
        brain = self.brain
        state = self._warm(drive, state)  # with a learned code: its phase in the warm start
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
            apical=brain.apical,
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
        report: dict[str, Any] = {
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
        if self.learned_code is not None and self._code_settlement is not None:
            report.update(self._code_settlement)  # the code phase this settle started from
        self._last_settlement = MappingProxyType(report)
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
        with self._code_attempt():
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

    @_operation
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
        current observation. The reward is credited to the action ``step`` issued.
        Where a teacher's or caregiver's action ran in its place, the issued action has
        no outcome: replace it with ``act(observations, greedy=True)``, which learns
        nothing from it, and teach the shown action with
        ``lesson(observations, labels)``; the next ``step`` then takes no reward.
        Reported to ``step``, the shown action's reward, or the zero of an omitted one,
        would credit an action that did not run. A moment in which a teacher acts can
        also be taught with ``lesson`` followed by a greedy ``act``, in place of
        ``step``. The first call has no previous action to reward. Later calls retain
        the batch's row identities; call ``reset`` before starting different streams.
        Salience is nonnegative, per row, and defaults to absolute reward for consolidation.
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

    @_operation
    def lesson(self, observations: Any, labels: Any) -> dict[str, float]:
        """Teach the labels of the present observations without acting on them.

        This is the lesson ``step(..., teacher=labels)`` gives before its act: the drive reads
        every memory the act reads, under ``learning_surprise`` the rows are judged by the act
        the brain would issue, its complete greedy choice on the settle the act reads (with
        ``memory_choice`` the readout with the episodic vote), and with a recall memory the
        address lesson follows at the taught rows. A row the brain would answer as shown is
        no surprise and gets no lesson, also where its episodic memory turns a wrong cortical
        answer into the right act: consolidating the cortex under an answer the memory
        supports is another objective than teaching by surprise, and a lesson-only curriculum
        leaves the cortex untaught on those rows. An episodic memory that remembers
        demonstrations records the taught rows. Returns the teaching report (with a
        recall memory, ``address_rows``, ``address_error`` and ``address_refused`` too, and
        with its position term ``position_rows`` and ``position_refused``; with several heads
        the rows and refusals are summed over them), which
        ``last_learning`` holds with the ``demonstration_`` prefix; a refused phase raises
        ``LearningPhaseError`` and leaves the parameters and memories as they were. The next
        ``act`` settles afresh under the changed parameters and writes the moment. The streams
        are the live ones; an action whose sampled outcome is pending is refused, since its
        eligibility belongs to the parameters before the lesson: report the outcome first
        (``step`` or ``learn``).
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
        acted = self._acted
        if (
            self.recall is not None
            and acted is not None
            and acted[1] is current
            and np.array_equal(acted[0], x)
        ):
            warnings.warn(
                "lesson on the observations the brain just acted on: the act already wrote "
                "this moment to the recall memory, so the lesson reads the moment itself, its "
                "successor still empty. Teach before acting, as step(..., teacher=labels) "
                "orders them.",
                RuntimeWarning,
                stacklevel=3,  # the caller, past the operation wrapper
            )
        self.last_learning = {}
        return dict(self._lesson(x, labels))

    def _lesson(self, x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """The lesson of validated observations and labels; ``last_learning`` takes its report
        with the ``demonstration_`` prefix."""
        with self._code_attempt():
            return self._teach(x, labels)

    def _teach(self, x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """The provisional code phase and the local demonstration consuming it."""
        from .learning import LearningPhaseError

        batch = len(x)
        self._prepare(x)
        drive = self.stimulus(x)
        code = self._coded(x)
        place = self._placed(x)
        key = self._decision_key(code)  # of the decision the lesson is shown for
        answer = None
        if self.learner.config.surprise:
            # the brain's own answer, the act it would issue: its complete greedy choice on the
            # settle its act reads, warm from the stream, with the episodic vote under the
            # decision's own key; a row that answer already gives as shown is no surprise
            answer = self._greedy(
                self._qualified(drive, self._activity(), operation="act"), place, key
            )
        try:
            # the lesson's free phase starts from rest, with a learned code's phase in it
            lesson, teaching = self.learner.step(
                drive, labels, warm=self._warm(drive, None), answer=answer
            )
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
        # lesson taught, those whose complete answer differed from the label.
        shown = np.ones(batch, bool) if lesson.taught is None else lesson.taught
        episodic = self._episodic
        if episodic is not None and episodic.demonstrations and shown.any():
            value = np.full(batch, episodic.demonstrations)
            episodic.remember(place, key, labels, value, shown)
        self._prepared = None
        self.basal_ganglia._drive = None
        self.last_learning["demonstrations"] = float(batch)
        return teaching

    @_operation
    def live(
        self, observations: Any, *, reward: Any = None, done: Any = None,
        salience: Any = None, decision_id: Any = None,
    ) -> np.ndarray:
        """One moment of a continuing life: routine while outcomes match, repair when not.

        ``reward`` and ``done`` describe the preceding action, as in ``step``. The
        brain's ``arousal`` decides what this moment costs. Calm, it answers with
        the greedy choice of one qualified settle: no eligibility phases, no
        learning, no write to the associative memory, no parameter changes; the
        eligibility of earlier sampled actions fades by one step, as time passes. The
        working trace, the heard history and the recall memory's slots advance at every
        accepted moment, calm or aroused; the recall memory's address map changes in lessons
        and, with ``recall_address_reward``, by the outcomes of sampled actions. Its
        forecast for the preceding action was the critic's value when it acted; the outcome
        is measured against it with the value of the present state, and a surprising outcome
        or a reward below what life usually pays raises the arousal (``Arousal`` gives the
        law). Only the outcome of the
        brain's own best guess can surprise it: the act it would issue greedily, read when the
        action is issued, with ``memory_choice`` the readout with the episodic vote, so a
        greedy act the vote decided is its own. Aroused, the brain samples its
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
        without reward, done or salience. Readings of the moment are in ``last_arousal``.

        An omitted reward is a zero outcome, not a missing one. When the outcome
        comes later than the next observations, sense them with ``wait`` and report
        the outcome here once it is observed: it is measured against the forecasts
        made when the action was chosen, and credited to that action's eligibility and
        situation as an immediate outcome would be. ``decision_id`` names the action
        the outcome belongs to (``Brain.decision_id``); any other value is refused
        with ``ValueError`` before anything changes.

        ``salience`` is an optional nonnegative one-row importance of that actual outcome,
        as in ``step``. When the outcome is learned, a positive mark makes even a zero
        outcome known to the episodic records without changing its reward or closing replay.
        It also sets consolidation's importance when an outcome wakes a routine brain.
        A mark does not itself wake the brain or permit a calm write. It needs a preceding
        action and is validated before feedback or forecasting changes anything.
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
        if not sampled and not routine and (
            reward is not None or done is not None or salience is not None
        ):
            raise RuntimeError("feedback needs a preceding action; start with live(observations)")
        r = np.zeros(1) if reward is None else np.asarray(reward, dtype=float)
        ended = np.zeros(1, bool) if done is None else np.asarray(done)
        if r.shape != (1,) or not np.isfinite(r).all():
            raise ValueError("reward must be one finite value for the stream")
        if ended.shape != (1,) or ended.dtype != np.bool_:
            raise ValueError("done must be one boolean for the stream")
        importance = (
            None if salience is None else SynapticMemory.salience_vector(salience, 1)
        )
        answer: tuple[np.ndarray, BrainState] | None = None
        error = surprise = want = 0.0
        sweeps = learning_sweeps = 0
        learned = recorded = False
        credited = 0
        replay = False
        if not routine:
            self._calm = []  # held calm moments belong to an unbroken run of routine outcomes
        if sampled:
            report = self.learn(r, ended, x, salience=importance)
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
                surprise, want = arousal.outcome(error, float(r[0]), own=own, learned=sampled)
            except ValueError as exc:
                if sampled:
                    raise ValueError(
                        "feedback was accepted but arousal could not update; "
                        "retry live(observations) without reward or done or salience"
                    ) from exc
                raise
            self._lived = None  # this outcome is taken, exactly once
            self._awaiting = None  # and the stream continues from where it settled
            issued, self._calm_issued = self._calm_issued, None
            if routine and arousal.config.hindsight:
                assert lived is not None
                if issued is not None and issued[1] is lived[4]:
                    self._calm.append((*issued, float(r[0])))
                    del self._calm[: -arousal.config.hindsight]
                replay = arousal.aroused and bool(self._calm)
            if routine:
                if ended[0] and self.working_memory is not None:
                    self.working_memory.reset(1, rows=np.array([0]))
                if ended[0] and self.efference is not None:
                    self.efference.reset(1, rows=np.array([0]))
                if ended[0] and self.history is not None:
                    self.history.reset(1, rows=np.array([0]))
                if ended[0] and self.recall is not None:
                    # the forecast read the stream as this end leaves it; once the end is
                    # committed that read is the moment's, for the act's eligibility and write
                    self.recall.reset(1, rows=np.array([0]))
                    if self._forecast_read is not None:
                        self.recall.hold(self._forecast_read)
                self._forecast_read = None
                # a moment without eligibility has passed: the credit of earlier sampled
                # actions fades as it does between two outcomes that are learned from; an end
                # credited in hindsight clears the traces at its replayed step instead
                agent.fade(None if replay else ended)
            if routine and arousal.aroused and self.hippocampus is not None:
                assert lived is not None
                if not self.memory_choice:
                    self._record(
                        self._key_of(lived[0], lived[6]), lived[1], r,
                        SynapticMemory.salience_vector(
                            np.abs(r) if importance is None else importance, 1
                        ),
                    )
                    recorded = True
                else:
                    # the outcome that woke the brain surprised it: known at any reward
                    assert self._episodic is not None
                    self._episodic.outcome(
                        self._key_of(lived[0], lived[6]), lived[1], r, ended,
                        np.ones(1, bool), gamma=agent.config.gamma, woken=True,
                        salience=importance,
                    )
                    recorded = True
                if recorded and not self.memory_choice:
                    assert answer is not None
                    # the record can change this moment's recall: settle again under it, warm
                    agent._free, agent._free_brain, agent._drive = (
                        answer[1], self.brain, answer[0]
                    )
                    answer = None
            if replay:
                now_drive, now_state = answer if answer is not None else (agent._drive, agent._free)
                assert now_drive is not None and now_state is not None
                held = len(self._calm)
                credited, work = self._credit_calm(now_drive, now_state, bool(ended[0]))
                learning_sweeps += work
                learned = learned or credited > 0
                if ended[0] and credited < held:
                    agent.fade(ended)  # the end's own step was not taken: its traces end here
                answer = None  # the parameters may have moved: settle this moment again
            if routine and (arousal.aroused or ended[0]):
                self._calm = []
            episodic = self._episodic
            if routine and episodic is not None:
                episodic.received(r)  # a calm outcome joins the path; replay may value it later
                if ended[0]:  # the next attempt begins with no context and no path
                    episodic.reset(1, rows=np.flatnonzero(ended))
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
        place = self._placed(x)
        # the brain's own best guess, read at issue time before the decision moves the
        # temporal context on: the act it would issue greedily, the readout with the episodic
        # vote under the decision's own key, one choice per slot
        guessed = self._greedy(answer[1], place, self._decision_key(code))
        best = [int(choice) for choice in np.atleast_1d(guessed[0])]
        action = self._choose(
            x, *answer, greedy=not aroused, temperature=temperatures, learn_code=aroused
        )
        state = agent.state
        assert state is not None
        if agent._pending is not None:
            learning_sweeps += int(agent._pending[1].steps) + int(agent._pending[2].steps)
        chosen = [int(choice) for choice in np.atleast_1d(action[0])]
        self._calm_issued = (
            (answer[0].copy(), state, np.asarray(action).copy())
            if not aroused and arousal.config.hindsight
            else None
        )
        self._lived = (
            x.copy(),
            action.copy(),
            float(agent.value(state)[0]),
            aroused,
            state,
            best == chosen,
            place.copy() if self.learned_code is not None else None,
        )
        self._last_arousal = MappingProxyType(
            {
                "mode": arousal.mode,
                "level": float(arousal.level),
                "error": float(error),
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
                "credited": credited,
                "sweeps": int(sweeps),
                "learning_sweeps": int(learning_sweeps),
            }
        )
        arousal.lived(sweeps, learning_sweeps)
        return action

    def _credit_calm(
        self, drive: np.ndarray, state: BrainState, ended: bool
    ) -> tuple[int, int]:
        """Credit the held calm transitions to the outcome that woke the brain: score each
        greedy action's eligibility, as a sampled action's, and step the actor and critic
        through them in order with their own rewards, the waking outcome last, its next
        moment ``drive`` and ``state``. Returns the moments credited and the settle steps.
        A refused step ends the credit; the steps taken stay learned."""
        agent = self.basal_ganglia
        moments, self._calm = self._calm, []
        try:
            agent.unfade(len(moments))
        except ValueError:
            return 0, 0  # the traces cannot be restored over this many moments: credit none
        beta = self.learner.config.beta
        credited = work = 0
        try:
            for i, (moment_drive, moment_state, action, reward) in enumerate(moments):
                last = i == len(moments) - 1
                next_drive, warm = (drive, state) if last else moments[i + 1][:2]
                target = self.learner.targets(action)
                plus = agent._nudged_groups(moment_drive, moment_state, target, beta)
                minus = agent._nudged_groups(moment_drive, moment_state, target, -beta)
                agent._free, agent._free_brain = moment_state, self.learner.brain
                agent._drive = moment_drive
                agent._pending = ("states", plus, minus, agent.value(moment_state))
                # the recall read of a held moment is gone: its address map is not credited
                agent._pending_memory = None
                done = np.array([ended and last])
                report = agent.learn(
                    np.array([reward]), done, next_drive, warm=None if done[0] else warm
                )
                credited += 1
                work += int(plus.steps) + int(minus.steps) + int(report["free_steps"])
                self.last_learning = report
        except BaseException as exc:
            # a refused step ends the credit, and the steps taken stay learned; the stream
            # continues from the waking moment, with no held action pending
            agent._pending = agent._pending_memory = None
            agent._free, agent._free_brain, agent._drive = state, self.learner.brain, drive
            if not isinstance(exc, RuntimeError):
                raise
        return credited, work

    @_operation
    def wait(self, observations: Any) -> None:
        """One moment of the stream while its issued action's outcome is still to come.

        The brain settles the present observation from its current activity, reading its
        working trace, the copy of its last command, its heard history, its recall memory and
        its associative memory; its working trace advances to that state, its heard history
        takes in the observation and its recall memory writes the moment. It issues no
        action and takes no outcome:
        the awaited action keeps the forecasts made before it, its eligibility and the
        situation it was chosen in, and ``live`` later credits its actual outcome to it
        once. Parameters, the critic, eligibility traces, associative memory, random
        state, the command copy and the arousal state stay as they were. A learned code
        follows the issued action's mode: routine ``live`` waits keep its tables still,
        and other waits take its local learning step. Eligibility,
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
        with self._code_attempt():
            self._prepare(x)
            drive = self.stimulus(x)
            state = self._qualified(drive, self._activity(), operation="wait")
        code = self._coded(x)
        lived = self._lived
        routine = lived is not None and lived[4] is current and not lived[3]
        if self.learned_code is not None and not routine:
            self._learn_code(state)  # routine live moments keep the region's tables still
        if self.working_memory is not None:
            self.working_memory.update(state)
        if self.history is not None:
            self.history.hear(self._heard(x, code))  # what was sensed joins the heard past
        if self.recall is not None:
            self.recall.witness(code, state)  # the sensed moment joins the stream's slots
        if self._episodic is not None:
            self._episodic.sense(code)  # what was sensed enters the temporal context
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

    def _forecast(self, x: np.ndarray, ended: bool) -> tuple[np.ndarray, BrainState]:
        """Read the next value without changing the stream. A finished episode's
        forecast starts from rest and fresh traces; accepting its outcome commits
        that reset, so a refused forecast or arousal update preserves feedback. The
        recall memory's read of the ended stream is kept for ``live`` to hold once the
        end is committed (``Recall.hold``): the act the forecast decides keeps the read's
        eligibility, and a refused outcome leaves the read cleared with the restore."""
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
        self._forecast_read = None
        try:
            for trace, _ in traces:
                trace.reset(1, rows=np.array([0]))
            if self.recall is not None:
                self.recall.reset(1, rows=np.array([0]))
            agent._free = None
            answer = self._settled(x)
            if self.recall is not None:
                # of the ended stream, as the forecast read it; live holds it once the
                # outcome is accepted and the end committed, so a refusal leaves no read
                self._forecast_read = self.recall.reading()
            return answer
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
        real outcomes are unchanged, and so are a learned code's step, work counts and
        latest phase. Motor responses are at ``motor_index`` in
        each returned state's activation. A refused phase ends the branch and
        remains in the result; inspect ``qualified`` before using a response.

        After each imagined moment the branch issues, privately, the act the brain would
        issue on it, its complete greedy choice; the branch's efference copy takes it in.
        With ``memory_choice`` that act includes the
        episodic vote, read under the branch's own temporal context, which the branch's acts
        move on as the stream's acts move the stream's. The vote joins the readout after the
        settle, so a returned state holds the cortex's response; a greedy ``act`` on a loaded
        copy reads the complete act on a supplied observation.

        This imagines the brain's responses, not an unobserved environment's
        dynamics.
        """
        from .brain import _copy_state

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
        # every memory's private branch: the traces and the history follow the branch, the
        # recall memory reads the live slots and writes none, the stores are read
        branch = Memories({name: memory.branch() for name, memory in self.memories.items()})
        trace, echo = branch.get("working_memory"), branch.get("efference")
        heard, recalled = branch.get("history"), branch.get("recall")
        # the branch issues the act the brain would issue where the act is heard, by the copy
        # of the command or by a recall memory that keeps it; the episodic vote is read under
        # the branch's own temporal context, the stream's moved on privately by the branch's
        # acts
        issues = echo is not None or (recalled is not None and recalled.action)
        episodic = branch.get("episodic")
        # the learned code's live state stays as it was: an imagined moment settles the region
        # as a real one does, at the step its competition needs, and the branch leaves the
        # region's step, its work counts and its latest phase, and the moment's code, as it
        # found them
        region = self.learned_code
        moment, settlement = self._coded_moment, self._code_settlement
        held = None if region is None else (
            region.halvings, region.graph, region.last, dict(region.work)
        )
        phases = []
        try:
            for value in sequence:
                self._prepare(value, trace=trace)  # a learned code in the branch's own context
                drive = self.stimulus(value, memory=False)
                code = self._coded(value)
                place = self._placed(value)
                drive = branch.stimulus(drive, self._read_moment(value, code, place))
                phase = self._equilibrate(
                    drive, current, budget=budget, tolerance=tolerance
                )
                phases.append(phase)
                if not np.all(phase.qualified):
                    break
                current = _copy_state(phase.state)
                if trace is not None:
                    trace.update(phase.state)
                command = None
                if issues:
                    # the imagined branch issues, privately, the act the brain would issue: its
                    # complete greedy choice, the readout with the episodic vote under the branch's
                    # own key, after which the branch's context moves on
                    key = None if episodic is None else episodic.key(code)
                    choice = self._greedy(phase.state, place, key)
                    if episodic is not None and key is not None:
                        episodic.move(key, choice)
                    command = self._command(choice)
                if echo is not None:
                    assert command is not None
                    echo.issue(command)
                if heard is not None:
                    # the imagined observation, as the cortex read it, joins the private history
                    heard.hear(self._heard(value, code))
                if recalled is not None:
                    # the branch's cortex moves on, and with ``recall_action`` its command
                    recalled.witness(code, phase.state, command if recalled.action else None)
        finally:
            self._coded_moment, self._code_settlement = moment, settlement
            if region is not None and held is not None:
                region.halvings, region.graph, region.last, work = held
                region.work.clear()
                region.work.update(work)
        return tuple(phases)

    @_operation
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
        action = self._choose(x, drive, free, greedy=greedy, temperature=temperature)
        self._acted = (x.copy(), self.basal_ganglia.state)
        return action

    def _settled(self, x: np.ndarray) -> tuple[np.ndarray, BrainState]:
        """The drive of validated observations and its qualified free state, warm from the
        stream's last state; the observations' code is computed here once for the moment.
        Refusal raises before anything changes."""
        with self._code_attempt():
            self._prepare(x)
            drive = self.stimulus(x)
            return drive, self._qualified(drive, self._activity(), operation="act")

    def _coded(self, x: np.ndarray) -> np.ndarray:
        """The observation code of the moment being acted on, computed by ``_settled``; of any
        other observations, computed here."""
        coded = self._coded_moment
        return coded[1] if coded is not None and coded[0] is x else self._code(x)

    def _placed(self, x: np.ndarray) -> np.ndarray:
        """The code the place stores key on (the associative store, the episodic place
        records): the moment's, kept by ``_prepare``; of any other observations, the
        context-free code with a learned code, else their observation code."""
        coded = self._coded_moment
        if coded is not None and coded[0] is x:
            return coded[3]
        code = self.learned_code
        if code is not None:
            return self._code(x, context=False)
        return self._code(x)

    def _choose(
        self,
        x: np.ndarray,
        drive: np.ndarray,
        free: BrainState,
        *,
        greedy: bool,
        temperature: float | Sequence[float] | np.ndarray | None = None,
        learn_code: bool = True,
    ) -> np.ndarray:
        """Issue the action of a qualified free state: the stream advances to it.

        ``live`` holds the code's tables still in routine; explicit acts keep its local
        learning rule, including greedy acts.
        """
        self.basal_ganglia._free = free
        self.basal_ganglia._free_brain = self.brain
        self.basal_ganglia._drive = drive.copy()
        self._prepared = None
        self._awaiting = None
        code = self._coded(x)  # the memories key on the observation code
        place = self._placed(x)  # the place stores on the place key
        episodic = self._episodic
        key = bias = None
        if episodic is not None:
            key = episodic.key(code)
            bias = episodic.vote(place, key)
        action = self.basal_ganglia.act(drive, greedy=greedy, temperature=temperature, bias=bias)
        state = self.basal_ganglia.state
        if learn_code and self.learned_code is not None and state is not None:
            self._learn_code(state)  # the region's step at every accepted moment
        if self.working_memory is not None and state is not None:
            self.working_memory.update(state)
        if self.efference is not None:
            self.efference.issue(self._command(action))
        if self.history is not None:
            # the observation acted on, as the cortex read it, joins the heard past; the next
            # drive holds the present
            self.history.hear(self._heard(x, code))
        if self.recall is not None:
            # the moment acted on joins the stream's slots, with ``recall_action`` with the
            # command executed on it; its settled cortex enters the next moment's address
            self.recall.witness(
                code, free, self._command(action) if self.recall.action else None
            )
        if episodic is not None:
            guess = None
            if episodic.steps:
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
                guess = np.minimum(guess, 1.0)
            # what the stream did joins its temporal context after what it saw, and with replay
            # the decision its path
            episodic.decide(place, key, action, guess)
        self._moment = None if greedy else (x.copy(), action.copy())
        # with a learned code the place key the decision was taken under, settled at this moment
        self._moment_code = (
            None if greedy or self.learned_code is None else np.array(place, copy=True)
        )
        return action

    def _greedy(self, state: BrainState, code: np.ndarray, key: np.ndarray | None) -> np.ndarray:
        """The brain's complete greedy choice on a settled state: the action a greedy act on it
        issues, one per output slot, ``(batch,)`` or ``(batch, slots)``. With
        ``memory_choice`` the records' vote for the observation ``code`` under ``key``, the
        decision's own temporal key (``_decision_key``, read before the decision moves the
        context on), joins the readout before each slot's softmax and the most probable
        action is taken, as ``_choose`` and ``ActorCritic.act`` do for a greedy act; without
        it the choice is the most active motor neuron of every slot, the readout's argmax,
        which differs from a greedy act's softmax argmax only at an exact tie
        of two motor activations. Pure: the records are read and nothing changes. The
        surprise lesson judges by this choice, ``live`` decides from it whether an issued
        action was the brain's own best guess, and an imagined branch issues it."""
        episodic = self._episodic
        if episodic is None:
            return self._choices(state)
        p = self.basal_ganglia.probabilities(state, bias=episodic.vote(code, key))
        return np.asarray(np.argmax(p, axis=-1), dtype=np.int64)

    def _decision_key(self, code: np.ndarray) -> np.ndarray | None:
        """The temporal key of a decision taken now on this observation code, before the
        decision moves the stream's context on; None without context records."""
        return None if self._episodic is None else self._episodic.key(code)

    def _command(self, action: np.ndarray) -> np.ndarray:
        """The ``(batch, motor)`` one-hot of an issued action: one per row, or one per slot."""
        return command(action, len(self.motor_index), self.learner.slot_offsets)

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

        A ``done`` row's working trace, efference copy and heard history begin afresh, and its
        recall memory's slots are emptied.
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
        # Preflight without reading or resetting memories: invalid input must not write an episode.
        # Only the next drive's shape is needed here; computing it would already settle code.
        shape = np.broadcast_to(0.0, (len(following), self.connectome.n))
        reward, done, _, bootstrap = self.basal_ganglia._validated_transition(
            reward, done, shape, bootstrap
        )
        importance = SynapticMemory.salience_vector(
            np.abs(reward) if salience is None else salience, len(following)
        )
        with self._code_attempt():
            return self._learn_outcome(
                reward, done, following, bootstrap, importance, salience is not None
            )

    def _learn_outcome(
        self, reward: np.ndarray, done: np.ndarray, following: np.ndarray,
        bootstrap: np.ndarray | None, importance: np.ndarray, supplied_salience: bool,
    ) -> dict[str, float]:
        """Apply a validated outcome, retaining pending feedback on numerical refusal."""
        self._prepare(following)
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
        heard = self.history
        heard_trace = None if heard is None or not done.any() else heard.trace.copy()
        recalled = self.recall
        recalled_state = None if recalled is None or not done.any() else recalled.snapshot()
        episodic = self._episodic  # its context records, temporal contexts and paths
        episode_values = None if episodic is None else episodic.snapshot()
        try:
            if memory is not None and self._moment is not None:
                keys, action = self._moment
                if episodic is None:
                    self._record(self._key_of(keys, self._moment_code), action, reward, importance)
                else:
                    episodic.received(reward)
                    episodic.outcome(
                        self._key_of(keys, self._moment_code), action, reward, done,
                        np.ones(len(action), bool), gamma=self.basal_ganglia.config.gamma,
                        salience=importance if supplied_salience else None,
                    )
            if episodic is not None and done.any():
                # the streams whose attempt ended begin the next with no context and no path
                episodic.reset(len(done), rows=np.flatnonzero(done))
            if trace is not None and done.any():
                trace.reset(len(done), rows=np.flatnonzero(done))
            if echo is not None and done.any():
                echo.reset(len(done), rows=np.flatnonzero(done))
            if heard is not None and done.any():
                heard.reset(len(done), rows=np.flatnonzero(done))
            if recalled is not None and done.any():
                recalled.reset(len(done), rows=np.flatnonzero(done))
            next_drive = self.stimulus(following)
            warm = self._awaited()
            if self.learned_code is not None:
                # the next state settles warm from the stream's state with the code's phase
                warm = self._warm(next_drive, self.basal_ganglia.state if warm is None else warm)
            report = self.basal_ganglia.learn(reward, done, next_drive, bootstrap, warm=warm)
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
            if episodic is not None and episode_values is not None:
                episodic.restore(episode_values)
            raise
        self._moment = None
        self._moment_code = None
        self._prepared = following.copy()
        self._awaiting = None
        return report

    def _record(
        self, keys: np.ndarray, action: np.ndarray, reward: np.ndarray, importance: np.ndarray
    ) -> None:
        """Write the outcome of a chosen action into the associative store for the situation
        it was chosen in, every stream."""
        assert self.hippocampus is not None
        record_action(
            self.hippocampus, keys, action, reward, importance, None, self.learner.slot_offsets
        )

    def reset(self, *, keep: Any = ("arousal",), **renamed: Any) -> None:
        """Start every stream afresh; the stores' records are kept. By default a brain with
        arousal carries its arousal state into the new stream (``keep=("arousal",)``).
        ``keep`` names what carries over, a name or a sequence of names; ``keep=()`` keeps
        neither:

        - ``"slots"``: the recall memory's slots. Each stream ends an episode there
          (``Recall.end``), its slots read by the next streams of the same batch, and
          everything else starts afresh; a brain without a recall memory resets as without it.
        - ``"arousal"``: the arousal state as it is. Its level, its recent and long-run
          reward, its usual errors, its reward scale, their counts, its age and its work
          counts carry into the new stream, as when the state saved before the reset
          (``state = brain.arousal.to_dict()``) is restored after it
          (``brain.arousal = Arousal.from_dict(state)``); a brain that was aroused when it
          was reset begins the new stream aroused. A brain without arousal resets as without
          it.

        After a reset that does not keep ``"arousal"`` (``keep=()`` or ``keep=("slots",)``),
        the arousal begins calm with its references cleared and its age kept, as after
        ``brain.arousal.reset()``: an adult brain (its youth over) has no income or error
        references and no reward scale, so at ``need=0`` it has no want, and the scaled need
        wants nothing until outcomes enter the new stream's scale; only a surprise at its own
        forecasts can rouse it, and a world that stops paying may leave it calm on a lapsed
        habit (``docs/start.md#reset-and-save``)."""
        for name in renamed:
            if name in ("keep_memory", "keep_arousal"):
                what = "slots" if name == "keep_memory" else "arousal"
                raise TypeError(f"{name}=True is reset(keep=({what!r},)) in this version")
            raise TypeError(f"reset() got an unexpected keyword argument {name!r}")
        names = (keep,) if isinstance(keep, str) else keep
        try:
            kept = set(names)
        except TypeError as exc:
            raise ValueError("keep names 'slots' and 'arousal'") from exc
        if not all(isinstance(name, str) for name in kept) or kept - {"slots", "arousal"}:
            raise ValueError("keep names 'slots' and 'arousal'")
        self.basal_ganglia.reset()
        for memory in self.memories.values():
            # the streams begin afresh; the stores' records stay, and kept, the recall
            # memory's slots
            memory.begin(keep="slots" in kept)
        self._moment = None
        self._moment_code = None
        self._coded_moment = None
        self._code_settlement = None
        if self.learned_code is not None:
            self.learned_code.last = None  # the tables stay, as the weights do
        self._prepared = None
        self.last_learning = {}
        self._last_settlement = None
        self._lived = None
        self._last_arousal = None
        self._calm, self._calm_issued = [], None
        self._acted = None
        self._awaiting = None
        if self.arousal is not None and "arousal" not in kept:
            self.arousal.reset()

    def parameters(self) -> int:
        """Actor/critic and consolidated weights, the recall memory's address and position
        maps and a learned code's tables; per-stream transient storage is additional."""
        recall = self.recall
        return (
            self.basal_ganglia.parameters()
            + (
                self.hippocampus.consolidated.size
                if isinstance(self.hippocampus, SynapticMemory)
                else 0
            )
            + (0 if self.context_memory is None else self.context_memory.consolidated.size)
            + (
                0 if recall is None
                else sum(sum(a.size for a in recall.head_map(h)) for h in range(recall.heads))
                + sum(a.size for maps in recall.position_maps for a in maps)
            )
            + (0 if self.learned_code is None else self.learned_code.parameters())
        )

    def save(self, path: str | Path) -> Path:
        """Save the complete composition, including an action awaiting its real outcome.

        The file is ``cadence-brain/1``: the learner's checkpoint, the actor (its config,
        critic, eligibility and random generator), the brain's random generator, the free
        phase, a pending action's nudged states, the state ``wait`` sensed since that action
        was chosen, the arousal, the observation code or the learned code, and one group
        ``memories/<name>/...`` per present memory (``working_memory``, ``efference``,
        ``history``, ``recall``, ``hippocampus``, ``episodic``) with the memory's settings
        under ``memories`` in the metadata. ``Learner.save`` saves the slow learned response
        alone.
        """
        from .checkpoint import _learner_data, _write
        from .receipts import canonical_json

        agent = self.basal_ganglia
        data = _learner_data(self.learner)
        memories: dict[str, Any] = {}
        metadata: dict[str, Any] = {
            "format": BRAIN_FORMAT,
            "actor": agent.config.to_dict(),
            "rng": self.rng.bit_generator.state,
            "actor_rng": agent.rng.bit_generator.state,
            "updates": agent.updates,
            "b_critic": agent.b_critic,
            "free_steps": None if agent.state is None else agent.state.steps,
            "free_current": agent._free_brain is self.brain,
            "pending": agent._pending is not None,
            "last_learning": (
                self.last_learning if self.learner.config.initial_damping
                else _checkpoint_learning_report(self.last_learning)
            ),
            "memories": memories,
        }
        if self._composition is not None:
            metadata["composition"] = plain(self._composition)
        if self.attention:
            metadata["attention"] = True
        if self.arousal is not None:
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
                }
                if lived[6] is not None:  # the code the lived action was keyed on
                    data["lived/code"] = lived[6]
        awaited = self._awaited()
        if awaited is not None:  # what the stream sensed while its outcome is awaited
            metadata["awaiting"] = {"steps": int(awaited.steps)}
            for name in ("v", "activation", "adaptation"):
                data["awaiting/" + name] = np.asarray(getattr(awaited, name))
        if self.sparse_code is not None:
            # the records have the code's width and the keys read its tables
            metadata["code"] = {**self.sparse_code.to_dict(), "cortex": self.coded_cortex}
            tables = self.sparse_code.arrays()
            data["code/index"], data["code/weight"] = tables["index"], tables["weight"]
        if self.learned_code is not None:
            # its tables are the brain's synapses and the memories' keys
            metadata["learned_code"] = self.learned_code.to_dict()
            for name, value in self.learned_code.arrays().items():
                data["learned_code/" + name] = value
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
                if self._moment_code is not None:  # the code the decision was keyed on
                    data["moment/code"] = self._moment_code
        for name in _REWARD_ARRAYS:
            value = getattr(agent, name)
            if value is not None:
                data["actor/" + name] = np.asarray(value)
        if agent._trace_device is not None:
            for name, tensor in zip(("trace", "trace_bias"), agent._trace_device, strict=True):
                data["actor/" + name] = tensor.detach().cpu().double().numpy()
        if agent.state is not None:
            for name in ("v", "activation", "adaptation"):
                data["free/" + name] = np.asarray(getattr(agent.state, name))
        if self._prepared is not None:
            data["prepared"] = self._prepared
        for name, (settings, arrays) in self._saved_memories().items():
            memories[name] = settings
            for array, value in arrays.items():
                data[f"memories/{name}/{array}"] = value
        data["brain"] = np.array(canonical_json(metadata))
        return _write(data, path)

    def _saved_memories(self) -> dict[str, tuple[Any, dict[str, np.ndarray]]]:
        """Each present memory's settings and arrays, in the order the brain reads them."""
        agent = self.basal_ganglia
        out: dict[str, tuple[Any, dict[str, np.ndarray]]] = {
            name: (memory.to_dict(), memory.arrays()) for name, memory in self.memories.items()
        }
        if "recall" in out:
            # the actor's eligibility of the address map, and the pending action's contrast
            arrays = out["recall"][1]
            if agent.trace_memory is not None:
                arrays["trace"] = agent.trace_memory
            if agent._pending is not None and agent._pending_memory is not None:
                arrays["pending"] = agent._pending_memory
        return out

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

        Reads ``cadence-brain/1`` and, through 0.85.x, the earlier formats:
        ``cadence-generic/1`` to ``/9``, which 0.84.3 reads, and ``/10``, which development
        builds of the learned code wrote. A setting of 0.84.3 that this version removed loads
        at the value whose law this version keeps and is refused otherwise, with a message
        naming 0.84.3. Keep the saved batch row identities, or call ``reset`` for new
        episodes. Cross-backend results are subject to floating-point differences.
        """
        import json

        with np.load(path, allow_pickle=False) as archive:
            data: Mapping[str, Any] = archive
            if "brain" in archive:
                saved = json.loads(str(archive["brain"]))
                if not isinstance(saved, dict) or saved.get("format") != BRAIN_FORMAT:
                    raise ValueError("unsupported Brain checkpoint format")
                meta, data = _earlier_layout(saved, archive)
            elif "generic" in archive:
                meta = json.loads(str(archive["generic"]))
            else:
                raise ValueError("not a Brain checkpoint; use Learner.load for a learner")
            formats = tuple(f"cadence-generic/{level}" for level in range(1, 11))
            if not isinstance(meta, dict) or meta.get("format") not in formats:
                raise ValueError("unsupported Brain checkpoint format")
            # each format carries the component it was introduced for, and may carry those
            # of the earlier formats; a component under an earlier format name is refused
            level = formats.index(meta["format"]) + 1
            if (level == 10) != ("learned_code" in meta):
                raise ValueError(
                    "a brain's learned code belongs to checkpoint format cadence-generic/10"
                )
            if (level == 9) != ("code" in meta):
                raise ValueError("a brain's code belongs to checkpoint format cadence-generic/9")
            if level == 8 and meta.get("recall") is None:
                raise ValueError("format cadence-generic/8 carries a recall memory")
            if meta.get("recall") is not None and level < 8:
                raise ValueError(
                    "a recall memory belongs to checkpoint formats cadence-generic/8 to /10"
                )
            if level == 7 and meta.get("history") is None:
                raise ValueError("format cadence-generic/7 carries a heard history")
            if meta.get("history") is not None and level < 7:
                raise ValueError(
                    "a heard history belongs to checkpoint formats cadence-generic/7 to /10"
                )
            if level == 6 and "episodic" not in meta:
                raise ValueError("format cadence-generic/6 carries an episodic memory's choice")
            if "episodic" in meta and level < 6:
                raise ValueError(
                    "an episodic memory's choice belongs to checkpoint formats cadence-generic/6 "
                    "to /10"
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
            # the resting bias 0.84.3 recorded was the initial bias of the processing regions,
            # provenance only: the saved biases hold whatever it and learning made of them
            learner = Learner.load(path, backend=backend, device=device, precision=precision)
            if learner.config.nudge != "cross_entropy":
                raise ValueError(
                    "this checkpoint was saved with learning_nudge='quadratic', which a composed "
                    "brain no longer takes; load it with cadence-net 0.84.3"
                )
            composition = _load_composition(meta.get("composition"), learner)
            populations = learner.brain.connectome.populations
            if not np.array_equal(learner.output_index, populations.get("motor", ())):
                raise ValueError("saved learner outputs must match the motor population in order")
            code = _load_code(
                meta, data, len(populations.get("sensory", populations.get("visual/input", ())))
            )
            learned = _validate_learned_code(meta, data, learner)
            try:
                _validate_life_state(meta, data, learner, code, learned)
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid or incomplete saved continuation state") from exc
            # a memory keyed on the code numbers its key coordinates from zero: the code's
            # width bounds them, with the motor neurons after it for the context records
            neurons = learner.brain.connectome.n
            coded: int | None
            if learned:  # the key width: the units, joined to the observation's identity
                coded = learned + len(populations["sensory"])
            else:
                coded = None if code is None else code[1].expansion
            memory = _load_memory(meta["hippocampus"], data, neurons, keys=coded)
            heard = meta.get("history")
            recalled = meta.get("recall")
            attention = meta.get("attention", False)
            if not isinstance(attention, bool):
                raise ValueError("invalid saved attention marker")
            if attention and (
                not learned
                or recalled is None
                or recalled.get("window", Recall.options()["window"]) < 2
                or not recalled.get("next", Recall.founders()["next"])
            ):
                raise ValueError(
                    "saved attention needs a learned code and contextual successor recall"
                )
            if composition is not None and attention != bool(composition.get("attention", False)):
                raise ValueError("saved attention disagrees with composition provenance")
            result = cls(
                learner.brain.connectome,
                episodic=False,
                reward=_actor_config(meta["reward"]),
                slots=[int(k) for k in learner.slot_sizes],
                code=0 if code is None else code[1].expansion,
                inputs=code[0] if code is not None and code[2] else None,
                learned_code=learned or None,
                **(
                    {}
                    if not learned
                    else {
                        f"learned_code_{name}": meta["learned_code"][name]
                        for name in LearnedCode.GENES
                    }
                ),
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
                        "recall_position": _saved_position(recalled, data),
                        "recall_compete": _saved_compete(recalled),
                    }
                ),
            )
            if code is not None:
                # the saved code as validated, its settings, seed and tables: the construction
                # above drew a code from the default seed
                result.sparse_code = code[1]
            if learned:
                # the saved tables, never ones drawn anew, and the saved synapses must be the
                # ones the tables give (the learner below carries them)
                region = result.learned_code
                assert region is not None
                region.seed = int(meta["learned_code"]["seed"])
                region.take(
                    data["learned_code/dictionary"], data["learned_code/weights"],
                    data["learned_code/thresholds"],
                )
                region.work.update(
                    {k: int(v) for k, v in meta["learned_code"]["work"].items() if k in region.work}
                )
                installed = result._installed(learner.brain)
                if not np.array_equal(installed.efficacy, learner.brain.efficacy) or not (
                    np.array_equal(installed.bias, learner.brain.bias)
                ):
                    raise ValueError(
                        "the saved learned code tables disagree with the saved synapses and "
                        "biases of the code region"
                    )
                # the region's phase at the step it was saved at, and its two switches
                region.restep(int(meta["learned_code"]["stepping"]))
                region.learning = bool(meta["learned_code"]["learning"])
                region.top_down = float(meta["learned_code"]["top_down"])
            if heard is not None:
                assert result.history is not None  # validated above: the population is there
                result.history.trace = data["history/trace"].astype(float)
            if recalled is not None:
                recall = result.recall
                assert recall is not None  # validated above: the populations are there
                for name in ("keys", "values", "nexts", "previous", "w_x", "w_c", "b"):
                    setattr(recall, name, data["recall/" + name].astype(float))
                recall.head_maps = [  # validated above: one map per further head
                    tuple(data[f"recall/{name}_{h}"].astype(float) for name in ("w_x", "w_c", "b"))
                    for h in range(1, recall.heads)
                ]
                recall.position_maps = [  # validated above: saved with a position term
                    tuple(data["recall/" + name].astype(float) for name in Recall.position_names(h))
                    for h in range(recall.heads if recall.position_maps else 0)
                ]
                recall.held = data["recall/held"].astype(bool)
                recall.ended = (
                    data["recall/ended"].astype(bool)
                    if "recall/ended" in data
                    else np.zeros(len(recall.keys), dtype=bool)
                )
                if recall.window > 1:  # validated above: saved with the window
                    recall.past = data["recall/past"].astype(float)
                if recall.action:  # validated above: saved with the command
                    recall.acted = data["recall/acted"].astype(float)
                recall.writes = int(recalled["writes"])
            result.learner = learner
            result._composition = composition
            result.attention = attention
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
            if not learner.config.initial_damping:
                result.last_learning = _checkpoint_learning_report(
                    result.last_learning, restore=True
                )
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
                    if "moment/code" in data:
                        result._moment_code = data["moment/code"].astype(float)
            if "recall/trace" in data:
                agent.trace_memory = data["recall/trace"].astype(float)
            working = meta["working_memory"]
            result.working_memory = None
            if working is not None:
                result.working_memory = Trace(
                    result.connectome,
                    **{
                        name: working[name]
                        for name in ("decay", "amplitude", "source", "target")
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
                    np.arange(result.code_width) if result._coded_keys else result.sensory_index
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
                assert memory is not None  # validated above: the place records are there
                temporal = _load_memory(
                    episodic["context_memory"], data, neurons, "temporal/",
                    keys=None if coded is None else coded + len(result.motor_index),
                )
                if temporal is not None and not isinstance(temporal, SynapticMemory):
                    raise ValueError("an episodic memory's context records consolidate")
                if temporal is not None and not np.array_equal(
                    temporal.pre,
                    np.arange(result.code_width + len(result.motor_index))
                    if result._coded_keys
                    else np.concatenate([result.sensory_index, result.motor_index]),
                ):
                    raise ValueError(
                        "the saved context records are keyed on other coordinates than the "
                        "brain's observation code and its motor neurons"
                    )
                records = EpisodicRecords(
                    memory,
                    temporal,
                    width=result.code_width,
                    offsets=result.learner.slot_offsets,
                    share=float(episodic["context"]),
                    steps=int(episodic["replay"]),
                    demonstrations=float(episodic["demonstrations"]),
                )
                result._episodic = records
                for name in ("context", "decision"):
                    if "episode/" + name in data:
                        setattr(records, name, data["episode/" + name].astype(float))
                if "episode/path/lengths" in data:
                    lengths = data["episode/path/lengths"]
                    keys = data.get("episode/path/keys")
                    actions = data["episode/path/actions"]
                    observations = data["episode/path/observations"]
                    rewards, received = data["episode/path/rewards"], data["episode/path/received"]
                    guesses = data["episode/path/guess"]
                    records.paths = [
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
                result._lived = (
                    data["lived/observations"].copy(),
                    data["lived/action"].copy(),
                    float(meta["lived"]["forecast"]),
                    bool(meta["lived"]["sampled"]),
                    state,
                    bool(meta["lived"]["own"]),
                    data["lived/code"].astype(float) if "lived/code" in data else None,
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
