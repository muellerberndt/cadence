"""Named overrides for the composed-brain configurations.

A public setting name is a prefix and a field, such as ``learning_beta``,
``actor_eta`` or ``arousal_threshold``, for ``LearnerConfig``,
``ActorCriticConfig`` and ``ArousalConfig``; ``temperature`` is the learner's
``temperature``, the one spelling. This module resolves the names and checks
scalar kinds; the dataclasses define the defaults and valid domains. Mutation
search spaces need not cover every valid configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

from .arousal import ArousalConfig
from .learning import LearnerConfig
from .plasticity import ActorCriticConfig

Config = TypeVar("Config", LearnerConfig, ActorCriticConfig, ArousalConfig)

_KINDS = {
    "learning": LearnerConfig,
    "actor": ActorCriticConfig,
    "arousal": ArousalConfig,
}
# every field of each configuration with the kind its annotation names: "bool", "int", "str" or
# a real number, with " | None" where None is valid
_FIELDS = {
    prefix: {field.name: str(field.type) for field in fields(kind)}
    for prefix, kind in _KINDS.items()
}
# Public names this library no longer takes, each with the release that retired it, the one
# value a checkpoint may still carry for it (None: never read) and what the brain does instead:
# the retired fields of the three configurations under their public names, and the brain's
# own. Refused with one sentence (``removed``) at ``compose`` and ``retune`` and in an arousal
# mapping; a checkpoint that carries one at another value is refused by the component that
# saved it (``removed_on_load``). The names of 0.84.3 left in 0.85.0, which cadence-net 0.84.3
# takes.
RETIRED: Mapping[str, tuple[str, Any, str]] = MappingProxyType({
    **{f"learning_{name}": entry for name, entry in LearnerConfig.RETIRED.items()},
    **{f"actor_{name}": entry for name, entry in ActorCriticConfig.RETIRED.items()},
    **{f"arousal_{name}": entry for name, entry in ArousalConfig.RETIRED.items()},
    "learning_nudge": ("0.85.0", "cross_entropy", "a composed brain learns with the "
                       "cross-entropy nudge, and the quadratic nudge is a standalone "
                       "Learner's; cadence-net 0.84.3 takes it"),
    "working_memory_focus": ("0.85.0", 0.0, "every trace decays toward its source; "
                             "cadence-net 0.84.3 takes it"),
    "resting_bias": ("0.85.0", None, "every bias starts at zero, and a checkpoint's biases "
                     "are in its arrays; cadence-net 0.84.3 takes it"),
    "learning_normalize_relative": ("0.93.0", 0.0, "teaching='deep' sets the "
                                    "projection-relative floor at 0.1 with nudged phases of "
                                    "a whole 32 sweeps, the one point ever read; "
                                    "learning=LearnerConfig(...) sets it as an expert; "
                                    "cadence 0.92.1 takes it"),
    "learning_nudged_fixed": ("0.93.0", False, "teaching='deep' runs the nudged phases their "
                              "whole 32 sweeps with the projection-relative floor at 0.1, the "
                              "one point ever read; learning=LearnerConfig(...) sets it as an "
                              "expert; cadence 0.92.1 takes it"),
    "learning_temperature": ("0.93.0", None, "temperature is the one spelling of the "
                             "learner's softmax temperature, at compose, retune and in "
                             "describe; cadence 0.92.1 takes it"),
    # the constants of the composed brain (``CONSTANTS``): the founder each reading left
    "learning_beta": ("0.93.0", 0.1, "a composed brain nudges at beta 0.1, a constant; "
                      "learning=LearnerConfig(beta=...) moves it as an expert; cadence 0.92.1 "
                      "takes it"),
    "learning_momentum": ("0.93.0", 0.9, "a composed brain steps each synapse on a running "
                          "average of its contrast at 0.9, a constant; "
                          "learning=LearnerConfig(momentum=...) moves it as an expert; cadence "
                          "0.92.1 takes it"),
    "actor_dopamine_cap": ("0.93.0", 1.0, "a composed brain's dopamine saturates at 1.0, a "
                           "constant; reward=ActorCriticConfig(dopamine_cap=...) moves it as "
                           "an expert; cadence 0.92.1 takes it"),
    "actor_eligibility_steps": ("0.93.0", 12, "a composed brain scores a sampled action's "
                                "eligibility in nudged phases of 12 sweeps, a constant; "
                                "reward=ActorCriticConfig(eligibility_steps=...) moves it as "
                                "an expert; cadence 0.92.1 takes it"),
    "arousal_need_scale": ("0.93.0", 0.1, "a composed brain's need in reward scales is 0.1, "
                           "a constant, the founder the reading selected; "
                           "arousal=ArousalConfig(need_scale=...) moves it as an expert; "
                           "cadence 0.92.1 takes it"),
    "memory_decay": ("0.93.0", 0.9, "a stream's record fades by 0.9 per later write, a "
                     "constant; cadence 0.92.1 takes it"),
    "memory_replay": ("0.93.0", 8, "a reward replays up to 8 decisions of the stream's path, "
                      "a constant; cadence 0.92.1 takes it"),
    "memory_demonstrations": ("0.93.0", 1.0, "a demonstrated action is recorded at 1.0, a "
                              "constant; cadence 0.92.1 takes it"),
    "memory_winners": ("0.93.0", 24, "the episodic separation keeps its 24 most active cells "
                       "per key, a constant; cadence 0.92.1 takes it"),
    "history_scale": ("0.93.0", 1.2, "the heard history's projection starts at magnitude 1.2, "
                      "a constant (scales={'history': 1.2} alone is taken); cadence 0.92.1 "
                      "takes scales={'history': ...}"),
    "recall_scale": ("0.93.0", 1.2, "the recall and next projections start at magnitude 1.2, "
                     "a constant (scales={'recall': 1.2} alone is taken); cadence 0.92.1 "
                     "takes scales={'recall': ...}"),
})
# Fields of the three configurations that are constants of the composed brain since 0.93.0,
# the founders every reading by name used: the public names are retired (``RETIRED``), in a
# mapping as well, and ``describe`` leaves a constant out at its value. A configuration passed
# whole (``learning=``, ``reward=``, ``arousal=``) is the expert route and keeps its fields, as
# the chambers' frozen recipes compose (``LearnerConfig(**recipe)`` at momentum 0): ``describe``
# then names a moved constant by its old name, which ``compose`` and ``retune`` do not take, and
# a checkpoint carries and loads the expert's value as it does every field.
CONSTANTS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "learning": MappingProxyType({"beta": 0.1, "momentum": 0.9}),
    "actor": MappingProxyType({"dopamine_cap": 1.0, "eligibility_steps": 12}),
    "arousal": MappingProxyType({"need_scale": 0.1}),
})
# Learning settings the teaching point sets (``teaching=``); ``learning=LearnerConfig(...)``
# sets them as an expert.
_TAUGHT = {"learning_eta", "learning_eta_bias", "learning_normalize", "learning_normalize_floor"}


# The fields of a teaching point: the steps, their normalization, the projection-relative floor
# and whether the nudged phases run their whole budget. ``describe`` names the point a learner
# is at by these (``teaching_point``).
POINT = ("eta", "eta_bias", "normalize", "normalize_floor", "normalize_relative", "nudged_fixed")
# The teaching points ``teaching=`` names, every other learning setting at its founder.
# ``"normalized"`` is the point for many lessons, as in imitation; ``"deep"`` the normalized
# point for a brain with an apical compartment, with the projection-relative floor and nudged
# phases of a whole 32 sweeps (the one such point ever read, 0.90.0 to 0.92.1 as
# ``learning_normalize_relative=0.1, learning_nudged_steps=32, learning_nudged_fixed=True``).
TEACHING: dict[str, dict[str, Any]] = {
    "founder": {"eta": 0.5, "eta_bias": 0.05, "normalize": 0.0, "normalize_floor": 1e-3,
                "normalize_relative": 0.0, "nudged_fixed": False},
    "normalized": {"eta": 0.003, "eta_bias": 0.0003, "normalize": 0.99, "normalize_floor": 1e-4,
                   "normalize_relative": 0.0, "nudged_fixed": False},
    "deep": {"eta": 0.003, "eta_bias": 0.0003, "normalize": 0.99, "normalize_floor": 1e-4,
             "normalize_relative": 0.1, "nudged_fixed": True, "nudged_steps": 32},
}
# The projection scales ``scales=`` names, with the argument of 0.84.3 each replaces and its
# founder. The heard history's and the recall memory's projections start at 1.2, constants
# since 0.93.0 (``RETIRED``: ``history_scale``, ``recall_scale``).
SCALES: dict[str, tuple[str, float]] = {
    "sensory": ("sensory_scale", 1.0),
    "trace": ("working_memory_scale", 1.2),
}
CONSTANT_SCALES = {"history": 1.2, "recall": 1.2}


def removed(name: str) -> str:
    """The one sentence that refuses a retired public name (``RETIRED``): the release that
    retired it and what the brain does instead."""
    release, _, instead = RETIRED[name]
    return f"{name} was retired in {release}: {instead}"


def removed_on_load(owner: str, field: str, value: Any, name: str) -> str:
    """The refusal of a checkpoint saved with a retired setting at another value than the one
    it loads at: ``owner``'s ``field`` as the file wrote it, ``name`` its public name."""
    release, _, instead = RETIRED[name]
    return (
        f"this checkpoint was saved with the {owner}'s {field}={value!r}, which was retired "
        f"in {release}: {instead}"
    )


def teaching_fields(point: Any) -> dict[str, Any]:
    """The learner fields a teaching point sets (``TEACHING``)."""
    if not isinstance(point, str) or point not in TEACHING:
        raise ValueError("teaching must be 'founder', 'normalized' or 'deep'")
    return dict(TEACHING[point])


def taught(point: Any, base: LearnerConfig) -> LearnerConfig:
    """``base`` at a teaching point: the point's fields replaced."""
    return replace(base, **teaching_fields(point))


def teaching_point(config: LearnerConfig) -> str:
    """The teaching point a learner's configuration is at, by the fields of ``POINT``;
    ``"custom"`` outside every point."""
    fields = {name: getattr(config, name) for name in POINT}
    return next(
        (point for point, values in TEACHING.items()
         if {name: values[name] for name in POINT} == fields),
        "custom",
    )


def rescaled(name: str) -> str | None:
    """The refusal of a projection scale argument of 0.84.3, or None for any other name."""
    for key, (old, _) in SCALES.items():
        if name == old:
            return f"{old} is scales={{'{key}': ...}} in this version, a construction setting"
    return None


def scales_of(value: Any) -> dict[str, float]:
    """The projection scales: the founders with ``value`` (a mapping of names) laid over."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("scales must be a mapping of 'sensory' and 'trace'")
    unknown = sorted(set(value) - set(SCALES))
    for key in unknown:
        # the history's and the recall memory's projections start at 1.2, a constant since
        # 0.93.0: the key is taken at the constant alone (the chambers' frozen recipes pass it)
        if key in CONSTANT_SCALES and value[key] == CONSTANT_SCALES[key]:
            continue
        if key in CONSTANT_SCALES:
            raise ValueError(removed(f"{key}_scale"))
        raise ValueError(f"unknown scale {key!r}: scales name 'sensory' and 'trace'")
    return {
        key: real(f"scales[{key!r}]", value.get(key, founder), low=0)
        for key, (_, founder) in SCALES.items()
    }


def real(name: str, value: Any, *, low: float = -np.inf, high: float = np.inf) -> float:
    """Accept real scalars, never truth values, arrays or coerced strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be finite and lie in [{low:g}, {high:g}]")
    return number


def split_genes(genes: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Resolve only unambiguous public names; refuse typos before any changes."""
    values = dict(genes)
    changes: dict[str, dict[str, Any]] = {prefix: {} for prefix in _KINDS}
    if "temperature" in values:  # the learner's softmax temperature, under its one name
        changes["learning"]["temperature"] = values.pop("temperature")
    for name, value in values.items():
        if name in ("eta", "eta_bias"):
            raise TypeError(f"{name} is ambiguous; use actor_{name} or learning_{name}")
        if name in RETIRED:
            raise ValueError(removed(name))
        if name in _TAUGHT:
            raise ValueError(
                f"{name} is set by teaching ('founder', 'normalized' or 'deep'); "
                "learning=LearnerConfig(...) sets it as an expert"
            )
        prefix, _, field = name.partition("_")
        if prefix not in _FIELDS or field not in _FIELDS[prefix]:
            raise TypeError(f"unknown brain setting: {name!r}")
        changes[prefix][field] = value
    return tuple(changes[prefix] for prefix in _KINDS)


def configured(base: Config, changes: Mapping[str, Any], prefix: str) -> Config:
    """Overlay validated fields on a config without losing unspecified values."""
    expected = _KINDS[prefix]
    if not isinstance(base, expected):
        raise ValueError(f"{prefix} configuration must be {expected.__name__}")
    updates = dict(changes)
    unknown = set(updates) - _FIELDS[prefix].keys()
    for name in sorted(unknown):
        if f"{prefix}_{name}" in RETIRED:
            raise ValueError(removed(f"{prefix}_{name}"))
    if unknown:
        raise TypeError(f"unknown {prefix} setting: {sorted(unknown)[0]!r}")
    # Config constructors validate ranges and relationships. Validate scalar kinds
    # here too: several constructors accept bools as Python numbers.
    for name, kind in _FIELDS[prefix].items():
        value = updates.get(name, getattr(base, name))
        qualified = f"{prefix}_{name}"
        if value is None and kind.endswith(" | None"):
            continue
        kind = kind.removesuffix(" | None")
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{qualified} must be boolean")
        elif kind == "int":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{qualified} must be an integer")
            if name in updates:
                updates[name] = int(value)
        elif kind == "str":
            if not isinstance(value, str):
                raise ValueError(f"{qualified} must be a string")
        else:
            number = real(qualified, value)
            if name in updates:
                updates[name] = number
    for name in sorted(set(updates) & set(CONSTANTS[prefix])):  # named, or in a mapping
        raise ValueError(removed(f"{prefix}_{name}"))
    try:
        result = replace(base, **updates) if updates else base
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {prefix} configuration: {exc}") from exc
    if isinstance(result, LearnerConfig) and result.nudge != "cross_entropy":
        raise ValueError(removed("learning_nudge"))
    return result


def arousal_config(
    value: ArousalConfig | Mapping[str, Any] | bool | None,
    changes: Mapping[str, Any],
    *,
    current: ArousalConfig | None = None,
    retuning: bool = False,
) -> ArousalConfig | None:
    """Build an arousal configuration; a mapping overlays ``current`` or the defaults."""
    if value is None:
        if changes:
            raise ValueError("arousal settings require arousal=True or an arousal configuration")
        return None
    if isinstance(value, Mapping):
        overlap = set(value) & set(changes)
        if overlap:
            name = sorted(overlap)[0]
            raise TypeError(f"arousal_{name} was supplied both in arousal and as a named setting")
        return configured(current or ArousalConfig(), {**value, **changes}, "arousal")
    if value is True and not retuning:
        value = ArousalConfig()
    if not isinstance(value, ArousalConfig):
        raise ValueError(
            "arousal must be True, an ArousalConfig, a mapping, or None at construction"
        )
    return configured(value, changes, "arousal")


def plain(value: Any) -> Any:
    """Return detached, JSON-safe descriptive values, including NumPy scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(item) for item in value]
    return value
