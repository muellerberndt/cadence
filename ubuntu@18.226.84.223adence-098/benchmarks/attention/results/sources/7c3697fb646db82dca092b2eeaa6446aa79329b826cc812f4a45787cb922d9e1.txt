"""Named overrides for the composed-brain configurations.

A public setting name is a prefix and a field, such as ``learning_beta``,
``actor_eta`` or ``arousal_threshold``, for ``LearnerConfig``,
``ActorCriticConfig`` and ``ArousalConfig``; ``temperature`` is accepted for
``learning_temperature``. This module resolves the names and checks scalar
kinds; the dataclasses define the defaults and valid domains. Mutation search
spaces need not cover every valid configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
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
_FIELDS = {prefix: {field.name for field in fields(kind)} for prefix, kind in _KINDS.items()}
_BOOLEAN = {"qualified", "surprise"}
_INTEGER = {"free_steps", "nudged_steps", "damping", "eligibility_steps", "youth"}
_STRING = {"nudge"}
_OPTIONAL = {"eta_bias", "tolerance", "eligibility_steps"}
# Settings of 0.84.3 that this version removed, refused by name with a message naming that
# release, at ``compose`` and ``retune`` and in an arousal mapping.
_REMOVED = {
    "arousal_record_surprise",
    "actor_dopamine_center",
    "actor_dopamine_floor",
    "actor_center_scale",
    "actor_critic_signal",
    "actor_critic_normalize",
    "actor_normalize",
    "actor_momentum",
    "learning_centered",
    "learning_decay",
    "learning_nudge",
    "working_memory_focus",
    "resting_bias",
}
# Learning settings the teaching point sets (``teaching=``); ``learning=LearnerConfig(...)``
# sets them as an expert.
_TAUGHT = {"learning_eta", "learning_eta_bias", "learning_normalize", "learning_normalize_floor"}


# The teaching points ``teaching=`` names: the steps and the normalization of the lesson's
# learner, every other learning setting at its founder. ``"normalized"`` is the point for many
# lessons, as in imitation.
TEACHING: dict[str, dict[str, float]] = {
    "founder": {"eta": 0.5, "eta_bias": 0.05, "normalize": 0.0, "normalize_floor": 1e-3},
    "normalized": {"eta": 0.003, "eta_bias": 0.0003, "normalize": 0.99, "normalize_floor": 1e-4},
}
# The projection scales ``scales=`` names, with the argument of 0.84.3 each replaces and its
# founder.
SCALES: dict[str, tuple[str, float]] = {
    "sensory": ("sensory_scale", 1.0),
    "trace": ("working_memory_scale", 1.2),
    "history": ("history_scale", 1.2),
    "recall": ("recall_scale", 1.2),
}


def removed(name: str) -> str:
    """The refusal of a setting this version removed."""
    return f"{name} was removed in this version; cadence-net 0.84.3 takes it"


def taught(point: Any, base: LearnerConfig) -> LearnerConfig:
    """``base`` at a teaching point: its steps and normalization replaced."""
    if not isinstance(point, str) or point not in TEACHING:
        raise ValueError("teaching must be 'founder' or 'normalized'")
    steps = TEACHING[point]
    return replace(
        base, eta=steps["eta"], eta_bias=steps["eta_bias"], normalize=steps["normalize"],
        normalize_floor=steps["normalize_floor"],
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
        raise ValueError("scales must be a mapping of 'sensory', 'trace', 'history', 'recall'")
    unknown = sorted(set(value) - set(SCALES))
    if unknown:
        raise ValueError(
            f"unknown scale {unknown[0]!r}: scales name 'sensory', 'trace', 'history', 'recall'"
        )
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
    if "temperature" in values:
        if "learning_temperature" in values:
            raise TypeError("use temperature or learning_temperature, not both")
        values["learning_temperature"] = values.pop("temperature")
    changes: dict[str, dict[str, Any]] = {prefix: {} for prefix in _KINDS}
    for name, value in values.items():
        if name in ("eta", "eta_bias"):
            raise TypeError(f"{name} is ambiguous; use actor_{name} or learning_{name}")
        if name in _REMOVED:
            raise ValueError(removed(name))
        if name in _TAUGHT:
            raise ValueError(
                f"{name} is set by teaching ('founder' or 'normalized'); "
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
    unknown = set(updates) - _FIELDS[prefix]
    for name in sorted(unknown):
        if f"{prefix}_{name}" in _REMOVED:
            raise ValueError(removed(f"{prefix}_{name}"))
    if unknown:
        raise TypeError(f"unknown {prefix} setting: {sorted(unknown)[0]!r}")
    # Config constructors validate ranges and relationships. Validate scalar kinds
    # here too: several constructors accept bools as Python numbers.
    for name in _FIELDS[prefix]:
        value = updates.get(name, getattr(base, name))
        qualified = f"{prefix}_{name}"
        if value is None and name in _OPTIONAL and prefix != "arousal":
            continue
        if name in _BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"{qualified} must be boolean")
        elif name in _INTEGER:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{qualified} must be an integer")
            if name in updates:
                updates[name] = int(value)
        elif name in _STRING:
            if not isinstance(value, str):
                raise ValueError(f"{qualified} must be a string")
        else:
            number = real(qualified, value)
            if name in updates:
                updates[name] = number
    try:
        result = replace(base, **updates) if updates else base
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {prefix} configuration: {exc}") from exc
    if isinstance(result, LearnerConfig) and result.nudge != "cross_entropy":
        raise ValueError(
            "a composed brain learns with the cross-entropy nudge; the quadratic nudge is a "
            "standalone Learner's (cadence-net 0.84.3 composes it)"
        )
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
