"""``cadence.stream.PatternSeparator`` in the core: the fixed random code's sizes, its fan-in
(``index`` and ``weight``) or its dense ``projection``, and the running mean of the keys."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...stream import PatternSeparator

CLASS = PatternSeparator
SIZES = ("inputs", "expansion", "winners", "fan_in", "cover")


def state_of(held: PatternSeparator) -> dict[str, Any]:
    out: dict[str, Any] = {f"config.{s}": np.int64(getattr(held, s)) for s in SIZES}
    out["config.center"] = np.float64(held.center)
    if held.fan_in:
        out["index"] = np.asarray(held.index, dtype=np.int64)
        out["weight"] = np.asarray(held.weight, dtype=np.float64)
    else:
        out["projection"] = np.asarray(held.projection, dtype=np.float64)
    out["mean"] = np.array(held.mean, dtype=np.float64)
    return out


def restore(held: PatternSeparator, state: dict[str, Any]) -> None:
    for s in SIZES:
        setattr(held, s, int(state[f"config.{s}"]))
    held.center = float(state["config.center"])
    if held.fan_in:
        held.index = np.array(state["index"], dtype=np.int64)
        held.weight = np.array(state["weight"], dtype=np.float64)
        held.projection = None
    else:
        held.projection = np.array(state["projection"], dtype=np.float64)
        held.index = held.weight = None
    held.mean = np.array(state["mean"], dtype=np.float64)


def _code(key: Any, learn: bool = False) -> dict[str, Any]:
    return {"key": np.array(key, dtype=np.float64), "learn": np.bool_(learn)}


METHODS = {
    "code": (_code, lambda value: {"code": np.array(value, dtype=np.float64)}),
}
