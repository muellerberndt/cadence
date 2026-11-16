"""``cadence.surprise.Surprise`` in the core: its genes, each unit's usual size, the latest
surprise per stream and unit, and the moments observed. A brain's surprise and a map's cells'
surprise (``Map(surprise=)``) are each one."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...surprise import Surprise

CLASS = Surprise


def state_of(held: Surprise) -> dict[str, Any]:
    out: dict[str, Any] = {f"config.{g}": np.float64(getattr(held, g)) for g in Surprise.GENES}
    out.update({
        "usual": np.array(held.usual, dtype=np.float64),
        "last": np.array(held.last, dtype=np.float64),
        "moments": np.int64(held.moments),
    })
    return out


def restore(held: Surprise, state: dict[str, Any]) -> None:
    for g in Surprise.GENES:
        setattr(held, g, float(state[f"config.{g}"]))
    held.usual = np.array(state["usual"], dtype=np.float64)
    held.size = int(held.usual.size)
    held.last = np.array(state["last"], dtype=np.float64)
    held.moments = int(state["moments"])


def _observe(sizes: Any, *, learn: bool = True, carried: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"sizes": np.asarray(sizes, dtype=np.float64),
                           "learn": np.bool_(bool(learn))}
    if carried is not None:
        out["carried"] = np.asarray(carried)
    return out


def _reset(streams: int = 1) -> dict[str, Any]:
    return {"streams": np.int64(streams)}


METHODS = {
    "observe": (_observe, lambda value: {"surprise": np.asarray(value, dtype=np.float64)}),
    "moment": (lambda: {}, lambda value: {"moment": np.asarray(value, dtype=np.float64)}),
    "reset": (_reset, lambda value: {}),
}
