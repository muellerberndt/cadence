"""``cadence.stream.History`` in the core: the heard history, one block per decay, its amplitude
and learned gains (``hearing``, one per block, or per stream and block; absent where it holds
none), and the neurons it stimulates (``glow``); before its first ``reset`` (its
construction's) it holds no streams."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...stream import History, columns
from .trace import METHODS as TRACE_METHODS
from .trace import _held

CLASS = History


def state_of(held: History) -> dict[str, Any]:
    out: dict[str, Any] = {
        "config.decays": np.asarray(held.decays, dtype=np.float64),
        "config.amplitude": np.float64(held.amplitude),
        "width": np.int64(held.width),
        "glow": np.asarray(held.glow, dtype=np.int64),
    }
    if held.hearing is not None:
        out["hearing"] = np.array(held.hearing, dtype=np.float64)
    out["trace"] = _held(held, "trace", len(held.glow))
    return out


def restore(held: History, state: dict[str, Any]) -> None:
    held.decays = tuple(float(d) for d in state["config.decays"])
    held.amplitude = float(state["config.amplitude"])
    held.width = int(state["width"])
    held.glow = np.array(state["glow"], dtype=np.int64)
    held._glow_columns = columns(held.glow)
    held._decay = np.repeat(np.asarray(held.decays), held.width)
    held.hearing = np.array(state["hearing"], dtype=np.float64) if "hearing" in state else None
    held.trace = np.array(state["trace"], dtype=np.float64)


def _hear(observations: Any) -> dict[str, Any]:
    return {"observations": np.array(observations, dtype=np.float64)}


METHODS = {
    "reset": TRACE_METHODS["reset"],
    "stimulate": TRACE_METHODS["stimulate"],
    "stimulus": TRACE_METHODS["stimulus"],
    "hear": (_hear, lambda value: {}),
}
