"""``cadence.stream.Trace`` in the core: the working trace of a range, kept per stream, its decay,
amplitude and learned gain (``hearing``), and the neurons it reads (``hidden``) and stimulates
(``glow``). A trace resting on a device leaves it first (the host's arrays cross); a trace
before its first ``reset`` (its construction's) holds no streams. What a moment changes in place
crosses as a copy, so a kept state stays as it was taken."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...stream import Trace, columns

CLASS = Trace


def _held(held: Any, name: str, cols: int, dtype: Any = np.float64) -> np.ndarray:
    """A copy of ``held.name``, or no streams before the first ``reset``."""
    value = held.__dict__.get(name)
    if value is None:
        return np.zeros((0, cols) if cols >= 0 else 0, dtype=dtype)
    return np.array(value, dtype=dtype)


def state_of(held: Trace) -> dict[str, Any]:
    held.leave_device()
    n = len(held.hidden)
    return {
        "config.decay": np.float64(held.decay),
        "config.amplitude": np.float64(held.amplitude),
        "hearing": np.float64(held.hearing),
        "glow": np.asarray(held.glow, dtype=np.int64),
        "hidden": np.asarray(held.hidden, dtype=np.int64),
        "trace": _held(held, "trace", n),
        "last": _held(held, "last", n),
        "cold": _held(held, "cold", -1, bool),
    }


def restore(held: Trace, state: dict[str, Any]) -> None:
    held.leave_device()
    held.decay = float(state["config.decay"])
    held.amplitude = float(state["config.amplitude"])
    held.hearing = float(state["hearing"])
    held.glow = np.array(state["glow"], dtype=np.int64)
    held.hidden = np.array(state["hidden"], dtype=np.int64)
    held._glow_columns = columns(held.glow)
    held._hidden_columns = columns(held.hidden)
    held.trace = np.array(state["trace"], dtype=np.float64)
    held.last = np.array(state["last"], dtype=np.float64)
    held.cold = np.array(state["cold"], dtype=bool)


def _reset(batch: int, rows: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"batch": np.int64(batch)}
    if rows is not None:
        rows = np.array(rows)
        out["rows"] = rows if rows.dtype == bool else rows.astype(np.int64)
    return out


def _activation(state: Any) -> dict[str, Any]:
    return {"activation": np.atleast_2d(np.array(state.activation, dtype=np.float64))}


def _drive(drive: Any, moment: Any = None) -> dict[str, Any]:
    return {"drive": np.array(drive, dtype=np.float64)}


def _array(name: str) -> Any:
    return lambda value: {name: np.array(value, dtype=np.float64)}


METHODS = {
    "reset": (_reset, lambda value: {}),
    "update": (_activation, lambda value: {}),
    "_source_activation": (_activation, _array("activation")),
    "stimulate": (_drive, _array("drive")),
    "stimulus": (_drive, _array("drive")),
}
