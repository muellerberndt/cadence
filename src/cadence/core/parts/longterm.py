"""``cadence.longterm.LongTermMemory`` in the core: its settings, its records and their sums.

The core holds a store read whole (no coarse stage), in float64, without a source's signature
(Wild's stores; the core refuses another)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...longterm import LongTermMemory

CLASS = LongTermMemory
# the settings in the order the core keeps them, each with its type
SETTINGS = (("width", np.int64), ("value_width", np.int64), ("capacity", np.int64),
            ("temperature", np.float64), ("candidates", np.int64), ("expansion", np.int64),
            ("winners", np.int64), ("depth", np.float64), ("idle", np.float64),
            ("hold", np.float64), ("revise", np.float64), ("fade", np.bool_),
            ("seed", np.int64), ("direct", np.int64), ("cosine", np.bool_), ("pairs", np.int64),
            ("apart", np.int64), ("blend", np.bool_), ("familiar", np.float64),
            ("keep", np.float64), ("forget", np.float64), ("novelty", np.float64),
            ("coarse", np.bool_))


def state_of(store: LongTermMemory) -> dict[str, Any]:
    if store.dtype != np.float64:
        raise ValueError("the core holds a long-term memory in float64")
    out: dict[str, Any] = {f"config.{name}": kind(getattr(store, name))
                           for name, kind in SETTINGS}
    for name in ("keys", "values", "deep", "recent", "faded", "norms"):
        out[name] = np.array(getattr(store, name), dtype=np.float64)
    out["kept"] = np.array(store.kept, dtype=bool)
    out["gone"] = np.array(store.gone, dtype=bool)
    out["state"] = np.array([store.count, store.writes, store._cursor], dtype=np.int64)
    return out


def restore(store: LongTermMemory, state: dict[str, Any]) -> None:
    """The records and their sums onto a store of the same settings (refused otherwise)."""
    for name, value in state_of(store).items():
        if name.startswith("config.") and not np.array_equal(
                np.asarray(state[name]), value):
            raise ValueError(f"{name}: a long-term memory of other settings")
    for name in ("keys", "values", "deep", "recent", "faded", "norms"):
        given = np.asarray(state[name], dtype=np.float64)
        if given.shape != getattr(store, name).shape:
            raise ValueError(f"{name} has another shape: a memory of other settings")
        setattr(store, name, given.copy())
    for name in ("kept", "gone"):
        given = np.asarray(state[name], dtype=bool)
        if given.shape != getattr(store, name).shape:
            raise ValueError(f"{name} has another shape: a memory of other settings")
        setattr(store, name, given.copy())
    store.count, store.writes, store._cursor = (int(v) for v in np.asarray(state["state"]))


def _write(key: Any, value: Any, recognised: int | None = None, source: Any = None,
           surprise: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"key": np.asarray(key, dtype=np.float64),
                           "value": np.asarray(value, dtype=np.float64)}
    if recognised is not None:
        out["recognised"] = np.int64(recognised)
    if source is not None:
        out["source"] = np.asarray(source, dtype=np.float64)
    if surprise is not None:
        out["surprise"] = np.float64(surprise)
    return out


def _written(value: tuple[int, bool]) -> dict[str, Any]:
    return {"slot": np.int64(value[0]), "revised": np.bool_(value[1])}


METHODS = {
    "write": (_write, _written),
    "moment": (lambda: {}, lambda value: {}),
}
