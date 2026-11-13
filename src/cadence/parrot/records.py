"""The parrot's heard records: a heard word stored with the thing in focus, kept by use.

A heard sound's record is written to a ``cadence.LongTermMemory`` with the thing in focus as its
key (its kind's code and its place's code) and the heard frames, pooled, as its value. A word
heard again with the same thing in focus is the same record told again: the store revises it and
it grows deeper, so a word heard often is kept and one heard once fades with disuse. A read by
the thing in focus fetches its word's record and the record's strength, which sets how readily
the voice says it (#333).

The hearing memory of #336 (surprise and repetition strengthen a record; a kept threshold; the
running mean of the takes) arrives as options of the store. This part passes a write's
``surprise`` and reads a record's ``strength`` only where the store takes them, and otherwise
uses what the store holds on main (a told-again write revises the record and deepens it; depth
slows the fading). Nothing here depends on #336.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np


class HeardRecords:
    """Heard sounds by the thing in focus; ``store`` is a ``cadence.LongTermMemory`` (one of
    ``width``, ``value_width`` and ``capacity`` by default)."""

    def __init__(self, width: int, value_width: int, capacity: int = 256, *,
                 store: Any = None):
        from ..longterm import LongTermMemory

        self.store: Any = store if store is not None else LongTermMemory(
            width=int(width), value_width=int(value_width), capacity=int(capacity))
        self.takes_surprise = "surprise" in inspect.signature(self.store.write).parameters

    def write(self, key: np.ndarray, value: np.ndarray, *,
              surprise: float | None = None) -> tuple[int, bool]:
        """Store a heard value with its key; ``surprise`` reaches a store that takes it (#336).
        Gives the slot and whether a held record was told again."""
        if surprise is not None and self.takes_surprise:
            out = self.store.write(key, value, surprise=float(surprise))
        else:
            out = self.store.write(key, value)
        return int(out[0]), bool(out[1])

    def strength(self, slot: int) -> float:
        """A record's strength: the store's own where it reads one (#336), else its depth."""
        reader = getattr(self.store, "strength", None)
        if callable(reader):
            return float(reader(int(slot)))
        return float(np.asarray(self.store.deep)[int(slot)])

    def fetch(self, cue: np.ndarray) -> tuple[np.ndarray, float] | None:
        """The record a cue (the thing in focus) fetches, as its value and strength; None where
        none is held."""
        read = self.store.read(np.asarray(cue, dtype=np.float64))
        slot = read.first()
        if slot < 0:
            return None
        return np.asarray(self.store.value(read)), self.strength(slot)


__all__ = ["HeardRecords"]
