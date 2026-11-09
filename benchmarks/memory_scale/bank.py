"""The candidate of cadence#288: a long-term bank of records under a fixed budget of slots.

A record is a key (an encoded meaning) with a value. The bank reads in two stages and keeps what
is used.

**The coarse stage: a sparse address.** Every key is coded as the library's pattern separation
codes it (``cadence.PatternSeparator``): a fixed random projection onto ``expansion`` units, each
unit's weights at unit length, of which the ``winners`` most driven are the key's address. Each
unit keeps the slots whose address holds it. A cue activates its own winners, every slot sums the
active units it listens to, and the ``candidates`` most driven slots go on. No slot's key is read
at this stage.

**The fine stage: the recall memory's read.** Over the candidates alone,

    p_i = softmax_i(cue . K_i / temperature)

and the read is ``sum_i p_i V_i``, as ``cadence.Recall`` reads its slots. ``temperature=0`` reads
the first choice whole. Several heads read in turn under the memory's own law
(``cadence.stream._share``): a slot is read in full once per moment, so each head takes of every
candidate at most the share the heads before it left. ``candidates=0`` scores every slot, the
read without a coarse stage.

**What stays: depth and recent use.** Every slot keeps three leaky sums and no count or time
stamp, after ``memory_fade`` (cadence#253):

* ``depth``, the attention the reads gave it plus one per write, kept at ``1 - 1 / hold`` a
  moment (``hold=inf`` keeps the whole sum, as ``memory_fade`` does);
* ``recent``, the reads' attention alone, kept at ``1 - 1 / idle`` a moment;
* ``faded``, how far the record has faded. Every moment (``moment``) it grows by

      idle_i / (1 + (depth_i / depth) ** ORDER),    idle_i = 1 / (1 + (recent_i / FLOOR) ** 2)

  and a read that attends the slot with weight ``p`` leaves ``1 - p`` of it.

A write into a full bank replaces the most faded slot: an idle shallow record goes first, a
record in use does not fade, and a deep one fades far more slowly. ``fade=False`` replaces the
slot written longest ago instead, the recall memory's rule.

**Revision.** A write first reads with its own key. Where its first choice holds a key at
cosine ``revise`` or above, the write is that record told again: it rewrites the slot's key and
value in place and deepens it, so a fact that changed is current and no second copy stays behind.
``revise=0`` appends every write.

This module is an instrument-level candidate; nothing here changes the library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cadence.stream import PatternSeparator, _left, _share

ORDER = 4.0  # the order of the depth's law, as ``SynapticMemory.FADE_ORDER``
FLOOR = 0.01  # the recent use below which a slot counts as idle, as ``FADE_FLOOR``


@dataclass
class Read:
    """One read: the candidate slots, and each head's weights over them."""

    slots: np.ndarray
    weights: list[np.ndarray]

    def first(self, head: int = 0) -> int:
        """The slot the head attends most; -1 for a read of nothing."""
        w = self.weights[head]
        return int(self.slots[int(np.argmax(w))]) if len(w) and w.max() > 0 else -1

    def mass(self, slot: int, head: int = 0) -> float:
        """The head's weight on ``slot``."""
        at = np.flatnonzero(self.slots == slot)
        return float(self.weights[head][at[0]]) if len(at) else 0.0


@dataclass
class Bank:
    width: int
    value_width: int
    capacity: int
    temperature: float = 0.0
    candidates: int = 256
    expansion: int = 8192
    winners: int = 64
    depth: float = 2.0
    idle: float | None = None
    hold: float | None = None
    revise: float = 0.9
    fade: bool = True
    seed: int = 0
    dtype: Any = np.float64
    count: int = field(init=False, default=0)
    writes: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for name in ("width", "value_width", "capacity"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.temperature < 0 or self.candidates < 0 or not 0 <= self.revise <= 1:
            raise ValueError("temperature and candidates are nonnegative, revise lies in [0, 1]")
        if self.depth <= 0:
            raise ValueError("depth must be positive (inf for a bank that reads recent use alone)")
        if self.idle is None:
            self.idle = max(self.capacity / 16.0, 2.0)
        if self.hold is None:
            self.hold = 8.0 * self.capacity
        if self.idle <= 1 or self.hold <= 1:
            raise ValueError("idle and hold must exceed one moment")
        n = int(self.capacity)
        self.keys = np.zeros((n, self.width), dtype=self.dtype)
        self.values = np.zeros((n, self.value_width), dtype=self.dtype)
        self.deep = np.zeros(n)
        self.recent = np.zeros(n)
        self.faded = np.zeros(n)
        self._work = np.zeros((2, n))  # the fading's scratch
        self._cursor = 0  # the slot written longest ago (``fade=False``)
        self.coarse = bool(self.candidates) and self.candidates < n
        if self.coarse:
            if not 0 < self.winners <= self.expansion:
                raise ValueError("winners must lie in [1, expansion]")
            separator = PatternSeparator(self.width, self.expansion, self.winners, seed=self.seed)
            assert separator.projection is not None
            # every unit's weights at unit length, so that the units win equally often and the
            # lists stay near one length
            weights = separator.projection / np.linalg.norm(separator.projection, axis=0)
            self.projection = np.ascontiguousarray(weights.astype(self.dtype))
            self.units = np.full((n, self.winners), -1, dtype=np.int32)
            self.places = np.zeros((n, self.winners), dtype=np.int32)
            room = max(8, int(2 * n * self.winners / self.expansion) + 8)
            self.lists = np.zeros((self.expansion, room), dtype=np.int32)
            self.counts = np.zeros(self.expansion, dtype=np.int32)

    # -- the coarse stage

    def address(self, key: np.ndarray) -> np.ndarray:
        """The key's address: its ``winners`` most driven units, those driven above zero."""
        drive = np.asarray(key, dtype=self.dtype) @ self.projection
        top = np.argpartition(drive, -self.winners)[-self.winners:]
        return top[drive[top] > 0].astype(np.int32)

    def _listen(self, units: np.ndarray) -> np.ndarray:
        """Every slot's sum over the active ``units`` it listens to."""
        lists, counts = self.lists, self.counts
        ids = np.concatenate([lists[u, : counts[u]] for u in units.tolist()] or [lists[0, :0]])
        return np.bincount(ids, minlength=self.count)

    def _candidates(
        self, key: np.ndarray, units: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The slots the fine stage reads for ``key`` and their keys: every held slot without a
        coarse stage, else the ``candidates`` slots its address drives most."""
        if not self.coarse or self.count <= self.candidates:
            return np.arange(self.count), self.keys[: self.count]
        chosen = self._driven(self.address(key) if units is None else units)
        return chosen, self.keys[chosen]

    def _driven(self, units: np.ndarray) -> np.ndarray:
        """The ``candidates`` slots the active ``units`` drive most."""
        heard = self._listen(units)
        # the slots driven at or above the lowest level that fits whole, then slots of the level
        # below, in slot order, until the candidates are full
        above = np.cumsum(np.bincount(heard)[::-1])[::-1]
        level = int(np.argmax(above <= self.candidates))
        chosen = np.flatnonzero(heard >= level)
        if len(chosen) < self.candidates and level:
            more = np.flatnonzero(heard == level - 1)[: self.candidates - len(chosen)]
            chosen = np.concatenate([chosen, more])
        return chosen

    def _file(self, slot: int, units: np.ndarray) -> None:
        """The slot joins the lists of its address's units."""
        at = self.counts[units]
        if len(at) and at.max() >= self.lists.shape[1]:
            wider = np.zeros((self.expansion, 2 * self.lists.shape[1]), dtype=np.int32)
            wider[:, : self.lists.shape[1]] = self.lists
            self.lists = wider
        self.lists[units, at] = slot
        self.units[slot] = -1
        self.units[slot, : len(units)] = units
        self.places[slot, : len(units)] = at
        self.counts[units] = at + 1

    def _unfile(self, slot: int) -> None:
        """The slot leaves the lists of its address's units: each list's last slot takes its
        place."""
        units = self.units[slot]
        units, places = units[units >= 0], self.places[slot][self.units[slot] >= 0]
        last = self.counts[units] - 1
        moved = self.lists[units, last]
        self.lists[units, places] = moved
        column = np.argmax(self.units[moved] == units[:, None], axis=1)
        self.places[moved, column] = places
        self.counts[units] = last

    # -- the fine stage

    def read(self, cue: np.ndarray, heads: int = 1, credit: bool = True) -> Read:
        """Read the bank with ``cue``: the candidates and each head's weights over them. With
        ``credit`` the read deepens the slots it attends; a probe passes False."""
        cue = np.asarray(cue, dtype=np.float64)
        if not self.count:
            return Read(np.zeros(0, dtype=np.int64), [np.zeros(0) for _ in range(heads)])
        slots, keys = self._candidates(cue)
        scores = np.asarray(keys @ cue.astype(self.dtype), dtype=np.float64)
        weights = self._weights(scores, heads)
        if credit:
            for w in weights:
                self.deep[slots] += w
                self.recent[slots] += w
                self.faded[slots] *= 1.0 - np.minimum(w, 1.0)
        return Read(slots, weights)

    def _weights(self, scores: np.ndarray, heads: int) -> list[np.ndarray]:
        if self.temperature == 0:  # each head reads its first choice among the slots left, whole
            out, left = [], np.ones(len(scores), dtype=bool)
            for _ in range(heads):
                w = np.zeros(len(scores))
                if left.any():
                    best = int(np.argmax(np.where(left, scores, -np.inf)))
                    w[best], left[best] = 1.0, False
                out.append(w)
            return out
        logits = scores[None, :] / self.temperature
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        out, left = [p[0]], _left(p)
        for _ in range(1, heads):
            p = _share(logits, left, np.ones(1))
            out.append(p[0])
            left = left * _left(p)
        return out

    def value(self, read: Read, head: int = 0) -> np.ndarray:
        """What the head holds: its weights over the candidates' values."""
        if not len(read.slots):
            return np.zeros(self.value_width)
        return np.asarray(read.weights[head] @ self.values[read.slots], dtype=np.float64)

    # -- the write

    def write(self, key: np.ndarray, value: np.ndarray) -> tuple[int, bool]:
        """Write one record and return its slot and whether the write revised a held record.
        A key the bank recognises (``revise``) rewrites that record; any other takes an empty
        slot, or in a full bank the most faded one."""
        key = np.asarray(key, dtype=np.float64)
        units = self.address(key) if self.coarse else None
        slot, revised = -1, False
        if self.revise and self.count:
            slots, keys = self._candidates(key, units)
            scores = np.asarray(keys @ key.astype(self.dtype), dtype=np.float64)
            best = int(np.argmax(scores))
            norm = float(np.linalg.norm(key) * np.linalg.norm(self.keys[slots[best]]))
            if norm > 0 and scores[best] / norm >= self.revise:
                slot, revised = int(slots[best]), True
        if slot < 0:
            if self.count < self.capacity:
                slot = self.count
                self.count += 1
            elif self.fade:
                slot = int(np.argmax(self.faded))
            else:
                slot = self._cursor
            if not self.fade:
                self._cursor = (slot + 1) % self.capacity
            self.deep[slot] = self.recent[slot] = 0.0
        if self.coarse:
            self._unfile(slot)
            assert units is not None
            self._file(slot, units)
        self.keys[slot], self.values[slot] = key, value
        self.deep[slot] += 1.0  # a write deepens; only a read marks recent use
        self.faded[slot] = 0.0
        self.writes += 1
        return slot, revised

    def moment(self) -> None:
        """One moment of the stream: every held slot fades by its depth's and its recent use's
        law, and the recent use and the depth decay."""
        if not self.fade or not self.count:
            return
        n = self.count
        assert self.idle is not None and self.hold is not None
        recent, deep = self.recent[:n], self.deep[:n]
        rate = self._work[0, :n]  # 1 + (recent / FLOOR) ** 2, then its product with the depth's
        np.multiply(recent, 1.0 / FLOOR, out=rate)
        np.multiply(rate, rate, out=rate)
        rate += 1.0
        if np.isfinite(self.depth):
            law = self._work[1, :n]  # 1 + (depth / depth scale) ** ORDER
            np.multiply(deep, 1.0 / self.depth, out=law)
            np.multiply(law, law, out=law)  # ORDER is 4: two squarings
            np.multiply(law, law, out=law)
            law += 1.0
            rate *= law
        np.divide(1.0, rate, out=rate)
        self.faded[:n] += rate
        recent *= 1.0 - 1.0 / self.idle
        if np.isfinite(self.hold):
            deep *= 1.0 - 1.0 / self.hold

    @property
    def nbytes(self) -> int:
        """The bytes the bank holds: records, traces and, with a coarse stage, its address."""
        held = [self.keys, self.values, self.deep, self.recent, self.faded]
        if self.coarse:
            held += [self.units, self.places, self.lists, self.counts, self.projection]
        return int(sum(a.nbytes for a in held))
