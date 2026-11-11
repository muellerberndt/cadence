"""A long-term memory of records under a fixed budget of slots.

``LongTermMemory`` holds records, each a key (an encoded meaning) with a value, for one stream.
It reads in two stages, keeps what is used, and revises what is told again. It stands beside a
brain as the other standalone stores do: the caller tells it records and holds what it reads on
input coordinates of the moment (docs/memory.md, "A long-term memory of records").

**The coarse stage: a sparse address.** Every key is coded as pattern separation codes it
(``PatternSeparator``): a fixed random projection onto ``expansion`` units, each unit's weights
at unit length, of which the ``winners`` most driven are the key's address. Each unit keeps the
slots whose address holds it. A cue activates its own winners, every slot sums the active units
it listens to, and the ``candidates`` most driven slots go on. No slot's key is read at this
stage. A memory of at most ``direct`` slots has no coarse stage and scores every slot, which costs
less than the address up to about twice the address's units; so does ``candidates=0``.

**The fine stage: the recall memory's read.** Over the candidates alone,

    p_i = softmax_i(cue . K_i / temperature)

and the read is ``sum_i p_i V_i``, as ``Recall`` reads its slots. With ``cosine=True`` the score
is the cosine, ``cue . K_i / (|cue| |K_i|)``, so a key that holds the cue and more scores below the
cue's own key where the dot product ties them. The founder ``temperature=0`` reads the first
choice whole. Several heads read in turn under the recall memory's law: a slot
is read in full once per moment, so each head takes of every candidate at most the share the
heads before it left; at temperature 0 each head reads its first choice among the slots left.

**What stays: depth and recent use.** Every slot keeps three leaky sums and no count or time
stamp, in the form cadence#253 gives the fading of the associative store's records:

* ``deep``, the attention the reads gave it plus one per write, kept at ``1 - 1 / hold`` a
  moment;
* ``recent``, the reads' attention alone, kept at ``1 - 1 / idle`` a moment;
* ``faded``, how far the record has faded. Every moment (``moment``) it grows by

      idle_i / (1 + (deep_i / depth) ** ORDER),    idle_i = 1 / (1 + (recent_i / FLOOR) ** 2)

  and a read that attends the slot with weight ``p`` leaves ``1 - p`` of it.

A write into a full memory replaces the most faded slot: an idle shallow record goes first, a
record in use does not fade, and a deep one fades far more slowly. ``fade=False`` replaces the
slot written longest ago instead, as the recall memory drops its oldest slot.

**Revision.** A write first reads with its own key. Where its first choice holds a key at cosine
``revise`` or above, the write is that record told again: it rewrites the slot's key and value in
place and deepens it, so a fact that changed is current and no second copy stays behind.
``revise=0`` appends every write.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .resident import _on_host
from .stream import PatternSeparator, _left, _share
from .stream import columns as _columns

__all__ = ["LongTermMemory", "LongTermRead", "LongTermRecords"]


def _norms(keys: np.ndarray) -> np.ndarray:
    """The length of each row of ``keys``, the same for a row read alone as among many."""
    rows = np.asarray(keys, dtype=np.float64)
    return np.asarray(np.sqrt((rows * rows).sum(axis=1)))


@dataclass
class LongTermRead:
    """One read of a ``LongTermMemory``: the candidate slots, and each head's weights over
    them."""

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
class LongTermMemory:
    """Records under a fixed budget of slots, read by a sparse address and then by content,
    kept by depth and recent use, revised where a key is told again (the module's docstring
    gives the laws).

    ``width`` and ``value_width`` are the widths of a key and a value, ``capacity`` the slots.
    ``temperature`` is the fine read's (0 reads the first choice whole), and ``cosine`` scores
    the fine read by the cosine in place of the dot product. ``candidates``,
    ``expansion`` and ``winners`` shape the coarse stage; a memory of at most ``direct`` slots
    (twice the expansion unless given), or one whose capacity the candidates hold, reads every
    slot. ``depth`` is the depth at which a record fades at half the rate,
    ``hold`` the moments the depth is kept over (8 capacities unless given; ``inf`` keeps the
    whole sum) and ``idle`` the moments the recent use is kept over (a sixteenth of the
    capacity unless given). ``revise`` is the cosine at which a written key is a held record
    told again. ``seed`` draws the address's projection, and ``dtype`` is the type the keys and
    values are held in (float64; float32 halves their bytes).
    """

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
    direct: int | None = None
    dtype: Any = np.float64
    cosine: bool = False
    count: int = field(init=False, default=0)
    writes: int = field(init=False, default=0)

    # the fading's constants: the order of the depth's law and the recent use below which a slot
    # counts as idle
    ORDER = 4.0
    FLOOR = 0.01
    # the settings a description holds, and the arrays a memory keeps
    SETTINGS = ("width", "value_width", "capacity", "temperature", "candidates", "expansion",
                "winners", "depth", "idle", "hold", "revise", "fade", "seed", "direct", "dtype",
                "cosine")
    ARRAYS = ("keys", "values", "deep", "recent", "faded")
    ADDRESS = ("units", "places", "lists", "counts")

    def __post_init__(self) -> None:
        for name in ("width", "value_width", "capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a nonnegative real: 0 reads the first choice")
        if (
            isinstance(self.candidates, bool)
            or not isinstance(self.candidates, (int, np.integer))
            or self.candidates < 0
        ):
            raise ValueError("candidates must be a nonnegative integer: 0 reads every slot")
        if not 0 <= self.revise <= 1:
            raise ValueError("revise must lie in [0, 1]: the cosine of a key told again")
        if not self.depth > 0:
            raise ValueError("depth must be positive (inf reads recent use alone)")
        if self.idle is None:
            self.idle = max(self.capacity / 16.0, 2.0)
        if self.hold is None:
            self.hold = 8.0 * self.capacity
        if not self.idle > 1 or not self.hold > 1:
            raise ValueError("idle and hold must exceed one moment")
        self.dtype = np.dtype(self.dtype)
        if self.dtype.kind != "f":
            raise ValueError("dtype must be a floating type: float64 or float32")
        n = self.capacity
        self.keys: np.ndarray = np.zeros((n, self.width), dtype=self.dtype)
        self.values: np.ndarray = np.zeros((n, self.value_width), dtype=self.dtype)
        self.deep: np.ndarray = np.zeros(n)
        self.recent: np.ndarray = np.zeros(n)
        self.faded: np.ndarray = np.zeros(n)
        # each key's length, which a cosine read divides by (held where ``cosine``)
        self.cosine = bool(self.cosine)
        self.norms: np.ndarray = np.zeros(n if self.cosine else 0)
        self._work = np.zeros((2, n))  # the fading's scratch
        self._cursor = 0  # the slot written longest ago (``fade=False``)
        for name in ("expansion", "winners"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.winners > self.expansion:
            raise ValueError("winners must not exceed the expansion")
        if self.direct is None:
            self.direct = 2 * int(self.expansion)
        if (
            isinstance(self.direct, bool)
            or not isinstance(self.direct, (int, np.integer))
            or self.direct < 0
        ):
            raise ValueError("direct must be a nonnegative integer: the slots read without an "
                             "address")
        self.coarse = bool(self.candidates) and n > max(self.candidates, self.direct)
        if self.coarse:
            separator = PatternSeparator(self.width, self.expansion, self.winners, seed=self.seed)
            assert separator.projection is not None
            # every unit's weights at unit length, so that the units win equally often and the
            # lists stay near one length
            weights = separator.projection / np.linalg.norm(separator.projection, axis=0)
            self.projection: np.ndarray = np.ascontiguousarray(weights.astype(self.dtype))
            self.units: np.ndarray = np.full((n, self.winners), -1, dtype=np.int32)
            self.places: np.ndarray = np.zeros((n, self.winners), dtype=np.int32)
            room = max(8, int(2 * n * self.winners / self.expansion) + 8)
            self.lists: np.ndarray = np.zeros((self.expansion, room), dtype=np.int32)
            self.counts: np.ndarray = np.zeros(self.expansion, dtype=np.int32)

    # -- the coarse stage

    def address(self, key: np.ndarray) -> np.ndarray:
        """The key's address: its ``winners`` most driven units, those driven above zero."""
        drive = np.asarray(key, dtype=self.dtype) @ self.projection
        top = np.argpartition(drive, -self.winners)[-self.winners:]
        return np.asarray(top[drive[top] > 0], dtype=np.int32)

    def _listen(self, units: np.ndarray) -> np.ndarray:
        """Every slot's sum over the active ``units`` it listens to."""
        lists, counts = self.lists, self.counts
        ids = np.concatenate([lists[u, : counts[u]] for u in units.tolist()] or [lists[0, :0]])
        return np.asarray(np.bincount(ids, minlength=self.count))

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
        return np.asarray(chosen)

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
        held = self.units[slot] >= 0
        units, places = self.units[slot][held], self.places[slot][held]
        last = self.counts[units] - 1
        moved = self.lists[units, last]
        self.lists[units, places] = moved
        column = np.argmax(self.units[moved] == units[:, None], axis=1)
        self.places[moved, column] = places
        self.counts[units] = last

    # -- the fine stage

    def read(self, cue: np.ndarray, heads: int = 1, credit: bool = True) -> LongTermRead:
        """Read the memory with ``cue``: the candidates and each head's weights over them. With
        ``credit`` the read deepens the slots it attends; a probe passes False."""
        cue = self._key(cue, "cue")
        if isinstance(heads, bool) or not isinstance(heads, (int, np.integer)) or heads < 1:
            raise ValueError("heads must be a positive integer")
        if not self.count:
            return LongTermRead(np.zeros(0, dtype=np.int64), [np.zeros(0) for _ in range(heads)])
        slots, keys = self._candidates(cue)
        scores = np.asarray(keys @ cue.astype(self.dtype), dtype=np.float64)
        if self.cosine:
            scores = self._cosines(slots, scores, cue)
        weights = self._weights(scores, int(heads))
        if credit:
            for w in weights:  # the slots a head attends; the others keep their sums
                at = np.flatnonzero(w)
                held, share = slots[at], w[at]
                self.deep[held] += share
                self.recent[held] += share
                self.faded[held] *= 1.0 - np.minimum(share, 1.0)
        return LongTermRead(slots, weights)

    def _cosines(self, slots: np.ndarray, dots: np.ndarray, cue: np.ndarray) -> np.ndarray:
        """The candidates' dot products with ``cue`` over their keys' lengths and the cue's: their
        cosines, 0 where a key or the cue is zeros."""
        lengths = self.norms[slots] * _norms(cue.astype(self.dtype)[None, :])[0]
        out = np.zeros(len(dots))
        np.divide(dots, lengths, out=out, where=lengths > 0)
        return out

    def _weights(self, scores: np.ndarray, heads: int) -> list[np.ndarray]:
        if self.temperature == 0:  # each head reads its first choice among the slots left, whole
            first = np.zeros(len(scores))
            first[int(np.argmax(scores))] = 1.0
            if heads == 1:
                return [first]
            out, open_ = [first], first == 0
            for _ in range(1, heads):
                w = np.zeros(len(scores))
                if open_.any():
                    best = int(np.argmax(np.where(open_, scores, -np.inf)))
                    w[best], open_[best] = 1.0, False
                out.append(w)
            return out
        logits = scores[None, :] / self.temperature
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        shares, left = [p[0]], _left(p)
        for _ in range(1, heads):
            p = _share(logits, left, np.ones(1))
            shares.append(p[0])
            left = left * _left(p)
        return shares

    def value(self, read: LongTermRead, head: int = 0) -> np.ndarray:
        """What the head holds: its weights over the candidates' values."""
        if not len(read.slots):
            return np.zeros(self.value_width)
        weights = read.weights[head]
        if self.temperature == 0:  # the first choice whole, or nothing
            first = read.first(head)
            if first < 0:
                return np.zeros(self.value_width)
            return np.array(self.values[first], dtype=np.float64)
        return np.asarray(weights @ self.values[read.slots], dtype=np.float64)

    # -- the write

    def _key(self, key: Any, name: str = "key") -> np.ndarray:
        out = np.asarray(key, dtype=np.float64)
        if out.shape != (self.width,) or not np.isfinite(out).all():
            raise ValueError(f"a {name} is one finite row of {self.width} values")
        return out

    def write(
        self, key: np.ndarray, value: np.ndarray, recognised: int | None = None
    ) -> tuple[int, bool]:
        """Write one record and return its slot and whether the write revised a held record.
        A key the memory recognises (``revise``) rewrites that record; any other takes an empty
        slot, or in a full memory the most faded one. ``recognised`` is the recognition a
        caller made by reading first, the slot the key is told again in or -1 for none: the
        write then reads nothing itself."""
        key = self._key(key)
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.value_width,) or not np.isfinite(value).all():
            raise ValueError(f"a value is one finite row of {self.value_width} values")
        if recognised is not None and not -1 <= int(recognised) < self.count:
            raise ValueError("recognised is a held slot, or -1 for none")
        units = self.address(key) if self.coarse else None
        slot, revised = -1, False
        if recognised is not None and self.revise:
            if recognised >= 0:
                slot, revised = int(recognised), True
        elif self.revise and self.count:
            slots, keys = self._candidates(key, units)
            scores = np.asarray(keys @ key.astype(self.dtype), dtype=np.float64)
            best = int(np.argmax(self._cosines(slots, scores, key) if self.cosine else scores))
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
        if self.cosine:
            self.norms[slot] = _norms(self.keys[slot : slot + 1])[0]
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
        np.multiply(recent, 1.0 / self.FLOOR, out=rate)
        np.multiply(rate, rate, out=rate)
        rate += 1.0
        if np.isfinite(self.depth):
            law = self._work[1, :n]  # 1 + (deep / depth) ** ORDER
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

    # -- what a caller saves

    def to_dict(self) -> dict[str, Any]:
        """The memory's settings, as ``LongTermMemory(**settings)`` takes them."""
        out = {name: getattr(self, name) for name in self.SETTINGS}
        out["dtype"] = self.dtype.name
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        """The memory's state by name: its records, its sums and, with a coarse stage, the lists
        of its address; ``restore`` takes them back."""
        names = (*self.ARRAYS, *(self.ADDRESS if self.coarse else ()))
        out = {name: np.array(getattr(self, name)) for name in names}
        out["state"] = np.array([self.count, self.writes, self._cursor], dtype=np.int64)
        return out

    def restore(self, arrays: dict[str, np.ndarray]) -> None:
        """Take back the ``arrays`` of a memory of the same settings."""
        names = (*self.ARRAYS, *(self.ADDRESS if self.coarse else ()))
        missing = [name for name in (*names, "state") if name not in arrays]
        if missing:
            raise ValueError(f"the arrays of this memory hold {', '.join(missing)} as well")
        for name in names:
            given, held = np.asarray(arrays[name]), getattr(self, name)
            if given.shape != held.shape and not (name == "lists" and given.ndim == 2
                                                  and given.shape[0] == held.shape[0]):
                raise ValueError(f"{name} has another shape: a memory of other settings")
            setattr(self, name, np.array(given, dtype=held.dtype))
        self.count, self.writes, self._cursor = (int(v) for v in np.asarray(arrays["state"]))
        if self.cosine:  # the keys' lengths, as their writes gave them
            self.norms = _norms(self.keys)

    @property
    def nbytes(self) -> int:
        """The bytes the memory holds: records, sums and, with a coarse stage, its address."""
        held = [self.keys, self.values, self.deep, self.recent, self.faded, self.norms]
        if self.coarse:
            held += [self.units, self.places, self.lists, self.counts, self.projection]
        return int(sum(a.nbytes for a in held))


class LongTermRecords:
    """A brain's own long-term memory (``Brain.compose(longterm=records)``): records keyed by
    the brain's held message, on regions the association reads.

    **The key.** The brain's held message (``message=units``) is held from a turn: a heard
    moment with the input channel ``message_boundary`` on, with ``message_boundary_holds``. From
    that moment the message keeps what the episode heard before the turn, so its key
    (``HeldMessage.key()``, the union over its blocks) says what the episode is about, and
    nothing heard after the turn changes it.

    **The write.** The moment after the turn is what the episode says: its observation, as the
    message hears it, is the record's value. The write reads the stream's records with the key
    first. Where a record holds the key (cosine ``revise`` or above) and the same value, the
    moment says nothing more and nothing is written; where it holds the key and another value,
    the write revises that record; where none holds the key, the write takes a slot. A silent
    moment after the turn writes nothing, so a question (the turn, then the brain's answer)
    leaves the records as they are.

    **The read.** Before every settle after a turn the key addresses the stream's records:
    ``heads`` heads read in turn, each its first choice by cosine to the message's key among the
    records the heads before it left (a record whose key holds the message's and more scores
    below the message's own), and a head holds its record only where the record's key lies at
    cosine ``match`` or above to the message's key. Head ``h`` holds ``amplitude`` times the
    record's value on the region ``longterm`` (head 0) or ``longterm_h``, which the association
    reads through plastic synapses, as it reads the recall memory's ports. Before a turn the
    regions are silent.

    **What stays.** Each stream keeps its records in a ``LongTermMemory`` of ``capacity``
    slots: a read deepens the records it holds, every moment the brain lives fades the idle
    ones by their depth and recent use, and a full memory replaces the most faded record. The
    records survive ``Brain.reset`` and every episode's end; ``clear()`` empties them.

    **Turns in a conversation (``turn=channel``).** Every heard moment with the input channel
    ``turn`` on is a turn, and an episode holds as many as it hears; the message is held as
    it is, with no boundary. A turn's key is the content of what was heard since the moment
    after the turn before: the units of the message's code (``Message.code``) of those moments
    that are rare, each unit on in at most a share ``rare`` of the moments the brain has heard
    (a running count over every stream, over at least ``10 / rare`` moments). A unit that the
    frequent words drive drops out, so a statement and a question about one person share the
    person's units. The moment after a turn is read with the key and written under it as
    above, and the next key begins after it.

    **Records read by what tells them apart (``distinct=share``, with turns).** Each distinct
    heard moment (its code under the message) is one entry of a codebook the records share, so
    two words whose codes share units stay two entries. A record holds the whole statement: the
    entries of the turn's window and of the moment after it. An entry's weight is its rarity
    among the stream's own records, ``log((n + 1) / (records holding it + 1))``: an entry every
    record holds weighs nothing and one only a person's records hold weighs most, in a lexicon of
    any size. A turn's cue keeps the window's entries whose weight is at least ``distinct``
    times the cue's heaviest (an entry no record holds weighs nothing), and the read takes the
    latest record that holds at least ``match`` of the cue's weight. Every value is written as a
    new record, so a fact told again is read from its latest telling. With a second head the
    read chains: of the first record's entries outside the cue, the one the most records hold
    (what the record is about: a person recurs in more records than the words around them)
    cues head 1, which reads the latest other record holding it. A stream's records are kept as
    lists: each record's entries, and for each entry the records that hold it, so a read touches
    only the records that hold the cue's entries, and the records and the codebook grow to any
    size.

    **The record that holds the most (``best=True``, with ``distinct``).** The read takes the
    record holding the largest share of the cue's weight, at least ``match``, and the latest
    among records holding the same share. A question that names a subject and a relation then
    reads the record of both, where the latest record above ``match`` may hold the subject
    alone; a fact told again holds the same share as its earlier telling, so the latest
    telling is read.

    **What the question asks beyond the first record (``rest=True``, with ``distinct`` and two
    heads).** Head 1 is cued by the first record's entries outside the cue together with the
    cue's entries that the first record does not hold and that weigh at least ``distinct`` of
    the cue's heaviest. A question of two facts ("the capital of the country of a city") reads
    the city's country with head 0, then the record that holds that country and "capital" with
    head 1. Where the first record holds every such entry, head 1 holds nothing.
    """

    # the store's settings that the brain does not name: the library store's founders, a first
    # choice read whole, and the read scored by cosine
    STORE: ClassVar[dict[str, Any]] = {"temperature": 0.0, "cosine": True}
    # the streams' records as one bank where they allow it (``batched``); False reads, writes
    # and fades them store by store, the reference the bank equals
    BATCHED: ClassVar[bool] = True

    def __init__(
        self,
        connectome: Any,
        message: Any,
        capacity: int,
        heads: int = 1,
        amplitude: float = 3.0,
        match: float = 0.75,
        revise: float = 0.999,
        seed: int = 0,
        turn: int | None = None,
        rare: float = 0.03,
        distinct: float | None = None,
        best: bool = False,
        rest: bool = False,
    ) -> None:
        if turn is None and (
            message is None or message.boundary is None or not message.boundary_holds
        ):
            raise ValueError(
                "longterm keys its records on the brain's held message from a turn: it needs "
                "message=units, message_boundary=channel and message_boundary_holds=True, or "
                "longterm_turn=channel"
            )
        if turn is not None and (message is None or not 0 <= int(turn) < int(message.inputs)):
            raise ValueError("longterm_turn is an input channel of the heard moment: it needs "
                             "message=units")
        if not 0.0 < float(rare) <= 1.0:
            raise ValueError("longterm_rare is a share of the moments heard: it lies in (0, 1]")
        if distinct is not None and (turn is None or not 0.0 < float(distinct) <= 1.0):
            raise ValueError("longterm_distinct is a share of a cue's heaviest weight in (0, 1], "
                             "and it reads turns: it needs longterm_turn=channel")
        if (best or rest) and distinct is None:
            raise ValueError("longterm_best and longterm_rest read records by what tells them "
                             "apart: they need longterm_distinct")
        if rest and int(heads) < 2:
            raise ValueError("longterm_rest cues the second head: it needs longterm_heads >= 2")
        self.distinct = None if distinct is None else float(distinct)
        self.best, self.rest = bool(best), bool(rest)
        self.message = message
        self.turn = None if turn is None else int(turn)
        self.rare = float(rare)
        self.capacity, self.heads = int(capacity), int(heads)
        self.amplitude, self.match, self.revise = float(amplitude), float(match), float(revise)
        self.seed = int(seed)
        self.width, self.value_width = int(message.units), int(message.inputs)
        populations = connectome.populations
        self.columns: list[slice | np.ndarray] = []
        for name in self.ports(self.heads):
            if name not in populations:
                raise ValueError(f"longterm needs a {name!r} population; Brain.compose adds it")
            neurons = np.asarray(populations[name], dtype=np.int64)
            if len(neurons) != self.value_width:
                raise ValueError(
                    f"the {name} population holds {len(neurons)} neurons, where a record's "
                    f"value holds {self.value_width}"
                )
            self.columns.append(_columns(neurons))
        self.stores: list[LongTermMemory] = []
        # whether a store reads through an address (past ``direct`` slots): then the stores
        # are read one by one; and whether it reads by cosine
        probe = self._store()
        self._store_coarse, self._cosine = probe.coarse, probe.cosine
        self._binary = True
        self._bank()
        self.since = np.full(0, -1, dtype=np.int64)  # moments since each stream's turn
        self.read_slots = np.full((0, self.heads), -1, dtype=np.int64)  # the latest read
        self.counts = {"writes": 0, "revisions": 0, "agreed": 0, "reads": 0}
        self._private = False
        # with turns: what each stream heard since its last turn's value, each stream's key at
        # its latest turn, and how often each code unit has been on among the moments heard
        self.window = np.zeros((0, self.width))
        self.key = np.zeros((0, self.width))
        self.heard = np.zeros(self.width)
        self.moments = 0.0
        # with ``distinct``: each stream's slots' write order (0 for none); the codebook of heard
        # moments (an entry for each code, and each entry's code); each stream's records as
        # entries, each slot's entries in ascending order and each entry's slots; and each
        # stream's window and cue as entries
        self.stamps = np.zeros((0, self.capacity))
        self._entries: dict[bytes, int] = {}
        self._codes: list[np.ndarray] = []
        self._members: list[dict[int, np.ndarray]] = []
        self._postings: list[dict[int, list[int]]] = []
        self._wsets: list[set[int]] = []
        self._cues: list[np.ndarray] = []

    @staticmethod
    def ports(heads: int) -> tuple[str, ...]:
        """The regions the heads hold their records on."""
        return ("longterm", *(f"longterm_{h}" for h in range(1, int(heads))))

    def _store(self) -> LongTermMemory:
        return LongTermMemory(self.width, self.value_width, self.capacity, revise=self.revise,
                              seed=self.seed, **self.STORE)

    # -- the streams' records as one bank

    @property
    def batched(self) -> bool:
        """Whether the streams' records are read, written and faded as one bank: every slot
        scored (no address), and keys of zeros and ones, which a held message gives."""
        return self.BATCHED and self._binary and not self._store_coarse

    def _bank(self) -> None:
        """The streams' sums (``deep``, ``recent``, ``faded``) as rows of three shared arrays,
        each store's arrays views of its row, and each stream's keys as an index of the slots
        that hold each unit, ``(streams, units, slots)``: a read sums the rows of the cue's
        units, one product over every stream."""
        streams, capacity = len(self.stores), self.capacity
        names = ("deep", "recent", "faded", *(("norms",) if self._cosine else ()))
        sums = [np.zeros((streams, capacity)) for _ in range(4)]
        index = np.zeros((streams, self.width, capacity), dtype=np.uint8)
        binary = True
        for s, store in enumerate(self.stores):
            for shared, name in zip(sums, names, strict=False):
                shared[s] = getattr(store, name)
                setattr(store, name, shared[s])
            held = np.asarray(store.keys[: store.count])
            binary = binary and bool(np.isin(held, (0.0, 1.0)).all())
            index[s, :, : store.count] = held.T.astype(np.uint8)
        self._deep, self._recent, self._faded, self._norms = sums
        self._index = index
        self._binary = binary
        self._cache: dict[int, tuple[int, np.ndarray, int]] = {}

    def _streams(self, batch: int) -> None:
        """Every stream of a batch of ``batch`` has its records; a stream that begins holds no
        turn."""
        if len(self.stores) < batch:
            while len(self.stores) < batch:
                self.stores.append(self._store())
            self._bank()
        if len(self.stamps) < len(self.stores):
            grown = np.zeros((len(self.stores), self.capacity))
            grown[: len(self.stamps)] = self.stamps
            self.stamps = grown
        while len(self._members) < len(self.stores):
            self._members.append({})
            self._postings.append({})
        if self.distinct is not None and len(self._wsets) != batch:
            self._wsets = [set() for _ in range(batch)]
            self._cues = [np.zeros(0, dtype=np.int64) for _ in range(batch)]
        if len(self.since) != batch:
            self.since = np.full(batch, -1, dtype=np.int64)
            self.read_slots = np.full((batch, self.heads), -1, dtype=np.int64)
        if self.turn is not None and len(self.window) != batch:
            self.window = np.zeros((batch, self.width))
            self.key = np.zeros((batch, self.width))

    # -- the records' entries (``distinct``), as arrays

    @property
    def entries(self) -> int:
        """The codebook's entries: the distinct moments heard."""
        return len(self._codes)

    @property
    def codebook(self) -> np.ndarray:
        """Each entry's code, ``(entries, units)``."""
        if not self._codes:
            return np.zeros((0, self.width))
        return np.asarray(np.stack(self._codes), dtype=np.float64)

    @property
    def sets(self) -> np.ndarray:
        """Which entries each stream's slots hold, ``(streams, capacity, entries)``, zeros and
        ones (built from the lists; for a small memory)."""
        out = np.zeros((len(self.stores), self.capacity, self.entries), dtype=np.uint8)
        for row, members in enumerate(self._members[: len(self.stores)]):
            for slot, held in members.items():
                out[row, slot, held] = 1
        return out

    @property
    def wset(self) -> np.ndarray:
        """Each stream's window as entries, ``(batch, entries)``."""
        out = np.zeros((len(self._wsets), self.entries), dtype=np.uint8)
        for row, held in enumerate(self._wsets):
            out[row, sorted(held)] = 1
        return out

    @property
    def cue(self) -> np.ndarray:
        """Each stream's cue at its latest turn as entries, ``(batch, entries)``."""
        out = np.zeros((len(self._cues), self.entries), dtype=np.uint8)
        for row, held in enumerate(self._cues):
            out[row, held] = 1
        return out

    def _keys(self, batch: int) -> np.ndarray | None:
        """Each stream's key, ``(batch, units)``, where the message holds that batch; with turns
        each stream's key at its latest turn."""
        if self.turn is not None:
            return self.key if len(self.key) == batch else None
        held = self.message
        if len(held.trace) != batch:
            return None
        return np.asarray(held.key(), dtype=np.float64)

    def _holds(self, store: LongTermMemory, slot: int, key: np.ndarray, bar: float) -> bool:
        """Whether the record in ``slot`` holds ``key``: their cosine at ``bar`` or above."""
        if slot < 0:
            return False
        held = np.asarray(store.keys[slot], dtype=np.float64)
        norm = float(np.linalg.norm(held) * np.linalg.norm(key))
        return norm > 0 and float(held @ key) / norm >= bar - 1e-12

    def _choices(self, rows: np.ndarray, cues: np.ndarray, heads: int) -> np.ndarray:
        """Each stream's first choices among its records, ``(rows, heads)`` (-1 for none), as
        the store's read at temperature 0 makes them: the slot whose key shares the most of
        the cue's units (over the key's length and the cue's, by cosine), the lowest among
        equals, and each head the best the heads before it left. One product over every
        stream: the index rows of the cues' units, summed per stream."""
        counts = np.array([self.stores[int(r)].count for r in rows])
        active = [np.flatnonzero(cue) for cue in cues]
        lengths = np.array([len(units) for units in active])
        gathered = self._index[np.repeat(rows, lengths), np.concatenate(active)]
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        shared = np.add.reduceat(gathered, starts, axis=0, dtype=np.int64)
        scores = np.asarray(shared, dtype=np.float64)
        if self._cosine:  # over each key's length and the cue's, as the store's read divides
            norm = self._norms[rows] * np.sqrt(lengths.astype(np.float64))[:, None]
            scores = np.zeros(shared.shape)
            np.divide(shared, norm, out=scores, where=norm > 0)
        scores[np.arange(self.capacity)[None, :] >= counts[:, None]] = -1
        out = np.full((len(rows), heads), -1, dtype=np.int64)
        each = np.arange(len(rows))
        for head in range(heads):
            pick = np.argmax(scores, axis=1)
            open_ = scores[each, pick] >= 0
            out[open_, head] = pick[open_]
            scores[each[open_], pick[open_]] = -2
        return out

    # -- the read, before the settle

    def stimulus(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The drive with each head's record on its region; silent where a stream holds no
        turn."""
        out = np.array(_on_host(drive), dtype=float)
        batch = len(out)
        self._streams(batch)
        for columns in self.columns:
            out[:, columns] = 0.0
        self.read_slots[:] = -1
        keys = self._keys(batch)
        if keys is None:
            return out
        reading = self.since >= 0 if self.turn is None else self.since == 0
        cued = np.asarray([len(c) > 0 for c in self._cues] if self.distinct is not None
                          else keys.any(axis=1), dtype=bool)
        rows = np.array([
            int(row) for row in np.flatnonzero(reading)
            if self.stores[int(row)].count and cued[row]
        ], dtype=np.int64)
        if not len(rows):
            return out
        if self.distinct is not None:
            for row in rows:
                store, cue = self.stores[int(row)], self._cues[int(row)]
                slots = [self._read_distinct(int(row), cue)]
                for _ in range(1, self.heads):  # each head chains from the record before
                    slots.append(self._rest(int(row), slots, cue) if self.rest
                                 else self._chain(int(row), slots, cue))
                for head, (slot, columns) in enumerate(zip(slots, self.columns, strict=True)):
                    if slot >= 0:
                        out[row, columns] = self.amplitude * store.values[slot]
                        self.read_slots[row, head] = slot
            self.counts["reads"] += 0 if self._private else len(rows)
            return out
        if self.batched:
            picks = self._choices(rows, keys[rows], self.heads)
            if not self._private:
                for head in range(self.heads):  # a read deepens what it holds, as the store's
                    held = picks[:, head] >= 0
                    at, slot = rows[held], picks[held, head]
                    self._deep[at, slot] += 1.0
                    self._recent[at, slot] += 1.0
                    self._faded[at, slot] *= 0.0
                self.counts["reads"] += len(rows)
                for i, row in enumerate(rows):  # the read a write after the settle reuses
                    store = self.stores[int(row)]
                    self._cache[int(row)] = (int(picks[i, 0]), keys[row].copy(), store.writes)
        else:
            picks = np.full((len(rows), self.heads), -1, dtype=np.int64)
            for i, row in enumerate(rows):
                read = self.stores[int(row)].read(
                    keys[row], heads=self.heads, credit=not self._private
                )
                picks[i] = [read.first(head) for head in range(self.heads)]
            self.counts["reads"] += 0 if self._private else len(rows)
        for i, row in enumerate(rows):
            store, key = self.stores[int(row)], keys[row]
            for head, columns in enumerate(self.columns):
                slot = int(picks[i, head])
                if self._holds(store, slot, key, self.match):
                    out[row, columns] = self.amplitude * store.values[slot]
                    self.read_slots[row, head] = slot
        return out

    # -- the write, after the settle

    def hear(self, observations: np.ndarray) -> None:
        """After the message has heard the moment: a stream whose message turned at this moment
        holds its turn, and the moment after a turn is written where it says more than the
        record the key finds; every stream's records live one moment."""
        x = np.asarray(observations, dtype=float)
        batch = len(x)
        self._streams(batch)
        if self.turn is not None:
            self._hear_turns(x)
            return
        held = self.message.held if len(self.message.held) == batch else np.zeros(batch, bool)
        keys = self._keys(batch)
        turned = held & (self.since < 0)
        after = self.since >= 0
        self.since = np.where(after, self.since + 1, np.where(turned, 0, -1))
        if self._private:
            return
        if keys is not None:
            for row in np.flatnonzero(self.since == 1):
                if x[row].any() and keys[row].any():
                    self.record(int(row), keys[row], x[row])
        self._cache = {}
        self._live(batch)

    def content(self, window: np.ndarray) -> np.ndarray:
        """The rare units of ``window``: those on in at most a share ``rare`` of the moments
        heard."""
        # until the brain has heard ten moments a unit's share allows, a unit is content until
        # it has been on ten times
        share = self.heard / max(self.moments, 10.0 / self.rare)
        return np.asarray(window * (share <= self.rare + 1e-12), dtype=np.float64)

    def weights(self, row: int, entries: np.ndarray | None = None) -> np.ndarray:
        """The weight among stream ``row``'s records of each of ``entries`` (every codebook
        entry where None): ``log((n + 1) / (held + 1))`` for an entry ``held`` of the ``n``
        records hold, 0 for an entry none holds."""
        n = self.stores[row].count
        postings = self._postings[row] if row < len(self._postings) else {}
        which = np.arange(self.entries) if entries is None else np.asarray(entries)
        held = np.array([len(postings.get(int(e), ())) for e in which], dtype=np.float64)
        return np.where(held > 0, np.log((n + 1.0) / (held + 1.0)), 0.0)

    def entry(self, code: np.ndarray) -> int:
        """The codebook entry of a heard moment's code, added where it is new."""
        name = np.asarray(code, dtype=np.uint8).tobytes()
        found = self._entries.get(name)
        if found is not None:
            return found
        found = len(self._codes)
        self._entries[name] = found
        self._codes.append(np.asarray(code, dtype=np.uint8).copy())
        return found

    def _hold(self, row: int, slot: int, entries: np.ndarray) -> None:
        """Slot ``slot`` of stream ``row`` holds ``entries`` (ascending), in place of what it
        held."""
        members, postings = self._members[row], self._postings[row]
        for e in members.pop(slot, np.zeros(0, dtype=np.int64)).tolist():
            held = postings[e]
            held.remove(slot)
            if not held:
                del postings[e]
        members[slot] = entries
        for e in entries.tolist():
            postings.setdefault(e, []).append(slot)

    def _read_distinct(self, row: int, cue: np.ndarray, exclude: list[int] | None = None) -> int:
        """The latest record of stream ``row`` that holds at least ``match`` of the weight of
        the cue's distinct entries (``cue`` the entries, ascending); with ``best`` the latest of
        those that hold the most; -1 for none."""
        n = self.stores[row].count
        cue = np.asarray(cue, dtype=np.int64)
        if not n or not len(cue):
            return -1
        assert self.distinct is not None
        weight = self.weights(row, cue)
        if not weight.any():
            return -1
        weight = weight * (weight >= self.distinct * weight.max() - 1e-12)
        kept = np.flatnonzero(weight)
        postings = self._postings[row]
        parts = [np.asarray(postings[int(cue[i])], dtype=np.int64) for i in kept]
        slots = np.concatenate(parts)
        shares = np.repeat(weight[kept], [len(p) for p in parts])
        found, at = np.unique(slots, return_inverse=True)
        held = np.bincount(at, weights=shares, minlength=len(found)) / weight.sum()
        if self.match <= 1e-12:  # every record holds a share of none at least
            every = np.zeros(n)
            every[found] = held
            found, held = np.arange(n), every
        able = held >= self.match - 1e-12
        if exclude:
            able &= ~np.isin(found, np.asarray(exclude, dtype=np.int64))
        if not able.any():
            return -1
        if self.best:  # the records that hold the most of the cue's weight
            able &= held >= held[able].max() - 1e-9
        stamps = self.stamps[row, found]
        return int(found[int(np.argmax(np.where(able, stamps, -1.0)))])

    def _chain(self, row: int, slots: list[int], cue: np.ndarray) -> int:
        """The next head's record: the latest other record holding the entry of the record
        before that the most records hold among those outside the cue with a weight (what that
        record is about: a person recurs in more records than the words around them)."""
        if slots[-1] < 0:
            return -1
        members = self._members[row][slots[-1]]
        outside = members[~np.isin(members, cue)]
        weight = self.weights(row, outside)
        if not weight.any():
            return -1
        about = int(outside[int(np.argmin(np.where(weight > 0, weight, np.inf)))])
        return self._read_distinct(row, np.array([about]), exclude=slots)

    def _rest(self, row: int, slots: list[int], cue: np.ndarray) -> int:
        """The next head's record with ``rest``: cued by the record before's entries outside
        the cue and the cue's entries of weight (at least ``distinct`` of its heaviest) that the
        record before does not hold; -1 where it holds them all."""
        if slots[-1] < 0:
            return -1
        assert self.distinct is not None
        members = self._members[row][slots[-1]]
        weight = self.weights(row, cue)
        heavy = weight >= max(self.distinct * float(weight.max(initial=0.0)), 1e-12)
        asked = cue[heavy & ~np.isin(cue, members)]
        if not len(asked):
            return -1
        told = members[~np.isin(members, cue)]
        return self._read_distinct(row, np.union1d(told, asked), exclude=slots)

    def _hear_turns(self, x: np.ndarray) -> None:
        """One moment with turns: the moment after a turn is written under the turn's key and
        begins the next window, a turn takes its key from its window, and any other moment
        joins the window."""
        batch = len(x)
        codes = np.asarray(self.message.code(x), dtype=np.float64)
        value = self.since == 0
        turned = (x[:, self.turn] > 0) & ~value
        if not self._private:
            self.heard += codes.sum(axis=0)
            self.moments += batch
            for row in np.flatnonzero(value):
                if not (x[row].any() and self.key[row].any()):
                    continue
                if self.distinct is None:
                    self.record(int(row), self.key[row], x[row])
                    continue
                # the whole statement, every telling a record of its own
                store = self.stores[int(row)]
                whole = np.maximum(self.key[row], codes[row])
                said = self.entry(codes[row])
                slot, _ = store.write(whole, x[row], recognised=-1)
                if self.batched:
                    self._index[row, :, slot] = whole.astype(np.uint8)
                self._hold(int(row), int(slot), np.union1d(self._cues[row], [said]))
                self.stamps[row, slot] = store.writes
                self.counts["writes"] += 1
        grow = ~value & ~turned
        self.window[grow] = np.maximum(self.window[grow], codes[grow])
        if self.distinct is not None:
            for row in np.flatnonzero(grow):
                if codes[row].any():
                    self._wsets[row].add(self.entry(codes[row]))
        for row in np.flatnonzero(turned):
            self.key[row] = (self.content(self.window[row]) if self.distinct is None
                             else self.window[row].copy())
            self.window[row] = 0.0
            if self.distinct is not None:
                self._cues[row] = np.array(sorted(self._wsets[row]), dtype=np.int64)
                self._wsets[row] = set()
        self.since = np.where(turned, 0, -1).astype(np.int64)
        if self._private:
            return
        self._cache = {}
        self._live(batch)

    def record(self, row: int, key: np.ndarray, value: np.ndarray) -> None:
        """A statement of stream ``row``: ``value`` under ``key``, written where it says more
        than the record the key finds (the same key and value write nothing, the same key and
        another value revise that record, a key no record holds takes a slot)."""
        store = self.stores[row]
        found = -1
        if store.count and self.batched:
            cached = self._cache.get(row)
            if cached is not None and cached[2] == store.writes and np.array_equal(cached[1], key):
                found = cached[0]  # the read before the settle, of the same records and key
            else:
                found = int(self._choices(np.array([row]), key[None, :], 1)[0, 0])
        elif store.count:
            found = store.read(key, credit=False).first()
        if store.count and self._holds(store, found, key, self.revise) and np.array_equal(
            np.asarray(store.values[found], dtype=np.float64), value
        ):
            self.counts["agreed"] += 1  # the moment says nothing the record does not
            return
        if self.batched:
            # the store's own recognition, on the slot the read found
            recognised = -1
            if found >= 0 and store.revise:
                score = float(np.asarray(store.keys[found]) @ key.astype(store.dtype))
                norm = float(np.linalg.norm(key) * np.linalg.norm(store.keys[found]))
                if norm > 0 and score / norm >= store.revise:
                    recognised = found
            slot, revised = store.write(key, value, recognised=recognised)
            if not np.isin(key, (0.0, 1.0)).all():
                self._binary = False  # a key of other values: the stores read one by one
            self._index[row, :, slot] = key.astype(np.uint8)
        else:
            slot, revised = store.write(key, value)
        self.counts["writes"] += 1
        self.counts["revisions"] += int(revised)

    def _live(self, batch: int) -> None:
        """One moment of every stream's records: each fades by the store's law."""
        if not self.batched:
            for store in self.stores[:batch]:
                store.moment()
            return
        held = max((store.count for store in self.stores[:batch]), default=0)
        if not held:
            return
        probe = self.stores[0]
        assert probe.idle is not None and probe.hold is not None
        recent = self._recent[:batch, :held]
        deep = self._deep[:batch, :held]
        rate = recent * (1.0 / probe.FLOOR)
        np.multiply(rate, rate, out=rate)
        rate += 1.0
        if np.isfinite(probe.depth):
            law = deep * (1.0 / probe.depth)
            np.multiply(law, law, out=law)  # ORDER is 4: two squarings
            np.multiply(law, law, out=law)
            law += 1.0
            rate *= law
        np.divide(1.0, rate, out=rate)
        self._faded[:batch, :held] += rate
        recent *= 1.0 - 1.0 / probe.idle
        if np.isfinite(probe.hold):
            deep *= 1.0 - 1.0 / probe.hold

    # -- the memory contract

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Streams that begin hold no turn; every record stays."""
        if rows is None or len(self.since) != batch:
            self.since = np.full(batch, -1, dtype=np.int64)
            self.read_slots = np.full((batch, self.heads), -1, dtype=np.int64)
            if self.turn is not None:
                self.window = np.zeros((batch, self.width))
                self.key = np.zeros((batch, self.width))
                self._wsets = [set() for _ in range(batch)]
                self._cues = [np.zeros(0, dtype=np.int64) for _ in range(batch)]
        else:
            self.since[rows] = -1
            self.read_slots[rows] = -1
            if self.turn is not None and len(self.window) == batch:
                self.window[rows] = 0.0
                self.key[rows] = 0.0
            if len(self._wsets) == batch:
                for row in np.arange(batch)[rows].tolist():
                    self._wsets[row] = set()
                    self._cues[row] = np.zeros(0, dtype=np.int64)
        self._cache = {}

    def keep(self, rows: np.ndarray) -> None:
        """Keep the streams of ``rows``, their records with them."""
        index = np.arange(len(self.since))[rows]
        self.stores = [self.stores[int(i)] for i in index if int(i) < len(self.stores)]
        self.since, self.read_slots = self.since[index], self.read_slots[index]
        if self.turn is not None and len(self.window) >= len(index):
            self.window, self.key = self.window[index], self.key[index]
        if len(self.stamps) >= len(index):
            self.stamps = self.stamps[index]
        kept = [int(i) for i in index if int(i) < len(self._members)]
        self._members = [self._members[i] for i in kept]
        self._postings = [self._postings[i] for i in kept]
        if len(self._wsets) >= len(index):
            self._wsets = [self._wsets[int(i)] for i in index]
            self._cues = [self._cues[int(i)] for i in index]
        self._bank()

    def clear(self) -> None:
        """Empty every stream's records."""
        self.stores = []
        self.stamps = np.zeros((0, self.capacity))
        self._members, self._postings = [], []
        self.counts = dict.fromkeys(self.counts, 0)
        self._bank()

    def branch(self) -> LongTermRecords:
        """A private copy for an imagined branch: it reads the live records, credits none and
        writes none."""
        out = copy(self)
        out.since, out.read_slots = self.since.copy(), self.read_slots.copy()
        out.window, out.key, out.heard = self.window.copy(), self.key.copy(), self.heard.copy()
        out._wsets = [set(held) for held in self._wsets]
        out._cues = [held.copy() for held in self._cues]
        out._entries, out._codes = dict(self._entries), list(self._codes)
        out.counts = dict(self.counts)
        out._cache = {}
        out._private = True
        return out

    def streams(self) -> dict[str, np.ndarray]:
        """What the streams hold, the records apart (with turns, their windows and keys)."""
        out: dict[str, np.ndarray] = {"since": self.since, "read_slots": self.read_slots}
        if self.turn is not None:
            out.update(window=self.window, key=self.key)
        return out

    # a record's arrays a checkpoint keeps, the first ``count`` rows of each, and the address's
    SAVED_ROWS = ("keys", "values", "deep", "recent", "faded")
    SAVED_ADDRESS_ROWS = ("units", "places")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "capacity": self.capacity, "heads": self.heads, "amplitude": self.amplitude,
            "match": self.match, "revise": self.revise, "seed": self.seed,
            "records": [s.count for s in self.stores], **self.counts,
        }
        if self.turn is not None:
            out.update(turn=self.turn, rare=self.rare)
        if self.distinct is not None:
            out["distinct"] = self.distinct
        if self.best:
            out["best"] = True
        if self.rest:
            out["rest"] = True
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        """What a checkpoint keeps (``Brain.save``): the streams' turns and every stream's
        records, the first ``count`` rows of each array (``store/<s>/<name>``), the store's
        counts (``store/<s>/state``) and, with an address, its lists."""
        out = dict(self.streams())
        if self.turn is not None:
            out.update(heard=self.heard.copy(), moments=np.array(self.moments))
        if self.distinct is not None:
            # the records' entries as (stream, slot, entry) rows, the windows' and cues' as
            # (stream, entry) rows
            held = [(row, slot, e) for row, members in enumerate(self._members[: len(self.stores)])
                    for slot, entries in sorted(members.items()) for e in entries.tolist()]
            window = [(row, e) for row, entries in enumerate(self._wsets) for e in sorted(entries)]
            cue = [(row, e) for row, entries in enumerate(self._cues) for e in entries.tolist()]
            out.update(stamps=self.stamps[: len(self.stores)].copy(),
                       codebook=self.codebook.astype(np.uint8),
                       held=np.array(held, dtype=np.int64).reshape(-1, 3),
                       window_entries=np.array(window, dtype=np.int64).reshape(-1, 2),
                       cue_entries=np.array(cue, dtype=np.int64).reshape(-1, 2))
        for s, store in enumerate(self.stores):
            held = store.arrays()
            rows = (*self.SAVED_ROWS, *(self.SAVED_ADDRESS_ROWS if store.coarse else ()))
            for name in rows:
                out[f"store/{s}/{name}"] = np.array(held[name][: store.count])
            for name in ("state", *(("lists", "counts") if store.coarse else ())):
                out[f"store/{s}/{name}"] = held[name]
        return out

    def take(self, arrays: Mapping[str, np.ndarray], stores: int,
             counts: Mapping[str, int] | None = None) -> None:
        """Take back what ``arrays`` kept of ``stores`` streams' records and of the streams'
        turns, into a memory of the same settings (``Brain.load``). Refused where a shape
        disagrees with the settings."""
        missing = [f"store/{s}/{name}" for s in range(int(stores)) for name in
                   (*self.SAVED_ROWS, "state") if f"store/{s}/{name}" not in arrays]
        missing += [name for name in ("since", "read_slots") if name not in arrays]
        if missing:
            raise ValueError(f"missing saved long-term records: {missing[0]}")
        self.stores = []
        for s in range(int(stores)):
            store = self._store()
            full = store.arrays()
            state = np.asarray(arrays[f"store/{s}/state"])
            if state.shape != (3,) or not 0 <= int(state[0]) <= self.capacity:
                raise ValueError("invalid saved long-term records: counts")
            count = int(state[0])
            rows = (*self.SAVED_ROWS, *(self.SAVED_ADDRESS_ROWS if store.coarse else ()))
            for name in rows:
                given = np.asarray(arrays[f"store/{s}/{name}"])
                if given.shape != (count, *full[name].shape[1:]):
                    raise ValueError(f"invalid saved long-term records: {name}")
                full[name][:count] = given
            for name in ("lists", "counts") if store.coarse else ():
                full[name] = np.asarray(arrays[f"store/{s}/{name}"])
            full["state"] = state
            store.restore(full)
            self.stores.append(store)
        since = np.asarray(arrays["since"]).astype(np.int64)
        slots = np.asarray(arrays["read_slots"]).astype(np.int64)
        if since.ndim != 1 or slots.shape != (len(since), self.heads):
            raise ValueError("invalid saved long-term records: the streams' turns")
        self.since, self.read_slots = since, slots
        if self.turn is not None:
            names = ("window", "key", "heard", "moments")
            if any(name not in arrays for name in names):
                raise ValueError("missing saved long-term records: the turns' windows")
            window, key = np.asarray(arrays["window"]), np.asarray(arrays["key"])
            heard = np.asarray(arrays["heard"])
            if window.shape != (len(since), self.width) or key.shape != window.shape \
                    or heard.shape != (self.width,):
                raise ValueError("invalid saved long-term records: the turns' windows")
            self.window, self.key = window.astype(float), key.astype(float)
            self.heard, self.moments = heard.astype(float), float(np.asarray(arrays["moments"]))
        self.stamps = np.zeros((len(self.stores), self.capacity))
        self._members = [{} for _ in self.stores]
        self._postings = [{} for _ in self.stores]
        self._entries, self._codes = {}, []
        self._wsets = [set() for _ in since]
        self._cues = [np.zeros(0, dtype=np.int64) for _ in since]
        if self.distinct is not None:
            saved = ("stamps", "codebook", "held", "window_entries", "cue_entries")
            if any(name not in arrays for name in saved):
                raise ValueError("missing saved long-term records: the records' entries")
            codebook = np.asarray(arrays["codebook"])
            entries = len(codebook)
            stamps, held = np.asarray(arrays["stamps"]), np.asarray(arrays["held"])
            window = np.asarray(arrays["window_entries"])
            cue = np.asarray(arrays["cue_entries"])
            if (codebook.shape != (entries, self.width) or stamps.shape != self.stamps.shape
                    or held.ndim != 2 or held.shape[1] != 3 or window.ndim != 2
                    or window.shape[1] != 2 or cue.ndim != 2 or cue.shape[1] != 2
                    or (len(held) and (held.min() < 0 or held[:, 0].max() >= len(self.stores)
                                       or held[:, 1].max() >= self.capacity
                                       or held[:, 2].max() >= entries))
                    or any(len(a) and (a.min() < 0 or a[:, 0].max() >= len(since)
                                       or a[:, 1].max() >= entries) for a in (window, cue))):
                raise ValueError("invalid saved long-term records: the records' entries")
            self.stamps = stamps.astype(float)
            for code in codebook:
                self.entry(code)
            by_slot: dict[tuple[int, int], list[int]] = {}
            for row, slot, e in held.tolist():
                by_slot.setdefault((row, slot), []).append(e)
            for (row, slot), got in by_slot.items():
                self._hold(row, slot, np.unique(np.asarray(got, dtype=np.int64)))
            for row, e in window.tolist():
                self._wsets[row].add(e)
            for row, e in cue.tolist():
                self._cues[row] = np.union1d(self._cues[row], [e]).astype(np.int64)
        if counts is not None:
            self.counts = {name: int(counts.get(name, 0)) for name in self.counts}
        self._bank()

