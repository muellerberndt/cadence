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

and the read is ``sum_i p_i V_i``, as ``Recall`` reads its slots. The founder ``temperature=0``
reads the first choice whole. Several heads read in turn under the recall memory's law: a slot
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
    ``temperature`` is the fine read's (0 reads the first choice whole). ``candidates``,
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
    count: int = field(init=False, default=0)
    writes: int = field(init=False, default=0)

    # the fading's constants: the order of the depth's law and the recent use below which a slot
    # counts as idle
    ORDER = 4.0
    FLOOR = 0.01
    # the settings a description holds, and the arrays a memory keeps
    SETTINGS = ("width", "value_width", "capacity", "temperature", "candidates", "expansion",
                "winners", "depth", "idle", "hold", "revise", "fade", "seed", "direct", "dtype")
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
        weights = self._weights(scores, int(heads))
        if credit:
            for w in weights:  # the slots a head attends; the others keep their sums
                at = np.flatnonzero(w)
                held, share = slots[at], w[at]
                self.deep[held] += share
                self.recent[held] += share
                self.faded[held] *= 1.0 - np.minimum(share, 1.0)
        return LongTermRead(slots, weights)

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

    def write(self, key: np.ndarray, value: np.ndarray) -> tuple[int, bool]:
        """Write one record and return its slot and whether the write revised a held record.
        A key the memory recognises (``revise``) rewrites that record; any other takes an empty
        slot, or in a full memory the most faded one."""
        key = self._key(key)
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.value_width,) or not np.isfinite(value).all():
            raise ValueError(f"a value is one finite row of {self.value_width} values")
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

    @property
    def nbytes(self) -> int:
        """The bytes the memory holds: records, sums and, with a coarse stage, its address."""
        held = [self.keys, self.values, self.deep, self.recent, self.faded]
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
    ``heads`` heads read in turn, each its first choice among the records the heads before it
    left, and a head holds its record only where the record's key lies at cosine ``match`` or
    above to the message's key. Head ``h`` holds ``amplitude`` times the record's value on the
    region ``longterm`` (head 0) or ``longterm_h``, which the association reads through
    plastic synapses, as it reads the recall memory's ports. Before a turn the regions are
    silent.

    **What stays.** Each stream keeps its records in a ``LongTermMemory`` of ``capacity``
    slots: a read deepens the records it holds, every moment the brain lives fades the idle
    ones by their depth and recent use, and a full memory replaces the most faded record. The
    records survive ``Brain.reset`` and every episode's end; ``clear()`` empties them.
    """

    # the store's settings that the brain does not name: the library store's founders, and a
    # first choice read whole
    STORE: ClassVar[dict[str, Any]] = {"temperature": 0.0}

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
    ) -> None:
        if message is None or message.boundary is None or not message.boundary_holds:
            raise ValueError(
                "longterm keys its records on the brain's held message from a turn: it needs "
                "message=units, message_boundary=channel and message_boundary_holds=True"
            )
        self.message = message
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
        self.since = np.full(0, -1, dtype=np.int64)  # moments since each stream's turn
        self.read_slots = np.full((0, self.heads), -1, dtype=np.int64)  # the latest read
        self.counts = {"writes": 0, "revisions": 0, "agreed": 0, "reads": 0}
        self._private = False

    @staticmethod
    def ports(heads: int) -> tuple[str, ...]:
        """The regions the heads hold their records on."""
        return ("longterm", *(f"longterm_{h}" for h in range(1, int(heads))))

    def _store(self) -> LongTermMemory:
        return LongTermMemory(self.width, self.value_width, self.capacity, revise=self.revise,
                              seed=self.seed, **self.STORE)

    def _streams(self, batch: int) -> None:
        """Every stream of a batch of ``batch`` has its records; a stream that begins holds no
        turn."""
        while len(self.stores) < batch:
            self.stores.append(self._store())
        if len(self.since) != batch:
            self.since = np.full(batch, -1, dtype=np.int64)
            self.read_slots = np.full((batch, self.heads), -1, dtype=np.int64)

    def _keys(self, batch: int) -> np.ndarray | None:
        """Each stream's key, ``(batch, units)``, where the message holds that batch."""
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
        for row in np.flatnonzero(self.since >= 0):
            store, key = self.stores[int(row)], keys[row]
            if not store.count or not key.any():
                continue
            read = store.read(key, heads=self.heads, credit=not self._private)
            self.counts["reads"] += 0 if self._private else 1
            for head, columns in enumerate(self.columns):
                slot = read.first(head)
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
                    self._write(int(row), keys[row], x[row])
        for store in self.stores[:batch]:
            store.moment()

    def _write(self, row: int, key: np.ndarray, value: np.ndarray) -> None:
        store = self.stores[row]
        if store.count:
            found = store.read(key, credit=False).first()
            if self._holds(store, found, key, self.revise) and np.array_equal(
                np.asarray(store.values[found], dtype=np.float64), value
            ):
                self.counts["agreed"] += 1  # the moment says nothing the record does not
                return
        _, revised = store.write(key, value)
        self.counts["writes"] += 1
        self.counts["revisions"] += int(revised)

    # -- the memory contract

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Streams that begin hold no turn; every record stays."""
        if rows is None or len(self.since) != batch:
            self.since = np.full(batch, -1, dtype=np.int64)
            self.read_slots = np.full((batch, self.heads), -1, dtype=np.int64)
        else:
            self.since[rows] = -1
            self.read_slots[rows] = -1

    def keep(self, rows: np.ndarray) -> None:
        """Keep the streams of ``rows``, their records with them."""
        index = np.arange(len(self.since))[rows]
        self.stores = [self.stores[int(i)] for i in index if int(i) < len(self.stores)]
        self.since, self.read_slots = self.since[index], self.read_slots[index]

    def clear(self) -> None:
        """Empty every stream's records."""
        self.stores = []
        self.counts = dict.fromkeys(self.counts, 0)

    def branch(self) -> LongTermRecords:
        """A private copy for an imagined branch: it reads the live records, credits none and
        writes none."""
        out = copy(self)
        out.since, out.read_slots = self.since.copy(), self.read_slots.copy()
        out.counts = dict(self.counts)
        out._private = True
        return out

    def streams(self) -> dict[str, np.ndarray]:
        """What the streams hold, the records apart."""
        return {"since": self.since, "read_slots": self.read_slots}

    # a record's arrays a checkpoint keeps, the first ``count`` rows of each, and the address's
    SAVED_ROWS = ("keys", "values", "deep", "recent", "faded")
    SAVED_ADDRESS_ROWS = ("units", "places")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity, "heads": self.heads, "amplitude": self.amplitude,
            "match": self.match, "revise": self.revise, "seed": self.seed,
            "records": [s.count for s in self.stores], **self.counts,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """What a checkpoint keeps (``Brain.save``): the streams' turns and every stream's
        records, the first ``count`` rows of each array (``store/<s>/<name>``), the store's
        counts (``store/<s>/state``) and, with an address, its lists."""
        out = dict(self.streams())
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
        if counts is not None:
            self.counts = {name: int(counts.get(name, 0)) for name in self.counts}

