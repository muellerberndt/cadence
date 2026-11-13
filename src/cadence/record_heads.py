"""The learned addresses of a brain's long-term heads (``Brain.compose(longterm_attention=rate)``).

A transformer answers from several places of its context at once: each attention head reads a
position of its own. A brain keeps what it heard in records outside its settled state
(``longterm=records``). With ``longterm_attention`` each of its ``longterm_heads`` heads reads a
record of its own in one answer, and where each head looks is learned by the brain's lessons.

**What a head attends.** A turn's question is the moments heard since the last value, in order.
Head ``h`` holds a gate per place in the question, a gate for what the head before it read (the
chain) and a bar,

    g_h = sigmoid(k M_h + m_h)

``k`` the question's key (the union of its moments' codes) and ``(M_h, m_h)`` the head's map.
A place's gate reads two terms of the map, one by the place counted from the question's first
moment and one by the place counted from its last, so a head can hold to the third word or to
the word before the last in a question of any length. The map is zeros at birth, so every gate
starts at one half; the bar starts at ``BAR``. Places past ``POSITIONS`` share the last term.

**What an entry weighs.** Each of the question's moments is an entry of the records' codebook.
An entry some record holds weighs ``1 - log(records holding it + 1) / log(records + 1)``: one
for an entry a single record holds, zero for an entry every record holds. An entry no record
holds weighs the same law among the questions heard: zero for a word every question holds, one
for a word heard for the first time. An entry at weight ``TELLING`` or above tells records
apart; the others are the question's context.

**The read.** A record is found by its key, the entries heard before its turn. The heads read
in turn, and two things are read once: a record, and a telling entry of the question, which
leaves the cue of every later head once the key of a record read holds it. Head ``h``'s cue is
the gated weight of the entries left to it, with the entry of the value the head before it read
as one more place. A record's level is the cue its key holds over the cue of the record the
head looks for: its most attended telling entry and every context entry. The head takes the
record of the highest level among the records left, the latest telling among equals, and holds
it where its level reaches the head's bar; a head whose record falls short holds nothing. The
records scored are those whose key holds a telling entry of the question or of the chain, so a
read's cost follows the records about the question's subjects and not the records the brain
holds.

**What the regions hold.** Head ``h``'s record lies on region ``h``. Where a head's record holds
the value the head before it read, the two are links of a chain, and a chain lies on its regions
from its end: the record that answers on the first of them, as ``longterm_rest`` lays its chain.

**The lesson.** After a lesson each head's choice is scored against the movement of the
neurons its regions reach between the nudged phases, read back through the regions' synapses.
For each record a head could have taken, and for taking none, the heads after it read again as
they would have, and the records the regions would then hold are scored. The head's map steps
toward the choices that score above its own, by the gradient of the levels its gates give them,
and its bar steps toward or away from holding nothing: the recall memory's address lesson over
the heads of the records.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

import numpy as np


class _Read(NamedTuple):
    """A head's read, kept for its lesson."""

    slots: np.ndarray  # the records scored
    holds: np.ndarray  # what each record's key holds of the head's places, (records, places)
    slope: np.ndarray  # each place's cue by its gate's logit
    ideal: np.ndarray  # the places of the record the head looks for: 1 or 0
    whole: float  # that record's cue
    levels: np.ndarray  # each record's level
    bar: float
    terms: list[np.ndarray]  # the terms of the map each place's gate reads


class _Turn:
    """One stream's question at a turn, and what its read found."""

    __slots__ = ("row", "key", "asked", "weight", "terms", "logits", "rows", "reads", "picks",
                 "held", "work")

    def __init__(self, row: int, key: np.ndarray, asked: np.ndarray, weight: np.ndarray,
                 terms: list[np.ndarray], logits: np.ndarray) -> None:
        self.row, self.key, self.asked, self.weight = row, key, asked, weight
        self.terms, self.logits = terms, logits
        self.rows: dict[int, np.ndarray] = {}  # a record's key against the question's places
        self.reads: list[_Read | None] = []
        self.picks: list[int] = []
        self.held: list[int] = []
        self.work = 0.0


class RecordHeads:
    """The heads of a ``LongTermRecords`` with learned addresses (the module's docstring)."""

    POSITIONS: ClassVar[int] = 24  # places from each end of a question with a term of their own
    TELLING: ClassVar[float] = 0.5  # the weight at which an entry tells records apart
    BAR: ClassVar[float] = -3.0  # a newborn head's bar, as its logit: a level of .05
    ALTERNATIVES: ClassVar[int] = 8  # the records a head's lesson weighs beside its own
    TEMPERATURE: ClassVar[float] = 0.1  # of a head's levels in its lesson
    SCORE_TEMPERATURE: ClassVar[float] = 0.1  # of the movement's scores, the recall memory's

    def __init__(self, records: Any, connectome: Any, rate: float) -> None:
        if isinstance(rate, (bool, np.bool_)) or not np.isfinite(rate) or rate < 0:
            raise ValueError("longterm_attention is the rate of the heads' lesson: a finite "
                             "nonnegative number (0.0 holds the maps as they are)")
        self.records = records
        self.rate = float(rate)
        self.heads, self.width = int(records.heads), int(records.width)
        # a map's terms: the places from the first moment, the places from the last, the chain
        # and the bar
        self.chain, self.bar = 2 * self.POSITIONS, 2 * self.POSITIONS + 1
        self.maps: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(self.heads):
            bias = np.zeros(self.bar + 1)
            bias[self.bar] = self.BAR
            self.maps.append((np.zeros((self.width, self.bar + 1)), bias))
        populations = connectome.populations
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        # each region's synapses: their source in the region and the neuron they reach (the
        # association's, and with ``longterm_motor`` the motor's)
        self._ports = []
        for name in records.ports(self.heads):
            port = np.asarray(populations[name], dtype=np.int64)
            source = -np.ones(connectome.n, dtype=np.int64)
            source[port] = np.arange(len(port))
            synapses = np.flatnonzero(source[pre] >= 0)
            self._ports.append((synapses, source[pre[synapses]], post[synapses]))
        self.last: dict[int, _Turn] = {}  # each stream's latest read, for its lesson

    @property
    def parameters(self) -> int:
        """The numbers the heads' maps hold."""
        return int(sum(m.size + b.size for m, b in self.maps))

    # -- the read

    def _key_row(self, turn: _Turn, slot: int) -> np.ndarray:
        """Which of the question's places record ``slot``'s key holds."""
        found = turn.rows.get(slot)
        if found is None:
            found = np.isin(turn.asked, self._key_of(turn.row, slot))
            turn.rows[slot] = found
            turn.work += len(turn.asked)
        return found

    def _key_of(self, row: int, slot: int) -> np.ndarray:
        """The entries of record ``slot``'s key: what was heard before its turn."""
        records = self.records
        members = records._members[row][slot]
        return np.asarray(members[members != records.said(row, slot)])

    def _weights(self, row: int, entries: np.ndarray) -> np.ndarray:
        """What each of ``entries`` weighs: its rarity among stream ``row``'s records, or for an
        entry no record holds its rarity among the questions heard."""
        records = self.records
        postings = records._postings[row]
        count = records.stores[row].count
        asked = records.asked_in
        out = np.zeros(len(entries))
        for i, e in enumerate(entries.tolist()):
            held = len(postings.get(e, ()))
            if held:
                out[i] = 1.0 - np.log(held + 1.0) / np.log(count + 1.0)
            else:
                out[i] = 1.0 - np.log(asked.get(e, 0) + 1.0) / np.log(records.questions + 2.0)
        return np.asarray(np.maximum(out, 0.0))

    def _begin(self, row: int) -> _Turn | None:
        """Stream ``row``'s question at its latest turn; None where nothing is asked or held."""
        records = self.records
        asked = records._asked[row] if row < len(records._asked) else np.zeros(0, np.int64)
        if not records.stores[row].count or not len(asked):
            return None
        key = np.asarray(records.key[row], dtype=np.float64)
        logits = np.stack([key @ m + b for m, b in self.maps])
        places = np.arange(len(asked))
        terms = [np.minimum(places, self.POSITIONS - 1),
                 self.POSITIONS + np.minimum(places[::-1], self.POSITIONS - 1)]
        return _Turn(row, key, asked, self._weights(row, asked), terms, logits)

    def _candidates(self, turn: _Turn, entries: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """The records that hold a telling entry among ``entries``."""
        postings = self.records._postings[turn.row]
        told = [postings.get(int(e), ()) for e in np.unique(entries[weight >= self.TELLING])]
        turn.work += float(sum(len(t) for t in told))
        if not told:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate([np.asarray(t, dtype=np.int64) for t in told]))

    def _head(self, turn: _Turn, head: int, picks: list[int]) -> tuple[int, _Read | None]:
        """Head ``head``'s read after the heads before it read ``picks`` (-1: none): its record
        (-1: none), and its read for its lesson."""
        records = self.records
        row = turn.row
        taken = [p for p in picks if p >= 0]
        entries, weight = turn.asked, turn.weight
        terms = list(turn.terms)
        logits = turn.logits[head, terms[0]] + turn.logits[head, terms[1]]
        if taken:  # a telling entry of the question is read once
            read = np.any([self._key_row(turn, p) for p in taken], axis=0)
            weight = np.where(read & (weight >= self.TELLING), 0.0, weight)
        chained = records.said(row, picks[-1]) if picks and picks[-1] >= 0 else -1
        if chained >= 0:
            entries = np.append(entries, chained)
            weight = np.append(weight, self._weights(row, np.array([chained])))
            logits = np.append(logits, turn.logits[head, self.chain])
            terms = [np.append(terms[0], self.chain), np.append(terms[1], -1)]
        slots = self._candidates(turn, entries, np.append(turn.weight, weight[len(turn.asked):]))
        if taken:
            slots = slots[~np.isin(slots, np.asarray(taken, dtype=np.int64))]
        gates = 1.0 / (1.0 + np.exp(-logits))
        cue = gates * weight
        telling = weight >= self.TELLING
        # the record the head looks for: its most attended telling entry and every context one
        ideal = (~telling).astype(np.float64)
        if telling.any():
            ideal[int(np.argmax(np.where(telling, cue, -1.0)))] = 1.0
        whole = float(cue @ ideal)
        if not len(slots) or whole <= 1e-12:
            return -1, None
        holds = np.stack([self._key_row(turn, int(s)) for s in slots]).astype(np.float64)
        if chained >= 0:
            link_held = np.array([chained in self._key_of(row, int(s)) for s in slots],
                                 dtype=np.float64)
            holds = np.concatenate([holds, link_held[:, None]], axis=1)
        levels = (holds @ cue) / whole
        turn.work += float(holds.size)
        bar = float(1.0 / (1.0 + np.exp(-turn.logits[head, self.bar])))
        best = float(levels.max())
        pick = -1
        if best >= bar and best > 1e-12:
            top = np.flatnonzero(levels >= best - 1e-9)
            pick = int(slots[top[int(np.argmax(records.stamps[row, slots[top]]))]])
        return pick, _Read(slots, holds, cue * (1.0 - gates), ideal, whole, levels, bar, terms)

    def _roll(self, turn: _Turn, picks: list[int], keep: bool = False) -> list[int]:
        """The heads from ``len(picks)`` on read in turn after ``picks``; with ``keep`` each
        head's read is kept for the lesson."""
        picks = list(picks)
        for head in range(len(picks), self.heads):
            pick, read = self._head(turn, head, picks)
            picks.append(pick)
            if keep:
                turn.reads.append(read)
        return picks

    def _laid(self, turn: _Turn, picks: list[int]) -> list[int]:
        """The record each region holds: head ``h``'s on region ``h``, and a chain (each record
        holding the value the head before it read) on its regions from its end."""
        records, row = self.records, turn.row
        out = list(picks)
        start = 0
        for head in range(1, self.heads + 1):
            linked = (
                head < self.heads and picks[head] >= 0 and picks[head - 1] >= 0
                and records.said(row, picks[head - 1]) in self._key_of(row, picks[head])
            )
            if not linked:
                out[start:head] = picks[start:head][::-1]
                start = head
        return out

    def read(self, row: int) -> list[int]:
        """Stream ``row``'s read at its turn: the record each region holds (-1: none)."""
        turn = self._begin(row)
        if turn is None:
            self.last.pop(row, None)
            return [-1] * self.heads
        turn.picks = self._roll(turn, [], keep=True)
        turn.held = self._laid(turn, turn.picks)
        self.last[row] = turn
        self.records.work = turn.work
        return turn.held

    # -- the lesson

    def _back(self, graph: Any, movement: np.ndarray) -> np.ndarray:
        """The ``movement`` of every neuron between the nudged phases, one row per stream, read
        back through each region's synapses: ``(regions, rows, value width)``, each row at
        unit length over its regions together."""
        width = int(self.records.value_width)
        out = np.zeros((self.heads, len(movement), width))
        weights = np.asarray(graph._weights)
        for port, (synapses, sources, targets) in enumerate(self._ports):
            reach = weights[synapses]
            for at, moved in enumerate(movement):
                out[port, at] = np.bincount(sources, weights=moved[targets] * reach,
                                            minlength=width)
        norm = np.sqrt(np.sum(out * out, axis=(0, 2)))[None, :, None]
        np.divide(out, norm, out=out, where=norm > 1e-12)
        return out

    def _named(self, turn: _Turn, picks: list[int], back: np.ndarray) -> float:
        """How far the regions holding ``picks``'s records lie along the movement."""
        values = self.records.stores[turn.row].values
        return float(sum(
            np.asarray(values[slot], dtype=np.float64) @ back[port]
            for port, slot in enumerate(self._laid(turn, picks)) if slot >= 0
        ))

    def learn(self, graph: Any, learned: Any) -> dict[str, float]:
        """The heads' lesson after a lesson whose phases are ``learned``, at its taught rows.
        Returns how many heads' maps moved (``heads_moved``) and how many a non-finite step
        left as they were (``heads_refused``)."""
        quiet = {"heads_moved": 0.0, "heads_refused": 0.0}
        if (
            not self.rate or not self.last or learned.free is None or learned.nudged is None
            or learned.nudged is learned.free or getattr(graph, "_weights", None) is None
        ):
            return quiet
        plus = np.atleast_2d(learned.nudged.activation)
        minus = np.atleast_2d(
            learned.free.activation if learned.opposite is None else learned.opposite.activation
        )
        taught = learned.taught
        rows = [
            row for row in sorted(self.last)
            if row < len(plus) and (taught is None or bool(np.asarray(taught)[row]))
        ]
        if not rows:
            return quiet
        back = self._back(graph, (plus - minus)[rows])
        moved = refused = 0
        for at, row in enumerate(rows):
            turn = self.last[row]
            bound = min(1.0, 1.0 / (self.rate * (float(turn.key @ turn.key) + 1.0)))
            for head, read in enumerate(turn.reads):
                if read is None:
                    continue
                able = np.flatnonzero(read.levels > 1e-12)
                order = able[np.argsort(-read.levels[able], kind="stable")][: self.ALTERNATIVES]
                own = turn.picks[head]
                choices = [int(read.slots[i]) for i in order]
                if own >= 0 and own not in choices:
                    order = np.append(order, int(np.flatnonzero(read.slots == own)[0]))
                    choices.append(own)
                if not choices:
                    continue
                levels = np.append(read.levels[order], read.bar)  # the last choice: no record
                named = np.array([
                    self._named(turn, self._roll(turn, [*turn.picks[:head], choice]),
                                back[:, at])
                    for choice in (*choices, -1)
                ])
                with np.errstate(over="ignore", invalid="ignore"):
                    p = np.exp((levels - levels.max()) / self.TEMPERATURE)
                    p /= p.sum()
                    q = np.log(np.maximum(p, 1e-300)) + named / self.SCORE_TEMPERATURE
                    q = np.exp(q - q.max())
                    q /= q.sum()
                    lean = (q - p) / self.TEMPERATURE
                    # a level's slope by a gate's logit: the place's cue, held or not, less the
                    # level where the place is one of the record the head looks for
                    step = read.slope * (lean[:-1] @ (
                        read.holds[order] - read.levels[order][:, None] * read.ideal[None, :]
                    )) / read.whole
                    raise_bar = float(lean[-1]) * read.bar * (1.0 - read.bar)
                if not (np.isfinite(step).all() and np.isfinite(raise_bar)):
                    refused += 1
                    continue
                gates = np.zeros(self.bar + 1)
                for terms in read.terms:
                    at_term = terms >= 0
                    np.add.at(gates, terms[at_term], step[at_term])
                gates[self.bar] = raise_bar
                if not gates.any():
                    continue
                m, b = self.maps[head]
                m += (self.rate * bound) * np.outer(turn.key, gates)
                b += (self.rate * bound) * gates
                moved += 1
        return {"heads_moved": float(moved), "heads_refused": float(refused)}

    # -- what a checkpoint keeps

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for head, (m, b) in enumerate(self.maps):
            out[f"address/{head}/x"] = m.copy()
            out[f"address/{head}/b"] = b.copy()
        return out

    def take(self, arrays: Mapping[str, np.ndarray]) -> None:
        maps = []
        for head, (m, b) in enumerate(self.maps):
            names = (f"address/{head}/x", f"address/{head}/b")
            if any(name not in arrays for name in names):
                raise ValueError("missing saved long-term records: the heads' maps")
            x, bias = (np.asarray(arrays[name], dtype=np.float64) for name in names)
            if x.shape != m.shape or bias.shape != b.shape or not (
                    np.isfinite(x).all() and np.isfinite(bias).all()):
                raise ValueError("invalid saved long-term records: the heads' maps")
            maps.append((x.copy(), bias.copy()))
        self.maps = maps
        self.last = {}


__all__ = ["RecordHeads"]
