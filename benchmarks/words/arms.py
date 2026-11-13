"""The lexicon's place taken by a lookup (cadence#326, W7): how much of a reading is the settling
lexicon's, and how much the program around it?

Each arm keeps the brain's program (``cadence.words.Words``: the map, the body, the reads' path,
the motor's letters, a word heard back before it is said) and puts a table in the lexicon's
place, with the lexicon's calls. A row a word, one value a concept: as many numbers as the
lexicon's reciprocal pairs. Hearing sums the heard words' rows; saying reads every word's row at
the concept. Both are controls by the lane's rules (a lookup, a feed-forward readout).

- ``Table``: a lesson moves each heard word's row by its share toward the target (the running
  mean of its lessons, 1 / n as the lexicon's);
- ``Delta``: a lesson moves each heard word's row by its share of one error, the target less the
  heard words' summed rows (a delta rule).

Under the game a mark holds a word's target and its strength (fading by the lexicon's ``keep``),
and a pistachio applies the marks, as the lexicon's three factors do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadence.words import Lexicon


class Table(Lexicon):
    """A count table in the lexicon's place."""

    delta = False

    def __init__(self, lexicon: Lexicon):
        told = dict(lexicon.settings)
        super().__init__(told.pop("concepts"), **told)
        self.rows = np.zeros((self.capacity, self.concepts))
        self.pending: dict[tuple[str, ...], dict[int, tuple[np.ndarray, float]]] = {}

    def _macs(self) -> None:
        self.work["settles"] += 1
        self.work["macs"] += int(self.rows.size)

    def _heard_units(self, words: Any, enter: bool) -> list[int]:
        rows = [words] if isinstance(words, str) else list(words)
        out = []
        for word in rows:
            unit = self.enter(word) if enter else self.unit(word)
            if unit is not None:
                out.append(unit)
        return sorted(set(out))

    def hear(self, words: Any) -> np.ndarray:
        self._macs()
        units = self._heard_units(words, enter=False)
        return np.asarray(self.rows[units].sum(axis=0)) if units else np.zeros(self.concepts)

    def words_for(self, concept: Any) -> np.ndarray:
        self._macs()
        vector = self._vector(concept)
        return np.asarray(self.rows[: len(self.forms)] @ vector)

    def _move(self, units: list[int], target: np.ndarray, outputs: np.ndarray,
              shares: dict[int, float]) -> None:
        if self.delta:
            error = target[outputs] - self.rows[np.ix_(units, outputs)].sum(axis=0)
            for u in units:
                self.rows[u, outputs] += shares[u] * error / len(units)
        else:
            for u in units:
                self.rows[u, outputs] += shares[u] * (target[outputs] - self.rows[u, outputs])

    def _outputs(self, groups: tuple[str, ...]) -> np.ndarray:
        return np.concatenate([np.arange(self.groups[g].start, self.groups[g].stop)
                               for g in groups])

    def name(self, words: Any, concepts: Any, *, groups: Any = None,
             weight: float = 1.0) -> float:
        groups = tuple(self.groups) if groups is None else tuple(groups)
        units = self._heard_units(words, enter=True)
        counts = self.counts.setdefault(groups, np.zeros(self.capacity))
        weight = min(max(float(weight), 0.0), 1.0)
        shares = {u: weight * max(1.0 / (counts[u] + 1), self.rate) for u in units}
        if not units or max(shares.values()) <= 0:
            return float("nan")
        target = self.level * self._concepts(concepts)
        self._move(units, target, self._outputs(groups), shares)
        for u in units:
            counts[u] += 1
        self.lessons += 1
        self._macs()
        return 0.0

    def mark(self, words: Any, concepts: Any, *, groups: Any = None) -> None:
        groups = tuple(self.groups) if groups is None else tuple(groups)
        units = self._heard_units(words, enter=True)
        target = self.level * self._concepts(concepts)
        pending = self.pending.setdefault(groups, {})
        for u in units:
            pending[u] = (target, 1.0)
        if units:
            self.heard_last = units[-1]
        self.marks = {g: np.zeros(1) for g in self.pending}  # the base class's marks: in use

    def confirm(self, amount: float = 1.0) -> int:
        made = 0
        for groups, pending in self.pending.items():
            counts = self.counts.setdefault(groups, np.zeros(self.capacity))
            outputs = self._outputs(groups)
            for u, (target, strength) in pending.items():
                share = float(amount) * strength * max(1.0 / (counts[u] + 1), self.rate)
                self._move([u], target, outputs, {u: share})
                counts[u] += float(amount) * strength
                made += 1
        if made:
            self.lessons += 1
        self.pending.clear()
        self.marks.clear()
        return made

    def fade(self) -> None:
        for groups in list(self.pending):
            pending = self.pending[groups]
            for u in list(pending):
                target, strength = pending[u]
                if strength * self.keep < 1e-3:
                    del pending[u]
                else:
                    pending[u] = (target, strength * self.keep)
            if not pending:
                del self.pending[groups]
        self.marks = {g: np.zeros(1) for g in self.pending}


class Delta(Table):
    """A delta rule in the lexicon's place."""

    delta = True


ARMS = {"table": Table, "delta": Delta}
