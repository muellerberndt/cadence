"""Corrections heard beside earlier utterances at a weight.

A correction teaches a brain one utterance it got wrong: the utterance's rows (one observation
per step) with a label per step, for a recogniser the letter targets of a forced alignment, or a
distribution over the actions per step (``targets``) for a brain taught soft targets. Taught
alone, a correction moves the brain toward the corrected speaker and away from the speakers it
knew before. ``Rehearsal`` hears the correction as one stream beside ``rows`` earlier utterances,
each its own stream, and teaches those at lesson weight ``weight`` (``Brain.lesson(weights=)``):
the earlier utterances pull back at a quarter of the correction's strength.

On the spoken-phrase chamber of issue 303 (six fresh founders, 18 new speakers and 18 dialect
speakers, two minutes of corrections, ``benchmarks/spoken``), three earlier phrases at weight .25
kept the earlier speakers (a change of +.005 and +.003 in their word accuracy, where a correction
alone lost .018 and .056) at nearly a plain correction's gain (+.075 against +.080 and +.086
against +.096). At weight zero the earlier rows teach nothing there (at the normalized teaching
points a composed brain takes, the lesson's step is the correction's alone; under the founder
teaching zero rows still shrink the step) and the loss halves; at weight one the earlier speakers
rise and the gain falls to +.062.

``Rehearsal`` keeps the earlier utterances in a bank under a byte budget. ``remember`` adds an
utterance; when the bank would pass its budget, utterances chosen at random (seeded) leave it.
``correct`` resets the brain, hears the correction and the chosen earlier utterances side by side
from rest, each step a lesson before a greedy act, and leaves the brain at rest. Earlier
utterances of the correction's own length are chosen first, then the nearest in length; each
stream past its own last row holds that row and is taught at weight zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Rehearsal"]


@dataclass
class Rehearsal:
    """A bank of earlier utterances and corrections heard beside them.

    ``rows`` earlier utterances (3) beside each correction, taught at ``weight`` (.25), kept
    within ``budget`` bytes (64 MiB); ``seed`` draws which utterances a correction hears and which
    leave a full bank."""

    rows: int = 3
    weight: float = 0.25
    budget: int = 64 * 2**20
    seed: int = 0
    _bank: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.rows, bool) or not isinstance(self.rows, (int, np.integer)):
            raise ValueError("rows must be a nonnegative integer")
        if self.rows < 0:
            raise ValueError("rows must be a nonnegative integer")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float, np.integer, np.floating))
            or not np.isfinite(self.weight)
            or self.weight < 0
        ):
            raise ValueError("weight must be a finite nonnegative number")
        if isinstance(self.budget, bool) or not isinstance(self.budget, (int, np.integer)):
            raise ValueError("budget must be a positive number of bytes")
        if self.budget <= 0:
            raise ValueError("budget must be a positive number of bytes")
        self.rows, self.weight, self.budget = int(self.rows), float(self.weight), int(self.budget)
        self._rng = np.random.default_rng(np.random.SeedSequence([30303, int(self.seed)]))

    # -- the bank

    def __len__(self) -> int:
        return len(self._bank)

    @property
    def nbytes(self) -> int:
        """Bytes the bank's utterances hold."""
        return sum(x.nbytes + y.nbytes for x, y in self._bank)

    def remember(self, rows: Any, labels: Any = None, *, targets: Any = None) -> None:
        """Keep one earlier utterance: its rows ``(steps, inputs)`` and a label per step, or
        ``targets``, a distribution over the actions per step ``(steps, actions)``."""
        x, y = _utterance(rows, labels, targets)
        size = x.nbytes + y.nbytes
        if size > self.budget:
            raise ValueError("one utterance holds more bytes than the bank's budget")
        while self._bank and self.nbytes + size > self.budget:
            self._bank.pop(int(self._rng.integers(len(self._bank))))
        self._bank.append((x, y))

    def chosen(self, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """The earlier utterances a correction of ``steps`` steps hears: those of its length
        first, then the nearest, each group in a seeded order."""
        if not self.rows or not self._bank:
            return []
        gap = np.array([abs(len(x) - steps) for x, _ in self._bank], dtype=np.float64)
        order = np.lexsort((self._rng.random(len(gap)), gap))
        return [self._bank[int(i)] for i in order[: self.rows]]

    # -- the correction

    def correct(
        self, brain: Any, rows: Any, labels: Any = None, *, targets: Any = None
    ) -> dict[str, Any]:
        """Teach ``brain`` one utterance, its rows ``(steps, inputs)`` and a label per step (or
        ``targets``, a distribution over the actions per step), beside earlier utterances of the
        bank at the rehearsal weight. Where the correction or a banked utterance holds targets,
        every stream is taught targets (a label as its one-hot row). The brain is reset before
        and left at rest after. Returns the streams heard, the steps, the lessons a learning phase
        refused and the actions the brain took on the correction."""
        from ..learning import LearningPhaseError

        x, y = _utterance(rows, labels, targets)
        streams = [(x, y), *self.chosen(len(x))]
        width = x.shape[1]
        if any(r.shape[1] != width for r, _ in streams):
            raise ValueError("earlier utterances must hold rows as wide as the correction")
        soft = [t for _, t in streams if t.ndim == 2]
        actions_count = soft[0].shape[1] if soft else 0
        if any(t.shape[1] != actions_count for t in soft):
            raise ValueError("targets of the correction and the bank must name as many actions")
        longest = max(len(r) for r, _ in streams)
        batch = np.zeros((longest, len(streams), width), dtype=np.float64)
        taught: np.ndarray = (
            np.zeros((longest, len(streams), actions_count), dtype=np.float64)
            if soft
            else np.zeros((longest, len(streams)), dtype=np.int64)
        )
        weights = np.zeros((longest, len(streams)), dtype=np.float64)
        for s, (r, labs) in enumerate(streams):
            if soft and labs.ndim == 1:
                if (labs >= actions_count).any():
                    raise ValueError("a label names an action the targets do not hold")
                labs = np.eye(actions_count)[labs]
            batch[: len(r), s] = r
            batch[len(r) :, s] = r[-1]
            taught[: len(r), s] = labs
            taught[len(r) :, s] = labs[-1]
            weights[: len(r), s] = 1.0 if s == 0 else self.weight
        actions = np.zeros(len(x), dtype=np.int64)
        refused = 0
        brain.reset()
        for t in range(longest):
            if weights[t].any():
                try:
                    if soft:
                        brain.lesson(batch[t], targets=taught[t], weights=weights[t])
                    else:
                        brain.lesson(batch[t], taught[t], weights=weights[t])
                except LearningPhaseError:
                    refused += 1
            acted = np.asarray(brain.act(batch[t], greedy=True)).reshape(-1)
            if t < len(x):
                actions[t] = int(acted[0])
        brain.reset()
        return {"streams": len(streams), "steps": longest, "refused": refused, "actions": actions}


def _utterance(rows: Any, labels: Any, targets: Any = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError(
            "an utterance's rows are a finite (steps, inputs) array of one step or more"
        )
    if (labels is None) == (targets is None):
        raise ValueError("an utterance holds labels or targets, one of them")
    if targets is not None:
        t = np.asarray(targets, dtype=np.float32)
        if (
            t.ndim != 2
            or len(t) != len(x)
            or not np.isfinite(t).all()
            or (t < 0).any()
            or not np.allclose(t.sum(axis=1), 1.0, atol=1e-4)
        ):
            raise ValueError("targets hold a distribution over the actions per step")
        return x, t
    y = np.asarray(labels)
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError("an utterance holds one nonnegative integer label per step")
    return x, y.astype(np.int64)
