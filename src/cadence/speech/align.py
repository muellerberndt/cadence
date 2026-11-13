"""Training from transcripts: a lesson for every step from the brain's own beliefs.

A transcript says which symbols an utterance holds and not when. The lesson gives each step a
target by the best monotonic path of the transcript through the brain's own per-step beliefs
(``forced_path``, the forced alignment of CTC: blank may fill any step and parts a doubled
letter). The path is computed outside the brain, by the teacher, and taught as labels; the brain
learns by its own rule. Before a brain believes anything, ``even_path`` spreads the symbols
evenly.

``soft_path`` gives each step a distribution in place of one symbol: the share of the
transcript's paths that say each symbol at that step, every path weighed by the brain's beliefs
along it (the occupation shares of CTC's forward and backward sums). Taught as a lesson's
``targets``, the pull on the motor neurons at every step is the descent direction of CTC's loss
on the brain's beliefs, computed, as the path is, outside the brain from its beliefs alone.
``soft_step`` is one moment of that lesson: the pull, then a greedy act.

``Alignments`` holds the paths of utterances to come, one per hearing: an utterance heard in two
passes takes a path in each, and using one never removes the other. ``log_beliefs`` reads a
frozen copy's beliefs for several utterances heard side by side.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from .decode import BLANK


def even_path(steps: int, transcript: Any) -> np.ndarray:
    """Each step's symbol on an even split of ``steps`` among the transcript's symbols."""
    symbols = np.asarray(transcript, dtype=np.int64)
    if steps < 1 or len(symbols) == 0:
        raise ValueError("even_path needs a step and a symbol")
    n = len(symbols)
    return symbols[np.minimum((np.arange(steps) * n) // steps, n - 1)]


def forced_path(log_beliefs: Any, transcript: Any, *, blank: int = BLANK,
                states: bool = False) -> np.ndarray:
    """The symbol taught at each step: the transcript's best monotonic path through
    ``(steps, symbols)`` log beliefs, blank allowed on any step and required between a symbol
    and the same symbol after it (CTC's forced alignment, by Viterbi). With ``states`` the
    path's state at each step instead: ``2k + 1`` on the transcript's k-th symbol, even on a
    blank. A transcript longer than the steps allow is refused."""
    scores = np.asarray(log_beliefs, dtype=np.float64)
    sequence = np.asarray(transcript, dtype=np.int64)
    if scores.ndim != 2 or len(sequence) == 0:
        raise ValueError("forced_path needs (steps, symbols) beliefs and a symbol")
    repeats = int(np.sum(sequence[1:] == sequence[:-1]))
    if len(scores) < len(sequence) + repeats:
        raise ValueError("the transcript needs more steps than the utterance holds")
    frames = len(scores)
    labels = np.full(2 * len(sequence) + 1, blank, dtype=np.int64)
    labels[1::2] = sequence
    n = len(labels)
    skip = np.zeros(n, dtype=bool)  # a symbol reachable from two states back
    skip[3::2] = labels[3::2] != labels[1:-2:2]
    best = np.full((frames, n), -np.inf)
    back = np.zeros((frames, n), dtype=np.int64)
    best[0, 0] = scores[0, labels[0]]
    best[0, 1] = scores[0, labels[1]]
    for t in range(1, frames):
        stay = best[t - 1]
        one = np.concatenate([[-np.inf], best[t - 1, :-1]])
        two = np.where(skip, np.concatenate([[-np.inf, -np.inf], best[t - 1, :-2]]), -np.inf)
        choice = np.argmax(np.stack([stay, one, two]), axis=0)
        best[t] = np.choose(choice, [stay, one, two]) + scores[t, labels]
        back[t] = choice
    k = n - 1 if best[-1, n - 1] >= best[-1, n - 2] else n - 2
    path = np.zeros(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = k if states else labels[k]
        k -= int(back[t, k])
    return path


def soft_path(log_beliefs: Any, transcript: Any, *, blank: int = BLANK) -> np.ndarray:
    """The distribution taught at each step: ``(steps, symbols)``, the share of the transcript's
    monotonic paths that say each symbol at that step, each path weighed by the product of the
    beliefs along it. The paths are ``forced_path``'s: blank allowed on any step and required
    between a symbol and the same symbol after it. Every row sums to one. Where the beliefs
    leave one path, the rows are that path's symbols; where they say nothing, the rows spread
    each symbol over the steps it can take. A transcript longer than the steps allow is
    refused."""
    scores = np.asarray(log_beliefs, dtype=np.float64)
    sequence = np.asarray(transcript, dtype=np.int64)
    if scores.ndim != 2 or len(sequence) == 0:
        raise ValueError("soft_path needs (steps, symbols) beliefs and a symbol")
    repeats = int(np.sum(sequence[1:] == sequence[:-1]))
    if len(scores) < len(sequence) + repeats:
        raise ValueError("the transcript needs more steps than the utterance holds")
    if np.isnan(scores).any():
        raise ValueError("soft_path needs beliefs without NaN")
    frames = len(scores)
    labels = np.full(2 * len(sequence) + 1, blank, dtype=np.int64)
    labels[1::2] = sequence
    n = len(labels)
    skip = np.zeros(n, dtype=bool)  # a symbol reachable from two states back
    skip[3::2] = labels[3::2] != labels[1:-2:2]
    emit = scores[:, labels]  # (steps, states)
    none = np.full(2, -np.inf)
    with np.errstate(invalid="ignore"):  # sums of impossible states stay impossible
        ahead = np.full((frames, n), -np.inf)  # every path to a state, its step's belief included
        ahead[0, :2] = emit[0, :2]
        for t in range(1, frames):
            stay = ahead[t - 1]
            one = np.concatenate([none[:1], stay[:-1]])
            two = np.where(skip, np.concatenate([none, stay[:-2]]), -np.inf)
            ahead[t] = np.logaddexp(np.logaddexp(stay, one), two) + emit[t]
        after = np.full((frames, n), -np.inf)  # every path from a state on, its step's excluded
        after[-1, -2:] = 0.0
        for t in range(frames - 2, -1, -1):
            onward = after[t + 1] + emit[t + 1]
            one = np.concatenate([onward[1:], none[:1]])
            two = np.concatenate([np.where(skip[2:], onward[2:], -np.inf), none])
            after[t] = np.logaddexp(np.logaddexp(onward, one), two)
        total = np.logaddexp(ahead[-1, -1], ahead[-1, -2])
    if not np.isfinite(total):
        raise ValueError("the beliefs leave the transcript no path")
    with np.errstate(invalid="ignore"):
        shares = np.exp(ahead + after - total)
    shares = np.nan_to_num(shares, nan=0.0)
    out = np.zeros_like(scores)
    np.add.at(out, (np.arange(frames)[:, None], labels[None, :]), shares)
    return np.asarray(out / out.sum(axis=1, keepdims=True))


def soft_step(brain: Any, rows: Any, targets: Any) -> dict[str, float]:
    """One moment of a soft lesson, in place of ``brain.step(rows, teacher=labels)``: the lesson
    pulls each stream toward its row of ``targets`` (``Brain.lesson(targets=...)``), then the
    brain answers the moment greedily, which carries its streams on. Learned fields take their
    one step of the moment at the act, as in ``step``, and are held during the lesson. Returns
    the lesson's report."""
    fields = getattr(brain, "fields", None)
    moving = fields is not None and bool(fields.learning)
    if moving:
        fields.learning = False
    try:
        report: dict[str, float] = brain.lesson(rows, targets=targets)
    finally:
        if moving:
            fields.learning = True
    brain.act(rows, greedy=True)
    return report


def log_beliefs(brain: Any, utterances: Sequence[Any]) -> list[np.ndarray]:
    """A brain's log beliefs over its symbols at each step of each utterance, the utterances
    heard side by side as streams from rest and answered greedily: the log-softmax of the settled
    motor activations at the brain's temperature. The brain's streams start again; give it a
    frozen copy (``Brain.load`` of a saved brain) where the life must go on untouched."""
    rows = [np.asarray(u, dtype=np.float64) for u in utterances]
    if not rows:
        return []
    longest = max(len(r) for r in rows)
    x = np.stack([np.concatenate([r, np.repeat(r[-1:], longest - len(r), axis=0)])
                  for r in rows], axis=1)
    brain.reset()
    motor = np.asarray(brain.connectome.populations["motor"], dtype=np.int64)
    temperature = float(brain.describe()["genes"]["temperature"])
    out = np.zeros((longest, len(rows), len(motor)))
    for t in range(longest):
        brain.act(x[t], greedy=True)
        drive = np.asarray(brain.basal_ganglia.state.activation)[:, motor] / temperature
        top = drive.max(axis=1, keepdims=True)
        out[t] = drive - top - np.log(np.sum(np.exp(drive - top), axis=1, keepdims=True))
    return [out[: len(r), k] for k, r in enumerate(rows)]


class Alignments:
    """The paths of hearings to come, one per hearing.

    ``key`` names a hearing (a pass and a place in it, for instance), not an utterance, so an
    utterance heard twice holds two paths and taking one leaves the other. ``even`` sets a
    hearing's path before any beliefs, ``realign`` from a frozen copy's log beliefs (``soft``:
    the soft path, a distribution per step), ``take`` returns it and forgets it. ``lesson``
    gives the labels of hearings heard side by side and ``targets`` their distributions.
    """

    def __init__(self, *, blank: int = BLANK) -> None:
        self.blank = blank
        self._paths: dict[Hashable, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._paths

    def even(self, key: Hashable, steps: int, transcript: Any) -> np.ndarray:
        self._paths[key] = even_path(steps, transcript)
        return self._paths[key]

    def realign(self, key: Hashable, log_beliefs: Any, transcript: Any, *,
                soft: bool = False) -> np.ndarray:
        """Hold the hearing's path through ``log_beliefs``: the best path's symbols
        (``forced_path``), or with ``soft`` the soft path's distributions (``soft_path``)."""
        self._paths[key] = (soft_path if soft else forced_path)(
            log_beliefs, transcript, blank=self.blank)
        return self._paths[key]

    def take(self, key: Hashable) -> np.ndarray:
        if key not in self._paths:
            raise KeyError(f"no path is held for the hearing {key!r}")
        return self._paths.pop(key)

    def lesson(self, keys: Sequence[Hashable], steps: int) -> np.ndarray:
        """The labels of hearings heard side by side: ``(steps, len(keys))``, each column its
        hearing's path, then blank on the steps after its end. Taking them forgets them. A
        hearing that holds a soft path is refused: its lesson is ``targets``."""
        for key in keys:
            if key in self._paths and self._paths[key].ndim != 1:
                raise ValueError(
                    f"the hearing {key!r} holds a soft path: teach it with targets()"
                )
        out = np.full((steps, len(keys)), self.blank, dtype=np.int64)
        for k, key in enumerate(keys):
            path = self.take(key)
            if len(path) > steps:
                raise ValueError("a hearing's path is longer than the steps of its lesson")
            out[: len(path), k] = path
        return out

    def targets(self, keys: Sequence[Hashable], steps: int, symbols: int) -> np.ndarray:
        """The distributions of hearings heard side by side: ``(steps, len(keys), symbols)``,
        row ``[t, k]`` what ``Brain.lesson(targets=...)`` teaches stream ``k`` at step ``t``. A
        hearing's soft path is given as held, a path of symbols as one-hot rows, and the steps
        after a hearing's end are blank. Taking them forgets them."""
        for key in keys:
            if key not in self._paths:
                raise KeyError(f"no path is held for the hearing {key!r}")
            held = self._paths[key]
            if len(held) > steps:
                raise ValueError("a hearing's path is longer than the steps of its lesson")
            if held.ndim == 2 and held.shape[1] != symbols:
                raise ValueError(
                    f"the hearing {key!r} holds distributions over {held.shape[1]} symbols, "
                    f"not {symbols}"
                )
            if held.ndim == 1 and len(held) and int(held.max()) >= symbols:
                raise ValueError(f"the hearing {key!r} holds a symbol outside the {symbols}")
        out = np.zeros((steps, len(keys), symbols))
        out[:, :, self.blank] = 1.0
        for k, key in enumerate(keys):
            path = self.take(key)
            if path.ndim == 1:
                out[: len(path), k] = np.eye(symbols)[path]
            else:
                out[: len(path), k] = path
        return out
