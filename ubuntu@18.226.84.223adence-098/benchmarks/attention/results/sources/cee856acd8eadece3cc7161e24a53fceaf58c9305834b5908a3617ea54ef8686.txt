"""Two shared-token bindings, required together in one current answer.

This deliberately small component task separates a contextual cortical state
from a context-free observation code. The same cue and query occur under every
binding. Swapping the two values preserves the unordered input-token multiset,
but changes both correct output slots. No record address is supplied.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

VALUES = 4
SLOTS = 2
# is, neutral, ambiguous cue, ask AB, ask BA; A/B names; four shared values;
# eight dormant sensory coordinates used ONLY by the exposed acquisition control.
INPUTS = 19
HELD_PAIRS = ((0, 1), (1, 0), (2, 3), (3, 2))
CASES = np.array(list(itertools.product(range(VALUES), range(VALUES), range(2), range(2))),
                 dtype=np.int64)
HELD = np.array([tuple(row[:2]) in HELD_PAIRS for row in CASES], dtype=bool)


def validate(rows: Any) -> np.ndarray:
    rows = np.asarray(rows)
    if (rows.ndim != 2 or rows.shape[1] != 4 or not len(rows)
            or rows.dtype.kind not in "iu" or (rows < 0).any()
            or (rows[:, :2] >= VALUES).any() or (rows[:, 2:] >= 2).any()):
        raise ValueError("rows require integer (A value, B value, prefix order, query order)")
    return rows.astype(np.int64, copy=False)


def targets(rows: Any) -> np.ndarray:
    rows = validate(rows)
    return np.where(rows[:, 3, None] == 0, rows[:, :2], rows[:, 1::-1]).copy()


def events(rows: Any, *, exposed: bool = False) -> tuple[list[np.ndarray], np.ndarray]:
    rows = validate(rows)
    batch, index = len(rows), np.arange(len(rows))
    facts = []
    for name in range(2):
        fact = np.zeros((batch, INPUTS))
        fact[:, 0] = 1.0
        fact[:, 5 + name] = 1.0
        fact[index, 7 + rows[:, name]] = 1.0
        facts.append(fact)
    first = np.where(rows[:, 2, None] == 0, facts[0], facts[1])
    second = np.where(rows[:, 2, None] == 0, facts[1], facts[0])
    neutral = np.zeros((batch, INPUTS))
    neutral[:, 1] = 1.0
    query = np.zeros((batch, INPUTS))
    query[:, 2] = 1.0
    query[index, 3 + rows[:, 3]] = 1.0
    if exposed:
        query[index, 11 + rows[:, 0]] = 1.0
        query[index, 15 + rows[:, 1]] = 1.0
    return [first, second, neutral], query


def panel(name: str) -> np.ndarray:
    if name == "train":
        return CASES[~HELD].copy()
    if name == "held_out":
        return CASES[HELD].copy()
    if name == "all":
        return CASES.copy()
    raise ValueError("unknown panel")


def draw(seed: int, rounds: int, batch: int = 16) -> np.ndarray:
    """A fixed schedule; held-out pairs never occur in a training prefix or teacher."""
    rng = np.random.default_rng(np.random.SeedSequence([622022, seed]))
    training = panel("train")
    return training[rng.integers(len(training), size=(rounds, batch))]


def swapped(rows: Any) -> np.ndarray:
    out = validate(rows).copy()
    out[:, :2] = out[:, 1::-1]
    return out


def scores(answers: Any, rows: Any) -> dict[str, Any]:
    rows = validate(rows)
    answers = np.asarray(answers)
    if (answers.shape != (len(rows), SLOTS) or answers.dtype.kind not in "iu"
            or not np.isin(answers, range(-1, VALUES)).all()):
        raise ValueError("answers need two integer choices per case, with -1 for unrun/refused")
    expected = targets(rows)
    correct = (answers == expected) & (answers >= 0)
    exact = correct.all(axis=1)
    unequal = rows[:, 0] != rows[:, 1]
    return {
        "exact_pair_accuracy": float(exact.mean()),
        "per_slot_accuracy": correct.mean(axis=0).tolist(),
        "per_query_accuracy": [float(exact[rows[:, 3] == q].mean())
                               if (rows[:, 3] == q).any() else None for q in range(2)],
        "unequal_pair_accuracy": float(exact[unequal].mean()) if unequal.any() else None,
        "unanswered_slots": int((answers < 0).sum()),
        "expected": expected.tolist(),
    }
