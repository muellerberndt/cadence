"""Eight full-factorial two-fact cases with a shared fixed-order query.

This task is a smaller component diagnostic. Every pair is trained; there is
no held-out split or transfer reading. The sensory/motor interface retains
19 inputs and two four-choice output slots, with only values 0/1 active.
No brain is constructed or run on import.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

INPUTS = 19
SLOTS = 2
MOTOR_CHOICES = 4
VALUES = MOTOR_CHOICES  # Existing compose call shape: actions=VALUES*SLOTS.
ACTIVE_VALUES = (0, 1)
CASES = np.asarray(
    [(*row, 0) for row in itertools.product(ACTIVE_VALUES, ACTIVE_VALUES, range(2))],
    dtype=np.int64,
)
CASES.flags.writeable = False


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate(rows: Any) -> np.ndarray:
    """Rows are (A value, B value, prefix order, fixed query code 0)."""
    rows = np.asarray(rows)
    if (
        rows.ndim != 2
        or rows.shape[1] != 4
        or not len(rows)
        or rows.dtype.kind not in "iu"
        or (rows < 0).any()
        or (rows[:, :2] > 1).any()
        or (rows[:, 2] > 1).any()
        or (rows[:, 3] != 0).any()
    ):
        raise ValueError("rows need active values0/1, prefix order0/1 and fixed query0")
    return rows.astype(np.int64, copy=False)


def targets(rows: Any) -> np.ndarray:
    """Both named values, in fixed A/B output order."""
    return validate(rows)[:, :2].copy()


def events(rows: Any, *, exposed: bool = False) -> tuple[list[np.ndarray], np.ndarray]:
    """Actual named facts, either order, neutral, then the shared query.

    Only the explicitly exposed control supplies both values in dormant
    query coordinates. Candidate queries contain no value or record address.
    """
    rows = validate(rows)
    if not isinstance(exposed, (bool, np.bool_)):
        raise TypeError("exposed must be boolean")
    batch, index = len(rows), np.arange(len(rows))
    facts = []
    for name in range(SLOTS):
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
    query[:, (2, 3)] = 1.0  # AMBIGUOUS_CUE + ASK_AB in every case.
    if exposed:
        query[index, 11 + rows[:, 0]] = 1.0
        query[index, 15 + rows[:, 1]] = 1.0
    return [first, second, neutral], query


def panel(name: str) -> np.ndarray:
    """The same eight cases serve acquisition and evaluation; none are held."""
    if name not in ("train", "all"):
        raise ValueError("only train/all exist; this full-factorial task has no held-out panel")
    return CASES.copy()


def draw(seed: int, rounds: int, batch: int = 16) -> np.ndarray:
    """Draw batches from all eight cases, using the existing schedule namespace."""
    seed = _integer("seed", seed)
    rounds = _integer("rounds", rounds)
    batch = _integer("batch", batch, 1)
    rng = np.random.default_rng(np.random.SeedSequence([622022, seed]))
    return CASES[rng.integers(len(CASES), size=(rounds, batch))].copy()


def swapped(rows: Any) -> np.ndarray:
    """Exchange the two actual values, retaining names, prefix and query."""
    out = validate(rows).copy()
    out[:, :2] = out[:, 1::-1]
    return out


def flipped(rows: Any, name: int) -> np.ndarray:
    """The actual counterpart donor case with one named binary value changed."""
    name = _integer("name", name)
    if name >= SLOTS:
        raise ValueError("name must be0(A) or1(B)")
    out = validate(rows).copy()
    out[:, name] = 1 - out[:, name]
    return out


def scores(answers: Any, rows: Any) -> dict[str, Any]:
    """Retain refused/unrun(-1) and inactive motor choices2/3 as wrong."""
    rows = validate(rows)
    answers = np.asarray(answers)
    if (
        answers.shape != (len(rows), SLOTS)
        or answers.dtype.kind not in "iu"
        or not np.isin(answers, range(-1, MOTOR_CHOICES)).all()
    ):
        raise ValueError("answers need two integer motor choices0..3, or-1 for unrun/refused")
    expected = targets(rows)
    correct = (answers == expected) & (answers >= 0)
    exact = correct.all(axis=1)
    unequal = rows[:, 0] != rows[:, 1]
    return {
        "cases": len(rows),
        "correct_pairs": int(exact.sum()),
        "per_slot_correct": correct.sum(axis=0).tolist(),
        "unequal_cases": int(unequal.sum()),
        "unequal_correct_pairs": int(exact[unequal].sum()),
        "exact_pair_accuracy": float(exact.mean()),
        "per_slot_accuracy": correct.mean(axis=0).tolist(),
        "per_query_accuracy": [float(exact.mean())],  # One fixed query, no missing second query.
        "per_prefix_accuracy": [
            float(exact[rows[:, 2] == order].mean()) if (rows[:, 2] == order).any() else None
            for order in range(2)
        ],
        "unequal_pair_accuracy": float(exact[unequal].mean()) if unequal.any() else None,
        "unanswered_slots": int((answers < 0).sum()),
        "expected": expected.tolist(),
    }


def random_reference(seed: int, *, policy: str, rows: Any = None) -> dict[str, Any]:
    """Actual independent random answers plus their analytic chance reference.

    Active-binary uniform is the task-aware baseline. Full-motor uniform is
    separately reported because the unchanged brain retains four choices.
    Neither random stream consumes the training or action RNG.
    """
    seed = _integer("seed", seed)
    rows = panel("all") if rows is None else validate(rows)
    if policy == "active_binary":
        count, namespace = len(ACTIVE_VALUES), 622024
    elif policy == "full_motor":
        count, namespace = MOTOR_CHOICES, 622023
    else:
        raise ValueError("random policy must be active_binary or full_motor")
    entropy = [namespace, seed]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    answer = rng.integers(count, size=(len(rows), SLOTS))
    return {
        "policy": policy,
        "seed_entropy": entropy,
        "choices": list(range(count)),
        "answers": answer.tolist(),
        "scores": scores(answer, rows),
        "expected_exact_pair_accuracy": 1 / count**SLOTS,
        "expected_per_slot_accuracy": 1 / count,
    }


def random_references(seed: int, rows: Any = None) -> dict[str, dict[str, Any]]:
    return {
        policy: random_reference(seed, policy=policy, rows=rows)
        for policy in ("active_binary", "full_motor")
    }
