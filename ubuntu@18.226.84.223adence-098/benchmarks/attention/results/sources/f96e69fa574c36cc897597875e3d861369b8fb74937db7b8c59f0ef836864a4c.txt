"""Pure witnessed-payload arrays; no brain, controller, teaching or run protocol."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

INPUTS = 9
NAME_A, NAME_B, PAYLOAD, IS, NEUTRAL, ASK_A, ASK_B, EXPOSED_A, EXPOSED_B = range(INPUTS)
HISTORIES = np.array(list(product(range(2), repeat=3)), dtype=np.int64)
HISTORIES.flags.writeable = False  # A class, B class, presentation order
SPLITS = {"train": 0, "held": 1, "donor": 2}
ENTROPY = 622031


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return int(value)


def _rng(seed, index, split, channel):
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError("choose an explicit train, held or donor stream")
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                ENTROPY,
                _integer(seed, "seed"),
                _integer(index, "index"),
                SPLITS[split],
                channel,
            ]
        )
    )


def _values(rows, rng):
    return np.where(rows[:, :2] == 0, 0.1, 0.6) + 0.3 * rng.random((8, 2))


@dataclass(frozen=True)
class Histories:
    """Eight immutable actual histories; both questions share these exact quantities.

    ``rows`` is metadata for the experiment/scorer, never an observation or an
    address supplied to a brain. Its columns are A class, B class and order.
    """

    rows: np.ndarray
    quantities: np.ndarray

    def __post_init__(self):
        rows, quantities = np.asarray(self.rows), np.asarray(self.quantities)
        if (
            rows.shape != (8, 3)
            or rows.dtype.kind not in "iu"
            or not np.array_equal(np.unique(rows, axis=0), HISTORIES)
        ):
            raise ValueError("one of each of the eight stratified histories is required")
        if quantities.shape != (8, 2) or quantities.dtype.kind not in "fiu":
            raise ValueError("quantities must be a finite8-by2 numeric array")
        lower = np.where(rows[:, :2] == 0, 0.1, 0.6)
        upper = np.where(rows[:, :2] == 0, 0.4, 0.9)
        if (
            not np.isfinite(quantities).all()
            or np.any(quantities < lower)
            or np.any(quantities > upper)
        ):
            raise ValueError("each quantity must lie in its declared low or high interval")
        for name, array, dtype in (("rows", rows, np.int64), ("quantities", quantities, float)):
            frozen = np.array(array, dtype=dtype, copy=True)
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)


def draw(seed, index, *, split="train"):
    """Independent payload/permutation channels; drawing one split cannot advance another."""
    quantities = _values(HISTORIES, _rng(seed, index, split, 0))
    order = _rng(seed, index, split, 1).permutation(8)
    return Histories(HISTORIES[order], quantities[order])


def schedule(seed, rounds, *, split="train"):
    """Caller chooses the count; return full quantities rather than only an RNG seed."""
    return tuple(draw(seed, tick, split=split) for tick in range(_integer(rounds, "rounds")))


def _query(query):
    query = _integer(query, "query")
    if query not in (0, 1):
        raise ValueError("query must name A or B")
    return query


def events(batch, query, *, exposed=False, neutral_amplitude=1.0, reverse_order=False):
    """Actual facts, neutral, and one identity-only question; no selected address.

    Reversing order or changing neutral amplitude preserves the exact quantities
    and question. Exposed controls receive BOTH actual unselected quantities.
    """
    query = _query(query)
    if not isinstance(exposed, (bool, np.bool_)) or not isinstance(reverse_order, (bool, np.bool_)):
        raise ValueError("exposed and reverse_order must be booleans")
    if (
        isinstance(neutral_amplitude, (bool, np.bool_))
        or not np.isscalar(neutral_amplitude)
        or not np.isfinite(neutral_amplitude)
        or not 0 <= neutral_amplitude <= 1
    ):
        raise ValueError("neutral amplitude must be finite in[0,1]")
    facts = np.zeros((2, 8, INPUTS))
    facts[:, :, IS] = 1
    for name in range(2):
        facts[name, :, name] = 1
        facts[name, :, PAYLOAD] = batch.quantities[:, name]
    order = batch.rows[:, 2] ^ int(reverse_order)
    rows = np.arange(8)
    first, second = facts[order, rows], facts[1 - order, rows]
    neutral = np.zeros((8, INPUTS))
    neutral[:, NEUTRAL] = neutral_amplitude
    question = np.zeros((8, INPUTS))
    question[:, ASK_A + query] = 1
    if exposed:
        question[:, EXPOSED_A : EXPOSED_B + 1] = batch.quantities
    return [first, second, neutral], question


def targets(batch, query):
    """Only the final binary answer, evaluated from the actual asked quantity."""
    return (batch.quantities[:, _query(query)] >= 0.5).astype(np.int64)


def donor(batch, name, seed, index, *, opposite_class=True):
    """Alternate actual histories: replace one quantity only, preserving name/order.

    A future controller must witness these events and transplant the chosen
    record's actual K AND V together. This array helper performs no memory edit.
    """
    name = _query(name)
    if not isinstance(opposite_class, (bool, np.bool_)):
        raise ValueError("opposite_class must be boolean")
    rows, quantities = batch.rows.copy(), batch.quantities.copy()
    rows[:, name] ^= int(opposite_class)
    replacement = _values(rows, _rng(seed, index, "donor", 2 + name))
    quantities[:, name] = replacement[:, name]
    return Histories(rows, quantities)


def record(batch):
    """Full retained values and sixteen query strata; metadata never enters the input."""
    rows = np.column_stack((np.repeat(batch.rows, 2, axis=0), np.tile([0, 1], 8)))
    return {"rows": rows.tolist(), "quantities": np.repeat(batch.quantities, 2, axis=0).tolist()}


def scores(answers, batch):
    """Descriptive sixteen-answer scores; this declares no acquisition gate."""
    answers = np.asarray(answers)
    if (
        answers.shape != (8, 2)
        or answers.dtype.kind not in "iu"
        or not np.isin(answers, (-1, 0, 1)).all()
    ):
        raise ValueError("answers must be eight paired binary questions; -1 denotes unavailable")
    correct = answers == (batch.quantities >= 0.5)
    unequal = batch.rows[:, 0] != batch.rows[:, 1]
    return {
        "accuracy": float(correct.mean()),
        "both_questions_correct": float(correct.all(axis=1).mean()),
        "per_query_accuracy": correct.mean(axis=0).tolist(),
        "per_order_accuracy": [float(correct[batch.rows[:, 2] == k].mean()) for k in (0, 1)],
        "unequal_fact_accuracy": float(correct[unequal].mean()),
        "unanswered": int((answers == -1).sum()),
    }
