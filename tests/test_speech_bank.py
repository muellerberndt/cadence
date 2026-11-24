"""The rehearsal bank holds utterances of its own (cadence.speech.correct.Rehearsal)."""

from __future__ import annotations

import numpy as np

from cadence import Brain
from cadence.speech.correct import Rehearsal


def brain(seed: int = 0) -> Brain:
    return Brain.compose(8, 4, seed=seed, backend="cpu", learning_surprise=False)


def utterance(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 4, size=steps)
    rows = np.eye(4)[labels].repeat(2, axis=1) + 0.1 * rng.standard_normal((steps, 8))
    return rows.astype(np.float32), labels


def test_arrays_filled_again_after_remember_leave_the_bank_as_it_was():
    rows = np.eye(2, dtype=np.float32)
    targets = rows.copy()
    labels = np.array([1, 0])
    longer = np.arange(40, dtype=np.float32).reshape(10, 4)
    bank = Rehearsal(rows=3)
    bank.remember(rows, targets=targets)
    bank.remember(rows, labels)
    bank.remember(longer[2:4, :2], labels)  # two rows of a larger array
    held = bank.nbytes
    assert held == 3 * rows.nbytes + targets.nbytes + 2 * labels.nbytes
    rows[:] = 0
    targets[:] = 0
    labels[:] = 3
    longer[:] = 0
    kept = sorted(bank.chosen(2), key=lambda u: (u[1].ndim, float(u[0].sum())))
    assert [x.tolist() for x, _ in kept] == [np.eye(2).tolist(), [[8.0, 9.0], [12.0, 13.0]],
                                             np.eye(2).tolist()]
    assert [y.tolist() for _, y in kept] == [[1, 0], [1, 0], np.eye(2).tolist()]
    assert bank.nbytes == held and len(bank) == 3


def test_one_buffer_for_every_utterance_fills_the_bank_with_each_as_it_was_given():
    buffer = np.zeros((10, 8), dtype=np.float32)
    said = np.zeros(10, dtype=np.int64)
    size = buffer.nbytes + said.nbytes
    bank, twin = (Rehearsal(rows=7, budget=3 * size, seed=1) for _ in range(2))
    for k in range(7):
        buffer[:] = k
        said[:] = k % 4
        bank.remember(buffer, said)
        twin.remember(buffer.copy(), said.copy())  # the twin is given arrays of its own
        assert bank.nbytes <= bank.budget
    assert len(bank) == 3 and bank.nbytes == twin.nbytes == 3 * size

    def held(of: Rehearsal) -> list[tuple[float, int]]:
        return sorted((float(x[0, 0]), int(y[0])) for x, y in of.chosen(10))

    mine = held(bank)
    assert mine == held(twin)  # the utterances the budget kept, each as it was given
    assert len({first for first, _ in mine}) == 3


def test_a_selection_is_a_copy_the_caller_may_change():
    rng = np.random.default_rng(5)
    bank, twin = (Rehearsal(rows=3, weight=0.25, seed=2) for _ in range(2))
    for steps in (6, 6, 9):
        rows, labels = utterance(rng, steps)
        bank.remember(rows, labels)
        twin.remember(rows.copy(), labels.copy())
    held = bank.nbytes
    for rows, labels in bank.chosen(6):
        rows[:] = 0.0
        labels[:] = 0
    twin.chosen(6)  # the twin draws as often
    assert bank.nbytes == held == twin.nbytes and len(bank) == 3
    for (rows, labels), (theirs, said) in zip(bank.chosen(6), twin.chosen(6), strict=True):
        np.testing.assert_array_equal(rows, theirs)
        np.testing.assert_array_equal(labels, said)
        assert rows.any()
    corrected, beside = brain(), brain()
    rows, labels = utterance(rng, 6)
    bank.correct(corrected, rows, labels)  # the correction hears what the bank was given
    twin.correct(beside, rows, labels)
    np.testing.assert_array_equal(np.asarray(corrected.brain.efficacy),
                                  np.asarray(beside.brain.efficacy))
