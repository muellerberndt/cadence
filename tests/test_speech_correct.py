"""Corrections heard beside earlier utterances at a weight (cadence.speech.correct)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cadence import Brain
from cadence.speech.correct import Rehearsal


def brain(seed: int = 0) -> Brain:
    return Brain.compose(8, 4, seed=seed, backend="cpu", learning_surprise=False)


def utterance(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 4, size=steps)
    rows = np.eye(4)[labels].repeat(2, axis=1) + 0.1 * rng.standard_normal((steps, 8))
    return rows, labels


def test_the_bank_keeps_its_budget_and_refuses_an_utterance_larger_than_it():
    rng = np.random.default_rng(0)
    rows, labels = utterance(rng, 10)
    size = rows.astype(np.float32).nbytes + labels.astype(np.int64).nbytes
    bank = Rehearsal(budget=3 * size, seed=1)
    for _ in range(7):
        bank.remember(*utterance(rng, 10))
        assert bank.nbytes <= bank.budget
    assert len(bank) == 3
    with pytest.raises(ValueError):
        Rehearsal(budget=size - 1).remember(rows, labels)
    for bad in ({"rows": -1}, {"weight": -0.5}, {"weight": float("nan")}, {"budget": 0}):
        with pytest.raises(ValueError):
            Rehearsal(**bad)
    with pytest.raises(ValueError):
        bank.remember(rows, labels[:-1])
    with pytest.raises(ValueError):
        bank.remember(rows[:, :3][None], labels)


def test_a_correction_hears_earlier_utterances_of_its_own_length_first():
    rng = np.random.default_rng(1)
    bank = Rehearsal(rows=2, seed=0)
    for steps in (5, 9, 7, 7, 12):
        bank.remember(*utterance(rng, steps))
    assert sorted(len(x) for x, _ in bank.chosen(7)) == [7, 7]
    assert sorted(len(x) for x, _ in bank.chosen(10)) == [9, 12]
    assert Rehearsal(rows=0).chosen(7) == []


def test_at_weight_zero_the_earlier_streams_teach_nothing():
    rng = np.random.default_rng(2)
    rows, labels = utterance(rng, 6)
    alone, beside, quarter = brain(), brain(), brain()
    before = np.array(alone.brain.efficacy, copy=True)
    Rehearsal(rows=0).correct(alone, rows, labels)
    held = Rehearsal(rows=3, weight=0.0, seed=3)
    rehearsed = Rehearsal(rows=3, weight=0.25, seed=3)
    for _ in range(3):
        earlier = utterance(rng, 6)
        held.remember(*earlier)
        rehearsed.remember(*earlier)
    report = held.correct(beside, rows, labels)
    assert report["streams"] == 4 and report["steps"] == 6 and report["refused"] == 0
    moved = np.asarray(alone.brain.efficacy) - before
    assert np.linalg.norm(moved) > 0
    assert np.allclose(np.asarray(beside.brain.efficacy) - before, moved, atol=1e-9 * (
        1 + np.linalg.norm(moved)))
    rehearsed.correct(quarter, rows, labels)
    assert not np.allclose(np.asarray(quarter.brain.efficacy) - before, moved, atol=1e-6)


def test_streams_of_other_lengths_hold_their_last_row_and_take_no_lesson_past_it():
    rng = np.random.default_rng(3)
    rows, labels = utterance(rng, 5)
    longer = utterance(rng, 9)
    plain, held = brain(), brain()
    Rehearsal(rows=0).correct(plain, rows, labels)
    bank = Rehearsal(rows=1, weight=0.0)
    bank.remember(*longer)
    report = bank.correct(held, rows, labels)
    assert report["steps"] == 9 and len(report["actions"]) == 5
    assert np.allclose(np.asarray(held.brain.efficacy), np.asarray(plain.brain.efficacy),
                       atol=1e-9)


def test_repeated_corrections_teach_the_corrected_utterance():
    rng = np.random.default_rng(4)
    rows, labels = utterance(rng, 8)
    learner = brain(5)
    bank = Rehearsal(rows=3, weight=0.25, seed=0)
    for _ in range(3):
        bank.remember(*utterance(rng, 8))
    first = bank.correct(copy.deepcopy(learner), rows, labels)["actions"]
    for _ in range(25):
        last = bank.correct(learner, rows, labels)["actions"]
    assert np.mean(last == labels) > np.mean(first == labels)
    assert np.mean(last == labels) >= 0.75


def test_import_cadence_loads_no_speech_piece():
    import subprocess
    import sys

    code = "import sys, cadence; assert not [m for m in sys.modules if m.startswith('cadence.speech')]"
    subprocess.run([sys.executable, "-c", code], check=True)
