"""Rehearsal from the lessons a brain was taught (cadence#238): after each lesson, rows drawn
from a reservoir of earlier lessons' rows (their drives and labels) are taught again under the
same rule, so types taught in sequence keep what earlier lessons taught."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def lesson_rows(kind: int, rng: np.random.Generator, rows: int = 8):
    """Type ``kind``: an observation in its own third of 12 inputs, labelled by its own rule."""
    x = np.zeros((rows, 12))
    index = rng.integers(0, 4, size=rows)
    x[np.arange(rows), kind * 4 + index] = 1.0
    return x, (index + kind) % 4


def taught_in_sequence(rehearsal: float, lessons: int = 150) -> cd.Brain:
    brain = cd.Brain.compose(12, 4, seed=0, learning_rehearsal=rehearsal, learning_surprise=True)
    rng = np.random.default_rng(0)
    for kind in (0, 1, 2):
        for _ in range(lessons):
            brain.reset()
            brain.lesson(*lesson_rows(kind, rng))
    return brain


def accuracy(brain: cd.Brain, kind: int) -> float:
    x, labels = lesson_rows(kind, np.random.default_rng(9 + kind), 64)
    brain.reset()
    return float(np.mean(brain.act(x, greedy=True) == labels))


def test_the_founder_keeps_no_reservoir_and_no_record_of_one(tmp_path):
    brain = cd.Brain.compose(12, 4, seed=0)
    brain.lesson(*lesson_rows(0, np.random.default_rng(0)))
    assert brain.learner.config.rehearsal == 0.0 and not len(brain.learner.rehearsal_drives)
    assert "learning_rehearsal" not in brain.describe()["genes"]
    with np.load(brain.save(tmp_path / "brain.npz")) as saved:
        assert "rehearsal_drives" not in saved.files


def test_types_taught_in_sequence_keep_what_earlier_lessons_taught():
    plain, rehearsed = taught_in_sequence(0.0), taught_in_sequence(1.0)
    kept = [accuracy(rehearsed, kind) for kind in (0, 1, 2)]
    assert min(kept) >= 0.95, kept
    assert np.mean(kept) > np.mean([accuracy(plain, kind) for kind in (0, 1, 2)])


def test_the_reservoir_keeps_a_uniform_sample_within_its_capacity():
    brain = taught_in_sequence(1.0, lessons=100)
    learner = brain.learner
    assert len(learner.rehearsal_drives) == learner.config.rehearsal_capacity == 512
    assert learner.rehearsal_seen == 3 * 100 * 8
    sensory = learner.brain.connectome.populations["sensory"]
    kinds = np.argmax(learner.rehearsal_drives[:, sensory], axis=1) // 4
    assert all(np.bincount(kinds, minlength=3) > 100)  # every type in the sample


def test_a_rehearsing_brain_saves_and_continues_exactly(tmp_path):
    brain = taught_in_sequence(1.0, lessons=20)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    for name in ("rehearsal_drives", "rehearsal_labels", "rehearsal_seen", "rehearsal_draws"):
        np.testing.assert_array_equal(getattr(copy.learner, name), getattr(brain.learner, name))
    rng = np.random.default_rng(3)
    for kind in (0, 1, 2):
        x, labels = lesson_rows(kind, rng)
        brain.lesson(x, labels)
        copy.lesson(x, labels)
    np.testing.assert_array_equal(copy.brain.efficacy, brain.brain.efficacy)


def test_rehearsal_settings_are_validated():
    with pytest.raises(ValueError, match="rehearsal must be"):
        cd.Brain.compose(12, 4, learning_rehearsal=-1.0)
    with pytest.raises(ValueError, match="rehearsal_capacity"):
        cd.Brain.compose(12, 4, learning_rehearsal=1.0, learning_rehearsal_capacity=0)


def test_the_capacity_is_a_named_integer_setting():
    brain = cd.Brain.compose(6, 4, seed=0, learning_rehearsal=1.0, learning_rehearsal_capacity=256)
    assert brain.learner.config.rehearsal_capacity == 256
    brain.retune(learning_rehearsal_capacity=128)
    assert brain.learner.config.rehearsal_capacity == 128


def _filled(rows: int = 8) -> tuple[cd.Brain, np.random.Generator]:
    brain = cd.Brain.compose(6, 4, seed=0, learning_rehearsal=1.0,
                             learning_rehearsal_capacity=rows)
    rng = np.random.default_rng(0)
    for _ in range(rows // 2):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))
    assert len(brain.learner.rehearsal_drives) == rows
    return brain, rng


def test_a_smaller_capacity_trims_the_reservoir_and_the_brain_saves_and_loads(tmp_path):
    brain, rng = _filled()
    brain.retune(learning_rehearsal_capacity=4)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert len(copy.learner.rehearsal_drives) == 4
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    assert len(brain.learner.rehearsal_drives) == 4
    np.testing.assert_array_equal(brain.learner.rehearsal_drives, copy.learner.rehearsal_drives)


def test_the_reservoir_is_saved_while_rehearsal_is_off(tmp_path):
    brain, rng = _filled()
    brain.retune(learning_rehearsal=0.0)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    np.testing.assert_array_equal(brain.learner.rehearsal_drives, copy.learner.rehearsal_drives)
    for twin in (brain, copy):
        twin.retune(learning_rehearsal=1.0)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
