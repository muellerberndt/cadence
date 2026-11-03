"""Rehearsal from the lessons a brain was taught (cadence#238): after each lesson, rows drawn
from a reservoir of earlier lessons' rows (their drives and labels) are taught again under the
same rule, so types taught in sequence keep what earlier lessons taught."""

from __future__ import annotations

from dataclasses import replace

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


def test_save_grow_and_graft_leave_the_living_reservoir_as_it_is(tmp_path):
    brain, _ = _filled()
    # set without retune: the next lesson trims
    brain.learner.config = replace(brain.learner.config, rehearsal_capacity=4)
    before = brain.learner.rehearsal_drives.copy()
    brain.save(tmp_path / "brain.npz")
    np.testing.assert_array_equal(brain.learner.rehearsal_drives, before)
    brain.retune(learning_rehearsal_capacity=4)  # retune trims at once
    assert len(brain.learner.rehearsal_drives) == 4


def test_a_rehearsing_brain_learns_on_a_torch_device_as_on_the_cpu():
    torch = pytest.importorskip("torch")
    del torch
    rng = np.random.default_rng(3)
    rows = [rng.random((8, 12)) for _ in range(3)]
    labels = [rng.integers(0, 4, size=8) for _ in range(3)]
    brains = []
    for device in (None, "cpu"):
        brain = cd.Brain.compose(12, 4, seed=0, learning_rehearsal=1.0, learning_surprise=True)
        if device is not None:
            graph = brain.brain
            brain.learner.brain = cd.NeuralGraph(
                graph.connectome, graph.neuron_model, efficacy=graph.efficacy, bias=graph.bias,
                log_gain=graph.log_gain, backend="torch", device=device, precision="float64",
            )
        for x, y in zip(rows, labels, strict=True):
            brain.lesson(x, y)
        brains.append(brain)
    cpu, moved = brains
    assert len(cpu.learner.rehearsal_drives) == 24 and cpu.learner.rehearsal_draws > 0
    np.testing.assert_allclose(np.asarray(moved.learner.brain.efficacy, dtype=float),
                               np.asarray(cpu.learner.brain.efficacy, dtype=float), rtol=0, atol=1e-9)
    assert moved.learner.rehearsal_draws == cpu.learner.rehearsal_draws


def test_a_refused_rehearsal_is_reported_and_the_lesson_stands():
    rng = np.random.default_rng(4)
    brain = cd.Brain.compose(12, 4, seed=0, learning_rehearsal=1.0)
    brain.lesson(rng.random((8, 12)), rng.integers(0, 4, size=8))  # fills the reservoir
    learner = brain.learner
    stored = np.array(learner.rehearsal_drives)
    step = learner.step
    strict = cd.Brain.compose(12, 4, seed=0, learning_qualified=True, learning_nudged_steps=1,
                              learning_tolerance=1e-12)
    with pytest.raises(cd.LearningPhaseError) as caught:  # a refusal as the learner raises it
        strict.lesson(rng.random((8, 12)), rng.integers(0, 4, size=8))

    def refusing(drive, labels, *args, **kwargs):
        if any((np.asarray(drive)[0] == row).all() for row in stored):  # a rehearsed row
            raise caught.value
        return step(drive, labels, *args, **kwargs)

    learner.step = refusing
    before = np.asarray(brain.brain.efficacy).copy()
    brain.lesson(rng.random((8, 12)), rng.integers(0, 4, size=8))
    assert brain.last_learning["rehearsal_refused"] == 1.0
    assert not np.array_equal(np.asarray(brain.brain.efficacy), before)  # the lesson itself
