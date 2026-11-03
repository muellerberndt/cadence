"""Wrong-row teaching retains the same witnessed streams through all three phases."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

SPEC = importlib.util.spec_from_file_location(
    "attention_coupled_rows", Path(__file__).with_name("coupled.py"))
assert SPEC is not None and SPEC.loader is not None
coupled = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coupled
SPEC.loader.exec_module(coupled)


def boundary():
    rng = np.random.default_rng(41)
    return coupled.Memory(
        np.array([0, 1]), np.array([2, 3, 4]),
        rng.normal(size=(3, 4, 2)), rng.normal(size=(3, 4, 3)),
        np.array([[True, False, True, True], [False] * 4, [False, True, False, True]]),
        np.arange(4.0), amplitude=0.3, temperature=0.8, recency=0.1, null=-0.2)


@pytest.mark.parametrize("rows,index", [
    ([2, 0], [2, 0]), (np.array([False, True, True]), [1, 2]), ([], []),
])
def test_subset_preserves_stream_order_masks_drives_and_immutable_boundaries(rows, index):
    memory = boundary()
    subset = memory.take(rows)
    index = np.asarray(index, dtype=np.int64)
    assert subset.streams == len(index)
    for name in ("keys", "values", "held"):
        np.testing.assert_array_equal(getattr(subset, name), getattr(memory, name)[index])
    for name in ("query", "read", "ages"):
        np.testing.assert_array_equal(getattr(subset, name), getattr(memory, name))
    state = np.random.default_rng(12).normal(size=(3, 6))
    np.testing.assert_allclose(subset.drive(state[index]), memory.drive(state)[index])
    np.testing.assert_allclose(subset.energy(state[index]), memory.energy(state)[index])
    for name in ("keys", "values", "held", "query", "read", "ages"):
        assert not getattr(subset, name).flags.writeable
        assert not np.shares_memory(getattr(subset, name), getattr(memory, name))


@pytest.mark.parametrize("rows", [
    [0, 0], [-1], [3], [0.5], [[0]], [True, False],
    np.array([2**64 - 1], dtype=np.uint64),
])
def test_invalid_stream_subsets_refuse_without_changing_the_original(rows):
    memory = boundary()
    before = memory.keys.copy()
    with pytest.raises(ValueError, match="row"):
        memory.take(rows)
    np.testing.assert_array_equal(memory.keys, before)


def captured_brain():
    brain = coupled.AttentionBrain.compose(
        3, 2, modules=(4,), episodic=False, seed=2,
        learning_free_steps=256, learning_nudged_steps=256)
    brain.act(np.eye(3), greedy=True)
    query = np.roll(np.eye(3), 1, axis=1)
    drive = brain.stimulus(query).copy()
    memory = brain.brain.memory
    assert memory is not None
    # The query joins live slots; the captured boundary must stay earlier.
    brain.act(query, greedy=True)
    assert brain.recall.slots == memory.keys.shape[1] + 1
    return brain, drive, memory


def forbidden(*args, **kwargs):
    raise AssertionError("a saved lesson must not read or write a new moment")


def test_saved_lesson_uses_one_reordered_snapshot_in_all_phases_and_parameter_copies(monkeypatch):
    brain, drive, memory = captured_brain()
    original = brain.brain
    before = {name: value.copy() for name, value in brain.recall.arrays().items()}
    reading = brain.recall.reading()
    phases = []
    qualified = brain.learner._qualified_phase

    def observed(*args, **kwargs):
        phases.append((brain.brain.memory, np.asarray(args[0]).copy()))
        return qualified(*args, **kwargs)

    monkeypatch.setattr(brain.learner, "_qualified_phase", observed)
    monkeypatch.setattr(brain, "stimulus", forbidden)
    monkeypatch.setattr(brain.recall, "witness", forbidden)
    monkeypatch.setattr(brain.recall, "learn", forbidden)
    learned, report = brain.teach_saved(drive, np.array([0, 1, 1]), [2, 0], memory)
    assert report["accepted_presentations"] == 2 and report["qualified"]
    assert len(phases) == 3
    subset = phases[0][0]
    assert all(phase[0] is subset for phase in phases)
    np.testing.assert_array_equal(subset.keys, memory.keys[[2, 0]])
    np.testing.assert_array_equal(subset.values, memory.values[[2, 0]])
    for _, actual_drive in phases:
        np.testing.assert_array_equal(actual_drive, drive[[2, 0]])
    assert learned.free.activation.shape[0] == 2
    assert brain.brain is not original
    assert brain.brain.memory is original.memory is memory
    after = brain.recall.arrays()
    for name, value in before.items():
        np.testing.assert_array_equal(after[name], value)
    assert brain.recall.reading()[0] is reading[0]
    assert brain.recall.reading()[1] is reading[1]


def test_refused_saved_lesson_restores_full_boundary_and_parameters():
    brain, drive, memory = captured_brain()
    original = brain.brain
    efficacy, bias = original.efficacy.copy(), original.bias.copy()
    brain.learner.config = replace(brain.learner.config, free_steps=0, tolerance=1e-12)
    with pytest.raises(cd.LearningPhaseError, match="free"):
        brain.teach_saved(drive, np.array([0, 1, 1]), [2, 0], memory)
    assert brain.brain is original and original.memory is memory
    np.testing.assert_array_equal(original.efficacy, efficacy)
    np.testing.assert_array_equal(original.bias, bias)
    assert brain.learner.updates == 0


def test_empty_wrong_mask_runs_no_phase_and_changes_no_boundary(monkeypatch):
    brain, drive, memory = captured_brain()
    original, live_boundary = brain.brain, brain.brain.memory
    monkeypatch.setattr(brain.learner, "step", forbidden)
    assert brain.teach_saved(drive, np.array([0, 1, 1]), [False] * 3, memory) is None
    assert brain.brain is original and original.memory is live_boundary
    assert brain.learner.updates == 0


@pytest.mark.parametrize("setting,value", [("surprise", True), ("qualified", False)])
def test_saved_lesson_refuses_unsupported_phase_policies_before_mutation(setting, value):
    brain, drive, memory = captured_brain()
    original, live_boundary = brain.brain, brain.brain.memory
    brain.learner.config = replace(brain.learner.config, **{setting: value})
    with pytest.raises(ValueError, match="qualified phases"):
        brain.teach_saved(drive, np.array([0, 1, 1]), [0], memory)
    assert brain.brain is original and original.memory is live_boundary
