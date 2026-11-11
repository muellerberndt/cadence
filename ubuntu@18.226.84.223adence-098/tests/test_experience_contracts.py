"""A continuing life must preserve evidence and act using its current learned state."""

import json

import numpy as np
import pytest
from checkpoints import flat, metadata, write

import cadence as cd


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_next_decision_refreshes_activity_after_a_reward_update(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    brain = cd.Brain.compose(
        3,
        2,
        modules=(8,),
        episodic=False,
        backend=backend,
        device="cpu" if backend == "torch" else None, recall=False,
    )
    actor = brain.basal_ganglia
    drive = brain.stimulus(np.eye(3))
    actor.act(drive)
    actor.learn(np.ones(3), np.zeros(3, bool), drive)
    bootstrap_state = actor.state
    expected = brain.learner.free(drive, warm=bootstrap_state)
    actor.act(drive, greedy=True)
    assert actor.state is not bootstrap_state
    np.testing.assert_allclose(actor.state.activation, expected.activation, atol=1e-12)


def test_actor_cache_detects_an_update_from_another_learning_head():
    brain = cd.Brain.compose(2, 2, modules=(6,), episodic=False, recall=False)
    actor = brain.basal_ganglia
    drive = brain.stimulus(np.eye(2))
    actor.act(drive)
    cached = actor.state
    brain.learner.apply(np.zeros(brain.connectome.synapses), np.ones(brain.connectome.n) * 0.5)
    expected = brain.learner.free(drive, warm=cached)
    actor.act(drive, greedy=True)
    assert actor.state is not cached
    np.testing.assert_array_equal(actor.state.activation, expected.activation)


def test_resume_between_learning_and_next_action_refreshes_the_same_way(tmp_path):
    brain = cd.Brain.compose(2, 2, modules=(8,), recall=False)
    brain.act(np.eye(2))
    brain.learn(np.ones(2), np.zeros(2, bool), np.eye(2))
    restored = cd.Brain.load(brain.save(tmp_path / "life"))
    np.testing.assert_array_equal(brain.act(np.eye(2)), restored.act(np.eye(2)))
    np.testing.assert_array_equal(
        brain.basal_ganglia.state.activation, restored.basal_ganglia.state.activation
    )


@pytest.mark.parametrize("memory_class", [cd.FastSynapses, cd.SynapticMemory])
def test_hypothetical_query_batch_cannot_erase_live_memories(memory_class):
    memory = memory_class(np.arange(2), np.arange(2, 4))
    key = np.eye(2)
    memory.observe(key, key[:, ::-1])
    strength, mass, writes = memory.strength.copy(), memory.mass.copy(), memory.writes
    original = memory.recall(key)
    hypothetical = memory.recall(key[:1])
    np.testing.assert_array_equal(memory.strength, strength)
    np.testing.assert_array_equal(memory.mass, mass)
    assert memory.writes == writes
    np.testing.assert_array_equal(memory.recall(key), original)
    expected = (
        key[:1] @ memory.consolidated if isinstance(memory, cd.SynapticMemory) else np.zeros((1, 2))
    )
    np.testing.assert_array_equal(hypothetical, expected)


def test_hypothetical_stimulus_cannot_erase_carried_context():
    brain = cd.Brain.compose(2, 2, modules=(8,), recall=False)
    brain.act(np.eye(2))
    memory = brain.working_memory
    before = [memory.trace.copy(), memory.last.copy(), memory.cold.copy()]
    brain.stimulus(np.ones((3, 2)))
    for current, saved in zip((memory.trace, memory.last, memory.cold), before, strict=True):
        np.testing.assert_array_equal(current, saved)


@pytest.mark.parametrize("array_name", ["free/v", "pending/value", "working/trace"])
def test_invalid_lifetime_arrays_are_rejected_at_load(tmp_path, array_name):
    brain = cd.Brain.compose(2, 2, modules=(6,), recall=False)
    brain.step(np.eye(2))
    path = brain.save(tmp_path / "life")
    with np.load(path, allow_pickle=False) as saved:
        data = flat({name: saved[name].copy() for name in saved.files})
    data[array_name] = np.full_like(data[array_name], np.nan, dtype=float)
    write(path, data)
    with pytest.raises(ValueError, match="saved|checkpoint"):
        cd.Brain.load(path)


@pytest.mark.parametrize("separated", [False, True])
def test_overflowing_write_preserves_live_memory_and_separator(separated):
    separator = cd.PatternSeparator(2, 4, 4, center=0.5) if separated else None
    if separator is not None:
        separator.projection[:] = 0.5
    memory = cd.FastSynapses(np.arange(2), np.arange(2, 4), decay=0.5, separator=separator)
    memory.observe(np.eye(2), np.eye(2))
    before = memory.strength.copy(), memory.mass.copy(), memory.writes
    mean = None if separator is None else separator.mean.copy()
    with pytest.raises(ValueError, match="overflow"):
        memory.observe(np.full((1, 2), 1e200), np.full((1, 2), 1e200))
    np.testing.assert_array_equal(memory.strength, before[0])
    np.testing.assert_array_equal(memory.mass, before[1])
    assert memory.writes == before[2]
    if separator is not None:
        np.testing.assert_array_equal(separator.mean, mean)


def test_separator_rejects_overflow_without_changing_its_address_frame():
    separator = cd.PatternSeparator(2, 4, 2, center=0.5)
    separator.projection[:] = 1e308
    before = separator.mean.copy()
    with pytest.raises(ValueError, match="overflow"):
        separator.code(np.full((1, 2), 100.0), learn=True)
    np.testing.assert_array_equal(separator.mean, before)


def test_separator_can_average_large_representable_keys():
    separator = cd.PatternSeparator(2, 4, 2, center=0.5)
    separator.projection[:] = 0.25
    key = np.full((4, 2), 1e308)
    separator.habituate(key)
    np.testing.assert_array_equal(separator.mean, key[0])
    assert np.isfinite(separator.code(key, learn=True)).all()


@pytest.mark.parametrize(
    "corruption",
    [
        "free-shape",
        "value-shape",
        "action-range",
        "cold-dtype",
        "negative-updates",
        "fractional-steps",
        "critic-shape",
        "missing-metadata",
    ],
)
def test_invalid_continuation_structure_is_rejected(tmp_path, corruption):
    brain = cd.Brain.compose(2, 2, modules=(6,), recall=False)
    brain.step(np.eye(2))
    path = brain.save(tmp_path / "life")
    with np.load(path, allow_pickle=False) as saved:
        data = flat({name: saved[name].copy() for name in saved.files})
    meta = metadata(data)
    if corruption == "free-shape":
        data["free/activation"] = data["free/activation"][:1]
    elif corruption == "value-shape":
        data["pending/value"] = np.zeros((2, 1))
    elif corruption == "action-range":
        data["moment/action"][:] = 2
    elif corruption == "cold-dtype":
        data["working/cold"] = data["working/cold"].astype(float)
    elif corruption == "negative-updates":
        meta["updates"] = -1
    elif corruption == "fractional-steps":
        meta["pending_plus_steps"] = 0.5
    elif corruption == "critic-shape":
        meta["b_critic"] = [0.0]
    elif corruption == "missing-metadata":
        del meta["free_steps"]
    data["generic"] = np.array(json.dumps(meta))
    write(path, data)
    with pytest.raises(ValueError, match="saved|checkpoint"):
        cd.Brain.load(path)
