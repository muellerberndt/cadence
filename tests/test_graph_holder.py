"""One holder of the current graph for several learners: ``Learner.sharing`` builds a learner
on another's ``GraphHolder``, so each one's updates reach the other."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import cadence as cd


def learners(**config: object) -> tuple[cd.Learner, cd.Learner, cd.Learner, np.ndarray]:
    """Two learners on one holder, each teaching two of the four outputs, and a third built
    on the same graph the old way; the drive they are taught on."""
    connectome = cd.layered(4, 8, 4, density=0.8, seed=1)
    outputs = list(connectome.populations["output"])
    settings = cd.LearnerConfig(**{"eta": 0.5, "tolerance": 1e-9, "free_steps": 2000,
                                   "nudged_steps": 2000, **config})
    first = cd.Learner(cd.NeuralGraph(connectome, cd.learning_neuron_model()), outputs[:2],
                       settings)
    second = cd.Learner.sharing(first, outputs[2:], config=settings)
    third = cd.Learner(first.brain, outputs[2:], settings)
    drive = np.zeros((2, connectome.n))
    drive[:, :4] = np.random.default_rng(2).random((2, 4))
    return first, second, third, drive


def test_two_learners_on_one_holder_see_each_others_updates():
    first, second, third, drive = learners()
    original = first.brain
    assert second.holder is first.holder and third.holder is not first.holder
    assert first.brain is second.brain is third.brain is original
    first.step(drive, np.array([0, 1]))
    assert second.brain is first.brain and first.brain is not original
    taught = first.brain.efficacy.copy()
    second.step(drive, np.array([1, 0]))  # teaches the graph the first update left
    assert first.brain is second.brain
    assert not np.array_equal(first.brain.efficacy, taught)
    assert (first.updates, second.updates) == (1, 1)
    # the third, built from the graph the old way, keeps its own: today's behavior
    assert third.brain is original and third.updates == 0
    third.step(drive, np.array([1, 0]))
    assert third.brain is not original and third.brain is not first.brain
    assert first.brain is second.brain


def test_a_graph_assigned_to_one_learner_is_every_sharing_learners_graph():
    first, second, third, _ = learners()
    replacement = first.brain.with_parameters(bias=first.brain.bias + 0.1)
    second.brain = replacement
    assert first.brain is replacement and first.holder.brain is replacement
    assert third.brain is not replacement


def test_a_refused_update_replaces_no_learners_graph():
    first, second, _, drive = learners()
    held = first.brain
    synapses, neurons = held.connectome.synapses, held.connectome.n
    with pytest.raises(ValueError, match="finite"):
        first.apply(np.full(synapses, np.nan), np.zeros(neurons))
    with pytest.raises(ValueError, match="finite"):
        second.apply(np.zeros(synapses), np.full(neurons, np.inf))
    assert first.brain is second.brain is held and (first.updates, second.updates) == (0, 0)
    refusing, sharing, _, _ = learners(qualified=True, free_steps=1, nudged_steps=1,
                                       tolerance=1e-12)
    before = refusing.brain
    with pytest.raises(cd.LearningPhaseError):
        refusing.step(drive, np.array([0, 1]))
    assert refusing.brain is sharing.brain is before and refusing.updates == 0


def test_a_saved_learner_loads_with_a_holder_of_its_own(tmp_path):
    first, second, _, drive = learners()
    first.step(drive, np.array([0, 1]))
    loaded = cd.Learner.load(first.save(tmp_path / "learner.npz"))
    assert loaded.holder is not first.holder
    np.testing.assert_array_equal(loaded.brain.efficacy, first.brain.efficacy)
    np.testing.assert_array_equal(loaded.brain.bias, first.brain.bias)
    again = cd.Learner.sharing(loaded, second.outputs, config=second.config)
    again.step(drive, np.array([1, 0]))
    assert loaded.brain is again.brain
    assert not np.array_equal(loaded.brain.efficacy, first.brain.efficacy)


def test_a_shallow_copy_shares_the_holder_and_a_rebuilt_learner_has_its_own():
    first, second, _, drive = learners()
    twin = copy.copy(first)
    rebuilt = dataclasses.replace(first)  # the constructor again, from the graph
    assert twin.holder is first.holder
    assert rebuilt.holder is not first.holder and rebuilt.brain is first.brain
    twin.step(drive, np.array([0, 1]))
    assert first.brain is twin.brain is second.brain and rebuilt.brain is not first.brain


def test_a_learner_sharing_a_composed_brains_holder_teaches_the_graph_the_brain_acts_with(
    tmp_path,
):
    brain = cd.Brain.compose(3, 2, seed=3, arousal=True)
    assert brain.learner.holder.brain is brain.brain
    readout = cd.Learner.sharing(brain.learner, list(brain.motor_index),
                                 config=brain.learner.config)
    x = np.array([[1.0, 0.0, 0.0]])
    brain.step(x)
    before = brain.brain
    readout.step(brain.stimulus(x, memory=False), np.array([1]))
    assert brain.brain is readout.brain and brain.brain is not before
    brain.step(x, reward=[1.0])  # the actor learns on the graph the readout left
    assert brain.brain is readout.brain
    resumed = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    np.testing.assert_array_equal(resumed.brain.efficacy, brain.brain.efficacy)
    np.testing.assert_array_equal(
        resumed.act(x, greedy=True), brain.act(x, greedy=True)
    )
