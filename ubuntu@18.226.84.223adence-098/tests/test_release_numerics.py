"""Independent regressions from the 0.17 numerical contract review."""

import numpy as np
import pytest

import cadence as cd


def make_learner(backend="cpu", *, rate=1.0, eta=.1):
    if backend == "torch":
        pytest.importorskip("torch")
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1])
    return cd.Learner(
        cd.NeuralGraph(graph, cd.NeuronModel(), backend=backend, device="cpu"), [1],
        cd.LearnerConfig(eta=eta, eta_bias=0, momentum=.2), reciprocal=False,
        synapse_rate=np.array([rate]),
    )


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_scaled_overflow_cannot_be_hidden_by_parameter_clipping(backend):
    learner = make_learner(backend, rate=1e200)
    old_brain = learner.brain
    original = old_brain.efficacy.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="scaled and tied steps"):
            learner.apply(np.array([1e200]), np.zeros(2))
    assert learner.brain is old_brain and learner.updates == 0
    np.testing.assert_array_equal(learner.brain.efficacy, original)


def test_resident_scaled_overflow_restores_optimizer_history():
    learner = make_learner("torch", rate=1e200, eta=1e200)
    free = learner.brain.settle_batch(np.zeros((1, 2)), steps=2)
    nudged = learner.brain.settle_batch(np.ones((1, 2)), steps=2)
    original = learner.brain.efficacy.copy()
    before = {name: getattr(learner, name).copy() for name in
              ("velocity", "velocity_bias", "second_moment", "second_moment_bias")}
    with pytest.raises(ValueError, match="scaled and tied steps"):
        learner.update(free, nudged)
    assert learner.updates == learner.contrast_updates == 0
    np.testing.assert_array_equal(learner.brain.efficacy, original)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(learner, name), value)
    assert free.__dict__.get("activation") is None
    assert nudged.__dict__.get("activation") is None


@pytest.mark.parametrize("backend", ["cpu", "torch"])
@pytest.mark.parametrize("method", ["contrast", "contrast_rows", "update"])
@pytest.mark.parametrize("centered", [False, True])
def test_learning_rejects_broadcasting_different_phase_streams(backend, method, centered):
    learner = make_learner(backend)
    free = learner.brain.settle_batch(np.zeros((1, 2)), steps=1)
    nudged = learner.brain.settle_batch(np.ones((2, 2)), steps=1)
    opposite = learner.brain.settle_batch(-np.ones((2, 2)), steps=1) if centered else None
    original = learner.brain.efficacy.copy()
    with pytest.raises(ValueError, match="same nonempty .* shape"):
        getattr(learner, method)(free, nudged, opposite)
    assert learner.updates == learner.contrast_updates == 0
    np.testing.assert_array_equal(learner.brain.efficacy, original)
    if backend == "torch":
        assert free.__dict__.get("activation") is None
        assert nudged.__dict__.get("activation") is None


def test_empty_phase_rows_cannot_update():
    learner = make_learner()
    empty = cd.BrainState(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), 0)
    with pytest.raises(ValueError, match="same nonempty .* shape"):
        learner.update(empty, empty)
    assert learner.updates == 0
