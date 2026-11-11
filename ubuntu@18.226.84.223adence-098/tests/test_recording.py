"""Record actual repairs, including nudged phases, without an invented replay."""

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_recording_captures_every_state_and_preserves_the_result(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    graph = cd.layered(2, 3, 2, seed=4)
    model = cd.learning_neuron_model().replace(adaptation=cd.Adaptation(tau_steps=10, strength=0.2))
    brain = cd.NeuralGraph(graph, model, backend=backend, device="cpu", precision="float64")
    drive = np.zeros((2, graph.n))
    drive[:, :2] = [[1, 0.5], [0.2, 0.8]]
    warm = brain.settle_batch(drive, steps=3)
    mask = np.ones(graph.n)
    mask[2] = 0
    nudge = cd.Nudge(np.ones_like(drive), np.ones(graph.n), 0.1)
    expected = brain.settle_batch(drive, state=warm, steps=7, mask=mask, nudge=nudge)
    records = []
    with cd.record_settlements(records.append, label="imagining"):
        actual = brain.settle_batch(
            drive, state=warm, steps=7, mask=mask, nudge=nudge, trajectory=True
        )
    (record,) = records
    assert record.label == "imagining"
    assert record.steps == actual.steps == 7
    assert record.potential.shape == record.activation.shape == (8, 2, graph.n)
    assert np.array_equal(record.potential[0], warm.v)
    assert np.allclose(actual.v, expected.v)
    assert np.allclose(actual.adaptation, expected.adaptation)
    assert np.array_equal(record.activation[1:], actual.trajectory)
    assert np.array_equal(record.potential[-1], actual.v)
    assert np.array_equal(record.adaptation[-1], actual.adaptation)
    assert np.any(np.diff(record.potential, axis=0))
    assert np.array_equal(record.weights, brain.weights)
    saved = record.drive.copy()
    drive[:] = 99
    assert np.array_equal(record.drive, saved)
    with pytest.raises(ValueError):
        record.activation[0] = 0


def test_zero_steps_nested_scopes_and_callback_diagnostics():
    brain = cd.NeuralGraph(cd.layered(1, 2, 1), cd.learning_neuron_model())
    outer, inner = [], []

    def observe(record):
        outer.append(record)
        brain.settle(steps=1)  # observer diagnostics do not recurse

    with cd.record_settlements(observe):
        brain.settle(steps=0)
        with cd.record_settlements(inner.append):
            brain.settle(steps=2)
        brain.settle(steps=3)
    brain.settle(steps=4)
    assert [r.steps for r in outer] == [0, 3]
    assert [r.steps for r in inner] == [2]
    assert outer[0].activation.shape[0] == 1


def test_callback_failure_restores_the_context():
    brain = cd.NeuralGraph(cd.layered(1, 2, 1), cd.learning_neuron_model())

    def fail(_record):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"), cd.record_settlements(fail):
        brain.settle(steps=1)
    brain.settle(steps=1)


def test_learning_records_free_and_nudged_phases():
    graph = cd.layered(2, 3, 2)
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model()),
        graph.populations["output"],
        config=cd.LearnerConfig(free_steps=4, nudged_steps=3),
    )
    records = []
    drive = np.zeros((1, learner.brain.connectome.n))
    drive[0, :2] = [1, 0.5]
    with cd.record_settlements(records.append):
        learner.step(drive, np.array([1]))
    assert len(records) >= 2
    assert records[0].nudge is None
    assert any(record.nudge is not None for record in records[1:])


@pytest.mark.parametrize("backend", ["cpu", "torch"])
@pytest.mark.parametrize("reward", [-2.0, 2.0])
def test_dopamine_report_preserves_the_sign(backend, reward):
    if backend == "torch":
        pytest.importorskip("torch")
    graph = cd.layered(1, 2, 1)
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model(), backend=backend, device="cpu"),
        graph.populations["output"],
    )
    actor = cd.ActorCritic(
        learner,
        graph.populations["hidden"],
        cd.ActorCriticConfig(eta=0, eta_bias=0, eta_critic=0),
    )
    actor.w_critic[:] = 0
    actor.b_critic = 0
    drive = np.zeros((1, graph.n))
    actor.act(drive)
    report = actor.learn(np.array([reward]), np.array([True]), drive)
    assert report["dopamine"] == np.sign(reward)
    assert report["delta"] == 1.0
