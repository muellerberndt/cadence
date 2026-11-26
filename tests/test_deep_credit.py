"""Observable motor-only credit, and a bounded proposal before writing taught weights."""

import warnings

import numpy as np
import pytest

import cadence as cd


def many_slots(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    motor = np.arange(2, 428)
    connectome = cd.Connectome.from_synapses(
        428, pre=np.r_[0, np.full(len(motor), 1), motor],
        post=np.r_[1, motor, np.full(len(motor), 1)],
        sign=np.r_[0.5, np.tile([0.10, -0.08], 426)],
        populations={"sensory": [0], "deep": [1], "motor": motor.tolist()},
    )
    graph = cd.NeuralGraph(
        connectome, cd.learning_neuron_model(dt=0.5), backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    config = cd.LearnerConfig(qualified=True, free_steps=256, nudged_steps=256, tolerance=3e-3)
    drive = np.zeros((1, 428))
    drive[0, 0] = 0.6
    return cd.Learner(graph, motor, config, slots=(2,) * 213), drive


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_qualified_many_slots_cannot_accept_unchanged_free_state(backend):
    learner, drive = many_slots(backend)
    target = learner.targets(np.ones((1, 213), dtype=int))
    free = learner.free(drive)
    pull = learner.nudge_for(target, learner.config.beta)
    # The released solver accepts the free state without a sweep.
    previous = learner.brain.equilibrate(drive, state=free, nudge=pull, tolerance=3e-3)
    assert previous.qualified.all() and previous.steps == 0
    before = learner.brain.efficacy.copy()
    states, report = learner.step(drive, np.ones((1, 213), dtype=int))
    assert report["qualified"] == 1
    assert min(report["nudged_steps"], report["opposite_steps"]) > 0
    assert states.nudged is not None and states.opposite is not None
    assert abs(states.nudged.activation[0, 1] - states.opposite.activation[0, 1]) > 1e-4
    assert learner.brain.efficacy[0] != before[0]


def test_opt_in_budget_limits_each_observed_projection_and_preserves_reciprocity():
    learner, drive = many_slots("cpu")
    learner.config = cd.LearnerConfig(drive_budget=0.35, credit_window=3)
    graph = learner.brain.connectome
    free = learner.free(drive)
    # Magnify the proposed feedback and sensory drive independently.
    step = np.r_[np.ones(1), np.tile([1.0, -1.0], 426)]
    report = learner._apply(step, np.zeros(graph.n), free)
    assert report["drive_limited_projections"] == 2
    weights = learner.brain.weights
    back, forward = (graph.post == 1) & (graph.pre >= 2), graph.post >= 2
    np.testing.assert_array_equal(weights[back], weights[forward])
    for source in (graph.pre == 0, graph.pre >= 2):
        incoming = (graph.post == 1) & source
        projected = free.activation[:, graph.pre[incoming]] @ weights[incoming]
        assert np.max(np.abs(projected)) <= 0.175 + 1e-14


def test_warns_once_per_streak_for_hidden_region_without_a_return_path():
    cn = cd.Connectome.from_synapses(
        5, pre=[0, 1, 1], post=[1, 3, 4], sign=[0.4, 0.3, -0.3],
        populations={"sense": [0], "silent": [1], "motor": [3, 4]},
    )
    learner = cd.Learner(cd.NeuralGraph(cn, cd.learning_neuron_model()), [3, 4],
                         cd.LearnerConfig(credit_window=2, nudged_fixed=True))
    drive = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            _, report = learner.step(drive, np.array([1]))
            assert report["credit_motor_contrast"] > 0
            assert report["credit_silent_contrast"] == 0
            assert report["credit_silent_slope"] > 0
    messages = [str(w.message) for w in caught if "motor teaching contrast" in str(w.message)]
    assert len(messages) == 1 and "silent" in messages[0]


def test_composed_settings_round_trip(tmp_path):
    brain = cd.Brain.compose(4, 6, modules=(8, 4), slots=(3, 3), seed=4, backend="cpu",
                             learning_drive_budget=4.0, learning_credit_window=2)
    path = tmp_path / "control.npz"
    brain.save(path)
    restored = cd.Brain.load(path, backend="cpu")
    assert restored.learner.config.drive_budget == 4.0
    assert restored.learner.config.credit_window == 2
    x, y = np.array([[0.0, 1.0, 0.0, 0.0]]), np.array([[1, 2]])
    assert brain.lesson(x, y) == restored.lesson(x, y)
    np.testing.assert_array_equal(brain.brain.weights, restored.brain.weights)
