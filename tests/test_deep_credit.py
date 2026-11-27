"""Observable motor-only credit, and a bounded proposal before writing taught weights."""

import warnings
from dataclasses import replace

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


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_finite_many_slots_keep_small_pulls_for_the_declared_budget(backend):
    learner, drive = many_slots(backend)
    learner.config = replace(learner.config, qualified=False, nudged_steps=24, credit_window=2)
    labels = np.ones((1, 213), dtype=int)
    # Start from a precisely solved free state, so its unrelated motion
    # can't mask the premature stopping defect.
    free = learner.brain.equilibrate(drive, tolerance=1e-9).state
    pull = learner.nudge_for(learner.targets(labels), learner.config.beta)
    push = replace(pull, beta=-pull.beta)
    before = [learner.brain.settle_batch(drive, state=free, nudge=nudge, steps=24, tolerance=3e-3)
              for nudge in (pull, push)]
    assert [state.steps for state in before] == [1, 1]
    assert before[0].activation[0, 1] == before[1].activation[0, 1]

    _, report = learner.step(drive, labels, warm=free)
    assert report["nudged_steps"] == report["opposite_steps"] == 24
    assert report["credit_deep_contrast"] > 1e-4
    assert report["qualification_required"] == 0


@pytest.mark.parametrize("compiled", [False, True])
def test_finite_tiny_single_slot_pull_uses_the_same_budget(compiled):
    brain = cd.Brain.compose(
        8, 2, modules=(12, 8, 6), backend="cpu", seed=3, recall=False,
        learning=cd.LearnerConfig(
            beta=1e-3, tolerance=3e-3, compiled_moment=compiled, nudged_steps=20,
        ),
    )
    report = brain.lesson(np.eye(8)[:2], np.array([1, 0]))
    assert report["nudged_steps"] == report["opposite_steps"] == 20


@pytest.mark.parametrize("apical", [False, True])
def test_motor_only_acquisition_through_three_hidden_regions(apical):
    slots, size = 32, 69
    motor = np.arange(5, size)
    outgoing = np.tile([0.3, -0.3], slots)
    graph = cd.Connectome.from_synapses(
        size, pre=np.r_[0, 1, 2, 3, 3, 4, np.full(2 * slots, 4), motor],
        post=np.r_[2, 2, 3, 2, 4, 3, motor, np.full(2 * slots, 4)],
        sign=np.r_[0, 0, 0.7, 0.7, 0.7, 0.7, outgoing, np.tile([0.01, -0.01], slots)],
        populations={"sensory": [0, 1], "deep": [2], "middle": [3], "upper": [4], "motor": motor},
    )
    compartment = (cd.Apical(
        (graph.pre > graph.post) & (graph.post >= 2) & (graph.post <= 4),
        np.r_[-1, -1, 0, 1, 2, np.full(2 * slots, -1)], 0.25,
    ) if apical else None)
    drive = np.zeros((2, size))
    drive[0, 0] = drive[1, 1] = 1.0
    labels = np.stack([np.zeros(slots, dtype=int), np.ones(slots, dtype=int)])
    bias = np.r_[0, 0, 0.5, 0.5, 0.5, np.full(2 * slots, 0.6)]
    neural = cd.NeuralGraph(
        graph, cd.learning_neuron_model(), apical=compartment, bias=bias, backend="cpu",
    )
    # Set a fixed decision threshold from the common untrained state. No
    # target-driven change to the motor or any other downstream parameter.
    quiet = neural.with_parameters(efficacy=np.where(graph.pre < 5, neural.efficacy, 0.0))
    reference = quiet.equilibrate(drive, tolerance=1e-9).state.activation[0, 4]
    bias[motor] -= reference * outgoing
    neural = neural.with_parameters(bias=bias)
    config = cd.LearnerConfig(
        eta=0.04, eta_bias=0, normalize=0.99, normalize_floor=1e-6,
        nudged_steps=32, free_steps=128, tolerance=3e-3, credit_window=100,
    )
    plastic = graph.pre < 2

    class PreviousFiniteStop(cd.Learner):
        def _finite_nudge_tolerance(self, nudge):
            return self.config.tolerance

    for kind in (cd.Learner, PreviousFiniteStop):
        learner = kind(
            neural, motor, config, reciprocal=False, slots=(2,) * slots,
            plastic_synapses=plastic, plastic_neurons=np.zeros(size, dtype=bool),
        )
        for _ in range(8):
            warm = learner.brain.equilibrate(drive, tolerance=1e-9).state
            _, report = learner.step(drive, labels, warm=warm)
        final = learner.brain.equilibrate(drive, tolerance=1e-8)
        assert final.qualified.all()
        accuracy = (learner._answer(final.state) == labels).mean()
        if kind is PreviousFiniteStop:
            assert report["credit_deep_contrast"] == 0
            assert accuracy == 0.5
        else:
            assert report["credit_deep_max_return_hops"] == 3
            assert report["credit_deep_contrast"] > 0
            assert accuracy == 1.0
        np.testing.assert_array_equal(learner.brain.efficacy[~plastic], neural.efficacy[~plastic])
        np.testing.assert_array_equal(learner.brain.bias, bias)


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


def test_the_budget_refuses_a_layout_that_fell_back_to_one_block():
    """One population per motor neuron cuts the connectome into more blocks than the layout
    takes: it is one block over the whole matrix, the budget has no projection to give a share
    to, and the refused update changes no parameter and no optimizer history."""
    motor = np.arange(2, 428)
    populations = {"sensory": [0], "deep": [1], "motor": motor.tolist()}
    populations.update({f"motor/m{k}": [int(neuron)] for k, neuron in enumerate(motor)})
    connectome = cd.Connectome.from_synapses(
        428, pre=np.r_[0, np.full(len(motor), 1), motor],
        post=np.r_[1, motor, np.full(len(motor), 1)],
        sign=np.r_[0.5, np.tile([0.10, -0.08], 426)], populations=populations,
    )
    with pytest.warns(RuntimeWarning, match="into 853 blocks, more than max_pairs=256"):
        graph = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=0.5))
    assert graph.layout.fragments == 853 and graph.layout.pairs == 1
    budgeted = cd.LearnerConfig(
        free_steps=256, nudged_steps=32, tolerance=3e-3, momentum=0.9, drive_budget=0.35
    )
    learner = cd.Learner(graph, motor, budgeted, slots=(2,) * 213)
    drive = np.zeros((1, 428))
    drive[0, 0] = 0.6
    labels = np.zeros((1, 213), dtype=int)
    held = {
        "efficacy": learner.brain.efficacy.copy(), "bias": learner.brain.bias.copy(),
        "velocity": learner.velocity.copy(), "updates": learner.updates,
        "contrast_updates": learner.contrast_updates,
    }
    with pytest.raises(ValueError, match="one block over the whole matrix") as refused:
        learner.step(drive, labels)
    assert "cut it into 853 blocks" in str(refused.value)
    np.testing.assert_array_equal(learner.brain.efficacy, held["efficacy"])
    np.testing.assert_array_equal(learner.brain.bias, held["bias"])
    np.testing.assert_array_equal(learner.velocity, held["velocity"])
    assert (learner.updates, learner.contrast_updates) == (held["updates"], 0)
    # without the budget the same learner teaches on that layout, as before
    learner.config = replace(budgeted, drive_budget=0.0)
    _, report = learner.step(drive, labels)
    assert report["scale_step"] > 0 and "drive_limited_projections" not in report
    assert learner.contrast_updates == 1


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
            assert report["credit_silent_voltage_contrast"] == 0
            assert report["credit_silent_slope"] > 0
            assert report["credit_silent_return_share"] == 0
            assert report["credit_silent_max_return_hops"] == -1
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
    np.testing.assert_array_equal(brain.brain.bias, restored.brain.bias)


def test_a_zero_contact_does_not_make_a_return_path():
    graph = cd.Connectome.from_synapses(
        4, pre=[0, 1, 2], post=[1, 2, 1], count=[1., 1., 0.],
        populations={"sense": [0], "hidden": [1], "motor": [2, 3]},
    )
    learner = cd.Learner(cd.NeuralGraph(graph, cd.learning_neuron_model()), [2, 3],
                         cd.LearnerConfig(credit_window=20))
    _, report = learner.step(np.array([[1., 0., 0., 0.]]), np.array([1]))
    assert report["credit_hidden_return_share"] == 0
    assert report["credit_hidden_max_return_hops"] == -1
