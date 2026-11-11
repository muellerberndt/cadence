"""Public inputs must not silently change labels, batch meaning, or stopping behavior."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize("slots", [2, (3, 2)])
def test_accuracy_counts_choices_across_rows_and_slots(slots: int | tuple[int, ...]) -> None:
    w = cd.layered(2, 3, 4 if isinstance(slots, int) else sum(slots), seed=7)
    learner = cd.Learner(
        cd.NeuralGraph(w, cd.learning_neuron_model()), w.populations["output"], slots=slots
    )
    drive = np.zeros((3, w.n))
    labels = learner.predict(drive)
    assert learner.accuracy(drive, labels, batch=2) == 1.0
    labels[1, 0] = (labels[1, 0] + 1) % learner.slot_sizes[0]
    assert learner.accuracy(drive, labels, batch=2) == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "labels", [np.array([-1]), np.array([2]), np.array([0.5]), np.array([[0]])]
)
def test_bad_labels_fail_before_learning(labels: np.ndarray) -> None:
    w = cd.layered(2, 3, 2, seed=1)
    learner = cd.Learner(cd.NeuralGraph(w, cd.learning_neuron_model()), w.populations["output"])
    before = learner.brain.efficacy.copy()
    with pytest.raises(ValueError, match="label"):
        learner.step(np.zeros((1, w.n)), labels)
    np.testing.assert_array_equal(learner.brain.efficacy, before)
    assert learner.updates == 0


def test_one_label_cannot_silently_train_an_entire_batch() -> None:
    w = cd.layered(2, 3, 2, seed=1)
    learner = cd.Learner(cd.NeuralGraph(w, cd.learning_neuron_model()), w.populations["output"])
    for operation in (learner.step, learner.accuracy):
        with pytest.raises(ValueError, match="same batch size"):
            operation(np.zeros((3, w.n)), np.array([0]))
    with pytest.raises(ValueError, match="at least one"):
        learner.accuracy(np.empty((0, w.n)), np.array([], dtype=int))
    with pytest.raises(ValueError, match="batch"):
        learner.accuracy(np.zeros((1, w.n)), np.array([0]), batch=0)


@pytest.mark.parametrize("outputs", [[], [-1], [99], [1, 1], [1.5]])
def test_output_ports_cannot_alias_or_duplicate_an_neuron(outputs: list[int | float]) -> None:
    w = cd.layered(2, 3, 2, seed=1)
    with pytest.raises(ValueError, match="output"):
        cd.Learner(cd.NeuralGraph(w, cd.learning_neuron_model()), outputs)  # type: ignore[arg-type]


def test_single_softmax_nudge_matches_analytic_pull_and_batched_form() -> None:
    nudge = cd.Nudge(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
        0.2,
        softmax_temperature=0.5,
        weight=np.array([2.0]),
    )
    single = nudge.drive(np.zeros(3))
    np.testing.assert_allclose(single, [0.0, 0.2, -0.2])
    np.testing.assert_array_equal(single, nudge.drive(np.zeros((1, 3)))[0])


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_zero_steps_and_inactive_nudges_agree_across_backends(backend: str) -> None:
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    w = cd.layered(2, 3, 2, seed=5)
    kw = {"device": "cpu"} if backend == "torch" else {}
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(), backend=backend, **kw)  # type: ignore[arg-type]
    drive = np.full((2, w.n), 0.1)
    warm = brain.settle_batch(drive, steps=3)
    unchanged = brain.settle_batch(drive, state=warm, steps=0, trajectory=True)
    assert unchanged.trajectory is not None and unchanged.trajectory.shape == (0, 2, w.n)
    assert unchanged.steps == 0
    np.testing.assert_array_equal(unchanged.v, warm.v)
    np.testing.assert_array_equal(unchanged.activation, warm.activation)

    expected = brain.settle_batch(drive, steps=3).activation
    for groups in (None, np.arange(w.n)):
        nudge = cd.Nudge(np.ones(w.n), np.zeros(w.n), 0.2, softmax_temperature=0.5, groups=groups)
        np.testing.assert_array_equal(nudge.drive(np.zeros(w.n)), np.zeros(w.n))
        actual = brain.settle_batch(drive, steps=3, nudge=nudge).activation
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    bad = dataclasses.replace(warm, adaptation=np.zeros((1, w.n)), device=None)
    with pytest.raises(ValueError, match="state batch"):
        brain.settle_batch(drive, state=bad)


@pytest.mark.parametrize("steps", [-1, 1.5])
def test_invalid_step_count_cannot_report_a_successful_no_op(steps: int | float) -> None:
    w = cd.layered(2, 3, 2, seed=1)
    with pytest.raises(ValueError, match="steps"):
        cd.NeuralGraph(w, cd.learning_neuron_model()).settle(steps=steps)  # type: ignore[arg-type]


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_invalid_temperature_cannot_poison_parameters(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        cd.LearnerConfig(temperature=temperature)
    with pytest.raises(ValueError, match="temperature"):
        cd.Nudge(np.ones(2), np.ones(2), 0.1, softmax_temperature=temperature)


@pytest.mark.parametrize("threshold", [-100.0, 1000.0])
def test_unrepresentable_resting_sigmoid_fails_before_backend_execution(threshold: float) -> None:
    # Re-basing divides by rest and 1-rest. Rounded endpoints made every backend
    # divide by zero even though the supplied slope and threshold were finite.
    with pytest.raises(ValueError, match="resting sigmoid"):
        cd.NeuronModel(slope=1.0, threshold=threshold)
