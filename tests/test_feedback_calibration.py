"""Motor-only calibration changes a physical feedback gain, not learned efficacies."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def chain(backend="cpu", dt=0.5):
    pre = np.array([0, 1, 2, 2, 2, 3, 4, 0, 5])
    post = np.array([1, 2, 1, 3, 4, 2, 2, 5, 3])
    weights = [0.9, 0.6, 0.6, 0.5, -0.3, 0.5, -0.3, 0.4, 0.1]
    cn = cd.Connectome.from_synapses(
        6, pre=pre, post=post, sign=weights,
        populations={"sensory": [0], "deep": [1], "shallow": [2], "motor": [3, 4], "one_way": [5]},
    )
    rows = np.array([-1, 0, 1, -1, -1, -1])
    top = (cn.pre > cn.post) & (rows[cn.post] >= 0)
    apical = cd.Apical(top, rows, 0.1)
    graph = cd.NeuralGraph(
        cn, cd.learning_neuron_model(dt=dt), bias=np.array([0, 0.3, 0.4, 0.2, 0.2, 0.2]),
        apical=apical, backend=backend, device="cpu" if backend == "torch" else None,
    )
    x = np.zeros((3, 6))
    x[:, 0] = [0.6, 0.8, 1.0]
    return cd.Learner(graph, [3, 4]), x


def test_improves_weak_hop_without_hidden_targets_or_parameter_changes():
    learner, drive = chain()
    original = learner.brain
    trial = learner.calibrate_credit(drive, apply=False)
    assert trial["status"] == "adjusted" and not trial["applied"]
    assert trial["unreachable"] == ["one_way"]
    assert trial["selected"]["regions"]["deep"] > 1.3 * trial["baseline"]["regions"]["deep"]
    assert learner.brain is original

    report = learner.calibrate_credit(drive)
    assert report["applied"] and report["fractions"] == trial["fractions"]
    assert all(0.1 / 3 <= f <= 0.1 * 3 for f in report["fractions"])
    assert learner.brain.apical.pool_fractions == tuple(report["fractions"])
    assert report["residual_checks"] and report["sweeps"]
    np.testing.assert_array_equal(learner.brain.weights, original.weights)
    np.testing.assert_array_equal(learner.brain.bias, original.bias)
    assert learner.updates == learner.contrast_updates == 0


def test_unsettled_probe_does_not_modify_the_live_brain():
    learner, drive = chain(dt=0.01)
    original = learner.brain
    report = learner.calibrate_credit(drive, budget=16)
    assert report["status"] == "baseline_unsettled" and not report["applied"]
    assert learner.brain is original


@pytest.mark.parametrize("backend", ["cpu", "torch", "mlx"])
def test_distinct_pool_gains_affect_actual_equations_and_roundtrip(backend, tmp_path):
    if backend != "cpu":
        pytest.importorskip("mlx.core" if backend == "mlx" else backend)
    learner, drive = chain(backend)
    apical = learner.brain.apical
    assert apical is not None
    learner.brain = learner.brain._with_apical(replace(apical, pool_fractions=(0.15, 0.25)))
    state = learner.brain.equilibrate(drive, budget=128, tolerance=1e-5)
    assert state.qualified.all()
    reference, _ = chain()
    reference.brain = reference.brain._with_apical(replace(apical, pool_fractions=(0.15, 0.25)))
    expected = reference.brain.equilibrate(drive, budget=128, tolerance=1e-5)
    np.testing.assert_allclose(state.state.activation, expected.state.activation, atol=2e-6)
    path = tmp_path / "calibrated.npz"
    learner.save(path)
    restored = cd.Learner.load(path, backend="cpu")
    assert restored.brain.apical.pool_fractions == (0.15, 0.25)
    np.testing.assert_array_equal(restored.brain.weights, reference.brain.weights)


def test_composed_teaching_rechecks_at_the_interval(tmp_path):
    b = cd.Brain.compose(8, 6, modules=(12, 8, 6), slots=(3, 3), apical=0.25,
                         seed=1, backend="cpu", learning_credit_calibration=2,
                         learning_surprise=False)
    x, targets = np.eye(8)[:2], np.array([[1, 2], [2, 1]])
    first = b.lesson(x, targets)
    assert "credit_calibration_sweeps" not in first
    path = tmp_path / "before-recheck.npz"
    b.save(path)
    restored = cd.Brain.load(path, backend="cpu")
    second = b.lesson(x, targets)
    assert second == restored.lesson(x, targets)
    assert second["credit_calibration_sweeps"] > 0
    assert b.learner.last_credit_calibration["contrast_updates"] == 2
    assert b.learner.contrast_updates == 2
    assert restored.brain.apical.pool_fractions == b.brain.apical.pool_fractions
