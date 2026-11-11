"""A nudge's anchor is an explicit fixed boundary, separate from teaching evidence."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_anchor_adds_without_target_division_or_row_gain_scaling():
    s = np.array([[0.2, 0.7], [0.5, -0.1]])
    y = np.array([[0.8, 0.1], [0.3, 0.9]])
    anchor = np.array([[0.1, 0.4], [0.9, 0.2]])
    gain = np.array([0.3, 0.6])
    weight = np.array([1.0, 0.0])
    nudge = cd.Nudge(y, np.ones(2), -0.3, weight=weight, anchor=anchor, anchor_gain=gain)
    expected = -0.3 * weight[:, None] * (y - s) + gain * (anchor - s)
    np.testing.assert_allclose(nudge.drive(s), expected, rtol=0, atol=1e-16)
    # First coordinate's stiffness exactly cancels, leaving a finite constant force.
    changed = s.copy()
    changed[0, 0] += 10
    assert nudge.drive(changed)[0, 0] == pytest.approx(expected[0, 0])
    # A zero target gain must not remove the second stream's temporal boundary.
    np.testing.assert_allclose(nudge.drive(s)[1], gain * (anchor[1] - s[1]))


@pytest.mark.parametrize("softmax", [False, True])
def test_anchor_settling_and_residual_match_independent_equations_on_backends(softmax):
    graph = cd.Connectome.from_synapses(
        3, pre=[0, 1, 1, 2], post=[1, 0, 2, 1], sign=[0.2, 0.2, -0.1, -0.1]
    )
    model = cd.learning_neuron_model(dt=0.2)
    drive = np.array([[0.6, 0.1, -0.1], [0.2, 0.8, 0.3]])
    y = np.array([[0.3, 0.7, 0.1], [0.5, 0.2, 0.8]])
    anchor = np.array([[0.8, -0.03, 0.2], [0.1, 0.3, 0.5]])
    gain, mask = np.array([0.3, 0.6, 0]), np.array([1.0, 1.0, 0])
    weight, beta = np.array([1.0, 0.0]), -0.3
    nudge = cd.Nudge(
        y,
        mask,
        beta,
        weight=weight,
        softmax_temperature=0.4 if softmax else None,
        anchor=anchor,
        anchor_gain=gain,
    )
    weights = np.zeros((3, 3))
    weights[graph.pre, graph.post] = graph.sign

    def force(v):
        s = model.activation(v)
        if softmax:
            probabilities = np.exp((s[:, :2] - s[:, :2].max(axis=1, keepdims=True)) / 0.4)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            teacher = np.zeros_like(s)
            teacher[:, :2] = beta * weight[:, None] * (y[:, :2] - probabilities)
        else:
            teacher = beta * weight[:, None] * mask * (y - s)
        return s @ weights + drive - v + teacher + gain * (anchor - s)

    expected = np.zeros_like(drive)
    for _ in range(23):
        expected += model.dt * force(expected)
    for backend in cd.available_backends():
        brain = cd.NeuralGraph(
            graph, model, backend=backend, device="cpu" if backend == "torch" else None
        )
        phase = brain.settle_batch(drive, steps=23, nudge=nudge)
        tol = 2e-6 if backend == "mlx" else 1e-12
        np.testing.assert_allclose(phase.v, expected, atol=tol, rtol=tol, err_msg=backend)
        actual = brain.residual(drive, phase, nudge=nudge)
        independent = np.max(np.abs(force(phase.v)), axis=1)
        np.testing.assert_allclose(actual, independent, atol=tol, rtol=tol, err_msg=backend)
        if backend == "cpu":
            # Capturing a trajectory selects the NumPy, rather than fused, step loop.
            reference = brain.settle_batch(drive, steps=23, nudge=nudge, trajectory=True)
            np.testing.assert_allclose(reference.v, expected, atol=1e-12, rtol=1e-12)


def test_fixed_boundary_contrast_matches_finite_difference_bias_gradient():
    graph = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], sign=[0.2, 0.2])
    model = cd.learning_neuron_model(leak=1.0, dt=0.2)
    brain = cd.NeuralGraph(graph, model)
    x, y = np.array([[0.7, 0.4]]), np.array([[0.0, 0.8]])
    anchor, gain = np.array([[0.6, 0.2]]), np.array([1.0, 1.0])
    mask = np.array([0.0, 1.0])
    beta = 1e-3

    def solve(owner, sign):
        return owner.equilibrate(
            x,
            budget=2000,
            chunk=16,
            tolerance=1e-13,
            nudge=cd.Nudge(y, mask, sign * beta, anchor=anchor, anchor_gain=gain),
        )

    plus, minus = solve(brain, 1), solve(brain, -1)
    assert np.all(plus.converged) and np.all(minus.converged)
    local_descent = (plus.state.activation - minus.state.activation)[0, 0] / (2 * beta)
    eps = 1e-5
    losses = []
    for sign in (1, -1):
        state = solve(brain.with_parameters(bias=np.array([sign * eps, 0.0])), 0)
        losses.append(0.5 * (state.state.activation[0, 1] - y[0, 1]) ** 2)
    numerical_descent = -(losses[0] - losses[1]) / (2 * eps)
    assert local_descent == pytest.approx(numerical_descent, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor": np.zeros(2)},
        {"anchor_gain": np.ones(2)},
        {"anchor": np.zeros(2), "anchor_gain": np.array([-1.0, 1.0])},
        {"anchor": np.array([np.nan, 0.0]), "anchor_gain": np.ones(2)},
    ],
)
def test_invalid_temporal_boundaries_are_rejected(kwargs):
    with pytest.raises(ValueError):
        cd.Nudge(np.zeros(2), np.ones(2), 0.1, **kwargs)
