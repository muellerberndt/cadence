"""Check fixed-point equations and the hypotheses behind contrast gradients."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def isolated(neuron_model: cd.NeuronModel) -> cd.NeuralGraph:
    return cd.NeuralGraph(cd.Connectome.from_synapses(1, pre=[], post=[]), neuron_model)


def test_activation_tolerance_can_stop_far_from_equilibrium() -> None:
    brain = isolated(cd.learning_neuron_model(dt=0.1))
    drive = np.array([100.0])
    stopped = brain.settle(drive, steps=300, tolerance=1e-8)
    assert stopped.steps < 10  # the activation saturates while potential still moves
    assert brain.residual(drive, stopped)[0] > 60
    settled = brain.settle(drive, steps=300)
    assert brain.residual(drive, settled)[0] < 1e-10
    slower = isolated(cd.learning_neuron_model(dt=0.001))
    np.testing.assert_allclose(slower.residual(drive, stopped), brain.residual(drive, stopped))


def test_residual_includes_slow_adaptation_equation() -> None:
    neuron_model = cd.learning_neuron_model().replace(
        adaptation=cd.Adaptation(tau_steps=1e6, strength=0.5)
    )
    brain = isolated(neuron_model)
    state = cd.BrainState(np.ones(1), neuron_model.activation(np.ones(1)), np.zeros(1), 0)
    # The potential equation already holds, but the adaptation is far from its limit.
    assert brain.residual(np.ones(1), state)[0] > 0.4
    a = state.activation.copy()
    equilibrium = cd.BrainState(state.v, state.activation, a, 0)
    assert brain.residual(np.ones(1) + 0.5 * a, equilibrium)[0] < 1e-12


def test_rule_replace_preserves_adaptation_object() -> None:
    adaptation = cd.Adaptation(tau_steps=20, strength=0.3)
    original = cd.learning_neuron_model().replace(adaptation=adaptation)
    changed = original.replace(gain=0.7)
    assert changed.adaptation is adaptation
    assert changed.gain == 0.7 and original.gain == 1.0
    state = isolated(changed).settle(np.ones(1), steps=10)
    assert np.isfinite(state.adaptation).all()


def test_receipt_success_message_names_only_checks_performed(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("original")
    sources = [("source.txt", source)]
    path = tmp_path / "receipt.json"
    cd.Receipt.build("test", {"answer": 42}, sources).write(path)
    ok, message = cd.Receipt.verify(path)
    assert ok and message == "canonical form, digest agree"
    ok, message = cd.Receipt.verify(path, sources=sources)
    assert ok and "sources" in message and "arithmetic" not in message
    ok, message = cd.Receipt.verify(path, check=lambda body: None)
    assert ok and "arithmetic" in message and "sources" not in message
    source.write_text("changed")
    assert not cd.Receipt.verify(path, sources=sources)[0]


@pytest.mark.parametrize("dense_limit", [1, 100])
@pytest.mark.parametrize("row_masks", [False, True])
def test_residual_matches_the_projected_neuron_equation_without_mutation(
    dense_limit: int, row_masks: bool
) -> None:
    connectome = cd.Connectome.from_synapses(
        3, pre=[0, 1, 2], post=[1, 2, 0], count=[2, 3, 4], sign=[0.2, -0.1, 0.3]
    )
    neuron_model = cd.learning_neuron_model(gain=0.7, dt=0.3).replace(
        adaptation=cd.Adaptation(tau_steps=8, strength=0.2)
    )
    brain = cd.NeuralGraph(
        connectome,
        neuron_model,
        log_gain=np.log([1, 2, 3]),
        bias=np.array([0.1, 0.2, 0.3]),
        dense_limit=dense_limit,
    )
    keep = np.array([1.0, 0.5, 0.0])
    if row_masks:
        keep = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3]])
    v = np.array([[0.3, 0.8, 0.0], [0.5, 0.2, 0.4]])
    a = np.array([[0.2, 0.1, 0.0], [0.1, 0.1, 0.2]])
    state = cd.BrainState(v.copy(), neuron_model.activation(v) * keep, a.copy(), 0)
    drive = np.array([[0.2, 0.1, 0.4], [0.1, 0.3, 0.2]])
    nudge = cd.Nudge(
        np.array([[0.2, 0.8, 0], [0.7, 0.3, 0]]),
        np.array([1.0, 1.0, 0]),
        0.2,
        softmax_temperature=0.3,
        weight=np.array([1.0, -0.5]),
    )
    following = brain.settle_batch(
        drive, steps=1, state=state, mask=keep, nudge=nudge, trajectory=True
    )
    expected = np.maximum(
        np.abs((following.v - v) / neuron_model.dt).max(axis=1),
        np.abs(neuron_model.activation(v) * keep - a).max(axis=1),
    )
    np.testing.assert_allclose(
        brain.residual(drive, state, nudge=nudge, mask=keep), expected, atol=1e-12
    )
    np.testing.assert_array_equal(state.v, v)
    np.testing.assert_array_equal(state.adaptation, a)
    assert brain.residual(drive, state).shape == (2,)


def test_ablation_requires_zero_potential_and_adaptation() -> None:
    neuron_model = cd.learning_neuron_model().replace(adaptation=cd.Adaptation())
    brain = isolated(neuron_model)
    state = cd.BrainState(np.array([1.0]), np.zeros(1), np.array([0.4]), 0)
    assert brain.residual(np.array([100.0]), state, mask=np.zeros(1))[0] >= 1.0
    dead = cd.BrainState(np.zeros(1), np.zeros(1), np.zeros(1), 0)
    assert brain.residual(np.array([100.0]), dead, mask=np.zeros(1))[0] == 0.0


def test_residual_does_not_claim_a_unique_equilibrium() -> None:
    connectome = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], sign=[6, 6])
    neuron_model = cd.NeuronModel(gain=1.0, dt=0.5)
    brain = cd.NeuralGraph(connectome, neuron_model)
    drive = np.zeros(2)
    cold = brain.settle(drive, steps=200)
    initial = cd.BrainState(
        np.full(2, 6.0), neuron_model.activation(np.full(2, 6.0)), np.zeros(2), 0
    )
    warm = brain.settle(drive, steps=200, state=initial)
    assert brain.residual(drive, cold)[0] < 1e-12
    assert brain.residual(drive, warm)[0] < 1e-12
    assert np.max(np.abs(cold.activation - warm.activation)) > 0.99


def test_residual_rejects_wrong_shapes_and_does_not_certify_nan() -> None:
    brain = isolated(cd.learning_neuron_model())
    state = brain.settle(np.zeros(1), steps=1)
    with pytest.raises(ValueError, match="column"):
        brain.residual(np.zeros(2), state)
    with pytest.raises(ValueError, match="batch"):
        brain.residual(np.zeros((2, 1)), state)
    with pytest.raises(ValueError, match="mask"):
        brain.residual(np.zeros(1), state, mask=np.zeros(2))
    invalid = cd.BrainState(np.array([np.nan]), np.array([np.nan]), np.zeros(1), 0)
    assert np.isinf(brain.residual(np.zeros(1), invalid)[0])


@pytest.mark.parametrize("nudge", ["quadratic", "cross_entropy"])
def test_raw_contrast_needs_parameter_and_loss_units_for_exact_gradient(nudge: str) -> None:
    """Keep the optimizer convention; verify the explicit mathematical conversion.

    Effective weights are symmetric, phases converge, adaptation is absent, and
    all neurons remain on a smooth activation branch. Each perturbed parameter
    moves both directions of one physical synapse, so one contrast enters.
    """
    connectome = cd.Connectome.from_synapses(
        3,
        pre=[0, 1, 0, 2, 1, 2],
        post=[1, 0, 2, 0, 2, 1],
        count=[1, 1, 2, 2, 3, 3],
        sign=[0.08] * 6,
    )
    brain = cd.NeuralGraph(
        connectome,
        cd.learning_neuron_model(gain=1.4),
        log_gain=np.full(3, np.log(1.3)),
        bias=np.array([0.3, 0.2, 0.1]),
    )
    config = cd.LearnerConfig(
        beta=1e-4, nudge=nudge, temperature=0.3, free_steps=500, nudged_steps=500, tolerance=1e-13
    )
    learner = cd.Learner(brain, [1, 2], config)
    drive = np.array([[0.3, 0.2, 0.4], [0.2, 0.1, 0.3]])
    target = np.array([[0, 0.3, 0.7], [0, 0.6, 0.4]])
    free = learner.free(drive)
    plus = learner.nudged(drive, free, target)
    minus = learner.nudged(drive, free, target, sign=-1.0)
    contrast, _ = learner.contrast(free, plus, minus)
    assert brain.residual(drive, free).max() < 1e-11
    assert brain.residual(drive, plus, nudge=learner.nudge_for(target, config.beta)).max() < 1e-11
    assert brain.residual(drive, minus, nudge=learner.nudge_for(target, -config.beta)).max() < 1e-11

    def loss(candidate: cd.NeuralGraph) -> float:
        s = candidate.settle_batch(drive, steps=500, tolerance=1e-13).activation[:, [1, 2]]
        if nudge == "quadratic":
            return float(0.5 * np.square(s - target[:, [1, 2]]).sum(axis=1).mean())
        z = s / config.temperature
        z -= z.max(axis=1, keepdims=True)
        log_prob = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-(log_prob * target[:, [1, 2]]).sum(axis=1).mean())

    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair = ((connectome.pre == i) & (connectome.post == j)) | (
            (connectome.pre == j) & (connectome.post == i)
        )
        edge = np.flatnonzero(pair)[0]
        up, down = brain.efficacy.copy(), brain.efficacy.copy()
        up[pair] += 1e-5
        down[pair] -= 1e-5
        negative_gradient = (
            -(loss(brain.with_parameters(efficacy=up)) - loss(brain.with_parameters(efficacy=down)))
            / 2e-5
        )
        parameter_units = (
            brain.neuron_model.gain
            * connectome.count[edge]
            * np.exp(brain.log_gain[connectome.pre[edge]])
        )
        loss_units = config.temperature if nudge == "cross_entropy" else 1.0
        np.testing.assert_allclose(
            parameter_units * contrast[edge] / loss_units, negative_gradient, rtol=1e-5, atol=3e-9
        )
        assert not np.isclose(contrast[edge], negative_gradient, rtol=0.01, atol=1e-5)
