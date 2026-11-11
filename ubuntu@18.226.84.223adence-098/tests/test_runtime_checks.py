"""Independent equation checks and false-green cases for resident residuals."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize("dense_limit", [1, 100])
@pytest.mark.parametrize("nudge_kind", [None, "quadratic", "slots"])
@pytest.mark.parametrize("adaptation", [False, True])
def test_resident_residual_matches_host_and_projected_equations(
    dense_limit: int, nudge_kind: str | None, adaptation: bool
) -> None:
    pytest.importorskip("torch")
    graph = cd.layered(3, 4, 4, density=1, seed=3)
    model = cd.learning_neuron_model(dt=0.3, gain=0.4)
    if adaptation:
        model = model.replace(adaptation=cd.Adaptation(tau_steps=1e6, strength=0.3))
    brain = cd.NeuralGraph(graph, model, backend="torch", device="cpu", dense_limit=dense_limit)
    rng = np.random.default_rng(6)
    drive = rng.uniform(0.1, 0.4, (2, graph.n))
    keep = rng.choice([0, 0.4, 1.0], (2, graph.n))
    nudge = None
    if nudge_kind:
        learner = cd.Learner(
            brain,
            graph.populations["output"],
            cd.LearnerConfig(nudge="quadratic" if nudge_kind == "quadratic" else "cross_entropy"),
            slots=2,
        )
        target = learner.targets(np.array([[0, 1], [1, 0]]))
        nudge = learner.nudge_for(target, -0.1, np.array([1.0, -0.4]))
    state = brain.settle_batch(drive, steps=7, mask=keep, nudge=nudge)
    assert state.__dict__["v"] is None
    actual = brain.residual(drive, state, mask=keep, nudge=nudge)
    assert state.__dict__["v"] is None and state.__dict__["adaptation"] is None
    np.testing.assert_allclose(
        actual, brain.residual(drive, state, mask=keep, nudge=nudge, on_device=False), atol=1e-14
    )
    # A fresh neuron step is an independent oracle for the projected potential equation.
    next_state = brain.settle_batch(drive, steps=1, state=state, mask=keep, nudge=nudge)
    expected = np.abs((next_state.v - state.v) / model.dt).max(axis=1)
    if adaptation:
        expected = np.maximum(
            expected, np.abs(model.activation(state.v) * keep - state.adaptation).max(axis=1)
        )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_residual_preserves_float64_reference_for_low_precision_and_edited_host_state() -> None:
    pytest.importorskip("torch")
    graph = cd.Connectome.from_synapses(1, pre=[], post=[])
    brain = cd.NeuralGraph(
        graph, cd.learning_neuron_model(), backend="torch", device="cpu", precision="float32"
    )
    state = brain.settle_batch(np.array([[1.00000001]]), steps=100)
    assert brain.residual(np.array([[1.00000001]]), state)[0] > 9e-9
    brain64 = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="torch", device="cpu")
    state64 = brain64.settle_batch(np.ones((1, 1)), steps=100)
    state64.v[:] = 0.0
    assert brain64.residual(np.ones((1, 1)), state64)[0] == 1.0


def test_resident_checks_do_not_fetch_state_or_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    graph = cd.layered(3, 4, 2, seed=2)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="torch", device="cpu")
    learner = cd.Learner(brain, graph.populations["output"])
    drive = np.ones((2, graph.n)) * 0.1
    learner.step(drive, np.array([0, 1]))  # parameters now live on the kernel
    brain = learner.brain
    state = brain.settle_batch(drive, steps=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected host materialization")

    monkeypatch.setattr(brain._torch, "host_scale", forbidden)
    monkeypatch.setattr(brain._torch, "host_bias", forbidden)
    state.device["fetch"] = forbidden
    actual = brain.residual(drive, state)
    assert np.isfinite(actual).all()
    assert brain._efficacy is None and brain._bias is None
    result = brain.equilibrate(drive, state=state, chunk=4, budget=128, tolerance=1e-7)
    assert result.converged.all()
    assert result.state.__dict__["v"] is None


def test_resident_residual_never_certifies_nonfinite_state() -> None:
    torch = pytest.importorskip("torch")
    graph = cd.Connectome.from_synapses(1, pre=[], post=[])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="torch", device="cpu")
    state = brain.settle_batch(np.zeros((1, 1)), steps=0)
    state.device["v"].fill_(torch.nan)
    assert np.isinf(brain.residual(np.zeros((1, 1)), state)[0])


@pytest.mark.parametrize("seed", range(5))
def test_residual_adaptive_work_follows_input_change_without_relaxing_tolerance(seed: int) -> None:
    graph = cd.layered(6, 10, 3, density=1, seed=seed)
    raw = cd.NeuralGraph(graph, cd.learning_neuron_model())
    brain = raw.with_parameters(efficacy=raw.efficacy / cd.row_mass(raw))
    rng = np.random.default_rng(seed)
    drive = np.zeros((4, graph.n))
    drive[:, :6] = rng.uniform(0.1, 0.9, (4, 6))
    old = brain.equilibrate(drive, tolerance=1e-12, chunk=1)
    unchanged = brain.equilibrate(drive, state=old.state, tolerance=1e-8, chunk=1)
    assert unchanged.steps == 0 and unchanged.converged.all()
    changed = drive.copy()
    changed[:, :6] += 1e-3
    warm = brain.equilibrate(changed, state=old.state, tolerance=1e-8, chunk=1)
    cold = brain.equilibrate(changed, tolerance=1e-8, chunk=1)
    assert warm.converged.all() and cold.converged.all() and warm.steps < cold.steps
    np.testing.assert_allclose(warm.state.v, cold.state.v, atol=5e-8)


def test_zero_rate_certificate_and_invalid_budgets() -> None:
    cert = cd.Certificate(row_mass=0.0, lipschitz=0.5, dt=1.0, adaptation=False)
    assert cert.certified and cert.rate == 0
    assert cert.steps_for(change=1, tolerance=1e-9) == 1
    assert cert.steps_for(change=1e-10, tolerance=1e-9) == 0
    ordinary = cd.Certificate(row_mass=1.0, lipschitz=0.5, dt=0.5, adaptation=False)
    assert ordinary.steps_for(change=1e300, tolerance=1e-300) > 4000
    for change, tolerance in [(0, 0), (-1, 1), (np.nan, 1), (1, np.inf)]:
        with pytest.raises(ValueError):
            cert.steps_for(change, tolerance)
    with pytest.raises(ValueError):
        cert.error_bound(-1)
    with pytest.raises(ValueError):
        cert.error_bound(np.inf)
    with pytest.raises(ValueError):
        cert.apriori_bound(1, -1)
    with pytest.raises(ValueError):
        cd.Certificate(row_mass=0.0, lipschitz=0.5, dt=3, adaptation=False)


def test_ep_structure_uses_effective_free_weights_and_rejects_hidden_feedback() -> None:
    graph = cd.layered(3, 4, 2, density=1, seed=3)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model())
    assert not cd.ep_structure(brain).compatible  # one-way input projections in the whole matrix
    structure = cd.ep_structure(brain, fixed_inputs=graph.populations["input"])
    assert structure.compatible and structure.free_asymmetry == 0
    unequal_gain = brain.with_parameters(log_gain=np.linspace(0, 0.5, graph.n))
    assert not cd.ep_structure(unequal_gain, fixed_inputs=graph.populations["input"]).compatible
    assert not cd.ep_structure(
        brain, fixed_inputs=[3]
    ).compatible  # naming a hidden neuron is not a clamp
    rhythmic = cd.NeuralGraph(graph, brain.neuron_model.replace(adaptation=cd.Adaptation()))
    assert not cd.ep_structure(rhythmic, fixed_inputs=graph.populations["input"]).compatible
