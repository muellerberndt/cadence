"""A learned reward-reversal boundary must not trap the default in a two-cycle."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from anatomy import single_region

from cadence import BrainState, NeuralGraph


def reversal_boundary(backend="cpu"):
    fixture = Path(__file__).parent / "fixtures" / "generic_reversal"
    provenance = json.loads((fixture / "provenance.json").read_text())
    data_path = fixture / "moment300.npz"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with np.load(data_path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}

    if backend in ("torch", "mps"):
        torch = pytest.importorskip("torch")
        if backend == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS hardware unavailable")
    # the fixture was recorded on the single-region layout of earlier releases
    agent = single_region(
        4,
        4,
        hidden=16,
        seed=0,
        backend="torch" if backend == "mps" else backend,
        device="mps" if backend == "mps" else "cpu" if backend == "torch" else None,
    )
    for name in ("pre", "post", "count", "sign"):
        np.testing.assert_array_equal(getattr(agent.connectome, name), data[name])
    agent.learner.brain = agent.brain.with_parameters(
        efficacy=data["efficacy"], log_gain=data["log_gain"], bias=data["bias"]
    )
    brain = agent.brain
    state = BrainState(
        v=data["v"],
        activation=brain.neuron_model.activation(data["v"]),
        adaptation=data["adaptation"],
        steps=0,
    )
    return agent, state, data["drive"]


@pytest.mark.parametrize("backend", ["cpu", "torch", "mps"])
def test_default_repairs_frozen_learned_boundary_that_cycles_without_damping(backend):
    agent, state, drive = reversal_boundary(backend)
    brain = agent.brain
    original_model = brain.neuron_model
    original_parameters = [x.copy() for x in (brain.efficacy, brain.bias, brain.log_gain)]
    original_state = state.v.copy(), state.adaptation.copy()
    undamped = NeuralGraph(
        brain.connectome,
        brain.neuron_model.replace(dt=1.0),
        efficacy=brain.efficacy,
        log_gain=brain.log_gain,
        bias=brain.bias,
    )
    failed = undamped.equilibrate(drive, state=state, budget=1024, tolerance=3e-3)
    assert not failed.qualified.any()
    assert failed.residual.min() > 1.0
    first = undamped.settle_batch(drive, state=failed.state, steps=1)
    second = undamped.settle_batch(drive, state=first, steps=1)
    np.testing.assert_allclose(second.v, failed.state.v, rtol=0, atol=1e-12)
    assert np.abs(first.v - failed.state.v).max() > 1.0

    # Use the actual Brain admission path, preserving its original tolerance.
    repaired = agent._qualified(drive, state)
    report = agent.last_settlement
    assert report["qualified"] and report["damping_halvings"] == 1
    assert 512 < report["steps"] <= report["budget"] == 1024
    np.testing.assert_array_equal(report["residual"], brain.residual(drive, repaired))
    assert brain.residual(drive, repaired).max() <= agent.learner.config.tolerance
    # Damping changes the numerical path, not the unmasked fixed-point equation.
    assert undamped.residual(drive, repaired).max() <= agent.learner.config.tolerance
    matrix = np.zeros((brain.connectome.n, brain.connectome.n))
    np.add.at(matrix, (brain.connectome.post, brain.connectome.pre), brain.weights)
    actual = brain.neuron_model.activation(repaired.v) @ matrix.T + drive + brain.bias - repaired.v
    assert np.abs(actual).max() <= agent.learner.config.tolerance
    assert agent.brain is brain and brain.neuron_model is original_model
    assert brain.neuron_model.dt == 1.0
    assert agent.learner.config.nudged_steps == 12
    assert agent.learner.config.tolerance == 3e-3
    for before, after in zip(
        original_parameters, (brain.efficacy, brain.bias, brain.log_gain), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(state.v, original_state[0])
    np.testing.assert_array_equal(state.adaptation, original_state[1])


@pytest.mark.parametrize("budget", [0, 1, 3, 25])
def test_fallback_respects_zero_and_odd_total_budgets(monkeypatch, budget):
    agent, state, drive = reversal_boundary()
    calls = []
    equilibrate = NeuralGraph.equilibrate

    def counted(brain, *args, **kwargs):
        phase = equilibrate(brain, *args, **kwargs)
        calls.append((brain.neuron_model.dt, kwargs["budget"], phase.steps))
        return phase

    monkeypatch.setattr(NeuralGraph, "equilibrate", counted)
    phase = agent._equilibrate(drive, state, budget=budget, tolerance=1e-14)
    assert not phase.qualified.any()
    assert phase.steps == sum(taken for _, _, taken in calls) == budget
    assert sum(limit for _, limit, _ in calls) == budget
    assert calls[0][:2] == (1.0, budget if budget < 2 else (budget + 1) // 2)
    if budget < 2:
        assert len(calls) == 1
    else:
        assert len(calls) == 2 and calls[1][:2] == (0.5, budget // 2)
    assert agent.brain.neuron_model.dt == 1.0
    calls.clear()
    agent.learner.config = replace(agent.learner.config, free_steps=budget, tolerance=1e-14)
    with pytest.raises(RuntimeError, match="no action issued"):
        agent._qualified(drive, state)
    report = agent.last_settlement
    assert not report["qualified"]
    # the budget, then a rescue of as many from where it stopped, on the damping ladder
    assert report["steps"] == sum(taken for _, _, taken in calls) == 2 * budget
    assert report["budget"] == budget and report.get("rescue_steps", 0) == budget
    assert report["damping_halvings"] == min(agent.learner.config.damping, max(budget - 1, 0))
