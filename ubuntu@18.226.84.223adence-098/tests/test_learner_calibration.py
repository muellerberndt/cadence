"""Gain selection admits actual free states and charges its bounded search."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def sparse_learner(**config):
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1])
    return cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model(dt=1)), [1],
        cd.LearnerConfig(qualified=True, free_steps=32, tolerance=1e-9, **config),
    )


def test_default_gain_grid_reaches_sparse_operating_points_above_one():
    learner = sparse_learner()
    drive = np.array([[1., 0.]])
    old_gain = learner.calibrate(drive, grid=[0.02 * 1.3**k for k in range(16)])
    old_gap = abs(np.tanh(old_gain * np.tanh(0.5) / 2) - 0.5)
    learner.brain = learner._with_gain(1)
    assert learner.calibrate(drive) == 2
    report = learner.last_calibration
    assert [x["gain"] for x in report["candidates"]] == [
        1., *(2.**k for k in range(-8, 9) if k)
    ]
    assert report["attempted_candidates"] == report["admitted_candidates"] == 17
    assert report["total_activation_checks"] == report["total_row_activation_checks"] == 17
    for candidate in report["candidates"]:
        # Independent closed form for this asymmetric graph's free equilibrium.
        expected = np.tanh(candidate["gain"] * np.tanh(0.5) / 2)
        assert candidate["mean_output"] == pytest.approx(expected, abs=1e-12)
        assert candidate["qualified"] and candidate["residual"] == [0.]
    assert abs(np.tanh(np.tanh(0.5)) - 0.5) < old_gap


def test_flat_objective_preserves_current_gain_by_default_and_explicit_grid_order():
    learner = sparse_learner()
    learner.brain = learner._with_gain(3)
    drive = np.zeros((1, 2))
    assert learner.calibrate(drive) == 3
    assert learner.calibrate(drive, grid=[4, 0.5, 4]) == 4
    assert [x["gain"] for x in learner.last_calibration["candidates"]] == [4, 0.5, 4]


def test_zero_budget_qualified_calibration_refuses_without_parameter_or_history_writes():
    learner = sparse_learner(momentum=0.8, normalize=0.9)
    learner.config = replace(learner.config, free_steps=0)
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        getattr(learner, name)[:] = 0.3
    learner.updates = learner.contrast_updates = 7
    graph = learner.brain
    before = {
        name: getattr(learner, name).copy()
        for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
    }
    params = [graph.efficacy.copy(), graph.bias.copy(), graph.log_gain.copy()]
    with pytest.raises(RuntimeError, match="no admissible calibration candidate"):
        learner.calibrate(np.ones((2, 2)), grid=[1, 2])
    assert learner.brain is graph
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(learner, name), value)
    for value, old in zip((graph.efficacy, graph.bias, graph.log_gain), params, strict=True):
        np.testing.assert_array_equal(value, old)
    assert learner.updates == learner.contrast_updates == 7
    report = learner.last_calibration
    assert report["selected_gain"] is None and report["admitted_candidates"] == 0
    assert report["attempted_presentations"] == 4
    assert report["total_steps"] == report["total_row_sweeps"] == 0
    assert report["total_activation_checks"] == 2
    assert report["total_row_activation_checks"] == 4
    assert report["total_residual_checks"] == 2
    assert report["total_row_residual_checks"] == 4
    assert all(x["residual"] == [1., 1.] for x in report["candidates"])


def test_calibration_damping_qualifies_the_original_equations_and_counts_work(monkeypatch):
    pre, post = np.where(~np.eye(36, dtype=bool))
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5)),
        cd.learning_neuron_model(dt=1),
    )
    learner = cd.Learner(graph, np.arange(36), cd.LearnerConfig(
        qualified=True, damping=3, free_steps=256, nudged_steps=256, tolerance=3e-3,
    ))
    checks, candidates = [], []
    residual, with_gain = cd.NeuralGraph.residual, cd.Learner._with_gain

    def counted(candidate, *args, **kwargs):
        checks.append(candidate.neuron_model.dt)
        return residual(candidate, *args, **kwargs)

    def constructed(instance, gain):
        candidate = with_gain(instance, gain)
        candidates.append(candidate)
        return candidate

    monkeypatch.setattr(cd.NeuralGraph, "residual", counted)
    monkeypatch.setattr(cd.Learner, "_with_gain", constructed)
    assert learner.calibrate(np.full((2, 36), 0.2), grid=[1]) == 1
    report = learner.last_calibration
    assert report["total_residual_checks"] == len(checks)
    assert report["total_row_residual_checks"] == 2 * len(checks)
    assert report["total_steps"] <= 256
    assert report["total_row_sweeps"] == 2 * report["total_steps"]
    assert report["candidates"][0]["damping_halvings"] == 3
    assert report["candidates"][0]["qualified"]
    assert learner.brain is candidates[0] and len(candidates) == 1
    assert graph.neuron_model.dt == learner.brain.neuron_model.dt == 1


def test_unqualified_candidate_cannot_win_even_when_closer_to_target(monkeypatch):
    learner = sparse_learner()
    solve = cd.NeuralGraph.equilibrate

    def candidates(graph, drive, **kwargs):
        phase = solve(graph, drive, **kwargs)
        if graph.neuron_model.gain == 2:
            return replace(phase, residual=np.full(len(drive), 1.0))
        return phase

    monkeypatch.setattr(cd.NeuralGraph, "equilibrate", candidates)
    assert learner.calibrate(np.array([[1., 0.]]), grid=[2, 1]) == 1
    attempts = learner.last_calibration["candidates"]
    assert not attempts[0]["admitted"] and attempts[1]["admitted"]
    assert learner.last_calibration["total_steps"] == sum(x["steps"] for x in attempts)


@pytest.mark.parametrize("broken", ["potential", "activation", "adaptation", "cache"])
def test_nonfinite_or_inconsistent_state_cannot_be_admitted(monkeypatch, broken):
    learner = sparse_learner()
    graph = learner.brain
    solve = cd.NeuralGraph.equilibrate

    def candidate(graph, drive, **kwargs):
        phase = solve(graph, drive, **kwargs)
        state = cd.BrainState(
            phase.state.v.copy(), phase.state.activation.copy(),
            phase.state.adaptation.copy(), phase.state.steps,
        )
        value = {"potential": state.v, "activation": state.activation,
                 "adaptation": state.adaptation, "cache": state.activation}[broken]
        value[0, 0] = 0.9 if broken == "cache" else np.nan
        return replace(phase, state=state)

    monkeypatch.setattr(cd.NeuralGraph, "equilibrate", candidate)
    with pytest.raises(RuntimeError, match="no admissible calibration"):
        learner.calibrate(np.array([[1., 0.]]), grid=[1])
    assert learner.brain is graph
    assert not learner.last_calibration["candidates"][0]["admitted"]


def test_finite_calibration_does_not_claim_required_equilibrium():
    learner = sparse_learner()
    learner.config = replace(learner.config, qualified=False, free_steps=0)
    assert learner.calibrate(np.ones((1, 2)), grid=[1]) == 1
    report = learner.last_calibration
    assert not report["qualification_required"]
    assert report["candidates"][0]["admitted"]
    assert not report["candidates"][0]["qualified"]
    assert report["total_steps"] == 0 and report["total_residual_checks"] == 1


@pytest.mark.parametrize("drive,grid,level", [
    (np.empty((0, 2)), [1], 0.5), (np.ones((1, 3)), [1], 0.5),
    (np.ones((2, 1, 2)), [1], 0.5), (np.array([[np.nan, 0]]), [1], 0.5),
    (np.array([[np.inf, 0]]), [1], 0.5), (np.ones(2), [], 0.5),
    (np.ones(2), [0], 0.5), (np.ones(2), [-1], 0.5),
    (np.ones(2), [np.nan], 0.5), (np.ones(2), [np.inf], 0.5),
    (np.ones(2), [[1]], 0.5), (np.ones(2), [1], np.inf),
    (np.ones(2), [1], [0.5]),
])
def test_invalid_calibration_input_is_rejected_before_search(drive, grid, level):
    learner = sparse_learner()
    graph = learner.brain
    learner.last_calibration = {"previous": True}
    with pytest.raises(ValueError):
        learner.calibrate(drive, grid=grid, level=level)
    assert learner.brain is graph and learner.last_calibration == {"previous": True}
