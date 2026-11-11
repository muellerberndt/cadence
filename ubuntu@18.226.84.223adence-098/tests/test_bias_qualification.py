"""Bias searches use qualified original equations without changing their source graph."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def motor_graph():
    pre, post = np.where(~np.eye(36, dtype=bool))
    wire = cd.Connectome.from_synapses(
        36, pre=pre, post=post, sign=np.full(len(pre), -0.5),
        populations={"motor": range(36)},
    )
    return cd.NeuralGraph(wire, cd.learning_neuron_model(dt=1))


def isolated_graph():
    wire = cd.Connectome.from_synapses(1, pre=[], post=[], populations={"motor": [0]})
    return cd.NeuralGraph(wire, cd.learning_neuron_model(dt=1))


def test_qualified_motor_bias_reaches_independent_equilibrium_and_counts_all_work(monkeypatch):
    graph = motor_graph()
    drive = np.zeros((2, 36))
    target = 0.25
    # Positive activation is tanh(v/2). Each motor receives 35 inhibitory peers.
    expected_bias = 2 * np.arctanh(target) + 0.5 * 35 * target
    assert -6 < expected_bias < 6
    original = graph.neuron_model.to_dict()
    checks, comparisons = [], []
    residual, repeated = cd.NeuralGraph.residual, cd.NeuralGraph._repeated_state

    def counted(candidate, *args, **kwargs):
        checks.append(candidate.neuron_model.dt)
        return residual(candidate, *args, **kwargs)

    def compared(candidate, *args, **kwargs):
        comparisons.append(candidate.neuron_model.dt)
        return repeated(candidate, *args, **kwargs)

    monkeypatch.setattr(cd.NeuralGraph, "residual", counted)
    monkeypatch.setattr(cd.NeuralGraph, "_repeated_state", compared)
    report = {"previous": True}
    bias = cd.calibrate_bias(
        graph, drive, {"motor": target}, rounds=1, iterations=12,
        steps=512, tolerance=1e-6, qualified=True, damping=3, report=report,
    )
    np.testing.assert_allclose(bias, expected_bias, atol=12 / 2**12, rtol=0)
    np.testing.assert_array_equal(graph.bias, np.zeros(36))
    assert graph.neuron_model.to_dict() == original
    assert "previous" not in report and report["completed"]
    assert report["attempted_trials"] == report["admitted_trials"] == 12
    assert report["attempted_solves"] == report["admitted_solves"] == 13
    assert [trial["kind"] for trial in report["solves"]] == ["midpoint"] * 12 + ["final"]
    assert all(trial["qualified"] and trial["activation_consistent"] for trial in report["solves"])
    assert report["solves"][-1]["damping_halvings"] == 3
    assert report["steps"] == 512 and report["tolerance"] == 1e-6
    assert report["damping"] == 3 and report["span"] == [-6, 6]
    for trial in report["solves"][:-1]:
        mean = trial["mean_output"]
        potential = 2 * np.arctanh(mean) if mean >= 0 else 2 * np.arctanh(mean / 0.1)
        assert abs(potential - (trial["trial_bias"] - 0.5 * 35 * mean)) < 1e-6
    assert report["total_residual_checks"] == len(checks)
    assert report["total_stagnation_checks"] == len(comparisons)
    assert report["total_activation_checks"] == 13
    assert report["total_row_activation_checks"] == report["attempted_presentations"] == 26
    for name in ("steps", "residual_checks", "stagnation_checks", "activation_checks",
                 "row_sweeps", "row_residual_checks", "row_activation_checks"):
        assert report["total_" + name] == sum(trial[name] for trial in report["solves"])
    assert all(trial["steps"] <= 512 for trial in report["solves"])
    assert report["total_row_sweeps"] == 2 * report["total_steps"]
    assert report["total_row_residual_checks"] == 2 * len(checks)
    assert report["targets"] == [{"population": "motor", "indices": list(range(36)), "target": target}]
    mean = report["final_means"][0]
    # Independent original potential equation at the returned symmetric activity.
    assert abs(2 * np.arctanh(mean) - (bias[0] - 0.5 * 35 * mean)) < 1e-6
    assert report["target_gaps"] == [abs(mean - target)]
    assert report["target_gaps"][0] < 1e-4


def test_finite_bias_search_keeps_its_output_and_reports_unqualified_endpoints():
    graph = motor_graph()
    drive = np.zeros((2, 36))
    plain = cd.calibrate_bias(graph, drive, {"motor": 0.25}, rounds=1, iterations=8)
    report = {}
    diagnosed = cd.calibrate_bias(graph, drive, {"motor": 0.25}, rounds=1, iterations=8, report=report)
    np.testing.assert_array_equal(plain, diagnosed)
    assert not report["qualification_required"] and report["completed"]
    assert report["total_residual_checks"] == report["attempted_solves"] == 9
    assert any(not trial["qualified"] for trial in report["solves"])
    assert report["solves"][-1]["residual"][0] > 1


@pytest.mark.parametrize("with_report", [False, True])
def test_finite_bias_search_refuses_numerical_overflow_before_using_its_output(with_report):
    # All inputs and weights are finite, but their transport overflows. Saturated
    # activation alone is finite and must not conceal the invalid potential.
    wire = cd.Connectome.from_synapses(
        3, pre=[0, 1], post=[2, 2], sign=[1e308, 1e308],
        populations={"motor": [2]},
    )
    graph = cd.NeuralGraph(wire, cd.learning_neuron_model(dt=1))
    before = (graph.efficacy.copy(), graph.bias.copy(), graph.log_gain.copy())
    report = {} if with_report else None
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(RuntimeError, match="invalid bias calibration midpoint"):
            cd.calibrate_bias(
                graph, np.array([[1., 1., 1e308]]), {"motor": 0.5},
                rounds=1, iterations=1, steps=4, report=report,
            )
    for actual, expected in zip((graph.efficacy, graph.bias, graph.log_gain), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    if report is not None:
        assert not report["completed"] and report["attempted_solves"] == 1
        assert report["admitted_solves"] == 0
        assert not report["solves"][0]["finite"]
        assert report["solves"][0]["mean_output"] is None
        assert report["final_means"] == report["target_gaps"] == []


@pytest.mark.parametrize("with_report", [False, True])
@pytest.mark.parametrize("kind", ["midpoint", "final"])
@pytest.mark.parametrize("broken", ["potential", "adaptation", "cache"])
def test_finite_bias_search_validates_all_candidate_state(monkeypatch, with_report, kind, broken):
    graph = isolated_graph()
    settle = cd.NeuralGraph.settle_batch
    calls = 0
    failing_call = 1 if kind == "midpoint" else 2

    def corrupt(candidate, drive, **kwargs):
        nonlocal calls
        calls += 1
        state = settle(candidate, drive, **kwargs)
        if calls == failing_call:
            if broken == "cache":
                state.activation[:] = 0.5
            elif broken == "potential":
                state.v[:] = np.nan
            else:
                state.adaptation[:] = np.nan
        return state

    monkeypatch.setattr(cd.NeuralGraph, "settle_batch", corrupt)
    report = {} if with_report else None
    with pytest.raises(RuntimeError, match="invalid bias calibration " + kind):
        cd.calibrate_bias(
            graph, np.zeros((1, 1)), {"motor": 0.25}, rounds=1, iterations=1,
            steps=4, report=report,
        )
    assert calls == failing_call
    np.testing.assert_array_equal(graph.bias, [0])
    if report is not None:
        assert not report["completed"] and not report["solves"][-1]["admitted"]
        assert report["solves"][-1]["kind"] == kind


def test_qualified_bias_does_not_promise_the_target_is_reachable():
    graph = isolated_graph()
    report = {}
    bias = cd.calibrate_bias(
        graph, np.zeros((1, 1)), {"motor": 0.9}, span=(-0.1, 0.1),
        rounds=1, iterations=8, steps=32, tolerance=1e-9, qualified=True, report=report,
    )
    assert report["completed"] and all(trial["qualified"] for trial in report["solves"])
    assert bias[0] < 0.1 and report["final_means"][0] == pytest.approx(np.tanh(bias[0] / 2))
    assert report["target_gaps"][0] > 0.8


def test_per_neuron_targets_and_final_means_keep_the_declared_group_order():
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(2, pre=[], post=[], populations={"motor": [0, 1]}),
        cd.learning_neuron_model(dt=1),
    )
    drive = np.array([[0.1, -0.2], [0.2, 0.1]])
    report = {}
    bias = cd.calibrate_bias(
        graph, drive, {"motor": 0.25}, per_neuron=True, rounds=1, iterations=12,
        steps=32, tolerance=1e-9, qualified=True, report=report,
    )
    assert [group["indices"] for group in report["targets"]] == [[0], [1]]
    assert report["attempted_trials"] == 24 and report["attempted_solves"] == 25
    expected = np.tanh((drive + bias) / 2).mean(axis=0)
    np.testing.assert_allclose(report["final_means"], expected, atol=1e-12, rtol=0)
    assert max(report["target_gaps"]) < 0.001


def test_refused_midpoint_is_not_used_to_advance_the_search():
    graph = isolated_graph()
    report = {}
    with pytest.raises(RuntimeError, match="unqualified bias calibration midpoint"):
        cd.calibrate_bias(
            graph, np.zeros((2, 1)), {"motor": 0.25}, rounds=3, iterations=16,
            steps=0, tolerance=1e-9, qualified=True, report=report,
        )
    # The zero midpoint qualifies at rest; the next positive one refuses at zero budget.
    assert not report["completed"]
    assert report["attempted_trials"] == report["attempted_solves"] == 2
    assert report["admitted_trials"] == 1
    assert report["solves"][-1]["mean_output"] is None
    assert report["total_steps"] == report["total_row_sweeps"] == 0
    assert report["total_residual_checks"] == 2
    assert report["final_means"] == report["target_gaps"] == []
    np.testing.assert_array_equal(graph.bias, [0])


@pytest.mark.parametrize("broken", ["residual", "cache", "adaptation", "final"])
def test_false_green_or_refused_final_state_cannot_return_bias(monkeypatch, broken):
    graph = isolated_graph()
    solve = cd.NeuralGraph.equilibrate
    calls = 0

    def corrupt(candidate, drive, **kwargs):
        nonlocal calls
        calls += 1
        phase = solve(candidate, drive, **kwargs)
        if broken == "residual" or (broken == "final" and calls == 2):
            return replace(phase, residual=np.ones(len(drive)))
        if broken in ("cache", "adaptation"):
            state = cd.BrainState(
                phase.state.v.copy(), phase.state.activation.copy(),
                phase.state.adaptation.copy(), phase.state.steps,
            )
            if broken == "cache":
                state.activation[:] = 0.5
            else:
                state.adaptation[:] = np.nan
            return replace(phase, state=state)
        return phase

    monkeypatch.setattr(cd.NeuralGraph, "equilibrate", corrupt)
    report = {}
    kind = "final" if broken == "final" else "midpoint"
    with pytest.raises(RuntimeError, match="unqualified bias calibration " + kind):
        cd.calibrate_bias(
            graph, np.zeros((1, 1)), {"motor": 0.25}, rounds=1, iterations=1,
            steps=32, qualified=True, report=report,
        )
    assert not report["completed"] and not report["solves"][-1]["admitted"]
    assert report["solves"][-1]["kind"] == kind
    np.testing.assert_array_equal(graph.bias, [0])


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_refused_torch_bias_search_preserves_lazy_source_and_device_parameters(precision):
    torch = pytest.importorskip("torch")
    wire = cd.layered(2, 3, 2, seed=3)
    graph = cd.NeuralGraph(
        wire, cd.learning_neuron_model(dt=1), backend="torch", device="cpu", precision=precision,
    )
    learner = cd.Learner(graph, wire.populations["output"], cd.LearnerConfig(momentum=0.6, normalize=0.7))
    drive = np.zeros((2, wire.n))
    drive[:, :2] = np.eye(2)
    learner.step(drive, np.array([0, 1]))
    source = learner.brain
    assert source._efficacy is source._bias is None
    held = learner.__dict__["_device_moments"]
    tensors = {name: value for name, value in held.items() if name != "holder"}
    values = {name: value.clone() for name, value in tensors.items()}
    scale, bias = source._torch.scale.clone(), source._torch.bias_param.clone()
    report = {}
    with pytest.raises(RuntimeError, match="unqualified bias calibration midpoint"):
        cd.calibrate_bias(source, drive, {"output": 0.25}, steps=0, qualified=True, report=report)
    assert learner.brain is source and source._efficacy is source._bias is None
    assert learner.__dict__["_device_moments"] is held
    assert held["holder"] is source._torch
    for name, tensor in tensors.items():
        assert held[name] is tensor
        torch.testing.assert_close(tensor, values[name], rtol=0, atol=0)
    torch.testing.assert_close(source._torch.scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(source._torch.bias_param, bias, rtol=0, atol=0)
    assert not report["completed"] and report["total_steps"] == 0


@pytest.mark.parametrize("kwargs", [
    {"rounds": 0}, {"rounds": True}, {"iterations": 1.5}, {"iterations": 0},
    {"steps": -1}, {"steps": True}, {"damping": -1}, {"damping": 0.5},
    {"qualified": 1}, {"per_neuron": "yes"}, {"tolerance": np.nan},
    {"tolerance": -1}, {"tolerance": "small"}, {"qualified": True, "tolerance": None},
    {"span": (0, 0)}, {"span": (2, -1)}, {"span": (-np.inf, 6)}, {"span": [1]},
])
def test_invalid_controls_leave_report_and_parameters_unchanged(kwargs):
    graph = isolated_graph()
    report = {"previous": True}
    with pytest.raises(ValueError):
        cd.calibrate_bias(graph, np.zeros((1, 1)), {"motor": 0.25}, report=report, **kwargs)
    assert report == {"previous": True}
    np.testing.assert_array_equal(graph.bias, [0])


@pytest.mark.parametrize("drive,targets", [
    (np.empty((0, 1)), {"motor": 0.25}), (np.zeros((1, 2)), {"motor": 0.25}),
    (np.full((1, 1), np.nan), {"motor": 0.25}), (np.zeros((1, 1, 1)), {"motor": 0.25}),
    (np.zeros((1, 1)), {"absent": 0.25}), (np.zeros((1, 1)), {"motor": np.nan}),
    (np.zeros((1, 1)), {"motor": [0.25]}), (np.zeros((1, 1)), {"motor": "small"}),
    (np.zeros((1, 1)), {"motor": 2}), (np.zeros((1, 1)), {(1,): 0.25}),
    (np.zeros((1, 1)), {(0.1,): 0.25}), (np.zeros((1, 1)), {(0, 0): 0.25}),
])
def test_invalid_drive_or_targets_are_validated_before_clearing_report(drive, targets):
    graph = isolated_graph()
    report = {"previous": True}
    with pytest.raises(ValueError):
        cd.calibrate_bias(graph, drive, targets, report=report)
    assert report == {"previous": True}
    np.testing.assert_array_equal(graph.bias, [0])
