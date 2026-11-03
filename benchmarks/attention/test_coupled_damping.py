"""Existing bounded halving preserves live memory equations, work and transactions."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


heads = load("attention_damping_heads", "heads.py")
fixed = load("attention_damping_fixed", "fixed_read.py")
coupled = heads.coupled
Graph = heads.Graph


def orbital_graph(kind="agreement", *, empty_anchor=True, zero_slots=False):
    offset = 76 if kind == "fixed" else 4
    motor = np.arange(offset, offset + 36)
    first, second = np.where(~np.eye(36, dtype=bool))
    pre, post = motor[first], motor[second]
    signs = np.full(len(pre), -0.5)
    if kind != "none":
        read = 19 if kind == "fixed" else 1
        pre, post = np.r_[pre, read, motor[0]], np.r_[post, motor[0], read]
        signs = np.r_[signs, 0.04, 0.04]
    connectome = cd.Connectome.from_synapses(
        offset + 36, pre=pre, post=post, sign=signs,
        populations={"motor": motor},
    )
    graph = Graph(connectome, cd.learning_neuron_model(dt=1.0))
    if kind == "agreement":
        slots = 0 if zero_slots else 2
        graph.memory = coupled.Memory(
            np.array([0]), np.array([1]), np.tile([[[-1.0], [1.0]]], (2, 1, 1))[:, :slots],
            np.tile([[[0.2], [0.6]]], (2, 1, 1))[:, :slots],
            np.array([[False, False], [True, True]])[:, :slots], np.zeros(slots),
            amplitude=0.4, temperature=0.9, empty_anchor=empty_anchor,
        )
    elif kind == "fixed":
        values = np.zeros((2, 2, 19))
        values[:, 0, [0, 5, 7]] = 1.0
        values[:, 1, [0, 6, 8]] = 1.0
        held = np.array([[False, False], [True, True]])
        member = coupled.Memory(
            np.arange(19), np.arange(19, 38), values, values, held, np.array([1., 0.]),
            amplitude=1.5, empty_anchor=empty_anchor,
        )
        other = replace(member, query=np.arange(38, 57), read=np.arange(57, 76))
        graph.memory = fixed.FrozenRead(heads.MultiMemory((member, other)))
    drive = np.zeros((2, connectome.n))
    drive[1, motor] = 0.2
    return graph, drive, motor


@pytest.mark.parametrize("kind", ["none", "agreement", "fixed"])
def test_zero_damping_delegates_the_original_exact_path(kind):
    graph, drive, _ = orbital_graph(kind)
    actual = graph.equilibrate(drive, budget=96, tolerance=3e-3)
    reference = cd.NeuralGraph.equilibrate(graph, drive, budget=96, tolerance=3e-3, damping=0)
    for name in ("v", "activation", "adaptation"):
        np.testing.assert_array_equal(getattr(actual.state, name), getattr(reference.state, name))
    np.testing.assert_array_equal(actual.residual, reference.residual)
    assert actual.steps == reference.steps and actual.residual_checks == reference.residual_checks
    assert actual.damping_halvings == reference.damping_halvings == 0


@pytest.mark.parametrize("kind", ["none", "agreement", "fixed"])
def test_clone_preserves_parameters_layout_and_exact_boundary_pointer(kind):
    graph, _, _ = orbital_graph(kind)
    memory = graph.memory
    changed = graph._with_dt(0.5)
    assert type(changed) is Graph and changed.memory is memory
    assert changed.connectome is graph.connectome and changed.layout is graph.layout
    assert changed.dense_limit == graph.dense_limit and changed.precision == graph.precision
    assert changed.backend == graph.backend == "cpu"
    assert changed.neuron_model == graph.neuron_model.replace(dt=0.5)
    for name in ("efficacy", "log_gain", "bias", "weights"):
        np.testing.assert_array_equal(getattr(changed, name), getattr(graph, name))
    assert graph.neuron_model.dt == 1.0
    for replacement in (changed.with_parameters(efficacy=changed.efficacy),
                        changed.with_parameters(log_gain=changed.log_gain),
                        changed.with_parameters(bias=changed.bias)):
        assert replacement.memory is memory


@pytest.mark.parametrize("kind", ["agreement", "fixed"])
def test_real_period_two_halving_keeps_memory_and_counts_every_attempt_and_check(monkeypatch, kind):
    graph, drive, _ = orbital_graph(kind)
    memory = graph.memory
    attempts, checks = [], []
    original_local, original_residual = Graph._equilibrate_local, Graph.residual

    def local(candidate, *args, **kwargs):
        assert candidate.memory is memory
        phase = original_local(candidate, *args, **kwargs)
        attempts.append((candidate.neuron_model.dt, kwargs["budget"], phase))
        return phase

    def residual(candidate, *args, **kwargs):
        assert candidate.memory is memory
        checks.append(candidate.neuron_model.dt)
        return original_residual(candidate, *args, **kwargs)

    monkeypatch.setattr(Graph, "_equilibrate_local", local)
    monkeypatch.setattr(Graph, "residual", residual)
    phase = graph.equilibrate(drive, budget=256, damping=3, tolerance=3e-3)
    assert phase.qualified.all() and phase.damping_halvings == 3
    assert [dt for dt, _, _ in attempts] == [1.0, 0.5, 0.25, 0.125]
    assert phase.steps == sum(p.steps for _, _, p in attempts) <= 256
    assert phase.residual_checks == len(checks)
    assert phase.stagnation_checks == sum(p.stagnation_checks for _, _, p in attempts) > 0
    assert graph.memory is memory and graph.neuron_model.dt == 1.0
    np.testing.assert_array_equal(phase.state.v[0], 0.0)
    # Evaluate the original model, including memory, independently of candidate reporting.
    s = graph.neuron_model.activation(phase.state.v)
    matrix = np.zeros((graph.connectome.n, graph.connectome.n))
    np.add.at(matrix, (graph.connectome.pre, graph.connectome.post), graph.weights)
    error = s @ matrix + memory.drive(s) + drive + graph.bias - phase.state.v
    expected = np.abs(error).max(axis=1)
    np.testing.assert_allclose(phase.residual, expected, atol=2e-14, rtol=0)
    assert expected.max() <= 3e-3
    assert np.abs(memory.drive(s)[1]).max() > 0.01


def test_none_boundary_matches_the_library_halving_schedule_exactly():
    graph, drive, _ = orbital_graph("none")
    reference = cd.NeuralGraph(graph.connectome, graph.neuron_model,
                               efficacy=graph.efficacy, bias=graph.bias, log_gain=graph.log_gain)
    actual = graph.equilibrate(drive, budget=256, damping=3, tolerance=3e-3)
    expected = reference.equilibrate(drive, budget=256, damping=3, tolerance=3e-3)
    for name in ("v", "activation", "adaptation"):
        np.testing.assert_array_equal(getattr(actual.state, name), getattr(expected.state, name))
    np.testing.assert_array_equal(actual.residual, expected.residual)
    for name in ("steps", "residual_checks", "stagnation_checks", "damping_halvings"):
        assert getattr(actual, name) == getattr(expected, name)
    assert graph.memory is None


@pytest.mark.parametrize("kind,zero_slots", [
    ("agreement", False), ("agreement", True), ("fixed", False),
])
def test_empty_and_subset_boundaries_remain_the_exact_held_object(kind, zero_slots):
    graph, drive, _ = orbital_graph(kind, zero_slots=zero_slots)
    graph.memory = graph.memory.take([1, 0])
    memory = graph.memory
    phase = graph.equilibrate(drive[[1, 0]], budget=256, damping=3, tolerance=3e-3)
    assert phase.qualified.all() and graph.memory is memory and memory.empty_anchor
    np.testing.assert_array_equal(graph.residual(drive[[1, 0]], phase.state), phase.residual)
    assert graph._with_dt(0.25).memory is memory


@pytest.mark.parametrize("setting,value", [
    ("budget", -1), ("budget", True), ("budget", 2.0),
    ("chunk", 0), ("chunk", False), ("chunk", 1.0),
    ("damping", -1), ("damping", True), ("damping", False),
    ("damping", 0.0), ("damping", np.bool_(True)),
])
def test_numerical_integer_domains_reject_before_any_attempt(monkeypatch, setting, value):
    graph, drive, _ = orbital_graph()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid numerical options started a phase")

    monkeypatch.setattr(Graph, "_equilibrate_local", forbidden)
    with pytest.raises(ValueError, match=setting):
        graph.equilibrate(drive, **{setting: value})


def test_numpy_integer_options_and_zero_budget_keep_warm_arrays_and_boundary():
    graph, drive, _ = orbital_graph()
    warm = cd.BrainState(np.zeros_like(drive), np.full_like(drive, 0.9),
                         np.zeros_like(drive), 9)
    before = [value.copy() for value in (warm.v, warm.activation, warm.adaptation)]
    memory = graph.memory
    phase = graph.equilibrate(
        drive, state=warm, budget=np.int64(0), chunk=np.int64(8), damping=np.int64(3))
    assert phase.steps == 0 and phase.residual_checks == 1 and phase.damping_halvings == 0
    assert not phase.qualified.all() and graph.memory is memory
    np.testing.assert_array_equal(
        phase.state.activation, graph.neuron_model.activation(phase.state.v))
    for actual, expected in zip((warm.v, warm.activation, warm.adaptation), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def snapshot(learner):
    return {name: getattr(learner, name).copy() for name in (
        "velocity", "velocity_bias", "second_moment", "second_moment_bias")}


def test_qualified_free_positive_and_negative_phases_use_same_memory_and_original_equations():
    graph, drive, motor = orbital_graph("fixed")
    memory = graph.memory
    learner = cd.Learner(graph, motor, cd.LearnerConfig(
        qualified=True, damping=3, free_steps=256, nudged_steps=256,
        tolerance=3e-3, eta=0.0, eta_bias=0.0, momentum=0.9, normalize=0.9))
    labels = np.array([0, 1])
    learned, report = learner.step(drive, labels)
    target = learner.targets(labels)
    for name, state, sign in (("free", learned.free, 0), ("nudged", learned.nudged, 1),
                              ("opposite", learned.opposite, -1)):
        nudge = learner.nudge_for(target, sign * learner.config.beta) if sign else None
        assert graph.residual(drive, state, nudge=nudge).max() <= 3e-3
        assert report[name + "_steps"] <= 256
        assert report[name + "_damping_halvings"] >= 0
    assert report["qualified"] == report["accepted"] == 1.0
    assert report["total_steps"] == sum(report[name + "_steps"]
                                         for name in ("free", "nudged", "opposite"))
    assert learner.brain.memory is graph.memory is memory
    assert learner.brain.neuron_model.dt == 1.0


def test_refused_damped_teaching_preserves_parameters_optimizer_and_boundary():
    graph, drive, motor = orbital_graph("fixed")
    with pytest.warns(RuntimeWarning, match="below free_steps"):
        cfg = cd.LearnerConfig(
            qualified=True, damping=3, free_steps=256, nudged_steps=0,
            tolerance=3e-3, momentum=0.9, normalize=0.9, eta=0.003, eta_bias=0.0003)
    learner = cd.Learner(graph, motor, cfg)
    for name, value in snapshot(learner).items():
        setattr(learner, name, value + 0.2)
    learner.updates = learner.contrast_updates = 3
    before = snapshot(learner)
    efficacy, bias, memory = graph.efficacy.copy(), graph.bias.copy(), graph.memory
    with pytest.raises(cd.LearningPhaseError, match="nudged") as caught:
        learner.step(drive, np.array([0, 1]))
    assert caught.value.phases["free"].qualified.all()
    assert caught.value.report["nudged_steps"] == 0
    assert caught.value.report["total_steps"] <= 256
    assert learner.brain is graph and graph.memory is memory
    assert learner.updates == learner.contrast_updates == 3
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(learner, name), value)
    np.testing.assert_array_equal(graph.efficacy, efficacy)
    np.testing.assert_array_equal(graph.bias, bias)


@pytest.mark.parametrize("count", [1, 2])
def test_composed_brains_accept_existing_damping_gene_and_saved_row_teaching(count):
    brain = heads.MultiHeadBrain.compose(
        3, 2, heads=count, modules=(4,), episodic=False, recall_next=False,
        dt=1.0, learning_damping=3, empty_anchor=True,
        learning_free_steps=256, learning_nudged_steps=256)
    assert brain.learner.config.damping == brain.describe()["genes"]["learning_damping"] == 3
    assert brain.brain.neuron_model.dt == 1.0
    brain.act([[1, 0, 0], [0, 1, 0]], greedy=True)
    drive = brain.stimulus([[0, 0, 1], [0, 0, 1]]).copy()
    memory, original = brain.brain.memory, brain.brain
    _, report = brain.teach_saved(drive, np.array([0, 1]), [1], memory)
    assert report["qualified"] and report["accepted_presentations"] == 1
    assert original.memory is brain.brain.memory is memory
    assert brain.brain.neuron_model.dt == 1.0


def test_composed_default_damping_and_dt_are_unchanged():
    for cls in (coupled.AttentionBrain, heads.MultiHeadBrain):
        brain = cls.compose(3, 2, modules=(4,), episodic=False)
        assert brain.learner.config.damping == 0
        assert brain.brain.neuron_model.dt == 0.1


@pytest.mark.parametrize("kind", ["none", "agreement", "fixed"])
def test_the_halved_copies_keep_current_parameters_boundary_and_original_residual(kind):
    graph, drive, _ = orbital_graph(kind)
    graph = graph.with_parameters(bias=graph.bias + .001)
    memory, model = graph.memory, graph.neuron_model
    actual = graph.equilibrate(drive, budget=256, damping=3, tolerance=3e-3)
    assert graph.memory is memory and graph.neuron_model is model
    same = graph._with_dt(.125)
    assert type(same) is type(graph) and same.memory is memory
    for name in ("efficacy", "bias", "log_gain"):
        np.testing.assert_array_equal(getattr(same, name), getattr(graph, name))
    np.testing.assert_array_equal(actual.residual, graph.residual(drive, actual.state))
    assert 0 <= actual.damping_halvings <= 3


def test_benchmark_life_archive_keeps_the_qualified_damping_policy(tmp_path):
    brain = heads.MultiHeadBrain.compose(
        3, 2, heads=1, modules=(4,), episodic=False, recall_next=False,
        dt=1., learning_qualified=True, learning_damping=3,
        learning_free_steps=256, learning_nudged_steps=256,
    )
    brain.act([[1., 0., 0.]])
    assert brain.pending_feedback
    path = brain.save(tmp_path / "qualified-policy.npz")
    resumed = heads.MultiHeadBrain.load(path)
    assert resumed.learner.config == brain.learner.config
    assert resumed.learner.config.damping == 3 and resumed.learner.config.qualified
    assert type(resumed.brain) is Graph
    for owner in (brain, resumed):
        owner.learn(np.array([.1]), np.array([False]), [[0., 1., 0.]])
    np.testing.assert_array_equal(brain.brain.efficacy, resumed.brain.efficacy)
    np.testing.assert_array_equal(brain.act([[0., 0., 1.]], greedy=True),
                                  resumed.act([[0., 0., 1.]], greedy=True))
