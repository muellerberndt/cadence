"""Actual-device System 1 transport, admission, memory and continuation checks."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


@pytest.fixture(
    params=[("cpu", "float64"), ("cuda:0", "float64"), ("cuda:0", "float32"), ("mps:0", "float32")],
    ids=["torch_cpu64", "cuda64", "cuda32", "mps32"],
)
def execution(request):
    torch = pytest.importorskip("torch")
    device, precision = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    if device.startswith("mps") and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    return device, precision


def graph_on(graph, execution, **changes):
    device, precision = execution
    return cd.NeuralGraph(
        graph.connectome,
        changes.pop("model", graph.neuron_model),
        efficacy=graph.efficacy,
        bias=graph.bias,
        log_gain=graph.log_gain,
        backend="torch",
        device=device,
        precision=precision,
        **changes,
    )


def composed(execution):
    brain = cd.Brain.compose(
        3,
        2,
        modules=(5, 4),
        observers=(2,),
        seed=3,
        learning=cd.LearnerConfig(
            qualified=True,
            free_steps=512,
            nudged_steps=512,
            eta=0.01,
            eta_bias=0.001,
            momentum=0.8,
            normalize=0.9,
        ),
    )
    brain.learner.brain = graph_on(brain.brain, execution)
    return brain


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def unchanged(brain, before, path):
    after = archive(brain.save(path))
    assert after.keys() == before.keys()
    for name, value in before.items():
        np.testing.assert_array_equal(after[name], value, err_msg=name)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("nudge_kind", [None, "quadratic", "softmax"])
def test_actual_device_transport_matches_independent_projected_equations(
    execution,
    sparse,
    nudge_kind,
):
    brain = composed(execution)
    graph = graph_on(
        brain.brain,
        execution,
        dense_limit=1 if sparse else 2048,
        model=cd.learning_neuron_model(dt=0.3).replace(
            adaptation=cd.Adaptation(tau_steps=7, strength=0.2)
        ),
    )
    rng = np.random.default_rng(12)
    n = graph.connectome.n
    drive = rng.uniform(-0.2, 0.8, (3, n))
    v, a = rng.uniform(-0.3, 0.5, (3, n)), rng.uniform(0, 0.3, (3, n))
    keep = rng.choice([0.0, 0.5, 1.0], (3, n))
    model = graph.neuron_model

    # Independent scalar equations and edge scatter; no production transport/nudge.
    def activity(potential):
        r = 1 / (1 + np.exp(-model.slope * (potential - model.threshold)))
        rest = 1 / (1 + np.exp(model.slope * model.threshold))
        delta = r - rest
        return np.maximum(delta, 0) / (1 - rest) + model.leak * np.minimum(delta, 0) / rest

    mask = np.zeros(n)
    outputs = brain.motor_index
    mask[outputs] = 1
    target = np.zeros((3, n))
    target[:, outputs[0]] = 1
    row_weight = np.array([1.0, -0.4, 0.0])
    nudge = (
        None
        if nudge_kind is None
        else cd.Nudge(
            target,
            mask,
            0.13,
            softmax_temperature=0.3 if nudge_kind == "softmax" else None,
            weight=row_weight,
            anchor=np.full((3, n), 0.1),
            anchor_gain=np.full(n, 0.02),
        )
    )
    s = activity(v) * keep
    matrix = np.zeros((n, n))
    wire = graph.connectome
    np.add.at(matrix, (wire.post, wire.pre), graph.weights)
    total = s @ matrix.T + drive + graph.bias - model.adaptation.strength * a
    if nudge is not None:
        push = 0.13 * (target - s) * mask
        if nudge_kind == "softmax":
            z = s[:, outputs] / 0.3
            p = np.exp(z - z.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            push[:, outputs] = 0.13 * (target[:, outputs] - p)
        total += push * row_weight[:, None] + 0.02 * (0.1 - s)
    expected_v = (v + model.dt * (total - v)) * keep
    expected_s = activity(expected_v) * keep
    expected_a = a + (expected_s - a) / model.adaptation.tau_steps
    warm = cd.BrainState(v.copy(), s.copy(), a.copy(), 0)
    result = graph.settle_batch(drive, state=warm, steps=1, mask=keep, nudge=nudge)
    torch = pytest.importorskip("torch")
    assert result.device["v"].device == torch.device(execution[0])
    assert result.device["v"].dtype == getattr(torch, execution[1])
    tol = 2e-6 if execution[1] == "float32" else 2e-12
    for actual, expected in (
        (result.v, expected_v),
        (result.activation, expected_s),
        (result.adaptation, expected_a),
    ):
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    expected_residual = np.maximum(
        np.abs((expected_v - v) / model.dt).max(axis=1), np.abs(s - a).max(axis=1)
    )
    np.testing.assert_allclose(
        graph.residual(drive, warm, mask=keep, nudge=nudge), expected_residual, rtol=tol, atol=tol
    )
    np.testing.assert_array_equal(warm.v, v)
    np.testing.assert_array_equal(warm.adaptation, a)


def test_observer_feedback_participates_in_the_same_device_solve(execution):
    brain = composed(execution)
    graph = brain.brain
    wire = graph.connectome
    observer = np.asarray(wire.populations["observer_0"])
    bias = graph.bias.copy()
    bias[observer] = 0.4
    graph = graph.with_parameters(bias=bias)
    drive = brain.stimulus([[0.5, 0.1, -0.1]], memory=False)
    full = graph.equilibrate(drive, budget=512, tolerance=3e-3, damping=3)
    assert full.qualified.all()
    returning = np.isin(wire.pre, observer) & ~np.isin(wire.post, observer)
    assert returning.any()
    efficacy = graph.efficacy.copy()
    efficacy[returning] = 0
    cut = graph.with_parameters(efficacy=efficacy)
    removed = cut.equilibrate(drive, budget=512, tolerance=3e-3, damping=3)
    assert removed.qualified.all()
    assert (
        np.max(
            np.abs(
                full.state.v[:, brain.association_index]
                - removed.state.v[:, brain.association_index]
            )
        )
        > 1e-4
    )
    assert graph.residual(drive, full.state, on_device=False).max() <= 3e-3


def test_overflowed_device_proposal_never_qualifies_or_changes_parameters(execution):
    maximum = np.finfo(execution[1]).max
    wire = cd.Connectome.from_synapses(
        3, pre=[0, 1], post=[2, 2], sign=[maximum * 0.8, maximum * 0.8]
    )
    graph = cd.NeuralGraph(
        wire,
        cd.learning_neuron_model(dt=1),
        backend="torch",
        device=execution[0],
        precision=execution[1],
    )
    before = graph.efficacy.copy(), graph.bias.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        phase = graph.equilibrate(
            np.array([[100.0, 100.0, 0.0]]), budget=4, chunk=2, tolerance=3e-3
        )
    assert not phase.qualified.any()
    assert not np.isfinite(phase.residual).all()
    assert phase.steps > 0
    np.testing.assert_array_equal(graph.efficacy, before[0])
    np.testing.assert_array_equal(graph.bias, before[1])


def test_frozen_parameters_survive_real_device_learning(execution):
    brain = composed(execution)
    learner = brain.learner
    learner.config = replace(learner.config, free_steps=1024, nudged_steps=1024)
    learner.plastic_synapses[::2] = False
    learner.plastic_neurons[::2] = False
    kernel = brain.brain._torch
    # MPS stores parameters in float32. Freeze their actual device values, not
    # the pre-upload float64 construction arrays, which can round on upload.
    efficacy, bias = kernel.scale.clone(), kernel.bias_param.clone()
    _, report = learner.step(brain.stimulus([[0.5, 0.2, 0.1]], memory=False), np.array([1]))
    assert report["accepted"] == report["qualified"] == 1
    assert report["total_steps"] > 0
    assert brain.brain._torch is kernel
    assert kernel.torch.equal(kernel.scale[::2], efficacy[::2])
    assert kernel.torch.equal(kernel.bias_param[::2], bias[::2])
    assert not kernel.torch.equal(kernel.scale[1::2], efficacy[1::2])


def test_refused_answer_and_lesson_preserve_pending_feedback_and_optimizer(execution, tmp_path):
    brain = composed(execution)
    cue = np.array([[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]])
    brain.step(cue, teacher=np.array([0, 1]))
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=1, tolerance=1e-14)
    before = archive(brain.save(tmp_path / "before.npz"))
    with pytest.raises(RuntimeError, match="no action issued"):
        brain.act(cue + 0.7)
    assert brain.last_settlement["steps"] == 1
    assert not brain.last_settlement["qualified"]
    unchanged(brain, before, tmp_path / "after-act.npz")
    with pytest.raises(cd.LearningPhaseError) as refused:
        brain.learner.step(brain.stimulus(cue + 0.7), np.array([1, 0]))
    assert refused.value.report["free_steps"] == 1
    unchanged(brain, before, tmp_path / "after-lesson.npz")
    brain.learner.config = config
    report = brain.learn(np.array([0.4, -0.2]), np.array([False, True]), cue)
    assert np.isfinite(report["td_error"])
    with pytest.raises(RuntimeError):
        brain.learn(np.array([0.4, -0.2]), np.array([False, True]), cue)


def test_device_allocation_failure_during_solve_keeps_live_continuation(
    execution,
    tmp_path,
    monkeypatch,
):
    brain = composed(execution)
    cue = np.array([[0.2, 0.1, 0.3]])
    brain.step(cue)
    before = archive(brain.save(tmp_path / "before.npz"))
    kernel = brain.brain._torch
    original = kernel._synaptic_input
    calls = []
    torch = kernel.torch

    def fail_after_transport(*args):
        result = original(*args)
        calls.append(result.device.type)
        raise torch.OutOfMemoryError("injected allocation failure after actual transport")

    monkeypatch.setattr(kernel, "_synaptic_input", fail_after_transport)
    with pytest.raises(torch.OutOfMemoryError):
        brain.act(cue + 0.2)
    assert calls == [torch.device(execution[0]).type]
    unchanged(brain, before, tmp_path / "after.npz")
    monkeypatch.setattr(kernel, "_synaptic_input", original)
    assert brain.act(cue + 0.2).shape == (1,)


def test_private_imagination_checkpoint_transfer_and_actual_outcome_custody(execution, tmp_path):
    brain = composed(execution)
    cue = np.array([[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]])
    action = brain.step(cue, teacher=np.array([0, 1]))
    saved = brain.save(tmp_path / "pending.npz")
    before = archive(saved)
    imagined = brain.imagine([cue * 0.5, cue * 0.8], tolerance=3e-3)
    assert len(imagined) == 2 and all(p.qualified.all() for p in imagined)
    unchanged(brain, before, tmp_path / "after-private.npz")
    for bad in (np.array([np.nan, 0]), np.array([0])):
        with pytest.raises(ValueError):
            brain.learn(bad, np.array([False, True]), cue)
        unchanged(brain, before, tmp_path / "after-invalid.npz")
    device, precision = execution
    restored = cd.Brain.load(saved, backend="torch", device=device, precision=precision)
    transferred = cd.Brain.load(saved, backend="cpu")
    reward = (action == np.array([0, 1])).astype(float)
    expected_updates = brain.basal_ganglia.updates + 1
    expected_writes = brain.hippocampus.writes + len(cue)
    for owner in (brain, restored, transferred):
        owner.learn(reward, np.array([False, True]), cue * 0.7)
        assert owner.basal_ganglia.updates == expected_updates
        assert owner.basal_ganglia._pending is None
        assert owner.hippocampus.writes == expected_writes
    tol = 2e-5 if precision == "float32" else 2e-10
    for owner in (restored, transferred):
        np.testing.assert_allclose(owner.brain.efficacy, brain.brain.efficacy, rtol=tol, atol=tol)
        np.testing.assert_allclose(owner.brain.bias, brain.brain.bias, rtol=tol, atol=tol)
        np.testing.assert_array_equal(owner.hippocampus.strength, brain.hippocampus.strength)
    np.testing.assert_array_equal(brain.act(cue * 0.7), restored.act(cue * 0.7))
