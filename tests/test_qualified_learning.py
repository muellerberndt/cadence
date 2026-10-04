"""Residual-gated contrast learning and its independent numerical boundary."""

import numpy as np
import pytest

import cadence as cd


def motor_graph(backend="cpu", precision=None):
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "mlx":
        pytest.importorskip("mlx.core")
    pre, post = np.where(~np.eye(36, dtype=bool))
    connectome = cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5))
    return cd.NeuralGraph(
        connectome, cd.learning_neuron_model(dt=1), backend=backend,
        device="cpu" if backend == "torch" else None,
        precision=precision,
    )


def snapshot(learner):
    return {
        name: getattr(learner, name).copy()
        for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
    } | {"efficacy": learner.brain.efficacy.copy(), "bias": learner.brain.bias.copy(),
         "updates": learner.updates, "contrast_updates": learner.contrast_updates}


def assert_snapshot(learner, before):
    for name, value in snapshot(learner).items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_qualified_learning_repairs_period_two_without_changing_the_live_model(backend):
    graph = motor_graph(backend)
    drive = np.full((2, 36), 0.2)
    config = cd.LearnerConfig(
        qualified=True, damping=3, free_steps=256, nudged_steps=256,
        tolerance=3e-3, eta=0, eta_bias=0,
    )
    learner = cd.Learner(graph, np.arange(36), config)
    failed = graph.settle_batch(drive, steps=256)
    one = graph.settle_batch(drive, state=failed, steps=1)
    two = graph.settle_batch(drive, state=one, steps=1)
    assert graph.residual(drive, failed).min() > 10
    assert np.abs(one.activation - failed.activation).max() > 0.8
    np.testing.assert_allclose(two.activation, failed.activation, rtol=0, atol=1e-12)
    # A single half-step fallback also remains in a numerical orbit here.
    assert not graph.equilibrate(drive, budget=256, tolerance=3e-3, damping=1).qualified.any()

    before = snapshot(learner)
    model = graph.neuron_model
    learned, report = learner.step(drive, np.array([0, 1]))
    target = learner.targets(np.array([0, 1]))
    for name, state, nudge in (
        ("free", learned.free, None),
        ("nudged", learned.nudged, learner.nudge_for(target, config.beta)),
        ("opposite", learned.opposite, learner.nudge_for(target, -config.beta)),
    ):
        assert state is not None
        assert graph.residual(drive, state, nudge=nudge).max() <= 3e-3
        np.testing.assert_allclose(state.activation, model.activation(state.v), atol=1e-15)
        assert report[f"{name}_residual"] <= 3e-3
        assert report[f"{name}_steps"] == state.steps <= 256
        assert report[f"{name}_damping_halvings"] == 3
    assert report["total_steps"] == sum(report[f"{name}_steps"] for name in ("free", "nudged", "opposite"))
    assert report["qualified"] == report["accepted"] == 1
    assert learner.brain.neuron_model is model and model.dt == 1
    np.testing.assert_array_equal(learner.brain.efficacy, before["efficacy"])
    np.testing.assert_array_equal(learner.brain.bias, before["bias"])


def test_phase_report_counts_every_residual_transport(monkeypatch):
    graph = motor_graph()
    learner = cd.Learner(graph, np.arange(36), cd.LearnerConfig(
        qualified=True, free_steps=128, nudged_steps=128, tolerance=3e-3,
    ))
    checks = []
    residual = cd.NeuralGraph.residual

    def counted(candidate, *args, **kwargs):
        checks.append(candidate.neuron_model.dt)
        return residual(candidate, *args, **kwargs)

    monkeypatch.setattr(cd.NeuralGraph, "residual", counted)
    _, report = learner.step(np.full((1, 36), 0.2), np.array([0]))
    assert report["total_residual_checks"] == len(checks)
    assert report["total_steps"] <= 3 * 128
    assert {1.0, 0.5, 0.25, 0.125} <= set(checks)


@pytest.mark.parametrize("backend,precision", [
    ("cpu", None), ("torch", "float64"), ("torch", "float32"), ("mlx", None),
])
def test_signed_slotted_lesson_qualifies_each_original_phase(backend, precision):
    graph = motor_graph(backend, precision)
    config = cd.LearnerConfig(
        qualified=True, free_steps=512, nudged_steps=512, tolerance=3e-3,
        momentum=0.9, normalize=0.9,
    )
    learner = cd.Learner(graph, np.arange(36), config, slots=(12, 24))
    matrix = np.zeros((36, 36))
    np.add.at(matrix, (graph.connectome.pre, graph.connectome.post), graph.weights)
    bias = graph.bias.copy()
    drive = np.full((3, 36), 0.2)
    labels = np.array([[0, 23], [11, 0], [3, 10]])
    weight = np.array([0.0, 1.0, -0.5])
    learned, report = learner.step(drive, labels, weight=weight)

    # Evaluate the literal original equations using parameters from BEFORE
    # the lesson. This reference constructs each slot's weighted nudge itself.
    model = graph.neuron_model
    for name, state, sign in (
        ("free", learned.free, 0), ("nudged", learned.nudged, 1),
        ("opposite", learned.opposite, -1),
    ):
        assert state is not None
        emission = 1 / (1 + np.exp(-model.slope * (state.v - model.threshold)))
        relative = emission - model.rest_emission
        activity = np.maximum(relative, 0) / (1 - model.rest_emission)
        activity += model.leak * np.minimum(relative, 0) / model.rest_emission
        defect = activity @ matrix + drive + bias - state.v
        if sign:
            for slot, (start, end) in enumerate(((0, 12), (12, 36))):
                logits = activity[:, start:end] / config.temperature
                exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities = exponents / exponents.sum(axis=1, keepdims=True)
                target = np.eye(end - start)[labels[:, slot]]
                defect[:, start:end] += (
                    sign * config.beta / 2 * (target - probabilities) * weight[:, None]
                )
        assert np.abs(defect).max() <= config.tolerance
        np.testing.assert_allclose(state.activation, activity, atol=2e-6, rtol=0)
        assert state.steps == report[name + "_steps"] <= 512
        assert report[name + "_damping_halvings"] > 0
    assert report["qualified"] == report["accepted"] == 1
    assert report["attempted_presentations"] == report["accepted_presentations"] == 3
    assert report["total_row_sweeps"] == 3 * report["total_steps"] <= 3 * 3 * 512
    assert learner.updates == learner.contrast_updates == 1
    assert graph.neuron_model.dt == 1


@pytest.mark.parametrize("backend,precision", [
    ("cpu", None), ("torch", "float64"), ("torch", "float32"), ("mlx", None),
])
def test_one_unqualified_weighted_row_refuses_whole_slotted_lesson(backend, precision):
    graph = motor_graph(backend, precision)
    learner = cd.Learner(graph, np.arange(36), cd.LearnerConfig(
        qualified=True, free_steps=512, nudged_steps=0, tolerance=3e-3,
        momentum=0.9, normalize=0.9,
    ), slots=(12, 24))
    learner.velocity = np.full_like(learner.velocity, 0.2)
    learner.velocity_bias = np.full_like(learner.velocity_bias, -0.3)
    learner.second_moment = np.full_like(learner.second_moment, 0.4)
    learner.second_moment_bias = np.full_like(learner.second_moment_bias, 0.5)
    learner.updates, learner.contrast_updates = 3, 2
    before = snapshot(learner)
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.step(
            np.full((3, 36), 0.2), np.array([[0, 23], [11, 0], [3, 10]]),
            weight=np.array([0.0, 1.0, -0.5]),
        )
    failure = caught.value
    assert failure.phase == "nudged" and list(failure.phases) == ["free", "nudged"]
    assert failure.phases["free"].qualified.all()
    np.testing.assert_array_equal(failure.phases["nudged"].qualified, [True, False, False])
    assert failure.report["attempted_presentations"] == 3
    assert failure.report["accepted_presentations"] == failure.report["accepted"] == 0
    assert failure.report["nudged_steps"] == 0
    assert failure.report["total_steps"] == failure.phases["free"].steps <= 512
    assert_snapshot(learner, before)


@pytest.mark.parametrize("failed_phase", ["free", "nudged"])
def test_refusal_preserves_parameters_optimizer_and_warm_state(failed_phase):
    graph = cd.NeuralGraph(cd.Connectome.from_synapses(2, pre=[], post=[]), cd.learning_neuron_model())
    config = cd.LearnerConfig(
        qualified=True, free_steps=0, nudged_steps=0, tolerance=3e-3,
        momentum=0.9, normalize=0.9,
    )
    learner = cd.Learner(graph, [0, 1], config)
    # A failed lesson must preserve an existing history as well as zero history.
    learner.velocity = np.full_like(learner.velocity, 0.2)
    learner.velocity_bias = np.full_like(learner.velocity_bias, -0.3)
    learner.second_moment = np.full_like(learner.second_moment, 0.4)
    learner.second_moment_bias = np.full_like(learner.second_moment_bias, 0.5)
    learner.updates, learner.contrast_updates = 3, 2
    drive = np.full((1, 2), 0.1 if failed_phase == "free" else 0.0)
    # A stale cached activation must not substitute for the state equation.
    warm = cd.BrainState(np.zeros((1, 2), dtype=np.float32), np.full((1, 2), 0.9), np.zeros((1, 2)), 9)
    original_warm = [x.copy() for x in (warm.v, warm.activation, warm.adaptation)]
    before = snapshot(learner)
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.step(drive, np.array([1]), warm=warm)
    failure = caught.value
    assert failure.phase == failed_phase
    assert list(failure.phases) == (["free"] if failed_phase == "free" else ["free", "nudged"])
    assert failure.report["accepted"] == 0
    assert failure.report["qualified"] == 0
    assert failure.report["qualification_required"] == 1
    assert failure.report["total_steps"] == 0
    assert failure.report["total_residual_checks"] == len(failure.phases)
    assert failure.report[f"{failed_phase}_residual"] > config.tolerance
    assert_snapshot(learner, before)
    for actual, expected in zip((warm.v, warm.activation, warm.adaptation), original_warm, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for phase in failure.phases.values():
        assert phase.state.v.dtype == np.float64
        np.testing.assert_array_equal(phase.state.activation, graph.neuron_model.activation(phase.state.v))


def test_opposite_refusal_keeps_all_work_before_any_optimizer_write(monkeypatch):
    graph = motor_graph()
    learner = cd.Learner(graph, np.arange(36), cd.LearnerConfig(
        qualified=True, free_steps=128, nudged_steps=128, tolerance=3e-3,
        momentum=0.9, normalize=0.9,
    ))
    solve = learner._qualified_phase

    def opposite_budget(drive, state, budget, nudge=None):
        # Exhaust the opposite phase deliberately, after two genuine qualified
        # solves. The independent residual must prevent committing their contrast.
        return solve(drive, state, 0 if nudge is not None and nudge.beta < 0 else budget, nudge)

    monkeypatch.setattr(learner, "_qualified_phase", opposite_budget)
    before = snapshot(learner)
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.step(np.full((1, 36), 0.2), np.array([0]))
    failure = caught.value
    assert failure.phase == "opposite" and list(failure.phases) == ["free", "nudged", "opposite"]
    assert failure.report["qualified"] == 0 and failure.report["qualification_required"] == 1
    assert failure.phases["free"].qualified.all() and failure.phases["nudged"].qualified.all()
    assert not failure.phases["opposite"].qualified.all()
    assert failure.report["total_steps"] > 0
    assert failure.report["opposite_steps"] == 0
    assert failure.report["total_steps"] == sum(x.steps for x in failure.phases.values())
    assert_snapshot(learner, before)


@pytest.mark.parametrize("nudge", ["quadratic", "cross_entropy"])
@pytest.mark.parametrize("precision", [None, "float64", "float32"], ids=["cpu", "cuda64", "cuda32"])
def test_qualified_contrast_matches_independent_newton_finite_differences(nudge, precision):
    if precision is not None:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware unavailable")
    low_precision = precision == "float32"
    tolerance = 2e-7 if low_precision else 1e-13
    gradient_rtol = 3e-3 if low_precision else 2e-5
    gradient_atol = 2e-5 if low_precision else 5e-9
    connectome = cd.Connectome.from_synapses(
        3, pre=[0, 1, 0, 2, 1, 2], post=[1, 0, 2, 0, 2, 1],
        count=[1, 1, 2, 2, 3, 3], sign=[0.08] * 6,
    )
    graph = cd.NeuralGraph(
        connectome, cd.learning_neuron_model(gain=1.4, leak=1),
        log_gain=np.full(3, np.log(1.3)), bias=np.array([0.3, 0.2, 0.1]),
        backend="cpu" if precision is None else "torch",
        device=None if precision is None else "cuda:0", precision=precision,
    )
    config = cd.LearnerConfig(
        qualified=True, beta=1e-2 if low_precision else 1e-4, nudge=nudge, temperature=0.3,
        free_steps=256, nudged_steps=256, tolerance=tolerance, eta=0, eta_bias=0,
    )
    learner = cd.Learner(graph, [1, 2], config)
    drive = np.array([[0.3, 0.2, 0.4], [0.2, 0.1, 0.3]])
    labels = np.array([0, 1])
    learned, report = learner.step(drive, labels)
    contrast, bias_contrast = learner.contrast(learned.free, learned.nudged, learned.opposite)
    assert all(report[f"{name}_residual"] <= tolerance for name in ("free", "nudged", "opposite"))
    if precision is not None:
        assert learned.free.device["v"].device == torch.device("cuda:0")
        assert learned.free.device["v"].dtype == getattr(torch, precision)
    contact_gain = 1.4 * connectome.count * 1.3

    def loss(efficacy, bias):
        # This Newton solver does not call production transport, activation,
        # settle, residual, nudge, or contrast. Its only input is the declared
        # symmetric effective matrix and smooth tanh activity equation.
        matrix = np.zeros((3, 3))
        np.add.at(matrix, (connectome.post, connectome.pre), contact_gain * efficacy)
        v = np.zeros_like(drive)
        for _ in range(12):
            activity = np.tanh(v / 2)
            error = v - drive - bias - activity @ matrix.T
            if np.abs(error).max() < 1e-15:
                break
            for row in range(len(v)):
                jacobian = np.eye(3) - matrix * (0.5 * (1 - activity[row] ** 2))[None, :]
                v[row] -= np.linalg.solve(jacobian, error[row])
        assert np.abs(v - drive - bias - np.tanh(v / 2) @ matrix.T).max() < 1e-14
        outputs = np.tanh(v[:, [1, 2]] / 2)
        target = np.eye(2)[labels]
        if nudge == "quadratic":
            return 0.5 * np.square(outputs - target).sum(axis=1).mean()
        z = outputs / config.temperature
        z -= z.max(axis=1, keepdims=True)
        return -(target * (z - np.log(np.exp(z).sum(axis=1, keepdims=True)))).sum(axis=1).mean()

    scale = config.temperature if nudge == "cross_entropy" else 1.0
    for first, second in ((0, 1), (0, 2), (1, 2)):
        pair = ((connectome.pre == first) & (connectome.post == second)) | ((connectome.pre == second) & (connectome.post == first))
        up, down = graph.efficacy.copy(), graph.efficacy.copy()
        up[pair] += 1e-5
        down[pair] -= 1e-5
        negative_gradient = -(loss(up, graph.bias) - loss(down, graph.bias)) / 2e-5
        edge = np.flatnonzero(pair)[0]
        np.testing.assert_allclose(
            contact_gain[edge] * contrast[edge] / scale, negative_gradient,
            rtol=gradient_rtol, atol=gradient_atol,
        )
    for neuron in range(3):
        up, down = graph.bias.copy(), graph.bias.copy()
        up[neuron] += 1e-5
        down[neuron] -= 1e-5
        negative_gradient = -(loss(graph.efficacy, up) - loss(graph.efficacy, down)) / 2e-5
        np.testing.assert_allclose(
            bias_contrast[neuron] / scale, negative_gradient,
            rtol=gradient_rtol, atol=gradient_atol,
        )


def same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_composed_qualified_learning_saved_continuation_and_private_memory(tmp_path):
    brain = cd.Brain.compose(2, 36, modules=(4,), seed=0, learning=cd.LearnerConfig(
        qualified=True, free_steps=256, nudged_steps=256, tolerance=3e-3,
        momentum=0.9, normalize=0.9,
    ))
    cue = np.array([[0.2, 0.8]])
    brain.step(cue, teacher=np.array([0]))
    before = brain.save(tmp_path / "before")
    restored = cd.Brain.load(before)
    assert restored.learner.config == brain.learner.config
    assert restored.learner.config.qualified and restored.learner.config.damping == 3
    phases = brain.imagine([cue, cue * 0.5])
    assert all(phase.qualified.all() for phase in phases)
    same_checkpoint(before, brain.save(tmp_path / "after-imagination"))
    for owner in (brain, restored):
        owner.learn(np.array([0.4]), np.array([False]), cue * 0.5)
    np.testing.assert_array_equal(brain.act(cue * 0.5), restored.act(cue * 0.5))
    same_checkpoint(brain.save(tmp_path / "continued"), restored.save(tmp_path / "restored"))


def test_finite_default_retains_its_update_and_reports_unqualified_phase_errors():
    graph = motor_graph()
    config = cd.LearnerConfig(free_steps=12, nudged_steps=12, tolerance=3e-3, momentum=0.9)
    first, second = cd.Learner(graph, np.arange(36), config), cd.Learner(graph, np.arange(36), config)
    drive, labels = np.full((1, 36), 0.2), np.array([0])
    target = second.targets(labels)
    free = second.free(drive)
    plus = second.nudged(drive, free, target)
    minus = second.nudged(drive, free, target, sign=-1)
    free_error = graph.residual(drive, free).max()
    second.update(free, plus, minus)
    _, report = first.step(drive, labels)
    assert_snapshot(first, snapshot(second))
    assert report["qualified"] == 0 and report["accepted"] == 1
    assert report["free_residual"] == free_error > 3e-3
    assert report["total_steps"] == free.steps + plus.steps + minus.steps
    assert report["total_residual_checks"] == 3


@pytest.mark.parametrize("options", [{"qualified": 1}, {"qualified": True, "tolerance": None}, {"damping": True}, {"damping": -1}])
def test_qualified_learning_configuration_rejects_ambiguous_contract(options):
    with pytest.raises(ValueError):
        cd.LearnerConfig(**options)


def test_public_qualified_phases_have_their_own_refusal_diagnostics():
    graph = cd.NeuralGraph(cd.Connectome.from_synapses(2, pre=[], post=[]), cd.learning_neuron_model())
    learner = cd.Learner(graph, [0, 1], cd.LearnerConfig(
        qualified=True, free_steps=0, nudged_steps=0, tolerance=3e-3,
    ))
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.free(np.ones((1, 2)))
    assert caught.value.phase == "free" and caught.value.report["total_steps"] == 0
    drive = np.zeros((1, 2))
    free = learner.free(drive)
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.nudged(drive, free, learner.targets(np.array([0])), sign=-1)
    assert caught.value.phase == "opposite" and caught.value.report["total_residual_checks"] == 1
    assert learner.updates == learner.contrast_updates == 0


def test_unrepresentable_damping_is_rejected_before_doing_any_phase_work(monkeypatch):
    graph = motor_graph()

    def unexpected(*args, **kwargs):
        raise AssertionError("an invalid numerical recipe must not start solving")

    monkeypatch.setattr(cd.NeuralGraph, "residual", unexpected)
    with pytest.raises(ValueError, match="dt"):
        graph.equilibrate(np.ones((1, 36)), budget=1101, damping=1100)
