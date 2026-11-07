"""The MLX launch: a chunk of sweeps in one Metal dispatch. It settles as the CPU does within float32
rounding, free and under every nudge it takes, stops under a movement tolerance where the CPU stops,
checks a solve's equations on the device within rounding of the float64 reference, leaves a state
that continues and reads back without another launch, and hands what it does not take to one MLX
operation at a time."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module
from cadence.brain import _MLX_COMPARTMENT_NEURONS, Apical, _MlxPlan, _MlxState
from cadence.genome import Genome, Projection, Region, develop

pytestmark = pytest.mark.skipif("mlx" not in cd.available_backends(), reason="mlx not installed")

ROWS = 4
# what a launch's claims are read on: the step of one, and the half step of an apical compartment
APICAL = {"modules": (32, 16), "apical": 0.25, "recall": False}
FORMS = pytest.mark.parametrize("form", [{}, APICAL], ids=["whole_step", "apical"])


def pair(**settings):
    settings = {"modules": (32,), "seed": 3, **settings}
    return cd.Brain.compose(8, 4, **settings), cd.Brain.compose(8, 4, backend="mlx", **settings)


def drive_of(brain, rows=ROWS, seed=0):
    x = np.random.default_rng(seed).random((rows, 8))
    drive = np.zeros((rows, brain.connectome.n))
    drive[:, brain.sensory_index] = x * brain.brain.neuron_model.stimulus_amplitude
    return drive


def target_of(brain, rows=ROWS):
    target = np.zeros((rows, brain.connectome.n))
    target[np.arange(rows), brain.motor_index[np.arange(rows) % len(brain.motor_index)]] = 1.0
    return target


def launched(state):
    return isinstance(state.device, _MlxState) and state.__dict__.get("activation") is None


@pytest.mark.parametrize("steps", [1, 2, 5, 32, 33, 100])
@FORMS
def test_a_launch_settles_as_the_cpu_at_any_count_of_sweeps(steps, form):
    cpu, mlx = pair(**form)
    drive = drive_of(cpu)
    a = cpu.brain.settle_batch(drive, steps=steps)
    b = mlx.brain.settle_batch(drive, steps=steps)
    assert launched(b) and b.steps == steps
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.v, a.v, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.activity_change, a.activity_change, rtol=0, atol=1e-3)


@FORMS
def test_the_first_sweep_reads_the_sources_it_starts_from(form):
    cpu, mlx = pair(**form)
    earlier, later = drive_of(cpu, seed=1), drive_of(cpu, seed=2)
    a = cpu.brain.settle_batch(later, steps=3, state=cpu.brain.settle_batch(earlier, steps=20))
    b = mlx.brain.settle_batch(later, steps=3, state=mlx.brain.settle_batch(earlier, steps=20))
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.activity_change, a.activity_change, rtol=0, atol=1e-3)


@pytest.mark.parametrize("tolerance", [1e-1, 1e-2, 3e-3, 1e-3])
@FORMS
def test_a_movement_tolerance_stops_where_the_cpu_stops(tolerance, form):
    cpu, mlx = pair(**form)
    drive = drive_of(cpu)
    a = cpu.brain.settle_batch(drive, steps=400, tolerance=tolerance)
    b = mlx.brain.settle_batch(drive, steps=400, tolerance=tolerance)
    assert b.steps == a.steps
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    assert b.attempted_steps >= b.steps  # a launch past the stop counts as work


@pytest.mark.parametrize(
    "kind", ["softmax", "softmax_weighted", "quadratic", "quadratic_weighted", "slots"]
)
@FORMS
def test_a_launch_takes_the_nudges_of_a_lesson(kind, form):
    settings = {"slots": 2} if kind == "slots" else {}
    cpu, mlx = pair(**{**form, **settings})
    drive = drive_of(cpu)
    weight = np.array([1.0, 0.5, -0.5, 0.0]) if kind.endswith("weighted") else None
    target = target_of(cpu)

    def nudge_of(brain):
        learner = brain.learner
        if kind.startswith("quadratic"):
            return cd.Nudge(target, learner.output_mask, 0.1, weight=weight)
        return learner.nudge_for(target, 0.1, weight)

    free_a, free_b = cpu.brain.settle_batch(drive, steps=40), mlx.brain.settle_batch(drive, steps=40)
    a = cpu.brain.settle_batch(drive, steps=32, state=free_a, nudge=nudge_of(cpu))
    nudge = nudge_of(mlx)
    b = mlx.brain.settle_batch(drive, steps=32, state=free_b, nudge=nudge)
    assert launched(b)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        mlx.brain.residual(drive, b, nudge=nudge),
        mlx.brain.residual(drive, b, nudge=nudge, on_device=False), rtol=0, atol=1e-4,
    )


@FORMS
def test_a_solve_checks_on_the_device_and_counts_as_on_the_cpu(form):
    cpu, mlx = pair(**form)
    drive = drive_of(cpu)
    a = cpu.brain.equilibrate(drive, budget=512, tolerance=3e-3)
    b = mlx.brain.equilibrate(drive, budget=512, tolerance=3e-3)
    assert launched(b.state)
    assert (b.steps, b.residual_checks, b.state.steps) == (a.steps, a.residual_checks, a.state.steps)
    np.testing.assert_allclose(b.residual, a.residual, rtol=0, atol=1e-4)
    # the check that decided the solve is within rounding of the float64 reference
    np.testing.assert_allclose(
        b.residual, mlx.brain.residual(drive, b.state, on_device=False), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-4)


@FORMS
def test_a_state_within_tolerance_takes_no_sweep_and_one_check(form):
    cpu, mlx = pair(**form)
    drive = drive_of(cpu)
    settled = mlx.brain.equilibrate(drive, budget=512, tolerance=3e-3)
    again = mlx.brain.equilibrate(drive, state=settled.state, budget=512, tolerance=3e-3)
    assert (again.steps, again.residual_checks) == (0, 1)
    assert (again.residual <= 3e-3).all()
    np.testing.assert_allclose(again.state.activation, settled.state.activation, rtol=0, atol=0)
    assert not np.abs(again.state.activity_change).max()
    # a new drive on the sources alone: the settling neurons are at rest under the old products,
    # and the check still reads the sources' equations
    moved = mlx.brain.equilibrate(drive_of(cpu, seed=4), state=settled.state, budget=512, tolerance=3e-3)
    assert moved.steps > 0


@FORMS
def test_a_budget_of_no_sweeps_reads_the_start(form):
    cpu, mlx = pair(**form)
    drive = drive_of(cpu)
    a = cpu.brain.equilibrate(drive, budget=0, tolerance=3e-3)
    b = mlx.brain.equilibrate(drive, budget=0, tolerance=3e-3)
    assert (b.steps, b.residual_checks) == (0, 1)
    np.testing.assert_allclose(b.residual, a.residual, rtol=0, atol=1e-5)


def test_a_tolerance_below_the_floor_is_checked_by_the_float64_reference():
    _, mlx = pair()
    drive = drive_of(mlx)
    reference = mlx.brain.residual
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("on_device", True))
        return reference(*args, **kwargs)

    solved = mlx.brain.equilibrate(drive, budget=64, tolerance=1e-5)
    assert solved.residual_checks >= 2
    np.testing.assert_array_equal(
        solved.residual, reference(drive, solved.state, on_device=False)
    )
    mlx.brain.residual = counted  # a wrapped check sees every one, as on the compiled CPU path
    mlx.brain.equilibrate(drive, budget=64, tolerance=3e-3)
    assert calls


def test_an_unsettling_drive_reads_infinity_and_stops():
    connectome = cd.layered(8, 12, 4, density=1.0, seed=2)
    neuron_model = cd.learning_neuron_model(dt=1.0)
    mlx = cd.NeuralGraph(
        connectome, neuron_model, backend="mlx", efficacy=np.full(connectome.synapses, 3e38)
    )
    assert mlx._mlx.plan is not None
    solved = mlx.equilibrate(np.ones((2, connectome.n)), budget=320, tolerance=1e-3)
    assert not np.isfinite(solved.residual).any()
    assert solved.steps == 32  # the first chunk that reads no finite residual ends the solve


@FORMS
def test_a_state_continues_on_the_device_and_reads_back_without_a_launch(form):
    cpu, mlx = pair(**form)
    drive = drive_of(cpu)
    free = mlx.brain.settle_batch(drive, steps=30)
    handle = free.device
    assert launched(free) and handle["holder"] is mlx.brain._mlx.root

    def forbidden(*_, **__):
        raise AssertionError("a read launched")

    eval_ = mlx.brain._mlx.mx.eval
    mlx.brain._mlx.mx.eval = forbidden
    try:
        s, v = free.activation, free.v
    finally:
        mlx.brain._mlx.mx.eval = eval_
    assert s.dtype == np.float64 and s.shape == (ROWS, cpu.connectome.n) and v.shape == s.shape
    a = cpu.brain.settle_batch(drive, steps=30)
    np.testing.assert_allclose(s, a.activation, rtol=0, atol=1e-5)
    # the state's whole arrays on the device are the host's
    np.testing.assert_array_equal(np.array(handle["s"], dtype=np.float64), s)
    np.testing.assert_array_equal(np.array(handle["v"], dtype=np.float64), v)
    np.testing.assert_array_equal(np.array(handle["a"], dtype=np.float64), free.adaptation)
    # from the device and from the host, the same continuation
    target = target_of(cpu)
    nudge = mlx.learner.nudge_for(target, 0.1)
    on = mlx.brain.settle_batch(drive, steps=10, state=free, nudge=nudge)
    host = cd.BrainState(free.v, free.activation, free.adaptation, free.steps)
    off = mlx.brain.settle_batch(drive, steps=10, state=host, nudge=nudge)
    np.testing.assert_allclose(on.activation, off.activation, rtol=0, atol=1e-5)


def test_what_the_launch_does_not_take_runs_one_operation_at_a_time():
    cpu, mlx = pair()
    drive = drive_of(cpu)
    keep = np.ones(cpu.connectome.n)
    keep[3] = 0.0
    a = cpu.brain.settle_batch(drive, steps=12, mask=keep)
    b = mlx.brain.settle_batch(drive, steps=12, mask=keep)
    assert not launched(b)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    # a state of that path continues in a launch
    target = target_of(cpu)
    nudge = mlx.learner.nudge_for(target, 0.1)
    c = cpu.brain.settle_batch(drive, steps=5, state=a, nudge=cpu.learner.nudge_for(target, 0.1))
    d = mlx.brain.settle_batch(drive, steps=5, state=b, nudge=nudge)
    assert launched(d)
    np.testing.assert_allclose(d.activation, c.activation, rtol=0, atol=1e-5)
    # a temperature below the launch's floor, and a group that is not a range
    cold = cd.Nudge(target, cpu.learner.output_mask, 0.1, softmax_temperature=0.001)
    assert mlx.brain._mlx.plan.nudge_tables(cold) is None
    apart = np.full(cpu.connectome.n, -1)
    apart[cpu.motor_index[[0, 2]]] = 0
    apart[cpu.motor_index[[1, 3]]] = 1
    split = cd.Nudge(target, cpu.learner.output_mask, 0.1, softmax_temperature=0.2, groups=apart)
    assert mlx.brain._mlx.plan.nudge_tables(split) is None
    e = cpu.brain.settle_batch(drive, steps=5, nudge=split)
    f = mlx.brain.settle_batch(drive, steps=5, nudge=split)
    assert not launched(f)
    np.testing.assert_allclose(f.activation, e.activation, rtol=0, atol=1e-5)
    trajectory = mlx.brain.settle_batch(drive, steps=3, trajectory=True)
    assert trajectory.trajectory is not None and trajectory.trajectory.shape[0] == 3


def test_the_environment_keeps_one_operation_at_a_time(monkeypatch):
    monkeypatch.setenv("CADENCE_MLX_LAUNCH", "0")
    cpu, mlx = pair()
    assert mlx.brain._mlx.plan is None
    drive = drive_of(cpu)
    a = cpu.brain.settle_batch(drive, steps=12)
    b = mlx.brain.settle_batch(drive, steps=12)
    assert not launched(b)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)


def test_a_rule_the_launch_does_not_hold_has_no_plan():
    connectome = cd.layered(8, 12, 4, density=1.0, seed=2)
    adapting = cd.learning_neuron_model(dt=1.0).replace(adaptation=cd.Adaptation(tau_steps=15))
    assert cd.NeuralGraph(connectome, adapting, backend="mlx")._mlx.plan is None
    plain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0), backend="mlx")
    assert isinstance(plain._mlx.plan, _MlxPlan)


def test_a_composed_brain_lives_in_launches_as_on_the_cpu(tmp_path):
    cpu, mlx = pair(recall=True)
    rng = np.random.default_rng(5)
    acts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            x = rng.random((ROWS, 8))
            labels = rng.integers(4, size=ROWS)
            cpu.lesson(x, labels)
            mlx.lesson(x, labels)
            acts.append((cpu.act(x, greedy=True), mlx.act(x, greedy=True)))
    for a, b in acts:
        np.testing.assert_array_equal(a, b)
    assert isinstance(mlx.basal_ganglia.state.device, _MlxState)
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=1e-4)
    # a checkpoint of a brain whose state rests on the device continues within rounding
    mlx.save(tmp_path / "brain.npz")
    again = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    x = rng.random((ROWS, 8))
    np.testing.assert_array_equal(mlx.act(x, greedy=True), again.act(x, greedy=True))


def test_a_compiled_sweep_takes_every_feature_and_a_new_temperature_traces_nothing():
    connectome = cd.layered(8, 12, 4, density=1.0, seed=2)
    model = cd.learning_neuron_model(dt=0.5, leak=0.2).replace(
        adaptation=cd.Adaptation(tau_steps=15, strength=0.1)
    )
    cpu = cd.NeuralGraph(connectome, model)
    mlx = cd.NeuralGraph(connectome, model, backend="mlx")
    assert mlx._mlx.plan is None
    rng = np.random.default_rng(3)
    drive = cpu.stimulus_levels(rng.random((ROWS, connectome.n)) * 0.5)
    out = list(connectome.populations["output"])
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    target = np.zeros((ROWS, connectome.n))
    target[np.arange(ROWS), [out[i % len(out)] for i in range(ROWS)]] = 1.0
    for temperature in (0.2, 0.3, 0.5):
        nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=temperature)
        a = cpu.settle_batch(drive, steps=12, nudge=nudge)
        b = mlx.settle_batch(drive, steps=12, nudge=nudge)
        np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
        assert mlx._mlx.sweeps  # compiled, and one sweep for all three temperatures
    assert len(mlx._mlx.sweeps) == 1
    # a temperature below float32's least normal reads the host's ladder, one operation at a time
    cold = cd.Nudge(target, mask, 0.1, softmax_temperature=1e-39)
    np.testing.assert_allclose(
        mlx.settle_batch(drive, steps=4, nudge=cold).activation,
        cpu.settle_batch(drive, steps=4, nudge=cold).activation, rtol=0, atol=1e-4,
    )


# -- a lesson's update and a large address map on the device


@pytest.fixture
def on_device(monkeypatch):
    """The sizes from which the device takes a lesson's update and the address map, lowered to the
    test's brain."""
    from cadence import brain as brain_module
    from cadence import stream as stream_module

    monkeypatch.setattr(brain_module, "_MLX_UPDATE_ENTRIES", 1)
    monkeypatch.setattr(stream_module, "_MLX_MAP_ENTRIES", 1)


RECIPE = dict(recall=True, episodic=False)  # the memory a new brain composes, its gains on
# float32 contrasts near a synapse's noise, through RMS normalization: the backend's own bound
SYNAPSES, MAPS = 2e-2, 2e-2


def lives(cpu, mlx, lessons=6, seed=9):
    """The same moments and labels on both brains: lessons before acts, as a chamber teaches."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(lessons):
            x = rng.random((ROWS, 8))
            labels = rng.integers(4, size=ROWS)
            a, b = cpu.lesson(x, labels), mlx.lesson(x, labels)
            np.testing.assert_allclose(b["scale_step"], a["scale_step"], rtol=2e-2)
            np.testing.assert_allclose(b["bias_step"], a["bias_step"], rtol=2e-2)
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))


def test_a_lessons_update_on_the_device_moves_as_on_the_host(on_device):
    cpu, mlx = pair(**RECIPE)
    lives(cpu, mlx)
    graph = mlx.brain
    assert graph._efficacy is None and graph._mlx.scale is not None  # nothing read it yet
    np.testing.assert_allclose(graph.bias, cpu.brain.bias, rtol=0, atol=1e-4)
    np.testing.assert_allclose(graph.efficacy, cpu.brain.efficacy, rtol=0, atol=SYNAPSES)
    for name in ("velocity", "second_moment", "velocity_bias", "second_moment_bias"):
        np.testing.assert_allclose(
            getattr(mlx.learner, name), getattr(cpu.learner, name), rtol=0, atol=SYNAPSES,
        )
    assert graph.efficacy.dtype == np.float64 and graph.efficacy.shape == cpu.brain.efficacy.shape
    # read on the host, the moments go on from where the device left them
    lives(cpu, mlx, lessons=2, seed=10)
    assert mlx.learner.contrast_updates == cpu.learner.contrast_updates
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=SYNAPSES)


def test_ties_move_one_synapse_and_the_bound_holds(on_device):
    settings = dict(RECIPE)
    _, mlx = pair(**settings)
    assert len(mlx.learner._paired)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlx.lesson(np.random.default_rng(0).random((ROWS, 8)), np.arange(ROWS) % 4)
    assert mlx.brain._mlx.scale is not None
    efficacy = mlx.brain.efficacy
    np.testing.assert_array_equal(
        efficacy[mlx.learner._paired], efficacy[mlx.learner._paired_reverse]
    )
    assert np.abs(efficacy).max() <= mlx.learner.config.scale_cap


def test_a_refused_update_on_the_device_changes_nothing(on_device):
    _, mlx = pair(**RECIPE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlx.lesson(np.random.default_rng(0).random((ROWS, 8)), np.arange(ROWS) % 4)
        graph, moments = mlx.brain, mlx.learner.__dict__["_device_moments"]
        count = mlx.learner.contrast_updates
        kernel = graph._mlx
        kernel.mx.eval(kernel.blocks)
        real = kernel.mx.isfinite

        def broken(x):
            return kernel.mx.array(False)

        kernel.mx.isfinite = broken
        try:
            with pytest.raises(ValueError, match="finite"):
                mlx.lesson(np.random.default_rng(1).random((ROWS, 8)), np.arange(ROWS) % 4)
        finally:
            kernel.mx.isfinite = real
    assert mlx.brain is graph and mlx.learner.__dict__["_device_moments"] is moments
    assert mlx.learner.contrast_updates == count


def test_a_brain_saves_and_continues_from_the_device(on_device, tmp_path):
    cpu, mlx = pair(**RECIPE)
    lives(cpu, mlx, lessons=3)
    mlx.save(tmp_path / "brain.npz")
    again = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    np.testing.assert_allclose(again.brain.efficacy, mlx.brain.efficacy, rtol=0, atol=1e-7)
    rng = np.random.default_rng(12)
    x = rng.random((ROWS, 8))
    labels = rng.integers(4, size=ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlx.lesson(x, labels)
        again.lesson(x, labels)
    np.testing.assert_array_equal(mlx.act(x, greedy=True), again.act(x, greedy=True))
    np.testing.assert_allclose(again.brain.efficacy, mlx.brain.efficacy, rtol=0, atol=SYNAPSES)
    # brought back to the CPU from a checkpoint the device wrote, it reads there
    host = cd.Brain.load(tmp_path / "brain.npz")
    assert host.brain.backend == "cpu"
    np.testing.assert_allclose(host.brain.efficacy, again.brain.efficacy, rtol=0, atol=SYNAPSES)


def test_a_host_update_after_a_device_update_starts_from_what_the_device_left(on_device):
    cpu, mlx = pair(**RECIPE)
    lives(cpu, mlx, lessons=2)
    assert mlx.brain._efficacy is None
    step_a = np.full(cpu.connectome.synapses, 1e-3)
    step_b = np.zeros(cpu.connectome.n)
    cpu.learner.apply(step_a, step_b)
    mlx.learner.apply(step_a, step_b)
    assert mlx.brain._mlx.scale is None  # on the host now, its blocks made from there
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=SYNAPSES)
    lives(cpu, mlx, lessons=2, seed=11)
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=SYNAPSES)


def test_what_the_update_on_the_device_does_not_take_stays_on_the_host(on_device):
    cpu, mlx = pair(**RECIPE)
    assert mlx.learner._mlx_update_tables(*states(mlx)) is not None
    # an entry without a synapse: a sparse graph's update stays on the host
    connectome = cd.layered(8, 12, 4, density=0.5, seed=2)
    sparse = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0), backend="mlx"),
        connectome.populations["output"],
    )
    assert sparse.brain._mlx.update_tables(
        sparse.brain._gain_pre, sparse._paired, sparse._paired_reverse
    ) is None
    # a frozen synapse, or a rate of its own, likewise
    mlx.learner.plastic_synapses = np.array(mlx.learner.plastic_synapses)
    mlx.learner.plastic_synapses[0] = False
    assert mlx.learner._mlx_update_tables(*states(mlx)) is None


def states(brain):
    drive = drive_of(brain)
    free = brain.brain.settle_batch(drive, steps=2)
    return free, free


def test_a_large_address_map_rests_on_the_device_and_moves_as_on_the_host(on_device):
    from cadence.brain import _MlxMatrix

    cpu, mlx = pair(**RECIPE)
    lives(cpu, mlx)
    w_x, w_c, b = mlx.recall.head_map(0)
    assert isinstance(w_x, _MlxMatrix) and isinstance(w_c, _MlxMatrix)
    host_x, host_c, host_b = cpu.recall.head_map(0)
    assert w_x.shape == host_x.shape and w_x.size == host_x.size and w_x.ndim == 2
    np.testing.assert_allclose(np.asarray(w_x), host_x, rtol=0, atol=MAPS)
    np.testing.assert_allclose(np.asarray(w_c), host_c, rtol=0, atol=MAPS)
    np.testing.assert_allclose(b, host_b, rtol=0, atol=MAPS)
    assert np.abs(host_x).max() > 0  # the lessons moved it
    assert np.asarray(w_x).dtype == np.float64 and w_x.finite() and w_c.finite()
    np.testing.assert_array_equal(w_x[0], np.asarray(w_x)[0])  # it reads as a host array
    assert len(w_x) == len(host_x) and w_x.dtype == host_x.dtype
    assert not np.asarray(w_x).flags.writeable
    # the slots and the attention read the same moments
    assert mlx.parameters() == cpu.parameters()
    np.testing.assert_allclose(mlx.recall.keys, cpu.recall.keys, rtol=0, atol=1e-12)


def test_a_step_that_is_not_finite_is_refused_and_the_map_stays(on_device, monkeypatch):
    from cadence.brain import _MlxMatrix

    _, mlx = pair(**RECIPE)
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            x = rng.random((ROWS, 8))
            mlx.lesson(x, rng.integers(4, size=ROWS))
            mlx.act(x, greedy=True)
        kept = mlx.recall.head_map(0)

        def overflowing(self, other, x, c, step, rate):
            mx = self.mx
            return _MlxMatrix(mx, mx.full(self.array.shape, mx.inf, dtype=mx.float32)), other

        monkeypatch.setattr(_MlxMatrix, "stepped", overflowing)
        x = rng.random((ROWS, 8))
        report = mlx.lesson(x, rng.integers(4, size=ROWS))
    assert report["address_refused"] > 0 and report["address_rows"] == 0
    assert mlx.recall.head_map(0)[0] is kept[0] and mlx.recall.head_map(0)[1] is kept[1]


def test_a_checkpoint_of_a_large_map_reads_the_device(on_device, tmp_path):
    cpu, mlx = pair(**RECIPE)
    lives(cpu, mlx, lessons=3)
    mlx.save(tmp_path / "brain.npz")
    again = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    np.testing.assert_allclose(
        np.asarray(again.recall.head_map(0)[0]), np.asarray(mlx.recall.head_map(0)[0]),
        rtol=0, atol=1e-6,
    )
    x = np.random.default_rng(4).random((ROWS, 8))
    np.testing.assert_array_equal(mlx.act(x, greedy=True), again.act(x, greedy=True))


def test_more_settling_neurons_than_threads_share_a_thread():
    connectome = cd.layered(8, 1400, 6, density=1.0, seed=2)
    model = cd.learning_neuron_model(dt=1.0)
    cpu, mlx = cd.NeuralGraph(connectome, model), cd.NeuralGraph(connectome, model, backend="mlx")
    plan = mlx._mlx.plan
    assert plan is not None and plan.per == 2 and plan.threads * plan.per >= plan.m
    drive = cpu.stimulus_levels(np.random.default_rng(1).random((3, connectome.n)) * 0.5)
    out = list(connectome.populations["output"])
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    target = np.zeros((3, connectome.n))
    target[np.arange(3), out[:3]] = 1.0
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=0.2)
    a = cpu.settle_batch(drive, steps=20, nudge=nudge)
    b = mlx.settle_batch(drive, steps=20, nudge=nudge)
    assert launched(b)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-4)


def test_a_mask_changed_in_place_is_read_anew():
    cpu, mlx = pair()
    drive = drive_of(cpu)
    mask = np.array(mlx.learner.output_mask, dtype=float)
    target = target_of(cpu)
    plan = mlx.brain._mlx.plan
    first = plan.nudge_tables(cd.Nudge(target, mask, 0.1, softmax_temperature=0.2))
    mask[np.flatnonzero(mask)[0]] = 0.5
    second = plan.nudge_tables(cd.Nudge(target, mask, 0.1, softmax_temperature=0.2))
    assert first is not second
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=0.2)
    a = cpu.brain.settle_batch(drive, steps=20, nudge=nudge)
    b = mlx.brain.settle_batch(drive, steps=20, nudge=nudge)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)


def test_a_deep_copy_lives_apart_on_the_device(on_device):
    import copy

    _, mlx = pair(**RECIPE)
    rng = np.random.default_rng(6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = rng.random((ROWS, 8))
        labels = rng.integers(4, size=ROWS)
        mlx.lesson(x, labels)
        mlx.act(x, greedy=True)
        twin = copy.deepcopy(mlx)
        x2 = rng.random((ROWS, 8))
        labels2 = rng.integers(4, size=ROWS)
        before = mlx.brain._mlx.version
        twin.lesson(x2, labels2)
        assert mlx.brain._mlx.version == before  # the copy's lesson leaves the original's kernel
        np.testing.assert_array_equal(twin.act(x2, greedy=True).shape, (ROWS,))
        mlx.lesson(x2, labels2)
        np.testing.assert_allclose(mlx.brain.efficacy, twin.brain.efficacy, rtol=0, atol=1e-6)


def test_a_launchs_check_holds_for_the_parameters_it_read(on_device):
    cpu, mlx = pair(**RECIPE)
    drive = drive_of(cpu)
    settled = mlx.brain.equilibrate(drive, budget=512, tolerance=3e-3)
    held = mlx.brain.residual(drive, settled.state)
    np.testing.assert_array_equal(held, settled.residual)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlx.lesson(np.random.default_rng(1).random((ROWS, 8)), np.arange(ROWS) % 4)
    # under new parameters the state's equations are read anew, by the float64 reference
    np.testing.assert_array_equal(
        mlx.brain.residual(drive, settled.state),
        mlx.brain.residual(drive, settled.state, on_device=False),
    )


def test_a_long_chunk_is_solved_one_launch_at_a_time_elsewhere():
    cpu, mlx = pair()
    drive = drive_of(cpu)
    a = cpu.brain.equilibrate(drive, budget=512, chunk=128, tolerance=3e-3)
    b = mlx.brain.equilibrate(drive, budget=512, chunk=128, tolerance=3e-3)
    assert (b.steps, b.residual_checks) == (a.steps, a.residual_checks)
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-4)


def test_the_environment_keeps_the_update_and_the_map_on_the_host(on_device, monkeypatch):
    monkeypatch.setenv("CADENCE_MLX_LAUNCH", "0")
    cpu, mlx = pair(**RECIPE)
    lives(cpu, mlx, lessons=2)
    assert mlx.brain._mlx.plan is None and mlx.brain._mlx.scale is None
    assert mlx.brain._efficacy is not None
    assert isinstance(mlx.recall.head_map(0)[0], np.ndarray)


def test_the_routes_gains_and_the_grid_learn_on_the_device_as_on_the_host(on_device):
    settings = dict(RECIPE, history_decays=(0.5, 0.9), working_memory_amplitude=0.0)
    cpu, mlx = pair(**settings)
    assert mlx.recall.grid and mlx.recall.drive_gain and mlx.carried_gains
    assert set(mlx.carried_gains) == set(cpu.carried_gains)
    lives(cpu, mlx, lessons=6)
    assert mlx.brain._efficacy is None  # the routes' lessons read the blocks, not the host
    for name, gain in cpu.carried_gains.items():
        assert mlx.carried_gains[name] == pytest.approx(gain, abs=5e-3)
    for kept, host in zip(mlx.recall.grid_maps[0], cpu.recall.grid_maps[0], strict=True):
        np.testing.assert_allclose(kept, host, rtol=0, atol=MAPS)
    # a long stream reads past the position table, where the grid answers
    rng = np.random.default_rng(13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(40):
            x = rng.random((ROWS, 8))
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
    assert mlx.recall.slots > mlx.recall.position


def test_a_default_brain_composes_and_lives_on_the_device(tmp_path):
    """The memory a new brain holds, its blocks at any width and a checkpoint of it."""
    cpu, mlx = cd.Brain.compose(64, 8, seed=2), cd.Brain.compose(64, 8, seed=2, backend="mlx")
    assert mlx.recall is not None and mlx.carried_gains
    rng = np.random.default_rng(14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            x = rng.random((ROWS, 64))
            labels = rng.integers(8, size=ROWS)
            cpu.lesson(x, labels)
            mlx.lesson(x, labels)
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
    mlx.save(tmp_path / "brain.npz")
    again = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    x = rng.random((ROWS, 64))
    np.testing.assert_array_equal(mlx.act(x, greedy=True), again.act(x, greedy=True))
    assert again.carried_gains == mlx.carried_gains


def test_a_default_brain_at_the_speech_size_composes_on_the_blocks():
    brain = cd.Brain.compose(3920, 37, modules=(512,), seed=1, backend="mlx")
    assert brain.brain._mlx is not None and brain.brain._blocked
    x = np.random.default_rng(0).random((1, 3920)) * 0.25
    assert np.asarray(brain.act(x, greedy=True)).shape == (1,)


# -- an apical compartment: the half step, the sources on their way and the pools' gains

def apical_form(form, backend="cpu", seed=3):
    inputs, actions, settings = APICAL_FORMS[form]
    return cd.Brain.compose(
        inputs, actions, seed=seed, backend=backend, apical=0.25, teaching="deep", **settings
    )


APICAL_FORMS = {
    "depth": (48, 7, {"modules": (128, 64, 64, 64), "recall": False}),
    "depth_memory": (48, 7, {"modules": (128, 64, 64, 64)}),
    "field": ((16, 16, 1), 6, {"modules": (64,), "features": 8, "field": 5, "recall": False}),
    "code": (64, 8, {"modules": (64, 32), "code": 128}),
    "learned_code": (64, 8, {"modules": (64, 32), "learned_code": 64}),
}
# Where a synapse's velocity stands from here, its step reads within rounding: below it a contrast
# is float32's noise, and RMS normalization makes a step of it whichever its sign (the torch
# backend at float32 parts from float64 the same way).
CONTRAST = 1e-6


def observation(brain_inputs, rng, rows=ROWS):
    shape = (rows, *brain_inputs) if isinstance(brain_inputs, tuple) else (rows, brain_inputs)
    return rng.random(shape) * 0.5


def read_a_lesson_alike(cpu, mlx, before):
    """One lesson's step from one set of weights: the raw contrast within rounding everywhere, the
    step within rounding wherever the contrast stands above it, the backend's bound elsewhere."""
    velocity = np.asarray(cpu.learner.velocity)
    np.testing.assert_allclose(np.asarray(mlx.learner.velocity), velocity, rtol=0, atol=1e-7)
    standing = np.abs(velocity) >= CONTRAST
    assert standing.any()
    step, device = cpu.brain.efficacy - before, mlx.brain.efficacy - before
    np.testing.assert_allclose(device[standing], step[standing], rtol=0, atol=3e-5)
    np.testing.assert_allclose(device, step, rtol=0, atol=SYNAPSES)


@pytest.mark.parametrize("form", list(APICAL_FORMS))
def test_each_apical_form_settles_learns_and_lives_as_the_cpu(form):
    cpu, mlx = apical_form(form), apical_form(form, "mlx")
    inputs = APICAL_FORMS[form][0]
    actions = APICAL_FORMS[form][1]
    plan = mlx.brain._mlx.plan
    assert isinstance(plan, _MlxPlan) and plan.pools == len(cpu.brain.apical.members)
    np.testing.assert_array_equal(mlx.brain.efficacy, cpu.brain.efficacy)
    rng = np.random.default_rng(5)
    # a settle, with the gains it leaves
    x = observation(inputs, rng)
    drive = cpu.stimulus(x, memory=False)
    a = cpu.brain.equilibrate(drive, budget=512, tolerance=3e-3)
    b = mlx.brain.equilibrate(drive, budget=512, tolerance=3e-3)
    assert launched(b.state)
    assert (b.steps, b.residual_checks) == (a.steps, a.residual_checks)
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.residual, a.residual, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        mlx.brain.apical_gain(drive, b.state), cpu.brain.apical_gain(drive, a.state),
        rtol=0, atol=1e-5,
    )
    # a continued life, on one set of weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(12):
            x = observation(inputs, rng)
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
            np.testing.assert_allclose(
                mlx.basal_ganglia.state.activation, cpu.basal_ganglia.state.activation,
                rtol=0, atol=2e-5,
            )
    # a lesson
    before = np.array(cpu.brain.efficacy)
    x, labels = observation(inputs, rng), rng.integers(actions, size=ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, q = cpu.lesson(x, labels), mlx.lesson(x, labels)
    assert (q["free_steps"], q["nudged_steps"]) == (p["free_steps"], p["nudged_steps"])
    np.testing.assert_allclose(q["scale_step"], p["scale_step"], rtol=2e-2)
    read_a_lesson_alike(cpu, mlx, before)
    np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))


def test_a_form_past_the_threads_shares_them_with_its_compartment():
    """The field form holds more settling neurons than threads: a thread reads its pools' gains
    once each."""
    _, mlx = apical_form("field"), apical_form("field", "mlx")
    plan = mlx.brain._mlx.plan
    assert 1024 < plan.m <= _MLX_COMPARTMENT_NEURONS and plan.per == 2


def test_a_nudged_settle_holds_the_gains_of_the_free_phase():
    cpu, mlx = pair(**APICAL)
    drive = drive_of(cpu)
    free_a = cpu.brain.equilibrate(drive, budget=512, tolerance=3e-3).state
    free_b = mlx.brain.equilibrate(drive, budget=512, tolerance=3e-3).state
    held_a = cpu.brain.apical_gain(drive, free_a)
    eval_ = mlx.brain._mlx.mx.eval

    def forbidden(*_, **__):
        raise AssertionError("a read launched")

    mlx.brain._mlx.mx.eval = forbidden  # the launch that made the state read them
    try:
        held_b = mlx.brain.apical_gain(drive, free_b)
    finally:
        mlx.brain._mlx.mx.eval = eval_
    assert held_b.shape == held_a.shape == (ROWS, len(cpu.brain.apical.members))
    np.testing.assert_allclose(held_b, held_a, rtol=0, atol=1e-5)
    target = target_of(cpu)
    nudge_a = cpu.learner.nudge_for(target, 0.1, None, held_a)
    nudge_b = mlx.learner.nudge_for(target, 0.1, None, held_b)
    a = cpu.brain.settle_batch(drive, steps=32, state=free_a, nudge=nudge_a)
    b = mlx.brain.settle_batch(drive, steps=32, state=free_b, nudge=nudge_b)
    assert launched(b)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        mlx.brain.residual(drive, b, nudge=nudge_b),
        mlx.brain.residual(drive, b, nudge=nudge_b, on_device=False), rtol=0, atol=1e-4,
    )
    # the settle kept to them: it moved off the gains its own state would read
    assert np.abs(cpu.brain.apical_gain(drive, a) - held_a).max() > 0.0
    # a state that keeps none reads them in a launch of no sweeps, and so does one from the host
    np.testing.assert_allclose(
        mlx.brain.apical_gain(drive, b), cpu.brain.apical_gain(drive, a), rtol=0, atol=1e-5,
    )
    host = cd.BrainState(free_b.v, free_b.activation, free_b.adaptation, free_b.steps)
    np.testing.assert_allclose(mlx.brain.apical_gain(drive, host), held_a, rtol=0, atol=1e-5)
    other = drive_of(cpu, seed=7)  # and a state read under another drive than its own
    np.testing.assert_allclose(
        mlx.brain.apical_gain(other, free_b), cpu.brain.apical_gain(other, free_a),
        rtol=0, atol=1e-5,
    )
    with pytest.raises(ValueError, match="batch"):
        mlx.brain.apical_gain(other[:2], free_b)


def half_step(backend="cpu"):
    connectome = cd.layered(8, 12, 4, density=1.0, seed=2)
    graph = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=0.5), backend=backend)
    rng = np.random.default_rng(4)
    return graph, rng.random((3, connectome.n)) * 0.5, rng.random((3, connectome.n)) * 0.5


@pytest.mark.parametrize("steps", [1, 3, 24, 25, 40, 64, 65, 130])
def test_another_step_reads_the_sources_on_their_way_to_the_drive(steps):
    """Under dt = 0.5 a source range moves for sweeps on end: each of them reads its own products,
    across launches and from a state under another drive."""
    cpu, one, other = half_step()
    mlx, *_ = half_step("mlx")
    assert mlx._mlx.plan is not None and mlx._mlx.plan.rho == 0.5
    a = cpu.settle_batch(one, steps=steps)
    b = mlx.settle_batch(one, steps=steps)
    assert launched(b)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.v, a.v, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.activity_change, a.activity_change, rtol=0, atol=1e-3)
    c = cpu.settle_batch(other, steps=steps, state=a)
    d = mlx.settle_batch(other, steps=steps, state=b)
    assert launched(d)
    np.testing.assert_allclose(d.activation, c.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(d.v, c.v, rtol=0, atol=1e-5)
    np.testing.assert_allclose(d.activity_change, c.activity_change, rtol=0, atol=1e-3)


@pytest.mark.parametrize("tolerance", [3e-1, 1e-1, 1e-2, 1e-3])
def test_another_step_stops_where_the_cpu_stops_and_goes_on_from_there(tolerance):
    cpu, one, other = half_step()
    mlx, *_ = half_step("mlx")
    a = cpu.settle_batch(one, steps=400, tolerance=tolerance)
    b = mlx.settle_batch(one, steps=400, tolerance=tolerance)
    assert b.steps == a.steps
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.v, a.v, rtol=0, atol=1e-5)
    # a stop inside the sources' way: the next settle starts from where they were left
    c = cpu.settle_batch(other, steps=7, state=a)
    d = mlx.settle_batch(other, steps=7, state=b)
    np.testing.assert_allclose(d.activation, c.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(d.v, c.v, rtol=0, atol=1e-5)
    # under the drive it stopped at, too
    e = cpu.settle_batch(one, steps=7, state=a)
    f = mlx.settle_batch(one, steps=7, state=b)
    np.testing.assert_allclose(f.activation, e.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(f.v, e.v, rtol=0, atol=1e-5)


def test_another_steps_solve_reads_the_sources_equations():
    cpu, one, other = half_step()
    mlx, *_ = half_step("mlx")
    a = cpu.equilibrate(one, budget=512, tolerance=1e-3)
    b = mlx.equilibrate(one, budget=512, tolerance=1e-3)
    assert launched(b.state)
    assert (b.steps, b.residual_checks) == (a.steps, a.residual_checks)
    np.testing.assert_allclose(b.residual, a.residual, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        b.residual, mlx.residual(one, b.state, on_device=False), rtol=0, atol=1e-5
    )
    c = cpu.equilibrate(other, state=a.state, budget=512, tolerance=1e-3)
    d = mlx.equilibrate(other, state=b.state, budget=512, tolerance=1e-3)
    assert (d.steps, d.residual_checks) == (c.steps, c.residual_checks)
    np.testing.assert_allclose(d.state.activation, c.state.activation, rtol=0, atol=1e-5)


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_a_graph_with_no_source_range_launches(dt):
    genome = Genome((Region("a", 8), Region("b", 8)), (Projection("a", "b", reciprocal=True),))
    connectome = develop(genome, seed=0)
    model = cd.learning_neuron_model(dt=dt)
    cpu, mlx = cd.NeuralGraph(connectome, model), cd.NeuralGraph(connectome, model, backend="mlx")
    assert mlx._mlx.plan is not None and not mlx._mlx.plan.sources
    drive = np.random.default_rng(2).random((ROWS, connectome.n)) * 0.5
    a = cpu.equilibrate(drive, budget=256, tolerance=1e-3)
    b = mlx.equilibrate(drive, budget=256, tolerance=1e-3)
    assert launched(b.state) and b.steps == a.steps
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-5)


def test_the_sources_way_made_in_parts_reads_the_same_bits(monkeypatch):
    mlx, one, other = half_step("mlx")
    whole = mlx.settle_batch(other, steps=40, state=mlx.settle_batch(one, steps=40))
    monkeypatch.setattr(brain_module, "_MLX_RAMP_FLOATS", 1)
    parts = mlx.settle_batch(other, steps=40, state=mlx.settle_batch(one, steps=40))
    np.testing.assert_array_equal(parts.activation, whole.activation)
    np.testing.assert_array_equal(parts.v, whole.v)
    np.testing.assert_array_equal(parts.activity_change, whole.activity_change)


def moments_alike(cpu, mlx, inputs, actions, atol=SYNAPSES):
    rng = np.random.default_rng(11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            x, labels = rng.random((ROWS, inputs)), rng.integers(actions, size=ROWS)
            cpu.lesson(x, labels)
            mlx.lesson(x, labels)
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=atol)


def test_a_compartment_past_threadgroup_memory_takes_the_compiled_sweep():
    settings = {"modules": (_MLX_COMPARTMENT_NEURONS + 8,), "apical": 0.25, "recall": False}
    cpu, mlx = pair(**settings)
    assert mlx.brain._mlx.plan is None
    drive = drive_of(cpu)
    a = cpu.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    b = mlx.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    assert not launched(b.state) and b.steps == a.steps
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-5)
    held = cpu.brain.apical_gain(drive, a.state)
    np.testing.assert_allclose(mlx.brain.apical_gain(drive, b.state), held, rtol=0, atol=1e-5)
    target = target_of(cpu)
    c = cpu.brain.settle_batch(
        drive, steps=16, state=a.state, nudge=cpu.learner.nudge_for(target, 0.1, None, held)
    )
    d = mlx.brain.settle_batch(
        drive, steps=16, state=b.state, nudge=mlx.learner.nudge_for(target, 0.1, None, held)
    )
    np.testing.assert_allclose(d.activation, c.activation, rtol=0, atol=1e-5)
    assert mlx.brain._mlx.sweeps  # compiled
    moments_alike(cpu, mlx, 8, 4)


def test_the_environment_keeps_a_compartment_one_operation_at_a_time(monkeypatch):
    monkeypatch.setenv("CADENCE_MLX_LAUNCH", "0")
    cpu, mlx = pair(**APICAL)
    assert mlx.brain._mlx.plan is None and not mlx.brain._mlx.sweeps
    drive = drive_of(cpu)
    a = cpu.brain.settle_batch(drive, steps=12)
    b = mlx.brain.settle_batch(drive, steps=12)
    assert not launched(b) and not mlx.brain._mlx.sweeps
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    moments_alike(cpu, mlx, 8, 4)


def test_a_block_of_both_compartments_takes_the_compiled_sweep():
    """A spec that puts part of a block on the apical side: its two sides in blocks of their own,
    and a lesson's update on the host, where the blocks are the synapses."""
    connectome = cd.layered(8, 12, 4, density=1.0, seed=2)
    model = cd.learning_neuron_model(dt=0.5)
    groups = np.full(connectome.n, -1)
    groups[list(connectome.populations["output"])] = 0
    hidden = [i for i in range(connectome.n) if i not in connectome.populations["output"]
              and i not in connectome.populations["input"]]
    groups[hidden] = 1
    post = np.asarray(connectome.post)
    synapses = (groups[post] >= 0) & (np.arange(connectome.synapses) % 2 == 0)
    spec = Apical(synapses, groups, 0.25)
    cpu = cd.NeuralGraph(connectome, model, apical=spec)
    mlx = cd.NeuralGraph(connectome, model, backend="mlx", apical=spec)
    assert mlx._mlx.mixed and mlx._mlx.plan is None
    drive = np.random.default_rng(6).random((ROWS, connectome.n)) * 0.5
    a = cpu.settle_batch(drive, steps=24)
    b = mlx.settle_batch(drive, steps=24)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    held = cpu.apical_gain(drive, a)
    np.testing.assert_allclose(mlx.apical_gain(drive, b), held, rtol=0, atol=1e-5)
    mask = np.zeros(connectome.n)
    mask[list(connectome.populations["output"])] = 1.0
    target = np.zeros((ROWS, connectome.n))
    nudge = cd.Nudge(target, mask, 0.1, apical_gain=held)
    np.testing.assert_allclose(
        mlx.settle_batch(drive, steps=8, state=b, nudge=nudge).activation,
        cpu.settle_batch(drive, steps=8, state=a, nudge=nudge).activation, rtol=0, atol=1e-5,
    )
    empty = np.zeros(0, int)
    assert mlx._mlx.update_tables(np.ones(connectome.synapses), empty, empty) is None


def test_the_speech_size_with_a_compartment_settles_and_learns_as_the_cpu():
    settings = {"modules": (256, 128), "apical": 0.25, "teaching": "deep", "seed": 3}
    cpu = cd.Brain.compose(3920, 37, **settings)
    mlx = cd.Brain.compose(3920, 37, backend="mlx", **settings)
    assert mlx.brain._mlx.plan is not None and mlx.brain._mlx.plan.pools
    rng = np.random.default_rng(5)
    x = rng.random((ROWS, 3920)) * 0.5
    drive = cpu.stimulus(x, memory=False)
    a = cpu.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    b = mlx.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    assert (b.steps, b.residual_checks) == (a.steps, a.residual_checks)
    np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        mlx.brain.apical_gain(drive, b.state), cpu.brain.apical_gain(drive, a.state),
        rtol=0, atol=1e-5,
    )
    with warnings.catch_warnings():  # a continued life, on one set of weights
        warnings.simplefilter("ignore")
        for _ in range(4):
            x = rng.random((ROWS, 3920)) * 0.5
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
            np.testing.assert_allclose(
                mlx.basal_ganglia.state.activation, cpu.basal_ganglia.state.activation,
                rtol=0, atol=2e-5,
            )
    before = np.array(cpu.brain.efficacy)
    labels = rng.integers(37, size=ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, q = cpu.lesson(x, labels), mlx.lesson(x, labels)
    assert (q["free_steps"], q["nudged_steps"]) == (p["free_steps"], p["nudged_steps"])
    assert mlx.brain._efficacy is None  # the update ran on the device
    read_a_lesson_alike(cpu, mlx, before)
    np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))


def test_a_lessons_rounding_on_a_compartment_is_float32s_own(tmp_path):
    """A lesson at float32 parts from float64 where the contrasts lie at float32's noise. The torch
    backend at float32 on the CPU, with no MLX in it, parts by as much."""
    pytest.importorskip("torch")
    cpu = apical_form("depth")
    cpu.save(tmp_path / "brain.npz")
    mlx = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    witness = cd.Brain.load(
        tmp_path / "brain.npz", backend="torch", device="cpu", precision="float32"
    )
    rng = np.random.default_rng(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            x, labels = observation(48, rng), rng.integers(7, size=ROWS)
            for brain in (cpu, mlx, witness):
                brain.lesson(x, labels)
    apart = np.abs(mlx.brain.efficacy - cpu.brain.efficacy).max()
    alone = np.abs(witness.brain.efficacy - cpu.brain.efficacy).max()
    assert 0.0 < apart < 2 * alone and alone < 2 * apart


def test_an_apical_brain_saves_and_goes_on_from_the_device(on_device, tmp_path):
    cpu, mlx = pair(**APICAL)
    moments_alike(cpu, mlx, 8, 4)
    rng = np.random.default_rng(12)
    x = rng.random((ROWS, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlx.lesson(x, rng.integers(4, size=ROWS))
    assert mlx.brain._efficacy is None  # what the save reads rests on the device
    mlx.act(x, greedy=True)
    mlx.save(tmp_path / "brain.npz")
    again = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    back = cd.Brain.load(tmp_path / "brain.npz", backend="cpu")
    assert again.brain.apical.fraction == back.brain.apical.fraction == 0.25
    for _ in range(3):
        x = rng.random((ROWS, 8))
        expected = mlx.act(x, greedy=True)
        np.testing.assert_array_equal(again.act(x, greedy=True), expected)
        np.testing.assert_array_equal(back.act(x, greedy=True), expected)
        kept = mlx.basal_ganglia.state.activation
        np.testing.assert_allclose(again.basal_ganglia.state.activation, kept, rtol=0, atol=1e-5)
        np.testing.assert_allclose(back.basal_ganglia.state.activation, kept, rtol=0, atol=1e-5)


# -- a reward on the device

def outcomes(cpu, mlx, moments=6, seed=21, rows=ROWS, done_at=None):
    """The same observations and rewards on both brains, sampled actions and their outcomes."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x = rng.random((rows, 8))
            reward = rng.random(rows) - 0.5
            done = np.zeros(rows, bool)
            if done_at == t:
                done[0] = True
            a = cpu.step(x, reward=reward, done=done) if t else cpu.step(x)
            b = mlx.step(x, reward=reward, done=done) if t else mlx.step(x)
            np.testing.assert_array_equal(a, b)


def test_a_rewards_eligibility_and_step_rest_on_the_device(on_device):
    cpu, mlx = pair(recall=False)
    outcomes(cpu, mlx, done_at=3)
    agent, host = mlx.basal_ganglia, cpu.basal_ganglia
    assert agent._trace_mlx is not None and agent.trace is None and agent.trace_bias is None
    assert mlx.brain._efficacy is None and mlx.brain._bias is None  # nothing read the parameters
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=1e-4)
    np.testing.assert_allclose(mlx.brain.bias, cpu.brain.bias, rtol=0, atol=1e-4)
    np.testing.assert_allclose(agent.w_critic, host.w_critic, rtol=0, atol=1e-4)
    for name in ("trace", "scale_step", "bias_step"):
        assert mlx.last_learning[name] == pytest.approx(cpu.last_learning[name], abs=1e-4), name
    # a moment that adds no eligibility fades it, on the host, and a finished row forgets it
    ended = np.zeros(ROWS, bool)
    ended[1] = True
    agent.fade(ended), host.fade(ended)
    assert agent._trace_mlx[2][1] == 0.0
    agent._host_trace()
    assert agent._trace_mlx is None
    np.testing.assert_allclose(agent.trace, host.trace, rtol=0, atol=1e-3)
    np.testing.assert_allclose(agent.trace_bias, host.trace_bias, rtol=0, atol=1e-3)
    outcomes(cpu, mlx, moments=3, seed=22)  # from the host's back onto the device
    assert agent._trace_mlx is not None and agent.trace is None
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=1e-3)


def test_a_refused_reward_on_the_device_changes_nothing(on_device):
    _, mlx = pair(recall=False)
    rng = np.random.default_rng(23)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(3):
            x = rng.random((ROWS, 8))
            mlx.step(x, reward=np.ones(ROWS)) if t else mlx.step(x)
        agent, graph = mlx.basal_ganglia, mlx.brain
        before = (agent._trace_mlx, agent.w_critic.copy(), graph.efficacy.copy())
        kernel = graph._mlx
        original = kernel.rewarded

        def refused(*args, **kwargs):
            raise ValueError("steps must be finite vectors, one per synapse and one per neuron")

        kernel.rewarded = refused
        try:
            with pytest.raises(ValueError, match="finite"):
                mlx.step(rng.random((ROWS, 8)), reward=np.ones(ROWS))
        finally:
            kernel.rewarded = original
    assert agent._trace_mlx is before[0]
    np.testing.assert_array_equal(agent.w_critic, before[1])
    np.testing.assert_array_equal(mlx.brain.efficacy, before[2])


def test_a_reward_on_the_device_saves_and_goes_on(on_device, tmp_path):
    cpu, mlx = pair(recall=False)
    outcomes(cpu, mlx)
    assert mlx.basal_ganglia._trace_mlx is not None
    mlx.save(tmp_path / "brain.npz")
    again = cd.Brain.load(tmp_path / "brain.npz", backend="mlx")
    back = cd.Brain.load(tmp_path / "brain.npz", backend="cpu")
    host = cpu.basal_ganglia
    np.testing.assert_allclose(back.basal_ganglia.trace_bias, host.trace_bias, rtol=0, atol=1e-3)
    np.testing.assert_allclose(back.basal_ganglia.trace, host.trace, rtol=0, atol=1e-3)
    rng = np.random.default_rng(24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            x, reward = rng.random((ROWS, 8)), rng.random(ROWS)
            np.testing.assert_array_equal(mlx.step(x, reward=reward), again.step(x, reward=reward))
    assert again.basal_ganglia._trace_mlx is not None
    np.testing.assert_allclose(again.brain.efficacy, mlx.brain.efficacy, rtol=0, atol=1e-4)


def test_a_reward_below_the_device_size_stays_on_the_host():
    cpu, mlx = pair(recall=False)
    outcomes(cpu, mlx)
    assert mlx.basal_ganglia._trace_mlx is None and mlx.basal_ganglia.trace is not None
    np.testing.assert_allclose(mlx.brain.efficacy, cpu.brain.efficacy, rtol=0, atol=1e-4)
