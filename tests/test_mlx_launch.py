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
from cadence.brain import _MlxPlan, _MlxState

pytestmark = pytest.mark.skipif("mlx" not in cd.available_backends(), reason="mlx not installed")

ROWS = 4


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
def test_a_launch_settles_as_the_cpu_at_any_count_of_sweeps(steps):
    cpu, mlx = pair()
    drive = drive_of(cpu)
    a = cpu.brain.settle_batch(drive, steps=steps)
    b = mlx.brain.settle_batch(drive, steps=steps)
    assert launched(b) and b.steps == steps
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.v, a.v, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.activity_change, a.activity_change, rtol=0, atol=1e-3)


def test_the_first_sweep_reads_the_sources_it_starts_from():
    cpu, mlx = pair()
    earlier, later = drive_of(cpu, seed=1), drive_of(cpu, seed=2)
    a = cpu.brain.settle_batch(later, steps=3, state=cpu.brain.settle_batch(earlier, steps=20))
    b = mlx.brain.settle_batch(later, steps=3, state=mlx.brain.settle_batch(earlier, steps=20))
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    np.testing.assert_allclose(b.activity_change, a.activity_change, rtol=0, atol=1e-3)


@pytest.mark.parametrize("tolerance", [1e-1, 1e-2, 3e-3, 1e-3])
def test_a_movement_tolerance_stops_where_the_cpu_stops(tolerance):
    cpu, mlx = pair()
    drive = drive_of(cpu)
    a = cpu.brain.settle_batch(drive, steps=400, tolerance=tolerance)
    b = mlx.brain.settle_batch(drive, steps=400, tolerance=tolerance)
    assert b.steps == a.steps
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=1e-5)
    assert b.attempted_steps >= b.steps  # a launch past the stop counts as work


@pytest.mark.parametrize(
    "kind", ["softmax", "softmax_weighted", "quadratic", "quadratic_weighted", "slots"]
)
def test_a_launch_takes_the_nudges_of_a_lesson(kind):
    settings = {"slots": 2} if kind == "slots" else {}
    cpu, mlx = pair(**settings)
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


def test_a_solve_checks_on_the_device_and_counts_as_on_the_cpu():
    cpu, mlx = pair()
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


def test_a_state_within_tolerance_takes_no_sweep_and_one_check():
    cpu, mlx = pair()
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


def test_a_budget_of_no_sweeps_reads_the_start():
    cpu, mlx = pair()
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


def test_a_state_continues_on_the_device_and_reads_back_without_a_launch():
    cpu, mlx = pair()
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
    half = cd.learning_neuron_model(dt=0.5)
    assert cd.NeuralGraph(connectome, half, backend="mlx")._mlx.plan is None
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


RECIPE = dict(
    recall=True, recall_recency=0.0, learning_surprise=True, recall_boundary=True,
    recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01, recall_relative=True,
    teaching="deep", recall_address_contrast=True, episodic=False,
)
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
