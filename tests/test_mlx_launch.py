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
