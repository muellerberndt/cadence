"""The brain: nudges, adaptation, transports, batches, validation, and the torch kernel."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.brain import Nudge


def ring(n: int = 6) -> cd.Connectome:
    return cd.Connectome.from_synapses(
        n, pre=list(range(n)), post=[(i + 1) % n for i in range(n)], count=[100] * n
    )


def test_available_backends_names_cpu() -> None:
    backends = cd.available_backends()
    assert backends["cpu"].startswith("numpy")


def test_settle_reports_steps_and_stops_at_tolerance() -> None:
    brain = cd.NeuralGraph(ring(), cd.NeuronModel(gain=0.03, dt=0.5))
    loose = brain.settle(stimulus={0: 3.0}, steps=200, tolerance=1e-2)
    tight = brain.settle(stimulus={0: 3.0}, steps=200, tolerance=1e-6)
    assert 0 < loose.steps <= tight.steps <= 200
    assert tight.activation.shape == (6,)
    with_trajectory = brain.settle(stimulus={0: 3.0}, steps=5, trajectory=True)
    assert (
        with_trajectory.trajectory is not None
        and with_trajectory.trajectory.shape[0] == with_trajectory.steps
    )


def test_settled_state_helpers() -> None:
    brain = cd.NeuralGraph(ring(), cd.NeuronModel(gain=0.03))
    batch = brain.settle_batch(
        np.stack([brain.stimulus_vector({0: 3.0}), brain.stimulus_vector({3: 3.0})]), steps=40
    )
    assert batch.batched and batch.activation.shape == (2, 6)
    assert batch.row(1).shape == (6,)
    assert 0 <= batch.mean([0, 1, 2]) <= 1 and batch.mean([]) == 0.0
    assert 0 <= batch.fraction_active() <= 1 and 0 <= batch.fraction_active([0, 1]) <= 1
    assert batch.active(0.5) <= 6
    single = brain.settle(stimulus={0: 3.0}, steps=40)
    assert not single.batched


def test_stimulus_vector_and_levels() -> None:
    brain = cd.NeuralGraph(ring(), cd.NeuronModel(stimulus_amplitude=2.0))
    v = brain.stimulus_vector({2: 1.5})
    assert v.shape == (6,) and v[2] == 3.0 and v.sum() == 3.0  # level times the stimulus amplitude
    levels = brain.stimulus_levels(np.array([[0.5, 0, 0, 0, 0, 1.0]]))
    assert np.allclose(levels[0], [1.0, 0, 0, 0, 0, 2.0])


def test_dense_and_segmented_transports_agree_with_nudges_and_adaptation() -> None:
    connectome = cd.layered(5, 4, 3, density=1.0, seed=3)
    neuron_model = cd.learning_neuron_model(dt=0.5).replace(
        adaptation=cd.Adaptation(tau_steps=10, strength=0.2)
    )
    dense = cd.NeuralGraph(connectome, neuron_model, dense_limit=10_000)
    segmented = cd.NeuralGraph(connectome, neuron_model, dense_limit=1)
    assert (
        dense.to_dict()["transport"] == "dense" and segmented.to_dict()["transport"] == "segmented"
    )
    drive = dense.stimulus_levels(np.random.default_rng(0).random((3, connectome.n)) * 0.5)
    mask = np.zeros(connectome.n)
    mask[list(connectome.populations["output"])] = 1.0
    target = np.zeros((3, connectome.n))
    target[:, list(connectome.populations["output"])] = 0.8
    for nudge in (
        None,
        Nudge(target, mask, 0.2),
        Nudge(target, mask, 0.2, softmax_temperature=0.3, weight=np.array([1.0, -0.5, 0.0])),
    ):
        a = dense.settle_batch(drive, steps=30, nudge=nudge)
        b = segmented.settle_batch(drive, steps=30, nudge=nudge)
        assert np.allclose(a.activation, b.activation, atol=1e-9)
        assert np.allclose(a.adaptation, b.adaptation, atol=1e-9)
    assert dense.dense().shape == (connectome.n, connectome.n) and dense.weights.shape == (
        connectome.synapses,
    )


def test_nudge_drive_shapes() -> None:
    mask = np.array([0.0, 1.0, 1.0])
    quadratic = Nudge(np.array([0.5, 0.5, 0.5]), mask, 0.1)
    push = quadratic.drive(np.array([0.2, 0.2, 0.2]))
    assert np.allclose(push, [0.0, 0.03, 0.03])
    weighted = Nudge(np.array([[0.5] * 3, [0.5] * 3]), mask, 0.1, weight=np.array([1.0, 2.0]))
    push2 = weighted.drive(np.array([[0.2] * 3, [0.2] * 3]))
    assert np.allclose(push2[1], 2 * push2[0])


def test_brain_validates_inputs() -> None:
    connectome = ring()
    neuron_model = cd.NeuronModel()
    with pytest.raises(ValueError):
        cd.NeuralGraph(connectome, neuron_model, efficacy=np.ones(3))
    with pytest.raises(ValueError):
        cd.NeuralGraph(connectome, neuron_model, bias=np.ones(2))
    with pytest.raises(ValueError):
        cd.NeuralGraph(connectome, neuron_model, backend="abacus")  # type: ignore[arg-type]
    brain = cd.NeuralGraph(connectome, neuron_model)
    with pytest.raises(ValueError):
        brain.settle_batch(np.zeros((2, 5)))
    state = brain.settle_batch(np.zeros((2, 6)), steps=2)
    with pytest.raises(ValueError):
        brain.settle_batch(np.zeros((3, 6)), state=state)
    one_row = brain.settle_batch(np.zeros(6), steps=2)
    assert one_row.activation.shape == (1, 6)


def test_with_parameters_and_readings() -> None:
    connectome = ring().with_populations(head=[0, 1], tail=[4, 5])
    brain = cd.NeuralGraph(connectome, cd.NeuronModel(gain=0.03))
    changed = brain.with_parameters(efficacy=brain.efficacy * 2.0, bias=np.full(6, 0.1))
    assert np.allclose(changed.efficacy, brain.efficacy * 2.0) and changed.bias[0] == 0.1
    state = brain.settle(stimulus={0: 3.0}, steps=40)
    readings = brain.readings(state, ["head", "tail"])
    assert set(readings) == {"head", "tail"} and set(readings["head"]) >= {"mean", "fraction"}


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_kernel_matches_cpu_with_every_feature() -> None:
    connectome = cd.layered(8, 6, 4, density=1.0, seed=5)
    neuron_model = cd.learning_neuron_model(dt=0.5, leak=0.2).replace(
        adaptation=cd.Adaptation(tau_steps=15, strength=0.1)
    )
    cpu = cd.NeuralGraph(connectome, neuron_model)
    torch_dense = cd.NeuralGraph(connectome, neuron_model, backend="torch", dense_limit=10_000)
    torch_segmented = cd.NeuralGraph(connectome, neuron_model, backend="torch", dense_limit=1)
    drive = cpu.stimulus_levels(np.random.default_rng(1).random((4, connectome.n)) * 0.5)
    out = list(connectome.populations["output"])
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    target = np.zeros((4, connectome.n))
    target[np.arange(4), [out[i % 4] for i in range(4)]] = 1.0
    for nudge in (
        None,
        Nudge(target, mask, 0.1),
        Nudge(target, mask, 0.1, softmax_temperature=0.2, weight=np.array([1.0, 0.5, -0.5, 0.0])),
    ):
        a = cpu.settle_batch(drive, steps=25, nudge=nudge, tolerance=None)
        for brain in (torch_dense, torch_segmented):
            b = brain.settle_batch(drive, steps=25, nudge=nudge, tolerance=None)
            assert np.allclose(a.activation, b.activation, atol=1e-4), brain.to_dict()["transport"]
            assert np.allclose(a.adaptation, b.adaptation, atol=1e-4)
    traced = torch_dense.settle_batch(drive, steps=6, trajectory=True, tolerance=1e-9)
    assert traced.trajectory is not None and traced.trajectory.shape == (
        traced.steps,
        4,
        connectome.n,
    )
    warm = torch_dense.settle_batch(drive, steps=3, state=traced)
    assert warm.activation.shape == (4, connectome.n)


@pytest.mark.skipif("mlx" not in cd.available_backends(), reason="mlx not installed")
def test_mlx_kernel_matches_cpu_with_every_feature() -> None:
    connectome = cd.layered(8, 6, 4, density=1.0, seed=5)
    neuron_model = cd.learning_neuron_model(dt=0.5, leak=0.2).replace(
        adaptation=cd.Adaptation(tau_steps=15, strength=0.1)
    )
    cpu = cd.NeuralGraph(connectome, neuron_model)
    mlx = cd.NeuralGraph(connectome, neuron_model, backend="mlx")
    assert mlx.to_dict()["transport"] == "dense"
    drive = cpu.stimulus_levels(np.random.default_rng(1).random((4, connectome.n)) * 0.5)
    out = list(connectome.populations["output"])
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    target = np.zeros((4, connectome.n))
    target[np.arange(4), [out[i % 4] for i in range(4)]] = 1.0
    keep = np.ones(connectome.n)
    keep[9] = 0.0
    for nudge in (
        None,
        Nudge(target, mask, 0.1),
        Nudge(target, mask, 0.1, softmax_temperature=0.2, weight=np.array([1.0, 0.5, -0.5, 0.0])),
    ):
        a = cpu.settle_batch(drive, steps=25, nudge=nudge, mask=keep, tolerance=1e-5)
        b = mlx.settle_batch(drive, steps=25, nudge=nudge, mask=keep, tolerance=1e-5)
        assert np.abs(a.activation - b.activation).max() < 1e-4
        assert np.abs(a.adaptation - b.adaptation).max() < 1e-4
        assert a.activity_change is not None and b.activity_change is not None
        assert np.abs(a.activity_change - b.activity_change).max() < 1e-3
        assert b.device is not None and b.device["kernel"] == "mlx"
    # a continuation from a state that stayed on the device is the same continuation
    free = mlx.settle_batch(drive, steps=30)
    again = mlx.settle_batch(drive, steps=10, state=free, nudge=Nudge(target, mask, 0.1))
    from_host = mlx.settle_batch(
        drive,
        steps=10,
        state=cd.BrainState(free.v, free.activation, free.adaptation, free.steps),
        nudge=Nudge(target, mask, 0.1),
    )
    assert np.abs(again.activation - from_host.activation).max() < 1e-5
    traj = mlx.settle_batch(drive, steps=5, trajectory=True)
    assert traj.trajectory is not None and traj.trajectory.shape == (5, 4, connectome.n)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_device_kernels_honour_softmax_groups(backend: str) -> None:
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    connectome = cd.layered(6, 5, 6, density=1.0, seed=8)  # two slots of three output neurons
    neuron_model = cd.learning_neuron_model(dt=1.0)
    cpu = cd.NeuralGraph(connectome, neuron_model)
    kw = {"device": "cpu"} if backend == "torch" else {}
    device = cd.NeuralGraph(connectome, neuron_model, backend=backend, **kw)  # type: ignore[arg-type]
    out = np.asarray(connectome.populations["output"])
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    groups = np.full(connectome.n, -1)
    groups[out[:3]] = 0
    groups[out[3:]] = 1
    target = np.zeros((3, connectome.n))
    target[np.arange(3), out[[0, 1, 2]]] = 1.0
    target[np.arange(3), out[[3, 4, 5]]] = 1.0
    drive = cpu.stimulus_levels(np.random.default_rng(9).random((3, connectome.n)) * 0.5)
    nudge = Nudge(target, mask, 0.2, softmax_temperature=0.2, groups=groups)
    a = cpu.settle_batch(drive, steps=25, nudge=nudge)
    b = device.settle_batch(drive, steps=25, nudge=nudge)
    tolerance = 1e-12 if backend == "torch" else 1e-4
    assert np.abs(a.activation - b.activation).max() < tolerance
    # and the grouped result differs from a single softmax over all six, so the groups matter
    single = cpu.settle_batch(
        drive, steps=25, nudge=Nudge(target, mask, 0.2, softmax_temperature=0.2)
    )
    assert np.abs(a.activation - single.activation).max() > 1e-3


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_zero_tolerance_runs_fixed_budget_without_device_scalar_read(monkeypatch) -> None:
    import torch

    connectome = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], sign=[0.4, -0.3])
    brain = cd.NeuralGraph(connectome, cd.NeuronModel(gain=1), backend="torch", device="cpu")
    expected = brain.settle([1.0, 0.0], steps=20, tolerance=None)

    def forbidden_scalar_read(self):
        raise AssertionError("fixed-budget settling must not synchronize a device scalar")

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "__float__", forbidden_scalar_read)
        actual = brain.settle([1.0, 0.0], steps=20, tolerance=0)
    assert actual.steps == 20
    np.testing.assert_array_equal(actual.activation, expected.activation)
