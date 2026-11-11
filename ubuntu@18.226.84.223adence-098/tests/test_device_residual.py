"""The residual of a state that rests on the torch device is computed there and agrees
with the host calculation, with and without a nudge, a mask and adaptation."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")


def _brains() -> tuple[cd.NeuralGraph, cd.NeuralGraph, cd.Connectome]:
    connectome = cd.layered(6, 8, 3, density=1.0, lateral=-0.3, seed=3)
    bias = np.random.default_rng(1).normal(0.0, 0.3, connectome.n)
    host = cd.NeuralGraph(connectome, cd.learning_neuron_model(), bias=bias)
    device = cd.NeuralGraph(connectome, cd.learning_neuron_model(), bias=bias, backend="torch", device="cpu")
    return host, device, connectome


@pytest.mark.parametrize("softmax", [None, 0.2])
@pytest.mark.parametrize("masked", [False, True])
def test_device_residual_matches_host(softmax: float | None, masked: bool) -> None:
    host, device, connectome = _brains()
    rng = np.random.default_rng(0)
    drive = np.zeros((4, connectome.n))
    drive[:, :6] = rng.random((4, 6))
    target = np.zeros((4, connectome.n))
    target[np.arange(4), 6 + 8 + rng.integers(0, 3, 4)] = 1.0
    mask = np.zeros(connectome.n)
    mask[-3:] = 1.0
    nudge = cd.Nudge(target, mask, 0.1, softmax_temperature=softmax, weight=np.array([1.0, -0.5, 2.0, 0.0]))
    keep = None
    if masked:
        keep = np.ones(connectome.n)
        keep[7] = 0.0
    on_device = device.settle_batch(drive, steps=7, nudge=nudge, mask=keep)
    assert on_device.device is not None and on_device.device.get("holder") is device._torch
    fetched = cd.BrainState(np.array(on_device.v), np.array(on_device.activation), np.array(on_device.adaptation), 7)
    expected = host.residual(drive, fetched, nudge=nudge, mask=keep)
    got = device.residual(drive, on_device, nudge=nudge, mask=keep)
    assert got.shape == (4,)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    assert expected.max() > 1e-3  # seven steps do not settle: the test compares real errors


def test_device_residual_with_adaptation_and_equilibrate() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=5)
    neuron_model = cd.learning_neuron_model().replace(adaptation=cd.Adaptation(tau_steps=5, strength=0.3))
    host = cd.NeuralGraph(connectome, neuron_model)
    device = cd.NeuralGraph(connectome, neuron_model, backend="torch", device="cpu")
    drive = np.zeros((3, connectome.n))
    drive[:, :4] = np.random.default_rng(2).random((3, 4))
    state = device.settle_batch(drive, steps=5)
    fetched = cd.BrainState(np.array(state.v), np.array(state.activation), np.array(state.adaptation), 5)
    np.testing.assert_allclose(device.residual(drive, state), host.residual(drive, fetched), rtol=1e-10, atol=1e-12)
    settled = device.equilibrate(drive, budget=2000, chunk=50, tolerance=1e-8)
    assert settled.converged.all()
    fetched = cd.BrainState(np.array(settled.state.v), np.array(settled.state.activation), np.array(settled.state.adaptation), 0)
    assert host.residual(drive, fetched).max() <= 1e-8


def test_device_residual_survives_a_learner_update() -> None:
    connectome = cd.layered(5, 7, 2, density=1.0, seed=8)
    device = cd.NeuralGraph(connectome, cd.learning_neuron_model(), backend="torch", device="cpu")
    learner = cd.Learner(device, connectome.populations["output"], cd.LearnerConfig(eta=0.5))
    drive = np.zeros((4, connectome.n))
    drive[:, :5] = np.random.default_rng(3).random((4, 5))
    learner.step(drive, np.array([0, 1, 1, 0]))
    brain = learner.brain  # parameters now live on the device
    settled = brain.equilibrate(drive, budget=1000, chunk=25, tolerance=1e-9)
    host = cd.NeuralGraph(connectome, cd.learning_neuron_model(), efficacy=brain.efficacy, bias=brain.bias)
    fetched = cd.BrainState(np.array(settled.state.v), np.array(settled.state.activation), np.array(settled.state.adaptation), 0)
    np.testing.assert_allclose(brain.residual(drive, settled.state), host.residual(drive, fetched), rtol=1e-9, atol=1e-12)
    assert settled.converged.all()
