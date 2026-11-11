"""Adaptive history stays on-device without changing masks or checkpoint continuation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")


def make(device: str | None, momentum: float, normalize: float) -> cd.Learner:
    w, groups = cd.embedded(7, 2, 4, 9, 3, seed=11)
    return cd.Learner(
        cd.NeuralGraph(
            w,
            cd.learning_neuron_model(dt=1),
            backend="cpu" if device is None else "torch",
            device=device,
        ),
        w.populations["output"],
        cd.LearnerConfig(
            momentum=momentum,
            normalize=normalize,
            eta=0.03,
            eta_bias=0.01,
            free_steps=20,
            nudged_steps=12,
            tolerance=None,
        ),
        tie_groups=groups,
    )


def sample(learner: cd.Learner, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    drive = np.zeros((5, learner.brain.connectome.n))
    drive[:, :14] = rng.uniform(0, 0.4, (5, 14))
    return drive, rng.integers(3, size=5)


@pytest.mark.parametrize("momentum,normalize", [(0.9, 0), (0, 0.999), (0.9, 0.999)])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_adaptive_device_matches_numpy_with_mutable_masks(
    momentum: float, normalize: float, device: str
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    host, dev = make(None, momentum, normalize), make(device, momentum, normalize)
    kernel = dev.brain._torch
    for i in range(7):
        if i == 3:
            # In-place mask edits must invalidate device caches, even with the same object id.
            for learner in (host, dev):
                learner.plastic_synapses[::3] = False
                learner.plastic_neurons[::2] = False
            frozen_host_scale = host.brain.efficacy[::3].copy()
            frozen_host_bias = host.brain.bias[::2].copy()
            frozen_device_scale = kernel.scale[::3].clone()
            frozen_device_bias = kernel.bias_param[::2].clone()
        d, labels = sample(host, i)
        phase_h, _ = host.step(d, labels)
        phase_d, _ = dev.step(d, labels)
        assert dev.brain._torch is kernel
        assert dev.__dict__["_device_moments"]["holder"] is kernel
        assert dev.brain._efficacy is None
        np.testing.assert_allclose(phase_d.free.activation, phase_h.free.activation, atol=2e-6)
        if i >= 3:
            # Mask correctness is exact, independent of backend trajectory rounding.
            np.testing.assert_array_equal(host.brain.efficacy[::3], frozen_host_scale)
            np.testing.assert_array_equal(host.brain.bias[::2], frozen_host_bias)
            assert torch.equal(kernel.scale[::3], frozen_device_scale)
            assert torch.equal(kernel.bias_param[::2], frozen_device_bias)
    for name in MOMENTS:
        np.testing.assert_allclose(getattr(dev, name), getattr(host, name), atol=2e-6)
    np.testing.assert_allclose(dev.brain.efficacy, host.brain.efficacy, atol=2e-6)
    np.testing.assert_allclose(dev.brain.bias, host.brain.bias, atol=2e-6)
    assert dev.updates == host.updates == 7


def test_history_read_edit_and_checkpoint_continue(tmp_path: Path) -> None:
    host, dev = make(None, 0.8, 0.9), make("cpu", 0.8, 0.9)
    for i in range(3):
        for learner in (host, dev):
            learner.step(*sample(learner, i))
    for learner in (host, dev):
        learner.velocity[::2] *= 0.5
        learner.second_moment_bias[:] += 0.03
    path = dev.save(tmp_path / "adaptive")
    loaded = cd.Learner.load(path, backend="torch", device="cpu")
    for i in range(3, 6):
        for learner in (host, dev, loaded):
            learner.step(*sample(learner, i))
    for learner in (dev, loaded):
        np.testing.assert_allclose(learner.brain.efficacy, host.brain.efficacy, atol=1e-11)
        for name in MOMENTS:
            np.testing.assert_allclose(getattr(learner, name), getattr(host, name), atol=1e-11)


def test_switch_to_host_contrast_and_back_preserves_history() -> None:
    host, dev = make(None, 0.8, 0.9), make("cpu", 0.8, 0.9)
    for i in range(5):
        d, labels = sample(host, i)
        host.step(d, labels)
        if i == 2:
            target = dev.targets(labels)
            free = dev.free(d)
            plus = dev.nudged(d, free, target)
            minus = dev.nudged(d, free, target, sign=-1)
            states = [
                cd.BrainState(s.v.copy(), s.activation.copy(), s.adaptation.copy(), s.steps)
                for s in (free, plus, minus)
            ]
            dev.update(*states)
            assert str(dev.brain._torch.device) == "cpu"
        else:
            dev.step(d, labels)
    np.testing.assert_allclose(dev.brain.efficacy, host.brain.efficacy, atol=1e-11)
    np.testing.assert_allclose(dev.second_moment, host.second_moment, atol=1e-11)
    # Calibration rebuilds the brain, and must retain an explicit device too.
    assert str(dev._with_gain(0.8)._torch.device) == "cpu"
    dev.config = replace(dev.config, momentum=0, normalize=0)
    dev.step(*sample(dev, 6))
    assert dev.updates == 6
