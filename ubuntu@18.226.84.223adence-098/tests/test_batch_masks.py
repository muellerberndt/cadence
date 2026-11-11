"""Shared and per-row ablation masks obey the same projected neuron update."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.brain import _FUSED


@pytest.mark.skipif(not _FUSED, reason="fused CPU kernel disabled or unavailable")
@pytest.mark.parametrize("mask_shape", ["shared", "shared_row", "per_row"])
@pytest.mark.parametrize("nudging", ["none", "quadratic", "softmax"])
@pytest.mark.parametrize("adaptation", [False, True])
def test_fused_masks_match_numpy_with_warm_states_and_nudges(
    mask_shape: str, nudging: str, adaptation: bool
) -> None:
    connectome = cd.layered(3, 4, 3, density=1.0, seed=9)
    neuron_model = cd.learning_neuron_model(dt=0.35, leak=0.4)
    if adaptation:
        neuron_model = neuron_model.replace(adaptation=cd.Adaptation(tau_steps=7, strength=0.2))
    brain = cd.NeuralGraph(connectome, neuron_model, efficacy=0.08 * connectome.sign)
    rng = np.random.default_rng(11)
    drive = rng.normal(scale=0.6, size=(3, connectome.n))
    keep = np.ones(connectome.n)
    keep[[0, 7]] = 0
    keep[[4, 8]] = 0.35
    if mask_shape == "shared_row":
        keep = keep[None, :]
    elif mask_shape == "per_row":
        keep = np.stack([keep, np.roll(keep, 1), np.roll(keep, 2)])
    nudge = None
    if nudging != "none":
        nmask = np.zeros(connectome.n)
        nmask[list(connectome.populations["output"])] = 1
        nudge = cd.Nudge(
            rng.uniform(0.1, 0.9, size=drive.shape),
            nmask,
            0.1,
            softmax_temperature=0.3 if nudging == "softmax" else None,
            weight=np.array([1.0, -0.5, 0.0]),
        )
    keep_before = keep.copy()
    warm = brain.settle_batch(drive, steps=7, mask=keep, trajectory=True)
    warm_before = [value.copy() for value in (warm.v, warm.activation, warm.adaptation)]
    for state in (None, warm):
        actual = brain.settle_batch(
            drive, steps=100, state=state, mask=keep, nudge=nudge, tolerance=1e-8
        )
        reference = brain.settle_batch(
            drive,
            steps=100,
            state=state,
            mask=keep,
            nudge=nudge,
            tolerance=1e-8,
            trajectory=True,
        )
        assert actual.steps == reference.steps
        for name in ("v", "activation", "adaptation", "activity_change"):
            np.testing.assert_allclose(
                getattr(actual, name), getattr(reference, name), atol=1e-12, rtol=0
            )
        dead = np.broadcast_to(keep == 0, drive.shape)
        np.testing.assert_array_equal(actual.v[dead], 0)
        np.testing.assert_array_equal(actual.activation[dead], 0)
    np.testing.assert_array_equal(keep, keep_before)
    for value, before in zip((warm.v, warm.activation, warm.adaptation), warm_before, strict=True):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("shape", [(), (2,), (3, 1), (2, 4), (1, 3, 4)])
def test_brain_rejects_masks_without_one_entry_per_neuron(shape: tuple[int, ...]) -> None:
    brain = cd.NeuralGraph(cd.Connectome.from_synapses(4, pre=[], post=[]), cd.learning_neuron_model())
    with pytest.raises(ValueError, match="mask"):
        brain.settle_batch(np.ones((3, 4)), mask=np.ones(shape), trajectory=True)


@pytest.mark.skipif(not _FUSED, reason="fused CPU kernel disabled or unavailable")
@pytest.mark.parametrize("next_mask", ["removed", "shared_ones", "per_row_ones", "changed"])
def test_warm_start_recomputes_publication_after_changing_mask(next_mask: str) -> None:
    connectome = cd.Connectome.from_synapses(3, pre=[0, 1], post=[1, 2], sign=[0.2, -0.1])
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=0.5, leak=1.0))
    drive = np.array([[0.6, 0.0, 0.1], [0.4, 0.1, 0.0]])
    previous_mask = np.array([[0.5, 1.0, 1.0], [0.3, 1.0, 0.5]])
    state = brain.settle_batch(drive, steps=7, mask=previous_mask)
    mask = {
        "removed": None,
        "shared_ones": np.ones(3),
        "per_row_ones": np.ones_like(drive),
        "changed": np.array([[0.8, 0.4, 1.0], [1.0, 0.0, 0.2]]),
    }[next_mask]
    keep = np.ones(3) if mask is None else mask
    published = brain.neuron_model.activation(state.v) * keep
    expected_v = (
        state.v + brain.neuron_model.dt * (published @ brain.dense() + drive - state.v)
    ) * keep
    expected_activation = brain.neuron_model.activation(expected_v) * keep
    expected_activity_change = np.abs(expected_activation - published).sum(axis=1)
    for trajectory in (False, True):
        actual = brain.settle_batch(drive, steps=1, state=state, mask=mask, trajectory=trajectory)
        np.testing.assert_allclose(actual.v, expected_v, atol=1e-14, rtol=0)
        np.testing.assert_allclose(actual.activation, expected_activation, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            actual.activity_change, expected_activity_change, atol=1e-14, rtol=0
        )


@pytest.mark.parametrize("backend", ["torch_cpu", "torch_mps", "mlx"])
@pytest.mark.parametrize("dense_limit", [1, 2048])
@pytest.mark.parametrize("transition", ["same", "removed", "changed"])
def test_device_mask_continuations_match_neuron_equations(
    backend: str, dense_limit: int, transition: str
) -> None:
    connectome = cd.Connectome.from_synapses(3, pre=[0, 1], post=[1, 2], sign=[0.2, -0.1])
    neuron_model = cd.learning_neuron_model(dt=0.5, leak=1.0).replace(
        adaptation=cd.Adaptation(tau_steps=7, strength=0.2)
    )
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        device = "cpu" if backend == "torch_cpu" else "mps"
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable")
        brain = cd.NeuralGraph(
            connectome, neuron_model, backend="torch", device=device, dense_limit=dense_limit
        )
    else:
        if dense_limit == 1:
            pytest.skip("MLX requires block transport")
        pytest.importorskip("mlx.core")
        brain = cd.NeuralGraph(connectome, neuron_model, backend="mlx")
    drive = np.array([[0.6, 0.0, 0.1], [0.4, 0.1, 0.0]])
    old_mask = np.array([[0.5, 1.0, 1.0], [0.3, 1.0, 0.5]])
    state = brain.settle_batch(drive, steps=7, mask=old_mask)
    assert state.device is not None
    if backend.startswith("torch"):
        initial_v = state.device["v"].cpu().double().numpy().copy()
        initial_a = state.device["a"].cpu().double().numpy().copy()
        assert state.__dict__["v"] is None and state.__dict__["adaptation"] is None

        def forbidden_fetch(key: str) -> np.ndarray:
            raise AssertionError(f"continuation fetched {key} from the device")

        state.device["fetch"] = forbidden_fetch
    else:
        initial_v = np.array(state.device["v"], dtype=float)
        initial_a = np.array(state.device["a"], dtype=float)
        # The resident handle is sufficient even when host arrays are unavailable.
        state.__dict__["v"] = state.__dict__["adaptation"] = None
    mask = {
        "same": old_mask,
        "removed": None,
        "changed": np.array([[0.8, 0.4, 1.0], [1.0, 0.0, 0.2]]),
    }[transition]
    keep = np.ones(3) if mask is None else mask
    nudge = cd.Nudge(
        np.array([[0.0, 0.8, 0.2], [0.0, 0.3, 0.7]]),
        np.array([0.0, 1.0, 1.0]),
        0.1,
        weight=np.array([1.0, -0.5]),
    )
    v, a = initial_v.copy(), initial_a.copy()
    published = neuron_model.activation(v) * keep
    activity_change = np.zeros(len(drive))
    for _ in range(3):
        total = published @ brain.dense() + drive - 0.2 * a + nudge.drive(published)
        v = (v + neuron_model.dt * (total - v)) * keep
        following = neuron_model.activation(v) * keep
        activity_change += np.abs(following - published).sum(axis=1)
        published = following
        a = a + (published - a) / 7
    actual = brain.settle_batch(drive, steps=3, state=state, mask=mask, nudge=nudge)
    tolerance = 1e-12 if backend == "torch_cpu" else 2e-6
    for value, expected in zip(
        (actual.v, actual.activation, actual.adaptation, actual.activity_change),
        (v, published, a, activity_change),
        strict=True,
    ):
        np.testing.assert_allclose(value, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("dense_limit", [0, 2048])
def test_cpu_warm_state_keeps_potentials_and_activations_consistent(dtype, dense_limit):
    brain = cd.NeuralGraph(
        cd.Connectome.from_synapses(2, pre=[0], post=[1], sign=[0.3]),
        cd.NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=1, stimulus_amplitude=1),
        dense_limit=dense_limit,
    )
    initial = cd.BrainState(
        v=np.zeros(2, dtype=dtype),
        activation=np.zeros(2, dtype=dtype),
        adaptation=np.zeros(2, dtype=dtype),
        steps=0,
    )
    drive = np.array([0.4, 0.0])
    state = brain.settle(drive, state=initial, steps=20, tolerance=0)
    assert state.v.dtype == np.float64 and state.adaptation.dtype == np.float64
    np.testing.assert_allclose(state.v, [0.4, 0.3 * np.tanh(0.4)], atol=1e-14)
    np.testing.assert_allclose(state.activation, brain.neuron_model.activation(state.v), atol=1e-14)
    assert brain.residual(drive, state)[0] < 1e-14
    np.testing.assert_array_equal(initial.v, [0, 0])
