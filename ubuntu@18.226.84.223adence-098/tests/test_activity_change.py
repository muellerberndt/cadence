"""The activity change of settling: the total movement of the published activations, per row."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_activity_change_is_the_path_length_of_the_activations_and_agrees_across_paths() -> None:
    w = cd.layered(6, 5, 3, density=1.0, seed=0)
    neuron_model = cd.learning_neuron_model(dt=1.0)
    brain = cd.NeuralGraph(w, neuron_model)
    drive = brain.stimulus_levels(np.random.default_rng(0).random((3, w.n)) * 0.5)
    fused = brain.settle_batch(drive, steps=40)
    loop = brain.settle_batch(drive, steps=40, trajectory=True)
    assert (
        fused.activity_change is not None
        and loop.activity_change is not None
        and loop.trajectory is not None
    )
    assert fused.activity_change.shape == (3,) and (fused.activity_change > 0).all()
    assert np.allclose(fused.activity_change, loop.activity_change, atol=1e-12)
    path = np.abs(np.diff(loop.trajectory, axis=0, prepend=0.0)).sum(axis=2).sum(axis=0)
    assert np.allclose(loop.activity_change, path, atol=1e-12)
    still = brain.settle_batch(drive, steps=5, state=fused)  # already at rest: nothing moves
    assert still.activity_change is not None and still.activity_change.max() < 40 * 1e-4
    single = brain.settle(drive[0], steps=40)
    assert single.activity_change is not None and np.isclose(
        float(single.activity_change), fused.activity_change[0]
    )


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_reports_the_same_activity_change() -> None:
    w = cd.layered(6, 5, 3, density=1.0, seed=1)
    neuron_model = cd.learning_neuron_model(dt=1.0)
    drive = cd.NeuralGraph(w, neuron_model).stimulus_levels(
        np.random.default_rng(1).random((2, w.n)) * 0.5
    )
    cpu = cd.NeuralGraph(w, neuron_model).settle_batch(drive, steps=30)
    acc = cd.NeuralGraph(w, neuron_model, backend="torch", device="cpu").settle_batch(drive, steps=30)
    assert cpu.activity_change is not None and acc.activity_change is not None
    assert np.allclose(cpu.activity_change, acc.activity_change, atol=1e-9)
