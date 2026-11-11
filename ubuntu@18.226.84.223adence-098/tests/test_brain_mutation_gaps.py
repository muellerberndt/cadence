"""NeuralGraph defaults, integer stimulus indices, and host warm states the kernels must not write.

The first three tests each fail on a fault that the rest of the suite does not detect.
"""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def ring(n: int = 6) -> cd.Connectome:
    return cd.Connectome.from_synapses(
        n, pre=list(range(n)), post=[(i + 1) % n for i in range(n)], count=[100] * n
    )


def test_fraction_active_defaults_to_row_zero() -> None:
    brain = cd.NeuralGraph(ring(), cd.NeuronModel(gain=0.03))
    batch = brain.settle_batch(
        np.stack([brain.stimulus_vector({0: 3.0, 1: 3.0, 2: 3.0}), brain.stimulus_vector(None)]),
        steps=60,
    )
    assert batch.fraction_active(i=0) != batch.fraction_active(i=1)  # the rows differ
    assert batch.fraction_active() == batch.fraction_active(i=0)


def test_stimulus_vector_integer_indices_are_neurons() -> None:
    brain = cd.NeuralGraph(ring(6), cd.NeuronModel(stimulus_amplitude=2.0))
    # a 1-D integer array names neurons at full amplitude; it is not a vector of levels 0..5
    assert np.allclose(brain.stimulus_vector(np.array([0, 1, 2, 3, 4, 5])), 2.0)


def test_settle_default_runs_sixty_steps() -> None:
    brain = cd.NeuralGraph(ring(), cd.NeuronModel(gain=0.03))
    assert brain.settle(stimulus={0: 3.0}).steps == 60


def test_torch_float64_kernel_leaves_the_host_warm_state_unchanged() -> None:
    pytest.importorskip("torch")
    model = cd.NeuronModel(gain=0.03, adaptation=cd.Adaptation(tau_steps=5.0, strength=0.5))
    drive = np.zeros((2, 6))
    drive[:, 0] = 3.0
    warm = cd.NeuralGraph(ring(), model).settle_batch(drive, steps=30)
    assert np.abs(warm.adaptation).max() > 0
    before = [x.copy() for x in (warm.v, warm.activation, warm.adaptation)]
    # float64 host arrays reach the CPU kernel through torch.from_numpy without a copy, so
    # the kernel's potential and adaptation updates must rebind rather than write in place
    brain = cd.NeuralGraph(ring(), model, backend="torch", device="cpu", precision="float64")
    brain.settle_batch(drive, steps=30, state=warm)
    for value, saved in zip((warm.v, warm.activation, warm.adaptation), before, strict=True):
        np.testing.assert_array_equal(value, saved)
