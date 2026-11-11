"""Zero sensory/action projections preserve the initial circuit and remain trainable."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


@pytest.mark.parametrize("seed", range(5))
def test_zero_skip_preserves_existing_parameters_predictions_and_learns(seed: int) -> None:
    original = cd.layered(4, 8, 2, density=1, seed=seed)
    expanded = cd.layered(4, 8, 2, density=1, seed=seed, skip=True, skip_init=0.0)
    before = cd.NeuralGraph(original, cd.learning_neuron_model(dt=1.0))
    brain = cd.NeuralGraph(expanded, before.neuron_model)
    np.testing.assert_array_equal(brain.dense(), before.dense())
    new_edges = (expanded.pre < 4) & (expanded.post >= 12)
    assert new_edges.sum() == 8 and expanded.synapses == original.synapses + 8
    assert np.all(expanded.count[new_edges] == 1)
    assert np.all(brain.efficacy[new_edges] == 0)
    drive = np.zeros((2, expanded.n))
    drive[0, :2] = 1
    drive[1, 2:4] = 1
    np.testing.assert_array_equal(
        brain.settle_batch(drive, steps=80).activation,
        before.settle_batch(drive, steps=80).activation,
    )
    learner = cd.Learner(brain, expanded.populations["output"])
    learner.step(drive, np.array([1, 0]))
    assert np.abs(learner.brain.efficacy[new_edges]).max() > 1e-6


def test_skip_scale_is_independent_and_default_preserves_current_initializer() -> None:
    default = cd.layered(4, 8, 2, seed=7, skip=True, init=0.3)
    explicit = cd.layered(4, 8, 2, seed=7, skip=True, init=0.3, skip_init=0.3)
    np.testing.assert_array_equal(default.sign, explicit.sign)
    changed = cd.layered(4, 8, 2, seed=7, skip=True, init=0.3, skip_init=0.6)
    skip = (changed.pre < 4) & (changed.post >= 12)
    np.testing.assert_array_equal(changed.sign[~skip], default.sign[~skip])
    np.testing.assert_array_equal(changed.sign[skip], 2 * default.sign[skip])
    assert np.any(cd.layered(4, 8, 2, seed=7, skip=True, init=0, skip_init=0.6).sign != 0)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_skip_scale_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="skip_init"):
        cd.layered(4, 8, 2, skip=True, skip_init=value)
