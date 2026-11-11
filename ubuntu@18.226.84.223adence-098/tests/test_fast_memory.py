"""Independent local-memory laws, legacy compatibility, and stream isolation."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadence import FastSynapses


@pytest.mark.parametrize("rate", [0.0, 0.2, 1.0])
def test_delta_matches_independent_normalized_lms(rate):
    rng = np.random.default_rng(37)
    memory = FastSynapses(np.arange(5), np.arange(5, 8), rule="delta", rate=rate, decay=0.97)
    weights = np.zeros((4, 5, 3))
    count = 0
    for _ in range(30):
        keys = rng.normal(size=(4, 5))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        values = rng.normal(size=(4, 3))
        gate = rng.random(4) > 0.25
        weights *= 0.97
        for stream in np.flatnonzero(gate):
            for output in range(3):
                error = values[stream, output] - np.dot(keys[stream], weights[stream, :, output])
                weights[stream, :, output] += rate * keys[stream] * error
        memory.observe(keys, values, gate)
        count += int(gate.sum())
        np.testing.assert_allclose(memory.strength, weights, atol=1e-13)
    assert memory.writes == count


def test_delta_latest_write_is_exact_and_a_repeat_is_quiet():
    memory = FastSynapses(np.arange(3), np.arange(3, 5), rule="delta", amplitude=2)
    key = np.array([[0.4, 0.5, 0.9]])
    memory.observe(key, np.array([[1.0, 0.0]]))
    memory.observe(key, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(memory.recall(10 * key), [[0.0, 2.0]], atol=1e-14)
    before = memory.strength.copy()
    memory.observe(key, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(memory.strength, before, atol=1e-14)


def test_delta_nonorthogonal_update_changes_other_keys():
    memory = FastSynapses(np.arange(2), np.arange(2, 4), rule="delta")
    first, second = np.array([[1.0, 0.0]]), np.array([[0.6, 0.8]])
    memory.observe(first, np.array([[1.0, 0.0]]))
    memory.observe(second, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(memory.recall(second), [[0.0, 1.0]], atol=1e-14)
    assert not np.allclose(memory.recall(first), [[1.0, 0.0]])


def test_zero_delta_keys_do_not_write_but_time_still_decays():
    memory = FastSynapses(np.arange(2), np.arange(2, 3), rule="delta", decay=0.5)
    memory.observe(np.array([[1.0, 0.0], [0.0, 1.0]]), np.ones((2, 1)))
    before, count = memory.strength.copy(), memory.writes
    memory.observe(np.zeros((2, 2)), np.ones((2, 1)))
    np.testing.assert_array_equal(memory.strength, before * 0.5)
    assert memory.writes == count


@pytest.mark.parametrize(
    "bad", ["key_nan", "value_inf", "key_shape", "value_shape", "batch", "mask_shape", "mask_type"]
)
def test_invalid_observation_does_not_decay_reset_or_partially_write(bad):
    memory = FastSynapses(np.arange(2), np.arange(2, 4), rule="delta", decay=0.5)
    memory.observe(np.eye(2), np.eye(2))
    old = memory.strength.copy(), memory.mass.copy(), memory.writes
    keys, values, gate = np.eye(2), np.eye(2), np.ones(2, dtype=bool)
    if bad == "key_nan":
        keys[0, 0] = np.nan
    elif bad == "value_inf":
        values[1, 1] = np.inf
    elif bad == "key_shape":
        keys = np.ones((2, 3))
    elif bad == "value_shape":
        values = np.ones((2, 3))
    elif bad == "batch":
        keys, gate = np.ones((3, 2)), np.ones(3, dtype=bool)
    elif bad == "mask_shape":
        gate = np.ones((2, 1), dtype=bool)
    elif bad == "mask_type":
        gate = np.ones(2, dtype=int)
    with pytest.raises(ValueError):
        memory.observe(keys, values, gate)
    np.testing.assert_array_equal(memory.strength, old[0])
    np.testing.assert_array_equal(memory.mass, old[1])
    assert memory.writes == old[2]


def test_invalid_recall_does_not_reset_memory():
    memory = FastSynapses(np.arange(2), np.arange(2, 3), rule="delta")
    memory.observe(np.eye(2), np.ones((2, 1)))
    before = memory.strength.copy()
    with pytest.raises(ValueError):
        memory.recall(np.full((3, 2), np.nan))
    np.testing.assert_array_equal(memory.strength, before)


def test_row_reset_keep_and_batch_change_are_isolated():
    memory = FastSynapses(np.arange(2), np.arange(2, 3), rule="delta")
    memory.observe(np.array([[1.0, 0.0]] * 3), np.array([[1.0], [2.0], [3.0]]))
    memory.reset(3, rows=np.array([False, True, False]))
    np.testing.assert_allclose(memory.recall(np.array([[1.0, 0.0]] * 3)), [[1.0], [0.0], [3.0]])
    memory.keep(np.array([2, 0]))
    np.testing.assert_allclose(memory.recall(np.array([[1.0, 0.0]] * 2)), [[3.0], [1.0]])
    np.testing.assert_array_equal(memory.recall(np.array([[1.0, 0.0]])), [[0.0]])


@pytest.mark.parametrize(
    "normalize,replace", [(False, False), (True, False), (False, True), (True, True)]
)
def test_legacy_hebb_update_matches_original_formula(normalize, replace):
    rng = np.random.default_rng(13)
    memory = FastSynapses(
        np.array([1, 3]),
        np.array([0, 4]),
        normalize=normalize,
        replace=replace,
        rate=0.7,
        decay=0.9,
        amplitude=1.8,
    )
    weights, mass = np.zeros((3, 2, 2)), np.zeros(3)
    for step in range(7):
        activation = rng.uniform(-1, 1, (3, 5))
        # Tiny keys exercise the legacy norm floor as well as ordinary vectors.
        if step == 2:
            activation[:, [1, 3]] *= 1e-14
        post = rng.normal(size=(3, 2))
        mask = None if step == 3 else np.array([True, step % 2 == 0, False])
        weights *= 0.9
        mass *= 0.9
        if mask is not None:
            for row in np.flatnonzero(mask):
                key = activation[row, [1, 3]]
                if normalize:
                    key = key / max(np.linalg.norm(key), 1e-12)
                if replace:
                    weights[row] *= (key <= 0)[:, None]
                weights[row] += 0.7 * np.outer(key, post[row])
                mass[row] += 0.7
        memory.update(SimpleNamespace(activation=activation), write=mask, post=post)
        np.testing.assert_allclose(memory.strength, weights, atol=1e-14)
        cue = activation[:, [1, 3]]
        if normalize:
            cue = cue / np.maximum(np.linalg.norm(cue, axis=1, keepdims=True), 1e-12)
        expected = np.einsum("bi,bij->bj", cue, weights)
        if normalize:
            expected /= np.maximum(mass, 1e-12)[:, None]
        np.testing.assert_allclose(memory.read(activation), 1.8 * expected, atol=1e-14)


@pytest.mark.parametrize("size", [1e-300, 1e300])
def test_delta_normalization_handles_finite_key_extremes(size):
    memory = FastSynapses(np.arange(2), np.arange(2, 3), rule="delta")
    key = size * np.array([[1.0, 2.0]])
    memory.observe(key, np.array([[0.7]]))
    np.testing.assert_allclose(memory.recall(key), [[0.7]], atol=1e-14)
