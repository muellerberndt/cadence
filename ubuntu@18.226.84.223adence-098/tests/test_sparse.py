"""CSR is a lowering of the neuron-local sum, including parameter changes and fallbacks."""

from __future__ import annotations

import types

import numpy as np
import pytest

import cadence as cd
from cadence import sparse


def connectome():
    # Duplicate synapses are legal in the raw Connectome constructor; isolated neurons too.
    return cd.Connectome(
        9,
        np.array([0, 0, 2, 5, 5, 3]),
        np.array([1, 1, 4, 4, 6, 0]),
        np.array([2.0, 3.0, 1.0, 0.0, 4.0, 2.0]),
        np.array([1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    )


def neuron_sum(brain, activation, blocks=None):
    synaptic_input = np.zeros_like(activation)
    for pre, post, weight in zip(
        brain.connectome.pre, brain.connectome.post, brain.weights, strict=True
    ):
        for row in range(len(activation)):
            synaptic_input[row, post] += activation[row, pre] * weight
    return synaptic_input


def test_csr_preserves_duplicates_signed_weights_and_isolated_neurons():
    pytest.importorskip("scipy.sparse")
    brain = cd.NeuralGraph(connectome(), cd.NeuronModel(), dense_limit=0)
    state = np.random.default_rng(12).normal(size=(32, 9))
    assert brain.to_dict()["sparse_kernel"] is None
    np.testing.assert_allclose(brain._synaptic_input(state), neuron_sum(brain, state), atol=1e-14)
    assert brain.to_dict()["sparse_kernel"] == "scipy_csr"
    assert brain._csr is not None
    assert np.shares_memory(brain._csr[1].data, brain.weights)


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("kind", ["free", "quadratic", "grouped_softmax"])
def test_changing_input_trajectories_match_neuron_sum_with_mask_nudge_and_adaptation(batch, kind):
    neuron_model = cd.NeuronModel(
        dt=0.3, gain=0.2, leak=0.02, adaptation=cd.Adaptation(tau_steps=7, strength=0.25)
    )
    brain = cd.NeuralGraph(connectome(), neuron_model, dense_limit=0)
    reference = cd.NeuralGraph(brain.connectome, neuron_model, dense_limit=0)
    reference._synaptic_input = types.MethodType(neuron_sum, reference)
    rng = np.random.default_rng(33)
    keep = np.ones(9)
    keep[3], keep[7] = 0.0, 0.5
    nudge = None
    if kind != "free":
        mask = np.zeros(9)
        mask[[0, 1, 5, 6]] = 1
        target = rng.uniform(size=(batch, 9))
        groups = np.array([0, 0, -1, -1, -1, 1, 1, -1, -1])
        nudge = cd.Nudge(
            target,
            mask,
            0.13,
            softmax_temperature=0.2 if kind == "grouped_softmax" else None,
            groups=groups,
            weight=rng.uniform(-1, 1, batch),
        )
    actual = expected = None
    for _ in range(3):
        drive = rng.uniform(0, 3, size=(batch, 9))
        actual = brain.settle_batch(
            drive, state=actual, steps=17, mask=keep, nudge=nudge, trajectory=True
        )
        expected = reference.settle_batch(
            drive, state=expected, steps=17, mask=keep, nudge=nudge, trajectory=True
        )
        for field in ("v", "activation", "adaptation", "trajectory", "activity_change"):
            np.testing.assert_allclose(
                getattr(actual, field), getattr(expected, field), rtol=1e-12, atol=1e-13
            )
        assert actual.steps == expected.steps == 17


def test_new_parameters_cannot_reuse_stale_csr_weights_or_mutate_old_brain():
    pytest.importorskip("scipy.sparse")
    brain = cd.NeuralGraph(connectome(), cd.NeuronModel(), dense_limit=0)
    state = np.random.default_rng(73).normal(size=(3, 9))
    original = brain._synaptic_input(state)
    original_weights = brain.weights.copy()
    changed = brain.with_parameters(efficacy=-brain.efficacy, bias=np.ones(9))
    assert changed._csr is None
    np.testing.assert_allclose(changed._synaptic_input(state), -original, atol=1e-14)
    np.testing.assert_array_equal(brain._synaptic_input(state), original)
    np.testing.assert_array_equal(brain.weights, original_weights)
    assert changed._csr[1] is not brain._csr[1]
    assert np.shares_memory(changed._csr[1].indices, brain._csr[1].indices)
    assert np.shares_memory(changed._csr[1].indptr, brain._csr[1].indptr)
    changed.efficacy = brain.efficacy * 2
    assert changed._csr is None
    np.testing.assert_allclose(changed._synaptic_input(state), 2 * original, atol=1e-14)
    with pytest.raises(ValueError, match="read-only"):
        changed.weights[:] *= 0.5
    changed.efficacy = changed.efficacy * 0.5
    assert changed._csr is None
    np.testing.assert_allclose(changed._synaptic_input(state), original, atol=1e-14)
    np.testing.assert_array_equal(brain._synaptic_input(state), original)
    gained = brain.with_parameters(log_gain=np.log(np.full(9, 3.0)))
    np.testing.assert_allclose(gained._synaptic_input(state), 3 * original, atol=1e-14)


def test_numpy_only_fallback_still_runs_the_original_segmented_sum(monkeypatch):
    monkeypatch.setattr(sparse, "_csr_type", lambda: None)
    brain = cd.NeuralGraph(connectome(), cd.NeuronModel(), dense_limit=0)
    state = np.random.default_rng(15).normal(size=(3, 9))
    np.testing.assert_allclose(brain._synaptic_input(state), neuron_sum(brain, state), atol=1e-14)
    assert brain.to_dict()["sparse_kernel"] == "numpy_segmented"


@pytest.mark.parametrize("neurons", [0, 8])
def test_empty_graph_has_no_sparse_dependency(neurons, monkeypatch):
    def forbidden():
        raise AssertionError("empty graph should never import a sparse transport")

    monkeypatch.setattr(sparse, "_csr_type", forbidden)
    w = cd.Connectome.from_synapses(neurons, pre=[], post=[])
    brain = cd.NeuralGraph(w, cd.NeuronModel(), dense_limit=0)
    state = np.ones((3, neurons))
    np.testing.assert_array_equal(brain._synaptic_input(state), np.zeros_like(state))
    assert brain._csr is None


def test_connectome_still_rejects_self_synapses_before_transport():
    with pytest.raises(ValueError, match="distinct neurons"):
        cd.Connectome(2, np.array([0]), np.array([0]), np.ones(1), np.ones(1))
