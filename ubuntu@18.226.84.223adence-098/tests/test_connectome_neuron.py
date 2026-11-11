"""Connectomes and neuron models: construction, validation, queries, digests."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_from_edges_merges_parallel_synapses_and_drops_self_loops() -> None:
    w = cd.Connectome.from_synapses(
        3, pre=[0, 0, 1, 2], post=[1, 1, 2, 2], count=[2, 3, 1, 9], sign=[1, -1, 1, 1]
    )
    assert w.synapses == 2  # the self loop 2->2 is dropped; 0->1 merged
    i = list(zip(w.pre.tolist(), w.post.tolist(), strict=True)).index((0, 1))
    assert w.count[i] == 5 and np.isclose(w.sign[i], (2 - 3) / 5)
    assert w.in_degree()[1] == 1 and w.out_degree()[0] == 1
    kept = cd.Connectome.from_synapses(3, pre=[0, 1], post=[1, 2], count=[1, 10], min_count=5)
    assert kept.synapses == 1


def test_connectome_validation_and_queries() -> None:
    with pytest.raises(ValueError):
        cd.Connectome(
            3, np.array([0, 1]), np.array([1]), np.array([1.0, 1.0]), np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError):
        cd.Connectome.from_synapses(2, pre=[0, 5], post=[1, 0])
    w = cd.Connectome.from_synapses(
        4, pre=[0, 1, 2], post=[1, 2, 3], populations={"a": [0, 1]}, label="chain"
    )
    w2 = w.with_populations(b=[2, 3])
    assert w2.populations == {"a": (0, 1), "b": (2, 3)} and w2.label == "chain"
    assert w2.members("b") == (2, 3)
    assert (
        w.digest()
        == cd.Connectome.from_synapses(
            4, pre=[2, 1, 0], post=[3, 2, 1], populations={"a": [0, 1]}, label="chain"
        ).digest()
    )
    assert w.digest() != w2.digest()
    summary = w2.summary()
    assert summary["neurons"] == 4 and summary["synapses"] == 3 and "b" in summary["populations"]


def test_layered_and_embedded_builders() -> None:
    dense = cd.layered(4, 3, 2, density=1.0, seed=0)
    assert set(dense.populations) == {"input", "hidden", "output"} and dense.n == 9
    out_degree, in_degree = dense.out_degree(), dense.in_degree()
    assert (
        out_degree[list(dense.populations["input"])] > 0
    ).all()  # every input neuron reaches the net
    assert (in_degree[list(dense.populations["hidden"])] > 0).all()
    assert (in_degree[list(dense.populations["output"])] > 0).all()
    sparse = cd.layered(10, 6, 2, density=0.3, seed=0)
    assert sparse.synapses < cd.layered(10, 6, 2, density=1.0, seed=0).synapses
    skip = cd.layered(4, 3, 2, density=1.0, seed=0, skip=True)
    assert skip.synapses > dense.synapses
    connectome, tie = cd.embedded(5, 3, 2, 4, 2, seed=0)
    assert set(connectome.populations) >= {"input", "embedding", "hidden", "output"}
    assert tie.shape == (connectome.synapses,) and (tie >= 0).sum() > 0
    counts = np.bincount(tie[tie >= 0])
    assert counts.min() >= 2  # a shared synapse appears in every position


@pytest.mark.parametrize(
    "pre, post", [([3], [0]), ([-1], [2]), ([0.5], [1.5]), ([0], [3]), ([np.nan], [1])]
)
def test_from_edges_rejects_invalid_indices_before_merging(pre, post) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        cd.Connectome.from_synapses(3, pre=pre, post=post)


@pytest.mark.parametrize(
    "pre, post, count, sign",
    [
        ([0, 1], [2], [1, 1], [1, 1]),
        ([[0]], [[1]], [[1]], [[1]]),
        ([0], [1], [np.nan], [1]),
        ([0], [1], [1], [np.inf]),
    ],
)
def test_connectome_rejects_malformed_edge_arrays(pre, post, count, sign) -> None:  # type: ignore[no-untyped-def]
    for construct in (
        lambda: cd.Connectome.from_synapses(3, pre=pre, post=post, count=count, sign=sign),
        lambda: cd.Connectome(
            3, np.asarray(pre), np.asarray(post), np.asarray(count), np.asarray(sign)
        ),
    ):
        with pytest.raises(ValueError):
            construct()


def test_small_contact_counts_preserve_signed_transport_when_merged() -> None:
    connectome = cd.Connectome.from_synapses(
        2, pre=[0, 0], post=[1, 1], count=[1e-14, 2e-14], sign=[1, -1]
    )
    np.testing.assert_allclose(connectome.count * connectome.sign, [-1e-14], rtol=1e-14, atol=0)


@pytest.mark.parametrize("pre, post", [([], []), ([0], [0])])
def test_empty_edge_construction_keeps_isolated_neurons(pre, post) -> None:  # type: ignore[no-untyped-def]
    connectome = cd.Connectome.from_synapses(
        3, pre=pre, post=post, populations={"isolated": [1, 2]}
    )
    assert connectome.n == 3 and connectome.synapses == 0
    assert connectome.count.dtype == np.float64 and connectome.sign.dtype == np.float64
    np.testing.assert_array_equal(connectome.in_degree(), [0, 0, 0])


@pytest.mark.parametrize("members", [[-1], [3], [0.5]])
def test_connectome_rejects_invalid_named_neurons(members) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        cd.Connectome.from_synapses(3, pre=[0], post=[1], populations={"readout": members})


def test_graded_rule_validation_activation_and_dict() -> None:
    with pytest.raises(ValueError):
        cd.NeuronModel(dt=0.0)
    with pytest.raises(ValueError):
        cd.NeuronModel(slope=-1.0)
    with pytest.raises(ValueError):
        cd.NeuronModel(leak=2.0)
    with pytest.raises(ValueError):
        cd.Adaptation(tau_steps=0.0)
    neuron_model = cd.learning_neuron_model(dt=1.0, leak=0.1)
    assert neuron_model.activation(np.array([0.0]))[0] == 0.0  # nothing at rest
    assert neuron_model.activation(np.array([2.0]))[0] > 0.5
    assert neuron_model.activation(np.array([-3.0]))[0] < 0.0  # the leak answers below rest
    assert cd.NeuronModel(leak=0.0).activation(np.array([-3.0]))[0] == 0.0
    assert neuron_model.slope_at(np.array([0.0]))[0] > 0
    d = neuron_model.to_dict()
    assert (
        d["adaptation"] is None
        and cd.NeuronModel(**{k: v for k, v in d.items() if k != "adaptation"}) == neuron_model
    )
    with_adapt = neuron_model.replace(adaptation=cd.Adaptation(tau_steps=5.0))
    assert with_adapt.to_dict()["adaptation"] == {"tau_steps": 5.0, "strength": 1.0}


@pytest.mark.parametrize("name", ["slope", "gain", "threshold", "stimulus_amplitude"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rule_rejects_nonfinite_parameters(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        cd.NeuronModel(**{name: value})


@pytest.mark.parametrize("name", ["tau_steps", "strength"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_adaptation_rejects_nonfinite_parameters(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        cd.Adaptation(**{name: value})
