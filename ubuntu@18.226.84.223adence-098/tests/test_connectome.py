"""Direct tests for cadence.connectome: Connectome construction and utility methods."""

import warnings

import numpy as np
import pytest

from cadence.connectome import Connectome


def _ring(n: int = 5) -> Connectome:
    pre = np.arange(n)
    post = (np.arange(n) + 1) % n
    return Connectome.from_synapses(n, pre=pre, post=post, sign=np.ones(n))


# ------------------------------------------------------------------ construction


def test_from_synapses_sets_n_and_sorts_edges_by_post_then_pre() -> None:
    c = _ring(4)
    assert c.n == 4
    assert c.synapses == 4
    assert c.post.tolist() == [0, 1, 2, 3]
    assert c.pre.tolist() == [3, 0, 1, 2]


def test_from_synapses_drops_autapses_and_the_constructor_refuses_them() -> None:
    c = Connectome.from_synapses(3, pre=[0, 1], post=[0, 2], sign=[1.0, 1.0])
    assert list(zip(c.pre.tolist(), c.post.tolist(), strict=True)) == [(1, 2)]
    with pytest.raises(ValueError, match="distinct neurons"):
        Connectome(3, np.array([0]), np.array([0]), np.ones(1), np.ones(1))


def test_parallel_synapses_merge_and_min_count_drops_weak_ones() -> None:
    c = Connectome.from_synapses(
        3, pre=[0, 0, 1], post=[1, 1, 2], count=[2.0, 3.0, 1.0], sign=[1.0, -1.0, 1.0], min_count=1.5
    )
    assert c.synapses == 1
    assert c.count.tolist() == [5.0]
    assert c.sign.tolist() == [pytest.approx((2.0 - 3.0) / 5.0)]  # count-weighted sign


def test_negative_n_is_rejected_and_an_empty_brain_is_allowed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        Connectome.from_synapses(-1, pre=[], post=[])
    assert Connectome.from_synapses(0, pre=[], post=[]).synapses == 0


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        Connectome.from_synapses(2, pre=[0], post=[1], count=[-1.0])


def test_arrays_are_read_only() -> None:
    with pytest.raises(ValueError):
        _ring(3).pre[0] = 2


def test_out_of_range_pre_post_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        Connectome.from_synapses(3, pre=[0, 5], post=[1, 2], sign=[1.0, 1.0])


# ------------------------------------------------------------------ degree


def test_in_degree_counts_incoming_synapses() -> None:
    c = _ring(4)
    # ring: each node has exactly one incoming synapse
    assert c.in_degree().tolist() == [1, 1, 1, 1]


def test_out_degree_counts_outgoing_synapses() -> None:
    c = _ring(4)
    assert c.out_degree().tolist() == [1, 1, 1, 1]


def test_degree_asymmetric_graph() -> None:
    # node 0 -> {1, 2}; node 1 -> {2}
    c = Connectome.from_synapses(3, pre=[0, 0, 1], post=[1, 2, 2], sign=[1.0, 1.0, 1.0])
    assert c.out_degree().tolist() == [2, 1, 0]
    assert c.in_degree().tolist() == [0, 1, 2]


# ------------------------------------------------------------------ populations


def test_with_populations_stores_named_sets() -> None:
    c = _ring(6).with_populations(left=[0, 1, 2], right=[3, 4, 5])
    assert set(c.populations["left"]) == {0, 1, 2}
    assert set(c.populations["right"]) == {3, 4, 5}


def test_a_population_out_of_order_warns_and_is_held_ascending() -> None:
    with pytest.warns(RuntimeWarning, match="population 'late' is not ascending"):
        c = _ring(6).with_populations(late=[3, 4, 5, 0, 1, 2])
    assert c.populations["late"] == (0, 1, 2, 3, 4, 5)
    with pytest.warns(RuntimeWarning, match="population 'twice' is not ascending"):
        assert _ring(6).with_populations(twice=[1, 1, 2]).populations["twice"] == (1, 2)
    with pytest.warns(RuntimeWarning, match="population 'late' is not ascending"):
        Connectome.from_synapses(3, pre=[0, 1], post=[1, 2], populations={"late": [2, 0]})


def test_a_population_in_order_or_a_set_stays_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = _ring(6).with_populations(left=range(3), right=np.arange(3, 6), bag={5, 0, 2})
        assert c.with_populations(left=[0, 1]).populations["right"] == (3, 4, 5)
        assert c.populations["bag"] == (0, 2, 5)
        Connectome.from_synapses(3, pre=[0, 1], post=[1, 2], populations={"bag": frozenset({2, 0})})


def test_inputs_take_a_population_in_ascending_order_whatever_it_was_given() -> None:
    """The order a brain reads, so a caller who laid regions out of order learns what happened."""
    pre = [0, 1, 2, 3, 4, 5, 6, 7]
    post = [8, 8, 8, 8, 9, 9, 9, 9]
    wiring = Connectome.from_synapses(10, pre=pre, post=post)
    populations = {"association": [8], "motor": [9]}
    ascending = wiring.with_populations(sensory=range(8), **populations)
    with pytest.warns(RuntimeWarning, match="population 'sensory' is not ascending"):
        given = wiring.with_populations(sensory=[4, 5, 6, 7, 0, 1, 2, 3], **populations)
    assert given.populations["sensory"] == ascending.populations["sensory"]
    assert given.digest() == ascending.digest()


def test_members_returns_union_of_named_populations() -> None:
    c = _ring(6).with_populations(left=[0, 1], right=[4, 5])
    m = c.members("left", "right")
    assert set(m) == {0, 1, 4, 5}


def test_members_unknown_population_raises() -> None:
    c = _ring(4)
    with pytest.raises(KeyError):
        c.members("nonexistent")


# ------------------------------------------------------------------ digest / summary


def test_digest_is_deterministic_hex_string() -> None:
    c = _ring(5)
    d = c.digest()
    assert isinstance(d, str) and len(d) == 64  # SHA-256 hex
    assert c.digest() == d  # stable across calls


def test_digest_differs_for_different_connectomes() -> None:
    assert _ring(4).digest() != _ring(5).digest()


def test_summary_contains_expected_keys() -> None:
    c = _ring(4)
    s = c.summary()
    assert s["neurons"] == 4 and s["synapses"] == 4
    assert s["excitatory"] == 4 and s["inhibitory"] == 0
    assert s["digest"] == c.digest()
