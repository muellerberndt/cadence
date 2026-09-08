"""Wirings and owner rules: construction, validation, queries, digests."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_from_edges_merges_parallel_overlaps_and_drops_self_loops() -> None:
    w = cd.Wiring.from_edges(
        3, pre=[0, 0, 1, 2], post=[1, 1, 2, 2], count=[2, 3, 1, 9], sign=[1, -1, 1, 1]
    )
    assert w.edges == 2  # the self loop 2->2 is dropped; 0->1 merged
    i = list(zip(w.pre.tolist(), w.post.tolist(), strict=True)).index((0, 1))
    assert w.count[i] == 5 and np.isclose(w.sign[i], (2 - 3) / 5)
    assert w.in_degree()[1] == 1 and w.out_degree()[0] == 1
    kept = cd.Wiring.from_edges(3, pre=[0, 1], post=[1, 2], count=[1, 10], min_count=5)
    assert kept.edges == 1


def test_wiring_validation_and_queries() -> None:
    with pytest.raises(ValueError):
        cd.Wiring(3, np.array([0, 1]), np.array([1]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        cd.Wiring.from_edges(2, pre=[0, 5], post=[1, 0])
    w = cd.Wiring.from_edges(4, pre=[0, 1, 2], post=[1, 2, 3], sets={"a": [0, 1]}, label="chain")
    w2 = w.with_sets(b=[2, 3])
    assert w2.sets == {"a": (0, 1), "b": (2, 3)} and w2.label == "chain"
    assert w2.members("b") == (2, 3)
    assert (
        w.digest()
        == cd.Wiring.from_edges(
            4, pre=[2, 1, 0], post=[3, 2, 1], sets={"a": [0, 1]}, label="chain"
        ).digest()
    )
    assert w.digest() != w2.digest()
    summary = w2.summary()
    assert summary["owners"] == 4 and summary["overlaps"] == 3 and "b" in summary["sets"]


def test_layered_and_embedded_builders() -> None:
    dense = cd.layered(4, 3, 2, density=1.0, seed=0)
    assert set(dense.sets) == {"input", "hidden", "output"} and dense.n == 9
    out_degree, in_degree = dense.out_degree(), dense.in_degree()
    assert (out_degree[list(dense.sets["input"])] > 0).all()  # every input owner reaches the net
    assert (in_degree[list(dense.sets["hidden"])] > 0).all()
    assert (in_degree[list(dense.sets["output"])] > 0).all()
    sparse = cd.layered(10, 6, 2, density=0.3, seed=0)
    assert sparse.edges < cd.layered(10, 6, 2, density=1.0, seed=0).edges
    skip = cd.layered(4, 3, 2, density=1.0, seed=0, skip=True)
    assert skip.edges > dense.edges
    wiring, tie = cd.embedded(5, 3, 2, 4, 2, seed=0)
    assert set(wiring.sets) >= {"input", "embedding", "hidden", "output"}
    assert tie.shape == (wiring.edges,) and (tie >= 0).sum() > 0
    counts = np.bincount(tie[tie >= 0])
    assert counts.min() >= 2  # a shared seam appears in every position


def test_shuffled_keeps_degrees_and_changes_wiring() -> None:
    w = cd.layered(6, 4, 2, density=0.6, seed=2)
    s = cd.shuffled(w, seed=1)
    assert s.edges == w.edges and s.n == w.n
    assert s.digest() != w.digest()
    kept = cd.shuffled(w, seed=1, keep=[0, 1])
    assert kept.edges == w.edges


def test_graded_rule_validation_activation_and_dict() -> None:
    with pytest.raises(ValueError):
        cd.GradedRule(dt=0.0)
    with pytest.raises(ValueError):
        cd.GradedRule(slope=-1.0)
    with pytest.raises(ValueError):
        cd.GradedRule(leak=2.0)
    with pytest.raises(ValueError):
        cd.Adaptation(tau_steps=0.0)
    rule = cd.learning_rule(dt=1.0, leak=0.1)
    assert rule.activation(np.array([0.0]))[0] == 0.0  # nothing at rest
    assert rule.activation(np.array([2.0]))[0] > 0.5
    assert rule.activation(np.array([-3.0]))[0] < 0.0  # the leak answers below rest
    assert cd.GradedRule(leak=0.0).activation(np.array([-3.0]))[0] == 0.0
    assert rule.slope_at(np.array([0.0]))[0] > 0
    d = rule.to_dict()
    assert (
        d["adaptation"] is None
        and cd.GradedRule(**{k: v for k, v in d.items() if k != "adaptation"}) == rule
    )
    with_adapt = rule.replace(adaptation=cd.Adaptation(tau_steps=5.0))
    assert with_adapt.to_dict()["adaptation"] == {"tau_steps": 5.0, "strength": 1.0}
