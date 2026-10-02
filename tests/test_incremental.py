"""Independent full-reference comparisons for exact cached activity arithmetic."""

import math
import random

import pytest

from cadence import _repair as R
from cadence._incremental import ActivityCache


def fresh(cache):
    return R._evaluate(
        cache.graph,
        cache.inputs,
        cache.state,
        cache.weights,
        cache.biases,
        cache._alpha,
        None,
        None,
        0.1,
        parameter_gradients=False,
    )


def same(cache, evaluated=None):
    expected = fresh(cache)
    actual = cache.evaluation() if evaluated is None else evaluated
    assert actual == {k: expected[k] for k in actual}
    for clamps in ({}, {0: cache.state[0]}, dict(enumerate(cache.state))):
        residual = R._stationarity(
            cache.state,
            cache.weights,
            cache.biases,
            expected,
            clamps,
            False,
            cache._state_bound,
            4.0,
        )
        assert cache.certificate(clamps=clamps) == {
            "stationarity": residual,
            "qualified": residual <= 1e-6,
        }


def graph_and_values(seed):
    rng = random.Random(seed)
    n, m = 7, 4
    order = list(range(n))
    rng.shuffle(order)
    rank = {node: i for i, node in enumerate(order)}
    edges = []
    for target in range(n):
        for source in range(m):
            if rng.random() < 0.35:
                edges.append(("input", source, target))
        for source in range(n):
            if rng.random() < 0.18:
                edges.append(("state", source, target))  # cycles/self-contacts legal
            if rank[source] < rank[target] and rng.random() < 0.25:
                edges.append(("residual", source, target))
    graph = R.Graph(m, n, tuple(edges))

    def values(size):
        return tuple(rng.uniform(-0.8, 0.8) for _ in range(size))

    return graph, values(m), values(n), values(len(edges)), values(n), rng


@pytest.mark.parametrize("seed", range(16))
def test_randomized_deep_residual_recurrent_exact_reference(seed):
    graph, u, x, w, b, rng = graph_and_values(seed)
    cache = ActivityCache(graph, u, x, w, b)
    same(cache)
    for step in range(20):
        u, x = list(cache.inputs), list(cache.state)
        if step % 3 != 0:
            u[rng.randrange(len(u))] = rng.uniform(-1, 1)
        if step % 3 != 1:
            x[rng.randrange(len(x))] = rng.uniform(-1, 1)
        same(cache, cache.update(inputs=u, state=x))


def test_dirty_reverse_slope_without_changed_error():
    graph = R.Graph(1, 2, (("input", 0, 0), ("residual", 0, 1), ("state", 0, 1)))
    cache = ActivityCache(graph, (0.0,), (0.0, 0.3), (1.0, 0.8, 0.2), (0.0, 0.0))
    cache.update(inputs=(0.5,), state=(math.tanh(0.5), 0.3))
    assert cache.evaluation()["errors"][0] == 0.0
    same(cache)


def test_untouched_component_reuses_gradient_but_full_certificate_counts_it():
    graph = R.Graph(2, 3, (("input", 0, 0), ("input", 1, 1), ("state", 1, 2)))
    cache = ActivityCache(graph, (0.0, 0.0), (0.0,) * 3, (0.5,) * 3, (0.0,) * 3)
    stamp = cache.stamp((1, 2))
    before = cache.work
    cache.update(inputs=(0.4, 0.0))
    assert cache.matches(stamp)
    assert cache.work["edge_visits"] - before["edge_visits"] == 1
    assert cache.evaluation()["gradient_state"][1:] == (0.0, 0.0)
    before = cache.work["certificate_checks"]
    assert not cache.certificate()["qualified"]
    assert cache.work["certificate_checks"] - before == 3
    same(cache)


def test_stamp_expands_error_ancestry_but_not_direct_state_prediction():
    graph = R.Graph(
        2, 4, (("input", 0, 0), ("state", 0, 1), ("input", 1, 2), ("residual", 2, 3))
    )
    cache = ActivityCache(graph, (0.0, 0.0), (0.0,) * 4, (0.5,) * 4, (0.0,) * 4)
    direct = cache.stamp((1,))
    recursive = cache.stamp((3,))
    cache.update(inputs=(0.5, 0.0))
    assert cache.matches(direct)  # e1 reads x0, not p0(u0)
    assert cache.matches(recursive)
    cache.update(inputs=(0.5, 0.5))
    assert not cache.matches(recursive)
    cache.update(state=(0.2, 0.0, 0.0, 0.0))
    assert not cache.matches(direct)


def test_zero_weight_contacts_remain_conservative_dependencies():
    graph = R.Graph(1, 2, (("input", 0, 0), ("residual", 0, 1)))
    cache = ActivityCache(graph, (0.0,), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    stamp = cache.stamp((1,))
    cache.update(inputs=(1.0,))
    assert not cache.matches(stamp)
    same(cache)


def test_scope_rejects_different_model_and_fork_but_not_irrelevant_updates():
    graph = R.Graph(1, 2, (("input", 0, 0),))
    a = ActivityCache(graph, (0.0,), (0.0, 0.0), (0.0,), (0.0, 0.0))
    b = ActivityCache(graph, a.inputs, a.state, (0.1,), a.biases)
    stamp = a.stamp((0,))
    assert not b.matches(stamp) and not a.fork().matches(stamp)
    a.update(state=(0.0, 0.2))
    assert a.matches(stamp)


def test_fork_isolated_values_and_accounting():
    graph = R.Graph(1, 2, (("input", 0, 0), ("residual", 0, 1)))
    parent = ActivityCache(graph, (0.1,), (0.0, 0.0), (0.7, 0.4), (0.0, 0.0))
    original, before = parent.evaluation(), parent.work
    child = parent.fork()
    assert all(v == 0 for v in child.work.values())
    assert parent.work["forks"] - before["forks"] == 1
    assert parent.work["copied_values"] > before["copied_values"]
    child.update(inputs=(0.8,), state=(0.3, -0.2))
    assert parent.evaluation() == original and child.evaluation() != original
    same(child)
    same(parent)


@pytest.mark.parametrize(
    "update", ({"inputs": (float("nan"),)}, {"state": (2.0,)}, {"inputs": (1e308,)})
)
def test_failed_update_atomic_values_and_stamps(update):
    cache = ActivityCache(
        R.Graph(1, 1, (("input", 0, 0),)), (0.0,), (0.0,), (4.0,), (0.0,)
    )
    before, stamp = cache.evaluation(), cache.stamp((0,))
    with pytest.raises(ValueError):
        cache.update(**update)
    assert cache.evaluation() == before and cache.matches(stamp)
    same(cache, cache.update(inputs=(0.2,)))


def test_exact_bound_projection_and_all_clamped_certificate():
    cache = ActivityCache(R.Graph(0, 2, ()), (), (1.0, -1.0), (), (4.0, -4.0))
    same(cache)
    assert cache.certificate(clamps={0: 1.0, 1: -1.0}) == {
        "qualified": True,
        "stationarity": 0.0,
    }
    for clamps in ({0: 0.5}, {2: 0.0}, {True: -1.0}, {0: float("inf")}):
        with pytest.raises(ValueError):
            cache.certificate(clamps=clamps)


def test_finite_extreme_cancellation_and_rejected_overflow():
    graph = R.Graph(2, 1, (("input", 0, 0), ("input", 1, 0)))
    cache = ActivityCache(graph, (1e307, -1e307), (0.0,), (4.0, 4.0), (0.0,))
    same(cache)
    same(cache, cache.update(inputs=(1e306, -1e306)))
    before = cache.evaluation()
    with pytest.raises(ValueError):
        cache.update(inputs=(1e308, 1e308))
    assert cache.evaluation() == before


def test_reverse_overflow_rolls_back_after_successful_forward_traversal():
    graph = R.Graph(
        1, 9, (("input", 0, 0),) + tuple(("residual", 0, i) for i in range(1, 9))
    )
    cache = ActivityCache(
        graph,
        (0.0,),
        (0.0,) * 9,
        (1.0,) + (1e308,) * 8,
        (0.0,) * 9,
        parameter_bound=1e308,
    )
    before, stamp = cache.evaluation(), cache.stamp((0,))
    with pytest.raises(ValueError):
        cache.update(inputs=(1e-308,))
    assert cache.evaluation() == before and cache.matches(stamp)


def test_cached_gradient_matches_independent_energy_finite_difference():
    graph, u, x, w, b, _ = graph_and_values(91)
    cache = ActivityCache(graph, u, x, w, b)
    x = tuple(v * 0.7 for v in x)
    actual = cache.update(state=x, inputs=tuple(-v for v in u))
    h = 1e-6
    for i, derivative in enumerate(actual["gradient_state"]):
        plus, minus = list(x), list(x)
        plus[i] += h
        minus[i] -= h

        def energy(state):
            return R._evaluate(
                graph,
                cache.inputs,
                state,
                w,
                b,
                0.01,
                None,
                None,
                0.1,
                parameter_gradients=False,
            )["energy"]

        assert derivative == pytest.approx(
            (energy(plus) - energy(minus)) / (2 * h), abs=2e-9
        )


def test_failed_vector_validation_charges_attempted_values_and_returned_value_stays_stale():
    cache = ActivityCache(
        R.Graph(1, 1, (("input", 0, 0),)), (0.0,), (0.0,), (1.0,), (0.0,)
    )
    stamp, before = cache.stamp((0,)), cache.work
    with pytest.raises(ValueError):
        cache.update(inputs=(float("nan"),))
    assert cache.work["finite_checks"] == before["finite_checks"] + 1
    assert cache.matches(stamp)
    cache.update(inputs=(0.4,))
    cache.update(inputs=(0.0,))
    assert not cache.matches(stamp)


def test_immutable_configuration_and_copied_work_receipt():
    cache = ActivityCache(R.Graph(0, 1, ()), (), (0.0,), (), (0.0,))
    for attr in ("graph", "inputs", "state", "weights", "biases"):
        with pytest.raises(AttributeError):
            setattr(cache, attr, None)
    work = cache.work
    work["edge_visits"] = -100
    assert cache.work["edge_visits"] >= 0
