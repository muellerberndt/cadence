"""A state the compiled kernels settled or checked keeps the block products of its activations:
a check of it, or a settle from it, reads them where the activations are the ones they were
made from and makes the rest, to the bits of making them all."""

from __future__ import annotations

import copy
import gc
import pickle
import weakref
from collections.abc import Iterator

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module
from cadence.brain import _copy_state

pytest.importorskip("numba")


def bare(state: cd.BrainState) -> cd.BrainState:
    """The same state without kept products."""
    return cd.BrainState(
        np.array(state.v), np.array(state.activation), np.array(state.adaptation), state.steps
    )


def same(a: cd.BrainState, b: cd.BrainState) -> bool:
    return (
        a.steps == b.steps
        and np.array_equal(a.v, b.v)
        and np.array_equal(a.activation, b.activation)
        and np.array_equal(a.adaptation, b.adaptation)
        and np.array_equal(a.activity_change, b.activity_change)
    )


def graphs() -> Iterator[tuple[cd.Brain, cd.NeuralGraph]]:
    """The founder's graph, the same with adaptation, and a deep graph with an apical
    compartment."""
    founder = cd.Brain.compose(10, 4, modules=(12, 8), seed=1)
    yield founder, founder.brain
    g = founder.brain
    yield founder, cd.NeuralGraph(
        g.connectome, g.neuron_model.replace(adaptation=cd.Adaptation(tau_steps=7.0, strength=0.4)),
        efficacy=g.efficacy, bias=g.bias,
    )
    deep = cd.Brain.compose(10, 4, modules=(12, 9, 7), seed=2, apical=0.25)
    yield deep, deep.brain


def kept(state: cd.BrainState) -> np.ndarray:
    return state.__dict__["_products"][3]


@pytest.mark.parametrize("batch", [1, 3])
def test_a_check_and_a_settle_read_the_kept_products_to_the_bits_of_making_them_all(
    batch: int,
) -> None:
    rng = np.random.default_rng(batch)
    for brain, graph in graphs():
        n = graph.connectome.n
        drive = brain.stimulus(rng.normal(size=(batch, 10)))
        motor = np.array(graph.connectome.populations["motor"])
        target = np.zeros((batch, n))
        target[np.arange(batch), motor[rng.integers(0, len(motor), batch)]] = 1.0
        out = np.zeros(n)
        out[motor] = 1.0
        free = graph.settle_batch(drive, steps=30)
        nudge = cd.Nudge(target, out, 0.3, softmax_temperature=0.5,
                         apical_gain=graph.apical_gain(drive, free))
        shared = np.ones(n)
        shared[motor] = 0.5  # every range a block reads loses some activation under it
        shared[: graph.layout.starts[1]] = 0.5
        for kw in (dict(), dict(mask=shared), dict(nudge=nudge), dict(nudge=nudge, mask=shared)):
            state = graph.settle_batch(drive, steps=9, state=free, **kw)  # type: ignore[arg-type]
            assert "_products" in state.__dict__
            np.testing.assert_array_equal(
                graph.residual(drive, state, **kw),  # type: ignore[arg-type]
                graph.residual(drive, bare(state), **kw),  # type: ignore[arg-type]
            )
            # the check left every product current, which a settle from the state reads
            after = graph.settle_batch(drive, steps=7, state=state, **kw)  # type: ignore[arg-type]
            assert same(after, graph.settle_batch(drive, steps=7, state=bare(state), **kw))  # type: ignore[arg-type]
            # a check under another mask makes the products whose activations differ
            graph.residual(drive, after, **kw)  # type: ignore[arg-type]
            np.testing.assert_array_equal(
                graph.residual(drive, after), graph.residual(drive, bare(after))
            )
            again = graph.settle_batch(drive, steps=5, state=after, tolerance=1e-6)
            assert same(again, graph.settle_batch(drive, steps=5, state=bare(after), tolerance=1e-6))


def test_kept_products_are_read_where_the_activations_hold_and_made_where_they_differ() -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=0)
    graph = brain.brain
    lay, n, batch = graph.layout, graph.connectome.n, 2
    drive = brain.stimulus(np.random.default_rng(0).random((batch, 8)))
    state = graph.settle_batch(drive, steps=12)
    graph.residual(drive, state)  # every kept product current
    # each block's products in the kept array, after the activations, in block order
    width = np.diff(lay.starts)
    offset = batch * n + np.concatenate([[0], np.cumsum(batch * width[lay.pair_post])])
    k = int(np.argmax(width[lay.pair_pre] * width[lay.pair_post]))  # the largest block
    kept(state)[offset[k]:offset[k + 1]] += 1.0  # a product that is no longer its block's
    remade = graph.residual(drive, bare(state))
    assert not np.array_equal(graph.residual(drive, state), remade)  # it is read
    # its source range's activations moved: the product is made anew
    r = int(lay.pair_pre[k])
    before = np.array(state.activation)
    state.v[:, lay.starts[r]:lay.starts[r + 1]] += 0.5
    moved = graph.neuron_model.activation(state.v)
    assert not np.array_equal(moved[:, lay.starts[r]:lay.starts[r + 1]],
                              before[:, lay.starts[r]:lay.starts[r + 1]])
    np.testing.assert_array_equal(graph.residual(drive, state), graph.residual(drive, bare(state)))


def test_kept_products_hold_only_for_the_blocks_they_were_made_with() -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=0)
    graph = brain.brain
    drive = brain.stimulus(np.random.default_rng(1).random((2, 8)))
    state = graph.settle_batch(drive, steps=12)
    graph.residual(drive, state)
    other = graph.with_parameters(efficacy=graph.efficacy * 0.5)
    np.testing.assert_array_equal(other.residual(drive, state), other.residual(drive, bare(state)))
    assert same(other.settle_batch(drive, steps=6, state=state),
                other.settle_batch(drive, steps=6, state=bare(state)))
    graph.residual(drive, state)
    graph.efficacy = graph.efficacy * 0.9  # the same graph, new blocks
    np.testing.assert_array_equal(graph.residual(drive, state), graph.residual(drive, bare(state)))
    # the products keep no blocks alive
    connectome, model = graph.connectome, graph.neuron_model
    alone = cd.NeuralGraph(connectome, model, efficacy=graph.efficacy, bias=graph.bias)
    blocks = weakref.ref(alone._blocks)
    held = alone.settle_batch(drive, steps=4)
    del alone
    gc.collect()
    assert blocks() is None and "_products" in held.__dict__


def test_copies_leave_the_kept_products_and_the_solver_copy_and_no_sweeps_keep_them() -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=0)
    graph = brain.brain
    drive = brain.stimulus(np.random.default_rng(2).random((2, 8)))
    state = graph.settle_batch(drive, steps=5)
    products = state.__dict__["_products"]
    for other in (copy.copy(state), copy.deepcopy(state), pickle.loads(pickle.dumps(state))):
        assert "_products" not in other.__dict__ and same(other, state)
    assert _copy_state(state).__dict__["_products"] is products
    assert graph.settle_batch(drive, steps=0, state=state).__dict__["_products"] is products
    single = graph.settle(drive[0], steps=5)
    assert "_products" in single.__dict__
    np.testing.assert_array_equal(graph.residual(drive[0], single),
                                  graph.residual(drive[0], bare(single)))
    original = brain_module._FUSED
    brain_module._FUSED = False
    try:  # the NumPy loop keeps none
        assert "_products" not in graph.settle_batch(drive, steps=5).__dict__
    finally:
        brain_module._FUSED = original
