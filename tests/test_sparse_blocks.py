"""Blocks read by their synapses (``fused.sparse_offset``): natively a block in which no neuron
receives more than one synapse (the map's one-to-one wiring, a block that holds none) sums each
column's one synapse where BLAS would multiply the whole block. Every other term of the whole
product is a zero, so a life on the synapses is the life on whole blocks to the bit. A block in
which a neuron receives several synapses keeps BLAS's product."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module
from cadence import fused
from cadence.space.map import Map

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(not brain_module._FUSED, reason="CADENCE_FUSED=0 reads the NumPy loop alone")


def whole(layout, apical=None) -> np.ndarray:
    return np.asarray(layout.offset, dtype=np.int64)


def a_map() -> Map:
    return Map((0.0, 24.0, 0.0, 24.0), (8, 8), units=4, values={"hue": 4}, reach=3.0, step=1.8,
               stride=6.0, margin=0.3, physics={"noise": 0.3})


def most_in_a_column(layout, k: int) -> int:
    """The most synapses a neuron receives within block ``k``."""
    _, _, b0, b1 = layout.bounds(k)
    inside = layout.edge_index[(layout.edge_index >= layout.offset[k])
                               & (layout.edge_index < layout.offset[k + 1])]
    columns = (np.unique(inside) - layout.offset[k]) % (b1 - b0)
    return int(np.bincount(columns, minlength=1).max())


def test_the_maps_one_to_one_sheet_is_read_by_its_synapses() -> None:
    graph = a_map()._plain
    lay = graph.layout
    table = fused.sparse_offset(lay)
    blocks = lay.pairs
    assert np.array_equal(table[: blocks + 1], lay.offset)
    heads = table[blocks + 1 : 2 * blocks + 1]
    sheet = [k for k in range(blocks) if most_in_a_column(lay, k) == 1]
    assert sheet and all(heads[k] >= 0 for k in sheet)  # held to held: one synapse a neuron
    for k in sheet:  # the form holds the block's synapses, each column's one
        inside = np.count_nonzero((lay.edge_index >= lay.offset[k])
                                  & (lay.edge_index < lay.offset[k + 1]))
        assert int(table[heads[k]]) == inside
    assert fused.sparse_offset(lay) is table  # made once


@pytest.mark.parametrize("graph", [
    lambda: cd.Brain.compose(12, 4, modules=(32, 32), backend="cpu", seed=1).learner.brain,
    lambda: cd.NeuralGraph(cd.layered(40, 128, 8, density=0.02, seed=1),
                           cd.learning_neuron_model()),
    lambda: cd.NeuralGraph(cd.layered(40, 128, 8, seed=1), cd.learning_neuron_model()),
])
def test_a_block_is_read_by_its_synapses_where_no_neuron_receives_two(graph) -> None:
    lay = graph().layout
    table = fused.sparse_offset(lay)
    several = 0
    for k in range(lay.pairs):
        most = most_in_a_column(lay, k)
        several += most > 1
        assert (table[lay.pairs + 1 + k] >= 0) == (most <= 1), k
    assert several  # each of these graphs holds a block BLAS keeps


def test_under_pyodide_every_block_is_read_whole(monkeypatch) -> None:
    lay = a_map()._plain.layout
    monkeypatch.setattr(fused, "_WHOLE_BLOCKS", True)
    assert np.array_equal(fused.sparse_offset(lay), lay.offset)


def map_life(moments: int = 40) -> list[np.ndarray]:
    held = a_map()
    rng = np.random.default_rng(1)
    out = []
    for t in range(moments):
        seen = [(int(k), rng.random(2) * 24.0, {"hue": int(rng.integers(4))})
                for k in range(2) if rng.random() < 0.8]
        held.moment(seen=seen, lit=lambda points, e=6.0 + (t % 9) * 2.0: points[:, 0] < e)
        out.extend([held.potential.copy(), held.observation()])
    return out


def deep_life(moments: int = 40) -> list[np.ndarray]:
    brain = cd.Brain.compose(10, 4, modules=(24, 24), apical=0.25, teaching="deep",
                             backend="cpu", seed=2, arousal=True)
    rng = np.random.default_rng(3)
    out = []
    for i in range(moments):
        x = rng.random((1, 10))
        action = brain.act(x, greedy=True) if i % 4 == 3 else brain.live(
            x, reward=[float(rng.random() < 0.4)] if brain.pending_feedback else None)
        out.append(np.atleast_1d(np.asarray(action, dtype=np.float64)).ravel())
    out.append(np.asarray(brain.learner.brain.efficacy, dtype=np.float64))
    return out


def sparse_graph_life(moments: int = 40) -> list[np.ndarray]:
    """Free solves of a layered graph wired at a fiftieth: some blocks one synapse a neuron,
    some several."""
    graph = cd.NeuralGraph(cd.layered(40, 128, 8, density=0.02, seed=1),
                           cd.learning_neuron_model())
    rng = np.random.default_rng(6)
    out, state = [], None
    for _ in range(moments):
        drive = rng.normal(size=(2, graph.connectome.n)) * 0.7
        state = graph.equilibrate(drive, budget=300, chunk=16, tolerance=1e-9, state=state).state
        out.extend([np.asarray(state.v).copy(), np.asarray(state.activation).copy()])
    return out


@pytest.mark.parametrize("life", [map_life, deep_life, sparse_graph_life])
def test_a_life_on_the_synapses_is_the_life_on_whole_blocks_to_the_bit(life, monkeypatch) -> None:
    by_synapses = life()
    monkeypatch.setattr(fused, "sparse_offset", whole)
    on_blocks = life()
    assert len(by_synapses) == len(on_blocks)
    for a, b in zip(by_synapses, on_blocks, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
