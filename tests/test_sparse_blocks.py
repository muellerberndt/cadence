"""Blocks read by their synapses (``fused.sparse_offset``): a block whose synapses fill less
than a quarter of it sums its product over them, each column's in the order of their
presynaptic neurons. Natively the whole blocks' products run in BLAS, so a life on the
synapses agrees with the same life on whole blocks within rounding; in a browser, where the
whole block's product sums its rows in order, to the bit (``kernels/wasm/browser/sparse.py``)."""

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


def test_the_maps_one_to_one_wiring_is_read_by_its_synapses() -> None:
    graph = a_map()._plain
    lay = graph.layout
    table = fused.sparse_offset(lay)
    blocks = lay.pairs
    assert np.array_equal(table[: blocks + 1], lay.offset)
    heads = table[blocks + 1 : 2 * blocks + 1]
    assert (heads >= 0).all()  # every block of the map's sheet is a sparse one
    held = sum(int(table[at]) for at in heads)
    assert held == len(np.unique(lay.edge_index)) == graph.connectome.synapses
    assert fused.sparse_offset(lay) is table  # made once


def test_a_brain_reads_its_dense_blocks_whole() -> None:
    graph = cd.Brain.compose(12, 4, modules=(32, 32), backend="cpu", seed=1).learner.brain
    lay = graph.layout
    table = fused.sparse_offset(lay)
    for k in range(lay.pairs):
        a0, a1, b0, b1 = lay.bounds(k)
        filled = np.count_nonzero((lay.edge_index >= lay.offset[k])
                                  & (lay.edge_index < lay.offset[k + 1]))
        assert (table[lay.pairs + 1 + k] >= 0) == (filled < fused.SPARSE_SHARE * (a1 - a0) * (b1 - b0))


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


@pytest.mark.parametrize("life", [map_life, deep_life])
def test_a_life_on_the_synapses_is_the_life_on_whole_blocks(life, monkeypatch) -> None:
    by_synapses = life()
    monkeypatch.setattr(fused, "sparse_offset", whole)
    on_blocks = life()
    assert len(by_synapses) == len(on_blocks)
    for a, b in zip(by_synapses, on_blocks, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
