"""The blocks read by their synapses (``fused.sparse_offset``) against the whole blocks: the
same lives with each low-density block's product summed over its synapses and over its whole
block (``sparse_offset`` handing the kernels ``layout.offset`` alone). A map of valued things
over lives whose things move in and out of view, and the composed brain (an apical compartment,
recall heads, bound records) living on random observations. Under Pyodide to the bit: the
product of a whole block sums its rows in order, as the synapses' sum does. Natively the whole
blocks' products run in BLAS, so the lives agree within rounding.

    python kernels/wasm/browser/run.py sparse.py [moments]      # Pyodide, the WebAssembly kernel
    python kernels/wasm/browser/sparse.py fused [moments]        # native, Numba
"""
from __future__ import annotations

import sys

import engine

engine.choose("wasm" if engine.BROWSER else "fused")
MOMENTS = int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 120

import numpy as np  # noqa: E402
from compositions import COMPOSED, observation, things_map  # noqa: E402

import cadence as cd  # noqa: E402
from cadence import fused  # noqa: E402



def map_life(moments):
    held = things_map()
    side = held.plane[1] - held.plane[0]
    rng = np.random.default_rng(2)
    kinds = len(held.codes)
    walkers = rng.random((kinds, 2)) * side
    pace = rng.normal(size=(kinds, 2)) * 0.03 * side
    out = []
    for t in range(moments):
        edge = side * (0.6 + 0.45 * np.sin(t / 2.0))
        walkers += pace
        pace[(walkers < 0.0) | (walkers > side)] *= -1.0
        walkers = np.clip(walkers, 0.0, side)
        seen = [(k, walkers[k].copy(), {"hue": int(rng.integers(4))}) for k in range(kinds)
                if walkers[k, 0] < edge]
        held.moment(seen=seen, lit=lambda points, e=edge: points[:, 0] < e)
        out.extend([held.potential.copy(), held.observation()])
        thing = held.thing_at(rng.random(2) * side)
        if thing is not None:
            out.append(np.asarray(thing["place"]))
    return out


def brain_life(moments):
    brain = cd.Brain.compose(**COMPOSED, backend="cpu", seed=4, arousal=True)
    space = brain.space = things_map()
    rng = np.random.default_rng(5)
    out = []
    for i in range(moments):
        seen = [(0, [6.0, 6.0], {"hue": np.eye(4)[0]}),
                (1, [4.0 + 0.1 * (i % 120), 12.0], {"hue": np.eye(4)[2]})]
        space.moment(seen=seen, lit=lambda points: points[:, 0] < 16.0)
        x = observation(COMPOSED, rng)
        if i % 5 == 4:
            action = brain.act(x, greedy=True)
        elif brain.pending_feedback:
            action = brain.live(x, reward=[float(rng.random() < 0.4)])
        else:
            action = brain.live(x)
        out.append(np.atleast_1d(np.asarray(action, dtype=np.float64)).ravel())
    graph = brain.learner.brain
    out.extend([np.asarray(graph.efficacy, dtype=np.float64), np.asarray(graph.bias)])
    return out


failures = 0
sparse = fused.sparse_offset
for label, life in (("valued map", map_life), ("composed brain", brain_life)):
    by_synapses = life(MOMENTS)
    fused.sparse_offset = lambda layout, apical=None: np.asarray(layout.offset, dtype=np.int64)
    try:
        whole = life(MOMENTS)
    finally:
        fused.sparse_offset = sparse
    same = len(whole) == len(by_synapses) and all(
        a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))
        for a, b in zip(by_synapses, whole, strict=True))
    gap = max(float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else
              (0.0 if a.shape == b.shape else np.inf) for a, b in zip(by_synapses, whole,
                                                                      strict=True))
    failures += not (same if engine.BROWSER else gap < 1e-9)
    print(f"{label:14s} {'bitwise' if same else f'apart, largest gap {gap:.1e}'} over "
          f"{len(whole)} arrays", flush=True)
engine.finish(not failures)
