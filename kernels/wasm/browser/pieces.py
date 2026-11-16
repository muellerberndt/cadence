"""The map's pieces compiled against ``Map._pieces``'s Python over random masks: the same
pieces, cells and order (the order a Python set pops their seeds), natively and in Chromium.

    python kernels/wasm/browser/run.py pieces.py
    python kernels/wasm/browser/pieces.py fused
"""
from __future__ import annotations

import sys

import engine

engine.choose("wasm" if engine.BROWSER else (sys.argv[1] if len(sys.argv) > 1 else "fused"))

import numpy as np  # noqa: E402

from cadence import compiled  # noqa: E402
from cadence.space.map import Map  # noqa: E402

assert compiled.available("_map_pieces"), "the map's pieces kernel is not in use"
rng = np.random.default_rng(11)
apart = cases = 0
for nx, ny in ((8, 8), (6, 5), (3, 17), (20, 20), (40, 25), (3, 9)):
    held = Map((0.0, 1.0 * nx, 0.0, 1.0 * ny), (nx, ny), units=2)
    for _ in range(300):
        active = rng.random(nx * ny) < rng.uniform(0.0, 0.7)
        fast = held._pieces(active)
        kernel, compiled._map_pieces = compiled._map_pieces, None
        try:
            slow = held._pieces(active)
        finally:
            compiled._map_pieces = kernel
        cases += 1
        apart += len(fast) != len(slow) or any(
            a.dtype != b.dtype or not np.array_equal(a, b) for a, b in zip(fast, slow, strict=True))
print(f"masks {cases}, apart {apart}")
engine.finish(apart == 0)
