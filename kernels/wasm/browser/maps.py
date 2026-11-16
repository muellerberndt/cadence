"""The map of things on the compiled path against its Python: moments with reports, sight and
physics, the moves its surprise reads, questions by place and by value, on Wild's map and two
others. Under Pyodide to the bit; natively within 1e-9 (NumPy's BLAS sums a piece's weighted
middle in another order than the compiled loop).

    python kernels/wasm/browser/run.py maps.py           # Pyodide, the WebAssembly kernel
    python kernels/wasm/browser/maps.py fused            # native, Numba
"""
from __future__ import annotations

import sys

import engine

engine.choose("wasm" if engine.BROWSER else (sys.argv[1] if len(sys.argv) > 1 else "fused"))

import numpy as np  # noqa: E402

from cadence import compiled  # noqa: E402
from cadence.space.map import Map  # noqa: E402

assert compiled.available(), "the map's kernel is not in use"


def maps():
    yield "Wild's map", lambda: Map((0.0, 24.0, 0.0, 24.0), (8, 8), units=4, values={"hue": 4},
                                    reach=3.0, step=1.8, stride=6.0, margin=0.3,
                                    physics={"noise": 0.3}, surprise=True)
    yield "plain map", lambda: Map((0.0, 10.0, 0.0, 10.0), (6, 5), units=3, empty="near")
    yield "kinds by code", lambda: Map((0.0, 12.0, 0.0, 12.0), (7, 7), units=9, kinds=8,
                                       values={"size": 3}, empty="clear")


def life(make, moments=60):
    rng = np.random.default_rng(3)
    held = make()
    out = []
    for _ in range(moments):
        kinds = len(held.codes)
        seen = []
        for _k in range(3):
            if rng.random() < 0.7:
                place = rng.random(2) * (held.plane[1] - held.plane[0])
                if held.values:
                    group, units = next(iter(held.values.items()))
                    seen.append((int(rng.integers(kinds)), place,
                                 {group: np.eye(units.stop - units.start)[int(rng.integers(
                                     units.stop - units.start))]}))
                else:
                    seen.append((int(rng.integers(kinds)), place))
        edge = float(rng.random()) * (held.plane[1] - held.plane[0])
        held.moment(seen=seen, lit=lambda points, e=edge: points[:, 0] < e)
        out.append(held.potential.copy())
        out.append(held.observation())
        out.append(held.moves(held.potential * 0.7 + 0.3))  # a moved sheet against this one
        if held.surprise is not None:
            out.append(np.asarray(held.surprise.last, dtype=np.float64).ravel())
        thing = held.thing_at(rng.random(2) * (held.plane[1] - held.plane[0]))
        if thing is not None:
            out.extend([np.array([thing["kind"]]), np.asarray(thing["place"])])
        if held.values:
            group = next(iter(held.values))
            out.append(held.where_is_value(group, 0).ravel())
    return out


failures = 0
for label, make in maps():
    kernels = {name: getattr(compiled, name) for name in ("_map_settle", "_map_pieces",
                                                            "_map_moves", "_map_write_values")}
    fast = life(make)
    for name in kernels:  # the map's Python: the reference
        setattr(compiled, name, None)
    try:
        slow = life(make)
    finally:
        for name, kernel in kernels.items():
            setattr(compiled, name, kernel)
    same = len(fast) == len(slow) and all(
        a.shape == b.shape and np.array_equal(a.view(np.uint64) if a.dtype == float else a,
                                              b.view(np.uint64) if b.dtype == float else b)
        for a, b in zip(fast, slow, strict=True))
    gap = max((float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else
               (0.0 if a.shape == b.shape else np.inf)) for a, b in zip(fast, slow, strict=True))
    # under Pyodide to the bit; natively NumPy's BLAS sums a weighted middle otherwise
    failures += not (same if engine.BROWSER else gap < 1e-9)
    print(f"{label:16s} {'bitwise' if same else f'apart, largest gap {gap:.1e}'} over "
          f"{len(fast)} arrays", flush=True)
engine.finish(not failures)
