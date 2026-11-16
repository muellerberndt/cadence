"""The core's shared primitives against NumPy: the generator draw for draw (``random``,
``integers`` over small and large ranges, ``choice`` with probabilities, and the interleaved
draws of a moment), the sums (pairwise, the dot product in order, the mean) and the selection
of the sparse code (``argpartition``, NumPy's introselect, ties and all). Bitwise, or it fails;
NumPy's own dot runs in BLAS natively, and its gap is shown. On x86 with AVX2 or AVX-512 NumPy
selects with its own SIMD routine, which arranges ties otherwise: run this on an ARM machine or
under Pyodide.

    python kernels/core/check/primitives.py
"""
from __future__ import annotations

import sys

import numpy as np

from cadence.core import Core, generator_state

core = Core()
failures = 0


def same(label: str, ours: np.ndarray, theirs: np.ndarray) -> None:
    global failures
    ok = ours.shape == theirs.shape and ours.tobytes() == theirs.tobytes()
    failures += not ok
    print(f"{label:34s} {'bitwise' if ok else 'APART'}")


for seed in (0, 1, 12345, 2**40 + 7):
    rng = np.random.default_rng(seed)
    rng.random()  # a buffered or advanced state, as a life leaves one
    rng.integers(0, 5)  # a 32-bit draw leaves half a draw buffered
    state = generator_state(rng)
    ref = np.random.Generator(np.random.PCG64())
    ref.bit_generator.state = rng.bit_generator.state
    same(f"random, seed {seed}", core.probe("random", n=np.int64(500), **state)["draws"],
         ref.random(500))
    for low, high in ((0, 2), (0, 7), (-3, 40), (0, 2**33 + 5), (5, 2**62)):
        ref = np.random.Generator(np.random.PCG64())
        ref.bit_generator.state = rng.bit_generator.state
        theirs = np.array([ref.integers(low, high) for _ in range(300)], dtype=np.int64)
        got = core.probe("integers", low=np.int64(low), high=np.int64(high), n=np.int64(300),
                         **state)
        same(f"integers [{low}, {high}), seed {seed}", got["draws"], theirs)
    p = np.random.default_rng(seed + 1).random(9)
    p /= p.sum()
    ref = np.random.Generator(np.random.PCG64())
    ref.bit_generator.state = rng.bit_generator.state
    theirs = np.array([ref.choice(len(p), p=p) for _ in range(300)], dtype=np.int64)
    same(f"choice with p, seed {seed}",
         core.probe("choice_p", p=p, n=np.int64(300), **state)["draws"], theirs)
    ref = np.random.Generator(np.random.PCG64())
    ref.bit_generator.state = rng.bit_generator.state
    theirs = []
    for k in range(200):
        theirs.append(ref.random())
        theirs.append(float(ref.integers(0, k % 7 + 2)))
    same(f"interleaved, seed {seed}", core.probe("mixed", n=np.int64(200), **state)["draws"],
         np.array(theirs))

for n in (1, 3, 7, 8, 9, 16, 31, 128, 129, 200, 1000, 4099):
    a = np.random.default_rng(n).normal(size=n) * 10.0 ** np.random.default_rng(n + 1).integers(
        -3, 4, size=n)
    got = core.probe("sum", a=a)
    same(f"sum of {n}", np.atleast_1d(got["sum"]), np.atleast_1d(np.sum(a)))
    same(f"mean of {n}", np.atleast_1d(got["mean"]), np.atleast_1d(np.mean(a)))
    seq = 0.0
    for x in a:
        seq += x * x
    same(f"dot of {n} (in order)", np.atleast_1d(got["dot"]), np.atleast_1d(seq))
    blas = abs(float(got["dot"]) - float(np.dot(a, a))) / max(abs(float(np.dot(a, a))), 1e-300)
    print(f"  numpy's dot of {n} (BLAS natively): relative gap {blas:.1e}")
# the selection of the sparse code: rows full of ties (zeros and repeated values), any kth
for seed in range(400):
    rng = np.random.default_rng(seed)
    cols, rows = int(rng.integers(2, 300)), 3
    v = np.round(rng.normal(size=(rows, cols)) * rng.choice([1, 3, 10]), 0) * rng.choice(
        [0, 1.0], size=(rows, cols), p=[0.3, 0.7])
    kth = int(rng.integers(0, cols))
    x = -np.maximum(v, 0.0)
    got = core.probe("argpartition", v=x, kth=np.int64(kth), rows=np.int64(rows),
                     cols=np.int64(cols))["index"]
    ok = np.array_equal(got, np.argpartition(x, kth, axis=1))
    failures += not ok
    if not ok:
        print(f"argpartition, seed {seed}: APART")
print("argpartition: 400 rows of ties against NumPy's introselect")
print("PASS" if not failures else f"FAIL ({failures})")
sys.exit(1 if failures else 0)
