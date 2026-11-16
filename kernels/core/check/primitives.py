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
# zeros of either sign and NaN in sums, maximum, minimum and clip (NumPy on ARM: between zeros
# maximum gives +0.0 and minimum -0.0 in either order; clip keeps x at a bound; a NaN bound is
# the answer; a sum starts from +0.0)
zeros = np.array([-0.0, 0.0, -0.0, 0.0, 1.5, np.nan, -0.0, 0.5, 0.5, 0.5])
others = np.array([0.0, -0.0, -0.0, 0.0, -2.0, 1.0, 0.0, 0.5, 0.5, 0.5])
lo = np.array([0.0, -1.0, 0.0, -0.0, 0.0, 0.0, -0.0, np.nan, 0.0, 0.5])
hi = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, np.nan, 0.5])
got = core.probe("elementwise", a=zeros, b=others, lo=lo, hi=hi)
same("maximum, zeros and NaN", got["maximum"], np.maximum(zeros, others))
same("minimum, zeros and NaN", got["minimum"], np.minimum(zeros, others))
same("clip, zeros and NaN bounds", got["clip"], np.array([np.clip(x, a, b) for x, a, b in
                                                          zip(zeros, lo, hi, strict=True)]))
for run in (np.array([-0.0]), np.full(9, -0.0), np.zeros(0), np.array([-0.0, 0.0])):
    same(f"sum of {run.tolist()}", np.atleast_1d(core.probe("sum", a=run)["sum"]),
         np.atleast_1d(np.sum(run)))

# einsum's sums of products on ARM: rows against rows (``bj,bj->b``), and a sparse code's
# gather against its weights (``bek,ek->be``) at a batch of one and of several
for seed in range(300):
    rng = np.random.default_rng(seed)
    rows, cols = int(rng.integers(1, 5)), int(rng.integers(1, 600))
    a = rng.normal(size=(rows, cols)) * (rng.random((rows, cols)) < 0.7)
    b = rng.normal(size=(rows, cols))
    got = core.probe("einsum", a=a, b=b, rows=np.int64(rows), cols=np.int64(cols))
    ok = got["dot"].tobytes() == np.einsum("bj,bj->b", a, b).tobytes()
    failures += not ok
    if not ok or seed < 3:
        print(f"einsum bj,bj->b of {cols}, seed {seed}: {'bitwise' if ok else 'APART'}")
    batch, units = int(rng.integers(1, 5)), int(rng.integers(1, 40))
    fan_in = int(rng.integers(1, 40))
    x = rng.normal(size=(batch, fan_in + 40)) * (rng.random((batch, fan_in + 40)) < 0.7)
    index = np.sort(np.stack([rng.choice(fan_in + 40, fan_in, replace=False)
                              for _ in range(units)]), axis=1)
    w = rng.normal(size=(units, fan_in))
    gathered = x[:, index]
    theirs = np.einsum("bek,ek->be", gathered, w)
    flat = gathered.reshape(batch * units, fan_in)
    got = core.probe("einsum", a=flat, b=np.tile(w, (batch, 1)), rows=np.int64(batch * units),
                     cols=np.int64(fan_in))
    ours = (got["dot"] if batch == 1 else got["fused"]).reshape(batch, units)
    ok = ours.tobytes() == theirs.tobytes()
    failures += not ok
    if not ok or seed < 3:
        print(f"einsum bek,ek->be at batch {batch}, seed {seed}: {'bitwise' if ok else 'APART'}")
print("einsum: 300 cases of each against NumPy's")
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
# the recall memory's orders: the product in order (BLAS natively, its gap shown), a run's
# maximum, running sum and stable order (ties, zeros of both signs, infinities and NaN), and
# logaddexp (the C library's log1p and exp, its gap shown)
for n, k, m in ((1, 128, 128), (1, 30, 19), (3, 7, 1), (2, 129, 5)):
    rng = np.random.default_rng(n * k * m)
    a, b = rng.normal(size=(n, k)), rng.normal(size=(k, m))
    seq = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for t in range(k):
                seq[i, j] += a[i, t] * b[t, j]
    got = core.probe("matmul", a=a, b=b)["product"]
    same(f"matmul {n}x{k} @ {k}x{m} (in order)", got, seq)
    print(f"  numpy's matmul (BLAS natively): largest gap {np.max(np.abs(got - a @ b)):.1e}")
for seed in range(200):
    rng = np.random.default_rng(seed)
    v = rng.choice([0.0, -0.0, 1.5, -2.0, np.inf, -np.inf, np.nan, 0.25], size=int(
        rng.integers(1, 40)), p=[0.2, 0.1, 0.2, 0.15, 0.05, 0.1, 0.05, 0.15])
    v = np.where(rng.random(len(v)) < 0.5, v, rng.normal(size=len(v)))
    x, y = rng.normal(size=50) * 30, rng.normal(size=50) * 30
    x[:5], y[:5] = (-np.inf, np.inf, 1.0, -np.inf, np.nan), (-np.inf, np.inf, 1.0, 3.0, 0.0)
    got = core.probe("orders", v=v, x=x, y=y)
    ok = (got["cumsum"].tobytes() == np.cumsum(v).tobytes()
          and np.array_equal(got["argsort"], np.argsort(v, kind="stable"))
          and (np.isnan(got["max"]) == np.isnan(np.max(v)))
          and (np.isnan(got["max"]) or got["max"] == np.max(v)))
    failures += not ok
    if not ok:
        print(f"orders, seed {seed}: APART")
    theirs = np.logaddexp(x, y)
    if not np.array_equal(got["logaddexp"], theirs, equal_nan=True):
        gap = np.nanmax(np.abs(got["logaddexp"] - theirs))
        print(f"  logaddexp, seed {seed}: largest gap {gap:.1e} (the C library's)")
print("orders: 200 runs of ties, signed zeros, infinities and NaN against NumPy")
print("PASS" if not failures else f"FAIL ({failures})")
sys.exit(1 if failures else 0)
