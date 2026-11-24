"""The bound records' compiled read against its Python, in every read of real lives: the
composed brain and a brain with a voice and records of its own, each read made both ways and
compared (to the bit under Pyodide; natively NumPy's BLAS sums its dot products otherwise, so
the gap is reported).

    python kernels/wasm/browser/run.py records.py [moments]
    python kernels/wasm/browser/records.py fused [moments]          # native
"""
from __future__ import annotations

import sys

import engine

ENGINE = engine.choose(
    "wasm" if engine.BROWSER else (sys.argv[1] if len(sys.argv) > 1 else "fused"))
MOMENTS = int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 150

import numpy as np  # noqa: E402
from compositions import COMPOSED, observation, things_map  # noqa: E402

import cadence as cd  # noqa: E402
from cadence import bound, compiled  # noqa: E402

assert compiled.available("_bound_recall"), "the bound records' kernel is not in use"
reads, apart, worst = 0, 0, 0.0
python_read = bound.BoundRecords._read


def both(self, code, row):
    global reads, apart, worst
    fast = compiled.bound_recall(self, code)
    slow = self._agreed(self._taking(code), row)
    reads += 1
    same = list(fast) == list(slow) and all(
        np.array_equal(fast[k].view(np.uint64), slow[k].view(np.uint64)) for k in fast)
    if not same:
        apart += 1
        if list(fast) == list(slow):
            worst = max([worst] + [float(np.abs(fast[k] - slow[k]).max()) for k in fast])
        else:
            worst = float("inf")
    return fast


bound.BoundRecords._read = both
VOICED = dict(senses={"eyes": 24, "heard": 16, "body": 4}, actions=4, code=64,
              bound=24, voice="heard", recall_capacity=8)
for label, options in (("the composed brain", COMPOSED), ("a voice and records", VOICED)):
    brain = cd.Brain.compose(**options, backend="cpu", seed=5, arousal=True)
    if options is COMPOSED:
        brain.space = things_map()
    rng = np.random.default_rng(9)
    words = np.eye(16)[rng.integers(0, 16, 6)]
    for i in range(MOMENTS):
        if options is COMPOSED:
            brain.space.moment(seen=[(0, [6.0, 6.0], {"hue": np.eye(4)[0]}),
                                     (1, [4.0 + 0.1 * (i % 120), 12.0], {"hue": np.eye(4)[2]})],
                               lit=lambda points: points[:, 0] < 16.0)
        x = observation(options, rng)
        if "heard" in options["senses"] and i % 3 == 0:
            x["heard"] = words[i % len(words)][None]
        if brain.pending_feedback:
            brain.live(x, reward=[float(rng.random() < 0.4)])
        else:
            brain.live(x)
    print(f"{label:22s} reads {reads}, apart {apart}, largest gap {worst:.2e}", flush=True)
bound.BoundRecords._read = python_read
engine.finish(apart == 0 if engine.BROWSER else worst < 1e-9)
