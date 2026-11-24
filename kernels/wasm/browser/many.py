"""N composed brains a moment, each with its map of things, all in this process: milliseconds
per moment of the N and per brain.

    python kernels/wasm/browser/run.py many.py wasm|numpy N [moments] [warm] [seed]
    python kernels/wasm/browser/run.py --workers K many.py wasm N [moments] [warm] {k}
    python kernels/wasm/browser/many.py fused|numpy N [moments] [warm] [seed]    # native
"""
from __future__ import annotations

import json
import sys
import time

import engine

ENGINE = engine.choose(sys.argv[1])
N = int(sys.argv[2])
MOMENTS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
WARM = int(sys.argv[4]) if len(sys.argv) > 4 else 3
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 1

import numpy as np  # noqa: E402
from compositions import COMPOSED, observation, things_map  # noqa: E402

import cadence as cd  # noqa: E402

rng = np.random.default_rng(SEED)
brains = []
for k in range(N):
    brain = cd.Brain.compose(**COMPOSED, backend="cpu", seed=1000 * SEED + k, arousal=True)
    brain.space = things_map()
    brains.append(brain)


def moment(i: int) -> None:
    for k, brain in enumerate(brains):
        seen = [(0, [6.0, 6.0], {"hue": np.eye(4)[0]}),
                (1, [4.0 + 0.1 * ((i + k) % 120), 12.0], {"hue": np.eye(4)[2]})]
        brain.space.moment(seen=seen, lit=lambda points: points[:, 0] < 16.0)
        x = observation(COMPOSED, rng)
        if brain.pending_feedback:
            brain.live(x, reward=[float(rng.random() < 0.4)])
        else:
            brain.live(x)


for i in range(WARM):
    moment(i)
took = []
for i in range(MOMENTS):
    began = time.perf_counter()
    moment(WARM + i)
    took.append(time.perf_counter() - began)
ms = 1e3 * float(np.median(took))
print("MANY " + json.dumps({"engine": ENGINE, "browser": engine.BROWSER, "brains": N,
                            "ms_per_moment": ms, "ms_per_brain": ms / N}), flush=True)
