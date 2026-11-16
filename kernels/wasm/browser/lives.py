"""Whole lives of several compositions on one engine: each moment's action and the arrays a
life ends on, one JSON line a life (``compare.py`` reads two engines' lines).

    python kernels/wasm/browser/run.py lives.py wasm [moments] > wasm.txt      # Pyodide
    python kernels/wasm/browser/run.py lives.py numpy [moments] > numpy.txt    # Pyodide
    python kernels/wasm/browser/lives.py fused [moments] > fused.txt           # native

A life acts and lives on random observations: every fifth moment a greedy act, the others a
``live`` with a reward drawn at 0.4, a learning lesson wherever its arousal opens one.
"""
from __future__ import annotations

import base64
import json
import sys
import time

import engine

ENGINE = engine.choose(sys.argv[1] if len(sys.argv) > 1 else "fused")
MOMENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
from compositions import COMPOSITIONS, POCKET, observation, pocket_map  # noqa: E402

engine.fit_32_bit_numpy()

def packed(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype=np.float64).tobytes()).decode()


def life(label, options, moments, seed=0):
    brain = cd.Brain.compose(**options, backend="cpu", seed=seed, arousal=True)
    space = None
    if options is POCKET:
        space = brain.space = pocket_map()
    rng = np.random.default_rng(seed + 1)
    actions = []
    began = time.perf_counter()
    for i in range(moments):
        if space is not None:  # two things seen, one of them moving
            seen = [(0, [6.0, 6.0], {"hue": np.eye(4)[0]}),
                    (1, [4.0 + 0.1 * (i % 120), 12.0], {"hue": np.eye(4)[2]})]
            space.moment(seen=seen, lit=lambda points: points[:, 0] < 16.0)
        x = observation(options, rng)
        if i % 5 == 4:
            action = brain.act(x, greedy=True)
        elif brain.pending_feedback:
            action = brain.live(x, reward=[float(rng.random() < 0.4)])
        else:
            action = brain.live(x)
        actions.append(int(np.atleast_1d(action[0])[0]))
    took = time.perf_counter() - began
    graph = brain.learner.brain
    print("LIFE " + json.dumps({
        "label": label, "engine": ENGINE, "browser": engine.BROWSER, "actions": actions,
        "ms_per_moment": 1e3 * took / moments, "lessons": int(brain.learner.updates),
        "efficacy": packed(graph.efficacy), "bias": packed(graph.bias)}), flush=True)


for label, options in COMPOSITIONS:
    life(label, options, MOMENTS)
