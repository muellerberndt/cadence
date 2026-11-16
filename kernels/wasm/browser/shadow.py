"""Real lives of several compositions, every compiled call checked against the kernels' own
Python (reference.py).

    python kernels/wasm/browser/run.py shadow.py [moments] [every]
"""
import sys
import time

import reference  # noqa: E402
from reference import checked, report  # noqa: E402

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402

MOMENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
reference.EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 1

COMPOSITIONS = [
    ("plain 8 inputs", dict(inputs=8, actions=3)),
    ("chamber 35 inputs", dict(inputs=35, actions=5)),
    ("deep apical (64, 32)", dict(inputs=24, actions=4, modules=(64, 32), apical=0.25,
                                 teaching="deep")),
    ("recall heads 2", dict(inputs=16, actions=4, recall_heads=2, recall_capacity=16)),
    ("no fast settle", dict(inputs=12, actions=3, learning_fast_settle=False)),
    ("momentum, normalize", dict(inputs=12, actions=3, learning=cd.LearnerConfig(
        momentum=0.9, normalize=0.999))),
    ("relative floor", dict(inputs=12, actions=3, learning=cd.LearnerConfig(
        momentum=0.9, normalize=0.999, normalize_relative=0.5))),
    ("normalize, no momentum", dict(inputs=12, actions=3, learning=cd.LearnerConfig(
        normalize=0.99, normalize_relative=0.25))),
]


def life(options, moments, seed=0):
    brain = cd.Brain.compose(**options, backend="cpu", seed=seed, arousal=True)
    rng = np.random.default_rng(seed)
    n = options["inputs"]
    actions = []
    for i in range(moments):
        x = rng.random((1, n))
        if i % 5 == 4:
            actions.append(int(np.atleast_1d(brain.act(x, greedy=True)[0])[0]))
        elif brain.pending_feedback:
            reward = [float(rng.random() < 0.4)]
            actions.append(int(np.atleast_1d(brain.live(x, reward=reward)[0])[0]))
        else:
            actions.append(int(np.atleast_1d(brain.live(x)[0])[0]))
    return actions


t0 = time.time()
for label, options in COMPOSITIONS:
    before = dict(checked)
    life(options, MOMENTS)
    got = {k: checked[k] - before[k] for k in checked if checked[k] - before[k]}
    print(f"{label:24s} checked {got}  ({time.time() - t0:.0f}s)", flush=True)
report()
