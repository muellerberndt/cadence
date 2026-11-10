"""Seconds of a whole life: compose, place, and step to the end, under a backend or ``auto``.

A life is a brain composed fresh and ``--moments`` moments of ``step`` on ``--rows`` streams, with
rewards from the second moment on; its seconds run from before ``Brain.compose`` to after the last
moment. ``--plan`` calibrates the torch transport at the life's rows before the first moment
(``NeuralGraph.calibrate_transport``), and its seconds are counted in the life::

    python benchmarks/device/life.py --brain chamber --rows 256 --moments 300
    python benchmarks/device/life.py --brain speech --rows 1 --moments 100 --plan
    python benchmarks/device/life.py --brain speech --rows 16 --backend cpu --moments 20

This is a benchmark of a source checkout: ``PYTHONPATH=src`` from the repository's root.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import BRAINS, _place, _sync, environment  # noqa: E402

import cadence as cd  # noqa: E402


def life(name: str, backend: str, rows: int, moments: int, plan: bool, seed: int) -> dict[str, Any]:
    inputs, motor, options = BRAINS[name]
    rng = np.random.default_rng(seed)
    xs = [rng.random((rows, inputs)) * (rng.random((rows, inputs)) < 0.3) for _ in range(moments)]
    rs = [(rng.random(rows) < 0.3).astype(float) for _ in range(moments)]
    place = {} if backend == "auto" else _place(backend)
    warnings.simplefilter("ignore")
    start = time.perf_counter()
    brain = cd.Brain.compose(inputs, motor, seed=seed, **options, **place)
    composed = time.perf_counter() - start
    brain.step(xs[0])
    _sync(brain)
    first = time.perf_counter() - start - composed
    calibrated = 0.0
    if plan and brain.brain._torch is not None:
        began = time.perf_counter()
        brain.brain.calibrate_transport(batches=(rows,))
        calibrated = time.perf_counter() - began
    for t in range(1, moments):
        brain.step(xs[t], reward=rs[t - 1])
    _sync(brain)
    total = time.perf_counter() - start
    placement = getattr(brain, "placement", {}) or {}
    return {
        "brain": name, "backend": backend, "rows": rows, "moments": moments, "plan": plan,
        "placed": placement.get("backend"), "device": placement.get("device"),
        "seconds": total, "composed": composed, "first": first, "calibrated": calibrated,
        "per_moment": (total - composed - first) / max(moments - 1, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--brain", default="chamber", choices=BRAINS)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--moments", type=int, default=100)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    row = life(args.brain, args.backend, args.rows, args.moments, args.plan, args.seed)
    line = json.dumps({**row, "environment": environment()})
    if args.out:
        with open(args.out, "a") as sink:
            print(line, file=sink)
    print(f"{row['brain']} {row['backend']} -> {row['placed']} rows {row['rows']} "
          f"plan {row['plan']}: {row['seconds']:.2f} s, {row['per_moment'] * 1e3:.2f} ms a moment")


if __name__ == "__main__":
    main()
