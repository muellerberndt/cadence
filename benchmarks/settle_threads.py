"""Seconds per lesson of a composed brain at several Numba thread counts.

Two brains by default: a deep apical brain (four processing regions with the apical compartment
at the depth benchmark's teaching point) and a flat founder (one module, founder settings).
For every thread count each brain is composed afresh from the same seed and taught the same
lessons, so the runs differ only in the threads; the learned weights must come out bit for bit
the same at every count, and the script checks that they do.

    PYTHONPATH=src python benchmarks/settle_threads.py
    PYTHONPATH=src python benchmarks/settle_threads.py --threads 1 2 4 8 16 32 64 --batch 64
    PYTHONPATH=src python benchmarks/settle_threads.py --split   # the lesson's time by part

The settle's rows run on Numba's threads (``NUMBA_NUM_THREADS`` caps them; this script sets the
count per run with ``numba.set_num_threads``). ``all`` stands for ``NUMBA_NUM_THREADS``. Cap the
BLAS threads (``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``) to taste: the block products of a
sweep already run on distinct Numba threads.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import time
from typing import Any

import numpy as np

import cadence as cd

# the depth benchmark's composed brain: apical compartment, normalized steps with the
# projection-relative floor, nudged phases of a whole 32 sweeps, no memories
DEEP: dict[str, Any] = {
    "apical": 0.25,
    "teaching": "deep",
    "learning_surprise": False,
    "episodic": False,
    "working_memory_amplitude": 0.0,
}
FLAT: dict[str, Any] = {"learning_surprise": False, "episodic": False}
# the parts of a lesson the split reports: function name -> label (cumulative time)
PARTS = {
    "settle_batch": "settles",
    "residual": "residual checks",
    "apical_gain": "held apical gains",
    "contrast": "contrast",
    "_update": "update (incl. contrast)",
    "_relative_rms": "relative floor",
    "apply": "apply step",
}


def compose(kind: str, inputs: int, modules: tuple[int, ...], classes: int) -> cd.Brain:
    if kind == "deep":
        return cd.Brain.compose(inputs, classes, modules=modules, seed=0, **DEEP)
    return cd.Brain.compose(inputs, classes, modules=modules[:1], seed=0, **FLAT)


def lessons(
    count: int, batch: int, inputs: int, classes: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(1)
    return [(rng.uniform(0.0, 1.0, (batch, inputs)), rng.integers(0, classes, batch))
            for _ in range(count)]


def run(kind: str, threads: int, args: argparse.Namespace) -> dict[str, Any]:
    import numba

    numba.set_num_threads(threads)
    brain = compose(kind, args.inputs, tuple(args.modules), args.classes)
    work = lessons(args.warmup + args.lessons, args.batch, args.inputs, args.classes)
    for x, y in work[: args.warmup]:  # compiles the kernels on first use; not timed
        brain.lesson(x, y)
    seconds, sweeps, phases = [], [], []
    profile = cProfile.Profile() if args.split else None
    for x, y in work[args.warmup :]:
        if profile is not None:
            profile.enable()
        started = time.perf_counter()
        report = brain.lesson(x, y)
        seconds.append(time.perf_counter() - started)
        if profile is not None:
            profile.disable()
        phase = [report.get(name, 0.0)
                 for name in ("free_steps", "nudged_steps", "opposite_steps")]
        phases.append(phase)
        sweeps.append(sum(phase))
    out: dict[str, Any] = {
        "kind": kind, "threads": threads, "neurons": int(brain.connectome.n),
        "synapses": int(brain.connectome.synapses),
        "seconds_min": float(np.min(seconds)), "seconds_median": float(np.median(seconds)),
        "sweeps_per_lesson": float(np.mean(sweeps)),
        "weights": brain.brain.efficacy.copy(), "bias": brain.brain.bias.copy(),
        "optimizer": {name: getattr(brain.learner, name).copy() for name in (
            "velocity", "velocity_bias", "second_moment", "second_moment_bias")},
        "phases": phases, "updates": brain.learner.updates,
        "contrast_updates": brain.learner.contrast_updates,
    }
    if profile is not None:
        split: dict[str, float] = {}
        for (filename, _, name), (_, _, _, cumulative, _) in pstats.Stats(profile).stats.items():
            if name in PARTS and "cadence" in filename:
                split[PARTS[name]] = split.get(PARTS[name], 0.0) + cumulative / len(seconds)
        out["split"] = split
    return out


def main() -> None:
    import numba

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threads", nargs="+", default=["1", "2", "4", "8", "all"])
    parser.add_argument("--brains", nargs="+", default=["deep", "flat"], choices=["deep", "flat"])
    parser.add_argument("--inputs", type=int, default=1024)
    parser.add_argument("--modules", type=int, nargs="+", default=[512, 256, 256, 256])
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lessons", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--split", action="store_true", help="time the parts of a lesson")
    parser.add_argument("--out", help="write the readings as JSON to this path")
    args = parser.parse_args()
    most = int(numba.config.NUMBA_NUM_THREADS)
    counts = sorted({most if t == "all" else min(int(t), most) for t in args.threads})
    print(f"{platform.platform()}, {os.cpu_count()} cpus, NUMBA_NUM_THREADS={most}, "
          f"batch {args.batch}, {args.lessons} timed lessons")
    readings = []
    for kind in args.brains:
        base: dict[str, Any] | None = None
        for threads in counts:
            reading = run(kind, threads, args)
            if base is None:
                base = dict(reading)
            same = (np.array_equal(reading["weights"], base["weights"])
                    and np.array_equal(reading["bias"], base["bias"])
                    and all(np.array_equal(value, base["optimizer"][name])
                            for name, value in reading["optimizer"].items())
                    and all(reading[key] == base[key]
                            for key in ("phases", "updates", "contrast_updates")))
            speedup = base["seconds_min"] / reading["seconds_min"]
            print(f"{kind:5s} n={reading['neurons']} synapses={reading['synapses']} "
                  f"threads={threads:3d}  {reading['seconds_min']:.3f} s/lesson (median "
                  f"{reading['seconds_median']:.3f})  speedup {speedup:.2f}x  "
                  f"sweeps/lesson {reading['sweeps_per_lesson']:.0f}  "
                  f"weights {'identical' if same else 'DIFFER'}")
            if "split" in reading:
                parts = ", ".join(f"{k} {v * 1e3:.1f} ms" for k, v in
                                  sorted(reading["split"].items(), key=lambda kv: -kv[1]))
                print(f"      split: {parts}")
            reading.update(speedup=speedup, identical=same)
            arrays = {"weights": reading["weights"], "bias": reading["bias"],
                      **reading["optimizer"]}
            reading["parameter_sha256"] = {
                name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()
            }
            del reading["weights"], reading["bias"]
            del reading["optimizer"]
            readings.append(reading)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"platform": platform.platform(), "readings": readings}, handle, indent=1)
    if not all(reading["identical"] for reading in readings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
