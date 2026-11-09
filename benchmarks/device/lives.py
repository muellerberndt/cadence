"""Seconds per life: many continuing lives of one topology, four ways.

A life is ``Brain.compose(35, 18, seed=k)``, the chamber's default brain, stepping ``--moments``
moments of ``step`` on observations and rewards of its own; ``--composition lives`` composes it
with ``recall=False, episodic=False``, the composition ``cadence.Lives`` holds. Four ways to run
``--lives`` of them:

* ``serial``: one process steps every life in turn on ``--backend``; each life is its own brain.
* ``pool``: ``--workers`` processes on the CPU, one thread each, each stepping its share of the
  lives in turn; each life is its own brain.
* ``streams``: one brain on ``--backend`` steps every life as one of its streams, in one call per
  moment. Its streams share the brain's parameters and learning, so this is one brain's lives, a
  bound on what a batch of independent lives in one call could reach on that device, and not
  independent lives.
* ``lives``: ``cadence.Lives`` on ``--backend``, every life a brain of its own with its own
  parameters, one call per moment for all of them (``--composition lives`` alone).

Each reading states the seconds to build the lives and the seconds per life and moment of the
stepping that follows (after ``--warm`` moments of each life)::

    python benchmarks/device/lives.py --ways serial pool --lives 12 128 --workers 8
    python benchmarks/device/lives.py --ways streams --backend mlx --lives 12 128
    python benchmarks/device/lives.py --ways streams serial --backend torch:cuda --lives 12 128
    python benchmarks/device/lives.py --composition lives --ways lives streams pool \
        --backend torch:cuda --lives 12 128

This is a benchmark of a source checkout: ``PYTHONPATH=src`` from the repository's root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import _place, _sync, environment  # noqa: E402

import cadence as cd  # noqa: E402

INPUTS, MOTOR = 35, 18
COMPOSITION: dict[str, object] = {}  # ``--composition lives``: recall=False, episodic=False


def _world(seed: int, rows: int, moments: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Every moment's observations and the rewards of the moment before, for ``rows`` lives."""
    rng = np.random.default_rng(seed)
    xs = [rng.random((rows, INPUTS)) * (rng.random((rows, INPUTS)) < 0.3) for _ in range(moments)]
    rs = [(rng.random(rows) < 0.3).astype(float) for _ in range(moments)]
    return xs, rs


def _compose(seed: int, backend: str) -> cd.Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(INPUTS, MOTOR, seed=seed, **_place(backend), **COMPOSITION)


def _live(brain: cd.Brain, xs: list[np.ndarray], rs: list[np.ndarray], first: int, last: int,
          row: slice) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(first, last):
            if t:
                brain.step(xs[t][row], reward=rs[t][row])
            else:
                brain.step(xs[t][row])


def _share(
    lives: list[int], backend: str, moments: int, warm: int,
    composition: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """A worker's lives, in turn: seconds to build them and seconds of their measured moments."""
    if composition is not None:
        COMPOSITION.update(composition)
    start = time.perf_counter()
    brains = [_compose(k, backend) for k in lives]
    built = time.perf_counter() - start
    worlds = [_world(1000 + k, 1, moments + warm) for k in lives]
    for brain, (xs, rs) in zip(brains, worlds, strict=True):
        _live(brain, xs, rs, 0, warm, slice(None))
        _sync(brain)
    start = time.perf_counter()
    for brain, (xs, rs) in zip(brains, worlds, strict=True):
        _live(brain, xs, rs, warm, warm + moments, slice(None))
    for brain in brains:
        _sync(brain)
    return built, time.perf_counter() - start


def serial(lives: int, backend: str, moments: int, warm: int) -> dict[str, Any]:
    built, stepped = _share(list(range(lives)), backend, moments, warm)
    return {"built": built, "per_life_moment": stepped / (lives * moments)}


def pool(lives: int, workers: int, moments: int, warm: int) -> dict[str, Any]:
    shares = [list(range(lives))[w::workers] for w in range(workers)]
    shares = [s for s in shares if s]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"  # the workers' arithmetic, one thread each
    with ProcessPoolExecutor(len(shares)) as executor:
        executor.submit(int, 0).result()  # the workers start before the clock
        start = time.perf_counter()
        results = list(executor.map(_share, shares, ["cpu"] * len(shares),
                                    [moments] * len(shares), [warm] * len(shares),
                                    [dict(COMPOSITION)] * len(shares)))
        wall = time.perf_counter() - start
    built = max(r[0] for r in results)
    stepped = max(r[1] for r in results)
    return {"built": built, "per_life_moment": stepped / (lives * moments), "wall": wall,
            "workers": len(shares)}


def streams(lives: int, backend: str, moments: int, warm: int) -> dict[str, Any]:
    start = time.perf_counter()
    brain = _compose(0, backend)
    built = time.perf_counter() - start
    xs, rs = _world(1000, lives, moments + warm)
    _live(brain, xs, rs, 0, warm, slice(None))
    _sync(brain)
    start = time.perf_counter()
    _live(brain, xs, rs, warm, warm + moments, slice(None))
    _sync(brain)
    return {"built": built, "per_life_moment": (time.perf_counter() - start) / (lives * moments),
            "placement": brain.placement["backend"]}


def batched(lives: int, backend: str, moments: int, warm: int) -> dict[str, Any]:
    """``cadence.Lives``: the lives' own parameters, one call per moment for all of them."""
    place = _place(backend)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = cd.Lives(INPUTS, MOTOR, lives, **place)
    built = time.perf_counter() - start
    xs, rs = _world(1000, lives, moments + warm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(warm):
            held.step(xs[t], **({"reward": rs[t]} if t else {}))
        start = time.perf_counter()
        for t in range(warm, warm + moments):
            held.step(xs[t], reward=rs[t])
    if held.brain is not None:
        _sync(held.brain)
    return {"built": built, "per_life_moment": (time.perf_counter() - start) / (lives * moments)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ways", nargs="+", default=["serial", "pool", "streams"],
                        choices=["serial", "pool", "streams", "lives"])
    parser.add_argument("--composition", default="default", choices=["default", "lives"])
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--lives", nargs="+", type=int, default=[12, 128])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--moments", type=int, default=20)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.composition == "lives":
        COMPOSITION.update(recall=False, episodic=False)
    elif "lives" in args.ways:
        parser.error("the lives way holds --composition lives")
    sink = open(args.out, "a") if args.out else sys.stdout
    print(json.dumps({"environment": environment()}), file=sink, flush=True)
    for lives in args.lives:
        for way in args.ways:
            if way == "pool":
                row = pool(lives, args.workers, args.moments, args.warm)
                backend = "cpu"
            elif way == "lives":
                row = batched(lives, args.backend, args.moments, args.warm)
                backend = args.backend
            elif way == "serial":
                row = serial(lives, args.backend, args.moments, args.warm)
                backend = args.backend
            else:
                row = streams(lives, args.backend, args.moments, args.warm)
                backend = args.backend
            row = {"way": way, "lives": lives, "backend": backend,
                   "composition": args.composition, **row}
            print(json.dumps(row), file=sink, flush=True)
            if sink is not sys.stdout:
                print(f"{way:8} {backend:11} {lives:4} lives  "
                      f"{row['per_life_moment'] * 1e3:8.3f} ms per life and moment, "
                      f"built in {row['built']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
