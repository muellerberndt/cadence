"""Which opt-in a brain taught on its own path can do without: one removed at a time.

    PYTHONPATH=src python benchmarks/coaching/ablation.py \\
        --out benchmarks/coaching/results/ablation.json

The cart-pole chamber's task, body (``PopulationCode``) and demonstrator (``cartpole.py``). A
brain of modules ``(64, 32)`` with a set of opt-ins watches eight demonstrated episodes, then is
taught on its own path only: three rounds of 16 episodes, every moment taught before its act.
After watching and after each round it is read frozen on the 32 held-out starts. The arms are the
whole set, the set less each opt-in in turn, and the default brain.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from cartpole import GATE, STEPS, CartPole, demonstrator, senses

from cadence import Brain

FULL = {"arousal": True, "maturity": True, "memory_fade": 1.2e-4, "memory_choice": True,
        "efference_amplitude": 1.0, "learning_compiled_moment": True}


def options(arm: str) -> dict:
    if arm == "default":
        return {}
    if arm in ("full", "full-after"):
        return dict(FULL)
    removed = arm.removeprefix("full-")
    return {k: v for k, v in FULL.items() if k != removed}


def episode(brain: Brain, see, seed: int, teach: bool, brain_acts: bool,
            after: bool = False) -> int:
    pole = CartPole(seed)
    brain.reset()
    running = True
    while running:
        x, label = see(pole.state), demonstrator(pole.state)
        if teach and not after:
            brain.lesson(x, [label])
        answer = int(brain.act(x, greedy=True)[0])
        if teach and after:  # the order step(teacher=) avoids: the act already wrote this moment
            brain.lesson(x, [label])
        running = pole.step(answer if brain_acts else label)
    return pole.steps


def read(brain: Brain, see) -> dict:
    frozen = copy.deepcopy(brain)
    steps = [episode(frozen, see, seed, teach=False, brain_acts=True) for seed in GATE]
    return {"balanced": sum(s == STEPS for s in steps), "mean_steps": float(np.mean(steps))}


def life(job: tuple[str, int]) -> dict:
    arm, founder = job
    inputs, see = senses("code")
    brain = Brain.compose(inputs, 2, modules=(64, 32), seed=founder, **options(arm))
    started = time.time()
    after = arm == "full-after"
    for e in range(8):
        episode(brain, see, 40_000 + 100 * founder + e, teach=True, brain_acts=False, after=after)
    reads = [read(brain, see)]
    for r in range(3):
        for e in range(16):
            episode(brain, see, 60_000 + 100 * founder + 16 * r + e, teach=True, brain_acts=True,
                    after=after)
        reads.append(read(brain, see))
    seconds = round(time.time() - started, 1)
    return {"arm": arm, "founder": founder, "reads": reads, "seconds": seconds}


def main(argv: list[str] | None = None) -> int:
    arms = ["full", *(f"full-{k}" for k in FULL), "default", "full-after"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+", default=arms)
    parser.add_argument("--founders", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    jobs = [(arm, f) for arm in args.arms for f in args.founders]
    with Pool(args.processes) as pool:
        rows = pool.map(life, jobs)
    for row in rows:
        reads = " ".join(f"{r['balanced']:2d}/32 {r['mean_steps']:5.1f}" for r in row["reads"])
        print(f"{row['arm']:36} founder {row['founder']}: {reads}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
