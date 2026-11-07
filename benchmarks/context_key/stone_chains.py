"""Stone Age chains for the episodic context key (cadence#202).

The chamber, its chains and its readings are Will Glynn's, from his open lab notebook
(github.com/WGlynn/cadence-teachers, GPL-3.0: EXP-56, EXP-57, EXP-59, EXP-60, EXP-66), after
Huncho's ambiguous chain (V7.2) and the V5 Stone Age chamber's event order. They are ported
here with credit. Where Will's instrument restored the stream's temporal context outside the
library, the brains here hold decisions out of it with ``memory_context_hold``.

The chamber: five actions (wood, stone, fire, warm, wait) to be taken in a strict order; a
wrong action changes nothing; at most ten decisions per attempt; +2 at success, -1 at the
timeout. The chains differ in what the brain sees at the four stages:

- ``ambiguous``: A, B, A, B, the two A's and the two B's byte-identical (Huncho's V7.2): only
  the brain's own history tells the stages apart;
- ``invisible``: A, A, B, C: the right wood changes nothing visible (EXP-59), the boundary of
  the observation hold, which keeps that decision out of the context;
- ``observed``: the inventory is visible at every stage (EXP-45 and EXP-60).

The brain: ``Brain.compose(7, 5, modules=(48, 24), arousal=True, memory_choice=True,
memory_context=0.5)``. The arms, paired by seed:

- ``founder``: the temporal context as composed;
- ``unchanged``: ``memory_context_hold="unchanged"``;
- ``cost``: a miss that does not end the attempt pays ``-COST`` instead of 0;
- ``punished``: the same cost with ``memory_context_hold="punished"``.

The signals: ``natural`` (the world's reward) and ``corrective`` (in the guided four of every
twelve decisions, a miss that does not end the attempt is followed by the right action as
``teacher=``). The readings, on a saved copy: the greedy exact pass (the chain in four
decisions), the sampled success over ``EVAL`` attempts and the first-try accuracy per stage.

    python benchmarks/context_key/stone_chains.py --chain ambiguous --arms founder unchanged \\
        --seeds 48 --out results/ambiguous.jsonl
    python benchmarks/context_key/stone_chains.py --summary results/ambiguous.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from math import comb
from multiprocessing import get_context
from pathlib import Path

import numpy as np

WOOD, STONE, FIRE, WARM, WAIT = range(5)
RIGHTS = (WOOD, STONE, FIRE, WARM)
DECISIONS, SUCCESS, FAILURE, COST = 10, 2.0, -1.0, 0.05
GUIDED, CYCLE = 4, 12
A = (0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
B = (0.35, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0)
C = (0.35, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0)
INVENTORY = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))  # wood, stone, fire held at each stage
CHAINS = {
    "ambiguous": (A, B, A, B),
    "invisible": (A, A, B, C),
    "observed": tuple((0.35, w, s, f, w / 2.0, s / 2.0, 1.0) for w, s, f in INVENTORY),
}
ARMS = {
    "founder": dict(hold=None, cost=0.0),
    "unchanged": dict(hold="unchanged", cost=0.0),
    "cost": dict(hold=None, cost=COST),
    "punished": dict(hold="punished", cost=COST),
}
SIGNALS = ("natural", "corrective")
SEED_BASE = 202_000  # fresh: Will's runs used 56_000 to 67_000


class Chamber:
    """One attempt at a chain: the stage reached, the decisions taken, how it ended."""

    def __init__(self, chain: str, cost: float = 0.0) -> None:
        self.views, self.cost = CHAINS[chain], float(cost)
        self.stage, self.decisions, self.done, self.success = 0, 0, False, False

    def x(self) -> np.ndarray:
        return np.array([self.views[self.stage]], dtype=float)

    def right(self) -> int:
        return RIGHTS[self.stage]

    def execute(self, action: int) -> tuple[bool, float]:
        ok = action == self.right()
        if ok:
            self.stage += 1
            self.success = self.done = self.stage == len(RIGHTS)
        self.decisions += 1
        if not self.done and self.decisions >= DECISIONS:
            self.done = True
        if self.done:
            return ok, SUCCESS if self.success else FAILURE
        return ok, -self.cost if not ok and self.cost else 0.0


def compose(arm: str, seed: int):
    import cadence as cd

    return cd.Brain.compose(
        7, 5, modules=(48, 24), seed=int(seed), arousal=True, memory_choice=True,
        memory_context=0.5, memory_context_hold=ARMS[arm]["hold"], recall=False,
    )


def train(brain, chain: str, arm: str, signal: str, episodes: int) -> dict:
    cost = ARMS[arm]["cost"]
    world, decisions, wins, labels = Chamber(chain, cost), 0, 0, 0
    action = int(brain.step(world.x())[0])
    for _ in range(episodes):
        while True:
            guided = decisions % CYCLE < GUIDED
            right = world.right()
            ok, reward = world.execute(action)
            decisions += 1
            if world.done:
                wins += world.success
                world = Chamber(chain, cost)
                action = int(brain.step(world.x(), reward=[reward], done=[True])[0])
                break
            teacher = None
            if signal == "corrective" and guided and not ok:
                teacher, labels = np.array([right]), labels + 1
            action = int(brain.step(world.x(), reward=[reward], done=[False], teacher=teacher)[0])
    brain.reset()
    return dict(train_success=wins / episodes, teacher_labels=labels, decisions=decisions)


def attempt(brain, chain: str, greedy: bool, log: list | None = None) -> tuple[bool, int, int]:
    brain.reset()
    world, misses = Chamber(chain), 0
    while not world.done:
        stage = world.stage
        action = int(np.asarray(brain.act(world.x(), greedy=greedy)).reshape(-1)[0])
        ok, _ = world.execute(action)
        if log is not None:
            log.append((stage, misses, bool(ok)))
        misses = 0 if ok else misses + 1
    return world.success, world.stage, world.decisions


def read(path: str, chain: str, attempts: int) -> dict:
    import cadence as cd

    success, stages, decisions = attempt(cd.Brain.load(path), chain, greedy=True)
    brain, log = cd.Brain.load(path), []
    sampled = float(np.mean([attempt(brain, chain, False, log)[0] for _ in range(attempts)]))
    first = []
    for stage in range(len(RIGHTS)):
        tries = [ok for k, missed, ok in log if k == stage and missed == 0]
        first.append(float(np.mean(tries)) if tries else None)
    return dict(
        greedy_exact=bool(success and decisions == len(RIGHTS)), greedy_success=bool(success),
        greedy_stages=int(stages), sampled=sampled, first_try=first,
    )


def life(job: tuple) -> dict:
    chain, arm, signal, seed, episodes, attempts = job
    import cadence as cd

    started = time.time()
    brain = compose(arm, SEED_BASE + seed)
    stats = train(brain, chain, arm, signal, episodes)
    with tempfile.TemporaryDirectory() as folder:
        reading = read(str(brain.save(os.path.join(folder, "brain.npz"))), chain, attempts)
    return dict(chain=chain, arm=arm, signal=signal, seed=seed, episodes=episodes, **stats,
                **reading, seconds=round(time.time() - started, 1), cadence=cd.__version__)


def mcnemar(first_only: int, second_only: int) -> float:
    """The exact two-sided McNemar test of discordant pairs."""
    n = first_only + second_only
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(first_only, second_only) + 1)) / 2**n)


def summary(rows: list[dict], arm: str, against: str) -> dict:
    """Paired readings of ``arm`` against ``against`` per chain, per signal and pooled."""
    by = {(r["chain"], r["arm"], r["signal"], r["seed"]): r for r in rows}
    out = {}
    for chain in sorted({r["chain"] for r in rows}):
        for scope in (*SIGNALS, "pooled"):
            signals = SIGNALS if scope == "pooled" else (scope,)
            pairs = [
                (by[(chain, arm, s, i)], by[(chain, against, s, i)])
                for (c, a, s, i) in by
                if c == chain and a == arm and s in signals and (chain, against, s, i) in by
            ]
            if not pairs:
                continue
            only = sum(x["greedy_exact"] and not y["greedy_exact"] for x, y in pairs)
            other = sum(y["greedy_exact"] and not x["greedy_exact"] for x, y in pairs)
            sampled = [x["sampled"] - y["sampled"] for x, y in pairs]
            out[f"{chain}/{scope}"] = dict(
                pairs=len(pairs),
                exact=(
                    sum(x["greedy_exact"] for x, _ in pairs),
                    sum(y["greedy_exact"] for _, y in pairs),
                ),
                discordant=(only, other), p=mcnemar(only, other),
                sampled=(float(np.mean([x["sampled"] for x, _ in pairs])),
                         float(np.mean([y["sampled"] for _, y in pairs]))),
                sampled_sem=(
                    float(np.std(sampled, ddof=1) / np.sqrt(len(sampled)))
                    if len(sampled) > 1 else 0.0
                ),
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chain", choices=sorted(CHAINS), default="ambiguous")
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=["founder", "unchanged"])
    parser.add_argument("--signals", nargs="+", choices=SIGNALS, default=list(SIGNALS))
    parser.add_argument("--seeds", type=int, default=48, help="seeds per signal, from SEED_BASE")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--eval", type=int, default=100, help="sampled attempts read")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="a JSON-lines file, resumed")
    parser.add_argument("--summary", type=Path, default=None, help="pair the rows of this file")
    parser.add_argument("--against", default="founder")
    args = parser.parse_args(argv)
    if args.summary is not None:
        rows = [json.loads(line) for line in args.summary.read_text().splitlines() if line]
        for arm in sorted({r["arm"] for r in rows} - {args.against}):
            costed = arm == "punished" and any(r["arm"] == "cost" for r in rows)
            against = "cost" if costed else args.against
            for key, value in summary(rows, arm, against).items():
                print(f"{arm} vs {against} {key}: {json.dumps(value)}")
        return 0
    done = set()
    if args.out is not None and args.out.exists():
        done = {(r["chain"], r["arm"], r["signal"], r["seed"], r["episodes"])
                for r in map(json.loads, args.out.read_text().splitlines()) if r}
    jobs = [
        (args.chain, arm, signal, seed, args.episodes, args.eval)
        for seed in range(args.first_seed, args.first_seed + args.seeds)
        for signal in args.signals
        for arm in args.arms
        if (args.chain, arm, signal, seed, args.episodes) not in done
    ]
    with get_context("spawn").Pool(max(1, min(args.workers, len(jobs) or 1))) as pool:
        for row in pool.imap_unordered(life, jobs):
            line = json.dumps(row)
            print(line, flush=True)
            if args.out is not None:
                with args.out.open("a") as handle:
                    handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
