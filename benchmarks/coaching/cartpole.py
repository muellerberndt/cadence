"""Teaching a brain to balance a pole: what it sees, and whose path its lessons follow.

    PYTHONPATH=src python benchmarks/coaching/cartpole.py \
        --out benchmarks/coaching/results/cartpole.json

The task is Gymnasium's CartPole-v1, reproduced here in NumPy (the same equations, constants,
Euler step, termination and 500-step limit), so the chamber needs nothing beyond the library. A
demonstrator pushes toward where the pole falls, with small position and velocity terms against
drift, and balances every one of the 32 held-out starts for 500 steps.

Every arm composes a default brain per founder and reads the 32 held-out starts frozen
(greedy acts on a copy, nothing taught). The arms:

- ``raw``: the four readings as they come, eight watched episodes;
- ``code``: the readings through ``PopulationCode`` (16 tuned inputs a reading), eight watched;
- ``code+dense``: as ``code``, beside 40 inputs that read 0.75 at every moment;
- ``coach``: ``code``, eight watched, then four rounds of 16 episodes on the brain's own path,
  every moment taught (surprise lessons skip the rows it answers right);
- ``mixed``: as ``coach``, each of 8 own episodes a round followed by a watched one;
- ``coach+efference``: as ``coach``, on a brain with the efference copy
  (``efference_amplitude=1.0``);
- ``percentile``: as ``code``, over ranges from the 1st to the 99th percentile of each reading in
  eight other demonstrated episodes, unclamped (``clamp=False``): a reading past its range fades
  and goes dark;
- ``percentile+clamp``: the same ranges, a reading past its range read at the nearer end;
- ``code+narrow``: ``code`` beside ``percentile+clamp``'s code, a wide code of the task's limits
  and a narrow one of the demonstrator's states (128 inputs).

Two more arms teach many poles at once, each pole one of the brain's 32 streams and every lesson
a row per stream (the own-path command in the README):

- ``streams``: watches 32 demonstrated poles side by side for 125 steps (4,000 rows, as the eight
  watched episodes), then four rounds of 250 steps in which 16 poles move by the brain's answer
  and 16 by the demonstrator's; every pole's state is taught at every step, before the act. A
  pole that falls or reaches 500 steps is replaced by a fresh one on the same stream;
- ``streams-own``: as ``streams``, every pole moving by the brain's answer in the rounds.

Two arms correct each of the brain's own episodes after it ends, through
``cadence.speech.correct.Rehearsal``: the brain plays the episode without lessons, and the episode's
moments with the demonstrator's answers are taught as one correction:

- ``correct``: the correction alone (``Rehearsal(rows=0)``);
- ``correct+rehearsal``: the correction heard beside three of the eight watched episodes at lesson
  weight 0.25 (``Rehearsal(rows=3, weight=0.25)``, the watched episodes in its bank).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cadence import Brain, PopulationCode

GRAVITY, CART, POLE, HALF_LENGTH, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
X_LIMIT, ANGLE_LIMIT, STEPS = 2.4, 12 * 2 * math.pi / 360, 500
GATE = range(10_000, 10_032)
STREAMS, WATCH_STEPS, ROUND_STEPS = 32, 125, 250
WATCHED, ROUNDS, OWN = 8, 4, 16  # watched episodes, rounds, own episodes a round
RANGES = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.0, 3.0)]


class CartPole:
    """CartPole-v1's dynamics: the state is (x, x_dot, theta, theta_dot); action 1 pushes right."""

    def __init__(self, seed: int):
        self.state = np.random.default_rng(seed).uniform(-0.05, 0.05, size=4)
        self.steps = 0

    def step(self, action: int) -> bool:
        """Advance one step; True while the pole stands and the episode runs."""
        x, x_dot, theta, theta_dot = self.state
        force = FORCE if action == 1 else -FORCE
        cos, sin = math.cos(theta), math.sin(theta)
        total, pole_moment = CART + POLE, POLE * HALF_LENGTH
        temp = (force + pole_moment * theta_dot**2 * sin) / total
        arm = HALF_LENGTH * (4.0 / 3.0 - POLE * cos**2 / total)
        theta_acc = (GRAVITY * sin - cos * temp) / arm
        x_acc = temp - pole_moment * theta_acc * cos / total
        self.state = np.array([x + TAU * x_dot, x_dot + TAU * x_acc,
                               theta + TAU * theta_dot, theta_dot + TAU * theta_acc])
        self.steps += 1
        fallen = abs(self.state[0]) > X_LIMIT or abs(self.state[2]) > ANGLE_LIMIT
        return not fallen and self.steps < STEPS


def demonstrator(state: np.ndarray) -> int:
    return int(state[2] + 0.5 * state[3] + 0.05 * state[0] + 0.2 * state[1] > 0)


def percentile_ranges() -> list[tuple[float, float]]:
    """The 1st to 99th percentile of each reading over eight demonstrated episodes."""
    states = []
    for seed in range(20_000, 20_008):
        pole = CartPole(seed)
        running = True
        while running:
            states.append(pole.state)
            running = pole.step(demonstrator(pole.state))
    low, high = np.percentile(np.array(states), [1, 99], axis=0)
    return list(zip(low.tolist(), high.tolist(), strict=True))


def senses(arm: str):
    if arm == "raw":
        return 4, lambda s: np.asarray(s, np.float64).reshape(1, 4)
    if arm.startswith("percentile"):
        code = PopulationCode(percentile_ranges(), per=16, clamp=arm == "percentile+clamp")
        return code.inputs, code
    if arm == "code+narrow":
        wide, narrow = PopulationCode(RANGES, per=16), PopulationCode(percentile_ranges(), per=16)
        return wide.inputs + narrow.inputs, lambda s: np.hstack([wide(s), narrow(s)])
    code = PopulationCode(RANGES, per=16)
    if arm == "code+dense":
        return code.inputs + 40, lambda s: np.hstack([code(s), np.full((1, 40), 0.75)])
    return code.inputs, code


def episode(brain: Brain, see, seed: int, mode: str) -> int:
    """``watch``: the demonstrator acts and every moment is taught. ``coach``: the brain acts and
    its moments that differ are taught. ``frozen``: the brain acts and nothing is taught."""
    pole = CartPole(seed)
    brain.reset()
    running = True
    while running:
        x, label = see(pole.state), demonstrator(pole.state)
        if mode in ("watch", "coach"):
            # before the act, as step(teacher=) orders them; a row it answers right is skipped
            brain.lesson(x, [label])
        answer = int(brain.act(x, greedy=True)[0])
        running = pole.step(label if mode == "watch" else answer)
    return pole.steps


def gate(brain: Brain, see) -> list[int]:
    frozen = copy.deepcopy(brain)
    return [episode(frozen, see, seed, "frozen") for seed in GATE]


def streams_life(arm: str, founder: int) -> dict:
    """``streams`` and ``streams-own``: 32 poles as the brain's 32 streams, a lesson of 32 rows a
    step, taught before the act."""
    inputs, code = senses("code")
    brain = Brain.compose(inputs, 2, seed=founder)
    started, made = time.time(), [0]

    def fresh() -> CartPole:
        made[0] += 1
        return CartPole(80_000 + 100_000 * founder + made[0])

    def run(steps: int, own: np.ndarray) -> None:
        poles = [fresh() for _ in range(STREAMS)]
        brain.reset()
        for _ in range(steps):
            states = np.array([pole.state for pole in poles])
            x, labels = code(states), np.array([demonstrator(state) for state in states])
            brain.lesson(x, labels)
            answers = brain.act(x, greedy=True).astype(int)
            for k, pole in enumerate(poles):
                if not pole.step(int(answers[k]) if own[k] else int(labels[k])):
                    poles[k] = fresh()

    run(WATCH_STEPS, np.zeros(STREAMS, bool))
    reads = [gate(brain, code)]
    own = np.arange(STREAMS) < (STREAMS if arm == "streams-own" else STREAMS // 2)
    for _ in range(ROUNDS):
        run(ROUND_STEPS, own)
        reads.append(gate(brain, code))
    summary = [{"balanced": sum(s == STEPS for s in g), "mean_steps": float(np.mean(g))}
               for g in reads]
    return {"arm": arm, "founder": founder, "reads": summary,
            "seconds": round(time.time() - started, 1)}


def correction_life(arm: str, founder: int) -> dict:
    """``correct`` and ``correct+rehearsal``: ``coach``'s watched episodes and own starts; each own
    episode played without lessons, then taught as one correction through ``Rehearsal``."""
    from cadence.speech.correct import Rehearsal

    inputs, code = senses("code")
    brain = Brain.compose(inputs, 2, seed=founder)
    rehearsal = Rehearsal(rows=3 if arm == "correct+rehearsal" else 0, weight=0.25, seed=founder)
    started = time.time()
    for e in range(WATCHED):  # the watched episodes of ``coach``, each kept in the bank
        pole, rows, labels = CartPole(40_000 + 100 * founder + e), [], []
        brain.reset()
        running = True
        while running:
            x, label = code(pole.state), demonstrator(pole.state)
            brain.lesson(x, [label])
            brain.act(x, greedy=True)
            rows.append(x[0])
            labels.append(label)
            running = pole.step(label)
        rehearsal.remember(np.array(rows), np.array(labels))
    reads = [gate(brain, code)]
    for r in range(ROUNDS):
        for e in range(OWN):
            pole, rows, labels = CartPole(60_000 + 100 * founder + 16 * r + e), [], []
            brain.reset()
            running = True
            while running:
                x = code(pole.state)
                rows.append(x[0])
                labels.append(demonstrator(pole.state))
                running = pole.step(int(brain.act(x, greedy=True)[0]))
            rehearsal.correct(brain, np.array(rows), np.array(labels))
        reads.append(gate(brain, code))
    summary = [{"balanced": sum(s == STEPS for s in g), "mean_steps": float(np.mean(g))}
               for g in reads]
    return {"arm": arm, "founder": founder, "reads": summary,
            "seconds": round(time.time() - started, 1)}


def life(arm: str, founder: int) -> dict:
    if arm.startswith("streams"):
        return streams_life(arm, founder)
    if arm.startswith("correct"):
        return correction_life(arm, founder)
    coached = arm in ("coach", "mixed", "coach+efference")
    inputs, see = senses("code" if coached else arm)
    copy_gain = 1.0 if arm == "coach+efference" else 0.0
    brain = Brain.compose(inputs, 2, seed=founder, efference_amplitude=copy_gain)
    started = time.time()
    for e in range(WATCHED):
        episode(brain, see, 40_000 + 100 * founder + e, "watch")
    reads = [gate(brain, see)]
    if coached:
        for r in range(ROUNDS):
            for e in range(OWN if arm == "coach" else OWN // 2):
                episode(brain, see, 60_000 + 100 * founder + 16 * r + e, "coach")
                if arm == "mixed":
                    episode(brain, see, 70_000 + 100 * founder + 16 * r + e, "watch")
            reads.append(gate(brain, see))
    summary = [{"balanced": sum(s == STEPS for s in g), "mean_steps": float(np.mean(g))}
               for g in reads]
    seconds = round(time.time() - started, 1)
    return {"arm": arm, "founder": founder, "reads": summary, "seconds": seconds}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+",
                        default=["raw", "code", "code+dense", "coach", "mixed", "coach+efference",
                                 "percentile", "percentile+clamp"])
    parser.add_argument("--founders", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    demo = [_demonstrate(seed) for seed in GATE]
    jobs = [(arm, f) for arm in args.arms for f in args.founders]
    partial = None if args.out is None else args.out.with_suffix(".partial.jsonl")
    rows = []
    with Pool(args.processes) as pool:  # each life written as it ends, in case the run is cut
        for row in pool.imap_unordered(_life, jobs):
            rows.append(row)
            if partial is not None:
                partial.parent.mkdir(parents=True, exist_ok=True)
                with partial.open("a") as held:
                    held.write(json.dumps(row) + "\n")
    rows.sort(key=lambda row: (jobs.index((row["arm"], row["founder"]))))
    for row in rows:
        reads = " ".join(f"{r['balanced']:2d}/32 {r['mean_steps']:5.1f}"
                         for r in row["reads"])
        print(f"{row['arm']:16} founder {row['founder']}: {reads}  ({row['seconds']} s)",
              flush=True)
    receipt = {"task": "CartPole-v1 dynamics in NumPy", "gate": [GATE.start, GATE.stop],
               "demonstrator": {"balanced": sum(s == STEPS for s in demo),
                                "mean_steps": float(np.mean(demo))},
               "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
    return 0


def _life(job: tuple[str, int]) -> dict:
    return life(*job)


def _demonstrate(seed: int) -> int:
    pole = CartPole(seed)
    while pole.step(demonstrator(pole.state)):
        pass
    return pole.steps


if __name__ == "__main__":
    raise SystemExit(main())
