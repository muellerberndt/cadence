"""A walker learns the beat from the world's reward alone: no teacher, no clock.

Run from this checkout: PYTHONPATH=src python examples/walking_for_reward.py

One continuing `Brain.live` life sees the same drive every moment. A step on the
other foot than the last one earns one unit, paid at the next moment as the
outcome of that step; a repeated step earns nothing. Nothing in the observation
says which foot moved last. The walker carries an efference copy of its own last
command (`efference_amplitude=3.0, efference_decay=0.0`); the control is the
same founder without the copy. Both run the key-door nursery's declared
operating point and arousal genes: aroused and learning through a youth of 100
moments, then routine while the outcomes match what it expects and aroused
again when they do not or when its need goes unmet.

The example prints, per block of 100 moments, how often each walker changed
foot, how often it was aroused, and its income; then the beat of the last 64
moments against a uniform-random walker, a greedy twin loaded from a mid-life
checkpoint (the acquired policy without exploration), the twin's identical
continuation beside the living walker, and the work of each mode. The
reward-rhythm chamber (`benchmarks/rhythm/reward_rhythm.py`) is the frozen
instrument with its fresh seeds, controls and receipts; this example shows one
founder and claims nothing beyond what it prints.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from cadence import ArousalConfig, Brain

DRIVE = np.array([[1.0, 0.0, 0.0, 0.0]])
FEET = "LR"
SEED = 3
MOMENTS = 600
PROBE_AT = 400
AROUSAL = {
    "threshold": 0.2,
    "decay": 0.9,
    "tolerance": 2.0,
    "floor": 0.1,
    "fast": 0.05,
    "slow": 0.005,
    "heat": 2.0,
    "youth": 100,
    "value_surprise": 1.0,
    "need": 0.5,
}


def founder(copy: bool) -> Brain:
    genes = {"efference_amplitude": 3.0, "efference_decay": 0.0} if copy else {}
    brain = Brain.compose(
        4,
        2, backend="cpu",
        modules=(32,),
        seed=SEED,
        episodic=False,
        working_memory_amplitude=0.3,
        working_memory_decay=0.1,
        arousal=ArousalConfig(**AROUSAL),
        **genes,
    )
    # the key-door nursery's declared point; the composed defaults are its control there
    brain.basal_ganglia.config = replace(
        brain.basal_ganglia.config, eta=0.1, eta_bias=0.01, lam=0.95, gamma=0.95, eta_critic=5.0
    )
    return brain


def pay(previous: int | None, action: int) -> float:
    return 0.0 if previous is None else float(action != previous)


def live(brain: Brain, moments: int, *, probe_at: int | None = None) -> dict:
    steps: list[int] = []
    aroused: list[bool] = []
    earned: list[float] = []
    work = {"routine": 0, "aroused": 0, "sweeps_routine": 0, "sweeps_aroused": 0, "learning": 0}
    previous: int | None = None
    reward: float | None = None
    probe = None
    twin_steps: list[int] = []
    twin = None
    twin_previous: int | None = None
    twin_reward: float | None = None
    with tempfile.TemporaryDirectory() as temporary:
        for t in range(moments):
            if t == probe_at:
                probe = brain.save(Path(temporary) / "mid-stride.npz")
                twin, twin_previous, twin_reward = Brain.load(probe), previous, reward
            if twin is not None and len(twin_steps) < 64:
                feedback = {} if twin_reward is None else {"reward": [twin_reward], "done": [False]}
                b = int(twin.live(DRIVE, **feedback)[0])
                twin_reward, twin_previous = pay(twin_previous, b), b
                twin_steps.append(b)
            feedback = {} if reward is None else {"reward": [reward], "done": [False]}
            action = int(brain.live(DRIVE, **feedback)[0])
            reading = brain.last_arousal
            mode = reading["mode"]
            work[mode] += 1
            work["sweeps_" + mode] += int(reading["sweeps"])
            work["learning"] += int(reading["learning_sweeps"])
            aroused.append(mode == "aroused")
            reward, previous = pay(previous, action), action
            steps.append(action)
            earned.append(reward)
        # Keep all block metrics on the same event boundaries, including the unpaid
        # first step. The final-window beat measures adjacent actions inside that window.
        window = steps[-64:]
        changed = [float(a != b) for a, b in zip(window, window[1:], strict=False)]
        out = {
            "changed_foot_by_block": [
                round(float(np.mean(earned[i : i + 100])), 2) for i in range(0, len(earned), 100)
            ],
            "aroused_by_block": [
                round(float(np.mean(aroused[i : i + 100])), 2) for i in range(0, len(aroused), 100)
            ],
            "income_by_block": [
                round(float(np.mean(earned[i : i + 100])), 2) for i in range(0, len(earned), 100)
            ],
            "last_64_beat": round(float(np.mean(changed)), 2) if changed else 0.0,
            "last_64_aroused": round(float(np.mean(aroused[-64:])), 2),
            "last_32_steps": "".join(FEET[s] for s in steps[-32:]),
            "work": work,
        }
        if probe is not None:
            copy = Brain.load(probe)
            greedy = [int(copy.act(DRIVE, greedy=True)[0]) for _ in range(32)]
            out["greedy_twin_beat"] = round(
                float(np.mean([a != b for a, b in zip(greedy, greedy[1:], strict=False)])), 2
            )
            out["twin_continued_identically"] = twin_steps == steps[probe_at : probe_at + 64]
        return out


def run() -> dict:
    walker = live(founder(True), MOMENTS, probe_at=PROBE_AT)
    control = live(founder(False), MOMENTS, probe_at=PROBE_AT)
    rng = np.random.default_rng(SEED)
    random_steps = rng.integers(0, 2, size=MOMENTS)
    random_beat = float(np.mean(random_steps[-64:][1:] != random_steps[-64:][:-1]))
    return {
        "seed": SEED,
        "world": "+1 when the foot changes, 0 when it repeats, paid at the next moment; "
        "the same drive every moment",
        "walker_with_copy": walker,
        "control_without_copy": control,
        "uniform_random_last_64_beat": round(random_beat, 2),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
