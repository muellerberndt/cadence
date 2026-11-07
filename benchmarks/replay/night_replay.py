"""A day of decisions, a night of replay: is a continuing brain's own day worth re-living?

An application can let a saved copy of a ``Brain.compose`` brain re-experience its own
day: it calls ``step(observation, reward=...)`` on the day's recorded observations, with
the reward each replayed choice would have earned, for several passes, and adopts the
copy if a gate passes ([issue 139](https://github.com/muellerberndt/cadence/issues/139)).
``step`` cannot tell such a night from a day, so the night is more experience of the
same day at the same rates, not a consolidation operation. This chamber measures whether
it helps, against the same number of fresh decisions awake, the equal-experience
comparison that [issue 112](https://github.com/muellerberndt/cadence/issues/112) asks for.

The world is one decision per interval. A cue ``c`` in {-1, +1} is in the observation
with lagged outcomes, their momentum and magnitude, the time of day and a constant. The
outcome ``o`` of the interval is signed, in small units. The three actions are to stay
(0), to move left (1) or to move right (2); moving left earns ``+o - cost``, moving
right ``-o - cost`` and staying nothing. In the ``signal`` world ``o = 4 c + noise``, so
the cue decides which move pays; in the ``noise`` world ``o`` is noise and the cue a
coin, so staying is the best policy. This is the contract of the reported paper-trading
loop (hold, buy and sell; an outcome in basis points; a fixed cost per trade).

What runs, per seed and setting:

1. the day: ``days`` decisions on one stream through ``step``;
2. the night: a saved copy replays ``rounds`` rounds of the day, each round the original
   day, its mirror (directional inputs and outcomes negated) and a half-interval shift,
   every pass starting with the reward owed to the pending action and ``done=True``;
3. the awake control: another saved copy continues for the same number of decisions on
   fresh days;
4. the readings, each on a frozen copy: the greedy choices on the day's observations
   with the trace carried as in life, the per-observation policies and their
   dependence on the observation (the mean total variation between each policy and the
   mean policy), the agreement with the cue rule in the signal world, the graph-only
   choice (``predict``, without trace or memory), the memory's recall drive per action,
   and the shares of clipped dopamine and saturated outputs during the day and night.

Nothing here is a trading claim; the market is synthetic and the outcome is a coin or
a cue. Run ``python benchmarks/replay/night_replay.py --help``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from typing import Any

import numpy as np

import cadence as cd

STAY, LEFT, RIGHT = 0, 1, 2
ACTIONS = ("stay", "left", "right")
FEATURES = 8
DIRECTIONAL = np.array([True, True, True, True, False, False, False, False])

SETTINGS: dict[str, dict[str, Any]] = {
    "defaults": {},
    "trace-1-0.8": {"trace_amplitude": 1.0, "trace_decay": 0.8},
    "trace-off": {"trace_amplitude": 0.0},
    "actor-0.1": {"actor_eta": 0.1},
    "trace-1-0.8+actor-0.1": {"trace_amplitude": 1.0, "trace_decay": 0.8, "actor_eta": 0.1},
    "trace-1-0.8+actor-0.03": {"trace_amplitude": 1.0, "trace_decay": 0.8, "actor_eta": 0.03},
    "trace-off+actor-0.1": {"trace_amplitude": 0.0, "actor_eta": 0.1},
    "no-memory+trace-1-0.8+actor-0.1": {
        "trace_amplitude": 1.0,
        "trace_decay": 0.8,
        "actor_eta": 0.1,
        "episodic": False,
    },
    "no-memory": {"episodic": False},
    "frozen-actor": {"frozen_actor": True},
    "no-cost": {"cost": 0.0},
    "reward/5": {"scale": 0.2},
}


def make_day(seed: int, days: int, signal: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observations ``(days, FEATURES)`` in about [-1, 1], outcomes and cues ``(days,)``."""
    rng = np.random.default_rng(seed)
    cue = rng.choice([-1.0, 1.0], size=days + 2)
    outcome = (
        4.0 * cue + rng.normal(0.0, 1.5, size=days + 2)
        if signal
        else rng.normal(0.0, 3.0, size=days + 2)
    )
    obs = np.zeros((days + 2, FEATURES))
    for t in range(days + 2):
        recent = outcome[max(0, t - 4) : t]
        obs[t, 0] = np.clip(outcome[t - 1] / 6.0, -1, 1) if t >= 1 else 0.0
        obs[t, 1] = np.clip(outcome[t - 2] / 6.0, -1, 1) if t >= 2 else 0.0
        obs[t, 2] = np.clip(recent.mean() / 6.0, -1, 1) if t >= 1 else 0.0
        obs[t, 3] = cue[t]
        obs[t, 4] = np.clip(np.abs(recent).mean() / 6.0, 0, 1) if t >= 1 else 0.5
        obs[t, 5] = np.sin(2 * np.pi * (t % 48) / 48)
        obs[t, 6] = np.cos(2 * np.pi * (t % 48) / 48)
        obs[t, 7] = 1.0
    return obs[:days], outcome[:days], cue[:days]


def reward_of(action: int, outcome: float, cost: float) -> float:
    if action == LEFT:
        return outcome - cost
    if action == RIGHT:
        return -outcome - cost
    return 0.0


def mirror(obs: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flipped = obs.copy()
    flipped[:, DIRECTIONAL] *= -1.0
    return flipped, -outcome


def shifted(obs: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (obs[:-1] + obs[1:]), 0.5 * (outcome[:-1] + outcome[1:])


def make_brain(setting: dict[str, Any], seed: int) -> cd.Brain:
    options: dict[str, Any] = {}
    if not setting.get("episodic", True):
        options["episodic"] = False
    if "trace_amplitude" in setting:
        options["working_memory_amplitude"] = setting["trace_amplitude"]
    if "trace_decay" in setting:
        options["working_memory_decay"] = setting["trace_decay"]
    brain = cd.Brain.compose(FEATURES, 3, backend="cpu", observers=(8,), seed=seed, **{"recall": False, **options})
    changes: dict[str, Any] = {}
    if setting.get("frozen_actor"):
        changes.update(eta=0.0, eta_bias=0.0, eta_critic=0.0)
    if "actor_eta" in setting:
        changes.update(eta=setting["actor_eta"], eta_bias=setting["actor_eta"] / 10)
    for name in ("dopamine_cap", "gamma", "lam", "eta_critic"):
        if name in setting:
            changes[name] = setting[name]
    if changes:
        brain.basal_ganglia.config = replace(brain.basal_ganglia.config, **changes)
    return brain


def live(
    brain: cd.Brain,
    obs: np.ndarray,
    outcome: np.ndarray,
    *,
    cost: float,
    scale: float,
    owed: float | None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """One pass of decisions through ``step``. ``owed`` is the reward of the brain's pending
    action from the previous pass (``None`` on the first), delivered with ``done=True``
    so the pass starts a new episode. Returns the sampled actions, the reward now owed to
    the pending action, and the mean learn-report readings of the pass."""
    cap = brain.basal_ganglia.config.dopamine_cap
    actions: list[int] = []
    capped, saturation = [], []
    if owed is None:
        action = brain.step([obs[0]])
    else:
        action = brain.step([obs[0]], reward=[owed * scale], done=[True])
    actions.append(int(action[0]))
    for t in range(1, len(obs)):
        reward = reward_of(actions[-1], float(outcome[t - 1]), cost) * scale
        action = brain.step([obs[t]], reward=[reward], done=[False])
        report = brain.last_learning
        # ``capped`` is reported from 0.73.2 on; older releases are read through td_error.
        capped.append(float(report.get("capped", float(report.get("td_error", 0.0) > cap))))
        saturation.append(float(report.get("saturation", 0.0)))
        actions.append(int(action[0]))
    owed = reward_of(actions[-1], float(outcome[-1]), cost)
    readings = {
        "capped": float(np.mean(capped)) if capped else 0.0,
        "saturation": float(np.mean(saturation)) if saturation else 0.0,
    }
    return np.asarray(actions), owed, readings


def frozen(brain: cd.Brain) -> cd.Brain:
    """A saved and reloaded copy; the live brain is never read for a measurement."""
    with tempfile.TemporaryDirectory() as directory:
        return cd.Brain.load(brain.save(os.path.join(directory, "brain.npz")))


def readings(brain: cd.Brain, obs: np.ndarray, cue: np.ndarray, signal: bool) -> dict[str, Any]:
    """Greedy choices and policies of a frozen copy on the day's observations, in order."""
    copy = frozen(brain)
    copy.reset()
    greedy, policies = [], []
    for observation in obs:
        greedy.append(int(copy.act([observation], greedy=True)[0]))
        policies.append(copy.basal_ganglia.probabilities(copy.basal_ganglia.state)[0])
    choices = np.asarray(greedy)
    p = np.asarray(policies)
    dependence = float(0.5 * np.abs(p - p.mean(axis=0)).sum(axis=1).mean())
    graph = frozen(brain)
    graph.reset()
    predicted = graph.predict(obs)
    out: dict[str, Any] = {
        "greedy": {name: int(np.sum(choices == k)) for k, name in enumerate(ACTIONS)},
        "graph_only": {name: int(np.sum(predicted == k)) for k, name in enumerate(ACTIONS)},
        "mean_policy": [float(v) for v in p.mean(axis=0)],
        "dependence": dependence,
        "distinct_greedy": int(len(set(greedy))),
    }
    if signal:
        rule = np.where(cue > 0, LEFT, RIGHT)
        out["cue_agreement"] = float(np.mean(choices == rule))
    if brain.hippocampus is not None:
        out["recall_drive"] = [float(v) for v in brain.hippocampus.recall(obs).mean(axis=0)]
    return out


def run_one(world: str, name: str, seed: int, *, days: int = 40, rounds: int = 3) -> dict[str, Any]:
    """One seed of one setting in one world: the day, the night, the awake control, and the
    readings before the night, after it and after the awake control."""
    if world not in ("signal", "noise"):
        raise ValueError("world must be 'signal' or 'noise'")
    setting = SETTINGS[name]
    signal = world == "signal"
    cost, scale = setting.get("cost", 2.0), setting.get("scale", 1.0)
    obs, outcome, cue = make_day(seed, days, signal)
    started = time.time()
    brain = make_brain(setting, seed)
    day_actions, owed, day_readings = live(brain, obs, outcome, cost=cost, scale=scale, owed=None)
    before = readings(brain, obs, cue, signal)
    passes = [
        ("day", obs, outcome),
        ("mirror", *mirror(obs, outcome)),
        ("shift", *shifted(obs, outcome)),
    ]
    sleeper = frozen(brain)
    night_actions, night_readings = [], []
    owed_night = owed
    for _ in range(rounds):
        for _, replayed, replayed_outcome in passes:
            actions, owed_night, reading = live(
                sleeper, replayed, replayed_outcome, cost=cost, scale=scale, owed=owed_night
            )
            night_actions.append(actions)
            night_readings.append(reading)
    after_night = readings(sleeper, obs, cue, signal)
    waker = frozen(brain)
    owed_awake, awake_decisions = owed, 0
    for k, (_, replayed, _) in enumerate(passes * rounds):
        fresh_obs, fresh_outcome, _ = make_day(10_000 + 100 * seed + k, len(replayed), signal)
        _, owed_awake, _ = live(
            waker, fresh_obs, fresh_outcome, cost=cost, scale=scale, owed=owed_awake
        )
        awake_decisions += len(fresh_obs)
    after_awake = readings(waker, obs, cue, signal)
    sampled = np.concatenate(night_actions)
    return {
        "world": world,
        "setting": name,
        "options": setting,
        "seed": seed,
        "days": days,
        "rounds": rounds,
        "seconds": round(time.time() - started, 2),
        "day_sampled": {a: int(np.sum(day_actions == k)) for k, a in enumerate(ACTIONS)},
        "day_capped": day_readings["capped"],
        "day_saturation": day_readings["saturation"],
        "night_decisions": int(sampled.size),
        "night_sampled": {a: int(np.sum(sampled == k)) for k, a in enumerate(ACTIONS)},
        "night_capped": float(np.mean([r["capped"] for r in night_readings])),
        "awake_decisions": awake_decisions,
        "before": before,
        "after_night": after_night,
        "after_awake": after_awake,
    }


def summarize(results: list[dict[str, Any]]) -> str:
    """Mean readings per world and setting, before the night, after it and after the awake
    control: dependence, and in the signal world the cue agreement."""
    lines = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault((result["world"], result["setting"]), []).append(result)
    for (world, name), rows in groups.items():
        dependence = [_mean(rows, stage, "dependence") for stage in STAGES]
        agreement = [_mean(rows, stage, "cue_agreement") for stage in STAGES]
        lines.append(
            f"{world:6} {name:34} seeds {len(rows)}  dependence {dependence[0]} -> night "
            f"{dependence[1]} / awake {dependence[2]}  cue agreement {agreement[0]} -> night "
            f"{agreement[1]} / awake {agreement[2]}  capped day "
            f"{np.mean([row['day_capped'] for row in rows]):.2f}"
        )
    return "\n".join(lines)


STAGES = ("before", "after_night", "after_awake")


def _mean(rows: list[dict[str, Any]], stage: str, key: str) -> str:
    values = [row[stage][key] for row in rows if key in row[stage]]
    return f"{np.mean(values):.2f}" if values else "-"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world", default="both", choices=("signal", "noise", "both"))
    parser.add_argument("--settings", default="all", help="comma-separated names, or all")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--days", type=int, default=40, help="decisions per day")
    parser.add_argument("--rounds", type=int, default=3, help="rounds of the three passes")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None, help="JSON path for every run's readings")
    args = parser.parse_args(argv)
    worlds = ("signal", "noise") if args.world == "both" else (args.world,)
    names = list(SETTINGS) if args.settings == "all" else args.settings.split(",")
    unknown = [name for name in names if name not in SETTINGS]
    if unknown:
        raise SystemExit(f"unknown settings {unknown}; known: {', '.join(SETTINGS)}")
    jobs = [(world, name, seed) for world in worlds for name in names for seed in range(args.seeds)]
    results: list[dict[str, Any]] = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_one, world, name, seed, days=args.days, rounds=args.rounds)
                for world, name, seed in jobs
            ]
            for future in futures:
                results.append(future.result())
                print(summarize(results[-1:]), flush=True)
    else:
        for world, name, seed in jobs:
            results.append(run_one(world, name, seed, days=args.days, rounds=args.rounds))
            print(summarize(results[-1:]), flush=True)
    print("\nmeans")
    print(summarize(results))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {"cadence": cd.__version__, "python": sys.version, "results": results},
                handle,
                indent=1,
            )


if __name__ == "__main__":
    main()
