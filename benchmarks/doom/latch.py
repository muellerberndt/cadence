"""Where a learning brain's score goes in the Doom body: its buttons, the saturation of its motor
readout and its dopamine, episode by episode (cadence#285).

    PYTHONPATH=src python benchmarks/doom/latch.py --scenario basic --episodes 300 --seed 0 \\
        --eta founder > results/latch/basic-founder-0.json
    PYTHONPATH=src python benchmarks/doom/latch.py --read results/latch/*.json

``--eta founder`` is the default composition; a number composes ``actor_eta`` at it with the
bias rate a tenth of it. ``--read`` prints each life's first and last quarter.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doom_body  # noqa: E402
import doom_players  # noqa: E402


def live(scenario: str, episodes: int, seed: int, eta: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if eta != "founder":
        options = {"actor_eta": float(eta), "actor_eta_bias": float(eta) / 10}
    body = doom_body.DoomBody(scenario, seed=seed)
    player = doom_players.BrainPlayer("84x84", body.actions, seed=seed, backend="cpu", **options)
    observation, previous = body.reset(), None
    buttons: collections.Counter[int] = collections.Counter()
    saturation: list[float] = []
    dopamine: list[float] = []
    out: list[dict[str, Any]] = []
    while True:
        action = player.act(observation, previous)
        buttons[action] += 1
        report = player.brain.last_learning or {}
        if "saturation" in report:
            saturation.append(float(report["saturation"]))
            dopamine.append(abs(float(report.get("dopamine", 0.0))))
        transition = body.step(action)
        if not transition.done:
            assert transition.observation is not None
            observation, previous = transition.observation, transition
            continue
        out.append({
            "e": body.episode, "outcome": transition.outcome, "return": body.returned,
            "actions": [buttons[i] for i in range(body.actions)],
            "saturation": float(np.mean(saturation)) if saturation else None,
            "dopamine": float(np.mean(dopamine)) if dopamine else None,
        })
        buttons, saturation, dopamine = collections.Counter(), [], []
        if len(out) == episodes:
            body.close()
            return {"eta": eta, "seed": seed, "scenario": scenario, "episodes": out}
        observation, previous = body.reset(), transition


def read(paths: list[Path]) -> None:
    print("| Scenario | `actor_eta` | Founder | Quarter | Button shares | Completion | Return "
          "| Saturation | Mean dopamine magnitude |")
    print("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for path in paths:
        life = json.loads(path.read_text())
        episodes = life["episodes"]
        quarter = max(len(episodes) // 4, 1)
        for name, part in (("first", episodes[:quarter]), ("last", episodes[-quarter:])):
            buttons = np.array([e["actions"] for e in part]).sum(axis=0)
            shares = ", ".join(f"{v:.2f}" for v in buttons / buttons.sum())
            completion = np.mean([e["outcome"] == "completion" for e in part])
            ret = np.mean([e["return"] for e in part])
            sat = np.mean([e["saturation"] for e in part if e["saturation"] is not None])
            dop = np.mean([e["dopamine"] for e in part if e["dopamine"] is not None])
            print(f"| {life['scenario']} | {life['eta']} | {life['seed']} | {name} | {shares} "
                  f"| {completion:.2f} | {ret:.3f} | {sat:.2f} | {dop:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", choices=sorted(doom_body.SCENARIOS), default="basic")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", default="founder")
    parser.add_argument("--read", nargs="*", type=Path)
    args = parser.parse_args()
    if args.read is not None:
        read(args.read)
        return
    print(json.dumps(live(args.scenario, args.episodes, args.seed, args.eta)))


if __name__ == "__main__":
    main()
