"""Matched-experience screen of the experimental credit horizon in Cadence 0.60.

The body exposes stage and the remembered first executed action as a one-hot
observation. It rewards that first decision only after the declared delay;
later choices are executed but do not alter that outcome. This isolates credit
from representation and memory. It is not visual navigation or strategic skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from temporal_credit import Reinforcement, context, layout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(seed, delay):
    return Reinforcement(
        layout(1 + 2 * delay, seed, "flat", ("q0", "q1")),
        actions=2,
        action_input=None,
        value_output=("q0", "q1"),
        discount=delay / (delay + 1),
        exploration=1.0,
        capacity=2048,
        batch_size=16,
        credit_horizon=delay + 1,
        seed=seed,
    )


def collect(agent, delay, preferred):
    outcomes = []
    for episode in range(12):
        chosen, actions = 0, []
        for stage in range(delay + 1):
            result = agent.act(context(stage, chosen, delay), explore=stage == 0)
            if not result["accepted"]:
                raise RuntimeError("collection decision refused")
            if stage == 0:
                chosen = result["action"]
            actions.append(result["action"])
            terminal = stage == delay
            reward = (1 if chosen == preferred else -1) if terminal else 0
            agent.feedback(
                reward,
                None if terminal else context(stage + 1, chosen, delay),
                decision_id=result["decision_id"],
                executed_action=result["action"],
                terminal=terminal,
                learn=False,
            )
        outcomes.append({"episode": episode, "actions": actions, "reward": reward})
    return outcomes


def assess(snapshot, delay, preferred):
    agent = Reinforcement.from_snapshot(snapshot)
    rows = []
    for episode in range(8):
        chosen, actions = 0, []
        for stage in range(delay + 1):
            result = agent.act(context(stage, chosen, delay), explore=False)
            if not result["accepted"]:
                raise RuntimeError("evaluation decision refused")
            if stage == 0:
                chosen = result["action"]
            actions.append(result["action"])
            terminal = stage == delay
            reward = (1 if chosen == preferred else -1) if terminal else 0
            agent.feedback(
                reward,
                None if terminal else context(stage + 1, chosen, delay),
                decision_id=result["decision_id"],
                executed_action=result["action"],
                terminal=terminal,
                learn=False,
            )
        rows.append({"episode": episode, "actions": actions, "reward": reward})
    return {
        "success": sum(row["reward"] > 0 for row in rows) / len(rows),
        "episodes": rows,
    }


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        Path(__file__).with_name("temporal_credit.py"),
        *sorted((root / "src/cadence").glob("*.py")),
    ]
    sources = {str(p.relative_to(root)): digest(p) for p in paths}
    protocol = {
        "seed": args.seed,
        "delays": [1, 8, 32, 128],
        "updates": args.updates,
        "collection_episodes": 12,
        "evaluation_episodes": 8,
        "discount": "delay/(delay+1), fixed analytic normalization control",
        "preferred": "seed parity",
        "arms": ["one_step", "greedy_cut"],
        "sources": sources,
        "seconds_per_delay": args.seconds,
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    reports = []
    for delay in protocol["delays"]:
        start = time.monotonic()
        preferred = args.seed % 2
        founder = make(args.seed, delay)
        collection = collect(founder, delay, preferred)
        original = json.loads(founder.snapshot())
        arms = []
        for name, horizon in (("one_step", 1), ("greedy_cut", delay + 1)):
            saved = {
                **original,
                "config": {**original["config"], "credit_horizon": horizon},
            }
            agent = Reinforcement.from_snapshot(json.dumps(saved))
            before = assess(agent.snapshot(), delay, preferred)
            ledger, work, started = [], Counter(), time.monotonic()
            for update in range(args.updates):
                if time.monotonic() - start > args.seconds:
                    break
                result = agent.replay()
                work.update(result["work"])
                ledger.append(
                    {
                        "update": update,
                        "accepted": result["accepted"],
                        "indices": result.get("indices"),
                        "targets": result.get("targets"),
                        "horizons": result.get("credit_horizons"),
                        "stops": result.get("credit_stops"),
                        "reason": result["reason"],
                    }
                )
            seconds = time.monotonic() - started
            after = assess(agent.snapshot(), delay, preferred)
            checkpoint = args.out / f"delay{delay}-{name}-brain.json"
            checkpoint.write_text(agent.brain.snapshot())
            arms.append(
                {
                    "name": name,
                    "credit_horizon": horizon,
                    "before": before,
                    "after": after,
                    "updates": ledger,
                    "work": dict(work),
                    "training_seconds": seconds,
                    "brain_sha256": digest(checkpoint),
                    "status": "complete"
                    if len(ledger) == args.updates
                    else "time_limit",
                }
            )
        matched = [row["indices"] for row in arms[0]["updates"]] == [
            row["indices"] for row in arms[1]["updates"]
        ]
        report = {
            "delay": delay,
            "preferred": preferred,
            "collection": collection,
            "arms": arms,
            "matched_row_order": matched,
            "wall_seconds": time.monotonic() - start,
        }
        reports.append(report)
        (args.out / "report.json").write_text(
            json.dumps({"protocol": protocol, "cases": reports}, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "delay": delay,
                    "matched": matched,
                    "arms": [
                        {
                            "name": arm["name"],
                            "success": arm["after"]["success"],
                            "seconds": arm["training_seconds"],
                            "accepted": sum(row["accepted"] for row in arm["updates"]),
                            "status": arm["status"],
                        }
                        for arm in arms
                    ],
                }
            ),
            flush=True,
        )
    final = {
        "protocol": protocol,
        "cases": reports,
        "sources_unchanged": all(digest(root / p) == h for p, h in sources.items()),
    }
    (args.out / "report.json").write_text(json.dumps(final, indent=2) + "\n")
    assert final["sources_unchanged"]
    assert all(math.isfinite(case["wall_seconds"]) for case in reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=180)
    run(parser.parse_args())
