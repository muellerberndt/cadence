"""Diagnose delayed-credit targets with matched observed experience.

Stage and the remembered first executed action are explicit body observations.
Only that first action affects the terminal reward. This artificial fixture
isolates target construction; it does not test learned memory or planning.
The episodic-return arm evaluates observed behavior, not general off-policy
optimal values. Stratified replay is a disclosed diagnostic, not a new API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_credit import Reinforcement, context, layout  # noqa: E402

EPISODES = 24
UPDATES = 32
ARM_SECONDS = 180
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32)
ARMS = ("one_step", "greedy_cut", "observed_return")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make(seed, delay):
    return Reinforcement(
        layout(1 + 2 * delay, seed, "ordinary", ("q0", "q1")),
        actions=2,
        action_input=None,
        value_output=("q0", "q1"),
        discount=delay / (delay + 1),
        exploration=1.0,
        capacity=4096,
        batch_size=16,
        credit_horizon=delay + 1,
        seed=seed,
    )


def collect(agent, delay, preferred, episodes=EPISODES):
    """Retain every qualified executed action and its actual body outcome."""
    records, outcomes, work = [], [], Counter()
    for episode in range(episodes):
        chosen, actions = 0, []
        for stage in range(delay + 1):
            inputs = context(stage, chosen, delay)
            result = agent.act(inputs, explore=True)
            work.update(result["work"])
            if not result["accepted"]:
                return records, outcomes, dict(work), "collection_refused"
            if stage == 0:
                chosen = result["action"]
            actions.append(result["action"])
            terminal = stage == delay
            reward = (1 if chosen == preferred else -1) if terminal else 0
            following = None if terminal else context(stage + 1, chosen, delay)
            agent.feedback(
                reward,
                following,
                decision_id=result["decision_id"],
                executed_action=result["action"],
                terminal=terminal,
                learn=False,
            )
            records.append(
                dict(
                    episode=episode,
                    stage=stage,
                    first_action=chosen,
                    context=inputs,
                    action=result["action"],
                    reward=reward,
                    following=following,
                )
            )
        outcomes.append(dict(episode=episode, actions=actions, reward=reward))
    return records, outcomes, dict(work), "complete"


def coverage(records, delay):
    counts = Counter((row["stage"], row["first_action"], row["action"]) for row in records)
    expected = [(0, action, action) for action in range(2)] + [
        (stage, first, action)
        for stage in range(1, delay + 1)
        for first in range(2)
        for action in range(2)
    ]
    rows = [dict(stage=s, first_action=f, action=a, count=counts[s, f, a]) for s, f, a in expected]
    return dict(
        rows=rows,
        missing=[row for row in rows if row["count"] == 0],
        root_adequate=all(counts[0, action, action] > 0 for action in range(2)),
        continuation_complete=all(counts[key] > 0 for key in expected),
    )


def schedule(records, seed, updates=UPDATES):
    """Two root-action strata, 13 other rows, and the latest terminal row."""
    roots = [
        [i for i, row in enumerate(records) if row["stage"] == 0 and row["action"] == action]
        for action in range(2)
    ]
    if not all(roots):
        raise ValueError("Both executed root actions require observed records")
    if records[-1]["following"] is not None:
        raise ValueError("Collection must finish at a real terminal transition")
    other = [i for i, row in enumerate(records[:-1]) if row["stage"] != 0]
    if len(other) < 13:
        raise ValueError("At least 13 non-root records are required")
    rng = random.Random(seed)
    return [
        [
            rng.choice(roots[0]),
            rng.choice(roots[1]),
            *rng.sample(other, 13),
            len(records) - 1,
        ]
        for _ in range(updates)
    ]


def observed_return(records, index, discount, value_scale, reward_scale):
    """Return only recorded rewards up to this same episode's actual end."""
    rewards = []
    while True:
        row = records[index]
        rewards.append((1 - discount) * value_scale * row["reward"] / reward_scale)
        if row["following"] is None:
            break
        if index + 1 == len(records):
            raise ValueError("An observed return requires a terminal outcome")
        next_row = records[index + 1]
        if next_row["episode"] != row["episode"] or next_row["context"] != row["following"]:
            raise ValueError("An observed return cannot cross a discontinuity")
        index += 1
    value = 0.0
    for reward in reversed(rewards):
        value = reward + discount * value
    return value, len(rewards), "terminal"


def fit(agent, records, indices, arm):
    """Same patch repair in every arm; only the declared estimates differ."""
    examples, targets, horizons, stops, work = [], [], [], [], Counter()
    for index in indices:
        row = records[index]
        if arm == "observed_return":
            target, horizon, stop = observed_return(
                records,
                index,
                agent.config["discount"],
                agent.config["value_scale"],
                agent.config["reward_scale"],
            )
        else:
            target, horizon, stop = agent._target(index, None, work)
        if target is None:
            return dict(accepted=False, reason="bootstrap_refused", work=dict(work))
        examples.append((row["context"], {f"q{row['action']}": target}))
        targets.append(target)
        horizons.append(horizon)
        stops.append(stop)
    result = agent.brain.observe_batch(examples, source="estimate")
    work.update(result["work"])
    return dict(
        accepted=result["accepted"],
        reason=result["reason"],
        source=result["source"],
        indices=indices,
        targets=targets,
        horizons=horizons,
        stops=stops,
        work=dict(work),
    )


def root_query(agent, delay, preferred):
    before = agent.brain.snapshot()
    result = agent.brain.settle(context(0, 0, delay))
    values = [result["outputs"][f"q{action}"][0] for action in range(2)]
    best = max(values)
    return dict(
        qualified=result["qualified"],
        query_pure=agent.brain.snapshot() == before,
        values=values,
        preferred_margin=values[preferred] - values[1 - preferred],
        greedy_actions=[action for action in range(2) if values[action] == best],
        work=result["work"],
    )


def evaluate(agent, delay, preferred):
    """Execute one complete free episode on a saved clone without learning."""
    owner = Reinforcement.from_snapshot(agent.snapshot())
    actions, work, chosen = [], Counter(), 0
    for stage in range(delay + 1):
        result = owner.act(context(stage, chosen, delay), explore=False)
        work.update(result["work"])
        if not result["accepted"]:
            return dict(status="evaluation_refused", actions=actions, work=dict(work))
        actions.append(result["action"])
        if stage == 0:
            chosen = result["action"]
        terminal = stage == delay
        reward = (1 if chosen == preferred else -1) if terminal else 0
        owner.feedback(
            reward,
            None if terminal else context(stage + 1, chosen, delay),
            decision_id=result["decision_id"],
            executed_action=result["action"],
            terminal=terminal,
            learn=False,
        )
    return dict(status="complete", actions=actions, reward=reward, work=dict(work))


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(__file__).with_name("temporal_credit.py"),
        *sorted((root / "src/cadence/experimental/equilibrium").glob("*.py")),
    ]
    sources = {path.relative_to(root).as_posix(): digest(path) for path in paths}
    protocol = dict(
        schema="credit-diagnostic/1",
        seed=args.seed,
        delay=args.delay,
        preferred=args.preferred,
        campaign_seeds=[2, 7, 11, 19, 29],
        campaign_delays=[8, 32, 128],
        campaign_preferred=[0, 1],
        episodes=EPISODES,
        updates=UPDATES,
        batch_size=16,
        arm_replay_seconds=ARM_SECONDS,
        checkpoints=CHECKPOINTS,
        arms=ARMS,
        exploration="1.0 at every decision; collection parameters frozen",
        replay="one observed root per action + 13 nonroots + latest terminal",
        discount="delay/(delay+1); same normalized reward scale in every arm",
        representation="explicit one-hot stage and remembered first action",
        scope="Diagnostic only; episodic returns do not estimate general optimal Q",
        cap="Independent replay wall cap per arm, checked between whole admissions",
        sources=sources,
    )
    write(args.out / "protocol.json", protocol)
    founder = make(args.seed, args.delay)
    records, outcomes, collection_work, status = collect(founder, args.delay, args.preferred)
    covered = coverage(records, args.delay)
    report = dict(
        protocol=protocol,
        records=records,
        outcomes=outcomes,
        coverage=covered,
        collection_work=collection_work,
        arms=[],
    )
    write(args.out / "collection.json", report)
    report["collection_sha256"] = digest(args.out / "collection.json")
    if status != "complete" or not covered["root_adequate"]:
        report["status"] = status if status != "complete" else "inadequate_root_coverage"
        write(args.out / "report.json", report)
        return report
    batches = schedule(records, args.seed + 1000003, updates=UPDATES)
    write(args.out / "batches.json", batches)
    report["batches_sha256"] = digest(args.out / "batches.json")
    original = json.loads(founder.snapshot())
    for name in ARMS:
        saved = {
            **original,
            "config": {
                **original["config"],
                "credit_horizon": 1 if name == "one_step" else args.delay + 1,
            },
        }
        agent = Reinforcement.from_snapshot(json.dumps(saved))
        queries = [dict(update=0, **root_query(agent, args.delay, args.preferred))]
        ledger, work, started = [], Counter(), time.monotonic()
        for update, indices in enumerate(batches, 1):
            if time.monotonic() - started >= ARM_SECONDS:
                break
            result = fit(agent, records, indices, name)
            work.update(result["work"])
            ledger.append(dict(update=update, **result))
            if not result["accepted"]:
                break
            if update in CHECKPOINTS:
                queries.append(dict(update=update, **root_query(agent, args.delay, args.preferred)))
        elapsed = time.monotonic() - started
        checkpoint = args.out / f"{name}-brain.json"
        checkpoint.write_text(agent.brain.snapshot())
        arm = dict(
            name=name,
            updates=ledger,
            queries=queries,
            work=dict(work),
            replay_seconds=elapsed,
            brain_sha256=digest(checkpoint),
            evaluation=evaluate(agent, args.delay, args.preferred),
            status="complete"
            if len(ledger) == UPDATES and all(row["accepted"] for row in ledger)
            else "incomplete",
        )
        total_work = Counter(work)
        total_work.update(arm["evaluation"]["work"])
        for query in queries:
            total_work.update(query["work"])
        arm["total_work"] = dict(total_work)
        arm["credit_stops"] = dict(
            Counter(stop for row in ledger if row["accepted"] for stop in row["stops"])
        )
        arm["root_terminal_targets"] = sum(
            stop == "terminal" for row in ledger if row["accepted"] for stop in row["stops"][:2]
        )
        report["arms"].append(arm)
        write(args.out / "report.json", report)
        print(
            json.dumps(
                dict(
                    arm=name,
                    status=arm["status"],
                    seconds=elapsed,
                    reward=arm["evaluation"].get("reward"),
                )
            ),
            flush=True,
        )
    report["matched_rows"] = all(
        [row.get("indices") for row in arm["updates"]] == batches for arm in report["arms"]
    )
    report["sources_unchanged"] = all(digest(root / p) == h for p, h in sources.items())
    total_work = Counter(collection_work)
    for arm in report["arms"]:
        total_work.update(arm["total_work"])
    report["total_work"] = dict(total_work)
    report["status"] = (
        "complete"
        if report["matched_rows"]
        and all(
            arm["status"] == "complete"
            and arm["evaluation"]["status"] == "complete"
            and all(q["qualified"] and q["query_pure"] for q in arm["queries"])
            for arm in report["arms"]
        )
        and report["sources_unchanged"]
        else "incomplete"
    )
    write(args.out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(2, 7, 11, 19, 29), required=True)
    parser.add_argument("--delay", type=int, choices=(8, 32, 128), required=True)
    parser.add_argument("--preferred", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    result = run(options)
    raise SystemExit(0 if result["status"] == "complete" else 1)
