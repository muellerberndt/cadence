"""Matched-exposure replay-order control for ordinary one-step Q estimates.

Reconstruct an existing diagnostic collection without adding experience. The
two arms share every sampled row multiplicity at 32-update boundaries; only
ordering and consequent batch membership differ. Observed returns are used for
diagnostics only, never for training. Every parameter admission uses Cadence's
ordinary patch repair with source='estimate'.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import credit_diagnostic as diagnostic  # noqa: E402

UPDATES = 512
BLOCK = 32
BATCH = 16
ARM_SECONDS = 360
CASE_SECONDS = 900
CHECKPOINTS = (0, 32, 128, 512)


def reverse_blocks(batches, records, block=BLOCK):
    """Preserve exact row multiplicities at each complete block boundary."""
    if len(batches) % block or not all(len(rows) == BATCH for rows in batches):
        raise ValueError("Replay requires complete equal-size blocks")
    result = []
    for offset in range(0, len(batches), block):
        rows = [index for batch in batches[offset : offset + block] for index in batch]
        # Stable ties retain their already seeded random order.
        rows.sort(key=lambda index: records[index]["stage"], reverse=True)
        result.extend(rows[i : i + BATCH] for i in range(0, len(rows), BATCH))
    return result


def matched_prefixes(left, right, checkpoints):
    return {
        str(point): Counter(i for row in left[:point] for i in row)
        == Counter(i for row in right[:point] for i in row)
        and len(left) >= point
        and len(right) >= point
        for point in checkpoints
    }


def reconstruct(collection):
    """Verify each original executed action and reward using the frozen body."""
    config = collection["protocol"]
    owner = diagnostic.make(config["seed"], config["delay"])
    records, outcomes, work, status = diagnostic.collect(
        owner, config["delay"], config["preferred"], episodes=config["episodes"]
    )
    if status != "complete" or records != collection["records"]:
        raise ValueError("Reconstructed observed transition identities differ")
    if outcomes != collection["outcomes"]:
        raise ValueError("Reconstructed executed actions or outcomes differ")
    return owner, dict(work)


def calibration(agent, records):
    """Compare free values with actual episodic returns without teaching them."""
    work, rows, seen = Counter(), [], set()
    before = agent.brain.snapshot()
    for index, record in enumerate(records):
        key = record["stage"], record["first_action"], record["action"]
        if key in seen:
            continue
        seen.add(key)
        result = agent.brain.settle(record["context"])
        work.update(result["work"])
        target, _, _ = diagnostic.observed_return(
            records,
            index,
            agent.config["discount"],
            agent.config["value_scale"],
            agent.config["reward_scale"],
        )
        value = result["outputs"][f"q{record['action']}"][0]
        rows.append(
            dict(
                stage=record["stage"],
                first_action=record["first_action"],
                action=record["action"],
                value=value,
                observed_return=target,
                absolute_error=abs(value - target),
                qualified=result["qualified"],
            )
        )
    return dict(
        rows=rows,
        work=dict(work),
        query_pure=agent.brain.snapshot() == before,
        qualified=all(row["qualified"] for row in rows),
    )


def checkpoint(agent, records, config, update):
    return dict(
        update=update,
        root=diagnostic.root_query(agent, config["delay"], config["preferred"]),
        calibration=calibration(agent, records),
        evaluation=diagnostic.evaluate(agent, config["delay"], config["preferred"]),
    )


def run(args):
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(diagnostic.__file__),
        Path(__file__).with_name("temporal_credit.py"),
        *sorted((root / "src/cadence/experimental/equilibrium").glob("*.py")),
    ]
    sources = {path.relative_to(root).as_posix(): diagnostic.digest(path) for path in paths}
    collection = json.loads(args.collection.read_text())
    config = collection["protocol"]
    protocol = dict(
        schema="credit-exposure/1",
        seed=config["seed"],
        delay=config["delay"],
        preferred=config["preferred"],
        collection_sha256=diagnostic.digest(args.collection),
        source_collection_protocol=config,
        sources=sources,
        updates=UPDATES,
        batch_size=BATCH,
        block=BLOCK,
        checkpoints=CHECKPOINTS,
        arm_seconds=ARM_SECONDS,
        case_seconds=CASE_SECONDS,
        arms=["stratified", "reverse_stage"],
        credit_horizon=1,
        matching="Exact per-record multiplicities at every32-update block boundary",
        scope="Replay-order diagnostic; no new learning law or episode-return targets",
        limits="Independent arm caps; case cap checked between calls; launcher must enforce900s",
    )
    diagnostic.write(args.out / "protocol.json", protocol)
    report = dict(protocol=protocol, status="starting", arms=[])
    try:
        expected = config["sources"]
        report["source_compatibility"] = {
            name: sources.get(name) == digest for name, digest in expected.items()
        }
        if not all(report["source_compatibility"].values()):
            raise ValueError("Frozen collection source identity is incompatible")
        founder, reconstruction_work = reconstruct(collection)
        report["reconstruction_work"] = reconstruction_work
        report["collection_verified"] = True
        records = collection["records"]
        forward = diagnostic.schedule(records, config["seed"] + 1000003, updates=UPDATES)
        schedules = {
            "stratified": forward,
            "reverse_stage": reverse_blocks(forward, records, block=BLOCK),
        }
        schedule_checks = matched_prefixes(
            forward, schedules["reverse_stage"], range(BLOCK, UPDATES + 1, BLOCK)
        )
        if not all(schedule_checks.values()):
            raise ValueError("Replay row multiplicities differ")
        diagnostic.write(args.out / "schedules.json", schedules)
        report["schedules_sha256"] = diagnostic.digest(args.out / "schedules.json")
        report["planned_matched_prefixes"] = schedule_checks
        original = json.loads(founder.snapshot())
        for name, batches in schedules.items():
            saved = {**original, "config": {**original["config"], "credit_horizon": 1}}
            agent = diagnostic.Reinforcement.from_snapshot(json.dumps(saved))
            arm = dict(name=name, status="running", updates=[], checkpoints=[])
            report["arms"].append(arm)
            arm_started, work = time.monotonic(), Counter()
            arm["checkpoints"].append(checkpoint(agent, records, config, 0))
            for update, indices in enumerate(batches, 1):
                if time.monotonic() - started >= CASE_SECONDS:
                    arm["status"] = "case_time_limit"
                    break
                if time.monotonic() - arm_started >= ARM_SECONDS:
                    arm["status"] = "arm_time_limit"
                    break
                result = diagnostic.fit(agent, records, indices, "one_step")
                work.update(result["work"])
                arm["updates"].append(dict(update=update, **result))
                if not result["accepted"]:
                    arm["status"] = "admission_refused"
                    break
                if update in CHECKPOINTS:
                    arm["checkpoints"].append(checkpoint(agent, records, config, update))
                    (args.out / f"{name}-{update}-brain.json").write_text(agent.brain.snapshot())
                if update % BLOCK == 0:
                    diagnostic.write(args.out / "report.json", report)
                    print(
                        json.dumps(
                            dict(
                                arm=name,
                                updates=update,
                                seconds=time.monotonic() - arm_started,
                            )
                        ),
                        flush=True,
                    )
            if len(arm["updates"]) == UPDATES and all(r["accepted"] for r in arm["updates"]):
                arm["status"] = "complete"
            arm["seconds"] = time.monotonic() - arm_started
            arm["training_work"] = dict(work)
            for saved_check in arm["checkpoints"]:
                for key in ("root", "calibration", "evaluation"):
                    work.update(saved_check[key]["work"])
            arm["total_work"] = dict(work)
            final = args.out / f"{name}-final-brain.json"
            final.write_text(agent.brain.snapshot())
            arm["final_brain_sha256"] = diagnostic.digest(final)
            diagnostic.write(args.out / "report.json", report)
        report["actual_matched_prefixes"] = matched_prefixes(
            [r["indices"] for r in report["arms"][0]["updates"] if r["accepted"]],
            [r["indices"] for r in report["arms"][1]["updates"] if r["accepted"]],
            CHECKPOINTS[1:],
        )
        checks_valid = all(
            check["root"]["qualified"]
            and check["root"]["query_pure"]
            and check["calibration"]["qualified"]
            and check["calibration"]["query_pure"]
            and check["evaluation"]["status"] == "complete"
            for arm in report["arms"]
            for check in arm["checkpoints"]
        )
        report["status"] = (
            "complete"
            if checks_valid
            and all(a["status"] == "complete" for a in report["arms"])
            and all(report["actual_matched_prefixes"].values())
            else "incomplete"
        )
        total_work = Counter(reconstruction_work)
        for arm in report["arms"]:
            total_work.update(arm["total_work"])
        report["total_work"] = dict(total_work)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        report["sources_unchanged"] = all(
            diagnostic.digest(root / p) == h for p, h in sources.items()
        )
        if not report["sources_unchanged"]:
            report["status"] = "source_changed"
        report["seconds"] = time.monotonic() - started
        diagnostic.write(args.out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    report = run(parser.parse_args())
    raise SystemExit(0 if report["status"] == "complete" else 1)
