"""Read the entire fixed confirmation cohort; missing or invalid attempts fail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path) -> dict:
    directory = Path(__file__).parent
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    admission_path = directory / "sources.json"
    admission = json.loads(admission_path.read_text())
    gate = protocol["acceptance"]
    rows, errors = [], []
    source = None
    for case in protocol["cases"]:
        folder = root / case["id"]
        try:
            receipt = json.loads((folder / "receipt.json").read_text())
            assert receipt["case"] == case and not receipt["smoke"], "wrong case"
            assert receipt["status"] == "complete", "attempt incomplete or failed"
            assert receipt["protocol"] == protocol, "wrong protocol"
            assert receipt["protocol_sha256"] == digest(protocol_path), "protocol hash differs"
            assert receipt["producer_sha256"] == digest(directory / "confirm.py"), (
                "producer differs"
            )
            assert receipt["source_unchanged_after"], "runtime source changed"
            assert receipt["producer_unchanged_after"], "producer/protocol changed"
            assert receipt["source_admitted"], "wrong admitted source"
            assert receipt["admission_sha256"] == digest(admission_path), "admission changed"
            assert receipt["source_sha256"] == admission["source_sha256"], "unadmitted source"
            if source is None:
                source = receipt["source_sha256"]
            assert receipt["source_sha256"] == source, "cohort runtime sources differ"
            required = {"trained.npz", "world.json", "fork-original.npz", "fork-loaded.npz"}
            assert required <= set(receipt["artifacts"]), "missing required custody artifacts"
            for name, item in receipt["artifacts"].items():
                assert digest(folder / name) == item["sha256"], f"artifact changed: {name}"
                assert (folder / name).stat().st_size == item["bytes"], f"artifact size: {name}"
            training, custody = receipt["training"]["total"], receipt["custody"]
            assert training["moments"] == protocol["training_moments"], "wrong training horizon"
            assert training["refused"] == 0, "training refusal"
            assert training["refusal_attempts"] == 0, "hidden retried training refusal"
            assert receipt["training"]["brain"]["refusals"] == 0, "wrapper refusals"
            assert receipt["training"]["blocks"][-1]["cumulative"] == training, "partial totals"
            assert custody["outcomes_received"] == protocol["training_moments"] - 1, "outcomes"
            assert custody["pending_action"] and custody["pending_done"] is False, "pending action"
            assert custody["wrapper_owed"] == 0, "wrapper refusal debt"
            world = json.loads((folder / "world.json").read_text())
            assert world["world"]["reward"] == custody["pending_actual_reward"], "owed reward"
            assert world["wrapper"]["owed"] == custody["wrapper_owed"], "wrapper sidecar"
            continuation = receipt["continuation"]
            assert continuation["outcomes_received_after"] == (
                protocol["training_moments"] - 1 + protocol["continuation_moments"]
            ), "resumed outcome count"
            assert continuation["equal"] and continuation["checkpoint_arrays_equal"], "continuation"
            assert continuation["moments_per_fork"] == protocol["continuation_moments"], (
                "fork length"
            )
            assert len(continuation["records"]) == protocol["continuation_moments"], "fork records"
            for pair in continuation["records"]:
                left = {k: v for k, v in pair["unbroken"].items() if k != "policy_ms"}
                right = {k: v for k, v in pair["reloaded"].items() if k != "policy_ms"}
                assert left == right and not left["refusal_attempts"], "fork action/work mismatch"
            import numpy as np

            with (
                np.load(folder / "fork-original.npz", allow_pickle=False) as left,
                np.load(folder / "fork-loaded.npz", allow_pickle=False) as right,
            ):
                assert set(left.files) == set(right.files), "fork checkpoint inventory"
                assert all(np.array_equal(left[k], right[k]) for k in left.files), "fork state"
            assert all(row["refused"] == 0 for row in receipt["fingerprint"]["rows"]), (
                "probe refusal"
            )
            assert all(not row["refusal_attempts"] for row in receipt["fingerprint"]["rows"]), (
                "hidden probe refusal"
            )
            evaluations = receipt["evaluations"]
            assert [row["world"] for row in evaluations] == protocol["evaluation_worlds"], "worlds"
            means = {}
            for name in ("trained", "newborn", "random", "ablated"):
                trials = [row[name]["total"] for row in evaluations]
                assert all(t["moments"] == protocol["evaluation_moments"] for t in trials), (
                    "horizon"
                )
                assert all(t["refused"] == 0 and t["learning_sweeps"] == 0 for t in trials), (
                    "frozen"
                )
                assert all(not t["refusal_attempts"] for t in trials), "hidden evaluation refusal"
                means[name] = {
                    metric: sum(t[metric] for t in trials) / len(trials)
                    for metric in ("progress_m", "dealt", "taken", "kills")
                }
            progress_gain = means["trained"]["progress_m"] - max(
                means["newborn"]["progress_m"],
                means["random"]["progress_m"],
            )
            damage_gain = means["trained"]["dealt"] - max(
                means["newborn"]["dealt"],
                means["random"]["dealt"],
            )
            damage_total = means["trained"]["dealt"] * len(evaluations)
            rows.append(
                {
                    **case,
                    "means": means,
                    "progress_gain": progress_gain,
                    "damage_gain": damage_gain,
                    "progress_success": progress_gain > gate["epsilon"],
                    "damage_success": (
                        damage_gain > gate["epsilon"]
                        and damage_total >= gate["minimum_total_damage"]
                    ),
                    "lesion_loss": means["trained"]["progress_m"] - means["ablated"]["progress_m"],
                    "total_seconds": receipt["total_seconds"],
                }
            )
        except (OSError, ValueError, KeyError, AssertionError) as error:
            errors.append({"case": case["id"], "error": str(error)})
    body_counts = {
        body: {
            metric: sum(row[metric] for row in rows if row["body"] == body)
            for metric in ("progress_success", "damage_success")
        }
        for body in sorted({case["body"] for case in protocol["cases"]})
    }
    denominator = sum(row["progress_gain"] for row in rows)
    numerator = sum(row["lesion_loss"] for row in rows)
    fraction = numerator / denominator if denominator > gate["epsilon"] else None
    checks = {
        "complete_valid_census": not errors and len(rows) == len(protocol["cases"]),
        "progress_lives": sum(row["progress_success"] for row in rows)
        >= gate["positive_progress_lives"],
        "damage_lives": sum(row["damage_success"] for row in rows) >= gate["positive_damage_lives"],
        "progress_each_body": all(
            counts["progress_success"] >= gate["minimum_progress_successes_each_body"]
            for counts in body_counts.values()
        ),
        "damage_each_body": all(
            counts["damage_success"] >= gate["minimum_damage_successes_each_body"]
            for counts in body_counts.values()
        ),
        "sensory_dependence": fraction is not None
        and fraction >= gate["minimum_sensory_gain_fraction"],
    }
    return {
        "protocol_sha256": digest(protocol_path),
        "pass": all(checks.values()),
        "checks": checks,
        "rows": rows,
        "errors": errors,
        "body_counts": body_counts,
        "lesion_gain_fraction": fraction,
        "lesion_numerator": numerator,
        "gain_denominator": denominator,
        "scope": "Fixed practical engineering gate; not a population success-probability bound",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["pass"] else 1)
