"""Independently verify a campaign and publish its source-bound scalar evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(folder):
    campaign = read(folder / "campaign.json")
    outcomes = read(folder / "outcomes.json")
    protocol = campaign["protocol"]
    checks, records, sources = [], [], {}

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    check(
        "ordered job inventory complete",
        [[job[k] for k in ("kind", "seed", "backend", "revision")] for job in outcomes]
        == campaign["jobs"],
    )
    check("unique job directories", len({job["directory"] for job in outcomes}) == len(outcomes))
    check(
        "harness remained frozen",
        read(folder / "completion.json")["harness_and_revisions_unchanged"],
    )
    for job in outcomes:
        directory = folder / job["directory"]
        manifest, result = read(directory / "manifest.json"), read(directory / "result.json")
        identity = job["directory"]
        check(identity + " exit", job["exit_code"] == 0)
        check(identity + " completed", result["passed"] and result["source_unchanged"])
        check(identity + " commit", manifest["commit"] == campaign["revisions"][job["revision"]])
        check(identity + " backend identity", manifest["backend"] == job["backend"])
        check(
            identity + " protocol", manifest["protocol_sha256_lf"] == campaign["protocol_sha256_lf"]
        )
        script = "run.py" if job["kind"] == "nursery" else job["kind"] + ".py"
        check(
            identity + " harness",
            manifest["harness_sha256_lf"] == campaign["harness_sha256_lf"][script],
        )
        if job["revision"] not in sources:
            sources[job["revision"]] = manifest["source_sha256_lf"]
        check(identity + " sources", sources[job["revision"]] == manifest["source_sha256_lf"])
        record = {k: v for k, v in job.items() if k != "command"}
        record["result"] = result
        if job["kind"] == "nursery":
            check(identity + " clean", not manifest["dirty"])
            check(identity + " seed", manifest["seed"] == job["seed"])
            check(identity + " declared protocol", manifest["protocol"] == protocol)
            execution = result["execution"]
            cuda = job["backend"].startswith("cuda")
            dtype = "float32" if job["backend"] == "cuda32" else "float64"
            check(
                identity + " actual execution",
                execution
                == {
                    "backend": "torch" if cuda else "cpu",
                    "device": "cuda:0" if cuda else "cpu",
                    "state_dtype": "torch." + dtype if cuda else dtype,
                    "parameter_dtype": "torch.float64" if cuda else "float64",
                },
            )
            with np.load(directory / "acquired.npz", allow_pickle=False) as saved:
                meta = json.loads(str(saved["meta"]))
                generic = json.loads(str(saved["generic"]))
            for name, effective in (("learning", meta["config"]), ("reward", generic["reward"])):
                check(identity + " saved " + name, result["graph"][name] == effective)
                check(
                    identity + " requested " + name,
                    all(effective.get(k) == value for k, value in protocol[name].items()),
                )
            check(
                identity + " acquisition",
                result["bootstrap_accuracy"] >= protocol["gate"]["bootstrap_held_out_accuracy"]
                and result["bootstrap_accuracy"] > result["founder_accuracy"],
            )
            check(
                identity + " retention",
                result["retained_accuracy"] >= protocol["gate"]["retained_held_out_accuracy"],
            )
            check(identity + " saved continuation", result["saved_actions_equal"])
            check(
                identity + " live inventory",
                [(r["stage"], r["tick"]) for r in result["live"]]
                == [(s["name"], tick) for s in protocol["stages"] for tick in range(s["calls"])],
            )
            check(
                identity + " free admission",
                all(
                    r["settlement"]["qualified"]
                    and r["settlement"]["row_qualified"] == [True] * protocol["live_batch"]
                    and len(r["settlement"]["residual"]) == protocol["live_batch"]
                    and all(
                        np.isfinite(value) and 0 <= value <= protocol["learning"]["tolerance"]
                        for value in r["settlement"]["residual"]
                    )
                    and r["settlement"]["max_residual"] == max(r["settlement"]["residual"])
                    and r["settlement"]["tolerance"] == protocol["learning"]["tolerance"]
                    and r["settlement"]["budget"] == protocol["learning"]["free_steps"]
                    for r in result["live"]
                ),
            )
            check(
                identity + " actual live scores",
                all(
                    len(r["actions"]) == len(r["labels"]) == protocol["live_batch"]
                    and all(0 <= a < protocol["actions"] for a in r["actions"] + r["labels"])
                    and r["correct"]
                    == sum(a == b for a, b in zip(r["actions"], r["labels"], strict=True))
                    for r in result["live"]
                ),
            )
            check(
                identity + " no failed calls",
                all(r["failure"] is None for r in result["measurements"]),
            )
            check(
                identity + " latency stages",
                set(result["latencies"]) == {s["name"] for s in protocol["stages"]},
            )
            for stage in protocol["stages"]:
                name = stage["name"]
                group = result["latencies"].get(name)
                rows = [r for r in result["measurements"] if r["stage"] == name]
                durations = [r["seconds"] for r in rows]
                check(
                    identity + " " + name + " complete timed calls",
                    len(rows) == stage["calls"]
                    and all(np.isfinite(d) and d > 0 for d in durations)
                    and all(r["decisions"] == protocol["live_batch"] for r in rows),
                )
                if group is None or not durations:
                    continue
                check(
                    identity + " " + name + " totals and deadline misses",
                    group["calls"] == stage["calls"]
                    and np.isclose(group["seconds"], sum(durations), rtol=1e-14, atol=0)
                    and group["max_seconds"] == max(durations)
                    and group["deadline_misses"]
                    == sum(d > protocol["latency"]["deadline_seconds"] for d in durations),
                )
                for percentile in (50, 95, 99):
                    check(
                        identity + f" {name} p{percentile}",
                        np.isclose(
                            group[f"p{percentile}_seconds"],
                            np.percentile(durations, percentile),
                            rtol=1e-14,
                            atol=0,
                        ),
                    )
                check(
                    identity + " " + name + " throughput",
                    np.isclose(
                        group["decisions_per_second"],
                        sum(r["decisions"] for r in rows) / sum(durations),
                    ),
                )
        elif job["kind"] == "transport":
            check(
                identity + " cases",
                [(c["modules"], c["batch"]) for c in result["cases"]]
                == [
                    (layout, batch)
                    for layout in protocol["transport_controls"]["layouts"]
                    for batch in protocol["transport_controls"]["batches"]
                ],
            )
            for case in result["cases"]:
                check(
                    identity + " states hash " + case["states_file"],
                    case["states_sha256"]
                    == hashlib.sha256((directory / case["states_file"]).read_bytes()).hexdigest(),
                )
        else:
            record["python_profile"] = (directory / "python-profile.txt").read_text(
                encoding="utf-8"
            )
            record["operators_table"] = (directory / "operators.txt").read_text(encoding="utf-8")
        records.append(record)

    lives = [r for r in records if r["kind"] == "nursery"]
    for right in lives:
        if right["revision"] != "candidate":
            continue
        matching = [
            r
            for r in lives
            if r["revision"] == "baseline"
            and (r["backend"], r["seed"]) == (right["backend"], right["seed"])
        ]
        check(right["directory"] + " baseline pairing", len(matching) == 1)
        if len(matching) != 1:
            continue
        before, after = matching[0]["result"], right["result"]
        check(right["directory"] + " matched counters", before["counts"] == after["counts"])
        check(
            right["directory"] + " matched live behavior",
            [
                [r[k] for k in ("stage", "tick", "actions", "labels", "correct")]
                for r in before["live"]
            ]
            == [
                [r[k] for k in ("stage", "tick", "actions", "labels", "correct")]
                for r in after["live"]
            ],
        )
        check(
            right["directory"] + " matched stage work",
            [(r["stage"], r["work"]) for r in before["measurements"]]
            == [(r["stage"], r["work"]) for r in after["measurements"]],
        )

    comparisons = []
    controls = [r for r in records if r["kind"] == "transport"]
    for right in controls:
        if right["revision"] != "candidate":
            continue
        for left in controls:
            pair = left["backend"] == right["backend"] and left["revision"] == "baseline"
            reference = (
                left["backend"] == "cpu"
                and left["revision"] == "candidate"
                and right["backend"] != "cpu"
            )
            if not (pair or reference):
                continue
            tolerance = protocol["transport_controls"][
                "float32_atol_rtol" if right["backend"] == "cuda32" else "float64_atol_rtol"
            ]
            for a, b in zip(left["result"]["cases"], right["result"]["cases"], strict=True):
                with np.load(folder / left["directory"] / a["states_file"]) as archive:
                    expected = archive["potentials"]
                with np.load(folder / right["directory"] / b["states_file"]) as archive:
                    actual = archive["potentials"]
                passed = np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
                label = left["directory"] + " -> " + right["directory"] + " " + a["states_file"]
                check(label, passed)
                comparisons.append(
                    {
                        "left": left["directory"],
                        "right": right["directory"],
                        "case": a["states_file"],
                        "max_abs_deviation": float(np.max(np.abs(actual - expected))),
                        "atol_rtol": tolerance,
                        "passed": bool(passed),
                    }
                )
    return {
        "passed": all(c["passed"] for c in checks),
        "campaign": campaign,
        "source_sha256_lf": sources,
        "records": records,
        "transport_comparisons": comparisons,
        "checks": checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.campaign)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "checks": len(receipt["checks"]),
                "failures": [c for c in receipt["checks"] if not c["passed"]],
            }
        )
    )
    raise SystemExit(0 if receipt["passed"] else 1)
