"""Saved-call custody and independent state arithmetic, without new model calls."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import body
import head_path as H
import probe as P
from verify_lesion import arithmetic


def expected(tape, schedule):
    def checks(rows, phase, corrected):
        for row in rows:
            yield (
                "settle",
                body.inputs(row),
                body.facts(row) if corrected else None,
                {"phase": phase, "id": row["id"], "corrected": corrected},
                row,
            )

    for update in range(129):
        if update in (0, 32, 64, 128):
            for corrected in (False, True):
                yield from checks(tape["clean"], f"check-{update}", corrected)
        if update < 128:
            rows = [tape["train"][i] for i in schedule[update]]
            yield (
                "observe_batch",
                [(body.inputs(r), body.facts(r, reveal=True)) for r in rows],
                None,
                {
                    "phase": "training",
                    "update": update + 1,
                    "ids": [r["id"] for r in rows],
                },
                None,
            )
    for row in tape["clean"][:8]:
        for teacher in (False, True):
            yield (
                "settle",
                body.inputs(row),
                body.facts(row, reveal=teacher),
                {
                    "phase": "teacher_clamp_diagnostic",
                    "id": row["id"],
                    "teacher": teacher,
                },
                row,
            )
    for policy in P.POLICIES:
        for i, row in enumerate(tape["live"]):
            yield (
                "settle",
                body.inputs(row),
                body.facts(row) if policy.startswith("corrected") else None,
                {"phase": "live", "policy": policy, "id": row["id"]},
                row,
            )
            if policy.endswith("fit") and i in (31, 47):
                rows = tape["live"][i - 15 : i + 1]
                yield (
                    "observe_batch",
                    [(body.inputs(r), body.facts(r, reveal=True)) for r in rows],
                    None,
                    {
                        "phase": "live_fit",
                        "policy": policy,
                        "ids": [r["id"] for r in rows],
                    },
                    None,
                )
        for dataset in ("clean", "shifted"):
            yield from checks(tape[dataset], policy + "-retained", False)


def qualified_state(brain, result, inputs, targets):
    assert len(result["state"]) == len(brain.biases)
    assert all(
        math.isfinite(x) and abs(x) <= brain.config["state_bound"]
        for x in result["state"]
    )
    flat = tuple(
        x
        for port in ("coarse", "command", "prior", "fine", "present")
        for x in inputs[port]
    )
    pred, error, gradient, energy = arithmetic(
        brain.graph,
        flat,
        result["state"],
        brain.weights,
        brain.biases,
        brain.config["state_prior"],
    )
    assert (
        pred == result["predictions"]
        and error == result["errors"]
        and energy == result["energy"]
    )
    info = brain.inspect()
    populations = {p["name"]: p["indices"] for p in info["populations"]}
    outputs = {
        o["name"]: populations[o["reads"]][o["indices"][0]] for o in info["outputs"]
    }
    clamps = {outputs[k]: v[0] for k, v in (targets or {}).items()}
    assert all(result["state"][i] == value for i, value in clamps.items())
    assert result["outputs"] == {
        name: [result["state"][i]] for name, i in outputs.items()
    }
    bound = brain.config["state_bound"]
    assert all(math.isfinite(x) and abs(x) <= bound for x in result["state"])
    norm = max(
        0 if (x <= -bound and g > 0) or (x >= bound and g < 0) else abs(g)
        for i, (x, g) in enumerate(zip(result["state"], gradient, strict=True))
        if i not in clamps
    )
    assert norm == result["stationarity"] and norm <= brain.config["tolerance"]
    assert result["parameter_sha256"] == P.digest([brain.weights, brain.biases])
    return norm


def inspect_case(root, job, tape, schedule):
    directory = root / job["name"]
    if not (directory / "report.json").exists():
        return {"job": job, "status": "missing", "complete": False}
    report = json.loads((directory / "report.json").read_text())
    assert report["job"] == job
    models = {}
    for path in directory.rglob("*.json"):
        if path.name in (
            "initial.json",
            "latest.json",
            "final.json",
        ) or path.name.startswith("checkpoint-"):
            snapshot = path.read_text()
            models[P.digest(snapshot)] = P.Brain.from_snapshot(snapshot)
    initial = (directory / "initial.json").read_text()
    assert initial == H.make(job["kind"], job["seed"]).snapshot()
    with gzip.open(directory / "calls.jsonl.gz", "rt") as stream:
        calls = [json.loads(line) for line in stream]
    plan = list(expected(tape, schedule))
    assert len(calls) <= len(plan) == 1428
    counts, work, groups = Counter(), Counter(), {}
    checked, unknown, unavailable = 0, 0, 0
    maximum = 0.0
    current = P.digest(initial)
    acquired_hash = None
    branch_fits = Counter()
    for ordinal, (saved, (method, argument, targets, meta, row)) in enumerate(
        zip(calls, plan, strict=False), 1
    ):
        assert saved["ordinal"] == ordinal and saved["method"] == method
        assert saved["meta"] == meta and saved["arguments_sha256"] == P.digest(
            [argument, targets]
        )
        if saved["status"] != "returned":
            assert ordinal == len(calls) and saved["unknown_solver_work"] is True
            unknown += 1
            continue
        if meta["phase"] == "live" and row["id"] == tape["live"][0]["id"]:
            current = acquired_hash
        assert saved["before"] == current
        if meta["phase"].startswith("check-"):
            update = int(meta["phase"].split("-")[1])
            assert current == P.digest(
                (directory / f"checkpoint-{update}.json").read_text()
            )
        if meta["phase"].endswith("-retained"):
            policy = meta["phase"].removesuffix("-retained")
            assert current == P.digest((directory / policy / "final.json").read_text())
        r = saved["result"]
        assert r["qualified"] is True
        counts[method] += 1
        work.update(r["work"])
        assert math.isfinite(saved["seconds"]) and saved["seconds"] >= 0
        if method == "observe_batch":
            assert r["accepted"] is True and r["source"] == "witness"
            assert r["batch_size"] == 16 and r["duplicate"] is False
            if meta["phase"] == "training":
                event = meta["update"] - 1
            else:
                event = 128 + branch_fits[meta["policy"]]
                branch_fits[meta["policy"]] += 1
            assert r["event_id"] == event
            for (_, labels), outputs in zip(argument, r["outputs"], strict=True):
                assert all(outputs[name] == value for name, value in labels.items())
            current = saved["after"]
            if meta["phase"] == "training" and meta["update"] == 128:
                acquired_hash = current
            continue  # Numeric learning trajectory explicitly outside this check.
        assert saved["before"] == saved["after"]
        if saved["before"] in models:
            maximum = max(
                maximum, qualified_state(models[saved["before"]], r, argument, targets)
            )
            checked += 1
        else:
            # Intermediate first-live-fit parameters were hashed, not checkpointed.
            assert meta["phase"] == "live" and meta["policy"].endswith("fit")
            assert row["id"] in {q["id"] for q in tape["live"][32:48]}
            unavailable += 1
        key = (meta["phase"], meta.get("policy"), meta.get("corrected"))
        groups.setdefault(key, []).append((row, r["outputs"]["f"][0], saved["before"]))
    inflight = json.loads((directory / "inflight.json").read_text())
    if calls:
        last = calls[-1]
        if inflight["ordinal"] == last["ordinal"]:
            for key in ("method", "meta", "before", "arguments_sha256"):
                assert inflight[key] == last[key]
            if last["status"] == "returned":
                assert inflight["status"] == "returned"
            else:
                assert inflight["status"] == "started"
        else:
            assert report["status"] != "complete" and report["unknown_calls"] > 0
            assert (
                inflight["status"] == "started"
                and inflight["ordinal"] == len(calls) + 1
            )
    assert dict(counts) == report["counts"] and dict(work) == report["work"]
    assert counts["observe_batch"] == report["admissions"]
    assert report["unknown_calls"] >= unknown
    complete = report["status"] == "complete"
    if complete:
        assert (
            len(calls) == 1428
            and checked == 1264
            and unavailable == 32
            and unknown == 0
        )
        assert report["unknown_calls"] == 0
        assert counts == {"observe_batch": 132, "settle": 1296}
        for update in (0, 32, 64, 128):
            for mode in ("routine", "corrected"):
                rows = groups[f"check-{update}", None, mode == "corrected"]
                values = [
                    {
                        "id": row["id"],
                        "prediction": prediction,
                        "actual": row["actual"][3],
                        "error": abs(prediction - row["actual"][3]),
                    }
                    for row, prediction, _ in rows
                ]
                assert report["checks"][str(update)][mode] == {
                    "mae": sum(v["error"] for v in values) / 64,
                    "rows": values,
                }
        acquired = all(
            report["checks"]["128"][mode]["mae"] <= 0.04
            and report["checks"]["128"][mode]["mae"]
            <= 0.5 * report["checks"]["0"][mode]["mae"]
            for mode in ("routine", "corrected")
        )
        assert report["acquired"] == acquired
        for policy in P.POLICIES:
            rows = groups["live", policy, None]
            detail = report["branches"][policy]
            outcomes = []
            for row, prediction, model in rows:
                outcomes.append(
                    {
                        "id": row["id"],
                        "inputs_sha256": P.digest(body.inputs(row)),
                        "issued_prediction": prediction,
                        "actual_f": row["actual"][3],
                        "body_record_sha256": P.digest(row),
                        "error": abs(prediction - row["actual"][3]),
                        "model_sha256": model,
                    }
                )
            assert (
                outcomes
                == detail["rows"]
                == json.loads((directory / policy / "outcomes.json").read_text())
            )
            assert detail["shift_mae"] == sum(r["error"] for r in outcomes[16:]) / 48
            assert detail["late_mae"] == sum(r["error"] for r in outcomes[48:]) / 16
            retained = groups[policy + "-retained", None, False]
            for offset, dataset in ((0, "clean"), (64, "shifted")):
                selected = retained[offset : offset + 64]
                values = [
                    {
                        "id": row["id"],
                        "prediction": prediction,
                        "actual": row["actual"][3],
                        "error": abs(prediction - row["actual"][3]),
                    }
                    for row, prediction, _ in selected
                ]
                assert detail["retained_" + dataset] == {
                    "mae": sum(r["error"] for r in values) / 64,
                    "rows": values,
                }
            assert detail["final_sha256"] == P.digest(
                (directory / policy / "final.json").read_text()
            )
        acquired_snapshot = (directory / "checkpoint-128.json").read_text()
        assert (
            acquired_snapshot
            == (directory / "routine/final.json").read_text()
            == (directory / "corrected/final.json").read_text()
        )
        assert (directory / "routine_fit/final.json").read_text() == (
            directory / "corrected_fit/final.json"
        ).read_text()
        b = report["branches"]
        retention = (
            b["routine_fit"]["retained_clean"]["mae"]
            <= b["routine"]["retained_clean"]["mae"] + 0.005
            and b["routine_fit"]["retained_shifted"]["mae"]
            <= 0.9 * b["routine"]["retained_shifted"]["mae"]
        )
        assert report["retention_gate"] == retention
    return {
        "job": job,
        "complete": complete,
        "status": report["status"],
        "counts": dict(counts),
        "admissions": report["admissions"],
        "reported_unknown_calls": report["unknown_calls"],
        "saved_states_requalified": checked,
        "states_without_retained_parameters": unavailable,
        "maximum_projected_gradient": maximum,
        "work": dict(work),
    }


def verify(root):
    p = H.bound(root)
    source_paths = (
        __file__,
        Path(__file__).with_name("test_verify_head_path.py"),
        Path(__file__).with_name("verify_lesion.py"),
    )
    source_before = {str(Path(name).resolve()): P.sha(name) for name in source_paths}
    before = {
        str(f): P.sha(f)
        for f in root.rglob("*")
        if f.is_file() and f.name != "saved-call-verification.json"
    }
    tape = json.loads((root / "data.json").read_text())
    assert tape == body.data()
    schedule = json.loads((root / "schedules.json").read_text())["3109"]
    assert schedule == P.schedule(3109)
    cases = [inspect_case(root, j, tape, schedule) for j in p["inventory"]]
    original_summary = json.loads((root / "summary.json").read_text())
    fresh = H.summary(root, json.loads((root / "execution.json").read_text()))
    for field in (
        "complete",
        "all_acquired",
        "cases",
        "comparison_pairs",
        "development_quality_gate",
        "confirmatory_claim",
        "integrated_three_capabilities",
    ):
        assert fresh[field] == original_summary[field]
    H.bound(root)
    assert all(P.sha(path) == value for path, value in before.items())
    sources = {
        str(Path(name).resolve()): P.sha(name)
        for name in (
            __file__,
            Path(__file__).with_name("test_verify_head_path.py"),
            Path(__file__).with_name("verify_lesion.py"),
        )
    }
    assert sources == source_before
    return {
        "valid": True,
        "complete": fresh["complete"],
        "all_acquired": fresh["all_acquired"],
        "development_quality_gate": fresh["development_quality_gate"],
        "cases": cases,
        "sources": sources,
        "artifact_pins": before,
        "new_numerical_calls": 0,
        "learning_trajectory_replayed": False,
        "scope": "Exact call/target/body/metric/work census plus independently recomputed saved-state prediction/error/energy/gradient qualification where numerical parameters are checkpointed. Intermediate first-live-fit parameters are hash-custody only (32 queries per complete arm); no learning trajectory, timing or solver-work reexecution.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    out = args.root / "saved-call-verification.json"
    if out.exists():
        raise ValueError("Preserve prior verification")
    P.write(out, verify(args.root.resolve()))
    print(P.sha(out))


if __name__ == "__main__":
    main()
