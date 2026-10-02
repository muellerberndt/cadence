"""Post-hoc, frozen-parameter lesions; no training or architecture confirmation."""

import argparse
import gzip
import json
import signal
import time
import traceback
from collections import Counter
from pathlib import Path

import body
import probe as P

from cadence import Brain

VARIANTS = ("unchanged", "zero_residual", "zero_state")
DESIGN = {
    "schema": "factual-completion-posthoc-lesion/1",
    "selected_parent": "observer-3109",
    "selection": "Root requested the one acquired preflight observer after seeing the failed/censored cohort. All original outcomes remain unchanged.",
    "variants": list(VARIANTS),
    "sets": ["clean", "shifted"],
    "modes": ["routine", "corrected"],
    "rows_per_set": 64,
    "expected_queries": 768,
    "fits": 0,
    "variant_seconds": 40,
    "total_bytes": 10_000_000,
    "intervention": "Fresh native Brain.from_snapshot clones; private _weights replacement zeros exactly the two learned P-error contacts into C or their two matching P-state contacts. No error/state/target fabrication, retraining, new optimizer or checkpoint claim of an admitted intervention. Full same-law qualification with resulting frozen coefficients remains mandatory.",
    "scope": "Post-hoc sensitivity to removal at one acquired initializer. Equal contact count, not equal removed weight norm. Residual lesions alter both representation and returning derivatives; this is not a pure stop-gradient ablation, exclusive recursion result, matched retraining result or release gate.",
}


def intervention(brain, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown lesion")
    info = brain.inspect()
    populations = {x["name"]: x["indices"] for x in info["populations"]}
    outputs = {
        x["name"]: populations[x["reads"]][x["indices"][0]] for x in info["outputs"]
    }
    p = {outputs["p0"], outputs["p1"]}
    c = outputs["c"]
    kind = "residual" if variant == "zero_residual" else "state"
    edges = (
        [
            i
            for i, (k, s, t) in enumerate(info["edges"])
            if k == kind and s in p and t == c
        ]
        if variant != "unchanged"
        else []
    )
    if variant != "unchanged" and len(edges) != 2:
        raise ValueError("Expected exact two corresponding P to C contacts")
    old = json.loads(brain.snapshot())
    weights = list(brain.weights)
    removed = [
        {"index": i, "edge": info["edges"][i], "old": weights[i], "new": 0.0}
        for i in edges
    ]
    for i in edges:
        weights[i] = 0.0
    # Explicit private research intervention, never presented as a public admission.
    brain._weights = tuple(weights)
    now = json.loads(brain.snapshot())
    assert {k: v for k, v in old.items() if k != "weights"} == {
        k: v for k, v in now.items() if k != "weights"
    }
    assert [w for i, w in enumerate(old["weights"]) if i not in edges] == [
        w for i, w in enumerate(now["weights"]) if i not in edges
    ]
    return removed


def pins(parent):
    own = Path(__file__).resolve().parent
    files = {
        str(parent / n): P.sha(parent / n)
        for n in (
            "protocol.json",
            "data.json",
            "summary.json",
            "execution.json",
            "observer-3109/checkpoint-128.json",
            "observer-3109/report.json",
            "observer-3109/calls.jsonl.gz",
        )
    }
    return (
        P.source_pins()
        | files
        | {str(own / n): P.sha(own / n) for n in ("lesion.py", "test_lesion.py")}
    )


def bound(root):
    protocol = json.loads((root / "protocol.json").read_text())
    parent = Path(protocol["parent"])
    if protocol["design"] != DESIGN or protocol["pins"] != pins(parent):
        raise ValueError("Lesion source or parent inputs changed")
    P.bound(parent)
    return protocol


def prepare(parent, root):
    P.bound(parent)
    if root.exists():
        raise ValueError("Preserve prior lesion outputs")
    report = json.loads((parent / "observer-3109/report.json").read_text())
    assert report["status"] == "complete" and report["acquired"]
    assert report["counts"] == {"settle": 1296, "observe_batch": 132}
    root.mkdir(parents=True)
    P.write(
        root / "protocol.json",
        {"design": DESIGN, "parent": str(parent), "pins": pins(parent)},
    )
    bound(root)


def run(root):
    protocol = bound(root)
    if (root / "execution.json").exists():
        raise ValueError("Never repeat an attempted lesion run")
    parent = Path(protocol["parent"])
    data = json.loads((parent / "data.json").read_text())
    parent_report = json.loads((parent / "observer-3109/report.json").read_text())
    saved = (parent / "observer-3109/checkpoint-128.json").read_text()
    execution = [{"variant": v, "status": "not_started"} for v in VARIANTS]
    P.write(root / "execution.json", execution)
    for item in execution:
        if (
            sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            >= DESIGN["total_bytes"]
        ):
            break
        variant = item["variant"]
        directory = root / variant
        directory.mkdir()
        brain = Brain.from_snapshot(saved)
        removed = intervention(brain, variant)
        before = brain.snapshot()
        (directory / "before-intervention.json").write_text(saved)
        (directory / "after-intervention.json").write_text(before)
        P.write(directory / "intervention.json", removed)
        started = time.perf_counter()
        signal.signal(signal.SIGALRM, P.alarm)
        signal.setitimer(signal.ITIMER_REAL, DESIGN["variant_seconds"])
        results, work, returned, unknown, query_seconds = {}, Counter(), 0, 0, 0.0
        with gzip.open(directory / "queries.jsonl.gz", "wt") as stream:
            try:
                for name in DESIGN["sets"]:
                    for mode in DESIGN["modes"]:
                        records = []
                        for row in data[name]:
                            args = body.inputs(row)
                            targets = body.facts(row) if mode == "corrected" else None
                            intent = {
                                "ordinal": returned + 1,
                                "set": name,
                                "mode": mode,
                                "id": row["id"],
                                "before": P.digest(before),
                                "arguments_sha256": P.digest([args, targets]),
                            }
                            P.write(
                                directory / "inflight.json",
                                intent | {"status": "started"},
                            )
                            tick = time.perf_counter()
                            result = brain.settle(args, targets=targets)
                            seconds = time.perf_counter() - tick
                            query_seconds += seconds
                            returned += 1
                            work.update(result["work"])
                            assert brain.snapshot() == before
                            stream.write(
                                P.canonical(
                                    intent
                                    | {"result": P.compact(result), "seconds": seconds}
                                )
                                + "\n"
                            )
                            stream.flush()
                            P.write(
                                directory / "inflight.json",
                                intent | {"status": "returned"},
                            )
                            if not result["qualified"]:
                                raise ValueError(
                                    "Refused lesion query; retained, never scored"
                                )
                            forecast = result["outputs"]["f"][0]
                            if variant == "unchanged" and name == "clean":
                                old = parent_report["checks"]["128"][mode]["rows"]
                                expected = next(r for r in old if r["id"] == row["id"])
                                assert forecast == expected["prediction"]
                            if (
                                variant == "unchanged"
                                and name == "shifted"
                                and mode == "routine"
                            ):
                                old = parent_report["branches"]["routine"][
                                    "retained_shifted"
                                ]["rows"]
                                expected = next(r for r in old if r["id"] == row["id"])
                                assert forecast == expected["prediction"]
                            records.append(
                                {
                                    "id": row["id"],
                                    "forecast": forecast,
                                    "actual": row["actual"][3],
                                    "absolute_error": abs(forecast - row["actual"][3]),
                                }
                            )
                        results[f"{name}/{mode}"] = {
                            "rows": records,
                            "mae": sum(r["absolute_error"] for r in records) / 64,
                        }
                item["status"] = "complete"
            except BaseException:
                item["status"] = "failed_or_censored"
                item["exception"] = traceback.format_exc()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        inflight_path = directory / "inflight.json"
        if inflight_path.exists():
            intent = json.loads(inflight_path.read_text())
            if intent["status"] == "started":
                with gzip.open(directory / "queries.jsonl.gz", "rt") as stream:
                    logged = {json.loads(line)["ordinal"] for line in stream}
                unknown = int(intent["ordinal"] not in logged)
        item.update(
            seconds=time.perf_counter() - started,
            returned_queries=returned,
            unknown_calls=unknown,
            query_seconds=query_seconds,
            work=dict(work),
            results=results,
            unchanged_snapshot=brain.snapshot() == before,
            removed=removed,
        )
        if (
            item["seconds"] >= DESIGN["variant_seconds"]
            or not item["unchanged_snapshot"]
        ):
            item["status"] = "failed_or_censored"
        P.write(directory / "report.json", item)
        P.write(root / "execution.json", execution)
    bound(root)
    complete = (
        all(
            r["status"] == "complete"
            and r["returned_queries"] == 256
            and r["unknown_calls"] == 0
            for r in execution
        )
        and sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        < DESIGN["total_bytes"]
    )
    contrasts = []
    if complete:
        baseline = execution[0]["results"]
        for lesion in execution[1:]:
            for name in DESIGN["sets"]:
                b0, b1 = [baseline[f"{name}/{mode}"]["mae"] for mode in DESIGN["modes"]]
                a0, a1 = [
                    lesion["results"][f"{name}/{mode}"]["mae"]
                    for mode in DESIGN["modes"]
                ]
                contrasts.append(
                    {
                        "variant": lesion["variant"],
                        "set": name,
                        "routine_mae_change": a0 - b0,
                        "corrected_mae_change": a1 - b1,
                        "baseline_clamp_benefit": b0 - b1,
                        "lesioned_clamp_benefit": a0 - a1,
                    }
                )
    P.write(
        root / "result.json",
        {
            "complete": complete,
            "source_unchanged": True,
            "queries": sum(r.get("returned_queries", 0) for r in execution),
            "admissions": 0,
            "variants": execution,
            "contrasts": contrasts,
            "scope": DESIGN["scope"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent.resolve(), args.root.resolve())
        print(P.sha(args.root / "protocol.json"))
    else:
        run(args.root.resolve())
        result = json.loads((args.root / "result.json").read_text())
        print(
            P.canonical({k: result[k] for k in ("complete", "queries", "admissions")})
        )


if __name__ == "__main__":
    main()
