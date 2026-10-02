"""Frozen development screen for factual, partially clamped world prediction.

No action policy, reward learning, automatic attention, or independent clocks.
All prediction methods are pure; two scheduled factual updates preserve activity.
"""

import argparse
import gzip
import hashlib
import json
import random
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import body

import cadence
from cadence import Brain, Cortex
from cadence.brain import IMPLEMENTATION

KINDS = ("ordinary", "observer", "wide")
POLICIES = ("routine", "corrected", "routine_fit", "corrected_fit")
CONFIG = dict(
    initial_scale=0.3,
    state_prior=0.01,
    parameter_prior=0.1,
    settle_budget=2048,
    tolerance=1e-6,
)
DESIGN = {
    "schema": "factual-completion-development/1",
    "preflight_seeds": [3109],
    "screen_seeds": [3119, 3121, 3137],
    "kinds": list(KINDS),
    "policies": list(POLICIES),
    "config": CONFIG,
    "updates": 128,
    "batch": 16,
    "checks": [0, 32, 64, 128],
    "live_steps": 64,
    "factual_fit_after_steps": [31, 47],
    "factual_fit_rows": 16,
    "arm_seconds": 60,
    "study_bytes": 70_000_000,
    "gates": {
        "acquisition_mae": 0.04,
        "newborn_reduction": 0.5,
        "observer_relative": 0.1,
        "observer_absolute": 0.002,
        "retention_old_mae_increase": 0.005,
        "retention_relative": 0.1,
    },
    "information": "Every layout receives identical coarse/observed-fine/presence/command/immutable-prior ports. Hidden state, calibration bias, unavailable fine coordinate and future F never enter inputs. All layouts have a raw-data head bypass.",
    "learning": "128 shared batches of16 actual sensor rows; same one-visible-P,C,F witness targets. The other P remains free, including training. Each live-fit branch gets exactly two scheduled16-row actual-outcome batches; no optimal action, corrected-state pseudo-target, reward or private label.",
    "comparison": "Ordinary and observer use H4/G4; wide uses H8/G8. C's two additional P-error contacts are the only ordinary/observer wiring difference; ordinary already backreacts. Equal numeric seeds do not guarantee equal common edge coefficients. Compare all three, count full work, do not claim exact capacity matching.",
    "live": "One common exogenous-command body tape, actual coarse sensor bias changes after step15. Predictions precede F reveal. Routine uses no P/C clamps but still gets every observed raw measurement. Corrected additionally clamps factual P/C, with F free. Fits after31/47 are scheduled diagnostic controls, not learned attention. Four pure-query branches preserve activity; fit twins must retain identical parameters.",
    "scope": "Development prediction/completion prerequisite, not inverse control, autonomous adaptation or all-three capability proof. One preflight seed, then a separately frozen three-seed screen only after review; no automatic launch or success-selected arms. Eight-seed confirmation requires a new frozen protocol, no reuse as fresh evidence.",
    "interpretation": "No fitting gate at a clamped F. Training/query latent changes are retained diagnostics. C residual now responds to actual factual constraints, but any positive observer result is finite-budget wiring/representation benefit, not exclusive expressivity or persistent memory. Prior actuators already used real P/C clamps.",
}
EXPECTED_CORE = {
    "brain.py": "93287ff9751a79065fadd462d94fc91f77a87650c28ec5c6c8bed30bd8363807",
    "cortex.py": "270f963fa6ca1c91b0deb19ce0f440cd4881f79205fdd115f64ef23942463482",
    "column.py": "d3752d27227ca34dfc03134d792cc046fc41d96ac181f19be852a193d1eee203",
    "ports.py": "f0ec46ba314ee20523ff603473256362ae84a1522e89d39751f082f278aa4dca",
    "_repair.py": "76f57fcdffad1e39b65e327af8a2ae26f89467dafa11520e96ac07a8ea67595a",
    "_validation.py": "05d09fd56ec96ef9315cfe569fa818a718795d1aedefd12d5b02bb619808b0ab",
    "_tensor.py": "d5765b286c931863cf540d19133500f734b70270c1bf702d60476c8945d86ce9",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def source_pins():
    directory = Path(__file__).resolve().parent
    package = Path(cadence.__file__).resolve().parent
    return {
        str(p): sha(p)
        for p in [
            *(package / n for n in IMPLEMENTATION),
            *(
                directory / n
                for n in ("body.py", "probe.py", "test_probe.py", "PROTOCOL.md")
            ),
        ]
    }


def make(kind, seed):
    if kind not in KINDS:
        raise ValueError("unknown layout")
    c = Cortex(seed=seed, **CONFIG)
    raw = {
        name: c.input(name, shape=n)
        for name, n in (
            ("coarse", 2),
            ("command", 2),
            ("prior", 3),
            ("fine", 3),
            ("present", 3),
        )
    }
    width = 8 if kind == "wide" else 4
    h = c.column(
        "latent", patches=width, inputs=(raw["coarse"], raw["prior"], raw["command"])
    )
    p = c.column("predicted_fine", patches=2, inputs=h)
    if kind == "observer":
        constraint = c.observer(
            "constraint", patches=1, inputs=(h, raw["coarse"]), observes=p
        )
    else:
        constraint = c.column("constraint", patches=1, inputs=(h, raw["coarse"], p))
    g = c.column(
        "nonlinear_head", patches=width, inputs=(*raw.values(), h, p, constraint)
    )
    f = c.column("future", patches=1, inputs=(g, *raw.values()))
    c.output("p0", shape=1, reads=p, indices=(0,))
    c.output("p1", shape=1, reads=p, indices=(1,))
    c.output("c", shape=1, reads=constraint)
    c.output("f", shape=1, reads=f)
    return c.build()


def schedule(seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(4):
        ids = list(range(512))
        rng.shuffle(ids)
        rows.extend(ids)
    return [rows[i : i + 16] for i in range(0, len(rows), 16)]


def bound(root):
    p = json.loads((root / "protocol.json").read_text())
    if p["design"] != DESIGN or p["sources"] != source_pins():
        raise ValueError("frozen source/design changed")
    if any(sha(root / name) != value for name, value in p["files"].items()):
        raise ValueError("frozen data/founder changed")
    if (
        dict(IMPLEMENTATION) != EXPECTED_CORE
        or p["mathematical_modules"] != EXPECTED_CORE
    ):
        raise ValueError("loaded mathematical implementation differs")
    return p


def prepare(root, phase):
    if dict(IMPLEMENTATION) != EXPECTED_CORE:
        raise ValueError("Requires exactly public 70e7a6a mathematical sources")
    if root.exists():
        raise ValueError("Preserve prior attempts; choose a new directory")
    root.mkdir(parents=True)
    seeds = DESIGN[f"{phase}_seeds"]
    write(root / "data.json", body.data())
    founders = {}
    inventory = []
    for seed in seeds:
        order = list(KINDS)
        shift = seeds.index(seed) % len(order)
        for kind in order[shift:] + order[:shift]:
            brain = make(kind, seed)
            name = f"{kind}-{seed}"
            founders[name] = brain.snapshot()
            info = brain.inspect()
            inventory.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "name": name,
                    "patches": info["patches"],
                    "edges": info["connections"],
                    "parameters": info["patches"] + info["connections"],
                }
            )
    write(root / "founders.json", founders)
    write(root / "schedules.json", {str(s): schedule(s) for s in seeds})
    write(
        root / "protocol.json",
        {
            "design": DESIGN,
            "phase": phase,
            "sources": source_pins(),
            "inventory": inventory,
            "mathematical_modules": dict(IMPLEMENTATION),
            "files": {
                n: sha(root / n)
                for n in ("data.json", "founders.json", "schedules.json")
            },
        },
    )
    bound(root)


class Cap(Exception):
    pass


def alarm(*_):
    raise Cap("independent per-arm wall cap")


def compact(result):
    return {k: v for k, v in result.items() if k not in ("weights", "biases")} | {
        "parameter_sha256": digest([result["weights"], result["biases"]])
    }


class Calls:
    def __init__(self, directory):
        self.directory = directory
        self.stream = gzip.open(directory / "calls.jsonl.gz", "wt")
        self.counts, self.work = Counter(), Counter()
        self.admissions = 0
        self.seconds = 0.0
        self.unknown = 0

    def call(self, brain, method, argument, targets, meta):
        before = brain.snapshot()
        intent = {
            "ordinal": sum(self.counts.values()) + self.unknown + 1,
            "method": method,
            "meta": meta,
            "before": digest(before),
            "arguments_sha256": digest([argument, targets]),
        }
        write(self.directory / "inflight.json", intent | {"status": "started"})
        started = time.perf_counter()
        try:
            if method == "observe_batch":
                result = brain.observe_batch(argument, source="witness")
            else:
                result = brain.settle(argument, targets=targets)
        except BaseException:
            self.unknown += 1
            self.seconds += time.perf_counter() - started
            self.stream.write(
                canonical(
                    intent
                    | {
                        "status": "interrupted_or_error",
                        "seconds": time.perf_counter() - started,
                        "unknown_solver_work": True,
                    }
                )
                + "\n"
            )
            self.stream.flush()
            if brain.snapshot() != before:
                raise AssertionError("failed operation mutated continuation") from None
            raise
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.counts[method] += 1
        self.work.update(result["work"])
        if method == "settle":
            assert brain.snapshot() == before, "pure query changed continuation"
        else:
            assert json.loads(brain.snapshot())["state"] == json.loads(before)["state"]
            if result["accepted"]:
                self.admissions += 1
                (self.directory / "latest.json").write_text(brain.snapshot())
        self.stream.write(
            canonical(
                intent
                | {
                    "status": "returned",
                    "seconds": elapsed,
                    "after": digest(brain.snapshot()),
                    "result": compact(result),
                }
            )
            + "\n"
        )
        self.stream.flush()
        write(self.directory / "inflight.json", intent | {"status": "returned"})
        if not result["qualified"]:
            raise Cap("returned unqualified result retained; no output promotion")
        return result


def pending_unknown(directory):
    path = directory / "inflight.json"
    if not path.exists():
        return False
    intent = json.loads(path.read_text())
    if intent["status"] != "started":
        return False
    with gzip.open(directory / "calls.jsonl.gz", "rt") as stream:
        return not any(
            row["ordinal"] == intent["ordinal"] and row["status"] == "returned"
            for row in (json.loads(line) for line in stream)
        )


def check(calls, brain, rows, label, corrected):
    output = []
    for row in rows:
        result = calls.call(
            brain,
            "settle",
            body.inputs(row),
            body.facts(row) if corrected else None,
            {"phase": label, "id": row["id"], "corrected": corrected},
        )
        prediction = result["outputs"]["f"][0]
        output.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "actual": row["actual"][3],
                "error": abs(prediction - row["actual"][3]),
            }
        )
    return {"mae": sum(r["error"] for r in output) / len(output), "rows": output}


def branch(calls, brain, rows, policy, directory):
    corrected = policy.startswith("corrected")
    learned = policy.endswith("fit")
    records = []
    for index, row in enumerate(rows):
        inputs = body.inputs(row)
        write(
            directory / "outcome-intent.json",
            {"id": row["id"], "policy": policy, "status": "awaiting_forecast"},
        )
        result = calls.call(
            brain,
            "settle",
            inputs,
            body.facts(row) if corrected else None,
            {"phase": "live", "policy": policy, "id": row["id"]},
        )
        predicted = result["outputs"]["f"][0]
        # The physical tape is fixed independently; the prediction is now immutable.
        receipt = {
            "id": row["id"],
            "inputs_sha256": digest(inputs),
            "issued_prediction": predicted,
            "actual_f": row["actual"][3],
            "body_record_sha256": digest(row),
            "error": abs(predicted - row["actual"][3]),
            "model_sha256": digest(brain.snapshot()),
        }
        records.append(receipt)
        write(directory / "outcomes.json", records)
        write(
            directory / "outcome-intent.json",
            {"id": row["id"], "status": "acknowledged"},
        )
        if learned and index in DESIGN["factual_fit_after_steps"]:
            actual = rows[index - 15 : index + 1]
            batch = [(body.inputs(r), body.facts(r, reveal=True)) for r in actual]
            calls.call(
                brain,
                "observe_batch",
                batch,
                None,
                {
                    "phase": "live_fit",
                    "policy": policy,
                    "ids": [r["id"] for r in actual],
                },
            )
    return {
        "rows": records,
        "late_mae": sum(r["error"] for r in records[48:]) / 16,
        "shift_mae": sum(r["error"] for r in records[16:]) / 48,
        "final_sha256": digest(brain.snapshot()),
    }


def run_arm(root, name):
    p = bound(root)
    job = next(j for j in p["inventory"] if j["name"] == name)
    directory = root / name
    directory.mkdir()
    tape = json.loads((root / "data.json").read_text())
    schedule_rows = json.loads((root / "schedules.json").read_text())[str(job["seed"])]
    brain = Brain.from_snapshot(json.loads((root / "founders.json").read_text())[name])
    (directory / "initial.json").write_text(brain.snapshot())
    calls = Calls(directory)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, DESIGN["arm_seconds"])
    report = {"job": job, "status": "started", "checks": {}, "branches": {}}
    try:
        for update in range(129):
            if update in DESIGN["checks"]:
                (directory / f"checkpoint-{update}.json").write_text(brain.snapshot())
                report["checks"][str(update)] = {
                    "routine": check(
                        calls, brain, tape["clean"], f"check-{update}", False
                    ),
                    "corrected": check(
                        calls, brain, tape["clean"], f"check-{update}", True
                    ),
                }
                write(directory / "report.json", report)
            if update == 128:
                break
            rows = [tape["train"][i] for i in schedule_rows[update]]
            calls.call(
                brain,
                "observe_batch",
                [(body.inputs(r), body.facts(r, reveal=True)) for r in rows],
                None,
                {
                    "phase": "training",
                    "update": update + 1,
                    "ids": [r["id"] for r in rows],
                },
            )
        acquired = brain.snapshot()
        report["acquired"] = all(
            report["checks"]["128"][mode]["mae"] <= 0.04
            and report["checks"]["128"][mode]["mae"]
            <= 0.5 * report["checks"]["0"][mode]["mae"]
            for mode in ("routine", "corrected")
        )
        diagnostics = []
        for row in tape["clean"][:8]:
            pair = []
            for teacher in (False, True):
                result = calls.call(
                    brain,
                    "settle",
                    body.inputs(row),
                    body.facts(row, reveal=teacher),
                    {
                        "phase": "teacher_clamp_diagnostic",
                        "id": row["id"],
                        "teacher": teacher,
                    },
                )
                pair.append(
                    {
                        "state": result["state"],
                        "errors": result["errors"],
                        "predictions": result["predictions"],
                    }
                )
            diagnostics.append(
                {"id": row["id"], "free_f": pair[0], "actual_f_clamped": pair[1]}
            )
        report["teacher_diagnostic"] = diagnostics
        # All branches are diagnostic and retained even if the acquisition gate fails.
        twins = {}
        for policy in POLICIES:
            child = Brain.from_snapshot(acquired)
            child_dir = directory / policy
            child_dir.mkdir()
            detail = branch(calls, child, tape["live"], policy, child_dir)
            detail["retained_clean"] = check(
                calls, child, tape["clean"], policy + "-retained", False
            )
            detail["retained_shifted"] = check(
                calls, child, tape["shifted"], policy + "-retained", False
            )
            (child_dir / "final.json").write_text(child.snapshot())
            twins[policy] = child.snapshot()
            report["branches"][policy] = detail
        assert twins["routine"] == twins["corrected"] == acquired
        assert twins["routine_fit"] == twins["corrected_fit"]
        report["same_learning_twins"] = True
        report["retention_gate"] = (
            report["branches"]["routine_fit"]["retained_clean"]["mae"]
            <= report["branches"]["routine"]["retained_clean"]["mae"] + 0.005
            and report["branches"]["routine_fit"]["retained_shifted"]["mae"]
            <= 0.9 * report["branches"]["routine"]["retained_shifted"]["mae"]
        )
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed_or_censored"
        report["exception"] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        calls.stream.close()
        if pending_unknown(directory):
            calls.unknown = max(calls.unknown, 1)
        report.update(
            seconds=time.perf_counter() - started,
            counts=dict(calls.counts),
            admissions=calls.admissions,
            work=dict(calls.work),
            call_seconds=calls.seconds,
            unknown_calls=calls.unknown,
        )
        report["source_unchanged"] = source_pins() == p["sources"]
        if not report["source_unchanged"] or report["seconds"] >= DESIGN["arm_seconds"]:
            report["status"] = "failed_or_censored"
        write(directory / "report.json", report)
    return report


def summary(root, execution):
    p = bound(root)
    if [r["name"] for r in execution] != [j["name"] for j in p["inventory"]]:
        raise ValueError("Full frozen execution census required")
    cases = []
    for job, outcome in zip(p["inventory"], execution, strict=True):
        path = root / job["name"] / "report.json"
        report = json.loads(path.read_text()) if path.exists() else None
        if report is not None and report["job"] != job:
            raise ValueError("case identity changed")
        eligible = bool(
            report
            and report["status"] == "complete"
            and report["source_unchanged"]
            and report["same_learning_twins"]
            and report["admissions"] == 132
            and report["counts"] == {"settle": 1296, "observe_batch": 132}
            and report["unknown_calls"] == 0
            and outcome["status"] == "returned"
            and outcome["returncode"] == 0
        )
        cases.append(
            {
                "job": job,
                "eligible": eligible,
                "acquired": eligible and report["acquired"],
                "report": report,
            }
        )
    complete = all(c["eligible"] for c in cases)
    acquired = all(c["acquired"] for c in cases)
    pairs = []
    if complete:
        for seed in DESIGN[f"{p['phase']}_seeds"]:
            reports = {
                c["job"]["kind"]: c["report"] for c in cases if c["job"]["seed"] == seed
            }
            for control in ("ordinary", "wide"):
                o, c = reports["observer"], reports[control]
                oe = o["branches"]["corrected"]["shift_mae"]
                ce = c["branches"]["corrected"]["shift_mae"]
                pairs.append(
                    {
                        "seed": seed,
                        "control": control,
                        "observer_mae": oe,
                        "control_mae": ce,
                        "improvement": ce - oe,
                        "quality_gate": oe <= 0.9 * ce
                        and ce - oe >= 0.002
                        and oe <= 0.9 * o["branches"]["routine"]["shift_mae"],
                        "total_edge_work_observer": o["work"]["edge_visits"],
                        "total_edge_work_control": c["work"]["edge_visits"],
                    }
                )
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return {
        "complete": complete,
        "all_acquired": acquired,
        "cases": cases,
        "comparison_pairs": pairs,
        "bytes_before_summary": size,
        "development_quality_gate": complete
        and acquired
        and bool(pairs)
        and all(r["quality_gate"] for r in pairs)
        and size <= DESIGN["study_bytes"],
        "confirmatory_claim": False,
        "integrated_three_capabilities": False,
    }


def run(root):
    p = bound(root)
    if (root / "execution.json").exists() or any(
        (root / j["name"]).exists() for j in p["inventory"]
    ):
        raise ValueError("Do not repeat or overwrite an attempted cohort")
    results = [
        {"name": j["name"], "status": "not_started", "returncode": None}
        for j in p["inventory"]
    ]
    write(root / "execution.json", results)
    # Independent processes; run serially for the initial bounded CPU preflight.
    for result in results:
        if (
            sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
            >= DESIGN["study_bytes"]
        ):
            break
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "arm",
            str(root),
            "--name",
            result["name"],
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=65
            )
            result.update(
                status="returned",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as error:
            result.update(status="outer_timeout", error=str(error))
        except Exception:
            result.update(status="launch_error", error=traceback.format_exc())
        write(root / "execution.json", results)
    report = summary(root, results)
    write(root / "summary.json", report)
    print(
        canonical(
            {
                k: report[k]
                for k in ("complete", "all_acquired", "development_quality_gate")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "arm", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--phase", choices=("preflight", "screen"), default="preflight")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root.resolve(), args.phase)
        print(canonical({"protocol_sha256": sha(args.root / "protocol.json")}))
    elif args.command == "run":
        run(args.root.resolve())
    else:
        r = run_arm(args.root.resolve(), args.name)
        print(
            canonical(
                {k: r[k] for k in ("job", "status", "counts", "admissions", "seconds")}
            )
        )
        sys.exit(0 if r["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
