"""Single post-hoc head-path development diagnostic; original collector unchanged."""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import body
import probe as P
from probe import (
    CONFIG,
    POLICIES,
    Brain,
    Calls,
    Cortex,
    alarm,
    branch,
    check,
    pending_unknown,
    write,
)

KINDS = ("head_ordinary", "head_observer", "head_wide", "raw_ordinary", "raw_observer")
DESIGN = P.DESIGN | {
    "schema": "factual-completion-head-path-development/1",
    "kinds": list(KINDS),
    "seeds": [3109],
    "arm_seconds": 120,
    "study_bytes": 70_000_000,
    "information": "Identical13raw actor coordinates for all layouts. G always receives every raw fact; only head layouts remove direct raw F contacts. Hidden state/bias/missing current P/F remain excluded.",
    "scope": "Exactlyfive posthoc development arms on already viewed seed3109; no automatic seed or architecture expansion. Same current-state completion, not future action control or integrated attention.",
    "reason": "Post-hoc test after saved arithmetic showed raw F drive about .19-.22 versus G-state drive about .001. Remove direct raw F contacts for all three head layouts; G still gets every raw fact. Original raw ordinary and observer are rerun controls with exact acquisition parity required. Same rows,128x16dose,checks,branches,learning and quality thresholds. This is development on a viewed seed, not confirmation.",
    "comparison": "Head observer must satisfy unchanged10%+.002 quality predicate versus head ordinary, head wide and original raw ordinary, and unchanged10% own-correction predicate. Raw observer is descriptive. Allfive acquisition/completeness gates required.120s is a prospective new per-arm ceiling; original wide60s censor remains unchanged. No further architecture/dose/seed expansion follows automatically.",
}


def source_pins():
    own = Path(__file__).resolve().parent
    return P.source_pins() | {
        str(own / n): P.sha(own / n)
        for n in ("head_path.py", "test_head_path.py", "HEAD_PROTOCOL.md")
    }


def make(kind, seed):
    if kind.startswith("raw_"):
        return P.make(kind[4:], seed)
    if kind not in KINDS:
        raise ValueError("unknown head-path layout")
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
    width = 8 if kind == "head_wide" else 4
    h = c.column(
        "latent", patches=width, inputs=(raw["coarse"], raw["prior"], raw["command"])
    )
    p = c.column("predicted_fine", patches=2, inputs=h)
    if kind == "head_observer":
        constraint = c.observer(
            "constraint", patches=1, inputs=(h, raw["coarse"]), observes=p
        )
    else:
        constraint = c.column("constraint", patches=1, inputs=(h, raw["coarse"], p))
    g = c.column(
        "nonlinear_head", patches=width, inputs=(*raw.values(), h, p, constraint)
    )
    f = c.column("future", patches=1, inputs=g)
    c.output("p0", shape=1, reads=p, indices=(0,))
    c.output("p1", shape=1, reads=p, indices=(1,))
    c.output("c", shape=1, reads=constraint)
    c.output("f", shape=1, reads=f)
    return c.build()


def bound(root):
    p = json.loads((root / "protocol.json").read_text())
    if p["design"] != DESIGN or p["sources"] != source_pins():
        raise ValueError("frozen head-path source/design changed")
    if [(j["kind"], j["seed"], j["name"]) for j in p["inventory"]] != [
        (kind, 3109, kind + "-3109") for kind in KINDS
    ]:
        raise ValueError("frozen five-arm inventory changed")
    P.bound(Path(p["parent"]))
    if any(P.sha(name) != value for name, value in p["parent_pins"].items()):
        raise ValueError("parent evidence changed")
    if any(P.sha(root / name) != value for name, value in p["files"].items()):
        raise ValueError("frozen data/founder changed")
    return p


def prepare(root, parent, lesion):
    P.bound(parent)
    if (
        P.sha(parent / "protocol.json")
        != "a100fb48e6ad1b849802014124626ea9932065875df3ee652adeb7396c88876f"
    ):
        raise ValueError("wrong original preflight")
    if (
        P.sha(lesion / "saved-state-verification.json")
        != "9eceeee157a85b60765e378a5997b3a63ba41d194a11dfba9276dfa154aa9bc5"
    ):
        raise ValueError("wrong saved-state attenuation diagnostic")
    if root.exists():
        raise ValueError("Preserve previous head-path attempt")
    root.mkdir(parents=True)
    parent_names = [
        "protocol.json",
        "data.json",
        "summary.json",
        "execution.json",
        "founders.json",
        "ordinary-3109/checkpoint-128.json",
        "observer-3109/checkpoint-128.json",
    ]
    parent_pins = {str(parent / n): P.sha(parent / n) for n in parent_names}
    parent_pins |= {
        str(lesion / n): P.sha(lesion / n)
        for n in ("protocol.json", "result.json", "saved-state-verification.json")
    }
    shutil.copyfile(parent / "data.json", root / "data.json")
    shutil.copyfile(parent / "schedules.json", root / "schedules.json")
    old = json.loads((parent / "founders.json").read_text())
    founders, inventory = {}, []
    for kind in KINDS:
        name = kind + "-3109"
        brain = make(kind, 3109)
        if kind.startswith("raw_"):
            assert brain.snapshot() == old[kind[4:] + "-3109"]
        founders[name] = brain.snapshot()
        info = brain.inspect()
        inventory.append(
            {
                "kind": kind,
                "seed": 3109,
                "name": name,
                "patches": info["patches"],
                "edges": info["connections"],
                "parameters": info["patches"] + info["connections"],
            }
        )
    write(root / "founders.json", founders)
    write(
        root / "protocol.json",
        {
            "design": DESIGN,
            "parent": str(parent),
            "parent_pins": parent_pins,
            "inventory": inventory,
            "sources": source_pins(),
            "files": {
                n: P.sha(root / n)
                for n in ("data.json", "schedules.json", "founders.json")
            },
        },
    )
    bound(root)


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
        if job["kind"].startswith("raw_"):
            expected = Path(p["parent"]) / f"{job['kind'][4:]}-3109/checkpoint-128.json"
            assert acquired == expected.read_text(), (
                "raw acquisition must exactly reproduce parent"
            )
            report["raw_acquisition_exact_parent"] = True
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
        # Include source checks and initial report serialization in the final veto.
        report["seconds"] = time.perf_counter() - started
        if report["seconds"] >= DESIGN["arm_seconds"]:
            report["status"] = "failed_or_censored"
        write(directory / "report.json", report)
    return report


def summary(root, execution):
    p = bound(root)
    if [r["name"] for r in execution] != [j["name"] for j in p["inventory"]]:
        raise ValueError("Full frozen five-arm census required")
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
            and (
                not job["kind"].startswith("raw_")
                or report["raw_acquisition_exact_parent"]
            )
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
        reports = {c["job"]["kind"]: c["report"] for c in cases}
        o = reports["head_observer"]
        oe = o["branches"]["corrected"]["shift_mae"]
        for control in ("head_ordinary", "head_wide", "raw_ordinary"):
            c = reports[control]
            ce = c["branches"]["corrected"]["shift_mae"]
            pairs.append(
                {
                    "control": control,
                    "observer_mae": oe,
                    "control_mae": ce,
                    "improvement": ce - oe,
                    "quality_gate": oe <= 0.9 * ce
                    and ce - oe >= 0.002
                    and oe <= 0.9 * o["branches"]["routine"]["shift_mae"],
                    "observer_total_edge_work": o["work"]["edge_visits"],
                    "control_total_edge_work": c["work"]["edge_visits"],
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
        result.update(status="started", started=time.time())
        write(root / "execution.json", results)
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=125
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
        P.canonical(
            {
                k: report[k]
                for k in ("complete", "all_acquired", "development_quality_gate")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "arm"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--lesion", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root.resolve(), args.parent.resolve(), args.lesion.resolve())
        print(P.sha(args.root / "protocol.json"))
    elif args.command == "run":
        run(args.root.resolve())
    else:
        r = run_arm(args.root.resolve(), args.name)
        print(
            P.canonical(
                {k: r[k] for k in ("job", "status", "counts", "admissions", "seconds")}
            )
        )
        sys.exit(0 if r["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
