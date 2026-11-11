"""One paired, fixed-budget native witness-coverage diagnostic, preparation only.

Uses two fresh copies of the same passed32-module founder and raw local rule.
Action order/frequency/exposure are matched; only18 actions gain a second TRAIN
witness. Those previously queried TRAIN rows now become explicit teaching data.
Reused development19 is scored after each fixed schedule, never taught. This
diagnostic cannot promote the immutable original native independent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import single_module_native_24 as extension

from cadence import Brain

native = extension.native
ARMS = ("selected-only", "expanded-train")
SCHEMA = "cadence-native-matched-witness-coverage-v1"
BUDGET = 1008
BOUNDS = {
    "teaching_seconds": 300,
    "worker_seconds": 360,
    "admission_mib": 144,
    "output_cap_mib": 160,
}
FIXED = {
    "schema": SCHEMA,
    "arms": list(ARMS),
    "founder_seeds": {name: 0 for name in ARMS},
    "teacher_presentations_per_arm": BUDGET,
    "presentations_per_action": 42,
    "maximum_teacher_calls": 2020,
    "maximum_query_rows": 170,
    "model_construction": extension.school.CONSTRUCTION,
    "query_policy": (
        "selected24 newborn; after fixed schedule selected24/TRAIN18/development19 once each"
    ),
    "role_change": "previously queried independent TRAIN18 now teaching data in expanded arm",
    "whole_issue_passed": False,
    **BOUNDS,
}


@contextmanager
def admission_threads(protocol):
    """Nested historical guards require their declared threads even without solves."""
    previous = {name: os.environ.get(name) for name in protocol["threads"]}
    os.environ.update(protocol["threads"])
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def read_panel(root, name):
    if name not in ("school", "independent_train", "development"):
        raise ValueError("coverage diagnostic never admits the requested panel")
    fixture = root / "source/fixture"
    with np.load(fixture / "school.npz", allow_pickle=False) as archive:
        x, y = archive[name + "_inputs"], archive[name + "_labels"]
    identities = json.loads((fixture / "provenance.json").read_text())["panels"][name]
    count = {"school": 24, "independent_train": 18, "development": 19}[name]
    if x.shape != (count, 650) or y.shape != (count,) or not np.isfinite(x).all():
        raise ValueError("coverage panel census differs")
    for row, label, identity in zip(x, y, identities, strict=True):
        if (
            native.harness.row_hash(row) != identity["input_sha256"]
            or int(label) != identity["label"]
        ):
            raise ValueError("coverage panel identity differs")
    return x, y, identities


def schedules(root, old_order):
    sx, sy, si = read_panel(root, "school")
    tx, ty, ti = read_panel(root, "independent_train")
    if len(set(sy.tolist())) != 24 or len(set(ty.tolist())) != 18 or not set(ty) <= set(sy):
        raise ValueError("coverage action-label census differs")
    other = {int(label): index for index, label in enumerate(ty)}
    occurrence = Counter()
    result = {name: [] for name in ARMS}
    for school_row in old_order[: BUDGET + 1]:
        label = int(sy[school_row])
        occurrence[label] += 1
        for arm in ARMS:
            expanded = arm == "expanded-train" and label in other and occurrence[label] % 2 == 0
            panel, index, identities = (
                ("independent_train", other[label], ti) if expanded else ("school", school_row, si)
            )
            result[arm].append(
                {
                    "panel": panel,
                    "panel_row": int(index),
                    "school_row": int(school_row),
                    "label": label,
                    "frame": identities[index]["frame"],
                    "input_sha256": identities[index]["input_sha256"],
                }
            )
    for arm, rows in result.items():
        if len(rows) != BUDGET + 1 or set(Counter(r["label"] for r in rows[:BUDGET]).values()) != {
            42
        }:
            raise ValueError("coverage fixed exposure differs")
        if arm == "expanded-train":
            counts = Counter((r["label"], r["panel"]) for r in rows[:BUDGET])
            if any(
                counts[(label, panel)] != 21
                for label in other
                for panel in ("school", "independent_train")
            ):
                raise ValueError("expanded coverage is not exactly21+21 per available action")
    return result


def check_sources(root, protocol, *, worker_runtime=True):
    if any(protocol.get(k) != v for k, v in FIXED.items()) or any(
        type(protocol["founder_seeds"][name]) is not int for name in ARMS
    ):
        raise ValueError("matched coverage diagnostic changed gene/gate/exposure/bound")
    for folder, pins in (
        ("source/library/cadence", protocol["library_sources"]),
        ("source", protocol["producing_sources"]),
        ("source/fixture", protocol["fixture"]),
        ("original-native", protocol["original_native_capsule"]),
        ("original-audit", protocol["original_audit_capsule"]),
    ):
        for name, expected in pins.items():
            if native.harness.sha256(root / folder / name) != expected:
                raise ValueError("matched coverage frozen source/capsule changed: " + name)
    for name, expected in protocol["producing_sources"].items():
        imported = sys.modules.get(Path(name).stem)
        path = Path(imported.__file__) if imported else Path(__file__).with_name(name)
        if native.harness.sha256(path) != expected:
            raise ValueError("matched coverage imported source differs: " + name)
    if (
        native.harness.sources() != protocol["library_sources"]
        or native.runtime_precision() != protocol["runtime_precision"]
    ):
        raise ValueError("matched coverage source/runtime differs")
    if worker_runtime and any(os.environ.get(k) != v for k, v in protocol["threads"].items()):
        raise ValueError("matched coverage thread settings differ")
    original = root / "original-native"
    prior = json.loads((original / "protocol.json").read_text())
    extension.check_sources(original, prior, worker_runtime=worker_runtime)
    summary = json.loads((original / "summary.json").read_text())
    if (
        not summary["native_24_gate_passed"]
        or summary["independent_gate_passed"]
        or summary["whole_issue_passed"]
    ):
        raise ValueError("matched coverage requires the immutable native independent failure")
    for key in (
        "library_sources",
        "learner_config",
        "neuron_model",
        "model_construction",
        "runtime_precision",
        "initial_model_identity",
        "fixture",
    ):
        if prior[key] != protocol[key]:
            raise ValueError("coverage changed original graph/genes: " + key)
    audit = json.loads((root / "original-audit/result.json").read_text())
    audit_job = json.loads((root / "original-audit/job-protocol.json").read_text())
    if (
        not audit["verified"]
        or audit["job_protocol_sha256"]
        != native.harness.sha256(root / "original-audit/job-protocol.json")
        or audit_job["script_sha256"] != native.harness.sha256(root / "original-audit/script.py")
        or audit_job["protocol_sha256"] != native.harness.sha256(original / "protocol.json")
        or audit_job["summary_sha256"] != native.harness.sha256(original / "summary.json")
    ):
        raise ValueError("coverage original retained arithmetic audit binding differs")
    old_order = prior["orders_including_next_lesson"]["24"]
    if schedules(root, old_order) != protocol["schedules_including_next_lesson"]:
        raise ValueError("coverage teaching schedule differs")
    for arm in ARMS:
        path = root / ("prepared-" + arm + ".npz")
        if native.harness.sha256(path) != protocol["prepared_founders_sha256"][arm]:
            raise ValueError("coverage frozen founder changed")
        model = Brain.load(path)
        if (
            native.model_identity(model) != protocol["initial_model_identity"]
            or model.learner.contrast_updates
            or model.learner.updates
        ):
            raise ValueError("coverage founder is not the exact untrained physical graph")
    job = json.loads((root / "job-protocol.json").read_text())
    if (
        job["protocol_sha256"] != native.harness.sha256(root / "protocol.json")
        or job["bounds"] != BOUNDS
        or job["interpreter_sha256"]
        != hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    ):
        raise ValueError("coverage reviewed execution job changed")


def prepare(root, original, audit_root):
    if root.exists():
        raise FileExistsError(root)
    prior = json.loads((original / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(original / "source", root / "source", ignore=ignored)
    shutil.copyfile(Path(__file__), root / "source/native_coverage_control.py")
    shutil.copytree(original, root / "original-native", ignore=ignored)
    (root / "original-audit").mkdir()
    for name in ("job-protocol.json", "result.json"):
        shutil.copyfile(audit_root / name, root / "original-audit" / name)
    shutil.copyfile(
        audit_root.parent / "verification-sources/single_module_24_retained_audit.py",
        root / "original-audit/script.py",
    )
    founders = {}
    for arm in ARMS:
        founder = extension.school.make_brain("qualified", 0, native.ARGS)
        path = founder.save(root / ("prepared-" + arm + ".npz"))
        founders[arm] = native.harness.sha256(path)
    if (
        len(set(founders.values())) != 1
        or next(iter(founders.values())) != prior["prepared_founder_sha256"]
    ):
        raise ValueError("coverage fresh instances differ from passed physical founder")
    protocol = {
        **FIXED,
        **{
            key: prior[key]
            for key in (
                "library_sources",
                "learner_config",
                "neuron_model",
                "model_construction",
                "runtime_precision",
                "threads",
                "initial_model_identity",
                "fixture",
            )
        },
        "producing_sources": {
            **prior["producing_sources"],
            "native_coverage_control.py": native.harness.sha256(
                root / "source/native_coverage_control.py"
            ),
        },
        "original_native_capsule": {
            str(p.relative_to(root / "original-native")): native.harness.sha256(p)
            for p in sorted((root / "original-native").rglob("*"))
            if p.is_file()
        },
        "original_audit_capsule": {
            p.name: native.harness.sha256(p) for p in (root / "original-audit").iterdir()
        },
        "prepared_founders_sha256": founders,
        "schedules_including_next_lesson": schedules(
            root, prior["orders_including_next_lesson"]["24"]
        ),
        "frozen_source_mode": (
            "verbatim source/library/teacher from passed one-module native24 attempt"
        ),
        "development_role": (
            "already queried19-row panel reused once per arm after fixed schedule, "
            "never teaching; diagnostic only"
        ),
        "continuation": (
            "one extra declared teacher lesson on original/load copies per arm, included in work"
        ),
        "primary_comparison": (
            "expanded minus selected-only development correct rows, only if both fixed "
            "schedules and continuations complete without any refusal"
        ),
        "positive_coverage_effect": (
            "expanded development correct rows strictly greater than selected-only; "
            "no broad significance/default/native acceptance claim"
        ),
        "random_control": {
            "seed": 11003232,
            "actions": 36,
            "expected_row_accuracy": 1 / 36,
            "scope": (
                "uniform draws on the same final panels; computed separately from neural queries"
            ),
        },
        "heldout": "never decoded or queried",
        "scope": (
            "one paired coverage diagnostic; source/genes/initial draw/label sequence/exposure "
            "fixed; no original#110 or continuing-animal/generalization promotion"
        ),
    }
    native.harness.write_json(root / "protocol.json", protocol)
    native.harness.write_json(
        root / "job-protocol.json",
        {
            "schema": "cadence-native-matched-coverage-job-v1",
            "status": "prepared_not_executed",
            "protocol_sha256": native.harness.sha256(root / "protocol.json"),
            "interpreter": sys.executable,
            "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "bounds": BOUNDS,
            "teacher_calls": 0,
            "query_rows": 0,
            "scope": protocol["scope"],
        },
    )
    protocol = json.loads((root / "protocol.json").read_text())
    with admission_threads(protocol):
        check_sources(root, protocol, worker_runtime=False)
    native.harness.write_json(
        root / "admission.json",
        {
            "protocol_sha256": native.harness.sha256(root / "protocol.json"),
            "status": "prepared_not_executed",
            "source_schedule_both_founders_preflight": True,
            "teacher_calls": 0,
            "query_rows": 0,
            "heldout_decoded": False,
            "development_decoded_during_preparation": False,
        },
    )
    return protocol


def arm(root, name, data, protocol, deadline):
    model = Brain.load(root / ("prepared-" + name + ".npz"))
    folder = root / name
    folder.mkdir()
    schedule = protocol["schedules_including_next_lesson"][name]
    report = {
        "arm": name,
        "examples": 24,
        "tolerance": 0.003,
        "lessons": [],
        "recall": [],
        "passed": False,
        "diagnostic_valid": False,
        "status": "running",
        "complete": False,
        "current_operation": None,
        "development_read": False,
        "heldout_read": False,
        "training_roles": {"school": True, "independent_train": name == "expanded-train"},
    }

    def operation(kind, **details):
        report["current_operation"] = {"kind": kind, **details}
        native.persist_stage(folder, report, full=False)

    def query(panel, value, lesson):
        operation("query", panel=panel, rows=len(value[1]), lesson=lesson)
        reading = native.harness.free_recall(model, value[0], value[1])
        report["recall"].append({"panel": panel, "lesson": lesson, **reading})
        report["current_operation"] = None
        return reading

    try:
        operation("initial_checkpoint")
        report["initial_sha256"] = native.harness.sha256(model.save(folder / "initial.npz"))
        report["current_operation"] = None
        query("school", data["school"], 0)
        native.persist_stage(folder, report)
        for number, item in enumerate(schedule[:BUDGET], 1):
            if (
                time.monotonic() >= deadline
                or native.online.tree_bytes(root) >= protocol["admission_mib"] * 1024**2
            ):
                report["status"] = "time_or_output_limit"
                break
            if any(r["refusals"] for r in report["recall"]):
                report["status"] = "refused_initial_query"
                break
            x, y, _ = data[item["panel"]]
            index = item["panel_row"]
            operation("lesson", number=number, panel=item["panel"], panel_row=index)
            lesson = {
                "lesson": number,
                **item,
                **native.continual.graph_lesson(
                    model, x[[index]], y[[index]], folder, f"phases-{number:04d}"
                ),
            }
            report["lessons"].append(lesson)
            report["current_operation"] = None
            with (folder / "lessons.jsonl").open("a") as stream:
                stream.write(json.dumps(lesson) + "\n")
            native.persist_stage(folder, report, full=False)
            if not native.lesson_passed(lesson, 0.003):
                report["status"] = "refused_or_unqualified_teacher"
                break
            if number % 96 == 0:
                operation("progress_checkpoint", number=number)
                report["progress_sha256"] = native.harness.sha256(
                    model.save(folder / "progress.npz")
                )
                report["current_operation"] = None
                native.persist_stage(folder, report)
        else:
            report["status"] = "fixed_schedule_complete"
        operation("final_checkpoint")
        report["final_sha256"] = native.harness.sha256(model.save(folder / "final.npz"))
        report["current_operation"] = None
        if len(report["lessons"]) == BUDGET:
            query("school", data["school"], BUDGET)
            query("independent_train", data["independent_train"], BUDGET)
            # The only development decode is admitted after the fixed schedule.
            development = read_panel(root, "development")
            report["development_read"] = True
            report["development_panel_decodes"] = 1
            query("development", development, BUDGET)
        next_item = schedule[len(report["lessons"])]
        x, y, _ = data[next_item["panel"]]
        index = next_item["panel_row"]
        operation("continuation", teacher_calls=2, **next_item)
        report["continuation"] = native.continual.graph_checkpoint_continuation(
            model, Brain.load(folder / "final.npz"), x[[index]], y[[index]], folder
        )
        report["current_operation"] = None
        report["continuation_passed"] = native.continuation_passed(report["continuation"], 0.003)
        report["diagnostic_valid"] = bool(
            len(report["lessons"]) == BUDGET
            and report["continuation_passed"]
            and all(native.lesson_passed(r, 0.003) for r in report["lessons"])
            and not any(r["refusals"] for r in report["recall"])
        )
        report["complete"] = True
    except BaseException as error:
        report.update(
            status="process_exception", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        native.persist_stage(folder, report)
    return native.stage_result(report)


def finish(root, summary, protocol):
    native.write_receipt(root / "summary.json", summary)
    size = native.online.tree_bytes(root)
    summary["output_bytes"] = size
    summary["output_cap_exceeded"] = size > protocol["output_cap_mib"] * 1024**2
    if summary["output_cap_exceeded"]:
        summary.update(
            status="output_cap_exceeded", comparison_valid=False, positive_coverage_effect=False
        )
    native.write_receipt(root / "summary.json", summary)
    size = native.online.tree_bytes(root)
    if size > protocol["output_cap_mib"] * 1024**2:
        summary.update(
            output_bytes=size,
            output_cap_exceeded=True,
            status="output_cap_exceeded",
            comparison_valid=False,
            positive_coverage_effect=False,
        )
        native.write_receipt(root / "summary.json", summary)


def process_failure(root, protocol, reason):
    path = root / "summary.json"
    summary = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "protocol_sha256": native.harness.sha256(root / "protocol.json"),
            "arms": [{"arm": name, "status": "scheduled", "complete": False} for name in ARMS],
        }
    )
    summary.update(
        status="process_failed",
        complete=False,
        process_failure=reason,
        comparison_valid=False,
        positive_coverage_effect=False,
        whole_issue_passed=False,
    )
    for index, name in enumerate(ARMS):
        state = root / name / "state.json"
        if state.exists():
            recovered = json.loads(state.read_text())
            if recovered.get("arm") != name:
                raise ValueError("coverage recovery state differs from scheduled arm")
            summary["arms"][index] = recovered
    finish(root, summary, protocol)


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol)
    with (root / "execution-started.json").open("x") as stream:
        stream.write(native.harness.sha256(root / "protocol.json") + "\n")
    data = {name: read_panel(root, name) for name in ("school", "independent_train")}
    began = time.monotonic()
    summary = {
        "schema": "cadence-native-matched-coverage-result-v1",
        "protocol_sha256": native.harness.sha256(root / "protocol.json"),
        "arms": [{"arm": name, "status": "scheduled", "complete": False} for name in ARMS],
        "comparison_valid": False,
        "positive_coverage_effect": False,
        "development_gain_rows": None,
        "whole_issue_passed": False,
        "complete": False,
        "status": "running",
        "current_arm": None,
        "heldout_read": False,
        "process_failure": None,
    }
    try:
        for index, name in enumerate(ARMS):
            if (
                time.monotonic() >= began + protocol["teaching_seconds"]
                or native.online.tree_bytes(root) >= protocol["admission_mib"] * 1024**2
            ):
                summary["arms"][index]["status"] = "not_run_resource_limit"
                continue
            summary["current_arm"] = name
            finish(root, summary, protocol)
            summary["arms"][index] = arm(
                root, name, data, protocol, began + protocol["teaching_seconds"]
            )
            summary["current_arm"] = None
            finish(root, summary, protocol)
        valid = all(
            row.get("diagnostic_valid") and row.get("work_census_complete")
            for row in summary["arms"]
        )
        summary["comparison_valid"] = valid
        if valid:
            scores = [
                next(q["correct"] for q in row["recall"] if q["panel"] == "development")
                for row in summary["arms"]
            ]
            summary["development_gain_rows"] = scores[1] - scores[0]
            summary["positive_coverage_effect"] = scores[1] > scores[0]
        summary.update(
            complete=True, status="comparison_complete" if valid else "comparison_incomplete"
        )
        rng = np.random.default_rng(protocol["random_control"]["seed"])
        random = {}
        for name in ("school", "independent_train", "development"):
            if name == "development" and not all(
                r.get("development_read") for r in summary["arms"]
            ):
                continue
            if name in data:
                labels = data[name][1]
            else:
                # Reuse labels already retained by the first arm's scored read;
                # the development arrays are decoded exactly once per arm.
                scored = next(
                    q for q in summary["arms"][0]["recall"] if q["panel"] == name
                )
                labels = np.asarray([row["label"] for row in scored["rows"]])
            predictions = rng.integers(0, 36, size=len(labels))
            random[name] = {
                "correct": int((predictions == labels).sum()),
                "rows": len(labels),
                "predictions": predictions.tolist(),
                "brain_queries": 0,
            }
        summary["random_control"] = {**protocol["random_control"], "panels": random}
    except BaseException as error:
        summary.update(
            status="process_failed",
            process_failure={"type": type(error).__name__, "message": str(error)},
        )
        if summary["current_arm"]:
            name = summary["current_arm"]
            path = root / name / "receipt.json"
            if path.exists():
                summary["arms"][ARMS.index(name)] = native.stage_result(
                    json.loads(path.read_text())
                )
        raise
    finally:
        summary["seconds"] = time.monotonic() - began
        finish(root, summary, protocol)
    return summary


def launch(root):
    protocol = json.loads((root / "protocol.json").read_text())
    with admission_threads(protocol):
        check_sources(root, protocol, worker_runtime=False)
    if (root / "execution-started.json").exists():
        raise ValueError("coverage diagnostic already attempted")
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update(protocol["threads"])
    with (root / "worker.log").open("w") as stream:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "source/native_coverage_control.py"),
                    "--worker",
                    "--out",
                    str(root),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=protocol["worker_seconds"],
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            process_failure(root, protocol, {"kind": "timeout", "seconds": error.timeout})
            raise
    if completed.returncode:
        process_failure(root, protocol, {"kind": "worker_exit", "exit_code": completed.returncode})
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    summary = json.loads((root / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("coverage worker exited without complete census")
    summary["process_exit_code"] = completed.returncode
    finish(root, summary, protocol)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
    elif args.execute_reviewed:
        result = launch(root)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "comparison_valid",
                        "development_gain_rows",
                        "positive_coverage_effect",
                        "seconds",
                    )
                }
            )
        )
    elif args.prepare_only and args.original and args.audit:
        prepare(root, args.original.resolve(), args.audit.resolve())
        print(
            json.dumps(
                {
                    "prepared": str(root),
                    "protocol_sha256": native.harness.sha256(root / "protocol.json"),
                    "teacher_calls": 0,
                    "query_rows": 0,
                }
            )
        )
    else:
        parser.error("prepare-only requires original/audit; execution awaits separate review")


if __name__ == "__main__":
    main()
