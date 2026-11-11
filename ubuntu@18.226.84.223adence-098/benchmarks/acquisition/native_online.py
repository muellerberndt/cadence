"""One source-frozen single-encounter transfer to the original movie school.

Uses the confirmed small-RMS teacher gene, never an external answer head. This
is a graph-plasticity microscope, not a continuing animal or gameplay claim.
Independent movie rows are read once after the selected-row screen; heldout
arrays stay sealed. The batch transfer remains a separate negative control.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import continual
import numpy as np
import online_curriculum as online
import run as harness

from cadence import Brain

CAPS = {2: 512, 4: 1024, 24: 4096}
POSITIONS = {2: [0, 9], 4: [0, 9, 17, 22], 24: list(range(24))}
PRODUCERS = (
    "native_online.py",
    "run.py",
    "extract.py",
    "verify.py",
    "continual.py",
    "online_curriculum.py",
    "relation_development.py",
    "relations.py",
)
ARGS = SimpleNamespace(
    rate=0.003,
    free_steps=4096,
    nudged_steps=4096,
    damping=3,
    tolerance=0.003,
    nudge="cross_entropy",
    gene="lateral0-small-rms",
)
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
PANEL_COUNTS = {"school": 24, "independent_train": 18, "development": 19}
MODEL_CONSTRUCTION = {
    "inputs": 650, "actions": 36, "modules": [32, 16], "observers": [], "lateral": 0.0,
}


def runtime_precision():
    return {
        "numpy": np.__version__, "dtype": "float64",
        "eps": float(np.finfo(np.float64).eps),
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
        "python": sys.version, "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }


def model_identity(model):
    graph = model.brain
    arrays = {
        "pre": model.connectome.pre, "post": model.connectome.post,
        "sign": model.connectome.sign, "count": model.connectome.count,
        "efficacy": graph.efficacy, "bias": graph.bias, "log_gain": graph.log_gain,
        "sensory_index": model.sensory_index, "motor_index": model.motor_index,
        "plastic_synapses": model.learner.plastic_synapses,
        "plastic_neurons": model.learner.plastic_neurons,
    }
    return {
        "arrays": {name: harness.row_hash(value) for name, value in arrays.items()},
        "populations": {name: harness.row_hash(value)
                        for name, value in model.connectome.populations.items()},
        "learner_config": model.learner.config.to_dict(),
        "neuron_model": graph.neuron_model.to_dict(),
        "layout": graph.layout.to_dict(), "backend": graph.backend,
        "precision": graph.precision, "dense_limit": graph.dense_limit,
    }


def check_reference(prior, summary, digest, model):
    outcomes = summary.get("outcomes", [])
    seeds = prior.get("founder_seeds", [])
    if (summary.get("all_founders_passed") is not True
            or summary.get("founder_denominator") != 5
            or len(seeds) != 5 or len(set(seeds)) != 5
            or len(outcomes) != 5 or {row.get("seed") for row in outcomes} != set(seeds)
            or summary.get("recipe_sha256") != digest
            or any(row.get("passed") is not True or row.get("status") != "confirmed"
                   or row.get("recipe_sha256") != digest
                   or row.get("process_exit_code") != 0
                   or row.get("old_passed") is not True
                   or row.get("development_passed") is not True
                   or row.get("continuation_passed") is not True
                   or row.get("heldout_read") is not True for row in outcomes)):
        raise ValueError("reference must be the completed hash-bound five-founder confirmation")
    if (model.learner.config.to_dict() != prior["learner_config"]
            or model.brain.neuron_model.to_dict() != prior["neuron_model"]
            or prior["model_construction"] != MODEL_CONSTRUCTION):
        raise ValueError("native transfer changed confirmed model/teaching genes")
    if harness.sources() != prior["library_sources"]:
        raise ValueError("native transfer changed confirmed library source")
    actual = runtime_precision()
    if any(actual[name] != prior["runtime_precision"][name]
           for name in ("numpy", "dtype", "eps")):
        raise ValueError("native transfer changed confirmed numerical runtime")


def load_fixture(fixture):
    """Check every permitted row before teaching; never materialize heldout arrays."""
    provenance = json.loads((fixture / "provenance.json").read_text())
    with np.load(fixture / "school.npz", allow_pickle=False) as archive:
        data = {name + suffix: archive[name + suffix]
                for name in PANEL_COUNTS for suffix in ("_inputs", "_labels")}
    for name, count in PANEL_COUNTS.items():
        x, y = data[name + "_inputs"], data[name + "_labels"]
        identities = provenance["panels"][name]
        if (x.shape != (count, 650) or x.dtype != np.dtype("float64")
                or y.shape != (count,) or not np.issubdtype(y.dtype, np.integer)
                or not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= 36)
                or len(identities) != count):
            raise ValueError("native panel shape/type/identity census differs: " + name)
        for row, identity in enumerate(identities):
            if (harness.row_hash(x[row]) != identity["input_sha256"]
                    or int(y[row]) != identity["label"]):
                raise ValueError("native row identity differs: " + name)
    return data


def check_protocol(protocol):
    if protocol.get("schema") != "cadence-native-online-transfer-v2":
        raise ValueError("use the frozen producer to reproduce a historical native protocol")
    if protocol["stage_counts"] != [2, 4, 24]:
        raise ValueError("native stage census differs")
    for count in protocol["stage_counts"]:
        key = str(count)
        cap, every = protocol["lesson_caps"][key], protocol["query_every"][key]
        positions, order = protocol["positions"][key], protocol["orders_including_next_lesson"][key]
        if (type(cap) is not int or cap <= 0 or type(every) is not int or every <= 0
                or len(positions) != count or len(set(positions)) != count
                or any(type(row) is not int or row < 0 or row >= 24 for row in positions)
                or len(order) != cap + 1 or any(row not in positions for row in order)):
            raise ValueError("native lesson/query census differs: " + key)
    for name in ("teaching_seconds", "worker_seconds", "admission_mib", "output_cap_mib"):
        if not np.isfinite(protocol[name]) or protocol[name] <= 0:
            raise ValueError("invalid native resource bound: " + name)
    if (protocol["worker_seconds"] <= protocol["teaching_seconds"]
            or protocol["admission_mib"] >= protocol["output_cap_mib"]):
        raise ValueError("native resource bounds leave no mandatory tail")


def check_sources(root, protocol, *, worker_runtime=True):
    check_protocol(protocol)
    admission = json.loads((root / "admission.json").read_text())
    if admission["protocol_sha256"] != harness.sha256(root / "protocol.json"):
        raise ValueError("native admitted protocol changed")
    for name, digest in protocol["library_sources"].items():
        if harness.sha256(root / "source/library/cadence" / name) != digest:
            raise ValueError("frozen library changed: " + name)
    for name, digest in protocol["producing_sources"].items():
        if harness.sha256(root / "source" / name) != digest:
            raise ValueError("frozen producer changed: " + name)
        imported = sys.modules.get(Path(name).stem)
        actual = Path(imported.__file__) if imported is not None else Path(__file__).with_name(name)
        if harness.sha256(actual) != digest:
            raise ValueError("imported producer changed: " + name)
    if (harness.sources() != protocol["library_sources"]
            or runtime_precision() != protocol["runtime_precision"]):
        raise ValueError("imported source/runtime differs from frozen protocol")
    if worker_runtime and any(os.environ.get(name) != value
                              for name, value in protocol["threads"].items()):
        raise ValueError("worker thread settings differ from frozen protocol")
    for name, digest in protocol["fixture"].items():
        if harness.sha256(root / "source/fixture" / name) != digest:
            raise ValueError("native fixture changed")
    for name, digest in protocol["reference"].items():
        filename = name.removesuffix("_sha256") + ".json"
        if harness.sha256(root / "reference" / filename) != digest:
            raise ValueError("native reference capsule changed")
    model = harness.make_brain("qualified", protocol["seed"], ARGS)
    if model_identity(model) != protocol["initial_model_identity"]:
        raise ValueError("native model differs from frozen construction")
    check_reference(
        json.loads((root / "reference/protocol.json").read_text()),
        json.loads((root / "reference/summary.json").read_text()),
        protocol["reference"]["protocol_sha256"], model,
    )


def prepare(root, fixture, reference):
    if root.exists():
        raise FileExistsError(root)
    protocol_bytes = (reference / "protocol.json").read_bytes()
    summary_bytes = (reference / "summary.json").read_bytes()
    prior = json.loads(protocol_bytes)
    summary = json.loads(summary_bytes)
    model = harness.make_brain("qualified", 0, ARGS)
    check_reference(prior, summary, harness.sha256(reference / "protocol.json"), model)
    load_fixture(fixture)
    root.mkdir(parents=True, exist_ok=False)
    harness.freeze_sources(root / "source", fixture)
    for name in PRODUCERS:
        shutil.copyfile(Path(__file__).with_name(name), root / "source" / name)
    control = root / "reference"
    control.mkdir()
    (control / "protocol.json").write_bytes(protocol_bytes)
    (control / "summary.json").write_bytes(summary_bytes)
    rng = np.random.default_rng(online.PEDAGOGY_SEED)
    orders = {}
    for count, positions in POSITIONS.items():
        order = []
        while len(order) < CAPS[count] + 1:
            order.extend(int(row) for row in rng.permutation(positions))
        orders[str(count)] = order[: CAPS[count] + 1]
    protocol = {
        "schema": "cadence-native-online-transfer-v2",
        "seed": 0,
        "learner_config": model.learner.config.to_dict(),
        "neuron_model": model.brain.neuron_model.to_dict(),
        "model_construction": MODEL_CONSTRUCTION,
        "initial_model_identity": model_identity(model),
        "library_sources": harness.sources(),
        "numpy": np.__version__,
        "dtype": "float64",
        "runtime_precision": runtime_precision(),
        "reference_runtime_precision": prior["runtime_precision"],
        "threads": {name: "1" for name in THREAD_VARIABLES},
        "producing_sources": {name: harness.sha256(root / "source" / name) for name in PRODUCERS},
        "fixture": {
            name: harness.sha256(root / "source/fixture" / name)
            for name in ("school.npz", "provenance.json")
        },
        "reference": {
            "protocol_sha256": harness.sha256(control / "protocol.json"),
            "summary_sha256": harness.sha256(control / "summary.json"),
        },
        "positions": POSITIONS,
        "stage_counts": list(CAPS),
        "orders_including_next_lesson": orders,
        "lesson_caps": CAPS,
        "query_every": {2: 32, 4: 32, 24: 96},
        "teaching_seconds": 300,
        "worker_seconds": 360,
        "admission_mib": 144,
        "output_cap_mib": 160,
        "selected_correct": {2: 2, 4: 4, 24: 18},
        "independent_correct": {"independent_train": 14, "development": 15},
        "selected_gate": "perfect2/4 before expansion; school>=18/24; positive founder gain; "
        "zero teaching/answer refusals; each selected stage fresh founder",
        "independent_gate": "read independentTRAIN18/development19 once after school; "
        ">=14/18 and>=15/19,zero refusals for this transfer qualification",
        "heldout": "sealed; no generation/read; five-founder native confirmation separate",
        "scope": "single-cue local contrast; zero biases/lateral, no memory/calibration; "
        "no label-family identification, biological fidelity or gameplay claim",
    }
    harness.write_json(root / "protocol.json", protocol)
    harness.write_json(root / "admission.json", {
        "protocol_sha256": harness.sha256(root / "protocol.json"),
        "source_fixture_reference_model_preflight": True,
    })
    # Compare serialized keys exactly as the independently launched worker will.
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol, worker_runtime=False)
    load_fixture(root / "source/fixture")
    return protocol


def lesson_passed(lesson, tolerance):
    """Agreement between two refusals is never a successful continuation."""
    phases = lesson.get("phases", {})
    if lesson.get("accepted") != 1 or lesson.get("failed") is not False or set(phases) != {
        "free", "nudged", "opposite"
    }:
        return False
    return all(
        bool(reading.get("qualified")) and all(reading["qualified"])
        and all(np.isfinite(reading.get(field, [])))
        and bool(reading.get(field)) and max(reading[field]) <= tolerance
        for reading in phases.values() for field in ("full_residual", "cache_defect")
    )


def continuation_passed(continuation, tolerance):
    return (
        continuation.get("continued_arrays_equal") is True
        and continuation.get("accepted_equal") is True
        and continuation.get("phase_payloads_retained") is True
        and len(continuation.get("lessons", [])) == 2
        and all(lesson_passed(row, tolerance) for row in continuation["lessons"])
    )


def work_census(report):
    lessons = report["lessons"] + report.get("continuation", {}).get("lessons", [])
    queries = [q["work"] for q in report["recall"]]
    queries += [report[name]["work"] for name in ("independent_train", "development")
                if name in report]
    return {
        "teacher_calls": len(lessons),
        "accepted_teacher_calls": sum(r["accepted"] for r in lessons),
        "refused_teacher_calls": sum(not r["accepted"] for r in lessons),
        "unqualified_teacher_calls": sum(
            not lesson_passed(row, report["tolerance"]) for row in lessons
        ),
        "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in lessons),
        "reported_phase_residual_checks": sum(r["reported_row_residual_checks"] for r in lessons),
        "phase_stagnation_checks": sum(
            r["report"].get("total_stagnation_checks", 0) for r in lessons
        ),
        "independent_phase_equation_cache_checks": sum(
            r["independent_residual_checks"] for r in lessons
        ),
        "query_calls": sum(q["calls"] for q in queries),
        "query_row_sweeps": sum(q["row_sweeps"] for q in queries),
        "reported_query_residual_checks": sum(q["reported_residual_checks"] for q in queries),
        "query_stagnation_checks": sum(q["reported_stagnation_checks"] for q in queries),
        "independent_query_equation_cache_checks": sum(
            q["independent_residual_checks"] for q in queries
        ),
        "unreported_phase_work": any(r.get("reported_residual_checks") is None for r in lessons),
        "unreported_query_work": any(
            q.get("unreported_residual_work", False)
            or q.get("unreported_stagnation_work", False) for q in queries
        ),
        "incomplete_operation": report.get("current_operation"),
        "memory_reads": 0, "memory_writes": 0,
    }


def write_receipt(path, payload):
    """A hard timeout must leave the preceding complete census readable."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    harness.write_json(temporary, payload)
    os.replace(temporary, path)


def persist_stage(folder, report, *, full=True):
    report["work"] = work_census(report)
    report["work_census_complete"] = bool(
        report.get("current_operation") is None
        and not report["work"]["unreported_phase_work"]
        and not report["work"]["unreported_query_work"]
    )
    # The small state census survives a hard kill between operations. Full
    # receipts are periodic; rewriting an ever-growing phase journal on every
    # lesson would add quadratic filesystem work to the instrument.
    write_receipt(folder / "state.json", stage_result(report))
    if full:
        write_receipt(folder / "receipt.json", report)


def stage_result(report):
    # The phase journal remains in its receipt; retain every acceptance gate and
    # independent score in the census without duplicating thousands of lessons.
    result = {name: value for name, value in report.items() if name != "lessons"}
    if "lessons" in report:
        result["completed_lessons"] = len(report["lessons"])
    return result


def stage(root, count, data, protocol, deadline):
    brain = harness.make_brain("qualified", protocol["seed"], ARGS)
    folder = root / f"selected-{count}"
    folder.mkdir()
    x, y = data["school_inputs"], data["school_labels"]
    key = str(count)
    positions = protocol["positions"][key]
    order = protocol["orders_including_next_lesson"][key]
    cap = protocol["lesson_caps"][key]
    report = {
        "examples": count, "tolerance": protocol["learner_config"]["tolerance"],
        "lessons": [], "recall": [], "passed": False, "selected_passed": False,
        "independent_passed": None, "independent_read": False,
        "continuation_passed": None, "status": "running", "complete": False,
        "current_operation": {"kind": "initial_checkpoint"},
    }
    persist_stage(folder, report)

    def operation(kind, **details):
        report["current_operation"] = {"kind": kind, **details}
        persist_stage(folder, report, full=False)

    def measured_recall(name, inputs, labels, lesson=None):
        operation("query", panel=name, rows=len(labels), lesson=lesson)
        reading = harness.free_recall(brain, inputs, labels)
        report["current_operation"] = None
        return reading

    try:
        report["initial_sha256"] = harness.sha256(brain.save(folder / "initial.npz"))
        report["recall"].append({"lesson": 0, **measured_recall(
            "selected", x[positions], y[positions], 0
        )})
        persist_stage(folder, report)
        for number, row in enumerate(order[:cap], 1):
            if (time.monotonic() >= deadline
                    or online.tree_bytes(root) >= protocol["admission_mib"] * 1024**2):
                report["status"] = "time_or_output_limit"
                break
            if any(r["refusals"] for r in report["recall"]):
                report["status"] = "refused_or_unqualified"
                break
            operation("lesson", lesson=number, school_row=row)
            lesson = {
                "lesson": number, "school_row": row,
                **continual.graph_lesson(brain, x[[row]], y[[row]], folder,
                                        f"phases-{number:04d}"),
            }
            report["lessons"].append(lesson)
            report["current_operation"] = None
            with (folder / "lessons.jsonl").open("a") as stream:
                stream.write(json.dumps(lesson) + "\n")
            persist_stage(folder, report, full=False)
            qualified = lesson_passed(lesson, report["tolerance"])
            if number % protocol["query_every"][key] == 0 or number == cap or not qualified:
                reading = {"lesson": number, **measured_recall(
                    "selected", x[positions], y[positions], number
                )}
                report["recall"].append(reading)
                operation("progress_checkpoint", lesson=number)
                report["progress_sha256"] = harness.sha256(brain.save(folder / "progress.npz"))
                report["current_operation"] = None
                if not qualified or any(r["refusals"] for r in report["recall"]):
                    report["status"] = "refused_or_unqualified"
                    break
                if (reading["correct"] >= protocol["selected_correct"][key]
                        and reading["correct"] > report["recall"][0]["correct"]):
                    report.update(selected_passed=True, status="selected_passed")
                    break
                persist_stage(folder, report)
        else:
            report["status"] = "lesson_limit"
        if report["recall"][-1]["lesson"] != len(report["lessons"]):
            report["recall"].append({"lesson": len(report["lessons"]), **measured_recall(
                "selected", x[positions], y[positions], len(report["lessons"])
            )})
        operation("final_checkpoint")
        report["final_sha256"] = harness.sha256(brain.save(folder / "final.npz"))
        report["current_operation"] = None
        if report["selected_passed"] and count == 24:
            for name in protocol["independent_correct"]:
                report[name] = measured_recall(name, data[name + "_inputs"], data[name + "_labels"])
                persist_stage(folder, report)
            report["independent_read"] = True
            report["independent_passed"] = all(
                report[name]["correct"] >= gate and not report[name]["refusals"]
                for name, gate in protocol["independent_correct"].items()
            )
        operation("continuation", teacher_calls=2, school_row=order[len(report["lessons"])])
        loaded = Brain.load(folder / "final.npz")
        next_row = order[len(report["lessons"]):len(report["lessons"]) + 1]
        report["continuation"] = continual.graph_checkpoint_continuation(
            brain, loaded, x[next_row], y[next_row], folder
        )
        report["current_operation"] = None
        report["continuation_passed"] = continuation_passed(
            report["continuation"], report["tolerance"]
        )
        report["passed"] = bool(
            report["selected_passed"] and report["continuation_passed"]
            and (count != 24 or report["independent_passed"])
        )
        if report["selected_passed"]:
            report["status"] = (
                "continuation_failed" if not report["continuation_passed"] else
                "independent_failed" if count == 24 and not report["independent_passed"] else
                "transfer_passed" if count == 24 else "selected_passed"
            )
        report["complete"] = True
    except BaseException as error:
        report.update(status="process_exception", passed=None,
                      error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        persist_stage(folder, report)
    return stage_result(report)


def initial_summary(root, protocol):
    return {
        "protocol_sha256": harness.sha256(root / "protocol.json"),
        "stage_denominator": len(protocol["stage_counts"]),
        "stages": [{"examples": count, "status": "scheduled", "passed": None,
                    "complete": False, "work_census_complete": True}
                   for count in protocol["stage_counts"]],
        "heldout_read": False, "independent_read": False,
        "passed": False, "transfer_passed": False, "complete": False,
        "status": "running", "current_stage": None, "process_failure": None,
    }


def finalize_summary(root, summary, protocol):
    """Enforce the final artifact cap, including the serialized census itself."""
    summary["independent_read"] = any(row.get("independent_read") for row in summary["stages"])
    summary["work_census_complete"] = all(
        row.get("work_census_complete", False) for row in summary["stages"]
    )
    summary["transfer_passed"] = summary["passed"] = bool(
        summary.get("process_failure") is None and summary.get("complete")
        and len(summary["stages"]) == len(protocol["stage_counts"])
        and all(row.get("passed") is True and row.get("selected_passed") is True
                and row.get("continuation_passed") is True
                and row.get("complete") is True and row.get("work_census_complete") is True
                and (row["examples"] != 24 or row.get("independent_passed") is True)
                for row in summary["stages"])
    )
    if summary.get("complete") and summary.get("process_failure") is None:
        summary["status"] = "transfer_passed" if summary["passed"] else "transfer_failed"
    write_receipt(root / "summary.json", summary)
    size = online.tree_bytes(root)
    if size > protocol["output_cap_mib"] * 1024**2:
        summary.update(status="output_cap_exceeded", passed=False, transfer_passed=False,
                       output_cap_exceeded=True)
    else:
        summary["output_cap_exceeded"] = False
    summary["output_bytes"] = size
    write_receipt(root / "summary.json", summary)
    # The last accounting write can itself cross the bound. Preserve evidence
    # and report the violation, rather than deleting attempted-work artifacts.
    size = online.tree_bytes(root)
    if size > protocol["output_cap_mib"] * 1024**2:
        summary.update(status="output_cap_exceeded", passed=False, transfer_passed=False,
                       output_cap_exceeded=True, output_bytes=size)
        write_receipt(root / "summary.json", summary)
    return summary


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol)
    data = load_fixture(root / "source/fixture")
    began = time.monotonic()
    summary = initial_summary(root, protocol)
    write_receipt(root / "summary.json", summary)
    try:
        for index, count in enumerate(protocol["stage_counts"]):
            summary["current_stage"] = count
            summary["stages"][index].update(status="running", work_census_complete=False)
            write_receipt(root / "summary.json", summary)
            result = stage(root, count, data, protocol, began + protocol["teaching_seconds"])
            summary["stages"][index] = result
            summary["current_stage"] = None
            write_receipt(root / "summary.json", summary)
            if not result["passed"]:
                for pending in summary["stages"][index + 1:]:
                    pending["status"] = "not_run_prior_failure"
                break
        summary.update(complete=True, status="transfer_passed" if all(
            row.get("passed") is True for row in summary["stages"]
        ) else "transfer_failed")
    except BaseException as error:
        summary["process_failure"] = {"kind": "worker_exception", "type": type(error).__name__,
                                      "message": str(error)}
        if summary["current_stage"] is not None:
            index = protocol["stage_counts"].index(summary["current_stage"])
            receipt = root / f"selected-{summary['current_stage']}" / "receipt.json"
            if receipt.exists():
                summary["stages"][index] = stage_result(json.loads(receipt.read_text()))
            else:
                summary["stages"][index].update(status="process_exception", passed=None)
            for pending in summary["stages"][index + 1:]:
                pending["status"] = "not_run_process_failure"
        summary["status"] = "process_failed"
        raise
    finally:
        summary["seconds"] = time.monotonic() - began
        finalize_summary(root, summary, protocol)
    return summary


def process_failure_summary(root, protocol, failure):
    path = root / "summary.json"
    try:
        summary = json.loads(path.read_text())
        if (summary["protocol_sha256"] != harness.sha256(root / "protocol.json")
                or [row["examples"] for row in summary["stages"]] != protocol["stage_counts"]):
            raise ValueError("worker summary differs from admitted census")
    except (OSError, ValueError, KeyError, TypeError):
        if path.exists():
            shutil.copyfile(path, root / "invalid-worker-summary.json")
        summary = initial_summary(root, protocol)
    for index, count in enumerate(protocol["stage_counts"]):
        row = summary["stages"][index]
        folder = root / f"selected-{count}"
        if row["status"] in ("running", "process_exception", "scheduled"):
            recovered = False
            for receipt in (folder / "state.json", folder / "receipt.json"):
                if not receipt.exists():
                    continue
                try:
                    known = json.loads(receipt.read_text())
                    if known["examples"] != count:
                        raise ValueError("current stage differs from census")
                    row = summary["stages"][index] = stage_result(known)
                    recovered = True
                    break
                except (OSError, ValueError, KeyError, TypeError):
                    row["state_readback_failed"] = True
            if recovered and row.get("complete") is True:
                continue
            if recovered or row["status"] != "scheduled" or folder.exists():
                row.update(status="process_interrupted", passed=None, complete=False,
                           work_census_complete=False)
                summary["current_stage"] = count
            else:
                row["status"] = "not_run_process_failure"
    summary.update(status="process_failed", passed=False, transfer_passed=False, complete=False,
                   process_failure=failure)
    return finalize_summary(root, summary, protocol)


def launch_worker(root, protocol):
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update(protocol["threads"])
    try:
        with (root / "worker.log").open("w") as stream:
            result = subprocess.run(
                [sys.executable, str(root / "source/native_online.py"),
                 "--worker", "--out", str(root)],
                env=env, stdout=stream, stderr=subprocess.STDOUT, check=False,
                timeout=protocol["worker_seconds"],
            )
    except subprocess.TimeoutExpired as error:
        process_failure_summary(root, protocol, {
            "kind": "worker_timeout", "seconds": error.timeout,
        })
        raise
    except BaseException as error:
        process_failure_summary(root, protocol, {
            "kind": "launcher_exception", "type": type(error).__name__, "message": str(error),
        })
        raise
    if result.returncode:
        process_failure_summary(root, protocol, {
            "kind": "worker_exit", "exit_code": result.returncode,
        })
        raise subprocess.CalledProcessError(result.returncode, result.args)
    try:
        summary = json.loads((root / "summary.json").read_text())
        if (summary["protocol_sha256"] != harness.sha256(root / "protocol.json")
                or [row["examples"] for row in summary["stages"]] != protocol["stage_counts"]
                or summary.get("complete") is not True):
            raise ValueError("worker exited without a complete admitted stage census")
    except (OSError, ValueError, KeyError, TypeError) as error:
        process_failure_summary(root, protocol, {
            "kind": "invalid_worker_summary", "type": type(error).__name__, "message": str(error),
        })
        raise
    summary["process_exit_code"] = result.returncode
    finalize_summary(root, summary, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixture")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
        return
    if args.reference is None:
        parser.error("source-bound confirmed reference required")
    protocol = prepare(root, args.fixture.resolve(), args.reference.resolve())
    launch_worker(root, protocol)
    print((root / "summary.json").read_text())


if __name__ == "__main__":
    main()
