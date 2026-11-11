"""One fresh24 extension of the passed source-frozen one-module native school.

The physical graph and local teacher are identical to the passed2/4 control.
Only the predeclared original24-row schedule and conditional independent
panels are admitted. Independent arrays are decoded once per panel after the
selected18/24 screen; heldout arrays remain sealed. No whole-issue claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import single_module_native as school

from cadence import Brain

native = school.native
SCHEMA = "cadence-native-single-module-24-extension-v1"
BOUNDS = {
    "teaching_seconds": 300,
    "worker_seconds": 360,
    "admission_mib": 144,
    "output_cap_mib": 160,
}
FIXED = {
    "schema": SCHEMA,
    "seed": 0,
    "stage_counts": [24],
    "model_construction": school.CONSTRUCTION,
    "lesson_caps": {"24": 4096},
    "positions": {"24": list(range(24))},
    "query_every": {"24": 96},
    "selected_correct": {"24": 18},
    "independent_correct": {"independent_train": 14, "development": 15},
    **BOUNDS,
}


def check_protocol(protocol):
    if any(protocol.get(key) != value for key, value in FIXED.items()):
        raise ValueError("native24 extension changed its source recipe/gate/bound")
    order = protocol["orders_including_next_lesson"]["24"]
    if len(order) != 4097 or any(type(row) is not int or row not in range(24) for row in order):
        raise ValueError("native24 extension order differs")


class SchoolPanels(dict):
    """Lazy sealed panels; no independent label is materialized before admission."""

    def __init__(self, root, data):
        super().__init__(data)
        self.root = root
        self.decoded = {}

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        names = {
            name + suffix: name
            for name in ("independent_train", "development")
            for suffix in ("_inputs", "_labels")
        }
        if key not in names:
            raise KeyError("this native24 job never decodes the requested panel: " + key)
        panel = names[key]
        state = json.loads((self.root / "selected-24/state.json").read_text())
        readings = state["recall"]
        if (
            state.get("selected_passed") is not True
            or readings[-1]["correct"] < 18
            or readings[-1]["correct"] <= readings[0]["correct"]
            or any(row["refusals"] for row in readings)
            or state["work"]["refused_teacher_calls"]
            or state["work"]["unqualified_teacher_calls"]
            or not (self.root / "selected-24/final.npz").exists()
        ):
            raise ValueError("independent movie panel requested before native24 screen passed")
        fixture = self.root / "source/fixture"
        provenance = json.loads((fixture / "provenance.json").read_text())
        count = {"independent_train": 18, "development": 19}[panel]
        with np.load(fixture / "school.npz", allow_pickle=False) as archive:
            x, y = archive[panel + "_inputs"], archive[panel + "_labels"]
        identities = provenance["panels"][panel]
        if (
            x.shape != (count, 650)
            or x.dtype != np.dtype("float64")
            or y.shape != (count,)
            or not np.issubdtype(y.dtype, np.integer)
            or not np.isfinite(x).all()
            or np.any(y < 0)
            or np.any(y >= 36)
            or len(identities) != count
        ):
            raise ValueError("independent movie panel census differs: " + panel)
        for row, label, identity in zip(x, y, identities, strict=True):
            if (
                native.harness.row_hash(row) != identity["input_sha256"]
                or int(label) != identity["label"]
            ):
                raise ValueError("independent movie panel identity differs: " + panel)
        self[panel + "_inputs"], self[panel + "_labels"] = x, y
        self.decoded[panel] = {
            "decodes": 1,
            "rows": count,
            "selected_checkpoint_sha256": native.harness.sha256(
                self.root / "selected-24/final.npz"
            ),
            "selected_score": readings[-1]["correct"],
            "selected_lessons": state["completed_lessons"],
            "labels_used_for_teaching": False,
        }
        native.harness.write_json(
            self.root / "independent-read-census.json",
            {
                "panels": self.decoded,
                "heldout_decoded": False,
                "scope": (
                    "arrays decoded on first conditional panel request; "
                    "query work separately retained"
                ),
            },
        )
        return dict.__getitem__(self, key)


def load_fixture(folder):
    return SchoolPanels(folder.parents[1], school.load_fixture(folder))


def check_school(root, protocol):
    capsule = root / "passed-school"
    for name, expected in protocol["passed_school_capsule"].items():
        if native.harness.sha256(capsule / name) != expected:
            raise ValueError("passed native2/4 school capsule changed: " + name)
    prior = json.loads((capsule / "protocol.json").read_text())
    school.check_sources(capsule, prior)
    summary = json.loads((capsule / "summary.json").read_text())
    if (
        summary["protocol_sha256"] != protocol["passed_school_capsule"]["protocol.json"]
        or not summary["passed"]
        or not summary["complete"]
        or summary["process_exit_code"] != 0
        or summary["independent_read"]
        or summary["heldout_read"]
        or summary["whole_issue_passed"]
        or summary["native_24_gate_passed"]
        or [
            (r["examples"], r["completed_lessons"], r["recall"][-1]["correct"])
            for r in summary["stages"]
        ]
        != [(2, 64, 2), (4, 512, 4)]
    ):
        raise ValueError("native24 extension requires the actual passed2/4 school")
    for count, stage in zip((2, 4), summary["stages"], strict=True):
        folder = capsule / f"selected-{count}"
        receipt = json.loads((folder / "receipt.json").read_text())
        if (
            native.stage_result(receipt) != stage
            or native.work_census(receipt) != receipt["work"]
            or not receipt["passed"]
            or not receipt["continuation_passed"]
            or not receipt["work_census_complete"]
            or receipt["initial_sha256"] != native.harness.sha256(folder / "initial.npz")
            or receipt["final_sha256"] != native.harness.sha256(folder / "final.npz")
            or native.model_identity(Brain.load(folder / "initial.npz"))
            != protocol["initial_model_identity"]
        ):
            raise ValueError("native24 passed-school complete receipt/initial identity differs")
    audit = json.loads((capsule / "audit/result.json").read_text())
    audit_job = json.loads((capsule / "audit/job-protocol.json").read_text())
    if (
        not audit["verified"]
        or audit["job_protocol_sha256"]
        != native.harness.sha256(capsule / "audit/job-protocol.json")
        or audit_job["script_sha256"] != native.harness.sha256(capsule / "audit/script.py")
        or audit_job["protocol_sha256"] != protocol["passed_school_capsule"]["protocol.json"]
        or audit_job["summary_sha256"] != protocol["passed_school_capsule"]["summary.json"]
        or any(
            audit["work"][key]
            for key in (
                "fresh_teacher_presentations",
                "fresh_query_rows",
                "fresh_equilibrium_solves",
            )
        )
    ):
        raise ValueError("native24 reference school retained-arithmetic audit differs")
    original = json.loads((capsule / "control/protocol.json").read_text())
    if (
        protocol["orders_including_next_lesson"]["24"]
        != original["orders_including_next_lesson"]["24"]
    ):
        raise ValueError("native24 extension changed the original fixed movie schedule")
    for key in (
        "learner_config",
        "neuron_model",
        "model_construction",
        "initial_model_identity",
        "library_sources",
        "runtime_precision",
        "fixture",
    ):
        if protocol[key] != prior[key]:
            raise ValueError("native24 extension changed the passed native school: " + key)


def check_sources(root, protocol, *, worker_runtime=True):
    check_protocol(protocol)
    for folder, pins in (
        ("source/library/cadence", protocol["library_sources"]),
        ("source", protocol["producing_sources"]),
        ("source/fixture", protocol["fixture"]),
    ):
        for name, expected in pins.items():
            if native.harness.sha256(root / folder / name) != expected:
                raise ValueError("native24 extension frozen source/data changed: " + name)
    for name, expected in protocol["producing_sources"].items():
        imported = sys.modules.get(Path(name).stem)
        path = Path(imported.__file__) if imported else Path(__file__).with_name(name)
        if native.harness.sha256(path) != expected:
            raise ValueError("native24 imported producer differs: " + name)
    if native.harness.sources() != protocol["library_sources"]:
        raise ValueError("native24 imported library differs from passed school")
    if native.runtime_precision() != protocol["runtime_precision"]:
        raise ValueError("native24 runtime differs from passed school")
    if worker_runtime and any(os.environ.get(k) != v for k, v in protocol["threads"].items()):
        raise ValueError("native24 worker thread settings differ")
    check_school(root, protocol)
    model = school.make_brain("qualified", 0, native.ARGS)
    if native.model_identity(model) != protocol["initial_model_identity"]:
        raise ValueError("native24 founder physical graph/genes differ")
    job = json.loads((root / "job-protocol.json").read_text())
    if (
        job["protocol_sha256"] != native.harness.sha256(root / "protocol.json")
        or job["bounds"] != BOUNDS
        or job["interpreter_sha256"]
        != hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        or native.harness.sha256(root / "prepared-founder.npz")
        != protocol["prepared_founder_sha256"]
        or native.model_identity(Brain.load(root / "prepared-founder.npz"))
        != protocol["initial_model_identity"]
    ):
        raise ValueError("native24 reviewed job/newborn changed")


def prepare(root, prior_root, audit_root):
    if root.exists():
        raise FileExistsError(root)
    prior = json.loads((prior_root / "protocol.json").read_text())
    school.check_sources(prior_root, prior, worker_runtime=False)
    root.mkdir(parents=True, exist_ok=False)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(prior_root / "source", root / "source", ignore=ignored)
    shutil.copyfile(Path(__file__), root / "source/single_module_native_24.py")
    capsule = root / "passed-school"
    capsule.mkdir()
    for name in ("source", "control", "reference", "prior-centered"):
        shutil.copytree(prior_root / name, capsule / name, ignore=ignored)
    for name in (
        "protocol.json",
        "summary.json",
        "job-protocol.json",
        "admission.json",
        "prepared-founder.npz",
    ):
        shutil.copyfile(prior_root / name, capsule / name)
    for count in (2, 4):
        (capsule / f"selected-{count}").mkdir()
        for name in (
            "receipt.json",
            "initial.npz",
            "final.npz",
            "continued-original.npz",
            "continued-loaded.npz",
        ):
            shutil.copyfile(
                prior_root / f"selected-{count}" / name, capsule / f"selected-{count}" / name
            )
    (capsule / "audit").mkdir()
    for name in ("job-protocol.json", "result.json"):
        shutil.copyfile(audit_root / name, capsule / "audit" / name)
    shutil.copyfile(
        audit_root.parent / "verification-sources/single_module_retained_audit.py",
        capsule / "audit/script.py",
    )
    control = json.loads((capsule / "control/protocol.json").read_text())
    model = school.make_brain("qualified", 0, native.ARGS)
    newborn = model.save(root / "prepared-founder.npz")
    protocol = {
        **FIXED,
        **{
            key: prior[key]
            for key in (
                "learner_config",
                "neuron_model",
                "initial_model_identity",
                "construction_proof",
                "library_sources",
                "runtime_precision",
                "threads",
                "fixture",
            )
        },
        "producing_sources": {
            **prior["producing_sources"],
            "single_module_native_24.py": native.harness.sha256(
                root / "source/single_module_native_24.py"
            ),
        },
        "passed_school_capsule": {
            str(path.relative_to(capsule)): native.harness.sha256(path)
            for path in sorted(capsule.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        },
        "prepared_founder_sha256": native.harness.sha256(newborn),
        "orders_including_next_lesson": {"24": control["orders_including_next_lesson"]["24"]},
        "selected_gate": (
            "original school>=18/24, positive newborn gain, zero teacher/query refusals; "
            "fresh seed0 brain identical to passed2/4 construction; full continuation"
        ),
        "independent_gate": (
            "only after selected screen, one independentTRAIN18 and one development19 query panel; "
            ">=14/18 and>=15/19, zero refusals; no further teaching"
        ),
        "fixture_decode": (
            "school TRAIN only before teaching; independent arrays lazily decoded "
            "after screen; heldout sealed"
        ),
        "heldout": "never decoded or queried in this extension",
        "scope": (
            "one native movie transfer of existing local equilibrium teacher; "
            "original24/independent qualification only; no default change, "
            "five-founder native retention, continuing-animal or whole-issue claim"
        ),
        "prior_cost_scope": (
            "passed2/4 and all historical negatives/audits remain separately charged; "
            "no work amortization claim"
        ),
        "failure_policy": (
            "single fixed schedule/cap/architecture; retain failure; no retry or tuning"
        ),
    }
    native.harness.write_json(root / "protocol.json", protocol)
    native.harness.write_json(
        root / "job-protocol.json",
        {
            "schema": "cadence-native-single-module-24-job-v1",
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
    # Preparation has no thread-dependent solve; the frozen school guard still
    # checks source/model/runtime, using its declared single-thread admission.
    prior_env = {name: os.environ.get(name) for name in native.THREAD_VARIABLES}
    os.environ.update(protocol["threads"])
    try:
        check_sources(root, protocol, worker_runtime=False)
    finally:
        for name, value in prior_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    native.harness.write_json(
        root / "admission.json",
        {
            "protocol_sha256": native.harness.sha256(root / "protocol.json"),
            "status": "prepared_not_executed",
            "source_reference_newborn_preflight": True,
            "teacher_calls": 0,
            "query_rows": 0,
            "independent_decoded": False,
            "heldout_decoded": False,
        },
    )
    return protocol


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol)
    with (root / "execution-started.json").open("x") as stream:
        stream.write(native.harness.sha256(root / "protocol.json") + "\n")
    native.harness.make_brain = school.make_brain
    native.check_sources = check_sources
    native.load_fixture = load_fixture
    try:
        result = native.worker(root)
    finally:
        path = root / "summary.json"
        if path.exists():
            result = json.loads(path.read_text())
            stage = result["stages"][0]
            census = root / "independent-read-census.json"
            result.update(
                native_24_gate_passed=stage.get("selected_passed") is True,
                independent_gate_passed=stage.get("independent_passed") is True,
                whole_issue_passed=False,
                scope=protocol["scope"],
                heldout_decoded=False,
                independent_data_decoded=(
                    json.loads(census.read_text())["panels"] if census.exists() else {}
                ),
            )
            native.finalize_summary(root, result, protocol)
    return result


def launch(root):
    protocol = json.loads((root / "protocol.json").read_text())
    # Capsule guards validate both imported producer source and original runtime.
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update(protocol["threads"])
    old = {name: os.environ.get(name) for name in native.THREAD_VARIABLES}
    os.environ.update(protocol["threads"])
    try:
        check_sources(root, protocol, worker_runtime=False)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if (root / "execution-started.json").exists():
        raise ValueError("native24 extension already attempted")
    with (root / "worker.log").open("w") as stream:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "source/single_module_native_24.py"),
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
            native.process_failure_summary(
                root, protocol, {"kind": "worker_timeout", "seconds": error.timeout}
            )
            raise
    if completed.returncode:
        native.process_failure_summary(
            root, protocol, {"kind": "worker_exit", "exit_code": completed.returncode}
        )
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    summary = json.loads((root / "summary.json").read_text())
    if not summary.get("complete"):
        raise ValueError("native24 worker exited without complete census")
    summary["process_exit_code"] = completed.returncode
    native.finalize_summary(root, summary, protocol)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--school", type=Path)
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
                    "passed": result["passed"],
                    "native_24_gate_passed": result["native_24_gate_passed"],
                    "independent_gate_passed": result["independent_gate_passed"],
                    "status": result["status"],
                    "seconds": result["seconds"],
                }
            )
        )
    elif args.prepare_only and args.school and args.audit:
        prepare(root, args.school.resolve(), args.audit.resolve())
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
        parser.error("prepare-only requires school/audit; execution awaits separate review")


if __name__ == "__main__":
    main()
