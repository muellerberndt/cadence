"""Frozen endpoint-only memory-amplitude diagnostic; never trains or writes memory.

Prepare first, review its hashes, then explicitly launch the one complete panel.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import actual_outcome_chamber as chamber
import numpy as np

SCHEMA = "cadence-retained-memory-amplitude/1"
ENDPOINTS = ("newborn", "acquired", "routine", "revision", "restored")
AMPLITUDES = (1.0, 2.0)
SECONDS, OUTPUT_MIB = 120, 32


def panels(summary):
    complete = [row for row in summary["lives"] if row["complete"]]
    if len(complete) != 4 or any(row["seed"] != 0 for row in complete):
        raise ValueError("diagnostic requires the four retained complete founder0 lives")
    return [
        {
            "condition": row["condition"],
            "exposure": row["exposure"],
            "endpoint": name,
            "world": row["endpoints"][name]["intact"]["world"],
            "baseline": row["endpoints"][name]["intact"],
            "amplitude": amplitude,
            "checkpoint": f"reference/seed-0/{row['condition']}-{row['exposure']}/{name}.npz",
        }
        for row in complete
        for name in ENDPOINTS
        for amplitude in AMPLITUDES
    ]


def numeric_gate(reading):
    return bool(
        reading["qualified_free"]
        and reading["refused"] == 0
        and reading["unrun"] == 0
        and all(
            q["prototype_correct"] == 1
            and q["correct"] >= 9
            and (q["cue"] == 2 or q["obsolete"] <= 1)
            for q in reading["cues"]
        )
    )


def same_baseline(reading, expected):
    return (
        reading["qualified_free"] == expected["qualified_free"]
        and reading["refused"] == expected["refused"]
        and all(
            q["prototype_prediction"] == e["prototype_prediction"]
            and q["predictions"] == e["predictions"]
            and np.allclose(q["margins"], e["margins"], rtol=0, atol=1e-12)
            and np.allclose(q["memory_values"], e["memory_values"], rtol=0, atol=1e-12)
            for q, e in zip(reading["cues"], expected["cues"], strict=True)
        )
    )


def only_amplitude_changed(original, changed, amplitude):
    with (
        np.load(original, allow_pickle=False) as left,
        np.load(changed, allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files):
            return False
        for name in left.files:
            if name in ("generic", "brain"):  # "brain" groups the memories, the format since 0.85
                before, after = json.loads(left[name].item()), json.loads(right[name].item())
                store = before["memories"] if name == "brain" else before
                if store["hippocampus"]["amplitude"] != 1:
                    return False
                store["hippocampus"]["amplitude"] = amplitude
                if before != after:
                    return False
            elif not np.array_equal(left[name], right[name]):
                return False
    return True


def prepare(root, reference):
    began = time.monotonic()
    if root.exists():
        raise ValueError("diagnostic output must be a new directory")
    original = json.loads((reference / "protocol.json").read_text())
    summary = json.loads((reference / "summary.json").read_text())
    jobs = panels(summary)
    for name, pin in (original["source_files"] | original["admitted_artifacts"]).items():
        if chamber.digest(reference / name) != pin:
            raise ValueError("retained reference source/artifact changed: " + name)
    root.mkdir(parents=True)
    for name in original["source_files"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
    shutil.copyfile(Path(__file__), root / "source/amplitude_readout.py")
    names = {"protocol.json", "summary.json", "inputs-0.npz", "initial-0.npz"}
    names.update(row["checkpoint"].removeprefix("reference/") for row in jobs)
    reference_pins = {}
    for name in sorted(names):
        target = root / "reference" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
        reference_pins["reference/" + name] = chamber.digest(target)
    protocol = {
        "schema": SCHEMA,
        "panels": jobs,
        "amplitudes": list(AMPLITUDES),
        "runtime": original["runtime"],
        "source_files": original["source_files"]
        | {"source/amplitude_readout.py": chamber.digest(root / "source/amplitude_readout.py")},
        "reference_files": reference_pins,
        "reference_origin": str(reference),
        "worker_seconds": SECONDS,
        "output_cap_mib": OUTPUT_MIB,
        "planned_panels": 40,
        "planned_query_rows": 1320,
        "planned_free_calls": 200,
        "free_row_sweep_bound": 1351680,
        "gates": original["gates"],
        "free_steps": 1024,
        "tolerance": 0.003,
        "teacher_calls": 0,
        "actual_outcomes": 0,
        "store_writes": 0,
        "changed_gene": "hippocampus.amplitude only; 1 exact baseline, 2 fixed counterfactual",
        "cold_semantics": (
            "Brain.reset plus hippocampus.reset before each original <=8-row batch; "
            "C preserved, F cleared"
        ),
        "baseline_gate": (
            "all40 panels complete and qualified; amplitude1 predictions/read values/margins "
            "match archived panels"
        ),
        "diagnostic_gate": (
            "amplitude2 all acquired/routine/revision/restored panels meet unchanged "
            "prototype+9/10/obsolete<=1 gates; rare loss<=1 versus archived endpoint"
        ),
        "equation": (
            "unchanged original NeuralGraph equilibrium; recalled memory amplitude scales "
            "drive through existing declared post ports"
        ),
        "scope": (
            "retained endpoint readout counterfactual only; "
            "no acquisition/confirmation/default/promotion claim"
        ),
        "campaign_failure_preserved": {
            k: summary[k]
            for k in ("complete", "passed", "status", "unrun_batches", "output_cap_exceeded")
        },
    }
    chamber.atomic_json(root / "protocol.json", protocol)
    chamber.atomic_json(
        root / "admission.json",
        {
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "preparation_seconds": time.monotonic() - began,
            "final_admission_io_allowance_seconds": 1,
        },
    )
    return protocol


def preflight(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol["schema"] != SCHEMA
        or protocol["worker_seconds"] != SECONDS
        or protocol["output_cap_mib"] != OUTPUT_MIB
        or protocol["runtime"] != chamber.runtime()
        or protocol["amplitudes"] != list(AMPLITUDES)
    ):
        raise ValueError("frozen diagnostic configuration/runtime differs")
    admission = json.loads((root / "admission.json").read_text())
    if admission["protocol_sha256"] != chamber.digest(root / "protocol.json"):
        raise ValueError("frozen diagnostic protocol changed")
    for name, pin in (protocol["source_files"] | protocol["reference_files"]).items():
        if chamber.digest(root / name) != pin:
            raise ValueError("frozen diagnostic source/reference changed: " + name)
    package = Path(chamber.cadence.__file__).resolve().parent
    if package != (root / "source/library/cadence").resolve():
        raise ValueError("diagnostic must import its frozen library")
    if chamber.digest(Path(__file__)) != protocol["source_files"]["source/amplitude_readout.py"]:
        raise ValueError("diagnostic must execute its frozen producer")
    if (
        chamber.digest(Path(chamber.__file__))
        != protocol["source_files"]["source/actual_outcome_chamber.py"]
    ):
        raise ValueError("imported chamber producer differs")
    expected = panels(json.loads((root / "reference/summary.json").read_text()))
    if protocol["panels"] != expected or len(expected) != 40:
        raise ValueError("frozen complete query denominator differs")
    if any(
        protocol[name] != value
        for name, value in {
            "planned_panels": 40,
            "planned_query_rows": 1320,
            "planned_free_calls": 200,
            "free_row_sweep_bound": 1351680,
            "free_steps": 1024,
            "tolerance": 0.003,
            "teacher_calls": 0,
            "actual_outcomes": 0,
            "store_writes": 0,
        }.items()
    ):
        raise ValueError("frozen query/resource contract differs")
    if chamber.tree_bytes(root) >= OUTPUT_MIB * 1024**2 - 1024**2:
        raise ValueError("prepared output has no resource reserve")
    return protocol


class Journal:
    def __init__(self, root, protocol, began):
        self.root, self.protocol, self.began = root, protocol, began
        self.summary = {
            "schema": SCHEMA,
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "complete": False,
            "passed": False,
            "current_operation": None,
            "unknown_current_work": False,
            "work": {},
            "panels": [{**p, "status": "unrun"} for p in protocol["panels"]],
        }
        self.persist()

    def bounds(self):
        if time.monotonic() - self.began >= SECONDS:
            raise chamber.ResourceLimit("diagnostic total wall bound reached")
        if chamber.tree_bytes(self.root) >= OUTPUT_MIB * 1024**2 - 1024**2:
            raise chamber.ResourceLimit("diagnostic output reserve reached")

    def persist(self):
        self.summary["seconds"] = time.monotonic() - self.began
        chamber.atomic_json(self.root / "summary.json", self.summary)

    def begin(self, operation, context):
        self.bounds()
        self.summary.update(
            current_operation={"kind": operation, **context}, unknown_current_work=True
        )
        self.persist()

    def completed(self, operation, context, body, work=None):
        for name, value in (work or {}).items():
            self.summary["work"][name] = self.summary["work"].get(name, 0) + value
        with gzip.open(self.root / "journal.jsonl.gz", "at") as stream:
            stream.write(
                chamber.canonical_json(
                    {"operation": operation, "context": context, "body": body, "work": work or {}}
                )
                + "\n"
            )
        self.summary.update(current_operation=None, unknown_current_work=False)
        self.persist()
        self.bounds()

    def load(self, path):
        self.begin("checkpoint_load", {"path": str(path.relative_to(self.root))})
        brain = chamber.Brain.load(path)
        self.completed("checkpoint_load", {}, {}, {"checkpoint_reads": 1})
        return brain

    def save(self, brain, path):
        context = {"path": str(path.relative_to(self.root))}
        self.begin("checkpoint_save", context)
        path.parent.mkdir(parents=True, exist_ok=True)
        began = time.monotonic()
        brain.save(path)
        self.completed("checkpoint_save", context, {"sha256": chamber.digest(path)},
                       {"checkpoint_writes": 1, "io_seconds": time.monotonic() - began})


def worker(root):
    began = time.monotonic()
    protocol = preflight(root)
    admission = json.loads((root / "admission.json").read_text())
    reserved = admission["preparation_seconds"] + admission["final_admission_io_allowance_seconds"]
    journal = Journal(root, protocol, began - reserved)
    arrays = chamber.inputs.load(root / "reference/inputs-0.npz", 0)
    try:
        for index, row in enumerate(journal.summary["panels"]):
            row["status"] = "running"
            brain = journal.load(root / row["checkpoint"])
            before = chamber.memory_arrays(brain.hippocampus)
            brain.hippocampus.amplitude = row["amplitude"]
            # Existing cold_probe loads a complete clone; serialize only the one declared gene.
            anchor = root / "query-clones" / f"panel-{index}.npz"
            journal.save(brain, anchor)
            if not only_amplitude_changed(root / row["checkpoint"], anchor, row["amplitude"]):
                raise AssertionError("complete clone changed beyond the declared amplitude gene")
            if any(
                not np.array_equal(before[k], chamber.memory_arrays(brain.hippocampus)[k])
                for k in before
            ):
                raise AssertionError("amplitude assignment changed stored observations")
            reading = chamber.cold_probe(
                journal,
                anchor,
                root / "reference/initial-0.npz",
                arrays[row["condition"] + "/probes"],
                arrays[row["condition"] + "/prototypes"],
                row["world"],
                {"panel": index, "amplitude": row["amplitude"]},
            )
            reading = json.loads(chamber.canonical_json(reading))
            row.update(
                status="completed",
                reading=reading,
                unchanged_numeric_gate=numeric_gate(reading),
                exact_baseline=same_baseline(reading, row["baseline"])
                if row["amplitude"] == 1
                else None,
                rare_loss=row["baseline"]["cues"][2]["correct"] - reading["cues"][2]["correct"],
                scaled_memory_read=all(
                    np.allclose(q["memory_values"],
                                row["amplitude"] * np.asarray(e["memory_values"]),
                                rtol=0, atol=1e-12)
                    for q, e in zip(reading["cues"], row["baseline"]["cues"], strict=True)
                ),
            )
            journal.completed(
                "panel_result", {"panel": index}, row, {"query_panels": 1}
            )
            print(
                chamber.canonical_json(
                    {"completed_panels": index + 1, "seconds": time.monotonic() - journal.began}
                ),
                flush=True,
            )
        rows = journal.summary["panels"]
        baseline = all(
            r["exact_baseline"] and r["reading"]["qualified_free"]
            for r in rows
            if r["amplitude"] == 1
        )
        diagnostic = all(
            r["unchanged_numeric_gate"] and r["rare_loss"] <= 1 and r["scaled_memory_read"]
            for r in rows
            if r["amplitude"] == 2 and r["endpoint"] != "newborn"
        )
        journal.summary.update(
            complete=True,
            baseline_passed=baseline,
            diagnostic_passed=diagnostic,
            passed=baseline and diagnostic,
            acquisition_claim=False,
            store_writes=0,
            actual_outcomes=0,
        )
    except BaseException as error:
        journal.summary.update(
            passed=False,
            complete=False,
            process_failure={"type": type(error).__name__, "message": str(error)},
        )
        for row in journal.summary["panels"]:
            if row["status"] == "running":
                row["status"] = "interrupted"
        raise
    finally:
        journal.persist()
    journal.bounds()


def launch(root):
    began = time.monotonic()
    protocol = json.loads((root / "protocol.json").read_text())
    admission = json.loads((root / "admission.json").read_text())
    reserved = admission["preparation_seconds"] + admission["final_admission_io_allowance_seconds"]
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update({name: "1" for name in chamber.THREADS})
    try:
        with (root / "worker.log").open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "source/amplitude_readout.py"),
                    "--worker",
                    "--out",
                    str(root),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=SECONDS - reserved,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"diagnostic worker exited {result.returncode}")
    except BaseException as error:
        path = root / "summary.json"
        summary = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "schema": SCHEMA,
                "protocol_sha256": chamber.digest(root / "protocol.json"),
                "panels": [{**row, "status": "unrun"} for row in protocol["panels"]],
            }
        )
        summary.update(
            complete=False,
            passed=False,
            process_failure={"type": type(error).__name__, "message": str(error)},
            unknown_current_work=True
            if not path.exists()
            else summary.get("unknown_current_work", True),
        )
        for row in summary["panels"]:
            if row["status"] == "running":
                row["status"] = "interrupted"
        chamber.atomic_json(path, summary)
        raise
    if (
        time.monotonic() - began + reserved >= SECONDS
        or chamber.tree_bytes(root) >= OUTPUT_MIB * 1024**2
    ):
        summary = json.loads((root / "summary.json").read_text())
        summary.update(passed=False, complete=False, process_failure={"type": "final_resource_cap"})
        chamber.atomic_json(root / "summary.json", summary)
        raise chamber.ResourceLimit("diagnostic final wall/output cap exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--worker", action="store_true")
    choice.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.prepare:
        prepare(root, args.reference.resolve())
        print(
            chamber.canonical_json(
                {
                    "prepared_no_queries": True,
                    "protocol_sha256": chamber.digest(root / "protocol.json"),
                }
            )
        )
    elif args.worker:
        worker(root)
    else:
        launch(root)


if __name__ == "__main__":
    main()
