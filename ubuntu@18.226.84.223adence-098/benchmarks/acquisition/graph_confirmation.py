"""Five frozen founders: durable graph acquisition, rehearsal, recall and custody.

Each continuing brain acquires old four relations, then learns the other twenty
with charged old rehearsal. Only qualified free/nudged graph contrasts learn.
The hippocampal store and working trace never contribute to these answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import relation_development as development
import relations
import run as harness

from cadence import Brain, LearningPhaseError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_brain(seed):
    args = SimpleNamespace(phase_steps=4096, tolerance=0.003)
    return development.make_brain("canonical", seed, args)


def recall(model, panel, folder, name):
    original = model._equilibrate
    previous_override = model.__dict__.get("_equilibrate")
    row = 0
    refused_states = []

    def capture(*args, **kwargs):
        nonlocal row
        phase = original(*args, **kwargs)
        if not np.all(phase.qualified):
            path = folder / f"refused-{name}-{row:04d}.npz"
            np.savez_compressed(
                path,
                v=phase.state.v,
                activation=phase.state.activation,
                adaptation=phase.state.adaptation,
            )
            refused_states.append({"file": path.name, "sha256": harness.sha256(path)})
        row += 1
        return phase

    model._equilibrate = capture
    try:
        result = harness.free_recall(
            model, panel["inputs"], panel["labels"], folder=folder, name=name
        )
    finally:
        if previous_override is None:
            del model._equilibrate
        else:
            model._equilibrate = previous_override
    result["refused_states"] = refused_states
    result["family_credits"] = development.credited_families(panel, result["predictions"])
    result["family_count"] = len(np.unique(panel["families"]))
    if len(panel["families"]) == 48:
        for group_name, families in (
            ("old", relations.OLD_FAMILIES),
            ("new", relations.NEW_FAMILIES),
        ):
            keep = np.isin(panel["families"], families)
            part = {key: value[keep] for key, value in panel.items()}
            result[group_name + "_family_credits"] = development.credited_families(
                part, np.asarray(result["predictions"])[keep]
            )
    harness.write_json(folder / (name + ".json"), result)
    return result


def admission(model, panel, folder, number):
    graph = model.brain
    weights = (
        graph.neuron_model.gain
        * graph.connectome.count
        * graph.efficacy
        * np.exp(graph.log_gain[graph.connectome.pre])
    )
    bias = graph.bias.copy()
    drive = model.stimulus(panel["inputs"], memory=False)
    phases, failed = {}, False
    try:
        state, report = model.learner.step(drive, panel["labels"])
        phases = {"free": state.free, "nudged": state.nudged, "opposite": state.opposite}
    except LearningPhaseError as error:
        phases = {name: phase.state for name, phase in error.phases.items()}
        report, failed = error.report, True
    readings = harness.phase_readback(model, drive, panel["labels"], phases, weights, bias)
    valid = (
        not failed
        and len(readings) == 3
        and all(
            all(reading["qualified"]) and max(reading["cache_defect"]) <= 0.003
            for reading in readings.values()
        )
    )
    for name in readings:
        readings[name]["steps"] = int(report[name + "_steps"])
    path = folder / f"phases-{number:04d}.npz"
    return {
        "update": number,
        "report": harness.json_value(report),
        "phases": readings,
        "phase_sha256": harness.phase_file(path, phases),
        "failed": failed,
        "independent_qualification_passed": valid,
        "phase_row_sweeps": len(panel["labels"]) * int(report["total_steps"]),
        "reported_phase_row_residual_checks": len(panel["labels"])
        * int(report["total_residual_checks"]),
        "independent_phase_row_residual_checks": len(panel["labels"]) * len(readings),
    }


def train_stage(model, train, dev, folder, *, mixed, deadline, args):
    folder.mkdir()
    model.save(folder / "initial.npz")
    receipt = {
        "stage": "mixed" if mixed else "old4",
        "status": "running",
        "initial_sha256": harness.sha256(folder / "initial.npz"),
        "labels": train["labels"].tolist(),
        "families": train["families"].tolist(),
        "input_sha256": development.array_hash(train["inputs"]),
        "learner_config": model.learner.config.to_dict(),
        "starting_updates": model.learner.updates,
        "attempted_updates": 0,
        "accepted_updates": 0,
        "accepted_row_exposures": 0,
        "updates": [],
        "recall": [],
    }
    receipt["recall"].append(
        {
            "update": 0,
            "train": recall(model, train, folder, "train-0000"),
            "development": recall(model, dev, folder, "development-0000"),
        }
    )
    receipt["refused_recall_rows"] = sum(
        q["refusals"] for q in (receipt["recall"][0]["train"], receipt["recall"][0]["development"])
    )
    for number in range(1, 513):
        if receipt["refused_recall_rows"]:
            receipt["status"] = "refused_recall"
            break
        if time.monotonic() >= deadline:
            receipt["status"] = "time_limit"
            break
        # Reserve room for the final bounded states/readback, including a refused phase.
        limit = harness.elapsed_limit(float("inf"), args.out, (args.max_output_mib - 8) * 1024**2)
        if limit:
            receipt["status"] = limit
            break
        receipt["attempted_updates"] += 1
        harness.write_json(folder / "receipt.json", receipt)
        record = admission(model, train, folder, number)
        receipt["updates"].append(record)
        accepted = int(record["report"]["accepted"])
        receipt["accepted_updates"] += accepted
        receipt["accepted_row_exposures"] += accepted * len(train["labels"])
        if (
            number % 8 == 0
            or number == 512
            or record["failed"]
            or not record["independent_qualification_passed"]
        ):
            reading = {
                "update": number,
                "train": recall(model, train, folder, f"train-{number:04d}"),
                "development": recall(model, dev, folder, f"development-{number:04d}"),
            }
            receipt["recall"].append(reading)
            receipt["refused_recall_rows"] += (
                reading["train"]["refusals"] + reading["development"]["refusals"]
            )
            if receipt["refused_recall_rows"]:
                receipt["status"] = "refused_recall"
                break
            if not record["independent_qualification_passed"]:
                receipt["status"] = (
                    "refused_learning" if record["failed"] else "qualification_violation"
                )
                break
            passed = (
                reading["train"]["refusals"] == 0
                and reading["development"]["refusals"] == 0
                and reading["train"]["family_credits"] >= (18 if mixed else 4)
                and reading["development"]["correct"] >= (36 if mixed else 8)
                and (
                    not mixed
                    or (
                        number >= 128
                        and reading["development"]["old_family_credits"] >= 3
                        and reading["development"]["new_family_credits"] >= 15
                    )
                )
                and reading["train"]["correct"] > receipt["recall"][0]["train"]["correct"]
            )
            if passed:
                receipt["status"] = "development_passed"
                break
        harness.write_json(folder / "receipt.json", receipt)
    else:
        receipt["status"] = "update_limit"
    if receipt["recall"][-1]["update"] != receipt["attempted_updates"]:
        number = receipt["attempted_updates"]
        receipt["recall"].append(
            {
                "update": number,
                "train": recall(model, train, folder, "train-final"),
                "development": recall(model, dev, folder, "development-final"),
            }
        )
        receipt["refused_recall_rows"] += sum(
            q["refusals"]
            for q in (receipt["recall"][-1]["train"], receipt["recall"][-1]["development"])
        )
    model.save(folder / "final.npz")
    receipt["final_sha256"] = harness.sha256(folder / "final.npz")
    receipt["passed"] = receipt["status"] == "development_passed"
    # A time/iteration endpoint can pass the same frozen gate; no earlier prefix is selected.
    final = receipt["recall"][-1]
    if receipt["status"] in ("time_limit", "update_limit") and mixed:
        receipt["passed"] = (
            receipt["accepted_updates"] >= 128
            and final["train"]["family_credits"] >= 18
            and final["development"]["correct"] >= 36
            and final["development"]["old_family_credits"] >= 3
            and final["development"]["new_family_credits"] >= 15
            and final["train"]["refusals"] == final["development"]["refusals"] == 0
            and receipt["refused_recall_rows"] == 0
        )
    receipt["work"] = {
        "attempted_learning_calls": receipt["attempted_updates"],
        "accepted_learning_calls": receipt["accepted_updates"],
        "refused_learning_calls": receipt["attempted_updates"] - receipt["accepted_updates"],
        "training_row_presentations": len(train["labels"]) * receipt["attempted_updates"],
        "accepted_training_row_presentations": receipt["accepted_row_exposures"],
        "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in receipt["updates"]),
        "reported_phase_row_residual_checks": sum(
            r["reported_phase_row_residual_checks"] for r in receipt["updates"]
        ),
        "independent_phase_row_residual_checks": sum(
            r["independent_phase_row_residual_checks"] for r in receipt["updates"]
        ),
        "recall_calls": sum(
            q["work"]["calls"] for r in receipt["recall"] for q in (r["train"], r["development"])
        ),
        "recall_row_sweeps": sum(
            q["work"]["row_sweeps"]
            for r in receipt["recall"]
            for q in (r["train"], r["development"])
        ),
        "reported_recall_residual_checks": sum(
            q["work"]["reported_residual_checks"]
            for r in receipt["recall"]
            for q in (r["train"], r["development"])
        ),
        "independent_recall_residual_checks": sum(
            q["work"]["independent_residual_checks"]
            for r in receipt["recall"]
            for q in (r["train"], r["development"])
        ),
        "refused_recall_rows": receipt["refused_recall_rows"],
    }
    if mixed:
        receipt["work"]["new_training_row_presentations"] = 80 * receipt["attempted_updates"]
        receipt["work"]["old_rehearsal_row_presentations"] = 16 * receipt["attempted_updates"]
    harness.write_json(folder / "receipt.json", receipt)
    return receipt


def protocol(args):
    recipe = {
        "gene": development.GENES["canonical"],
        "config": make_brain(0).learner.config.to_dict(),
        "architecture": {"inputs": 650, "motor": 36, "modules": [32, 16], "observers": []},
    }
    return {
        "schema": "cadence-integrated-graph-confirmation-v1",
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
        "relations_protocol_sha256": relations.protocol_hash(),
        "relations_protocol": relations.protocol(),
        "founder_seeds": list(relations.CONFIRMATION_FOUNDERS),
        "schedule": {
            "old4": "all16 TRAIN rows until all4 TRAIN/DEV families are acquired",
            "mixed": "same brain; all80 new TRAIN rows plus16 old rehearsal rows per update",
            "old_update_limit": 512,
            "mixed_update_limit": 512,
            "mixed_minimum": 128,
            "check_every": 8,
            "seconds_per_founder": 1200,
        },
        "stopping": (
            "First passing cold TRAIN>=18families,DEV>=36/48,old>=3,new>=15 after128 "
            "mixed updates; otherwise final frozen endpoint. No earlier prefix selection."
        ),
        "heldout": (
            "Generated only after this founder's matching development pass; >=36/48 required. "
            "No recipe change or response training follows; only the separately charged "
            "TRAIN-labelled saved-continuation audit applies a next update."
        ),
        "checkpoint": (
            "Exact save/load predictions/config and actual next accepted/refused teaching "
            "outcome and complete checkpoint arrays."
        ),
        "zero_refusal_boundary": (
            "Every required old/mixed learning admission, cold recall prefix, final DEV/TEST, "
            "saved readback and actual next-teaching audit must have zero refusals."
        ),
        "library_sources": harness.sources(),
        "producing_sources": {
            name: harness.sha256(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "relation_development.py",
                "relations.py",
                "run.py",
                "extract.py",
            )
        },
        "max_output_mib": args.max_output_mib,
        "time_boundary": (
            "1200sec admission deadline per founder; finish the current bounded phase and "
            "mandatory endpoint/heldout/checkpoint readbacks, charging all their work."
        ),
        "recording": (
            "Accepted vector/update hashes and independent residual/cache/cost; refused "
            "raw states; full initial/old/mixed/continued checkpoints. Full trajectory "
            "replay remains separate."
        ),
        "scope": (
            "Same continuing graph learns by qualified local contrasts; explicit charged "
            "rehearsal. No hippocampal/Trace read or write, external classifier, native "
            "policy competence, general lifelong retention or efficiency advantage."
        ),
    }


def stage_summary(receipt, folder):
    return {
        key: receipt[key]
        for key in (
            "status",
            "passed",
            "accepted_updates",
            "attempted_updates",
            "accepted_row_exposures",
            "initial_sha256",
            "final_sha256",
            "work",
        )
    } | {
        "folder": folder.name,
        "receipt_sha256": harness.sha256(folder / "receipt.json"),
        "recall": [receipt["recall"][0], receipt["recall"][-1]],
    }


def complete_work(result):
    work = {}
    for name in ("old4", "mixed"):
        for key, count in result.get(name, {}).get("work", {}).items():
            work[key] = work.get(key, 0) + count
    queries = []
    if "heldout" in result:
        queries.append(result["heldout"]["work"])
    if "save_load" in result:
        queries.append(result["save_load"]["query_work"])
    if "continuation" in result:
        queries.extend(result["continuation"]["query_work"])
        for attempt in result["continuation"]["attempts"]:
            report = attempt["report"]
            for key, count in {
                "attempted_learning_calls": 1,
                "accepted_learning_calls": int(attempt["accepted"]),
                "refused_learning_calls": int(not attempt["accepted"]),
                "training_row_presentations": 96,
                "accepted_training_row_presentations": 96 * int(attempt["accepted"]),
                "phase_row_sweeps": 96 * int(report["total_steps"]),
                "reported_phase_row_residual_checks": 96 * int(report["total_residual_checks"]),
            }.items():
                work[key] = work.get(key, 0) + count
    for query in queries:
        for key, field in {
            "recall_calls": "calls",
            "recall_row_sweeps": "row_sweeps",
            "reported_recall_residual_checks": "reported_residual_checks",
            "independent_recall_residual_checks": "independent_residual_checks",
        }.items():
            work[key] = work.get(key, 0) + query[field]
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-output-mib", type=float, default=80)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 16 < args.max_output_mib <= 80:
        parser.error("campaign bound must lie in (16,80] MiB")
    if not args.out.exists():
        args.out.mkdir(parents=True)
        harness.write_json(args.out / "protocol.json", protocol(args))
        development.freeze_sources(args.out)
        shutil.copyfile(__file__, args.out / "source" / Path(__file__).name)
    declaration = json.loads((args.out / "protocol.json").read_text())
    if declaration != protocol(args):
        raise ValueError("frozen source, recipe or resource contract differs")
    if args.prepare_only:
        print(
            json.dumps(
                {"prepared": True, "protocol_sha256": harness.sha256(args.out / "protocol.json")}
            )
        )
        return
    if (args.out / "summary.json").exists():
        raise FileExistsError(
            "preserve and audit the existing campaign; do not overwrite founder outcomes"
        )
    development.compact_phase_recording()
    summary = {
        "schema": "cadence-integrated-graph-confirmation-results-v1",
        "protocol_sha256": harness.sha256(args.out / "protocol.json"),
        "founder_denominator": 5,
        "founders": [
            {"seed": seed, "status": "not_run", "passed": False}
            for seed in declaration["founder_seeds"]
        ],
    }
    harness.write_json(args.out / "summary.json", summary)
    for result in summary["founders"]:
        seed = result["seed"]
        folder = args.out / f"founder-{seed}"
        folder.mkdir()
        deadline = time.monotonic() + 1200
        began = time.monotonic()
        model = make_brain(seed)
        old_train = relations.load_panel("train", families=relations.OLD_FAMILIES)
        old_dev = relations.load_panel("development", families=relations.OLD_FAMILIES)
        old_receipt = train_stage(
            model, old_train, old_dev, folder / "old4", mixed=False, deadline=deadline, args=args
        )
        result["old4"] = stage_summary(old_receipt, folder / "old4")
        if not result["old4"]["passed"]:
            result["status"] = "old_acquisition_failed"
        else:
            train, dev = relations.load_panel("train"), relations.load_panel("development")
            mixed_receipt = train_stage(
                model, train, dev, folder / "mixed", mixed=True, deadline=deadline, args=args
            )
            result["mixed"] = stage_summary(mixed_receipt, folder / "mixed")
            if not result["mixed"]["passed"]:
                result["status"] = "mixed_development_failed"
            else:
                unlock = {
                    "relations_protocol_sha256": declaration["relations_protocol_sha256"],
                    "recipe_sha256": declaration["recipe_sha256"],
                    "development_passed": True,
                    "zero_refusals": True,
                    "recipe_frozen_before_development": True,
                    "founder_seed": seed,
                    "checkpoint_sha256": result["mixed"]["final_sha256"],
                }
                harness.write_json(folder / "development-unlock.json", unlock)
                heldout = relations.load_panel("heldout", development_receipt=unlock)
                result["heldout"] = recall(model, heldout, folder, "heldout")
                loaded = Brain.load(folder / "mixed/final.npz")
                loaded_recall = recall(loaded, dev, folder, "saved_development")
                before = result["mixed"]["recall"][-1]["development"]
                result["save_load"] = {
                    "predictions_equal": loaded_recall["predictions"] == before["predictions"],
                    "config_equal": loaded.learner.config == model.learner.config,
                    "query_work": loaded_recall["work"],
                }
                result["continuation"] = harness.continuation(
                    model, train["inputs"], train["labels"], folder / "mixed"
                )
                result["passed"] = (
                    result["heldout"]["correct"] >= 36
                    and result["heldout"]["refusals"] == 0
                    and result["save_load"]["predictions_equal"]
                    and result["save_load"]["config_equal"]
                    and result["continuation"]["passed"]
                    and all(attempt["accepted"] for attempt in result["continuation"]["attempts"])
                )
                result["status"] = "passed" if result["passed"] else "confirmation_failed"
        result["seconds"] = time.monotonic() - began
        result["work"] = complete_work(result)
        harness.write_json(folder / "receipt.json", result)
        harness.write_json(args.out / "summary.json", summary)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": result["status"],
                    "seconds": result["seconds"],
                    "old_updates": result["old4"]["accepted_updates"],
                    "mixed_updates": result.get("mixed", {}).get("accepted_updates"),
                    "heldout_correct": result.get("heldout", {}).get("correct"),
                }
            ),
            flush=True,
        )
    summary["passed"] = all(result["passed"] for result in summary["founders"])
    summary["completed_founders"] = sum(
        result["status"] != "not_run" for result in summary["founders"]
    )
    harness.write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
