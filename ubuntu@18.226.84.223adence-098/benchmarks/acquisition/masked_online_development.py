"""One declared canonical contrast control with fixed motor competition.

Only the existing motor-to-motor plastic mask changes from the canonical online
control. Tiny two relations use a fresh founder; old four and new twenty then
use one other continuing founder. No heldout or associative memory contributes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import continual
import graph_confirmation as graph_recording
import numpy as np
import online_curriculum as online
import relation_development as development
import relations
import run as harness

from cadence import Brain

OUTPUT_MIB = 64
SOURCE_NAMES = (
    "masked_online_development.py",
    "online_curriculum.py",
    "continual.py",
    "graph_confirmation.py",
    "relation_development.py",
    "relations.py",
    "run.py",
    "extract.py",
)


def make_brain():
    model = development.make_brain(
        "canonical", 0, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    wire = model.connectome
    motor = model.motor_index
    fixed = np.isin(wire.pre, motor) & np.isin(wire.post, motor)
    model.learner.plastic_synapses = ~fixed
    assert model.learner.config.eta == 0.2 and model.learner.config.eta_bias == 0.02
    assert model.learner.plastic_neurons.all()
    return model


def panels():
    return {
        name: (
            relations.load_panel("train", families=families),
            relations.load_panel("development", families=families),
        )
        for name, families in (
            ("tiny2", relations.TWO_FAMILIES),
            ("old4", relations.OLD_FAMILIES),
            ("mixed", None),
        )
    }


def orders(data):
    return {
        name: online.curriculum(data[name][0], cap + 1, 2 if name == "mixed" else 1)
        for name, cap in (("tiny2", 256), ("old4", 512), ("mixed", 2048))
    }


def declaration(data, lesson_orders):
    model = make_brain()
    return {
        "schema": "cadence-masked-canonical-online-development-v1",
        "brain_seed": 0,
        "pedagogy_seed": online.PEDAGOGY_SEED,
        "gene": development.GENES["canonical"],
        "sole_gene_change": "existing motor-to-motor contacts are nonplastic from birth",
        "plastic_mask_sha256": development.array_hash(model.learner.plastic_synapses),
        "frozen_contacts": int((~model.learner.plastic_synapses).sum()),
        "learner_config": model.learner.config.to_dict(),
        "neuron_model": model.brain.neuron_model.to_dict(),
        "relations_protocol_sha256": relations.protocol_hash(),
        "library_sources": harness.sources(),
        "producing_sources": {
            name: harness.sha256(Path(__file__).with_name(name)) for name in SOURCE_NAMES
        },
        "orders": lesson_orders,
        "input_sha256": {
            name: development.array_hash(train["inputs"]) for name, (train, _) in data.items()
        },
        "schedule": {
            "tiny2": "fresh canonical founder; up to256 single observed cues",
            "old4": "another fresh founder; up to512 single observed cues",
            "mixed": "same old4 brain; up to2048 balanced old/new single cues",
            "minimum_mixed_lessons": 128,
            "query_every": {"tiny2": 32, "old4": 32, "mixed": 96},
            "seconds_total": 900,
            "output_bound_mib": OUTPUT_MIB,
            "output_admission_mib": OUTPUT_MIB - 8,
        },
        "gates": {
            "tiny2_and_old4": "perfect all TRAIN/DEV rows, positive TRAIN gain, accepted teaching",
            "mixed": "TRAIN>=18complete families,DEV>=36/48,old>=3/4,new>=15/20 after128lessons",
            "zero_refusals": (
                "all learning, required query prefixes/endpoints and checkpoint audits"
            ),
            "endpoint": (
                "first passing fixed query, or final resource endpoint; no prefix selection"
            ),
        },
        "checkpoint": (
            "Save/load identical config and cold DEV answers; actual next scheduled TRAIN cue "
            "on both original and resumed models, independent phase checks and exact arrays."
        ),
        "recording": (
            "Every phase includes pre-update independent original-equation/cache checks, "
            "raw-vector and post-update optimizer/counter pins; refused raw states retained. "
            "Separate cold query records retain every refusal and numerical work."
        ),
        "deadline": "900sec teaching admission; finish bounded phase and charge mandatory tail",
        "scope": (
            "One development control, not a five-founder confirmation or efficiency claim. "
            "No heldout generated/read; no hippocampal/Trace read/write or external classifier. "
            "Prior batch/online/RMS/sensory-bias failures remain evidence."
        ),
    }


def gate(stage, train, dev, *, mixed):
    final = stage["recall"][-1]
    admitted = stage["accepted_updates"] == len(stage["lessons"]) > 0
    qualified = admitted and all(not item["failed"] for item in stage["lessons"])
    zero_queries = not any(
        q["refusals"] for r in stage["recall"] for q in (r["train"], r["development"])
    )
    gain = final["train"]["correct"] > stage["recall"][0]["train"]["correct"]
    if mixed:
        behavior = (
            stage["accepted_updates"] >= 128
            and final["train"]["family_credits"] >= 18
            and final["development"]["correct"] >= 36
            and final["development"]["old_family_credits"] >= 3
            and final["development"]["new_family_credits"] >= 15
        )
    else:
        behavior = final["train"]["correct"] == len(train["labels"]) and final["development"][
            "correct"
        ] == len(dev["labels"])
    return bool(qualified and zero_queries and gain and behavior)


def stage_brief(report, passed):
    return {
        "status": report["status"],
        "gate_passed": passed,
        "attempted_lessons": len(report["lessons"]),
        "accepted_lessons": report["accepted_updates"],
        "work": report["work"],
        "initial_sha256": report["initial_sha256"],
        "final_sha256": report["final_sha256"],
        "initial": report["recall"][0],
        "final": report["recall"][-1],
    }


def checkpoint(model, train, dev, next_row, source, folder, record):
    folder.mkdir()
    record.update(folder=folder, number=0)
    loaded = Brain.load(source)
    before = [online.recall(m, dev) for m in (model, loaded)]
    same_config = loaded.learner.config == model.learner.config
    mask_equal = np.array_equal(loaded.learner.plastic_synapses, model.learner.plastic_synapses)
    custody = continual.graph_checkpoint_continuation(
        model, loaded, train["inputs"][[next_row]], train["labels"][[next_row]], folder
    )
    after = [online.recall(m, dev) for m in (model, loaded)]
    zero = not any(q["refusals"] for q in before + after)
    result = custody | {
        "scheduled_next_row": int(next_row),
        "before": before,
        "after": after,
        "scheduled_next_family": int(train["families"][next_row]),
        "scheduled_next_variant": int(train["instances"][next_row]),
        "config_equal": same_config,
        "plastic_mask_equal": mask_equal,
        "passed": bool(
            same_config
            and mask_equal
            and zero
            and before[0]["predictions"] == before[1]["predictions"]
            and after[0]["predictions"] == after[1]["predictions"]
            and custody["continued_arrays_equal"]
            and all(r["accepted"] and not r["failed"] for r in custody["lessons"])
        ),
    }
    harness.write_json(folder / "receipt.json", result)
    return result


def aggregate(summary):
    work = {}
    for name in ("tiny2", "old4", "mixed"):
        for key, count in summary.get(name, {}).get("work", {}).items():
            work[key] = work.get(key, 0) + count
    for custody in summary.get("checkpoint", {}).values():
        for lesson in custody["lessons"]:
            for key, count in {
                "attempted_single_cue_lessons": 1,
                "observed_row_presentations": 1,
                "accepted_single_cue_lessons": lesson["accepted"],
                "refused_single_cue_lessons": int(not lesson["accepted"]),
                "phase_row_sweeps": lesson["phase_row_sweeps"],
                "reported_phase_row_residual_checks": lesson["reported_row_residual_checks"],
                "independent_phase_equation_cache_checks": lesson["independent_residual_checks"],
                "checkpoint_teaching_row_presentations": 1,
            }.items():
                work[key] = work.get(key, 0) + count
        for q in custody["before"] + custody["after"]:
            for key, count in {
                "query_calls": q["work"]["calls"],
                "query_row_sweeps": q["work"]["row_sweeps"],
                "reported_query_residual_checks": q["work"]["reported_residual_checks"],
                "independent_query_equation_cache_checks": q["work"]["independent_residual_checks"],
                "refused_query_rows": q["refusals"],
            }.items():
                work[key] = work.get(key, 0) + count
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data = panels()
    lesson_orders = orders(data)
    protocol = declaration(data, lesson_orders)
    if not args.out.exists():
        if shutil.disk_usage(args.out.parent).free <= 20 * 1024**3:
            raise RuntimeError("require>20GiB free")
        args.out.mkdir(parents=True)
        harness.write_json(args.out / "protocol.json", protocol)
        development.freeze_sources(args.out)
        for name in SOURCE_NAMES:
            shutil.copyfile(Path(__file__).with_name(name), args.out / "source" / name)
    if json.loads((args.out / "protocol.json").read_text()) != protocol:
        raise ValueError("frozen source/recipe/resource declaration differs")
    if args.prepare_only:
        print(
            json.dumps(
                {"prepared": True, "protocol_sha256": harness.sha256(args.out / "protocol.json")}
            )
        )
        return
    if (args.out / "summary.json").exists():
        raise FileExistsError("preserve existing probe outcomes")
    began = time.monotonic()
    deadline = began + 900
    development.compact_phase_recording()
    recording = {"folder": None, "number": 0}

    def recorded_recall(model, panel):
        model.reset()
        model.hippocampus.reset(1)
        number = recording["number"]
        recording["number"] += 1
        return graph_recording.recall(model, panel, recording["folder"], f"query-{number:04d}")

    online.recall = recorded_recall
    summary = {
        "protocol_sha256": harness.sha256(args.out / "protocol.json"),
        "passed": False,
        "status": "running",
        "heldout_read": False,
        "checkpoint": {},
    }
    harness.write_json(args.out / "summary.json", summary)
    models = []
    for name in ("tiny2", "old4", "mixed"):
        if name == "mixed":
            model = models[-1]
        else:
            model = make_brain()
            models.append(model)
        train, dev = data[name]
        folder = args.out / name
        recording.update(folder=folder, number=0)
        report = online.stage(
            model,
            train,
            dev,
            lesson_orders[name][:-1],
            folder,
            deadline,
            mixed=name == "mixed",
            output_admission_mib=OUTPUT_MIB - 8,
        )
        passed = gate(report, train, dev, mixed=name == "mixed")
        summary[name] = stage_brief(report, passed)
        harness.write_json(args.out / "summary.json", summary)
        if name == "tiny2" or name == "mixed" or not passed:
            custody = checkpoint(
                model,
                train,
                dev,
                lesson_orders[name][len(report["lessons"])],
                folder / "final.npz",
                args.out / (name + "-checkpoint"),
                recording,
            )
            summary["checkpoint"][name] = custody
            passed = passed and custody["passed"]
        if not passed:
            summary["status"] = name + "_failed"
            break
        if name == "mixed":
            summary["status"], summary["passed"] = "development_passed", True
    for model in models:
        if model.hippocampus.writes or np.any(model.hippocampus.consolidated):
            raise AssertionError("unexpected associative memory write")
        fixed = ~model.learner.plastic_synapses
        if not np.array_equal(model.brain.efficacy[fixed], model.connectome.sign[fixed]):
            raise AssertionError("fixed motor competition changed")
    summary["work"] = aggregate(summary)
    summary["seconds"] = time.monotonic() - began
    summary["source_library_unchanged"] = harness.sources() == protocol["library_sources"]
    summary["source_producers_unchanged"] = all(
        harness.sha256(Path(__file__).with_name(name)) == digest
        for name, digest in protocol["producing_sources"].items()
    )
    summary["passed"] &= (
        summary["source_library_unchanged"] and summary["source_producers_unchanged"]
    )
    summary["final_output_bytes"] = sum(
        p.stat().st_size for p in args.out.rglob("*") if p.is_file()
    )
    if summary["final_output_bytes"] >= OUTPUT_MIB * 1024**2:
        summary["passed"], summary["status"] = False, "output_boundary_violation"
    harness.write_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "passed", "seconds", "work")}), flush=True)


if __name__ == "__main__":
    main()
