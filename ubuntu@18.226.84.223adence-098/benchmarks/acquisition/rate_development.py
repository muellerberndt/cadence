"""One source-frozen quarter-rate development candidate; no heldout access.

The local contrast law and historical graph remain unchanged. Smaller fixed
rates are a candidate gene, motivated by the preserved large-step collapse in
the half-rate confirmation. Development success is not fresh confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import continual
import numpy as np
import online_curriculum as online
import relation_development as development
import relations
import run as harness

SEED = 0
CAPS = {"tiny2": 512, "old4": 1024, "mixed": 4096}
SECONDS = 900
ADMISSION_MIB = 144
CAP_MIB = 160
PRODUCERS = (
    "rate_development.py", "online_curriculum.py", "continual.py",
    "relation_development.py", "relations.py", "run.py", "extract.py",
)


def make_brain():
    model = development.make_brain(
        "canonical", SEED, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    baseline = model.learner.config
    model.learner.config = replace(baseline, eta=0.05, eta_bias=0.005)
    before, after = baseline.to_dict(), model.learner.config.to_dict()
    if {name for name in before if before[name] != after[name]} != {"eta", "eta_bias"}:
        raise AssertionError("candidate changed more than its two fixed learning rates")
    return model


def identity(model):
    graph = model.brain
    arrays = {
        "pre": graph.connectome.pre, "post": graph.connectome.post,
        "count": graph.connectome.count, "efficacy": graph.efficacy,
        "bias": graph.bias, "log_gain": graph.log_gain,
        "sensory_index": model.sensory_index, "motor_index": model.motor_index,
        "plastic_synapses": model.learner.plastic_synapses,
        "plastic_neurons": model.learner.plastic_neurons,
    }
    return {name: relations.array_hash(value) for name, value in arrays.items()}


def check_sources(protocol, root):
    source = root / "source"
    for name, digest in protocol["library_sources"].items():
        if harness.sha256(source / "library/cadence" / name) != digest:
            raise ValueError("frozen library source changed: " + name)
    for name, digest in protocol["producing_sources"].items():
        if harness.sha256(source / name) != digest:
            raise ValueError("frozen producer source changed: " + name)
    if harness.sources() != protocol["library_sources"]:
        raise ValueError("imported library differs from frozen source")
    if relations.protocol_hash() != protocol["relations_protocol_sha256"]:
        raise ValueError("relation task differs from frozen source")


def prepare(root, reference, canonical, negative):
    if root.exists():
        raise FileExistsError(root)
    parent = root.parent if root.parent.exists() else Path.cwd()
    if shutil.disk_usage(parent).free <= 20 * 1024**3:
        raise RuntimeError("require >20 GiB free before source admission")
    prior = json.loads((reference / "protocol.json").read_text())
    previous = json.loads((reference / "summary.json").read_text())
    if previous.get("passed") is not True or previous.get("protocol_sha256") != harness.sha256(
        reference / "protocol.json"
    ):
        raise ValueError("reference must retain its passed, hash-bound development summary")
    if prior["relations_protocol_sha256"] != relations.protocol_hash():
        raise ValueError("candidate changed the reference relation task")
    panels = {
        "tiny2": relations.load_panel("train", families=relations.TWO_FAMILIES),
        "old4": relations.load_panel("train", families=relations.OLD_FAMILIES),
        "mixed": relations.load_panel("train"),
    }
    orders = {name: online.curriculum(panel, CAPS[name] + 1, stage)
              for stage, (name, panel) in enumerate(panels.items())}
    for name, key in (("tiny2", "tiny_order"), ("old4", "old_order"), ("mixed", "mixed_order")):
        if orders[name][:len(prior[key])] != prior[key]:
            raise ValueError("candidate changed the frozen balanced curriculum prefix")
    model = make_brain()
    selected = model.learner.config.to_dict()
    if any(selected[key] != value for key, value in prior["learner_config"].items()
           if key not in ("eta", "eta_bias")):
        raise ValueError("candidate changed a non-rate learner parameter")
    if model.brain.neuron_model.to_dict() != prior["neuron_model"]:
        raise ValueError("candidate changed the neuron model")
    root.mkdir(parents=True, exist_ok=False)
    controls = {}
    for name, folder in (("canonical", canonical), ("half-rate-development", reference),
                         ("half-rate-confirmation", negative)):
        target = root / "controls" / name
        target.mkdir(parents=True)
        controls[name] = {"original_directory": str(folder), "artifacts": {}}
        for filename in ("protocol.json", "summary.json"):
            original = folder / filename
            shutil.copyfile(original, target / filename)
            controls[name]["artifacts"][filename] = harness.sha256(original)
        if name == "half-rate-confirmation":
            custody = folder / "custody/seed-4.json"
            if custody.exists():
                shutil.copyfile(custody, target / "seed-4-custody.json")
                controls[name]["artifacts"]["seed-4-custody.json"] = harness.sha256(custody)
    protocol = {
        "schema": "cadence-online-fixed-rate-development-v1",
        "candidate_gene": {"step_scale": 0.25, "eta": 0.05, "eta_bias": 0.005},
        "brain_seed": SEED, "founder_seeds": [SEED], "founder_denominator": 1,
        "model_construction": {"inputs": 650, "actions": 36, "modules": [32, 16],
                               "observers": [], "lateral": -0.5},
        "learner_config": selected, "neuron_model": model.brain.neuron_model.to_dict(),
        "initial_model_identity": identity(model), "pedagogy_seed": online.PEDAGOGY_SEED,
        "tiny_order": orders["tiny2"][:-1], "old_order": orders["old4"][:-1],
        "mixed_order": orders["mixed"][:-1],
        "after_cap_next_rows": {name: order[-1] for name, order in orders.items()},
        "tiny_cap": CAPS["tiny2"], "old_cap": CAPS["old4"], "mixed_cap": CAPS["mixed"],
        "teaching_admission_seconds": SECONDS, "output_admission_mib": ADMISSION_MIB,
        "output_cap_mib": CAP_MIB, "tiny_old_query_every": 32, "mixed_query_every": 96,
        "tiny_old_gate": "perfect TRAIN and DEV; positive TRAIN gain; zero refusals; fresh stages",
        "mixed_gate": ">=18 TRAIN families,>=36/48 DEV,old>=3/4,new>=15/20,"
                      ">=128 mixed lessons,positive TRAIN gain,zero refusals",
        "curriculum": "unchanged reference prefixes; balanced shuffled family cycles and "
                      "rotated TRAIN variants; paid old-family rehearsal in mixed stage",
        "mandatory_tail": "finish the current bounded phase; endpoint TRAIN/DEV reads and "
                          "actual original/load next scheduled learning; all separately charged",
        "library_sources": harness.sources(),
        "producing_sources": {name: harness.sha256(Path(__file__).with_name(name))
                              for name in PRODUCERS},
        "relations_protocol_sha256": relations.protocol_hash(),
        "runtime_precision": {"numpy": np.__version__, "dtype": "float64",
                              "eps": float(np.finfo(np.float64).eps)},
        "preserved_controls": controls,
        "scope": "one fixed rate development gene; original reciprocal graph/local contrast; "
                 "no memory, reward, classifier, default change or efficiency comparison",
        "heldout": "no generation or reads; development only; new fresh confirmation required",
        "recording": "accepted phase vector/parameter hashes and independent residual reports; "
                     "raw refused phases, lesson journals and checkpoints; "
                     "trajectory replay separate",
    }
    harness.write_json(root / "protocol.json", protocol)
    development.freeze_sources(root)
    for name in PRODUCERS:
        shutil.copyfile(Path(__file__).with_name(name), root / "source" / name)
    check_sources(protocol, root)
    harness.write_json(root / "summary.json", {
        "protocol_sha256": harness.sha256(root / "protocol.json"),
        "status": "prepared", "passed": False, "heldout_read": False,
    })
    return protocol


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(protocol, root)
    model = make_brain()
    if model.learner.config.to_dict() != protocol["learner_config"] or identity(model) != protocol[
        "initial_model_identity"
    ]:
        raise ValueError("worker initial model differs from the frozen candidate")
    development.compact_phase_recording()
    began = time.monotonic()
    deadline = began + protocol["teaching_admission_seconds"]
    stages = []
    endpoint_train = endpoint_dev = None
    endpoint_name = "tiny2"
    for name, families, order_key in (
        ("tiny2", relations.TWO_FAMILIES, "tiny_order"),
        ("old4", relations.OLD_FAMILIES, "old_order"),
        ("mixed", None, "mixed_order"),
    ):
        if name == "old4":
            model = make_brain()
        train = relations.load_panel("train", families=families)
        dev = relations.load_panel("development", families=families)
        report = online.stage(model, train, dev, protocol[order_key], root / name,
                              deadline, mixed=name == "mixed",
                              output_admission_mib=protocol["output_admission_mib"])
        stages.append((name, report))
        endpoint_name, endpoint_train, endpoint_dev = name, train, dev
        print(json.dumps({"stage": name, "passed": report["passed"],
                          "status": report["status"], "lessons": len(report["lessons"])}),
              flush=True)
        if not report["passed"]:
            break
    teaching_seconds = time.monotonic() - began
    tail_began = time.monotonic()
    checkpoint = root / endpoint_name / "final.npz"
    loaded = continual.Brain.load(checkpoint)
    originals = [online.recall(model, panel) for panel in (endpoint_train, endpoint_dev)]
    restored = [online.recall(loaded, panel) for panel in (endpoint_train, endpoint_dev)]
    equal_reads = all(a["predictions"] == b["predictions"]
                      for a, b in zip(originals, restored, strict=True))
    order_key = {"tiny2": "tiny_order", "old4": "old_order", "mixed": "mixed_order"}[endpoint_name]
    next_order = protocol[order_key] + [protocol["after_cap_next_rows"][endpoint_name]]
    next_row = next_order[len(stages[-1][1]["lessons"])]
    custody = continual.graph_checkpoint_continuation(
        model, loaded, endpoint_train["inputs"][[next_row]],
        endpoint_train["labels"][[next_row]], root,
    )
    continuation = {"saved_cold_predictions_equal": equal_reads,
                    "original_cold_train_development": originals,
                    "loaded_cold_train_development": restored,
                    "next_scheduled_row": next_row,
                    "next_family": int(endpoint_train["families"][next_row]),
                    "next_variant": int(endpoint_train["instances"][next_row]), **custody}
    harness.write_json(root / "continuation.json", continuation)
    lessons = [lesson for _, report in stages for lesson in report["lessons"]] + custody["lessons"]
    readings = [query for _, report in stages for reading in report["recall"]
                for query in (reading["train"], reading["development"])] + originals + restored
    query_work = [query["work"] for query in readings]
    zero_refusals = all(not lesson["failed"] for lesson in lessons) and all(
        not query["refusals"] for query in readings
    )
    continuation_passed = bool(equal_reads and custody["continued_arrays_equal"]
                               and custody["accepted_equal"] and zero_refusals)
    complete = len(stages) == 3 and all(report["passed"] for _, report in stages)
    passed = complete and continuation_passed
    if model.hippocampus.writes or np.any(model.hippocampus.consolidated):
        raise AssertionError("graph-only development unexpectedly wrote associative memory")
    if online.tree_bytes(root) >= CAP_MIB * 1024**2:
        passed = False
    last = stages[-1][1]["recall"][-1]
    summary = {
        "protocol_sha256": harness.sha256(root / "protocol.json"),
        "status": "development_passed" if passed else "development_failed",
        "passed": passed, "heldout_read": False, "founder_denominator": 1,
        "seed": SEED, "candidate_gene": protocol["candidate_gene"],
        "stages": [{"stage": name, "status": report["status"], "passed": report["passed"],
                    "lessons": len(report["lessons"]), "work": report["work"],
                    "receipt_sha256": harness.sha256(root / name / "receipt.json")}
                   for name, report in stages],
        "train_family_credits": last["train"]["family_credits"],
        "development_correct": last["development"]["correct"],
        "old_family_credits": last["development"].get("old_family_credits"),
        "new_family_credits": last["development"].get("new_family_credits"),
        "continuation_passed": continuation_passed, "zero_refusals": zero_refusals,
        "endpoint_checkpoint": str(checkpoint.relative_to(root)),
        "endpoint_checkpoint_sha256": harness.sha256(checkpoint),
        "continuation_sha256": harness.sha256(root / "continuation.json"),
        "teaching_and_instage_readback_seconds": teaching_seconds,
        "mandatory_tail_seconds": time.monotonic() - tail_began,
        "elapsed_seconds": time.monotonic() - began,
        "output_bytes_before_summary": online.tree_bytes(root),
        "work": {
            "single_cue_teaching_calls": len(lessons),
            "accepted_single_cue_teaching_calls": sum(row["accepted"] for row in lessons),
            "refused_single_cue_teaching_calls": sum(not row["accepted"] for row in lessons),
            "actual_continuation_teaching_calls": 2,
            "phase_row_sweeps": sum(row["phase_row_sweeps"] for row in lessons),
            "reported_phase_row_residual_checks": sum(row["reported_row_residual_checks"]
                                                      for row in lessons),
            "phase_stagnation_checks": sum(row["report"].get("total_stagnation_checks", 0)
                                            for row in lessons),
            "independent_phase_equation_cache_checks": sum(row["independent_residual_checks"]
                                                           for row in lessons),
            "cold_helper_query_calls": sum(row["calls"] for row in query_work),
            "query_row_sweeps": sum(row["row_sweeps"] for row in query_work),
            "reported_query_residual_checks": sum(row["reported_residual_checks"]
                                                  for row in query_work),
            "query_stagnation_checks": sum(row.get("reported_stagnation_checks", 0)
                                           for row in query_work),
            "independent_query_equation_cache_checks": sum(row["independent_residual_checks"]
                                                           for row in query_work),
            "refused_query_answers": sum(row["refusals"] for row in readings),
            "public_act_calls": 0, "associative_memory_reads": 0, "associative_memory_writes": 0,
        },
        "boundary": "one development founder; heldout sealed; all trajectory replay and fresh "
                    "confirmation remain separate; no general default or efficiency claim",
    }
    check_sources(protocol, root)
    harness.write_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--canonical-control", type=Path)
    parser.add_argument("--negative-confirmation", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
        return
    controls = (args.reference, args.canonical_control, args.negative_confirmation)
    if any(path is None for path in controls):
        parser.error("reference, canonical-control and negative-confirmation are required")
    prepare(root, args.reference.resolve(), args.canonical_control.resolve(),
            args.negative_confirmation.resolve())
    if args.prepare_only:
        return
    env = dict(os.environ)
    env.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env.update(PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(root / "source" / Path(__file__).name),
               "--out", str(root), "--worker"]
    with (root / "worker.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if code:
            summary = json.loads((root / "summary.json").read_text())
            summary.update(status="process_failed", passed=False, process_exit_code=code)
            harness.write_json(root / "summary.json", summary)
            raise RuntimeError("development worker failed; preserved worker.log and attempt")
    print((root / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
