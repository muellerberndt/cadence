"""Five source-frozen founders, each acquiring then retaining in one brain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
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

SEEDS = (1, 2, 3, 4, 5)
SECONDS = 1800
ADMISSION_MIB = 80
CAP_MIB = 90


def model_identity(brain):
    """Initial numerical model identity, independent of checkpoint metadata."""
    graph = brain.brain
    arrays = {
        "pre": graph.connectome.pre,
        "post": graph.connectome.post,
        "count": graph.connectome.count,
        "efficacy": graph.efficacy,
        "bias": graph.bias,
        "log_gain": graph.log_gain,
        "sensory_index": brain.sensory_index,
        "motor_index": brain.motor_index,
        "plastic_synapses": brain.learner.plastic_synapses,
        "plastic_neurons": brain.learner.plastic_neurons,
    }
    return {
        "arrays": {name: relations.array_hash(value) for name, value in arrays.items()},
        "learner_config": brain.learner.config.to_dict(),
        "neuron_model": graph.neuron_model.to_dict(),
    }


def check_sources(protocol, source):
    """Refuse source or fixture drift before a worker can teach or read a panel."""
    for name, expected in protocol["library_sources"].items():
        if harness.sha256(source / "library/cadence" / name) != expected:
            raise ValueError(f"frozen library source changed: {name}")
    for name, expected in protocol["producing_sources"].items():
        if harness.sha256(source / name) != expected:
            raise ValueError(f"frozen producing source changed: {name}")
    if harness.sources() != protocol["library_sources"]:
        raise ValueError("imported library differs from frozen source")
    if relations.protocol_hash() != protocol["relations_protocol_sha256"]:
        raise ValueError("relations protocol differs from frozen source")


def make_brain(
    seed, *, eta=0.1, eta_bias=0.01, leak=0.1, lateral=-0.5,
    normalize=0.0, momentum=0.0,
):
    # Historical archived constructors have no lateral keyword and already use
    # the original circuit. Only the new declared wiring gene needs this hook.
    wiring = {} if lateral == -0.5 else {"lateral": lateral}
    brain = development.make_brain(
        "canonical", seed, SimpleNamespace(phase_steps=4096, tolerance=0.003),
        **wiring,
    )
    baseline = brain.learner.config
    brain.learner.config = replace(
        baseline, eta=eta, eta_bias=eta_bias, normalize=normalize, momentum=momentum,
    )
    changed = {
        name
        for name in baseline.to_dict()
        if baseline.to_dict()[name] != brain.learner.config.to_dict()[name]
    }
    if not changed <= {"eta", "eta_bias", "normalize", "momentum"}:
        raise AssertionError("candidate changed an undeclared learning configuration")
    graph = brain.brain
    if leak != graph.neuron_model.leak:
        brain.learner.brain = type(graph)(
            graph.connectome,
            graph.neuron_model.replace(leak=leak),
            efficacy=graph.efficacy,
            bias=graph.bias,
            log_gain=graph.log_gain,
            backend=graph.backend,
            dense_limit=graph.dense_limit,
            layout=graph.layout,
            precision=graph.precision,
        )
    return brain


def tree_bytes(root):
    return online.tree_bytes(root)


def worker(root, seed):
    protocol = json.loads((root / "protocol.json").read_text())
    recipe_sha = harness.sha256(root / "protocol.json")
    if protocol.get("schema", "").endswith(("-v2", "-v3", "-v4", "-v5", "-v6")):
        check_sources(protocol, root / "source")
    if seed not in protocol["founder_seeds"]:
        raise ValueError("worker seed is not in the frozen founder census")
    folder = root / f"seed-{seed}"
    folder.mkdir(exist_ok=False)
    old_train = relations.load_panel("train", families=relations.OLD_FAMILIES)
    old_dev = relations.load_panel("development", families=relations.OLD_FAMILIES)
    train, dev = relations.load_panel("train"), relations.load_panel("development")
    brain = make_brain(
        seed, eta=protocol["learner_config"]["eta"],
        eta_bias=protocol["learner_config"]["eta_bias"],
        leak=protocol["neuron_model"]["leak"],
        lateral=protocol.get("model_construction", {}).get("lateral", -0.5),
        normalize=protocol["learner_config"]["normalize"],
        momentum=protocol["learner_config"]["momentum"],
    )
    if "initial_model_identities" in protocol:
        if model_identity(brain) != protocol["initial_model_identities"][str(seed)]:
            raise AssertionError("worker initial graph differs from frozen founder")
    if brain.learner.config.to_dict() != protocol["learner_config"]:
        raise AssertionError("worker learning configuration differs from frozen recipe")
    if brain.brain.neuron_model.to_dict() != protocol["neuron_model"]:
        raise AssertionError("worker neural model differs from frozen recipe")
    development.compact_phase_recording()
    began = time.monotonic()
    deadline = began + SECONDS
    old_folder, mixed_folder = root / f"seed-{seed}-old4", root / f"seed-{seed}-mixed"
    old = online.stage(
        brain,
        old_train,
        old_dev,
        protocol["old_order"],
        old_folder,
        deadline,
        mixed=False,
        output_admission_mib=protocol["output_admission_mib"],
    )
    outcome = {
        "seed": seed,
        "recipe_sha256": recipe_sha,
        "protocol_sha256": recipe_sha,
        "old_status": old["status"],
        "old_lessons": len(old["lessons"]),
        "old_passed": old["passed"],
        "stages": [str(old_folder.name)],
        "passed": False,
        "heldout_read": False,
    }
    endpoint_train, endpoint_dev = old_train, old_dev
    checkpoint = old_folder / "final.npz"
    next_order = protocol["old_order"] + [protocol["old_after_cap_next_row"]]
    next_row = next_order[len(old["lessons"])]
    readings = [q for r in old["recall"] for q in (r["train"], r["development"])]
    lessons = list(old["lessons"])
    mixed = None
    if old["passed"]:
        mixed = online.stage(
            brain,
            train,
            dev,
            protocol["mixed_order"],
            mixed_folder,
            deadline,
            mixed=True,
            output_admission_mib=protocol["output_admission_mib"],
        )
        endpoint_train, endpoint_dev = train, dev
        checkpoint = mixed_folder / "final.npz"
        next_order = protocol["mixed_order"] + [protocol["mixed_after_cap_next_row"]]
        next_row = next_order[len(mixed["lessons"])]
        readings.extend(q for r in mixed["recall"] for q in (r["train"], r["development"]))
        lessons.extend(mixed["lessons"])
        last = mixed["recall"][-1]
        outcome.update(
            mixed_status=mixed["status"],
            mixed_lessons=len(mixed["lessons"]),
            train_family_credits=last["train"]["family_credits"],
            development_correct=last["development"]["correct"],
            old_family_credits=last["development"]["old_family_credits"],
            new_family_credits=last["development"]["new_family_credits"],
        )
        outcome["stages"].append(str(mixed_folder.name))
    outcome["teaching_and_instage_readback_seconds"] = time.monotonic() - began
    tail_began = time.monotonic()
    # Both actual endpoint brains receive identical cold TRAIN and DEV reads.
    loaded = continual.Brain.load(checkpoint)
    original_reads = [online.recall(brain, p) for p in (endpoint_train, endpoint_dev)]
    loaded_reads = [online.recall(loaded, p) for p in (endpoint_train, endpoint_dev)]
    endpoint_equal = all(
        a["predictions"] == b["predictions"]
        for a, b in zip(original_reads, loaded_reads, strict=True)
    )
    zero_refusals = all(not row["failed"] for row in lessons) and all(
        row["refusals"] == 0 for row in readings + original_reads + loaded_reads
    )
    dev_passed = bool(
        old["passed"]
        and mixed is not None
        and mixed["passed"]
        and endpoint_equal
        and zero_refusals
        and original_reads[0]["family_credits"] >= 18
        and original_reads[1]["correct"] >= 36
        and original_reads[1]["old_family_credits"] >= 3
        and original_reads[1]["new_family_credits"] >= 15
    )
    unlock = {
        "seed": seed,
        "relations_protocol_sha256": relations.protocol_hash(),
        "recipe_sha256": recipe_sha,
        "recipe_frozen_before_development": True,
        "development_passed": dev_passed,
        "zero_refusals": zero_refusals,
        "endpoint_checkpoint_sha256": harness.sha256(checkpoint),
        "old_receipt_sha256": harness.sha256(old_folder / "receipt.json"),
        "mixed_receipt_sha256": harness.sha256(mixed_folder / "receipt.json") if mixed else None,
        "train_family_credits": original_reads[0]["family_credits"],
        "development_correct": original_reads[1]["correct"],
        "old_family_credits": original_reads[1].get("old_family_credits"),
        "new_family_credits": original_reads[1].get("new_family_credits"),
    }
    harness.write_json(folder / "development-receipt.json", unlock)
    heldout = None
    if dev_passed:
        # The persisted unlock above precedes every generation/read of test values.
        panel = relations.load_panel(
            "heldout", development_receipt=folder / "development-receipt.json"
        )
        test_brain = continual.Brain.load(checkpoint)
        heldout = online.recall(test_brain, panel)
        harness.write_json(
            folder / "heldout-receipt.json",
            {
                "development_receipt_sha256": harness.sha256(folder / "development-receipt.json"),
                "checkpoint_sha256": harness.sha256(checkpoint),
                "panel_hashes": {
                    name: relations.array_hash(value) for name, value in panel.items()
                },
                "recall": heldout,
            },
        )
        outcome.update(heldout_read=True, heldout_correct=heldout["correct"])
    x, y = endpoint_train["inputs"][[next_row]], endpoint_train["labels"][[next_row]]
    custody = continual.graph_checkpoint_continuation(brain, loaded, x, y, folder)
    continuation = {
        "saved_cold_predictions_equal": endpoint_equal,
        "original_cold_train_development": original_reads,
        "loaded_cold_train_development": loaded_reads,
        "next_scheduled_row": next_row,
        "next_family": int(endpoint_train["families"][next_row]),
        "next_variant": int(endpoint_train["instances"][next_row]),
        "next_label": int(y[0]),
        "next_single_cue_lessons": custody["lessons"],
        **{name: value for name, value in custody.items() if name != "lessons"},
    }
    harness.write_json(folder / "continuation.json", continuation)
    continuation_pass = bool(
        endpoint_equal
        and custody["continued_arrays_equal"]
        and custody["accepted_equal"]
        and all(not row["failed"] for row in custody["lessons"])
        and zero_refusals
    )
    all_lessons = lessons + custody["lessons"]
    all_reads = readings + original_reads + loaded_reads + ([heldout] if heldout else [])
    query_work = [row["work"] for row in all_reads]
    outcome.update(
        development_passed=dev_passed,
        continuation_passed=continuation_pass,
        passed=bool(
            dev_passed
            and heldout is not None
            and heldout["correct"] >= 36
            and heldout["refusals"] == 0
            and continuation_pass
        ),
        mandatory_tail_seconds=time.monotonic() - tail_began,
        elapsed_seconds=time.monotonic() - began,
        endpoint_checkpoint_sha256=harness.sha256(checkpoint),
        continuation_phase_payloads_retained=custody["phase_payloads_retained"],
        work={
            "single_cue_teaching_calls": len(all_lessons),
            "accepted_single_cue_teaching_calls": sum(row["accepted"] for row in all_lessons),
            "refused_single_cue_teaching_calls": sum(not row["accepted"] for row in all_lessons),
            "actual_continuation_teaching_calls": 2,
            "old_acquisition_presentations": len(old["lessons"]),
            "mixed_presentations": len(mixed["lessons"]) if mixed else 0,
            "mixed_old_rehearsal_presentations": mixed["work"]["old_rehearsal_presentations"]
            if mixed
            else 0,
            "mixed_new_relation_presentations": mixed["work"]["new_presentations"] if mixed else 0,
            "continuation_old_presentations": 2
            if continuation["next_family"] in relations.OLD_FAMILIES
            else 0,
            "continuation_new_presentations": 2
            if continuation["next_family"] in relations.NEW_FAMILIES
            else 0,
            "phase_row_sweeps": sum(row["phase_row_sweeps"] for row in all_lessons),
            "phase_stagnation_checks": sum(
                row["report"].get("total_stagnation_checks", 0) for row in all_lessons
            ),
            "query_stagnation_checks": sum(
                row.get("reported_stagnation_checks", 0) for row in query_work
            ),
            "reported_phase_row_residual_checks": int(
                sum(row["reported_row_residual_checks"] for row in all_lessons)
            ),
            "independent_phase_equation_cache_checks": sum(
                row["independent_residual_checks"] for row in all_lessons
            ),
            "cold_helper_query_calls": sum(row["calls"] for row in query_work),
            "public_act_calls": 0,
            "query_row_sweeps": sum(row["row_sweeps"] for row in query_work),
            "reported_query_residual_checks": sum(
                row["reported_residual_checks"] for row in query_work
            ),
            "independent_query_equation_cache_checks": sum(
                row["independent_residual_checks"] for row in query_work
            ),
            "refused_query_answers": sum(row["refusals"] for row in all_reads),
            "associative_memory_reads": 0,
            "associative_memory_writes": 0,
        },
    )
    if brain.hippocampus.writes or np.any(brain.hippocampus.consolidated):
        raise AssertionError("graph confirmation unexpectedly wrote associative memory")
    outcome["status"] = "confirmed" if outcome["passed"] else "confirmation_failed"
    harness.write_json(folder / "summary.json", outcome)
    print(json.dumps(outcome), flush=True)


def archive_founder(root, seed):
    # Numeric prefixes are not founder identities: seed-1 must never include
    # the still-running seed-10, its stages or its evidence.
    paths = [
        root / name for name in (f"seed-{seed}", f"seed-{seed}-old4", f"seed-{seed}-mixed")
        if (root / name).is_dir()
    ]
    hashes = {
        str(p.relative_to(root)): harness.sha256(p)
        for folder in paths
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }
    archive = root / f"seed-{seed}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=9) as bundle:
        for folder in paths:
            bundle.add(folder, arcname=folder.name)
    observed = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if member.isfile():
                observed[member.name] = hashlib.sha256(
                    bundle.extractfile(member).read()
                ).hexdigest()
    if observed != hashes:
        raise AssertionError("outer archive does not preserve every original member byte")
    receipt = {
        "seed": seed,
        "archive_sha256": harness.sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "original_member_sha256": hashes,
        "every_original_member_verified": True,
        "original_bytes_preserved": True,
        "loose_duplicates_removed_after_verification": True,
    }
    harness.write_json(root / "custody" / f"seed-{seed}.json", receipt)
    for folder in paths:
        shutil.rmtree(folder)
    return receipt


def launch(
    root, development_root, *, current_source=False, prepare_only=False,
    output_cap_mib=CAP_MIB, founder_seeds=SEEDS,
):
    if not 90 <= output_cap_mib <= 512:
        raise ValueError("output cap must be predeclared between 90 and 512 MiB")
    reference_raw = (development_root / "protocol.json").read_bytes()
    reference_sha = hashlib.sha256(reference_raw).hexdigest()
    reference = json.loads(reference_raw)
    development_seeds = reference.get("founder_seeds", [reference.get("brain_seed", 0)])
    if (
        len(founder_seeds) != 5
        or len(set(founder_seeds)) != 5
        or any(type(seed) is not int or seed < 0 for seed in founder_seeds)
        or not set(founder_seeds).isdisjoint(development_seeds)
    ):
        raise ValueError("confirmation requires five distinct fresh nonnegative founder seeds")
    reference_summary_raw = (development_root / "summary.json").read_bytes()
    reference_summary_sha = hashlib.sha256(reference_summary_raw).hexdigest()
    reference_summary = json.loads(reference_summary_raw)
    if reference_summary.get("passed") is not True or reference_summary.get(
        "protocol_sha256"
    ) != reference_sha:
        raise ValueError("reference is not a passed source-bound development result")
    if reference["relations_protocol_sha256"] != relations.protocol_hash():
        raise ValueError("confirmation changed the passed development's relation task")
    old_cap, mixed_cap = len(reference["old_order"]), len(reference["mixed_order"])
    for name, cap in (("old", old_cap), ("mixed", mixed_cap)):
        if name + "_cap" in reference and reference[name + "_cap"] != cap:
            raise ValueError("passed development's declared cap differs from its lesson order")
    parent = root.parent if root.parent.exists() else Path.cwd()
    if shutil.disk_usage(parent).free <= 20 * 1024**3:
        raise RuntimeError("require >20GiB free")
    old = relations.load_panel("train", families=relations.OLD_FAMILIES)
    train = relations.load_panel("train")
    old_order = online.curriculum(old, old_cap + 1, 1)
    mixed_order = online.curriculum(train, mixed_cap + 1, 2)
    if old_order[:old_cap] != reference["old_order"] or (
        mixed_order[:mixed_cap] != reference["mixed_order"]
    ):
        raise AssertionError("confirmation changed the successful frozen curriculum")
    rates = {
        "eta": reference["learner_config"]["eta"],
        "eta_bias": reference["learner_config"]["eta_bias"],
    }
    lateral = reference.get("model_construction", {}).get("lateral", -0.5)
    if type(lateral) not in (float, int) or lateral not in (-0.5, 0.0):
        raise ValueError("confirmation supports only the declared historical or zero lateral gene")
    gene = {
        **rates, "leak": reference["neuron_model"]["leak"], "lateral": lateral,
        "normalize": reference["learner_config"]["normalize"],
        "momentum": reference["learner_config"]["momentum"],
    }
    original_recipe = (
        founder_seeds == SEEDS and rates == {"eta": 0.1, "eta_bias": 0.01}
        and gene["leak"] == 0.1 and lateral == -0.5
        and gene["normalize"] == 0.0 and gene["momentum"] == 0.0
        and (old_cap, mixed_cap) == (512, 2048)
    )
    if not current_source and not original_recipe:
        raise ValueError("new candidate genes or founder census require --current-source")
    brain = make_brain(0, **gene)
    if (
        brain.learner.config.to_dict() != reference["learner_config"]
        or brain.brain.neuron_model.to_dict() != reference["neuron_model"]
    ):
        raise AssertionError("confirmation changed the successful half-step configuration")
    if "initial_model_identity" in reference:
        recorded = reference["initial_model_identity"]
        recorded_arrays = recorded.get("arrays", recorded)
        if model_identity(brain)["arrays"] != recorded_arrays:
            raise AssertionError("confirmation changed the development founder's initial graph")
    root.mkdir(exist_ok=False)
    (root / "custody").mkdir()
    reference_capsule = root / "reference-development"
    reference_capsule.mkdir()
    (reference_capsule / "protocol.json").write_bytes(reference_raw)
    (reference_capsule / "summary.json").write_bytes(reference_summary_raw)
    if current_source:
        development.freeze_sources(root)
        for name in reference["producing_sources"]:
            if Path(name).name != name:
                raise ValueError("producer source names must be local filenames")
            shutil.copyfile(Path(__file__).with_name(name), root / "source" / name)
    else:
        shutil.copytree(development_root / "source", root / "source")
    shutil.copyfile(root / "source/fixture/relations.py", root / "source/relations.py")
    shutil.copyfile(Path(__file__), root / "source" / Path(__file__).name)
    producing_sources = dict(reference["producing_sources"])
    if current_source:
        producing_sources = {
            name: harness.sha256(root / "source" / name) for name in producing_sources
        }
    producing_sources[Path(__file__).name] = harness.sha256(Path(__file__))
    protocol = {
        "schema": (
            "cadence-online-half-step-five-founder-confirmation-v6"
            if gene["normalize"] != 0.0 or gene["momentum"] != 0.0
            else "cadence-online-half-step-five-founder-confirmation-v5"
            if lateral != -0.5
            else "cadence-online-half-step-five-founder-confirmation-v4"
            if gene["leak"] != 0.1
            else "cadence-online-half-step-five-founder-confirmation-v3"
            if not original_recipe
            else "cadence-online-half-step-five-founder-confirmation-v2"
        ) if current_source else "cadence-online-half-step-five-founder-confirmation-v1",
        "founder_seeds": list(founder_seeds),
        "founder_denominator": 5,
        "candidate_gene": reference["candidate_gene"],
        "learner_config": reference["learner_config"],
        "neuron_model": reference["neuron_model"],
        "library_sources": harness.sources() if current_source else reference["library_sources"],
        "producing_sources": producing_sources,
        "runtime_precision": reference["runtime_precision"],
        "successful_development_protocol_sha256": reference_sha,
        "successful_development_summary_sha256": reference_summary_sha,
        "relations_protocol_sha256": relations.protocol_hash(),
        "pedagogy_seed": online.PEDAGOGY_SEED,
        "old_order": old_order[:old_cap],
        "mixed_order": mixed_order[:mixed_cap],
        "old_after_cap_next_row": old_order[old_cap],
        "mixed_after_cap_next_row": mixed_order[mixed_cap],
        "old_lesson_cap": old_cap,
        "mixed_lesson_cap": mixed_cap,
        "soft_teaching_admission_seconds_per_founder": SECONDS,
        "mandatory_tail": (
            "bounded endpoint/heldout/actual original-loaded next scheduled "
            "lesson charged separately"
        ),
        "concurrency": (
            "two frozen waves of three then two founders; independent processes; "
            "all numerical thread counts 1"
        ),
        "output_admission_mib": output_cap_mib - 10,
        "output_cap_mib": output_cap_mib,
        "gate": (
            "perfect old TRAIN+DEV then same brain mixed>=128; TRAIN>=18 complete "
            "families,DEV>=36/48,old>=3/4,new>=15/20,positive TRAIN gain,zero refusals"
        ),
        "heldout_gate": (
            ">=36/48,zero refusals; persisted matching development unlock "
            "before generating any heldout values"
        ),
        "archive": (
            "verify every original member SHA256 in lossless completed-founder "
            "tar.gz before removing loose duplicates"
        ),
        "query_boundary": (
            "producer cold helper queries; independent actual Brain.act audit separately charged"
        ),
        "no_replacement": (
            "all five seeds remain in census including failures, caps and "
            "process errors; no gene tuning"
        ),
        "mechanism": (
            "unchanged canonical graph/local contrast except declared half-step "
            "rates; no hippocampal/Trace/reward/classifier/default change"
        ),
    }
    if current_source:
        import cadence

        protocol.update(
            source_boundary="new current-source confirmation; original development retained",
            source_version=cadence.__version__,
            model_construction={
                "inputs": 650,
                "actions": 36,
                "modules": [32, 16],
                "observers": [],
                "lateral": lateral,
            },
            initial_model_identities={
                str(seed): model_identity(make_brain(seed, **gene)) for seed in founder_seeds
            },
            historical_library_sources=reference["library_sources"],
            historical_runtime_precision=reference["runtime_precision"],
            runtime_precision={
                "numpy": np.__version__, "dtype": "float64",
                "eps": float(np.finfo(np.float64).eps),
            },
        )
        if gene["leak"] != 0.1:
            protocol["mechanism"] = (
                "unchanged local contrast law with fixed declared learning rates and "
                "neuron leak gene; historical wiring; no memory/Trace/reward/classifier "
                "or library default change"
            )
        if lateral != -0.5:
            protocol["mechanism"] = (
                "unchanged local contrast law with fixed declared learning rates and "
                "zero motor lateral gene; no memory/Trace/reward/classifier "
                "or library default change"
            )
        if gene["normalize"] != 0.0 or gene["momentum"] != 0.0:
            protocol["mechanism"] = (
                "existing local contrast law and per-synapse optimizer history with "
                "fixed declared teacher rates and normalization; no memory/Trace/reward/"
                "classifier or library default change"
            )
    harness.write_json(root / "protocol.json", protocol)
    for name, expected in producing_sources.items():
        if harness.sha256(root / "source" / name) != expected:
            raise AssertionError("frozen producing source differs from declared recipe")
    for name, expected in protocol["library_sources"].items():
        if harness.sha256(root / "source/library/cadence" / name) != expected:
            raise AssertionError("frozen library differs from successful development source")
    recipe_sha = harness.sha256(root / "protocol.json")
    outcomes = {
        seed: {"seed": seed, "status": "scheduled", "passed": False} for seed in founder_seeds
    }

    def census():
        harness.write_json(
            root / "summary.json",
            {
                "recipe_sha256": recipe_sha,
                "founder_denominator": 5,
                "passed": sum(row["passed"] for row in outcomes.values()),
                "all_founders_passed": all(row["passed"] for row in outcomes.values()),
                "outcomes": list(outcomes.values()),
                "campaign_bytes": tree_bytes(root),
            },
        )

    census()
    if prepare_only:
        print(json.dumps({"prepared": True, "protocol_sha256": recipe_sha}), flush=True)
        return
    print(
        json.dumps({
            "protocol_sha256": recipe_sha, "seeds": founder_seeds,
            "seconds_per_founder": SECONDS,
        }),
        flush=True,
    )
    env = dict(os.environ)
    env.update(
        {
            name: "1"
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    env.update(PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    processes = {}
    try:
        for wave in (founder_seeds[:3], founder_seeds[3:]):
            processes = {}
            for seed in wave:
                log = (root / f"worker-{seed}.log").open("w")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(root / "source" / Path(__file__).name),
                        "--out",
                        str(root),
                        "--worker-seed",
                        str(seed),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                processes[seed] = (process, log)
                outcomes[seed]["status"] = "running"
            census()
            while processes:
                for seed, (process, log) in list(processes.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    summary = root / f"seed-{seed}/summary.json"
                    outcomes[seed] = (
                        json.loads(summary.read_text())
                        if summary.exists()
                        else {
                            "seed": seed,
                            "status": "process_failed",
                            "passed": False,
                            "exit_code": code,
                        }
                    )
                    outcomes[seed]["process_exit_code"] = code
                    receipt = archive_founder(root, seed)
                    outcomes[seed]["outer_archive_sha256"] = receipt["archive_sha256"]
                    print(
                        json.dumps(
                            {"completed": outcomes[seed], "archive_bytes": receipt["archive_bytes"]}
                        ),
                        flush=True,
                    )
                    del processes[seed]
                    census()
                if tree_bytes(root) >= output_cap_mib * 1024**2:
                    raise RuntimeError(
                        "declared output guard exceeded; preserve all files and "
                        "mark census resource failure"
                    )
                time.sleep(1)
    finally:
        # Only processes owned by this launch are stopped on a failed resource
        # guard or interruption; their attempted files remain for the census.
        for seed, (process, log) in processes.items():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            outcomes[seed].update(
                status="interrupted", passed=False, process_exit_code=process.returncode
            )
        if processes:
            census()
    census()
    print(
        json.dumps(
            {
                "finished": True,
                "passed": sum(row["passed"] for row in outcomes.values()),
                "founder_denominator": 5,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--current-source", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-cap-mib", type=int, default=CAP_MIB)
    parser.add_argument("--founder-seeds", type=int, nargs=5, default=SEEDS)
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker_seed is not None:
        worker(root, args.worker_seed)
    else:
        if args.development is None:
            parser.error("--development frozen successful campaign required")
        launch(
            root,
            args.development.resolve(),
            current_source=args.current_source,
            prepare_only=args.prepare_only,
            output_cap_mib=args.output_cap_mib,
            founder_seeds=tuple(args.founder_seeds),
        )


if __name__ == "__main__":
    main()
