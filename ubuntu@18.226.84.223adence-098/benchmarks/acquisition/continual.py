"""Explicit observed-label associative memory and rehearsal controls for System 1.

This benchmark separates graph contrasts from categorical associative-store
writes. A memory control does not discharge a contrast-learning contract.
Answers always come from the qualified whole neural graph, with remembered
values entering its motor ports as boundary drive. No reward is fabricated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import run as harness
from extract import row_hash, sha256, write_json

from cadence import Brain, LearnerConfig

ANCHORS = [0, 9, 17, 22]
PUBLIC_ACT_PARITY = False
COMPACT_QUERIES = False


def make_brain(seed: int, consolidation: float) -> Brain:
    config = LearnerConfig(
        beta=0.1,
        eta=0.005,
        eta_bias=0.02,
        normalize=0.99,
        momentum=0.9,
        free_steps=4096,
        nudged_steps=4096,
        tolerance=0.003,
        qualified=True,
        damping=3,
    )
    brain = Brain.compose(
        650, 36, backend="cpu", modules=(32, 16), seed=seed, consolidation=consolidation, learning=config,
            recall=False
    )
    graph = brain.brain
    bias = graph.bias.copy()
    bias[brain.sensory_index] = 0.6
    brain.learner.brain = graph.with_parameters(bias=bias)
    motor_pre = np.isin(graph.connectome.pre, brain.motor_index)
    motor_post = np.isin(graph.connectome.post, brain.motor_index)
    brain.learner.plastic_synapses = ~(motor_pre & motor_post)
    return brain


def durable_recall(brain: Brain, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """Clear live/working/fast state before every cold, durable-memory answer."""
    rows, sweeps, checks = [], 0, 0
    public_sweeps, public_checks, public_independent_checks = 0, 0, 0
    before = brain.hippocampus.consolidated.copy()
    writes = brain.hippocampus.writes
    for index, (observation, label) in enumerate(zip(inputs, labels, strict=True)):
        brain.reset()
        brain.hippocampus.reset(1)
        x = observation[None, :]
        drive = brain.stimulus(x, memory=True)
        phase = brain._equilibrate(drive, None, budget=4096, tolerance=0.003)
        residual, cache = harness.independent_residual(brain, drive, phase.state)
        valid = bool(phase.qualified.all() and residual.max() <= 0.003 and cache.max() <= 0.003)
        prediction = int(np.argmax(phase.state.activation[0, brain.motor_index])) if valid else -1
        row = {
            "row": index,
            "label": int(label),
            "prediction": prediction,
            "full_residual": float(residual.max()),
            "cache_defect": float(cache.max()),
            "steps": phase.steps,
            "residual_checks": phase.residual_checks,
        }
        if not COMPACT_QUERIES:
            row["memory_value"] = brain.hippocampus.recall(x)[0].tolist()
        if PUBLIC_ACT_PARITY:
            brain.reset()
            brain.hippocampus.reset(1)
            original = brain._equilibrate
            captured = []

            def recording_solve(drive, state, *, original=original, captured=captured, **kwargs):
                answer = original(drive, state, **kwargs)
                captured.append((drive.copy(), answer))
                return answer

            brain._equilibrate = recording_solve
            try:
                public_prediction = int(brain.act(x, greedy=True)[0])
            except RuntimeError:
                public_prediction = -1
            finally:
                del brain._equilibrate
            if len(captured) != 1:
                raise AssertionError(
                    "greedy public answer did not use exactly one whole-graph solve"
                )
            public_drive, public_phase = captured[0]
            public_residual, public_cache = harness.independent_residual(
                brain, public_drive, public_phase.state
            )
            public_sweeps += public_phase.steps
            public_checks += public_phase.residual_checks
            public_independent_checks += 1
            public_valid = (
                public_prediction >= 0
                and public_phase.qualified.all()
                and public_residual.max() <= 0.003
                and public_cache.max() <= 0.003
            )
            row.update(
                public_act_prediction=public_prediction,
                public_act_equal=bool(public_prediction == prediction),
                public_act_qualified=bool(public_valid),
                public_act_full_residual=float(public_residual.max()),
                public_act_cache_defect=float(public_cache.max()),
                public_act_steps=public_phase.steps,
                public_act_residual_checks=public_phase.residual_checks,
            )
        rows.append(row)
        sweeps += phase.steps
        checks += phase.residual_checks
    if (
        not np.array_equal(before, brain.hippocampus.consolidated)
        or writes != brain.hippocampus.writes
    ):
        raise AssertionError("a durable measurement wrote persistent memory")
    predictions = np.array([row["prediction"] for row in rows])
    return {
        "correct": int(np.sum(predictions == labels)),
        "examples": len(labels),
        "refusals": int(np.sum(predictions < 0)),
        "predictions": predictions.tolist(),
        "public_act_equal": all(r.get("public_act_equal", True) for r in rows),
        "public_act_refusals": sum(not r.get("public_act_qualified", True) for r in rows),
        "rows": rows,
        "work": {
            "calls": len(labels),
            "row_sweeps": sweeps,
            "reported_residual_checks": checks,
            "independent_residual_checks": len(labels),
            "durable_memory_reads": len(labels) * (1 + (not COMPACT_QUERIES) + PUBLIC_ACT_PARITY),
            "live_working_fast_resets": len(labels) * (1 + PUBLIC_ACT_PARITY),
            "public_act_calls": len(labels) if PUBLIC_ACT_PARITY else 0,
            "public_act_row_sweeps": public_sweeps,
            "public_act_reported_residual_checks": public_checks,
            "public_act_independent_residual_checks": public_independent_checks,
        },
    }


def memory_write(brain: Brain, observation: np.ndarray, label: int) -> float:
    """An independently checked existing normalized-delta association write."""
    memory = brain.hippocampus
    old = memory.consolidated.copy()
    memory.reset(1)
    target = np.eye(36)[[label]]
    memory.observe(observation[None, :], target)
    # Check the persistent matrix law with an independent formula and no fast
    # state. Record the runtime's longdouble precision: it can equal float64.
    cue = observation.astype(np.longdouble)
    scale = np.max(np.abs(cue))
    cue = cue / scale if scale else cue
    norm = np.linalg.norm(cue)
    cue = cue / norm if norm else cue
    error = target[0].astype(np.longdouble) - cue @ old.astype(np.longdouble)
    expected = old.astype(np.longdouble) + memory.consolidation * np.outer(cue, error)
    deviation = float(np.abs(memory.consolidated - expected).max())
    if deviation > 1e-12:
        raise AssertionError("observed-label persistent write differs from local delta law")
    return deviation


def family_credits(recall, labels, families):
    predicted = np.array(recall["predictions"])
    credited = [
        int(f)
        for f in np.unique(families)
        if np.all(predicted[families == f] == labels[families == f])
    ]
    return {"family_credits": len(credited), "credited_families": credited}


def train_memory(
    brain, inputs, labels, positions, rehearsal, folder, args, deadline, families=None
):
    folder.mkdir()
    report = {
        "mechanism": "explicit observed-label associative store",
        "status": "running",
        "rounds": [],
        "recall": [],
        "memory_writes": 0,
        "rehearsal_exposures": 0,
        "graph_contrast_updates": 0,
        "labels": labels.tolist(),
        "input_sha256": row_hash(inputs),
    }
    report["recall"].append({"round": 0, **durable_recall(brain, inputs, labels)})
    if families is not None:
        report["families"] = families.tolist()
        report["recall"][-1].update(family_credits(report["recall"][-1], labels, families))
    brain.save(folder / "initial.npz")
    replay = [i for i, position in enumerate(positions) if position in ANCHORS]
    for round_index in range(1, args.rounds + 1):
        if time.monotonic() >= deadline:
            report["status"] = "time_limit"
            break
        order = list(range(len(labels))) + (replay if rehearsal else [])
        max_error = 0.0
        for row in order:
            max_error = max(max_error, memory_write(brain, inputs[row], int(labels[row])))
        report["memory_writes"] += len(order)
        report["rehearsal_exposures"] += len(replay) if rehearsal else 0
        report["rounds"].append(
            {"round": round_index, "writes": len(order), "max_independent_matrix_error": max_error}
        )
        if round_index in {1, 8, 16, 32, 64, 128, args.rounds}:
            recall = durable_recall(brain, inputs, labels)
            report["recall"].append({"round": round_index, **recall})
            if families is not None:
                recall.update(family_credits(recall, labels, families))
                report["recall"][-1].update(family_credits(recall, labels, families))
            write_json(folder / "receipt.json", report)
            count = len(labels) if families is None else len(np.unique(families))
            gate = count if count < 24 else 18
            credit = recall["correct"] if families is None else recall["family_credits"]
            if credit >= gate and recall["refusals"] == 0:
                report["status"] = "acquired"
                break
    else:
        report["status"] = "round_limit"
    last_round = len(report["rounds"])
    if report["recall"][-1]["round"] != last_round:
        report["recall"].append({"round": last_round, **durable_recall(brain, inputs, labels)})
        if families is not None:
            report["recall"][-1].update(family_credits(report["recall"][-1], labels, families))
    if brain.learner.contrast_updates:
        raise AssertionError("associative control unexpectedly performed graph contrast updates")
    report["checkpoint_sha256"] = sha256(brain.save(folder / "final.npz"))
    report["persistent_parameters"] = brain.hippocampus.consolidated.size
    report["memory_write_key_value_macs_estimate"] = report["memory_writes"] * 4 * 650 * 36
    report["work"] = {
        "memory_writes": report["memory_writes"],
        "rehearsal_exposures": report["rehearsal_exposures"],
        "query_work": [r["work"] for r in report["recall"]],
        "independent_matrix_checks": report["memory_writes"],
    }
    write_json(folder / "receipt.json", report)
    return report


def retention_memory(
    brain, inputs, labels, rehearsal, folder, args, deadline, development=None, families=None
):
    """Same persistent store; either family-disjoint new writes or charged rehearsal."""
    folder.mkdir()
    old = ANCHORS if families is None else np.flatnonzero(np.isin(families, ANCHORS)).tolist()
    new = [i for i in range(len(labels)) if i not in old]
    if set(labels[old]).intersection(labels[new]):
        raise ValueError("interference action families must be disjoint")
    brain.save(folder / "anchor.npz")
    report = {
        "status": "running",
        "rounds": [],
        "recall": [],
        "memory_writes": 0,
        "new_exposures": 0,
        "rehearsal_exposures": 0,
        "graph_contrast_updates": 0,
    }

    def measure():
        if development is None:
            return {
                "old": durable_recall(brain, inputs[old], labels[old]),
                "new": durable_recall(brain, inputs[new], labels[new]),
            }
        readings = {}
        for name, family_set in (
            ("old", ANCHORS),
            ("new", [i for i in range(24) if i not in ANCHORS]),
        ):
            chosen = np.isin(development["families"], family_set)
            readings[name] = durable_recall(
                brain, development["inputs"][chosen], development["labels"][chosen]
            )
            readings[name].update(
                family_credits(
                    readings[name], development["labels"][chosen], development["families"][chosen]
                )
            )
        return readings

    for round_index in range(args.rounds + 1):
        if round_index in {0, 16, 32, 64, args.rounds}:
            report["recall"].append({"round": round_index, **measure()})
            write_json(folder / "receipt.json", report)
        if round_index == args.rounds:
            report["status"] = "complete"
            break
        if time.monotonic() >= deadline:
            report["status"] = "time_limit"
            break
        order = new + (old if rehearsal else [])
        max_error = max(memory_write(brain, inputs[i], int(labels[i])) for i in order)
        report["rounds"].append(
            {
                "round": round_index + 1,
                "writes": len(order),
                "max_independent_matrix_error": max_error,
            }
        )
        report["memory_writes"] += len(order)
        report["new_exposures"] += len(new)
        report["rehearsal_exposures"] += len(old) if rehearsal else 0
    if report["recall"][-1]["round"] != len(report["rounds"]):
        report["recall"].append({"round": len(report["rounds"]), **measure()})
    endpoint = report["recall"][-1]
    key = "correct" if development is None else "family_credits"
    report["passed"] = (
        report["status"] == "complete"
        and report["recall"][0]["old"][key] == 4
        and endpoint["old"][key] >= 3
        and endpoint["new"][key] >= 15
        and endpoint["old"]["refusals"] == endpoint["new"]["refusals"] == 0
    )
    report["final_sha256"] = sha256(brain.save(folder / "final.npz"))
    loaded = Brain.load(folder / "final.npz")
    original_brain, brain = brain, loaded
    validation = measure()
    brain = original_brain
    report["saved_durable_recall_equal"] = all(
        validation[name]["predictions"] == endpoint[name]["predictions"] for name in validation
    )
    # Compare actual live pre-save continuation with the loaded continuation.
    for model in (brain, loaded):
        memory_write(model, inputs[new[0]], int(labels[new[0]]))
    a, b = (
        brain.save(folder / "continued-original.npz"),
        loaded.save(folder / "continued-loaded.npz"),
    )
    with np.load(a, allow_pickle=False) as aa, np.load(b, allow_pickle=False) as bb:
        report["continued_arrays_equal"] = set(aa.files) == set(bb.files) and all(
            np.array_equal(aa[k], bb[k]) for k in aa.files
        )
    report["work"] = {
        "new_exposures": report["new_exposures"],
        "rehearsal_exposures": report["rehearsal_exposures"],
        "memory_writes": report["memory_writes"],
        "independent_matrix_checks": report["memory_writes"] + 2,
        "query_work": [r[name]["work"] for r in report["recall"] for name in ("old", "new")],
        "validation_query_work": [r["work"] for r in validation.values()],
        "continued_observation_writes": 2,
        "graph_contrast_updates": 0,
        "key_value_macs_estimate": (report["memory_writes"] + 2) * 4 * 650 * 36,
    }
    write_json(folder / "receipt.json", report)
    return report


def confirm_memory(args):
    """Five predeclared founders of the successful categorical-store recipe."""
    import relations

    global PUBLIC_ACT_PARITY, COMPACT_QUERIES
    PUBLIC_ACT_PARITY = COMPACT_QUERIES = True
    if args.rounds != 128 or args.consolidation != 0.05:
        raise ValueError("confirmation freezes 128 rounds and default consolidation .05")
    prior = args.out.parent / "relation-associative-retention"
    development = json.loads((prior / "memory-rehearsal/receipt.json").read_text())
    prior_protocol = json.loads((prior / "protocol.json").read_text())
    if not development["passed"] or prior_protocol["library_sources"] != harness.sources():
        raise ValueError("confirmation requires the passed, unchanged seed0 recipe sources")
    if prior_protocol["relations_protocol_sha256"] != relations.protocol_hash():
        raise ValueError("confirmation task differs from development")
    recipe = {
        "schema": "cadence-observed-label-rehearsal-recipe-v1",
        "mechanism": "existing categorical associative-store normalized delta; not graph contrast",
        "composition": {"inputs": 650, "actions": 36, "modules": [32, 16], "observers": []},
        "library_sources": harness.sources(),
        "relations_protocol_sha256": relations.protocol_hash(),
        ("brain_parameters"): (
            "make_brain: sensory bias .6; fixed motor lateral mask; no contrast updates"
        ),
        "consolidation": 0.05,
        "fast_boundary": "reset before each write and cold query",
        "teacher": "explicit observed one-hot label value; no reward invented",
        ("anchor_acquisition"): (
            "all16 old TRAIN rows then all16 old replay rows per round; "
            "checkpoints0,1,8,16,32,64,128; stop at perfect qualified TRAIN"
        ),
        "interference": "128 rounds; ascending80 new TRAIN rows then ascending16 old replay rows",
        "query_rounds": [0, 16, 32, 64, 128],
        "qualification": {"tolerance": 0.003, "budget": 4096, "live_dt": 1, "damping": 3},
        "founders": list(relations.CONFIRMATION_FOUNDERS),
        "founder_denominator": 5,
        "seconds_per_founder": args.seconds_per_arm,
        ("development_gate"): (
            "old>=3/4,new>=15/20 families; >=36/48 instances; zero refused; 128 full rounds"
        ),
        ("heldout_gate"): (
            ">=36/48 instances with zero refused; read only after matching founder "
            "development passes"
        ),
        ("public_action"): (
            "cold public Brain.act(greedy=True) and helper solve separately; "
            "require identical qualified predictions; both charged"
        ),
        ("persistence"): (
            "save-load cold predictions and actual live-versus-loaded next "
            "observed-label write arrays equal"
        ),
        "development_receipt_sha256": sha256(prior / "memory-rehearsal/receipt.json"),
        "development_protocol_sha256": sha256(prior / "protocol.json"),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "recipe.json", recipe)
    recipe_hash = sha256(args.out / "recipe.json")
    protocol = {
        "schema": "cadence-five-founder-associative-confirmation-v1",
        "recipe_sha256": recipe_hash,
        "harness_sha256": sha256(__file__),
        "recipe_frozen_before_development": True,
        "founder_denominator": 5,
        "source_bytes": "source/library/cadence and source/continual.py",
        ("scope"): (
            "existing durable associative-store capability; no local-contrast or "
            "action-category claim"
        ),
    }
    write_json(args.out / "protocol.json", protocol)
    harness.freeze_sources(args.out / "source", args.fixture)
    shutil.copyfile(__file__, args.out / "source/continual.py")
    relations.freeze(args.out / "source/relations")
    old_panel = relations.load_panel("train", families=ANCHORS)
    train_panel = relations.load_panel("train")
    dev_panel = relations.load_panel("development")
    results = []
    for seed in relations.CONFIRMATION_FOUNDERS:
        size = sum(p.stat().st_size for p in args.out.parent.rglob("*") if p.is_file())
        outcome = {"seed": seed, "passed": False, "heldout_read": False}
        results.append(outcome)
        if size >= 88 * 1024**2 or shutil.disk_usage(args.out).free <= 20 * 1024**3:
            outcome["status"] = "storage_limit"
        else:
            folder = args.out / f"founder-{seed}"
            folder.mkdir()
            start = time.monotonic()
            deadline = start + args.seconds_per_arm
            brain = make_brain(seed, 0.05)
            acquired = train_memory(
                brain,
                old_panel["inputs"],
                old_panel["labels"],
                old_panel["families"].tolist(),
                True,
                folder / "old",
                args,
                deadline,
                families=old_panel["families"],
            )
            outcome.update(
                acquisition_status=acquired["status"],
                acquisition_writes=acquired["memory_writes"],
                acquisition_rehearsal_exposures=acquired["rehearsal_exposures"],
            )
            if acquired["status"] != "acquired":
                outcome["status"] = "acquisition_failed"
            else:
                retained = retention_memory(
                    brain,
                    train_panel["inputs"],
                    train_panel["labels"],
                    True,
                    folder / "retention",
                    args,
                    deadline,
                    development=dev_panel,
                    families=train_panel["families"],
                )
                endpoint = retained["recall"][-1]
                readings = [r[name] for r in retained["recall"] for name in ("old", "new")]
                readings += acquired["recall"]
                public_ok = all(
                    r["public_act_equal"] and r["public_act_refusals"] == 0 for r in readings
                )
                dev_correct = endpoint["old"]["correct"] + endpoint["new"]["correct"]
                dev_pass = bool(
                    retained["passed"]
                    and dev_correct >= 36
                    and public_ok
                    and retained["saved_durable_recall_equal"]
                    and retained["continued_arrays_equal"]
                )
                unlock = {
                    "relations_protocol_sha256": relations.protocol_hash(),
                    "recipe_sha256": recipe_hash,
                    "recipe_frozen_before_development": True,
                    "development_passed": dev_pass,
                    "zero_refusals": public_ok and all(r["refusals"] == 0 for r in readings),
                    "seed": seed,
                    "retention_receipt_sha256": sha256(folder / "retention/receipt.json"),
                    "old_family_credits": endpoint["old"]["family_credits"],
                    "new_family_credits": endpoint["new"]["family_credits"],
                    "development_correct": dev_correct,
                }
                write_json(folder / "development-receipt.json", unlock)
                outcome.update(
                    status="development_passed" if dev_pass else "development_failed",
                    development_correct=dev_correct,
                    old_family_credits=endpoint["old"]["family_credits"],
                    new_family_credits=endpoint["new"]["family_credits"],
                    memory_writes=retained["memory_writes"],
                    rehearsal_exposures=retained["rehearsal_exposures"],
                    public_act_equal=public_ok,
                    saved_durable_recall_equal=retained["saved_durable_recall_equal"],
                    continued_arrays_equal=retained["continued_arrays_equal"],
                )
                if dev_pass:
                    # Unlock first, then generate any heldout values. Query the
                    # saved endpoint, before the charged continuation lesson.
                    heldout = relations.load_panel(
                        "heldout", development_receipt=folder / "development-receipt.json"
                    )
                    model = Brain.load(folder / "retention/final.npz")
                    read = durable_recall(model, heldout["inputs"], heldout["labels"])
                    read.update(family_credits(read, heldout["labels"], heldout["families"]))
                    write_json(
                        folder / "heldout-receipt.json",
                        {
                            "recipe_sha256": recipe_hash,
                            "development_receipt_sha256": sha256(
                                folder / "development-receipt.json"
                            ),
                            "checkpoint_sha256": sha256(folder / "retention/final.npz"),
                            "panel_hashes": {
                                k: relations.array_hash(v) for k, v in heldout.items()
                            },
                            "recall": read,
                        },
                    )
                    outcome.update(
                        heldout_read=True,
                        heldout_correct=read["correct"],
                        heldout_family_credits=read["family_credits"],
                        passed=bool(
                            read["correct"] >= 36
                            and read["refusals"] == 0
                            and read["public_act_equal"]
                            and read["public_act_refusals"] == 0
                        ),
                    )
                    outcome["status"] = "confirmed" if outcome["passed"] else "heldout_failed"
            outcome["elapsed_seconds"] = time.monotonic() - start
        write_json(
            args.out / "summary.json",
            {
                "recipe_sha256": recipe_hash,
                "founder_denominator": 5,
                "passed": sum(r["passed"] for r in results),
                "outcomes": results,
            },
        )
        print(json.dumps(outcome), flush=True)


def graph_lesson(brain, inputs, labels, folder, name):
    """A qualified graph lesson with exact costs and independent endpoint checks."""
    drive = brain.stimulus(inputs, memory=False)
    graph = brain.brain
    weights = (
        graph.neuron_model.gain
        * graph.connectome.count
        * graph.efficacy
        * np.exp(graph.log_gain[graph.connectome.pre])
    )
    bias = graph.bias.copy()
    began = time.monotonic()
    failed = False
    try:
        state, report = brain.learner.step(drive, labels)
        phases = {"free": state.free, "nudged": state.nudged, "opposite": state.opposite}
        accepted = int(report.get("accepted", 1))
    except RuntimeError as error:
        if not hasattr(error, "phases") or not hasattr(error, "report"):
            raise
        phases = {key: phase.state for key, phase in error.phases.items()}
        report, accepted, failed = error.report, 0, True
    readings = harness.phase_readback(brain, drive, labels, phases, weights, bias)
    valid = len(phases) == 3 and all(
        all(r["qualified"]) and max(r["cache_defect"]) <= brain.learner.config.tolerance
        for r in readings.values()
    )
    if accepted and not valid:
        failed = True
    for phase, state in phases.items():
        readings[phase]["steps"] = int(report.get(phase + "_steps", state.steps))
    steps = sum(r["steps"] for r in readings.values())
    return {
        "accepted": accepted,
        "failed": failed,
        "report": harness.json_value(report),
        "seconds": time.monotonic() - began,
        "phases": readings,
        "phase_file": name + ".npz",
        "phase_sha256": harness.phase_file(folder / (name + ".npz"), phases),
        "phase_row_sweeps": len(labels) * steps,
        "reported_residual_checks": report.get("total_residual_checks"),
        "reported_row_residual_checks": len(labels) * report.get("total_residual_checks", 0),
        "independent_residual_checks": len(labels) * len(phases),
        "accepted_row_exposures": accepted * len(labels),
    }


def graph_checkpoint_continuation(brain, loaded, inputs, labels, folder):
    """Keep each real next-learning phase and checkpoint in distinct paths."""
    lessons = [
        graph_lesson(model, inputs, labels, folder, "continued-phases-" + name)
        for model, name in ((brain, "original"), (loaded, "loaded"))
    ]
    checkpoints = [
        model.save(folder / ("continued-" + name + ".npz"))
        for model, name in ((brain, "original"), (loaded, "loaded"))
    ]
    for lesson in lessons:
        if sha256(folder / lesson["phase_file"]) != lesson["phase_sha256"]:
            raise AssertionError("checkpoint save changed a continuation phase payload")
    with (
        np.load(checkpoints[0], allow_pickle=False) as aa,
        np.load(checkpoints[1], allow_pickle=False) as bb,
    ):
        equal = set(aa.files) == set(bb.files) and all(
            np.array_equal(aa[key], bb[key]) for key in aa.files
        )
    return {
        "lessons": lessons,
        "continued_arrays_equal": equal,
        "accepted_equal": lessons[0]["accepted"] == lessons[1]["accepted"],
        "phase_payloads_retained": True,
        "checkpoint_sha256": [sha256(path) for path in checkpoints],
    }


def graph_retention(args):
    """Original local contrasts, with explicitly separate old-replay exposure."""
    import relation_development as contrast
    import relations

    if args.rounds != 128:
        raise ValueError("graph retention freezes 128 full rounds")
    source = args.graph_retention_from
    source_receipt = json.loads((source.parent / "receipt.json").read_text())
    if sha256(source) != source_receipt["final_sha256"]:
        raise ValueError("anchor checkpoint differs from its acquisition receipt")
    model = Brain.load(source)
    cfg = model.learner.config
    expected = {
        "beta": 0.1,
        "eta": 0.2,
        "eta_bias": 0.02,
        "normalize": 0.0,
        "momentum": 0.0,
        "free_steps": 4096,
        "nudged_steps": 4096,
        "tolerance": 0.003,
        "qualified": True,
        "damping": 3,
    }
    if any(getattr(cfg, name) != value for name, value in expected.items()) or (
        model.brain.neuron_model.dt != 1
        or model.brain.neuron_model.leak != 0.1
        or not model.learner.plastic_synapses.all()
        or not model.learner.plastic_neurons.all()
    ):
        raise ValueError("anchor is not the unchanged canonical graph genotype")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "cadence-canonical-graph-retention-v1",
        "relations_protocol_sha256": relations.protocol_hash(),
        "library_sources": harness.sources(),
        "runner_sha256": sha256(__file__),
        "phase_recorder_sha256": sha256(contrast.__file__),
        "anchor_source": str(source),
        "anchor_source_sha256": sha256(source),
        "anchor_receipt_sha256": sha256(source.parent / "receipt.json"),
        "learner_config": cfg.to_dict(),
        "neuron_model": model.brain.neuron_model.to_dict(),
        "rounds": 128,
        "seconds_per_arm": args.seconds_per_arm,
        "arms": {
            "graph": "all80 newTRAIN rows per qualified batch",
            ("graph-rehearsal"): (
                "all80 new plus all16 oldTRAIN rows per qualified batch; each family "
                "equally weighted"
            ),
        },
        "query_rounds": list(range(0, 129, 8)),
        ("gate"): (
            "128 full rounds, initialold4/4, finalold>=3/4 andnew>=15/20 all2DEV "
            "rows per family; no refused teaching or answers; saved continuation "
            "equal"
        ),
        ("memory_boundary"): (
            "cold graph-only answers; no working/associative read or write; no reward fabricated"
        ),
        ("recording"): (
            "compact accepted vector hashes and independent residuals; refused "
            "full states; actual live/load continuation"
        ),
        ("promotion_boundary"): (
            "retention assay; no heldout generated; full24 acquisition/development "
            "remains separate prerequisite"
        ),
    }
    write_json(args.out / "protocol.json", protocol)
    harness.freeze_sources(args.out / "source", args.fixture)
    shutil.copyfile(__file__, args.out / "source/continual.py")
    shutil.copyfile(contrast.__file__, args.out / "source/relation_development.py")
    relations.freeze(args.out / "source/relations")
    shutil.copyfile(source, args.out / "old.npz")
    contrast.compact_phase_recording()
    train, development = relations.load_panel("train"), relations.load_panel("development")
    old_rows = np.flatnonzero(np.isin(train["families"], ANCHORS))
    new_rows = np.flatnonzero(~np.isin(train["families"], ANCHORS))
    results = []
    for arm in ("graph", "graph-rehearsal"):
        folder = args.out / arm
        folder.mkdir()
        brain = Brain.load(args.out / "old.npz")
        initial_writes = brain.hippocampus.writes
        persistent = brain.hippocampus.consolidated.copy()
        brain.save(folder / "initial.npz")
        rows = new_rows if arm == "graph" else np.concatenate((new_rows, old_rows))
        x, y = train["inputs"][rows], train["labels"][rows]

        def measure(model):
            model.reset()
            model.hippocampus.reset(1)
            reading = harness.free_recall(model, development["inputs"], development["labels"])
            predicted = np.asarray(reading["predictions"])
            for name, families in (("old", ANCHORS), ("new", relations.NEW_FAMILIES)):
                selected = np.isin(development["families"], families)
                sub = {"predictions": predicted[selected].tolist()}
                reading[name] = family_credits(
                    sub, development["labels"][selected], development["families"][selected]
                )
            return reading

        began = time.monotonic()
        deadline = began + args.seconds_per_arm
        report = {
            "status": "running",
            "rounds": [],
            "recall": [{"round": 0, **measure(brain)}],
            "accepted_updates": 0,
            "new_exposures": 0,
            "rehearsal_exposures": 0,
        }
        for round_index in range(1, 129):
            size = sum(p.stat().st_size for p in args.out.parent.rglob("*") if p.is_file())
            if size >= 88 * 1024**2 or time.monotonic() >= deadline:
                report["status"] = "storage_limit" if size >= 88 * 1024**2 else "time_limit"
                break
            lesson = graph_lesson(brain, x, y, folder, f"phases-{round_index:04d}")
            report["rounds"].append({"round": round_index, **lesson})
            report["accepted_updates"] += lesson["accepted"]
            report["new_exposures"] += len(new_rows)
            report["rehearsal_exposures"] += len(old_rows) if arm == "graph-rehearsal" else 0
            if round_index % 8 == 0 or lesson["failed"]:
                report["recall"].append({"round": round_index, **measure(brain)})
            write_json(folder / "receipt.json", report)
            if lesson["failed"]:
                report["status"] = (
                    "refused_learning" if not lesson["accepted"] else "qualification_violation"
                )
                break
        else:
            report["status"] = "complete"
        if report["recall"][-1]["round"] != len(report["rounds"]):
            report["recall"].append({"round": len(report["rounds"]), **measure(brain)})
        endpoint = report["recall"][-1]
        report["final_sha256"] = sha256(brain.save(folder / "final.npz"))
        loaded = Brain.load(folder / "final.npz")
        before, after = measure(brain), measure(loaded)
        custody = graph_checkpoint_continuation(brain, loaded, x, y, folder)
        continued, same = custody["lessons"], custody["continued_arrays_equal"]
        report["continuation"] = {
            "saved_durable_recall_equal": before["predictions"] == after["predictions"],
            **custody,
            "query_work": [before["work"], after["work"]],
        }
        if brain.hippocampus.writes != initial_writes or not np.array_equal(
            persistent, brain.hippocampus.consolidated
        ):
            raise AssertionError("graph-only retention unexpectedly wrote associative memory")
        report["passed"] = bool(
            report["status"] == "complete"
            and report["recall"][0]["old"]["family_credits"] == 4
            and endpoint["old"]["family_credits"] >= 3
            and endpoint["new"]["family_credits"] >= 15
            and all(r["refusals"] == 0 for r in report["recall"])
            and all(not r["failed"] for r in report["rounds"])
            and report["continuation"]["saved_durable_recall_equal"]
            and same
        )
        all_lessons = report["rounds"] + continued
        report["work"] = {
            "attempted_updates": len(report["rounds"]),
            "accepted_updates": report["accepted_updates"],
            "new_exposures": report["new_exposures"],
            "rehearsal_exposures": report["rehearsal_exposures"],
            "continuation_attempts": 2,
            "continuation_row_exposures": 2 * len(y),
            "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in all_lessons),
            "reported_phase_row_residual_checks": sum(
                r["reported_row_residual_checks"] for r in all_lessons
            ),
            "independent_phase_residual_checks": sum(
                r["independent_residual_checks"] for r in all_lessons
            ),
            "query_work": [r["work"] for r in report["recall"]] + [before["work"], after["work"]],
            "memory_writes": 0,
            "memory_reads": 0,
            "elapsed_seconds": time.monotonic() - began,
        }
        write_json(folder / "receipt.json", report)
        outcome = {
            "arm": arm,
            "status": report["status"],
            "passed": report["passed"],
            "accepted_updates": report["accepted_updates"],
            "old_family_credits": endpoint["old"]["family_credits"],
            "new_family_credits": endpoint["new"]["family_credits"],
            "saved_durable_recall_equal": report["continuation"]["saved_durable_recall_equal"],
            "continued_arrays_equal": same,
        }
        results.append(outcome)
        write_json(
            args.out / "summary.json",
            {"protocol_sha256": sha256(args.out / "protocol.json"), "outcomes": results},
        )
        print(json.dumps(outcome), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixture")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("memory", "memory-rehearsal"),
        default=["memory", "memory-rehearsal"],
    )
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--seconds-per-arm", type=float, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--consolidation", type=float, default=0.05)
    parser.add_argument(
        "--retention-from",
        type=Path,
        help="acquired raw control folder; reuse each arm's actual four-anchor checkpoint",
    )
    parser.add_argument(
        "--relations",
        action="store_true",
        help="declared cue-family task; no movie action semantics",
    )
    parser.add_argument(
        "--confirm-memory",
        action="store_true",
        help="freeze the successful rehearsal recipe and confirm founders1-5",
    )
    parser.add_argument(
        "--graph-retention-from", type=Path, help="canonical graph's acquired old4 Brain checkpoint"
    )
    args = parser.parse_args()
    maximum_seconds = 600 if args.graph_retention_from else 300
    if not 1 <= args.rounds <= 256 or not 0 < args.seconds_per_arm <= maximum_seconds:
        parser.error("bounded local controls require <=256 rounds and <=300 seconds per arm")
    if (
        shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path.cwd()).free
        < 20 * 1024**3
    ):
        raise RuntimeError("require more than20GiB free before starting")
    if args.confirm_memory:
        confirm_memory(args)
        return
    if args.graph_retention_from:
        graph_retention(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((args.fixture / "provenance.json").read_text())
    if sha256(args.fixture / "school.npz") != provenance["fixture_sha256"]:
        raise ValueError("frozen fixture differs")
    with np.load(args.fixture / "school.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files if not k.startswith("heldout_")}
    protocol = {
        "schema": "cadence-explicit-associative-continual-control-v1",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "library_sources": harness.sources(),
        "harness_sha256": sha256(__file__),
        "fixture_sha256": provenance["fixture_sha256"],
        "two_positions": [0, 9],
        "four_positions": ANCHORS,
        ("acquisition_gates"): (
            "fresh tiny2/4 perfect qualified durable answers, then fresh24>=18 with no refusal"
        ),
        ("memory_boundary"): (
            "explicit observed teacher-label indicators, not rewards; no graph contrast updates"
        ),
        ("query_boundary"): (
            "live state and working trace reset, fast associative residual reset; "
            "durable matrix retained; every neuron equation independently checked"
        ),
        "store_capacity": "650x36 persistent synapses plus one650x36 fast stream matrix; fixed",
        ("rehearsal"): (
            "one extra observed-label write per present old anchor each round; "
            "every replay exposure charged"
        ),
        ("confirmation"): (
            "independentTRAIN18/DEV19 read once only after24 passes; final heldout remains sealed"
        ),
        ("retention_gate"): (
            "all declared rounds completed, initialold4/4, "
            "finalold>=3/4,new>=15/20, no refused answers"
        ),
        "retention_query_rounds": [0, 16, 32, 64, args.rounds],
        ("scope"): (
            "associative-store controls; success does not discharge contrast "
            "learning or establish generalization"
        ),
        "memory_defaults": make_brain(args.seed, args.consolidation).hippocampus.to_dict(),
    }
    if args.relations:
        import relations

        train_panel, development_panel = (
            relations.load_panel("train"),
            relations.load_panel("development"),
        )
        protocol.update(
            fixture_kind="declared identifiable cue relations",
            relations_protocol_sha256=relations.protocol_hash(),
            relations_generator_sha256=sha256(relations.__file__),
            acquisition_gates=(
                "fresh2/4 families perfect all4TRAIN instances; fresh24>=18 family "
                "credits; zero refusals"
            ),
            development_gate="at least36/48 independentDEV correct; zero refusals",
            retention_gate=(
                "old>=3/4 andnew>=15/20 families, all2DEV instances correct per "
                "family; all rounds completed; zero refusals"
            ),
        )
    write_json(args.out / "protocol.json", protocol)
    harness.freeze_sources(args.out / "source", args.fixture)
    shutil.copyfile(__file__, args.out / "source/continual.py")
    if args.relations:
        relations.freeze(args.out / "source/relations")
    results = []
    for arm in args.arms:
        deadline = time.monotonic() + args.seconds_per_arm
        outcome = {"arm": arm, "stages": [], "status": "running"}
        results.append(outcome)
        if args.retention_from:
            source = args.retention_from / f"{arm}-4/final.npz"
            source_receipt = json.loads((source.parent / "receipt.json").read_text())
            if sha256(source) != source_receipt["checkpoint_sha256"]:
                raise ValueError("acquired anchor checkpoint hash differs")
            brain = Brain.load(source)
            if args.relations:
                report = retention_memory(
                    brain,
                    train_panel["inputs"],
                    train_panel["labels"],
                    arm == "memory-rehearsal",
                    args.out / arm,
                    args,
                    deadline,
                    development=development_panel,
                    families=train_panel["families"],
                )
            else:
                report = retention_memory(
                    brain,
                    data["school_inputs"],
                    data["school_labels"],
                    arm == "memory-rehearsal",
                    args.out / arm,
                    args,
                    deadline,
                )
            outcome.update(
                status=report["status"],
                passed=report["passed"],
                anchor_source_sha256=sha256(source),
                endpoint={
                    name: report["recall"][-1][name].get(
                        "family_credits", report["recall"][-1][name]["correct"]
                    )
                    for name in ("old", "new")
                },
                memory_writes=report["memory_writes"],
                rehearsal_exposures=report["rehearsal_exposures"],
                saved_durable_recall_equal=report["saved_durable_recall_equal"],
                continued_arrays_equal=report["continued_arrays_equal"],
            )
            write_json(
                args.out / "summary.json",
                {"protocol_sha256": sha256(args.out / "protocol.json"), "outcomes": results},
            )
            print(json.dumps(outcome), flush=True)
            continue
        for positions in ([0, 9], ANCHORS, list(range(24))):
            brain = make_brain(args.seed, args.consolidation)
            families = None
            if args.relations:
                panel = relations.load_panel("train", families=positions)
                x, y, families = panel["inputs"], panel["labels"], panel["families"]
                memory_positions = families.tolist()
            else:
                x, y = data["school_inputs"][positions], data["school_labels"][positions]
                memory_positions = positions
            folder = args.out / f"{arm}-{len(positions)}"
            report = train_memory(
                brain,
                x,
                y,
                memory_positions,
                arm == "memory-rehearsal",
                folder,
                args,
                deadline,
                families=families,
            )
            final = report["recall"][-1]
            stage = {
                "examples": len(positions),
                "status": report["status"],
                "correct": final["correct"],
                "refusals": final["refusals"],
                "writes": report["memory_writes"],
                "rehearsal_exposures": report["rehearsal_exposures"],
            }
            outcome["stages"].append(stage)
            if families is not None:
                stage["family_credits"] = final["family_credits"]
            loaded = Brain.load(folder / "final.npz")
            replay = durable_recall(loaded, x, y)
            stage["saved_durable_recall_equal"] = replay["predictions"] == final["predictions"]
            stage["validation_query_work"] = replay["work"]
            if report["status"] != "acquired":
                outcome["status"] = (
                    "prerequisite_failed" if len(positions) < 24 else "screen_failed"
                )
                break
            if len(positions) == 24:
                if args.relations:
                    stage["development"] = durable_recall(
                        brain, development_panel["inputs"], development_panel["labels"]
                    )
                    stage["development"].update(
                        family_credits(
                            stage["development"],
                            development_panel["labels"],
                            development_panel["families"],
                        )
                    )
                    stage["development_passed"] = (
                        stage["development"]["correct"] >= 36
                        and stage["development"]["refusals"] == 0
                    )
                else:
                    for panel in ("independent_train", "development"):
                        stage[panel] = durable_recall(
                            brain, data[panel + "_inputs"], data[panel + "_labels"]
                        )
                outcome["status"] = "screen_passed_associative_control_only"
        write_json(
            args.out / "summary.json",
            {"protocol_sha256": sha256(args.out / "protocol.json"), "outcomes": results},
        )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "status": outcome["status"],
                    "stages": [
                        {
                            k: stage[k]
                            for k in (
                                "examples",
                                "status",
                                "correct",
                                "writes",
                                "rehearsal_exposures",
                            )
                        }
                        for stage in outcome["stages"]
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
