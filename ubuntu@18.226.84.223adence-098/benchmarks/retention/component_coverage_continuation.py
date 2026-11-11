"""Prepare one unchanged-gene #85 actual component-experience continuation.

No scientific work during preparation. Execute only the reviewed frozen capsule.
The original negative chamber stays immutable, including its unrun lives.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import actual_outcome_chamber as chamber
import component_coverage_inputs as inputs
import numpy as np

SCHEMA = "cadence-actual-component-coverage/1"
SECONDS, WATCHDOG_SECONDS, OUTPUT_MIB = 600, 630, 192
RESERVE_BYTES = 8 * 1024**2
CURVE = (32, 64, 128)
SEAMS = (("continued", 128), ("revision", 128), ("restored", 128), ("restored", 256))
PRODUCERS = (
    "component_coverage_continuation.py",
    "component_coverage_inputs.py",
    "COMPONENT_COVERAGE_PROTOCOL.md",
    "actual_outcome_chamber.py",
    "chamber_inputs.py",
)


def declared_lives():
    return [
        {
            "specimen": f"{condition}-{exposure}",
            "seed": 0,
            "condition": condition,
            "exposure": exposure,
            "arm": arm,
            "required": True,
            "planned_batches": 768,
            "actual_batches": 0,
            "executed_batches": 0,
            "pending_outcome": None,
            "status": "unrun",
            "complete": False,
            "passed": None,
            "unknown_current_work": False,
            "phases": {
                name: {
                    "planned_batches": 256,
                    "actual_batches": 0,
                    "executed_batches": 0,
                    "status": "unrun",
                }
                for name in inputs.PHASES
            },
        }
        for condition, exposure in inputs.SPECIMENS
        for arm in inputs.ARMS
    ]


def fixed_fields():
    return {
        "schema": SCHEMA,
        "mode": "reused-founder-development",
        "founders": [0],
        "lives": declared_lives(),
        "gates": chamber.GATES,
        "worker_seconds": SECONDS,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "output_cap_mib": OUTPUT_MIB,
        "output_reserve_bytes": RESERVE_BYTES,
        "storage_reconciliation_seconds": 5,
        "main_streams": 8,
        "teacher_calls": 0,
        "terminal_done": True,
        "decay": 0,
        "world": inputs.WORLD,
        "curve_batches": list(CURVE),
        "pending_seams": [list(value) for value in SEAMS],
        "planned_main_batches": 6144,
        "planned_main_outcomes": 49152,
        "planned_cold_panels": 200,
        "planned_cold_rows": 6600,
        "planned_cold_free_calls": 1000,
        "planned_pending_seams": 32,
        "planned_reprocessed_feedback_calls": 64,
        "planned_branch_only_pending_actions": 64,
        "planned_private_imagination_calls": 32,
        "planned_deliberate_refusals": 32,
        "planned_duplicate_refusals": 32,
    }


def specimen_rows(summary):
    rows = [row for row in summary["lives"] if row["complete"]]
    if [(r["condition"], r["exposure"]) for r in rows] != list(inputs.SPECIMENS):
        raise ValueError("reference requires all four declared completed specimens")
    if any(row["seed"] != 0 or row["actual_batches"] != row["planned_batches"] for row in rows):
        raise ValueError("reference completed life identity/census differs")
    return rows


def same_baseline(reading, expected):
    return bool(
        reading["qualified_free"] == expected["qualified_free"]
        and reading["refused"] == expected["refused"]
        and reading["planned"] == reading["attempted"] == expected["planned"]
        and reading["unrun"] == expected["unrun"] == 0
        and all(
            q["prototype_prediction"] == e["prototype_prediction"]
            and np.array_equal(q["predictions"], e["predictions"])
            and np.allclose(q["margins"], e["margins"], rtol=0, atol=1e-12)
            and np.allclose(q["memory_values"], e["memory_values"], rtol=0, atol=1e-12)
            for q, e in zip(reading["cues"], expected["cues"], strict=True)
        )
    )


def next_observations(arrays, phase, condition, arm, batch):
    if batch < 255:
        return arrays[f"{condition}/{phase}/{arm}"][batch + 1]
    position = inputs.PHASES.index(phase) + 1
    if position < len(inputs.PHASES):
        return arrays[f"{condition}/{inputs.PHASES[position]}/{arm}"][0]
    # Declared terminal afterlife observation; no subsequent world action/reward.
    return np.repeat(chamber.inputs.prototypes(condition)[[3]], 8, axis=0)


def row_gate(row):
    required = set(inputs.PHASES)
    if (
        not row.get("complete")
        or not required.issubset(row.get("endpoints", {}))
        or row.get("actual_batches") != row["planned_batches"]
        or row.get("executed_batches") != row["planned_batches"]
        or row.get("pending_outcome") is not None
        or not row.get("baseline_reproduced")
        or not row.get("starting_full_array_parity")
    ):
        return {"passed": False, "reason": "incomplete-census-or-baseline"}
    baseline_rare = row["endpoints"]["initial"]["intact"]["cues"][2]["correct"]
    endpoints_passed, rare_drop = True, 0
    for name in inputs.PHASES:
        reading = row["endpoints"][name]["intact"]
        endpoints_passed &= bool(
            reading["qualified_free"]
            and not reading["refused"]
            and not reading["unrun"]
            and all(
                q["correct"] >= chamber.GATES["variants_correct"]
                and q["prototype_correct"] == 1
                and not q["refused"]
                and not q["unrun"]
                and (q["cue"] == 2 or q["obsolete"] <= chamber.GATES["obsolete_max"])
                for q in reading["cues"]
            )
        )
        rare_drop = max(rare_drop, baseline_rare - reading["cues"][2]["correct"])
    custody = [(s.get("phase"), s.get("batch")) for s in row.get("seams", [])] == list(
        SEAMS
    ) and all(seam["passed"] is True for seam in row["seams"])
    panels_present = all(
        {"intact", "graph_only", "initial_graph", "joint_reset", "uniform"}.issubset(
            row["endpoints"].get(name, {})
        )
        for name in ("initial", *inputs.PHASES)
    ) and all(
        {"0", "32", "64", "128", "256"}.issubset(row.get("curves", {}).get(name, {}))
        for name in inputs.PHASES
    )
    all_probes = panels_present and chamber.probes_complete(
        [row["endpoints"], row.get("curves", {})]
    )
    return {
        "passed": bool(endpoints_passed and rare_drop <= 1 and custody and all_probes),
        "endpoint_gates_passed": endpoints_passed,
        "rare_max_drop": rare_drop,
        "custody_passed": custody,
        "all_planned_probes_qualified": all_probes,
    }


def cohort(lives, arm):
    correct = uniform = reset = planned = 0
    count = 0
    for row in lives:
        if row["arm"] != arm:
            continue
        for phase in inputs.PHASES:
            count += 1
            planned += 30
            endpoint = row.get("endpoints", {}).get(phase)
            if endpoint is None:
                continue
            correct += sum(q["correct"] for q in endpoint["intact"]["cues"])
            uniform += endpoint["uniform"]["correct"]
            reset += sum(q["correct"] for q in endpoint["joint_reset"]["cues"])
    # Missing panels never shrink the fixed denominator or earn a baseline pass.
    complete = count == 12 and all(
        set(inputs.PHASES).issubset(row.get("endpoints", {})) for row in lives if row["arm"] == arm
    )
    return {
        "planned": planned,
        "correct": correct,
        "uniform_correct": uniform,
        "joint_reset_correct": reset,
        "complete": complete,
        "gap_uniform": (correct - uniform) / planned if planned else None,
        "gap_joint_reset": (correct - reset) / planned if planned else None,
        "passed": bool(
            complete
            and (correct - uniform) / planned >= 0.20
            and (correct - reset) / planned >= 0.20
        ),
    }


def prepare(root, reference):
    began = time.monotonic()
    if root.exists():
        raise ValueError("continuation needs a new output directory")
    if any(os.environ.get(name) != "1" for name in chamber.THREADS):
        raise ValueError("all declared numerical threads must equal1")
    original = json.loads((reference / "protocol.json").read_text())
    summary = json.loads((reference / "summary.json").read_text())
    rows = specimen_rows(summary)
    for name, expected in (original["source_files"] | original["admitted_artifacts"]).items():
        if chamber.digest(reference / name) != expected:
            raise ValueError("original negative source/artifact changed: " + name)
    if (
        original["gates"] != chamber.GATES
        or original["runtime"] != chamber.runtime()
        or chamber.model_identity(chamber.make_brain(0)) != original["initial_models"]["0"]
    ):
        raise ValueError("original gene/initial arrays/runtime drifted")
    package = Path(chamber.cadence.__file__).resolve().parent
    current_library = {
        "source/library/cadence/" + p.relative_to(package).as_posix(): chamber.digest(p)
        for p in package.rglob("*.py")
    }
    if current_library != {
        k: v for k, v in original["source_files"].items() if k.startswith("source/library/cadence/")
    }:
        raise ValueError("continuation library differs from the retained negative")
    root.mkdir(parents=True)
    source_files, reference_files = {}, {}
    for name in current_library:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
        source_files[name] = chamber.digest(target)
    for name in PRODUCERS:
        target = root / "source" / name
        shutil.copyfile(Path(__file__).with_name(name), target)
        source_files["source/" + name] = chamber.digest(target)
    repo = Path(__file__).resolve().parents[2]
    for name in (
        "test_component_coverage_continuation.py",
        "test_retention_chamber.py",
        "test_retention_amplitude.py",
        "test_actual_outcome_association.py",
    ):
        target = root / "source/contracts" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(repo / "tests" / name, target)
        source_files[target.relative_to(root).as_posix()] = chamber.digest(target)
    names = {"protocol.json", "summary.json", "journal.jsonl.gz", "inputs-0.npz", "initial-0.npz"}
    names.update(f"seed-0/{r['condition']}-{r['exposure']}/restored.npz" for r in rows)
    for name in sorted(names):
        target = root / "reference" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
        reference_files["reference/" + name] = chamber.digest(target)
    arrays = inputs.frozen_arrays()
    np.savez_compressed(root / "continuation-inputs.npz", **arrays)
    reserved = chamber.inputs.load(root / "reference/inputs-0.npz", 0)
    audit = inputs.coverage_audit(arrays, reserved)
    chamber.atomic_json(root / "input-audit.json", audit)
    models = {}
    for condition, exposure in inputs.SPECIMENS:
        name = f"{condition}-{exposure}"
        brain = chamber.Brain.load(root / f"reference/seed-0/{name}/restored.npz")
        # the learner has no leak (a leak was removed from the library after 0.84.3)
        if brain._moment is not None:
            raise ValueError("reference final specimen is not the accepted decay0 life")
        models[name] = chamber.model_identity(brain)
    protocol = {
        **fixed_fields(),
        "runtime": original["runtime"],
        "interpreter_sha256": chamber.digest(Path(sys.executable)),
        "source_files": source_files,
        "reference_files": reference_files,
        "admitted_artifacts": {
            name: chamber.digest(root / name)
            for name in ("continuation-inputs.npz", "input-audit.json")
        },
        "initial_models": models,
        "initial_model": original["initial_models"]["0"],
        "reference_origin": str(reference.resolve()),
        "immutable_original_failure": {
            k: summary[k]
            for k in ("complete", "passed", "status", "unrun_batches", "output_cap_exceeded")
        },
        "scope": "one reused-founder causal development; no whole85/confirmation claim",
        "capacity": {"C": [8, 2], "strength_C_plus_F": [8, 8, 2]},
        "extra_information": "actual partial TRAIN experience atamplitude1; heldout.75/1.25",
        "sequence_acceptance": "remaining; terminal isolated odors",
        "integrated_replay": "remaining; no replay in this pairedcontinuation",
        "eligibility_scope": (
            "finite12-step actor eligibility/finite bootstrap residuals retained; "
            "only1024/.003 public free answers require equation qualification"
        ),
    }
    chamber.atomic_json(root / "protocol.json", protocol)
    chamber.atomic_json(
        root / "job-protocol.json",
        {
            "schema": "cadence.component-coverage-job/1",
            "status": "prepared-not-executed",
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "helper_sha256": source_files["source/component_coverage_continuation.py"],
            "interpreter_sha256": protocol["interpreter_sha256"],
            "science_calls": 0,
            "planned_main_batches": 6144,
        },
    )
    chamber.atomic_json(
        root / "admission.json",
        {
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "preparation_seconds": time.monotonic() - began,
            "final_admission_io_allowance_seconds": 1,
            "science_calls": 0,
        },
    )
    guard_work = {}
    preflight(root, worker=False, work=guard_work)
    chamber.atomic_json(
        root / "admission.json",
        {
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "preparation_seconds": time.monotonic() - began,
            "final_admission_io_allowance_seconds": 1,
            "science_calls": 0,
            "source_admission_work": guard_work,
            "brain_constructor_calls": 1,
            "checkpoint_load_calls": 9,
            "copied_source_files": len(source_files),
            "copied_reference_files": len(reference_files),
            "copied_source_reference_bytes": sum(
                (root / name).stat().st_size for name in source_files | reference_files
            ),
            "prepared_input_logical_bytes": sum(a.nbytes for a in arrays.values()),
            "heldout_probe_changes": 0,
            "new_solves": 0,
            "new_learning": 0,
            "new_actual_outcomes": 0,
        },
    )
    return protocol


def preflight(root, *, worker=True, work=None):
    began = time.monotonic()
    work = {} if work is None else work
    work["source_admission_calls"] = work.get("source_admission_calls", 0) + 1

    def digest(path):
        result = chamber.digest(path)
        work["admission_hash_reads"] = work.get("admission_hash_reads", 0) + 1
        work["admission_hashed_bytes"] = (
            work.get("admission_hashed_bytes", 0) + Path(path).stat().st_size
        )
        return result

    def load_model(path):
        result = chamber.Brain.load(path)
        work["admission_checkpoint_reads"] = work.get("admission_checkpoint_reads", 0) + 1
        work["admission_checkpoint_bytes"] = (
            work.get("admission_checkpoint_bytes", 0) + path.stat().st_size
        )
        return result

    protocol = json.loads((root / "protocol.json").read_text())
    if any(protocol.get(k) != v for k, v in fixed_fields().items()):
        raise ValueError("frozen continuation scope/gates/resources differ")
    admission = json.loads((root / "admission.json").read_text())
    job = json.loads((root / "job-protocol.json").read_text())
    if (
        admission["protocol_sha256"] != digest(root / "protocol.json")
        or job["protocol_sha256"] != digest(root / "protocol.json")
        or job["helper_sha256"]
        != protocol["source_files"]["source/component_coverage_continuation.py"]
        or protocol["runtime"] != chamber.runtime()
        or protocol["interpreter_sha256"] != digest(Path(sys.executable))
    ):
        raise ValueError("continuation admission/source job/runtime differs")
    reserved = admission["preparation_seconds"] + admission["final_admission_io_allowance_seconds"]
    if not np.isfinite(reserved) or not 0 <= reserved < SECONDS:
        raise ValueError("invalid preparation/resource allowance")
    for name, expected in (
        protocol["source_files"] | protocol["reference_files"] | protocol["admitted_artifacts"]
    ).items():
        if digest(root / name) != expected:
            raise ValueError("continuation frozen source/reference/input changed: " + name)
    imported = {
        "component_coverage_continuation.py": Path(__file__),
        "component_coverage_inputs.py": Path(inputs.__file__),
        "actual_outcome_chamber.py": Path(chamber.__file__),
        "chamber_inputs.py": Path(chamber.inputs.__file__),
    }
    if any(
        digest(path) != protocol["source_files"]["source/" + name]
        for name, path in imported.items()
    ):
        raise ValueError("imported continuation helper differs")
    package = Path(chamber.cadence.__file__).resolve().parent
    if worker and package != (root / "source/library/cadence").resolve():
        raise ValueError("continuation must import the frozen library")
    current = {
        "source/library/cadence/" + p.relative_to(package).as_posix(): digest(p)
        for p in package.rglob("*.py")
    }
    if current != {
        k: v for k, v in protocol["source_files"].items() if k.startswith("source/library/cadence/")
    }:
        raise ValueError("imported continuation library differs")
    original = json.loads((root / "reference/protocol.json").read_text())
    summary = json.loads((root / "reference/summary.json").read_text())
    specimen_rows(summary)
    if original["gates"] != chamber.GATES or original["runtime"] != protocol["runtime"]:
        raise ValueError("reference gates/runtime differs")
    reserved_inputs = chamber.inputs.load(root / "reference/inputs-0.npz", 0)
    inputs.coverage_audit(inputs.load(root / "continuation-inputs.npz"), reserved_inputs)
    if (
        chamber.model_identity(load_model(root / "reference/initial-0.npz"))
        != protocol["initial_model"]
    ):
        raise ValueError("reference newborn model differs")
    for specimen, expected in protocol["initial_models"].items():
        brain = load_model(root / f"reference/seed-0/{specimen}/restored.npz")
        if chamber.model_identity(brain) != expected:
            raise ValueError("complete acquired specimen model differs")
    if chamber.tree_bytes(root) >= OUTPUT_MIB * 1024**2 - RESERVE_BYTES:
        raise ValueError("prepared continuation has no output reserve")
    work["source_admission_seconds"] = (
        work.get("source_admission_seconds", 0.0) + time.monotonic() - began
    )
    return protocol


class Journal(chamber.Journal):
    def __init__(self, root, protocol, began):
        self.root, self.protocol, self.began = root, protocol, began
        self.last_progress, self.serial, self.totals = began, 0, {}
        self.work_by_arm, self.file_sizes, self.output_bytes = {}, {}, 0
        self.last_storage_reconciliation = began
        self.resource_accounting = {
            "managed_file_size_checks": 0,
            "full_reconciliations": 0,
            "files_scanned": 0,
            "check_seconds": 0.0,
            "largest_managed_operation_bytes": 0,
        }
        self.active_index, self.active_brain = None, None
        self.reconcile_storage()
        self.summary = {
            "schema": SCHEMA,
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "lives": declared_lives(),
            "status": "running",
            "complete": False,
            "passed": False,
            "whole_issue_passed": False,
            "process_failure": None,
            "current_operation": None,
            "unknown_current_work": False,
            "completed_calls": 0,
            "life_denominator": 8,
            "sequence_acceptance": "remaining",
            "integrated_replay": "remaining",
        }
        self.persist()

    def completed(self, operation, context, body, work=None):
        # Executed world outcomes and accepted feedback have distinct custody.
        # Both commits precede the append/post-call cap guard.
        if (
            operation in ("actual_world_event", "actual_feedback")
            and body["accepted"]
            and context.get("main_life")
            and "branch" not in context
        ):
            row = self.summary["lives"][context["life"]]
            witness, phase = body["witness"], context["phase"]
        else:
            super().completed(operation, context, body, work)
            return
        if operation == "actual_feedback":
            row["actual_batches"] += 1
            row["phases"][phase]["actual_batches"] += 1
            row["pending_outcome"] = None
        else:
            row["executed_batches"] += 1
            row["phases"][phase]["executed_batches"] += 1
            row["pending_outcome"] = {"phase": phase, "batch": context["batch"], "witness": witness}
            stats = row["actual_world"][phase]
            identities = np.asarray(witness["identities"])
            actions, reward = np.asarray(witness["actions"]), np.asarray(witness["reward"])
            stats["events"] += len(actions)
            stats["reward_sum"] += float(reward.sum())
            stats["non_neutral"] += int((identities != 3).sum())
            stats["correct_non_neutral"] += int(
                (
                    (actions == chamber.inputs.MAPPING[inputs.WORLD[phase], identities])
                    & (identities != 3)
                ).sum()
            )
            for cue, action in zip(identities, actions, strict=True):
                stats["actions_per_cue"][cue][action] += 1
        super().completed(operation, context, body, work)


def sampled_world_event(journal, brain, key, identities, world, following, context):
    action, refused = chamber.charge(
        journal, "sampled_act", context, lambda: brain.act(key, greedy=False)
    )
    if refused:
        return None
    if (
        brain._moment is None
        or not np.array_equal(brain._moment[0], key)
        or not np.array_equal(brain._moment[1], action)
    ):
        raise AssertionError("public pending proposal differs from executed world action")
    # Preflight before executing the action in the world. No world outcome exists
    # if this guard stops after a sampled proposal but before actual execution.
    journal.begin("actual_world_event", context)
    reward = chamber.inputs.actual_reward(identities, action, world)
    witness = {
        "key": key.copy(),
        "identities": identities.copy(),
        "actions": action.copy(),
        "reward": reward,
        "world": world,
        "done": np.ones(8, bool),
        "event_ids": [f"{context}/{row}" for row in range(8)],
        "probabilities": brain.basal_ganglia.probabilities(brain.basal_ganglia.state),
    }
    journal.completed(
        "actual_world_event",
        context,
        {"accepted": True, "witness": witness},
        {"actual_world_event_calls": 1, "actual_world_outcome_rows": 8},
    )
    return witness


def life(journal, index, arrays, reserved, baselines):
    row = journal.summary["lives"][index]
    condition, exposure, arm = row["condition"], row["exposure"], row["arm"]
    prefix = f"lives/{condition}-{exposure}/{arm}"
    initial = journal.root / "reference/initial-0.npz"
    anchor = journal.root / f"reference/seed-0/{condition}-{exposure}/restored.npz"
    brain = journal.load(anchor)
    journal.active_index, journal.active_brain = index, brain
    fixed_capacity = chamber.capacity(brain)
    row.update(
        status="running",
        endpoints={},
        curves={},
        seams=[],
        actual_world={
            name: {
                "events": 0,
                "reward_sum": 0.0,
                "non_neutral": 0,
                "correct_non_neutral": 0,
                "actions_per_cue": [[0, 0] for _ in range(4)],
            }
            for name in inputs.PHASES
        },
    )
    context = {"life": index, "specimen": row["specimen"], "arm": arm, "main_life": True}
    starting_checkpoint, baseline = chamber.probe_endpoint(
        journal, brain, initial, reserved, condition, 0, prefix + "/initial", context
    )
    row["endpoints"]["initial"] = baseline
    row["starting_full_array_parity"] = chamber.checkpoints_equal(starting_checkpoint, anchor)
    if not row["starting_full_array_parity"]:
        raise AssertionError("complete loaded specimen changed before continuation")
    row["baseline_reproduced"] = same_baseline(baseline["intact"], baselines[row["specimen"]])
    if not row["baseline_reproduced"]:
        raise AssertionError("original retained baseline did not reproduce")
    previous = baseline["intact"]
    for phase in inputs.PHASES:
        world, identities = inputs.WORLD[phase], arrays[phase + "/identities"]
        keys = arrays[f"{condition}/{phase}/{arm}"]
        row["phases"][phase]["status"] = "running"
        row["curves"][phase] = {"0": chamber.rescore_previous(previous, world)}
        journal.completed(
            "world_change_rescore",
            {**context, "phase": phase},
            row["curves"][phase]["0"],
            {"reused_probe_comparisons": 33},
        )
        for batch in range(256):
            detail = {**context, "phase": phase, "batch": batch + 1}
            row.update(
                current_batch={"phase": phase, "batch": batch + 1}, unknown_current_work=True
            )
            following = next_observations(arrays, phase, condition, arm, batch)
            witness = sampled_world_event(
                journal, brain, keys[batch], identities[batch], world, following, detail
            )
            if witness is None:
                row.update(status="refused_action", passed=False, unknown_current_work=False)
                row["phases"][phase]["status"] = "refused_action"
                journal.save(brain, prefix + "/refused-main-action.npz")
                journal.active_index, journal.active_brain = None, None
                return
            if (phase, batch + 1) in SEAMS:
                row["seams"].append(
                    chamber.checkpoint_seam(
                        journal,
                        brain,
                        witness,
                        following,
                        prefix + f"/seam-{phase}-{batch + 1}",
                        detail,
                    )
                    | {"phase": phase, "batch": batch + 1}
                )
            _, refused = chamber.actual_feedback(journal, brain, witness, following, detail)
            if refused:
                row.update(status="refused_feedback", passed=False, unknown_current_work=False)
                row["phases"][phase]["status"] = "refused_feedback"
                journal.save(brain, prefix + "/refused-main-feedback.npz")
                journal.active_index, journal.active_brain = None, None
                return
            if chamber.capacity(brain) != fixed_capacity:
                raise AssertionError("continuing acquired capacity changed")
            row.update(current_batch=None, unknown_current_work=False)
            if batch + 1 in CURVE:
                _, reading = chamber.probe_endpoint(
                    journal,
                    brain,
                    initial,
                    reserved,
                    condition,
                    world,
                    prefix + f"/curve-{phase}-{batch + 1}",
                    detail,
                    lesions=False,
                )
                row["curves"][phase][str(batch + 1)] = reading["intact"]
            journal.persist()
        _, result = chamber.probe_endpoint(
            journal,
            brain,
            initial,
            reserved,
            condition,
            world,
            prefix + "/" + phase,
            {**context, "phase": phase},
        )
        row["endpoints"][phase] = result
        row["curves"][phase]["256"] = result["intact"]
        row["phases"][phase]["status"] = "completed"
        previous = result["intact"]
        journal.persist(full=True)
    row.update(complete=True, status="completed", current_batch=None, unknown_current_work=False)
    row["gate"] = row_gate(row)
    row["passed"] = row["gate"]["passed"]
    journal.active_index, journal.active_brain = None, None
    journal.persist(full=True)


def finish(journal):
    summary = journal.summary
    lives = summary["lives"]
    summary["cohorts"] = {arm: cohort(lives, arm) for arm in inputs.ARMS}
    summary["unrun_batches"] = sum(r["planned_batches"] - r["actual_batches"] for r in lives)
    summary["unexecuted_world_batches"] = sum(
        r["planned_batches"] - r["executed_batches"] for r in lives
    )
    summary["executed_without_feedback_batches"] = sum(
        r["executed_batches"] - r["actual_batches"] for r in lives
    )
    summary["complete"] = all(r["complete"] for r in lives)
    summary["passed"] = bool(
        summary["complete"]
        and not summary["process_failure"]
        and not summary["unknown_current_work"]
        and all(r["passed"] is True for r in lives if r["arm"] == "component-coverage")
        and all(
            r.get("gate", {}).get("custody_passed") is True
            and r.get("gate", {}).get("all_planned_probes_qualified") is True
            for r in lives
        )
        and summary["cohorts"]["component-coverage"]["passed"]
    )
    summary["status"] = (
        "development_passed"
        if summary["passed"]
        else ("development_failed" if summary["complete"] else "incomplete_census")
    )
    summary["seconds"] = time.monotonic() - journal.began
    summary["output_bytes"] = chamber.tree_bytes(journal.root)
    if summary["seconds"] >= SECONDS or summary["output_bytes"] >= OUTPUT_MIB * 1024**2:
        summary.update(passed=False, status="resource_cap_exceeded")
    journal.persist(full=True)
    final_bytes = chamber.tree_bytes(journal.root)
    final_seconds = time.monotonic() - journal.began
    if final_bytes >= OUTPUT_MIB * 1024**2 or final_seconds >= SECONDS:
        summary.update(
            passed=False,
            status="resource_cap_exceeded",
            output_bytes=final_bytes,
            seconds=final_seconds,
            final_flush_resource_qualified=False,
        )
        journal.persist(full=True)
    return summary


def worker(root):
    guard_work = {}
    protocol = preflight(root, work=guard_work)
    marker = json.loads((root / "execution-started.json").read_text())
    if marker["protocol_sha256"] != chamber.digest(root / "protocol.json"):
        raise ValueError("execution marker differs")
    admission = json.loads((root / "admission.json").read_text())
    began = marker["monotonic_start"] - admission["preparation_seconds"] - 1
    journal = Journal(root, protocol, began)
    journal.completed("source_admission", {"stage": "worker-entry"}, {}, guard_work)
    reserved = chamber.inputs.load(root / "reference/inputs-0.npz", 0)
    arrays = inputs.load(root / "continuation-inputs.npz")
    summary = json.loads((root / "reference/summary.json").read_text())
    baselines = {
        f"{r['condition']}-{r['exposure']}": r["endpoints"]["restored"]["intact"]
        for r in specimen_rows(summary)
    }
    try:
        for index in range(len(journal.summary["lives"])):
            life(journal, index, arrays, reserved, baselines)
        journal.bounds(storage=True)
        final_guard_work = {}
        journal.begin("source_admission", {"stage": "worker-final"})
        preflight(root, work=final_guard_work)
        journal.completed("source_admission", {"stage": "worker-final"}, {}, final_guard_work)
    except BaseException as error:
        journal.summary.update(
            process_failure={"type": type(error).__name__, "message": str(error)}
        )
        for row in journal.summary["lives"]:
            if row["status"] == "running":
                row.update(
                    status="interrupted",
                    complete=False,
                    passed=None,
                    unknown_current_work=journal.summary["unknown_current_work"],
                )
        if journal.active_brain is not None:
            # Reserve-only custody write: no new dynamics, reward or learning.
            path = root / "interrupted-main.npz"
            journal.active_brain.save(path)
            journal.track_file(path)
            journal.summary["interrupted_state"] = {
                "life": journal.active_index,
                "path": path.name,
                "sha256": chamber.digest(path),
                "sampled_proposal_without_executed_outcome": (
                    journal.active_brain._moment is not None
                    and journal.summary["lives"][journal.active_index]["pending_outcome"] is None
                ),
            }
            journal.totals["reserve_tail_checkpoint_writes"] = 1
        raise
    finally:
        finish(journal)
    return journal.summary


def close_execution_receipt(root, began, preparation_seconds, receipt):
    """Include receipt flushes in the final bound and bind any failed summary."""
    summary_path, receipt_path = root / "summary.json", root / "execution.json"

    def store_receipt():
        receipt.update(
            summary_sha256=chamber.digest(summary_path), launch_seconds=time.monotonic() - began
        )
        chamber.atomic_json(receipt_path, receipt)

    store_receipt()
    for _ in range(2):
        seconds = time.monotonic() - began + preparation_seconds
        retained = chamber.tree_bytes(root)
        bad = seconds >= SECONDS or retained >= OUTPUT_MIB * 1024**2
        receipt["final_resources"] = {
            "total_seconds_including_preparation": seconds,
            "retained_bytes": retained,
            "qualified": not bad,
        }
        if bad:
            summary = json.loads(summary_path.read_text())
            summary.update(
                passed=False,
                status="resource_cap_exceeded",
                seconds=seconds,
                output_bytes=retained,
                final_flush_resource_qualified=False,
            )
            chamber.atomic_json(summary_path, summary)
        # A resource failure may have changed the summary; bind its new bytes.
        store_receipt()
        if not bad and (
            time.monotonic() - began + preparation_seconds >= SECONDS
            or chamber.tree_bytes(root) >= OUTPUT_MIB * 1024**2
        ):
            continue
        return json.loads(summary_path.read_text())
    # Crossing on the last final write still cannot retain a positive claim.
    summary = json.loads(summary_path.read_text())
    summary.update(
        passed=False, status="resource_cap_exceeded", final_flush_resource_qualified=False
    )
    chamber.atomic_json(summary_path, summary)
    receipt["final_resources"]["qualified"] = False
    store_receipt()
    return summary


def launch(root):
    began = time.monotonic()
    launch_guard_work = {}
    protocol = preflight(root, worker=False, work=launch_guard_work)
    marker = {
        "protocol_sha256": chamber.digest(root / "protocol.json"),
        "monotonic_start": began,
        "one_attempt": True,
    }
    with (root / "execution-started.json").open("x") as stream:
        stream.write(chamber.canonical_json(marker) + "\n")
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update({name: "1" for name in chamber.THREADS})
    failure = None
    try:
        with (root / "worker.log").open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "source/component_coverage_continuation.py"),
                    "--worker",
                    "--out",
                    str(root),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=WATCHDOG_SECONDS,
                check=False,
            )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        summary = json.loads((root / "summary.json").read_text())
        if summary["protocol_sha256"] != chamber.digest(root / "protocol.json") or len(
            summary["lives"]
        ) != len(protocol["lives"]):
            raise RuntimeError("paired continuation census identity differs")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        try:
            summary = json.loads((root / "summary.json").read_text())
        except (OSError, ValueError):
            summary = {
                "schema": SCHEMA,
                "protocol_sha256": chamber.digest(root / "protocol.json"),
                "lives": declared_lives(),
                "work": {},
                "unknown_current_work": True,
            }
        for row in summary["lives"]:
            if row["status"] == "running":
                row.update(
                    status="interrupted", passed=None, complete=False, unknown_current_work=True
                )
        summary.update(
            passed=False, complete=False, status="process_failed", process_failure=failure
        )
        chamber.atomic_json(root / "summary.json", summary)
        raise
    finally:
        receipt = {
            "protocol_sha256": chamber.digest(root / "protocol.json"),
            "job_protocol_sha256": chamber.digest(root / "job-protocol.json"),
            "launch_seconds": time.monotonic() - began,
            "launch_source_admission_work": launch_guard_work,
            "failure": failure,
            "whole_issue_passed": False,
        }
        admission = json.loads((root / "admission.json").read_text())
        summary = close_execution_receipt(
            root,
            began,
            admission["preparation_seconds"] + admission["final_admission_io_allowance_seconds"],
            receipt,
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
    elif args.execute_reviewed:
        result = launch(root)
        print(chamber.canonical_json({k: result[k] for k in ("complete", "passed", "cohorts")}))
    elif args.prepare_only and args.reference:
        prepare(root, args.reference.resolve())
        print(
            chamber.canonical_json(
                {"prepared": str(root), "protocol_sha256": chamber.digest(root / "protocol.json")}
            )
        )
    else:
        parser.error("prepare needs reference; execution needs separately reviewed capsule")


if __name__ == "__main__":
    main()
