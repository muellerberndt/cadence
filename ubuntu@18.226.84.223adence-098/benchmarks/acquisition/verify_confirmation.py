"""Audit retained five-founder confirmation and export a small, evidence-backed demo.

The default audit recomputes scores, gates, schedules and work from retained records
and checks source, phase and checkpoint custody. It does not reconstruct historical
learning from compact phase hashes. Optional endpoint replay invokes public cold
actions using the run's archived library and independently checks the equations.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

SEEDS = (1, 2, 3, 4, 5)
OLD = (0, 9, 17, 22)
TOLERANCE = 0.003
SCHEMAS = (
    "cadence-online-half-step-five-founder-confirmation-v1",
    "cadence-online-half-step-five-founder-confirmation-v2",
    "cadence-online-half-step-five-founder-confirmation-v3",
    "cadence-online-half-step-five-founder-confirmation-v4",
    "cadence-online-half-step-five-founder-confirmation-v5",
    "cadence-online-half-step-five-founder-confirmation-v6",
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def safe_name(name):
    path = PurePosixPath(name)
    check(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
    return str(path)


class Artifacts:
    def __init__(self, root, seed, outcome):
        self.root = root
        self.files = None
        archive = root / f"seed-{seed}.tar.gz"
        if archive.exists():
            custody = json.loads((root / "custody" / f"seed-{seed}.json").read_text())
            archive_hash = digest(archive.read_bytes())
            check(archive_hash == custody["archive_sha256"], "archive hash differs")
            check(archive_hash == outcome["outer_archive_sha256"], "census archive hash differs")
            self.files = {}
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle.getmembers():
                    name = safe_name(member.name)
                    check(member.isdir() or member.isfile(), "archive contains a non-file member")
                    if member.isfile():
                        check(name not in self.files, "duplicate archive member")
                        self.files[name] = bundle.extractfile(member).read()
            check(
                {name: digest(raw) for name, raw in self.files.items()}
                == custody["original_member_sha256"],
                "archive member custody differs",
            )

    def read(self, name):
        name = safe_name(name)
        return self.files[name] if self.files is not None else (self.root / name).read_bytes()

    def exists(self, name):
        return name in self.files if self.files is not None else (self.root / name).exists()

    def json(self, name):
        return json.loads(self.read(name))

    def pin(self, name, expected):
        check(digest(self.read(name)) == expected, f"artifact hash differs: {name}")

    def arrays(self, name):
        with np.load(io.BytesIO(self.read(name)), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}


def panel_identity(task, split, old_only=False):
    counts = task["panel_observations_per_family"]
    families = np.repeat(np.arange(24), counts[split])
    variants = np.tile(np.arange(counts[split]), 24)
    labels = np.asarray(task["family_to_label"])[families]
    if old_only:
        keep = np.isin(families, OLD)
        families, variants, labels = (a[keep] for a in (families, variants, labels))
    return families, variants, labels


def frozen_order(task, count, stage, old_only=False):
    """Independently reconstruct the declared class-balanced four-variant schedule."""
    family_ids, variants, _ = panel_identity(task, "train", old_only)
    families = np.unique(family_ids)
    rng = np.random.default_rng(np.random.SeedSequence([1100301, stage]))
    visits = {int(family): 0 for family in families}
    rows = []
    while len(rows) < count:
        for family in rng.permutation(families):
            variant = visits[int(family)] % 4
            rows.append(int(np.flatnonzero((family_ids == family) & (variants == variant))[0]))
            visits[int(family)] += 1
            if len(rows) == count:
                break
    return rows


def query_audit(query, identity):
    families, _, labels = identity
    predictions = np.asarray(query["predictions"], dtype=int)
    check(predictions.shape == labels.shape, "query prediction census differs")
    check(np.all((predictions >= -1) & (predictions < 36)), "invalid query action")
    correct = predictions == labels

    def credit(ids):
        return sum(bool(np.all(correct[families == f])) for f in ids)

    expected = {
        "correct": int(correct.sum()),
        "examples": len(labels),
        "refusals": int((predictions == -1).sum()),
        "family_credits": credit(np.unique(families)),
    }
    for name, ids in (("old", OLD), ("new", tuple(i for i in range(24) if i not in OLD))):
        present = [f for f in ids if np.any(families == f)]
        if present:
            expected[name + "_family_credits"] = credit(present)
    for name, value in expected.items():
        check(query.get(name) == value, f"query {name} differs from raw predictions")
    rows = query["rows"]
    check(len(rows) == len(labels), "query row census differs")
    for index, (row, truth, answer) in enumerate(zip(rows, labels, predictions, strict=True)):
        check(row["row"] == index and row["label"] == truth, "query truth or row differs")
        check(row["prediction"] == answer, "query row prediction differs")
        if answer != -1:
            check(
                all(
                    math.isfinite(row[k]) and 0 <= row[k] <= TOLERANCE
                    for k in ("residual", "cache_defect")
                ),
                "accepted query fails its recorded equation gate",
            )
    work = {
        "calls": len(rows),
        "row_sweeps": sum(row["steps"] for row in rows),
        "reported_residual_checks": sum(row["residual_checks"] or 0 for row in rows),
        "independent_residual_checks": len(rows),
        "unreported_residual_work": any(row["residual_checks"] is None for row in rows),
    }
    if "reported_stagnation_checks" in query["work"]:
        work["reported_stagnation_checks"] = sum(row.get("stagnation_checks") or 0 for row in rows)
        work["unreported_stagnation_work"] = any(
            row.get("stagnation_checks") is None for row in rows
        )
    for name, value in work.items():
        check(query["work"].get(name) == value, f"query work differs: {name}")
    return expected


def lesson_audit(lesson, files, prefix):
    files.pin(f"{prefix}/{lesson['phase_file']}", lesson["phase_sha256"])
    payload = files.arrays(f"{prefix}/{lesson['phase_file']}")
    phases, report = lesson["phases"], lesson["report"]
    check(lesson["accepted"] in (0, 1), "invalid accepted lesson count")
    check(report["accepted"] == lesson["accepted"], "lesson admission count differs")
    if lesson["accepted"]:
        check(set(phases) == {"free", "nudged", "opposite"}, "accepted lesson misses a phase")
        check(not lesson["failed"], "accepted lesson reports failure")
    sweeps = checks = 0
    for name, phase in phases.items():
        check(name in ("free", "nudged", "opposite"), "unknown teaching phase")
        sweeps += phase["steps"]
        checks += report[name + "_residual_checks"]
        check(phase["steps"] == report[name + "_steps"], "phase work differs")
        if lesson["accepted"]:
            check(all(phase["qualified"]), "accepted phase is unqualified")
            check(
                all(
                    math.isfinite(v) and 0 <= v <= TOLERANCE
                    for key in ("full_residual", "cache_defect")
                    for v in phase[key]
                ),
                "accepted teaching fails its recorded equation gate",
            )
        for field, expected in phase.get("vector_sha256", {}).items():
            key = name + "_" + field
            if key in payload:
                observed = digest(np.ascontiguousarray(payload[key]).tobytes())
            else:
                check(key + "_sha256" in payload, "phase vector hash missing")
                observed = str(payload[key + "_sha256"].item())
            check(observed == expected, "phase vector hash differs")
    expected = {
        "phase_row_sweeps": sweeps,
        "reported_row_residual_checks": checks,
        "independent_residual_checks": len(phases),
        "accepted_row_exposures": lesson["accepted"],
    }
    check(report["total_steps"] == sweeps, "total teaching sweeps differ")
    check(report["total_residual_checks"] == checks, "total teaching checks differ")
    for name, value in expected.items():
        check(lesson[name] == value, f"lesson work differs: {name}")


def stage_gate(reading, initial, lesson, mixed):
    train, dev = reading["train"], reading["development"]
    small_family_count = train["examples"] // 4
    return bool(
        not train["refusals"]
        and not dev["refusals"]
        and train["correct"] > initial["train"]["correct"]
        and train["family_credits"] >= (18 if mixed else small_family_count)
        and dev["correct"] >= (36 if mixed else 2 * small_family_count)
        and (
            not mixed
            or (
                lesson >= 128 and dev["old_family_credits"] >= 3 and dev["new_family_credits"] >= 15
            )
        )
    )


def stage_audit(
    stage, files, prefix, protocol, task, mixed, partial=False, family_filter=None, order_key=None
):
    lessons, readings = stage["lessons"], stage["recall"]
    order = protocol[order_key or ("mixed_order" if mixed else "old_order")]
    identity = panel_identity(task, "train", not mixed)
    dev_identity = panel_identity(task, "development", not mixed)
    if family_filter:
        identity = tuple(a[np.isin(identity[0], family_filter)] for a in identity)
        dev_identity = tuple(a[np.isin(dev_identity[0], family_filter)] for a in dev_identity)
    check(len(lessons) <= len(order), "lesson cap exceeded")
    for number, lesson in enumerate(lessons, 1):
        row = order[number - 1]
        check(lesson["lesson"] == number and lesson["row"] == row, "teaching schedule differs")
        check(lesson["family"] == identity[0][row], "teaching family differs")
        check(lesson["variant"] == identity[1][row], "teaching variant differs")
        lesson_audit(lesson, files, prefix)
    accepted = sum(row["accepted"] for row in lessons)
    check(stage["accepted_updates"] == accepted, "stage accepted count differs")
    check(stage["refused_updates"] == len(lessons) - accepted, "stage refusal count differs")
    check(readings and readings[0]["lesson"] == 0, "newborn recall missing")
    check(readings[-1]["lesson"] == len(lessons), "final recall is not the endpoint")
    period = 96 if mixed else 32
    expected_readbacks = [0, *range(period, len(lessons) + 1, period)]
    if expected_readbacks[-1] != len(lessons):
        expected_readbacks.append(len(lessons))
    check(
        [r["lesson"] for r in readings] == expected_readbacks,
        "frozen recall checkpoint schedule differs",
    )
    previous = -1
    for reading in readings:
        check(previous < reading["lesson"] <= len(lessons), "recall schedule is unordered")
        previous = reading["lesson"]
        query_audit(reading["train"], identity)
        query_audit(reading["development"], dev_identity)
    passed = stage_gate(readings[-1], readings[0], len(lessons), mixed)
    no_refusals = not any(row["failed"] or not row["accepted"] for row in lessons) and not any(
        q["refusals"] for r in readings for q in (r["train"], r["development"])
    )
    expected_pass = stage["status"] == "development_passed" and passed and no_refusals
    check(stage["passed"] == expected_pass, "stage pass disagrees with frozen gate")
    for reading in readings[1:-1]:
        check(
            not stage_gate(reading, readings[0], reading["lesson"], mixed),
            "stage continued after its first passing frozen check",
        )
    terminal = "progress" if partial else "final"
    if partial:
        check(stage["progress_lessons"] == len(lessons), "progress checkpoint exposure differs")
    for name in ("initial", terminal):
        files.pin(f"{prefix}/{name}.npz", stage[name + "_sha256"])
    initial_arrays = files.arrays(prefix + "/initial.npz")
    final_arrays = files.arrays(prefix + f"/{terminal}.npz")
    initial_meta = json.loads(str(initial_arrays["meta"].item()))
    final_meta = json.loads(str(final_arrays["meta"].item()))
    for meta in (initial_meta, final_meta):
        check(meta["config"] == protocol["learner_config"], "checkpoint learning recipe differs")
        check(meta["neuron_model"] == protocol["neuron_model"], "checkpoint neuron recipe differs")
    check(initial_meta["updates"] == stage["starting_updates"], "initial update count differs")
    check(
        final_meta["updates"] == stage["starting_updates"] + accepted, "final update count differs"
    )
    if not mixed and "initial_model_identities" in protocol:
        seed = prefix.split("-")[1]
        identity = protocol["initial_model_identities"][seed]
        aliases = {
            "sensory_index": np.asarray(initial_meta["populations"]["sensory"]),
            "motor_index": np.asarray(initial_meta["populations"]["motor"]),
        }
        for key, expected in identity["arrays"].items():
            array = aliases[key] if key in aliases else initial_arrays[key]
            check(
                digest(np.ascontiguousarray(array).tobytes()) == expected,
                "initial frozen model identity differs",
            )
    for array in (initial_arrays, final_arrays):
        # the store's consolidated records, under the name of either checkpoint format
        for name in ("episodic/consolidated", "memories/hippocampus/consolidated"):
            if name in array:
                check(not np.any(array[name]), "associative memory was written")
    if lessons and lessons[-1]["accepted"]:
        recorded = lessons[-1]["phases"]["free"].get("parameter_readback")
        if recorded:
            for key in ("efficacy", "bias"):
                check(
                    digest(np.ascontiguousarray(final_arrays[key]).tobytes())
                    == recorded["post_" + key + "_sha256"],
                    "final parameter readback differs",
                )
            for key, expected in recorded["post_optimizer_sha256"].items():
                check(
                    digest(np.ascontiguousarray(final_arrays[key]).tobytes()) == expected,
                    "final optimizer readback differs",
                )
    work = {
        "attempted_single_cue_lessons": len(lessons),
        "accepted_single_cue_lessons": accepted,
        "refused_single_cue_lessons": len(lessons) - accepted,
        "observed_row_presentations": len(lessons),
        "old_rehearsal_presentations": sum(r["family"] in OLD for r in lessons) if mixed else 0,
        "new_presentations": sum(r["family"] not in OLD for r in lessons) if mixed else 0,
        "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in lessons),
        "reported_phase_row_residual_checks": sum(
            r["reported_row_residual_checks"] for r in lessons
        ),
        "independent_phase_equation_cache_checks": sum(
            r["independent_residual_checks"] for r in lessons
        ),
        "query_calls": sum(
            q["work"]["calls"] for r in readings for q in (r["train"], r["development"])
        ),
        "query_row_sweeps": sum(
            q["work"]["row_sweeps"] for r in readings for q in (r["train"], r["development"])
        ),
        "reported_query_residual_checks": sum(
            q["work"]["reported_residual_checks"]
            for r in readings
            for q in (r["train"], r["development"])
        ),
        "independent_query_equation_cache_checks": sum(
            q["work"]["independent_residual_checks"]
            for r in readings
            for q in (r["train"], r["development"])
        ),
        "refused_query_rows": sum(
            q["refusals"] for r in readings for q in (r["train"], r["development"])
        ),
        "memory_reads": 0,
        "memory_writes": 0,
    }
    if "phase_stagnation_checks" in stage.get("work", {}):
        work["phase_stagnation_checks"] = sum(
            r["report"].get("total_stagnation_checks", 0) for r in lessons
        )
        work["query_stagnation_checks"] = sum(
            q["work"].get("reported_stagnation_checks", 0)
            for r in readings
            for q in (r["train"], r["development"])
        )
    for name, value in work.items():
        if not partial:
            check(stage["work"].get(name) == value, f"stage work differs: {name}")
    return readings, lessons


def arrays_equal(first, second):
    return set(first) == set(second) and all(np.array_equal(first[k], second[k]) for k in first)


def random_control(identity, seed):
    """Score actual uniform draws and disclose their sampling variability."""
    families, _, labels = identity
    control_seed = [11020261004, int(seed)]
    answers = np.random.default_rng(np.random.SeedSequence(control_seed)).integers(
        36, size=len(labels)
    )
    correct = answers == labels
    return {
        "description": "Uniform-random actions scored on the same endpoint observations.",
        "seed": control_seed,
        "measured": True,
        "correct_rows": int(correct.sum()),
        "total_rows": len(labels),
        "correct_families": sum(bool(np.all(correct[families == f])) for f in np.unique(families)),
        "total_families": len(np.unique(families)),
        "expected_row_accuracy": 1 / 36,
        "row_accuracy_standard_deviation": math.sqrt((1 / 36) * (35 / 36) / len(labels)),
        "predictions": answers.tolist(),
        "labels": labels.tolist(),
        "work": {"evaluated_rows": len(labels), "neural_sweeps": 0, "teacher_presentations": 0},
    }


def failed_progress(files, seed, protocol, task):
    """Verify the last durable prefix without inventing interrupted tail work."""
    prefix = f"seed-{seed}"
    stages, queries, lessons = [], [], []
    last = endpoint = None
    for mixed in (False, True):
        name = prefix + ("-mixed" if mixed else "-old4")
        if not files.exists(name + "/receipt.json"):
            continue
        stage = files.json(name + "/receipt.json")
        partial = not files.exists(name + "/final.npz")
        if partial and not files.exists(name + "/progress.npz"):
            continue
        readings, teaching = stage_audit(stage, files, name, protocol, task, mixed, partial)
        stages.append(
            {
                "name": "mixed" if mixed else "old4",
                "lessons": len(teaching),
                "recall": [demo_reading(r, task, mixed) for r in readings],
            }
        )
        queries.extend(q for r in readings for q in (r["train"], r["development"]))
        lessons.extend(teaching)
        last = readings[-1]
        endpoint = name + ("/progress.npz" if partial else "/final.npz")
    if last is None:
        return {"stages": [], "curve": [], "work": {}}
    identity = panel_identity(task, "development", "-old4/" in endpoint)
    return {
        "stages": stages,
        "curve": [],
        "endpoint_checkpoint": endpoint,
        "replay_queries": last,
        "final": {
            "old_correct": last["development"].get("old_family_credits", 0),
            "old_total": 4,
            "new_correct": last["development"].get("new_family_credits", 0),
            "new_total": 20,
            "heldout_correct": None,
            "heldout_total": None,
        },
        "work": {
            "recorded_prefix_teaching_presentations": len(lessons),
            "recorded_prefix_phase_sweeps": sum(r["phase_row_sweeps"] for r in lessons),
            "recorded_prefix_query_sweeps": sum(q["work"]["row_sweeps"] for q in queries),
            "recorded_prefix_teacher_stagnation_checks": sum(
                r["report"].get("total_stagnation_checks", 0) for r in lessons
            ),
            "interrupted_tail_work_known": False,
            "boundary": "Verified durable prefix; later work and next-lesson continuation unknown.",
        },
        "controls": {"random": random_control(identity, seed)},
        "continuation": {"passed": False, "description": "No completed next-lesson audit."},
    }


def founder_audit(root, outcome, protocol, task, recipe_hash):
    seed = outcome["seed"]
    files = Artifacts(root, seed, outcome)
    prefix = f"seed-{seed}"
    if outcome["status"] == "process_failed":
        check(not outcome["passed"], "failed process claims acceptance")
        return {
            "seed": seed,
            "status": "process_failed",
            "failure_category": "operational",
            "passed": False,
            "exact_work_available": False,
            **failed_progress(files, seed, protocol, task),
        }
    summary = files.json(prefix + "/summary.json")
    check(
        summary["seed"] == seed and summary["recipe_sha256"] == recipe_hash,
        "founder recipe differs",
    )
    for key, value in summary.items():
        check(outcome.get(key) == value, f"founder census differs from worker: {key}")
    stages, curve, demo_stages, lessons, queries = {}, [], [], [], []
    for name in summary["stages"]:
        check(name in (prefix + "-old4", prefix + "-mixed"), "unknown founder stage")
        stage = files.json(name + "/receipt.json")
        mixed = name.endswith("-mixed")
        readings, teaching = stage_audit(stage, files, name, protocol, task, mixed)
        stages[name] = stage
        demo_stages.append(
            {
                "name": "mixed" if mixed else "old4",
                "lessons": len(teaching),
                "recall": [demo_reading(r, task, mixed) for r in readings],
            }
        )
        lessons.extend(teaching)
        for reading in readings:
            queries.extend((reading["train"], reading["development"]))
            curve.append(
                {
                    "stage": "mixed" if mixed else "old4",
                    "lesson": reading["lesson"],
                    **{k: reading["train"][k] for k in ("correct", "examples", "family_credits")},
                    "development_correct": reading["development"]["correct"],
                    "development_examples": reading["development"]["examples"],
                    "old_family_credits": reading["development"].get("old_family_credits", 0),
                    "new_family_credits": reading["development"].get("new_family_credits", 0),
                    "refusals": reading["train"]["refusals"] + reading["development"]["refusals"],
                }
            )
    check(prefix + "-old4" in stages, "old acquisition stage missing")
    mixed = stages.get(prefix + "-mixed")
    old = stages[prefix + "-old4"]
    if mixed:
        check(old["passed"], "mixed teaching started before old acquisition")
        check(
            old["final_sha256"] == mixed["initial_sha256"],
            "old-to-mixed checkpoint continuity differs",
        )
    endpoint = prefix + ("-mixed" if mixed else "-old4") + "/final.npz"
    files.pin(endpoint, summary["endpoint_checkpoint_sha256"])
    continuation = files.json(prefix + "/continuation.json")
    unlock = files.json(prefix + "/development-receipt.json")
    original = continuation["original_cold_train_development"]
    loaded = continuation["loaded_cold_train_development"]
    for pair in (original, loaded):
        check(len(pair) == 2, "continuation readback census differs")
        query_audit(pair[0], panel_identity(task, "train", not bool(mixed)))
        query_audit(pair[1], panel_identity(task, "development", not bool(mixed)))
    endpoint_equal = all(
        a["predictions"] == b["predictions"] for a, b in zip(original, loaded, strict=True)
    )
    check(
        continuation["saved_cold_predictions_equal"] == endpoint_equal,
        "saved prediction parity differs",
    )
    queries.extend(original + loaded)
    zero_refusals = not any(r["failed"] or not r["accepted"] for r in lessons) and not any(
        q["refusals"] for q in queries
    )
    dev_pass = bool(
        old["passed"]
        and mixed
        and mixed["passed"]
        and endpoint_equal
        and zero_refusals
        and original[0]["family_credits"] >= 18
        and original[1]["correct"] >= 36
        and original[1]["old_family_credits"] >= 3
        and original[1]["new_family_credits"] >= 15
    )
    check(summary["development_passed"] == dev_pass, "founder development gate differs")
    check(
        unlock["development_passed"] == dev_pass and unlock["zero_refusals"] == zero_refusals,
        "heldout unlock gate differs",
    )
    check(
        unlock["recipe_sha256"] == recipe_hash
        and unlock["relations_protocol_sha256"] == protocol["relations_protocol_sha256"],
        "heldout unlock identity differs",
    )
    check(unlock["recipe_frozen_before_development"] is True, "heldout recipe was not frozen")
    files.pin(endpoint, unlock["endpoint_checkpoint_sha256"])
    heldout = None
    heldout_name = prefix + "/heldout-receipt.json"
    if files.exists(heldout_name):
        check(dev_pass, "heldout opened after failed development")
        receipt = files.json(heldout_name)
        files.pin(prefix + "/development-receipt.json", receipt["development_receipt_sha256"])
        files.pin(endpoint, receipt["checkpoint_sha256"])
        heldout = receipt["recall"]
        query_audit(heldout, panel_identity(task, "heldout"))
        queries.append(heldout)
    check(summary["heldout_read"] == bool(heldout), "heldout read flag differs")
    next_lessons = continuation["next_single_cue_lessons"]
    order = protocol["mixed_order" if mixed else "old_order"]
    count = len(mixed["lessons"] if mixed else old["lessons"])
    next_row = (
        order[count]
        if count < len(order)
        else protocol["mixed_after_cap_next_row" if mixed else "old_after_cap_next_row"]
    )
    next_ids = panel_identity(task, "train", not bool(mixed))
    for key, expected in (
        ("next_scheduled_row", next_row),
        ("next_family", int(next_ids[0][next_row])),
        ("next_variant", int(next_ids[1][next_row])),
        ("next_label", int(next_ids[2][next_row])),
    ):
        check(continuation[key] == expected, "saved continuation schedule differs")
    check(len(next_lessons) == 2, "next-learning census differs")
    for lesson in next_lessons:
        lesson_audit(lesson, files, prefix)
    kept = []
    for name, expected in zip(
        ("continued-original.npz", "continued-loaded.npz"),
        continuation["checkpoint_sha256"],
        strict=True,
    ):
        files.pin(prefix + "/" + name, expected)
        kept.append(files.arrays(prefix + "/" + name))
    continued_equal = arrays_equal(*kept)
    accepted_equal = next_lessons[0]["accepted"] == next_lessons[1]["accepted"]
    check(
        continuation["continued_arrays_equal"] == continued_equal,
        "continued complete arrays differ",
    )
    check(continuation["accepted_equal"] == accepted_equal, "continued admission parity differs")
    continuation_pass = bool(
        endpoint_equal
        and continued_equal
        and accepted_equal
        and all(r["accepted"] and not r["failed"] for r in next_lessons)
        and zero_refusals
    )
    check(summary["continuation_passed"] == continuation_pass, "continuation gate differs")
    passed = bool(
        dev_pass
        and heldout
        and heldout["correct"] >= 36
        and not heldout["refusals"]
        and continuation_pass
    )
    check(summary["passed"] == passed, "founder acceptance differs from retained evidence")
    check(
        summary["status"] == ("confirmed" if passed else "confirmation_failed"),
        "founder terminal status differs",
    )
    all_lessons = lessons + next_lessons
    work = {
        "single_cue_teaching_calls": len(all_lessons),
        "accepted_single_cue_teaching_calls": sum(r["accepted"] for r in all_lessons),
        "refused_single_cue_teaching_calls": sum(not r["accepted"] for r in all_lessons),
        "actual_continuation_teaching_calls": 2,
        "old_acquisition_presentations": len(old["lessons"]),
        "mixed_presentations": len(mixed["lessons"]) if mixed else 0,
        "mixed_old_rehearsal_presentations": mixed["work"]["old_rehearsal_presentations"]
        if mixed
        else 0,
        "mixed_new_relation_presentations": mixed["work"]["new_presentations"] if mixed else 0,
        "continuation_old_presentations": 2 if continuation["next_family"] in OLD else 0,
        "continuation_new_presentations": 0 if continuation["next_family"] in OLD else 2,
        "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in all_lessons),
        "reported_phase_row_residual_checks": sum(
            r["reported_row_residual_checks"] for r in all_lessons
        ),
        "independent_phase_equation_cache_checks": sum(
            r["independent_residual_checks"] for r in all_lessons
        ),
        "cold_helper_query_calls": sum(q["work"]["calls"] for q in queries),
        "public_act_calls": 0,
        "query_row_sweeps": sum(q["work"]["row_sweeps"] for q in queries),
        "reported_query_residual_checks": sum(
            q["work"]["reported_residual_checks"] for q in queries
        ),
        "independent_query_equation_cache_checks": sum(
            q["work"]["independent_residual_checks"] for q in queries
        ),
        "refused_query_answers": sum(q["refusals"] for q in queries),
        "associative_memory_reads": 0,
        "associative_memory_writes": 0,
    }
    if "phase_stagnation_checks" in summary["work"]:
        work["phase_stagnation_checks"] = sum(
            r["report"].get("total_stagnation_checks", 0) for r in all_lessons
        )
        work["query_stagnation_checks"] = sum(
            q["work"].get("reported_stagnation_checks", 0) for q in queries
        )
    for name, value in work.items():
        check(summary["work"].get(name) == value, f"founder work differs: {name}")
    work["teacher_stagnation_checks"] = sum(
        r["report"].get("total_stagnation_checks", 0) for r in all_lessons
    )
    work["refused_teaching_attempts"] = work["refused_single_cue_teaching_calls"]
    work["query_stagnation_checks_available"] = all(
        "reported_stagnation_checks" in q["work"]
        and not q["work"].get("unreported_stagnation_work")
        for q in queries
    )
    work["boundary"] = "Solver sweeps and checks; no physical energy or efficiency claim."
    final_query = heldout or original[1]
    ids = panel_identity(task, "heldout" if heldout else "development", not bool(mixed))
    return {
        "seed": seed,
        "status": (
            "resource_capped"
            if summary.get("mixed_status", summary["old_status"]) in ("output_limit", "time_limit")
            else summary["status"]
        ),
        "stopping_reason": summary.get("mixed_status", summary["old_status"]),
        "passed": passed,
        "exact_work_available": True,
        "curve": curve,
        "stages": demo_stages,
        "work": work,
        "elapsed_seconds": summary["elapsed_seconds"],
        "endpoint_checkpoint": endpoint,
        "endpoint": {
            "train_family_credits": original[0]["family_credits"],
            "development_correct": original[1]["correct"],
            "old_family_credits": original[1].get("old_family_credits", 0),
            "new_family_credits": original[1].get("new_family_credits", 0),
            "heldout_correct": heldout["correct"] if heldout else None,
            "heldout_examples": 48 if heldout else None,
        },
        "final": {
            "old_correct": original[1].get("old_family_credits", 0),
            "old_total": 4,
            "new_correct": original[1].get("new_family_credits") if mixed else None,
            "new_total": 20 if mixed else None,
            "heldout_correct": heldout["correct"] if heldout else None,
            "heldout_total": 48 if heldout else None,
            "heldout_unit": "rows",
        },
        "continuation": {
            "passed": continuation_pass,
            "description": "Saved predictions and retained next-lesson checkpoints match.",
        },
        "controls": {
            "frozen": {
                "description": "Same founder before graph teaching.",
                "correct_families": old["recall"][0]["development"]["family_credits"],
                "total_families": 4,
                "correct_rows": old["recall"][0]["development"]["correct"],
                "total_rows": 8,
            },
            "random": random_control(ids, seed),
        },
        "examples": [
            {"family": int(f), "truth": int(y), "prediction": int(a)}
            for f, y, a in zip(ids[0], ids[2], final_query["predictions"], strict=True)
        ],
    }


def demo_reading(reading, task, mixed):
    result = {"accepted_updates": reading["lesson"]}
    for name, split in (("train", "train"), ("development", "development")):
        query = reading[name]
        identity = panel_identity(task, split, not mixed)
        result[name] = {
            "family_count": len(np.unique(identity[0])),
            "correct_families": query["family_credits"],
            "correct_rows": query["correct"],
            "total_rows": query["examples"],
            "refused_rows": query["refusals"],
            "old_correct_families": query.get("old_family_credits", 0),
            "new_correct_families": query.get("new_family_credits", 0),
            "old_correct": query.get("old_family_credits", 0),
            "new_correct": query.get("new_family_credits", 0),
            "old_total": 4,
            "new_total": 20 if mixed else 0,
            "predictions": query["predictions"],
            "labels": identity[2].tolist(),
        }
    return result


def verify(root):
    root = Path(root).resolve()
    protocol_raw = (root / "protocol.json").read_bytes()
    protocol, census = json.loads(protocol_raw), json.loads((root / "summary.json").read_text())
    recipe_hash = digest(protocol_raw)
    check(protocol["schema"] in SCHEMAS, "unknown confirmation protocol")
    source_bound_candidate = protocol["schema"].endswith(("v3", "v4", "v5", "v6"))
    normalized_candidate = protocol["schema"].endswith("v6")
    seeds = protocol["founder_seeds"]
    check(
        len(seeds) == 5
        and len(set(seeds)) == 5
        and all(type(seed) is int and seed >= 0 for seed in seeds)
        and protocol["founder_denominator"] == 5,
        "frozen founder census differs",
    )
    if not source_bound_candidate:
        check(tuple(seeds) == SEEDS, "original frozen founder seeds differ")
    check(
        census["recipe_sha256"] == recipe_hash and census["founder_denominator"] == 5,
        "campaign recipe differs",
    )
    check([r["seed"] for r in census["outcomes"]] == seeds, "campaign founder census differs")
    check(
        all(r["status"] not in ("running", "scheduled", "not_run") for r in census["outcomes"]),
        "campaign is incomplete",
    )
    cfg = protocol["learner_config"]
    rates = (
        {
            "eta": protocol["candidate_gene"]["eta"],
            "eta_bias": protocol["candidate_gene"]["eta_bias"],
        }
        if source_bound_candidate
        else {"eta": 0.1, "eta_bias": 0.01}
    )
    check(all(math.isfinite(v) and v > 0 for v in rates.values()), "invalid frozen rates")
    expected_config = {
        **rates,
        "beta": 0.1,
        "centered": True,
        "nudge": "cross_entropy",
        "temperature": 0.2,
        "normalize_floor": 0.001,
        "decay": 0.0,
        "scale_cap": 8.0,
        "normalize": 0.99 if normalized_candidate else 0.0,
        "momentum": 0.0,
        "free_steps": 4096,
        "nudged_steps": 4096,
        "qualified": True,
        "damping": 3,
        "tolerance": TOLERANCE,
    }
    check(set(cfg) == set(expected_config), "frozen learning configuration inventory differs")
    for name, value in expected_config.items():
        check(cfg[name] == value, f"frozen half-step recipe differs: {name}")
    if normalized_candidate:
        for name in ("normalize", "normalize_floor", "momentum"):
            if name in protocol["candidate_gene"]:
                check(
                    protocol["candidate_gene"][name] == cfg[name],
                    f"declared normalized gene differs: {name}",
                )
    expected_neuron = {
        "dt": 1.0,
        "slope": 1.0,
        "threshold": 0.0,
        "gain": 1.0,
        "stimulus_amplitude": 1.0,
        "adaptation": None,
        "leak": 0.1,
    }
    if protocol["schema"].endswith("v4"):
        leak = protocol["neuron_model"]["leak"]
        check(
            type(leak) in (float, int) and math.isfinite(leak) and 0 < leak <= 1,
            "invalid frozen leak gene",
        )
        expected_neuron["leak"] = leak
    if protocol["schema"].endswith(("v4", "v5", "v6")):
        check(
            (root / "reference-development").exists(),
            "source-bound model candidate requires passed development reference",
        )
    check(protocol["neuron_model"] == expected_neuron, "frozen neuron rule differs")
    if not protocol["schema"].endswith("v1"):
        check(
            protocol["model_construction"]
            == {
                "inputs": 650,
                "actions": 36,
                "modules": [32, 16],
                "observers": [],
                "lateral": 0.0 if protocol["schema"].endswith(("v5", "v6")) else -0.5,
            },
            "frozen construction differs",
        )
    for group, base in (
        ("library_sources", "source/library/cadence"),
        ("producing_sources", "source"),
    ):
        for name, expected in protocol[group].items():
            check(
                digest((root / base / safe_name(name)).read_bytes()) == expected,
                f"frozen source hash differs: {name}",
            )
    observed_library = {
        str(path.relative_to(root / "source/library/cadence"))
        for path in (root / "source/library/cadence").rglob("*.py")
    }
    check(
        observed_library == set(protocol["library_sources"]),
        "frozen library source inventory differs",
    )
    task_raw = (root / "source/fixture/protocol.json").read_bytes()
    task = json.loads(task_raw)
    check(digest(canonical(task)) == protocol["relations_protocol_sha256"], "task protocol differs")
    check(task["task_seed"] == 11020261003 and task["family_count"] == 24, "relation task differs")
    check(
        task["family_to_label"]
        == np.random.default_rng(np.random.SeedSequence([11020261003, 0]))
        .permutation(36)[:24]
        .tolist(),
        "task label mapping differs",
    )
    if "old_order" in protocol:
        check(protocol["pedagogy_seed"] == 1100301, "curriculum seed differs")
        for mixed, stage in ((False, 1), (True, 2)):
            name = "mixed" if mixed else "old"
            count = protocol[name + "_lesson_cap"]
            check(type(count) is int and count >= (128 if mixed else 1), "invalid lesson cap")
            if not source_bound_candidate:
                check(count == (2048 if mixed else 512), "original lesson cap differs")
            expected = frozen_order(task, count + 1, stage, not mixed)
            check(protocol[name + "_order"] == expected[:count], "frozen curriculum differs")
            check(
                protocol[name + "_after_cap_next_row"] == expected[count],
                "after-cap continuation schedule differs",
            )
    founders = [founder_audit(root, row, protocol, task, recipe_hash) for row in census["outcomes"]]
    passed = sum(r["passed"] for r in founders)
    check(census["passed"] == passed, "campaign pass count differs")
    if "all_founders_passed" in census:
        check(census["all_founders_passed"] == (passed == 5), "campaign all-founder gate differs")
    reference_verified = False
    reference = root / "reference-development"
    if reference.exists():
        for name in ("protocol", "summary"):
            raw = (reference / (name + ".json")).read_bytes()
            check(
                digest(raw) == protocol["successful_development_" + name + "_sha256"],
                "historical development reference differs",
            )
        development = json.loads((reference / "summary.json").read_text())
        check(
            development["passed"] is True and development["heldout_read"] is False,
            "historical development reference did not pass before heldout",
        )
        check(
            development["protocol_sha256"] == protocol["successful_development_protocol_sha256"],
            "historical development result is not bound to its protocol",
        )
        if source_bound_candidate:
            development_protocol = json.loads((reference / "protocol.json").read_text())
            prior_seeds = development_protocol.get(
                "founder_seeds", [development_protocol.get("brain_seed")]
            )
            check(set(seeds).isdisjoint(prior_seeds), "confirmation reuses a development founder")
            check(
                development_protocol["learner_config"] == cfg,
                "confirmation rates differ from passed development",
            )
            check(
                development_protocol["neuron_model"] == protocol["neuron_model"],
                "confirmation neuron model differs from passed development",
            )
            check(
                development_protocol["model_construction"] == protocol["model_construction"],
                "confirmation construction differs from passed development",
            )
            if "old_lesson_cap" in protocol:
                check(
                    protocol["old_lesson_cap"] == development_protocol["old_cap"]
                    and protocol["mixed_lesson_cap"] == development_protocol["mixed_cap"],
                    "confirmation lesson caps differ from passed development",
                )
        reference_verified = True
    return {
        "schema": "cadence.acquisition-demo.v1",
        "protocol_sha256": recipe_hash,
        "protocol": {
            "sha256": recipe_hash,
            "schema": protocol["schema"],
            "candidate_gene": protocol.get(
                "candidate_gene", {"step_scale": 0.5, "eta": 0.1, "eta_bias": 0.01}
            ),
            "learner_config": cfg,
            "neuron_model": protocol["neuron_model"],
            "resource_limits": {
                name: protocol[name]
                for name in (
                    "old_lesson_cap",
                    "mixed_lesson_cap",
                    "soft_teaching_admission_seconds_per_founder",
                    "output_admission_mib",
                    "output_cap_mib",
                )
                if name in protocol
            },
        },
        "source": {
            "version": protocol.get("source_version"),
            "commit": protocol.get("source_commit"),
            "library_sources": protocol["library_sources"],
        },
        "scope": "Controlled acquisition and retention of associations in graph weights.",
        "founder_denominator": 5,
        "passed_founders": passed,
        "all_founders_passed": passed == 5,
        "verification_scope": {
            "verifier_source_sha256": digest(Path(__file__).read_bytes()),
            "source_artifact_hashes": True,
            "scores_gates_schedule_work_recomputed": True,
            "continued_checkpoint_arrays_compared": True,
            "historical_trajectory_replayed": False,
            "endpoint_public_actions_replayed": False,
            "historical_development_reference_hashes": reference_verified,
            "boundary": (
                "Recorded equation checks and compact phase hashes do not reconstruct "
                "the historical learning trajectory."
            ),
        },
        "model_construction": protocol.get(
            "model_construction",
            {"inputs": 650, "actions": 36, "modules": [32, 16], "observers": [], "lateral": -0.5},
        ),
        "gates": {
            "train_families": 18,
            "development_correct": 36,
            "old_families": 3,
            "new_families": 15,
            "heldout_correct": 36,
            "refusals": 0,
        },
        "founders": founders,
    }


def replay_worker(job_path):
    """Read-only public endpoint queries under the checked archived library."""
    import time

    job = json.loads(Path(job_path).read_text())
    root = Path(job["root"])
    sys.path.insert(0, str(root / "source"))
    sys.path.insert(0, str(root / "source/library"))
    import relations
    import run as frozen_harness

    import cadence

    check(
        Path(cadence.__file__).resolve().parent == root / "source/library/cadence",
        "endpoint replay imported a different library",
    )
    began = time.monotonic()
    results = []
    for entry in job["entries"]:
        brain = cadence.Brain.load(entry["checkpoint"])
        work = {
            "public_act_calls": 0,
            "row_sweeps": 0,
            "reported_residual_checks": 0,
            "independent_equation_cache_checks": 0,
            "teacher_presentations": 0,
            "stagnation_checks": 0,
        }
        for selected in entry["panels"]:
            panel = relations.load_panel(
                selected["name"],
                families=OLD if entry["old_only"] else None,
                development_receipt=entry["unlock"] if selected["name"] == "heldout" else None,
            )
            answers = []
            for observation in panel["inputs"]:
                brain.reset()
                brain.hippocampus.reset(1)
                x = observation[None, :]
                drive = brain.stimulus(x, memory=False)
                answer = int(brain.act(x, greedy=True)[0])
                residual, cache = frozen_harness.independent_residual(
                    brain, drive, brain.basal_ganglia.state
                )
                check(
                    np.all(residual <= TOLERANCE) and np.all(cache <= TOLERANCE),
                    "public endpoint replay fails independent equations",
                )
                answers.append(answer)
                work["public_act_calls"] += 1
                work["independent_equation_cache_checks"] += 1
                report = brain.last_settlement
                work["row_sweeps"] += report["steps"]
                work["reported_residual_checks"] += report["residual_checks"]
                work["stagnation_checks"] += report.get("stagnation_checks", 0)
            check(answers == selected["predictions"], "public endpoint predictions differ")
        results.append({"seed": entry["seed"], "passed": True, "work": work})
    print(
        json.dumps(
            {"passed": True, "elapsed_seconds": time.monotonic() - began, "founders": results}
        )
    )


def replay_endpoints(root, demo):
    """Keep replay work separate; no parameter update is performed."""
    root = Path(root).resolve()
    census = json.loads((root / "summary.json").read_text())
    with tempfile.TemporaryDirectory(prefix="cadence-endpoint-verification-") as temporary:
        folder = Path(temporary)
        entries = []
        for outcome, founder in zip(census["outcomes"], demo["founders"], strict=True):
            if "endpoint_checkpoint" not in founder:
                continue
            seed = outcome["seed"]
            files = Artifacts(root, seed, outcome)
            prefix = f"seed-{seed}"
            checkpoint = folder / f"endpoint-{seed}.npz"
            checkpoint.write_bytes(files.read(founder["endpoint_checkpoint"]))
            if founder["exact_work_available"]:
                readings = files.json(prefix + "/continuation.json")[
                    "original_cold_train_development"
                ]
                unlock = files.json(prefix + "/development-receipt.json")
            else:
                readings = [founder["replay_queries"][name] for name in ("train", "development")]
                unlock = None
            panels = [
                {"name": name, "predictions": reading["predictions"]}
                for name, reading in zip(
                    ("train", "development"),
                    readings,
                    strict=True,
                )
            ]
            heldout = prefix + "/heldout-receipt.json"
            if files.exists(heldout):
                panels.append(
                    {"name": "heldout", "predictions": files.json(heldout)["recall"]["predictions"]}
                )
            entries.append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "panels": panels,
                    "old_only": "-old4/" in founder["endpoint_checkpoint"],
                    "unlock": unlock,
                }
            )
        job = folder / "job.json"
        job.write_bytes(canonical({"root": str(root), "entries": entries}))
        env = dict(os.environ)
        env.update(PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--replay-worker", str(job)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        result = json.loads(completed.stdout)
    demo["verification_scope"]["endpoint_public_actions_replayed"] = True
    demo["endpoint_verification"] = result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, help="write verified demo JSON to a new path")
    parser.add_argument(
        "--replay-endpoints",
        action="store_true",
        help="replay free public actions in an isolated archived-source subprocess",
    )
    parser.add_argument("--replay-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.replay_worker:
        replay_worker(args.replay_worker)
        return
    if args.run is None:
        parser.error("run directory required")
    verifier_hash = digest(Path(__file__).read_bytes())
    result = verify(args.run)
    if args.replay_endpoints:
        replay_endpoints(args.run, result)
    check(
        digest(Path(__file__).read_bytes()) == verifier_hash,
        "verifier source changed during the audit",
    )
    if args.out:
        with args.out.open("xb") as target:
            target.write(canonical(result))
    print(
        json.dumps(
            {
                "verified": True,
                "passed_founders": result["passed_founders"],
                "founder_denominator": 5,
                "verification_scope": result["verification_scope"],
            }
        )
    )


if __name__ == "__main__":
    main()
