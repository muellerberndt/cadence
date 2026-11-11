"""Frozen online curricula for canonical and half-step local contrasts.

This development control changes presentation order and batch size. It keeps
the graph, update law and qualification contract unchanged. The optional .5
step-scale is one explicit plasticity gene, alongside the recorded canonical
control. No hippocampal/working-memory read/write or heldout access occurs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import continual
import numpy as np
import relation_development as development
import relations
import run as harness

PEDAGOGY_SEED = 1100301


def tree_bytes(root):
    """Sample artifact sizes while completed siblings may be archived/removed."""
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # The coordinator removes loose duplicates only after verifying its
            # archive. Disappearance between is_file and stat is expected.
            continue
    return total


def curriculum(panel, count, stage):
    """Balanced shuffled family cycles; rotate the four observed variants."""
    rng = np.random.default_rng(np.random.SeedSequence([PEDAGOGY_SEED, stage]))
    families = np.unique(panel["families"])
    visits = {int(family): 0 for family in families}
    rows = []
    while len(rows) < count:
        for family in rng.permutation(families):
            family = int(family)
            variant = visits[family] % 4
            row = np.flatnonzero((panel["families"] == family) & (panel["instances"] == variant))[0]
            rows.append(int(row))
            visits[family] += 1
            if len(rows) == count:
                break
    return rows


def recall(brain, panel):
    brain.reset()
    brain.hippocampus.reset(1)
    result = harness.free_recall(brain, panel["inputs"], panel["labels"])
    result["family_credits"] = development.credited_families(panel, result["predictions"])
    predictions = np.array(result["predictions"])
    for name, families in (("old", relations.OLD_FAMILIES), ("new", relations.NEW_FAMILIES)):
        chosen = np.isin(panel["families"], families)
        if chosen.any():
            part = {key: value[chosen] for key, value in panel.items()}
            result[name + "_family_credits"] = development.credited_families(
                part, predictions[chosen]
            )
    return result


def stage(brain, train, dev, order, folder, deadline, *, mixed, output_admission_mib=32):
    folder.mkdir()
    brain.save(folder / "initial.npz")
    journal = folder / "lessons.jsonl"
    report = {
        "status": "running",
        "lessons": [],
        "recall": [],
        "accepted_updates": 0,
        "refused_updates": 0,
        "passed": False,
        "initial_sha256": harness.sha256(folder / "initial.npz"),
        "starting_updates": brain.learner.updates,
    }
    report["recall"].append(
        {"lesson": 0, "train": recall(brain, train), "development": recall(brain, dev)}
    )
    every = 96 if mixed else 32
    family_count = len(np.unique(train["families"]))
    for number, row in enumerate(order, 1):
        if time.monotonic() >= deadline:
            report["status"] = "time_limit"
            break
        if tree_bytes(folder.parent) >= output_admission_mib * 1024**2:
            report["status"] = "output_limit"
            break
        x, y = train["inputs"][[row]], train["labels"][[row]]
        lesson = continual.graph_lesson(brain, x, y, folder, f"phases-{number:04d}")
        report["lessons"].append(
            {
                "lesson": number,
                "row": row,
                "family": int(train["families"][row]),
                "variant": int(train["instances"][row]),
                **lesson,
            }
        )
        # Keep each attempted lesson even if the process stops between readbacks.
        with journal.open("a") as stream:
            stream.write(json.dumps(harness.json_value(report["lessons"][-1])) + "\n")
        report["accepted_updates"] += lesson["accepted"]
        report["refused_updates"] += not lesson["accepted"]
        if number % every == 0 or number == len(order) or lesson["failed"]:
            reading = {
                "lesson": number,
                "train": recall(brain, train),
                "development": recall(brain, dev),
            }
            report["recall"].append(reading)
            progress = folder / "progress.npz"
            report["progress_sha256"] = harness.sha256(brain.save(progress))
            report["progress_lessons"] = number
            if lesson["failed"]:
                report["status"] = (
                    "refused_learning" if not lesson["accepted"] else "qualification_violation"
                )
                break
            if any(q["refusals"] for r in report["recall"] for q in (r["train"], r["development"])):
                report["status"] = "refused_answer"
                break
            passed = (
                reading["train"]["family_credits"] >= (18 if mixed else family_count)
                and reading["development"]["correct"] >= (36 if mixed else len(dev["labels"]))
                and (
                    not mixed
                    or (
                        number >= 128
                        and reading["development"]["old_family_credits"] >= 3
                        and reading["development"]["new_family_credits"] >= 15
                    )
                )
                and reading["train"]["correct"] > report["recall"][0]["train"]["correct"]
            )
            if passed:
                report["status"], report["passed"] = "development_passed", True
                break
            temporary = folder / "receipt.pending.json"
            harness.write_json(temporary, report)
            temporary.replace(folder / "receipt.json")
    else:
        report["status"] = "lesson_limit"
    count = len(report["lessons"])
    if report["recall"][-1]["lesson"] != count:
        report["recall"].append(
            {"lesson": count, "train": recall(brain, train), "development": recall(brain, dev)}
        )
    report["final_sha256"] = harness.sha256(brain.save(folder / "final.npz"))
    lessons, queries = (
        report["lessons"],
        [q["work"] for r in report["recall"] for q in (r["train"], r["development"])],
    )
    report["work"] = {
        "attempted_single_cue_lessons": count,
        "accepted_single_cue_lessons": report["accepted_updates"],
        "refused_single_cue_lessons": report["refused_updates"],
        "observed_row_presentations": count,
        "old_rehearsal_presentations": sum(r["family"] in relations.OLD_FAMILIES for r in lessons)
        if mixed
        else 0,
        "new_presentations": sum(r["family"] in relations.NEW_FAMILIES for r in lessons)
        if mixed
        else 0,
        "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in lessons),
        "phase_stagnation_checks": sum(r["report"].get("total_stagnation_checks", 0)
                                       for r in lessons),
        "query_stagnation_checks": sum(r.get("reported_stagnation_checks", 0)
                                       for r in queries),
        "reported_phase_row_residual_checks": sum(
            r["reported_row_residual_checks"] for r in lessons
        ),
        "independent_phase_equation_cache_checks": sum(
            r["independent_residual_checks"] for r in lessons
        ),
        "query_calls": sum(r["calls"] for r in queries),
        "query_row_sweeps": sum(r["row_sweeps"] for r in queries),
        "reported_query_residual_checks": sum(r["reported_residual_checks"] for r in queries),
        "independent_query_equation_cache_checks": sum(
            r["independent_residual_checks"] for r in queries
        ),
        "refused_query_rows": sum(
            q["refusals"] for r in report["recall"] for q in (r["train"], r["development"])
        ),
        "memory_reads": 0,
        "memory_writes": 0,
    }
    harness.write_json(folder / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path.cwd()).free
        <= 20 * 1024**3
    ):
        raise RuntimeError("require >20GiB free")
    args.out.mkdir(parents=True, exist_ok=False)
    brain = development.make_brain(
        "canonical", 0, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    old_train = relations.load_panel("train", families=relations.OLD_FAMILIES)
    old_dev = relations.load_panel("development", families=relations.OLD_FAMILIES)
    train, dev = relations.load_panel("train"), relations.load_panel("development")
    old_order, mixed_order = curriculum(old_train, 512, 1), curriculum(train, 1536, 2)
    protocol = {
        "schema": "cadence-canonical-online-curriculum-development-v1",
        ("curriculum_gene"): (
            "one observed cue per local-contrast lesson; class-balanced shuffled "
            "family cycles with rotatedTRAIN variants; full-batch controls "
            "preserved"
        ),
        "brain_seed": 0,
        "pedagogy_seed": PEDAGOGY_SEED,
        "old_order": old_order,
        "mixed_order": mixed_order,
        "old_cap": 512,
        "mixed_cap": 1536,
        "seconds_total": 300,
        "output_cap_mib": 32,
        "old_query_every": 32,
        "mixed_query_every": 96,
        "learner_config": brain.learner.config.to_dict(),
        "gene": development.GENES["canonical"],
        "neuron_model": brain.brain.neuron_model.to_dict(),
        "relations_protocol_sha256": relations.protocol_hash(),
        "library_sources": harness.sources(),
        "producing_sources": {
            name: harness.sha256(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "continual.py",
                "relation_development.py",
                "relations.py",
                "run.py",
                "extract.py",
            )
        },
        "old_gate": "perfect all16TRAIN/all8DEV; zero phase/answer refusals; positive TRAIN gain",
        ("mixed_gate"): (
            ">=18TRAIN families,>=36/48DEV,old>=3/4,new>=15/20DEV all2instances; "
            "at least128 mixed lessons; zero refusals; positive TRAIN gain"
        ),
        ("experience"): (
            "every single cue, old rehearsal, free/\u00b1nudged phase, query and "
            "saved-continuation lesson charged"
        ),
        ("boundary"): (
            "same continuing graph, unchanged canonical local rule and all "
            "parameters; no memory/Trace/classifier, no heldout or efficiency claim"
        ),
        ("deadline"): (
            "300-second teaching admission cap; charge mandatory "
            "endpoint/checkpoint reads and the current bounded phase"
        ),
    }
    harness.write_json(args.out / "protocol.json", protocol)
    development.freeze_sources(args.out)
    for name in (Path(__file__).name, "continual.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.out / "source" / name)
    development.compact_phase_recording()
    began = time.monotonic()
    deadline = began + 300
    initial_writes = brain.hippocampus.writes
    initial_memory = brain.hippocampus.consolidated.copy()
    old = stage(brain, old_train, old_dev, old_order, args.out / "old4", deadline, mixed=False)
    summary = {
        "protocol_sha256": harness.sha256(args.out / "protocol.json"),
        "old_status": old["status"],
        "old_passed": old["passed"],
        "old_work": old["work"],
        "old_lessons": len(old["lessons"]),
        "passed": False,
        "heldout_read": False,
    }
    if old["passed"]:
        mixed = stage(brain, train, dev, mixed_order, args.out / "mixed", deadline, mixed=True)
        last = mixed["recall"][-1]
        summary.update(
            mixed_status=mixed["status"],
            mixed_lessons=len(mixed["lessons"]),
            mixed_work=mixed["work"],
            train_family_credits=last["train"]["family_credits"],
            development_correct=last["development"]["correct"],
            old_family_credits=last["development"]["old_family_credits"],
            new_family_credits=last["development"]["new_family_credits"],
            passed=mixed["passed"],
        )
        checkpoint = args.out / "mixed/final.npz"
    else:
        checkpoint = args.out / "old4/final.npz"
    loaded = continual.Brain.load(checkpoint)
    original_read, loaded_read = recall(brain, dev), recall(loaded, dev)
    next_row = mixed_order[0] if old["passed"] else int(old_order[0])
    next_panel = train if old["passed"] else old_train
    x, y = next_panel["inputs"][[next_row]], next_panel["labels"][[next_row]]
    custody = continual.graph_checkpoint_continuation(brain, loaded, x, y, args.out)
    continued, equal = custody["lessons"], custody["continued_arrays_equal"]
    continuation = {
        "saved_cold_predictions_equal": original_read["predictions"] == loaded_read["predictions"],
        "query_work": [original_read["work"], loaded_read["work"]],
        "next_single_cue_lessons": continued,
        "continued_arrays_equal": equal,
        "phase_payloads_retained": custody["phase_payloads_retained"],
        "refused_readbacks": original_read["refusals"] + loaded_read["refusals"],
    }
    harness.write_json(args.out / "continuation.json", continuation)
    summary["passed"] = bool(
        summary["passed"]
        and continuation["saved_cold_predictions_equal"]
        and equal
        and not continuation["refused_readbacks"]
        and all(not r["failed"] for r in continued)
    )
    summary.update(
        continuation_passed=equal and continuation["saved_cold_predictions_equal"],
        elapsed_seconds=time.monotonic() - began,
    )
    if initial_writes != brain.hippocampus.writes or not np.array_equal(
        initial_memory, brain.hippocampus.consolidated
    ):
        raise AssertionError("online graph curriculum unexpectedly wrote associative memory")
    harness.write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
