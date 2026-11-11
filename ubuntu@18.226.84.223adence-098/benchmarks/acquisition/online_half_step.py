"""One frozen half-step plasticity gene against the completed online control.

Only eta and eta_bias change, by step_scale=.5. This is a candidate gene,
not a library default or a new learning rule. No heldout access occurs here.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def make_brain():
    brain = development.make_brain(
        "canonical", 0, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    baseline = brain.learner.config
    brain.learner.config = replace(
        baseline, eta=baseline.eta * 0.5, eta_bias=baseline.eta_bias * 0.5
    )
    changes = {
        key
        for key in baseline.to_dict()
        if baseline.to_dict()[key] != brain.learner.config.to_dict()[key]
    }
    if changes != {"eta", "eta_bias"}:
        raise AssertionError("half-step candidate changed another learning parameter")
    return brain


def precision(kind):
    dtype = np.dtype(kind)
    info = np.finfo(dtype)
    return {
        "dtype": str(dtype),
        "bytes": dtype.itemsize,
        "fraction_bits": info.nmant,
        "eps": float(info.eps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path.cwd()).free
        <= 20 * 1024**3
    ):
        raise RuntimeError("require >20GiB free")
    baseline_path = args.out.parent / "cadence-online-curriculum-20261003/protocol.json"
    baseline = json.loads(baseline_path.read_text())
    tiny_train = relations.load_panel("train", families=relations.TWO_FAMILIES)
    tiny_dev = relations.load_panel("development", families=relations.TWO_FAMILIES)
    old_train = relations.load_panel("train", families=relations.OLD_FAMILIES)
    old_dev = relations.load_panel("development", families=relations.OLD_FAMILIES)
    train, dev = relations.load_panel("train"), relations.load_panel("development")
    tiny_order = online.curriculum(tiny_train, 256, 0)
    old_order = online.curriculum(old_train, 512, 1)
    mixed_order = online.curriculum(train, 2048, 2)
    if old_order != baseline["old_order"] or mixed_order[:1536] != baseline["mixed_order"]:
        raise AssertionError("half-step curriculum differs from the frozen control")
    candidate = make_brain()
    protocol = {
        "schema": "cadence-online-half-step-development-v1",
        "candidate_gene": {"step_scale": 0.5, "eta": 0.1, "eta_bias": 0.01},
        "baseline_gene": {"step_scale": 1.0, "eta": 0.2, "eta_bias": 0.02},
        "baseline_protocol_sha256": harness.sha256(baseline_path),
        "baseline_preserved": "completed control and all failures retain their original bytes",
        "brain_seed": 0,
        "pedagogy_seed": online.PEDAGOGY_SEED,
        "tiny_order": tiny_order,
        "old_order": old_order,
        "mixed_order": mixed_order,
        ("curriculum"): (
            "same frozen shuffled class-balanced cycles and rotated TRAIN "
            "variants; mixed contains four old families as charged rehearsal"
        ),
        "tiny_cap": 256,
        "old_cap": 512,
        "mixed_cap": 2048,
        "teaching_admission_seconds": 900,
        ("mandatory_tail"): (
            "finish current bounded phase, cold endpoint queries and actual "
            "original/load next-learning custody; record tail separately"
        ),
        "output_cap_mib": 80,
        "output_admission_cap_mib": 72,
        "tiny_old_query_every": 32,
        "mixed_query_every": 96,
        ("tiny_old_gate"): (
            "all TRAIN and DEV instances correct; zero phase/answer refusals; "
            "positive TRAIN gain; each stage starts fresh"
        ),
        ("mixed_gate"): (
            ">=18 complete TRAIN families,>=36/48DEV,old>=3/4,new>=15/20(all2DEV "
            "instances),zero refusals,positiveTRAIN gain,>=128 mixed presentations"
        ),
        ("continuation"): (
            "distinct phase/checkpoint paths; check phase hashes after checkpoint "
            "save, accepted outcomes and complete arrays"
        ),
        "learner_config": candidate.learner.config.to_dict(),
        "neuron_model": candidate.brain.neuron_model.to_dict(),
        "canonical_gene_control": development.GENES["canonical"],
        "relations_protocol_sha256": relations.protocol_hash(),
        "library_sources": harness.sources(),
        "producing_sources": {
            name: harness.sha256(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "online_curriculum.py",
                "continual.py",
                "relation_development.py",
                "relations.py",
                "run.py",
                "extract.py",
            )
        },
        "runtime_precision": {
            "numpy": np.__version__,
            "float64": precision(np.float64),
            "longdouble": precision(np.longdouble),
            ("boundary"): (
                "independent formulas at runtime float64 precision; longdouble equals float64 here"
            ),
        },
        ("scope"): (
            "one targeted plasticity-rate gene; original local rule/graph and "
            "other parameters; no memory/Trace/reward/classifier/default change"
        ),
        ("heldout"): (
            "unread/ungenerated; notify root after development pass before any "
            "confirmation proposal"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    harness.write_json(args.out / "protocol.json", protocol)
    development.freeze_sources(args.out)
    for name in (Path(__file__).name, "online_curriculum.py", "continual.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.out / "source" / name)
    development.compact_phase_recording()
    began = time.monotonic()
    deadline = began + 900
    stages = []
    tiny = online.stage(
        candidate,
        tiny_train,
        tiny_dev,
        tiny_order,
        args.out / "tiny2",
        deadline,
        mixed=False,
        output_admission_mib=72,
    )
    stages.append(
        {
            "stage": "tiny2",
            "status": tiny["status"],
            "passed": tiny["passed"],
            "lessons": len(tiny["lessons"]),
            "work": tiny["work"],
        }
    )
    brain, endpoint_panel, next_panel, next_row = candidate, tiny_dev, tiny_train, tiny_order[0]
    checkpoint = args.out / "tiny2/final.npz"
    summary = {
        "protocol_sha256": harness.sha256(args.out / "protocol.json"),
        "stages": stages,
        "passed": False,
        "heldout_read": False,
        "founder_denominator": 1,
    }
    if tiny["passed"]:
        brain = make_brain()
        old = online.stage(
            brain,
            old_train,
            old_dev,
            old_order,
            args.out / "old4",
            deadline,
            mixed=False,
            output_admission_mib=72,
        )
        stages.append(
            {
                "stage": "old4",
                "status": old["status"],
                "passed": old["passed"],
                "lessons": len(old["lessons"]),
                "work": old["work"],
            }
        )
        endpoint_panel, next_panel, next_row = old_dev, old_train, old_order[0]
        checkpoint = args.out / "old4/final.npz"
        if old["passed"]:
            mixed = online.stage(
                brain,
                train,
                dev,
                mixed_order,
                args.out / "mixed",
                deadline,
                mixed=True,
                output_admission_mib=72,
            )
            stages.append(
                {
                    "stage": "mixed",
                    "status": mixed["status"],
                    "passed": mixed["passed"],
                    "lessons": len(mixed["lessons"]),
                    "work": mixed["work"],
                }
            )
            last = mixed["recall"][-1]
            summary.update(
                train_family_credits=last["train"]["family_credits"],
                development_correct=last["development"]["correct"],
                old_family_credits=last["development"]["old_family_credits"],
                new_family_credits=last["development"]["new_family_credits"],
                passed=mixed["passed"],
            )
            endpoint_panel, next_panel, next_row = dev, train, mixed_order[0]
            checkpoint = args.out / "mixed/final.npz"
    summary["teaching_and_instage_readback_elapsed_seconds"] = time.monotonic() - began
    tail_began = time.monotonic()
    loaded = continual.Brain.load(checkpoint)
    original_read, loaded_read = (
        online.recall(brain, endpoint_panel),
        online.recall(loaded, endpoint_panel),
    )
    x, y = next_panel["inputs"][[next_row]], next_panel["labels"][[next_row]]
    custody = continual.graph_checkpoint_continuation(brain, loaded, x, y, args.out)
    continuation = {
        "saved_cold_predictions_equal": original_read["predictions"] == loaded_read["predictions"],
        "query_work": [original_read["work"], loaded_read["work"]],
        "refused_readbacks": original_read["refusals"] + loaded_read["refusals"],
        "next_family": int(next_panel["families"][next_row]),
        "next_label": int(y[0]),
        "next_single_cue_lessons": custody["lessons"],
        **{key: value for key, value in custody.items() if key != "lessons"},
    }
    harness.write_json(args.out / "continuation.json", continuation)
    continuation_passed = (
        custody["continued_arrays_equal"]
        and custody["accepted_equal"]
        and continuation["saved_cold_predictions_equal"]
        and not continuation["refused_readbacks"]
        and all(not record["failed"] for record in custody["lessons"])
    )
    summary.update(
        passed=bool(summary["passed"] and continuation_passed),
        continuation_passed=bool(continuation_passed),
        mandatory_continuation_tail_seconds=time.monotonic() - tail_began,
        elapsed_seconds=time.monotonic() - began,
    )
    for model in (candidate, brain, loaded):
        if model.hippocampus.writes != 0 or np.any(model.hippocampus.consolidated):
            raise AssertionError("candidate unexpectedly wrote associative memory")
    size = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
    summary["output_bytes_before_summary"] = size
    if size >= 80 * 1024**2:
        summary.update(passed=False, output_cap_exceeded=True)
    harness.write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
