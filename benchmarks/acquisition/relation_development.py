"""Bounded local-contrast development on identifiable cue/action relations.

This is a diagnostic school, not a native-body or lifelong-memory result. Every
answer is the original graph's qualified free equilibrium. Target labels enter
only a teaching nudge. No associative-memory writes or readout classifier occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import relations
import run as harness

from cadence import Brain, LearnerConfig

GENES = {
    "canonical": {
        "eta": 0.2,
        "normalize": 0.0,
        "momentum": 0.0,
        "sensory_bias": 0.0,
        "hidden_bias": 0.0,
        "fixed_motor_lateral": False,
        "leak": 0.1,
        "damping": 3,
    },
    "fixed-lateral-local-rms": {
        "eta": 0.005,
        "normalize": 0.99,
        "momentum": 0.9,
        "sensory_bias": 0.6,
        "hidden_bias": 0.0,
        "fixed_motor_lateral": True,
        "leak": 0.1,
        "damping": 3,
    },
    "responsive-local-rms": {
        "eta": 0.005,
        "normalize": 0.99,
        "momentum": 0.9,
        "sensory_bias": 0.6,
        "hidden_bias": 0.25,
        "fixed_motor_lateral": True,
        "leak": 0.1,
        "damping": 3,
    },
    "smooth-responsive-local-rms": {
        "eta": 0.005,
        "normalize": 0.99,
        "momentum": 0.9,
        "sensory_bias": 0.0,
        "hidden_bias": 0.25,
        "fixed_motor_lateral": True,
        "leak": 1.0,
        "damping": 3,
    },
}


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compact_phase_recording():
    """Retain independent readback and exact vector pins; keep refused raw states.

    Only this runner's in-process recorder changes. The native experiment harness
    and all historical receipt files retain their original recording behavior.
    """
    original_readback, original_file = harness.phase_readback, harness.phase_file
    latest = {"refused": False}

    def readback(model, drive, labels, phases, weights, bias):
        reports = original_readback(model, drive, labels, phases, weights, bias)
        latest["refused"] = any(
            not all(report["qualified"])
            or max(report["cache_defect"]) > model.learner.config.tolerance
            for report in reports.values()
        )
        for name, report in reports.items():
            report.pop("motor_activity")
            report["vector_sha256"] = {
                field: array_hash(getattr(phases[name], field))
                for field in ("v", "activation", "adaptation")
            }
        first = next(iter(reports.values()))
        first["parameter_readback"] = {
            "pre_effective_weight_sha256": array_hash(weights),
            "pre_bias_sha256": array_hash(bias),
            "post_efficacy_sha256": array_hash(model.brain.efficacy),
            "post_bias_sha256": array_hash(model.brain.bias),
            "post_optimizer_sha256": {
                name: array_hash(getattr(model.learner, name))
                for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
            },
            "updates": model.learner.updates,
            "contrast_updates": model.learner.contrast_updates,
        }
        return reports

    def phase_file(path, phases):
        if latest["refused"]:
            return original_file(path, phases)
        arrays = {}
        for name, state in phases.items():
            for field in ("v", "activation", "adaptation"):
                value = getattr(state, field)
                arrays[name + "_" + field + "_sha256"] = np.array(array_hash(value))
                arrays[name + "_" + field + "_shape"] = np.array(value.shape)
        np.savez_compressed(path, **arrays)
        return harness.sha256(path)

    harness.phase_readback, harness.phase_file = readback, phase_file


def make_brain(name, seed, args, *, lateral=-0.5):
    gene = GENES[name]
    cfg = LearnerConfig(
        beta=0.1,
        eta=gene["eta"],
        eta_bias=0.02,
        free_steps=args.phase_steps,
        nudged_steps=args.phase_steps,
        tolerance=args.tolerance,
        temperature=0.2,
        normalize=gene["normalize"],
        momentum=gene["momentum"],
        qualified=True,
        damping=gene["damping"],
    )
    # This microscope predates the width-dependent compose default. Its original
    # motor circuit is part of each frozen gene, rather than a library default.
    # A different lateral gene must be selected explicitly by its own capsule.
    model = Brain.compose(
        650, 36, modules=(32, 16), observers=(), lateral=lateral, seed=seed, learning=cfg,
            recall=False
    )
    graph = model.brain
    bias = graph.bias.copy()
    bias[model.sensory_index] = gene["sensory_bias"]
    for population in ("module_0", "association"):
        bias[np.asarray(graph.connectome.populations[population])] = gene["hidden_bias"]
    model.learner.brain = graph.with_parameters(bias=bias)
    if gene["leak"] != graph.neuron_model.leak:
        # The rule form is unchanged; a neuron-local leak is an explicit candidate gene.
        graph = model.brain
        model.learner.brain = type(graph)(
            graph.connectome,
            graph.neuron_model.replace(leak=gene["leak"]),
            efficacy=graph.efficacy,
            bias=graph.bias,
            log_gain=graph.log_gain,
            backend=graph.backend,
        )
    if gene["fixed_motor_lateral"]:
        wire = graph.connectome
        model.learner.plastic_synapses = ~(
            np.isin(wire.pre, model.motor_index) & np.isin(wire.post, model.motor_index)
        )
    return model


def credited_families(panel, predictions):
    return sum(
        bool(
            np.all(
                np.asarray(predictions)[panel["families"] == family]
                == panel["labels"][panel["families"] == family]
            )
        )
        for family in np.unique(panel["families"])
    )


def freeze_sources(out):
    import cadence

    root = Path(cadence.__file__).resolve().parent
    for source in sorted(root.rglob("*.py")):
        target = out / "source/library/cadence" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source in (
        Path(__file__),
        Path(harness.__file__),
        Path(harness.__file__).with_name("extract.py"),
    ):
        shutil.copyfile(source, out / "source" / source.name)
    relations.freeze(out / "source/fixture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--genes", nargs="+", choices=tuple(GENES), default=list(GENES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--check-every", type=int, default=8)
    parser.add_argument("--phase-steps", type=int, default=4096)
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument("--seconds-per-gene", type=float, default=90)
    parser.add_argument("--max-output-mib", type=float, default=64)
    args = parser.parse_args()
    if not 0 < args.tolerance <= 0.003:
        parser.error("tolerance must be positive and no weaker than .003")
    if min(args.updates, args.check_every, args.phase_steps, args.seconds_per_gene) <= 0:
        parser.error("resource bounds must be positive")
    if not 0 < args.max_output_mib < 90 or args.seed < 0:
        parser.error("output bound must be below 90 MiB and seed nonnegative")
    if len(set(args.genes)) != len(args.genes):
        parser.error("each gene is evaluated once")
    args.out.mkdir(parents=True, exist_ok=False)
    stages = [[0, 9], [0, 9, 17, 22], list(range(24))]
    protocol = {
        "schema": "cadence-identifiable-relation-development-v1",
        "relations_protocol_sha256": relations.protocol_hash(),
        "relations_protocol": relations.protocol(),
        "library_sources": harness.sources(),
        "runner_sha256": harness.sha256(Path(__file__)),
        "phase_harness_sha256": harness.sha256(Path(harness.__file__)),
        "phase_harness_dependency_sha256": {
            "extract.py": harness.sha256(Path(harness.__file__).with_name("extract.py"))
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "genes": {name: GENES[name] for name in args.genes},
        "founder_seeds": [args.seed],
        "stages": stages,
        "architecture": {"inputs": 650, "actions": 36, "modules": [32, 16], "observers": []},
        "resource_bounds": vars(args) | {"out": str(args.out)},
        "learning_config": {
            name: make_brain(name, args.seed, args).learner.config.to_dict() for name in args.genes
        },
        "gate": (
            "Fresh 2 then 4 perfect TRAIN; 24 has >=18 TRAIN family credits "
            "and >=36/48 independent DEV rows correct; zero refused phases and answers."
        ),
        "random_target_control": (
            "Same DEV free answers rescored against uniform labels from "
            "SeedSequence([11020261003,99,N]); no additional learning or equilibrium calls."
        ),
        "memory_boundary": (
            "No working trace, associative-memory read or write, rehearsal, "
            "private imagination or external readout."
        ),
        "scope": (
            "Development of an identifiable relation; no native-body generalization, "
            "lifelong retention or efficiency advantage."
        ),
        "test_boundary": "No TEST arrays are accessed.",
        "recording": (
            "Accepted phases retain independent per-row residual/cache/cost and raw-vector "
            "hashes; updates retain pre/post parameter/optimizer hashes. Full raw states "
            "are saved for refusals only. Initial/final checkpoints and producing source "
            "permit deterministic replay; accepted trajectories remain independently "
            "unverified until that replay is performed."
        ),
    }
    harness.write_json(args.out / "protocol.json", protocol)
    freeze_sources(args.out)
    compact_phase_recording()
    summary = {"protocol_sha256": harness.sha256(args.out / "protocol.json"), "outcomes": []}
    for gene in args.genes:
        deadline = time.monotonic() + args.seconds_per_gene
        outcome = {"gene": gene, "stages": []}
        summary["outcomes"].append(outcome)
        for classes in stages:
            if time.monotonic() >= deadline:
                outcome["status"] = "time_limit"
                break
            model = make_brain(gene, args.seed, args)
            train_panel = relations.load_panel("train", families=classes)
            dev_panel = relations.load_panel("development", families=classes)
            inputs, labels = train_panel["inputs"], train_panel["labels"]
            newborn_dev = harness.free_recall(model, dev_panel["inputs"], dev_panel["labels"])
            folder = args.out / (gene + "-" + str(len(classes)))
            receipt = harness.train(model, inputs, labels, folder, args, deadline, args.out)
            record = {
                "classes": len(classes),
                "folder": folder.name,
                "status": receipt["status"],
                "accepted_updates": receipt["accepted_updates"],
                "accepted_row_exposures": receipt["accepted_row_exposures"],
                "work": receipt["work"],
                "train": receipt["recall"][-1],
                "newborn": receipt["recall"][0],
                "newborn_development": newborn_dev,
            }
            outcome["stages"].append(record)
            record["development"] = harness.free_recall(
                model, dev_panel["inputs"], dev_panel["labels"], folder=folder, name="development"
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([relations.TASK_SEED, 99, len(classes)])
            )
            random_labels = rng.integers(0, 36, len(dev_panel["labels"]))
            record["random_target"] = {
                "labels": random_labels.tolist(),
                "correct": int(np.sum(random_labels == record["development"]["predictions"])),
                "examples": len(random_labels),
                "additional_equilibrium_calls": 0,
                "label_comparisons": len(random_labels),
            }
            record["train_family_credits"] = credited_families(
                train_panel, record["train"]["predictions"]
            )
            record["development_family_credits"] = credited_families(
                dev_panel, record["development"]["predictions"]
            )
            record["passed"] = (
                receipt["status"] not in ("refused_learning", "qualification_violation")
                and receipt["accepted_row_exposures"] > 0
                and record["train"]["correct"] > record["newborn"]["correct"]
                and record["train_family_credits"] >= (18 if len(classes) == 24 else len(classes))
                and record["train"]["refusals"] == 0
                and (len(classes) < 24 or record["development"]["correct"] >= 36)
                and record["development"]["refusals"] == 0
            )
            resumed = Brain.load(folder / "final.npz")
            loaded = harness.free_recall(resumed, inputs, labels, folder=folder, name="saved_train")
            record["save_load"] = {
                "predictions_equal": loaded["predictions"] == record["train"]["predictions"],
                "config_equal": resumed.learner.config == model.learner.config,
                "query_work": loaded["work"],
            }
            record["acquisition_passed"] = record["passed"]
            record["checkpoint_passed"] = all(
                record["save_load"][name] for name in ("predictions_equal", "config_equal")
            )
            if record["acquisition_passed"] and record["checkpoint_passed"]:
                record["continuation"] = harness.continuation(model, inputs, labels, folder)
                record["checkpoint_passed"] = record["continuation"]["passed"]
            record["passed"] = record["acquisition_passed"] and record["checkpoint_passed"]
            harness.write_json(args.out / "summary.json", summary)
            print(
                json.dumps(
                    {
                        "gene": gene,
                        "classes": len(classes),
                        "updates": receipt["accepted_updates"],
                        "train": record["train"]["correct"],
                        "development": record["development"]["correct"],
                        "passed": record["passed"],
                        "status": receipt["status"],
                    }
                ),
                flush=True,
            )
            if not record["passed"]:
                outcome["status"] = "prerequisite_failed"
                break
        else:
            outcome["status"] = "acquisition_passed"
        harness.write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
