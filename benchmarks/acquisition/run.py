"""Bounded finite/qualified phase comparison on actual sensory/action witnesses."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from extract import row_hash, sha256, write_json

import cadence
from cadence import Brain, LearnerConfig


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def sources():
    root = Path(cadence.__file__).resolve().parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def freeze_sources(folder, fixture):
    """Retain exact source bytes, not only hashes of a mutable dirty checkout."""
    root = Path(cadence.__file__).resolve().parent
    for source in sorted(root.rglob("*.py")):
        target = folder / "library" / "cadence" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("run.py", "extract.py", "verify.py"):
        shutil.copyfile(Path(__file__).with_name(name), folder / name)
    for name in ("school.npz", "provenance.json"):
        target = folder / "fixture" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture / name, target)


def activation(model, voltage):
    """Independent sigmoid, rebased at rest, including the negative leak."""
    with np.errstate(over="ignore"):
        rest = 1 / (1 + math.exp(model.slope * model.threshold))
        sigmoid = 1 / (1 + np.exp(-model.slope * (voltage - model.threshold)))
    difference = sigmoid - rest
    return np.where(difference > 0, difference / (1 - rest), difference * model.leak / rest)


def independent_residual(brain, drive, state, labels=None, sign=0, *, weights=None, bias=None):
    """Literal edge scatter against the original, undamped equations."""
    graph, cfg = brain.brain, brain.learner.config
    wire, model = graph.connectome, graph.neuron_model
    if weights is None:
        weights = model.gain * wire.count * graph.efficacy * np.exp(graph.log_gain[wire.pre])
    if bias is None:
        bias = graph.bias
    s = activation(model, np.atleast_2d(state.v))
    incoming = np.stack(
        [np.bincount(wire.post, weights=weights * row[wire.pre], minlength=wire.n) for row in s]
    )
    defect = incoming + drive + bias - state.v
    if model.adaptation is not None:
        defect -= model.adaptation.strength * state.adaptation
    if sign:
        motor = brain.motor_index
        target = np.eye(len(motor))[labels]
        if cfg.nudge == "cross_entropy":
            logits = s[:, motor] / cfg.temperature
            exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
            prediction = exponents / exponents.sum(axis=1, keepdims=True)
        else:
            prediction = s[:, motor]
        defect[:, motor] += sign * cfg.beta * (target - prediction)
    residual = np.abs(defect).max(axis=1)
    if model.adaptation is not None:
        residual = np.maximum(residual, np.abs(s - state.adaptation).max(axis=1))
    return residual, np.abs(s - state.activation).max(axis=1)


def reference_contrast(brain, phases):
    """Extended-precision unfactored products, independent of optimized contrasts."""
    if "nudged" not in phases or "opposite" not in phases:
        return None
    free, plus, minus = (phases[name] for name in ("free", "nudged", "opposite"))
    actual_edges, actual_bias = brain.learner.contrast(free, plus, minus)
    plus, minus = plus.activation.astype(np.longdouble), minus.activation.astype(np.longdouble)
    wire, span = brain.connectome, 2 * brain.learner.config.beta
    edges = (
        (plus[:, wire.pre] * plus[:, wire.post] - minus[:, wire.pre] * minus[:, wire.post]).mean(
            axis=0
        )
        / span
    ).astype(float)
    biases = ((plus - minus).mean(axis=0) / span).astype(float)
    return {
        "max_edge_deviation": float(np.abs(edges - actual_edges).max(initial=0)),
        "max_bias_deviation": float(np.abs(biases - actual_bias).max(initial=0)),
        "reference": "unfactored extended-precision endpoint products",
        "gradient_boundary": (
            "Arithmetic parity alone is not a gradient certificate; effective-weight symmetry, "
            "stable smooth branch, contact/gain derivatives and small nudge remain necessary."
        ),
    }


def phase_readback(brain, drive, labels, phases, weights, bias):
    reports = {}
    for name, state in phases.items():
        residual, cache_defect = independent_residual(
            brain,
            drive,
            state,
            labels,
            {"free": 0, "nudged": 1, "opposite": -1}[name],
            weights=weights,
            bias=bias,
        )
        reports[name] = {
            "steps": int(state.steps),
            "full_residual": residual.tolist(),
            "cache_defect": cache_defect.tolist(),
            "qualified": (residual <= brain.learner.config.tolerance).tolist(),
            "motor_activity": state.activation[:, brain.motor_index].tolist(),
        }
    return reports


def phase_file(path, phases):
    arrays = {
        name + "_" + field: getattr(state, field)
        for name, state in phases.items()
        for field in ("v", "activation", "adaptation")
    }
    np.savez_compressed(path, **arrays)
    return sha256(path)


def recipe_configuration(recipe, args):
    """Resolve effective genes before numerical work, including fixed overrides."""
    cfg = LearnerConfig(
        beta=0.1,
        eta=0.5,
        eta_bias=0.02,
        free_steps=1024,
        nudged_steps=12,
        tolerance=0.003,
        nudge="cross_entropy",
        temperature=0.2,
        momentum=0.9,
    )
    if recipe == "qualified":
        if not hasattr(cfg, "qualified"):
            raise RuntimeError("this source does not provide opt-in qualified learning")
        small_rms = getattr(args, "gene", "canonical") == "lateral0-small-rms"
        cfg = replace(
            cfg,
            qualified=True,
            damping=args.damping,
            eta=0.003 if small_rms else args.rate,
            eta_bias=0.0003 if small_rms else cfg.eta_bias,
            free_steps=args.free_steps,
            nudged_steps=args.nudged_steps,
            nudge=args.nudge,
            tolerance=getattr(args, "tolerance", 0.003),
        )
        if getattr(args, "gene", "canonical") == "fixed-lateral-local-rms":
            cfg = replace(cfg, eta=0.005, normalize=0.99)
        if getattr(args, "gene", "canonical") in ("lateral0-local-rms", "lateral0-resting"):
            # Transfer the no-lateral/local-RMS choices from the keyword control;
            # this school has different inputs, architecture, rates and data.
            cfg = replace(cfg, eta=args.rate, normalize=0.99, normalize_floor=1e-4)
        if getattr(args, "gene", "canonical") == "lateral0-small-rms":
            # Transfer the separately confirmed cue-school teacher recipe. The
            # native movie is a different task and still needs its own evidence.
            cfg = replace(cfg, eta=0.003, eta_bias=0.0003, normalize=0.99, momentum=0.0)
    gene = getattr(args, "gene", "canonical") if recipe == "qualified" else "canonical"
    return {
        "gene": gene,
        "learning": cfg.to_dict(),
        "lateral": 0.0
        if gene in ("lateral0-local-rms", "lateral0-resting", "lateral0-small-rms")
        else -0.5,
        "resting_bias": 0.5 if gene == "lateral0-resting" else 0.0,
        "sensory_bias": 0.6 if gene == "fixed-lateral-local-rms" else 0.0,
        "freeze_motor_lateral": gene == "fixed-lateral-local-rms",
    }


def resting(brain: Brain, value: float) -> Brain:
    """The resting bias the composed brain took up to 0.84.3 (``resting_bias``): ``value`` as
    the initial bias of every processing neuron, the modules, the observers and the
    association region, with the boundary regions (sensory, visual, prefrontal, motor,
    efference, history, recall, next and code) at zero. The arrays are those the option
    built."""
    populations = brain.connectome.populations
    boundary = {"sensory", "visual", "prefrontal", "motor", "efference", "history", "recall",
                "next", "code"}
    selected = np.zeros(brain.connectome.n, dtype=bool)
    excluded = np.zeros(brain.connectome.n, dtype=bool)
    for name, members in populations.items():
        target = excluded if name.split("/", 1)[0] in boundary else selected
        target[np.asarray(members, dtype=np.int64)] = True
    bias = np.asarray(brain.brain.bias, dtype=float).copy()
    bias[selected & ~excluded] = value
    brain.learner.brain = brain.learner.brain.with_parameters(bias=bias)
    return brain


def make_brain(recipe, seed, args):
    effective = recipe_configuration(recipe, args)
    cfg = LearnerConfig(**effective["learning"])
    brain = Brain.compose(
        backend="cpu", inputs=650, actions=36, modules=(32, 16), observers=(), seed=seed, learning=cfg,
        lateral=effective["lateral"], recall=False,
    )
    if effective["resting_bias"]:
        resting(brain, effective["resting_bias"])
    if effective["freeze_motor_lateral"]:
        graph = brain.brain
        bias = graph.bias.copy()
        bias[brain.sensory_index] = effective["sensory_bias"]
        brain.learner.brain = graph.with_parameters(bias=bias)
        motor_pre = np.isin(graph.connectome.pre, brain.motor_index)
        motor_post = np.isin(graph.connectome.post, brain.motor_index)
        brain.learner.plastic_synapses = ~(motor_pre & motor_post)
    return brain


def library_identity():
    """The imported source version may differ from installed wheel metadata."""
    try:
        installed = importlib.metadata.version("cadence-net")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    return {
        "cadence_version": cadence.__version__,
        "installed_distribution_version": installed,
        "cadence_module": str(Path(cadence.__file__).resolve()),
    }


def free_recall(brain, inputs, labels, *, folder=None, name="recall"):
    """The exact public predict solve path, retaining refused states and their work."""
    predictions, records, sweeps, checks = [], [], 0, 0
    drive = brain.stimulus(inputs, memory=False)
    for row, truth in enumerate(labels):
        phase = brain._equilibrate(
            drive[row : row + 1],
            None,
            budget=brain.learner.config.free_steps,
            tolerance=brain.learner.config.tolerance,
        )
        residual, cache = independent_residual(brain, drive[row : row + 1], phase.state)
        qualified = bool(
            np.all(phase.qualified)
            and np.all(residual <= brain.learner.config.tolerance)
            and np.all(cache <= brain.learner.config.tolerance)
        )
        prediction = (
            int(np.argmax(phase.state.activation[0, brain.motor_index])) if qualified else -1
        )
        predictions.append(prediction)
        sweeps += phase.steps
        # New qualified API reports exact residual transport counts. The historical
        # free wrapper lacks this counter: mark unavailable rather than infer it.
        count = getattr(phase, "residual_checks", None)
        checks += count if count is not None else 0
        records.append(
            {
                "row": row,
                "label": int(truth),
                "prediction": prediction,
                "steps": int(phase.steps),
                "residual": float(residual.max()),
                "cache_defect": float(cache.max()),
                "residual_checks": count,
                "stagnation_checks": getattr(phase, "stagnation_checks", None),
            }
        )
    result = {
        "correct": int(np.sum(np.asarray(predictions) == labels)),
        "examples": len(labels),
        "refusals": predictions.count(-1),
        "predictions": predictions,
        "rows": records,
        "work": {
            "calls": len(labels),
            "row_sweeps": int(sweeps),
            "reported_residual_checks": int(checks),
            "reported_stagnation_checks": sum(r["stagnation_checks"] or 0 for r in records),
            "unreported_stagnation_work": any(r["stagnation_checks"] is None for r in records),
            "independent_residual_checks": len(labels),
            "unreported_residual_work": any(r["residual_checks"] is None for r in records),
        },
    }
    if folder is not None:
        write_json(folder / (name + ".json"), result)
    return result


def microscope(brain, inputs, labels, folder):
    """Read the finite 0.70 control, before any optimizer update."""
    cfg = brain.learner.config
    if getattr(cfg, "qualified", False):
        brain.learner.config = replace(cfg, qualified=False)
    learner, graph = brain.learner, brain.brain
    drive = brain.stimulus(inputs, memory=False)
    target = learner.targets(labels)
    free = learner.free(drive)
    plus = learner.nudged(drive, free, target)
    minus = learner.nudged(drive, free, target, sign=-1)
    phases = {"free": free, "nudged": plus, "opposite": minus}
    one = graph.settle_batch(drive, state=free, steps=1)
    two = graph.settle_batch(drive, state=one, steps=1)
    weights = (
        graph.neuron_model.gain
        * graph.connectome.count
        * graph.efficacy
        * np.exp(graph.log_gain[graph.connectome.pre])
    )
    report = {
        "phases": phase_readback(brain, drive, labels, phases, weights, graph.bias),
        "phase_sha256": phase_file(folder / "microscope-phases.npz", phases),
        "contrast": reference_contrast(brain, phases),
        "one_step_voltage_movement": np.abs(one.v - free.v).max(axis=1).tolist(),
        "two_step_voltage_movement": np.abs(two.v - free.v).max(axis=1).tolist(),
        "one_step_motor_movement": np.abs(
            one.activation[:, brain.motor_index] - free.activation[:, brain.motor_index]
        )
        .max(axis=1)
        .tolist(),
        "two_step_motor_movement": np.abs(
            two.activation[:, brain.motor_index] - free.activation[:, brain.motor_index]
        )
        .max(axis=1)
        .tolist(),
        "work": {
            "phase_row_sweeps": len(labels) * sum(s.steps for s in phases.values()),
            "orbit_row_sweeps": 2 * len(labels),
            "independent_residual_checks": 3 * len(labels),
            "contrast_reference_rows": len(labels),
        },
    }
    # Check the separate library residual path before any parameter mutation.
    differences = []
    for name, state in phases.items():
        sign = {"free": 0, "nudged": 1, "opposite": -1}[name]
        nudge = None if not sign else learner.nudge_for(target, sign * cfg.beta)
        original = graph.residual(drive, state, nudge=nudge, on_device=False)
        independent = np.array(report["phases"][name]["full_residual"])
        differences.append(float(np.abs(original - independent).max()))
    report["library_residual_max_deviation"] = max(differences)
    report["work"]["library_residual_checks"] = 3 * len(labels)
    brain.learner.config = cfg
    write_json(folder / "microscope.json", report)
    return report


def elapsed_limit(deadline, folder, output_limit):
    if time.monotonic() >= deadline:
        return "time_limit"
    if sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) >= output_limit:
        return "output_limit"
    return None


def train(brain, inputs, labels, folder, args, deadline, output_root):
    folder.mkdir()
    brain.save(folder / "initial.npz")
    drive = brain.stimulus(inputs, memory=False)
    receipt = {
        "status": "running",
        "attempted_updates": 0,
        "accepted_updates": 0,
        "accepted_row_exposures": 0,
        "updates": [],
        "recall": [],
        "initial_sha256": sha256(folder / "initial.npz"),
        "learner_config": brain.learner.config.to_dict(),
        "input_sha256": row_hash(inputs),
        "labels": labels.tolist(),
    }
    receipt["recall"].append({"update": 0, **free_recall(brain, inputs, labels)})
    write_json(folder / "receipt.json", receipt)
    for update in range(1, args.updates + 1):
        limit = elapsed_limit(deadline, output_root, args.max_output_mib * 1024**2)
        if limit:
            receipt["status"] = limit
            break
        graph = brain.brain
        weights = (
            graph.neuron_model.gain
            * graph.connectome.count
            * graph.efficacy
            * np.exp(graph.log_gain[graph.connectome.pre])
        )
        bias = graph.bias.copy()
        phases, failed = {}, False
        receipt["attempted_updates"] += 1
        write_json(folder / "receipt.json", receipt)
        began = time.monotonic()
        try:
            state, report = brain.learner.step(drive, labels)
            phases = {"free": state.free, "nudged": state.nudged}
            if state.opposite is not None:
                phases["opposite"] = state.opposite
            accepted = int(report.get("accepted", 1))
        except RuntimeError as error:
            if not hasattr(error, "phases") or not hasattr(error, "report"):
                raise
            phases = {name: phase.state for name, phase in error.phases.items()}
            report, accepted, failed = error.report, 0, True
            receipt["failure"] = {"phase": error.phase, "reason": str(error)}
        record = {
            "update": update,
            "report": json_value(report),
            "seconds": time.monotonic() - began,
            "phases": phase_readback(brain, drive, labels, phases, weights, bias),
            "phase_sha256": phase_file(folder / f"phases-{update:04d}.npz", phases),
        }
        for name, state in phases.items():
            record["phases"][name]["steps"] = int(report.get(name + "_steps", state.steps))
        # Work is a declared dense/sparse transport count, not elapsed hardware MACs.
        phase_sweeps = sum(
            int(report.get(name + "_steps", state.steps)) for name, state in phases.items()
        )
        record["phase_row_sweeps"] = len(labels) * phase_sweeps
        record["independent_residual_checks"] = len(labels) * len(phases)
        reported_checks = report.get("total_residual_checks")
        record["reported_residual_checks"] = reported_checks
        record["reported_row_residual_checks"] = (
            None if reported_checks is None else len(labels) * reported_checks
        )
        if accepted and getattr(brain.learner.config, "qualified", False):
            valid = len(phases) == 3 and all(
                all(reading["qualified"])
                and max(reading["cache_defect"]) <= brain.learner.config.tolerance
                for reading in record["phases"].values()
            )
            if not valid:
                failed = True
                record["qualification_violation"] = True
                receipt["failure"] = {"reason": "accepted phases fail independent qualification"}
        receipt["updates"].append(record)
        receipt["accepted_updates"] += accepted
        receipt["accepted_row_exposures"] += accepted * len(labels)
        if update % args.check_every == 0 or update == args.updates or failed:
            recall = free_recall(brain, inputs, labels)
            receipt["recall"].append({"update": update, **recall})
            if not failed and recall["correct"] == len(labels) and recall["refusals"] == 0:
                receipt["status"] = "perfect_recall"
                break
        write_json(folder / "receipt.json", receipt)
        if failed:
            receipt["status"] = "qualification_violation" if accepted else "refused_learning"
            break
    else:
        receipt["status"] = "update_limit"
    if receipt["recall"][-1]["update"] != receipt["attempted_updates"]:
        receipt["recall"].append(
            {"update": receipt["attempted_updates"], **free_recall(brain, inputs, labels)}
        )
    brain.save(folder / "final.npz")
    receipt["final_sha256"] = sha256(folder / "final.npz")
    receipt["work"] = {
        "phase_row_sweeps": sum(r["phase_row_sweeps"] for r in receipt["updates"]),
        "recall_row_sweeps": sum(r["work"]["row_sweeps"] for r in receipt["recall"]),
        "independent_phase_residual_checks": sum(
            r["independent_residual_checks"] for r in receipt["updates"]
        ),
        "reported_phase_residual_checks": sum(
            r["reported_residual_checks"] or 0 for r in receipt["updates"]
        ),
        "reported_phase_row_residual_checks": sum(
            r["reported_row_residual_checks"] or 0 for r in receipt["updates"]
        ),
        "unreported_phase_residual_work": any(
            r["reported_residual_checks"] is None for r in receipt["updates"]
        ),
        "reported_recall_residual_checks": sum(
            r["work"]["reported_residual_checks"] for r in receipt["recall"]
        ),
        "independent_recall_residual_checks": sum(
            r["work"]["independent_residual_checks"] for r in receipt["recall"]
        ),
        "refused_recall_rows": sum(r["refusals"] for r in receipt["recall"]),
        "phase_transport_estimate": brain.connectome.synapses
        * sum(r["phase_row_sweeps"] for r in receipt["updates"]),
        "refused_learning_calls": int(receipt["status"] == "refused_learning"),
        "calibration_calls": 0,
        "search_candidates": 1,
        "replay_calls": 0,
    }
    write_json(folder / "receipt.json", receipt)
    return receipt


def continuation(brain, inputs, labels, folder):
    """Verify checkpoint equivalence under one further real teaching admission."""
    resumed = Brain.load(folder / "final.npz")
    before = free_recall(brain, inputs, labels)
    loaded = free_recall(resumed, inputs, labels)
    equality = before["predictions"] == loaded["predictions"]
    attempts = []
    for label, model in (("original", brain), ("resumed", resumed)):
        try:
            _, report = model.learner.step(model.stimulus(inputs, memory=False), labels)
            attempts.append({"arm": label, "accepted": True, "report": json_value(report)})
        except RuntimeError as error:
            if not hasattr(error, "report"):
                raise
            attempts.append({"arm": label, "accepted": False, "report": json_value(error.report)})
    original_path, resumed_path = (
        folder / "continued-original.npz",
        folder / "continued-resumed.npz",
    )
    brain.save(original_path)
    resumed.save(resumed_path)
    with (
        np.load(original_path, allow_pickle=False) as left,
        np.load(resumed_path, allow_pickle=False) as right,
    ):
        arrays_equal = set(left.files) == set(right.files) and all(
            np.array_equal(left[key], right[key]) for key in left.files
        )
    result = {
        "initial_recall_equal": equality,
        "continued_checkpoint_arrays_equal": arrays_equal,
        "accepted_outcomes_equal": attempts[0]["accepted"] == attempts[1]["accepted"],
        "attempts": attempts,
        "query_work": [before["work"], loaded["work"]],
        "passed": equality and arrays_equal and attempts[0]["accepted"] == attempts[1]["accepted"],
        "scope": (
            "Exact saved continuation for this independent teaching protocol; "
            "live memory/pending body feedback have separate capability tests."
        ),
    }
    write_json(folder / "continuation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixture")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", choices=("finite", "qualified", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--check-every", type=int, default=8)
    parser.add_argument("--free-steps", type=int, default=1024)
    parser.add_argument("--nudged-steps", type=int, default=12)
    parser.add_argument("--damping", type=int, default=3)
    # a composed brain learns with the cross-entropy nudge
    parser.add_argument("--nudge", choices=("cross_entropy",), default="cross_entropy")
    parser.add_argument(
        "--rate", type=float, default=0.5,
        help="qualified synaptic rate; fixed-lateral-local-rms uses .005, lateral0-small-rms .003",
    )
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument(
        "--gene",
        choices=("canonical", "fixed-lateral-local-rms", "lateral0-local-rms",
                 "lateral0-resting", "lateral0-small-rms"),
        default="canonical",
    )
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--max-output-mib", type=int, default=64)
    args = parser.parse_args()
    if (
        not 1 <= args.updates <= 128
        or not 1 <= args.check_every <= args.updates
        or not 1 <= args.free_steps <= 4096
        or not 1 <= args.nudged_steps <= 4096
        or not 0 < args.seconds <= 300
        or not 1 <= args.max_output_mib <= 80
        or not 0 < args.rate <= 1
        or not 0 < args.tolerance <= 0.003
        or args.seed < 0
        or not 0 <= args.damping <= 8
    ):
        parser.error(
            "bounded local protocol: <=128 updates, <=4096 phase sweeps, <=300 seconds, <=80MiB"
        )
    provenance = json.loads((args.fixture / "provenance.json").read_text())
    if sha256(args.fixture / "school.npz") != provenance["fixture_sha256"]:
        raise ValueError("fixture hash mismatch")
    with np.load(args.fixture / "school.npz", allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files if not key.startswith("heldout_")}
    for name, rows in provenance["panels"].items():
        if name == "heldout":
            continue
        for row, identity in enumerate(rows):
            if row_hash(data[name + "_inputs"][row]) != identity["input_sha256"]:
                raise ValueError("panel row hash mismatch")
            if int(data[name + "_labels"][row]) != identity["label"]:
                raise ValueError("panel label mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    recipes = ("finite", "qualified") if args.recipe == "both" else (args.recipe,)
    protocol = {
        "schema": "cadence-acquisition-protocol-v1",
        "arguments": json_value(vars(args)),
        "fixture_sha256": provenance["fixture_sha256"],
        "provenance_sha256": sha256(args.fixture / "provenance.json"),
        "harness_sha256": sha256(__file__),
        "extractor_sha256": sha256(Path(__file__).with_name("extract.py")),
        "library_sources": sources(),
        **library_identity(),
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "dtype": "float64 CPU",
        "recipes": list(recipes),
        "effective_recipes": {recipe: recipe_configuration(recipe, args) for recipe in recipes},
        "seed": args.seed,
        "two_positions": [0, 9],
        "four_positions": [0, 9, 17, 22],
        "expansion_gate": (
            "perfect qualified free recall on two, then four; each stage has a fresh founder"
        ),
        "school_gate": (
            "at least18/24 with zero recall refusals, positive accepted exposure and "
            "improvement over the founder; no refused lesson or qualification violation"
        ),
        "refusal_policy": (
            "A refused teaching phase applies no update, charges its attempted work, "
            "and terminates the stage. No skip, retry or budget increase within this run."
        ),
        "confirmation_rule": (
            "Independent TRAIN18/development19 are read once after the screen. "
            "Heldout14 is sealed; five fresh-seed confirmation remains a separate frozen campaign."
        ),
        "retention_rule": (
            "Do not claim retention from same-set replay; retain pre-interference checkpoints "
            "for a separate family-disjoint teaching test."
        ),
        "resource_model": (
            "one local process, phase/recall row-sweeps and residual transports; every failed "
            "admission and saved-continuation replay charged; no calibration/search or cloud"
        ),
        "outcomes": (
            "screen_passed_confirmation_pending, screen_failed, prerequisite_failed, "
            "perfect_recall, refused_learning, qualification_violation, update_limit, "
            "time_limit, output_limit, time_or_output_limit, failed"
        ),
    }
    protocol["arguments"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in protocol["arguments"].items()
    }
    write_json(args.out / "protocol.json", protocol)
    freeze_sources(args.out / "source", args.fixture)
    began, outcomes = time.monotonic(), []
    deadline = began + args.seconds
    try:
        base = make_brain("finite", args.seed, args)
        microscope(base, data["school_inputs"], data["school_labels"], args.out)
        for recipe in recipes:
            outcome = {"recipe": recipe, "stages": [], "status": "running"}
            outcomes.append(outcome)
            for positions in ([0, 9], [0, 9, 17, 22], list(range(24))):
                if elapsed_limit(deadline, args.out, args.max_output_mib * 1024**2):
                    outcome["status"] = "time_or_output_limit"
                    break
                brain = make_brain(recipe, args.seed, args)
                folder = args.out / f"{recipe}-{len(positions)}"
                x, y = data["school_inputs"][positions], data["school_labels"][positions]
                receipt = train(brain, x, y, folder, args, deadline, args.out)
                recall = receipt["recall"][-1]
                passed = (
                    receipt["status"]
                    not in ("qualification_violation", "refused_learning", "failed")
                    and receipt["accepted_row_exposures"] > 0
                    and recall["refusals"] == 0
                    and recall["correct"] >= (len(y) if len(y) < 24 else 18)
                    and recall["correct"] > receipt["recall"][0]["correct"]
                )
                stage = {
                    "examples": len(y),
                    "status": receipt["status"],
                    "correct": recall["correct"],
                    "refusals": recall["refusals"],
                    "accepted_row_exposures": receipt["accepted_row_exposures"],
                    "newborn_correct": receipt["recall"][0]["correct"],
                    "acquired_correct_gain": recall["correct"] - receipt["recall"][0]["correct"],
                    "passed": passed,
                }
                outcome["stages"].append(stage)
                if not passed:
                    outcome["status"] = "prerequisite_failed" if len(y) < 24 else "screen_failed"
                    break
                if len(y) == 24:
                    for name in ("independent_train", "development"):
                        stage[name] = free_recall(
                            brain,
                            data[name + "_inputs"],
                            data[name + "_labels"],
                            folder=folder,
                            name=name,
                        )
                    stage["continuation"] = continuation(brain, x, y, folder)
                    outcome["status"] = "screen_passed_confirmation_pending"
            write_json(
                args.out / "summary.json",
                {"outcomes": outcomes, "seconds": time.monotonic() - began},
            )
    except Exception as error:
        write_json(args.out / "failure.json", {"type": type(error).__name__, "reason": str(error)})
        raise
    finally:
        summary = {
            "protocol_sha256": sha256(args.out / "protocol.json"),
            "outcomes": outcomes,
            "seconds": time.monotonic() - began,
            "library_sources_unchanged": sources() == protocol["library_sources"],
            "harness_source_unchanged": sha256(__file__) == protocol["harness_sha256"],
            "boundary": (
                "Selected-row acquisition is not held-out generalization, "
                "lifelong retention or gameplay improvement."
            ),
        }
        write_json(args.out / "summary.json", summary)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
