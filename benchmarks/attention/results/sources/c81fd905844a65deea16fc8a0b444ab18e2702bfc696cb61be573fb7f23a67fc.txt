"""Finish a frozen H2 recipe from research arrays at a reset round boundary.

This is not a Brain checkpoint. The original negative comparison and all failed
comparators remain evidence; resuming two capped lives establishes no head advantage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "heads-continuation-protocol.json"
SPEC = importlib.util.spec_from_file_location("heads_suffix_relevance", HERE / "heads_relevance.py")
assert SPEC is not None and SPEC.loader is not None
relevance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = relevance
SPEC.loader.exec_module(relevance)
admission, task, sr = relevance.admission, relevance.task, relevance.sr
PARAMETERS = ("efficacy", "bias", "log_gain")
OPTIMIZER = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
KIND = "cadence-attention-heads-continuation/1"


def declared():
    return json.loads(PROTOCOL.read_text())


def recipe():
    admission.PROTOCOL = admission.protocol_path(declared()["base_protocol_file"])
    return json.loads(admission.PROTOCOL.read_text())


def sources():
    recipe()
    paths = [("benchmarks/attention/heads_continuation.py", Path(__file__).resolve()),
             ("benchmarks/attention/heads-continuation-protocol.json", PROTOCOL),
             *relevance.sources(),
             ("benchmarks/attention/results/" + declared()["parent_relevance_file"],
              HERE / "results" / declared()["parent_relevance_file"]),
             ("benchmarks/attention/results/" + declared()["intervening_diagnostic"]["file"],
              HERE / "results" / declared()["intervening_diagnostic"]["file"]),
             diagnostic_helper()]
    assert len(dict(paths)) == len(paths)
    return paths


def diagnostic_helper():
    relative = ("benchmarks/attention/results/sources/"
                + declared()["intervening_diagnostic"]["helper_sha256"] + ".txt")
    return relative, HERE.parent.parent / relative


def diagnostic(parent):
    """Bind the intervening diagnostic and charge its measured host work once."""
    spec = declared()
    frozen = spec["intervening_diagnostic"]
    path = HERE / "results" / frozen["file"]
    receipt = Receipt.read(path)
    valid, message = Receipt.verify(path, sources=[*relevance.sources(), diagnostic_helper()])
    body = receipt.body
    if (not valid or receipt.kind != frozen["kind"] or receipt.digest != frozen["digest"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != frozen["archive_sha256"]
            or body["reference_receipt_digest"] != parent.digest
            or body["previous_host_seconds"] != parent.body["cumulative_host_seconds"]
            or body["elapsed_host_seconds"] != frozen["elapsed_host_seconds"]
            or body["summary"]["elapsed_host_seconds"] != frozen["elapsed_host_seconds"]
            or body["summary"]["helper_sha256"] != frozen["helper_sha256"]
            or body["summary"]["new_training_updates"] != 0
            or body["summary"]["alternate_dt_tolerance_sweep_probes"] != 0
            or body["cumulative_host_seconds"] != spec["previous_host_seconds"]
            or spec["previous_host_seconds"] != (spec["parent_relevance_host_seconds"]
                                                 + frozen["elapsed_host_seconds"])):
        raise ValueError("intervening diagnostic custody or charge differs: " + message)
    return receipt


def parents():
    spec = declared()
    recipe()
    path = HERE / "results" / spec["parent_relevance_file"]
    parent = Receipt.read(path)
    valid, message = Receipt.verify(path, sources=relevance.sources(), check=relevance.check)
    capacity = relevance.verify_capacity(HERE / "results" / spec["capacity_file"])
    if (not valid or parent.kind != "cadence-attention-heads-relevance/1"
            or parent.digest != spec["parent_relevance_digest"]
            or capacity.digest != spec["capacity_digest"]
            or parent.body["capacity_reference"] != capacity.digest
            or parent.body["passed"] is not False or parent.body["selected"] is not None
            or parent.body["candidate_rounds_used"] != 1
            or parent.body["confirmation_seeds_used"]
            or parent.body["cumulative_host_seconds"] != spec["parent_relevance_host_seconds"]):
        raise ValueError("continuation parent evidence differs: " + message)
    diagnostic(parent)
    for point in spec["points"]:
        original = parent_point(parent, point["seed"])
        train = original["training"]
        if (train["stop"] != "wall_budget" or train["completed"]
                or train["rounds"] != point["completed_rounds"]
                or train["rounds"] + point["remaining_rounds"] != 512
                or train["work"]["action_refusals"] or train["work"]["lesson_refusals"]
                or original["passed"] or original["mode"] != "agreement"):
            raise ValueError("continuation needs the exact capped H2 round boundary")
    return parent


def parent_point(parent, seed):
    points = [r for r in parent.body["runs"] if r["arm"] == "h2" and r["seed"] == seed]
    if len(points) != 1:
        raise ValueError("missing or duplicated parent H2 founder")
    return points[0]


def state_digest(brain):
    arrays = {name: getattr(brain.brain, name) for name in PARAMETERS}
    arrays.update({"optimizer_" + name: getattr(brain.learner, name) for name in OPTIMIZER})
    return canonical_sha256({
        "arrays": {name: {"shape": value.shape, "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                   for name, value in arrays.items()},
        "updates": brain.learner.updates, "contrast_updates": brain.learner.contrast_updates})


def require_reset_recipe(brain):
    if (brain.learned_code is not None or brain.sparse_code is not None
            or brain.hippocampus is not None or brain.memory_choice or brain.arousal is not None
            or brain.learner.config.surprise or not brain.learner.config.qualified
            or brain.recall is None or brain.recall.address != 0
            or brain.recall.address_reward != 0 or brain.recall.window != 1):
        raise ValueError("research restoration supports only this raw greedy reset recipe")


def restore_arrays(brain, arrays, metadata):
    """Restore parameters, every optimizer array and both counters; perform no solve."""
    require_reset_recipe(brain)
    if metadata["learning"] != brain.learner.config.to_dict():
        raise ValueError("research learning config differs")
    for name in PARAMETERS + tuple("optimizer_" + key for key in OPTIMIZER):
        target = (getattr(brain.learner, name.removeprefix("optimizer_"))
                  if name.startswith("optimizer_") else getattr(brain.brain, name))
        value = np.asarray(arrays[name])
        if (value.dtype != target.dtype or value.shape != target.shape
                or not np.isfinite(value).all()
                or ("second_moment" in name and (value < 0).any())):
            raise ValueError("invalid research array: " + name)
    updates, contrasts = metadata["updates"], metadata["contrast_updates"]
    if (type(updates) is not int or type(contrasts) is not int
            or not 0 <= contrasts <= updates):
        raise ValueError("invalid research update counters")
    # Validation is complete before any assignment.
    brain.learner.brain = brain.brain.with_parameters(
        **{name: arrays[name] for name in PARAMETERS})
    for name in OPTIMIZER:
        setattr(brain.learner, name, arrays["optimizer_" + name].copy())
    brain.learner.updates, brain.learner.contrast_updates = updates, contrasts


def restore(path, original, parent, protocol):
    brain = admission.make_brain("h2", original["seed"], protocol, "agreement")
    require_reset_recipe(brain)
    initial = brain.brain.efficacy.copy()
    if (sr.parameter_digest(brain) != original["initial_parameters"]
            or canonical_sha256(initial) != original["initial_interface_sha256"]
            or brain.brain.neuron_model.to_dict() != original["neuron_model"]
            or brain.describe()["genes"] != original["resolved"]["genes"]):
        raise ValueError("fresh composed anatomy/parameters/config differ from parent")
    record = original["parameter_snapshot"]
    if path.name != record["file"] or hashlib.sha256(path.read_bytes()).hexdigest() != record[
            "archive_sha256"]:
        raise ValueError("research snapshot archive differs")
    with np.load(path, allow_pickle=False) as saved:
        expected = {*PARAMETERS, *("optimizer_" + name for name in OPTIMIZER), "metadata"}
        if set(saved.files) != expected:
            raise ValueError("research snapshot inventory differs")
        arrays = {name: saved[name].copy() for name in saved.files}
    if canonical_sha256({name: value.tolist() for name, value in arrays.items()}) != record[
            "content_sha256"]:
        raise ValueError("research snapshot content differs")
    metadata = json.loads(str(arrays.pop("metadata")))
    point = next(p for p in declared()["points"] if p["seed"] == original["seed"])
    if (metadata["schema"] != "cadence-attention-research-parameters/1"
            or metadata["arm"] != "h2" or metadata["seed"] != original["seed"]
            or metadata["mode"] != "agreement" or metadata["source"] != parent.source
            or metadata["source_revision"] != parent.body["source_revision"]
            or metadata["parameter_sha256"] != original["final_parameters"]
            or canonical_sha256(metadata) != point["metadata_sha256"]):
        raise ValueError("research snapshot source/point metadata differs")
    restore_arrays(brain, arrays, metadata)
    if (sr.parameter_digest(brain) != original["final_parameters"]
            or state_digest(brain) != point["restored_state_sha256"]):
        raise ValueError("restored parameters differ from parent")
    custody = {"archive": record, "metadata": metadata,
               "initial_parameters": original["initial_parameters"],
               "initial_interface_sha256": canonical_sha256(initial),
               "restored_parameters": sr.parameter_digest(brain),
               "restored_state_sha256": state_digest(brain)}
    return brain, initial, custody


def add_work(first, second):
    if first.keys() != second.keys():
        raise ValueError("work inventories differ")
    return {name: (max(first[name], second[name]) if name == "max_action_residual" else
                   first[name] + second[name]) for name in first}


def joined_training(before, suffix, rounds):
    completed = before["rounds"] + suffix["rounds"]
    return {"completed": completed == rounds and suffix["completed"],
            "rounds": completed, "stop": suffix["stop"],
            "wrong_counts": before["wrong_counts"] + suffix["wrong_counts"],
            "work": add_work(before["work"], suffix["work"])}


def finish(brain, initial, original, protocol, deadline, directory):
    """Existing training/evaluation/retention procedures, only on the missing suffix."""
    cost = {name: 0 for name in original["work"]}
    schedule = task.draw(original["seed"], 512, 16)[original["training"]["rounds"]:]
    with admission.metered(cost):
        suffix = admission.train(brain, "h2", schedule, min(
            deadline, time.monotonic() + declared()["budget"]["training_host_seconds_per_call"]))
        final = sr.parameter_digest(brain)
        evaluation = {name: relevance.evaluate_conditions(
            brain, "h2", task.panel(name), initial, deadline,
            interventions=False)["conditions"]["intact"] for name in ("train", "held_out", "all")}
        causal = relevance.evaluate_conditions(brain, "h2", task.panel("all"), initial, deadline)
        result = deepcopy(original)
        result.update(training=joined_training(original["training"], suffix, 512),
                      final_parameters=final, evaluation=evaluation, causal=causal,
                      parameters_preserved=sr.parameter_digest(brain) == final,
                      resolved=brain.describe())
        result["metrics"] = relevance.metrics(result, protocol)
        retention = {"status": "not_run_primary_failed", "passed": False}
        if result["metrics"]["primary_passed"]:
            retention = {"status": "not_run_budget", "passed": False}
            if time.monotonic() < deadline:
                twin = admission.binding.fork(brain)
                continued = admission.train(twin, "h2", relevance.retention_schedule(
                    original["seed"], protocol), min(deadline, time.monotonic()
                    + declared()["budget"]["training_host_seconds_per_call"]))
                retained = {name: relevance.evaluate_conditions(
                    twin, "h2", task.panel(name), initial, deadline,
                    interventions=False)["conditions"]["intact"] for name in ("train", "held_out")}
                retention = {"status": "completed" if continued["completed"] else "capped",
                    "training": continued, "evaluation": retained,
                    "loss": result["metrics"]["held_out_exact_pair_accuracy"]
                    - retained["held_out"]["scores"]["exact_pair_accuracy"]}
                retention["passed"] = relevance.retention_passes(retention, protocol)
        result["retention"] = retention
    result["work"] = add_work(original["work"], cost)
    result["passed"] = result["metrics"]["primary_passed"] and retention["passed"]
    result["representation_probe"] = relevance.representation_probe(result, deadline)
    result["parameter_snapshot"] = relevance.parameter_snapshot(
        brain, directory, arm="h2", seed=original["seed"], mode="agreement")
    return {"seed": original["seed"], "suffix_training": suffix,
            "suffix_schedule_sha256": canonical_sha256(schedule), "additional_work": cost,
            "final_state_sha256": state_digest(brain), "result": result}


def combined(parent, points, elapsed):
    body = deepcopy(parent.body)
    new = {p["seed"]: p["result"] for p in points}
    body["runs"] = [new.get(r["seed"], r) if r["arm"] == "h2" else r for r in body["runs"]]
    passed = len(points) == 2 and all(p["result"]["passed"] for p in points)
    combined_elapsed = (parent.body["elapsed_host_seconds"]
                        + declared()["intervening_diagnostic"]["elapsed_host_seconds"] + elapsed)
    body.update(passed=passed, selected={"mode": "agreement", "arm": "h2"} if passed else None,
                elapsed_host_seconds=combined_elapsed,
                cumulative_host_seconds=body["capacity_host_seconds"] + combined_elapsed)
    return body


def check(body):
    try:
        spec, parent = declared(), parents()
        if (body["protocol"] != spec or body["parent_reference"] != parent.digest
                or body["diagnostic_reference"] != spec["intervening_diagnostic"]["digest"]
                or body["previous_host_seconds"] != spec["previous_host_seconds"]
                or body["new_candidate_rounds_used"] != 0 or body["confirmation_seeds_used"]
                or body["head_advantage_claimed"] is not False):
            return "continuation scope or parent differs"
        points = body["points"]
        if [p["seed"] for p in points] != [0, 1]:
            return "continuation founders differ"
        for point, declared_point in zip(points, spec["points"], strict=True):
            old = parent_point(parent, point["seed"])
            custody = point["restoration"]
            if (custody["archive"] != old["parameter_snapshot"]
                    or canonical_sha256(custody["metadata"]) != declared_point["metadata_sha256"]
                    or custody["restored_state_sha256"] != declared_point["restored_state_sha256"]
                    or custody["metadata"]["source"] != parent.source
                    or custody["metadata"]["source_revision"] != parent.body["source_revision"]
                    or custody["metadata"]["learning"] != old["learning_config"]
                    or custody["metadata"]["parameter_sha256"] != old["final_parameters"]
                    or custody["initial_parameters"] != old["initial_parameters"]
                    or custody["initial_interface_sha256"] != old["initial_interface_sha256"]
                    or custody["restored_parameters"] != old["final_parameters"]):
                return "restoration custody differs"
            if (point["additional_work"].keys() != old["work"].keys()
                    or any(type(value) is not int or value < 0
                           for value in point["additional_work"].values())):
                return "additional work differs"
            if (not np.isfinite(point["elapsed_host_seconds"])
                    or point["elapsed_host_seconds"] < 0
                    or point["result"]["elapsed_host_seconds"]
                    != old["elapsed_host_seconds"] + point["elapsed_host_seconds"]):
                return "point host charge differs"
            suffix = point["suffix_training"]
            count = suffix["rounds"]
            schedule = task.draw(point["seed"], 512, 16)[declared_point["completed_rounds"]:]
            work = suffix["work"]
            template = admission.asdict(sr.Work())
            if (work.keys() != template.keys()
                    or any(not isinstance(value, (int, float)) or not np.isfinite(value)
                           or value < 0
                           or (name != "max_action_residual" and type(value) is not int)
                           for name, value in work.items())):
                return "suffix work is invalid"
            if (type(count) is not int or not 0 <= count <= len(schedule)
                    or len(suffix["wrong_counts"]) != count
                    or any(type(n) is not int or not 0 <= n <= 16 for n in suffix["wrong_counts"])
                    or suffix["completed"] != (count == len(schedule))
                    or (count == len(schedule) and suffix["stop"] is not None)
                    or (count < len(schedule) and suffix["stop"] not in (
                        "wall_budget", "refused_prefix", "refused_query"))
                    or point["suffix_schedule_sha256"] != canonical_sha256(schedule)
                    or point["result"]["training"]
                    != joined_training(old["training"], suffix, 512)
                    or point["result"]["work"] != add_work(old["work"], point["additional_work"])):
                return "suffix schedule, completion or work differs"
        elapsed = body["elapsed_host_seconds"]
        if (not np.isfinite(elapsed) or elapsed < 0
                or sum(p["elapsed_host_seconds"] for p in points) > elapsed
                or body["cumulative_host_seconds"] != spec["previous_host_seconds"] + elapsed):
            return "continuation cumulative charge differs"
        expected = combined(parent, points, elapsed)
        if expected != body["combined_relevance"]:
            return "original comparisons or combined readings differ"
        problem = relevance.check(expected)
        if problem:
            return "combined relevance: " + problem
        if body["passed"] != expected["passed"]:
            return "continuation gate differs"
    except (KeyError, TypeError, ValueError, OSError):
        return "incomplete or invalid continuation receipt"
    return None


def run(parameters: Path, output: Path):
    started = time.monotonic()
    spec, protocol, parent = declared(), recipe(), parents()
    manifest = source_manifest(sources())
    deadline = started + spec["budget"]["additional_host_seconds"]
    points, journal = [], output.with_suffix(".partial.json")
    directory = output.parent / "parameters"
    for point in spec["points"]:
        point_started = time.monotonic()
        old = parent_point(parent, point["seed"])
        brain, initial, custody = restore(parameters / old["parameter_snapshot"]["file"],
                                          old, parent, protocol)
        output.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(canonical_json({"status": "point_started_unverified",
            "seed": point["seed"], "restoration": custody, "completed_points": points,
            "protocol": spec, "parent_reference": parent.digest, "source": manifest,
            "elapsed_host_seconds": time.monotonic() - started}) + "\n")
        result = finish(brain, initial, old, protocol, deadline, directory)
        result["restoration"] = custody
        result["elapsed_host_seconds"] = time.monotonic() - point_started
        result["result"]["elapsed_host_seconds"] = (
            old["elapsed_host_seconds"] + result["elapsed_host_seconds"])
        points.append(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(canonical_json({"status": "running_unverified", "points": points,
            "protocol": spec, "parent_reference": parent.digest, "source": manifest,
            "elapsed_host_seconds": time.monotonic() - started}) + "\n")
    elapsed = time.monotonic() - started
    readings = combined(parent, points, elapsed)
    body = {"protocol": spec, "parent_reference": parent.digest, "points": points,
            "diagnostic_reference": spec["intervening_diagnostic"]["digest"],
            "previous_host_seconds": spec["previous_host_seconds"],
            "elapsed_host_seconds": elapsed,
            "cumulative_host_seconds": spec["previous_host_seconds"] + elapsed,
            "new_candidate_rounds_used": 0, "confirmation_seeds_used": [],
            "head_advantage_claimed": False, "combined_relevance": readings,
            "passed": readings["passed"], "source_revision": sr.revision()}
    journal.write_text(canonical_json({"status": "complete_unverified", "body": body,
                                      "source": manifest}) + "\n")
    if manifest != source_manifest(sources()):
        raise RuntimeError("source changed; retained journal is not a verified result")
    problem = check(body)
    if problem:
        raise ValueError(problem)
    return Receipt.build(KIND, body, sources())


def verify(path):
    if Receipt.read(path).kind != KIND:
        return False, "not an attention continuation receipt"
    return Receipt.verify(path, sources=sources(), check=check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = verify(args.verify)
        print(message)
        raise SystemExit(0 if ok else 1)
    if args.parameters is None or args.out is None:
        parser.error("provide --parameters and --out")
    if args.out.exists() or args.out.with_suffix(".partial.json").exists():
        raise FileExistsError(args.out)
    run(args.parameters, args.out).write(args.out)
