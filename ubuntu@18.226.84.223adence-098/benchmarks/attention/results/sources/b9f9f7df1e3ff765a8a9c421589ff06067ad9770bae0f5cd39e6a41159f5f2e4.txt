"""Source-bound coupled-head relevance development, after verified capacity.

All interventions fork one captured pre-query host, retaining its query, drive
and source-only cortical boundary. Teachers label final answers only. This is
a raw-token component experiment; representation and learned semantics remain
separate gates. No campaign is run on import.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from cadence.receipts import Receipt, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "heads_relevance_capacity", HERE / "heads_admission.py"
)
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)
ARMS = ("h1", "h2", "h1_wide", "h1_wide_ports", "h1_more_steps")
task, heads, sr = admission.task, admission.heads, admission.sr
representation = admission.load("heads_relevance_representation", HERE / "representation.py")
ANCHORED_PROTOCOL = "heads-anchored-protocol.json"
EMPTY_KIND = "cadence-attention-empty-boundary-admission/1"
_empty_owner = None


def empty_owner():
    global _empty_owner
    if _empty_owner is None:
        _empty_owner = admission.load(
            "heads_relevance_empty_boundary", HERE / "empty_boundary_admission.py")
    return _empty_owner


def anchored_protocol():
    return json.loads(admission.protocol_path(ANCHORED_PROTOCOL).read_text())


def sources() -> list[tuple[str, Path]]:
    paths = admission.sources() + [
        ("benchmarks/attention/heads_relevance.py", Path(__file__)),
        ("benchmarks/attention/representation.py", HERE / "representation.py"),
    ]
    if admission.PROTOCOL.name == ANCHORED_PROTOCOL:
        declared = anchored_protocol()["prerequisites"]
        paths += empty_owner().sources() + [(
            "benchmarks/attention/results/" + declared["capacity_receipt_file"],
            HERE / "results" / declared["capacity_receipt_file"])]
    if admission.PROTOCOL.name in admission.FOUNDER_PROTOCOLS:
        declared = json.loads(admission.PROTOCOL.read_text())["relevance_stage"]
        paths.append(("benchmarks/attention/results/" + declared["capacity_receipt_file"],
                      HERE / "results" / declared["capacity_receipt_file"]))
    unique = {}
    for name, path in paths:
        if name in unique and unique[name].resolve() != path.resolve():
            raise ValueError("a source path has conflicting owners")
        unique[name] = path
    return list(unique.items())


def anchored_capacity(path: Path) -> Receipt:
    """New positive evidence uses exact current sources; old exposed evidence is archived."""
    owner, protocol = empty_owner(), anchored_protocol()
    prerequisite = protocol["prerequisites"]
    receipt = Receipt.read(path)
    if receipt.kind != EMPTY_KIND or prerequisite["capacity_kind"] != EMPTY_KIND:
        raise ValueError("anchored capacity prerequisite: wrong receipt kind")
    if owner.PROTOCOL.name != prerequisite["capacity_protocol_file"]:
        raise ValueError("anchored capacity prerequisite: protocol owner differs")
    valid, message = Receipt.verify(path, sources=owner.sources(), check=owner.check)
    if not valid:
        raise ValueError("anchored capacity prerequisite: " + message)
    body = receipt.body
    if (body["passed"] is not True or body["masked_historical_reproduced"] is not True
            or body["occupied_query_overlap_identical"] is not True
            or body["candidate_rounds_used"] != 0 or body["confirmation_seeds_used"]):
        raise ValueError("anchored capacity prerequisite: acquisition or matched control failed")
    positive = [run for run in body["runs"] if run["variant"] == "anchored"]
    if ([run["seed"] for run in positive] != protocol["development_seeds"]
            or any(run["empty_anchor"] is not True or not owner.admitted(run, body["protocol"])
                   for run in positive)):
        raise ValueError("anchored capacity prerequisite: missing or failed anchored founder")
    base = json.loads(owner.BASE.read_text())
    expected_brain = {**base["brain"], "empty_anchor": True}
    admitted_settings = {
        "rounds": protocol["task"]["rounds"], "batch": protocol["task"]["batch"],
        "dt": protocol["dt"], "tolerance": protocol["brain"]["learning_tolerance"],
        "free_steps": protocol["brain"]["learning_free_steps"],
        "nudged_steps": protocol["brain"]["learning_nudged_steps"],
        "total_read_amplitude": protocol["brain"]["recall_amplitude"],
    }
    if (protocol["brain"] != expected_brain or protocol["task"] != base["task"]
            or protocol["dt"] != base["dt"] or protocol["gates"] != base["gates"]
            or any(body["protocol"].get(key) != value for key, value in admitted_settings.items())
            or body["protocol"]["gate"]["trained_floor"] != protocol["gates"]["content_use_floor"]
            or protocol["candidate_order"] != ["agreement"]
            or protocol["maximum_candidate_rounds"] != 1
            or protocol["budget"].get("additional_stage") is not True):
        raise ValueError("anchored relevance recipe differs from admitted capacity")
    old = admission.verify_prior(HERE / "results" / prerequisite["exposed_receipt_file"])
    if (old.digest != prerequisite["exposed_receipt_digest"]
            or body["reference_receipt_digest"] != old.digest or old.body["protocol"] != base):
        raise ValueError("anchored capacity prerequisite: exposed reference differs")
    exposed = [run for run in old.body["runs"] if run["arm"] == "exposed"]
    if ([run["seed"] for run in exposed] != protocol["development_seeds"]
            or any(run["training"]["rounds"] != protocol["task"]["rounds"]
                   or not admission.acquired(run, protocol["gates"]["exposed_acquisition_floor"])
                   for run in exposed)):
        raise ValueError("anchored capacity prerequisite: exposed acquisition failed")
    old_files = {entry["path"]: entry["sha256"] for entry in old.source["files"]}
    for entry in receipt.source["files"]:
        name = entry["path"]
        if ((name.startswith("src/cadence/") or name == "benchmarks/attention/dual_binding.py")
                and old_files.get(name) != entry["sha256"]):
            raise ValueError("anchored capacity prerequisite: exposed model or task differs")
    return receipt


def capacity_check(body: dict[str, Any]) -> str | None:
    """Independently recompute the prerequisite, including every declared founder."""
    try:
        protocol = json.loads(admission.PROTOCOL.read_text())
        problem = admission.check(body)
        if problem:
            return "capacity verifier: " + problem
        if body["protocol"] != protocol:
            return "capacity protocol differs"
        runs = body["runs"]
        if [(r["arm"], r["seed"]) for r in runs] != [
            (arm, seed) for arm in ("exposed", "supplied_reads")
            for seed in protocol["development_seeds"]
        ]:
            return "capacity founders or arms differ"
        for run in runs:
            schedule = task.draw(run["seed"], protocol["task"]["rounds"], protocol["task"]["batch"])
            if run["schedule_sha256"] != canonical_sha256(schedule):
                return "capacity schedule differs"
            for name in ("train", "held_out", "all"):
                measured = run["evaluation"][name]
                if measured["scores"] != task.scores(measured["answers"], task.panel(name)):
                    return "capacity scores differ from answers"
            train = run["training"]
            if (not train["completed"] or train["rounds"] != protocol["task"]["rounds"]
                    or not run["parameters_preserved"]):
                return "capacity training or preservation failed"
            score = task.scores(run["evaluation"]["train"]["answers"], task.panel("train"))
            floor = protocol["gates"]["exposed_acquisition_floor" if run["arm"] == "exposed"
                                       else "content_use_floor"]
            if (score["unanswered_slots"] or min(score["exact_pair_accuracy"],
                    score["unequal_pair_accuracy"], *score["per_slot_accuracy"],
                    *score["per_query_accuracy"]) < floor or not run["acquired"]):
                return "capacity acquisition failed"
        elapsed = body["elapsed_host_seconds"]
        previous = body.get("previous_host_seconds", 0.0)
        if (not np.isfinite([elapsed, previous]).all() or min(elapsed, previous) < 0
                or not body["capacity_admitted"]):
            return "capacity admission or work differs"
        if body["candidate_rounds_used"] != 0 or body["confirmation_seeds_used"]:
            return "capacity already opened selection or confirmation"
    except (KeyError, TypeError, ValueError):
        return "malformed capacity readings"
    return None


def verify_capacity(path: Path) -> Receipt:
    receipt = Receipt.read(path)
    if receipt.kind == EMPTY_KIND:
        receipt = anchored_capacity(path)
        admission.PROTOCOL = admission.protocol_path(ANCHORED_PROTOCOL)
        return receipt
    admission.PROTOCOL = admission.protocol_path(
        receipt.body.get("protocol_file", "heads-protocol.json"))
    expected = admission.sources()
    if admission.PROTOCOL.name in admission.FOUNDER_PROTOCOLS:
        valid, message = Receipt.verify(path, sources=expected, check=capacity_check)
        if not valid or receipt.kind != "cadence-attention-heads-admission/1":
            raise ValueError("founder capacity prerequisite: " + message)
        return receipt
    if [entry["path"] for entry in receipt.source["files"]] != [name for name, _ in expected]:
        raise ValueError("capacity prerequisite: source manifest paths differ")
    bound = []
    for entry, (name, current) in zip(receipt.source["files"], expected, strict=True):
        digest = entry["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("capacity prerequisite: invalid source digest")
        if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
            snapshot = HERE / "results" / "sources" / (digest + ".txt")
            if (name != "benchmarks/attention/heads_admission.py" or not snapshot.exists()
                    or ast.dump(ast.parse(snapshot.read_text()))
                    != ast.dump(ast.parse(current.read_text()))):
                raise ValueError("capacity prerequisite: source manifest differs")
            current = snapshot
        bound.append((name, current))
    valid, message = Receipt.verify(path, sources=bound, check=capacity_check)
    if not valid:
        raise ValueError("capacity prerequisite: " + message)
    if receipt.kind != "cadence-attention-heads-admission/1":
        raise ValueError("capacity prerequisite: wrong receipt kind")
    return receipt


def bank_change(memory, operation: str, brain=None, rows=None):
    """Change witnessed data jointly across heads, never the host query or drive."""
    members = memory.memories
    first = members[0]
    keys, values, held, ages = (getattr(first, name).copy()
                                for name in ("keys", "values", "held", "ages"))
    if operation.startswith("erase_"):
        name = int(operation[-1])
        selected = (values[:, :, 0] > 0) & (values[:, :, 5 + name] > 0)
        held &= ~selected
    elif operation == "neutral_sham":
        neutral = values[:, :, 1] > 0
        keys[neutral] *= 0.5
        values[neutral] *= 0.5
    elif operation == "reorder":
        order = np.arange(keys.shape[1])[::-1]
        keys, values, held, ages = (keys[:, order], values[:, order], held[:, order], ages[order])
    elif operation == "donor_swap":
        if brain is None or rows is None:
            raise ValueError("donor transplant requires actual donor fact events")
        donor = task.swapped(rows)
        prefix, _ = task.events(donor)
        copies = brain.port_copies
        for name in range(2):
            event = np.where(donor[:, 2, None] == name, prefix[0], prefix[1])
            embedded = np.tile(event, (1, copies)) / np.sqrt(copies)
            selected = (values[:, :, 0] > 0) & (values[:, :, 5 + name] > 0)
            values = np.where(selected[:, :, None], embedded[:, None, :], values)
            keys = np.where(selected[:, :, None],
                            brain.recall.heard * embedded[:, None, :], keys)
    else:
        raise ValueError("unknown witnessed-bank intervention")
    return replace(memory, memories=tuple(replace(
        member, keys=keys, values=values, held=held, ages=ages,
    ) for member in members))


def interface_change(brain, condition: str, initial: np.ndarray) -> None:
    """Acquired-copy seam diagnostics, with reciprocal transport kept explicit."""
    graph = brain.brain
    pre, post = graph.connectome.pre, graph.connectome.post
    association = np.asarray(brain.association_index)
    ports = np.concatenate((*brain.head_queries, *brain.head_reads))
    seam = ((np.isin(pre, ports) & np.isin(post, association))
            | (np.isin(post, ports) & np.isin(pre, association)))
    weights, bias = graph.efficacy.copy(), graph.bias.copy()
    if condition == "frozen_interface":
        weights[seam] = initial[seam]
        bias[ports] = 0.0  # The raw-token anatomy's actual newborn port biases.
    else:
        mapping = np.arange(graph.connectome.n)
        if condition == "shuffled_interface":
            mapping[association] = association[::-1]
        elif condition == "cloned_heads":
            if brain.heads < 2:
                return
            for query, read in zip(brain.head_queries[1:], brain.head_reads[1:], strict=True):
                mapping[query] = brain.head_queries[0]
                mapping[read] = brain.head_reads[0]
            bias[ports] = graph.bias[mapping[ports]]
        else:
            raise ValueError("unknown selection-interface intervention")
        lookup = {(int(a), int(b)): edge for edge, (a, b) in enumerate(zip(pre, post, strict=True))}
        for edge in np.flatnonzero(seam):
            weights[edge] = graph.efficacy[lookup[(int(mapping[pre[edge]]),
                                                 int(mapping[post[edge]]))]]
    brain.learner.brain = graph.with_parameters(efficacy=weights, bias=bias)


def reading(answers, rows, work=None, *, status="completed", association=None, attention=None):
    return {"rows": np.asarray(rows).tolist(), "answers": np.asarray(answers).tolist(),
            "scores": task.scores(answers, rows), "status": status,
            "work": asdict(sr.Work() if work is None else work),
            "association": association, "attention": attention}


def unrun(rows, reason="wall_budget"):
    return reading(np.full((len(rows), 2), -1, dtype=np.int64), rows, status=reason)


def answered(measured):
    return bool(measured["status"] == "completed" and not measured["work"]["action_refusals"]
                and not task.scores(measured["answers"], measured["rows"])["unanswered_slots"])


def evaluate_conditions(brain, arm, rows, initial, deadline, *, interventions=True):
    """One witnessed host; every condition forks its exact complete pre-query state."""
    host = admission.binding.fork(brain)
    host.reset()
    prefix_work = sr.Work()
    prefix, _ = task.events(rows)
    conditions = ["intact"]
    if interventions:
        conditions += ["erase_0", "erase_1", "neutral_sham", "reorder", "donor_swap",
                       "frozen_interface", "shuffled_interface", "head_order"]
        conditions += [f"head_{head}" for head in range(brain.heads)]
        if brain.heads > 1:
            conditions.append("cloned_heads")
    results = {}
    for observation in prefix:
        if time.monotonic() >= deadline or prefix_work.act(host, observation) is None:
            return {"prefix_work": asdict(prefix_work), "conditions": {
                condition: unrun(task.swapped(rows) if condition == "donor_swap" else rows,
                                 "prefix_unavailable") for condition in conditions}}
    query, drive, memory = admission.prepare(host, rows, arm)
    for condition in conditions:
        expected_rows = task.swapped(rows) if condition == "donor_swap" else rows
        if time.monotonic() >= deadline:
            results[condition] = unrun(expected_rows)
            continue
        twin = admission.binding.fork(host)
        if condition in ("erase_0", "erase_1", "neutral_sham", "reorder", "donor_swap"):
            twin.brain.memory = bank_change(memory, condition, twin, rows)
        elif condition.startswith("head_") and condition != "head_order":
            removed = int(condition.split("_")[1])
            twin.brain.memory = replace(memory, memories=tuple(
                replace(member, amplitude=0.0) if head == removed else member
                for head, member in enumerate(memory.memories)))
        elif condition == "head_order":
            # Descriptor ordering only: physical ports and downstream synapses stay fixed.
            twin.brain.memory = replace(memory, memories=tuple(reversed(memory.memories)))
        elif condition in ("frozen_interface", "shuffled_interface", "cloned_heads"):
            interface_change(twin, condition, initial)
        work = sr.Work()
        answer = admission.act_drive(twin, query, drive.copy(), work)
        activity = attention = None
        if answer is None:
            answer = np.full((len(rows), 2), -1, dtype=np.int64)
        else:
            state = np.atleast_2d(twin._activity().activation)
            activity = state[:, twin.association_index].tolist()
            if twin.brain.memory.mode != "linear":
                attention = twin.brain.memory.weights(state).tolist()
        results[condition] = reading(answer, expected_rows, work,
                                     status="completed" if not work.action_refusals else "refused",
                                     association=activity, attention=attention)
    return {"prefix_work": asdict(prefix_work), "conditions": results}


def metrics(result, protocol):
    """Derive causal effects from full answer arrays, including refused/unrun cases."""
    gates = protocol["gates"]
    train, held = result["evaluation"]["train"], result["evaluation"]["held_out"]
    train_score = task.scores(train["answers"], train["rows"])
    held_score = task.scores(held["answers"], held["rows"])
    conditions = result["causal"]["conditions"]
    baseline = conditions["intact"]
    rows, answers = np.asarray(baseline["rows"]), np.asarray(baseline["answers"])
    expected, index = task.targets(rows), np.arange(len(rows))
    unequal = rows[:, 0] != rows[:, 1]
    losses = []
    other_losses = []
    for name in range(2):
        slot = rows[:, 3] ^ name
        erased = np.asarray(conditions[f"erase_{name}"]["answers"])
        losses.append(float(np.mean(answers[index, slot] == expected[index, slot])
                            - np.mean(erased[index, slot] == expected[index, slot])))
        other = 1 - slot
        other_losses.append(float(np.mean(answers[index, other] == expected[index, other])
                                  - np.mean(erased[index, other] == expected[index, other])))
    changes = {name: float(np.mean(np.any(np.asarray(conditions[name]["answers"]) != answers,
                                          axis=1)))
               for name in ("neutral_sham", "reorder", "head_order")}
    donor = conditions["donor_swap"]
    donor_correct = (np.asarray(donor["answers"]) == task.targets(task.swapped(rows))).all(axis=1)
    donor_follow = float(donor_correct[unequal].mean())
    swapped_score = task.scores(donor["answers"], donor["rows"])
    donor_rows = np.asarray(donor["rows"])
    held_mask = np.array([tuple(row[:2]) in task.HELD_PAIRS for row in donor_rows])
    swapped_held = task.scores(np.asarray(donor["answers"])[held_mask], donor_rows[held_mask])
    required = ("intact", "erase_0", "erase_1", "neutral_sham", "reorder", "donor_swap",
                "head_order")
    qualified = all(answered(conditions[name]) for name in required)
    training = result["training"]
    acquired = bool(training["completed"] and result["parameters_preserved"]
                    and training["rounds"] == protocol["task"]["rounds"]
                    and len(training["wrong_counts"]) == training["rounds"]
                    and not training["work"]["action_refusals"]
                    and not training["work"]["lesson_refusals"]
                    and answered(train)
                    and min(train_score["exact_pair_accuracy"],
                            train_score["unequal_pair_accuracy"],
                            *train_score["per_slot_accuracy"], *train_score["per_query_accuracy"])
                    >= gates["content_use_floor"])
    causal = bool(qualified and min(losses) >= gates["relevant_record_erasure_slot_loss_minimum"]
                  and donor_follow >= gates["donor_pair_follow_minimum"]
                  and max(changes.values())
                  <= gates["sham_or_record_reordering_answer_change_maximum"])
    return {
        "acquired": acquired, "held_out_exact_pair_accuracy": held_score["exact_pair_accuracy"],
        "swapped_unequal_exact_pair_accuracy": swapped_score["unequal_pair_accuracy"],
        "swapped_held_out_exact_pair_accuracy": swapped_held["exact_pair_accuracy"],
        "swapped_held_out_scores": swapped_held,
        "record_erasure_slot_loss": losses, "record_erasure_other_slot_loss": other_losses,
        "donor_pair_follow": donor_follow, "donor_unequal_cases": int(unequal.sum()),
        "sham_or_reorder_pair_change": changes, "required_causal_answers_qualified": qualified,
        "causal_passed": causal,
        "primary_passed": bool(acquired and causal and answered(held)
                               and held_score["exact_pair_accuracy"]
                               >= gates["held_out_exact_pair_floor"]
                               and swapped_score["unequal_pair_accuracy"]
                               >= gates["swapped_unequal_exact_pair_floor"]
                               and swapped_held["exact_pair_accuracy"]
                               >= gates["swapped_unequal_exact_pair_floor"]),
    }


def retention_passes(retention, protocol):
    if retention["status"] not in ("completed", "capped"):
        return False
    training = retention["training"]
    evaluation = retention["evaluation"]
    return bool(
        training["rounds"] == protocol["gates"]["retention_rounds"]
        and admission.acquired(
            {"training": training, "evaluation": evaluation, "parameters_preserved": True},
            protocol["gates"]["content_use_floor"])
        and answered(evaluation["held_out"])
        and answered(evaluation["train"])
        and retention["loss"] <= protocol["gates"]["retention_loss_maximum"]
    )


def retention_schedule(seed, protocol):
    base = task.panel("train")
    base = base[base[:, 3] == 0]
    order = np.random.default_rng(np.random.SeedSequence([622024, seed])).permutation(len(base))
    base = base[order]
    batch = protocol["task"]["batch"]
    schedule = []
    for tick in range(protocol["gates"]["retention_rounds"]):
        rows = base[(np.arange(batch) + tick * batch) % len(base)].copy()
        rows[:, 3] = tick % 2
        schedule.append(rows)
    return np.stack(schedule)


def representation_inputs(result):
    """Use only already-recorded free association states, preserving row custody."""
    def subset(measured, mask):
        association = measured.get("association")
        return {
            "rows": np.asarray(measured.get("rows", task.CASES))[mask].tolist(),
            "answers": np.asarray(measured["answers"])[mask].tolist(),
            "association": (None if association is None
                            else np.asarray(association)[mask].tolist()),
            "work": measured["work"], "status": measured.get("status", "completed"),
        }

    conditions = result["causal"]["conditions"]
    return {
        "train_features": result["evaluation"]["train"],
        "held_features": result["evaluation"]["held_out"],
        "frozen_train": subset(result["initial"], ~task.HELD),
        "frozen_held": subset(result["initial"], task.HELD),
        "swapped_features": subset(conditions["donor_swap"], task.HELD),
        "erased_features": {name: subset(conditions[name], task.HELD)
                            for name in ("erase_0", "erase_1")},
    }


def representation_probe(result, deadline):
    if not result["passed"]:
        return {"status": "pending_behavioral_gates", "passed": False}
    if time.monotonic() >= deadline:
        return {"status": "pending_budget", "passed": False}
    began = time.monotonic()
    try:
        diagnostic = representation.analyze(**representation_inputs(result))
        probe = {"status": "completed", "result": diagnostic, "passed": diagnostic["passed"]}
    except (ValueError, np.linalg.LinAlgError) as error:
        probe = {"status": "refused_inputs", "error": str(error), "passed": False}
    probe["host_seconds"] = time.monotonic() - began
    return probe


def same_diagnostic(stored, recomputed):
    """Decisions/gates are exact; diagnostic floating fits allow rounding noise."""
    if isinstance(recomputed, dict):
        return (isinstance(stored, dict) and stored.keys() == recomputed.keys()
                and all(same_diagnostic(stored[key], value) for key, value in recomputed.items()))
    if isinstance(recomputed, list):
        return (isinstance(stored, list) and len(stored) == len(recomputed)
                and all(same_diagnostic(a, b) for a, b in zip(stored, recomputed, strict=True)))
    if isinstance(recomputed, float):
        return bool(isinstance(stored, (float, int))
                    and np.isclose(stored, recomputed, rtol=1e-10, atol=1e-12))
    return stored == recomputed


def parameter_snapshot(brain, directory, *, arm, seed, mode):
    """Local research arrays for later diagnostics, never a Brain checkpoint."""
    if directory is None:
        return None
    began = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{mode}-{arm}-{seed}-parameters.npz"
    arrays = {name: np.asarray(getattr(brain.brain, name)).copy()
              for name in ("efficacy", "bias", "log_gain")}
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        if hasattr(brain.learner, name):
            arrays["optimizer_" + name] = np.asarray(getattr(brain.learner, name)).copy()
    metadata = {
        "schema": "cadence-attention-research-parameters/1", "arm": arm, "seed": seed,
        "mode": mode, "source": source_manifest(sources()), "source_revision": sr.revision(),
        "parameter_sha256": sr.parameter_digest(brain),
        "learning": brain.learner.config.to_dict(),
        "updates": int(getattr(brain.learner, "updates", 0)),
        "contrast_updates": int(getattr(brain.learner, "contrast_updates", 0)),
        "scope": "Research parameter arrays only; not a supported continuing Brain checkpoint",
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    content = canonical_sha256({name: value.tolist() for name, value in arrays.items()})
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)
    return {"file": path.name, "content_sha256": content,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "host_seconds": time.monotonic() - began,
            "scope": "Research arrays; no checkpoint continuation claim"}


def trial(arm, seed, mode, protocol, deadline, parameters=None):
    began = time.monotonic()
    brain = admission.make_brain(arm, seed, protocol, mode)
    if protocol["brain"].get("empty_anchor", False) and not brain.empty_anchor:
        raise ValueError("anchored relevance requires the declared empty-read boundary")
    initial = brain.brain.efficacy.copy()
    initial_parameters = sr.parameter_digest(brain)
    cost = {key: 0 for key in ("cortical_synapse_transports", "memory_scan_macs",
                              "softmax_elements", "memory_drive_evaluations")}
    schedule = task.draw(seed, protocol["task"]["rounds"], protocol["task"]["batch"])
    with admission.metered(cost):
        newborn = admission.evaluate(brain, arm, task.panel("all"))
        training = admission.train(brain, arm, schedule, min(
            deadline, time.monotonic() + protocol["budget"]["training_seconds_per_arm"]))
        final = sr.parameter_digest(brain)
        evaluation = {name: evaluate_conditions(brain, arm, task.panel(name), initial, deadline,
                                              interventions=False)["conditions"]["intact"]
                      for name in ("train", "held_out", "all")}
        causal = evaluate_conditions(brain, arm, task.panel("all"), initial, deadline)
        result = {
            "arm": arm, "seed": seed, "mode": mode, "training": training,
            "empty_anchor": bool(getattr(brain, "empty_anchor", False)),
            "initial": newborn, "evaluation": evaluation, "causal": causal,
            "schedule_sha256": canonical_sha256(schedule),
            "initial_parameters": initial_parameters, "final_parameters": final,
            "initial_interface_sha256": canonical_sha256(initial),
            "parameters_preserved": sr.parameter_digest(brain) == final,
            "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
            "resolved": brain.describe(), "neuron_model": brain.brain.neuron_model.to_dict(),
        }
        if "numerical_control" in protocol:
            result.update(numerical_policy=admission.numerical_settings(brain),
                          learning_config=brain.learner.config.to_dict())
        result["metrics"] = metrics(result, protocol)
        retention = {"status": "not_run_primary_failed", "passed": False}
        if result["metrics"]["primary_passed"]:
            retention = {"status": "not_run_budget", "passed": False}
            if time.monotonic() < deadline:
                twin = admission.binding.fork(brain)
                continued = admission.train(twin, arm, retention_schedule(seed, protocol), min(
                    deadline, time.monotonic() + protocol["budget"]["training_seconds_per_arm"]))
                retained = {
                    name: evaluate_conditions(twin, arm, task.panel(name), initial, deadline,
                                              interventions=False)["conditions"]["intact"]
                    for name in ("train", "held_out")
                }
                loss = (result["metrics"]["held_out_exact_pair_accuracy"]
                        - retained["held_out"]["scores"]["exact_pair_accuracy"])
                retention = {"status": "completed" if continued["completed"] else "capped",
                             "training": continued, "evaluation": retained, "loss": loss}
                retention["passed"] = retention_passes(retention, protocol)
        result["retention"] = retention
    result["work"] = cost
    result["passed"] = result["metrics"]["primary_passed"] and retention["passed"]
    result["representation_probe"] = representation_probe(result, deadline)
    result["parameter_snapshot"] = parameter_snapshot(
        brain, parameters, arm=arm, seed=seed, mode=mode)
    result["elapsed_host_seconds"] = time.monotonic() - began
    return result


def run(capacity_path: Path, parameters: Path | None = None) -> Receipt:
    capacity = verify_capacity(capacity_path)
    protocol = json.loads(admission.PROTOCOL.read_text())
    anchored = getattr(capacity, "kind", None) == EMPTY_KIND
    founder = admission.PROTOCOL.name in admission.FOUNDER_PROTOCOLS
    if anchored or founder:
        stage = protocol["relevance_stage"] if founder else protocol["prerequisites"]
        retained = HERE / "results" / stage["capacity_receipt_file"]
        if Receipt.read(retained).digest != capacity.digest:
            raise ValueError("retain the verified capacity receipt at its declared filename")
    manifest = source_manifest(sources())
    began = time.monotonic()
    spent = (capacity.body["cumulative_host_seconds"] if anchored or founder else
             capacity.body["elapsed_host_seconds"]
             + capacity.body.get("previous_host_seconds", 0.0))
    allowance = (protocol["relevance_stage"]["development_total_host_seconds"] if founder else
                 protocol["budget"]["development_total_host_seconds"])
    deadline = began + (allowance if anchored or founder else max(0.0, allowance - spent))
    runs, selected = [], None
    for mode in protocol["candidate_order"]:
        for arm in ARMS:
            for seed in protocol["development_seeds"]:
                if time.monotonic() >= deadline:
                    result = {"mode": mode, "arm": arm, "seed": seed, "status": "unrun_budget",
                              "passed": False}
                else:
                    result = trial(arm, seed, mode, protocol, deadline, parameters=parameters)
                runs.append(result)
                print(json.dumps({key: result[key] for key in ("mode", "arm", "seed", "passed")}),
                      flush=True)
        for arm in ARMS:
            group = [r for r in runs if r["mode"] == mode and r["arm"] == arm]
            if all(r["passed"] for r in group):
                selected = {"mode": mode, "arm": arm}
                break
        if selected is not None:
            break
    if manifest != source_manifest(sources()):
        raise RuntimeError("source changed during relevance development")
    body = {"protocol": protocol, "protocol_file": admission.PROTOCOL.name,
            "runs": runs, "selected": selected,
            "passed": selected is not None, "capacity_reference": capacity.digest,
            "capacity_host_seconds": spent, "elapsed_host_seconds": time.monotonic() - began,
            "candidate_rounds_used": len({r["mode"] for r in runs}),
            "confirmation_seeds_used": [],
            "source_revision": sr.revision(),
            "head_advantage": "Conditional comparison only: inspect absolute scores and measured "
                              "transports, including retention; unmatched work proves no advantage",
            "representation_probe": "Per-run diagnostic after behavioral and retention gates; "
                                    "never installed in the brain; no learned-code claim"}
    if anchored:
        body.update(
            capacity_receipt_file=protocol["prerequisites"]["capacity_receipt_file"],
            exposed_capacity_reference=protocol["prerequisites"]["exposed_receipt_digest"],
            budget_scope="additional_relevance_stage",
            previous_capacity_host_seconds=capacity.body["previous_capacity_host_seconds"],
            previous_diagnostic_host_seconds=capacity.body["previous_diagnostic_host_seconds"],
            empty_boundary_host_seconds=capacity.body["elapsed_host_seconds"],
            cumulative_host_seconds=spent + body["elapsed_host_seconds"],
        )
    if founder:
        body.update(capacity_receipt_file=protocol["relevance_stage"]["capacity_receipt_file"],
                    budget_scope="additional_relevance_stage",
                    previous_host_seconds=capacity.body["previous_host_seconds"],
                    founder_capacity_host_seconds=capacity.body["elapsed_host_seconds"],
                    cumulative_host_seconds=spent + body["elapsed_host_seconds"])
    problem = check(body)
    if problem:
        raise AssertionError(problem)
    return Receipt.build("cadence-attention-heads-relevance/1", body, sources())


def check(body):
    try:
        protocol = json.loads(admission.PROTOCOL.read_text())
        if (body.get("protocol_file", "heads-protocol.json") != admission.PROTOCOL.name
                or body["protocol"] != protocol or body["confirmation_seeds_used"]):
            return "protocol or unopened confirmation differs"
        anchored = admission.PROTOCOL.name == ANCHORED_PROTOCOL
        founder = admission.PROTOCOL.name in admission.FOUNDER_PROTOCOLS
        if anchored:
            declared = protocol["prerequisites"]
            capacity = anchored_capacity(HERE / "results" / declared["capacity_receipt_file"])
            prior = capacity.body
            if (body["capacity_receipt_file"] != declared["capacity_receipt_file"]
                    or body["capacity_reference"] != capacity.digest
                    or body["exposed_capacity_reference"] != declared["exposed_receipt_digest"]
                    or body["budget_scope"] != "additional_relevance_stage"
                    or body["previous_capacity_host_seconds"]
                    != prior["previous_capacity_host_seconds"]
                    or body["previous_diagnostic_host_seconds"]
                    != prior["previous_diagnostic_host_seconds"]
                    or body["empty_boundary_host_seconds"] != prior["elapsed_host_seconds"]
                    or body["capacity_host_seconds"] != prior["cumulative_host_seconds"]
                    or not np.isfinite(body["elapsed_host_seconds"])
                    or body["elapsed_host_seconds"] < 0
                    or body["cumulative_host_seconds"]
                    != prior["cumulative_host_seconds"] + body["elapsed_host_seconds"]):
                return "anchored prerequisite or cumulative budget differs"
        if founder:
            declared = protocol["relevance_stage"]
            capacity = verify_capacity(HERE / "results" / declared["capacity_receipt_file"])
            prior = capacity.body
            elapsed = body["elapsed_host_seconds"]
            if (body["capacity_receipt_file"] != declared["capacity_receipt_file"]
                    or body["capacity_reference"] != capacity.digest
                    or body["budget_scope"] != "additional_relevance_stage"
                    or body["previous_host_seconds"] != prior["previous_host_seconds"]
                    or body["founder_capacity_host_seconds"] != prior["elapsed_host_seconds"]
                    or body["capacity_host_seconds"] != prior["cumulative_host_seconds"]
                    or not np.isfinite(elapsed) or elapsed < 0
                    or body["cumulative_host_seconds"]
                    != prior["cumulative_host_seconds"] + elapsed):
                return "founder prerequisite or cumulative budget differs"
        runs = body["runs"]
        selected = None
        used_modes = []
        for mode in protocol["candidate_order"]:
            used_modes.append(mode)
            for arm in ARMS:
                group = [r for r in runs if r["mode"] == mode and r["arm"] == arm]
                if [r["seed"] for r in group] != protocol["development_seeds"]:
                    return "relevance founders or arms differ"
                for result in group:
                    if result.get("status") == "unrun_budget":
                        if result["passed"]:
                            return "unrun relevance cell cannot pass"
                        continue
                    if anchored and (result["empty_anchor"] is not True or mode != "agreement"):
                        return "anchored relevance boundary or equation differs"
                    if founder:
                        problem = admission.numerical_check(result, protocol)
                        if problem or result["empty_anchor"] is not True:
                            return problem or "founder relevance empty boundary differs"
                    if set(result["evaluation"]) != {"train", "held_out", "all"}:
                        return "missing relevance panel"
                    conditions = {"intact", "erase_0", "erase_1", "neutral_sham", "reorder",
                                  "donor_swap", "frozen_interface", "shuffled_interface",
                                  "head_order", "head_0"}
                    if arm == "h2":
                        conditions |= {"head_1", "cloned_heads"}
                    if set(result["causal"]["conditions"]) != conditions:
                        return "missing causal condition"
                    schedule = task.draw(result["seed"], protocol["task"]["rounds"],
                                         protocol["task"]["batch"])
                    if result["schedule_sha256"] != canonical_sha256(schedule):
                        return "relevance schedule differs"
                    for name, measured in result["evaluation"].items():
                        if (measured["rows"] != task.panel(name).tolist()
                                or measured["scores"] != task.scores(
                                    measured["answers"], measured["rows"])):
                            return "relevance panel or scores differ"
                    for name, measured in result["causal"]["conditions"].items():
                        rows = task.panel("all")
                        if name == "donor_swap":
                            rows = task.swapped(rows)
                        if (measured["rows"] != rows.tolist()
                                or measured["scores"] != task.scores(measured["answers"], rows)):
                            return "causal panel or scores differ"
                    expected = metrics(result, protocol)
                    if expected != result["metrics"]:
                        return "recomputed relevance metrics differ"
                    retention = result["retention"]
                    retention_passed = False
                    if retention["status"] in ("completed", "capped"):
                        for name, measured in retention["evaluation"].items():
                            if measured["scores"] != task.scores(
                                measured["answers"], task.panel(name)
                            ):
                                return "retention scores differ"
                        held_accuracy = retention["evaluation"]["held_out"]["scores"][
                            "exact_pair_accuracy"]
                        loss = expected["held_out_exact_pair_accuracy"] - held_accuracy
                        retention_passed = retention_passes(retention, protocol)
                        if loss != retention["loss"]:
                            return "retention loss differs"
                    if retention["passed"] != retention_passed or result["passed"] != (
                        expected["primary_passed"] and retention_passed
                    ):
                        return "relevance or retention gate differs"
                    probe = result["representation_probe"]
                    if probe["status"] == "completed":
                        if not result["passed"]:
                            return "representation opened before behavioral gates"
                        diagnostic = representation.analyze(**representation_inputs(result))
                        if (not same_diagnostic(probe["result"], diagnostic)
                                or probe["passed"] != diagnostic["passed"]):
                            return "representation diagnostic differs"
                    elif (probe["status"] not in ("pending_behavioral_gates", "pending_budget",
                                                   "refused_inputs") or probe["passed"]):
                        return "invalid pending representation diagnostic"
                if selected is None and all(r["passed"] for r in group):
                    selected = {"mode": mode, "arm": arm}
            if selected is not None:
                break
        if [(r["mode"], r["arm"], r["seed"]) for r in runs] != [
            (mode, arm, seed) for mode in used_modes for arm in ARMS
            for seed in protocol["development_seeds"]
        ]:
            return "relevance run order differs"
        if (body["selected"] != selected or body["passed"] != (selected is not None)
                or body["candidate_rounds_used"] != len(used_modes)):
            return "relevance selection differs"
    except (KeyError, TypeError, ValueError, OSError):
        return "malformed relevance readings"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--parameters", type=Path,
                        help="Retain local research arrays; not supported Brain checkpoints")
    args = parser.parse_args()
    if args.verify:
        stored = Receipt.read(args.verify)
        admission.PROTOCOL = admission.protocol_path(
            stored.body.get("protocol_file", "heads-protocol.json"))
        valid, message = Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        raise SystemExit(0 if valid else 1)
    if args.capacity is None or args.out is None:
        parser.error("provide --capacity and --out")
    if args.out.exists():
        raise FileExistsError(args.out)
    run(args.capacity, args.parameters).write(args.out)
