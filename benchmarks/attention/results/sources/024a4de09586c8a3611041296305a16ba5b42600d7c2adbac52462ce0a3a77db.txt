"""A bounded eight-case prerequisite, with new matched capacity and typed results.

No training or model construction occurs on import. This does not reinterpret
the original task, establish transfer, or open resource comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "minimal-binding-protocol.json"
TASK = HERE / "minimal_binding_task.py"
STAGES = ("capacity", "h2")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
PARAMETERS = ("efficacy", "bias", "log_gain")
COST_KEYS = (
    "cortical_synapse_transports",
    "memory_scan_macs",
    "softmax_elements",
    "memory_drive_evaluations",
    "fixed_read_coordinate_updates",
)
CONDITIONS = (
    "intact",
    "flip_A",
    "flip_B",
    "erase_0",
    "erase_1",
    "neutral_sham",
    "reorder",
    "head_order",
    "head_0",
    "head_1",
    "cloned_heads",
    "frozen_interface",
    "shuffled_interface",
)
CONTENT_CONDITIONS = CONDITIONS[:8]
_CONTEXT = None
_CAPACITY = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(runtime, capacity=None):
    global _CONTEXT, _CAPACITY
    runtime = Path(runtime).resolve()
    campaign = load(
        "minimal_frozen_query_campaign", runtime / "benchmarks/attention/query_founder_campaign.py"
    )
    old = campaign.context()
    _CONTEXT = SimpleNamespace(
        runtime=runtime,
        campaign=campaign,
        admission=old.admission,
        relevance=old.relevance,
        initializer=old.initializer,
        sr=old.sr,
        task=load("minimal_binding_task", TASK),
        prior_cache=None,
    )
    _CAPACITY = None if capacity is None else Path(capacity).resolve()
    return _CONTEXT


def context():
    if _CONTEXT is None:
        raise ValueError("configure the exact runtime before verification or execution")
    return _CONTEXT


def declared():
    spec = json.loads(PROTOCOL.read_text())
    if (
        spec["schema"] != "cadence-minimal-two-binding-campaign/1"
        or spec["query_scale"] != 1
        or spec["development_seeds"] != [0, 1]
        or set(spec["stages"]) != set(STAGES)
    ):
        raise ValueError("undeclared minimal-binding point")
    return spec


def profile(stage):
    point = context().campaign.point_protocol("capacity")
    spec = declared()
    point["budget"]["training_seconds_per_arm"] = spec["stages"][stage]["training_seconds_per_call"]
    if (
        point["brain"] != spec["brain"]
        or point["dt"] != spec["dt"]
        or point["task"]["rounds"] != spec["task"]["rounds"]
        or point["task"]["batch"] != spec["task"]["batch"]
    ):
        raise ValueError("frozen numerical/learning point differs")
    return point


def sources(stage):
    ctx, spec = context(), declared()
    paths = [
        ("benchmarks/attention/minimal_binding_campaign.py", Path(__file__).resolve()),
        ("benchmarks/attention/minimal_binding_task.py", TASK),
        ("benchmarks/attention/minimal-binding-protocol.json", PROTOCOL),
        *ctx.campaign.sources("h2"),
        (
            "benchmarks/attention/results/" + spec["prior"]["file"],
            ctx.runtime / "benchmarks/attention/results" / spec["prior"]["file"],
        ),
    ]
    if stage == "h2":
        if _CAPACITY is None:
            raise ValueError("provide the new matched capacity receipt")
        paths.append(("benchmarks/attention/results/minimal-binding-capacity.json", _CAPACITY))
    unique = {}
    for name, path in paths:
        if name in unique and path.resolve() != unique[name].resolve():
            raise ValueError("conflicting source dependency")
        unique.setdefault(name, path)
    return list(unique.items())


def prior():
    ctx, spec = context(), declared()["prior"]
    path = ctx.runtime / "benchmarks/attention/results" / spec["file"]
    # Cache verification arithmetic only, after exact complete dependency hashes match.
    # No model, parameter, optimizer, memory or result-state cache is introduced.
    identity = (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(source_manifest(ctx.campaign.sources("h2"))),
        canonical_sha256(spec),
    )
    if ctx.prior_cache is not None and ctx.prior_cache[0] == identity:
        return Receipt.read(path)  # A caller cannot mutate the cached verified body.
    receipt = Receipt.read(path)
    valid, message = ctx.campaign.verify(path)
    if (
        not valid
        or receipt.kind != spec["kind"]
        or receipt.digest != spec["digest"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != spec["archive_sha256"]
        or receipt.body["h2_admitted"] is not False
        or receipt.body["cumulative_host_seconds"] != spec["cumulative_host_seconds"]
    ):
        raise ValueError("minimal-binding negative prior differs: " + message)
    if identity != (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(source_manifest(ctx.campaign.sources("h2"))),
        canonical_sha256(declared()["prior"]),
    ):
        raise ValueError("negative prior dependencies changed during verification")
    ctx.prior_cache = (identity, receipt)
    return receipt


def make_brain(arm, seed, stage):
    ctx = context()
    return ctx.initializer.make_brain(
        arm, seed, profile(stage), "agreement", query_scale=1, factory=ctx.admission.make_brain
    )


def state_digest(brain):
    return canonical_sha256(
        {
            "parameters": {k: getattr(brain.brain, k) for k in PARAMETERS},
            "optimizer": {k: getattr(brain.learner, k) for k in MOMENTS},
            "updates": brain.learner.updates,
            "contrast_updates": brain.learner.contrast_updates,
        }
    )


def snapshot(brain, directory, *, stage, arm, seed, phase="final"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}-{arm}-{seed}-{phase}.npz"
    arrays = {k: np.array(getattr(brain.brain, k), copy=True) for k in PARAMETERS}
    arrays.update(
        {"optimizer_" + k: np.array(getattr(brain.learner, k), copy=True) for k in MOMENTS}
    )
    metadata = {
        "schema": "cadence-attention-minimal-binding-parameters/1",
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "phase": phase,
        "mode": "agreement",
        "source": source_manifest(sources(stage)),
        "source_revision": context().sr.revision(),
        "query_founder": deepcopy(brain.query_initialization),
        "parameter_sha256": context().sr.parameter_digest(brain),
        "state_sha256": state_digest(brain),
        "learning": brain.learner.config.to_dict(),
        "updates": brain.learner.updates,
        "contrast_updates": brain.learner.contrast_updates,
        "scope": "Research arrays; no supported continuing checkpoint or admitted capability",
    }
    arrays["metadata"] = np.asarray(canonical_json(metadata))
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)
    return {
        "file": path.name,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": canonical_sha256({k: v.tolist() for k, v in arrays.items()}),
        "metadata": metadata,
    }


def reading(answers, rows, work, status, *, association=None, attention=None):
    return {
        "rows": rows.tolist(),
        "answers": answers.tolist(),
        "scores": context().task.scores(answers, rows),
        "work": asdict(work),
        "status": status,
        "association": association,
        "attention": attention,
    }


def unrun(rows, status="unrun_budget"):
    return reading(np.full((len(rows), 2), -1, dtype=np.int64), rows, context().sr.Work(), status)


def bank_change(memory, condition, brain, rows):
    ctx = context()
    if condition not in ("flip_A", "flip_B"):
        return ctx.relevance.bank_change(memory, condition, brain, rows)
    name = int(condition == "flip_B")
    donor = ctx.task.flipped(rows, name)
    prefix, _ = ctx.task.events(donor)
    event = np.where(donor[:, 2, None] == name, prefix[0], prefix[1])
    embedded = np.tile(event, (1, brain.port_copies)) / np.sqrt(brain.port_copies)
    first = memory.memories[0]
    mask = (first.values[:, :, 0] > 0) & (first.values[:, :, 5 + name] > 0) & first.held
    if not np.all(mask.sum(axis=1) == 1):
        raise ValueError("donor requires exactly one witnessed named fact per host")
    values = np.where(mask[:, :, None], embedded[:, None, :], first.values)
    keys = np.where(mask[:, :, None], brain.recall.heard * embedded[:, None, :], first.keys)
    return replace(
        memory, memories=tuple(replace(m, keys=keys, values=values) for m in memory.memories)
    )


def evaluate(brain, arm, initial, deadline, *, causal=False):
    """One actual prefix; every condition forks its identical pre-query boundary."""
    ctx = context()
    rows = ctx.task.panel("all")
    conditions = CONDITIONS if causal else ("intact",)
    prefix_work = ctx.sr.Work()
    host = ctx.admission.binding.fork(brain)
    host.reset()
    prefix, _ = ctx.task.events(rows)
    for observation in prefix:
        if time.monotonic() >= deadline:
            return {
                "prefix_work": asdict(prefix_work),
                "conditions": {
                    c: unrun(
                        ctx.task.flipped(rows, int(c == "flip_B"))
                        if c.startswith("flip_")
                        else rows
                    )
                    for c in conditions
                },
            }
        if prefix_work.act(host, observation) is None:
            return {
                "prefix_work": asdict(prefix_work),
                "conditions": {
                    c: unrun(
                        ctx.task.flipped(rows, int(c == "flip_B"))
                        if c.startswith("flip_")
                        else rows,
                        "prefix_refused",
                    )
                    for c in conditions
                },
            }
    query, drive, memory = ctx.admission.prepare(host, rows, arm)
    result = {}
    for condition in conditions:
        expected = (
            ctx.task.flipped(rows, int(condition == "flip_B"))
            if condition.startswith("flip_")
            else rows
        )
        if time.monotonic() >= deadline:
            result[condition] = unrun(expected)
            continue
        twin = ctx.admission.binding.fork(host)
        if condition in ("flip_A", "flip_B", "erase_0", "erase_1", "neutral_sham", "reorder"):
            twin.brain.memory = bank_change(memory, condition, twin, rows)
        elif condition in ("head_0", "head_1"):
            removed = int(condition[-1])
            twin.brain.memory = replace(
                memory,
                memories=tuple(
                    replace(m, amplitude=0.0) if h == removed else m
                    for h, m in enumerate(memory.memories)
                ),
            )
        elif condition == "head_order":
            twin.brain.memory = replace(memory, memories=tuple(reversed(memory.memories)))
        elif condition in ("cloned_heads", "frozen_interface", "shuffled_interface"):
            ctx.relevance.interface_change(twin, condition, initial)
        work = ctx.sr.Work()
        answer = ctx.admission.act_drive(twin, query, drive.copy(), work)
        activity = attention = None
        if answer is None:
            answer = np.full((len(rows), 2), -1, dtype=np.int64)
        else:
            state = np.atleast_2d(twin._activity().activation)
            activity = state[:, twin.association_index].tolist()
            if memory is not None:
                attention = twin.brain.memory.weights(state).tolist()
        result[condition] = reading(
            answer,
            expected,
            work,
            "completed" if not work.action_refusals else "refused",
            association=activity,
            attention=attention,
        )
    return {"prefix_work": asdict(prefix_work), "conditions": result}


def answered(panel):
    return bool(
        panel["status"] == "completed"
        and not panel["work"]["action_refusals"]
        and not panel["scores"]["unanswered_slots"]
    )


def qualified_panel(panel, conditions=None):
    conditions = panel["conditions"] if conditions is None else conditions
    return not panel["prefix_work"]["action_refusals"] and all(
        answered(panel["conditions"][name]) for name in conditions
    )


def floor(reading):
    score = reading["scores"]
    values = [
        score["exact_pair_accuracy"],
        score["unequal_pair_accuracy"],
        *score["per_slot_accuracy"],
        *score["per_prefix_accuracy"],
    ]
    return all(v is not None and v >= declared()["gates"]["acquisition_floor"] for v in values)


def trained(training, rounds):
    return (
        training["completed"]
        and training["rounds"] == rounds
        and not training["work"]["action_refusals"]
        and not training["work"]["lesson_refusals"]
    )


def metrics(result):
    ctx, spec = context(), declared()
    panel = result["evaluation"]
    intact = panel["conditions"]["intact"]
    acquisition = (
        trained(result["training"], spec["task"]["rounds"])
        and qualified_panel(result["initial"])
        and qualified_panel(panel)
        and floor(intact)
        and result["parameters_preserved"]
    )
    out = {
        "acquisition_passed": bool(acquisition),
        "content_use_passed": False,
        "primary_passed": bool(acquisition),
        "distinct_useful_heads": False,
    }
    if result["arm"] != "h2":
        return out
    readings = result["causal"]["conditions"]
    host = np.asarray(readings["intact"]["answers"])
    base = readings["intact"]["scores"]
    donor, erase, sham = {}, {}, {}
    for name, label in enumerate(("A", "B")):
        other = 1 - name
        d = readings["flip_" + label]
        donor[label] = {
            "pair_follow": d["scores"]["exact_pair_accuracy"],
            "other_slot_change": float(
                (np.asarray(d["answers"])[:, other] != host[:, other]).mean()
            ),
        }
        erase[label] = (
            base["per_slot_accuracy"][name]
            - readings[f"erase_{name}"]["scores"]["per_slot_accuracy"][name]
        )
    for name in ("neutral_sham", "reorder", "head_order"):
        sham[name] = float((np.asarray(readings[name]["answers"]) != host).any(axis=1).mean())
    gates = spec["gates"]
    content = (
        acquisition
        and qualified_panel(result["causal"], CONTENT_CONDITIONS)
        and floor(readings["intact"])
        and all(
            d["pair_follow"] >= gates["donor_pair_follow_minimum"]
            and d["other_slot_change"] <= gates["other_slot_change_maximum"]
            for d in donor.values()
        )
        and min(erase.values()) >= gates["record_erasure_slot_loss_minimum"]
        and max(sham.values()) <= gates["sham_answer_change_maximum"]
    )
    attention = readings["intact"]["attention"]
    specialization = {"passed": False, "unequal_rows": [], "predicted_slot_losses": []}
    if attention is not None:
        weights = np.asarray(attention, dtype=float)
        rows = ctx.task.panel("all")
        if (
            weights.shape != (2, 8, 3)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or (weights.sum(axis=2) > 1 + 1e-12).any()
        ):
            raise ValueError("invalid all-eight attention boundary")
        row_index = np.arange(8)
        mass = np.stack(
            (
                weights[:, row_index, rows[:, 2]],
                weights[:, row_index, 1 - rows[:, 2]],
                weights[:, :, 2],
                1 - weights.sum(axis=2),
            ),
            axis=2,
        )
        unequal = rows[:, 0] != rows[:, 1]
        preferred = mass[:, :, :2].argmax(axis=2)
        selective = np.ones((2, 8), dtype=bool)
        for h in range(2):
            for i, name in enumerate(preferred[h]):
                selective[h, i] = (
                    mass[h, i, name] >= gates["distinct_record_mass_minimum"]
                    and mass[h, i, name] - np.max(mass[h, i, np.arange(4) != name])
                    >= gates["distinct_record_margin_minimum"]
                )
        ok = selective[:, unequal].all() and np.all(preferred[0, unequal] != preferred[1, unequal])
        losses_by_head, unequal_losses = [], []
        indices = np.flatnonzero(unequal)
        target = ctx.task.targets(rows)
        for h in range(2):
            slots = preferred[h]  # NumPy argmax deterministically chooses A on a tie.
            lesion = np.asarray(readings[f"head_{h}"]["answers"])
            before = host[np.arange(8), slots] == target[np.arange(8), slots]
            after = lesion[np.arange(8), slots] == target[np.arange(8), slots]
            losses_by_head.append(float(before.mean() - after.mean()))
            unequal_losses.append(float(before[unequal].mean() - after[unequal].mean()))
        specialization = {
            "passed": bool(ok),
            "unequal_case_ids": indices.tolist(),
            "head_favored_names": preferred.tolist(),
            "selective": selective[:, unequal].tolist(),
            "predicted_slot_losses": losses_by_head,
            "slot_loss_denominator": 8,
            "unequal_predicted_slot_losses": unequal_losses,
            "unequal_slot_loss_denominator": len(indices),
        }
    losses = {
        name: base["exact_pair_accuracy"] - readings[name]["scores"]["exact_pair_accuracy"]
        for name in ("head_0", "head_1", "cloned_heads", "frozen_interface", "shuffled_interface")
    }
    useful = bool(
        content
        and qualified_panel(result["causal"])
        and specialization["passed"]
        and min(specialization["predicted_slot_losses"]) >= gates["head_slot_loss_minimum"]
        and min(losses[n] for n in ("head_0", "head_1", "cloned_heads"))
        >= gates["head_pair_loss_minimum"]
        and min(losses[n] for n in ("frozen_interface", "shuffled_interface"))
        >= gates["interface_pair_loss_minimum"]
    )
    out.update(
        content_use_passed=bool(content),
        primary_passed=bool(content),
        distinct_useful_heads=useful,
        donor_flips=donor,
        erasure_slot_losses=erase,
        sham_answer_changes=sham,
        head_and_interface_pair_losses=losses,
        specialization=specialization,
    )
    return out


def retention_passes(retention, before):
    if retention["status"] != "measured":
        return False
    intact = retention["evaluation"]["conditions"]["intact"]
    return bool(
        trained(retention["training"], declared()["retention"]["rounds"])
        and qualified_panel(retention["evaluation"])
        and floor(intact)
        and before - intact["scores"]["exact_pair_accuracy"]
        <= declared()["gates"]["retention_loss_maximum"]
        and retention["parameters_preserved"]
    )


def retention_schedule(seed):
    spec, task = declared(), context().task
    base = task.panel("all")
    order = np.random.default_rng(np.random.SeedSequence([622024, seed])).permutation(len(base))
    base = base[order]
    batch = spec["task"]["batch"]
    return np.stack(
        [
            base[(np.arange(batch) + tick * batch) % len(base)].copy()
            for tick in range(spec["retention"]["rounds"])
        ]
    )


def trial(arm, seed, stage, deadline, parameters, captures):
    ctx, spec = context(), declared()
    began = time.monotonic()
    brain = make_brain(arm, seed, stage)
    captures["brain"] = brain
    cost = dict.fromkeys(COST_KEYS, 0)
    captures["actual_transport_work"] = cost
    initial, initial_parameters = brain.brain.efficacy.copy(), ctx.sr.parameter_digest(brain)
    schedule = ctx.task.draw(seed, spec["task"]["rounds"], spec["task"]["batch"])
    with ctx.admission.metered(cost):
        newborn = evaluate(brain, arm, initial, deadline)
        training = ctx.admission.train(
            brain,
            arm,
            schedule,
            min(deadline, time.monotonic() + spec["stages"][stage]["training_seconds_per_call"]),
        )
        final, final_state = ctx.sr.parameter_digest(brain), state_digest(brain)
        evaluation = evaluate(brain, arm, initial, deadline)
        result = {
            "status": "measured",
            "arm": arm,
            "seed": seed,
            "mode": "agreement",
            "initial": newborn,
            "training": training,
            "evaluation": evaluation,
            "initial_parameters": initial_parameters,
            "initial_interface_sha256": canonical_sha256(initial),
            "final_parameters": final,
            "final_state": final_state,
            "evaluation_parameters": ctx.sr.parameter_digest(brain),
            "evaluation_state": state_digest(brain),
            "parameters_preserved": state_digest(brain) == final_state,
            "schedule_sha256": canonical_sha256(schedule),
            "query_initialization": deepcopy(brain.query_initialization),
            "resolved": brain.describe(),
            "neuron_model": brain.brain.neuron_model.to_dict(),
            "numerical_policy": ctx.admission.numerical_settings(brain),
            "learning_config": brain.learner.config.to_dict(),
            "neurons": brain.connectome.n,
            "synapses": brain.connectome.synapses,
            "random_references": ctx.task.random_references(seed),
        }
        if stage == "h2":
            result["causal"] = evaluate(brain, arm, initial, deadline, causal=True)
            result["evaluation_parameters"] = ctx.sr.parameter_digest(brain)
            result["evaluation_state"] = state_digest(brain)
            result["parameters_preserved"] = state_digest(brain) == final_state
        result["metrics"] = metrics(result)
        retention = {
            "status": "not_applicable_capacity"
            if stage == "capacity"
            else "not_run_primary_failed",
            "passed": False,
        }
        if stage == "h2" and result["metrics"]["primary_passed"]:
            retention = {"status": "unrun_budget", "passed": False}
            if time.monotonic() < deadline:
                twin = ctx.admission.binding.fork(brain)
                captures["retention_brain"] = twin
                schedule = retention_schedule(seed)
                continued = ctx.admission.train(
                    twin,
                    arm,
                    schedule,
                    min(
                        deadline,
                        time.monotonic() + spec["stages"][stage]["training_seconds_per_call"],
                    ),
                )
                retained_state = state_digest(twin)
                retained_eval = evaluate(twin, arm, initial, deadline)
                retention = {
                    "status": "measured",
                    "training": continued,
                    "evaluation": retained_eval,
                    "final_parameters": ctx.sr.parameter_digest(twin),
                    "final_state": retained_state,
                    "evaluation_state": state_digest(twin),
                    "parameters_preserved": retained_state == state_digest(twin),
                    "schedule_sha256": canonical_sha256(schedule),
                    "parameter_snapshot": snapshot(
                        twin, parameters, stage=stage, arm=arm, seed=seed, phase="retention"
                    ),
                }
                retention["passed"] = retention_passes(
                    retention,
                    result["evaluation"]["conditions"]["intact"]["scores"]["exact_pair_accuracy"],
                )
        result["retention"] = retention
    result["parameter_snapshot"] = snapshot(brain, parameters, stage=stage, arm=arm, seed=seed)
    result["work"] = cost
    result["passed"] = result["metrics"]["primary_passed"] and (
        stage == "capacity" or retention["passed"]
    )
    result["elapsed_host_seconds"] = time.monotonic() - began
    return result


def work_check(work):
    template = asdict(context().sr.Work())
    if work.keys() != template.keys() or any(
        not context().campaign.finite(v) or (k != "max_action_residual" and type(v) is not int)
        for k, v in work.items()
    ):
        raise ValueError("invalid actual phase work")


def panel_check(panel, *, causal=False):
    ctx = context()
    work_check(panel["prefix_work"])
    if set(panel["conditions"]) != set(CONDITIONS if causal else ("intact",)):
        raise ValueError("missing or extra minimal-binding condition")
    rows = ctx.task.panel("all")
    for condition, r in panel["conditions"].items():
        expected = (
            ctx.task.flipped(rows, int(condition == "flip_B"))
            if condition.startswith("flip_")
            else rows
        )
        if r["rows"] != expected.tolist() or r["scores"] != ctx.task.scores(r["answers"], expected):
            raise ValueError("minimal-binding actual rows or scores differ")
        work_check(r["work"])
        if r["status"] not in ("completed", "refused", "prefix_refused", "unrun_budget"):
            raise ValueError("undeclared panel status")
        if r["status"] in ("completed", "refused"):
            if (
                panel["prefix_work"]["actions"] != 3
                or panel["prefix_work"]["action_rows"] != 24
                or panel["prefix_work"]["action_refusals"]
                or panel["prefix_work"]["max_action_residual"]
                > declared()["brain"]["learning_tolerance"]
                or r["work"]["actions"] != 1
                or r["work"]["action_rows"] != 8
                or r["work"]["action_refusals"] != int(r["status"] == "refused")
            ):
                raise ValueError("completed/refused query work differs")
        elif r["work"] != asdict(ctx.sr.Work()):
            raise ValueError("unrun query has neural work")
        if r["status"] == "prefix_refused" and (
            panel["prefix_work"]["action_refusals"] != 1
            or not 1 <= panel["prefix_work"]["actions"] <= 3
            or panel["prefix_work"]["action_rows"] != 8 * panel["prefix_work"]["actions"]
        ):
            raise ValueError("refused-prefix work differs")
        if r["status"] != "completed" and np.any(np.asarray(r["answers"]) != -1):
            raise ValueError("unavailable query contains answered slots")
        if any(
            v
            for k, v in r["work"].items()
            if k.startswith("teaching_")
            or k in ("lessons", "lesson_refusals", "presentations", "accepted_presentations")
        ):
            raise ValueError("evaluation contains teaching work")
        if (
            answered(r)
            and r["work"]["max_action_residual"] > declared()["brain"]["learning_tolerance"]
        ):
            raise ValueError("answered panel did not qualify")


def snapshot_check(descriptor, result, stage, revision, *, phase="final"):
    m = descriptor["metadata"]
    if (
        m["schema"] != "cadence-attention-minimal-binding-parameters/1"
        or m["phase"] != phase
        or m["arm"] != result["arm"]
        or m["seed"] != result["seed"]
        or m["stage"] != stage
        or m["mode"] != "agreement"
        or m["source_revision"] != revision
        or m["source"] != source_manifest(sources(stage))
        or m["query_founder"] != result["query_initialization"]
        or m["learning"] != result["learning_config"]
    ):
        raise ValueError("minimal-binding parameter source or point differs")
    for key in ("archive_sha256", "content_sha256"):
        if len(descriptor[key]) != 64 or any(c not in "0123456789abcdef" for c in descriptor[key]):
            raise ValueError("invalid parameter archive digest")
    if (
        type(m["updates"]) is not int
        or type(m["contrast_updates"]) is not int
        or not 0 <= m["contrast_updates"] <= m["updates"]
    ):
        raise ValueError("invalid parameter update counters")
    if phase == "final" and (
        m["parameter_sha256"] != result["final_parameters"]
        or m["state_sha256"] != result["final_state"]
    ):
        raise ValueError("parameter archive does not describe final state")


def check(body):
    try:
        ctx, spec = context(), declared()
        stage = body["stage"]
        if stage not in STAGES or body["protocol"] != spec:
            return "minimal-binding protocol differs"
        old = prior()
        previous = old.body["cumulative_host_seconds"]
        expected_prerequisites = {"negative_q1": old.digest}
        if stage == "h2":
            valid, message = verify(_CAPACITY)
            capacity = Receipt.read(_CAPACITY)
            if (
                not valid
                or capacity.kind != spec["stages"]["capacity"]["receipt_kind"]
                or capacity.body["capacity_admitted"] is not True
            ):
                return "new matched minimal capacity required: " + message
            previous = capacity.body["cumulative_host_seconds"]
            expected_prerequisites["capacity"] = capacity.digest
        revision = body["source_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or body["prerequisites"] != expected_prerequisites
            or body["previous_host_seconds"] != previous
            or not ctx.campaign.finite(body["elapsed_host_seconds"])
            or body["cumulative_host_seconds"] != previous + body["elapsed_host_seconds"]
            or body["confirmation_seeds_used"]
            or body["candidate_points_used"] != 1
            or body["head_advantage_claimed"] is not False
            or body["comparators_opened"] is not False
            or body["stone_age_opened"] is not False
        ):
            return "minimal-binding custody, charge or claim differs"
        runs = body["runs"]
        if [[r["arm"], r["seed"]] for r in runs] != spec["stages"][stage]["order"]:
            return "missing, duplicate or extra minimal-binding cells"
        if (
            any(r["status"] == "unrun_budget" for r in runs)
            and body["elapsed_host_seconds"] < spec["stages"][stage]["total_host_seconds"]
        ):
            return "unrun cell before the declared stage deadline"
        if sum(r.get("elapsed_host_seconds", 0.0) for r in runs) > body["elapsed_host_seconds"]:
            return "cell host charges exceed actual stage charge"
        for r in runs:
            if r["status"] == "unrun_budget":
                if r["passed"] is not False:
                    return "unrun minimal-binding cell optimism"
                continue
            if r["status"] != "measured" or r["mode"] != "agreement":
                return "invalid measured minimal-binding cell"
            problem = ctx.campaign.training_check(r["training"], profile(stage))
            if problem:
                return problem
            problem = ctx.admission.numerical_check(r, profile(stage))
            if problem:
                return problem
            expected = make_brain(r["arm"], r["seed"], stage)
            if (
                r["query_initialization"] != expected.query_initialization
                or r["initial_parameters"] != ctx.sr.parameter_digest(expected)
                or r["initial_interface_sha256"] != canonical_sha256(expected.brain.efficacy)
                or r["random_references"] != ctx.task.random_references(r["seed"])
                or r["schedule_sha256"]
                != canonical_sha256(
                    ctx.task.draw(r["seed"], spec["task"]["rounds"], spec["task"]["batch"])
                )
            ):
                return "minimal-binding founder or schedule differs"
            panel_check(r["initial"])
            panel_check(r["evaluation"])
            if (
                r["parameters_preserved"] != (r["final_state"] == r["evaluation_state"])
                or r["final_parameters"] != r["evaluation_parameters"]
                or set(r["work"]) != set(COST_KEYS)
                or any(type(v) is not int or v < 0 for v in r["work"].values())
                or not ctx.campaign.finite(r["elapsed_host_seconds"])
            ):
                return "minimal-binding evaluation state or cost differs"
            if stage == "h2":
                panel_check(r["causal"], causal=True)
            if r["metrics"] != metrics(r):
                return "minimal-binding recomputed metrics differ"
            snapshot_check(r["parameter_snapshot"], r, stage, revision)
            retained = r["retention"]
            if retained["status"] == "measured":
                if stage != "h2" or not r["metrics"]["primary_passed"]:
                    return "unpermitted retention"
                problem = ctx.campaign.training_check(
                    retained["training"], profile(stage), rounds=spec["retention"]["rounds"]
                )
                if problem:
                    return problem
                panel_check(retained["evaluation"])
                if retained["parameters_preserved"] != (
                    retained["final_state"] == retained["evaluation_state"]
                ):
                    return "retention state preservation differs"
                suffix = retention_schedule(r["seed"])
                if retained["schedule_sha256"] != canonical_sha256(suffix):
                    return "retention suffix differs"
                snapshot_check(
                    retained["parameter_snapshot"], r, stage, revision, phase="retention"
                )
                if (
                    retained["parameter_snapshot"]["metadata"]["parameter_sha256"]
                    != retained["final_parameters"]
                    or retained["parameter_snapshot"]["metadata"]["state_sha256"]
                    != retained["final_state"]
                ):
                    return "retention snapshot final state differs"
            elif retained["status"] != (
                "not_applicable_capacity"
                if stage == "capacity"
                else "unrun_budget"
                if r["metrics"]["primary_passed"]
                else "not_run_primary_failed"
            ):
                return "invalid conditional retention status"
            retention_gate = retention_passes(
                retained, r["evaluation"]["conditions"]["intact"]["scores"]["exact_pair_accuracy"]
            )
            if retained["passed"] != retention_gate:
                return "retention optimism"
            if r["passed"] != (
                r["metrics"]["primary_passed"] and (stage == "capacity" or retention_gate)
            ):
                return "minimal-binding cell optimism"
        admitted = all(r["passed"] for r in runs)
        if body["capacity_admitted" if stage == "capacity" else "h2_admitted"] != admitted:
            return "minimal-binding aggregate optimism"
        if stage == "h2" and body["distinct_useful_heads"] != bool(
            admitted and all(r.get("metrics", {}).get("distinct_useful_heads", False) for r in runs)
        ):
            return "distinct-useful-head optimism"
    except (KeyError, TypeError, ValueError, OSError, AssertionError):
        return "incomplete or invalid minimal-binding receipt"
    return None


def verify(path):
    if path is None:
        return False, "missing new matched capacity receipt"
    receipt = Receipt.read(path)
    stage = receipt.body.get("stage")
    if stage not in STAGES or receipt.kind != declared()["stages"][stage]["receipt_kind"]:
        return False, "not a minimal-binding stage receipt"
    return Receipt.verify(path, sources=sources(stage), check=check)


def run(stage, output, parameters, commit):
    began = time.monotonic()
    ctx, spec = context(), declared()
    if ctx.sr.revision() != commit:
        raise ValueError("runtime execution commit differs")
    old = prior()  # Verify the old negative before any fresh brain construction.
    previous, dependencies = old.body["cumulative_host_seconds"], {"negative_q1": old.digest}
    if stage == "h2":
        valid, message = verify(_CAPACITY)
        capacity = Receipt.read(_CAPACITY) if _CAPACITY is not None else None
        if not valid or capacity.body["capacity_admitted"] is not True:
            raise ValueError("new matched minimal capacity required: " + message)
        previous, dependencies["capacity"] = (
            capacity.body["cumulative_host_seconds"],
            capacity.digest,
        )
    manifest = source_manifest(sources(stage))
    deadline = began + spec["stages"][stage]["total_host_seconds"]
    runs, captures = [], {}
    journal = output.with_suffix(".partial.json")

    def flush(status, **extra):
        output.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - began
        journal.write_text(
            canonical_json(
                {
                    "status": status,
                    "stage": stage,
                    "source": manifest,
                    "protocol": spec,
                    "runs": runs,
                    "source_revision": commit,
                    "prerequisites": dependencies,
                    "previous_host_seconds": previous,
                    "elapsed_host_seconds": elapsed,
                    "cumulative_host_seconds": previous + elapsed,
                    **extra,
                }
            )
            + "\n"
        )

    for arm, seed in spec["stages"][stage]["order"]:
        captures.clear()
        flush("cell_started_unverified", current={"arm": arm, "seed": seed})
        try:
            r = (
                trial(arm, seed, stage, deadline, parameters, captures)
                if time.monotonic() < deadline
                else {"arm": arm, "seed": seed, "status": "unrun_budget", "passed": False}
            )
        except Exception as error:
            detail = {
                "current": {"arm": arm, "seed": seed},
                "error": {"type": type(error).__name__, "message": str(error)},
                "actual_transport_work": captures.get("actual_transport_work"),
            }
            for label in ("brain", "retention_brain"):
                if label in captures:
                    try:
                        detail[label + "_snapshot"] = snapshot(
                            captures[label],
                            Path(parameters) / "interrupted",
                            stage=stage,
                            arm=arm,
                            seed=seed,
                            phase=label,
                        )
                    except Exception as capture_error:  # noqa: BLE001 -- retain original error too.
                        detail[label + "_snapshot_error"] = type(capture_error).__name__
            flush("cell_failed_unverified", **detail)
            raise
        runs.append(r)
        flush("running_unverified")
    elapsed = time.monotonic() - began
    body = {
        "stage": stage,
        "protocol": spec,
        "runs": runs,
        "prerequisites": dependencies,
        "source_revision": commit,
        "previous_host_seconds": previous,
        "elapsed_host_seconds": elapsed,
        "cumulative_host_seconds": previous + elapsed,
        "candidate_points_used": 1,
        "confirmation_seeds_used": [],
        "head_advantage_claimed": False,
        "comparators_opened": False,
        "stone_age_opened": False,
        "capacity_admitted" if stage == "capacity" else "h2_admitted": all(
            r["passed"] for r in runs
        ),
    }
    if stage == "h2":
        body["distinct_useful_heads"] = bool(
            body["h2_admitted"]
            and all(r.get("metrics", {}).get("distinct_useful_heads", False) for r in runs)
        )
    flush("complete_unverified", body=body)
    if ctx.sr.revision() != commit or source_manifest(sources(stage)) != manifest:
        raise ValueError("runtime revision or exact source changed during execution")
    problem = check(body)
    if problem:
        raise ValueError(problem)
    if ctx.sr.revision() != commit or source_manifest(sources(stage)) != manifest:
        raise ValueError("runtime revision or source changed during final verification")
    body["elapsed_host_seconds"] = time.monotonic() - began
    body["cumulative_host_seconds"] = previous + body["elapsed_host_seconds"]
    flush("verified_readings", body=body)
    return Receipt.build(spec["stages"][stage]["receipt_kind"], body, sources(stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--capacity", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    configure(args.runtime, args.capacity)
    if args.verify:
        valid, message = verify(args.verify)
        print(message)
        raise SystemExit(0 if valid else 1)
    if not args.stage or args.out is None or args.parameters is None or not args.commit:
        parser.error("provide --stage, --out, --parameters and --commit")
    if args.out.exists() or args.out.with_suffix(".partial.json").exists():
        raise FileExistsError(args.out)
    run(args.stage, args.out, args.parameters, args.commit).write(args.out)
