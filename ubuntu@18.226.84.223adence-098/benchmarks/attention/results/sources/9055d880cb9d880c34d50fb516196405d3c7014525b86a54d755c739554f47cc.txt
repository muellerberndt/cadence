"""One fixed single-answer relevance assay, with its own matched capacity.

Imports perform no model construction or neural work. All cases are trained;
this measures a raw witnessed-memory component, not transfer or a continuing life.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadence import Brain
from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "single-relevance-protocol.json"
STAGES = ("capacity", "learned_h1")
CONDITIONS = (
    "intact",
    "erase_A",
    "erase_B",
    "all_erased",
    "flip_relevant",
    "flip_irrelevant",
    "donor_both",
    "erase_neutral",
    "joint_K_V_held_age_reorder",
    "frozen_interface",
    "shuffled_interface",
)
REQUIRED = tuple(c for c in CONDITIONS if c != "all_erased")
PARAMETERS = ("efficacy", "bias", "log_gain")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
_CONTEXT = None
_CAPACITY = None


def configure(runtime, capacity=None):
    import importlib.util
    import sys

    global _CONTEXT, _CAPACITY
    runtime = Path(runtime).resolve()
    path = runtime / "benchmarks/attention/minimal_binding_campaign.py"
    spec = importlib.util.spec_from_file_location("single_frozen_minimal", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = module.configure(
        runtime, runtime / "benchmarks/attention/results/minimal-binding-capacity.json"
    )
    oracle = module.load("single_fixed_read", HERE / "single_fixed_read.py")

    class AskedBrain(old.admission.heads.MultiHeadBrain):
        def stimulus(self, observations, *, memory=True):
            drive = super().stimulus(observations, memory=memory)
            if self.brain.memory is not None:
                self.brain.memory = oracle.AskedRead(self.brain.memory, observations)
            return drive

    _CONTEXT = SimpleNamespace(
        runtime=runtime,
        base=module,
        admission=old.admission,
        relevance=old.relevance,
        initializer=old.initializer,
        sr=old.sr,
        oracle=oracle,
        oracle_brain=AskedBrain,
        prior_cache=None,
    )
    _CAPACITY = None if capacity is None else Path(capacity).resolve()
    return _CONTEXT


def context():
    if _CONTEXT is None:
        raise ValueError("configure the exact runtime first")
    return _CONTEXT


def declared():
    p = json.loads(PROTOCOL.read_text())
    if (
        p["schema"] != "cadence-single-relevance-campaign/1"
        or p["candidate_points"] != 1
        or p["development_seeds"] != [0, 1]
        or p["factory"]["inputs"] != 11
        or p["factory"]["actions"] != 2
        or p["factory"]["heads"] != 1
        or p["factory"]["query_scale"] != 1
        or p["causal"]["conditions"] != list(CONDITIONS)
        or set(p["stages"]) != set(STAGES)
    ):
        raise ValueError("undeclared single-relevance point")
    return p


def sources(stage):
    ctx, p = context(), declared()
    paths = [
        (f"benchmarks/attention/{name}", HERE / name)
        for name in ("single_relevance_campaign.py", "single_fixed_read.py", PROTOCOL.name)
    ]
    paths += ctx.base.sources("h2")
    paths.append(
        (
            "benchmarks/attention/results/" + p["prior"]["file"],
            ctx.runtime / "benchmarks/attention/results" / p["prior"]["file"],
        )
    )
    if stage == "learned_h1":
        if _CAPACITY is None:
            raise ValueError("supply the new matched single capacity")
        paths.append(
            ("benchmarks/attention/results/" + p["stages"][stage]["capacity_file"], _CAPACITY)
        )
    unique = {}
    for name, path in paths:
        if name in unique and unique[name].resolve() != path.resolve():
            raise ValueError("conflicting source dependency")
        unique.setdefault(name, path)
    return list(unique.items())


def prior():
    ctx, spec = context(), declared()["prior"]
    if any(
        spec.get(k) is None
        for k in ("digest", "archive_sha256", "cumulative_host_seconds", "h2_admitted")
    ):
        raise ValueError("finished verified multihead prior has not been declared")
    path = ctx.runtime / "benchmarks/attention/results" / spec["file"]
    identity = (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(source_manifest(ctx.base.sources("h2"))),
        canonical_sha256(spec),
    )
    if ctx.prior_cache == identity:
        return Receipt.read(path)
    receipt = Receipt.read(path)
    valid, message = ctx.base.verify(path)
    if (
        not valid
        or receipt.kind != spec["kind"]
        or receipt.digest != spec["digest"]
        or identity[0] != spec["archive_sha256"]
        or receipt.body["h2_admitted"] != spec["h2_admitted"]
        or receipt.body["cumulative_host_seconds"] != spec["cumulative_host_seconds"]
    ):
        raise ValueError("multihead prior differs: " + message)
    if identity != (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(source_manifest(ctx.base.sources("h2"))),
        canonical_sha256(declared()["prior"]),
    ):
        raise ValueError("prior dependencies changed during verification")
    ctx.prior_cache = identity
    return receipt


def make_brain(arm, seed):
    ctx, f = context(), declared()["factory"]
    options = deepcopy(f["brain"])
    options["modules"] = tuple(options["modules"])
    if arm == "exposed":
        options.pop("empty_anchor")
        options = {k: v for k, v in options.items() if not k.startswith("recall_")}
        brain = Brain.compose(f["inputs"], f["actions"], seed=seed, **options)
        g = brain.brain
        if g.neuron_model.dt != f["dt"]:
            raise ValueError("exposed composed dt differs")
        brain.learner.brain = ctx.admission.heads.Graph(
            g.connectome, g.neuron_model, efficacy=g.efficacy, bias=g.bias, log_gain=g.log_gain
        )
    elif arm in ("supplied_address", "h1"):
        cls = ctx.oracle_brain if arm == "supplied_address" else ctx.admission.heads.MultiHeadBrain
        brain = cls.compose(
            f["inputs"],
            f["actions"],
            heads=1,
            port_copies=1,
            mode="agreement",
            dt=f["dt"],
            seed=seed,
            **options,
        )
    else:
        raise ValueError("unknown single-relevance arm")
    ctx.initializer.initialize(brain, seed=seed, query_scale=1)
    config, model = brain.learner.config.to_dict(), brain.brain.neuron_model.to_dict()
    if any(config[k] != v for k, v in f["resolved_learning_guard"].items()) or any(
        model[k] != v for k, v in f["resolved_neuron_guard"].items()
    ):
        raise ValueError("resolved numerical/learning point differs")
    return brain


def schedule(seed, rounds, *, retention=False):
    rows = context().sr.CASES
    rng = np.random.default_rng(np.random.SeedSequence([622024 if retention else 31014, seed]))
    if retention:
        base = rows[rng.permutation(16)]
        return np.stack([base.copy() for _ in range(rounds)])
    return np.stack([rows[rng.permutation(16)] for _ in range(rounds)])


def prepare(brain, rows, arm):
    ctx = context()
    _, query, target = ctx.sr.inputs(rows[:, :3], rows[:, 3], 1, exposed=arm == "exposed")
    ctx.admission.boundary(brain, len(rows))
    drive = np.array(brain.stimulus(query), copy=True)
    return query, drive, brain.brain.memory, target


def train(brain, arm, rows_by_round, deadline):
    ctx, work, wrong, stop = context(), context().sr.Work(), [], None
    for rows in rows_by_round:
        if time.monotonic() >= deadline:
            stop = "wall_budget"
            break
        brain.reset()
        prefix, _, _ = ctx.sr.inputs(rows[:, :3], rows[:, 3], 1)
        for observation in prefix:
            if work.act(brain, observation) is None:
                stop = "refused_prefix"
                break
        if stop:
            break
        query, drive, memory, target = prepare(brain, rows, arm)
        answer = ctx.admission.act_drive(brain, query, drive, work)
        if answer is None:
            stop = "refused_query"
            break
        selected = np.asarray(answer).reshape(-1) != target
        wrong.append(int(selected.sum()))
        ctx.admission.teach(brain, drive, target, selected, memory, work)
    return {
        "completed": len(wrong) == len(rows_by_round),
        "rounds": len(wrong),
        "wrong_counts": wrong,
        "stop": stop,
        "work": asdict(work),
    }


def bank_data(memory):
    if memory is None:
        return None
    return {k: getattr(memory, k).tolist() for k in ("keys", "values", "held", "ages")}


def bank_change(memory, condition, query, brain):
    """Only actual witnessed K/V and custody masks change, on an immutable copy."""
    if condition in ("intact", "frozen_interface", "shuffled_interface"):
        return memory
    member = memory.memories[0]
    values, keys, held, ages = [
        np.array(getattr(member, k), copy=True) for k in ("values", "keys", "held", "ages")
    ]
    if condition == "joint_K_V_held_age_reorder":
        return replace(
            memory,
            memories=(
                replace(
                    member,
                    keys=keys[:, ::-1],
                    values=values[:, ::-1],
                    held=held[:, ::-1],
                    ages=ages[::-1],
                ),
            ),
        )
    if condition == "all_erased":
        held[:] = False
    else:
        names = (
            (0, 1)
            if condition == "donor_both"
            else (
                (int(condition[-1] == "B"),)
                if condition.startswith("erase_")
                else (query if condition == "flip_relevant" else 1 - query,)
            )
        )
        if condition == "erase_neutral":
            names = ()
            neutral = held & (values[:, :, 4] > 0)
            if not np.all(neutral.sum(axis=1) == 1):
                raise ValueError("neutral sham requires one actual neutral witness")
            # Existing neutral-record sham: remove only its held participation.
            held[neutral] = False
        for name in names:
            selected = held & (values[:, :, 2 * name : 2 * name + 2].sum(axis=2) > 0)
            if not np.all(selected.sum(axis=1) == 1):
                raise ValueError("requires exactly one actual named witness")
            if condition.startswith("erase_"):
                held[selected] = False
            else:
                donor = context().sr.FACTS.copy()
                donor[:, name] = 1 - donor[:, name]
                event = np.eye(11)[2 * name + donor[:, name]]
                values = np.where(selected[:, :, None], event[:, None, :], values)
                keys = np.where(selected[:, :, None], brain.recall.heard * event[:, None, :], keys)
    return replace(
        memory, memories=(replace(member, keys=keys, values=values, held=held, ages=ages),)
    )


def expected(condition):
    target = context().sr.FACTS[:, :2].copy()
    return 1 - target if condition in ("flip_relevant", "donor_both") else target


def scores(answers, condition="intact"):
    a = np.asarray(answers)
    native = context().sr.scores(a)
    right = a == expected(condition)
    unequal = context().sr.FACTS[:, 0] != context().sr.FACTS[:, 1]
    return {
        "accuracy": float(right.mean()),
        "unequal_fact_accuracy": float(right[unequal].mean()),
        "per_query_accuracy": right.mean(axis=0).tolist(),
        "unanswered": int((a < 0).sum()),
        "source_task_scores": native,
    }


def evaluate(brain, arm, initial, deadline, *, causal=False):
    ctx = context()
    conditions = CONDITIONS if causal else ("intact",)
    prefix_work = ctx.sr.Work()
    host = ctx.admission.binding.fork(brain)
    host.reset()
    prefix, _, _ = ctx.sr.inputs(ctx.sr.FACTS, np.zeros(8, dtype=np.int64), 1)
    stop = None
    for event in prefix:
        if time.monotonic() >= deadline:
            stop = "unrun_budget"
            break
        if prefix_work.act(host, event) is None:
            stop = "prefix_refused"
            break
    result = {}
    for condition in conditions:
        answer = np.full((8, 2), -1, dtype=np.int64)
        queries = []
        for q in range(2):
            work = ctx.sr.Work()
            record = {
                "query": q,
                "status": stop or "unrun_budget",
                "work": asdict(work),
                "bank": None,
                "attention": None,
                "activation": None,
                "drive_sha256": None,
            }
            if stop is None and time.monotonic() < deadline:
                twin = ctx.admission.binding.fork(host)
                rows = np.column_stack((ctx.sr.FACTS, np.full(8, q, dtype=np.int64)))
                query, drive, memory, _ = prepare(twin, rows, arm)
                if causal:
                    twin.brain.memory = bank_change(memory, condition, q, twin)
                    if condition in ("frozen_interface", "shuffled_interface"):
                        ctx.relevance.interface_change(twin, condition, initial)
                memory = twin.brain.memory
                record["bank"] = bank_data(memory)
                record["drive_sha256"] = canonical_sha256(drive)
                actual = ctx.admission.act_drive(twin, query, drive, work)
                record["status"] = "refused" if actual is None else "completed"
                if actual is not None:
                    answer[:, q] = np.asarray(actual).reshape(-1)
                    state = np.atleast_2d(twin._activity().activation)
                    record["activation"] = {
                        "association": state[:, twin.association_index].tolist()
                    }
                    if memory is not None:
                        record["attention"] = memory.weights(state)[0].tolist()
                        record["activation"].update(
                            query=state[:, twin.head_queries[0]].tolist(),
                            read=state[:, twin.head_reads[0]].tolist(),
                        )
                record["work"] = asdict(work)
            queries.append(record)
        result[condition] = {
            "answers": answer.tolist(),
            "scores": scores(answer, condition),
            "queries": queries,
        }
    return {"rows": ctx.sr.CASES.tolist(), "prefix_work": asdict(prefix_work), "conditions": result}


def complete(panel, conditions=("intact",)):
    return not panel["prefix_work"]["action_refusals"] and all(
        q["status"] == "completed"
        and not q["work"]["action_refusals"]
        and not panel["conditions"][name]["scores"]["unanswered"]
        for name in conditions
        for q in panel["conditions"][name]["queries"]
    )


def floor(reading):
    s = reading["scores"]
    return (
        min(s["accuracy"], s["unequal_fact_accuracy"], *s["per_query_accuracy"])
        >= declared()["gates"]["acquisition_floor"]
    )


def trained(t, rounds):
    return (
        t["completed"]
        and t["rounds"] == rounds
        and not t["work"]["action_refusals"]
        and not t["work"]["lesson_refusals"]
    )


def metrics(result):
    p, ctx = declared(), context()
    acquisition = bool(
        trained(result["training"], p["data"]["rounds"])
        and complete(result["initial"])
        and complete(result["evaluation"])
        and result["initial_parameters_preserved"]
        and floor(result["evaluation"]["conditions"]["intact"])
        and result["parameters_preserved"]
    )
    out = {
        "acquisition_passed": acquisition,
        "causal_passed": False,
        "selection_passed": False,
        "interface_passed": False,
        "primary_passed": acquisition,
    }
    if result["arm"] != "h1":
        return out
    readings = result["causal"]["conditions"]
    host = np.asarray(readings["intact"]["answers"])
    accurate = host == expected("intact")
    changes, erase, follows, interfaces = {}, {}, {}, {}
    for name, c in enumerate(("erase_A", "erase_B")):
        after = np.asarray(readings[c]["answers"]) == expected("intact")
        erase[c] = float(accurate[:, name].mean() - after[:, name].mean())
        changes[c + "_irrelevant"] = float(
            (np.asarray(readings[c]["answers"])[:, 1 - name] != host[:, 1 - name]).mean()
        )
    for c in ("flip_relevant", "donor_both"):
        follows[c] = readings[c]["scores"]["per_query_accuracy"]
    for c in ("flip_irrelevant", "erase_neutral", "joint_K_V_held_age_reorder"):
        changes[c] = (np.asarray(readings[c]["answers"]) != host).mean(axis=0).tolist()
    for c in ("frozen_interface", "shuffled_interface"):
        interfaces[c] = readings["intact"]["scores"]["accuracy"] - readings[c]["scores"]["accuracy"]
    selection = []
    for q in range(2):
        r = readings["intact"]["queries"][q]
        if r["attention"] is None or r["bank"] is None:
            selection.append({"passed": False, "mass": None})
            continue
        weights = np.asarray(r["attention"])
        values, held = np.asarray(r["bank"]["values"]), np.asarray(r["bank"]["held"])
        masses = np.stack(
            [
                np.sum(weights * held * (values[:, :, 2 * n : 2 * n + 2].sum(axis=2) > 0), axis=1)
                for n in range(2)
            ]
            + [np.sum(weights * held * (values[:, :, 4] > 0), axis=1), 1 - weights.sum(axis=1)],
            axis=1,
        )
        unequal = ctx.sr.FACTS[:, 0] != ctx.sr.FACTS[:, 1]
        g = p["gates"]
        selected = (masses[:, q] >= g["relevant_record_mass_minimum"]) & (
            masses[:, q] - np.max(masses[:, np.arange(4) != q], axis=1)
            >= g["relevant_record_margin_minimum"]
        )
        selection.append(
            {
                "passed": bool(
                    selected[unequal].mean() >= g["selective_unequal_share_each_query_minimum"]
                ),
                "mass": masses.tolist(),
                "selected": selected.tolist(),
            }
        )
    g = p["gates"]
    causal = (
        acquisition
        and complete(result["causal"], REQUIRED)
        and floor(readings["intact"])
        and min(erase.values()) >= g["relevant_erasure_loss_minimum"]
        and all(min(v) >= g["donor_follow_minimum"] for v in follows.values())
        and all(np.max(v) <= g["irrelevant_change_maximum"] for v in changes.values())
    )
    interface = causal and min(interfaces.values()) >= g["learned_interface_loss_minimum"]
    selected = causal and all(s["passed"] for s in selection)
    out.update(
        causal_passed=bool(causal),
        selection_passed=bool(selected),
        interface_passed=bool(interface),
        primary_passed=bool(causal and selected and interface),
        erasure_losses=erase,
        answer_changes=changes,
        donor_follow=follows,
        interface_losses=interfaces,
        selector=selection,
    )
    return out


def state_digest(brain):
    return canonical_sha256(
        {
            "parameters": {k: getattr(brain.brain, k) for k in PARAMETERS},
            "optimizer": {k: getattr(brain.learner, k) for k in MOMENTS},
            "updates": brain.learner.updates,
            "contrast_updates": brain.learner.contrast_updates,
        }
    )


def snapshot(brain, directory, stage, arm, seed, phase="final"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}-{arm}-{seed}-{phase}.npz"
    arrays = {k: np.array(getattr(brain.brain, k), copy=True) for k in PARAMETERS}
    arrays.update(
        {"optimizer_" + k: np.array(getattr(brain.learner, k), copy=True) for k in MOMENTS}
    )
    metadata = {
        "schema": "cadence-attention-single-relevance-parameters/1",
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
        "scope": "Research arrays; no supported continuing checkpoint",
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


def retained(retention, baseline):
    if retention["status"] != "measured":
        return False
    return bool(
        trained(retention["training"], declared()["retention"]["rounds"])
        and complete(retention["evaluation"])
        and floor(retention["evaluation"]["conditions"]["intact"])
        and baseline - retention["evaluation"]["conditions"]["intact"]["scores"]["accuracy"]
        <= declared()["gates"]["retention_loss_maximum"]
        and retention["parameters_preserved"]
    )


def trial(arm, seed, stage, deadline, parameters, captures):
    ctx, p, began = context(), declared(), time.monotonic()
    brain = make_brain(arm, seed)
    captures["brain"] = brain
    initial = brain.brain.efficacy.copy()
    initial_parameters = ctx.sr.parameter_digest(brain)
    initial_state = state_digest(brain)
    cost = dict.fromkeys(ctx.base.COST_KEYS, 0)
    captures["actual_transport_work"] = cost
    order = schedule(seed, p["data"]["rounds"])
    with ctx.admission.metered(cost):
        newborn = evaluate(brain, arm, initial, deadline)
        newborn_state = state_digest(brain)
        teaching = train(
            brain,
            arm,
            order,
            min(deadline, time.monotonic() + p["stages"][stage]["training_seconds_per_call"]),
        )
        final, final_state = ctx.sr.parameter_digest(brain), state_digest(brain)
        evaluation = evaluate(brain, arm, initial, deadline)
        causal = (
            evaluate(brain, arm, initial, deadline, causal=True) if stage == "learned_h1" else None
        )
        result = {
            "status": "measured",
            "arm": arm,
            "seed": seed,
            "mode": "agreement",
            "initial": newborn,
            "training": teaching,
            "evaluation": evaluation,
            "initial_parameters": initial_parameters,
            "initial_interface_sha256": canonical_sha256(initial),
            "initial_state": initial_state,
            "initial_evaluation_state": newborn_state,
            "initial_parameters_preserved": initial_state == newborn_state,
            "final_parameters": final,
            "final_state": final_state,
            "evaluation_parameters": ctx.sr.parameter_digest(brain),
            "evaluation_state": state_digest(brain),
            "parameters_preserved": final_state == state_digest(brain),
            "query_initialization": deepcopy(brain.query_initialization),
            "learning_config": brain.learner.config.to_dict(),
            "resolved": brain.describe(),
            "neuron_model": brain.brain.neuron_model.to_dict(),
            "numerical_policy": ctx.admission.numerical_settings(brain),
            "neurons": brain.connectome.n,
            "synapses": brain.connectome.synapses,
            "schedule_sha256": canonical_sha256(order),
            "random_reference": np.random.default_rng(np.random.SeedSequence([31015, seed]))
            .integers(2, size=(8, 2))
            .tolist(),
        }
        if causal is not None:
            result["causal"] = causal
        result["metrics"] = metrics(result)
        retention = {
            "status": "not_applicable_capacity"
            if stage == "capacity"
            else "not_run_primary_failed",
            "passed": False,
        }
        if stage == "learned_h1" and result["metrics"]["primary_passed"]:
            retention = {"status": "unrun_budget", "passed": False}
            if time.monotonic() < deadline:
                twin = ctx.admission.binding.fork(brain)
                captures["retention_brain"] = twin
                suffix = schedule(seed, p["retention"]["rounds"], retention=True)
                continued = train(
                    twin,
                    arm,
                    suffix,
                    min(
                        deadline, time.monotonic() + p["stages"][stage]["training_seconds_per_call"]
                    ),
                )
                after, after_state = ctx.sr.parameter_digest(twin), state_digest(twin)
                panel = evaluate(twin, arm, initial, deadline)
                retention = {
                    "status": "measured",
                    "training": continued,
                    "evaluation": panel,
                    "final_parameters": after,
                    "final_state": after_state,
                    "evaluation_parameters": ctx.sr.parameter_digest(twin),
                    "evaluation_state": state_digest(twin),
                    "parameters_preserved": after_state == state_digest(twin),
                    "schedule_sha256": canonical_sha256(suffix),
                    "parameter_snapshot": snapshot(twin, parameters, stage, arm, seed, "retention"),
                }
                retention["passed"] = retained(
                    retention, evaluation["conditions"]["intact"]["scores"]["accuracy"]
                )
        result["retention"] = retention
    result["parameter_snapshot"] = snapshot(brain, parameters, stage, arm, seed)
    result["work"] = cost
    result["passed"] = result["metrics"]["primary_passed"] and (
        stage == "capacity" or retention["passed"]
    )
    result["elapsed_host_seconds"] = time.monotonic() - began
    return result


def work_check(work):
    template = asdict(context().sr.Work())
    if (
        work.keys() != template.keys()
        or any(
            not isinstance(v, (int, float))
            or isinstance(v, bool)
            or not np.isfinite(v)
            or v < 0
            or (k != "max_action_residual" and type(v) is not int)
            for k, v in work.items()
        )
        or work["action_refusals"] > work["actions"]
        or work["lesson_refusals"] > work["lessons"]
        or work["accepted_presentations"] > work["presentations"]
    ):
        raise ValueError("invalid attempted phase work")


def training_check(training, rows_by_round):
    work_check(training["work"])
    counts, work = training["wrong_counts"], training["work"]
    rounds = training["rounds"]
    if (
        type(rounds) is not int
        or not 0 <= rounds <= len(rows_by_round)
        or rounds != len(counts)
        or any(type(n) is not int or not 0 <= n <= 16 for n in counts)
        or training["completed"] != (rounds == len(rows_by_round))
        or training["stop"] not in (None, "wall_budget", "refused_prefix", "refused_query")
        or (training["completed"] and training["stop"] is not None)
        or work["lessons"] != sum(n > 0 for n in counts)
        or work["presentations"] != sum(counts)
        or not 4 * rounds <= work["actions"] <= 4 * rounds + 4
        or work["action_rows"] != 16 * work["actions"]
    ):
        raise ValueError("training schedule, lesson or attempt count differs")
    if (
        not work["action_refusals"]
        and work["max_action_residual"] > declared()["factory"]["brain"]["learning_tolerance"]
    ):
        raise ValueError("answered training did not qualify")


def expected_boundary(brain, q, condition):
    ctx = context()
    prefix, observation, _ = ctx.sr.inputs(ctx.sr.FACTS, np.full(8, q, dtype=np.int64), 1)
    values = np.stack(prefix, axis=1)
    m = ctx.admission.heads.Memory(
        brain.head_queries[0],
        brain.head_reads[0],
        brain.recall.heard * values,
        values,
        np.ones((8, 3), dtype=bool),
        np.array([2.0, 1.0, 0.0]),
        amplitude=brain.recall.amplitude,
        temperature=brain.recall.temperature,
        recency=brain.recall.recency,
        null=brain.recall.null,
        mode="agreement",
        empty_anchor=True,
    )
    bank = ctx.admission.heads.MultiMemory((m,))
    if type(brain) is ctx.oracle_brain:
        return ctx.oracle.AskedRead(bank, observation)
    return bank_change(bank, condition, q, brain)


def panel_check(panel, causal=False, *, brain=None, arm=None):
    ctx, tolerance = context(), declared()["factory"]["brain"]["learning_tolerance"]
    work_check(panel["prefix_work"])
    if panel["rows"] != ctx.sr.CASES.tolist() or set(panel["conditions"]) != set(
        CONDITIONS if causal else ("intact",)
    ):
        raise ValueError("single-reader panel inventory differs")
    for name, r in panel["conditions"].items():
        answers = np.asarray(r["answers"])
        if (
            answers.shape != (8, 2)
            or r["scores"] != scores(answers, name)
            or len(r["queries"]) != 2
        ):
            raise ValueError("answer table or recomputed score differs")
        for q, record in enumerate(r["queries"]):
            work = record["work"]
            work_check(work)
            status = record["status"]
            if record["query"] != q or status not in (
                "completed",
                "refused",
                "prefix_refused",
                "unrun_budget",
            ):
                raise ValueError("invalid query identity/status")
            if status in ("completed", "refused"):
                if (
                    panel["prefix_work"]["actions"] != 3
                    or panel["prefix_work"]["action_rows"] != 24
                    or panel["prefix_work"]["action_refusals"]
                    or panel["prefix_work"]["max_action_residual"] > tolerance
                    or work["actions"] != 1
                    or work["action_rows"] != 8
                    or work["action_refusals"] != int(status == "refused")
                ):
                    raise ValueError("actual prefix/query work differs")
            elif work != asdict(ctx.sr.Work()):
                raise ValueError("unrun query contains attempted work")
            if status == "prefix_refused" and not panel["prefix_work"]["action_refusals"]:
                raise ValueError("prefix refusal not attempted")
            if status != "completed" and np.any(answers[:, q] != -1):
                raise ValueError("unavailable query has answers")
            if status == "completed" and (
                np.any(answers[:, q] < 0) or work["max_action_residual"] > tolerance
            ):
                raise ValueError("answered query did not qualify")
            if work["lessons"] or work["presentations"] or work["teaching_sweeps"]:
                raise ValueError("evaluation contains teaching")
            attention, bank = record["attention"], record["bank"]
            if brain is not None and status in ("completed", "refused"):
                rows = np.column_stack((ctx.sr.FACTS, np.full(8, q, dtype=np.int64)))
                _, drive, _, _ = prepare(brain, rows, arm)
                boundary = None if arm == "exposed" else expected_boundary(brain, q, name)
                if record["drive_sha256"] != canonical_sha256(drive) or bank != bank_data(boundary):
                    raise ValueError("captured actual drive or witnessed bank differs")
                if status == "completed":
                    activity = record["activation"]
                    expected_shapes = {"association": (8, 32)}
                    if boundary is not None:
                        expected_shapes.update(query=(8, 11), read=(8, 11))
                    if (
                        activity is None
                        or set(activity) != set(expected_shapes)
                        or any(
                            np.asarray(activity[k]).shape != shape
                            or not np.isfinite(activity[k]).all()
                            for k, shape in expected_shapes.items()
                        )
                    ):
                        raise ValueError("completed query is missing its actual activity")
                    if boundary is not None:
                        state = np.zeros((8, brain.connectome.n))
                        state[:, brain.head_queries[0]] = activity["query"]
                        state[:, brain.head_reads[0]] = activity["read"]
                        if attention is None or not np.allclose(
                            np.asarray(attention), boundary.weights(state)[0], rtol=0, atol=1e-12
                        ):
                            raise ValueError(
                                "actual selector disagrees with recorded query activity"
                            )
                    elif attention is not None:
                        raise ValueError("exposed control has an undeclared selector")
            if attention is not None:
                values, keys = np.asarray(bank["values"]), np.asarray(bank["keys"])
                held, ages, weights = (
                    np.asarray(bank["held"]),
                    np.asarray(bank["ages"]),
                    np.asarray(attention),
                )
                if (
                    values.ndim != 3
                    or values.shape[0] != 8
                    or values.shape[2] != 11
                    or keys.shape != values.shape
                    or held.shape != values.shape[:2]
                    or ages.shape != (values.shape[1],)
                    or weights.shape != held.shape
                    or held.dtype.kind != "b"
                    or not np.isfinite(values).all()
                    or not np.isfinite(keys).all()
                    or not np.isfinite(ages).all()
                    or (ages < 0).any()
                    or not np.isfinite(weights).all()
                    or (weights < 0).any()
                    or (weights.sum(axis=1) > 1 + 1e-12).any()
                    or np.any(weights[~held] != 0)
                ):
                    raise ValueError("invalid witnessed selector boundary")
    # Both questions start with the same actual prefix bank; interventions may then change it.
    if not causal:
        a, b = panel["conditions"]["intact"]["queries"]
        if a["bank"] is not None and b["bank"] is not None and a["bank"] != b["bank"]:
            raise ValueError("paired questions do not share their witnessed bank")


def snapshot_check(descriptor, result, stage, revision, phase="final"):
    m = descriptor["metadata"]
    if (
        m["schema"] != "cadence-attention-single-relevance-parameters/1"
        or m["stage"] != stage
        or m["arm"] != result["arm"]
        or m["seed"] != result["seed"]
        or m["phase"] != phase
        or m["mode"] != "agreement"
        or m["source_revision"] != revision
        or m["source"] != source_manifest(sources(stage))
        or m["query_founder"] != result["query_initialization"]
        or m["learning"] != result["learning_config"]
        or type(m["updates"]) is not int
        or type(m["contrast_updates"]) is not int
        or not 0 <= m["contrast_updates"] <= m["updates"]
    ):
        raise ValueError("single-reader snapshot source/config/counters differ")
    state = result if phase == "final" else result["retention"]
    if (
        m["parameter_sha256"] != state["final_parameters"]
        or m["state_sha256"] != state["final_state"]
        or any(
            not isinstance(descriptor[k], str)
            or len(descriptor[k]) != 64
            or any(c not in "0123456789abcdef" for c in descriptor[k])
            for k in ("archive_sha256", "content_sha256")
        )
    ):
        raise ValueError("snapshot does not describe retained final state")


def prerequisites(stage):
    old = prior()
    previous, dependencies = old.body["cumulative_host_seconds"], {"multihead": old.digest}
    if stage == "learned_h1":
        valid, message = verify(_CAPACITY)
        capacity = Receipt.read(_CAPACITY) if _CAPACITY is not None else None
        if not valid or capacity.body["capacity_admitted"] is not True:
            raise ValueError("new matched single capacity required: " + message)
        previous, dependencies["capacity"] = (
            capacity.body["cumulative_host_seconds"],
            capacity.digest,
        )
    return previous, dependencies


def check(body):
    try:
        ctx, p, stage = context(), declared(), body["stage"]
        if stage not in STAGES or body["protocol"] != p:
            return "single-relevance protocol differs"
        previous, dependencies = prerequisites(stage)
        revision = body["source_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or body["prerequisites"] != dependencies
            or body["previous_host_seconds"] != previous
            or not np.isfinite(body["elapsed_host_seconds"])
            or body["elapsed_host_seconds"] < 0
            or body["cumulative_host_seconds"] != previous + body["elapsed_host_seconds"]
            or body["candidate_points_used"] != 1
            or body["confirmation_seeds_used"]
            or body["aws_opened"] is not False
            or body["world_opened"] is not False
        ):
            return "single-relevance provenance, cost or scope differs"
        runs = body["runs"]
        if (
            [[r["arm"], r["seed"]] for r in runs] != p["stages"][stage]["order"]
            or sum(r.get("elapsed_host_seconds", 0) for r in runs) > body["elapsed_host_seconds"]
            or (
                any(r["status"] == "unrun_budget" for r in runs)
                and body["elapsed_host_seconds"] < p["stages"][stage]["total_host_seconds"]
            )
        ):
            return "missing/duplicate/unrun or host-charge cell differs"
        for r in runs:
            if r["status"] == "unrun_budget":
                if r["passed"] is not False:
                    return "unrun optimism"
                continue
            if r["status"] != "measured" or r["mode"] != "agreement":
                return "invalid measured single-reader cell"
            expected_brain = make_brain(r["arm"], r["seed"])
            for field, value in (
                ("query_initialization", expected_brain.query_initialization),
                ("initial_parameters", ctx.sr.parameter_digest(expected_brain)),
                ("initial_state", state_digest(expected_brain)),
                ("initial_interface_sha256", canonical_sha256(expected_brain.brain.efficacy)),
                ("learning_config", expected_brain.learner.config.to_dict()),
                ("neuron_model", expected_brain.brain.neuron_model.to_dict()),
                ("numerical_policy", ctx.admission.numerical_settings(expected_brain)),
                ("neurons", expected_brain.connectome.n),
                ("synapses", expected_brain.connectome.synapses),
            ):
                if r[field] != value:
                    return "single-reader founder/config differs: " + field
            if (
                r["resolved"]["genes"] != expected_brain.describe()["genes"]
                or r["resolved"]["layout"] != expected_brain.describe()["layout"]
            ):
                return "single-reader resolved anatomy/genes differ"
            order = schedule(r["seed"], p["data"]["rounds"])
            if (
                r["schedule_sha256"] != canonical_sha256(order)
                or r["random_reference"]
                != np.random.default_rng(np.random.SeedSequence([31015, r["seed"]]))
                .integers(2, size=(8, 2))
                .tolist()
            ):
                return "single-reader schedule differs"
            training_check(r["training"], order)
            panel_check(r["initial"], brain=expected_brain, arm=r["arm"])
            panel_check(r["evaluation"], brain=expected_brain, arm=r["arm"])
            if stage == "learned_h1":
                panel_check(r["causal"], True, brain=expected_brain, arm=r["arm"])
            if (
                r["initial_parameters_preserved"]
                != (r["initial_state"] == r["initial_evaluation_state"])
                or r["evaluation_parameters"] != r["final_parameters"]
                or r["parameters_preserved"] != (r["final_state"] == r["evaluation_state"])
                or set(r["work"]) != set(ctx.base.COST_KEYS)
                or any(type(v) is not int or v < 0 for v in r["work"].values())
                or not np.isfinite(r["elapsed_host_seconds"])
                or r["elapsed_host_seconds"] < 0
                or r["metrics"] != metrics(r)
            ):
                return "single-reader evaluation, recomputed gate or work differs"
            snapshot_check(r["parameter_snapshot"], r, stage, revision)
            t = r["retention"]
            if t["status"] == "measured":
                if stage != "learned_h1" or not r["metrics"]["primary_passed"]:
                    return "unpermitted retention"
                suffix = schedule(r["seed"], p["retention"]["rounds"], retention=True)
                training_check(t["training"], suffix)
                panel_check(t["evaluation"], brain=expected_brain, arm=r["arm"])
                if (
                    t["schedule_sha256"] != canonical_sha256(suffix)
                    or t["evaluation_parameters"] != t["final_parameters"]
                    or t["parameters_preserved"] != (t["final_state"] == t["evaluation_state"])
                ):
                    return "retained schedule/state differs"
                snapshot_check(t["parameter_snapshot"], r, stage, revision, "retention")
            elif t["status"] != (
                "not_applicable_capacity"
                if stage == "capacity"
                else "unrun_budget"
                if r["metrics"]["primary_passed"]
                else "not_run_primary_failed"
            ):
                return "invalid conditional retention status"
            passes = retained(t, r["evaluation"]["conditions"]["intact"]["scores"]["accuracy"])
            if t["passed"] != passes or r["passed"] != (
                r["metrics"]["primary_passed"] and (stage == "capacity" or passes)
            ):
                return "single-reader retention/cell optimism"
        if body["capacity_admitted" if stage == "capacity" else "h1_admitted"] != all(
            r["passed"] for r in runs
        ):
            return "single-reader aggregate optimism"
    except (KeyError, ValueError, TypeError, OSError, AssertionError):
        return "incomplete or invalid single-relevance receipt"
    return None


def verify(path):
    if path is None:
        return False, "missing matched single capacity receipt"
    r = Receipt.read(path)
    stage = r.body.get("stage")
    if stage not in STAGES or r.kind != declared()["stages"][stage]["receipt_kind"]:
        return False, "not a single-relevance receipt"
    return Receipt.verify(path, sources=sources(stage), check=check)


def run(stage, output, parameters, commit):
    ctx, p, began = context(), declared(), time.monotonic()
    if ctx.sr.revision() != commit:
        raise ValueError("runtime execution commit differs")
    previous, dependencies = prerequisites(stage)  # All old/current custody before a fresh model.
    manifest = source_manifest(sources(stage))
    deadline = began + p["stages"][stage]["total_host_seconds"]
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
                    "protocol": p,
                    "source": manifest,
                    "source_revision": commit,
                    "prerequisites": dependencies,
                    "previous_host_seconds": previous,
                    "elapsed_host_seconds": elapsed,
                    "cumulative_host_seconds": previous + elapsed,
                    "runs": runs,
                    **extra,
                }
            )
            + "\n"
        )

    for arm, seed in p["stages"][stage]["order"]:
        captures.clear()
        flush("cell_started_unverified", current={"arm": arm, "seed": seed})
        try:
            r = (
                trial(arm, seed, stage, deadline, parameters, captures)
                if time.monotonic() < deadline
                else {"arm": arm, "seed": seed, "status": "unrun_budget", "passed": False}
            )
        except Exception as error:
            details = {
                "current": {"arm": arm, "seed": seed},
                "error": {"type": type(error).__name__, "message": str(error)},
                "actual_transport_work": captures.get("actual_transport_work"),
            }
            for label in ("brain", "retention_brain"):
                if label in captures:
                    try:
                        details[label + "_snapshot"] = snapshot(
                            captures[label],
                            Path(parameters) / "interrupted",
                            stage,
                            arm,
                            seed,
                            label,
                        )
                    except Exception as capture_error:
                        details[label + "_snapshot_error"] = type(capture_error).__name__
            flush("cell_failed_unverified", **details)
            raise
        runs.append(r)
        flush("running_unverified")
    elapsed = time.monotonic() - began
    body = {
        "stage": stage,
        "protocol": p,
        "runs": runs,
        "source_revision": commit,
        "prerequisites": dependencies,
        "previous_host_seconds": previous,
        "elapsed_host_seconds": elapsed,
        "cumulative_host_seconds": previous + elapsed,
        "candidate_points_used": 1,
        "confirmation_seeds_used": [],
        "aws_opened": False,
        "world_opened": False,
        "capacity_admitted" if stage == "capacity" else "h1_admitted": all(
            r["passed"] for r in runs
        ),
    }
    flush("complete_unverified", body=body)
    if ctx.sr.revision() != commit or source_manifest(sources(stage)) != manifest:
        raise ValueError("runtime/source changed during execution")
    problem = check(body)
    if problem:
        raise ValueError(problem)
    if ctx.sr.revision() != commit or source_manifest(sources(stage)) != manifest:
        raise ValueError("runtime/source changed during verification")
    body["elapsed_host_seconds"] = time.monotonic() - began
    body["cumulative_host_seconds"] = previous + body["elapsed_host_seconds"]
    flush("verified_readings", body=body)
    return Receipt.build(p["stages"][stage]["receipt_kind"], body, sources(stage))


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
