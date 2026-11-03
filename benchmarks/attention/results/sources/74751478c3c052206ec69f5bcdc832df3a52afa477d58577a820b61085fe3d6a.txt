"""Direct raw9 H1 component: source-bound fresh-payload relevance controls."""

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

from cadence import Brain
from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "continuous-relevance-protocol.json"
STAGES = ("capacity", "learned_h1")
CONDITIONS = (
    "intact",
    "erase_A",
    "erase_B",
    "all_erased",
    "flip_relevant",
    "flip_irrelevant",
    "neutral_half",
    "erase_neutral",
    "reorder",
    "frozen_interface",
    "shuffled_interface",
)
REQUIRED = tuple(c for c in CONDITIONS if c != "all_erased")
PARAMETERS = ("efficacy", "bias", "log_gain")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
_CONTEXT = _CAPACITY = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(runtime, capacity=None):
    global _CONTEXT, _CAPACITY
    runtime = Path(runtime).resolve()
    old = load(
        "continuous_old_single", runtime / "benchmarks/attention/single_relevance_campaign.py"
    )
    inherited = old.configure(runtime)
    neutral = load(
        "continuous_old_neutral", runtime / "benchmarks/attention/neutral_variation_campaign.py"
    )
    neutral.configure(
        runtime, runtime / "benchmarks/attention/results/neutral-variation-capacity.json"
    )
    oracle = load("continuous_asked_oracle", HERE / "continuous_fixed_read.py").asked_read_class(
        runtime
    )

    class AskedBrain(inherited.admission.heads.MultiHeadBrain):
        def stimulus(self, observations, *, memory=True):
            drive = super().stimulus(observations, memory=memory)
            if self.brain.memory is not None:
                self.brain.memory = oracle(self.brain.memory, observations)
            return drive

    _CONTEXT = SimpleNamespace(
        runtime=runtime,
        base=old,
        neutral=neutral,
        admission=inherited.admission,
        relevance=inherited.relevance,
        initializer=inherited.initializer,
        sr=inherited.sr,
        heads=inherited.admission.heads,
        oracle=oracle,
        oracle_brain=AskedBrain,
        task=load("continuous_payload_task", HERE / "continuous_relevance_task.py"),
        cost_keys=inherited.base.COST_KEYS,
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
    f = deepcopy(context().base.declared()["factory"])
    f["inputs"] = 9
    if (
        p["schema"] != "cadence-continuous-relevance-campaign/1"
        or p["factory"] != f
        or p["candidate_points"] != 1
        or p["development_seeds"] != [0, 1]
        or set(p["stages"]) != set(STAGES)
        or p["conditions"] != list(CONDITIONS)
        or p["task_sha256"]
        != hashlib.sha256((HERE / "continuous_relevance_task.py").read_bytes()).hexdigest()
    ):
        raise ValueError("undeclared continuous relevance point or task")
    return p


def finished_prior():
    p = declared()["prior"]
    if (
        p["status"] != "finished_verified_neutral_h2"
        or p["source_stage"] != "h2"
        or any(
            p[k] is None
            for k in (
                "file",
                "kind",
                "digest",
                "archive_sha256",
                "cumulative_host_seconds",
                "capacity_file",
            )
        )
    ):
        raise ValueError("unfinished actual neutral-H2 prior; execution is blocked")
    return p


def sources(stage):
    ctx, prior = context(), finished_prior()
    paths = [
        (f"benchmarks/attention/{name}", HERE / name)
        for name in (
            "continuous_relevance_campaign.py",
            "continuous_relevance_task.py",
            "continuous_fixed_read.py",
            PROTOCOL.name,
        )
    ]
    paths += ctx.neutral.sources("h2")
    paths.append(
        (
            "benchmarks/attention/results/" + prior["file"],
            ctx.runtime / "benchmarks/attention/results" / prior["file"],
        )
    )
    if stage == "learned_h1":
        if _CAPACITY is None:
            raise ValueError("provide new continuous matched capacity")
        paths.append(("benchmarks/attention/results/continuous-relevance-capacity.json", _CAPACITY))
    unique = {}
    for name, path in paths:
        if name in unique and unique[name].resolve() != path.resolve():
            raise ValueError("conflicting source dependency")
        unique.setdefault(name, path)
    return list(unique.items())


def prior():
    ctx, p = context(), finished_prior()
    expected_capacity = ctx.runtime / "benchmarks/attention/results" / p["capacity_file"]
    if ctx.neutral._CAPACITY != expected_capacity.resolve():
        raise ValueError("declared neutral capacity dependency differs")
    path = ctx.runtime / "benchmarks/attention/results" / p["file"]
    key = (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(source_manifest(ctx.neutral.sources("h2"))),
        canonical_sha256(p),
    )
    if ctx.prior_cache == key:
        return Receipt.read(path)
    receipt = Receipt.read(path)
    valid, message = ctx.neutral.verify(path)
    if (
        not valid
        or receipt.kind != p["kind"]
        or receipt.digest != p["digest"]
        or key[0] != p["archive_sha256"]
        or receipt.body["stage"] != "h2"
        or receipt.body["cumulative_host_seconds"] != p["cumulative_host_seconds"]
    ):
        raise ValueError("completed neutral-H2 prior differs: " + message)
    after = (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(source_manifest(ctx.neutral.sources("h2"))),
        canonical_sha256(finished_prior()),
    )
    if key != after:
        raise ValueError("prior dependency changed during verification")
    ctx.prior_cache = key
    return receipt


def make_brain(arm, seed):
    ctx, f = context(), declared()["factory"]
    options = deepcopy(f["brain"])
    options["modules"] = tuple(options["modules"])
    if arm == "exposed":
        options.pop("empty_anchor")
        options = {k: v for k, v in options.items() if not k.startswith("recall_")}
        brain = Brain.compose(9, 2, seed=seed, **options)
        g = brain.brain
        if g.neuron_model.dt != f["dt"]:
            raise ValueError("composed exposed dt differs")
        brain.learner.brain = ctx.heads.Graph(
            g.connectome, g.neuron_model, efficacy=g.efficacy, bias=g.bias, log_gain=g.log_gain
        )
    elif arm in ("supplied_address", "h1"):
        cls = ctx.oracle_brain if arm == "supplied_address" else ctx.heads.MultiHeadBrain
        brain = cls.compose(
            9, 2, heads=1, port_copies=1, mode="agreement", dt=1.0, seed=seed, **options
        )
    else:
        raise ValueError("unknown continuous relevance arm")
    ctx.initializer.initialize(brain, seed=seed, query_scale=1)
    if any(
        brain.learner.config.to_dict()[k] != v for k, v in f["resolved_learning_guard"].items()
    ) or any(
        brain.brain.neuron_model.to_dict()[k] != v for k, v in f["resolved_neuron_guard"].items()
    ):
        raise ValueError("resolved numeric/learning point differs")
    return brain


def state_digest(brain):
    return context().base.state_digest(brain)


def schedule(seed, rounds, *, offset=0):
    task = context().task
    return [
        {
            "index": tick,
            "task": task.record(task.draw(seed, tick)),
            "neutral_amplitude": 1.0 if tick % 2 == 0 else 0.5,
        }
        for tick in range(offset, offset + rounds)
    ]


def history(record):
    task = context().task
    rows, quantities = np.asarray(record["rows"]), np.asarray(record["quantities"])
    batch = task.Histories(rows[::2, :3], quantities[::2])
    if task.record(batch) != record:
        raise ValueError("paired questions do not share actual full payloads")
    return batch


def prepare(brain, batch, query, arm):
    _, question = context().task.events(batch, query, exposed=arm == "exposed")
    context().admission.boundary(brain, 8)
    drive = np.array(brain.stimulus(question), copy=True)
    return question, drive, brain.brain.memory


def training_events(batch, arm, amplitude):
    task = context().task
    prefix, a = task.events(batch, 0, exposed=arm == "exposed", neutral_amplitude=amplitude)
    _, b = task.events(batch, 1, exposed=arm == "exposed")
    question = np.stack((a, b), axis=1).reshape(16, 9)
    target = np.column_stack((task.targets(batch, 0), task.targets(batch, 1))).reshape(16)
    return [np.repeat(x, 2, axis=0) for x in prefix], question, target


def train(brain, arm, records, deadline, progress=None):
    ctx, work, wrong, stop = context(), context().sr.Work(), [], None
    for record in records:
        if time.monotonic() >= deadline:
            stop = "wall_budget"
            break
        brain.reset()
        prefix, question, target = training_events(
            history(record["task"]), arm, record["neutral_amplitude"]
        )
        for event in prefix:
            if work.act(brain, event) is None:
                stop = "refused_prefix"
                break
        if stop:
            break
        ctx.admission.boundary(brain, 16)
        drive = np.array(brain.stimulus(question), copy=True)
        memory = brain.brain.memory
        answer = ctx.admission.act_drive(brain, question, drive, work)
        if answer is None:
            stop = "refused_query"
            break
        selected = np.asarray(answer).reshape(-1) != target
        wrong.append(int(selected.sum()))
        ctx.admission.teach(brain, drive, target, selected, memory, work)
        if progress is not None and len(wrong) % 32 == 0:
            progress(rounds=len(wrong), work=asdict(work), wrong_last=wrong[-1])
    return {
        "completed": len(wrong) == len(records),
        "rounds": len(wrong),
        "wrong_counts": wrong,
        "stop": stop,
        "work": asdict(work),
        "schedule": records,
        "schedule_sha256": canonical_sha256(records),
    }


def bank_data(memory):
    return context().base.bank_data(memory)


def bank_change(memory, condition, query, *, donor_bank=None):
    if memory is None:
        return None
    fixed = getattr(memory, "fixed_selector", False)
    bank = memory.bank if fixed else memory
    member = bank.memories[0]
    keys, values, held, ages = (
        getattr(member, k).copy() for k in ("keys", "values", "held", "ages")
    )
    if condition in ("erase_A", "erase_B", "all_erased", "erase_neutral", "neutral_half"):
        coordinate = {"erase_A": 0, "erase_B": 1, "erase_neutral": 4, "neutral_half": 4}.get(
            condition
        )
        selected = held.copy() if coordinate is None else held & (keys[:, :, coordinate] > 0)
        if not np.all(selected.sum(axis=1) == (3 if condition == "all_erased" else 1)):
            raise ValueError("actual bank does not contain the declared unique witnesses")
        if condition == "neutral_half":
            keys[selected] *= 0.5
            values[selected] *= 0.5
        else:
            held[selected] = False
    elif condition.startswith("flip_"):
        if donor_bank is None:
            raise ValueError("actual witnessed donor bank required")
        name = query if condition == "flip_relevant" else 1 - query
        source = donor_bank.memories[0]
        target_mask = held & (keys[:, :, name] > 0) & (keys[:, :, 3] > 0)
        source_mask = source.held & (source.keys[:, :, name] > 0) & (source.keys[:, :, 3] > 0)
        if not (np.all(target_mask.sum(axis=1) == 1) and np.all(source_mask.sum(axis=1) == 1)):
            raise ValueError("donor requires one actual named fact per stream")
        for row in range(len(held)):
            a, b = np.flatnonzero(target_mask[row])[0], np.flatnonzero(source_mask[row])[0]
            keys[row, a], values[row, a] = source.keys[row, b], source.values[row, b]
    elif condition == "reorder":
        order = np.arange(keys.shape[1])[::-1]
        keys, values, held, ages = keys[:, order], values[:, order], held[:, order], ages[order]
    elif condition not in ("intact", "frozen_interface", "shuffled_interface"):
        raise ValueError("unknown actual-bank condition")
    changed = replace(
        bank, memories=(replace(member, keys=keys, values=values, held=held, ages=ages),)
    )
    return type(memory)(changed, memory.observations) if fixed else changed


def expected_memory(brain, batch, query, arm, condition="intact", *, donor_batch=None):
    if arm == "exposed":
        return None
    ctx = context()
    prefix, question = ctx.task.events(batch, query)
    values = np.stack(prefix, axis=1)
    member = ctx.heads.Memory(
        brain.head_queries[0],
        brain.head_reads[0],
        brain.recall.heard * values,
        values,
        np.ones((8, 3), bool),
        np.array([2.0, 1.0, 0.0]),
        amplitude=3.0,
        temperature=0.25,
        mode="agreement",
        empty_anchor=True,
    )
    memory = ctx.heads.MultiMemory((member,))
    if arm == "supplied_address":
        memory = ctx.oracle(memory, question)
    donor_bank = None if donor_batch is None else expected_memory(brain, donor_batch, query, "h1")
    return bank_change(memory, condition, query, donor_bank=donor_bank)


def witnessed_memory(brain):
    """Capture the actual completed raw9 prefix store without choosing a question."""
    ctx, recall = context(), brain.recall
    if (
        recall is None
        or recall.window != 1
        or recall.next
        or brain.heads != 1
        or brain.port_copies != 1
    ):
        raise ValueError("continuous prefix requires the native one-head raw9 bank")
    member = ctx.heads.Memory(
        brain.head_queries[0],
        brain.head_reads[0],
        recall.keys,
        recall.values,
        recall.held,
        recall.ages(),
        amplitude=recall.amplitude,
        temperature=recall.temperature,
        recency=recall.recency,
        mode=brain.mode,
        empty_anchor=brain.empty_anchor,
    )
    return ctx.heads.MultiMemory((member,))


def witness(brain, arm, batch, deadline):
    ctx, work = context(), context().sr.Work()
    twin = ctx.admission.binding.fork(brain)
    twin.reset()
    prefix, _ = ctx.task.events(batch, 0)
    status = "completed"
    for event in prefix:
        if time.monotonic() >= deadline:
            status = "unrun_budget"
            break
        if work.act(twin, event) is None:
            status = "refused"
            break
    memory = None
    if status == "completed" and arm != "exposed":
        memory = witnessed_memory(twin)
    return twin, memory, {"status": status, "work": asdict(work), "bank": bank_data(memory)}


def evaluate(brain, arm, initial, seed, deadline, *, causal=False, start=0, progress=None):
    ctx, p = context(), declared()
    blocks = []
    for index in range(start, start + p["data"]["evaluation_blocks"]):
        batch = ctx.task.draw(seed, index, split="held")
        host, _, prefix = witness(brain, arm, batch, deadline)
        donor_banks, donors = {}, []
        if causal:
            for name in range(2):
                donor = ctx.task.donor(batch, name, seed, index)
                _, bank, recorded = witness(brain, arm, donor, deadline)
                donor_banks[name] = bank
                donors.append({"name": name, "task": ctx.task.record(donor), **recorded})
        conditions = {}
        for condition in CONDITIONS if causal else ("intact",):
            answers = np.full((8, 2), -1, dtype=np.int64)
            queries = []
            for q in range(2):
                work = ctx.sr.Work()
                row = {
                    "query": q,
                    "status": "unrun_budget",
                    "work": asdict(work),
                    "bank": None,
                    "drive": None,
                    "attention": None,
                    "activation": None,
                    "potential": None,
                }
                donor_name = q if condition == "flip_relevant" else 1 - q
                donor_ok = (
                    not condition.startswith("flip_") or donors[donor_name]["status"] == "completed"
                )
                if prefix["status"] != "completed":
                    row["status"] = "prefix_unavailable"
                elif not donor_ok:
                    row["status"] = "donor_unavailable"
                elif time.monotonic() < deadline:
                    twin = ctx.admission.binding.fork(host)
                    question, drive, memory = prepare(twin, batch, q, arm)
                    twin.brain.memory = bank_change(
                        memory, condition, q, donor_bank=donor_banks.get(donor_name)
                    )
                    if condition in ("frozen_interface", "shuffled_interface"):
                        ctx.relevance.interface_change(twin, condition, initial)
                    memory = twin.brain.memory
                    row.update(bank=bank_data(memory), drive=drive.tolist())
                    actual = ctx.admission.act_drive(twin, question, drive, work)
                    row["status"] = "refused" if actual is None else "completed"
                    if actual is not None:
                        answers[:, q] = np.asarray(actual).reshape(-1)
                        state = twin._activity()
                        row.update(
                            activation=np.atleast_2d(state.activation).tolist(),
                            potential=np.atleast_2d(state.v).tolist(),
                            attention=None
                            if memory is None
                            else memory.weights(np.atleast_2d(state.activation))[0].tolist(),
                        )
                    row["work"] = asdict(work)
                queries.append(row)
            conditions[condition] = {"answers": answers.tolist(), "queries": queries}
        blocks.append(
            {
                "index": index,
                "task": ctx.task.record(batch),
                "prefix": prefix,
                "donors": donors,
                "conditions": conditions,
            }
        )
        if progress is not None:
            progress(blocks=len(blocks), planned=p["data"]["evaluation_blocks"])
    panel = {"seed": seed, "start": start, "blocks": blocks}
    panel["scores"] = {
        c: pooled(panel, c)["scores"] for c in (CONDITIONS if causal else ("intact",))
    }
    return panel


def pooled(panel, condition):
    answers = np.concatenate(
        [np.asarray(b["conditions"][condition]["answers"]) for b in panel["blocks"]]
    )
    rows = np.concatenate([np.asarray(b["task"]["rows"])[::2, :3] for b in panel["blocks"]])
    target = rows[:, :2].copy()
    if condition == "flip_relevant":
        target = 1 - target
    right = answers == target
    unequal = rows[:, 0] != rows[:, 1]
    scores = {
        "accuracy": float(right.mean()),
        "per_query_accuracy": right.mean(0).tolist(),
        "per_order_accuracy": [float(right[rows[:, 2] == k].mean()) for k in (0, 1)],
        "unequal_fact_accuracy": float(right[unequal].mean()),
        "unanswered": int((answers < 0).sum()),
    }
    return {"answers": answers, "rows": rows, "correct": right, "scores": scores}


def complete(panel, conditions=("intact",)):
    return all(
        b["prefix"]["status"] == "completed"
        and all(
            r["status"] == "completed" and not r["work"]["action_refusals"]
            for c in conditions
            for r in b["conditions"][c]["queries"]
        )
        for b in panel["blocks"]
    )


def floor(panel):
    s = pooled(panel, "intact")["scores"]
    return (
        min(
            s["accuracy"],
            s["unequal_fact_accuracy"],
            *s["per_query_accuracy"],
            *s["per_order_accuracy"],
        )
        >= declared()["gates"]["acquisition_floor"]
    )


def metrics(result):
    ctx, p, g = context(), declared(), declared()["gates"]
    acquisition = bool(
        ctx.base.trained(result["training"], p["data"]["training_rounds"])
        and complete(result["initial"])
        and complete(result["evaluation"])
        and result["initial_parameters_preserved"]
        and result["parameters_preserved"]
        and floor(result["evaluation"])
    )
    out = {
        "acquisition_passed": acquisition,
        "primary_passed": acquisition,
        "causal_passed": False,
        "selection_passed": False,
        "interface_passed": False,
    }
    if result["arm"] != "h1":
        return out
    panel = result["causal"]
    host = pooled(panel, "intact")
    follows = pooled(panel, "flip_relevant")["correct"].mean(0)
    changes = {
        c: (pooled(panel, c)["answers"] != host["answers"]).mean(0).tolist()
        for c in ("flip_irrelevant", "neutral_half", "erase_neutral", "reorder")
    }
    erased = [
        float(
            host["correct"][:, q].mean()
            - pooled(panel, ("erase_A", "erase_B")[q])["correct"][:, q].mean()
        )
        for q in range(2)
    ]
    interfaces = {
        c: float(host["correct"].mean() - pooled(panel, c)["correct"].mean())
        for c in ("frozen_interface", "shuffled_interface")
    }
    selective, shifts = [[], []], []
    for block in panel["blocks"]:
        batch = history(block["task"])
        unequal = batch.rows[:, 0] != batch.rows[:, 1]
        ps = []
        for q, row in enumerate(block["conditions"]["intact"]["queries"]):
            if row["attention"] is None or row["bank"] is None:
                selective[q].extend([False] * int(unequal.sum()))
                ps.append(None)
                continue
            attention, bank = np.asarray(row["attention"]), row["bank"]
            keys, held = np.asarray(bank["keys"]), np.asarray(bank["held"])
            requested = held & (keys[:, :, q] > 0) & (keys[:, :, 3] > 0)
            mass = (attention * requested).sum(1)
            competition = np.maximum(
                np.where(requested, 0.0, attention).max(1), 1.0 - attention.sum(1)
            )
            selective[q].extend(
                (
                    (mass >= g["record_mass_minimum"])
                    & (mass - competition >= g["record_margin_minimum"])
                )[unequal].tolist()
            )
            ps.append(attention)
        if all(v is not None for v in ps):
            shifts.extend(np.abs(ps[0] - ps[1]).sum(1).tolist())
    shares = [float(np.mean(v)) for v in selective]
    qualified = complete(panel, REQUIRED)
    causal = bool(
        acquisition
        and qualified
        and floor(panel)
        and min(follows) >= g["donor_follow_each_query_minimum"]
        and max(v for values in changes.values() for v in values)
        <= g["invariant_change_each_query_maximum"]
        and min(erased) >= g["relevant_erasure_loss_each_query_minimum"]
    )
    selection = bool(qualified and min(shares) >= g["selective_unequal_share_each_query_minimum"])
    interface = bool(qualified and min(interfaces.values()) >= g["interface_loss_overall_minimum"])
    out.update(
        causal_passed=causal,
        selection_passed=selection,
        interface_passed=interface,
        primary_passed=bool(causal and selection and interface),
        donor_follow_each_query=follows.tolist(),
        invariant_changes_each_query=changes,
        relevant_erasure_losses=erased,
        interface_losses=interfaces,
        interface_losses_each_query={
            c: (host["correct"].mean(0) - pooled(panel, c)["correct"].mean(0)).tolist()
            for c in ("frozen_interface", "shuffled_interface")
        },
        selective_unequal_counts=[sum(v) for v in selective],
        selective_unequal_denominators=[len(v) for v in selective],
        selective_unequal_shares=shares,
        paired_question_attention_l1_mean=None if not shifts else float(np.mean(shifts)),
    )
    return out


def retained(retention, baseline):
    return bool(
        retention["status"] == "measured"
        and context().base.trained(retention["training"], declared()["retention"]["rounds"])
        and complete(retention["evaluation"])
        and floor(retention["evaluation"])
        and retention["parameters_preserved"]
        and baseline - pooled(retention["evaluation"], "intact")["scores"]["accuracy"]
        <= declared()["gates"]["retention_loss_maximum"]
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
        "schema": "cadence-attention-continuous-relevance-parameters/1",
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
        "scope": "Reset-round research arrays, not a supported continuing checkpoint",
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


def trial(arm, seed, stage, deadline, parameters, captures, progress=None):
    ctx, p, began = context(), declared(), time.monotonic()
    brain = make_brain(arm, seed)
    captures["brain"] = brain
    cost = dict.fromkeys(ctx.cost_keys, 0)
    captures["actual_transport_work"] = cost
    initial = brain.brain.efficacy.copy()
    initial_parameters, initial_state = ctx.sr.parameter_digest(brain), state_digest(brain)

    def notify(phase, **details):
        if progress is not None:
            progress(phase=phase, actual_transport_work=dict(cost), **details)

    with ctx.admission.metered(cost):
        notify("newborn_started")
        newborn = evaluate(
            brain, arm, initial, seed, deadline, progress=lambda **d: notify("newborn", **d)
        )
        newborn_state = state_digest(brain)
        order = schedule(seed, p["data"]["training_rounds"])
        captures["training_schedule"] = order
        notify("main_started")
        training = train(
            brain,
            arm,
            order,
            min(deadline, time.monotonic() + p["stages"][stage]["training_seconds_per_call"]),
            progress=lambda **d: notify("main", **d),
        )
        final, final_state = ctx.sr.parameter_digest(brain), state_digest(brain)
        notify("evaluation_started", rounds=training["rounds"])
        evaluation = evaluate(
            brain, arm, initial, seed, deadline, progress=lambda **d: notify("evaluation", **d)
        )
        causal = None
        if stage == "learned_h1":
            notify("causal_started")
            causal = evaluate(
                brain,
                arm,
                initial,
                seed,
                deadline,
                causal=True,
                progress=lambda **d: notify("causal", **d),
            )
        result = {
            "status": "measured",
            "arm": arm,
            "seed": seed,
            "mode": "agreement",
            "initial": newborn,
            "training": training,
            "evaluation": evaluation,
            "initial_parameters": initial_parameters,
            "initial_state": initial_state,
            "initial_evaluation_state": newborn_state,
            "initial_parameters_preserved": initial_state == newborn_state,
            "initial_interface_sha256": canonical_sha256(initial),
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
            "random_reference": np.random.default_rng(np.random.SeedSequence([622032, seed]))
            .integers(2, size=(256, 2))
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
                suffix = schedule(seed, p["retention"]["rounds"], offset=512)
                captures["retention_schedule"] = suffix
                notify("retention_started")
                continued = train(
                    twin,
                    arm,
                    suffix,
                    min(
                        deadline, time.monotonic() + p["stages"][stage]["training_seconds_per_call"]
                    ),
                    progress=lambda **d: notify("retention", **d),
                )
                after, after_state = ctx.sr.parameter_digest(twin), state_digest(twin)
                panel = evaluate(
                    twin,
                    arm,
                    initial,
                    seed,
                    deadline,
                    start=32,
                    progress=lambda **d: notify("retention_evaluation", **d),
                )
                retention = {
                    "status": "measured",
                    "training": continued,
                    "evaluation": panel,
                    "final_parameters": after,
                    "final_state": after_state,
                    "evaluation_parameters": ctx.sr.parameter_digest(twin),
                    "evaluation_state": state_digest(twin),
                    "parameters_preserved": after_state == state_digest(twin),
                    "parameter_snapshot": snapshot(twin, parameters, stage, arm, seed, "retention"),
                }
                retention["passed"] = retained(
                    retention, pooled(evaluation, "intact")["scores"]["accuracy"]
                )
        result["retention"] = retention
    result.update(
        parameter_snapshot=snapshot(brain, parameters, stage, arm, seed),
        work=cost,
        elapsed_host_seconds=time.monotonic() - began,
        passed=bool(
            result["metrics"]["primary_passed"] and (stage == "capacity" or retention["passed"])
        ),
    )
    notify("cell_completed", passed=result["passed"])
    return result


def prerequisites(stage):
    old = prior()
    previous, dependencies = old.body["cumulative_host_seconds"], {"neutral_h2": old.digest}
    if stage == "learned_h1":
        valid, message = verify(_CAPACITY)
        capacity = Receipt.read(_CAPACITY) if _CAPACITY is not None else None
        if not valid or capacity.body["capacity_admitted"] is not True:
            raise ValueError("new matched continuous capacity required: " + message)
        previous, dependencies["capacity"] = (
            capacity.body["cumulative_host_seconds"],
            capacity.digest,
        )
    return previous, dependencies


def panel_check(panel, brain, arm, seed, *, causal=False, start=0):
    ctx = context()
    if panel["seed"] != seed or panel["start"] != start or len(panel["blocks"]) != 32:
        raise ValueError("held payload panel census differs")
    for index, block in enumerate(panel["blocks"], start):
        batch = ctx.task.draw(seed, index, split="held")
        if block["index"] != index or block["task"] != ctx.task.record(batch):
            raise ValueError("actual held quantities or paired rows differ")
        prefix = block["prefix"]
        ctx.base.work_check(prefix["work"])
        if prefix["status"] == "completed":
            if (
                prefix["work"]["actions"] != 3
                or prefix["work"]["action_rows"] != 24
                or prefix["work"]["action_refusals"]
                or prefix["work"]["max_action_residual"] > 0.003
                or prefix["bank"] != bank_data(expected_memory(brain, batch, 0, arm))
            ):
                raise ValueError("actual prefix bank or work differs")
        elif prefix["status"] not in ("refused", "unrun_budget"):
            raise ValueError("unknown prefix status")
        if len(block["donors"]) != (2 if causal else 0):
            raise ValueError("actual donor census differs")
        for name, donor in enumerate(block["donors"]):
            actual = ctx.task.donor(batch, name, seed, index)
            ctx.base.work_check(donor["work"])
            if donor["name"] != name or donor["task"] != ctx.task.record(actual):
                raise ValueError("actual donor payload differs")
            if donor["status"] == "completed" and (
                donor["work"]["actions"] != 3
                or donor["work"]["action_rows"] != 24
                or donor["work"]["action_refusals"]
                or donor["work"]["max_action_residual"] > 0.003
                or donor["bank"] != bank_data(expected_memory(brain, actual, 0, arm))
            ):
                raise ValueError("actual donor bank or qualified work differs")
            if donor["status"] not in ("completed", "refused", "unrun_budget"):
                raise ValueError("unknown donor status")
        if set(block["conditions"]) != set(CONDITIONS if causal else ("intact",)):
            raise ValueError("condition census differs")
        for condition, reading in block["conditions"].items():
            answers = np.asarray(reading["answers"])
            if (
                answers.shape != (8, 2)
                or not np.isin(answers, (-1, 0, 1)).all()
                or len(reading["queries"]) != 2
            ):
                raise ValueError("invalid paired answers")
            for q, row in enumerate(reading["queries"]):
                ctx.base.work_check(row["work"])
                if row["query"] != q:
                    raise ValueError("question ownership differs")
                status = row["status"]
                if status not in (
                    "completed",
                    "refused",
                    "prefix_unavailable",
                    "donor_unavailable",
                    "unrun_budget",
                ):
                    raise ValueError("unknown query status")
                if status != "completed" and np.any(answers[:, q] != -1):
                    raise ValueError("unavailable query contains answered rows")
                if status in ("completed", "refused"):
                    if (
                        prefix["status"] != "completed"
                        or row["work"]["actions"] != 1
                        or row["work"]["action_rows"] != 8
                        or row["work"]["action_refusals"] != int(status == "refused")
                    ):
                        raise ValueError("query phase ownership/work differs")
                    donor_name = q if condition == "flip_relevant" else 1 - q
                    donor = (
                        ctx.task.donor(batch, donor_name, seed, index)
                        if condition.startswith("flip_")
                        else None
                    )
                    expected = expected_memory(brain, batch, q, arm, condition, donor_batch=donor)
                    _, query = ctx.task.events(batch, q, exposed=arm == "exposed")
                    clean = ctx.admission.binding.fork(brain)
                    _, drive, _ = prepare(clean, batch, q, arm)
                    if row["bank"] != bank_data(expected) or row["drive"] != drive.tolist():
                        raise ValueError(
                            "captured complete drive or actual witnessed boundary differs"
                        )
                    if (
                        condition.startswith("flip_")
                        and block["donors"][donor_name]["status"] != "completed"
                    ):
                        raise ValueError("answered donor condition lacks actual donor witness")
                    if status == "completed":
                        activity, potential = (
                            np.asarray(row["activation"]),
                            np.asarray(row["potential"]),
                        )
                        if (
                            activity.shape != (8, brain.connectome.n)
                            or potential.shape != activity.shape
                            or not np.isfinite(activity).all()
                            or not np.isfinite(potential).all()
                            or row["work"]["max_action_residual"] > 0.003
                            or not np.array_equal(
                                answers[:, q], np.argmax(activity[:, brain.motor_index], axis=1)
                            )
                        ):
                            raise ValueError("completed actual neural state/answer differs")
                        if expected is None:
                            if row["attention"] is not None:
                                raise ValueError("exposed control has an undeclared selector")
                        elif row["attention"] is None or not np.allclose(
                            expected.weights(activity)[0], row["attention"], rtol=0, atol=1e-12
                        ):
                            raise ValueError("captured selector disagrees with live query activity")
                elif row["work"] != asdict(ctx.sr.Work()):
                    raise ValueError("unrun query contains neural work")
                if any(
                    v
                    for k, v in row["work"].items()
                    if k.startswith("teaching_")
                    or k
                    in ("lessons", "lesson_refusals", "presentations", "accepted_presentations")
                ):
                    raise ValueError("evaluation contains teaching")
    if panel["scores"] != {
        c: pooled(panel, c)["scores"] for c in (CONDITIONS if causal else ("intact",))
    }:
        raise ValueError("held scores do not recompute")


def training_check(training, seed, *, retention=False):
    ctx, p = context(), declared()
    rounds, offset = (
        (p["retention"]["rounds"], 512) if retention else (p["data"]["training_rounds"], 0)
    )
    records = schedule(seed, rounds, offset=offset)
    if (
        training["schedule"] != records
        or training["schedule_sha256"] != canonical_sha256(records)
        or type(training["rounds"]) is not int
        or not 0 <= training["rounds"] <= rounds
        or training["completed"] != (training["rounds"] == rounds)
        or len(training["wrong_counts"]) != training["rounds"]
        or any(type(v) is not int or not 0 <= v <= 16 for v in training["wrong_counts"])
        or training["stop"]
        not in (
            (None,) if training["completed"] else ("wall_budget", "refused_prefix", "refused_query")
        )
    ):
        raise ValueError("joint payload/question/neutral schedule or training completion differs")
    ctx.base.work_check(training["work"])
    work, rounds, counts = training["work"], training["rounds"], training["wrong_counts"]
    if (
        work["lessons"] != sum(n > 0 for n in counts)
        or work["presentations"] != sum(counts)
        or not 4 * rounds <= work["actions"] <= 4 * rounds + 4
        or work["action_rows"] != 16 * work["actions"]
        or (training["completed"] and work["actions"] != 4 * rounds)
        or (not work["action_refusals"] and work["max_action_residual"] > 0.003)
    ):
        raise ValueError("actual training lesson, presentation or action census differs")


def snapshot_check(descriptor, result, stage, revision, phase="final"):
    state = result if phase == "final" else result["retention"]
    m = descriptor["metadata"]
    if (
        m["schema"] != "cadence-attention-continuous-relevance-parameters/1"
        or m["stage"] != stage
        or m["arm"] != result["arm"]
        or m["seed"] != result["seed"]
        or m["phase"] != phase
        or m["mode"] != "agreement"
        or m["source_revision"] != revision
        or m["source"] != source_manifest(sources(stage))
        or m["query_founder"] != result["query_initialization"]
        or m["learning"] != result["learning_config"]
        or m["parameter_sha256"] != state["final_parameters"]
        or m["state_sha256"] != state["final_state"]
        or type(m["updates"]) is not int
        or type(m["contrast_updates"]) is not int
        or not 0 <= m["contrast_updates"] <= m["updates"]
    ):
        raise ValueError("continuous snapshot custody differs")
    for name in ("archive_sha256", "content_sha256"):
        value = descriptor[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("invalid actual parameter archive digest")


def check(body):
    try:
        ctx, p, stage = context(), declared(), body["stage"]
        if stage not in STAGES or body["protocol"] != p:
            return "continuous protocol differs"
        previous, dependencies = prerequisites(stage)
        revision = body["source_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or body["previous_host_seconds"] != previous
            or body["prerequisites"] != dependencies
            or not ctx.base.context().base.context().campaign.finite(body["elapsed_host_seconds"])
            or body["cumulative_host_seconds"] != previous + body["elapsed_host_seconds"]
            or body["candidate_points_used"] != 1
            or body["confirmation_seeds_used"]
            or body["head_advantage_claimed"] is not False
            or body["world_runs_opened"] is not False
        ):
            return "continuous stage custody or cost differs"
        runs = body["runs"]
        if [[r["arm"], r["seed"]] for r in runs] != p["stages"][stage]["order"]:
            return "missing, duplicate or extra cells"
        if (
            any(r["status"] == "unrun_budget" for r in runs)
            and body["elapsed_host_seconds"] < p["stages"][stage]["total_host_seconds"]
        ):
            return "unrun cell before stage budget"
        if sum(r.get("elapsed_host_seconds", 0.0) for r in runs) > body["elapsed_host_seconds"]:
            return "cell charges exceed actual stage charge"
        for r in runs:
            if r["status"] == "unrun_budget":
                if r["passed"] is not False:
                    return "unrun optimism"
                continue
            if r["status"] != "measured" or r["mode"] != "agreement":
                return "unknown measured cell"
            training_check(r["training"], r["seed"])
            expected = make_brain(r["arm"], r["seed"])
            if (
                r["query_initialization"] != expected.query_initialization
                or r["initial_parameters"] != ctx.sr.parameter_digest(expected)
                or r["initial_state"] != state_digest(expected)
                or r["initial_interface_sha256"] != canonical_sha256(expected.brain.efficacy)
                or r["learning_config"] != expected.learner.config.to_dict()
                or r["neuron_model"] != expected.brain.neuron_model.to_dict()
                or r["resolved"]["genes"] != expected.describe()["genes"]
                or r["neurons"] != expected.connectome.n
                or r["synapses"] != expected.connectome.synapses
                or r["numerical_policy"] != ctx.admission.numerical_settings(expected)
                or r["initial_parameters_preserved"]
                != (r["initial_state"] == r["initial_evaluation_state"])
                or r["parameters_preserved"] != (r["final_state"] == r["evaluation_state"])
                or r["evaluation_parameters"] != r["final_parameters"]
                or set(r["work"]) != set(ctx.cost_keys)
                or any(type(v) is not int or v < 0 for v in r["work"].values())
                or not np.isfinite(r["elapsed_host_seconds"])
                or r["elapsed_host_seconds"] < 0
                or r["random_reference"]
                != np.random.default_rng(np.random.SeedSequence([622032, r["seed"]]))
                .integers(2, size=(256, 2))
                .tolist()
            ):
                return "founder/config/state/random/work differs"
            for name in ("initial", "evaluation"):
                panel_check(r[name], expected, r["arm"], r["seed"])
            if stage == "learned_h1":
                panel_check(r["causal"], expected, r["arm"], r["seed"], causal=True)
            if r["metrics"] != metrics(r):
                return "continuous metrics do not recompute"
            snapshot_check(r["parameter_snapshot"], r, stage, revision)
            ret = r["retention"]
            baseline = pooled(r["evaluation"], "intact")["scores"]["accuracy"]
            if ret["status"] == "measured":
                if stage != "learned_h1" or not r["metrics"]["primary_passed"]:
                    return "unpermitted retention"
                training_check(ret["training"], r["seed"], retention=True)
                panel_check(ret["evaluation"], expected, r["arm"], r["seed"], start=32)
                if (
                    ret["parameters_preserved"] != (ret["final_state"] == ret["evaluation_state"])
                    or ret["evaluation_parameters"] != ret["final_parameters"]
                ):
                    return "retention state differs"
                snapshot_check(ret["parameter_snapshot"], r, stage, revision, "retention")
            elif ret["status"] != (
                "not_applicable_capacity"
                if stage == "capacity"
                else "unrun_budget"
                if r["metrics"]["primary_passed"]
                else "not_run_primary_failed"
            ):
                return "conditional retention status differs"
            if ret["passed"] != retained(ret, baseline) or r["passed"] != bool(
                r["metrics"]["primary_passed"] and (stage == "capacity" or ret["passed"])
            ):
                return "continuous cell optimism"
        admitted = all(r["passed"] for r in runs)
        if body["capacity_admitted" if stage == "capacity" else "h1_admitted"] != admitted:
            return "continuous aggregate optimism"
    except (KeyError, TypeError, ValueError, OSError, AssertionError, IndexError):
        return "incomplete or invalid continuous relevance receipt"
    return None


def verify(path):
    if path is None:
        return False, "missing new matched capacity receipt"
    receipt = Receipt.read(path)
    stage = receipt.body.get("stage")
    if stage not in STAGES or receipt.kind != declared()["stages"][stage]["receipt_kind"]:
        return False, "wrong continuous receipt kind"
    return Receipt.verify(path, sources=sources(stage), check=check)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n")
    temporary.replace(path)


def run(stage, output, parameters, commit):
    began, ctx, p = time.monotonic(), context(), declared()
    if ctx.sr.revision() != commit:
        raise ValueError("runtime execution commit differs")
    previous, dependencies = prerequisites(stage)  # Before any fresh candidate.
    manifest = source_manifest(sources(stage))
    deadline, runs, captures = began + p["stages"][stage]["total_host_seconds"], [], {}
    journal = output.with_suffix(".partial.json")

    def flush(status, **extra):
        elapsed = time.monotonic() - began
        atomic_json(
            journal,
            {
                "status": status,
                "stage": stage,
                "source": manifest,
                "protocol": p,
                "runs": runs,
                "source_revision": commit,
                "prerequisites": dependencies,
                "previous_host_seconds": previous,
                "elapsed_host_seconds": elapsed,
                "cumulative_host_seconds": previous + elapsed,
                **extra,
            },
        )

    for arm, seed in p["stages"][stage]["order"]:
        captures.clear()
        flush("cell_started_unverified", current={"arm": arm, "seed": seed})

        def progress(_arm=arm, _seed=seed, **detail):
            elapsed = time.monotonic() - began
            atomic_json(
                output.with_suffix(".progress.json"),
                {
                    "status": "progress_unverified",
                    "stage": stage,
                    "arm": _arm,
                    "seed": _seed,
                    "source_revision": commit,
                    "elapsed_host_seconds": elapsed,
                    "cumulative_host_seconds": previous + elapsed,
                    **detail,
                },
            )

        try:
            result = (
                trial(arm, seed, stage, deadline, parameters, captures, progress)
                if time.monotonic() < deadline
                else {"arm": arm, "seed": seed, "status": "unrun_budget", "passed": False}
            )
        except Exception as error:
            details = {
                "current": {"arm": arm, "seed": seed},
                "error": {"type": type(error).__name__, "message": str(error)},
                "actual_transport_work": captures.get("actual_transport_work"),
                "training_schedule": captures.get("training_schedule"),
                "retention_schedule": captures.get("retention_schedule"),
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
        runs.append(result)
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
        "head_advantage_claimed": False,
        "world_runs_opened": False,
        "capacity_admitted" if stage == "capacity" else "h1_admitted": all(
            r["passed"] for r in runs
        ),
    }
    flush("complete_unverified", body=body)
    if ctx.sr.revision() != commit or source_manifest(sources(stage)) != manifest:
        raise ValueError("runtime revision/source changed during execution")
    problem = check(body)
    if problem:
        raise ValueError(problem)
    if ctx.sr.revision() != commit or source_manifest(sources(stage)) != manifest:
        raise ValueError("runtime revision/source changed during final verification")
    body.update(elapsed_host_seconds=time.monotonic() - began)
    body["cumulative_host_seconds"] = previous + body["elapsed_host_seconds"]
    flush("verified_readings", body=body)
    return Receipt.build(p["stages"][stage]["receipt_kind"], body, sources(stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--capacity", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    configure(args.runtime, args.capacity)
    if args.verify:
        ok, message = verify(args.verify)
        print(message)
        raise SystemExit(0 if ok else 1)
    if not args.stage or args.out is None or args.parameters is None or not args.commit:
        parser.error("provide --stage, --out, --parameters and --commit")
    if args.out.exists() or args.out.with_suffix(".partial.json").exists():
        raise FileExistsError(args.out)
    run(args.stage, args.out, args.parameters, args.commit).write(args.out)
