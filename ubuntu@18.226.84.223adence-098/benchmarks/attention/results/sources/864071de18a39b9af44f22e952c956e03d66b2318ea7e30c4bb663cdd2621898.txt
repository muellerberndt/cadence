"""One-factor witnessed-neutral training variation with an exact completed H1 prior.

Prepared but unrun: freeze complete sources before explicit bounded stages.
Imports construct no brain. All original minimal-task gates remain exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "neutral-variation-protocol.json"
STAGES = ("capacity", "h2")
PARAMETERS = ("efficacy", "bias", "log_gain")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
COST_KEYS = (
    "cortical_synapse_transports",
    "memory_scan_macs",
    "softmax_elements",
    "memory_drive_evaluations",
    "fixed_read_coordinate_updates",
)
_CONTEXT = None
_CAPACITY = None


def configure(runtime, capacity=None):
    # Independent normal module configuration, never a temporary replacement of
    # any runtime task/factory/train/evaluate function.
    import importlib.util
    import sys

    global _CONTEXT, _CAPACITY
    runtime = Path(runtime).resolve()
    path = runtime / "benchmarks/attention/minimal_binding_campaign.py"
    spec = importlib.util.spec_from_file_location("neutral_frozen_minimal", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = module.configure(
        runtime, runtime / "benchmarks/attention/results/minimal-binding-capacity.json"
    )
    _CONTEXT = SimpleNamespace(
        runtime=runtime,
        base=module,
        campaign=old.campaign,
        admission=old.admission,
        relevance=old.relevance,
        initializer=old.initializer,
        sr=old.sr,
        task=old.task,
        h1=None,
        h1_spec_key=None,
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
        p["schema"] != "cadence-neutral-variation-two-binding-campaign/1"
        or p["query_scale"] != 1
        or p["development_seeds"] != [0, 1]
        or set(p["stages"]) != set(STAGES)
    ):
        raise ValueError("undeclared neutral-variation point")
    old = context().base.declared()
    for key in (
        "brain",
        "dt",
        "task",
        "gates",
        "retention",
        "initialization",
        "resolved_learning_guard",
        "resolved_neuron_guard",
        "causal_controls",
        "random_references",
        "future_head_benefit_assessment",
    ):
        if p[key] != old[key]:
            raise ValueError("frozen minimal-task point/gates differ: " + key)
    original = context().runtime / "benchmarks/attention/minimal-binding-protocol.json"
    if hashlib.sha256(original.read_bytes()).hexdigest() != p["base_protocol_sha256"]:
        raise ValueError("original minimal protocol bytes differ")
    for stage in STAGES:
        for key in ("order", "total_host_seconds", "training_seconds_per_call", "gate"):
            if p["stages"][stage][key] != old["stages"][stage][key]:
                raise ValueError("frozen stage order/budget/gate differs")
    return p


def finished_prior_spec():
    spec = declared()["immediate_prior"]
    if (
        spec["status"] != "finished_verified_h1_prior"
        or any(
            spec[k] is None
            for k in (
                "file",
                "kind",
                "digest",
                "archive_sha256",
                "cumulative_host_seconds",
                "source_stage",
            )
        )
        or spec["source_stage"] not in ("capacity", "learned_h1")
        or (spec["source_stage"] == "learned_h1" and spec["capacity_file"] is None)
    ):
        raise ValueError("unfinished actual H1 prior: scientific execution is blocked")
    return spec


def h1_verifier():
    ctx = context()
    spec = finished_prior_spec()
    identity = (str(ctx.runtime), canonical_sha256(spec))
    if ctx.h1 is None or ctx.h1_spec_key != identity:
        h1 = ctx.base.load(
            "neutral_frozen_single",
            ctx.runtime / "benchmarks/attention/single_relevance_campaign.py",
        )
        h1.configure(
            ctx.runtime,
            None
            if spec["capacity_file"] is None
            else ctx.runtime / "benchmarks/attention/results" / spec["capacity_file"],
        )
        ctx.h1, ctx.h1_spec_key = h1, identity
    return ctx.h1


def sources(stage):
    ctx, p = context(), declared()
    h1 = h1_verifier()
    prior = finished_prior_spec()
    paths = [
        ("benchmarks/attention/neutral_variation_campaign.py", Path(__file__).resolve()),
        ("benchmarks/attention/neutral-variation-protocol.json", PROTOCOL),
        *ctx.base.sources("h2"),
        (
            "benchmarks/attention/results/" + p["retained_negative"]["file"],
            ctx.runtime / "benchmarks/attention/results" / p["retained_negative"]["file"],
        ),
        *h1.sources(prior["source_stage"]),
        (
            "benchmarks/attention/results/" + prior["file"],
            ctx.runtime / "benchmarks/attention/results" / prior["file"],
        ),
    ]
    if stage == "h2":
        if _CAPACITY is None:
            raise ValueError("provide new changed-training matched capacity")
        paths.append(("benchmarks/attention/results/neutral-variation-capacity.json", _CAPACITY))
    unique = {}
    for name, path in paths:
        if name in unique and path.resolve() != unique[name].resolve():
            raise ValueError("conflicting source dependency")
        unique.setdefault(name, path)
    return list(unique.items())


def prior():
    ctx, p = context(), declared()
    current = finished_prior_spec()  # Fail before any model construction.
    h1 = h1_verifier()
    oldspec = p["retained_negative"]
    oldpath = ctx.runtime / "benchmarks/attention/results" / oldspec["file"]
    path = ctx.runtime / "benchmarks/attention/results" / current["file"]
    identity = (
        hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(
            source_manifest([*ctx.base.sources("h2"), *h1.sources(current["source_stage"])])
        ),
        canonical_sha256({"old": oldspec, "current": current}),
    )
    if ctx.prior_cache == identity:
        return Receipt.read(path)
    old, receipt = Receipt.read(oldpath), Receipt.read(path)
    oldvalid, oldmessage = ctx.base.verify(oldpath)
    valid, message = h1.verify(path)
    if (
        not oldvalid
        or old.kind != oldspec["kind"]
        or old.digest != oldspec["digest"]
        or identity[0] != oldspec["archive_sha256"]
        or old.body["cumulative_host_seconds"] != oldspec["cumulative_host_seconds"]
        or old.body["h2_admitted"] is not False
    ):
        raise ValueError("retained negative differs: " + oldmessage)
    if (
        not valid
        or receipt.kind != current["kind"]
        or receipt.digest != current["digest"]
        or identity[1] != current["archive_sha256"]
        or receipt.body["stage"] != current["source_stage"]
        or (
            current["source_stage"] == "capacity" and receipt.body["capacity_admitted"] is not False
        )
        or receipt.body["cumulative_host_seconds"] != current["cumulative_host_seconds"]
        or receipt.body["cumulative_host_seconds"] < old.body["cumulative_host_seconds"]
    ):
        raise ValueError("exact finished H1 prior differs: " + message)
    after = (
        hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_sha256(
            source_manifest([*ctx.base.sources("h2"), *h1.sources(current["source_stage"])])
        ),
        canonical_sha256(
            {"old": declared()["retained_negative"], "current": finished_prior_spec()}
        ),
    )
    if identity != after:
        raise ValueError("prior dependency changed during verification")
    ctx.prior_cache = identity
    return receipt


def profile(stage):
    declared()
    return context().base.profile(stage)


def make_brain(arm, seed, stage):
    return context().base.make_brain(arm, seed, stage)


def state_digest(brain):
    return context().base.state_digest(brain)


def evaluate(*args, **kwargs):
    return context().base.evaluate(*args, **kwargs)


def metrics(result):
    return context().base.metrics(result)


def retention_passes(retention, before):
    return context().base.retention_passes(retention, before)


def retention_schedule(seed):
    return context().base.retention_schedule(seed)


def panel_check(panel, **kwargs):
    return context().base.panel_check(panel, **kwargs)


def neutral_amplitudes(rounds, *, offset=0):
    if type(rounds) is not int or rounds < 0 or type(offset) is not int or offset < 0:
        raise ValueError("rounds and offset must be nonnegative integers")
    return np.where((np.arange(rounds) + offset) % 2 == 0, 1.0, 0.5)


def joint_schedule_digest(schedule, *, offset=0):
    return canonical_sha256(
        {
            "rows": schedule,
            "neutral_amplitudes": neutral_amplitudes(len(schedule), offset=offset),
            "global_round_offset": offset,
        }
    )


def training_events(rows, amplitude):
    if amplitude not in (1.0, 0.5):
        raise ValueError("undeclared actual neutral amplitude")
    prefix, query = context().task.events(rows)
    # This observation is subsequently accepted/witnessed normally. No training
    # Memory/held/key/value is overwritten by this data preparation.
    prefix[2] = prefix[2] * amplitude
    return prefix, query


def train(brain, arm, schedule, deadline, *, offset=0):
    ctx = context()
    work, wrong_counts, stop = ctx.sr.Work(), [], None
    amplitudes = neutral_amplitudes(len(schedule), offset=offset)
    for tick, rows in enumerate(schedule):
        if time.monotonic() >= deadline:
            stop = "wall_budget"
            break
        brain.reset()
        prefix, _ = training_events(rows, amplitudes[tick])
        for observation in prefix:
            if work.act(brain, observation) is None:
                stop = "refused_prefix"
                break
        if stop:
            break
        query, drive, memory = ctx.admission.prepare(brain, rows, arm)
        answer = ctx.admission.act_drive(brain, query, drive, work)
        if answer is None:
            stop = "refused_query"
            break
        target = ctx.task.targets(rows)
        selected = (np.asarray(answer) != target).any(axis=1)
        wrong_counts.append(int(selected.sum()))
        ctx.admission.teach(brain, drive, target, selected, memory, work)
    return {
        "completed": len(wrong_counts) == len(schedule),
        "rounds": len(wrong_counts),
        "wrong_counts": wrong_counts,
        "stop": stop,
        "work": asdict(work),
        "global_round_offset": offset,
        "joint_schedule_sha256": joint_schedule_digest(schedule, offset=offset),
        "completed_neutral_amplitudes": amplitudes[: len(wrong_counts)].tolist(),
    }


def training_data_check(training, schedule, *, offset=0):
    if (
        training["global_round_offset"] != offset
        or training["joint_schedule_sha256"] != joint_schedule_digest(schedule, offset=offset)
        or training["completed_neutral_amplitudes"]
        != neutral_amplitudes(len(schedule), offset=offset)[: training["rounds"]].tolist()
    ):
        raise ValueError("actual neutral training schedule differs")


def snapshot(brain, directory, *, stage, arm, seed, phase="final"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}-{arm}-{seed}-{phase}.npz"
    arrays = {k: np.array(getattr(brain.brain, k), copy=True) for k in PARAMETERS}
    arrays.update(
        {"optimizer_" + k: np.array(getattr(brain.learner, k), copy=True) for k in MOMENTS}
    )
    metadata = {
        "schema": "cadence-attention-neutral-variation-parameters/1",
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
        training = train(
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
                continued = train(
                    twin,
                    arm,
                    schedule,
                    min(
                        deadline,
                        time.monotonic() + spec["stages"][stage]["training_seconds_per_call"],
                    ),
                    offset=spec["task"]["rounds"],
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


def snapshot_check(descriptor, result, stage, revision, *, phase="final"):
    m = descriptor["metadata"]
    if (
        m["schema"] != "cadence-attention-neutral-variation-parameters/1"
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
        expected_prerequisites = {
            "completed_h1": old.digest,
            "negative_minimal_h2": spec["retained_negative"]["digest"],
        }
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
            training_data_check(
                r["training"],
                ctx.task.draw(r["seed"], spec["task"]["rounds"], spec["task"]["batch"]),
            )
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
                training_data_check(retained["training"], suffix, offset=spec["task"]["rounds"])
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
    previous, dependencies = (
        old.body["cumulative_host_seconds"],
        {"completed_h1": old.digest, "negative_minimal_h2": spec["retained_negative"]["digest"]},
    )
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
