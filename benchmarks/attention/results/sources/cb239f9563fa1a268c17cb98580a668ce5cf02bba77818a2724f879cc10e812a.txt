"""Staged ordinary-query founder control, with its own matched prerequisites.

The old zero-query capacity remains historical evidence. It cannot admit this
initializer. Stage completion never establishes distinct useful heads or a
head-count advantage. This benchmark wrapper changes no production API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / "query-founder-protocol.json"
PARAMETERS = ("efficacy", "bias", "log_gain")
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
STAGES = ("capacity", "h2", "comparators")
_CONTEXT = None


def declared():
    spec = json.loads(PROTOCOL.read_text())
    if (spec["schema"] != "cadence-attention-query-founder-campaign/1"
            or spec["query_scale"] != 1 or spec["development_seeds"] != [0, 1]):
        raise ValueError("undeclared query founder point")
    return spec


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def context():
    global _CONTEXT
    if _CONTEXT is None:
        owner = load("query_campaign_frozen_owner", HERE / "heads_continuation.py")
        initializer = load("query_campaign_initializer", HERE / "query_founder.py")
        owner.recipe()
        _CONTEXT = SimpleNamespace(owner=owner, admission=owner.admission,
            relevance=owner.relevance, task=owner.task, sr=owner.sr, initializer=initializer,
            frozen_sources=owner.sources, frozen_snapshot=owner.relevance.parameter_snapshot)
    return _CONTEXT


def point_protocol(stage):
    spec = declared()
    path = HERE / spec["base_protocol_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["base_protocol_sha256"]:
        raise ValueError("frozen base profile bytes differ")
    profile = json.loads(path.read_text())
    profile["budget"]["training_seconds_per_arm"] = spec["stages"][stage][
        "training_seconds_per_call"]
    return profile


def ledger_paths():
    path = HERE / "results" / declared()["prior"]["file"]
    ledger = Receipt.read(path)
    return [(entry["path"], ROOT / entry["path"]) for entry in ledger.source["files"]]


def sources(stage):
    ctx, spec = context(), declared()
    paths = [("benchmarks/attention/query_founder_campaign.py", Path(__file__).resolve()),
             ("benchmarks/attention/query-founder-protocol.json", PROTOCOL),
             ("benchmarks/attention/query_founder.py", HERE / "query_founder.py"),
             ("benchmarks/attention/" + spec["head_assessment_file"],
              HERE / spec["head_assessment_file"]),
             *ctx.frozen_sources(),
             ("benchmarks/attention/results/" + spec["prior"]["file"],
              HERE / "results" / spec["prior"]["file"]), *ledger_paths()]
    for previous in STAGES[:STAGES.index(stage)]:
        name = spec["stages"][previous]["receipt_file"]
        paths.append(("benchmarks/attention/results/" + name, HERE / "results" / name))
    unique = {}
    for name, path in paths:
        if name in unique and path.resolve() != unique[name].resolve():
            raise ValueError("source path has conflicting dependencies")
        unique.setdefault(name, path)
    return list(unique.items())


def verify_prior():
    """Validate the actual q0 life and the complete JSON cost ledger before injection."""
    ctx, frozen = context(), declared()["prior"]
    path = HERE / "results" / frozen["file"]
    receipt = Receipt.read(path)
    if (receipt.kind != frozen["kind"] or receipt.digest != frozen["digest"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != frozen["archive_sha256"]):
        raise ValueError("query-founder prior identity differs")
    valid, message = Receipt.verify(path, sources=ledger_paths())
    if not valid:
        raise ValueError("query-founder prior sources differ: " + message)
    body = receipt.body
    parent = body["continuation"]
    parent_path = ROOT / parent["file"]
    valid, message = ctx.owner.verify(parent_path)
    previous = Receipt.read(parent_path)
    if (not valid or previous.digest != frozen["continuation_digest"]
            or parent["digest"] != previous.digest
            or hashlib.sha256(parent_path.read_bytes()).hexdigest() != parent["archive_sha256"]
            or previous.body["cumulative_host_seconds"] != parent["cumulative_host_seconds"]):
        raise ValueError("query-founder q0 baseline custody differs: " + message)
    cumulative, elapsed = parent["cumulative_host_seconds"], 0.
    for entry in body["diagnostics"]:
        journal_path = ROOT / entry["journal_source"]
        journal = json.loads(journal_path.read_text())
        files = {row["path"]: row["sha256"] for row in journal["source"]["files"]}
        phases = journal["phases"]
        qualified = all(all(phase["qualified"]) for phase in phases) if phases else None
        if (hashlib.sha256(journal_path.read_bytes()).hexdigest() != entry["journal_sha256"]
                or journal["status"] != entry["status"]
                or journal["elapsed_host_seconds"] != entry["elapsed_host_seconds"]
                or journal.get("error") != entry["error"]
                or files["diagnostic/diagnose.py"] != entry["helper_sha256"]
                or files["diagnostic/protocol.json"] != entry["protocol_sha256"]
                or len(phases) != entry["completed_phases"]
                or qualified != entry["all_completed_phases_qualified"]
                or entry["previous_host_seconds"] != cumulative):
            raise ValueError("prior diagnostic custody or charge differs")
        cumulative += entry["elapsed_host_seconds"]
        elapsed += entry["elapsed_host_seconds"]
        if cumulative != entry["cumulative_host_seconds"]:
            raise ValueError("prior diagnostic cumulative charge differs")
    if (len(body["diagnostics"]) != 3 or body["confirmation_seeds_used"]
            or body["new_candidate_rounds_used"] != 0
            or elapsed != body["elapsed_host_seconds"]
            or body["previous_host_seconds"] != parent["cumulative_host_seconds"]
            or cumulative != body["cumulative_host_seconds"]
            or cumulative != frozen["cumulative_host_seconds"]):
        raise ValueError("query-founder prior aggregate charge differs")
    return receipt


def prerequisites(stage):
    prior, spec = verify_prior(), declared()
    result = {"prior": prior}
    for previous in STAGES[:STAGES.index(stage)]:
        path = HERE / "results" / spec["stages"][previous]["receipt_file"]
        valid, message = verify(path)
        receipt = Receipt.read(path)
        required = "capacity_admitted" if previous == "capacity" else "h2_admitted"
        if (not valid or receipt.kind != spec["stages"][previous]["receipt_kind"]
                or receipt.body[required] is not True):
            raise ValueError("query-founder stage prerequisite: " + message)
        result[previous] = receipt
    return result


def snapshot(brain, directory, *, stage, arm, seed, mode):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}-{mode}-{arm}-{seed}-parameters.npz"
    arrays = {name: np.array(getattr(brain.brain, name), copy=True) for name in PARAMETERS}
    arrays.update({"optimizer_" + name: np.array(getattr(brain.learner, name), copy=True)
                   for name in MOMENTS})
    metadata = {"schema": "cadence-attention-research-parameters/1", "stage": stage,
        "arm": arm, "seed": seed, "mode": mode,
        "source": source_manifest(sources(stage)), "source_revision": context().sr.revision(),
        "query_founder": deepcopy(brain.query_initialization),
        "parameter_sha256": context().sr.parameter_digest(brain),
        "learning": brain.learner.config.to_dict(), "updates": brain.learner.updates,
        "contrast_updates": brain.learner.contrast_updates,
        "scope": "Research arrays, not a supported continuing checkpoint or admitted capability"}
    arrays["metadata"] = np.asarray(canonical_json(metadata))
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)
    return {"file": path.name, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": canonical_sha256({name: value.tolist()
                                                 for name, value in arrays.items()}),
            "metadata": metadata}


@contextmanager
def factory_scope(stage, captures=None):
    """Only this uniquely loaded owner is patched; every hook is restored in finally."""
    ctx = context()
    original = ctx.admission.make_brain
    old_snapshot = ctx.relevance.parameter_snapshot
    old_meter = ctx.admission.metered

    def make(arm, seed, profile, mode="agreement"):
        brain = original(arm, seed, profile, mode)
        ctx.initializer.initialize(brain, seed=seed, query_scale=declared()["query_scale"])
        if captures is not None:
            captures[(arm, seed)] = brain
        return brain

    def save(brain, directory, *, arm, seed, mode):
        return snapshot(brain, directory, stage=stage, arm=arm, seed=seed, mode=mode)

    @contextmanager
    def metered(work):
        if captures is not None:
            captures["actual_transport_work"] = work
        with old_meter(work):
            yield

    ctx.admission.make_brain, ctx.relevance.parameter_snapshot = make, save
    if captures is not None:
        ctx.admission.metered = metered
    try:
        yield
    finally:
        ctx.admission.make_brain, ctx.relevance.parameter_snapshot = original, old_snapshot
        ctx.admission.metered = old_meter


def finite(value):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float))
            and np.isfinite(value) and value >= 0)


def training_check(training, profile, *, rounds=None):
    count = training["rounds"]
    cap = profile["task"]["rounds"] if rounds is None else rounds
    if (type(count) is not int or not 0 <= count <= cap
            or training["completed"] != (count == cap)
            or len(training["wrong_counts"]) != count
            or any(type(value) is not int or not 0 <= value <= profile["task"]["batch"]
                   for value in training["wrong_counts"])):
        return "training completion or wrong counts differ"
    if training["stop"] not in ((None,) if count == cap else (
        "wall_budget", "refused_prefix", "refused_query"
    )):
        return "training stop differs"
    template = context().admission.asdict(context().sr.Work())
    work = training["work"]
    if work.keys() != template.keys() or any(
        not finite(value) or (name != "max_action_residual" and type(value) is not int)
        for name, value in work.items()
    ):
        return "invalid training work"
    return None


def reading_check(reading, rows):
    if ("rows" in reading and reading["rows"] != rows.tolist()) or (
        context().task.scores(reading["answers"], rows) != reading["scores"]
    ):
        return "reading panel or scores differ"
    template = context().admission.asdict(context().sr.Work())
    work = reading["work"]
    if work.keys() != template.keys() or any(
        not finite(value) or (name != "max_action_residual" and type(value) is not int)
        for name, value in work.items()
    ):
        return "invalid reading work"
    return None


def common_check(result, stage, profile, *, revision):
    ctx = context()
    if result.get("status") != "measured" or result["mode"] != "agreement":
        return "invalid measured cell status or equation"
    problem = training_check(result["training"], profile)
    if problem:
        return problem
    problem = ctx.admission.numerical_check(result, profile)
    if problem:
        return problem
    expected = ctx.admission.make_brain(result["arm"], result["seed"], profile, "agreement")
    record = result["query_initialization"]
    if (record != expected.query_initialization
            or result["initial_parameters"] != ctx.sr.parameter_digest(expected)
            or record["query_scale"] != declared()["query_scale"]):
        return "query founder or newborn parameters differ"
    if stage != "capacity" and result["initial_interface_sha256"] != canonical_sha256(
        expected.brain.efficacy
    ):
        return "newborn interface differs"
    schedule = ctx.task.draw(result["seed"], profile["task"]["rounds"], profile["task"]["batch"])
    if result["schedule_sha256"] != canonical_sha256(schedule):
        return "training schedule differs"
    if not {"train", "held_out", "all"} <= set(result["evaluation"]):
        return "missing intact panel"
    for name in ("train", "held_out", "all"):
        problem = reading_check(result["evaluation"][name], ctx.task.panel(name))
        if problem:
            return problem
    problem = reading_check(result["initial"], ctx.task.panel("all"))
    if problem:
        return problem
    if not finite(result["elapsed_host_seconds"]):
        return "invalid cell host charge"
    work_keys = {"cortical_synapse_transports", "memory_scan_macs", "softmax_elements",
                 "memory_drive_evaluations"}
    if stage == "capacity":
        work_keys.add("fixed_read_coordinate_updates")
    if set(result["work"]) != work_keys or any(
        type(value) is not int or value < 0 for value in result["work"].values()
    ):
        return "invalid actual transport work"
    descriptor = result["parameter_snapshot"]
    metadata = descriptor["metadata"]
    if (metadata["schema"] != "cadence-attention-research-parameters/1"
            or metadata["mode"] != "agreement" or metadata["source_revision"] != revision
            or metadata["source"] != source_manifest(sources(stage))
            or metadata["query_founder"] != record or metadata["arm"] != result["arm"]
            or metadata["seed"] != result["seed"] or metadata["stage"] != stage
            or metadata["parameter_sha256"] != result["final_parameters"]
            or metadata["learning"] != result["learning_config"]):
        return "research snapshot source or point differs"
    if (type(metadata["updates"]) is not int or type(metadata["contrast_updates"]) is not int
            or not 0 <= metadata["contrast_updates"] <= metadata["updates"]):
        return "research snapshot update counters differ"
    if stage == "capacity" and result["parameters_preserved"] != (
        result["final_parameters"] == result["evaluation_parameters"]
    ):
        return "evaluation parameter preservation differs"
    return None


def relevance_check(result, profile):
    ctx = context()
    required = {"intact", "erase_0", "erase_1", "neutral_sham", "reorder", "donor_swap",
                "frozen_interface", "shuffled_interface", "head_order", "head_0"}
    if result["arm"] == "h2":
        required |= {"head_1", "cloned_heads"}
    conditions = result["causal"]["conditions"]
    if set(conditions) != required:
        return "missing causal condition"
    for name, reading in conditions.items():
        rows = ctx.task.panel("all")
        if name == "donor_swap":
            rows = ctx.task.swapped(rows)
        problem = reading_check(reading, rows)
        if problem:
            return problem
    expected = ctx.relevance.metrics(result, profile)
    if result["metrics"] != expected:
        return "recomputed primary metrics differ"
    retention = result["retention"]
    passed = False
    if retention["status"] in ("completed", "capped"):
        problem = training_check(retention["training"], profile,
                                 rounds=profile["gates"]["retention_rounds"])
        if problem:
            return problem
        if set(retention["evaluation"]) != {"train", "held_out"}:
            return "missing retention panel"
        for name, reading in retention["evaluation"].items():
            problem = reading_check(reading, ctx.task.panel(name))
            if problem:
                return problem
        loss = expected["held_out_exact_pair_accuracy"] - retention["evaluation"]["held_out"][
            "scores"]["exact_pair_accuracy"]
        passed = ctx.relevance.retention_passes(retention, profile)
        if retention["loss"] != loss:
            return "retention loss differs"
    elif retention["status"] not in ("not_run_primary_failed", "not_run_budget"):
        return "unknown retention status"
    if retention["passed"] != passed or result["passed"] != (
        expected["primary_passed"] and passed
    ):
        return "H2 or retention optimism"
    probe = result["representation_probe"]
    if probe["status"] == "completed":
        if not result["passed"]:
            return "probe opened before behavioral gates"
        diagnostic = ctx.relevance.representation.analyze(
            **ctx.relevance.representation_inputs(result))
        if (not ctx.relevance.same_diagnostic(probe["result"], diagnostic)
                or probe["passed"] != diagnostic["passed"]):
            return "probe recomputation differs"
    elif (probe["status"] not in ("pending_behavioral_gates", "pending_budget", "refused_inputs")
          or probe["passed"]):
        return "invalid pending probe"
    return None


def complete_comparator(result):
    if result.get("status") != "measured":
        return False
    train = result["training"]
    return bool(train["completed"] and not train["work"]["action_refusals"]
                and not train["work"]["lesson_refusals"] and result["parameters_preserved"]
                and all(context().relevance.answered(result["evaluation"][name])
                        for name in ("train", "held_out", "all")))


def check(body):
    try:
        spec, ctx = declared(), context()
        stage = body["stage"]
        if stage not in STAGES or body["protocol"] != spec:
            return "query-founder stage or protocol differs"
        revision = body["source_revision"]
        if (not isinstance(revision, str) or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)):
            return "invalid recorded execution revision"
        prior = prerequisites(stage)  # Must run before changing any owner factory.
        expected_refs = {name: receipt.digest for name, receipt in prior.items()}
        previous = next(reversed(prior.values())).body["cumulative_host_seconds"]
        elapsed = body["elapsed_host_seconds"]
        if (body["prerequisites"] != expected_refs or body["query_scale"] != 1
                or body["previous_host_seconds"] != previous or not finite(elapsed)
                or body["cumulative_host_seconds"] != previous + elapsed
                or body["head_advantage_claimed"] is not False or body["confirmation_seeds_used"]
                or body["stone_age_opened"] is not False or body["candidate_points_used"] != 1):
            return "query-founder ownership, scope or cumulative charge differs"
        runs = body["runs"]
        if [[run["arm"], run["seed"]] for run in runs] != spec["stages"][stage]["order"]:
            return "missing, duplicate or reordered query-founder cells"
        profile = point_protocol(stage)
        with factory_scope(stage):
            for result in runs:
                if result.get("status") == "unrun_budget":
                    if (set(result) != {"arm", "seed", "mode", "status", "passed", "acquired"}
                            or result["mode"] != "agreement" or result["passed"] is not False
                            or result["acquired"] is not False
                            or elapsed < spec["stages"][stage]["total_host_seconds"]):
                        return "unrun cell optimism or false host cap"
                    continue
                problem = common_check(result, stage, profile, revision=revision)
                if problem:
                    return problem
                if stage == "capacity":
                    floor_name = ("exposed_acquisition_floor" if result["arm"] == "exposed"
                                  else "content_use_floor")
                    floor = profile["gates"][floor_name]
                    acquired = (ctx.admission.acquired(result, floor)
                                and ctx.admission.founder_evaluated(result))
                    if result["acquired"] != acquired:
                        return "recomputed capacity gate differs"
                else:
                    problem = relevance_check(result, profile)
                    if problem:
                        return problem
        if sum(result.get("elapsed_host_seconds", 0.) for result in runs) > elapsed:
            return "cell host work exceeds stage charge"
        if stage == "capacity":
            admitted = all(result.get("acquired", False) for result in runs)
            if body["capacity_admitted"] != admitted:
                return "capacity stage optimism"
        elif stage == "h2":
            admitted = all(result.get("passed", False) for result in runs)
            if (body["h2_admitted"] != admitted
                    or body["comparators_opened"] is not False
                    or body["resource_stage_permitted"] != admitted):
                return "H2 stage optimism or comparator work"
        elif body["comparison_measurements_complete"] != all(complete_comparator(r) for r in runs):
            return "comparator completeness differs"
    except (KeyError, TypeError, ValueError, OSError):
        return "incomplete or invalid query-founder receipt"
    return None


def verify(path):
    receipt = Receipt.read(path)
    kinds = {value["receipt_kind"]: name for name, value in declared()["stages"].items()}
    if receipt.kind not in kinds or receipt.body.get("stage") != kinds[receipt.kind]:
        return False, "not a matched query-founder stage receipt"
    stage = kinds[receipt.kind]
    return Receipt.verify(path, sources=sources(stage), check=check)


def run(stage, output, parameters, commit):
    began = time.monotonic()
    spec, ctx = declared(), context()
    if ctx.sr.revision() != commit:
        raise ValueError("actual execution commit differs from --commit")
    prior = prerequisites(stage)  # The old strict q0 verifier runs before injection.
    manifest = source_manifest(sources(stage))
    previous = next(reversed(prior.values())).body["cumulative_host_seconds"]
    deadline = began + spec["stages"][stage]["total_host_seconds"]
    profile = point_protocol(stage)
    runs, captures = [], {}
    journal = output.with_suffix(".partial.json")

    def flush(status, **extra):
        output.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(canonical_json({"status": status, "stage": stage,
            "protocol": spec, "source": manifest, "runs": runs,
            "elapsed_host_seconds": time.monotonic() - began, **extra}) + "\n")

    flush("started_unverified")
    with factory_scope(stage, captures):
        for arm, seed in spec["stages"][stage]["order"]:
            cell_began = time.monotonic()
            flush("cell_started_unverified", current={"arm": arm, "seed": seed})
            captures.pop("actual_transport_work", None)
            try:
                if time.monotonic() >= deadline:
                    result = {"arm": arm, "seed": seed, "mode": "agreement",
                              "status": "unrun_budget", "passed": False, "acquired": False}
                elif stage == "capacity":
                    result = ctx.admission.trial(arm, seed, profile, deadline, "agreement")
                    brain = captures[(arm, seed)]
                    result.update(status="measured",
                        query_initialization=deepcopy(brain.query_initialization),
                        parameter_snapshot=snapshot(brain, parameters, stage=stage, arm=arm,
                                                    seed=seed, mode="agreement"))
                else:
                    result = ctx.relevance.trial(arm, seed, "agreement", profile, deadline,
                                                 parameters=parameters)
                    brain = captures[(arm, seed)]
                    result.update(status="measured",
                                  query_initialization=deepcopy(brain.query_initialization))
            except Exception as error:  # noqa: BLE001 -- retain unexpected interrupted work.
                failure = {"current": {"arm": arm, "seed": seed},
                           "error": {"type": type(error).__name__, "message": str(error)},
                           "actual_transport_work": captures.get("actual_transport_work")}
                brain = captures.get((arm, seed))
                if brain is not None:
                    try:
                        failure["interrupted_parameters"] = snapshot(
                            brain, Path(parameters) / "interrupted", stage=stage,
                            arm=arm, seed=seed, mode="agreement")
                    except Exception as capture_error:  # noqa: BLE001 -- keep the original error.
                        failure["snapshot_error"] = {"type": type(capture_error).__name__,
                                                     "message": str(capture_error)}
                flush("cell_failed_unverified", **failure)
                raise
            if result["status"] == "measured":
                result["elapsed_host_seconds"] = time.monotonic() - cell_began
            runs.append(result)
            flush("running_unverified")
    if ctx.sr.revision() != commit:
        flush("revision_changed_unverified")
        raise ValueError("execution commit changed during the stage")
    elapsed = time.monotonic() - began
    body = {"stage": stage, "protocol": spec, "query_scale": 1, "runs": runs,
            "prerequisites": {name: receipt.digest for name, receipt in prior.items()},
            "previous_host_seconds": previous, "elapsed_host_seconds": elapsed,
            "cumulative_host_seconds": previous + elapsed, "source_revision": commit,
            "candidate_points_used": 1, "confirmation_seeds_used": [],
            "head_advantage_claimed": False, "stone_age_opened": False}
    if stage == "capacity":
        body["capacity_admitted"] = all(result.get("acquired", False) for result in runs)
    elif stage == "h2":
        body.update(h2_admitted=all(result.get("passed", False) for result in runs),
                    comparators_opened=False,
                    resource_stage_permitted=all(result.get("passed", False) for result in runs))
    else:
        body["comparison_measurements_complete"] = all(complete_comparator(result)
                                                       for result in runs)
    flush("complete_unverified", body=body)
    if source_manifest(sources(stage)) != manifest:
        raise ValueError("executed source or prerequisite changed; journal is unverified")
    problem = check(body)
    if problem:
        raise ValueError(problem)
    if ctx.sr.revision() != commit:
        raise ValueError("execution commit changed during verification")
    body["elapsed_host_seconds"] = time.monotonic() - began
    body["cumulative_host_seconds"] = previous + body["elapsed_host_seconds"]
    flush("verified_readings", body=body)
    return Receipt.build(spec["stages"][stage]["receipt_kind"], body, sources(stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        valid, message = verify(args.verify)
        print(message)
        raise SystemExit(0 if valid else 1)
    if not args.stage or args.out is None or args.parameters is None or not args.commit:
        parser.error("provide --stage, --out, --parameters and --commit")
    if args.out.exists() or args.out.with_suffix(".partial.json").exists():
        raise FileExistsError(args.out)
    run(args.stage, args.out, args.parameters, args.commit).write(args.out)
