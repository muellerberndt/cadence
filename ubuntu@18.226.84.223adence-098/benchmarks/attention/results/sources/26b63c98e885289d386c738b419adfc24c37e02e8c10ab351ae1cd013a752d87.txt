"""One matched empty-read-boundary repair, using the existing fixed-address control.

The masked arm must reproduce the retained refusal. The anchored arm changes only
empty-bank agreement. Hashes audit overlapping occupied-query phases without
extra neural solves; they are custody checks, not independent equation replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import types
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import cadence
from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "empty-boundary-protocol.json"
BASE = HERE / "heads-fixed-read-protocol.json"
REFERENCE = HERE / "results" / "heads-capacity-fixed-read.json"
SPEC = importlib.util.spec_from_file_location("empty_boundary_heads", HERE / "heads_admission.py")
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)
task, sr = admission.task, admission.sr


def sources() -> list[tuple[str, Path]]:
    return [("benchmarks/attention/empty_boundary_admission.py", Path(__file__).resolve()),
            ("benchmarks/attention/empty-boundary-protocol.json", PROTOCOL),
            ("benchmarks/attention/results/heads-capacity-fixed-read.json", REFERENCE),
            *admission.sources(BASE)]


def reference() -> Receipt:
    return admission.verify_prior(REFERENCE)


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return canonical_sha256({"shape": array.shape, "dtype": str(array.dtype),
                             "data": hashlib.sha256(array.tobytes()).hexdigest()})


def learner_hashes(brain: Any) -> dict[str, Any]:
    learner = brain.learner
    return {
        "parameters": canonical_sha256({name: array_hash(getattr(brain.brain, name))
                                        for name in ("efficacy", "bias", "log_gain")}),
        "optimizer": canonical_sha256({name: array_hash(getattr(learner, name)) for name in (
            "velocity", "velocity_bias", "second_moment", "second_moment_bias")}),
        "updates": int(learner.updates), "contrast_updates": int(learner.contrast_updates),
    }


def bank_hash(memory: Any) -> str:
    """Content/ports/force scale; empty_anchor is intentionally recorded separately.

    On an occupied query boundary both flag values define the same equations.
    """
    return canonical_sha256({
        "heads": [{name: array_hash(getattr(head, name))
                   for name in ("query", "read", "keys", "values", "held", "ages")}
                  | {"amplitude": float(head.amplitude)} for head in memory.memories],
        "targets": array_hash(memory.targets), "selector": array_hash(memory.p_fixed),
    })


def phase_record(phase: Any) -> dict[str, Any]:
    return {"state": canonical_sha256({name: array_hash(getattr(phase.state, name))
                                       for name in ("v", "activation", "adaptation")}),
            "steps": int(phase.steps), "residual_checks": int(phase.residual_checks),
            "residual": np.asarray(phase.residual).tolist(),
            "qualified": np.asarray(phase.qualified).tolist()}


@contextmanager
def capture(brain: Any, current: dict[str, Any]):
    """Observe actual accepted/refused phases before any memory rollback."""
    graph_type = admission.heads.Graph
    original_solve = graph_type.equilibrate
    original_phase = brain.learner._qualified_phase

    def solve(graph, drive, *args, **kwargs):
        phase = original_solve(graph, drive, *args, **kwargs)
        if not phase.qualified.all() and "failed_boundary" not in current:
            memory = graph.memory
            current["failed_boundary"] = {
                **phase_record(phase), "drive": array_hash(drive), "bank": bank_hash(memory),
                "held_any": memory.held.any(axis=1).tolist(),
                "prefix_index": current.get("prefix_index"),
                "parameters": learner_hashes(brain)["parameters"],
                "empty_anchor": memory.empty_anchor,
            }
        return phase

    def teacher_phase(self, drive, state, budget, nudge=None):
        phase = original_phase(drive, state, budget, nudge)
        current["phase_empty_anchor"].append(self.brain.memory.empty_anchor)
        current["phases"].append({
            **phase_record(phase), "drive": array_hash(drive),
            "bank": bank_hash(self.brain.memory),
            "parameters": learner_hashes(brain)["parameters"],
        })
        return phase

    graph_type.equilibrate = solve
    brain.learner._qualified_phase = types.MethodType(teacher_phase, brain.learner)
    try:
        yield
    finally:
        graph_type.equilibrate = original_solve
        del brain.learner._qualified_phase


def train(brain: Any, schedule: np.ndarray, deadline: float) -> tuple[dict[str, Any], list]:
    work, wrong_counts, trace = sr.Work(), [], []
    current: dict[str, Any] = {}
    stop = None
    with capture(brain, current):
        for index, rows in enumerate(schedule):
            if time.monotonic() >= deadline:
                stop = "wall_budget"
                break
            current.clear()
            current.update(round=index + 1, before=learner_hashes(brain), prefix=[], phases=[],
                           query=None, query_drive=None, bank=None, held_any=None,
                           answer=None, selected=[], teacher_drive=None, teacher_bank=None,
                           memory_empty_anchor=None, phase_empty_anchor=[])
            brain.reset()
            prefix, _ = task.events(rows)
            for position, observation in enumerate(prefix):
                current["prefix_index"] = position
                answer = work.act(brain, observation)
                current["prefix"].append(dict(brain.last_settlement))
                if answer is None:
                    stop = "refused_prefix"
                    break
            if not stop:
                query, drive, memory = admission.prepare(brain, rows, "supplied_reads")
                current.update(prefix_index=None, query_drive=array_hash(drive),
                               bank=bank_hash(memory), held_any=memory.held.any(axis=1).tolist(),
                               memory_empty_anchor=memory.empty_anchor)
                answer = admission.act_drive(brain, query, drive, work)
                current["query"] = dict(brain.last_settlement)
                if answer is None:
                    stop = "refused_query"
                else:
                    current["answer"] = np.asarray(answer).tolist()
                    selected = (np.asarray(answer) != task.targets(rows)).any(axis=1)
                    current["selected"] = np.flatnonzero(selected).tolist()
                    wrong_counts.append(int(selected.sum()))
                    if selected.any():
                        current.update(teacher_drive=array_hash(drive[selected]),
                                       teacher_bank=bank_hash(memory.take(selected)))
                    admission.teach(brain, drive, task.targets(rows), selected, memory, work)
            current.update(after=learner_hashes(brain), stop=stop)
            trace.append(deepcopy(current))
            if stop:
                break
    return {"completed": len(wrong_counts) == len(schedule), "rounds": len(wrong_counts),
            "wrong_counts": wrong_counts, "stop": stop, "work": asdict(work)}, trace


def reading(brain: Any, panel: str, deadline: float) -> dict[str, Any]:
    rows = task.panel(panel)
    if time.monotonic() >= deadline:
        answers = np.full((len(rows), 2), -1, dtype=np.int64)
        result = {"answers": answers.tolist(), "scores": task.scores(answers, rows),
                  "work": asdict(sr.Work()), "association": None, "attention": None}
    else:
        result = admission.evaluate(brain, "supplied_reads", rows)
    result["completed"] = (not result["scores"]["unanswered_slots"]
                           and not result["work"]["action_refusals"])
    return result


def admitted(result: dict[str, Any], declared: dict[str, Any]) -> bool:
    try:
        training, evaluation = result["training"], result["evaluation"]["train"]
        scores = task.scores(evaluation["answers"], task.panel("train"))
        values = [scores["exact_pair_accuracy"], scores["unequal_pair_accuracy"],
                  *scores["per_slot_accuracy"], *scores["per_query_accuracy"]]
        return bool(training["completed"] is True and training["rounds"] == declared["rounds"]
                    and not training["work"]["action_refusals"]
                    and not training["work"]["lesson_refusals"]
                    and result["parameters_preserved"] is True and evaluation["completed"] is True
                    and not evaluation["work"]["action_refusals"]
                    and not scores["unanswered_slots"]
                    and min(values) >= declared["gate"]["trained_floor"])
    except (KeyError, TypeError, ValueError):
        return False


def trial(variant: dict[str, Any], seed: int, declared: dict[str, Any],
          recipe: dict[str, Any], deadline: float) -> dict[str, Any]:
    began = time.monotonic()
    options = deepcopy(recipe)
    options["brain"]["empty_anchor"] = variant["empty_anchor"]
    brain = admission.make_brain("supplied_reads", seed, options)
    initial_parameters, initial_state = sr.parameter_digest(brain), learner_hashes(brain)
    schedule = task.draw(seed, declared["rounds"], declared["batch"])
    work = {name: 0 for name in ("cortical_synapse_transports", "memory_scan_macs",
                                "softmax_elements", "memory_drive_evaluations",
                                "fixed_read_coordinate_updates")}
    with admission.metered(work):
        initial = reading(brain, "all", deadline)
        training, trace = train(brain, schedule, min(
            deadline, time.monotonic() + declared["budget"]["training_host_seconds_per_arm"]))
        final_parameters, final_state = sr.parameter_digest(brain), learner_hashes(brain)
        evaluation = {panel: reading(brain, panel, deadline) for panel in ("train", "held_out")}
    evaluation_parameters, evaluation_state = sr.parameter_digest(brain), learner_hashes(brain)
    result = {
        "variant": variant["name"], "empty_anchor": variant["empty_anchor"], "seed": seed,
        "training": training, "trace": trace, "initial": initial, "evaluation": evaluation,
        "initial_parameters": initial_parameters, "final_parameters": final_parameters,
        "initial_state": initial_state, "final_state": final_state,
        "evaluation_parameters": evaluation_parameters, "evaluation_state": evaluation_state,
        "parameters_preserved": (evaluation_parameters == final_parameters
                                 and evaluation_state == final_state),
        "schedule_sha256": canonical_sha256(schedule), "work": work,
        "neuron_model": brain.brain.neuron_model.to_dict(), "resolved": brain.describe(),
        "learning_config": brain.learner.config.to_dict(),
        "elapsed_host_seconds": time.monotonic() - began,
    }
    result["admitted"] = admitted(result, declared)
    return result


def matched(masked: dict[str, Any], anchored: dict[str, Any]) -> bool:
    """Require exact occupied-query identity; prefix activities are not compared."""
    if (masked["initial_parameters"] != anchored["initial_parameters"]
            or masked["initial_state"] != anchored["initial_state"]):
        return False
    for index, old in enumerate(masked["trace"]):
        if index >= len(anchored["trace"]):
            return False
        new = anchored["trace"][index]
        if old["before"] != new["before"]:
            return False
        if old["answer"] is None:
            continue
        if new["answer"] is None:
            return False
        if any(old[key] != new[key] for key in (
                "before", "after", "query_drive", "bank", "held_any", "query", "answer",
                "selected", "teacher_drive", "teacher_bank", "phases")):
            return False
    return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _hash(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _phase(phase: dict[str, Any], rows: int, tolerance: float, budget: int) -> None:
    _require(isinstance(phase["residual"], (list, tuple))
             and all(type(x) in (int, float) for x in phase["residual"]),
             "invalid residual scalar kinds")
    residual = np.asarray(phase["residual"], dtype=float)
    flags = phase["qualified"] if isinstance(phase["qualified"], list) else phase["row_qualified"]
    _require(isinstance(flags, (list, tuple)), "missing phase row flags")
    flags = list(flags)
    _require(residual.shape == (rows,) and np.isfinite(residual).all()
             and (residual >= 0).all(), "phase residuals differ")
    _require(all(type(x) is bool for x in flags)
             and flags == (residual <= tolerance).tolist(), "phase qualification differs")
    _require(_int(phase["steps"]) and phase["steps"] <= budget
             and _int(phase["residual_checks"]), "phase work differs")
    if type(phase["qualified"]) is bool:
        _require(phase["qualified"] == all(flags)
                 and phase["max_residual"] == float(residual.max())
                 and phase["tolerance"] == tolerance and phase["budget"] == budget,
                 "action qualification differs")
    else:
        _require(_hash(phase["state"]), "phase state hash differs")


def trace_work(trace: list[dict[str, Any]], batch: int) -> dict[str, Any]:
    """Derive all original Work counters from actual action/teacher phase records."""
    work = asdict(sr.Work())
    for row in trace:
        actions = row["prefix"] + ([] if row["query"] is None else [row["query"]])
        for action in actions:
            work["actions"] += 1
            work["action_rows"] += batch
            work["action_refusals"] += int(not action["qualified"])
            for target, source in (("action_sweeps", "steps"),
                                   ("action_residual_checks", "residual_checks")):
                work[target] += action[source]
                work[target.replace("action_", "action_row_")] += batch * action[source]
            work["max_action_residual"] = max(work["max_action_residual"],
                                               max(action["residual"]))
        if row["answer"] is not None and row["selected"]:
            count = len(row["selected"])
            work["lessons"] += 1
            work["presentations"] += count
            accepted = len(row["phases"]) == 3 and all(
                all(p["qualified"]) for p in row["phases"])
            work["accepted_presentations"] += count * int(accepted)
            work["lesson_refusals"] += int(not accepted)
            for phase in row["phases"]:
                work["teaching_sweeps"] += phase["steps"]
                work["teaching_row_sweeps"] += count * phase["steps"]
                work["teaching_residual_checks"] += phase["residual_checks"]
                work["teaching_row_residual_checks"] += count * phase["residual_checks"]
    return work


def _reading(reading: dict[str, Any], panel: str, tolerance: float, budget: int) -> None:
    rows = task.panel(panel)
    scores = task.scores(reading["answers"], rows)
    _require(scores == reading["scores"], "evaluation scores differ")
    work = reading["work"]
    _require(all(_int(value) for key, value in work.items() if key != "max_action_residual")
             and work["actions"] <= 4 and work["action_refusals"] <= 1
             and work["action_rows"] == len(rows) * work["actions"]
             and work["action_row_sweeps"] == len(rows) * work["action_sweeps"]
             and work["action_row_residual_checks"] == len(rows) * work["action_residual_checks"]
             and work["action_sweeps"] <= budget * work["actions"], "evaluation work differs")
    maximum = work["max_action_residual"]
    _require(type(maximum) in (int, float) and math.isfinite(maximum) and maximum >= 0,
             "invalid evaluation residual")
    complete = scores["unanswered_slots"] == 0 and work["action_refusals"] == 0
    _require(type(reading["completed"]) is bool and reading["completed"] == complete,
             "evaluation completion differs")
    _require((complete and work["actions"] == 4 and work["max_action_residual"] <= tolerance)
             or (not complete and scores["unanswered_slots"] == len(rows) * 2
                 and (work["action_refusals"] == 1 or work["actions"] == 0)),
             "missing or refused evaluation differs")
    _require(not any(work[k] for k in ("lessons", "lesson_refusals", "presentations",
                                      "accepted_presentations", "teaching_sweeps")),
             "evaluation learned")


def transport_work(result: dict[str, Any]) -> dict[str, int]:
    """Every sweep/check performs one cortical and two fixed-read transports."""
    works = [result["training"]["work"], result["initial"]["work"],
             *(reading["work"] for reading in result["evaluation"].values())]
    calls = sum(w["action_sweeps"] + w["action_residual_checks"]
                + w["teaching_sweeps"] + w["teaching_residual_checks"] for w in works)
    rows = sum(w["action_row_sweeps"] + w["action_row_residual_checks"]
               + w["teaching_row_sweeps"] + w["teaching_row_residual_checks"] for w in works)
    return {"cortical_synapse_transports": rows * result["resolved"]["layout"]["synapses"],
            "memory_scan_macs": 0, "softmax_elements": 0,
            "memory_drive_evaluations": calls * 2, "fixed_read_coordinate_updates": rows * 38}


def check(body: dict[str, Any]) -> str | None:
    """Recompute completion, historical parity, overlap and acquisition from raw records."""
    try:
        declared = json.loads(PROTOCOL.read_text())
        _require(body["protocol_file"] == PROTOCOL.name and body["protocol"] == declared,
                 "empty-boundary protocol differs")
        prior = reference()
        _require(body["reference_receipt_digest"] == prior.digest,
                 "historical reference differs")
        runs = body["runs"]
        expected = [(v["name"], seed) for v in declared["variants"]
                    for seed in declared["development_seeds"]]
        _require([(r["variant"], r["seed"]) for r in runs] == expected,
                 "missing, duplicate or unpermitted variants/founders")
        for result in runs:
            variant = next(v for v in declared["variants"] if v["name"] == result["variant"])
            _require(type(result["empty_anchor"]) is bool
                     and result["empty_anchor"] == variant["empty_anchor"], "wrong boundary flag")
            historical_run = next(p for p in prior.body["runs"]
                                  if p["arm"] == "supplied_reads" and p["seed"] == result["seed"])
            _require(canonical_sha256(result["neuron_model"])
                     == canonical_sha256(historical_run["neuron_model"])
                     and canonical_sha256(result["resolved"]["genes"])
                     == canonical_sha256(historical_run["resolved"]["genes"])
                     and canonical_sha256(result["resolved"]["layout"])
                     == canonical_sha256(historical_run["resolved"]["layout"])
                     and canonical_sha256(result["learning_config"])
                     == canonical_sha256(declared["learning_config"])
                     and _hash(result["initial_parameters"])
                     and _hash(result["final_parameters"]), "resolved model or parameters differ")
            schedule = task.draw(result["seed"], declared["rounds"], declared["batch"])
            _require(result["schedule_sha256"] == canonical_sha256(schedule), "schedule differs")
            trace, training = result["trace"], result["training"]
            wrong = []
            previous = result["initial_state"]
            for index, row in enumerate(trace):
                _require(index < len(schedule) and row["round"] == index + 1
                         and row["before"] == previous, "state continuity differs")
                for state in (row["before"], row["after"]):
                    _require(_hash(state["parameters"]) and _hash(state["optimizer"])
                             and _int(state["updates"]) and _int(state["contrast_updates"]),
                             "invalid learned-state hash")
                prefix = row["prefix"]
                _require(1 <= len(prefix) <= 3, "prefix actions differ")
                for action in prefix:
                    _phase(action, declared["batch"], declared["tolerance"],
                           declared["free_steps"])
                if row["answer"] is not None:
                    _require(len(prefix) == 3 and all(p["qualified"] for p in prefix)
                             and row["query"] is not None and row["query"]["qualified"] is True
                             and _hash(row["query_drive"]) and _hash(row["bank"])
                             and row["held_any"] == [True] * declared["batch"]
                             and row["memory_empty_anchor"] is result["empty_anchor"],
                             "query boundary is not occupied and qualified")
                    _phase(row["query"], declared["batch"], declared["tolerance"],
                           declared["free_steps"])
                    answer = np.asarray(row["answer"])
                    _require(answer.shape == (declared["batch"], 2)
                             and answer.dtype.kind in "iu" and np.isin(answer, range(4)).all(),
                             "invalid actual query decisions")
                    selected = np.flatnonzero((answer != task.targets(schedule[index])).any(axis=1))
                    _require(row["selected"] == selected.tolist(), "wrong-row selection differs")
                    wrong.append(len(selected))
                    phases = row["phases"]
                    _require(len(phases) <= 3 and (bool(phases) == bool(len(selected))),
                             "teacher phase count differs")
                    _require(len(row["phase_empty_anchor"]) == len(phases)
                             and all(flag is result["empty_anchor"]
                                     for flag in row["phase_empty_anchor"]),
                             "actual teacher empty-bank setting differs")
                    for phase in phases:
                        _phase(phase, len(selected), declared["tolerance"],
                               declared["nudged_steps"])
                        _require(_hash(phase["drive"]) and _hash(phase["bank"])
                                 and phase["drive"] == row["teacher_drive"]
                                 and phase["bank"] == row["teacher_bank"]
                                 and phase["parameters"] == row["before"]["parameters"],
                                 "teacher boundary or parameter hash differs")
                    accepted = bool(len(selected) and len(phases) == 3
                                    and all(all(p["qualified"]) for p in phases))
                    for counter in ("updates", "contrast_updates"):
                        _require(row["after"][counter] == row["before"][counter] + int(accepted),
                                 "optimizer update count differs")
                    if not accepted:
                        _require(row["after"] == row["before"], "untaught/refused state changed")
                    _require(row["stop"] is None, "answered query marked stopped")
                else:
                    _require(index == len(trace) - 1 and row["stop"] in (
                        "refused_prefix", "refused_query") and not row["phases"]
                        and row["after"] == row["before"], "refused boundary changed state")
                    failure = row["failed_boundary"]
                    _phase(failure, declared["batch"], declared["tolerance"],
                           declared["free_steps"])
                    _require(not all(failure["qualified"]) and _hash(failure["drive"])
                             and _hash(failure["bank"])
                             and failure["parameters"] == row["before"]["parameters"]
                             and failure["empty_anchor"] is result["empty_anchor"]
                             and len(failure["held_any"]) == declared["batch"]
                             and all(type(x) is bool for x in failure["held_any"]),
                             "failed boundary custody differs")
                    if row["stop"] == "refused_prefix":
                        _require(prefix[-1]["qualified"] is False
                                 and all(p["qualified"] for p in prefix[:-1])
                                 and row["query"] is None
                                 and failure["prefix_index"] == len(prefix) - 1,
                                 "prefix refusal differs")
                        if len(prefix) == 1:
                            _require(failure["held_any"] == [False] * declared["batch"],
                                     "first-prefix bank is not empty")
                    else:
                        _require(len(prefix) == 3 and all(p["qualified"] for p in prefix)
                                 and row["query"]["qualified"] is False,
                                 "query refusal differs")
                        _phase(row["query"], declared["batch"], declared["tolerance"],
                               declared["free_steps"])
                    refused = prefix[-1] if row["stop"] == "refused_prefix" else row["query"]
                    _require(failure["steps"] == refused["steps"]
                             and failure["residual_checks"] == refused["residual_checks"]
                             and failure["residual"] == list(refused["residual"])
                             and failure["qualified"] == list(refused["row_qualified"]),
                             "failed phase differs from the refused action")
                previous = row["after"]
            complete = len(wrong) == declared["rounds"]
            _require(training["rounds"] == len(wrong) and training["wrong_counts"] == wrong
                     and type(training["completed"]) is bool and training["completed"] == complete
                     and training["work"] == trace_work(trace, declared["batch"])
                     and previous == result["final_state"], "training work/completion differs")
            _require(training["stop"] == (trace[-1]["stop"] if trace and trace[-1]["stop"]
                                          else None if complete else "wall_budget"),
                     "training stop differs")
            _reading(result["initial"], "all", declared["tolerance"], declared["free_steps"])
            _require(set(result["evaluation"]) == {"train", "held_out"},
                     "unexpected evaluation panel")
            for panel in ("train", "held_out"):
                _reading(result["evaluation"][panel], panel, declared["tolerance"],
                         declared["free_steps"])
            preserved = (result["evaluation_parameters"] == result["final_parameters"]
                         and result["evaluation_state"] == result["final_state"])
            _require(type(result["parameters_preserved"]) is bool
                     and result["parameters_preserved"] == preserved
                     and type(result["admitted"]) is bool
                     and result["admitted"] == admitted(result, declared),
                     "preservation/admission differs")
            _require(result["work"] == transport_work(result), "transport work differs")
        historical = all(
            r["training"] == next(p for p in prior.body["runs"] if p["arm"] == "supplied_reads"
                                  and p["seed"] == r["seed"])["training"]
            and r["final_parameters"] == next(
                p for p in prior.body["runs"] if p["arm"] == "supplied_reads"
                and p["seed"] == r["seed"])["final_parameters"]
            for r in runs if r["variant"] == "masked")
        overlap = all(matched(runs[i], runs[i + len(declared["development_seeds"])])
                      for i in range(len(declared["development_seeds"])))
        passed = historical and overlap and all(r["admitted"] for r in runs
                                               if r["variant"] == "anchored")
        _require(body["masked_historical_reproduced"] is historical
                 and body["occupied_query_overlap_identical"] is overlap
                 and body["passed"] is passed
                 and body["decision"] == ("empty_boundary_acquisition_admitted; selection_pending"
                                          if passed else "empty_boundary_repair_failed"),
                 "aggregate decision differs")
        _require(body["candidate_rounds_used"] == 0 and body["confirmation_seeds_used"] == [],
                 "this admission opened candidates or confirmation")
        elapsed = body["elapsed_host_seconds"]
        _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
                 and body["previous_capacity_host_seconds"]
                 == declared["budget"]["old_capacity_phase_host_seconds"]
                 and body["previous_diagnostic_host_seconds"]
                 == declared["budget"]["preceding_separate_diagnostic_host_seconds"]
                 and body["cumulative_host_seconds"] == body["previous_capacity_host_seconds"]
                 + body["previous_diagnostic_host_seconds"] + elapsed, "host charges differ")
        return None
    except (KeyError, TypeError, ValueError, OSError, StopIteration, IndexError):
        return "invalid empty-boundary admission: decisions, phase custody, work or charges differ"


def journal_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(data) + "\n")
    temporary.replace(path)


def run(journal: Path | None = None) -> Receipt:
    declared, recipe = json.loads(PROTOCOL.read_text()), json.loads(BASE.read_text())
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != declared["base_protocol_sha256"]:
        raise ValueError("fixed-read base protocol differs")
    prior = reference()
    manifest = source_manifest(sources())
    if journal is None:
        journal = HERE / "results" / "empty-boundary-admission.partial.json"
    if journal.exists():
        raise FileExistsError("retain the previous campaign journal")
    began = time.monotonic()
    deadline = began + declared["budget"]["new_phase_total_host_seconds"]
    runs = []
    journal_write(journal, {"status": "running", "protocol": declared, "source": manifest,
                            "source_revision": sr.revision(), "runs": runs})
    for variant in declared["variants"]:
        for seed in declared["development_seeds"]:
            result = trial(variant, seed, declared, recipe, deadline)
            runs.append(result)
            journal_write(journal, {"status": "running", "protocol": declared,
                                    "source": manifest, "source_revision": sr.revision(),
                                    "elapsed_host_seconds": time.monotonic() - began,
                                    "runs": runs})
            print(json.dumps({"variant": variant["name"], "seed": seed,
                              "rounds": result["training"]["rounds"],
                              "stop": result["training"]["stop"],
                              "train": result["evaluation"]["train"]["scores"],
                              "admitted": result["admitted"]}), flush=True)
    historical = all(
        r["training"] == next(p for p in prior.body["runs"] if p["arm"] == "supplied_reads"
                              and p["seed"] == r["seed"])["training"]
        and r["final_parameters"] == next(p for p in prior.body["runs"]
                                          if p["arm"] == "supplied_reads"
                                          and p["seed"] == r["seed"])["final_parameters"]
        for r in runs if r["variant"] == "masked")
    count = len(declared["development_seeds"])
    overlap = all(matched(runs[i], runs[i + count]) for i in range(count))
    passed = historical and overlap and all(r["admitted"] for r in runs
                                           if r["variant"] == "anchored")
    elapsed = time.monotonic() - began
    body = {
        "protocol_file": PROTOCOL.name, "protocol": declared, "runs": runs,
        "reference_receipt_digest": prior.digest, "masked_historical_reproduced": historical,
        "occupied_query_overlap_identical": overlap, "passed": passed,
        "decision": ("empty_boundary_acquisition_admitted; selection_pending" if passed
                     else "empty_boundary_repair_failed"),
        "candidate_rounds_used": 0, "confirmation_seeds_used": [],
        "elapsed_host_seconds": elapsed,
        "previous_capacity_host_seconds": declared["budget"]["old_capacity_phase_host_seconds"],
        "previous_diagnostic_host_seconds": declared["budget"][
            "preceding_separate_diagnostic_host_seconds"],
        "cumulative_host_seconds": declared["budget"]["old_capacity_phase_host_seconds"]
        + declared["budget"]["preceding_separate_diagnostic_host_seconds"] + elapsed,
        "source_revision": sr.revision(), "cadence_version": cadence.__version__,
        "numpy_version": np.__version__,
    }
    journal_write(journal, {"status": "unverified", "body": body, "source": manifest})
    if manifest != source_manifest(sources()):
        raise RuntimeError("sources changed during the campaign; retain evidence")
    problem = check(body)
    if problem:
        journal_write(journal, {"status": "check_failed", "check_error": problem,
                                "body": body, "source": manifest})
        raise AssertionError(problem)
    receipt = Receipt.build("cadence-attention-empty-boundary-admission/1", body, sources())
    journal_write(journal, {"status": "verified", "receipt": receipt.to_dict()})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; retain the earlier campaign")
    receipt = run(journal=args.out.with_suffix(args.out.suffix + ".partial.json"))
    receipt.write(args.out)
    print(receipt.body["decision"])
