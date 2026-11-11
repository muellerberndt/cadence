"""Bounded dual-binding acquisition and coupled-head development instrument.

The source-only query boundary deliberately clears the working trace and warm
activity while retaining the actually witnessed records. This is a component
test, not a continuing-life or learned semantic-code claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import cadence
from cadence import Brain
from cadence.learning import LearningPhaseError
from cadence.receipts import Receipt, canonical_sha256, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "heads-protocol.json"
PROTOCOL_NAMES = ("heads-protocol.json", "heads-longer-protocol.json",
                  "heads-fixed-read-protocol.json", "heads-anchored-protocol.json")


def protocol_path(name: str) -> Path:
    """Only declared, repository-owned recipes may select a run."""
    if name not in PROTOCOL_NAMES:
        raise ValueError("choose a declared heads protocol")
    return HERE / name


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


task = load("heads_dual_binding", HERE / "dual_binding.py")
heads = load("heads_instrument", HERE / "heads.py")
repair = load("heads_acquisition_repair", HERE / "repair_admission.py")
binding = load("heads_binding_forks", HERE.parent / "binding" / "token_binding.py")
sr = repair.sr
_fixed_module = None


class FixedReadBrain(heads.MultiHeadBrain):
    """Capacity-only fixed addresses with the candidate's read agreement term."""

    _fixed_read_capacity = True

    def stimulus(self, observations, *, memory=True):
        global _fixed_module
        drive = super().stimulus(observations, memory=memory)
        if self.brain.memory is not None:
            if _fixed_module is None:
                _fixed_module = load("heads_fixed_read_capacity", HERE / "fixed_read.py")
            self.brain.memory = _fixed_module.FrozenRead(self.brain.memory)
        return drive


def replace_members(memory, members):
    if getattr(memory, "fixed_selector", False):
        return type(memory)(replace(memory.bank, memories=tuple(members)))
    return replace(memory, memories=tuple(members))


def sources(protocol: Path | None = None) -> list[tuple[str, Path]]:
    active = PROTOCOL if protocol is None else protocol
    package = Path(cadence.__file__).resolve().parent
    names = ["heads_admission.py", active.name]
    if active.name != "heads-protocol.json":
        names.append("heads-protocol.json")
    if active.name == "heads-fixed-read-protocol.json":
        names += ["heads-longer-protocol.json", "fixed_read.py"]
    names += ["dual_binding.py", "heads.py", "coupled.py", "repair_admission.py",
              "repair-protocol.json"]
    paths = [("benchmarks/attention/" + name, active if name == active.name else HERE / name)
             for name in names]
    paths += [("benchmarks/binding/token_binding.py", Path(binding.__file__)),
              ("benchmarks/relevance/selective_recall.py", Path(sr.__file__)),
              ("benchmarks/relevance/acquisition_controls.py", Path(repair.controls.__file__))]
    return paths + [("src/cadence/" + str(p.relative_to(package)), p)
                    for p in sorted(package.rglob("*.py"))]


@contextmanager
def metered(work: dict[str, int]):
    """Count actual graph transports, including every residual check and refused phase.

    MAC counts describe the reference equations, not measured physical energy or
    total interpreter instructions. Softmax elements are counted separately.
    """
    original = heads.Graph._synaptic_input

    def transport(graph, state, blocks=None):
        work["cortical_synapse_transports"] += len(state) * graph.connectome.synapses
        memory = graph.memory
        if memory is not None:
            members = getattr(memory, "memories", (memory,))
            for member in members:
                batch, slots, key = member.keys.shape
                value = member.values.shape[-1]
                if getattr(memory, "fixed_selector", False):
                    work.setdefault("fixed_read_coordinate_updates", 0)
                    work["fixed_read_coordinate_updates"] += batch * value
                else:
                    work["memory_scan_macs"] += 2 * batch * slots * (key + value)
                    if member.mode != "linear":
                        work["softmax_elements"] += batch * (slots + 1)
                work["memory_drive_evaluations"] += 1
        return original(graph, state, blocks)

    heads.Graph._synaptic_input = transport
    try:
        yield
    finally:
        heads.Graph._synaptic_input = original


def boundary(brain: Brain, batch: int) -> None:
    if brain.working_memory is not None:
        brain.working_memory.reset(batch)
    brain.basal_ganglia.reset()


def make_brain(arm: str, seed: int, protocol: dict[str, Any], mode="agreement") -> Brain:
    options = dict(protocol["brain"])
    options["modules"] = tuple(options["modules"])
    options["slots"] = 2
    if arm == "exposed":
        options.pop("empty_anchor", None)  # Experimental boundary, not a plain Brain gene.
        options = {k: v for k, v in options.items() if not k.startswith("recall_")}
        brain = Brain.compose(task.INPUTS, task.VALUES * task.SLOTS, seed=seed, **options)
        original = brain.brain
        brain.learner.brain = heads.Graph(
            original.connectome, original.neuron_model.replace(dt=protocol["dt"]),
            efficacy=original.efficacy, bias=original.bias, log_gain=original.log_gain)
        return brain
    if arm == "h1_wide":
        options["modules"] = (64,)
    if arm == "h1_more_steps":
        options["learning_free_steps"] *= 2
        options["learning_nudged_steps"] *= 2
    options.update(recall_next=False)
    content_amplitude = options["recall_amplitude"]
    fixed_capacity = (arm == "supplied_reads"
                      and protocol.get("capacity_control") == "held-address-agreement")
    if arm == "supplied_reads" and not fixed_capacity:
        options["recall_amplitude"] = 0.0  # Disable coupling in the PREFIX as well as the query.
    brain_type = FixedReadBrain if fixed_capacity else heads.MultiHeadBrain
    brain = brain_type.compose(
        task.INPUTS, task.VALUES * task.SLOTS, seed=seed, dt=protocol["dt"], mode=mode,
        heads=2 if arm in ("h2", "supplied_reads") else 1,
        port_copies=2 if arm == "h1_wide_ports" else 1, **options)
    if arm == "supplied_reads" and not fixed_capacity:
        brain._supplied_content_amplitude = content_amplitude
    return brain


def prepare(brain: Brain, rows: np.ndarray, arm: str, condition="intact"):
    """Capture one drive/boundary before any actual query can append a slot."""
    _, query = task.events(rows, exposed=arm == "exposed")
    boundary(brain, len(rows))
    drive = np.array(brain.stimulus(query), copy=True)
    memory = brain.brain.memory
    if arm == "supplied_reads" and not getattr(brain, "_fixed_read_capacity", False):
        # An explicit capacity control: expose BOTH complete unselected facts in
        # separate read ports. No selector is taught or claimed by this arm.
        prefix, _ = task.events(rows)
        for h, member in enumerate(memory.memories):
            fact = np.where(rows[:, 2, None] == h, prefix[0], prefix[1])
            drive[:, member.read] = protocol_amplitude(brain) * fact
        memory = replace(memory, memories=tuple(replace(m, amplitude=0.0)
                                                for m in memory.memories))
    if condition == "erased" and memory is not None:
        memory = replace_members(memory, (
            replace(m, held=np.zeros_like(m.held)) for m in memory.memories))
    if condition.startswith("head_") and memory is not None:
        removed = int(condition.split("_")[1])
        memory = replace_members(memory, (
            replace(m, amplitude=0.0) if i == removed else m
            for i, m in enumerate(memory.memories)))
    brain.brain.memory = memory
    return query, drive, memory


def protocol_amplitude(brain: Brain) -> float:
    return float(getattr(brain, "_supplied_content_amplitude", brain.recall.amplitude))


def act_drive(brain: Brain, query: np.ndarray, drive: np.ndarray, work):
    work.actions += 1
    work.action_rows += len(query)
    previous = brain.last_settlement
    try:
        free = brain._qualified(drive, None, operation="act")
        answer = brain._choose(query, drive, free, greedy=True)
    except RuntimeError:
        report = brain.last_settlement
        if report is None or report is previous or report["qualified"]:
            raise
        work.action_refusals += 1
        answer = None
    report = brain.last_settlement
    for field, source in (("action_sweeps", "steps"),
                          ("action_residual_checks", "residual_checks")):
        setattr(work, field, getattr(work, field) + int(report[source]))
        row_field = field.replace("action_", "action_row_")
        setattr(work, row_field, getattr(work, row_field) + len(query) * int(report[source]))
    if np.isfinite(report["max_residual"]):
        work.max_action_residual = max(work.max_action_residual, float(report["max_residual"]))
    return answer


def teach(brain, drive, target, selected, memory, work):
    if not selected.any():
        return
    if memory is None:
        repair.controls.teach_drive(work, brain, drive[selected], target[selected])
        return
    work.lessons += 1
    try:
        _, report = brain.teach_saved(drive, target, selected, memory)
    except LearningPhaseError as error:
        work.lesson_refusals += 1
        report = error.report
    for field, source in (("presentations", "attempted_presentations"),
                          ("accepted_presentations", "accepted_presentations"),
                          ("teaching_sweeps", "total_steps"),
                          ("teaching_row_sweeps", "total_row_sweeps"),
                          ("teaching_residual_checks", "total_residual_checks"),
                          ("teaching_row_residual_checks", "total_row_residual_checks")):
        setattr(work, field, getattr(work, field) + int(report[source]))


def train(brain, arm, schedule, deadline):
    work, wrong_counts, stop = sr.Work(), [], None
    for rows in schedule:
        if time.monotonic() >= deadline:
            stop = "wall_budget"
            break
        brain.reset()
        prefix, _ = task.events(rows)
        for observation in prefix:
            if work.act(brain, observation) is None:
                stop = "refused_prefix"
                break
        if stop:
            break
        query, drive, memory = prepare(brain, rows, arm)
        answer = act_drive(brain, query, drive, work)
        if answer is None:
            stop = "refused_query"
            break
        target = task.targets(rows)
        selected = (np.asarray(answer) != target).any(axis=1)
        wrong_counts.append(int(selected.sum()))
        teach(brain, drive, target, selected, memory, work)
    return {"completed": len(wrong_counts) == len(schedule), "rounds": len(wrong_counts),
            "wrong_counts": wrong_counts, "stop": stop, "work": asdict(work)}


def evaluate(brain, arm, rows, condition="intact"):
    twin = binding.fork(brain)
    twin.reset()
    work = sr.Work()
    prefix, _ = task.events(rows)
    answers = np.full((len(rows), 2), -1, dtype=np.int64)
    for observation in prefix:
        if work.act(twin, observation) is None:
            return {"answers": answers.tolist(), "scores": task.scores(answers, rows),
                    "work": asdict(work), "association": None, "attention": None}
    query, drive, memory = prepare(twin, rows, arm, condition)
    answer = act_drive(twin, query, drive, work)
    activity = attention = None
    if answer is not None:
        answers[:] = answer
        state = np.atleast_2d(twin._activity().activation)
        activity = state[:, twin.connectome.populations["association"]].tolist()
        if memory is not None and memory.mode != "linear":
            attention = memory.weights(state).tolist()
    return {"answers": answers.tolist(), "scores": task.scores(answers, rows),
            "work": asdict(work), "association": activity, "attention": attention}


def acquired(result, floor):
    try:
        train, reading = result["training"], result["evaluation"]["train"]
        score = task.scores(reading["answers"], task.panel("train"))
        if score != reading["scores"]:
            return False
        values = [score["exact_pair_accuracy"], score["unequal_pair_accuracy"],
                  *score["per_slot_accuracy"], *score["per_query_accuracy"]]
        return bool(np.isfinite(values).all() and np.isfinite(floor)
                    and train["completed"] and result["parameters_preserved"]
                    and train["rounds"] == len(train["wrong_counts"]) > 0
                    and not train["work"]["action_refusals"]
                    and not train["work"]["lesson_refusals"]
                    and not reading["work"]["action_refusals"]
                    and not score["unanswered_slots"] and min(values) >= floor)
    except (KeyError, TypeError, ValueError):
        return False


def trial(arm, seed, protocol, deadline, mode="agreement"):
    began = time.monotonic()
    brain = make_brain(arm, seed, protocol, mode)
    initial = sr.parameter_digest(brain)
    cost = {k: 0 for k in ("cortical_synapse_transports", "memory_scan_macs",
                           "softmax_elements", "memory_drive_evaluations",
                           "fixed_read_coordinate_updates")}
    schedule = task.draw(seed, protocol["task"]["rounds"], protocol["task"]["batch"])
    with metered(cost):
        frozen = evaluate(brain, arm, task.panel("all"))
        training = train(brain, arm, schedule, min(
            deadline, time.monotonic() + protocol["budget"]["training_seconds_per_arm"]))
        final = sr.parameter_digest(brain)
        evaluation = {name: evaluate(brain, arm, task.panel(name))
                      for name in ("train", "held_out", "all")}
        if arm != "exposed":
            evaluation["erased"] = evaluate(brain, arm, task.panel("all"), "erased")
            if arm == "h2":
                for h in range(2):
                    evaluation[f"head_{h}"] = evaluate(brain, arm, task.panel("all"), f"head_{h}")
    random = np.random.default_rng(np.random.SeedSequence([622023, seed])).integers(
        task.VALUES, size=(len(task.CASES), 2))
    result = {
        "arm": arm, "mode": mode, "seed": seed, "training": training,
        "resolved": brain.describe(), "neuron_model": brain.brain.neuron_model.to_dict(),
        "initial": frozen, "evaluation": evaluation, "work": cost,
        "initial_parameters": initial, "final_parameters": final,
        "parameters_preserved": sr.parameter_digest(brain) == final,
        "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
        "schedule_sha256": canonical_sha256(schedule),
        "uniform_random": {"answers": random.tolist(), "scores": task.scores(random, task.CASES),
                           "expected_exact_pair_accuracy": 1 / task.VALUES ** 2},
        "elapsed_host_seconds": time.monotonic() - began,
    }
    floor = protocol["gates"]["exposed_acquisition_floor" if arm == "exposed"
                               else "content_use_floor"]
    result["acquired"] = acquired(result, floor)
    return result


def check(body, protocol=None):
    """Recompute admission from decisions, complete panels and the frozen protocol."""
    try:
        selector = body.get("protocol_file", "heads-protocol.json")
        if selector not in PROTOCOL_NAMES:
            return "undeclared heads protocol"
        if selector == "heads-anchored-protocol.json":
            return "anchored relevance protocol cannot certify capacity"
        if protocol is None:
            if selector != PROTOCOL.name:
                return "heads protocol selector differs"
            protocol = json.loads(PROTOCOL.read_text())
        if body["protocol"] != protocol:
            return "heads protocol differs"
        runs = body["runs"]
        seeds = protocol["development_seeds"]
        first = [("exposed", seed) for seed in seeds]
        if [(r["arm"], r["seed"]) for r in runs[:len(seeds)]] != first:
            return "exposed founders differ"
        exposed_passed = all(r["acquired"] for r in runs[:len(seeds)])
        expected = first + ([("supplied_reads", seed) for seed in seeds] if exposed_passed else [])
        if [(r["arm"], r["seed"]) for r in runs] != expected:
            return "missing, duplicate or unpermitted capacity arms"
        for result in runs:
            training = result["training"]
            rounds = training["rounds"]
            if (type(rounds) is not int or not 0 <= rounds <= protocol["task"]["rounds"]
                    or len(training["wrong_counts"]) != rounds
                    or any(type(n) is not int or not 0 <= n <= protocol["task"]["batch"]
                           for n in training["wrong_counts"])
                    or training["completed"] != (rounds == protocol["task"]["rounds"])):
                return "training rounds differ"
            for panel in ("train", "held_out", "all"):
                reading = result["evaluation"][panel]
                if task.scores(reading["answers"], task.panel(panel)) != reading["scores"]:
                    return "capacity scores differ"
            frozen = result["initial"]
            if task.scores(frozen["answers"], task.CASES) != frozen["scores"]:
                return "frozen scores differ"
            schedule = task.draw(result["seed"], protocol["task"]["rounds"],
                                 protocol["task"]["batch"])
            if result["schedule_sha256"] != canonical_sha256(schedule):
                return "capacity schedule differs"
            floor = protocol["gates"]["exposed_acquisition_floor" if result["arm"] == "exposed"
                                       else "content_use_floor"]
            if result["acquired"] != acquired(result, floor):
                return "capacity gate differs"
        capacity = len(runs) == 2 * len(seeds) and all(r["acquired"] for r in runs)
        decision = ("capacity_admitted; selection_and_causal_gates_pending" if capacity else
                    "supplied_reads_acquisition_failed" if exposed_passed else
                    "exposed_acquisition_failed")
        if body["capacity_admitted"] != capacity or body["decision"] != decision:
            return "capacity decision differs"
        if body["candidate_rounds_used"] != 0 or body["confirmation_seeds_used"]:
            return "capacity admission cannot open candidates or confirmation"
        return None
    except (KeyError, TypeError, ValueError):
        return "incomplete or invalid capacity receipt"


def verify_prior(prior: Path) -> Receipt:
    """A historical failure is checked against ITS protocol and source snapshots."""
    receipt = Receipt.read(prior)
    prior_path = protocol_path(receipt.body.get("protocol_file", "heads-protocol.json"))
    prior_protocol = json.loads(prior_path.read_text())
    expected = sources(prior_path)
    if [entry["path"] for entry in receipt.source["files"]] != [name for name, _ in expected]:
        raise ValueError("prior source manifest paths differ")
    frozen = []
    for entry, (name, current) in zip(receipt.source["files"], expected, strict=True):
        digest = entry["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("invalid prior source digest")
        snapshot = HERE / "results" / "sources" / (digest + ".txt")
        frozen.append((name, snapshot if snapshot.exists() else current))
    ok, message = Receipt.verify(
        prior, sources=frozen, check=lambda body: check(body, protocol=prior_protocol))
    if (not ok or receipt.kind != "cadence-attention-heads-admission/1"
            or receipt.body["capacity_admitted"]):
        raise ValueError("prior failed admission must verify: " + message)
    previous = [receipt.body["elapsed_host_seconds"],
                receipt.body.get("previous_host_seconds", 0.0)]
    if not np.isfinite(previous).all() or min(previous) < 0:
        raise ValueError("prior host work must be finite and nonnegative")
    return receipt


def run(prior: Path | None = None):
    if PROTOCOL.name == "heads-anchored-protocol.json":
        raise ValueError("use heads_relevance.py after verified empty-boundary capacity")
    protocol = json.loads(PROTOCOL.read_text())
    manifest = source_manifest(sources())
    began = time.monotonic()
    previous, reference = 0.0, None
    if PROTOCOL.name == "heads-longer-protocol.json" and prior is None:
        raise ValueError("the numerical retry requires its retained failed base admission")
    if PROTOCOL.name == "heads-fixed-read-protocol.json" and prior is None:
        raise ValueError("the anchored control revision requires its failed longer admission")
    if prior is not None:
        receipt = verify_prior(prior)
        if (PROTOCOL.name == "heads-longer-protocol.json"
                and receipt.body.get("protocol_file", "heads-protocol.json")
                != "heads-protocol.json"):
            raise ValueError("the one numerical retry must follow the base protocol")
        if (PROTOCOL.name == "heads-fixed-read-protocol.json"
                and receipt.body.get("protocol_file", "heads-protocol.json")
                != "heads-longer-protocol.json"):
            raise ValueError("the one anchored control revision must follow the longer protocol")
        previous = (receipt.body["elapsed_host_seconds"]
                    + receipt.body.get("previous_host_seconds", 0.0))
        reference = receipt.digest
    diagnostic = protocol.get("diagnostic_charge_seconds", 0.0)
    if not np.isfinite(diagnostic) or diagnostic < 0:
        raise ValueError("diagnostic host charge must be finite and nonnegative")
    previous += diagnostic
    deadline = began + max(0, protocol["budget"]["development_total_host_seconds"] - previous)
    runs = []
    stop = None
    for arm in ("exposed", "supplied_reads"):
        for seed in protocol["development_seeds"]:
            result = trial(arm, seed, protocol, deadline)
            runs.append(result)
            print(json.dumps({"arm": arm, "seed": seed, "acquired": result["acquired"],
                              "train": result["evaluation"]["train"]["scores"],
                              "rounds": result["training"]["rounds"]}), flush=True)
        if not all(r["acquired"] for r in runs if r["arm"] == arm):
            stop = f"{arm}_acquisition_failed"
            break
    # This first receipt admits capacity only. Selection experiments require the
    # acquired controls, then the declared causal runner; no optimistic pass flag.
    if manifest != source_manifest(sources()):
        raise RuntimeError("source changed during run; preserve diagnostics and rerun")
    body = {
        "protocol": protocol, "protocol_file": PROTOCOL.name,
        "runs": runs, "capacity_admitted": stop is None,
        "decision": stop or "capacity_admitted; selection_and_causal_gates_pending",
        "candidate_rounds_used": 0, "confirmation_seeds_used": [],
        "source_revision": sr.revision(),
        "elapsed_host_seconds": time.monotonic() - began,
        "previous_host_seconds": previous, "prior_reference": reference,
        "cadence_version": cadence.__version__, "numpy_version": np.__version__,
    }
    problem = check(body)
    if problem:
        raise AssertionError(problem)
    return Receipt.build("cadence-attention-heads-admission/1", body, sources())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", type=Path,
                        help="Retained failed admission charged to this budget")
    parser.add_argument("--protocol", choices=PROTOCOL_NAMES, default=PROTOCOL.name)
    args = parser.parse_args()
    PROTOCOL = protocol_path(args.protocol)
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt = run(args.prior)
    receipt.write(args.out)
    print(receipt.body["decision"])
