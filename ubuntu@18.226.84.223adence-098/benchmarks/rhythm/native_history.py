"""Native sensory-history configuration for the bounded period-four loop chamber.

Use --screen --out NEW_DIRECTORY for seeds 0/1, or --full for development and
conditionally admitted confirmation. --verify DIRECTORY checks archived sources,
raw score arithmetic, gates and checkpoint hashes. No runtime/default change or
general repair of issue 140 is claimed. Teaching is an explicit label policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import cadence as cd
from cadence.receipts import Receipt, source_manifest

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol-native-history.json"
SCHEMA = "native-history/1"
spec = importlib.util.spec_from_file_location("native_history_loop", HERE / "loop_rhythm.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def initial_arrays(brain):
    values = {"graph/" + k: getattr(brain.connectome, k) for k in ("pre", "post", "count", "sign")}
    values.update({"brain/" + k: getattr(brain.brain, k) for k in ("efficacy", "bias")})
    values.update(
        {
            "learner/" + k: getattr(brain.learner, k)
            for k in (
                "plastic_synapses",
                "plastic_neurons",
                "synapse_rate",
                "tie_groups",
                "velocity",
                "velocity_bias",
                "second_moment",
                "second_moment_bias",
            )
        }
    )
    values["critic"] = brain.basal_ganglia.w_critic
    return {k: v for k, v in values.items() if v is not None}


def make_brain(seed, source="sensory", protocol=None):
    """Configure public Trace once; its update remains owned by Brain.act."""
    if source not in ("association", "sensory"):
        raise ValueError("source must be association or sensory")
    protocol = load_protocol() if protocol is None else protocol
    original = loop.make_brain(seed, protocol, "nocopy")
    if source == "association":
        return original
    populations = dict(original.connectome.populations)
    target = protocol["history"]["target"]
    populations[target] = populations["prefrontal"][: protocol["history"]["width"]]
    connectome = replace(original.connectome, populations=populations)
    brain = cd.Brain(
        connectome,
        episodic=False,
        seed=seed,
        learning=original.learner.config,
        reward=original.basal_ganglia.config,
        working_memory_amplitude=protocol["recipe"]["trace_amplitude"],
        working_memory_decay=protocol["recipe"]["trace_decay"],
        slots=original.learner.slot_sizes.tolist(),
    )
    # the initial biases of the recipe's resting bias
    brain.learner.brain = brain.learner.brain.with_parameters(
        bias=np.asarray(original.brain.bias, dtype=float).copy()
    )
    for key, value in initial_arrays(original).items():
        np.testing.assert_array_equal(value, initial_arrays(brain)[key])
    assert original.learner.config == brain.learner.config
    assert original.basal_ganglia.config == brain.basal_ganglia.config
    assert original.basal_ganglia.b_critic == brain.basal_ganglia.b_critic
    assert original.brain.neuron_model == brain.brain.neuron_model
    brain.working_memory = cd.Trace(
        connectome,
        source="sensory",
        target=target,
        amplitude=protocol["recipe"]["trace_amplitude"],
        decay=protocol["recipe"]["trace_decay"],
    )
    return brain


class Checkpoints:
    def __init__(self):
        self.work = {"writes": 0, "reads": 0, "bytes": 0, "seconds": 0.0}

    def save(self, brain, path):
        started = time.monotonic()
        path = brain.save(path)
        self.work["writes"] += 1
        self.work["bytes"] += path.stat().st_size
        self.work["seconds"] += time.monotonic() - started
        return path

    def load(self, path):
        started = time.monotonic()
        brain = cd.Brain.load(path)
        self.work["reads"] += 1
        self.work["bytes"] += path.stat().st_size
        self.work["seconds"] += time.monotonic() - started
        return brain

    def equal(self, first, second):
        started = time.monotonic()
        with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
            same = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
        self.work["reads"] += 2
        self.work["bytes"] += first.stat().st_size + second.stat().st_size
        self.work["seconds"] += time.monotonic() - started
        return same


def scores(answers, phase, protocol):
    period = protocol["pattern"]["period"]
    futures = [
        loop.continuation(
            period, loop.pattern(period, protocol["play"]["prime"], p), protocol["play"]["rows"]
        )
        for p in range(period)
    ]
    values = [loop.score_play(answers, truth) for truth in futures]
    return {"phase": phase, "answers": answers, "scores": values, "own": values[phase]}


def assess(brain, work, protocol, *, lesion=False):
    watches, plays = [], []
    period, play = protocol["pattern"]["period"], protocol["play"]
    for phase in play["phases"]:
        if not lesion:
            watches.append(
                loop.watch_brain(
                    brain, "nocopy", loop.pattern(period, protocol["teaching"]["rows"], phase), work
                )
            )
        prime = loop.pattern(period, play["prime"], phase)
        constant = []
        if lesion:
            brain.reset()

            def act(event, constant=constant):
                answer = loop.act(brain, loop.observation(event), work)
                memory = brain.working_memory
                kept = memory.trace[:, 2].copy()
                memory.trace[:, :2] = 0.0
                np.testing.assert_array_equal(memory.trace[:, 2], kept)
                constant.append(kept.tolist())
                return answer

            work.phase = f"lesion-prime:{phase}"
            answer = None
            for event in prime:
                answer = act(int(event))
            answers = [answer]
            work.phase = f"lesion-play:{phase}"
            for _ in range(play["rows"] - 1):
                answer = act(loop.HOLD if answer is None else answer)
                answers.append(answer)
        else:
            answers = loop.play_brain(brain, "nocopy", prime, play["rows"], work, phase=phase)
        reading = scores(answers, phase, protocol)
        if lesion:
            reading["constant_trace"] = constant
        plays.append(reading)
    return {"watches": watches, "plays": plays}


def preservation(path, directory, io, failure_context=None):
    """Check acquired private imagination and sampled next-learning continuation."""
    brain = io.load(path)
    context = {} if failure_context is None else failure_context
    context["brain"] = brain
    brain.reset()
    work = loop.Work()
    for event in loop.pattern(4, 7):
        loop.act(brain, loop.observation(int(event)), work)
    before = io.save(brain, directory / "before-imagine.npz")
    imagined = brain.imagine(
        [loop.observation(0), loop.observation(1), loop.observation(0)], budget=200
    )
    after = io.save(brain, directory / "after-imagine.npz")
    unchanged = io.equal(before, after)
    brain = io.load(path)
    context["brain"] = brain
    brain.reset()
    previous = int(brain.act(loop.observation(loop.ONSET), greedy=False)[0])
    sweeps = int(brain.last_settlement["steps"])
    eligibility_phases = [[int(p.steps) for p in brain.basal_ganglia._pending[1:3]]]
    eligibility = sum(eligibility_phases[-1])
    action_steps, eligibility_steps = [sweeps], [eligibility]
    pending = io.save(brain, directory / "pending.npz")
    twin = io.load(pending)
    context["twin"] = twin
    choices, twin_choices, reports, equal_actions = [], [], [], True
    for event in (loop.HOLD, loop.HOLD, loop.HOLD, loop.ONSET):
        reward = [1.0 if previous == event else -1.0]
        a = brain.step(loop.observation(event), reward=reward, done=[False])
        b = twin.step(loop.observation(event), reward=reward, done=[False])
        equal_actions &= np.array_equal(a, b)
        previous = int(a[0])
        choices.append(previous)
        twin_choices.append(int(b[0]))
        for value in (brain, twin):
            action_steps.append(int(value.last_settlement["steps"]))
            eligibility_phases.append([int(p.steps) for p in value.basal_ganglia._pending[1:3]])
            eligibility_steps.append(sum(eligibility_phases[-1]))
            sweeps += action_steps[-1]
            eligibility += eligibility_steps[-1]
            reports.append(value.last_learning)
    continued = io.save(brain, directory / "continued.npz")
    restored = io.save(twin, directory / "restored.npz")
    same = io.equal(continued, restored)
    qualified = all(bool(np.all(p.qualified)) for p in imagined)
    free_budget = brain.learner.config.free_steps
    eligibility_budget = brain.basal_ganglia.config.eligibility_steps
    if eligibility_budget is None:
        eligibility_budget = brain.learner.config.nudged_steps
    not_exhausted = bool(
        all(n < free_budget for n in action_steps)
        and all(e["steps"] < free_budget for e in work.events if e["kind"] == "act")
        and all(n < eligibility_budget for pair in eligibility_phases for n in pair)
        and all(r["free_steps"] < free_budget for r in reports)
        and all(p.steps < 200 for p in imagined)
    )
    return {
        "passed": bool(
            unchanged
            and qualified
            and same
            and equal_actions
            and not_exhausted
            and not work.refused_acts
        ),
        "not_exhausted": not_exhausted,
        "phase_budgets": {"free": free_budget, "eligibility": eligibility_budget, "imagine": 200},
        "imagination_unchanged": unchanged,
        "imagination_qualified": qualified,
        "imagination_sweeps": sum(int(p.steps) for p in imagined),
        "imagination_phases": [
            {
                "steps": int(p.steps),
                "qualified": bool(np.all(p.qualified)),
                "residual": np.asarray(p.residual).tolist(),
            }
            for p in imagined
        ],
        "pending_arrays_equal": same,
        "pending_actions_equal": bool(equal_actions),
        "choices": choices,
        "twin_choices": twin_choices,
        "sampled_action_sweeps": sweeps,
        "sampled_action_steps": action_steps,
        "eligibility_sweeps": eligibility,
        "eligibility_steps": eligibility_steps,
        "eligibility_phase_steps": eligibility_phases,
        "learning_reports": reports,
        "work": work.summary(),
        "events": work.events,
    }


def train(brain, work, protocol, phase_reports):
    original_step = brain.learner.step

    def counted(*args, **kwargs):
        try:
            states, report = original_step(*args, **kwargs)
        except cd.LearningPhaseError as error:
            phase_reports.append(error.report)
            raise
        phase_reports.append(report)
        return states, report

    brain.learner.step = counted
    try:
        teaching = protocol["teaching"]
        return loop.teach_brain(
            brain,
            "nocopy",
            loop.pattern(protocol["pattern"]["period"], teaching["rows"], teaching["phase"]),
            teaching["passes"],
            work,
        )
    finally:
        brain.learner.step = original_step


def exhausted(reports, config):
    return {
        name: sum(r.get(name + "_steps", 0) >= budget for r in reports)
        for name, budget in (
            ("free", config["free_steps"]),
            ("nudged", config["nudged_steps"]),
            ("opposite", config["nudged_steps"]),
        )
    }


def founder_passes(row, protocol):
    if row.get("status") != "ok" or not row.get("preservation", {}).get("passed"):
        return False
    if not row.get("restored_outputs_equal"):
        return False
    gates, phases = protocol["gates"], protocol["play"]["phases"]
    for arm in ("sensory", "untaught", "lesion", "restored"):
        reading = row.get(arm, {})
        if len(reading.get("plays", [])) != len(phases):
            return False
        if [p.get("phase") for p in reading["plays"]] != phases:
            return False
        if any(len(p.get("answers", [])) != protocol["play"]["rows"] for p in reading["plays"]):
            return False
        work = reading.get("work", {})
        if work.get("refused_acts", 1) or work.get("refused_lessons", 1):
            return False
        if any(
            e["steps"] >= row["learning"]["free_steps"]
            for e in reading.get("events", [])
            if e["kind"] == "act"
        ):
            return False
        if any(exhausted(reading.get("phase_reports", []), row["learning"]).values()):
            return False
    for phase in phases:
        sensory = row["sensory"]["plays"][phase]
        if sensory["phase"] != phase or len(sensory["answers"]) != protocol["play"]["rows"]:
            return False
        measured = scores(sensory["answers"], phase, protocol)
        own = measured["own"]
        if not (
            own["agreement"] > gates["agreement_strictly_above"]
            and gates["onset_min"] <= own["onset_rate"] <= gates["onset_max"]
            and own["refusals"] == 0
        ):
            return False
        comparison = [s["agreement"] for i, s in enumerate(measured["scores"]) if i != phase]
        comparison.extend(
            scores(row[arm]["plays"][phase]["answers"], phase, protocol)["own"]["agreement"]
            for arm in ("untaught", "lesion")
        )
        if not all(own["agreement"] > value for value in comparison):
            return False
    return True


def stage_gate(rows, seeds, required, protocol):
    present = [row.get("seed") for row in rows]
    census = len(present) == len(set(present)) and set(present) == set(seeds)
    count = sum(founder_passes(row, protocol) for row in rows) if census else 0
    preservation_ok = census and all(
        row.get("preservation", {}).get("passed", False) for row in rows
    )
    return {
        "planned": len(seeds),
        "passed_founders": count,
        "preservation": preservation_ok,
        "passed": bool(preservation_ok and count >= required),
    }


def run_founder(seed, directory, protocol):
    directory.mkdir()
    began, io = time.monotonic(), Checkpoints()
    row = {
        "seed": seed,
        "status": "running",
        "work_complete": False,
        "candidate_work_complete": False,
        "constructed_graphs": 0,
        "checkpoints": io.work,
    }
    active = current = brain = None
    context_name, proof_context = "construction", {}
    try:
        brain = make_brain(seed, "sensory", protocol)
        row["constructed_graphs"] += 2
        row["learning"] = brain.learner.config.to_dict()
        row["initial_arrays"] = {
            k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in initial_arrays(brain).items()
        }
        for arm in ("untaught", "sensory"):
            context_name = arm
            active = loop.Work()
            reading = {"phase_reports": [], "work": active.summary(), "events": active.events}
            row[arm] = reading
            if arm != "sensory":
                current = make_brain(seed, "sensory", protocol)
                row["constructed_graphs"] += 2
            else:
                current = brain
            if arm != "untaught":
                reading["teaching"] = train(current, active, protocol, reading["phase_reports"])
                if arm == "sensory":
                    acquired = io.save(current, directory / "acquired.npz")
            reading.update(assess(current, active, protocol))
            reading["work"] = active.summary()
        active = loop.Work()
        context_name = "lesion"
        row["lesion"] = {"work": active.summary(), "events": active.events}
        current = io.load(acquired)
        row["lesion"].update(assess(current, active, protocol, lesion=True))
        row["lesion"]["work"] = active.summary()
        active = loop.Work()
        context_name = "restored"
        current = io.load(acquired)
        restored = assess(current, active, protocol)
        row["restored"] = {**restored, "work": active.summary(), "events": active.events}
        row["restored_outputs_equal"] = all(
            a["answers"] == b["answers"]
            for a, b in zip(restored["plays"], row["sensory"]["plays"], strict=True)
        )
        context_name = "preservation"
        row["preservation"] = preservation(acquired, directory, io, proof_context)
        row.update(status="ok", candidate_work_complete=True)
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
        if active is not None:
            row["partial_active_work"] = active.summary()
        failed = proof_context.get("brain", current if current is not None else brain)
        row["failure_context"] = {
            "stage": context_name,
            "last_event": active.events[-1] if active is not None and active.events else None,
        }
        if failed is not None:
            try:
                path = io.save(failed, directory / "failed-context.npz")
                row["failure_context"]["checkpoint"] = path.name
                row["failure_context"]["sha256"] = sha256(path)
                if "twin" in proof_context:
                    io.save(proof_context["twin"], directory / "failed-twin.npz")
            except Exception as save_error:
                row["failure_context"]["save_error"] = f"{type(save_error).__name__}: {save_error}"
    # The old association recipe is an independent failure control. Its failure
    # must not erase an acquired candidate or veto its separately measured gate.
    active, current = loop.Work(), None
    original = {
        "status": "running",
        "work_complete": False,
        "phase_reports": [],
        "events": active.events,
    }
    row["association"] = original
    try:
        current = make_brain(seed, "association", protocol)
        row["constructed_graphs"] += 1
        original["learning"] = current.learner.config.to_dict()
        original["teaching"] = train(current, active, protocol, original["phase_reports"])
        original.update(assess(current, active, protocol))
        original.update(status="ok", work_complete=True)
    except Exception as error:
        original.update(status="error", error=f"{type(error).__name__}: {error}")
        if current is not None:
            try:
                path = io.save(current, directory / "association-failed.npz")
                original["failure_checkpoint"] = {"path": path.name, "sha256": sha256(path)}
            except Exception as save_error:
                original["save_error"] = f"{type(save_error).__name__}: {save_error}"
    original["work"] = active.summary()
    original["quality_complete"] = bool(
        original["status"] == "ok"
        and not active.refused_acts
        and not active.refused_lessons
        and not any(exhausted(original["phase_reports"], current.learner.config.to_dict()).values())
    )
    row["work_complete"] = row["candidate_work_complete"] and original["work_complete"]
    row["elapsed_seconds"] = time.monotonic() - began
    row["passed"] = founder_passes(row, protocol)
    return row


def baseline(seed, protocol):
    rng = np.random.default_rng(seed)
    return {
        arm: [
            scores(
                (
                    [0] * protocol["play"]["rows"]
                    if arm == "hold"
                    else rng.integers(2, size=protocol["play"]["rows"]).tolist()
                ),
                p,
                protocol,
            )
            for p in protocol["play"]["phases"]
        ]
        for arm in ("hold", "random")
    }


def source_files():
    package = Path(cd.__file__).resolve().parent
    return [
        ("source/" + p.name, p)
        for p in (Path(__file__).resolve(), PROTOCOL, HERE / "loop_rhythm.py")
    ] + [
        ("source/cadence/" + p.relative_to(package).as_posix(), p)
        for p in sorted(package.rglob("*.py"))
    ]


def _verified_work(reading, *, complete):
    """Recompute successful ledger totals; failed rows retain explicitly partial work."""
    events = reading.get("events", [])
    totals = dict(
        acts=0, lessons=0, refused_acts=0, refused_lessons=0, act_sweeps=0, lesson_sweeps=0
    )
    for event in events:
        assert event["kind"] in ("act", "lesson")
        steps = event["steps"]
        assert isinstance(steps, int) and steps >= 0
        if event["kind"] == "act":
            assert event["answer"] in (None, 0, 1)
            totals["acts"] += 1
            totals["refused_acts"] += event["answer"] is None
            totals["act_sweeps"] += steps
        else:
            assert isinstance(event["refused"], bool)
            totals["lessons"] += 1
            totals["refused_lessons"] += event["refused"]
            totals["lesson_sweeps"] += steps
    work = reading.get("work", {})
    for key, value in totals.items():
        assert isinstance(work[key], int) and work[key] >= 0
        if complete:
            assert work[key] == value, f"work counter {key} disagrees with events"
    reports = reading.get("phase_reports")
    if reports is not None:
        for report in reports:
            assert report["total_steps"] == sum(
                report.get(name + "_steps", 0) for name in ("free", "nudged", "opposite")
            )
        if complete:
            assert len(reports) == totals["lessons"]
            assert sum(r["total_steps"] for r in reports) == totals["lesson_sweeps"]
    return totals


def _verified_reading(reading, protocol, *, complete, lesion=False):
    phases, play = protocol["play"]["phases"], protocol["play"]
    plays = reading.get("plays", [])
    present = [p["phase"] for p in plays]
    assert len(present) == len(set(present)) and set(present) <= set(phases)
    if complete:
        assert present == phases
    for value in plays:
        assert len(value["answers"]) == play["rows"]
        assert all(answer in (None, 0, 1) for answer in value["answers"])
        recomputed = scores(value["answers"], value["phase"], protocol)
        assert all(value[key] == recomputed[key] for key in recomputed)
        if lesion:
            retained = np.asarray(value["constant_trace"])
            assert retained.shape == (play["prime"] + play["rows"] - 1, 1)
            assert np.isfinite(retained).all()
        if complete:
            prefix = "lesion-" if lesion else ""
            primed = [
                e["answer"]
                for e in reading["events"]
                if e["kind"] == "act" and e["phase"] == f"{prefix}prime:{value['phase']}"
            ]
            played = [
                e["answer"]
                for e in reading["events"]
                if e["kind"] == "act" and e["phase"] == f"{prefix}play:{value['phase']}"
            ]
            assert len(primed) == play["prime"] and len(played) == play["rows"] - 1
            assert value["answers"] == primed[-1:] + played
    watches = reading.get("watches", [])
    if complete:
        assert len(watches) == (0 if lesion else len(phases))
    watched = []
    for phase, watch in zip(phases, watches, strict=False):
        truth = loop.pattern(protocol["pattern"]["period"], protocol["teaching"]["rows"], phase)[1:]
        answers = watch["answers"]
        assert len(answers) == len(truth) and all(a in (None, 0, 1) for a in answers)
        assert watch["agreement"] == sum(
            a == int(t) for a, t in zip(answers, truth, strict=True)
        ) / len(truth)
        onsets = int(np.sum(truth == loop.ONSET))
        announced = sum(
            a == loop.ONSET and t == loop.ONSET for a, t in zip(answers, truth, strict=True)
        )
        assert watch["onset_recall"] == announced / max(1, onsets)
        assert watch["rows"] == sum(a is not None for a in answers)
        assert watch["refusals"] == sum(a is None for a in answers)
        watched.extend(answers)
    if complete:
        actual = [
            e["answer"]
            for e in reading["events"]
            if e["kind"] == "act" and e["phase"] == "watching"
        ]
        assert actual == watched, "watch answers disagree with charged acts"
    teaching = reading.get("teaching")
    if teaching is not None:
        truth = loop.pattern(
            protocol["pattern"]["period"],
            protocol["teaching"]["rows"],
            protocol["teaching"]["phase"],
        )[1:]
        agreements, lessons = [], []
        assert len(teaching["answers"]) == protocol["teaching"]["passes"]
        for batch, answers in enumerate(teaching["answers"]):
            assert len(answers) == len(truth)
            agreements.append(
                sum(a == int(t) for a, t in zip(answers, truth, strict=True)) / len(truth)
            )
            events = [e for e in reading["events"] if e["phase"] == f"teaching:{batch}"]
            assert answers == [e["answer"] for e in events if e["kind"] == "act"]
            lessons.append(sum(e["kind"] == "lesson" for e in events))
        assert teaching["agreement_per_pass"] == agreements
        assert teaching["lessons_per_pass"] == lessons
        assert teaching["last_pass_agreement"] == (agreements[-1] if agreements else None)
    _verified_work(reading, complete=complete)


def _verified_preservation(directory, row, protocol):
    proof = row["preservation"]

    def equal(first, second):
        with (
            np.load(directory / first, allow_pickle=False) as a,
            np.load(directory / second, allow_pickle=False) as b,
        ):
            return a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)

    unchanged = equal("before-imagine.npz", "after-imagine.npz")
    continued = equal("continued.npz", "restored.npz")
    with np.load(directory / "pending.npz", allow_pickle=False) as saved:
        if "brain" in saved.files:  # one group per memory, the format since 0.85
            meta = json.loads(str(saved["brain"]))
            working = meta["memories"]["working_memory"]
        else:
            meta = json.loads(str(saved["generic"]))
            working = meta["working_memory"]
        assert meta["pending"] is True
        assert working["source"] == protocol["history"]["source"]
        assert working["target"] == protocol["history"]["target"]
    assert proof["imagination_unchanged"] == unchanged
    assert proof["pending_arrays_equal"] == continued
    assert len(proof["choices"]) == len(proof["twin_choices"]) == 4
    actions_equal = proof["choices"] == proof["twin_choices"]
    assert proof["pending_actions_equal"] == actions_equal
    phases = proof["imagination_phases"]
    assert len(phases) == 3
    qualified = all(p["qualified"] for p in phases)
    assert proof["imagination_qualified"] == qualified
    assert proof["imagination_sweeps"] == sum(p["steps"] for p in phases)
    for raw, total in (
        ("sampled_action_steps", "sampled_action_sweeps"),
        ("eligibility_steps", "eligibility_sweeps"),
    ):
        assert len(proof[raw]) == 9 and all(v >= 0 for v in proof[raw])
        assert proof[total] == sum(proof[raw])
    assert len(proof["learning_reports"]) == 8
    raw_eligibility = proof["eligibility_phase_steps"]
    assert len(raw_eligibility) == 9 and all(len(pair) == 2 for pair in raw_eligibility)
    assert [sum(pair) for pair in raw_eligibility] == proof["eligibility_steps"]
    # Configuration is bound to the saved pending life, not a verifier default.
    learned = cd.Brain.load(directory / "pending.npz")
    assert row["learning"] == learned.learner.config.to_dict()
    eligibility_budget = learned.basal_ganglia.config.eligibility_steps
    if eligibility_budget is None:
        eligibility_budget = learned.learner.config.nudged_steps
    budgets = {
        "free": learned.learner.config.free_steps,
        "eligibility": eligibility_budget,
        "imagine": 200,
    }
    assert proof["phase_budgets"] == budgets
    not_exhausted = bool(
        all(n < budgets["free"] for n in proof["sampled_action_steps"])
        and all(e["steps"] < budgets["free"] for e in proof["events"] if e["kind"] == "act")
        and all(0 <= n < budgets["eligibility"] for pair in raw_eligibility for n in pair)
        and all(0 <= r["free_steps"] < budgets["free"] for r in proof["learning_reports"])
        and all(p["steps"] < budgets["imagine"] for p in phases)
    )
    assert proof["not_exhausted"] == not_exhausted
    _verified_work(proof, complete=True)
    assert proof["work"]["acts"] == 7 and proof["work"]["lessons"] == 0
    assert proof["passed"] == bool(
        unchanged
        and continued
        and actions_equal
        and qualified
        and not_exhausted
        and not proof["work"]["refused_acts"]
    )


def verify(directory):
    try:
        directory = Path(directory)
        receipt = Receipt.read(directory / "summary.json")
        names = [item["path"] for item in receipt.source["files"]]
        assert len(names) == len(set(names)), "duplicate source inventory entry"
        expected_sources = {name for name, _ in source_files()}
        assert set(names) == expected_sources, "incomplete source inventory"
        actual_sources = {
            p.relative_to(directory).as_posix()
            for p in (directory / "source").rglob("*")
            if p.is_file()
        }
        assert actual_sources == set(names), "source archive differs from inventory"
        files = [(name, directory / name) for name in names]
        valid, reason = Receipt.verify(directory / "summary.json", sources=files)
        if not valid:
            return valid, reason
        body = receipt.body
        assert receipt.kind == SCHEMA
        protocol = body["protocol"]
        assert protocol == json.loads(
            (directory / "source/protocol-native-history.json").read_text()
        )
        actual_artifacts = {p.relative_to(directory).as_posix() for p in directory.rglob("*.npz")}
        assert set(body["artifacts"]) == actual_artifacts, "incomplete checkpoint inventory"
        for path, digest in body["artifacts"].items():
            assert sha256(directory / path) == digest, path
        baseline_seeds = set()
        for stage in ("development", "confirmation"):
            for row in body[stage]:
                baseline_seeds.add(str(row["seed"]))
                complete = row.get("status") == "ok"
                assert row.get("candidate_work_complete", False) == complete
                original_complete = row.get("association", {}).get("status") == "ok"
                assert row.get("work_complete", False) == (complete and original_complete)
                original = row["association"]
                assert original["work_complete"] == original_complete
                assert original["quality_complete"] == bool(
                    original_complete
                    and not original["work"]["refused_acts"]
                    and not original["work"]["refused_lessons"]
                    and not any(exhausted(original["phase_reports"], original["learning"]).values())
                )
                for arm in ("sensory", "untaught", "lesion", "association", "restored"):
                    if complete or arm in row:
                        _verified_reading(
                            row[arm],
                            protocol,
                            complete=original_complete if arm == "association" else complete,
                            lesion=arm == "lesion",
                        )
                if complete:
                    folder = f"{stage}-{row['seed']}"
                    required = {
                        f"{folder}/{name}.npz"
                        for name in (
                            "acquired",
                            "before-imagine",
                            "after-imagine",
                            "pending",
                            "continued",
                            "restored",
                        )
                    }
                    assert required <= actual_artifacts, "missing preservation checkpoint"
                    restored_equal = all(
                        a["answers"] == b["answers"]
                        for a, b in zip(
                            row["sensory"]["plays"], row["restored"]["plays"], strict=True
                        )
                    )
                    assert row["restored_outputs_equal"] == restored_equal
                    _verified_preservation(directory / folder, row, protocol)
                assert row["passed"] == founder_passes(row, protocol)
        assert set(body["baselines"]) == baseline_seeds
        assert body["random_expected_accuracy"] == 0.5
        for seed in baseline_seeds:
            assert body["baselines"][seed] == baseline(int(seed), protocol), (
                "control scores/draws disagree"
            )
        expected = protocol["seeds"]["screen" if body["mode"] == "screen" else "development"]
        required = (
            len(expected) if body["mode"] == "screen" else protocol["gates"]["development_passes"]
        )
        assert body["development_gate"] == stage_gate(
            body["development"], expected, required, protocol
        )
        admitted = body["mode"] == "full" and body["development_gate"]["passed"]
        assert body["confirmation_admitted"] == admitted
        if admitted:
            assert body["confirmation_gate"] == stage_gate(
                body["confirmation"],
                protocol["seeds"]["confirmation"],
                protocol["gates"]["confirmation_passes"],
                protocol,
            )
        else:
            assert not body["confirmation"] and body["confirmation_gate"] is None
        return True, "sources, inventory, checkpoints, raw scores, work and gates agree"
    except (OSError, ValueError, TypeError, KeyError, AssertionError, IndexError) as error:
        return False, str(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--screen", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        valid, reason = verify(args.verify)
        print(reason)
        return 0 if valid else 1
    if args.out is None or not (args.screen or args.full):
        parser.error("choose --screen or --full and --out NEW_DIRECTORY")
    args.out.mkdir(parents=True, exist_ok=False)
    files = source_files()
    manifest = source_manifest(files)
    for name, path in files:
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    archived = [(name, args.out / name) for name, _ in files]
    protocol = load_protocol()
    git = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    body = {
        "mode": "screen" if args.screen else "full",
        "protocol": protocol,
        "provenance": {
            "commit": git.stdout.strip(),
            "python": platform.python_version(),
            "cadence": cd.__version__,
            "numpy": np.__version__,
        },
        "development": [],
        "confirmation": [],
        "confirmation_admitted": False,
        "confirmation_gate": None,
        "random_expected_accuracy": 0.5,
        "baselines": {},
    }
    seeds = protocol["seeds"]["screen" if args.screen else "development"]
    required = len(seeds) if args.screen else protocol["gates"]["development_passes"]

    def run_stage(name, chosen):
        for seed in chosen:
            row = run_founder(seed, args.out / f"{name}-{seed}", protocol)
            body[name].append(row)
            body["baselines"][str(seed)] = baseline(seed, protocol)
            print(name, seed, row["status"], row["passed"], flush=True)

    run_stage("development", seeds)
    body["development_gate"] = stage_gate(body["development"], seeds, required, protocol)
    if args.full and body["development_gate"]["passed"]:
        body["confirmation_admitted"] = True
        run_stage("confirmation", protocol["seeds"]["confirmation"])
        body["confirmation_gate"] = stage_gate(
            body["confirmation"],
            protocol["seeds"]["confirmation"],
            protocol["gates"]["confirmation_passes"],
            protocol,
        )
    assert source_manifest(files) == manifest, "sources changed during execution"
    body["artifacts"] = {
        p.relative_to(args.out).as_posix(): sha256(p) for p in sorted(args.out.rglob("*.npz"))
    }
    Receipt.build(SCHEMA, body, archived).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    print(reason)
    gate = body["confirmation_gate"] if args.full else body["development_gate"]
    return 0 if valid and gate is not None and gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
