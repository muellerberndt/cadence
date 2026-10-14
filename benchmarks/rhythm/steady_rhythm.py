"""Steady-rhythm chamber: alternate two learned actions under constant drive.

One continuing ``Brain.compose`` life is taught to alternate A, B, A, B while every
observation after an onset cue is identical. The working trace and optional native
own-command efference carry phase inside the brain. Controls fork the same complete
checkpoint and use the same instrument. Event time is one accepted observation; the
real-time runs declare a physical cadence per event and vary solver budget and
host load at the same event schedule. Use --verify RUN_DIRECTORY to check a
completed run's source and artifact custody.
"""

from __future__ import annotations

import os

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
# NumPy wheels on macOS use Accelerate, whose thread limit is separate from OpenBLAS.
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import asdict, dataclass, field, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import rhythm_inputs as inputs  # noqa: E402
import timing_acceptance  # noqa: E402

import cadence  # noqa: E402
from cadence import Brain, LearnerConfig  # noqa: E402
from cadence.learning import LearningPhaseError  # noqa: E402
from cadence.receipts import Receipt  # noqa: E402

SCHEMA = "steady-rhythm/1"
CONTROLS = (
    "intact",
    "restored",
    "erased",
    "shuffled",
    "reset",
    "static",
    "static_cold",
    "flipflop",
    "random",
)
REFUSED = -1


@dataclass
class CheckpointIO:
    """Every checkpoint read/write/comparison, including conventional controls."""

    operations: list[dict] = field(default_factory=list)

    def save(self, model, path: Path) -> Path:
        began = time.perf_counter()
        model.save(path)
        self.operations.append(
            {
                "operation": "write",
                "file": path.name,
                "bytes": path.stat().st_size,
                "seconds": time.perf_counter() - began,
            }
        )
        return path

    def load(self, kind, path: Path):
        began = time.perf_counter()
        model = kind.load(path)
        self.operations.append(
            {
                "operation": "read",
                "file": path.name,
                "bytes": path.stat().st_size,
                "seconds": time.perf_counter() - began,
            }
        )
        return model

    def equal(self, first: Path, second: Path) -> bool:
        began = time.perf_counter()
        same = same_saved_arrays(first, second)
        self.operations.append(
            {
                "operation": "compare",
                "files": [first.name, second.name],
                "bytes": first.stat().st_size + second.stat().st_size,
                "seconds": time.perf_counter() - began,
            }
        )
        return same

    def save_world(self, value: dict, path: Path) -> None:
        began = time.perf_counter()
        path.write_text(json.dumps(value, indent=2) + "\n")
        self.operations.append(
            {
                "operation": "write",
                "file": path.name,
                "bytes": path.stat().st_size,
                "seconds": time.perf_counter() - began,
            }
        )

    def load_world(self, path: Path) -> dict:
        began = time.perf_counter()
        value = json.loads(path.read_text())
        self.operations.append(
            {
                "operation": "read",
                "file": path.name,
                "bytes": path.stat().st_size,
                "seconds": time.perf_counter() - began,
            }
        )
        return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_brain(seed: int, protocol: dict, recipe: str = "selected") -> Brain:
    """The declared System 1: one processing module, working trace, no associative store."""
    spec = protocol["recipes"][recipe]
    learning = LearnerConfig(**spec["learning"])
    return Brain.compose(
        inputs.INPUTS,
        inputs.ACTIONS,
        modules=tuple(spec["modules"]),
        lateral=spec["lateral"],
        seed=seed,
        learning=learning,
        episodic=spec["episodic"],
        working_memory_decay=spec["working_memory_decay"],
        working_memory_amplitude=spec["working_memory_amplitude"],
        # the composition of releases before 0.82.0 read the trace back at scale 12
        working_memory_scale=spec.get("working_memory_scale", 12.0),
        # rhythm/2: the efference copy as a gene; absent keys build the rhythm/1 brain
        **{key: spec[key] for key in ("efference_amplitude", "efference_decay") if key in spec},
    )


@dataclass
class Work:
    """Actual solver work of one branch. Sweeps are not joules; seconds exclude checkpoint IO."""

    action_attempts: int = 0
    action_refusals: int = 0
    action_sweeps: int = 0
    action_row_sweeps: int = 0
    action_residual_checks: int = 0
    action_damping_halvings: int = 0
    action_cpu_seconds: float = 0.0
    teacher_attempts: int = 0
    teacher_refusals: int = 0
    teacher_presentations: int = 0
    teacher_sweeps: int = 0
    teacher_row_sweeps: int = 0
    teacher_budget_exhausted: int = 0
    trace_row_updates: int = 0
    calls_seconds: float = 0.0
    reports: list[dict] = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        out = asdict(self)
        del out["reports"]
        return out

    def act(self, brain: Brain, x: np.ndarray) -> np.ndarray | None:
        """A refused act is a missed action: state, trace and the clock's event are preserved."""
        start = time.perf_counter()
        cpu_start = time.process_time()
        self.action_attempts += 1
        prior = brain.last_settlement
        try:
            answer = brain.act(x, greedy=True)
        except RuntimeError:
            report = brain.last_settlement
            if report is None or report is prior or report["qualified"]:
                raise
            self.action_refusals += 1
            answer = None
        finally:
            elapsed = time.perf_counter() - start
            self.calls_seconds += elapsed
        cpu_seconds = time.process_time() - cpu_start
        self.action_cpu_seconds += cpu_seconds
        report = dict(brain.last_settlement)
        self.reports.append(
            {
                "operation": "act",
                "answer": None if answer is None else answer.tolist(),
                "call_seconds": elapsed,
                "process_cpu_seconds": cpu_seconds,
                "rows": len(x),
                **report,
            }
        )
        self.action_sweeps += int(report["steps"])
        self.action_row_sweeps += len(x) * int(report["steps"])
        self.action_residual_checks += int(report["residual_checks"])
        self.action_damping_halvings += int(report["damping_halvings"])
        if answer is not None:
            self.trace_row_updates += len(x)
        return answer

    def teach(self, brain: Brain, drive: np.ndarray, labels: np.ndarray) -> bool:
        """One finite lesson on the supplied drive; refusals are charged and skipped."""
        start = time.perf_counter()
        self.teacher_attempts += 1
        accepted = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, report = brain.learner.step(drive, labels)
        except LearningPhaseError as error:
            self.teacher_refusals += 1
            report = error.report
            accepted = False
        finally:
            elapsed = time.perf_counter() - start
            self.calls_seconds += elapsed
        self.reports.append(
            {"operation": "teach", "accepted": accepted, "call_seconds": elapsed, **report}
        )
        self.teacher_presentations += int(report["attempted_presentations"])
        self.teacher_sweeps += int(report["total_steps"])
        self.teacher_row_sweeps += int(report["total_row_sweeps"])
        self.teacher_budget_exhausted += int(report.get("free_budget_exhausted", 0.0))
        return accepted


class FlipFlop:
    """Competent recurrent timing control with matched information.

    A linear softmax policy over [1, observation, one-hot of its own previous action],
    trained online by the same lesson schedule and labels as the brain. It sees the
    constant drive, the cue and its own previous output; the brain sees the same
    observations and its own working trace. It is a control, never the answer path.
    """

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)
        self.weights = np.zeros((1 + inputs.INPUTS + inputs.ACTIONS, inputs.ACTIONS))
        self.previous = np.zeros(inputs.ROWS, dtype=np.int64)
        self.lessons = 0

    def features(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.ones((len(x), 1)), x, np.eye(inputs.ACTIONS)[self.previous]], axis=1
        )

    def act(self, x: np.ndarray) -> np.ndarray:
        action = np.argmax(self.features(x) @ self.weights, axis=1)
        self.previous = action.astype(np.int64)
        return self.previous.copy()

    def teach(self, features: np.ndarray, labels: np.ndarray) -> None:
        logits = features @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        gradient = features.T @ (p - np.eye(inputs.ACTIONS)[labels]) / len(features)
        self.weights -= self.rate * gradient
        self.lessons += len(labels)

    def save(self, path: Path) -> Path:
        np.savez(
            path,
            weights=self.weights,
            previous=self.previous,
            lessons=np.array(self.lessons),
            rate=np.array(self.rate),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> FlipFlop:
        with np.load(path, allow_pickle=False) as data:
            out = cls(float(data["rate"]))
            out.weights = data["weights"].copy()
            out.previous = data["previous"].copy()
            out.lessons = int(data["lessons"])
        return out


# -- scoring


def score_window(actions: np.ndarray, anchor: np.ndarray, block: int) -> dict:
    """Alternation, repeats, agreement with the continued ideal alternation, period, refusals.

    ``actions`` has one row per event with -1 for a refused act. The ideal alternation
    continues ``anchor``, the last executed action of each row before the window.
    A refused event counts as wrong and breaks no pair; blocks report alternation
    per declared block of events, distinguishing a transient from continuing activity.
    """
    actions = np.asarray(actions, dtype=np.int64)
    events = len(actions)
    ideal = inputs.ideal_alternation(anchor, events)
    valid = actions >= 0
    rows = []
    for r in range(inputs.ROWS):
        a, ok = actions[:, r], valid[:, r]
        pairs = ok[1:] & ok[:-1]
        alternations = int(np.sum((a[1:] != a[:-1]) & pairs))
        repeats = int(np.sum((a[1:] == a[:-1]) & pairs))
        zeros = np.flatnonzero(a == 0)
        rows.append(
            {
                "alternation_rate": alternations / int(pairs.sum()) if pairs.any() else None,
                "alternations": alternations,
                "repeats": repeats,
                "refusals": int(events - ok.sum()),
                "agreement": float(np.mean((a == ideal[:, r]) & ok)),
                "first_agrees": bool(ok[0] and a[0] == ideal[0, r]),
                "period": float(np.mean(np.diff(zeros))) if len(zeros) > 1 else None,
            }
        )
    blocks = []
    for start in range(0, events, block):
        a = actions[start : start + block]
        ok = valid[start : start + block]
        pairs = ok[1:] & ok[:-1]
        blocks.append(
            float(np.sum((a[1:] != a[:-1]) & pairs) / pairs.sum()) if pairs.any() else None
        )
    rates = [row["alternation_rate"] for row in rows if row["alternation_rate"] is not None]
    periods = [row["period"] for row in rows if row["period"] is not None]
    return {
        "rows": rows,
        "alternation_rate": float(np.mean(rates)) if rates else None,
        "repeats": int(sum(row["repeats"] for row in rows)),
        "refusals": int(sum(row["refusals"] for row in rows)),
        "agreement": float(np.mean([row["agreement"] for row in rows])),
        "period": float(np.mean(periods)) if periods else None,
        "blocks": blocks,
    }


def recovery_events(actions: np.ndarray) -> list[int | None]:
    """Per row: events after the disturbance before uninterrupted alternation holds.

    Zero means every post-disturbance pair alternates; ``None`` means alternation did
    not hold through the end of the post window.
    """
    actions = np.asarray(actions, dtype=np.int64)
    out: list[int | None] = []
    for r in range(inputs.ROWS):
        a = actions[:, r]
        good = (a[1:] != a[:-1]) & (a[1:] >= 0) & (a[:-1] >= 0)
        bad = np.flatnonzero(~good)
        if not len(bad):
            out.append(0)
        elif bad[-1] + 1 >= len(a) - 1:
            out.append(None)
        else:
            out.append(int(bad[-1] + 1))
    return out


def last_executed(actions: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """The last non-refused action per row, or the fallback where no act was executed."""
    actions = np.asarray(actions, dtype=np.int64)
    out = np.asarray(fallback, dtype=np.int64).copy()
    for r in range(inputs.ROWS):
        executed = actions[:, r][actions[:, r] >= 0]
        if len(executed):
            out[r] = executed[-1]
    return out


def run_events(brain: Brain, observations: np.ndarray, work: Work, before=None) -> np.ndarray:
    actions = []
    for x in observations:
        if before is not None:
            before(brain)
        answer = work.act(brain, x)
        actions.append(np.full(inputs.ROWS, REFUSED) if answer is None else answer)
    return np.stack(actions) if actions else np.zeros((0, inputs.ROWS), dtype=np.int64)


def same_saved_arrays(first: Path, second: Path) -> bool:
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        return set(a.files) == set(b.files) and all(
            np.array_equal(a[name], b[name]) for name in a.files
        )


# -- stages


def teach_life(brain: Brain, flip: FlipFlop, frozen: dict, arm: str, work: Work) -> dict:
    """Teaching bouts: a cue event labels the cued action; every later label is the opposite
    of the row's own last executed action. ``every`` teaches before each free act;
    ``mismatch`` acts first and teaches only the rows whose free act was wrong."""
    if arm not in ("every", "mismatch"):
        raise ValueError("arm must be 'every' or 'mismatch'")
    observations, kinds = frozen["teach/observations"], frozen["teach/kinds"]
    cued = inputs.cue_labels(kinds, frozen["teach/cues"], frozen["teach/bout"])
    previous = np.zeros(inputs.ROWS, dtype=np.int64)
    flip_previous = np.zeros(inputs.ROWS, dtype=np.int64)
    actions, labels_log, flip_actions, flip_labels_log = [], [], [], []
    for t, x in enumerate(observations):
        labels = np.where(cued[t] >= 0, cued[t], 1 - previous)
        flip_labels = np.where(cued[t] >= 0, cued[t], 1 - flip_previous)
        drive = brain.stimulus(x)
        features = flip.features(x)
        if arm == "every":
            work.teach(brain, drive, labels)
            flip.teach(features, flip_labels)
        answer = work.act(brain, x)
        flip_answer = flip.act(x)
        if arm == "mismatch":
            if answer is not None:
                wrong = answer != labels
                if wrong.any():
                    work.teach(brain, drive[wrong], labels[wrong])
            wrong = flip_answer != flip_labels
            if wrong.any():
                flip.teach(features[wrong], flip_labels[wrong])
        actions.append(np.full(inputs.ROWS, REFUSED) if answer is None else answer)
        labels_log.append(labels)
        flip_actions.append(flip_answer)
        flip_labels_log.append(flip_labels)
        if answer is not None:
            previous = answer.astype(np.int64)
        flip_previous = flip_answer
    actions, labels_log = np.stack(actions), np.stack(labels_log)
    flip_actions, flip_labels_log = np.stack(flip_actions), np.stack(flip_labels_log)
    bouts = frozen["teach/bout"]
    agreement, flip_agreement = [], []
    for index in range(int(bouts.max()) + 1):
        rows = bouts == index
        agreement.append(float(np.mean(actions[rows] == labels_log[rows])))
        flip_agreement.append(float(np.mean(flip_actions[rows] == flip_labels_log[rows])))
    return {
        "actions": actions.tolist(),
        "labels": labels_log.tolist(),
        "flipflop_actions": flip_actions.tolist(),
        "agreement_per_bout": agreement,
        "last_bout_agreement": agreement[-1],
        "flipflop_agreement_per_bout": flip_agreement,
        "lessons_attempted": work.teacher_attempts,
        "lessons_presentations": work.teacher_presentations,
        "flipflop_lessons": flip.lessons,
        "work": work.summary(),
    }


def carried(brain: Brain) -> list:
    """The carried history of the brain: its working trace and, under rhythm/2, its
    efference copy. Every history control acts on all of it."""
    trace = brain.working_memory
    assert trace is not None
    return [trace] if brain.efference is None else [trace, brain.efference]


def erase_trace(brain: Brain) -> None:
    for memory in carried(brain):
        memory.reset(inputs.ROWS)


def transplant_trace(brain: Brain, permutation: np.ndarray) -> None:
    for memory in carried(brain):
        for name in ("trace", "last", "cold"):
            setattr(memory, name, getattr(memory, name)[permutation].copy())


def window_stage(
    brain: Brain,
    flip: FlipFlop,
    frozen: dict,
    protocol: dict,
    directory: Path,
    ledger: list,
    io: CheckpointIO | None = None,
) -> dict:
    """Cue, lead events, a probe checkpoint between two actions, then every control."""
    window = protocol["window"]
    io = CheckpointIO() if io is None else io
    block = window["block"]
    observations = frozen["window/observations"]
    lead_count = 1 + window["lead"]
    lead_work = Work()
    lead_actions = run_events(brain, observations[:lead_count], lead_work)
    flip_lead = np.stack([flip.act(x) for x in observations[:lead_count]])
    anchor = last_executed(lead_actions, frozen["window/cues"])
    probe = io.save(brain, directory / "probe.npz")
    flip_probe = io.save(flip, directory / "probe-flipflop.npz")
    rest = observations[lead_count:]
    if "timing_acceptance" in protocol:
        io.save_world(
            {
                "next_event": lead_count,
                "anchor": anchor.tolist(),
                "input_key": "window/observations",
                "clock": "event index; paced forks reanchor physical time",
            },
            directory / "probe-world.json",
        )
    permutation = frozen["shuffle/permutation"]
    branches: dict[str, dict] = {}
    restored_after = None

    def fork(name: str, prepare=None, before=None) -> np.ndarray:
        nonlocal restored_after
        branch = io.load(Brain, probe)
        if prepare is not None:
            prepare(branch)
        work = Work()
        continuation = rest
        if name == "restored" and "timing_acceptance" in protocol:
            world = io.load_world(directory / "probe-world.json")
            assert world["anchor"] == anchor.tolist()
            continuation = observations[world["next_event"] :]
        actions = run_events(branch, continuation, work, before=before)
        ledger.extend({"stage": "window", "branch": name, **r} for r in work.reports)
        branches[name] = {
            "actions": actions.tolist(),
            "anchor": anchor.tolist(),
            "work": work.summary(),
            "score": score_window(actions, anchor, block),
        }
        if name == "restored":
            restored_after = branch
        return actions

    restored_actions = fork("restored")
    fork("erased", prepare=erase_trace)
    fork("shuffled", prepare=lambda b: transplant_trace(b, permutation))
    fork("reset", prepare=lambda b: b.reset())
    fork("static", before=erase_trace)
    fork("static_cold", before=lambda b: b.reset())
    intact_work = Work()
    intact_actions = run_events(brain, rest, intact_work)
    ledger.extend({"stage": "window", "branch": "intact", **r} for r in intact_work.reports)
    ledger.extend({"stage": "window", "branch": "lead", **r} for r in lead_work.reports)
    branches["intact"] = {
        "actions": intact_actions.tolist(),
        "anchor": anchor.tolist(),
        "work": intact_work.summary(),
        "score": score_window(intact_actions, anchor, block),
    }
    assert restored_after is not None
    intact_after = io.save(brain, directory / "intact-after.npz")
    second = io.save(restored_after, directory / "restored-after.npz")
    continuation = {
        "actions_equal": bool(np.array_equal(intact_actions, restored_actions)),
        "saved_arrays_equal": io.equal(intact_after, second),
    }
    flip_branch = io.load(FlipFlop, flip_probe)
    flip_actions = np.stack([flip_branch.act(x) for x in rest])
    flip_anchor = last_executed(flip_lead, anchor)
    branches["flipflop"] = {
        "actions": flip_actions.tolist(),
        "anchor": flip_anchor.tolist(),
        "work": None,
        "score": score_window(flip_actions, flip_anchor, block),
    }
    random_actions = frozen["random/window"]
    branches["random"] = {
        "actions": random_actions.tolist(),
        "anchor": anchor.tolist(),
        "work": None,
        "score": score_window(random_actions, anchor, block),
    }
    if "timing_acceptance" in protocol:
        untaught = io.load(Brain, directory / "initial.npz")
        work = Work()
        lead = run_events(untaught, observations[:lead_count], work)
        untaught_anchor = last_executed(lead, frozen["window/cues"])
        actions = run_events(untaught, rest, work)
        ledger.extend({"stage": "window", "branch": "untaught", **r} for r in work.reports)
        branches["untaught"] = {
            "lead_actions": lead.tolist(),
            "actions": actions.tolist(),
            "anchor": untaught_anchor.tolist(),
            "work": work.summary(),
            "score": score_window(actions, untaught_anchor, block),
        }
    donor_anchor = anchor[permutation]
    branches["shuffled"]["donor_score"] = score_window(
        np.asarray(branches["shuffled"]["actions"]), donor_anchor, block
    )
    branches["shuffled"]["donor_anchor_differs"] = (donor_anchor != anchor).tolist()
    shuffled_actions = np.asarray(branches["shuffled"]["actions"])
    # A transplanted trace that reproduces the donor row's intact sequence shows that the
    # trace alone determines the trajectory and the warm neural state adds nothing.
    branches["shuffled"]["donor_identity"] = [
        bool(np.array_equal(shuffled_actions[:, r], intact_actions[:, permutation[r]]))
        for r in range(inputs.ROWS)
    ]
    identities = {
        "erased_equals_reset": bool(
            np.array_equal(branches["erased"]["actions"], branches["reset"]["actions"])
        ),
        "static_equals_static_cold": bool(
            np.array_equal(branches["static"]["actions"], branches["static_cold"]["actions"])
        ),
    }
    return {
        "cues": frozen["window/cues"].tolist(),
        "lead_actions": lead_actions.tolist(),
        "lead_work": lead_work.summary(),
        "anchor": anchor.tolist(),
        "cue_followed": (
            last_executed(lead_actions[:1], 1 - frozen["window/cues"]) == frozen["window/cues"]
        ).tolist(),
        "branches": branches,
        "continuation": continuation,
        "identities": identities,
        "probe": "probe.npz",
    }


def disturbance_stage(
    name: str,
    probe: Path,
    flip_probe: Path,
    anchor: np.ndarray,
    frozen: dict,
    protocol: dict,
    directory: Path,
    ledger: list,
    io: CheckpointIO | None = None,
) -> dict:
    """From the probe: pre events, the disturbance, post events; recovery and retention."""
    disturbances = protocol["disturbances"]
    io = CheckpointIO() if io is None else io
    pre, post = disturbances["pre"], disturbances["post"]
    observations = frozen[f"disturbance/{name}/observations"]
    kinds = frozen[f"disturbance/{name}/kinds"]
    middle = len(observations) - pre - post
    brain = io.load(Brain, probe)
    work = Work()
    pre_actions = run_events(brain, observations[:pre], work)
    custody = None
    if name == "pause2":
        first = run_events(brain, observations[pre : pre + 1], work)
        saved = io.save(brain, directory / f"{name}-mid.npz")
        twin = io.load(Brain, saved)
        twin_work = Work()
        remaining = observations[pre + 1 :]
        if "timing_acceptance" in protocol:
            io.save_world(
                {
                    "next_event": pre + 1,
                    "input_key": f"disturbance/{name}/observations",
                    "remaining_kinds": kinds[pre + 1 :].tolist(),
                },
                directory / f"{name}-world.json",
            )
            world = io.load_world(directory / f"{name}-world.json")
            assert world["remaining_kinds"] == kinds[world["next_event"] :].tolist()
            remaining = observations[world["next_event"] :]
        twin_actions = run_events(twin, remaining, twin_work)
        own_actions = run_events(brain, remaining, work)
        custody = {
            "actions_equal": bool(np.array_equal(own_actions, twin_actions)),
            "saved_arrays_equal": io.equal(
                io.save(brain, directory / f"{name}-after.npz"),
                io.save(twin, directory / f"{name}-twin-after.npz"),
            ),
            "twin_work": twin_work.summary(),
        }
        ledger.extend({"stage": name, "branch": "twin", **r} for r in twin_work.reports)
        during = np.concatenate([first, own_actions[: middle - 1]])
        post_actions = own_actions[middle - 1 :]
    else:
        during = run_events(brain, observations[pre : pre + middle], work)
        post_actions = run_events(brain, observations[pre + middle :], work)
    ledger.extend({"stage": name, "branch": "brain", **r} for r in work.reports)
    flip = io.load(FlipFlop, flip_probe)
    flip_actions = np.stack([flip.act(x) for x in observations])
    random_actions = frozen[f"random/disturbance/{name}"]
    block = protocol["window"]["block"]

    def scores(all_actions: np.ndarray, fallback: np.ndarray) -> dict:
        before = all_actions[:pre]
        after = all_actions[pre + middle :]
        hold = last_executed(before, fallback)
        continued = last_executed(all_actions[: pre + middle], fallback)
        return {
            "pre": score_window(before, fallback, block),
            "during_actions": all_actions[pre : pre + middle].tolist(),
            "post_hold": score_window(after, hold, block),
            "post_continue": score_window(after, continued, block),
            "recovery_events": recovery_events(after),
        }

    brain_all = np.concatenate([pre_actions, during, post_actions])
    return {
        "kinds": [inputs.KIND_NAMES[k] for k in kinds],
        "brain": {
            "actions": brain_all.tolist(),
            "work": work.summary(),
            **scores(brain_all, anchor),
        },
        "flipflop": {"actions": flip_actions.tolist(), **scores(flip_actions, anchor)},
        "random": {"actions": random_actions.tolist(), **scores(random_actions, anchor)},
        "custody": custody,
    }


def _burn(stop) -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 256))
    while not stop.is_set():
        a = np.tanh(a @ a / 256.0)


def real_time_run(brain: Brain, observations: np.ndarray, due_ms: np.ndarray, work: Work) -> dict:
    """Act at declared physical due times; record lateness, solve time and missed deadlines."""
    if len(observations) != len(due_ms):
        raise ValueError("one declared due time per event")
    start = time.perf_counter()
    actions, lateness, solve, starts, finishes = [], [], [], [], []
    for x, due in zip(observations, due_ms, strict=True):
        target = start + float(due) / 1000.0
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        began = time.perf_counter()
        answer = work.act(brain, x)
        finished = time.perf_counter()
        actions.append(np.full(inputs.ROWS, REFUSED) if answer is None else answer)
        lateness.append((began - target) * 1000.0)
        solve.append((finished - began) * 1000.0)
        starts.append((began - start) * 1000.0)
        finishes.append((finished - start) * 1000.0)
    distinct = np.unique(due_ms)
    interval = distinct[-1] - distinct[-2] if len(distinct) > 1 else 0
    # The deadline of an event is the next distinct due time; a doubled slot shares one.
    next_due = np.array(
        [
            distinct[distinct > due][0] if np.any(distinct > due) else distinct[-1] + interval
            for due in due_ms
        ]
    )
    missed = (np.asarray(finishes) > next_due).tolist()
    return {
        "actions": np.stack(actions).tolist(),
        "lateness_ms": [round(v, 3) for v in lateness],
        "solve_ms": [round(v, 3) for v in solve],
        "missed_deadlines": int(sum(missed)),
        "timing": {
            "due_ms": np.asarray(due_ms).tolist(),
            "begin_ms": starts,
            "end_ms": finishes,
            "deadline_ms": next_due.tolist(),
        },
        "wall_seconds": time.perf_counter() - start,
    }


def cadence_stage(
    probe: Path,
    anchor: np.ndarray,
    frozen: dict,
    protocol: dict,
    ledger: list,
    burners: int,
    io: CheckpointIO | None = None,
) -> dict:
    """The same events under different physical cadences, slot disturbances, solver budgets,
    tolerances and host load; per-event actions are compared with an unpaced reference."""
    cadence = protocol["cadence"]
    io = CheckpointIO() if io is None else io
    events = cadence["events"]
    observations = frozen["window/observations"][1 + protocol["window"]["lead"] :][:events]
    if len(observations) < events:
        raise ValueError("the window must supply the cadence events")
    block = protocol["window"]["block"]
    reference_brain = io.load(Brain, probe)
    reference_work = Work()
    reference_observations = observations
    if "timing_acceptance" in protocol:
        reference_observations = observations[0][None].repeat(events + 1, axis=0)
    reference = run_events(reference_brain, reference_observations, reference_work)
    ledger.extend({"stage": "cadence", "branch": "reference", **r} for r in reference_work.reports)
    variants: dict[str, dict] = {
        "reference": {
            "actions": reference.tolist(),
            "work": reference_work.summary(),
            "score": score_window(reference, anchor, block),
            "paced": False,
        },
    }

    def compare(actions: np.ndarray) -> dict:
        n = min(len(actions), len(reference))
        differing = int(np.sum(np.any(actions[:n] != reference[:n], axis=1)))
        return {
            "identical_to_reference": differing == 0 and len(actions) >= n,
            "differing_events": differing,
            "compared_events": n,
        }

    def paced(name: str, schedule: str, *, load: bool = False) -> None:
        slots = frozen[f"cadence/{schedule}/slot"]
        due = frozen[f"cadence/{schedule}/due_ms"]
        obs = observations[0][None].repeat(len(due), axis=0)
        branch = io.load(Brain, probe)
        work = Work()
        processes = []
        stop = None
        if load:
            context = multiprocessing.get_context("spawn")
            stop = context.Event()
            processes = [
                context.Process(target=_burn, args=(stop,), daemon=True) for _ in range(burners)
            ]
            for process in processes:
                process.start()
            time.sleep(0.5)
        try:
            result = real_time_run(branch, obs, due, work)
        finally:
            if stop is not None:
                stop.set()
                for process in processes:
                    process.join(timeout=5)
        ledger.extend({"stage": "cadence", "branch": name, **r} for r in work.reports)
        actions = np.asarray(result["actions"])
        slot_ideal = (anchor[None, :] + 1 + slots[:, None]) % inputs.ACTIONS
        valid = actions >= 0
        variants[name] = {
            **result,
            "work": work.summary(),
            "paced": True,
            "schedule": schedule,
            "burners": burners if load else 0,
            "score": score_window(actions, anchor, block),
            "slot_agreement": float(np.mean((actions == slot_ideal) & valid)),
            "slots": slots.tolist(),
            **compare(actions),
        }

    paced("paced_regular", "regular")
    for variant in protocol["event"]["cadence_variants_ms"]:
        paced(f"paced_regular{variant}", f"regular{variant}")
    paced("paced_extra", "extra")
    paced("paced_skipped", "skipped")
    paced("paced_regular_load", "regular", load=True)
    if "intervals_ms" in cadence:
        paced("paced_ordered", "ordered")
        paced("paced_shuffled_time", "shuffled_time")

    def numerical(branch: Brain, work: Work) -> dict:
        if "timing_acceptance" not in protocol:
            return {"actions": run_events(branch, observations, work).tolist(), "paced": False}
        due = frozen["cadence/regular/due_ms"]
        return {
            **real_time_run(branch, observations, due, work),
            "paced": True,
            "schedule": "regular",
            "slots": frozen["cadence/regular/slot"].tolist(),
            "burners": 0,
        }

    for budget in cadence["budgets"]:
        branch = io.load(Brain, probe)
        branch.learner.config = replace(branch.learner.config, free_steps=int(budget))
        work = Work()
        measured = numerical(branch, work)
        actions = np.asarray(measured["actions"])
        ledger.extend({"stage": "cadence", "branch": f"budget_{budget}", **r} for r in work.reports)
        variants[f"budget_{budget}"] = {
            **measured,
            "work": work.summary(),
            "free_steps": int(budget),
            "score": score_window(actions, anchor, block),
            **compare(actions),
        }
    for tolerance in cadence["tolerances"]:
        branch = io.load(Brain, probe)
        branch.learner.config = replace(branch.learner.config, tolerance=float(tolerance))
        work = Work()
        measured = numerical(branch, work)
        actions = np.asarray(measured["actions"])
        ledger.extend(
            {"stage": "cadence", "branch": f"tolerance_{tolerance}", **r} for r in work.reports
        )
        variants[f"tolerance_{tolerance}"] = {
            **measured,
            "work": work.summary(),
            "tolerance": float(tolerance),
            "score": score_window(actions, anchor, block),
            **compare(actions),
        }
    return {"anchor": anchor.tolist(), "events": events, "variants": variants}


# -- one founder


def run_founder(
    seed: int,
    arm: str,
    recipe: str,
    frozen: dict,
    protocol: dict,
    directory: Path,
    ledger: list,
    *,
    cadence: bool,
    burners: int,
) -> dict:
    began = time.perf_counter()
    ledger_start = len(ledger)
    io = CheckpointIO()
    directory.mkdir(parents=True)
    brain = make_brain(seed, protocol, recipe)
    flip = FlipFlop(protocol["controls_config"]["flipflop_rate"])
    io.save(brain, directory / "initial.npz")
    teaching_work = Work()
    teaching = teach_life(brain, flip, frozen, arm, teaching_work)
    ledger.extend({"stage": "teaching", "branch": "brain", **r} for r in teaching_work.reports)
    io.save(brain, directory / "taught.npz")
    window = window_stage(brain, flip, frozen, protocol, directory, ledger, io)
    probe, flip_probe = directory / "probe.npz", directory / "probe-flipflop.npz"
    anchor = np.asarray(window["anchor"], dtype=np.int64)
    disturbances = {
        name: disturbance_stage(
            name, probe, flip_probe, anchor, frozen, protocol, directory, ledger, io
        )
        for name in [f"pause{k}" for k in protocol["disturbances"]["pauses"]] + ["distractor"]
    }
    result = {
        "seed": seed,
        "arm": arm,
        "recipe": recipe,
        "teaching": teaching,
        "window": window,
        "disturbances": disturbances,
        "cadence": None,
    }
    if cadence:
        result["cadence"] = cadence_stage(probe, anchor, frozen, protocol, ledger, burners, io)
    for report in ledger[ledger_start:]:
        report.update(seed=seed, recipe=recipe, arm=arm)
    result["checkpoint_io"] = io.operations
    result["elapsed_seconds"] = time.perf_counter() - began
    result["control_work"] = {
        "flipflop_teacher_rows": flip.lessons,
        "flipflop_action_rows": inputs.ROWS
        * (
            len(teaching["flipflop_actions"])
            + len(window["lead_actions"])
            + len(window["branches"]["flipflop"]["actions"])
            + sum(len(d["flipflop"]["actions"]) for d in disturbances.values())
        ),
        "random_action_rows": inputs.ROWS
        * (
            len(window["branches"]["random"]["actions"])
            + sum(len(d["random"]["actions"]) for d in disturbances.values())
        ),
    }
    return result


def aggregate(runs: list[dict]) -> dict:
    """Means over every founder, with failed, refused and capped founders in the denominator."""
    out: dict = {}
    for recipe, arm in sorted({(run["recipe"], run["arm"]) for run in runs}):
        selected = [run for run in runs if run["arm"] == arm and run["recipe"] == recipe]
        controls = {}
        for name in CONTROLS:
            scores = [run["window"]["branches"][name]["score"] for run in selected]
            controls[name] = _mean_scores(scores)
        donor = [run["window"]["branches"]["shuffled"]["donor_score"] for run in selected]
        controls["shuffled_donor"] = _mean_scores(donor)
        disturbances = {}
        for name in selected[0]["disturbances"]:
            disturbances[name] = {
                model: {
                    "post_hold": _mean_scores(
                        [run["disturbances"][name][model]["post_hold"] for run in selected]
                    ),
                    "post_continue": _mean_scores(
                        [run["disturbances"][name][model]["post_continue"] for run in selected]
                    ),
                    "recovered_rows": int(
                        sum(
                            sum(v == 0 for v in run["disturbances"][name][model]["recovery_events"])
                            for run in selected
                        )
                    ),
                    "rows": int(inputs.ROWS * len(selected)),
                }
                for model in ("brain", "flipflop", "random")
            }
        cadence = None
        if all(run["cadence"] is not None for run in selected):
            cadence = {}
            for name in selected[0]["cadence"]["variants"]:
                items = [run["cadence"]["variants"][name] for run in selected]
                cadence[name] = {
                    "identical_to_reference": int(
                        sum(item.get("identical_to_reference", True) for item in items)
                    ),
                    "founders": len(items),
                    "refusals": int(sum(item["score"]["refusals"] for item in items)),
                    "mean_alternation_rate": _mean(
                        [item["score"]["alternation_rate"] for item in items]
                    ),
                    "mean_sweeps_per_event": _mean(
                        [
                            item["work"]["action_sweeps"] / max(1, item["work"]["action_attempts"])
                            for item in items
                        ]
                    ),
                    "mean_solve_ms": _mean(
                        [np.mean(item["solve_ms"]) for item in items if "solve_ms" in item]
                    ),
                    "max_lateness_ms": max(
                        [max(item["lateness_ms"]) for item in items if "lateness_ms" in item],
                        default=None,
                    ),
                    "missed_deadlines": int(sum(item.get("missed_deadlines", 0) for item in items)),
                    "mean_slot_agreement": _mean(
                        [item["slot_agreement"] for item in items if "slot_agreement" in item]
                    ),
                }
        out[f"{recipe}/{arm}"] = {
            "recipe": recipe,
            "arm": arm,
            "founders": len(selected),
            "teaching_last_bout_agreement": _mean(
                [run["teaching"]["last_bout_agreement"] for run in selected]
            ),
            "teaching_lessons": int(sum(run["teaching"]["lessons_attempted"] for run in selected)),
            "cue_followed_rows": int(sum(sum(run["window"]["cue_followed"]) for run in selected)),
            "shuffled_donor_identity_rows": int(
                sum(
                    sum(run["window"]["branches"]["shuffled"]["donor_identity"]) for run in selected
                )
            ),
            "erased_equals_reset_founders": int(
                sum(run["window"]["identities"]["erased_equals_reset"] for run in selected)
            ),
            "continuation_equal": all(
                run["window"]["continuation"]["actions_equal"]
                and run["window"]["continuation"]["saved_arrays_equal"]
                for run in selected
            ),
            "pause_custody_equal": all(
                run["disturbances"]["pause2"]["custody"]["actions_equal"]
                and run["disturbances"]["pause2"]["custody"]["saved_arrays_equal"]
                for run in selected
                if "pause2" in run["disturbances"]
            ),
            "controls": controls,
            "disturbances": disturbances,
            "cadence": cadence,
        }
    return out


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _mean_scores(scores: list[dict]) -> dict:
    return {
        "alternation_rate": _mean([s["alternation_rate"] for s in scores]),
        "agreement": _mean([s["agreement"] for s in scores]),
        "period": _mean([s["period"] for s in scores]),
        "repeats": int(sum(s["repeats"] for s in scores)),
        "refusals": int(sum(s["refusals"] for s in scores)),
        "blocks": [
            _mean([s["blocks"][i] for s in scores]) for i in range(len(scores[0]["blocks"]))
        ],
        "founders": len(scores),
    }


# -- custody


def _verify(directory: Path) -> tuple[bool, str]:
    path = directory / "summary.json"
    raw = json.loads(path.read_text())
    sources = [(item["path"], directory / item["path"]) for item in raw["source"]["files"]]

    def check(body: dict) -> str | None:
        for name, expected in body["artifacts"].items():
            if sha256(directory / name) != expected:
                return f"artifact differs: {name}"
        if body["protocol_sha256"] != sha256(directory / "protocol.json"):
            return "protocol copy differs from the recorded hash"
        if "timing_acceptance" in body["protocol"]:
            timing_acceptance.verify_body(body, directory)
        return None

    valid, reason = Receipt.verify(path, sources=sources, check=check)
    return valid, "canonical form, digest, sources and artifact hashes agree" if valid else reason


def verify(directory: Path) -> tuple[bool, str]:
    try:
        return _verify(directory)
    except (OSError, ValueError, KeyError, AssertionError, TypeError, IndexError) as error:
        return False, f"cannot verify artifact: {error}"


def apply_overrides(protocol: dict, args: argparse.Namespace) -> dict:
    """Smoke and development overrides; any override marks the run as not frozen."""
    overrides = {}
    if args.bouts is not None:
        protocol["teaching"]["bouts"] = args.bouts
        overrides["bouts"] = args.bouts
    if args.events_per_bout is not None:
        protocol["teaching"]["events_per_bout"] = args.events_per_bout
        overrides["events_per_bout"] = args.events_per_bout
    if args.window is not None:
        protocol["window"]["events"] = args.window
        overrides["window"] = args.window
    if args.post is not None:
        protocol["disturbances"]["post"] = args.post
        overrides["post"] = args.post
    if args.cadence_events is not None:
        protocol["cadence"]["events"] = args.cadence_events
        overrides["cadence_events"] = args.cadence_events
        if protocol["cadence"]["disturbed_slot"] >= args.cadence_events - 1:
            protocol["cadence"]["disturbed_slot"] = args.cadence_events // 2
            overrides["disturbed_slot"] = protocol["cadence"]["disturbed_slot"]
    recipes = [value for value in protocol["recipes"].values() if isinstance(value, dict)]
    for recipe in recipes:
        if args.decay is not None:
            recipe["working_memory_decay"] = args.decay
            overrides["working_memory_decay"] = args.decay
        if args.amplitude is not None:
            recipe["working_memory_amplitude"] = args.amplitude
            overrides["working_memory_amplitude"] = args.amplitude
        if args.eta is not None:
            recipe["learning"]["eta"] = args.eta
            recipe["learning"]["eta_bias"] = args.eta / 10.0
            overrides["eta"] = args.eta
        if args.normalize is not None:
            recipe["learning"]["normalize"] = args.normalize
            overrides["normalize"] = args.normalize
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--verify", type=Path)
    parser.add_argument("--protocol", type=Path, default=inputs.PROTOCOL_PATH)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="default: the protocol's confirmation seeds",
    )
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--recipes", nargs="+", default=None)
    parser.add_argument("--no-cadence", action="store_true")
    parser.add_argument("--burners", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--time-cap", type=float, default=None)
    parser.add_argument("--bouts", type=int)
    parser.add_argument("--events-per-bout", type=int)
    parser.add_argument("--window", type=int)
    parser.add_argument("--post", type=int)
    parser.add_argument("--cadence-events", type=int)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--normalize", type=float)
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    protocol, protocol_sha = inputs.load_protocol(args.protocol)
    args.arms = args.arms or protocol.get("run_arms", ["every", "mismatch"])
    args.recipes = args.recipes or protocol.get("run_recipes", ["selected", "compose_default"])
    overrides = apply_overrides(protocol, args)
    seeds = list(protocol["seeds"]["confirmation"]) if args.seeds is None else list(args.seeds)
    if (
        len(seeds) != len(set(seeds))
        or any(seed < 0 for seed in seeds)
        or any(arm not in ("every", "mismatch") for arm in args.arms)
        or len(args.arms) != len(set(args.arms))
        or any(not isinstance(protocol["recipes"].get(name), dict) for name in args.recipes)
        or len(args.recipes) != len(set(args.recipes))
        or args.burners < 1
        or any(
            value is not None and value < 1
            for value in (
                args.bouts,
                args.events_per_bout,
                args.window,
                args.post,
                args.cadence_events,
            )
        )
        or (args.events_per_bout is not None and args.events_per_bout < 2)
        or (args.window is not None and args.window < protocol["window"]["block"])
        or (args.decay is not None and not 0 <= args.decay < 1)
        or (
            args.amplitude is not None and not (np.isfinite(args.amplitude) and args.amplitude >= 0)
        )
        or (args.eta is not None and not (np.isfinite(args.eta) and args.eta > 0))
        or (args.normalize is not None and not 0 <= args.normalize < 1)
    ):
        parser.error("invalid seeds, arms, burners or overrides")
    if not args.no_cadence and protocol["cadence"]["events"] > protocol["window"]["events"]:
        parser.error("cadence events must fit inside the window")
    if "timing_acceptance" in protocol:
        for name, actual, expected in (
            ("seeds", seeds, protocol["seeds"]["confirmation"]),
            ("recipes", args.recipes, protocol["run_recipes"]),
            ("arms", args.arms, protocol["run_arms"]),
            ("cadence_runs", not args.no_cadence, True),
            ("burners", args.burners, max(1, os.cpu_count() or 2)),
        ):
            if actual != expected:
                overrides[name] = actual
    cap = float(protocol["caps"]["seconds"] if args.time_cap is None else args.time_cap)
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    protocol_copy = args.out / "protocol.json"
    protocol_copy.write_bytes(Path(args.protocol).read_bytes())
    declaration = {
        "schema": SCHEMA,
        "instrument_revision": 2,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": not overrides,
        "overrides": overrides,
        "seeds": seeds,
        "arms": list(args.arms),
        "recipes": list(args.recipes),
        "cadence_runs": not args.no_cadence,
        "burners": args.burners,
        "command": [Path(__file__).name, *(sys.argv[1:] if argv is None else argv)],
        "cadence": cadence.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "cadence_import": str(Path(cadence.__file__).resolve()),
    }
    (args.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    sources, origins = [], []
    package = Path(cadence.__file__).resolve().parent
    own = [
        Path(__file__).resolve(),
        Path(inputs.__file__).resolve(),
        Path(timing_acceptance.__file__).resolve(),
    ]
    for source in [*own, *sorted(package.rglob("*.py"))]:
        relative = (
            "source/" + source.name
            if source in own
            else "source/cadence/" + source.relative_to(package).as_posix()
        )
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources.append((relative, target))
        origins.append((source, target))
    frozen = {
        seed: inputs.freeze_inputs(
            args.out / f"inputs-seed{seed}.npz", seed=seed, protocol=protocol
        )
        for seed in seeds
    }
    runs: list[dict] = []
    ledger: list[dict] = []
    capped = False
    planned = [
        (seed, recipe, arm) for seed in seeds for recipe in args.recipes for arm in args.arms
    ]
    for seed, recipe, arm in planned:
        if time.perf_counter() - began > cap:
            capped = True
            break
        directory = args.out / f"seed{seed}-{recipe}-{arm}"
        result = run_founder(
            seed,
            arm,
            recipe,
            frozen[seed],
            protocol,
            directory,
            ledger,
            cadence=not args.no_cadence
            and (
                "timing_acceptance" not in protocol
                or (
                    recipe == protocol["timing_acceptance"]["primary_recipe"]
                    and arm == protocol["timing_acceptance"]["primary_arm"]
                )
            ),
            burners=args.burners,
        )
        runs.append(result)
        with (args.out / "runs.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        window = result["window"]["branches"]
        print(
            json.dumps(
                {
                    "seed": seed,
                    "recipe": recipe,
                    "arm": arm,
                    "intact_alternation": window["intact"]["score"]["alternation_rate"],
                    "intact_agreement": window["intact"]["score"]["agreement"],
                    "flipflop_alternation": window["flipflop"]["score"]["alternation_rate"],
                    "seconds": round(time.perf_counter() - began, 1),
                }
            ),
            flush=True,
        )
    with (args.out / "reports.jsonl").open("w") as handle:
        for report in ledger:
            handle.write(json.dumps(report) + "\n")
    completed = [(run["seed"], run["recipe"], run["arm"]) for run in runs]
    assert all(sha256(original) == sha256(copy) for original, copy in origins), (
        "source changed during the run; retain this incomplete attempt and rerun"
    )
    artifacts = {
        path.relative_to(args.out).as_posix(): sha256(path)
        for path in sorted(args.out.rglob("*"))
        if path.is_file()
        and "source" not in path.relative_to(args.out).parts
        and path.name != "summary.json"
    }
    summary = {
        "declaration": declaration,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": not overrides,
        "runs": runs,
        "planned_founders": [list(item) for item in planned],
        "completed_founders": [list(item) for item in completed],
        "capped": capped,
        "seconds": time.perf_counter() - began,
        "aggregate": aggregate(runs) if runs else {},
        "artifacts": artifacts,
        "work_scope": "Every teacher and free solve, control fork, disturbance, paced run and "
        "continuation check is charged. calls_seconds exclude checkpoint IO; "
        "run seconds include IO. No reward, eligibility or associative writes "
        "occur. Sweeps are not joules.",
    }
    if "timing_acceptance" in protocol:
        summary["timing_acceptance"] = timing_acceptance.evaluate(
            runs, protocol, declaration, capped
        )
    Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(
        json.dumps({"verified": valid, "capped": capped, "summary": str(args.out / "summary.json")})
    )
    return 3 if capped else 0


if __name__ == "__main__":
    raise SystemExit(main())
