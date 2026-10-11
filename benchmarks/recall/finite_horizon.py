"""The finite continuing recall chamber of issue 84, roadmap row 02: can one continuing
``Brain.compose`` remember one of two observed tokens after its cue disappears, at a fixed
state size, through its working trace rather than a supplied answer or a warm numerical
guess? The reviewed protocol is ``FINITE_HORIZON_PROTOCOL.md``; the frozen inputs are built by
``finite_horizon_inputs.py`` before any brain runs; ``protocol-finite.json`` declares every
setting, the founders, the caps and the gates.

Arms, one continuing life each per founder, on the same frozen episodes:

- ``vanished``  the declared recipe: trace decay 0.8, amplitude 1, the trace's write fixed,
                its read learned at the QUERY lessons only; it never sees a history coordinate;
- ``default``   the same brain with the composed working-trace defaults (amplitude 3,
                decay 0.2): the simpler setting, retained as the control of the declared gene;
- ``history``   the external-history comparator: byte-equal initial arrays, the same lessons,
                and the actual observed payloads of the last four WRITE events appended at QUERY
                by the world; supplied assistance, counted separately;
- ``random``    the frozen uniform-random actions of every row.

Revision 3 retains raw accepted-event trace transitions, paired neural/trace separation,
motor margins, history transport and both event-time forks. Each founder runs in a bounded
child process. A deadline kills and reaps that worker, preserves completed journals and
leaves all unfinished planned rows in the denominator. The byte cap is checked between
operations; neither a capped nor a failed worker can pass closure.

At every evaluation query the vanished and default lives are saved and forked: intact (the
life continues), erased trace, shuffled trace (transplanted from the paired row with the
opposite value) and full reset. The supported horizon is the largest contiguous passed prefix
over 0, 1, 2 intervening events under the prewritten gates; nuisance, replacement, order,
continuation, purity and resource gates must also pass. A refusal is wrong; unrun planned rows
are wrong. Run ``python benchmarks/recall/finite_horizon.py --out DIR``; verify with
``--verify DIR``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(name, "1")

import numpy as np  # noqa: E402  (after the thread environment is set)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_horizon_inputs as inputs  # noqa: E402

import cadence  # noqa: E402
from cadence import Brain, LearnerConfig, Receipt  # noqa: E402
from cadence.learning import LearningPhaseError  # noqa: E402

SCHEMA = "finite-recall/1"
INSTRUMENT_REVISION = 3
TRACE_FIELDS = (
    "before_trace",
    "before_last",
    "before_cold",
    "after_trace",
    "after_last",
    "after_cold",
    "activation",
    "decay",
)
DIAGNOSTIC_FIELDS = ("trace", "neural", "motor", "potential")
PROTOCOL_PATH = Path(__file__).with_name("protocol-finite.json")
ARMS = ("vanished", "default", "history", "random")
CONTROLS = ("intact", "erased", "shuffled", "reset")
CONDITIONS = {c.name: c for c in inputs.TEST_CONDITIONS}
PREFIX = (("clean-0",), ("clean-1", "distractor-1"), ("clean-2", "distractor-2"))
NUISANCE = ("partial-1", "noise-1", "replacement-1", "order-latest-1")
REQUIRED = ("seeds", "brain", "learning", "arms", "training", "evaluation", "caps", "gates")
UNMET_PROTOCOL_OBLIGATIONS: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}")
    if protocol.get("schema") != SCHEMA or protocol["arms"] != list(ARMS):
        raise ValueError("unsupported protocol schema or arm census")
    for phase, conditions in (
        ("training", inputs.TRAIN_CONDITIONS),
        ("evaluation", inputs.TEST_CONDITIONS),
    ):
        count = protocol[phase]["repeats"]
        if (
            type(count) is not int
            or count < 1
            or protocol[phase]["episodes"] != count * len(conditions)
        ):
            raise ValueError(f"{phase} repeats and episode census disagree")
    if protocol["training"].get("rule", "every") not in ("every", "surprise"):
        raise ValueError("training.rule must be 'every' or 'surprise'")
    if any(not np.isfinite(value) or value <= 0 for value in protocol["caps"].values()):
        raise ValueError("caps must be finite and positive")
    return protocol, hashlib.sha256(raw).hexdigest()


# -- brains


def make_brain(protocol: dict, seed: int, *, decay: float, amplitude: float) -> Brain:
    genes = protocol["brain"]
    learning = LearnerConfig(**protocol["learning"])
    return Brain.compose(
        inputs.INPUTS,
        2,
        modules=tuple(genes["modules"]),
        lateral=genes["lateral"],
        seed=seed,
        learning=learning,
        episodic=False,
        working_memory_decay=decay,
        working_memory_amplitude=amplitude,
        sensory_scale=float(genes.get("sensory_scale", 1.0)),
    )


def brain_for(arm: str, protocol: dict, seed: int) -> Brain:
    genes = protocol["brain"]
    if arm == "default":
        return make_brain(
            protocol, seed, decay=genes["default_decay"], amplitude=genes["default_amplitude"]
        )
    return make_brain(
        protocol, seed, decay=genes["trace_decay"], amplitude=genes["trace_amplitude"]
    )


# -- work


@dataclass
class Work:
    """Actual solver work, with the trace audit; sweeps are not joules."""

    action_attempts: int = 0
    action_refusals: int = 0
    action_sweeps: int = 0
    action_row_sweeps: int = 0
    action_residual_checks: int = 0
    action_row_residual_checks: int = 0
    expected_refusals: int = 0
    teacher_attempts: int = 0
    teacher_refusals: int = 0
    teacher_presentations: int = 0
    teacher_sweeps: int = 0
    teacher_row_sweeps: int = 0
    teacher_residual_checks: int = 0
    teacher_row_residual_checks: int = 0
    imagined_phases: int = 0
    imagined_refusals: int = 0
    imagined_sweeps: int = 0
    imagined_row_sweeps: int = 0
    imagined_residual_checks: int = 0
    imagined_row_residual_checks: int = 0
    trace_row_updates: int = 0
    trace_audits: int = 0
    trace_audit_max_error: float = 0.0
    retained_trace_records: int = 0
    trace_chunks: int = 0
    checkpoints: int = 0
    checkpoint_seconds: float = 0.0
    calls_seconds: float = 0.0
    reports: list[dict] = field(default_factory=list, repr=False)
    journal: Path | None = field(default=None, repr=False)
    trace_directory: Path | None = field(default=None, repr=False)
    trace_sink: Work | None = field(default=None, repr=False)
    pending_traces: list[dict] = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.repr}

    def retain_trace(self, before: dict, after: dict, h: np.ndarray, decay: float) -> None:
        if self.trace_sink is not None:
            self.trace_sink.retain_trace(before, after, h, decay)
            return
        if self.trace_directory is None:
            return
        if self.pending_traces and self.pending_traces[-1]["activation"].shape != h.shape:
            self.flush_traces()
        self.pending_traces.append(
            {
                **{f"before_{k}": np.asarray(v).copy() for k, v in before.items()},
                **{f"after_{k}": np.asarray(v).copy() for k, v in after.items()},
                "activation": h.copy(),
                "decay": np.asarray(decay),
            }
        )
        if len(self.pending_traces) >= 128:
            self.flush_traces()

    def flush_traces(self) -> None:
        if not self.pending_traces:
            return
        assert self.trace_directory is not None
        self.trace_directory.mkdir(exist_ok=True)
        path = self.trace_directory / f"chunk-{self.trace_chunks:05d}.npz"
        np.savez_compressed(path, **pack_traces(self.pending_traces))
        count = len(self.pending_traces)
        self.trace_chunks += 1
        self.retained_trace_records += count
        self.record(
            "trace_chunk",
            {
                "path": str(path.relative_to(self.trace_directory.parent)),
                "count": count,
                "encoding": "stacked-trace/1",
            },
        )
        self.pending_traces.clear()

    def record(self, operation: str, report: dict) -> None:
        entry = {"operation": operation, **report}
        self.reports.append(entry)
        if self.journal is not None:
            from cadence.receipts import canonical_json

            with self.journal.open("a") as stream:
                stream.write(canonical_json(entry) + "\n")

    def merge(self, other: Work) -> None:
        for name, value in other.summary().items():
            setattr(
                self,
                name,
                max(getattr(self, name), value)
                if name == "trace_audit_max_error"
                else getattr(self, name) + value,
            )
        self.reports.extend(other.reports)

    def save(self, brain: Brain, path: Path) -> Path:
        started = time.perf_counter()
        result = brain.save(path)
        elapsed = time.perf_counter() - started
        self.checkpoint_seconds += elapsed
        self.checkpoints += 1
        self.record("save", {"path": path.name, "io_seconds": elapsed})
        return result

    def load(self, path: Path) -> Brain:
        started = time.perf_counter()
        result = Brain.load(path)
        elapsed = time.perf_counter() - started
        self.checkpoint_seconds += elapsed
        self.checkpoints += 1
        self.record("load", {"path": path.name, "io_seconds": elapsed})
        return result

    def imagine(self, brain: Brain, observations: list[np.ndarray]) -> bool:
        start = time.perf_counter()
        try:
            phases = brain.imagine(observations)
        finally:
            elapsed = time.perf_counter() - start
            self.calls_seconds += elapsed
        for index, phase in enumerate(phases):
            rows = int(np.size(phase.residual))
            refused = not bool(np.all(phase.qualified))
            self.imagined_phases += 1
            self.imagined_refusals += int(refused)
            self.imagined_sweeps += phase.steps
            self.imagined_row_sweeps += rows * phase.steps
            self.imagined_residual_checks += phase.residual_checks
            self.imagined_row_residual_checks += rows * phase.residual_checks
            self.record(
                "imagine",
                {
                    "steps": phase.steps,
                    "rows": rows,
                    "residual_checks": phase.residual_checks,
                    "refused": refused,
                    "seconds": elapsed if index == 0 else 0.0,
                },
            )
        return len(phases) == len(observations) and all(np.all(p.qualified) for p in phases)

    def act(
        self, brain: Brain, x: np.ndarray, *, audit: bool = True, expected_refusal: bool = False
    ) -> np.ndarray | None:
        """One greedy act of every row; a refused act is a missed answer. The trace's update
        is checked against the literal source recurrence of the protocol."""
        trace = brain.working_memory
        before = (
            None
            if trace is None
            else {name: getattr(trace, name).copy() for name in ("trace", "last", "cold")}
        )
        if before is not None and len(before["trace"]) != len(x):
            before = {
                "trace": np.zeros((len(x), len(trace.hidden))),
                "last": np.zeros((len(x), len(trace.hidden))),
                "cold": np.ones(len(x), dtype=bool),
            }
        start = time.perf_counter()
        self.action_attempts += 1
        prior = brain.last_settlement
        try:
            answer = brain.act(x, greedy=True)
        except RuntimeError:
            report = brain.last_settlement
            if report is None or report is prior or report["qualified"]:
                raise
            self.action_refusals += 1
            self.expected_refusals += int(expected_refusal)
            answer = None
        finally:
            elapsed = time.perf_counter() - start
            self.calls_seconds += elapsed
        report = dict(brain.last_settlement)
        self.record(
            "act",
            {**report, "rows": len(x), "expected_refusal": expected_refusal, "seconds": elapsed},
        )
        self.action_sweeps += int(report["steps"])
        self.action_row_sweeps += len(x) * int(report["steps"])
        self.action_residual_checks += int(report["residual_checks"])
        self.action_row_residual_checks += len(x) * int(report["residual_checks"])
        if answer is not None:
            self.trace_row_updates += len(x)
            if before is not None and len(before["trace"]) == len(x):
                state = brain.basal_ganglia.state
                assert state is not None and trace is not None
                h = np.atleast_2d(np.asarray(state.activation))[:, brain.association_index]
                after = {name: getattr(trace, name) for name in ("trace", "last", "cold")}
                error = audit_trace(before, after, h, trace.decay)
                self.trace_audits += 1
                self.trace_audit_max_error = max(self.trace_audit_max_error, error)
                self.record("trace_audit", {"error": error})
                self.retain_trace(before, after, h, trace.decay)
        return answer

    def teach(
        self, brain: Brain, x: np.ndarray, labels: np.ndarray, *, drive: np.ndarray | None = None
    ) -> bool:
        """One lesson on the rows of ``labels``; ``drive`` is the stimulus the rows' free act
        read when the lesson follows that act (the trace has moved on since)."""
        start = time.perf_counter()
        self.teacher_attempts += 1
        accepted = True
        try:
            _, report = brain.learner.step(brain.stimulus(x) if drive is None else drive, labels)
        except LearningPhaseError as error:
            self.teacher_refusals += 1
            report = error.report
            accepted = False
        finally:
            elapsed = time.perf_counter() - start
            self.calls_seconds += elapsed
        self.record("teach", {**report, "seconds": elapsed})
        self.teacher_presentations += int(report["attempted_presentations"])
        self.teacher_sweeps += int(report["total_steps"])
        self.teacher_row_sweeps += int(report["total_row_sweeps"])
        self.teacher_residual_checks += int(report["total_residual_checks"])
        self.teacher_row_residual_checks += int(report["total_row_residual_checks"])
        return accepted


def audit_trace(before: dict, after: dict, h: np.ndarray, decay: float) -> float:
    """An independent literal recurrence for the arm's declared decay, ``decay * trace +
    (1 - decay) * h``; the declared recipe's decay of 0.8 is also checked by the reviewed
    fixture's own function, which hard-codes that recipe."""
    expected = decay * np.asarray(before["trace"]) + (1.0 - decay) * np.asarray(h)
    error = float(np.max(np.abs(np.asarray(after["trace"]) - expected)))
    if not np.array_equal(after["last"], h) or np.asarray(after["cold"]).any():
        raise ValueError("last/cold do not record exactly one admitted real event")
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError("trace does not match one declared accepted-event update")
    if decay == 0.8:
        error = max(error, inputs.check_trace_transition(before, after, h))
    return error


def pack_traces(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Lossless leading event axis; compression shares repeated values across events."""
    if not records or any(set(r) != set(TRACE_FIELDS) for r in records):
        raise ValueError("trace records lack the exact declared fields")
    for name in TRACE_FIELDS:
        first = records[0][name]
        if any(r[name].shape != first.shape or r[name].dtype != first.dtype for r in records):
            raise ValueError("trace packing cannot change shapes or promote dtypes")
    return {name: np.stack([record[name] for record in records]) for name in TRACE_FIELDS}


def unpack_traces(arrays: Any, count: int) -> list[dict[str, np.ndarray]]:
    if set(arrays) != set(TRACE_FIELDS):
        raise ValueError("packed trace fields differ from the declared schema")
    packed = {name: arrays[name] for name in TRACE_FIELDS}
    if any(value.ndim < 1 or len(value) != count for value in packed.values()):
        raise ValueError("packed trace event census differs")
    return [
        {name: np.asarray(packed[name][index]) for name in TRACE_FIELDS} for index in range(count)
    ]


def same_arrays(first: Path, second: Path) -> bool:
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        return set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


# -- the frozen episodes


def episode_arrays(frozen: dict[str, np.ndarray], phase: str, index: int) -> dict[str, np.ndarray]:
    return {
        name: frozen[f"{phase}/{index:04d}/{name}"]
        for name in (
            "observations",
            "labels",
            "token_values",
            "permutation",
            "uniform_actions",
            "timestamps",
            "irregular_timestamps",
        )
    }


def condition_name(frozen: dict[str, np.ndarray], phase: str, index: int) -> str:
    conditions = inputs.TRAIN_CONDITIONS if phase == "train" else inputs.TEST_CONDITIONS
    return conditions[int(frozen[f"{phase}/condition"][index])].name


# -- one arm's life


class Capped(RuntimeError):
    pass


def write_progress(path: Path, value: dict) -> None:
    """Only complete JSON snapshots replace the last durable progress record."""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def history_traffic(observations: np.ndarray) -> dict:
    """Bytes in the world's actual four-payload history buffer; never teacher labels."""
    writes = int(np.count_nonzero(observations[:, :, inputs.WRITE] == 1))
    queries = int(np.count_nonzero(observations[:, :, inputs.QUERY].any(axis=2)))
    return {
        "buffer_bytes": inputs.STREAMS * (16 * 8 + 8),
        "input_copy_bytes": int(observations.nbytes),
        "payload_bytes_read": writes * 4 * 8,
        "payload_bytes_written": writes * 4 * 8,
        "query_bytes_transported": queries * 16 * 8,
    }


def query_diagnostics(brain: Brain, labels: np.ndarray) -> dict:
    """Recorded after an evaluation act; labels only measure its motor margin."""
    state = brain.basal_ganglia.state
    assert state is not None and brain.working_memory is not None
    neural = np.atleast_2d(np.asarray(state.activation))
    trace = brain.working_memory.trace
    motor = neural[:, brain.motor_index]
    result = diagnostics_from_arrays(trace, neural, motor, labels)
    potential = np.atleast_2d(np.asarray(state.v))
    result["potential"] = potential.tolist()
    result["paired_potential_distance"] = np.linalg.norm(
        potential[::2] - potential[1::2], axis=1
    ).tolist()
    return result


def diagnostics_from_arrays(
    trace: np.ndarray, neural: np.ndarray, motor: np.ndarray, labels: np.ndarray
) -> dict:
    def pairs(x):
        return np.linalg.norm(x[::2] - x[1::2], axis=1).tolist()

    return {
        "trace": trace.tolist(),
        "neural": neural.tolist(),
        "motor": motor.tolist(),
        "paired_trace_distance": pairs(trace),
        "paired_neural_distance": pairs(neural),
        "motor_margin": (
            motor[np.arange(len(labels)), labels] - motor[np.arange(len(labels)), 1 - labels]
        ).tolist(),
    }


def retain_diagnostics(directory: Path, index: int, diagnostic: dict) -> dict:
    """Keep raw vectors once, outside repeated progress/receipt JSON."""
    target = directory / "query-diagnostics" / f"{index:04d}.npz"
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        target, **{name: np.asarray(diagnostic[name]) for name in DIAGNOSTIC_FIELDS}
    )
    return {
        **{k: v for k, v in diagnostic.items() if k not in DIAGNOSTIC_FIELDS},
        "raw": {
            "encoding": "query-arrays/1",
            "path": target.relative_to(directory).as_posix(),
            "sha256": sha256(target),
        },
    }


def check_caps(began: float, directory: Path, caps: dict) -> None:
    if time.time() - began > caps["seconds"]:
        raise Capped("the worker's wall cap was reached")
    root = directory.parent if directory.name in ARMS else directory
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if size > caps["bytes"]:
        raise Capped("the retained output cap was reached")


def train_arm(
    arm: str,
    brain: Brain,
    frozen: dict,
    protocol: dict,
    began: float,
    directory: Path,
    *,
    work: Work | None = None,
    progress: dict | None = None,
) -> dict:
    """The frozen training episodes: greedy acts and, at every QUERY, under the ``every``
    rule one lesson before the act (recall/1 and recall/2); under the ``surprise`` rule the
    act first and a lesson only on the rows it answered wrong, on the drive that act read, so
    that a right answer is routine and teaches nothing (recall/3)."""
    work = Work() if work is None else work
    count = int(len(frozen["train/condition"]))
    refused_life = False
    rule = str(protocol["training"].get("rule", "every"))
    if rule not in ("every", "surprise"):
        raise ValueError("training.rule must be 'every' or 'surprise'")
    progress = {} if progress is None else progress
    progress.update(episodes=count, completed_episodes=0, completed=False, rule=rule, queries=[])
    if arm == "history":
        progress["history_traffic"] = []
    for index in range(count):
        check_caps(began, directory, protocol["caps"])
        arrays = episode_arrays(frozen, "train", index)
        observations = arrays["observations"]
        if arm == "history":
            observations = inputs.append_observed_history(observations)
            progress["history_traffic"].append(history_traffic(arrays["observations"]))
        for event, x in enumerate(observations):
            check_caps(began, directory, protocol["caps"])
            progress["current_event"] = [index, event]
            query = event == len(observations) - 1
            if query:
                reading = {
                    "index": index,
                    "answer": None,
                    "labels": arrays["labels"].tolist(),
                    "lesson_rows": [],
                    "lesson_accepted": None,
                }
                progress["queries"].append(reading)
            if query and rule == "every":
                reading["lesson_rows"] = list(range(inputs.STREAMS))
                reading["lesson_accepted"] = work.teach(brain, x, arrays["labels"])
            drive = brain.stimulus(x) if query and rule == "surprise" else None
            answer = work.act(brain, x)
            if query:
                reading["answer"] = None if answer is None else answer.tolist()
            if answer is None:
                refused_life = True
                break
            if drive is not None:
                wrong = np.asarray(answer) != np.asarray(arrays["labels"])
                if wrong.any():
                    reading["lesson_rows"] = np.flatnonzero(wrong).tolist()
                    reading["lesson_accepted"] = work.teach(
                        brain, x, np.asarray(arrays["labels"])[wrong], drive=drive[wrong]
                    )
        if refused_life:
            write_progress(directory / "training-progress.json", progress)
            break
        progress["completed_episodes"] += 1
        write_progress(directory / "training-progress.json", progress)
    progress.update(completed=not refused_life, work=work.summary())
    write_progress(directory / "training-progress.json", progress)
    return progress


def fork_answers(checkpoint: Path, x: np.ndarray, permutation: np.ndarray, work: Work) -> dict:
    """Every control begins at the same complete pre-query state, with learning disabled."""
    out = {}
    for control in CONTROLS[1:]:
        branch = work.load(checkpoint)
        trace = branch.working_memory
        assert trace is not None
        if control == "erased":
            trace.reset(len(x))
        elif control == "shuffled":
            for name in ("trace", "last", "cold"):
                setattr(trace, name, getattr(trace, name)[permutation].copy())
        elif control == "reset":
            branch.reset()
        answer = work.act(branch, x, audit=False)
        out[control] = None if answer is None else answer.tolist()
    return out


def seam_checks(
    brain: Brain,
    observations: np.ndarray,
    cue_index: int,
    replacement_index: int | None,
    directory: Path,
    work: Work,
    protocol: dict,
) -> dict:
    """Saved seams of one episode: after the first cue, before a replacement when present, and
    before QUERY, each resumed by a clone that must execute the remaining observations with equal
    answers and equal complete saved arrays; private imagination and a refused act must leave a
    checkpoint unchanged; the brain never sees a timestamp."""
    from dataclasses import replace

    seams = {}
    saved: dict[str, Path] = {}
    stops = {"after_cue": cue_index + 1, "before_query": len(observations) - 1}
    if replacement_index is not None:
        stops["before_replacement"] = replacement_index
    answers: list[list[int] | None] = []
    for event, x in enumerate(observations):
        for name, stop in stops.items():
            if event == stop:
                saved[name] = work.save(brain, directory / f"seam-{name}.npz")
        answer = work.act(brain, x)
        answers.append(None if answer is None else answer.tolist())
        if answer is None:
            break
    final = work.save(brain, directory / "seam-final.npz")
    reference_reports = [report for report in work.reports if report["operation"] == "act"][
        -len(answers) :
    ]
    for name, stop in stops.items():
        if name not in saved:
            continue
        clone = work.load(saved[name])
        clone_work = Work(journal=work.journal, trace_sink=work)
        clone_answers = []
        for x in observations[stop:]:
            answer = clone_work.act(clone, x, audit=False)
            clone_answers.append(None if answer is None else answer.tolist())
            if answer is None:
                break
        resumed = work.save(clone, directory / f"seam-{name}-resumed.npz")
        seams[name] = {
            "answers_equal": clone_answers == answers[stop:],
            "reports_equal": [
                {k: v for k, v in r.items() if k != "seconds"}
                for r in clone_work.reports
                if r["operation"] == "act"
            ]
            == [{k: v for k, v in r.items() if k != "seconds"} for r in reference_reports[stop:]],
            "saved_arrays_equal": same_arrays(final, resumed),
            "work": clone_work.summary(),
        }
        work.merge(clone_work)
    # private imagination and a refused act from the pre-query seam leave it unchanged
    if "before_query" not in saved:
        return {
            "seams": seams,
            "imagine_leaves_checkpoint": False,
            "refused_act_leaves_checkpoint": False,
            "continuation_after_refusal_equal": False,
        }
    pre = saved["before_query"]
    twin = work.load(pre)
    imagined = work.imagine(twin, [observations[-1], observations[-1]])
    after_imagine = work.save(twin, directory / "seam-after-imagine.npz")
    unchanged_imagine = same_arrays(pre, after_imagine)
    twin = work.load(pre)
    kept = twin.learner.config
    twin.learner.config = replace(kept, free_steps=0, tolerance=1e-15)
    changed = observations[-1].copy()
    changed[:, inputs.PAYLOAD] = 1.0  # a changed cue the brain never trained on
    try:
        refused = work.act(twin, changed, audit=False, expected_refusal=True) is None
    finally:
        twin.learner.config = kept
    after_refusal = work.save(twin, directory / "seam-after-refusal.npz")
    unchanged_refusal = same_arrays(pre, after_refusal)
    twin_answer = work.act(twin, observations[-1], audit=False)
    continues = twin_answer is not None and twin_answer.tolist() == answers[-1]
    return {
        "seams": seams,
        "imagine_leaves_checkpoint": imagined and unchanged_imagine,
        "refused_act_leaves_checkpoint": refused and unchanged_refusal,
        "continuation_after_refusal_equal": continues,
    }


def evaluate_arm(
    arm: str,
    brain: Brain,
    frozen: dict,
    protocol: dict,
    began: float,
    directory: Path,
    *,
    work: Work | None = None,
    progress: dict | None = None,
) -> dict:
    """The 24 frozen evaluation episodes per condition: forks at QUERY for the brain arms with a
    trace, the plain answer for the history arm; seams on the first episode of every condition."""
    work = Work() if work is None else work
    count = int(len(frozen["test/condition"]))
    trials: list[dict] = []
    seams: dict[str, dict] = {}
    progress = {} if progress is None else progress
    progress.update(completed=False, trials=trials, seams=seams, timing=None)
    completed = True
    seen: set[str] = set()
    for index in range(count):
        check_caps(began, directory, protocol["caps"])
        arrays = episode_arrays(frozen, "test", index)
        name = condition_name(frozen, "test", index)
        observations = arrays["observations"]
        if arm == "history":
            observations = inputs.append_observed_history(observations)
        trial: dict[str, Any] = {
            "condition": name,
            "index": index,
            "labels": arrays["labels"].tolist(),
            "transplanted": arrays["labels"][arrays["permutation"]].tolist(),
        }
        if arm == "history":
            trial["history_traffic"] = history_traffic(arrays["observations"])
        if name not in seen:
            seen.add(name)
            # the seam episode runs on a loaded clone of the life so the life's own answer at
            # this episode is read from the life itself afterwards
            clone = work.load(work.save(brain, directory / "life-before-seams.npz"))
            writes = [i for i, x in enumerate(observations) if x[0, inputs.WRITE] == 1]
            replacement = writes[1] if CONDITIONS[name].opposite and len(writes) > 1 else None
            seam_directory = directory / f"seams-{name}"
            seam_directory.mkdir()
            seams[name] = seam_checks(
                clone, observations, writes[0], replacement, seam_directory, work, protocol
            )
            (directory / "life-before-seams.npz").unlink()
        answers: list[list[int] | None] = []
        for event, x in enumerate(observations[:-1]):
            check_caps(began, directory, protocol["caps"])
            progress["current_event"] = [index, event]
            answer = work.act(brain, x)
            answers.append(None if answer is None else answer.tolist())
            if answer is None:
                completed = False
                break
        if not completed:
            trial["branches"] = (
                {control: None for control in CONTROLS} if arm != "history" else {"answer": None}
            )
            trials.append(trial)
            write_progress(directory / "evaluation-progress.json", progress)
            break
        query = observations[-1]
        progress["current_event"] = [index, len(observations) - 1]
        if arm == "history":
            answer = work.act(brain, query)
            trial["branches"] = {"answer": None if answer is None else answer.tolist()}
        else:
            checkpoint = work.save(brain, directory / f"query-{index:04d}.npz")
            branches = fork_answers(checkpoint, query, arrays["permutation"], work)
            answer = work.act(brain, query)
            branches["intact"] = None if answer is None else answer.tolist()
            trial["branches"] = branches
            checkpoint.unlink()
        if answer is not None:
            trial["diagnostics"] = retain_diagnostics(
                directory, index, query_diagnostics(brain, arrays["labels"])
            )
        trial["random"] = arrays["uniform_actions"].tolist()
        trials.append(trial)
        write_progress(directory / "evaluation-progress.json", progress)
        if answer is None:
            completed = False
            break
    # the timestamp fork: identical events under two schedules must give equal arrays; the
    # brain never sees a timestamp, and this is recorded rather than assumed
    timing = None
    if arm in ("vanished", "default") and completed:
        index = next(i for i in range(count) if condition_name(frozen, "test", i) == "clean-2")
        arrays = episode_arrays(frozen, "test", index)
        observations = arrays["observations"]
        saves = []
        timing_work = []
        cue_clone = work.load(work.save(brain, directory / "timing-base.npz"))
        cue_end = next(i + 1 for i, x in enumerate(observations) if x[0, inputs.WRITE] == 1)
        cue_qualified = True
        for x in observations[:cue_end]:
            if work.act(cue_clone, x) is None:
                cue_qualified = False
                break
        cue_path = work.save(cue_clone, directory / "timing-cue.npz")
        for schedule in ("timestamps", "irregular_timestamps"):
            clone = work.load(cue_path)
            clone_work = Work(journal=work.journal, trace_sink=work)
            for x in observations[cue_end:] if cue_qualified else []:
                if clone_work.act(clone, x, audit=False) is None:
                    break
            saves.append(work.save(clone, directory / f"timing-{schedule}.npz"))
            timing_work.append(clone_work.summary())
            work.merge(clone_work)
        timing = {
            "schedules": ["regular", "irregular"],
            "equal_arrays": same_arrays(*saves)
            and cue_qualified
            and all(w["action_refusals"] == 0 for w in timing_work),
            "cue_qualified": cue_qualified,
            "work": timing_work,
            "elapsed_regular": float(arrays["timestamps"][-1]),
            "elapsed_irregular": float(arrays["irregular_timestamps"][-1]),
        }
        timing["matched_elapsed_events"] = event_count_fork(brain, observations, directory, work)
    work.save(brain, directory / "final.npz")
    check_caps(began, directory, protocol["caps"])
    progress.update(completed=completed, timing=timing, work=work.summary())
    write_progress(directory / "evaluation-progress.json", progress)
    return progress


def event_count_fork(brain: Brain, observations: np.ndarray, directory: Path, work: Work) -> dict:
    """Same saved post-cue state, one/two neutral events ending at the same physical time.
    Time is metadata only. This diagnostic measures event-count sensitivity without a new gate.
    """
    clone = work.load(work.save(brain, directory / "event-base.npz"))
    writes = [i for i, x in enumerate(observations) if x[0, inputs.WRITE] == 1]
    admitted = True
    for x in observations[: writes[-1] + 1]:
        if work.act(clone, x) is None:
            admitted = False
            break
    cue = work.save(clone, directory / "event-cue.npz")
    branches = []
    for count in (1, 2):
        branch = work.load(cue)
        accepted = admitted
        for _ in range(count):
            if work.act(branch, observations[-2]) is None:
                accepted = False
                break
        pre_query_trace = branch.working_memory.trace.copy()
        answer = work.act(branch, observations[-1]) if accepted else None
        branches.append(
            {
                "events": count,
                "timestamps": np.linspace(0, 2, count + 1).tolist(),
                "elapsed": 2.0,
                "qualified": answer is not None,
                "answer": None if answer is None else answer.tolist(),
                "pre_query_trace": pre_query_trace.tolist(),
            }
        )
    return {
        "branches": branches,
        "trace_distance": float(
            np.linalg.norm(
                np.asarray(branches[0]["pre_query_trace"])
                - np.asarray(branches[1]["pre_query_trace"])
            )
        ),
    }


def run_random(frozen: dict) -> dict:
    trials = []
    for index in range(int(len(frozen["test/condition"]))):
        arrays = episode_arrays(frozen, "test", index)
        trials.append(
            {
                "condition": condition_name(frozen, "test", index),
                "index": index,
                "labels": arrays["labels"].tolist(),
                "branches": {"answer": arrays["uniform_actions"].tolist()},
            }
        )
    return {"completed": True, "trials": trials, "work": Work().summary()}


# -- scores and gates


def accuracy(answers: list[list[int] | None], labels: list[list[int]], planned: int) -> dict:
    correct = attempted = 0
    for a, lab in zip(answers, labels, strict=True):
        if a is None:
            continue
        attempted += len(lab)
        correct += int(sum(int(x == y) for x, y in zip(a, lab, strict=True)))
    return {
        "correct": correct,
        "attempted": attempted,
        "planned": planned,
        "accuracy": correct / planned if planned else None,
        "accuracy_attempted": correct / attempted if attempted else None,
    }


def paired(
    answers: list[list[int] | None], labels: list[list[int]], planned_pairs: int
) -> float | None:
    both = 0
    for a, lab in zip(answers, labels, strict=True):
        if a is None:
            continue
        for i in range(0, len(lab), 2):
            both += int(a[i] == lab[i] and a[i + 1] == lab[i + 1])
    return both / planned_pairs if planned_pairs else None


def score_arm(result: dict, repeats: int, arm: str | None = None) -> dict:
    """Per condition: accuracy over planned rows for every branch, paired accuracy for the
    intact answer, and for the shuffled branch the accuracy against the transplanted value."""
    out = {}
    for name in CONDITIONS:
        trials = [t for t in result["trials"] if t["condition"] == name]
        planned = repeats * inputs.STREAMS
        labels = [t["labels"] for t in trials]
        entry: dict[str, Any] = {"planned_rows": planned, "attempted_episodes": len(trials)}
        if arm in ("history", "random") or (trials and "answer" in trials[0]["branches"]):
            answers = [t["branches"]["answer"] for t in trials]
            entry["answer"] = accuracy(answers, labels, planned)
            entry["paired"] = paired(answers, labels, repeats * inputs.STREAMS // 2)
        else:
            for control in CONTROLS:
                answers = [t["branches"].get(control) for t in trials]
                entry[control] = accuracy(answers, labels, planned)
            entry["paired"] = paired(
                [t["branches"].get("intact") for t in trials], labels, repeats * inputs.STREAMS // 2
            )
            entry["shuffled_vs_transplanted"] = accuracy(
                [t["branches"].get("shuffled") for t in trials],
                [t["transplanted"] for t in trials],
                planned,
            )
        out[name] = entry
    return out


def gates_for(founder: dict, protocol: dict) -> dict:
    g = protocol["gates"]
    scores = founder["scores"]
    vanished, history, random = scores["vanished"], scores["history"], scores["random"]

    def passes(name: str) -> dict:
        v, h, r = vanished[name], history[name], random[name]
        delay = CONDITIONS[name].delay
        checks = {
            "intact": v["intact"]["accuracy"] is not None
            and v["intact"]["accuracy"] >= g["intact"],
            "paired": v["paired"] is not None and v["paired"] >= g["paired"],
            "history": h["answer"]["accuracy"] is not None
            and h["answer"]["accuracy"] >= g["history"],
            "above_random": v["intact"]["accuracy"] is not None
            and v["intact"]["accuracy"] >= r["answer"]["accuracy"] + g["random_margin"],
            "complete": v["intact"]["attempted"] == v["intact"]["planned"]
            and v["intact"]["accuracy"] is not None
            and all(v[control]["attempted"] == v[control]["planned"] for control in CONTROLS)
            and h["answer"]["attempted"] == h["answer"]["planned"],
        }
        if delay > 0:
            checks["erased"] = (
                v["erased"]["accuracy"] <= g["lesion"]
                and v["intact"]["accuracy"] >= v["erased"]["accuracy"] + g["lesion_margin"]
            )
            checks["reset"] = (
                v["reset"]["accuracy"] <= g["lesion"]
                and v["intact"]["accuracy"] >= v["reset"]["accuracy"] + g["lesion_margin"]
            )
            checks["shuffled"] = (
                v["shuffled_vs_transplanted"]["accuracy"] >= g["shuffled_transplanted"]
                and v["shuffled"]["accuracy"] <= g["shuffled_original"]
            )
        return {"passed": all(checks.values()), **checks}

    report = {name: passes(name) for name in CONDITIONS}
    horizon = -1
    for depth, names in enumerate(PREFIX):
        if all(report[name]["passed"] for name in names):
            horizon = depth
        else:
            break
    nuisance = all(report[name]["passed"] for name in NUISANCE)

    def custody_for(arm: str) -> bool:
        seams = founder["arms"][arm].get("seams", {})
        return set(seams) == set(CONDITIONS) and all(
            {"after_cue", "before_query"} <= set(entry["seams"])
            and (not CONDITIONS[name].opposite or "before_replacement" in entry["seams"])
            and all(
                s["answers_equal"] and s["saved_arrays_equal"] and s.get("reports_equal", False)
                for s in entry["seams"].values()
            )
            and entry["imagine_leaves_checkpoint"]
            and entry["refused_act_leaves_checkpoint"]
            and entry["continuation_after_refusal_equal"]
            for name, entry in seams.items()
        )

    custody = all(custody_for(arm) for arm in ARMS[:-1])
    timing = founder["arms"]["vanished"].get("timing")
    audit = all(
        stage.get("work", {}).get("trace_audit_max_error", float("inf")) <= g["trace_audit"]
        and stage.get("work", {}).get("trace_audits", 0) > 0
        and stage["work"].get("retained_trace_records") == stage["work"]["trace_audits"]
        for arm in ARMS[:-1]
        for stage in (founder["arms"][arm], founder["arms"][arm].get("training", {}))
    )
    refusals = all(
        stage.get("work", {}).get("action_refusals", 0)
        == stage.get("work", {}).get("expected_refusals", 0)
        and stage.get("work", {}).get("imagined_refusals", 0) == 0
        and stage.get("work", {}).get("teacher_refusals", 0) == 0
        for arm in ARMS[:-1]
        for stage in (founder["arms"][arm], founder["arms"][arm].get("training", {}))
    )
    complete = all(
        founder["arms"][arm]["completed"]
        and founder["arms"][arm].get("training", {}).get("completed", arm == "random")
        for arm in ARMS
    )
    diagnostics = (
        complete
        and all(
            "diagnostics" in trial for arm in ARMS[:-1] for trial in founder["arms"][arm]["trials"]
        )
        and all(
            founder["arms"][arm].get("timing") is not None
            and len(
                founder["arms"][arm]["timing"].get("matched_elapsed_events", {}).get("branches", [])
            )
            == 2
            and all(
                b["qualified"]
                for b in founder["arms"][arm]["timing"]["matched_elapsed_events"]["branches"]
            )
            for arm in ("vanished", "default")
        )
    )
    guarded = founder.get("hard_guard", {}).get("completed", False) and founder.get(
        "hard_guard", {}
    ).get("reaped", False)
    return {
        "conditions": report,
        "horizon": horizon,
        "nuisance": nuisance,
        "custody": custody,
        "timing": timing is not None and timing["equal_arrays"],
        "trace_audit": audit,
        "no_refusals": refusals,
        "capped": founder.get("capped", False),
        "complete": complete,
        "diagnostics": diagnostics,
        "hard_guard": guarded,
        "protocol_obligations_complete": not founder.get("unmet_protocol_obligations", []),
        "closure": horizon >= g["minimum_horizon"]
        and nuisance
        and custody
        and (timing is not None and timing["equal_arrays"])
        and audit
        and refusals
        and complete
        and diagnostics
        and guarded
        and founder.get("history_initial_equal", False) is True
        and not founder.get("unmet_protocol_obligations", [])
        and not founder.get("capped", False),
    }


# -- a founder


def run_founder(seed: int, protocol: dict, directory: Path, repeats: dict) -> dict:
    directory.mkdir(parents=True)
    began = time.time()
    effective = repeats or {
        "train": protocol["training"]["repeats"],
        "test": protocol["evaluation"]["repeats"],
    }
    frozen = freeze_small(directory / "episodes.npz", seed, effective)
    founder: dict[str, Any] = {
        "seed": seed,
        "arms": {},
        "capped": False,
        "unmet_protocol_obligations": list(UNMET_PROTOCOL_OBLIGATIONS),
    }
    accounting: dict[str, tuple[Work, Work]] = {}
    for arm in protocol["arms"]:
        founder["arms"][arm] = {
            "completed": False,
            "trials": [],
            "seams": {},
            "timing": None,
            "work": Work().summary(),
        }
        if arm != "random":
            founder["arms"][arm]["training"] = {"completed": False, "work": Work().summary()}
    # The baseline census survives an interrupted learner.
    founder["arms"]["random"] = run_random(frozen)
    try:
        for arm in protocol["arms"]:
            if arm == "random":
                continue
            check_caps(began, directory, protocol["caps"])
            brain = brain_for(arm, protocol, seed)
            arm_dir = directory / arm
            arm_dir.mkdir()
            train_work, eval_work = (
                Work(
                    journal=arm_dir / "training-calls.jsonl",
                    trace_directory=arm_dir / "training-traces",
                ),
                Work(
                    journal=arm_dir / "evaluation-calls.jsonl",
                    trace_directory=arm_dir / "evaluation-traces",
                ),
            )
            accounting[arm] = train_work, eval_work
            result = founder["arms"][arm]
            train_work.save(brain, arm_dir / "initial.npz")
            training = train_arm(
                arm,
                brain,
                frozen,
                protocol,
                began,
                arm_dir,
                work=train_work,
                progress=result["training"],
            )
            train_work.save(brain, arm_dir / "trained.npz")
            train_work.flush_traces()
            if training["completed"]:
                evaluate_arm(
                    arm, brain, frozen, protocol, began, arm_dir, work=eval_work, progress=result
                )
            result["training"]["work"] = train_work.summary()
            eval_work.flush_traces()
            result["work"] = eval_work.summary()
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "trained": training["completed"],
                        "evaluated": result["completed"],
                        "seconds": round(time.time() - began, 1),
                    }
                ),
                flush=True,
            )
    except Capped as error:
        founder["capped"] = True
        founder["cap_reason"] = str(error)
        founder["incomplete_work"] = (
            "Completed calls retained; interrupted current stage is incomplete."
        )
    finally:
        for arm, (train_work, eval_work) in accounting.items():
            train_work.flush_traces()
            eval_work.flush_traces()
            founder["arms"][arm]["training"]["work"] = train_work.summary()
            founder["arms"][arm]["work"] = eval_work.summary()
    if (
        sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        > protocol["caps"]["bytes"]
    ):
        founder["capped"] = True
        founder["cap_reason"] = "the retained output cap was reached"
    test_repeats = int(effective["test"])
    founder["scores"] = {
        arm: score_arm(founder["arms"][arm], test_repeats, arm)
        for arm in founder["arms"]
        if "trials" in founder["arms"][arm]
    }
    founder["seconds"] = round(time.time() - began, 1)
    # the initial arrays of the vanished and history brains are byte-equal
    founder["history_initial_equal"] = (
        same_arrays(directory / "vanished" / "initial.npz", directory / "history" / "initial.npz")
        if (directory / "history" / "initial.npz").exists()
        and (directory / "vanished" / "initial.npz").exists()
        else None
    )
    founder["gates"] = gates_for(founder, protocol)
    return founder


def guarded_process(command: list[str], log: Path, seconds: float) -> dict:
    """One worker, a hard wall deadline, and kill/wait custody even on interruption."""
    started = time.monotonic()
    timed_out = False
    with log.open("w") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name == "posix"
        )
        try:
            try:
                child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if child.poll() is None:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
    return {
        "completed": not timed_out and child.returncode == 0,
        "timed_out": timed_out,
        "exit_code": child.returncode,
        "limit_seconds": seconds,
        "elapsed_seconds": time.monotonic() - started,
        "pid": child.pid,
        "reaped": True,
    }


def recover_founder(seed: int, protocol: dict, directory: Path, repeats: dict) -> dict:
    """Recover completed journals only; a killed call has unknown unfinished work.
    Every originally planned row remains in the denominator, including absent arms.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays = episode_fixture(seed, repeats)
    episode_path = directory / "episodes.npz"
    reconstructed = not episode_path.exists()
    if not reconstructed:
        try:
            with np.load(episode_path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[k], v) for k, v in arrays.items()
                ):
                    raise ValueError("unfinished input archive")
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            episode_path.rename(directory / "episodes-incomplete.npz")
            reconstructed = True
    if reconstructed:
        np.savez_compressed(episode_path, **arrays)
    founder = {
        "seed": seed,
        "arms": {},
        "capped": True,
        "seconds": None,
        "unmet_protocol_obligations": list(UNMET_PROTOCOL_OBLIGATIONS),
        "incomplete_work": "Worker failed or was terminated. Only completed durable calls "
        "are counted; current-call work and unflushed trace records are unknown.",
        "inputs_reconstructed_after_termination": reconstructed,
        "history_initial_equal": None,
    }
    for arm in ARMS[:-1]:
        base = directory / arm
        result = {"completed": False, "trials": [], "seams": {}, "timing": None}
        training = {"completed": False}
        for phase, target in (("training", training), ("evaluation", result)):
            path = base / f"{phase}-progress.json"
            if path.exists():
                target.update(json.loads(path.read_text()))
            target["work"] = work_from_journal(base / f"{phase}-calls.jsonl", incomplete=True)
        result["training"] = training
        founder["arms"][arm] = result
    founder["arms"]["random"] = run_random(arrays)
    founder["scores"] = {
        arm: score_arm(result, repeats["test"], arm) for arm, result in founder["arms"].items()
    }
    return founder


def guarded_founder(
    seed: int, protocol: dict, directory: Path, repeats: dict, protocol_path: Path
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(directory),
        "--protocol",
        str(protocol_path.resolve()),
        "--worker-seed",
        str(seed),
        "--repeats",
        str(repeats["train"]),
        str(repeats["test"]),
    ]
    guard = guarded_process(
        command, directory.parent / f"founder-{seed}-worker.log", protocol["caps"]["seconds"]
    )
    result_path = directory / "worker-result.json"
    if guard["completed"] and result_path.exists():
        founder = json.loads(result_path.read_text())
    else:
        founder = recover_founder(seed, protocol, directory, repeats)
    founder["hard_guard"] = guard
    founder["gates"] = gates_for(founder, protocol)
    write_progress(directory / "guard.json", guard)
    return founder


def freeze_small(path: Path, seed: int, repeats: dict) -> dict[str, np.ndarray]:
    """Freeze the explicitly resolved repeat counts, with the original seeding."""
    arrays = episode_fixture(seed, repeats)
    np.savez_compressed(path, **arrays)
    return arrays


def episode_fixture(seed: int, repeats: dict) -> dict[str, np.ndarray]:
    """Deterministic pure-input reconstruction, also used by the receipt verifier."""
    arrays = {}
    for phase, count, conditions, salt in (
        ("train", int(repeats["train"]), inputs.TRAIN_CONDITIONS, 0),
        ("test", int(repeats["test"]), inputs.TEST_CONDITIONS, 1),
    ):
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
        order = np.tile(np.arange(len(conditions)), count)
        if phase == "train":
            rng.shuffle(order)
        arrays[phase + "/condition"] = order
        for index, selected in enumerate(order):
            episode = inputs.make_episode(rng, conditions[selected])
            for name in episode.__dataclass_fields__:
                arrays[f"{phase}/{index:04d}/{name}"] = getattr(episode, name)
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


# -- custody


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the declared inputs or recorded events")


def retained_bytes(directory: Path, seed: int) -> int:
    """Conservatively charge shared sources and the complete receipt to each founder."""
    own = {f"founder-{seed}", f"founder-{seed}-worker.log"}
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
        and (
            not path.relative_to(directory).parts[0].startswith("founder-")
            or path.relative_to(directory).parts[0] in own
        )
    )


def journal_rows(journal: Path, *, incomplete: bool = False) -> list[dict]:
    if not journal.exists():
        return []
    lines = journal.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if not incomplete or index != len(lines) - 1:
                raise
    return rows


def work_from_journal(journal: Path, *, incomplete: bool = False) -> dict:
    expected = Work().summary()
    rows = journal_rows(journal, incomplete=incomplete)
    for row in rows:
        operation = row["operation"]
        elapsed = row.get("seconds", 0.0)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid completed call duration")
        expected["calls_seconds"] += elapsed
        if operation in ("save", "load"):
            expected["checkpoints"] += 1
            expected["checkpoint_seconds"] += row["io_seconds"]
        elif operation == "act":
            refused = not row["qualified"]
            expected["action_attempts"] += 1
            expected["action_refusals"] += int(refused)
            expected["expected_refusals"] += int(refused and row["expected_refusal"])
            expected["action_sweeps"] += row["steps"]
            expected["action_row_sweeps"] += row["rows"] * row["steps"]
            expected["action_residual_checks"] += row["residual_checks"]
            expected["action_row_residual_checks"] += row["rows"] * row["residual_checks"]
            expected["trace_row_updates"] += 0 if refused else row["rows"]
        elif operation == "teach":
            expected["teacher_attempts"] += 1
            expected["teacher_refusals"] += int(not row["accepted"])
            for target, source in (
                ("presentations", "attempted_presentations"),
                ("sweeps", "total_steps"),
                ("row_sweeps", "total_row_sweeps"),
                ("residual_checks", "total_residual_checks"),
                ("row_residual_checks", "total_row_residual_checks"),
            ):
                expected["teacher_" + target] += int(row[source])
        elif operation == "imagine":
            expected["imagined_phases"] += 1
            expected["imagined_refusals"] += int(row["refused"])
            expected["imagined_sweeps"] += row["steps"]
            expected["imagined_row_sweeps"] += row["rows"] * row["steps"]
            expected["imagined_residual_checks"] += row["residual_checks"]
            expected["imagined_row_residual_checks"] += row["rows"] * row["residual_checks"]
        elif operation == "trace_audit":
            expected["trace_audits"] += 1
            expected["trace_audit_max_error"] = max(expected["trace_audit_max_error"], row["error"])
        elif operation == "trace_chunk":
            expected["trace_chunks"] += 1
            expected["retained_trace_records"] += row["count"]
        else:
            raise ValueError("unknown completed-call operation")
    return expected


def audit_work(work: dict, journal: Path, *, incomplete: bool = False) -> None:
    expected = work_from_journal(journal, incomplete=incomplete)
    for key, value in expected.items():
        if key not in ("calls_seconds", "checkpoint_seconds"):
            require_equal(work[key], value, f"work.{key}")
        elif not np.isclose(work[key], value, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{key} differs from completed journal")
    count, max_error = 0, 0.0
    for row in journal_rows(journal, incomplete=incomplete):
        if row["operation"] != "trace_chunk":
            continue
        require_equal(row["encoding"], "stacked-trace/1", "trace encoding")
        with np.load(journal.parent / row["path"], allow_pickle=False) as arrays:
            for record in unpack_traces(arrays, row["count"]):
                before = {k: record[f"before_{k}"] for k in ("trace", "last", "cold")}
                after = {k: record[f"after_{k}"] for k in ("trace", "last", "cold")}
                error = audit_trace(before, after, record["activation"], float(record["decay"]))
                max_error = max(max_error, error)
                count += 1
    require_equal(count, work["retained_trace_records"], "retained trace count")
    if not incomplete:
        require_equal(count, work["trace_audits"], "complete raw trace census")
        require_equal(
            count, work["action_attempts"] - work["action_refusals"], "admitted event census"
        )
        require_equal(max_error, work["trace_audit_max_error"], "raw trace replay error")


def audit_current(body: dict, directory: Path) -> None:
    declaration = json.loads((directory / "declaration.json").read_text())
    require_equal(body["declaration"], declaration, "declaration copy")
    require_equal(declaration["trace_encoding"], "stacked-trace/1", "declared trace encoding")
    require_equal(declaration["query_encoding"], "query-arrays/1", "declared query encoding")
    protocol, digest = load_protocol(directory / "protocol.json")
    require_equal(body["protocol"], protocol, "protocol body")
    require_equal(declaration["protocol_sha256"], digest, "declared protocol hash")
    seeds = declaration["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("invalid founder census")
    repeats = declaration["repeats"] or {
        "train": protocol["training"]["repeats"],
        "test": protocol["evaluation"]["repeats"],
    }
    if set(repeats) != {"train", "test"} or any(
        type(n) is not int or n < 1 for n in repeats.values()
    ):
        raise ValueError("invalid repeat census")
    require_equal(declaration["effective_repeats"], repeats, "effective repeats")
    frozen = declaration["repeats"] is None and not declaration["seed_override"]
    if frozen:
        require_equal(seeds, protocol["seeds"]["founders"], "frozen founders")
    require_equal(declaration["frozen_protocol"], frozen, "frozen declaration")
    require_equal(body["frozen_protocol"], frozen, "frozen body")
    require_equal([f["seed"] for f in body["founders"]], seeds, "founder census")
    require_equal(
        body["unmet_protocol_obligations"], list(UNMET_PROTOCOL_OBLIGATIONS), "protocol scope"
    )
    for founder in body["founders"]:
        if (
            retained_bytes(directory, founder["seed"]) > protocol["caps"]["bytes"]
            and not founder["capped"]
        ):
            raise ValueError("retained byte cap exceeded without failed resource admission")
        require_equal(
            founder["unmet_protocol_obligations"], list(UNMET_PROTOCOL_OBLIGATIONS), "founder scope"
        )
        require_equal(sorted(founder["arms"]), sorted(ARMS), "arm census")
        base = directory / f"founder-{founder['seed']}"
        guard = json.loads((base / "guard.json").read_text())
        require_equal(founder["hard_guard"], guard, "worker guard custody")
        require_equal(guard["limit_seconds"], protocol["caps"]["seconds"], "worker deadline")
        if not guard["reaped"] or (
            guard["completed"] and (guard["timed_out"] or guard["exit_code"] != 0)
        ):
            raise ValueError("inconsistent worker completion")
        incomplete = not guard["completed"]
        if incomplete and (not founder["capped"] or not founder.get("incomplete_work")):
            raise ValueError("terminated worker lacks explicit incomplete custody")
        arrays = episode_fixture(founder["seed"], repeats)
        with np.load(base / "episodes.npz", allow_pickle=False) as stored:
            if set(stored.files) != set(arrays) or any(
                not np.array_equal(stored[name], values) for name, values in arrays.items()
            ):
                raise ValueError("frozen input arrays differ from the declared seed/repeats")
        expected_scores = {}
        for arm, result in founder["arms"].items():
            trials = result["trials"]
            require_equal([t["index"] for t in trials], list(range(len(trials))), "trial census")
            if len(trials) > len(arrays["test/condition"]):
                raise ValueError("extra evaluation trials")
            for trial in trials:
                index = trial["index"]
                episode = episode_arrays(arrays, "test", index)
                require_equal(
                    trial["condition"], condition_name(arrays, "test", index), "condition"
                )
                require_equal(trial["labels"], episode["labels"].tolist(), "trial labels")
                keys = {"answer"} if arm in ("history", "random") else set(CONTROLS)
                require_equal(set(trial["branches"]), keys, "control census")
                for answer in trial["branches"].values():
                    if answer is not None and (
                        len(answer) != inputs.STREAMS or any(a not in (0, 1) for a in answer)
                    ):
                        raise ValueError("invalid answer rows")
                if arm == "random":
                    require_equal(
                        trial["branches"]["answer"],
                        episode["uniform_actions"].tolist(),
                        "random baseline",
                    )
                elif arm != "history":
                    require_equal(
                        trial["transplanted"],
                        episode["labels"][episode["permutation"]].tolist(),
                        "transplanted labels",
                    )
                if arm != "random" and "diagnostics" in trial:
                    d = trial["diagnostics"]
                    require_equal(d["raw"]["encoding"], "query-arrays/1", "query encoding")
                    require_equal(
                        d["raw"]["path"],
                        f"query-diagnostics/{index:04d}.npz",
                        "query diagnostic identity",
                    )
                    diagnostic_path = base / arm / d["raw"]["path"]
                    require_equal(
                        sha256(diagnostic_path), d["raw"]["sha256"], "query diagnostic hash"
                    )
                    with np.load(diagnostic_path, allow_pickle=False) as archive:
                        require_equal(
                            set(archive.files), set(DIAGNOSTIC_FIELDS), "query raw array census"
                        )
                        raw_d = {name: archive[name] for name in DIAGNOSTIC_FIELDS}
                    expected_d = diagnostics_from_arrays(
                        raw_d["trace"],
                        raw_d["neural"],
                        raw_d["motor"],
                        episode["labels"],
                    )
                    potential = raw_d["potential"]
                    expected_d.update(
                        paired_potential_distance=np.linalg.norm(
                            potential[::2] - potential[1::2], axis=1
                        ).tolist(),
                    )
                    expected_d = {k: v for k, v in expected_d.items() if k not in DIAGNOSTIC_FIELDS}
                    expected_d["raw"] = d["raw"]
                    require_equal(d, expected_d, "query diagnostic arithmetic")
                if arm == "history":
                    require_equal(
                        trial["history_traffic"],
                        history_traffic(episode["observations"]),
                        "observed history traffic",
                    )
            if result["completed"]:
                require_equal(len(trials), len(arrays["test/condition"]), "complete trial census")
            if arm != "random":
                training = result["training"]
                if "episodes" in training:
                    require_equal(
                        training["episodes"], len(arrays["train/condition"]), "training episodes"
                    )
                    require_equal(
                        training["rule"], protocol["training"].get("rule", "every"), "teaching rule"
                    )
                    queries = training["queries"]
                    require_equal(
                        [q["index"] for q in queries],
                        list(range(len(queries))),
                        "training query census",
                    )
                    for query in queries:
                        labels = episode_arrays(arrays, "train", query["index"])["labels"].tolist()
                        require_equal(query["labels"], labels, "training labels")
                        selected = (
                            list(range(inputs.STREAMS))
                            if training["rule"] == "every"
                            else (
                                []
                                if query["answer"] is None
                                else [
                                    i
                                    for i, (a, b) in enumerate(
                                        zip(query["answer"], labels, strict=True)
                                    )
                                    if a != b
                                ]
                            )
                        )
                        require_equal(query["lesson_rows"], selected, "selected teaching rows")
                    if training["completed"]:
                        require_equal(
                            training["completed_episodes"],
                            len(arrays["train/condition"]),
                            "completed training",
                        )
                        require_equal(len(queries), training["episodes"], "completed queries")
                    if arm == "history":
                        for index, traffic in enumerate(training["history_traffic"]):
                            require_equal(
                                traffic,
                                history_traffic(
                                    episode_arrays(arrays, "train", index)["observations"]
                                ),
                                "training history traffic",
                            )
                audit_work(
                    training["work"], base / arm / "training-calls.jsonl", incomplete=incomplete
                )
                audit_work(
                    result["work"], base / arm / "evaluation-calls.jsonl", incomplete=incomplete
                )
                if result.get("timing") is not None:
                    require_equal(
                        result["timing"]["equal_arrays"],
                        same_arrays(
                            base / arm / "timing-timestamps.npz",
                            base / arm / "timing-irregular_timestamps.npz",
                        )
                        and result["timing"]["cue_qualified"]
                        and all(w["action_refusals"] == 0 for w in result["timing"]["work"]),
                        "timestamp fork arrays",
                    )
                    event = result["timing"]["matched_elapsed_events"]
                    require_equal(
                        [b["events"] for b in event["branches"]], [1, 2], "event-count fork census"
                    )
                    for branch in event["branches"]:
                        require_equal(branch["elapsed"], 2.0, "matched elapsed time")
                        require_equal(
                            branch["timestamps"],
                            np.linspace(0, 2, branch["events"] + 1).tolist(),
                            "matched event timestamps",
                        )
                    require_equal(
                        event["trace_distance"],
                        float(
                            np.linalg.norm(
                                np.asarray(event["branches"][0]["pre_query_trace"])
                                - np.asarray(event["branches"][1]["pre_query_trace"])
                            )
                        ),
                        "event-count trace distance",
                    )
                for name, seam in result.get("seams", {}).items():
                    seam_base = base / arm / f"seams-{name}"
                    for checkpoint_name, checkpoint in seam["seams"].items():
                        require_equal(
                            checkpoint["saved_arrays_equal"],
                            same_arrays(
                                seam_base / "seam-final.npz",
                                seam_base / f"seam-{checkpoint_name}-resumed.npz",
                            ),
                            "saved continuation arrays",
                        )
                    if seam["imagine_leaves_checkpoint"]:
                        require_equal(
                            same_arrays(
                                seam_base / "seam-before_query.npz",
                                seam_base / "seam-after-imagine.npz",
                            ),
                            True,
                            "private imagination arrays",
                        )
                    if seam["refused_act_leaves_checkpoint"]:
                        require_equal(
                            same_arrays(
                                seam_base / "seam-before_query.npz",
                                seam_base / "seam-after-refusal.npz",
                            ),
                            True,
                            "refused event arrays",
                        )
            expected_scores[arm] = score_arm(result, repeats["test"], arm)
        require_equal(founder["scores"], expected_scores, "trial scores")
        require_equal(founder["gates"], gates_for(founder, protocol), "founder gates")
    require_equal(
        body["closure"],
        frozen and all(f["gates"]["closure"] for f in body["founders"]),
        "overall closure",
    )


def verify(directory: Path) -> tuple[bool, str]:
    try:
        path = directory / "summary.json"
        raw = json.loads(path.read_text())
        sources = [(item["path"], directory / item["path"]) for item in raw["source"]["files"]]

        def check(body: dict) -> str | None:
            for name, expected in body["artifacts"].items():
                if sha256(directory / name) != expected:
                    return f"artifact differs: {name}"
            if body["protocol_sha256"] != sha256(directory / "protocol.json"):
                return "protocol copy differs from the recorded hash"
            revision = body["declaration"].get("instrument_revision")
            if revision is not None:
                if revision != INSTRUMENT_REVISION:
                    return "unsupported instrument revision"
                for source in (Path(__file__).resolve(), Path(inputs.__file__).resolve()):
                    if sha256(directory / "source" / source.name) != sha256(source):
                        return (
                            "corrected-instrument source differs; use its retained source verifier"
                        )
                require_equal(
                    json.loads((directory / "source-manifest.json").read_text()),
                    {name: sha256(path) for name, path in sources},
                    "pre-run source manifest",
                )
                audit_current(body, directory)
            return None

        valid, reason = Receipt.verify(path, sources=sources, check=check)
        if not valid:
            return False, reason
        if raw["body"]["declaration"].get("instrument_revision") is None:
            return (
                True,
                "legacy custody verified; historical schedule, work and gate claims are not "
                "corrected-instrument validation",
            )
        return (
            True,
            "canonical form, source/artifact custody, declared inputs, trials, work tallies "
            "raw trace replay, diagnostics and gates agree",
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return False, f"cannot verify artifact: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--verify", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--repeats",
        type=int,
        nargs=2,
        metavar=("TRAIN", "TEST"),
        default=None,
        help="development sizes; marks the receipt not frozen",
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    protocol, protocol_sha = load_protocol(args.protocol)
    seeds = list(protocol["seeds"]["founders"]) if args.seeds is None else list(args.seeds)
    repeats = None if args.repeats is None else {"train": args.repeats[0], "test": args.repeats[1]}
    if (
        len(seeds) != len(set(seeds))
        or any(s < 0 for s in seeds)
        or (repeats and min(repeats.values()) < 1)
    ):
        parser.error("invalid seeds or repeats")
    if args.worker_seed is not None:
        founder = run_founder(args.worker_seed, protocol, args.out, repeats)
        # Independent raw recurrence replay is inside the same hard worker deadline.
        for arm in ARMS[:-1]:
            for phase, stage in (
                ("training", founder["arms"][arm]["training"]),
                ("evaluation", founder["arms"][arm]),
            ):
                audit_work(stage["work"], args.out / arm / f"{phase}-calls.jsonl")
        write_progress(args.out / "worker-result.json", founder)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
    began = time.perf_counter()
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.__config__.show()
    declaration = {
        "schema": SCHEMA,
        "instrument_revision": INSTRUMENT_REVISION,
        "trace_encoding": "stacked-trace/1",
        "query_encoding": "query-arrays/1",
        "evidence_scope": "Corrected instrument; previously used seeds provide an audit, "
        "not fresh confirmation.",
        "protocol_sha256": protocol_sha,
        "frozen_protocol": repeats is None and args.seeds is None,
        "repeats": repeats,
        "effective_repeats": repeats
        or {"train": protocol["training"]["repeats"], "test": protocol["evaluation"]["repeats"]},
        "seeds": seeds,
        "seed_override": args.seeds is not None,
        "python": sys.version,
        "platform": platform.platform(),
        "cadence_version": cadence.__version__,
        "cadence_import": str(Path(cadence.__file__).resolve()),
        "numpy": np.__version__,
        "blas_configuration": blas.getvalue(),
        "float64_eps": float(np.finfo(np.float64).eps),
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    (args.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    sources, origins = [], []
    package = Path(cadence.__file__).resolve().parent
    own = [Path(__file__).resolve(), Path(inputs.__file__).resolve()]
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
    write_progress(
        args.out / "source-manifest.json", {name: sha256(path) for name, path in sources}
    )
    founders = []
    for seed in seeds:
        founder = guarded_founder(
            seed,
            protocol,
            args.out / f"founder-{seed}",
            declaration["effective_repeats"],
            args.out / "protocol.json",
        )
        founders.append(founder)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "horizon": founder["gates"]["horizon"],
                    "nuisance": founder["gates"]["nuisance"],
                    "closure": founder["gates"]["closure"],
                    "capped": founder["capped"],
                    "seconds": founder["seconds"],
                }
            ),
            flush=True,
        )
    assert all(sha256(original) == sha256(copy) for original, copy in origins), (
        "source changed during the run; retain this incomplete attempt and rerun"
    )
    artifacts = {
        p.relative_to(args.out).as_posix(): sha256(p)
        for p in sorted(args.out.rglob("*"))
        if p.is_file()
        and "source" not in p.relative_to(args.out).parts
        and p.name != "summary.json"
    }
    summary = {
        "declaration": declaration,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": declaration["frozen_protocol"],
        "founders": founders,
        "unmet_protocol_obligations": list(UNMET_PROTOCOL_OBLIGATIONS),
        "seconds": time.perf_counter() - began,
        "artifacts": artifacts,
        "closure": declaration["frozen_protocol"]
        and all(f["gates"]["closure"] for f in founders)
        and bool(founders),
        "work_scope": (
            "Completed teacher, free, fork, seam, imagined and refused phases and save/load "
            "calls are charged per arm; nested seam/timing work is included in the arm totals. "
            "Calls_seconds excludes checkpoint IO. Each founder has a hard child-process wall "
            "deadline; retained bytes are checked between operations. Failed/terminated workers "
            "retain durable progress, completed journals and full planned denominators, with "
            "unknown current-call work and unflushed trace arrays explicitly incomplete. "
            "Complete accepted events retain raw trace recurrence arrays in bounded chunks. "
            "Sweeps are not joules."
        ),
    }
    Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    # Publication adds metadata: enforce the original retained-output cap again.
    changed = False
    for founder in founders:
        if retained_bytes(args.out, founder["seed"]) > protocol["caps"]["bytes"]:
            founder["capped"] = True
            founder["cap_reason"] = "retained output cap exceeded after receipt publication"
            founder["gates"] = gates_for(founder, protocol)
            changed = True
    if changed:
        summary["closure"] = False
        Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(
        json.dumps(
            {
                "verified": valid,
                "closure": summary["closure"],
                "summary": str(args.out / "summary.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
