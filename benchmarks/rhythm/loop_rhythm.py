"""The loop chamber of issue 140: a brain taught to predict the next event of a periodic
pattern plays it in the closed loop, hearing its own output, from a prime.

A brain that predicts well on held-out rows can collapse, when fed its own output, to one
trajectory that does not depend on the prime: on the C64 rows of 0.73.1 the loop held forever,
because a brain that cannot count rows since its last onset, hearing its own holds, has hold as
its only consistent answer. This chamber is the smallest version of that observation: one
voice, an onset every ``period`` rows, the heard event and a constant drive as the only input,
the next event as the only label.

This is an engineered sensory-history comparison, not a supported use of the public
own-command efference contract or a repair of issue 140. The adapter writes heard inputs
directly into the efference storage and restores that storage after acts. It supplies a
decaying input-history feature; the next event remains a teaching label only. The
heard event enters the copy before the brain answers a row, in teaching, in watching and in
the prime of a play, and the brain's own command is taken back out of it, so that the copy
carries the heard stream alone; in the closed loop the brain's own command is the heard event
and stays. Without that contract a brain whose copy holds only its own greedy commands never
has an onset in it while it still emits holds, and the majority fixed point seals itself.
Teaching uses an explicit label-based mismatch policy, not ``Brain.live``: a row answered
right teaches nothing, a wrong or refused answer is followed by one lesson on that row;
a lesson on every row (the
watching recipe of the C64 lane) made the pattern come and go from pass to pass on the
development seeds. Arms on the same founder weights and the same rows:

- ``copy``      the copy at the protocol's decay under the declared contract;
- ``own``       the same copy, written only by the brain's own greedy commands (the simpler
                control of the contract);
- ``nocopy``    no copy: the working trace alone (the 0.73.1 setting);
- ``frozen``    the ``copy`` founder without lessons, played under the same contract (a beat
                from birth is not credited);
- ``ngram``     a table over the last ``period`` events, taught on the same rows: the matched
                conventional learner that counts explicitly;
- ``hold``      hold forever; ``random`` uniform random events.

Every brain arm is first watched: from reset it hears the taught rows once more with free
greedy acts and no lesson, and its agreement with the next event is the open-loop reading (the
agreement inside teaching precedes any conditional lesson on the same row). Then every arm
plays closed loops of ``rows`` rows from primes at every
phase of the pattern. Readings: onset rate against the truth's, agreement with the primed
pattern's own continuation, agreement with the other phases' continuations (prime dependence)
and the pairwise correlation of the plays across primes (the issue's own measure). Receipts
bind the chamber and the library. Run ``python benchmarks/rhythm/loop_rhythm.py --out DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd
from cadence import Brain, LearnerConfig, Receipt

SCHEMA = "loop-rhythm/1"
INSTRUMENT_REVISION = 2
IDENTITY = "engineered heard-event history adapter; not public own-command efference or Brain.live"
PROTOCOL_PATH = Path(__file__).with_name("protocol-loop.json")
ARMS = ("copy", "own", "nocopy", "frozen", "ngram", "hold", "random")
BRAIN_ARMS = ("copy", "own", "nocopy", "frozen")
CONTRACT_ARMS = ("copy", "frozen")  # the heard event enters the copy before the brain answers
HOLD, ONSET = 0, 1
INPUTS = 3  # the heard event (hold, onset) and a constant drive
REQUIRED = ("schema", "pattern", "teaching", "play", "recipe", "copy", "ngram", "seeds", "gates")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}")
    if set(protocol["seeds"]["development"]) & set(protocol["seeds"]["confirmation"]):
        raise ValueError("confirmation seeds must be fresh")
    if (
        protocol["pattern"]["period"] < 2
        or protocol["teaching"]["rows"] < 2
        or protocol["teaching"]["passes"] < 0
        or protocol["play"]["rows"] < 2
        or protocol["play"]["prime"] < protocol["pattern"]["period"]
    ):
        raise ValueError("invalid pattern, teaching or play length")
    return protocol, hashlib.sha256(raw).hexdigest()


# -- the pattern


def pattern(period: int, rows: int, phase: int = 0) -> np.ndarray:
    """An onset every ``period`` rows, the first at row ``phase``."""
    return np.array([ONSET if (t - phase) % period == 0 else HOLD for t in range(rows)])


def observation(event: int) -> np.ndarray:
    x = np.zeros((1, INPUTS))
    x[0, 2] = 1.0
    x[0, event] = 1.0
    return x


def continuation(period: int, prime: np.ndarray, rows: int) -> np.ndarray:
    """What the primed pattern itself does next: the onsets keep their period and phase."""
    last_onset = int(np.flatnonzero(prime == ONSET)[-1])
    start = len(prime)
    return np.array(
        [ONSET if (t - last_onset) % period == 0 else HOLD for t in range(start, start + rows)]
    )


# -- brains


def make_brain(seed: int, protocol: dict, arm: str) -> Brain:
    recipe = protocol["recipe"]
    options: dict[str, Any] = {
        "episodic": False,
        "lateral": recipe["lateral"],
        "resting_bias": recipe["resting_bias"],
        "working_memory_amplitude": recipe["trace_amplitude"],
        "working_memory_decay": recipe["trace_decay"],
        # the composition of releases before 0.82.0 read the trace back at scale 12
        "working_memory_scale": recipe.get("trace_scale", 12.0),
        "learning": LearnerConfig(**recipe["learning"]),
    }
    if arm in ("copy", "own", "frozen"):
        options["efference_amplitude"] = protocol["copy"]["amplitude"]
        options["efference_decay"] = protocol["copy"]["decay"]
    return Brain.compose(INPUTS, 2, modules=tuple(recipe["modules"]), seed=seed, **options)


class Work:
    def __init__(self) -> None:
        self.lessons = self.refused_lessons = self.acts = self.refused_acts = 0
        self.lesson_sweeps = self.act_sweeps = 0
        self.seconds = 0.0
        self.events: list[dict] = []
        self.adapter_rows = self.adapter_updates = 0
        self.adapter_seconds = 0.0
        self.checkpoint_writes = self.checkpoint_reads = self.checkpoint_bytes = 0
        self.checkpoint_seconds = 0.0
        self.control_updates = self.control_predictions = 0
        self.control_seconds = 0.0
        self.phase = "unclassified"

    def summary(self) -> dict:
        return {
            name: value for name, value in vars(self).items() if name not in ("events", "phase")
        }


def act(brain: Brain, x: np.ndarray, work: Work) -> int | None:
    start = time.perf_counter()
    work.acts += 1
    prior = brain.last_settlement
    try:
        answer = int(brain.act(x, greedy=True)[0])
    except RuntimeError:
        report = brain.last_settlement
        if report is None or report is prior or report["qualified"]:
            raise
        work.refused_acts += 1
        answer = None
    finally:
        work.seconds += time.perf_counter() - start
    work.act_sweeps += int(brain.last_settlement["steps"])
    work.events.append(
        {
            "kind": "act",
            "phase": work.phase,
            "answer": answer,
            "steps": int(brain.last_settlement["steps"]),
        }
    )
    return answer


def teach(brain: Brain, x: np.ndarray, label: int, work: Work) -> bool:
    from cadence.learning import LearningPhaseError

    start = time.perf_counter()
    work.lessons += 1
    try:
        _, report = brain.learner.step(brain.stimulus(x), np.array([label]))
    except LearningPhaseError as error:
        work.refused_lessons += 1
        report = error.report
        work.lesson_sweeps += int(report["total_steps"])
        work.events.append(
            {
                "kind": "lesson",
                "phase": work.phase,
                "refused": True,
                "steps": int(report["total_steps"]),
            }
        )
        return False
    finally:
        work.seconds += time.perf_counter() - start
    work.lesson_sweeps += int(report["total_steps"])
    work.events.append(
        {
            "kind": "lesson",
            "phase": work.phase,
            "refused": False,
            "steps": int(report["total_steps"]),
        }
    )
    return True


def memory_state(brain: Brain) -> dict[str, dict[str, np.ndarray]]:
    """The working trace and the copy as they stand: what the next settle will read."""
    state: dict[str, dict[str, np.ndarray]] = {}
    for name in ("working_memory", "efference"):
        part = getattr(brain, name)
        if part is not None:
            state[name] = {k: getattr(part, k).copy() for k in ("trace", "last", "cold")}
    return state


def restore_memory(brain: Brain, state: dict[str, dict[str, np.ndarray]]) -> None:
    for name, values in state.items():
        part = getattr(brain, name)
        for k, v in values.items():
            setattr(part, k, v.copy())


def hear(
    brain: Brain,
    arm: str,
    event: int,
    work: Work,
    label: int | None = None,
    keep: bool = False,
) -> int | None:
    """One row under the declared contract: the heard event enters the copy, the brain answers
    the row with a free greedy act and, when ``label`` is given and the answer is not the
    label, a lesson on the same drive follows (the trace and the copy are put back to what the
    act read for the lesson and left as the act left them after it): a right answer is routine
    and teaches nothing. Unless ``keep``, the brain's own command is then taken back out of
    the copy, so the copy carries the heard stream alone, one update per row. Under ``own`` the
    copy carries nothing but the brain's own commands."""
    echo = brain.efference if arm in CONTRACT_ARMS else None
    began = time.perf_counter()
    settling_before = work.seconds
    work.adapter_rows += 1
    if echo is not None:
        if not len(echo.trace):
            echo.reset(1)
        echo.issue(np.eye(2)[[event]])
        work.adapter_updates += 1
    x = observation(event)
    before = memory_state(brain)
    answer = act(brain, x, work)
    if label is not None and answer != label:
        after = memory_state(brain)
        restore_memory(brain, before)
        try:
            teach(brain, x, label, work)
        finally:
            restore_memory(brain, after)
    if echo is not None and not keep:
        for k, v in before["efference"].items():
            setattr(echo, k, v)
    work.adapter_seconds += time.perf_counter() - began - (work.seconds - settling_before)
    return answer


def teach_brain(brain: Brain, arm: str, rows: np.ndarray, passes: int, work: Work) -> dict:
    """The teaching contract: every row is heard and answered with a free greedy act, and only
    a wrong or refused answer is followed by a lesson on that row with the next event as the
    label. The agreement per pass is the share of right answers, each given before any lesson
    on its row (the online reading); ``watch_brain`` is the reading after teaching."""
    agreement, lessons, answers = [], [], []
    for batch in range(passes):
        work.phase = f"teaching:{batch}"
        brain.reset()
        hits = 0
        answered = []
        given = work.lessons
        for t in range(len(rows) - 1):
            label = int(rows[t + 1])
            answer = hear(brain, arm, int(rows[t]), work, label=label)
            answered.append(answer)
            hits += int(answer == label)
        agreement.append(hits / max(1, len(rows) - 1))
        lessons.append(work.lessons - given)
        answers.append(answered)
    return {
        "agreement_per_pass": agreement,
        "lessons_per_pass": lessons,
        "last_pass_agreement": agreement[-1] if agreement else None,
        "answers": answers,
    }


def watch_brain(brain: Brain, arm: str, rows: np.ndarray, work: Work) -> dict:
    """The open-loop reading: from reset, the brain hears the rows once with free greedy acts
    and no lesson; agreement is the share of its answers equal to the next event, onset recall
    the share of onsets it announced."""
    brain.reset()
    work.phase = "watching"
    hits = total = onsets = announced = 0
    answers = []
    for t in range(len(rows) - 1):
        label = int(rows[t + 1])
        answer = hear(brain, arm, int(rows[t]), work)
        answers.append(answer)
        total += int(answer is not None)
        hits += int(answer == label)
        if label == ONSET:
            onsets += 1
            announced += int(answer == ONSET)
    return {
        "agreement": hits / max(1, len(rows) - 1),
        "onset_recall": announced / max(1, onsets),
        "rows": total,
        "refusals": len(rows) - 1 - total,
        "answers": answers,
    }


def checkpoint(brain: Brain, path: Path, work: Work) -> None:
    began = time.perf_counter()
    brain.save(path)
    work.checkpoint_writes += 1
    work.checkpoint_bytes += path.stat().st_size
    work.checkpoint_seconds += time.perf_counter() - began


def play_brain(
    brain: Brain,
    arm: str,
    prime: np.ndarray,
    rows: int,
    work: Work,
    *,
    phase: int = 0,
    directory: Path | None = None,
    custody: dict | None = None,
) -> list[int | None]:
    """Hear the prime, then hear your own output for ``rows`` rows. The answer to the last
    prime row is the first event of the loop, and from there every heard event is the brain's
    own last answer (hold when it refused)."""
    brain.reset()
    answer: int | None = None
    last = len(prime) - 1
    work.phase = f"prime:{phase}"
    for t, event in enumerate(prime):
        answer = hear(brain, arm, int(event), work, keep=t == last)
    out: list[int | None] = [answer]
    heard = HOLD if answer is None else answer
    twin = None
    twin_heard = HOLD
    checkpoint_heard = HOLD
    twin_out: list[int | None] = []
    probe_at = min(8, rows - 1)
    for _ in range(rows - 1):
        if directory is not None and len(out) == probe_at:
            path = directory / "mid-loop.npz"
            checkpoint(brain, path, work)
            began = time.perf_counter()
            twin = Brain.load(path)
            work.checkpoint_reads += 1
            work.checkpoint_bytes += path.stat().st_size
            work.checkpoint_seconds += time.perf_counter() - began
            twin_heard = heard
            checkpoint_heard = heard
        work.phase = f"play:{phase}"
        answer = act(brain, observation(heard), work)
        out.append(answer)
        heard = HOLD if answer is None else answer
        if twin is not None:
            work.phase = "continuation"
            restored = act(twin, observation(twin_heard), work)
            twin_out.append(restored)
            twin_heard = HOLD if restored is None else restored
    if twin is not None:
        assert directory is not None and custody is not None
        original_path, restored_path = directory / "continued.npz", directory / "restored.npz"
        checkpoint(brain, original_path, work)
        checkpoint(twin, restored_path, work)
        began = time.perf_counter()
        with (
            np.load(original_path, allow_pickle=False) as original,
            np.load(restored_path, allow_pickle=False) as restored,
        ):
            equal = original.files == restored.files and all(
                np.array_equal(original[name], restored[name]) for name in original.files
            )
        work.checkpoint_reads += 2
        work.checkpoint_bytes += original_path.stat().st_size + restored_path.stat().st_size
        work.checkpoint_seconds += time.perf_counter() - began
        custody.update(
            after_rows=probe_at,
            heard=checkpoint_heard,
            answers=twin_out,
            actions_equal=twin_out == out[probe_at:],
            arrays_equal=equal,
        )
    return out


class NGram:
    """A table over the last ``order`` events, taught on the same rows; the matched learner
    that counts explicitly."""

    def __init__(self, order: int) -> None:
        self.order = order
        self.counts: dict[tuple[int, ...], np.ndarray] = {}

    def teach(self, rows: np.ndarray) -> None:
        history: list[int] = []
        for t in range(len(rows) - 1):
            history.append(int(rows[t]))  # the key ends with the current event, as in play
            key = tuple(history[-self.order :])
            self.counts.setdefault(key, np.zeros(2))[int(rows[t + 1])] += 1

    def answer(self, history: list[int]) -> int:
        key = tuple(history[-self.order :])
        counts = self.counts.get(key)
        return HOLD if counts is None else int(np.argmax(counts))

    def play(self, prime: np.ndarray, rows: int) -> list[int]:
        history = [int(e) for e in prime]
        out = []
        for _ in range(rows):
            nxt = self.answer(history)
            out.append(nxt)
            history.append(nxt)
        return out


# -- scoring


def score_play(out: list[int | None], truth: np.ndarray) -> dict:
    played = np.array([-1 if v is None else v for v in out])
    return {
        "onset_rate": float(np.mean(played == ONSET)),
        "agreement": float(np.mean(played == truth)),
        "refusals": int(sum(v is None for v in out)),
    }


def correlation(a: list[int | None], b: list[int | None]) -> float | None:
    if any(v is None for v in a + b):
        return None
    x = np.array([HOLD if v is None else v for v in a], dtype=float)
    y = np.array([HOLD if v is None else v for v in b], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_founder(seed: int, arm: str, protocol: dict, directory: Path) -> dict:
    began = time.perf_counter()
    directory.mkdir(parents=True)
    warnings.simplefilter("ignore")
    period = int(protocol["pattern"]["period"])
    rows = pattern(period, int(protocol["teaching"]["rows"]))
    prime_rows = int(protocol["play"]["prime"])
    play_rows = int(protocol["play"]["rows"])
    primes = [pattern(period, prime_rows, phase) for phase in range(period)]
    truths = [continuation(period, prime, play_rows) for prime in primes]
    work = Work()
    result: dict[str, Any] = {"arm": arm, "seed": seed}
    rng = np.random.default_rng(seed)
    if arm in BRAIN_ARMS:
        brain = make_brain(seed, protocol, arm)
        checkpoint(brain, directory / "initial.npz", work)
        if arm != "frozen":
            result["teaching"] = teach_brain(
                brain, arm, rows, int(protocol["teaching"]["passes"]), work
            )
            checkpoint(brain, directory / "taught.npz", work)
        result["watching"] = watch_brain(brain, arm, rows, work)
        result["continuation"] = {}
        plays = [
            play_brain(
                brain,
                arm,
                prime,
                play_rows,
                work,
                phase=phase,
                directory=directory if phase == 0 else None,
                custody=result["continuation"] if phase == 0 else None,
            )
            for phase, prime in enumerate(primes)
        ]
    elif arm == "ngram":
        started = time.perf_counter()
        table = NGram(int(protocol["ngram"]["order"]))
        table.teach(rows)
        plays = [table.play(prime, play_rows) for prime in primes]
        work.control_updates = len(rows) - 1
        work.control_predictions = period * play_rows
        work.control_seconds = time.perf_counter() - started
    elif arm == "hold":
        started = time.perf_counter()
        plays = [[HOLD] * play_rows for _ in primes]
        work.control_predictions = period * play_rows
        work.control_seconds = time.perf_counter() - started
    else:
        started = time.perf_counter()
        plays = [[int(v) for v in rng.integers(0, 2, play_rows)] for _ in primes]
        work.control_predictions = period * play_rows
        work.control_seconds = time.perf_counter() - started
    result["plays"] = [[-1 if v is None else v for v in play] for play in plays]
    result.update(score_plays(plays, truths))
    result["work"] = work.summary()
    result["events"] = work.events
    result["elapsed_seconds"] = time.perf_counter() - began
    return result


def score_plays(plays: list[list[int | None]], truths: list[np.ndarray]) -> dict:
    result: dict[str, Any] = {}
    result["scores"] = [score_play(play, truth) for play, truth in zip(plays, truths, strict=True)]
    # prime dependence: a play should agree with its own prime's continuation more than with
    # the continuations of the other phases
    cross = [
        [
            float(np.mean(np.array([-1 if v is None else v for v in play]) == other))
            for other in truths
        ]
        for play in plays
    ]
    result["cross_agreement"] = cross
    result["prime_dependent"] = all(
        row[i] > max(row[j] for j in range(len(row)) if j != i) for i, row in enumerate(cross)
    )
    pairs = [
        correlation(plays[i], plays[j]) for i in range(len(plays)) for j in range(i + 1, len(plays))
    ]
    present = [c for c in pairs if c is not None]
    result["cross_prime_correlation"] = float(np.mean(present)) if present else None
    result["mean_onset_rate"] = float(np.mean([s["onset_rate"] for s in result["scores"]]))
    result["mean_agreement"] = float(np.mean([s["agreement"] for s in result["scores"]]))
    result["refusals"] = int(sum(s["refusals"] for s in result["scores"]))
    return result


def gates(
    rows: list[dict], protocol: dict, seeds: list[int] | None = None,
    *, frozen_protocol: bool = True,
) -> dict:
    g = protocol["gates"]
    period = int(protocol["pattern"]["period"])
    truth_rate = 1.0 / period
    hold_agreement = 1.0 - truth_rate
    seeds = protocol["seeds"]["confirmation"] if seeds is None else seeds
    pairs = [(r["seed"], r["arm"]) for r in rows]
    expected = {(seed, arm) for seed in seeds for arm in ARMS}
    complete = len(pairs) == len(set(pairs)) and set(pairs) == expected
    admitted = complete and frozen_protocol and seeds == protocol["seeds"]["confirmation"]
    out: dict[str, Any] = {"complete": complete, "admitted": admitted, "passed": False}
    for arm in sorted({r["arm"] for r in rows}, key=ARMS.index):
        group = [r for r in rows if r["arm"] == arm]
        fires = sum(
            all(
                abs(s["onset_rate"] - truth_rate) <= g["onset_tolerance"] * truth_rate
                for s in r["scores"]
            )
            for r in group
        )
        follows = sum(
            all(s["agreement"] >= hold_agreement + g["agreement_margin"] for s in r["scores"])
            for r in group
        )
        depends = sum(bool(r["prime_dependent"]) for r in group)
        out[arm] = {
            "founders": len(group),
            "fires": fires,
            "follows": follows,
            "prime_dependent": depends,
            "mean_onset_rate": float(np.mean([r["mean_onset_rate"] for r in group])),
            "mean_agreement": float(np.mean([r["mean_agreement"] for r in group])),
            "refusals": int(sum(r["work"]["refused_acts"] for r in group)),
        }
        if all("watching" in r for r in group):
            out[arm]["mean_watching"] = float(np.mean([r["watching"]["agreement"] for r in group]))
    if "copy" in out and "frozen" in out:
        frozen = {r["seed"]: r for r in rows if r["arm"] == "frozen"}
        learned = sum(
            all(
                abs(s["onset_rate"] - truth_rate) <= g["onset_tolerance"] * truth_rate
                and s["agreement"] >= hold_agreement + g["agreement_margin"]
                for s in r["scores"]
            )
            and bool(r["prime_dependent"])
            and r["work"]["refused_acts"] == 0
            and r["seed"] in frozen
            and frozen[r["seed"]]["work"]["refused_acts"] == 0
            and frozen[r["seed"]]["mean_agreement"] < hold_agreement + g["agreement_margin"]
            for r in rows
            if r["arm"] == "copy"
        )
        out["copy"]["learned"] = learned
        out["passed"] = (
            admitted and learned >= g["share"] and all(r["work"]["refused_acts"] == 0 for r in rows)
        )
    return out


def check_run(run: dict, protocol: dict, directory: Path) -> None:
    """Recompute observations, work and scores; summary fields are not evidence by themselves."""
    period, count = protocol["pattern"]["period"], protocol["play"]["rows"]
    primes = [pattern(period, protocol["play"]["prime"], phase) for phase in range(period)]
    truths = [continuation(period, prime, count) for prime in primes]
    assert len(run["plays"]) == period
    assert all(
        len(play) == count and all(type(v) is int and v in (-1, HOLD, ONSET) for v in play)
        for play in run["plays"]
    )
    plays = [[None if v == -1 else v for v in play] for play in run["plays"]]
    for key, value in score_plays(plays, truths).items():
        assert run[key] == value, f"play arithmetic differs: {key}"
    work, events = run["work"], run["events"]
    assert all(
        e["kind"] in ("act", "lesson") and type(e["steps"]) is int and e["steps"] >= 0
        for e in events
    )
    acts = [e for e in events if e["kind"] == "act"]
    lessons = [e for e in events if e["kind"] == "lesson"]
    assert work["acts"] == len(acts) and work["lessons"] == len(lessons)
    assert work["act_sweeps"] == sum(e["steps"] for e in acts)
    assert work["lesson_sweeps"] == sum(e["steps"] for e in lessons)
    assert work["refused_acts"] == sum(e["answer"] is None for e in acts)
    assert work["refused_lessons"] == sum(e["refused"] for e in lessons)
    assert all(np.isfinite(v) and v >= 0 for v in work.values())
    assert np.isfinite(run["elapsed_seconds"]) and run["elapsed_seconds"] >= 0
    arm = run["arm"]
    rows = pattern(period, protocol["teaching"]["rows"])
    if arm not in BRAIN_ARMS:
        assert not events and work["adapter_rows"] == work["adapter_updates"] == 0
        assert (
            work["checkpoint_writes"] == work["checkpoint_reads"] == work["checkpoint_bytes"] == 0
        )
        assert work["control_predictions"] == period * count
        assert work["control_updates"] == (len(rows) - 1 if arm == "ngram" else 0)
        if arm == "ngram":
            table = NGram(protocol["ngram"]["order"])
            table.teach(rows)
            expected = [table.play(prime, count) for prime in primes]
        elif arm == "hold":
            expected = [[HOLD] * count for _ in primes]
        else:
            rng = np.random.default_rng(run["seed"])
            expected = [rng.integers(0, 2, count).tolist() for _ in primes]
        assert run["plays"] == expected
        return

    def answers(phase: str) -> list[int | None]:
        return [e["answer"] for e in acts if e["phase"] == phase]

    expected_phases = {"watching", "continuation"}
    expected_phases.update(
        f"{kind}:{phase}" for kind in ("prime", "play") for phase in range(period)
    )
    if arm != "frozen":
        teaching = run["teaching"]
        assert len(teaching["answers"]) == protocol["teaching"]["passes"]
        agreements, given = [], []
        for batch, values in enumerate(teaching["answers"]):
            phase = f"teaching:{batch}"
            expected_phases.add(phase)
            assert len(values) == len(rows) - 1 and values == answers(phase)
            agreements.append(
                sum(a == int(b) for a, b in zip(values, rows[1:], strict=True)) / len(values)
            )
            given.append(sum(e["phase"] == phase for e in lessons))
            assert given[-1] == sum(a != int(b) for a, b in zip(values, rows[1:], strict=True))
        assert teaching["agreement_per_pass"] == agreements
        assert teaching["lessons_per_pass"] == given
        assert teaching["last_pass_agreement"] == (agreements[-1] if agreements else None)
    else:
        assert "teaching" not in run and not lessons
    assert all(e["phase"] in expected_phases for e in events)
    assert all(e["phase"].startswith("teaching:") for e in lessons)
    watching = run["watching"]
    values = answers("watching")
    assert values == watching["answers"] and len(values) == len(rows) - 1
    assert watching["rows"] == sum(a is not None for a in values)
    assert watching["refusals"] == sum(a is None for a in values)
    assert watching["agreement"] == sum(
        a == int(b) for a, b in zip(values, rows[1:], strict=True)
    ) / len(values)
    assert watching["onset_recall"] == sum(
        a == ONSET and b == ONSET for a, b in zip(values, rows[1:], strict=True)
    ) / max(1, int(np.sum(rows[1:] == ONSET)))
    for phase in range(period):
        primed = answers(f"prime:{phase}")
        assert len(primed) == len(primes[phase]) and primed[-1] == plays[phase][0]
        assert answers(f"play:{phase}") == plays[phase][1:]
    custody = run["continuation"]
    probe_at = min(8, count - 1)
    expected_heard = HOLD if plays[0][probe_at - 1] is None else plays[0][probe_at - 1]
    assert custody["after_rows"] == probe_at and custody["heard"] == expected_heard
    assert custody["answers"] == answers("continuation") == plays[0][probe_at:]
    assert custody["actions_equal"] and custody["arrays_equal"]
    folder = directory / f"seed{run['seed']}-{arm}"
    names = ["initial.npz", "mid-loop.npz", "continued.npz", "restored.npz"]
    if arm != "frozen":
        names.append("taught.npz")
    assert work["checkpoint_writes"] == len(names) and work["checkpoint_reads"] == 3
    assert work["checkpoint_bytes"] == sum(
        (folder / name).stat().st_size for name in names + names[1:4]
    )
    with (
        np.load(folder / "continued.npz", allow_pickle=False) as original,
        np.load(folder / "restored.npz", allow_pickle=False) as restored,
    ):
        assert original.files == restored.files and all(
            np.array_equal(original[k], restored[k]) for k in original.files
        )
    heard_rows = sum(
        not e["phase"].startswith("play:") and e["phase"] != "continuation" for e in acts
    )
    assert work["adapter_rows"] == heard_rows
    assert work["adapter_updates"] == (heard_rows if arm in CONTRACT_ARMS else 0)
    assert work["control_updates"] == work["control_predictions"] == 0


def verify(directory: Path) -> tuple[bool, str]:
    try:
        path = directory / "summary.json"
        raw = json.loads(path.read_text())
        assert raw["kind"] == SCHEMA
        sources = [(item["path"], directory / item["path"]) for item in raw["source"]["files"]]
        required_sources = {"source/loop_rhythm.py"} | {
            "source/cadence/" + p.relative_to(Path(cd.__file__).parent).as_posix()
            for p in Path(cd.__file__).parent.rglob("*.py")
        }
        assert {name for name, _ in sources} == required_sources, "incomplete source manifest"

        def check(body: dict) -> str | None:
            assert (
                body["instrument_revision"] == INSTRUMENT_REVISION and body["identity"] == IDENTITY
            )
            declaration = body["declaration"]
            assert declaration == json.loads((directory / "declaration.json").read_text())
            protocol = json.loads((directory / "protocol.json").read_text())
            overrides = declaration["overrides"]
            assert set(overrides) <= {"passes", "seeds", "arms", "protocol"}
            if "passes" in overrides:
                protocol["teaching"]["passes"] = overrides["passes"]
            assert protocol == body["protocol"]
            seeds, arms = declaration["seeds"], declaration["arms"]
            assert len(seeds) == len(set(seeds)) and len(arms) == len(set(arms))
            assert all(type(s) is int and s >= 0 for s in seeds) and set(arms) <= set(ARMS)
            assert seeds == overrides.get("seeds", protocol["seeds"]["confirmation"])
            assert arms == overrides.get("arms", list(ARMS))
            expected = {(seed, arm) for seed in seeds for arm in arms}
            actual = [(r["seed"], r["arm"]) for r in body["runs"]]
            assert len(actual) == len(set(actual)) and set(actual) == expected
            assert body["frozen_protocol"] == declaration["frozen_protocol"] == (not overrides)
            assert declaration["protocol_sha256"] == body["protocol_sha256"]
            assert body["protocol_sha256"] == overrides.get("protocol", sha256(PROTOCOL_PATH))
            expected_artifacts = {"protocol.json", "declaration.json"}
            for run in body["runs"]:
                check_run(run, protocol, directory)
                if run["arm"] in BRAIN_ARMS:
                    names = ["initial.npz", "mid-loop.npz", "continued.npz", "restored.npz"]
                    if run["arm"] != "frozen":
                        names.append("taught.npz")
                    expected_artifacts.update(f"seed{run['seed']}-{run['arm']}/{n}" for n in names)
            assert set(body["artifacts"]) == expected_artifacts
            for name, expected in body["artifacts"].items():
                if sha256(directory / name) != expected:
                    return f"artifact differs: {name}"
            if body["protocol_sha256"] != sha256(directory / "protocol.json"):
                return "protocol copy differs from the recorded hash"
            if gates(
                body["runs"], body["protocol"], seeds, frozen_protocol=body["frozen_protocol"]
            ) != body["gates"]:
                return "the stored gates do not follow from the runs"
            return None

        valid, reason = Receipt.verify(path, sources=sources, check=check)
        return valid, (
            "canonical form, digest, sources, artifact hashes and gates agree" if valid else reason
        )
    except (OSError, ValueError, KeyError, TypeError, AssertionError, IndexError) as error:
        return False, f"cannot verify artifact: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--verify", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--passes", type=int, default=None)
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    protocol, protocol_sha = load_protocol(args.protocol)
    overrides: dict[str, Any] = {}
    if args.passes is not None:
        if args.passes < 0:
            parser.error("passes must be nonnegative")
        protocol["teaching"]["passes"] = args.passes
        overrides["passes"] = args.passes
    seeds = list(protocol["seeds"]["confirmation"]) if args.seeds is None else list(args.seeds)
    if args.seeds is not None:
        overrides["seeds"] = seeds
    if args.arms != list(ARMS):
        overrides["arms"] = list(args.arms)
    if protocol_sha != sha256(PROTOCOL_PATH):
        overrides["protocol"] = protocol_sha
    if (
        len(seeds) != len(set(seeds))
        or any(s < 0 for s in seeds)
        or any(a not in ARMS for a in args.arms)
        or len(args.arms) != len(set(args.arms))
    ):
        parser.error("invalid seeds or arms")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
    began = time.perf_counter()
    declaration = {
        "schema": SCHEMA,
        "instrument_revision": INSTRUMENT_REVISION,
        "identity": IDENTITY,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": not overrides,
        "overrides": overrides,
        "seeds": seeds,
        "arms": list(args.arms),
        "python": sys.version,
        "platform": platform.platform(),
        "cadence_version": cd.__version__,
        "cadence_import": str(Path(cd.__file__).resolve()),
        "cpu_count": os.cpu_count(),
    }
    (args.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    sources, origins = [], []
    package = Path(cd.__file__).resolve().parent
    own = [Path(__file__).resolve()]
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
    runs = []
    for seed in seeds:
        for arm in args.arms:
            result = run_founder(seed, arm, protocol, args.out / f"seed{seed}-{arm}")
            runs.append(result)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "watching": (
                            round(result["watching"]["agreement"], 3)
                            if "watching" in result
                            else None
                        ),
                        "onset_rate": round(result["mean_onset_rate"], 3),
                        "agreement": round(result["mean_agreement"], 3),
                        "prime_dependent": result["prime_dependent"],
                        "seconds": round(time.perf_counter() - began, 1),
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
        "instrument_revision": INSTRUMENT_REVISION,
        "identity": IDENTITY,
        "declaration": declaration,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": declaration["frozen_protocol"],
        "runs": runs,
        "gates": gates(runs, protocol, seeds, frozen_protocol=declaration["frozen_protocol"]),
        "seconds": time.perf_counter() - began,
        "artifacts": artifacts,
        "work_scope": (
            "All brain acts and lessons, including refused and continuation work, are recorded. "
            "Adapter, conventional-control and checkpoint operations have separate counts/times; "
            "founder elapsed time includes scoring and bookkeeping. Receipt/source copying and "
            "verification are outside these founder counters. "
            "Sweeps are not joules; no efficiency claim."
        ),
    }
    Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(
        json.dumps(
            {
                "verified": valid,
                "gates": summary["gates"].get("passed"),
                "summary": str(args.out / "summary.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
