"""Paired settling cost of qualified nudged phases from different starting states.

Every measured lesson settles one qualified free phase, then settles the positive
and negative nudged phases once from each arm's starting state on the same
acquired brain, drive, labels and parameters. Only the control's update is
applied, so all arms see the same brain sequence. See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import cadence
from cadence import Brain, BrainState, LearnerConfig, LearningPhaseError, RecoveryStart

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "acquisition" / "fixture" / "school.npz"
ARMS = ("default", "copy", "fitted", "mirror")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose(case: dict, seed: int) -> Brain:
    learning = LearnerConfig(
        beta=0.1, eta=0.5, eta_bias=0.02, centered=True, temperature=0.2, momentum=0.9,
        qualified=True, damping=3, tolerance=case["tolerance"],
        free_steps=case["budget"], nudged_steps=case["budget"],
    )
    return Brain.compose(
        inputs=650, actions=36, modules=tuple(case["modules"]), observers=(), seed=seed,
        learning=learning, lateral=case["lateral"],
    )


def mirror(free: BrainState, plus: BrainState, brain) -> BrainState:
    """Control start for the negative phase: the settled positive change, reflected."""
    v = 2.0 * np.asarray(free.v) - np.asarray(plus.v)
    s = brain.neuron_model.activation(v)
    return BrainState(v=v, activation=s, adaptation=np.asarray(free.adaptation) + s
                      - np.asarray(free.activation), steps=0)


def settle(learner, drive, start, nudge, *, exact: bool):
    """The learner's qualified phase; ``exact`` re-solves with one-sweep checks."""
    cfg = learner.config
    began = time.perf_counter()
    if exact:
        phase = learner.brain.equilibrate(
            drive, state=start, budget=cfg.nudged_steps, tolerance=cfg.tolerance,
            nudge=nudge, damping=cfg.damping, chunk=1,
        )
    else:
        phase = learner._qualified_phase(drive, start, cfg.nudged_steps, nudge)
    return phase, time.perf_counter() - began


def lesson_pairs(brain, recoveries, drive, labels):
    learner = brain.learner
    cfg = learner.config
    target = learner.targets(labels)
    free = learner.free(drive)
    nudges = {sign: learner.nudge_for(target, sign * cfg.beta) for sign in (1.0, -1.0)}
    out: dict = {"free_steps": int(free.steps)}
    states: dict = {}
    for arm in ARMS:
        states[arm] = {}
        for sign in (1.0, -1.0):
            nudge = nudges[sign]
            if arm == "default" or (arm == "mirror" and sign > 0):
                start, render = free, 0.0
            elif arm == "mirror":
                start, render = mirror(free, states["default"][1.0], learner.brain), 0.0
            else:
                began = time.perf_counter()
                start = recoveries[arm].start(learner, free, nudge)
                render = time.perf_counter() - began
            phase, seconds = settle(learner, drive, start, nudge, exact=False)
            exact, _ = settle(learner, drive, start, nudge, exact=True)
            states[arm][sign] = phase.state
            out[f"{arm}_{'plus' if sign > 0 else 'minus'}"] = {
                "sweeps": int(phase.steps),
                "qualified": bool(np.all(phase.qualified)),
                "seconds": seconds + render,
                "render_seconds": render,
                "sweeps_to_tolerance": int(exact.steps),
                "residual": float(np.max(phase.residual)),
            }
    reference = learner.contrast(free, states["default"][1.0], states["default"][-1.0])
    scale = float(np.linalg.norm(reference[0]))
    for arm in ARMS[1:]:
        edges, neurons = learner.contrast(free, states[arm][1.0], states[arm][-1.0])
        out[f"{arm}_contrast_relative_difference"] = float(
            np.linalg.norm(edges - reference[0]) / scale
        )
        out[f"{arm}_bias_contrast_max_difference"] = float(np.abs(neurons - reference[1]).max())
        out[f"{arm}_state_max_difference"] = float(max(
            np.abs(np.asarray(states[arm][s].activation)
                   - np.asarray(states["default"][s].activation)).max()
            for s in (1.0, -1.0)
        ))
    learner.update(free, states["default"][1.0], states["default"][-1.0])
    for arm in ("copy", "fitted"):  # each start fits only from its own accepted phases
        for sign in (1.0, -1.0):
            recoveries[arm].observe(learner, free, states[arm][sign], nudges[sign])
    return out


def run_case(case: dict, seed: int, school, args) -> dict:
    inputs, labels = school
    brain = compose(case, seed)
    rng = np.random.default_rng(1000 + seed)
    refusals = 0
    for _ in range(args.acquire):
        rows = rng.choice(len(labels), args.batch, replace=False)
        try:
            brain.learner.step(brain.stimulus(inputs[rows], memory=False), labels[rows])
        except LearningPhaseError:
            refusals += 1
    recoveries = {"copy": RecoveryStart(fit=False), "fitted": RecoveryStart()}
    lessons = []
    for _ in range(args.lessons):
        rows = rng.choice(len(labels), args.batch, replace=False)
        try:
            lessons.append(
                lesson_pairs(brain, recoveries, brain.stimulus(inputs[rows], memory=False),
                             labels[rows])
            )
        except LearningPhaseError:
            refusals += 1
    return {
        "seed": seed,
        "refusals": refusals,
        "lessons": lessons,
        "fidelity": recoveries["fitted"].fidelity(),
        "recall_after": float(brain.learner.accuracy(
            brain.stimulus(inputs, memory=False), labels)),
    }


def trajectory(case: dict, seed: int, school, args) -> dict:
    """Fresh founders learn the same lesson sequence with and without the start."""
    inputs, labels = school
    result = {}
    for arm in ("default", "fitted"):
        brain = compose(case, seed)
        if arm == "fitted":
            brain.learner.recovery = RecoveryStart()
        rng = np.random.default_rng(2000 + seed)
        sweeps = refusals = 0
        began = time.perf_counter()
        for _ in range(args.trajectory):
            rows = rng.choice(len(labels), args.batch, replace=False)
            try:
                _, report = brain.learner.step(
                    brain.stimulus(inputs[rows], memory=False), labels[rows])
            except LearningPhaseError as refused:
                refusals += 1
                sweeps += sum(refused.report.get(f"{name}_steps", 0.0)
                              for name in ("nudged", "opposite"))
                continue
            sweeps += report["nudged_steps"] + report["opposite_steps"]
        seconds = time.perf_counter() - began
        result[arm] = {
            "refusals": refusals,
            "nudged_sweeps": int(sweeps),
            "seconds": seconds,
            "efficacy": brain.learner.brain.efficacy.copy(),
            "bias": brain.learner.brain.bias.copy(),
            "recall": float(brain.learner.accuracy(
                brain.stimulus(inputs, memory=False), labels)),
            "answers": brain.learner.predict(brain.stimulus(inputs, memory=False)).tolist(),
        }
    base, cand = result["default"], result["fitted"]
    summary = {
        arm: {k: v for k, v in result[arm].items() if k not in ("efficacy", "bias")}
        for arm in result
    }
    summary["efficacy_relative_difference"] = float(
        np.linalg.norm(cand["efficacy"] - base["efficacy"]) / np.linalg.norm(base["efficacy"]))
    summary["bias_max_difference"] = float(np.abs(cand["bias"] - base["bias"]).max())
    summary["same_answers"] = base["answers"] == cand["answers"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for key in ("acquire", "lessons", "batch", "trajectory"):
        setattr(args, key, int(protocol[key]))
    seeds = args.seeds or protocol["seeds"]
    cases = {name: case for name, case in protocol["cases"].items()
             if args.cases is None or name in args.cases}
    with np.load(FIXTURE, allow_pickle=False) as arrays:
        school = (arrays["school_inputs"], arrays["school_labels"])
    record = {
        "schema": "cadence-recovery-start-v1",
        "protocol": protocol,
        "protocol_sha256": sha256(args.protocol),
        "harness_sha256": sha256(Path(__file__)),
        "recovery_sha256": sha256(Path(cadence.__file__).with_name("recovery.py")),
        "learning_sha256": sha256(Path(cadence.__file__).with_name("learning.py")),
        "fixture_sha256": sha256(FIXTURE),
        "cadence_version": cadence.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "seeds": seeds,
        "cases": {},
    }
    for name, case in cases.items():
        record["cases"][name] = {
            "paired": [run_case(case, seed, school, args) for seed in seeds],
            "trajectory": [trajectory(case, seed, school, args) for seed in seeds]
            if case.get("trajectory") else [],
        }
        print(json.dumps(summarize(record["cases"][name]), indent=1), flush=True)
    record["summary"] = {name: summarize(case) for name, case in record["cases"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(record, indent=1))


def summarize(case: dict) -> dict:
    lessons = [lesson for founder in case["paired"] for lesson in founder["lessons"]]
    out: dict = {"lessons": len(lessons)}
    for arm in ARMS:
        sweeps = np.array([lesson[f"{arm}_plus"]["sweeps"] + lesson[f"{arm}_minus"]["sweeps"]
                           for lesson in lessons])
        exact = np.array([lesson[f"{arm}_plus"]["sweeps_to_tolerance"]
                          + lesson[f"{arm}_minus"]["sweeps_to_tolerance"] for lesson in lessons])
        seconds = np.array([lesson[f"{arm}_plus"]["seconds"] + lesson[f"{arm}_minus"]["seconds"]
                            for lesson in lessons])
        qualified = all(lesson[f"{arm}_plus"]["qualified"] and lesson[f"{arm}_minus"]["qualified"]
                        for lesson in lessons)
        out[arm] = {
            "nudged_sweeps_mean": float(sweeps.mean()),
            "sweeps_to_tolerance_mean": float(exact.mean()),
            "milliseconds_mean": float(1e3 * seconds.mean()),
            "all_qualified": qualified,
        }
        if arm != "default":
            base = np.array([lesson["default_plus"]["sweeps"] + lesson["default_minus"]["sweeps"]
                             for lesson in lessons])
            out[arm]["sweep_ratio_to_default"] = float(sweeps.sum() / base.sum())
            out[arm]["contrast_relative_difference_max"] = float(max(
                lesson[f"{arm}_contrast_relative_difference"] for lesson in lessons))
            out[arm]["state_max_difference"] = float(max(
                lesson[f"{arm}_state_max_difference"] for lesson in lessons))
    out["fidelity"] = [founder["fidelity"] for founder in case["paired"]]
    if case["trajectory"]:
        out["trajectory"] = [
            {k: v for k, v in t.items() if k not in ("default", "fitted")}
            | {arm: {k: t[arm][k] for k in ("nudged_sweeps", "seconds", "recall")}
               for arm in ("default", "fitted")}
            for t in case["trajectory"]
        ]
    return out


if __name__ == "__main__":
    main()
