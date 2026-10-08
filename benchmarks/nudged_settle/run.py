"""Paired cost and agreement of qualified nudged phases under different integration paths.

Every measured lesson settles one qualified free phase, then both nudged phases
once per arm on the same acquired brain, drive, labels and parameters, and once
more as a tight reference. Only the default arm's update is applied. See README.md.
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
from cadence import Brain, BrainState, LearnerConfig, LearningPhaseError, NudgedSettle

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "acquisition" / "fixture" / "school.npz"
ARMS = ("default", "halve1", "halve2", "candidate", "local_step")


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


def local_step(learner, drive, free, nudge, chunk=32):
    """Per-neuron step 1/(1 + g_i); the fixed point and the residual are the model's own."""
    graph, cfg = learner.brain, learner.config
    model = graph.neuron_model
    w = graph.dense()
    h = 1e-6
    slope = (model.activation(free.v + h) - model.activation(free.v - h)) / (2 * h)
    eta = 1.0 / (1.0 + slope @ np.abs(w))
    v = np.array(free.v, dtype=float)
    s = model.activation(v)
    standing = drive + graph.bias
    used = 0
    state = BrainState(v=v, activation=s, adaptation=np.asarray(free.adaptation), steps=0)
    error = graph.residual(drive, state, nudge=nudge)
    while used < cfg.nudged_steps and error.max() > cfg.tolerance:
        for _ in range(min(chunk, cfg.nudged_steps - used)):
            v = v + eta * (s @ w + standing + nudge.drive(s) - v)
            s = model.activation(v)
            used += 1
        state = BrainState(v=v, activation=s, adaptation=np.asarray(free.adaptation), steps=used)
        error = graph.residual(drive, state, nudge=nudge)
    return state, used, bool(error.max() <= cfg.tolerance)


def solve(arm, learner, drive, free, nudge):
    cfg = learner.config
    work = {"probe": 0.0, "discarded": 0.0, "fallback": 0.0}
    if arm == "local_step":
        state, sweeps, qualified = local_step(learner, drive, free, nudge)
        return state, sweeps, qualified, work
    if arm == "candidate":
        phase, extra = NudgedSettle().solve(learner, drive, free, nudge)
        work = {"probe": extra["probe_sweeps"], "discarded": extra["discarded_sweeps"],
                "fallback": extra["fallback"]}
    else:
        first = {"default": 0, "halve1": 1, "halve2": 2}[arm]
        phase = learner.brain.equilibrate(
            drive, state=free, budget=cfg.nudged_steps, tolerance=cfg.tolerance, nudge=nudge,
            damping=cfg.damping, first_halving=first,
        )
    return phase.state, int(phase.steps), bool(np.all(phase.qualified)), work


def lesson_pairs(brain, drive, labels):
    learner = brain.learner
    cfg = learner.config
    target = learner.targets(labels)
    free = learner.free(drive)
    nudges = {sign: learner.nudge_for(target, sign * cfg.beta) for sign in (1.0, -1.0)}
    reference = {
        sign: learner.brain.equilibrate(
            drive, state=free, budget=80000, tolerance=1e-10, nudge=nudge, damping=8,
            first_halving=4,
        )
        for sign, nudge in nudges.items()
    }
    out: dict = {"free_steps": int(free.steps),
                 "reference_qualified": all(bool(np.all(r.qualified)) for r in reference.values())}
    exact = learner.contrast(free, reference[1.0].state, reference[-1.0].state)[0]
    states: dict = {}
    for arm in ARMS:
        states[arm] = {}
        record = {"sweeps": 0, "qualified": True, "seconds": 0.0, "probe": 0.0,
                  "discarded": 0.0, "fallbacks": 0.0}
        for sign, nudge in nudges.items():
            began = time.perf_counter()
            state, sweeps, qualified, work = solve(arm, learner, drive, free, nudge)
            record["seconds"] += time.perf_counter() - began
            record["sweeps"] += sweeps
            record["qualified"] = record["qualified"] and qualified
            record["probe"] += work["probe"]
            record["discarded"] += work["discarded"]
            record["fallbacks"] += work["fallback"]
            states[arm][sign] = state
        edges = learner.contrast(free, states[arm][1.0], states[arm][-1.0])[0]
        record["contrast_error_to_reference"] = float(
            np.linalg.norm(edges - exact) / np.linalg.norm(exact))
        record["state_distance_to_reference"] = float(max(
            np.abs(np.asarray(states[arm][s].activation)
                   - np.asarray(reference[s].state.activation)).max() for s in nudges))
        out[arm] = record
    base = learner.contrast(free, states["default"][1.0], states["default"][-1.0])[0]
    for arm in ARMS[1:]:
        edges = learner.contrast(free, states[arm][1.0], states[arm][-1.0])[0]
        out[arm]["contrast_difference_to_default"] = float(
            np.linalg.norm(edges - base) / np.linalg.norm(base))
    learner.update(free, states["default"][1.0], states["default"][-1.0])
    return out


def run_case(case, seed, school, args):
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
    lessons = []
    for _ in range(args.lessons):
        rows = rng.choice(len(labels), args.batch, replace=False)
        try:
            lessons.append(lesson_pairs(brain, brain.stimulus(inputs[rows], memory=False),
                                        labels[rows]))
        except LearningPhaseError:
            refusals += 1
    return {"seed": seed, "refusals": refusals, "lessons": lessons}


def trajectory(case, seed, school, args):
    inputs, labels = school
    result = {}
    for arm in ("default", "candidate"):
        brain = compose(case, seed)
        if arm == "candidate":
            brain.learner.nudged_settle = NudgedSettle()
        rng = np.random.default_rng(2000 + seed)
        sweeps = refusals = fallbacks = 0.0
        began = time.perf_counter()
        for _ in range(args.trajectory):
            rows = rng.choice(len(labels), args.batch, replace=False)
            try:
                _, report = brain.learner.step(
                    brain.stimulus(inputs[rows], memory=False), labels[rows])
            except LearningPhaseError:
                refusals += 1
                continue
            sweeps += (report["nudged_steps"] + report["opposite_steps"]
                       + report.get("nudged_settle_probe_steps", 0.0)
                       + report.get("nudged_settle_discarded_steps", 0.0))
            fallbacks += report.get("nudged_settle_fallbacks", 0.0)
        result[arm] = {
            "refusals": refusals, "fallbacks": fallbacks, "nudged_sweeps": sweeps,
            "seconds": time.perf_counter() - began,
            "efficacy": brain.learner.brain.efficacy.copy(),
            "answers": brain.learner.predict(brain.stimulus(inputs, memory=False)).tolist(),
        }
    base, cand = result["default"], result["candidate"]
    return {
        arm: {k: v for k, v in result[arm].items() if k != "efficacy"} for arm in result
    } | {
        "efficacy_relative_difference": float(
            np.linalg.norm(cand["efficacy"] - base["efficacy"]) / np.linalg.norm(base["efficacy"])),
        "same_answers": base["answers"] == cand["answers"],
    }


def summarize(case):
    lessons = [lesson for founder in case["paired"] for lesson in founder["lessons"]]
    out: dict = {"lessons": len(lessons),
                 "references_qualified": all(lesson["reference_qualified"] for lesson in lessons)}
    base = sum(lesson["default"]["sweeps"] for lesson in lessons)
    for arm in ARMS:
        rows = [lesson[arm] for lesson in lessons]
        charged = sum(r["sweeps"] + r["probe"] + r["discarded"] for r in rows)
        out[arm] = {
            "nudged_sweeps_per_lesson": charged / len(rows),
            "ratio_to_default": charged / base,
            "milliseconds_per_lesson": 1e3 * sum(r["seconds"] for r in rows) / len(rows),
            "all_qualified": all(r["qualified"] for r in rows),
            "contrast_error_to_reference_max": max(r["contrast_error_to_reference"] for r in rows),
            "contrast_error_to_reference_mean": float(np.mean(
                [r["contrast_error_to_reference"] for r in rows])),
            "state_distance_to_reference_max": max(r["state_distance_to_reference"] for r in rows),
            "fallback_phases": sum(r["fallbacks"] for r in rows),
        }
    out["trajectory"] = [
        {key: ({k: v for k, v in value.items() if k != "answers"}
               if isinstance(value, dict) else value) for key, value in founder.items()}
        for founder in case["trajectory"]
    ]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(args.protocol.read_text())
    for key in ("acquire", "lessons", "batch", "trajectory"):
        setattr(args, key, int(protocol[key]))
    seeds = args.seeds or protocol["seeds"]
    cases = {name: case for name, case in protocol["cases"].items()
             if args.cases is None or name in args.cases}
    with np.load(FIXTURE, allow_pickle=False) as arrays:
        school = (arrays["school_inputs"], arrays["school_labels"])
    source = Path(cadence.__file__).parent
    record = {
        "schema": "cadence-nudged-settle-v1",
        "protocol": protocol,
        "protocol_sha256": sha256(args.protocol),
        "harness_sha256": sha256(Path(__file__)),
        "sources_sha256": {name: sha256(source / name)
                           for name in ("nudged_settle.py", "learning.py", "brain.py")},
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
        print(name, json.dumps(summarize(record["cases"][name]), indent=1), flush=True)
    record["summary"] = {name: summarize(case) for name, case in record["cases"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
