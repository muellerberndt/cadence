"""One System 1 life through bootstrap, disruption, correction and continued use.

Run from this checkout: PYTHONPATH=src python examples/continuing_brain.py
The example uses the current source API, including Brain.last_settlement.

Two sensory cues have observed action labels. The independent environment
rewards the action actually executed, then changes its rule during the same
life. The body reveals the observed target after execution. During bootstrap
and repair, the application repeats a mistaken cue and teaches that witnessed
target. Successful actions receive their actual reward without a teacher.
This application policy does not gate reward learning or memory writes.
Brain.compose supplies reciprocal regions, working trace and consolidating
memory. Optional observers can extend the same graph. Teaching uses the default
finite rule; qualified teaching is an explicit choice.

Each executed action retains its free-answer residual and sweep count. Recorded
step sweeps also include feedback, teaching and reward eligibility, but exclude
residual checks, memory work and optimizer work. They are not joules or total
computation. Recording itself adds overhead and uses the inspectable CPU path.
Outcome counts describe this short run, not demonstrated acquisition, recovery,
or long-term retention. See docs/start.md#one-continuing-equilibrium-brain for the integration boundaries.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cadence import Brain, record_settlements


def observation(cue):
    return np.eye(2)[[cue]]


def label(cue, *, changed=False):
    """The independent body's rule, reversed after the declared disturbance."""
    return np.array([cue if changed else 1 - cue])


def execute(action, cue, *, changed=False):
    """Execute one action, measure its outcome, then reveal the next cue."""
    reward = (action == label(cue, changed=changed)).astype(float)
    return reward, 1 - cue


def same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as aa, np.load(second, allow_pickle=False) as bb:
        return set(aa.files) == set(bb.files) and all(
            np.array_equal(aa[key], bb[key]) for key in aa.files
        )


def issue(brain, cue, **feedback):
    """Keep the free answer separate from the other settlements inside step."""
    work = {"calls": 0, "sweeps": 0, "nudged_sweeps": 0}

    def count(record):
        work["calls"] += 1
        work["sweeps"] += record.steps
        if record.nudge is not None:
            work["nudged_sweeps"] += record.steps

    with record_settlements(count, label="continuing_brain.step"):
        action = brain.step(observation(cue), **feedback)
    # Save a snapshot now: the next call will replace last_settlement.
    free_answer = dict(brain.last_settlement)
    return action, free_answer, work


def run():
    brain = Brain.compose(2, 2, modules=(8,), observers=(), backend="cpu", seed=7)
    cue = 0
    # First try is free of a teacher; there is no previous action to reward.
    action, free_answer, step_work = issue(brain, cue)
    outcomes = []
    stages = []
    for phase, transitions, changed, correct_mistakes in (
        ("bootstrap", 16, False, True),
        ("unchanged_environment", 8, False, False),
        ("changed_environment", 8, True, False),
        ("repair", 16, True, True),
        ("continued_use", 8, True, False),
    ):
        correct = 0
        phase_outcomes = []
        for trial in range(transitions):
            # Score the issued action before revealing its measured outcome.
            reward, next_cue = execute(action, cue, changed=changed)
            observed_target = int(label(cue, changed=changed)[0])
            # The next issued action must belong to this teaching phase.
            # Otherwise its label would leak into the following free-use phase.
            correction = correct_mistakes and trial + 1 < transitions and not bool(reward[0])
            if correction:
                # Reward belongs to the executed action. Teacher belongs to
                # the CURRENT observation, so repeat the witnessed cue.
                next_cue = cue
            outcome = {
                "phase": phase, "cue": cue, "action": int(action[0]),
                "observed_target": observed_target,
                "reward": float(reward[0]),
                "next_cue": next_cue,
                "next_teacher": observed_target if correction else None,
                "free_answer": free_answer,
                # This work issued the action above, including any reward
                # for its predecessor and any teacher for its own cue.
                "step_settlement_work": step_work,
            }
            outcomes.append(outcome)
            phase_outcomes.append(outcome)
            correct += int(reward[0])
            cue = next_cue
            action, free_answer, step_work = issue(
                brain, cue, reward=reward, done=np.array([False]),
                teacher=np.array([observed_target]) if correction else None,
            )
        stages.append({
            "phase": phase, "executed_transitions": transitions,
            "correct_actions": correct, "witnessed_mistakes": transitions - correct,
            "corrective_lessons": sum(row["next_teacher"] is not None for row in phase_outcomes),
            "free_answer_sweeps": sum(row["free_answer"]["steps"] for row in phase_outcomes),
            "free_answer_residual_checks": sum(
                row["free_answer"]["residual_checks"] for row in phase_outcomes
            ),
            "max_free_answer_residual": max(
                row["free_answer"]["max_residual"] for row in phase_outcomes
            ),
        })

    # No reset or replacement across stages: both memory pathways remain live.
    assert brain.working_memory is not None
    assert brain.hippocampus is not None
    assert brain.hippocampus.writes == len(outcomes)

    with tempfile.TemporaryDirectory(prefix="cadence-continuing-") as folder:
        folder = Path(folder)
        # This checkpoint owns an issued action still awaiting its actual outcome.
        saved = brain.save(folder / "pending.npz")
        resumed = Brain.load(saved)
        reward, cue = execute(action, cue, changed=True)
        next_observation = observation(cue)
        next_action = brain.step(
            next_observation, reward=reward, done=np.array([False]),
        )
        resumed_action = resumed.step(
            next_observation, reward=reward, done=np.array([False]),
        )
        np.testing.assert_array_equal(next_action, resumed_action)
        before = brain.save(folder / "continued.npz")
        assert same_checkpoint(before, resumed.save(folder / "resumed.npz"))

        # Separate unit control: predict omits both memory pathways. It does
        # not measure the continuing policy or consume its pending feedback.
        recall = brain.predict(np.eye(2))
        assert recall.shape == (2,)
        assert same_checkpoint(before, brain.save(folder / "after-recall.npz"))

    return {
        "measurement_scope": (
            "Per-action free-answer diagnostics; step settlement sweeps also include "
            "previous feedback, current teaching and eligibility. They exclude residual "
            "checks, memory, optimizer and recording overhead. No energy or total-cost claim."
        ),
        "stages": stages,
        "outcomes": outcomes,
        "executed_transitions": len(outcomes) + 1,
        "associative_writes": brain.hippocampus.writes,
        "memory_free_graph_control": {
            "predictions": recall.tolist(), "changed_environment_labels": [0, 1],
        },
        "saved_feedback_continuation_identical": True,
        "free_queries_preserved_live_checkpoint": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
