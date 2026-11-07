"""A walking rhythm: left, right, left, right under one constant drive.

Run from this checkout: PYTHONPATH=src python examples/walking_rhythm.py

Two founders with identical weights are taught the same lessons. Each has four
continuing streams (four legs on four treadmills, if you like). A bout opens with
one cue event that names the first foot and continues with eleven identical drive
events; the lesson at each event is the opposite of the stream's own last step.
The brain sees nothing of its own steps except what it carries forward itself.

The control carries only its working trace, a copy of the settled association
state before the decision, so under identical drive it carries which foot it moved
only through whatever margin the motor competition left. The walker also carries
an efference copy of the command it issued (`Brain.compose(...,
efference_amplitude=3.0, efference_decay=0.0)`): one efference neuron per motor
neuron, driven by the one-hot of the last step and read by the association region
through a plastic projection. Both are the same reciprocal patch net, the same
local repair law and the same lessons; the copy is a gene with zero as its control.

After teaching, both walk freely for 32 events after a cue, then through a pause
of two silent events and one distractor event, and the walker is saved mid-stride
and resumed as a twin. A uniform-random policy is the baseline for the alternation
score. The steady-rhythm chamber (`benchmarks/rhythm`) is the frozen instrument
with its five fresh seeds and receipts; this example shows one founder and claims
nothing beyond what it prints.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cadence import Brain

ROWS, INPUTS = 4, 4
DRIVE, DISTRACTOR, CUE_LEFT, CUE_RIGHT = range(INPUTS)
FEET = "LR"
SEED = 5


def founder(efference: float) -> Brain:
    """The steady-rhythm chamber's selected System 1; the copy is the only difference."""
    genes = {"efference_amplitude": efference, "efference_decay": 0.0} if efference else {}
    return Brain.compose(
        INPUTS,
        2, backend="cpu",
        modules=(32,),
        seed=SEED,
        episodic=False,
        working_memory_decay=0.1,
        working_memory_amplitude=3.0,
        **genes,
    )


def observation(kind: str, cues: np.ndarray | None = None) -> np.ndarray:
    x = np.zeros((ROWS, INPUTS))
    if kind == "pause":
        return x
    if kind == "distractor":
        x[:, DISTRACTOR] = 1.0
        return x
    x[:, DRIVE] = 1.0
    if kind == "cue":
        assert cues is not None
        x[np.arange(ROWS), CUE_LEFT + cues] = 1.0
    return x


def act(brain: Brain, x: np.ndarray, work: dict) -> np.ndarray:
    answer = brain.act(x, greedy=True)
    work["acts"] += 1
    work["act_sweeps"] += int(brain.last_settlement["steps"])
    return answer


def teach(brain: Brain, rng: np.random.Generator, bouts: int, work: dict) -> list[float]:
    """The chamber's `every` arm: a lesson on the live stimulus, then the free step."""
    previous = np.zeros(ROWS, dtype=np.int64)
    agreement = []
    for _ in range(bouts):
        cues = rng.permutation([0, 0, 1, 1])
        hits = 0
        for t in range(12):
            x = observation("cue", cues) if t == 0 else observation("drive")
            labels = cues if t == 0 else 1 - previous
            _, report = brain.learner.step(brain.stimulus(x), labels)
            work["lessons"] += 1
            work["lesson_sweeps"] += int(report["total_steps"])
            previous = act(brain, x, work)
            hits += int(np.sum(previous == labels))
        agreement.append(hits / (12 * ROWS))
    return agreement


def alternation(steps: np.ndarray) -> float:
    """The share of successive events at which a stream changed foot."""
    return float(np.mean(steps[1:] != steps[:-1]))


def walk(brain: Brain, events: list[np.ndarray], work: dict) -> np.ndarray:
    return np.stack([act(brain, x, work) for x in events])


def run() -> dict:
    walker, control = founder(3.0), founder(0.0)
    work = {
        name: {"lessons": 0, "lesson_sweeps": 0, "acts": 0, "act_sweeps": 0}
        for name in ("walker", "control")
    }
    # One cue schedule for both founders: the same generator seed, drawn twice.
    taught = {
        "walker": teach(walker, np.random.default_rng(SEED + 1), 24, work["walker"]),
        "control": teach(control, np.random.default_rng(SEED + 1), 24, work["control"]),
    }

    # Free walking after a cue: the cued foot first, then the beat with no further cue.
    cues = np.array([0, 1, 0, 1])
    events = [observation("cue", cues)] + [observation("drive")] * 32
    free = {
        name: walk(brain, events, work[name])
        for name, brain in (("walker", walker), ("control", control))
    }
    random_steps = np.random.default_rng(SEED + 2).integers(0, 2, size=free["walker"].shape)

    # A pause of two silent events and one distractor, then the beat again.
    disturbance = (
        [observation("pause")] * 2 + [observation("distractor")] + [observation("drive")] * 16
    )
    after = {
        name: walk(brain, disturbance, work[name])
        for name, brain in (("walker", walker), ("control", control))
    }

    # Saved mid-stride, resumed as a twin, and walked beside the original.
    with tempfile.TemporaryDirectory() as directory:
        probe = walker.save(Path(directory) / "mid-stride.npz")
        twin = Brain.load(probe)
        more = [observation("drive")] * 16
        own, resumed = (
            walk(walker, more, work["walker"]),
            walk(twin, more, {"acts": 0, "act_sweeps": 0}),
        )
        continued = bool(np.array_equal(own, resumed))

    return {
        "seed": SEED,
        "genes": {
            "walker": {"efference_amplitude": 3.0, "efference_decay": 0.0},
            "control": "no copy",
        },
        "taught_agreement_by_bout": {
            name: [round(v, 2) for v in values] for name, values in taught.items()
        },
        "free_walk_steps_row0": {
            name: "".join(FEET[s] for s in steps[:, 0]) for name, steps in free.items()
        },
        "free_walk_alternation": {
            **{name: round(alternation(steps[1:]), 2) for name, steps in free.items()},
            "uniform_random": round(alternation(random_steps[1:]), 2),
        },
        "after_disturbance_alternation": {
            name: round(alternation(steps[3:]), 2) for name, steps in after.items()
        },
        "after_disturbance_steps_row0": {
            name: "".join(FEET[s] for s in steps[:, 0]) for name, steps in after.items()
        },
        "resumed_twin_walks_identically": continued,
        "work": work,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
