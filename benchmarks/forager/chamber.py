"""The forager chamber: a body taught by imitation, its cortex reading the code or the code and
its senses (cadence 0.102.0).

A brain with the senses of a small animal (``world.SENSES``: eyes, ears, a heard call, the thing
it attends, the nearest food and its body; 104 inputs), two modules of 32 neurons and a fixed
code of 128 units where the senses meet is taught by a forager that turns toward the nearest
plant in sight, walks to it and bites it (``world.teach``). The arms differ in what the cortex
reads (``ARMS``): the code (the founder), the code and the senses (``cortex_reads="both"``), the
senses with the memories keyed on the code (``cortex_reads="observation"``), and the senses
with no code at all.

1. **The life.** ``BODIES`` bodies, one stream each, live ``MOMENTS`` moments in worlds of their
   own. At every moment the brain is shown the teacher's action (``Brain.lesson``, each action's
   rows weighed by the inverse of its share, every action pulling alike, at most ``MOST``) and
   then acts greedily (``Brain.act``). Half of the bodies move as the brain answers and half as the
   teacher shows. Every ``SWAP`` moments the halves change places, the worlds are drawn anew and
   the streams begin a new scene (``Brain.reset``).
2. **Agreement while taught**: the share of the brain's answers that are the teacher's, over the
   last ``SWAP`` moments of lessons before each reading.
3. **Agreement on its own path**: a frozen copy (``copy.deepcopy``, greedy acts, no lesson) lives
   ``HELD`` moments in ``BODIES`` worlds no lesson used, every body moving as the copy answers;
   the share of its answers that are the teacher's at the states it reaches, by action, and the
   plants it eats. The same worlds with the teacher acting give the teacher's plants, and with a
   uniform random choice a random body's.
4. **On the teacher's path**: the same worlds with the teacher acting, a frozen copy and the
   newborn brain of the same founder settling on every moment (greedy acts the world does not
   take), so every arm is read on the same states: the copy's agreement with the teacher, and the
   turn in the cortex. On the moments at which a plant is in sight and the teacher turns left,
   turns right or walks on (toward the plant, or from a rock or a wall in the way), a ridge
   readout of that turn is fitted on every other sample, in the order the moments and the bodies
   give them, and read on the rest, from the observation itself, its code, and the association's
   settled activity of the copy and of the newborn. Balanced accuracy, chance one third. The same
   readout fitted on the even bodies and read on the odd ones (``<source>_bodies``, added after
   the declared run) keeps each body's moments on one side.

    PYTHONPATH=src python benchmarks/forager/chamber.py --founders 0 1 --out dev.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from world import ACTIONS, BITE, LEFT, RIGHT, SENSES, WALK, Worlds, flat, teach  # noqa: E402

import cadence as cd  # noqa: E402

CODE = 128
MODULES = (32, 32)
RECALL = 16  # slots a stream
BODIES = 16
MOMENTS = 1200
READ = 300  # moments of lessons between readings
SWAP = 100
HELD = 150
MOST = 8.0
RIDGE = 1.0
ARMS: dict[str, dict[str, Any]] = {
    "code": {"code": CODE},
    "both": {"code": CODE, "cortex_reads": "both"},
    "observation": {"code": CODE, "cortex_reads": "observation"},
    "none": {},
}


def compose(founder: int, arm: str) -> cd.Brain:
    return cd.Brain.compose(senses=SENSES, actions=len(ACTIONS), modules=MODULES,
                            recall_capacity=RECALL, seed=founder, backend="cpu", **ARMS[arm])


def association(brain: cd.Brain) -> np.ndarray:
    state = np.asarray(brain.basal_ganglia.state.activation, dtype=float)
    return state[:, np.asarray(brain.connectome.populations["association"])]


def weights_of(labels: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Each row's weight: every action pulls with the same total weight, at most ``MOST``."""
    share = seen / seen.sum()
    return np.asarray(np.minimum(
        1.0 / (np.count_nonzero(share) * np.maximum(share[labels], 1e-9)), MOST))


class Lessons:
    """The life of lessons of one brain: ``bodies`` worlds of their own, taught ``moments`` at a
    time (``Lessons.next``)."""

    def __init__(self, brain: cd.Brain, founder: int, bodies: int = BODIES) -> None:
        self.brain, self.bodies, self.founder = brain, bodies, founder
        self.world = Worlds(bodies, seed=founder * 7 + 1)
        self.seen = np.full(len(ACTIONS), 1e-9)
        self.own = np.arange(bodies) % 2 == 0
        self.t = 0

    def next(self, moments: int) -> dict:
        """``moments`` more moments of lessons; the agreement over the last ``SWAP``."""
        brain, world, bodies = self.brain, self.world, self.bodies
        agree: list[float] = []
        for _ in range(moments):
            if self.t and self.t % SWAP == 0:  # new worlds, a new scene; the halves change places
                self.own = ~self.own
                self.world = world = Worlds(bodies, seed=self.founder * 7 + 1 + 7919 * self.t)
                brain.reset()
            self.t += 1
            sensed = world.senses()
            shown = teach(world)
            self.seen += np.bincount(shown, minlength=len(ACTIONS))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                brain.lesson(sensed, shown, weights=weights_of(shown, self.seen))
            answered = np.asarray(brain.act(sensed, greedy=True)).reshape(bodies)
            agree.append(float((answered == shown).mean()))
            world.step(np.where(self.own, answered, shown))
        return {"agreement": float(np.mean(agree[-SWAP:]))}


def own_path(brain: cd.Brain | None, founder: int, *, mind: str = "brain",
             moments: int = HELD, bodies: int = BODIES) -> dict:
    """``mind`` on worlds no lesson used, every body moving as it answers: a frozen copy of
    ``brain`` (greedy acts, no lesson), the teacher, or a uniform random choice. The plants
    eaten, and for the brain its agreement with the teacher at the states it reaches, by
    action."""
    world = Worlds(bodies, seed=founder * 7 + 1000)
    rng = np.random.default_rng(founder * 7 + 2000)
    frozen = None if brain is None else copy.deepcopy(brain)
    if frozen is not None:
        frozen.reset()
    right = np.zeros((len(ACTIONS), 2))
    for _ in range(moments):
        sensed = world.senses()
        shown = teach(world)
        if mind == "teacher":
            answered = shown
        elif mind == "random":
            answered = rng.integers(0, len(ACTIONS), bodies)
        else:
            assert frozen is not None
            answered = np.asarray(frozen.act(sensed, greedy=True)).reshape(bodies)
            np.add.at(right, (shown, (answered == shown).astype(int)), 1)
        world.step(answered)
    out: dict[str, Any] = {"eaten": float(world.eaten.mean())}
    if mind == "brain":
        total = right.sum(1)
        out["agreement"] = round(float(right[:, 1].sum() / right.sum()), 3)
        out["by_action"] = {ACTIONS[a]: (round(float(right[a, 1] / total[a]), 3), int(total[a]))
                            for a in range(len(ACTIONS)) if total[a]}
    return out


def teacher_path(brain: cd.Brain, newborn: cd.Brain, founder: int, *, moments: int = HELD,
                 bodies: int = BODIES) -> dict:
    """The teacher's path on the held-out worlds, every arm read on the same states: a frozen
    copy of ``brain`` and of the ``newborn`` settle on each moment (greedy acts the world does
    not take). The copy's agreement with the teacher, and the turn readout from the observation,
    its code and the two association states."""
    world = Worlds(bodies, seed=founder * 7 + 1000)
    frozen, infant = copy.deepcopy(brain), copy.deepcopy(newborn)
    frozen.reset()
    infant.reset()
    agree: list[np.ndarray] = []
    right = np.zeros((len(ACTIONS), 2))
    samples: dict[str, list] = {"turn": [], "observation": [], "code": [], "state": [],
                                "newborn": [], "body": []}
    for _ in range(moments):
        sensed = world.senses()
        shown = teach(world)
        answered = np.asarray(frozen.act(sensed, greedy=True)).reshape(bodies)
        infant.act(sensed, greedy=True)
        agree.append(answered == shown)
        np.add.at(right, (shown, (answered == shown).astype(int)), 1)
        seen, _, _ = world.food()
        toward = seen & (shown != BITE)
        if toward.any():
            x = flat(sensed)
            samples["turn"].append(shown[toward])
            samples["body"].append(np.flatnonzero(toward))
            samples["observation"].append(x[toward])
            samples["code"].append(frozen.observation_code(sensed)[toward])
            samples["state"].append(association(frozen)[toward])
            samples["newborn"].append(association(infant)[toward])
        world.step(shown)
    total = right.sum(1)
    return {"agreement": round(float(np.mean(agree)), 3), "readout": readout(samples),
            "by_action": {ACTIONS[a]: (round(float(right[a, 1] / total[a]), 3), int(total[a]))
                          for a in range(len(ACTIONS)) if total[a]}}


def readout(samples: dict[str, list]) -> dict[str, float]:
    """Balanced accuracy of a ridge readout of the turn (left, walk on, right) from each source,
    fitted on every other sample and read on the rest (the declared split), and fitted on the
    even bodies and read on the odd ones (``<source>_bodies``)."""
    turn = np.concatenate(samples["turn"])
    classes = [LEFT, WALK, RIGHT]
    keep = np.isin(turn, classes)
    turn = turn[keep]
    splits = {"": np.arange(len(turn)) % 2 == 0}
    if samples.get("body"):
        splits["_bodies"] = np.concatenate(samples["body"])[keep] % 2 == 0
    out = {}
    for source in ("observation", "code", "state", "newborn"):
        if not samples[source]:
            continue
        x = np.concatenate(samples[source])[keep]
        x = np.hstack([x, np.ones((len(x), 1))])
        y = np.eye(3)[[classes.index(t) for t in turn]]
        for tag, half in splits.items():
            w = np.linalg.solve(x[half].T @ x[half] + RIDGE * np.eye(x.shape[1]),
                                x[half].T @ y[half])
            said = np.asarray(classes)[np.argmax(x[~half] @ w, axis=1)]
            recalls = [float(np.mean(said[turn[~half] == c] == c)) for c in classes
                       if (turn[~half] == c).any()]
            out[source + tag] = round(float(np.mean(recalls)), 3)
    out["moments"] = int(len(turn))
    return out


def life(founder: int, arm: str, moments: int = MOMENTS) -> dict:
    """One brain's life of lessons, read every ``READ`` moments; ``mean`` holds the readings
    averaged over the readings of the life."""
    began = time.time()
    brain = compose(founder, arm)
    newborn = copy.deepcopy(brain)
    lessons = Lessons(brain, founder)
    out: dict[str, Any] = {"founder": founder, "arm": arm, "parameters": int(brain.parameters()),
                           "readings": []}
    for _ in range(moments // READ):
        reading = {"moments": lessons.t + READ, "taught": lessons.next(READ)}
        reading["own"] = own_path(brain, founder)
        reading["teacher"] = teacher_path(brain, newborn, founder)
        out["readings"].append(reading)
    out["mean"] = mean(out["readings"])
    out["seconds"] = round(time.time() - began, 1)
    return out


def mean(readings: list[dict]) -> dict:
    """The readings of a life averaged over its readings."""
    pick = {
        "taught": lambda r: r["taught"]["agreement"],
        "own": lambda r: r["own"]["agreement"],
        "eaten": lambda r: r["own"]["eaten"],
        "teacher": lambda r: r["teacher"]["agreement"],
    }
    read = readings[0]["teacher"]["readout"]
    for source in [key for key in read if key != "moments"]:
        pick[f"turn_{source}"] = lambda r, s=source: r["teacher"]["readout"][s]
    return {name: round(float(np.mean([f(r) for r in readings])), 3)
            for name, f in pick.items()}


def baselines(founder: int) -> dict:
    """The teacher's and a random body's plants on the held-out worlds of ``founder``."""
    return {"teacher": own_path(None, founder, mind="teacher")["eaten"],
            "random": own_path(None, founder, mind="random")["eaten"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--founders", nargs="+", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = []
    for founder in args.founders:
        for arm in args.arms:
            results.append(life(founder, arm))
            results[-1]["baselines"] = baselines(founder)
            print(json.dumps(results[-1]), flush=True)
            args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
