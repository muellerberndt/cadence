"""New associations by pistachios alone: a never-seen thing (a lemon) and its new word ("zabu").

After pretraining, on a copy of the parrot, one exchange a trial, the parrot's voice sampled from
its own motor activity (``temperature``):

- odd trials, **the trainer names it**: the lemon alone in view, the trainer says "zabu" (the
  echo game of the words stage); a pistachio where the parrot says "zabu" back;
- even trials, **the trainer asks**: the lemon alone in view, "what is it ?"; a pistachio where
  the parrot says "zabu".

No label and no teacher's answer: a pistachio makes the exchange's acts lasting (the
three-factor rule: each act's trace, the pistachio the third factor): the brain is taught its
own acts of the rewarded turn, moment by moment, as it lived them, every one of them (the
lessons only where the answer differs are set aside for it). An exchange without a pistachio
teaches nothing. Every two trials a frozen copy answers "what is it ?" about 16 fresh
lemons (other places and glimpse noise), greedily; the reading is the trials (and pistachios)
until it says "zabu" at .9. Then, without any further lesson: "what objects are round ?" and
"what can you eat ?" in scenes with a lemon among trained things (does the new thing inherit
round and food under its new name?), and the trained kinds' names (anything forgotten?).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textworld as tw  # noqa: E402

NEW_WORD = "zabu"
NEW_KIND = "lemon"


class Sampler:
    """A brain policy whose voice is sampled from its motor activity at ``temperature``; it
    keeps every moment it lived and the acts it took, for a pistachio's lesson."""

    def __init__(self, policy: Any, rng: np.random.Generator, temperature: float):
        self.policy, self.rng, self.temperature = policy, rng, temperature
        self.lived: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
        self.phases: list[str] = []  # set by the chamber each moment
        layout = policy.layout
        motor = np.asarray(list(policy.brain.connectome.populations["motor"]))
        self.voice = motor[:layout.voice]

    def begin(self, n: int) -> None:
        self.lived = []
        self.policy.begin(n)

    def moment(self, x: np.ndarray, labels: Any, mask: np.ndarray, learn: bool) -> np.ndarray:
        a = self.policy.moment(x, None, mask, False)
        act = np.asarray(self.policy.brain.basal_ganglia.state.activation)[:, self.voice]
        z = act / self.temperature
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        a = a.copy()
        for k in range(len(a)):
            # the voice is held while the trainer speaks and the eyes search (born suppression
            # while hearing); it is free in the parrot's turn
            if self.phases and self.phases[k] in ("answer", "closing"):
                a[k, 0] = int(self.rng.choice(len(p[k]), p=p[k]))
        self.lived.append((x.copy(), a.copy(), list(self.phases)))
        return a


def lemon_alone(rng: np.random.Generator) -> tw.Scene:
    place = tw.PLACES[int(rng.integers(len(tw.PLACES)))]
    return tw.Scene([tw.Thing(NEW_KIND, "yellow", place, tw.look(NEW_KIND, "yellow", place, rng))])


def reward_lesson(policy: Any, lived: list[tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
    """The pistachio's lesson: the exchange relived, each moment of the parrot's turn taught the
    act it took there. The pistachio gates the whole turn's plasticity (the third factor), so the
    turn's acts are taught whether or not the brain would take them again unprompted: lessons
    only where the answer differs are set aside for the pistachio's lesson."""
    brain = policy.brain
    surprise = bool(brain.learner.config.surprise)
    if surprise:
        brain.retune(learning_surprise=False)
    try:
        policy.begin(len(lived[0][0]))
        for x, a, phases in lived:
            w = np.array([p in ("answer", "closing") for p in phases], float)
            if w.any():
                brain.lesson(x, a, weights=None if w.all() else w)
            brain.act(x, greedy=True)
    finally:
        if surprise:
            brain.retune(learning_surprise=True)


def read_name(policy: Any, rng: np.random.Generator, n: int = 16) -> float:
    items = []
    for _ in range(n):
        s = lemon_alone(rng)
        items.append((s, tw.Question("it", ["what", "is", "it", "?"], None, {0: [NEW_WORD]}, [])))
    eps = tc.run(policy, policy.layout, items, "free", learn=False)
    return float(np.mean([e.reply() == [NEW_WORD, "."] for e in eps]))


def inherit(policy: Any, rng: np.random.Generator, n: int = 32) -> dict[str, float]:
    """Round and food under the new name: the lemon among trained things."""
    out = {}
    for form, value in (("which_shape", "round"), ("which_food", "food")):
        items = []
        while len(items) < n:
            sc = tw.scene(rng, int(rng.integers(2, 4)), include=((NEW_KIND, "yellow"),))
            q = tw.question(form, sc, rng)
            if q is None or q.meta.get("value") != value:
                continue
            clauses = {i: [NEW_WORD if t.kind == NEW_KIND else t.kind]
                       for i, t in enumerate(sc.things)
                       if (t.shape == "round" if value == "round" else t.food)}
            items.append((sc, tw.Question(form, q.words, None, clauses, [])))
        eps = tc.run(policy, policy.layout, items, "free", learn=False)
        out[form] = float(np.mean([e.reply() == e.q.reply(e.scene) for e in eps]))
    return out


def new_word(policy: Any, rng: np.random.Generator, *, trials: int = 60,
             temperature: float = 0.1, every: int = 2) -> dict[str, Any]:
    q = copy.copy(policy)
    q.brain = copy.deepcopy(policy.brain)
    sampler = Sampler(q, rng, temperature)
    pistachios, curve, learned = 0, [], None
    for trial in range(1, trials + 1):
        sc = lemon_alone(rng)
        if trial % 2 == 1:
            item = (sc, tw.Question("echo", [NEW_WORD], None, {0: [NEW_WORD]}, []))
        else:
            item = (sc, tw.Question("it", ["what", "is", "it", "?"], None, {0: [NEW_WORD]}, []))
        eps = tc.run(sampler, q.layout, [item], "free", learn=False)
        if eps[0].reply() == [NEW_WORD, "."]:
            pistachios += 1
            reward_lesson(q, sampler.lived)
        if trial % every == 0:
            r = read_name(q, np.random.default_rng([trial, 5]))
            curve.append({"trial": trial, "pistachios": pistachios, "read": r})
            if r >= 0.9 and learned is None:
                learned = {"trial": trial, "pistachios": pistachios}
                break
    import textparrot as tp

    old = tp.read(q, q.layout, np.random.default_rng(99), ("it",), 64)["it"]["right"]
    return {"learned": learned, "curve": curve, "inherit": inherit(q, np.random.default_rng(98)),
            "old_names_after": old, "parrot": q}
