"""A founder's life in the meaning chamber (cadence#334): imitation first, then pistachios.

The parrot is one deep composed brain (``cadence.meaning.compose``). It lives ``streams`` rooms
side by side (rows of one batch), all in one kind of trial at a time. A trial is a few moments;
at each moment the parrot sees the things it holds (their features, ``world.seen``) and hears a
word, silence, or its own voice, settles, and acts. At the trial's turn (the moment the last
word is heard) the voice says what the brain chose: a word the parrot has heard, or nothing.
The next moment it hears itself.

**The teacher's trials** (only the world and the teacher are designed):

- ``babble``: silence and the parrot's turn; it hears what it said. No reward.
- ``imitate``: the teacher says a word, and that moment is the parrot's turn. The parrot's own
  comparison of what it hears itself say with the word it heard (``self_reward``, a born
  evaluator) is its reward.
- ``name`` (``scaffold``): the teacher holds a thing up and says its name, the parrot's turn;
  a pistachio when it says the name.
- every question form (``turn``): the teacher asks; the parrot's turn; a pistachio for the right
  answer. In ``scaffold`` a second trainer answers first (the model/rival game) and the parrot
  may say it after.
- ``test``: the parrot's turn on held-out rooms, things and combinations, greedy, on a copy of
  the brain that learns nothing.

**The self-hearing lesson** (``babble_lesson``): hearing itself say a word, the parrot takes a
lesson that the act it just made gives this sound (the inverse model of its voice, built by
babbling). Its label is the parrot's own act; no lesson names a meaning.
"""

from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import meaning_world as world  # noqa: E402

from cadence.meaning import HAND_SET, Eyes, Forms, compose, observation  # noqa: E402

SELF = "<self>"  # the parrot hears its own voice at this moment
ELIGIBILITY_STEPS = 12  # each reward-nudged phase of a composed brain (a constant)


@dataclass
class Trial:
    room: world.Room
    words: list[str | None]  # the word heard at each moment (SELF: its own voice)
    turn: int                # the moment whose act the voice says
    answer: str | None       # the right word (None: no right word)
    template: str | None     # the word the parrot compares its own voice with (imitation)
    form: str


class Parrot:
    """The brain with its senses and its voice for one founder."""

    def __init__(self, seed: int, genome: dict[str, Any] | None = None, *, streams: int = 8,
                 slots: int = 5, backend: str = "cpu"):
        self.seed, self.streams = int(seed), int(streams)
        self.genome = {**HAND_SET, **(genome or {})}
        self.lex = world.lexicon(seed)
        self.eyes = Eyes(slots=slots)
        self.forms = Forms(seed=seed)
        words = self.lex.every()
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 334, 2])).permutation(
            len(words))
        self.voice = [words[int(i)] for i in order]  # motor neuron i says voice[i]; last: silence
        self.unit = {w: i for i, w in enumerate(self.voice)}
        self.silence = len(self.voice)
        for w in words:  # the ear has heard every word of the world before the life begins
            self.forms(w)
        self.inputs = self.eyes.width + self.forms.width
        self.brain = compose(self.genome, self.inputs, len(self.voice), ear=self.forms.width,
                             seed=self.seed, backend=backend)
        self.work = {"moments": 0, "act_sweeps": 0.0, "lesson_rows": 0.0, "lesson_sweeps": 0.0,
                     "reward_moments": 0, "reward_row_sweeps": 0.0}
        self.synapses = int(self.brain.connectome.synapses)

    def said(self, action: int) -> str | None:
        return None if int(action) >= self.silence else self.voice[int(action)]

    def _observe(self, trials: list[Trial], t: int, last: list[str | None],
                 rng: np.random.Generator) -> np.ndarray:
        rows = []
        for i, tr in enumerate(trials):
            heard = tr.words[t]
            heard = last[i] if heard == SELF else heard
            rows.append(observation(self.eyes, self.forms, world.seen(tr.room, rng), heard))
        return np.asarray(rows)

    def live(self, trials: list[Trial], rng: np.random.Generator, *,
             pistachio: bool = True) -> list[bool]:
        """One block of trials, one a stream, learning; returns which turns were right."""
        brain, g = self.brain, self.genome
        length = len(trials[0].words)
        last: list[str | None] = [None] * len(trials)
        right = [False] * len(trials)
        voiced = np.zeros(len(trials), dtype=np.int64)
        for t in range(length):
            x = self._observe(trials, t, last, rng)
            reward = np.zeros(len(trials))
            if t == trials[0].turn + 1:
                for i, tr in enumerate(trials):
                    if pistachio and tr.answer is not None and last[i] == tr.answer:
                        reward[i] += 1.0
                    if tr.template is not None and last[i] == tr.template:
                        reward[i] += float(g["self_reward"])
            if brain.pending_feedback:
                report = brain.learn(reward, np.full(len(trials), t == length - 1), x)
                self.work["reward_moments"] += 1
                # the free solve, and the two eligibility phases of 12 sweeps each (a constant)
                self.work["reward_row_sweeps"] += len(trials) * (
                    float(report.get("free_steps", 0.0)) + 2 * ELIGIBILITY_STEPS)
            if t == trials[0].turn + 1 and int(g["babble_lesson"]):
                taught = np.array([last[i] is not None for i in range(len(trials))], float)
                if taught.any():
                    report = brain.lesson(x, voiced, weights=taught)
                    self.work["lesson_rows"] += float(report.get("attempted_presentations",
                                                                 taught.sum()))
                    self.work["lesson_sweeps"] += float(report.get("total_row_sweeps", 0.0))
            action = brain.act(x)
            self.work["moments"] += 1
            self.work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
            if t == trials[0].turn:
                voiced = np.asarray(action).reshape(len(trials), -1)[:, 0].astype(np.int64)
                last = [self.said(a) for a in voiced]
                right = [w is not None and w == (tr.answer if tr.answer is not None
                                                 else tr.template)
                         for tr, w in zip(trials, last, strict=True)]
            elif t != trials[0].turn:
                last = [None] * len(trials) if t > trials[0].turn else last
        return right

    def test(self, trials: list[Trial], rng: np.random.Generator) -> list[str | None]:
        """The words said at the turn, greedily, by a copy of the brain that learns nothing."""
        frozen = copy.deepcopy(self.brain)
        frozen.reset()
        said: list[str | None] = [None] * len(trials)
        for t in range(trials[0].turn + 1):
            x = self._observe(trials, t, [None] * len(trials), rng)
            action = frozen.act(x, greedy=True)
            if t == trials[0].turn:
                said = [self.said(a) for a in np.asarray(action).reshape(len(trials), -1)[:, 0]]
        return said


# -- the teacher's trials


def _q(lex: world.Lexicon, *names: str) -> list[str | None]:
    return [lex.question[n] for n in names]


def trial(form: str, mode: str, p: Parrot, rng: np.random.Generator, *,
          held: set[tuple[int, int]], test: bool = False) -> Trial:
    """One trial of ``form`` (babble, imitate, name, colour_of, by_colour, by_round, where,
    by_place, near_colour) in ``mode`` (scaffold, turn). Kind-colour pairs in ``held`` appear
    only in test rooms."""
    lex = p.lex
    kinds = list(range(world.TRAINED))
    if form == "babble":
        r = world.room(rng, kinds, int(rng.integers(1, 4)))
        return Trial(r, [None, SELF], 0, None, None, form)
    if form == "imitate":
        r = world.room(rng, kinds, int(rng.integers(1, 4)))
        w = lex.every()[int(rng.integers(len(lex.every())))]
        return Trial(r, [w, SELF], 0, None, w, form)
    for _ in range(200):
        r = world.room(rng, kinds, int(rng.integers(2, 5)))
        pairs = {(t.kind, t.colour) for t in r.things}
        if test == bool(pairs & held):
            break
    things = r.things
    pick = [t for t in things if (t.kind, t.colour) in held] if test else things
    pick = pick or things

    def choose() -> world.Thing:
        return pick[int(rng.integers(len(pick)))]

    if form == "name":
        x = things[int(rng.integers(len(things)))]
        x.salient = True
        x.place = np.array([0.7 * math.cos(r.heading), 0.7 * math.sin(r.heading)])
        words, answer = _q(lex, "what", "is", "this"), lex.kind[x.kind]
        if mode == "scaffold":
            words = [answer]
    elif form == "colour_of":
        x = choose()
        words = _q(lex, "what", "colour", "is", "the") + [lex.kind[x.kind]]
        answer = lex.colour[x.colour]
    elif form == "by_colour":
        x = choose()
        words, answer = _q(lex, "what", "is") + [lex.colour[x.colour]], lex.kind[x.kind]
    elif form == "where":
        x = choose()
        words = _q(lex, "where", "is", "the") + [lex.kind[x.kind]]
        answer = lex.place[world.place_of(r.heading, x.place)]
    elif form == "by_place":
        quarters = [world.place_of(r.heading, t.place) for t in things]
        alone = [i for i, q in enumerate(quarters) if quarters.count(q) == 1]
        if not alone:
            return trial(form, mode, p, rng, held=held, test=test)
        x = things[int(rng.choice(alone))]
        words = _q(lex, "what", "is") + [lex.place[world.place_of(r.heading, x.place)]] + _q(
            lex, "you")
        answer = lex.kind[x.kind]
    else:
        raise ValueError(f"unknown form {form}")
    if mode == "scaffold" and form != "name":
        words = words + [answer]
    words = words + [SELF]  # the turn is the moment the last word is heard
    turn = len(words) - 2
    template = answer if mode == "scaffold" else None
    return Trial(r, words, turn, answer, template, form)


def block(form: str, mode: str, p: Parrot, rng: np.random.Generator, held: set[tuple[int, int]],
          test: bool = False) -> list[Trial]:
    out = [trial(form, mode, p, rng, held=held, test=test) for _ in range(p.streams)]
    width = max(len(tr.words) for tr in out)
    for tr in out:  # lockstep: silence first where a trial is shorter
        pad = width - len(tr.words)
        tr.words, tr.turn = [None] * pad + tr.words, tr.turn + pad
    return out


STAGES: list[tuple[str, list[tuple[str, str, float]]]] = [
    ("babble", [("babble", "turn", 1.0)]),
    ("imitate", [("imitate", "turn", 1.0)]),
    ("names", [("name", "scaffold", 0.5), ("name", "turn", 0.5)]),
    ("attributes", [("colour_of", "scaffold", 0.25), ("colour_of", "turn", 0.25),
                    ("by_colour", "scaffold", 0.25), ("by_colour", "turn", 0.25)]),
    ("places", [("where", "scaffold", 0.25), ("where", "turn", 0.25),
                ("by_place", "scaffold", 0.25), ("by_place", "turn", 0.25)]),
]
TESTS = {"imitate": ["imitate"], "names": ["name"], "attributes": ["colour_of", "by_colour"],
         "places": ["where", "by_place"]}


def life(seed: int, genome: dict[str, Any] | None = None, *, blocks: dict[str, int] | None = None,
         streams: int = 8, test_blocks: int = 4, backend: str = "cpu",
         log: Any = None) -> dict[str, Any]:
    """One founder's life through the stages; each stage's test after it. ``blocks`` gives the
    blocks of trials a stage lives (each block one trial a stream)."""
    budget = {"babble": 20, "imitate": 60, "names": 80, "attributes": 120, "places": 120,
              **(blocks or {})}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 334, 3]))
    p = Parrot(seed, genome, streams=streams, backend=backend)
    held = {(k, int(rng.integers(len(world.COLOURS)))) for k in range(world.TRAINED)}
    out: dict[str, Any] = {"seed": int(seed), "parameters": int(p.brain.parameters()),
                           "synapses": p.synapses, "inputs": p.inputs, "voice": len(p.voice)}
    curve: dict[str, list[float]] = {}
    began = time.time()
    for stage, mix in STAGES:
        forms = [(f, m) for f, m, _ in mix]
        weights = np.array([w for _, _, w in mix])
        for b in range(int(budget[stage])):
            f, m = forms[int(rng.choice(len(forms), p=weights / weights.sum()))]
            right = p.live(block(f, m, p, rng, held), rng)
            if m == "turn" and f != "babble":
                curve.setdefault(f"{stage}:{f}", []).append(float(np.mean(right)))
            if log is not None and b % 10 == 0:
                log(f"{stage} {b} {f} {m} right={np.mean(right):.2f} t={time.time() - began:.0f}s")
        for f in TESTS.get(stage, []):
            said, right = [], []
            for _ in range(test_blocks):
                trials = block(f, "turn", p, rng, held, test=f not in ("imitate", "name"))
                if f == "imitate":
                    for tr in trials:
                        tr.answer = tr.template
                words = p.test(trials, rng)
                said += words
                right += [w is not None and w == tr.answer for tr, w in zip(trials, words,
                                                                              strict=True)]
            out[f"test_{f}"] = float(np.mean(right))
            out[f"silent_{f}"] = float(np.mean([w is None for w in said]))
    out["curve"] = {k: [float(np.mean(v[i:i + 5])) for i in range(0, len(v), 5)]
                    for k, v in curve.items()}
    out["work"] = dict(p.work)
    out["macs"] = float((p.work["act_sweeps"] * streams + p.work["lesson_sweeps"]
                         + p.work["reward_row_sweeps"]) * p.synapses)
    out["seconds"] = time.time() - began
    return out


if __name__ == "__main__":
    import json

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    given = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    result = life(seed, given.get("genome"), blocks=given.get("blocks"),
                  streams=int(given.get("streams", 8)),
                  log=lambda s: print(s, flush=True))
    print(json.dumps(result, default=float))
