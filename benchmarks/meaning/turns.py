"""Turns before words (cadence#334, rungs T0 to T2): the parrot learns that a sound may call for a
sound of its own, and when.

The voice acts at every moment: a word the parrot has heard, or silence. Nothing in the brain
detects a question or times a reply. The ear hears the trainer and the parrot's own voice at once
(the codes of the sounds of one moment summed), with the pitch contour of the word heard (rising,
flat or falling: a born auditory dimension).

The teacher pays a pistachio for a reply in the gap after the trainer stops (moments 1 to
``GAP`` after the last word: 0.25 to 2 s at 250 ms a moment); speaking over the trainer, or later,
earns nothing, and one turn earns one pistachio at most. The pistachio is thrown and reaches the
beak ``CATCH`` moments after the act it pays: the reward enters then, and the eligibility trace
bridges the delay.

- **T0, contingency:** any reply in the gap pays. The **yoked parrot** lives the same trials and
  receives, at the same moments, the pistachios its partner earned, whatever it says itself
  (Goldstein, King and West 2003): it must not take turns.
- **T1, questions and statements:** only a reply after a question pays. A question ends rising
  (``RISING`` of them) and holds a question word; a statement ends falling. Probes: a statement's
  words with a rising end (prosody alone), a question's words with a falling end (words alone).
- **T2, what to say:** first an echo of the trainer's last word pays (``echo``), then only the
  answer to the question about the focused thing (``answer``); an echo of the question's words
  earns nothing then.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import meaning_world as world  # noqa: E402

from cadence.meaning import HAND_SET, Eyes, Forms, compose  # noqa: E402

GAP = 8      # moments after the trainer's last word in which a reply pays (0.25 to 2 s)
CATCH = 2    # moments from the paid act to the nut in the beak (0.5 s)
TAIL = 3     # silent moments after the gap before the trial ends
RISING = 0.8
CONTOURS = ("rise", "flat", "fall")


@dataclass
class Turn:
    """One trial: the trainer's words with their contours, the room, and what pays."""

    room: world.Room
    words: list[str]
    contours: list[str]
    question: bool
    pays: str            # "any": any reply; "question": a reply after a question; "echo"; "answer"
    answer: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Parrot:
    """The deep brain with its eyes, its ear (heard-form codes and the pitch contour) and its
    voice, for one founder; ``live`` runs a block of turns on every stream at once."""

    def __init__(self, seed: int, genome: dict[str, Any] | None = None, *, streams: int = 8,
                 slots: int = 5, backend: str = "cpu"):
        self.seed, self.streams = int(seed), int(streams)
        self.genome = {**HAND_SET, **(genome or {})}
        self.lex = world.lexicon(seed)
        self.statements = world.nonce(seed, 12, salt=world.SALT + 3, taken=self.lex.every())
        self.eyes = Eyes(slots=slots)
        self.forms = Forms(seed=seed)
        words = self.lex.every() + self.statements
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 334, 2])).permutation(
            len(words))
        self.voice = [words[int(i)] for i in order]
        self.silence = len(self.voice)
        for w in words:
            self.forms(w)
        self.ear = self.forms.width + len(CONTOURS)
        self.inputs = self.eyes.width + self.ear
        self.brain = compose(self.genome, self.inputs, len(self.voice), ear=self.ear,
                             seed=self.seed, backend=backend)
        self.synapses = int(self.brain.connectome.synapses)
        self.work = {"moments": 0, "act_sweeps": 0.0, "lesson_sweeps": 0.0,
                     "reward_row_sweeps": 0.0}

    def said(self, action: int) -> str | None:
        return None if int(action) >= self.silence else self.voice[int(action)]

    def hearing(self, trainer: str | None, contour: str | None, own: str | None) -> np.ndarray:
        """What the ear takes in at one moment: the trainer's word and the parrot's own voice
        summed, and the contour of the trainer's word."""
        code = self.forms(trainer) + self.forms(own)
        tune = np.zeros(len(CONTOURS))
        if trainer is not None and contour is not None:
            tune[CONTOURS.index(contour)] = 1.0
        elif own is not None:
            tune[CONTOURS.index("flat")] = 1.0
        return np.concatenate([code, tune])

    def observe(self, turns: list[Turn], t: int, own: list[str | None],
                rng: np.random.Generator) -> np.ndarray:
        rows = []
        for i, tr in enumerate(turns):
            trainer = tr.words[t] if t < len(tr.words) else None
            contour = tr.contours[t] if t < len(tr.contours) else None
            rows.append(np.concatenate([self.eyes.scene(world.seen(tr.room, rng)),
                                        self.hearing(trainer, contour, own[i])]))
        return np.asarray(rows)


def length(turns: list[Turn]) -> int:
    return max(len(tr.words) for tr in turns) + GAP + TAIL


def pay(tr: Turn, t: int, said: str | None, paid: bool) -> bool:
    """Whether the act ``said`` at moment ``t`` earns the turn's pistachio."""
    last = len(tr.words) - 1
    if paid or said is None or not last < t <= last + GAP:
        return False
    if tr.pays == "any":
        return True
    if tr.pays == "question":
        return tr.question
    if tr.pays == "echo":
        return tr.question and said == tr.words[-1]
    return tr.question and said == tr.answer


def live(p: Parrot, turns: list[Turn], rng: np.random.Generator, *,
         given: np.ndarray | None = None) -> dict[str, Any]:
    """One block of turns, learning. ``given`` (moments by streams) is a schedule of pistachios
    that replaces the parrot's own (the yoked parrot). Returns what was said at each moment,
    the pistachios earned (as a schedule) and the replies."""
    brain, g = p.brain, p.genome
    n, moments = len(turns), length(turns)
    own: list[str | None] = [None] * n
    voiced = np.full(n, p.silence, dtype=np.int64)
    earned = np.zeros((moments + CATCH + 1, n))
    paid = [False] * n
    spoken: list[list[str | None]] = []
    for t in range(moments):
        x = p.observe(turns, t, own, rng)
        reward = (earned[t] if given is None else given[t]).copy()
        if brain.pending_feedback:
            report = brain.learn(reward, np.full(n, t == moments - 1), x)
            p.work["reward_row_sweeps"] += n * (float(report.get("free_steps", 0.0)) + 24.0)
        if int(g["babble_lesson"]):
            taught = np.array([w is not None for w in own], float)
            if taught.any():
                report = brain.lesson(x, voiced, weights=taught)
                p.work["lesson_sweeps"] += float(report.get("total_row_sweeps", 0.0))
        action = np.asarray(brain.act(x)).reshape(n, -1)[:, 0].astype(np.int64)
        p.work["moments"] += 1
        p.work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
        voiced = action
        own = [p.said(a) for a in action]
        spoken.append(list(own))
        for i, tr in enumerate(turns):
            if pay(tr, t, own[i], paid[i]):
                paid[i] = True
                earned[t + CATCH, i] += 1.0
    return {"spoken": spoken, "earned": earned, "paid": paid}


def read(p: Parrot, turns: list[Turn], rng: np.random.Generator) -> dict[str, Any]:
    """The turns on a copy of the brain that learns nothing, greedily: when it speaks and
    what it says first in the gap."""
    frozen = copy.deepcopy(p.brain)
    frozen.reset()
    n = len(turns)
    own: list[str | None] = [None] * n
    out = {"in_gap": 0, "over": 0, "late": 0, "replied": [False] * n,
           "first": [None] * n}
    for t in range(length(turns)):
        action = np.asarray(frozen.act(p.observe(turns, t, own, rng), greedy=True))
        own = [p.said(a) for a in action.reshape(n, -1)[:, 0]]
        for i, tr in enumerate(turns):
            if own[i] is None:
                continue
            last = len(tr.words) - 1
            if t <= last:
                out["over"] += 1
            elif t <= last + GAP:
                out["in_gap"] += 1
                if not out["replied"][i]:
                    out["replied"][i], out["first"][i] = True, own[i]
            else:
                out["late"] += 1
    return out


# -- the teacher's turns


def turn(p: Parrot, rng: np.random.Generator, rung: str, *,
         probe: str | None = None) -> Turn:
    """A trial of rung ``T0``, ``T1``, ``T2echo`` or ``T2answer``; ``probe`` ("prosody" or
    "words") swaps the end contour of T1's sentences."""
    lex = p.lex
    r = world.room(rng, list(range(world.TRAINED)), int(rng.integers(1, 4)))
    question = bool(rng.random() < 0.5)
    if rung.startswith("T2"):
        question = True
        x = r.things[0]
        x.salient = True
        words = [lex.question[w] for w in ("what", "is", "this")]
        answer: str | None = lex.kind[x.kind]
    elif question:
        q = [lex.question[w] for w in ("what", "where", "is", "the", "you")]
        words = [q[int(rng.integers(2))]] + [str(rng.choice(q[2:] + lex.kind[:8]))
                                             for _ in range(int(rng.integers(1, 5)))]
        answer = None
    else:
        words = [str(rng.choice(p.statements + lex.kind[:8]))
                 for _ in range(int(rng.integers(1, 6)))]
        answer = None
    end = ("rise" if rng.random() < RISING else "fall") if question else (
        "fall" if rng.random() < 0.9 else "flat")
    if probe == "prosody":
        end = "fall" if question else "rise"
    contours = ["flat"] * (len(words) - 1) + [end]
    pays = {"T0": "any", "T1": "question", "T2echo": "echo", "T2answer": "answer"}[rung]
    return Turn(r, words, contours, question, pays, answer)


def block(p: Parrot, rng: np.random.Generator, rung: str, **kw: Any) -> list[Turn]:
    out = [turn(p, rng, rung, **kw) for _ in range(p.streams)]
    width = max(len(tr.words) for tr in out)
    for tr in out:  # lockstep: the trainer starts later where a turn is shorter
        pad = width - len(tr.words)
        tr.words = [None] * pad + tr.words  # type: ignore[list-item]
        tr.contours = [None] * pad + tr.contours  # type: ignore[list-item]
    return out
