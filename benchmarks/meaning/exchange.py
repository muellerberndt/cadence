"""Step S2 of the meaning plan (cadence#334): sounds as symbols, in exchanges with a trainer.

Ten moments a second, as on the page. A word lasts ``WORD`` moments: the ear hears its code
throughout, with the pitch contour of the utterance's end on its last word. The voice acts at
every moment: a word the parrot has heard, or silence. Nothing in the brain detects a question,
times a reply or names a meaning.

An **exchange** is the trainer's utterance, the gap after it (``GAP`` moments: 0.3 to 2 s pay a
reply), and a tail. The first word the parrot says in the gap is its reply. Speaking before the
gap opens (over the trainer, or within 0.3 s of its end) forfeits the turn: the reply that
follows earns nothing, as an interruption goes unanswered. Speaking later earns nothing. A right
reply earns the trainer's **praise word** (a nonce word, heard at once) and a **pistachio**,
thrown and in the beak ``CATCH`` moments later; the exchange ends with ``done`` when the
pistachio is paid (else at its end). The eligibility trace bridges the catch (``actor_hold``).

The trainer's exchanges (only the world and the teacher are designed):

- ``babble``: silence; the parrot's sounds are heard and pay nothing.
- ``T0``: 1 to 6 words; any reply in the gap pays.
- ``T1``: a question (a question word, a rising end) or a statement (a falling end); a reply
  pays after a question alone.
- ``imitate``: one word; the reply that is that word pays (the trainer's echo game). The parrot's
  own comparison of its voice with the word (``self_reward``) adds its born reward.
- ``name_model``: the trainer holds a thing up (salient, in front, near) and a second trainer
  says its name; the parrot's reply that is the name pays (the model/rival game).
- ``name``: the thing held up, "what is this?"; the name pays.
- ``look``: several things; a name; the head's turn (the gaze slot) to that thing pays.

Each moment the parrot also hears itself: the codes of the trainer's word and its own voice
summed. Hearing itself say a word, it takes the self-hearing lesson (``babble_lesson``).
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import meaning_world as world  # noqa: E402

from cadence.meaning import HAND_SET, Eyes, Forms, compose  # noqa: E402

WORD = 3        # moments a word lasts (300 ms)
GAP = (3, 20)   # a reply pays from 0.3 s to 2 s after the trainer's last word
CATCH = 5       # moments from a paid reply to the pistachio in the beak (0.5 s)
TAIL = 6        # moments after the gap before the exchange ends
GAZE = 8        # directions the head turns to
CONTOURS = ("rise", "flat", "fall")
S2_HAND_SET = {"actor_hold": 0.93, "actor_eta": 0.1, "history_decays": (0.0, 0.5, 0.8),
               "ear_entry": "module_0", "self_reward": 0.1}


@dataclass
class Exchange:
    room: world.Room
    words: list[str]          # the trainer's words
    end: str                  # the contour of the last word
    kind: str
    question: bool = False
    answer: str | None = None  # the word that pays (None: any reply in T0 and T1)
    look: int | None = None    # the direction (gaze bin) that pays in ``look``

    @property
    def spoken(self) -> int:
        return WORD * len(self.words)


class Parrot:
    """The deep brain, its eyes, its ear and its voice and gaze, for one founder."""

    def __init__(self, seed: int, genome: dict[str, Any] | None = None, *, streams: int = 8,
                 slots: int = 5, backend: str = "cpu", gaze: bool = True):
        self.seed, self.streams = int(seed), int(streams)
        self.genome = {**HAND_SET, **S2_HAND_SET, **(genome or {})}
        self.lex = world.lexicon(seed)
        extra = world.nonce(seed, 13, salt=world.SALT + 3, taken=self.lex.every())
        self.praise, self.statements = extra[0], extra[1:]
        self.eyes = Eyes(slots=slots)
        self.forms = Forms(seed=seed)
        words = self.lex.every() + self.statements + [self.praise]
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 334, 2])).permutation(
            len(words))
        self.voice = [words[int(i)] for i in order]
        self.silence = len(self.voice)
        for w in words:
            self.forms(w)
        self.ear = self.forms.width + len(CONTOURS)
        self.inputs = self.eyes.width + self.ear
        self.gaze = GAZE if gaze else 0
        self.brain = compose(self.genome, self.inputs, len(self.voice), ear=self.ear,
                             gaze=self.gaze, seed=self.seed, backend=backend)
        self.synapses = int(self.brain.connectome.synapses)
        self.work = {"moments": 0, "stream_moments": 0, "act_sweeps": 0.0,
                     "lesson_sweeps": 0.0, "reward_row_sweeps": 0.0}

    def said(self, action: int) -> str | None:
        return None if int(action) >= self.silence else self.voice[int(action)]

    def heard(self, ex: Exchange, t: int) -> tuple[str | None, str | None]:
        """The trainer's word at moment ``t`` and its contour."""
        if t >= ex.spoken:
            return None, None
        i = t // WORD
        return ex.words[i], (ex.end if i == len(ex.words) - 1 else "flat")

    def observe(self, exs: list[Exchange], t: int, own: list[str | None],
                praise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rows = []
        for i, ex in enumerate(exs):
            word, contour = self.heard(ex, t)
            code = self.forms(word) + self.forms(own[i])
            if praise[i]:
                code = code + self.forms(self.praise)
            tune = np.zeros(len(CONTOURS))
            if contour is not None:
                tune[CONTOURS.index(contour)] = 1.0
            rows.append(np.concatenate([self.eyes.scene(world.seen(ex.room, rng)), code, tune]))
        return np.asarray(rows)


def length(exs: list[Exchange]) -> int:
    return max(ex.spoken for ex in exs) + GAP[1] + CATCH + TAIL


def in_gap(ex: Exchange, t: int) -> bool:
    return ex.spoken - 1 + GAP[0] <= t <= ex.spoken - 1 + GAP[1]


def pays(ex: Exchange, said: str | None) -> bool:
    if said is None or ex.kind == "babble":
        return False
    if ex.kind == "T0":
        return True
    if ex.kind == "T1":
        return ex.question
    return ex.answer is not None and said == ex.answer


def live(p: Parrot, exs: list[Exchange], rng: np.random.Generator, *,
         learn: bool = True, given: np.ndarray | None = None) -> dict[str, Any]:
    """One block of exchanges, one a stream. With ``learn`` the brain learns from its
    outcomes; without it a frozen copy acts greedily (a reading). ``given`` (moments by streams)
    replaces the parrot's own pistachios (the yoked parrot)."""
    brain = p.brain if learn else copy.deepcopy(p.brain)
    if not learn:
        brain.reset()
    g = p.genome
    n, moments = len(exs), length(exs)
    own: list[str | None] = [None] * n
    voiced = np.full(n, p.silence, dtype=np.int64)
    pistachio = np.zeros((moments + CATCH + 2, n))
    praise_at = np.zeros((moments + 2, n), dtype=bool)
    replied: list[str | None] = [None] * n
    reply_at = [-1] * n
    looked: list[int | None] = [None] * n
    ended = np.zeros(n, dtype=bool)
    interrupted = np.zeros(n, dtype=bool)  # spoke before the gap opened: the turn earns nothing
    over = late = 0
    for t in range(moments):
        x = p.observe(exs, t, own, praise_at[t], rng)
        if learn:
            reward = (pistachio[t] if given is None else given[t]).copy()
            for i, ex in enumerate(exs):  # the born comparison of its voice with the word heard
                if ex.kind == "imitate" and own[i] is not None and own[i] == ex.words[-1] \
                        and in_gap(ex, t - 1):
                    reward[i] += float(g["self_reward"])
            done = (pistachio[t] > 0) | np.full(n, t == moments - 1)
            if brain.pending_feedback:
                report = brain.learn(reward, done & ~ended, x)
                p.work["reward_row_sweeps"] += n * (float(report.get("free_steps", 0.0)) + 24)
            ended |= done
            if int(g["babble_lesson"]):
                taught = np.array([w is not None for w in own], float)
                if taught.any():
                    labels = voiced if not p.gaze else np.stack(
                        [voiced, np.zeros(n, dtype=np.int64)], axis=1)
                    report = brain.lesson(x, labels, weights=taught)
                    p.work["lesson_sweeps"] += float(report.get("total_row_sweeps", 0.0))
            action = np.asarray(brain.act(x)).reshape(n, -1)
        else:
            action = np.asarray(brain.act(x, greedy=True)).reshape(n, -1)
        p.work["moments"] += 1
        p.work["stream_moments"] += n
        p.work["act_sweeps"] += n * float((brain.last_settlement or {}).get("steps", 0))
        voiced = action[:, 0].astype(np.int64)
        own = [p.said(a) for a in voiced]
        for i, ex in enumerate(exs):
            if p.gaze and ex.kind == "look" and looked[i] is None and in_gap(ex, t):
                if int(action[i, 1]) != 0:  # bin 0 keeps the head where it is
                    looked[i] = int(action[i, 1])
                    if learn and looked[i] == ex.look and not interrupted[i]:
                        pistachio[t + CATCH, i] += 1.0
                        praise_at[t + 1, i] = True
            if own[i] is None:
                continue
            if t < ex.spoken - 1 + GAP[0]:
                over += 1
                interrupted[i] = interrupted[i] or ex.kind != "babble"
            elif in_gap(ex, t):
                if replied[i] is None:
                    replied[i], reply_at[i] = own[i], t
                    if learn and ex.kind != "look" and not interrupted[i] and pays(ex, own[i]):
                        pistachio[t + CATCH, i] += 1.0
                        praise_at[t + 1, i] = True
            else:
                late += 1
    return {"replied": replied, "reply_at": reply_at, "looked": looked, "over": over,
            "late": late, "interrupted": interrupted.tolist(),
            "pistachios": pistachio[:moments + CATCH]}


# -- the trainer's exchanges


def exchange(p: Parrot, rng: np.random.Generator, kind: str, *,
             probe: str | None = None) -> Exchange:
    lex = p.lex
    r = world.room(rng, list(range(world.TRAINED)), int(rng.integers(2, 5)))
    if kind == "babble":
        return Exchange(r, [], "flat", kind)
    if kind in ("T0", "T1"):
        question = bool(rng.random() < 0.5)
        q = [lex.question[w] for w in ("what", "where", "is", "the", "you")]
        if question:
            words = [q[int(rng.integers(2))]] + [str(rng.choice(q[2:] + lex.kind[:8]))
                                                 for _ in range(int(rng.integers(1, 4)))]
        else:
            words = [str(rng.choice(p.statements + lex.kind[:8]))
                     for _ in range(int(rng.integers(1, 5)))]
        end = ("rise" if rng.random() < 0.8 else "fall") if question else "fall"
        if probe == "prosody":
            end = "fall" if question else "rise"
        return Exchange(r, words, end, kind, question=question)
    if kind == "imitate":
        w = lex.every()[int(rng.integers(len(lex.every())))]
        return Exchange(r, [w], "flat", kind, answer=w)
    x = r.things[int(rng.integers(len(r.things)))]
    name = lex.kind[x.kind]
    if kind in ("name", "name_model"):
        x.salient = True
        x.place = np.array([0.7 * math.cos(r.heading), 0.7 * math.sin(r.heading)])
        if kind == "name_model":
            return Exchange(r, [name], "fall", kind, answer=name)
        return Exchange(r, [lex.question[w] for w in ("what", "is", "this")], "rise", kind,
                        question=True, answer=name)
    if kind == "look":
        a = world.direction(r.heading, x.place)
        bin_ = 1 + int(((a + math.pi) / (2 * math.pi) * (GAZE - 1)) % (GAZE - 1))
        return Exchange(r, [name], "flat", kind, look=bin_)
    raise ValueError(kind)


def block(p: Parrot, rng: np.random.Generator, kind: str, **kw: Any) -> list[Exchange]:
    return [exchange(p, rng, kind, **kw) for _ in range(p.streams)]
