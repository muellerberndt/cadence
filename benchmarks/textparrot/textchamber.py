"""The text parrot's exchanges, moment by moment, for any learner.

**An exchange.** The teacher types a question, one word a moment (the parrot's eyes rest). The
moment after the last word is the gap (an input channel of its own: the teacher has stopped), and
the eyes sweep over the scene (``textworld.SCAN``, born serial search), one thing a moment: the
search. At each thing the parrot's gaze either passes on or holds; a held thing stays in view in
a place of its own, the focus (born: the tectum keeps the selected thing's features; what to hold
is learned). Then the eyes sweep again, the answer: at each thing the parrot says words, one a
moment, and hears each word it said at the next moment; while it speaks the eyes stay on the
thing, and its silence moves them on. After the sweep the eyes rest and the parrot says the
closing words and ".", which ends the exchange.

**What a learner sees each moment** (``Layout``): the word heard (one channel a word form, each
form at a channel drawn for the founder, so the forms carry no spelling), the gap, the glimpse
(what the eyes report of the thing in view, ``textworld.look``, and a "seen" flag) and the focus
(the held thing's glimpse and its flag). **What it does**: one voice act (a word or silence) and
one gaze act (pass or hold). The voice is held while the teacher speaks and while the eyes
search (born: the songbird's song nuclei are inhibited while it listens); no lesson teaches the
question's moments, and what the voice does then is not heard.

**Teaching.** ``teacher``: each moment's label is the teacher's, and the next moment follows the
teacher's words (teacher forcing). ``dagger``: the parrot's own greedy words are heard and move
its eyes; each moment's label is what the teacher would say from there (the rest of the clause
where the parrot's words so far begin it, else silence). The focus follows the teacher's hold in
both. ``free``: nothing taught; the parrot's own acts drive the exchange, its focus included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import textworld as tw

PASS, HOLD = 0, 1
MAX_CLAUSE = 12  # most words the parrot may say at one thing
MAX_CLOSING = 3
STATS = {"dagger_moments": 0, "dagger_own_right": 0}


@dataclass
class Layout:
    """The input and action layout of one founder."""
    forms: np.ndarray  # forms[word index] = the ear channel of that word

    @classmethod
    def founder(cls, seed: int) -> Layout:
        rng = np.random.default_rng([seed, 7])
        return cls(rng.permutation(len(tw.VOCABULARY)))

    @property
    def words(self) -> int:
        return len(tw.VOCABULARY)

    @property
    def inputs(self) -> int:
        return self.words + 1 + 2 * (tw.GLIMPSE + 1)

    @property
    def gap(self) -> int:
        return self.words

    @property
    def glimpse(self) -> slice:
        return slice(self.words + 1, self.words + 2 + tw.GLIMPSE)

    @property
    def focus(self) -> slice:
        return slice(self.words + 2 + tw.GLIMPSE, self.inputs)

    @property
    def voice(self) -> int:  # the voice slot's acts: every word and silence (the last)
        return self.words + 1

    def say(self, act: int) -> str:
        return tw.SILENCE if act == self.words else tw.VOCABULARY[act]

    def act_of(self, word: str) -> int:
        return self.words if word == tw.SILENCE else tw.VOCABULARY.index(word)

    def ear(self, word: str | None) -> np.ndarray:
        x = np.zeros(self.words)
        if word is not None and word != tw.SILENCE:
            x[self.forms[tw.VOCABULARY.index(word)]] = 1.0
        return x


class Episode:
    """One exchange of one stream: the state machine of its moments."""

    def __init__(self, layout: Layout, scene: tw.Scene, q: tw.Question, mode: str):
        self.layout, self.scene, self.q, self.mode = layout, scene, q, mode
        self.order = scene.scanned()
        self.phase = "question"
        self.i = 0                    # position in the question, the search or the answer sweep
        self.heard: str | None = None  # the word heard this moment
        self.focus: tw.Thing | None = None
        self.held: list[tw.Thing] = []
        self.said: list[str] = []     # every word the parrot said, in order
        self.at: list[str] = []       # words said at the present thing (or in the closing)
        self.gap = False
        self.done = False
        self.right = True             # every act so far the teacher's
        self.moments = 0
        self._enter_question()

    # -- the moment the learner sees ---------------------------------------------------------
    def _enter_question(self) -> None:
        self.heard = self.q.words[0]

    def thing(self) -> tw.Thing | None:
        if self.phase in ("search", "answer") and self.i < len(self.order):
            return self.order[self.i]
        return None

    def observe(self) -> np.ndarray:
        L = self.layout
        x = np.zeros(L.inputs)
        x[:L.words] = L.ear(self.heard)
        x[L.gap] = 1.0 if self.gap else 0.0
        t = self.thing()
        if t is not None:
            x[L.glimpse] = np.concatenate([t.look, [1.0]])
        if self.focus is not None:
            x[L.focus] = np.concatenate([self.focus.look, [1.0]])
        return x

    # -- the teacher -------------------------------------------------------------------------
    def target_words(self) -> list[str]:
        if self.phase == "answer":
            t = self.thing()
            assert t is not None
            return self.q.clauses.get(self.scene.things.index(t), [])
        if self.phase == "closing":
            return self.q.closing + ["."]
        return []

    def expert(self) -> tuple[str, int]:
        """The teacher's word and gaze for this moment, from where the exchange stands."""
        if self.phase == "question":
            return tw.SILENCE, PASS
        if self.phase == "search":
            return tw.SILENCE, HOLD if self.thing() is self.q.referent else PASS
        target = self.target_words()
        n = len(self.at)
        if self.at == target[:n] and n < len(target):
            return target[n], PASS
        return tw.SILENCE, PASS

    # -- the step -------------------------------------------------------------------------------
    def advance(self, word: str, gaze: int) -> None:
        """Moves on by the acts taken this moment (``word``, ``gaze``)."""
        expert_word, expert_gaze = self.expert()
        if self.phase == "closing" or self.phase == "answer":
            if word != expert_word:
                self.right = False
        elif self.phase == "search":
            if gaze != expert_gaze:
                self.right = False
        self.moments += 1
        self.gap = False
        self.heard = None
        if self.phase == "question":
            self.i += 1
            if self.i < len(self.q.words):
                self.heard = self.q.words[self.i]
            else:
                self.phase, self.i, self.gap = "search", 0, True
                if not self.order:
                    self._enter_answer_end()
            return
        if self.phase == "search":
            t = self.order[self.i]
            if gaze == HOLD:
                self.held.append(t)
                self.focus = t
            self.i += 1
            if self.i >= len(self.order):
                self.phase, self.i, self.at = "answer", 0, []
            return
        if self.phase == "answer":
            if word != tw.SILENCE and len(self.at) < MAX_CLAUSE:
                self.at.append(word)
                self.said.append(word)
                self.heard = word
                return
            self.i += 1
            self.at = []
            if word != tw.SILENCE:  # a clause past its length: the eyes move on regardless
                self.said.append(word)
                self.heard = word
            if self.i >= len(self.order):
                self._enter_answer_end()
            return
        # closing
        if word != tw.SILENCE:
            self.said.append(word)
            self.heard = word
            self.at.append(word)
        if word == "." or word == tw.SILENCE or len(self.at) >= MAX_CLOSING + 1:
            self.done = True

    def _enter_answer_end(self) -> None:
        self.phase, self.at = "closing", []

    def reply(self) -> list[str]:
        return list(self.said)


class Policy(Protocol):
    """A learner the chamber drives: ``begin(n)`` starts n streams, ``moment(x, labels, mask,
    learn)`` takes one moment of every stream (labels (n, 2) of voice and gaze acts, mask the
    streams still in an exchange) and returns the greedy acts (n, 2)."""

    def begin(self, n: int) -> None: ...

    def moment(self, x: np.ndarray, labels: np.ndarray | None, mask: np.ndarray,
               learn: bool) -> np.ndarray: ...


def run(policy: Policy, layout: Layout, items: list[tuple[tw.Scene, tw.Question]], mode: str,
        *, learn: bool = True, cap: int = 120) -> list[Episode]:
    """One exchange a stream, all streams in lockstep. ``mode`` is ``teacher``, ``dagger`` or
    ``free`` (``free`` never teaches)."""
    eps = [Episode(layout, s, q, mode) for s, q in items]
    n = len(eps)
    policy.begin(n)
    for _ in range(cap):
        mask = np.array([not e.done for e in eps])
        if not mask.any():
            break
        x = np.stack([e.observe() for e in eps])
        labels = np.zeros((n, 2), dtype=np.int64)
        for k, e in enumerate(eps):
            w, g = e.expert() if not e.done else (tw.SILENCE, PASS)
            labels[k] = (layout.act_of(w), g)
        teach = learn and mode != "free"
        if hasattr(policy, "phases"):  # a learner that acts by the exchange's phase
            policy.phases = [e.phase for e in eps]
        if teach:
            # the voice is held while the teacher speaks (born suppression while hearing): the
            # question's moments teach nothing; every learner lives them
            mask = mask & np.array([e.phase != "question" for e in eps])
        if getattr(policy, "keyed", False):  # a learner that reads the situation's symbols
            acts = policy.moment_keys([policy.key(e) for e in eps], labels if teach else None,
                                      mask, teach)
        else:
            acts = policy.moment(x, labels if teach else None, mask, teach)
        for k, e in enumerate(eps):
            if e.done:
                continue
            if mode == "dagger" and e.phase in ("answer", "closing"):
                # how often the parrot's own word is the teacher's label (the label is the
                # teacher's word from where the parrot's own words left the exchange)
                STATS["dagger_moments"] += 1
                STATS["dagger_own_right"] += int(acts[k, 0] == labels[k, 0])
            if mode == "teacher":
                e.advance(layout.say(int(labels[k, 0])), int(labels[k, 1]))
            elif mode == "dagger":
                # the parrot's own words; the teacher's hold
                e.advance(layout.say(int(acts[k, 0])), int(labels[k, 1]))
            else:
                e.advance(layout.say(int(acts[k, 0])), int(acts[k, 1]))
    return eps


def score(e: Episode) -> dict[str, Any]:
    """Whether the reply is the teacher's reply, word for word; for the sentence forms, whether
    the parser reads it back to the scene."""
    want = e.q.reply(e.scene)
    got = e.reply()
    out: dict[str, Any] = {"form": e.q.form, "right": float(got == want), "said": got,
                           "want": want}
    if e.q.form == "describe":
        p = tw.parse(got)
        out["parsed"] = float(p["bad"] == 0 and sorted(p["things"]) == tw.described(e.scene))
    elif e.q.form == "describe_rel":
        p = tw.parse(got, relational=True)
        ref = e.q.referent
        assert ref is not None
        rel = e.q.meta["relation"]
        truth = sorted((t.colour, t.kind, rel, ref.colour, ref.kind)
                       for t in e.scene.things if tw.related(t, ref, rel))
        ok = p["bad"] == 0 and sorted(p["relations"]) == truth
        out["parsed"] = float(ok)
    return out
