"""Several things of one kind, answers that name several things, and how many (cadence#328).

``Several`` is a part of ``cadence.words.Words`` (``Words(map, parts=[Several()])``):

- **What a question names.** Each of the question's words is heard alone, and the lexicon
  settles on it: the word names its most active concept of the senses (a kind, a value, a place
  around the body) where that concept is above the brain's ``floor``. "The red ball" names the
  kind ball and the value red; "the green" names the value green alone, though the word holds a
  little of the kind and the place the green things of its namings had.
- **The things it names.** ``Words`` finds the things of the concept the question's words name
  most (the map settles on a kind's places, a value's places, or what is at a place around the
  body); of those, ``Several`` keeps the things the senses give every other named concept of:
  of two balls of two colours, "the red ball" is the one whose value the map settles on as red.
- **Two reads of the things found together**: ``number``, how many there are, and ``truth``,
  whether there is one. The number sense is born: one neuron a count of the things found, from
  none to ``numbers - 1`` (the last counts that many or more), as an animal is born with a sense
  of small numbers. The words for the counts are learned. Where the question's most active read
  is one of these, the things found are one: their count and their truth, said in one word.
- **Practice.** The part takes the lessons of a question whose answer is several words (a thing
  each) or one word no naming has shown (a number word, yes, no, "I do not know"). The
  question's words of asking are taught each read whose concepts the lexicon's motor would say
  as the teacher's words (a read that finds nothing teaches nothing): the places of two balls,
  heard together, hold the ball twice from their namings, and the read the motor says them by
  is the place. An answer of one word is named with what the brain found, at the question's
  reads as gains: the count found, whether there is one, and the read that finds nothing for a
  read of a thing that found none. A read whose concepts hold another word than
  the teacher's takes no part (a concept holds a word where the word the motor says for it,
  heard, names it above ``floor``); of the reads that take part, the most active one the
  question's words drive takes the whole gain, and where none is heard above ``floor`` they
  share it alike. A word answered to "how many" questions so keeps the count its answers
  share, and a word answered to "is there" questions keeps whether there was one. Where the
  teacher asks "how many" before "is there", "no" finds the truth free of a word; asked together
  from the first question, "zero" and "no" may each take the other's concept
  (``benchmarks/words/several.py`` reads both orders).
"""

from __future__ import annotations

from typing import Any

import numpy as np

NOTHING = ("none", 0)


class Several:
    """A part of ``cadence.words.Words`` for several things: what a question names, word by
    word; the things found that are all of it; their count (``numbers`` counts, from none) and
    whether there is one, as two more reads."""

    answers = ("number", "truth")

    def __init__(self, numbers: int = 6):
        if int(numbers) < 2:
            raise ValueError("a part of several things counts 2 or more")
        self.numbers = int(numbers)
        self.groups = {"number": self.numbers, "truth": 2}
        self.named: dict[str, int] = {}  # what the latest question named, word by word
        self.kept: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []  # the things it named

    # -- a question

    def name_words(self, brain: Any, words: list[str]) -> dict[str, int]:
        """What a question names, ``{group: concept}``: each word heard alone names its most
        active concept of the senses, above ``floor``; of two words that name one group, the
        more active."""
        groups = brain.lexicon.groups
        out: dict[str, int] = {}
        strength: dict[str, float] = {}
        for word in words:
            heard = brain.lexicon.hear(word)
            group = max(brain.senses.groups, key=lambda g: float(heard[groups[g]].max()))
            part = heard[groups[group]]
            if part.max() > brain.floor and part.max() > strength.get(group, 0.0):
                out[group], strength[group] = int(np.argmax(part)), float(part.max())
        return out

    def together(self, count: int) -> dict[str, np.ndarray]:
        """The things found together as concepts: their count and whether there is one."""
        return {"number": np.eye(self.numbers)[min(int(count), self.numbers - 1)],
                "truth": np.eye(2)[int(count > 0)]}

    def find(self, brain: Any, heard: Any, things: list[Any]) -> list[Any]:
        """Of the things found, those the senses give every concept the question names; where
        the question's most active read is a count or a truth, the things found together, as
        one."""
        self.named = self.name_words(brain, heard.words)
        if heard.about is None and not self.named:
            self.kept = []
            return things
        self.kept = [(place, described) for place, described in things
                     if all(g not in described or (described[g].max() > 0 and int(
                         np.argmax(described[g])) == v) for g, v in self.named.items())]
        live = {g: r for g, r in heard.reads.items() if r > brain.floor}
        if live and max(live, key=lambda g: live[g]) in self.answers:
            return [(np.full(2, np.nan), self.together(len(self.kept)))]
        return self.kept

    # -- practice

    def takes(self, brain: Any, words: list[str], answer: list[str], heard: Any) -> bool:
        """Whether the part takes a question's lessons: an answer of several words, or of one
        word that no naming has shown."""
        return len(answer) > 1 or not self._shown(brain, answer[0])

    def practice(self, brain: Any, words: list[str], answer: list[str], heard: Any) -> None:
        """The lessons of a question the part takes: the reads of its words of asking, and an
        answer of one word named with what was found."""
        if not self.takes(brain, words, answer, heard):
            return
        wanted = sorted(" ".join(str(w).lower().split()) for w in answer)
        right, open_ = [], []
        for read in brain.answers:
            found = [self.together(len(self.kept))] if read in self.answers else [
                described for _, described in self.kept]
            vectors = [v for v in (brain.concepts(d, {read: 1.0}) for d in found) if v.max() > 0]
            if not vectors:  # a read of a thing that finds none: the read that finds nothing
                if self._word_for(brain, NOTHING) in (None, *wanted):
                    open_.append(read)
                continue
            if sorted(brain.lexicon.say(v) or "" for v in vectors) == wanted:
                right.append(read)
                open_.append(read)
            elif not any(self._word_for(brain, v) for v in vectors):
                open_.append(read)  # its concepts hold no word
        asking = brain.asking(words)
        if right and asking:
            brain.lexicon.name(asking, {("read", brain.answers.index(r)): 1.0 / len(right)
                                        for r in right}, groups=("read",))
        target = self._found(brain, heard, open_) if len(answer) == 1 else {}
        if target:
            brain.lexicon.name(answer[0], target, groups=("number", "truth", "none"))

    def _found(self, brain: Any, heard: Any, open_: list[str]) -> dict[tuple[str, int], float]:
        """What was found, as concepts at the question's reads as gains over the reads that
        take part: the count and whether there is one, and the read that finds nothing for a
        read of a thing that found none (a thing's own concepts are named by showing it). A
        question that names nothing the brain knows gives the read that finds nothing alone."""
        if heard.about is None and not self.named:
            return {NOTHING: 1.0}
        if not open_:
            return {}
        reads = {g: heard.reads[g] for g in open_}
        if max(reads.values()) > brain.floor:  # the read the question asks, of those taking part
            top = max(reads, key=lambda g: reads[g])
            gains = {g: float(g == top) for g in open_}
        else:
            gains = dict.fromkeys(open_, 1.0 / len(open_))
        count = len(self.kept)
        out: dict[tuple[str, int], float] = {}
        for read, gain in gains.items():
            if not gain > 0.0:
                continue
            if read == "number":
                concept = ("number", min(count, self.numbers - 1))
            elif read == "truth":
                concept = ("truth", int(count > 0))
            elif not count:
                concept = NOTHING
            else:
                continue
            out[concept] = out.get(concept, 0.0) + gain
        return out

    @staticmethod
    def _word_for(brain: Any, concept: Any) -> str | None:
        """The word a concept holds: the word the motor says for it, where that word heard
        names the concept above ``floor``; None for a concept no word holds so."""
        word = brain.lexicon.say(concept)
        if word is None:
            return None
        heard = brain.lexicon.hear(word)
        meant = int(np.argmax(brain.lexicon._vector(concept)))
        return str(word) if float(heard[meant]) > brain.floor else None

    @staticmethod
    def _shown(brain: Any, word: str) -> bool:
        """Whether a naming has shown a thing with the word."""
        unit = brain.lexicon.unit(word)
        counts = brain.lexicon.counts.get(tuple(brain.shares))
        return unit is not None and counts is not None and bool(counts[unit] > 0)


__all__ = ["NOTHING", "Several"]
