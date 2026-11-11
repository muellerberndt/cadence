"""The encyclopedia's instrument (cadence#311): facts told as sentences and questions about them,
heard one word a moment, each answer said whole as one word of the closed answer vocabulary.

**Words.** Every word is heard as a fixed form code: ``ON`` of ``WIDTH`` input channels, drawn
from a hash of the word (an instrument's stand-in for a word's heard form, hand-engineered). The
channel ``WIDTH`` marks a turn (``TURN``). A told fact is "the {relation} of {subject} is
{object} .", the turn marked before the object; a question is its words, the turn, then a silent
moment at which the answer is said. The answers are the corpus's closed answer vocabulary
(``corpus.py``), one action each, an instrument's choice.

**Two sets of facts.** A subject belongs to the lessons where a hash of its item number falls
under ``LESSON``; every other subject's facts are the reading corpus, in the corpus's order, so
the corpus of ``size`` facts is the first ``size`` of them. No lesson tells or asks a fact of
the reading corpus.

**Wordings,** frozen here before any life:

- ``trained``: the wordings lessons ask in (``ASK``);
- ``held``: wordings no lesson asks in, made of the relation's and the subject's words with
  question words of their own (``ASK_HELD``);
- ``synonym``: the relation said by another word (``SYNONYM``), read without a bar;
- ``two``: a question of two facts, "what is the {second} of the {first} of {subject} ?", where
  the first fact's object is the subject of the second (``CHAINS``).

**Revision.** A fact told again with another object of its relation: the later telling is the
current one.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TURN = "<turn>"
WIDTH, ON = 256, 5
LESSON = 0.2  # the share of subjects whose facts the lessons use
TELL = "the {r} of {s} is {o} ."
ASK = ("what is the {r} of {s} ?", "the {r} of {s} is what ?")
ASK_HELD = ("which {r} does {s} have ?", "{s} 's {r} is which ?", "name the {r} of {s} ?")
ASK_TWO = "what is the {r2} of the {r1} of {s} ?"
SYNONYM = {"capital": "seat", "continent": "landmass", "currency": "money",
           "language": "speech", "citizenship": "nationality", "occupation": "job",
           "birthplace": "hometown", "mother tongue": "first speech", "sport": "game",
           "instrument": "axe", "country": "nation"}
# a question of two facts: the first relation's object is the second's subject
CHAINS = (("citizenship", "capital"), ("citizenship", "continent"), ("citizenship", "currency"),
          ("citizenship", "language"), ("birthplace", "country"))


@dataclass(frozen=True)
class Fact:
    subject: str  # the item, "Q..."
    name: tuple[str, ...]  # the subject's words
    relation: str
    answer: str


@dataclass
class Turn:
    """One turn: ``said`` the words heard; ``answer`` the word said at the silent moment (None
    for a told turn); ``turn_at`` the word the turn is marked before (a told turn)."""

    said: list[str]
    answer: str | None = None
    turn_at: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def asked(self) -> bool:
        return self.answer is not None


def form(word: str) -> np.ndarray:
    """A word's form code: the ``ON`` channels of ``WIDTH`` a hash of the word draws."""
    seed = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
    return np.sort(np.random.default_rng(seed).choice(WIDTH, ON, replace=False))


class Lexicon:
    """Words to moments (form codes and the turn) and answers to actions."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = sorted(set(answers))
        self.action = {a: i for i, a in enumerate(self.answers)}
        self._moments: dict[str, np.ndarray] = {}

    @property
    def inputs(self) -> int:
        return WIDTH + 1

    @property
    def actions(self) -> int:
        return len(self.answers)

    def moment(self, word: str | None) -> np.ndarray:
        if word is None:
            return np.zeros(self.inputs)
        x = self._moments.get(word)
        if x is None:
            x = np.zeros(self.inputs)
            if word == TURN:
                x[WIDTH] = 1.0
            else:
                x[form(word)] = 1.0
            self._moments[word] = x
        return x

    def moments(self, turn: Turn) -> list[tuple[np.ndarray, int | None]]:
        """The moments of a turn and the action taught or scored at each (None where none)."""
        out: list[tuple[np.ndarray, int | None]] = []
        for i, w in enumerate(turn.said):
            if turn.turn_at == i and not turn.asked:
                out.append((self.moment(TURN), None))
            out.append((self.moment(w), None))
        if turn.asked:
            out.append((self.moment(TURN), None))
            out.append((self.moment(None), self.action[str(turn.answer)]))
        return out


def lesson_subject(subject: str) -> bool:
    digest = hashlib.blake2b(subject.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2.0**64 < LESSON


class Corpus:
    """The facts of ``path`` (``corpus.py``'s table): the lesson facts and the reading corpus.
    ``keep`` keeps the first ``keep`` facts of the reading corpus (every fact where None); the
    answers and each relation's answers are read from every fact either way."""

    def __init__(self, path: Path, keep: int | None = None) -> None:
        self.lessons: list[Fact] = []
        self.reading: list[Fact] = []
        answers: dict[str, set[str]] = defaultdict(set)
        with open(path, newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                answers[row["relation"]].add(row["answer"])
                lesson = lesson_subject(row["subject"])
                if not lesson and keep is not None and len(self.reading) >= keep:
                    continue
                fact = Fact(row["subject"], tuple(row["name"].split()), row["relation"],
                            row["answer"])
                (self.lessons if lesson else self.reading).append(fact)
        self.lexicon = Lexicon([a for held in answers.values() for a in held])
        self.relation_answers = {r: sorted(a) for r, a in answers.items()}

    def corpus(self, size: int) -> list[Fact]:
        return self.reading[:size]


# --------------------------------------------------------------------------------- the turns


def words(text: str) -> list[str]:
    return text.split()


def tell(fact: Fact, answer: str | None = None) -> Turn:
    said = words(TELL.format(r=fact.relation, s=" ".join(fact.name), o="\0"))
    at = said.index("\0")
    said[at] = answer or fact.answer
    return Turn(said, None, at, {"relation": fact.relation})


def ask(fact: Fact, wording: str, template: int, answer: str | None = None) -> Turn:
    s, r = " ".join(fact.name), fact.relation
    if wording == "trained":
        text = ASK[template % len(ASK)].format(r=r, s=s)
    elif wording == "held":
        text = ASK_HELD[template % len(ASK_HELD)].format(r=r, s=s)
    elif wording == "synonym":
        text = ASK[template % len(ASK)].format(r=SYNONYM[r], s=s)
    else:
        raise ValueError(wording)
    return Turn(words(text), answer or fact.answer, None,
                {"wording": wording, "relation": r})


def ask_two(first: Fact, second: Fact) -> Turn:
    text = ASK_TWO.format(r2=second.relation, r1=first.relation, s=" ".join(first.name))
    return Turn(words(text), second.answer, None,
                {"wording": "two", "relation": f"{first.relation}>{second.relation}"})


def chains(facts: list[Fact]) -> list[tuple[Fact, Fact]]:
    """The questions of two facts among ``facts``: a fact whose object is the subject of a fact
    of the chained relation."""
    by_subject: dict[tuple[str, str], Fact] = {}
    for f in facts:
        if len(f.name) == 1:
            by_subject[(f.name[0], f.relation)] = f
    out = []
    for f in facts:
        for first, second in CHAINS:
            if f.relation == first and (f.answer, second) in by_subject:
                out.append((f, by_subject[(f.answer, second)]))
    return out


def revised(fact: Fact, corpus: Corpus, rng: np.random.Generator) -> str:
    """Another answer of the fact's relation, for its revision."""
    others = [a for a in corpus.relation_answers[fact.relation] if a != fact.answer]
    return str(others[int(rng.integers(len(others)))])


def balanced(facts: list[Fact], rng: np.random.Generator, n: int) -> list[Fact]:
    """``n`` facts drawn answer first: an answer uniformly, then one of its facts (so every
    answer is taught about as often)."""
    by_answer: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        by_answer[f.answer].append(f)
    answers = sorted(by_answer)
    out = []
    for i in rng.integers(len(answers), size=n):
        pool = by_answer[answers[int(i)]]
        out.append(pool[int(rng.integers(len(pool)))])
    return out


def lesson_session(corpus: Corpus, rng: np.random.Generator, facts: int = 6,
                   two: float = 0.2, revise: float = 0.1) -> list[Turn]:
    """One lesson session: ``facts`` lesson facts told, a share of them told again with another
    object, then a question about each in a trained wording; a share ``two`` of the questions
    is a question of two facts, its two facts told in the session."""
    told = balanced(corpus.lessons, rng, facts)
    pool = getattr(corpus, "_lesson_chains", None)
    if pool is None:
        pool = chains(corpus.lessons)
        corpus._lesson_chains = pool  # type: ignore[attr-defined]
    turns: list[Turn] = []
    asked: list[Turn] = []
    for fact in told:
        if pool and rng.random() < two:
            first, second = pool[int(rng.integers(len(pool)))]
            turns += [tell(first), tell(second)]
            asked.append(ask_two(first, second))
            continue
        turns.append(tell(fact))
        answer = fact.answer
        if rng.random() < revise:
            answer = revised(fact, corpus, rng)
            turns.append(tell(fact, answer))
        asked.append(ask(fact, "trained", int(rng.integers(len(ASK))), answer))
    order = rng.permutation(len(asked))
    return turns + [asked[int(i)] for i in order]
