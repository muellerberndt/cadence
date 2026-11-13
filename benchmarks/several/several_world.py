"""The world of the several-records chamber (cadence#325): facts told as words, and questions
whose one answer needs several of them.

**Words.** Every word is heard as a fixed form code, ``ON`` of ``WIDTH`` input channels drawn
from a hash of the word (the encyclopedia chamber's stand-in for a word's heard form). The
channel ``WIDTH`` marks a turn.

**Facts.** A population of subjects, each a word of its own. A subject has attributes
(``ATTRIBUTES``: a subject to one of ``PLACES`` places) and links (``LINKS``: a subject to
another subject), each present for half of the subjects. A fact is told as "the {relation} of
{subject} is {object} .", the turn marked before the object.

**Questions.** The answer is said whole at the silent moment after the question's turn, by
``SLOTS`` motor slots that settle together, each saying a place or ``NOTHING``:

- ``parts`` of k subjects, "what is the {r} of {s1} and {s2} ... ?": slot j says the place of
  the j-th named subject, and ``NOTHING`` where no fact tells it or j is past the parts;
- ``chain`` of h hops, "what is the {r} of the {l2} of the {l1} of {s} ?": slot 0 says the
  place at the chain's end, ``NOTHING`` where the chain breaks, and every other slot
  ``NOTHING``.

A part is absent where the named subject holds no fact of the relation (``lacks``) or is a word
no fact holds (``unknown``). Each question carries the records its answer needs
(``Question.needs``, a fact's index per part or hop) so a reading can score what each head
delivered.

**Two populations.** The lessons use one population and the readings another: no word of a
reading subject is heard in a lesson.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TURN = "<turn>"
NOTHING = "<nothing>"
WIDTH, ON = 256, 5
SLOTS = 8
ATTRIBUTES = ("home", "work", "school", "market")
LINKS = ("boss", "friend", "mother")
PLACES = tuple(f"place{i:02d}" for i in range(32))
PRESENT = 0.5  # the share of the subjects that hold each relation


def form(word: str) -> np.ndarray:
    """A word's form code: the ``ON`` channels of ``WIDTH`` a hash of the word draws."""
    seed = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
    return np.sort(np.random.default_rng(seed).choice(WIDTH, ON, replace=False))


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    answer: str  # a place, or with a link another subject


@dataclass
class Question:
    """One question: ``said`` its words, ``answer`` what each slot says, ``needs`` the fact each
    part or hop is answered by (an index into the population's facts, -1 where none answers
    it), ``kind`` and ``k`` its family and size, ``tags`` what the reading groups it by."""

    said: list[str]
    answer: list[str]
    needs: list[int]
    kind: str
    k: int
    tags: dict[str, Any] = field(default_factory=dict)


class Lexicon:
    """Words to moments (form codes and the turn) and each slot's answers to actions."""

    def __init__(self) -> None:
        self.answers = [*PLACES, NOTHING]
        self.action = {a: i for i, a in enumerate(self.answers)}
        self._moments: dict[str, np.ndarray] = {}

    @property
    def inputs(self) -> int:
        return WIDTH + 1

    @property
    def slot_actions(self) -> int:
        return len(self.answers)

    @property
    def actions(self) -> int:
        return SLOTS * len(self.answers)

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

    def told(self, fact: Fact) -> list[np.ndarray]:
        """The moments of a told fact, the turn before its object."""
        said = ["the", fact.relation, "of", fact.subject, "is", TURN, fact.answer, "."]
        return [self.moment(w) for w in said]

    def asked(self, question: Question) -> tuple[list[np.ndarray], np.ndarray]:
        """The moments of a question (its words, the turn, the silent moment of the answer) and
        the action each slot says there."""
        moments = [self.moment(w) for w in (*question.said, TURN, None)]
        return moments, np.array([self.action[a] for a in question.answer], dtype=np.int64)


class Population:
    """``subjects`` subjects named ``{prefix}{i}`` and their facts, drawn from ``seed``."""

    def __init__(self, seed: Any, subjects: int, prefix: str) -> None:
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        self.prefix = prefix
        self.subjects = [f"{prefix}{i}" for i in range(int(subjects))]
        self.facts: list[Fact] = []
        self.where: dict[tuple[str, str], int] = {}  # (relation, subject): its fact
        for s in self.subjects:
            held = [r for r in ATTRIBUTES if rng.random() < PRESENT]
            if not held:
                held = [ATTRIBUTES[int(rng.integers(len(ATTRIBUTES)))]]
            for r in held:
                self._add(Fact(s, r, PLACES[int(rng.integers(len(PLACES)))]))
            for r in LINKS:
                if rng.random() < PRESENT and len(self.subjects) > 1:
                    other = s
                    while other == s:
                        other = self.subjects[int(rng.integers(len(self.subjects)))]
                    self._add(Fact(s, r, other))
        order = rng.permutation(len(self.facts))
        self.facts = [self.facts[int(i)] for i in order]  # the order they are told in
        self.where = {(f.relation, f.subject): i for i, f in enumerate(self.facts)}
        self._strangers = 0
        self._holders: dict[str, list[str]] = {}

    def _add(self, fact: Fact) -> None:
        self.where[(fact.relation, fact.subject)] = len(self.facts)
        self.facts.append(fact)

    def keep(self, size: int) -> Population:
        """The population of the first ``size`` facts told: the same subjects, a fact past
        ``size`` untold."""
        out = Population.__new__(Population)
        out.prefix, out.subjects = self.prefix, self.subjects
        out.facts = self.facts[: int(size)]
        out.where = {(f.relation, f.subject): i for i, f in enumerate(out.facts)}
        out._strangers = 0
        out._holders = {}
        return out

    def _stranger(self) -> str:
        self._strangers += 1
        return f"{self.prefix}x{self._strangers}"  # a word no fact holds

    def parts(self, rng: np.random.Generator, k: int, absent: float = 0.0) -> Question:
        """A question of ``k`` parts; each part is absent with probability ``absent``."""
        if not self._holders:
            for f in self.facts:
                self._holders.setdefault(f.relation, []).append(f.subject)
        able = [r for r in ATTRIBUTES if len(self._holders.get(r, ())) >= 2 * k]
        if not able:
            raise RuntimeError(f"the population holds no relation of {2 * k} subjects")
        relation = able[int(rng.integers(len(able)))]
        holders = self._holders[relation]
        names: list[str] = []
        needs: list[int] = []
        lost = 0
        while len(names) < k:
            if rng.random() < absent:
                lost += 1
                if rng.random() < 0.5:
                    names.append(self._stranger())
                    needs.append(-1)
                    continue
                s = self.subjects[int(rng.integers(len(self.subjects)))]
                if (relation, s) in self.where or s in names:
                    lost -= 1
                    continue
                names.append(s)
                needs.append(-1)
                continue
            s = holders[int(rng.integers(len(holders)))]
            if s in names:
                continue
            names.append(s)
            needs.append(self.where[(relation, s)])
        said = ["what", "is", "the", relation, "of"]
        for j, s in enumerate(names):
            said += ([] if j == 0 else ["and"]) + [s]
        answer = [self.facts[i].answer if i >= 0 else NOTHING for i in needs]
        answer += [NOTHING] * (SLOTS - k)
        return Question(said + ["?"], answer, needs, "parts", k, {"absent": lost})

    def chain(self, rng: np.random.Generator, hops: int, broken: float = 0.0) -> Question:
        """A question of ``hops`` hops: ``hops - 1`` links and an attribute. With probability
        ``broken`` the chain breaks at one hop: that hop's subject holds no such fact."""
        cut = int(rng.integers(hops)) if rng.random() < broken else -1
        for _ in range(100_000):
            s = self.subjects[int(rng.integers(len(self.subjects)))]
            links = [LINKS[int(i)] for i in rng.integers(len(LINKS), size=hops - 1)]
            relation = ATTRIBUTES[int(rng.integers(len(ATTRIBUTES)))]
            at, needs, whole = s, [], True
            for h, r in enumerate([*links, relation]):
                found = self.where.get((r, at), -1)
                if (found < 0) != (h == cut):
                    whole = False
                    break
                needs.append(found)
                if found < 0:
                    break
                at = self.facts[found].answer
            if not whole:
                continue
            said = ["what", "is", "the", relation]
            for r in links[::-1]:
                said += ["of", "the", r]
            said += ["of", s, "?"]
            needs += [-1] * (hops - len(needs))
            answer = [at if cut < 0 else NOTHING] + [NOTHING] * (SLOTS - 1)
            return Question(said, answer, needs, "chain", hops, {"absent": int(cut >= 0)})
        raise RuntimeError("the population holds no such chain")
