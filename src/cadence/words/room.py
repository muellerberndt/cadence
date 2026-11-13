"""A brain that answers questions about its room in words it learned from naming.

``Words`` holds three parts and sequences their settles:

- **the map** (``cadence.space.map.Map`` with ``values=``): what is where, and what each thing
  is like (its colour), carried with the thing, in view and out of it;
- **the body**: where the brain stands and which way it faces, so a place on the map is a
  place around the body (in front, to the right, behind, to the left), as a robot knows its own
  pose;
- **the lexicon** (``cadence.words.Lexicon``): words bound to the concepts, learned from
  naming. Its concepts are the map's kinds, the colours, the four places around the body, the
  three reads of a question (the kind, the colour or the place of what it finds) and the read
  that finds nothing.

**Naming.** The brain attends to the thing at a place; the map settles on what is there (its
kind and colour) and the body gives where it is around the body. A word heard then is taught to
the lexicon with those concepts, each at its share (``shares``: the kind, the colour, the
place). Across namings a word keeps the concept its namings share.

**A question.** Its words are heard together and the lexicon settles: the concept neurons give
what is asked about (the most active kind, colour or place) and the read neurons what is asked
(the most active read). The map is asked by settling: a kind by ``where_is``, a colour by
``where_is_value``, a place around the body by the cells it covers (``what_is_at`` on them,
one thing at a time). For each thing found the read gives a concept (its kind, its colour or its
place around the body), and the lexicon's motor says the word for it. Where nothing is found, or
the question names no concept the lexicon holds, the concept is the read that finds nothing.

**Practice.** A question with the teacher's answer: the brain hears the answer's words and
reads, in its own lexicon, which group of concepts they name; the question's words are taught
that read. An answer to a read that found nothing is named with that concept.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .lexicon import Lexicon

PLACES = ("front", "right", "behind", "left")  # the places around the body, in their order
READS = ("kind", "colour", "place")  # what a question asks of what it finds


class Body:
    """Where a brain stands on the map's plane and which way it faces: ``(x, y)`` in metres and
    ``heading`` in radians from the plane's x axis, counter-clockwise."""

    def __init__(self, at: Any = (0.0, 0.0), heading: float = 0.0):
        self.at = np.asarray(at, dtype=np.float64).reshape(2).copy()
        self.heading = float(heading)

    def angle(self, points: np.ndarray) -> np.ndarray:
        """Each place's direction from the body, relative to where it faces, in (-pi, pi]."""
        d = np.asarray(points, dtype=np.float64).reshape(-1, 2) - self.at
        a = np.arctan2(d[:, 1], d[:, 0]) - self.heading
        return np.asarray(np.angle(np.exp(1j * a)))

    def place(self, point: Any) -> int:
        """The place around the body a point lies in: 0 in front, 1 to the right, 2 behind,
        3 to the left, each a quarter of the turn around the body."""
        a = float(self.angle(np.asarray(point))[0])
        return int(((-a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))

    def cells(self, place: int, centers: np.ndarray) -> np.ndarray:
        """The cells a place around the body covers, as flags on ``centers``."""
        a = self.angle(centers)
        quarter = (((-a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2)).astype(int)
        return np.asarray(quarter == int(place))


class Words:
    """A brain that answers questions about its room: a ``map`` with values of ``group`` (a
    ``cadence.space.map.Map`` made with ``values=``), a ``Body`` and a ``Lexicon`` over the
    map's kinds, the group's values, the places around the body, the reads and the read that
    finds nothing (made here unless given). ``shares`` are a naming's shares of the thing's
    kind, its value and its place; ``floor`` is the activity a concept needs to count as
    named."""

    def __init__(self, map: Any, *, group: str = "colour", lexicon: Lexicon | None = None,
                 body: Body | None = None, shares: tuple[float, float, float] = (0.4, 0.3, 0.3),
                 floor: float = 0.05, rate: float = 0.05, capacity: int = 256,
                 backend: Any = "cpu"):
        if group not in map.values:
            raise ValueError(f"the map holds no values of {group!r}: Map(values={{{group!r}: n}})")
        self.map, self.group = map, group
        self.body = body if body is not None else Body()
        self.kinds = len(map.codes)
        self.values = map.values[group].stop - map.values[group].start
        self.lexicon = lexicon if lexicon is not None else Lexicon(
            {"kind": self.kinds, "colour": self.values, "place": len(PLACES),
             "read": len(READS), "none": 1}, rate=rate, capacity=capacity, backend=backend)
        self.shares = tuple(float(s) for s in shares)
        self.floor = float(floor)
        self.last: dict[str, Any] = {}  # what the latest question heard, attended and found

    # -- what is at a place, as concepts

    def attend(self, place: Any) -> dict[tuple[str, int], float]:
        """The concepts of the thing at a place, each at its naming share: its kind and value
        as the map settles on them there, and its place around the body. Empty where the map
        holds nothing there."""
        kind = self.map.what_is_at(place)
        if kind is None:
            return {}
        out = {("kind", int(kind)): self.shares[0],
               ("place", self.body.place(place)): self.shares[2]}
        value = self.map.values_at(place)[self.group]
        if value is not None:
            out[("colour", int(value))] = self.shares[1]
        return out

    def name(self, word: str, at: Any) -> float:
        """A naming: the brain attends to the thing at ``at`` and hears ``word``. Gives the
        lesson's error (as a share of its target), or nan where the map holds nothing there."""
        concepts = self.attend(at)
        if not concepts:
            return float("nan")
        return self.lexicon.name(word, concepts, groups=("kind", "colour", "place"))

    # -- a question

    def _heard(self, words: list[str]) -> tuple[tuple[str, int] | None, int | None]:
        """What a question asks about and what it asks, from the lexicon's settle."""
        heard = self.lexicon.hear(words)
        groups = self.lexicon.groups
        best: tuple[str, int] | None = None
        top = self.floor
        for g in ("kind", "colour", "place"):
            part = heard[groups[g]]
            if part.max() > top:
                top, best = float(part.max()), (g, int(np.argmax(part)))
        reads = heard[groups["read"]]
        read = int(np.argmax(reads)) if reads.max() > self.floor else None
        self.last = {"heard": heard, "about": best, "read": read}
        return best, read

    def find(self, about: tuple[str, int] | None, read: int | None) -> list[tuple[str, int]]:
        """The concepts a read gives of the things a concept names on the map: one a thing
        found, or the read that finds nothing."""
        nothing = [("none", 0)]
        if about is None or read is None:
            return nothing
        group, value = about
        if group == "kind":
            places = list(self.map.where_is(value))
        elif group == "colour":
            places = list(self.map.where_is_value(self.group, value))
        else:
            places = self._things_in(value)
        out = []
        for place in places:
            if READS[read] == "kind":
                kind = self.map.what_is_at(place)
                if kind is not None:
                    out.append(("kind", int(kind)))
            elif READS[read] == "colour":
                found = self.map.values_at(place)[self.group]
                if found is not None:
                    out.append(("colour", int(found)))
            else:
                out.append(("place", self.body.place(place)))
        self.last["places"] = places
        return out or nothing

    def _things_in(self, around: int, most: int = 8) -> list[np.ndarray]:
        """The places of the things at a place around the body, one at a time: the map settles
        on what is on the cells that place covers, the places of that kind are found, those the
        body places there are kept, and the cells around every place of the kind are left out
        of the next settle."""
        left = self.body.cells(around, self.map.centers)
        out: list[np.ndarray] = []
        reach = 1.5 * float(self.map.size.max())
        for _ in range(most):
            kind = self.map.what_is_at(None, cells=left)
            if kind is None:
                break
            places = self.map.where_is(kind)
            out += [p for p in places if self.body.place(p) == around]
            before = int(left.sum())
            for p in places:
                left &= np.linalg.norm(self.map.centers - p, axis=1) > 2.0 * reach
            if int(left.sum()) == before:
                break
        return out

    def ask(self, question: str) -> list[str]:
        """The words the brain says to a question: one a thing found, said by the lexicon's
        motor; the word for the read that finds nothing where nothing is found."""
        about, read = self._heard(self._words(question))
        said = []
        for concept in self.find(about, read):
            word = self.lexicon.say(concept)
            said.append(word if word is not None else "")
        self.last["said"] = said
        return said

    @staticmethod
    def _words(question: str) -> list[str]:
        return [w for w in "".join(c if c.isalpha() or c in " '" else " "
                                   for c in question.lower()).split()]

    def practice(self, question: str, answer: str | list[str]) -> None:
        """A question with the teacher's answer: one word (a form, which may hold spaces, as
        ``"i do not know"``) or a list of them. The answer's words are heard: where the
        lexicon names a kind, a colour or a place with them, the question's words are taught
        that read. Where they name none of those and the brain's read finds nothing, the answer
        is named with the read that finds nothing."""
        words = self._words(question)
        about, read = self._heard(words)
        found = self.find(about, read)
        groups = self.lexicon.groups
        said = [answer] if isinstance(answer, str) else list(answer)
        heard = self.lexicon.hear(said)
        named = max(("kind", "colour", "place"), key=lambda g: float(heard[groups[g]].max()))
        if heard[groups[named]].max() > self.floor:
            self.lexicon.name(words, {("read", READS.index(named)): 1.0}, groups=("read",))
        elif found == [("none", 0)] and len(said) == 1:
            self.lexicon.name(said[0], {("none", 0): 1.0}, groups=("none",))
