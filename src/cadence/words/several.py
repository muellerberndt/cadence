"""Several things of one kind, answers that name several things, and how many (cadence#328).

``Several`` is the brain of ``cadence.words.Words`` with what a question names read word by
word, the things it names found by every concept it names, and two more reads of what it
finds:

- **What a question names.** Each of the question's words is heard, and the lexicon settles on
  it: the word names its most active concept of a thing (a kind, a value or a place around
  the body) where that concept is above ``floor``. "The red ball" names the kind ball and the
  value red; "the green" names the value green alone, though the word holds a little of the
  kind and the place the green things of its namings had. The question's words heard together
  give the read it asks.
- **The things it names** are found on the map by settling. The map settles on one named
  concept (a kind's places, a value's places, or what is at a place around the body), and a
  place is kept where the map's settle at it gives every other named concept: its kind
  (``what_is_at``), its value (``values_at``), its place around the body. Two balls of two
  colours are two places; "the red ball" is the place whose value the map settles on as red.
- **Reads.** A read gives a concept for each thing found (its kind, its value or its place
  around the body, as ``Words`` does) or one concept for the things found together:
  ``number``, how many there are, and ``truth``, whether there is one. The number sense is
  born: one neuron a count of the places a settle finds, from none to ``numbers - 1`` (the last
  counts that many or more), as an animal is born with a sense of small numbers. The words for
  the counts are learned.
- **Practice.** A question with the teacher's answer teaches two things. The question's words
  are taught each read whose concepts the lexicon's motor would say as the teacher's words; a
  read that finds nothing teaches nothing. An answer of one word that no naming has shown is
  named with what the brain found, at the question's reads as gains: the count found, whether
  there is one, and the read that finds nothing for a read of a thing that found none. A read
  whose concepts hold other words than the teacher's takes no part (a concept holds a word
  where the word the motor says for it, heard, names it above ``floor``). Of the reads that
  take part, the most active one the question's words drive takes the whole gain; where none
  is heard above ``floor`` they share it alike. A word answered to "how many" questions so
  keeps the count its answers share, and a word answered to "is there" questions keeps whether
  there was one. Where the teacher asks "how many" before "is there", "no" finds the truth
  free of a word; asked together from the first question, "zero" and "no" may each take the
  other's concept (``benchmarks/words/several.py`` reads both orders).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .lexicon import Lexicon
from .room import PLACES, Body, Words

READS = ("kind", "colour", "place", "number", "truth")  # what a question asks of what it finds
THING = ("kind", "colour", "place")  # a thing's concepts, named by showing the thing
NOTHING = ("none", 0)


class Several(Words):
    """A brain that answers questions about several things: a ``map`` with values of
    ``group``, a ``Body`` and a ``Lexicon`` over the map's kinds, the group's values, the places
    around the body, the five reads (``READS``), the read that finds nothing, the counts from
    none to ``numbers - 1`` and the two truths (there is none, there is one); made here unless
    given. The settings are those of ``Words``."""

    def __init__(self, map: Any, *, group: str = "colour", numbers: int = 6,
                 lexicon: Lexicon | None = None,
                 body: Body | None = None, shares: tuple[float, float, float] = (0.4, 0.3, 0.3),
                 floor: float = 0.05, rate: float = 0.05, capacity: int = 256,
                 backend: Any = "cpu"):
        if int(numbers) < 2:
            raise ValueError("a brain of several things counts 2 or more")
        if lexicon is None and group in map.values:
            held = map.values[group]
            lexicon = Lexicon({"kind": len(map.codes), "colour": held.stop - held.start,
                               "place": len(PLACES), "read": len(READS), "none": 1,
                               "number": int(numbers), "truth": 2},
                              rate=rate, capacity=capacity, backend=backend)
        super().__init__(map, group=group, lexicon=lexicon, body=body, shares=shares,
                         floor=floor, rate=rate, capacity=capacity, backend=backend)
        groups = self.lexicon.groups
        sizes = {g: s.stop - s.start for g, s in groups.items()}
        if sizes.get("read") != len(READS) or sizes.get("number") != int(numbers) or (
                sizes.get("truth") != 2):
            raise ValueError(f"the lexicon of a brain of several things holds {len(READS)} "
                             f"reads, {int(numbers)} counts and 2 truths")
        self.numbers = int(numbers)

    # -- a question

    def named(self, words: list[str]) -> tuple[dict[str, int], int | None]:
        """What a question names (``{group: concept}``: each word heard alone names its most
        active concept of a thing, above ``floor``; of two words that name one group, the more
        active) and the read it asks (None where no read is heard), from the lexicon's settle
        on its words together."""
        groups = self.lexicon.groups
        about: dict[str, int] = {}
        strength: dict[str, float] = {}
        for word in words:
            one = self.lexicon.hear(word)
            group = max(THING, key=lambda g: float(one[groups[g]].max()))
            part = one[groups[group]]
            if part.max() > self.floor and part.max() > strength.get(group, 0.0):
                about[group], strength[group] = int(np.argmax(part)), float(part.max())
        heard = self.lexicon.hear(words)
        reads = heard[groups["read"]]
        read = int(np.argmax(reads)) if reads.max() > self.floor else None
        self.last = {"heard": heard, "about": about, "read": read}
        return about, read

    def concept_at(self, group: str, place: Any) -> int | None:
        """A thing's concept of ``group`` at a place, as the map and the body give it: its
        kind (``what_is_at``), its value (``values_at``) or its place around the body."""
        found: int | None
        if group == "kind":
            found = self.map.what_is_at(place)
        elif group == "colour":
            found = self.map.values_at(place)[self.group]
        else:
            found = self.body.place(place)
        return found

    def found(self, about: dict[str, int]) -> list[np.ndarray]:
        """The places of the things a question names: the map settles on one named concept,
        and a place is kept where the map's settle at it gives every other one."""
        if not about:
            return []
        if "kind" in about:
            cued, places = "kind", list(self.map.where_is(about["kind"]))
        elif "colour" in about:
            cued, places = "colour", list(self.map.where_is_value(self.group, about["colour"]))
        else:
            cued, places = "place", self._things_in(about["place"])
        return [p for p in places if all(self.concept_at(g, p) == v
                                          for g, v in about.items() if g != cued)]

    def read_out(self, about: dict[str, int], read: int | None,
                 places: list[np.ndarray]) -> list[tuple[str, int]]:
        """The concepts a read gives of the things found: one a thing for a thing's read, one
        for a read of the things together (their count, whether there is one); the read that
        finds nothing where the question names nothing, no read is heard, or a thing's read
        finds none."""
        if not about or read is None:
            return [NOTHING]
        name = READS[read]
        if name == "number":
            return [("number", min(len(places), self.numbers - 1))]
        if name == "truth":
            return [("truth", int(bool(places)))]
        out = []
        for place in places:
            concept = self.concept_at(name, place)
            if concept is not None:
                out.append((name, int(concept)))
        return out or [NOTHING]

    def ask(self, question: str) -> list[str]:
        """The words the brain says to a question: one a thing found for a thing's read, one
        for a read of the things together, said by the lexicon's motor."""
        about, read = self.named(self._words(question))
        places = self.found(about)
        said = []
        for concept in self.read_out(about, read, places):
            word = self.lexicon.say(concept)
            said.append(word if word is not None else "")
        self.last.update({"places": places, "said": said})
        return said

    # -- practice

    def practice(self, question: str, answer: str | list[str]) -> None:
        """A question with the teacher's answer: one word (a form, which may hold spaces) or a
        list of them. The question's words are taught each read whose concepts the motor would
        say as the answer's words. An answer of one word that no naming has shown is named
        with what the brain found, at the question's reads as gains, among the reads whose
        concepts the motor would say as no other word."""
        words = self._words(question)
        about, read = self.named(words)
        places = self.found(about)
        said = [answer] if isinstance(answer, str) else list(answer)
        wanted = sorted(" ".join(str(w).lower().split()) for w in said)
        right, open_ = [], []
        for r in range(len(READS)):
            concepts = self.read_out(about, r, places)
            spoken = sorted(self.lexicon.say(c) or "" for c in concepts)
            if spoken == wanted and concepts != [NOTHING]:
                right.append(r)  # a read that finds nothing tells nothing of what was asked
            held = [self._word_for(c) for c in concepts]
            if spoken == wanted or not any(held):
                open_.append(r)  # the motor says the teacher's words, or the concepts hold none
        target = self._found_concepts(about, places, open_) if (
            len(said) == 1 and not self._shown(said[0])) else {}
        if right:
            self.lexicon.name(words, {("read", r): 1.0 / len(right) for r in right},
                              groups=("read",))
        if target:
            self.lexicon.name(said[0], target, groups=("number", "truth", "none"))

    def _word_for(self, concept: tuple[str, int]) -> str | None:
        """The word a concept holds: the word the motor says for it, where that word heard
        names the concept above ``floor``; None for a concept no word holds so."""
        word = self.lexicon.say(concept)
        if word is None:
            return None
        heard = self.lexicon.hear(word)
        return word if heard[self.lexicon.index(*concept)] > self.floor else None

    def _shown(self, word: str) -> bool:
        """Whether a naming has shown a thing with the word."""
        unit = self.lexicon.unit(word)
        counts = self.lexicon.counts.get(THING)
        return unit is not None and counts is not None and bool(counts[unit] > 0)

    def _found_concepts(self, about: dict[str, int], places: list[np.ndarray],
                        open_: list[int]) -> dict[tuple[str, int], float]:
        """What the brain found, as concepts at the question's reads as gains, over the reads
        ``open_`` takes part: the count and whether there is one, and the read that finds
        nothing for a read of a thing that found none (a thing's own concepts are named by
        showing it). The most active read that takes part takes the whole gain; where none is
        heard above ``floor`` they share it alike. A
        question that names nothing the brain knows gives the read that finds nothing alone."""
        if not about:
            return {NOTHING: 1.0}
        if not open_:
            return {}
        taking = np.zeros(len(READS))
        taking[open_] = 1.0
        reads = np.where(taking > 0, self.last["heard"][self.lexicon.groups["read"]], -np.inf)
        if reads.max() > self.floor:  # the read the question asks, of those taking part
            gains = np.zeros(len(READS))
            gains[int(np.argmax(reads))] = 1.0
        else:
            gains = taking / taking.sum()
        out: dict[tuple[str, int], float] = {}
        for r, name in enumerate(READS):
            if not gains[r] > 0.0:
                continue
            if name == "number":
                concept = ("number", min(len(places), self.numbers - 1))
            elif name == "truth":
                concept = ("truth", int(bool(places)))
            elif not places:
                concept = NOTHING
            else:
                continue
            out[concept] = out.get(concept, 0.0) + float(gains[r])
        return {c: g for c, g in out.items() if g > 0.0}


__all__ = ["NOTHING", "READS", "THING", "Several"]
