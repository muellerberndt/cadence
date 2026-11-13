"""A brain that answers questions about its room in words it learned from naming.

``Words`` holds its parts and sequences their settles:

- **the senses** (``Senses``): what the brain senses of a place, in groups of concepts. Born
  with the brain: the map of things (``cadence.space.map.Map``, what is where and, with
  ``values=``, what each thing is like, carried with the thing in view and out of it) and the
  body (where the brain stands and which way it faces, so a place on the map is a place around
  the body). A learned set of concepts takes their place where it gives the same calls;
- **the lexicon** (``cadence.words.Lexicon``): words bound to the concepts, learned from
  naming. Its concepts are the senses' groups, one read a group (what a question asks for), the
  groups the plug-in parts add, and the read that finds nothing;
- **parts** that plug into living, finding and practice (records of the past, a forecast,
  relations, counting): each adds its own concept groups to the lexicon and acts through them.

**Naming.** The brain attends to the thing at a place; the senses give what is there (its kind,
what it is like, where it is around the body), and a word heard then is taught to the lexicon
with those concepts, each group at its share (``shares``). Across namings a word keeps the
concept its namings share.

**A question.** Its words are heard together and the lexicon settles. The concept neurons give
what is asked about: the group whose most active concept is the most active of all. The map is
asked by settling for the things that concept names. The read neurons are gains: each found
thing's concepts, every group times its read's settled activity, drive the lexicon's saying
settle, and its motor says the word. So what a question word asks for is in the synapses its
practice taught, the reads compete for their gains, and no table maps a question word, or a
read, to an answer. Where nothing is
found, or the question names no concept the lexicon holds, the concept is the read that finds
nothing.

**Practice.** A question with the teacher's answer: the brain hears the answer's words and
reads, in its own lexicon, which group of concepts they name; the question's words that name no
concept (the words of asking, not the word asked about) are taught that group's read. An answer
that names no concept, given where the brain's read finds nothing, is named with the read that
finds nothing. **Reward.** With the answer never told, the brain tries a read (``ask(explore=
True)``), and a right answer earns a reward (``reward``: a pistachio, 1): the words of asking are
taught the read it tried, at the reward's surprise.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .lexicon import Lexicon

PLACES = ("front", "right", "behind", "left")  # the places around the body, in their order


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


class Senses:
    """What a brain senses of a place, in groups of concepts: ``kind`` (the map's kinds), one
    group of the map's values each (``colour``), and ``place`` (the four places around the
    body). Born with the brain: the map and the body.

    ``describe(place)`` gives each group's pattern of the thing there (None where the map holds
    nothing) and ``find(group, pattern)`` the places of the things whose ``group`` the pattern
    names, by settling: a kind by ``where_is``, a value by ``where_is_value``, a place around
    the body by the map's settle on the cells it covers, one thing at a time. A learned set of
    concepts gives the same three: ``groups``, ``describe`` and ``find``."""

    def __init__(self, map: Any, body: Body):
        self.map, self.body = map, body
        self.groups: dict[str, int] = {"kind": len(map.codes)}
        for name, units in map.values.items():
            self.groups[name] = units.stop - units.start
        self.groups["place"] = len(PLACES)

    def describe(self, place: Any) -> dict[str, np.ndarray] | None:
        kind = self.map.what_is_at(place)
        if kind is None:
            return None
        out = {"kind": np.eye(self.groups["kind"])[int(kind)]}
        for name, value in self.map.values_at(place).items():
            out[name] = (np.eye(self.groups[name])[int(value)] if value is not None
                         else np.zeros(self.groups[name]))
        out["place"] = np.eye(len(PLACES))[self.body.place(place)]
        return out

    def find(self, group: str, pattern: np.ndarray) -> list[np.ndarray]:
        value = int(np.argmax(pattern))
        if group == "kind":
            return list(self.map.where_is(value))
        if group == "place":
            return self.things_in(value)
        return list(self.map.where_is_value(group, value))

    def things_in(self, around: int, most: int = 8) -> list[np.ndarray]:
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


class Heard:
    """What a question's words settled on: the concepts' activity (``concepts``), what is asked
    about (``about``: a group and its pattern, or None) and each read's activity (``reads``,
    one a group the brain can answer with)."""

    def __init__(self, words: list[str], concepts: np.ndarray,
                 about: tuple[str, np.ndarray] | None, reads: dict[str, float]):
        self.words, self.concepts, self.about, self.reads = words, concepts, about, reads


class Words:
    """A brain that answers questions about its room: ``map`` (a ``cadence.space.map.Map``,
    with ``values=`` for what things are like), a ``Body``, ``Senses`` of the two (made here
    unless given), plug-in ``parts`` and a ``Lexicon`` over the senses' groups, a read a group,
    the parts' groups and the read that finds nothing (made here unless given).

    ``shares`` are a naming's shares of the senses' groups, by name: the kind's (0.4 unless
    given), the rest split evenly among the other groups. ``floor`` is the activity a concept
    needs to count as named.

    A part is an object with ``groups`` (the concept groups it adds to the lexicon, by name and
    size) and any of: ``answers`` (those of its groups a question may ask for, each given a
    read), ``sense(brain, seen, lit)`` (called at each moment ``look`` passes to the map, before
    the map takes it), ``find(brain, heard, things)`` (gives the things found, from those found
    before it: a list of ``(place, concepts)``), ``practice(brain, words, answer, heard)``
    (its lessons from a question answered), ``takes(brain, words, answer, heard)`` (True where
    the part takes a question's lessons: ``practice`` then takes none of its own)."""

    def __init__(self, map: Any, *, senses: Any = None, parts: Any = (),
                 lexicon: Lexicon | None = None, body: Body | None = None,
                 shares: dict[str, float] | None = None, floor: float = 0.05,
                 rate: float = 0.05, capacity: int = 256, temperature: float = 0.1,
                 sharpness: float = 0.02, seed: int = 0, backend: Any = "cpu"):
        self.map = map
        self.body = body if body is not None else Body()
        self.senses = senses if senses is not None else Senses(map, self.body)
        self.parts = list(parts)
        sensed = dict(self.senses.groups)
        answers = list(sensed)
        added: dict[str, int] = {}
        for part in self.parts:
            added.update(getattr(part, "groups", {}))
            answers += list(getattr(part, "answers", ()))
        self.answers = answers  # the groups a question may ask for, a read each
        concepts = {**sensed, "read": len(answers), **added, "none": 1}
        self.lexicon = lexicon if lexicon is not None else Lexicon(
            concepts, rate=rate, capacity=capacity, backend=backend)
        given = dict(shares or {})
        kind = given.pop("kind", 0.4)
        rest = [g for g in sensed if g != "kind"]
        self.shares = {"kind": float(kind), **{g: float(given.get(g, (1.0 - kind) / max(
            len(rest), 1))) for g in rest}}
        self.floor = float(floor)
        self.temperature = float(temperature)  # how a read is chosen where the brain tries one
        self.sharpness = float(sharpness)  # how the reads compete when the brain answers
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 11]))
        self.last: dict[str, Any] = {}  # what the latest question heard and found

    # -- living and naming

    def look(self, seen: Any = (), lit: Any = None, heard: Any = (), felt: Any = ()) -> None:
        """One moment of the senses: each part takes it first, then the map."""
        for part in self.parts:
            if hasattr(part, "sense"):
                part.sense(self, seen, lit)
        self.map.moment(seen=seen, heard=heard, felt=felt, lit=lit)

    def concepts(self, described: dict[str, np.ndarray], gains: dict[str, float]
                 ) -> np.ndarray:
        """A vector over the lexicon's concepts: each group's pattern times its gain."""
        out = np.zeros(self.lexicon.concepts)
        for group, pattern in described.items():
            if group in self.lexicon.groups and gains.get(group, 0.0):
                out[self.lexicon.groups[group]] += gains[group] * np.asarray(pattern)
        return out

    def attend(self, place: Any) -> np.ndarray | None:
        """The concepts of the thing at a place, each group at its naming share; None where the
        map holds nothing there."""
        described = self.senses.describe(place)
        return None if described is None else self.concepts(described, self.shares)

    def name(self, word: str, at: Any) -> float:
        """A naming: the brain attends to the thing at ``at`` and hears ``word``. Gives the
        lesson's error (as a share of its target), or nan where the map holds nothing there."""
        concepts = self.attend(at)
        if concepts is None:
            return float("nan")
        return self.lexicon.name(word, concepts, groups=tuple(self.shares))

    # -- a question

    def hear(self, words: list[str]) -> Heard:
        """What the question's words settle on."""
        concepts = self.lexicon.hear(words)
        groups = self.lexicon.groups
        about: tuple[str, np.ndarray] | None = None
        top = self.floor
        for g in self.senses.groups:
            part = concepts[groups[g]]
            if part.max() > top:
                top, about = float(part.max()), (g, part)
        reads = concepts[groups["read"]]
        heard = Heard(words, concepts, about, {g: float(max(reads[i], 0.0))
                                                for i, g in enumerate(self.answers)})
        self.last = {"heard": heard}
        return heard

    def find(self, heard: Heard) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        """The things the question names, each a place and what the senses give of it; each
        part then gives the things from those found before it."""
        things: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        if heard.about is not None:
            for place in self.senses.find(*heard.about):
                described = self.senses.describe(place)
                if described is not None:
                    things.append((np.asarray(place), described))
        for part in self.parts:
            if hasattr(part, "find"):
                things = part.find(self, heard, things)
        self.last["things"] = things
        return things

    def answer(self, heard: Heard, described: dict[str, np.ndarray],
               gains: dict[str, float] | None = None) -> np.ndarray | None:
        """The concepts the motor says of a thing found: every group of it times its read's
        gain, at a peak of one; None where no read is above the floor. The reads compete for
        their gains, a softmax of their settled activity at ``sharpness``, unless ``gains``
        are given."""
        if gains is None:
            live = {g: r for g, r in heard.reads.items() if r > self.floor}
            if live:
                acts = np.array(list(live.values()))
                odds = np.exp((acts - acts.max()) / max(self.sharpness, 1e-9))
                gains = dict(zip(live, odds / odds.sum(), strict=True))
            else:
                gains = {}
        vector = self.concepts(described, gains)
        return vector / vector.max() if vector.max() > 0 else None

    def choose(self, heard: Heard) -> tuple[str, float]:
        """A read tried: drawn from the reads' settled activity at ``temperature``, and the
        chance it was drawn with. Where no read is learned every read is as likely."""
        groups = list(heard.reads)
        acts = np.array([heard.reads[g] for g in groups])
        odds = np.exp((acts - acts.max()) / max(self.temperature, 1e-9))
        odds /= odds.sum()
        pick = int(self.rng.choice(len(groups), p=odds))
        return groups[pick], float(odds[pick])

    def ask(self, question: str, *, explore: bool = False) -> list[str]:
        """The words the brain says to a question: one a thing found, said by the lexicon's
        motor; the word for the read that finds nothing where nothing is found. With
        ``explore`` the brain tries one read, drawn by ``choose``, in place of the reads'
        gains: what a reward (``reward``) teaches."""
        heard = self.hear(self._words(question))
        if explore:
            taken, chance = self.choose(heard)
        else:  # the read the gains favour, and how sure the brain is of it
            taken = max(heard.reads, key=lambda g: heard.reads[g])
            chance = self._odds(heard, taken)
        self.last.update({"taken": taken, "chance": chance})
        said = []
        for _place, described in self.find(heard):
            vector = self.answer(heard, described, {taken: 1.0} if explore else None)
            word = self.lexicon.say(vector) if vector is not None else None
            if word is not None:
                said.append(word)
        if not said:
            said = [self.lexicon.say(("none", 0)) or ""]
        self.last["said"] = said
        return said

    def _odds(self, heard: Heard, taken: str) -> float:
        acts = np.array(list(heard.reads.values()))
        odds = np.exp((acts - acts.max()) / max(self.temperature, 1e-9))
        return float(odds[list(heard.reads).index(taken)] / odds.sum())

    def reward(self, amount: float = 1.0) -> float:
        """A reward for the latest answer (a pistachio for a right one: 1): the question's
        words of asking are taught the read the answer took, at the reward's surprise (the
        reward less the chance the brain gave that read). A reward of zero teaches nothing.
        Gives the surprise."""
        heard = self.last.get("heard")
        taken = self.last.get("taken")
        if heard is None or taken is None:
            return 0.0
        surprise = float(amount) - float(self.last.get("chance", 0.0))
        asking = self.asking(heard.words)
        if float(amount) > 0 and surprise > 0 and asking:
            self.lexicon.name(asking, {("read", self.answers.index(taken)): 1.0},
                              groups=("read",), weight=surprise)
        return surprise

    def asking(self, words: list[str]) -> list[str]:
        """The words of a question that learn what it asks for: those that name no concept of
        the senses (a word that names a kind, a colour or a place is what the question is
        about, and keeps no read)."""
        groups = self.lexicon.groups
        out = []
        for word in words:
            heard = self.lexicon.hear(word)
            if max(float(heard[groups[g]].max()) for g in self.senses.groups) <= self.floor:
                out.append(word)
        return out

    @staticmethod
    def _words(question: str) -> list[str]:
        return [w for w in "".join(c if c.isalpha() or c in " '" else " "
                                   for c in question.lower()).split()]

    def practice(self, question: str, answer: str | list[str]) -> None:
        """A question with the teacher's answer: one word (a form, which may hold spaces, as
        ``"i do not know"``) or a list of them. The answer's words are heard: where the lexicon
        names a group the brain can answer with, the question's words are taught that group's
        read. Where they name none and the brain's read finds nothing, the answer is named with
        the read that finds nothing. Each part then takes its own lessons."""
        words = self._words(question)
        heard = self.hear(words)
        found = self.find(heard)
        groups = self.lexicon.groups
        said = [answer] if isinstance(answer, str) else list(answer)
        told = self.lexicon.hear(said)
        named = max(self.answers, key=lambda g: float(told[groups[g]].max()))
        asking = self.asking(words)
        if any(hasattr(part, "takes") and part.takes(self, words, said, heard)
               for part in self.parts):
            pass  # a part takes this question's lessons, in its practice below
        elif told[groups[named]].max() > self.floor and asking:
            self.lexicon.name(asking, {("read", self.answers.index(named)): 1.0},
                              groups=("read",))
        elif not found and len(said) == 1:
            self.lexicon.name(said[0], {("none", 0): 1.0}, groups=("none",))
        for part in self.parts:
            if hasattr(part, "practice"):
                part.practice(self, words, said, heard)
