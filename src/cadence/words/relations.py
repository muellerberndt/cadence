"""Relations between things, in words learned from questions answered for the parrot
(cadence#327): "what is left of the ball?", "is the cup near the apple?", "where is the ball
from the cup?".

``Relations`` is a part of ``cadence.words.Words`` (``Words(map, parts=[Relations()])``). It adds
five groups of concepts to the lexicon and acts through them:

- ``direction`` and ``distance``: vector cells, a sense the parrot is born with. For a reference
  (a thing's place on the map, or the parrot itself) and a thing, ``direction`` holds the
  thing's direction from the reference in the frame of where the parrot faces (units tuned
  around the turn, each a von Mises curve) and ``distance`` its distance (units tuned on a log
  scale). No unit is a relation: a relation word's meaning is the pattern its lessons leave on
  these units, and where "left" ends or how near is near is where another word's pattern wins.
- ``self``: the parrot itself, the concept a word ("you") is named with while the parrot
  attends to itself (``name_self``).
- ``truth``: whether a relation a question names holds (it holds, it does not).
- ``anchor``: which of two things a question names the other is placed from (the first named or
  the last named): a read, learned from questions answered.

**Placing a thing from a reference.** The map settles on the reference's place and the thing's
place, and the vector cells give the thing's code from the reference. The parrot describes the
thing in words, one word a relation group: the code of the group drives the lexicon's concepts,
the word neurons settle, and the motor chooses the most active word whose own hearing settles
on that group (``describe``). Both directions read the same synapses: a word named with a code
is the word said for codes like it.

**Questions.** Each word is heard alone: a word whose hearing settles on a kind names a thing,
one that settles on the parrot itself names the parrot, and a word the parrot says back for its
own hearing on the relation groups is a relation word. The parrot is not on its own map, so it
is never the thing placed; of two things named, the anchor read says which is the reference.

- "What is left of the ball?" names one thing and a relation: every other thing on the map is
  placed from the ball and kept where its description holds a relation word of the question.
- "Where is the ball from the cup?" names two things and no relation: the ball is found and
  described from the cup; the question's reads (``Words``) choose what is said of it.
- "Is the cup near the apple?" names two things and a relation: the cup is found with
  ``truth``, whether its description from the apple holds the relation word.

**Lessons, from questions answered.** "Where is the X from the Y?" answered with a word: the
answer word is named with the X's code from the Y, as a word is named with a thing; where the
answer decides which of the two things the X was placed from, the question's words are taught
that anchor. "What is left of the X?" answered with things: the question's relation word is
named with each one's code. "Is the X left of the Y?" answered with a word: the word is named
with the parrot's own verdict. A relation word's lesson teaches each relation group at the
sharpness of the word's own hearing there, so the group where its lessons agree keeps its
pattern and the other falls toward zero; after each lesson its synapses onto the relation groups
are held at one length (``norm``, synaptic scaling). No program maps a relation word to a
geometric test.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RELATION_GROUPS = ("direction", "distance")
Named = tuple[Any, ...]  # a thing a question names: ("kind", k) or ("self",)


class Relations:
    """Relations between things, a part of ``cadence.words.Words``. ``directions`` direction
    cells, each a von Mises curve of ``sharpness`` around its preferred direction (evenly
    spaced around the turn, the first straight ahead of the parrot, counter-clockwise);
    ``distances`` distance cells, log-normal curves of ``width`` (natural log units) centred
    from ``nearest`` to ``farthest`` metres on a log scale. ``tries`` is how many of the most
    active words the motor hears back, in order, for one whose hearing settles on the group it
    says. ``norm`` holds each relation word's synapses onto the relation groups at that length
    after each of its lessons (synaptic scaling: the motor's choice then turns on the
    pattern, not its size; None leaves them as the lessons set them). ``contrast`` is the
    power of a group's sharpness in the gain a word is taught at there."""

    answers = ("direction", "distance", "truth")

    def __init__(self, *, directions: int = 8, sharpness: float = 2.5, distances: int = 6,
                 nearest: float = 0.3, farthest: float = 3.0, width: float = 0.37,
                 tries: int = 64, norm: float | None = 1.2, contrast: float = 1.0):
        if int(directions) < 3 or int(distances) < 2:
            raise ValueError("relations take 3 or more direction cells and 2 or more distance "
                             "cells")
        if not 0.0 < float(nearest) < float(farthest) or float(width) <= 0.0:
            raise ValueError("distance cells take 0 < nearest < farthest and a width above 0")
        self.groups = {"direction": int(directions), "distance": int(distances), "self": 1,
                       "truth": 2, "anchor": 2}
        self.sharpness, self.width, self.tries = float(sharpness), float(width), int(tries)
        self.norm = None if norm is None else float(norm)
        self.contrast = float(contrast)
        self.preferred = 2.0 * math.pi * np.arange(int(directions)) / int(directions)
        self.centres = np.geomspace(float(nearest), float(farthest), int(distances))
        self.settings = {"directions": int(directions), "sharpness": self.sharpness,
                         "distances": int(distances), "nearest": float(nearest),
                         "farthest": float(farthest), "width": self.width, "tries": self.tries,
                         "norm": self.norm, "contrast": self.contrast}
        self.last: dict[str, Any] = {}  # what the latest question named, placed and found
        self._back: dict[int, np.ndarray] = {}  # words heard back within one question

    # -- the sense: vector cells

    def code(self, origin: Any, point: Any, heading: float) -> dict[str, np.ndarray]:
        """The vector cells' code of ``point`` from ``origin``, the parrot facing ``heading``
        (radians from the plane's x axis): ``{"direction": (directions,), "distance":
        (distances,)}``, each cell's activity in (0, 1]."""
        d = np.asarray(point, dtype=np.float64).reshape(2) - np.asarray(
            origin, dtype=np.float64).reshape(2)
        angle = math.atan2(float(d[1]), float(d[0])) - float(heading)
        far = max(float(np.hypot(d[0], d[1])), 1e-6)
        return {"direction": np.exp(self.sharpness * (np.cos(angle - self.preferred) - 1.0)),
                "distance": np.exp(-np.log(far / self.centres) ** 2 / (2.0 * self.width ** 2))}

    # -- naming the parrot itself

    def name_self(self, brain: Any, word: str) -> float:
        """A naming of the parrot itself: it attends to itself and hears ``word``. Gives the
        lesson's error."""
        return float(brain.lexicon.name(word, {("self", 0): 1.0}, groups=("self",)))

    # -- hearing a question word by word

    def _hearing(self, brain: Any, words: list[str]) -> dict[str, np.ndarray]:
        """Each known word of a question heard alone."""
        lexicon = brain.lexicon
        return {w: lexicon.hear([w]) for w in dict.fromkeys(words) if lexicon.unit(w) is not None}

    def _named(self, brain: Any, words: list[str], hearing: dict[str, np.ndarray]
               ) -> list[Named]:
        """The things a question names, in the order heard: ``("kind", k)`` or ``("self",)``."""
        groups = brain.lexicon.groups
        out: list[Named] = []
        for w in words:
            if w not in hearing:
                continue
            kinds, me = hearing[w][groups["kind"]], float(hearing[w][groups["self"]][0])
            if max(float(kinds.max()), me) <= brain.floor:
                continue
            out.append(("self",) if me > kinds.max() else ("kind", int(np.argmax(kinds))))
        return out

    def _departures(self, brain: Any, heard: np.ndarray) -> dict[str, float]:
        """How sharply a hearing settles on each relation group: its peak above its mean. A
        word whose lessons' codes agree in a group is sharp there; one whose codes vary keeps a
        broad mean there, and as its gain there falls, a smaller one."""
        groups = brain.lexicon.groups
        out = {}
        for g in RELATION_GROUPS:
            held = np.maximum(heard[groups[g]], 0.0)
            out[g] = float(held.max() - held.mean())
        return out

    def _relational(self, brain: Any, heard: np.ndarray) -> np.ndarray:
        """A hearing on the relation groups alone."""
        groups = brain.lexicon.groups
        return np.concatenate([heard[groups[g]] for g in RELATION_GROUPS])

    def _group_of(self, brain: Any, heard: np.ndarray) -> str | None:
        """The relation group a hearing names: the one it departs most on; None where it
        settles on neither."""
        groups = brain.lexicon.groups
        if max(float(heard[groups[g]].max()) for g in RELATION_GROUPS) <= brain.floor:
            return None
        depart = self._departures(brain, heard)
        return max(depart, key=lambda g: depart[g])

    def _heard_back(self, brain: Any, unit: int) -> np.ndarray:
        if unit not in self._back:
            self._back[unit] = brain.lexicon.hear(brain.lexicon.forms[unit])
        return self._back[unit]

    def _say(self, brain: Any, group: str, pattern: np.ndarray) -> int | None:
        """The word the motor says for a pattern on one relation group, as a word neuron: the
        pattern drives the group's concepts, the word neurons settle, and the motor takes the
        most active word whose own hearing settles on that group; None where none is active."""
        lexicon = brain.lexicon
        drive = np.zeros(lexicon.concepts)
        drive[lexicon.groups[group]] = pattern
        active = lexicon.words_for(drive)
        for unit in np.argsort(-active)[: self.tries]:
            if active[unit] <= 1e-3:
                break
            if self._group_of(brain, self._heard_back(brain, int(unit))) == group:
                return int(unit)
        return None

    def _relations(self, brain: Any, hearing: dict[str, np.ndarray]) -> list[int]:
        """The relation words of a question, as word neurons: each word the parrot says back
        for its own hearing on the relation group it settles on."""
        groups = brain.lexicon.groups
        out: list[int] = []
        for w, heard in hearing.items():
            group = self._group_of(brain, heard)
            unit = brain.lexicon.unit(w)
            if group is not None and self._say(brain, group, heard[groups[group]]) == unit:
                out.append(int(unit))
        return out

    # -- placing a thing from a reference

    def describe(self, brain: Any, code: dict[str, np.ndarray]) -> dict[str, int | None]:
        """The words the parrot says of a code, one a relation group, as word neurons (None
        where no word is said)."""
        return {g: self._say(brain, g, code[g]) for g in RELATION_GROUPS}

    def _place(self, brain: Any, thing: Named) -> np.ndarray | None:
        """A named thing's place: the parrot's own, or the map's settle on the kind."""
        if thing[0] == "self":
            return np.asarray(brain.body.at, dtype=np.float64)
        places = brain.map.where_is(int(thing[1]))
        return np.asarray(places[0]) if len(places) else None

    def _others(self, brain: Any, origin: np.ndarray, reference: Named,
                most: int = 16) -> list[np.ndarray]:
        """The places of the things on the map other than the reference, one at a time: the map
        settles on what is on the cells left, that kind's places are found, and the cells
        around them are left out of the next settle."""
        held = brain.map
        reach = 2.0 * 1.5 * float(held.size.max())
        left = np.ones(held.count, dtype=bool)
        if reference[0] == "kind":
            left &= np.linalg.norm(held.centers - origin, axis=1) > reach
        out: list[np.ndarray] = []
        for _ in range(most):
            kind = held.what_is_at(None, cells=left)
            if kind is None:
                break
            before = int(left.sum())
            for place in held.where_is(kind):
                place = np.asarray(place)
                if not (reference == ("kind", kind)
                        and np.linalg.norm(place - origin) <= reach):
                    out.append(place)
                left &= np.linalg.norm(held.centers - place, axis=1) > reach
            if int(left.sum()) == before:
                break
        return out

    def _described(self, brain: Any, place: np.ndarray, code: dict[str, np.ndarray]
                   ) -> dict[str, np.ndarray]:
        """What the senses give of a thing, and its code from the reference: a thing placed
        from a reference is described from it, so the senses' place around the parrot is left
        out."""
        described = {g: p for g, p in (brain.senses.describe(place) or {}).items()
                     if g != "place"}
        described.update(code)
        return described

    def _roles(self, brain: Any, heard: Any, named: list[Named]
               ) -> tuple[Named, Named | None] | None:
        """The reference and the thing placed from it (None where the question names one
        thing): the parrot is never the thing placed, and of two things the anchor read says
        which is the reference; None where it reads neither."""
        if len(named) == 1:
            return named[0], None
        first, last = named[0], named[-1]
        if ("self",) in (first, last):
            return ("self",), (last if first == ("self",) else first)
        anchor = heard.concepts[brain.lexicon.groups["anchor"]]
        if anchor.max() <= brain.floor:
            return None
        return (last, first) if int(np.argmax(anchor)) == 1 else (first, last)

    def _parse(self, brain: Any, heard: Any) -> dict[str, Any]:
        self._back = {}
        hearing = self._hearing(brain, heard.words)
        named = self._named(brain, heard.words, hearing)
        relation = self._relations(brain, hearing)
        return {"words": list(heard.words), "hearing": hearing, "named": named,
                "relation": relation}

    def find(self, brain: Any, heard: Any, things: list[Any]) -> list[Any]:
        """The things a relation question names (``Words`` calls it after the senses found
        theirs): a question that names neither a relation nor two things is left to the
        senses."""
        parse = self._parse(brain, heard)
        self.last = parse
        named, relation = parse["named"], parse["relation"]
        if not named or (not relation and len(named) < 2):
            return things
        roles = self._roles(brain, heard, named)
        parse["roles"] = roles
        if roles is None:
            return things  # the order is not read yet: nothing is placed, and nothing is absent
        reference, target = roles
        origin = self._place(brain, reference)
        if origin is None:
            return []
        heading = float(brain.body.heading)
        if target is not None:
            point = self._place(brain, target)
            if point is None:
                return []
            code = self.code(origin, point, heading)
            described = self._described(brain, point, code)
            if relation:
                said = self.describe(brain, code)
                parse["said"] = said
                holds = any(u is not None and u in relation for u in said.values())
                described["truth"] = np.array([1.0, 0.0] if holds else [0.0, 1.0])
            return [(point, described)]
        found = []
        for point in self._others(brain, origin, reference):
            code = self.code(origin, point, heading)
            said = self.describe(brain, code)
            if any(u is not None and u in relation for u in said.values()):
                found.append((point, self._described(brain, point, code)))
        return found

    # -- lessons from questions answered

    def _teach(self, brain: Any, word: str, code: dict[str, np.ndarray]) -> None:
        """A relation word named with a code, each relation group at the word's gain there
        (``_gains``; the other group's synapses are taught toward zero), then scaled."""
        gains = self._gains(brain, word)
        concepts = np.zeros(brain.lexicon.concepts)
        for g in RELATION_GROUPS:
            concepts[brain.lexicon.groups[g]] = gains[g] * code[g]
        brain.lexicon.name(word, concepts, groups=RELATION_GROUPS)
        self._scale(brain, word)

    def _gains(self, brain: Any, word: str) -> dict[str, float]:
        """The gains a relation word is taught at, one a relation group: how far its hearing
        departs there from the cells' adapted mean, at a peak of one (in shape, whatever its
        size, so a group where its lessons agree keeps its pattern and one where they vary
        falls away); both groups at one for a word that names neither."""
        lexicon = brain.lexicon
        if lexicon.unit(word) is None:
            return dict.fromkeys(RELATION_GROUPS, 1.0)
        depart = self._departures(brain, lexicon.hear([word]))
        top = max(depart.values())
        if top <= 0.0:
            return dict.fromkeys(RELATION_GROUPS, 1.0)
        return {g: (v / top) ** self.contrast for g, v in depart.items()}

    def _scale(self, brain: Any, word: str) -> None:
        """Synaptic scaling: the word's synapses onto the relation groups held at ``norm``
        together (both of each reciprocal pair), so the balance between the groups is the
        lessons' own."""
        lexicon = brain.lexicon
        unit = lexicon.unit(word)
        if self.norm is None or unit is None:
            return
        pre, post = np.asarray(lexicon.connectome.pre), np.asarray(lexicon.connectome.post)
        cells = np.concatenate([lexicon.concept[lexicon.groups[g]] for g in RELATION_GROUPS])
        neuron = lexicon.word[unit]
        ahead = np.flatnonzero((pre == neuron) & np.isin(post, cells))
        back = np.flatnonzero((post == neuron) & np.isin(pre, cells))
        efficacy = np.array(lexicon.graph.efficacy, dtype=np.float64)
        length = float(np.linalg.norm(efficacy[ahead]))
        if length <= 0.0:
            return
        efficacy[ahead] *= self.norm / length
        efficacy[back] *= self.norm / length
        lexicon.graph.efficacy = efficacy
        self._back.pop(int(unit), None)

    def practice(self, brain: Any, words: list[str], said: list[str], heard: Any) -> None:
        """The part's lessons from a question answered (``Words.practice`` calls it)."""
        parse = self.last if self.last.get("words") == list(words) else self._parse(brain,
                                                                                    heard)
        self._back = {}  # the read lesson since the question moved synapses
        lexicon = brain.lexicon
        groups = lexicon.groups
        named, relation = parse["named"], parse["relation"]
        if not named:
            return
        heading = float(brain.body.heading)
        told = [lexicon.hear([w]) if lexicon.unit(w) is not None else None for w in said]
        names_thing = [t is not None and t[groups["kind"]].max() > brain.floor for t in told]
        if len(named) == 1 and relation:
            # "what is left of the X?" answered with things: each relation word of the
            # question is named with each one's code from the reference
            if not all(names_thing):
                return
            origin = self._place(brain, named[0])
            if origin is None:
                return
            for t in told:
                assert t is not None
                place = self._place(brain, ("kind", int(np.argmax(t[groups["kind"]]))))
                if place is None:
                    continue
                code = self.code(origin, place, heading)
                for unit in relation:
                    self._teach(brain, lexicon.forms[unit], code)
            return
        if len(named) < 2 or len(said) != 1 or names_thing[0]:
            return
        # a question with a word that hears on the relation groups names a relation, even one
        # too weak yet to be said back: its answer is a verdict, never a relation word
        relational = any(float(self._relational(brain, heard_word).max()) > brain.floor
                         for heard_word in parse["hearing"].values())
        if relational and not relation:
            return
        answer, heard_back = said[0], told[0]
        if heard_back is not None and max(float(heard_back[groups["none"]].max()),
                                          float(heard_back[groups["truth"]].max())
                                          ) > brain.floor and not relation:
            return  # the nothing answer, or a word for a verdict, is not a relation
        first, last = named[0], named[-1]
        if relation:
            # "is the X left of the Y?" answered: the word is named with the parrot's verdict
            roles = self._roles(brain, heard, named)
            if roles is None or roles[1] is None:
                return
            origin, point = self._place(brain, roles[0]), self._place(brain, roles[1])
            if origin is None or point is None:
                return
            described = self.describe(brain, self.code(origin, point, heading))
            holds = any(u is not None and u in relation for u in described.values())
            lexicon.name(answer, {("truth", 0 if holds else 1): 1.0}, groups=("truth",))
            return
        # "where is the X from the Y?" answered with a word
        places = {thing: self._place(brain, thing) for thing in (first, last)}
        if any(p is None for p in places.values()):
            return
        order: tuple[Named, Named | None] | None = None
        if ("self",) in (first, last):
            order = self._roles(brain, heard, named)
        else:
            unit = lexicon.unit(answer)
            decided = []
            if unit is not None and heard_back is not None and unit in self._relations(
                    brain, {answer: heard_back}):
                for slot, (ref, thing) in enumerate(((first, last), (last, first))):
                    code = self.code(places[ref], places[thing], heading)
                    if unit in self.describe(brain, code).values():
                        decided.append((slot, (ref, thing)))
            if len(decided) == 1:
                slot, (ref, thing) = decided[0]
                lexicon.name(words, {("anchor", slot): 1.0}, groups=("anchor",))
                order = (ref, thing)
            elif len(decided) == 2:
                order = decided[0][1]  # both orders say the answer (how far): either teaches it
            else:
                order = self._roles(brain, heard, named)
        if order is None or order[1] is None:
            return
        reference, placed = order
        self._teach(brain, answer, self.code(places[reference], places[placed], heading))
