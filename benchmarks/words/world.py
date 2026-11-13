"""The words chamber's world: rooms of things, a robot that looks around, a teacher who names
things and asks questions, and a lexicon of nonce words drawn for each founder.

A room is a plane of 4 m by 4 m with the robot at its middle. A thing is a kind, a colour and a
place, at least 0.6 m from the robot and 0.45 m from any other thing; the kinds in a room are
distinct. The robot's camera sees the things within 60 degrees either side of where it faces.

The concepts are the kinds (8 trained, 4 new, taught live), the colours (6), the four places
around the robot (in front, to the right, behind, to the left: quarters of the turn), the reads
of a question and the read that finds nothing. Every word is a nonce word drawn for the founder
from consonant-vowel syllables, so no word can be built in, and the English names below serve
the reader alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PLANE = (-2.0, 2.0, -2.0, 2.0)
CELLS = (32, 32)
VIEW = math.radians(60.0)  # the camera sees this far either side of the robot's heading
TRAINED, NEW, COLOURS = 8, 4, 6
KINDS = TRAINED + NEW
KIND_NAMES = ("ball", "cup", "box", "book", "shoe", "key", "apple", "lamp",
              "kettle", "drum", "vase", "clock")
COLOUR_NAMES = ("red", "blue", "green", "yellow", "white", "black")
PLACE_NAMES = ("front", "right", "behind", "left")
QUESTION_WORDS = ("where", "what", "colour", "is", "the", "you")
FOILS = 12  # nonce words never taught: six heard in practice, six in the test
CONSONANTS, VOWELS = "bdfgklmnprstvz", "aeiou"
NAMING = (0.5, 0.25, 0.25)  # the teacher names a thing's kind, colour or place at these rates


@dataclass
class Lexicon:
    """A founder's nonce words: one for each concept, the question words, the answer that says
    nothing is found, and the foils."""

    kind: list[str]
    colour: list[str]
    place: list[str]
    question: dict[str, str]
    none: str
    foils: list[str]

    def english(self) -> dict[str, str]:
        """Each nonce word's English name, for a reader."""
        out = {w: n for w, n in zip(self.kind, KIND_NAMES, strict=True)}
        out.update(zip(self.colour, COLOUR_NAMES, strict=True))
        out.update(zip(self.place, PLACE_NAMES, strict=True))
        out.update({w: n for n, w in self.question.items()})
        out[self.none] = "i do not know"
        return out


def lexicon(seed: int) -> Lexicon:
    """The founder's nonce words: distinct forms of two or three consonant-vowel syllables."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 1]))
    need = KINDS + COLOURS + len(PLACE_NAMES) + len(QUESTION_WORDS) + 1 + FOILS
    forms: list[str] = []
    while len(forms) < need:
        form = "".join(str(rng.choice(list(CONSONANTS))) + str(rng.choice(list(VOWELS)))
                       for _ in range(int(rng.integers(2, 4))))
        if form not in forms:
            forms.append(form)
    order = iter(forms)
    take = lambda n: [next(order) for _ in range(n)]  # noqa: E731
    return Lexicon(kind=take(KINDS), colour=take(COLOURS), place=take(len(PLACE_NAMES)),
                   question=dict(zip(QUESTION_WORDS, take(len(QUESTION_WORDS)), strict=True)),
                   none=take(1)[0], foils=take(FOILS))


def unseen_colours(seed: int) -> dict[int, int]:
    """Each trained kind's colour that no training room holds (W3): drawn for the founder."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 2]))
    return {k: int(rng.integers(COLOURS)) for k in range(TRAINED)}


@dataclass
class Thing:
    kind: int
    colour: int
    place: np.ndarray


@dataclass
class Room:
    things: list[Thing]
    heading: float = 0.0  # where the robot faces, radians from the x axis
    extra: dict[str, Any] = field(default_factory=dict)

    def kinds(self) -> list[int]:
        return [t.kind for t in self.things]


def room(rng: np.random.Generator, kinds: list[int], count: int, *,
         colour: Any = None, avoid: dict[int, int] | None = None) -> Room:
    """A room of ``count`` things of distinct kinds drawn from ``kinds``; a thing's colour is
    ``colour(kind)`` where given, else drawn, never a kind's colour in ``avoid``."""
    chosen = [int(k) for k in rng.choice(kinds, size=count, replace=False)]
    things: list[Thing] = []
    for kind in chosen:
        if colour is not None and colour(kind) is not None:
            c = int(colour(kind))
        else:
            options = [c for c in range(COLOURS) if not (avoid and avoid.get(kind) == c)]
            c = int(rng.choice(options))
        for _ in range(1000):
            r = rng.uniform(0.6, 1.8)
            a = rng.uniform(-math.pi, math.pi)
            p = np.array([r * math.cos(a), r * math.sin(a)])
            if all(np.linalg.norm(p - t.place) >= 0.45 for t in things):
                break
        things.append(Thing(kind, c, p))
    return Room(things, heading=float(rng.uniform(-math.pi, math.pi)))


def in_view(heading: float, points: np.ndarray) -> np.ndarray:
    """Which places the camera sees from the room's middle, facing ``heading``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.angle(np.exp(1j * (np.arctan2(p[:, 1], p[:, 0]) - heading)))
    return np.asarray(np.abs(a) <= VIEW)


def camera(heading: float):
    """The camera's sight as the map takes it: a function of places."""
    return lambda points: in_view(heading, points)


def seen(r: Room, heading: float) -> list[tuple[int, np.ndarray, dict[str, int]]]:
    """What the camera reports facing ``heading``: each thing in view, its kind, place and
    colour."""
    return [(t.kind, t.place.copy(), {"colour": t.colour}) for t in r.things
            if in_view(heading, t.place)[0]]


def place_of(heading: float, point: np.ndarray) -> int:
    """The place around the robot of a point: 0 in front, 1 to the right, 2 behind, 3 left."""
    a = float(np.angle(np.exp(1j * (math.atan2(point[1], point[0]) - heading))))
    return int(((-a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))


# -- the teacher's questions


@dataclass
class Question:
    text: str  # the question's nonce words
    form: str  # where, colour, by_colour, by_place
    answer: list[str]  # the words of the right answer (one a thing, or the nothing answer)
    about: str  # the thing's state: in_view, out_of_view, absent, foil, several
    kind: int | None = None


def where(lex: Lexicon, r: Room, kind: int) -> Question:
    q = lex.question
    word = lex.kind[kind] if kind >= 0 else lex.foils[-kind - 1]
    text = f"{q['where']} {q['is']} {q['the']} {word}"
    things = [t for t in r.things if t.kind == kind]
    if not things:
        return Question(text, "where", [lex.none], "absent" if kind >= 0 else "foil", kind)
    t = things[0]
    return Question(text, "where", [lex.place[place_of(r.heading, t.place)]],
                    "in_view" if in_view(r.heading, t.place)[0] else "out_of_view", kind)


def colour_of(lex: Lexicon, r: Room, kind: int) -> Question:
    q = lex.question
    word = lex.kind[kind] if kind >= 0 else lex.foils[-kind - 1]
    text = f"{q['what']} {q['colour']} {q['is']} {q['the']} {word}"
    things = [t for t in r.things if t.kind == kind]
    if not things:
        return Question(text, "colour", [lex.none], "absent" if kind >= 0 else "foil", kind)
    t = things[0]
    return Question(text, "colour", [lex.colour[t.colour]],
                    "in_view" if in_view(r.heading, t.place)[0] else "out_of_view", kind)


def by_colour(lex: Lexicon, r: Room, colour: int) -> Question:
    q = lex.question
    text = f"{q['what']} {q['is']} {lex.colour[colour]}"
    things = [t for t in r.things if t.colour == colour]
    answer = sorted(lex.kind[t.kind] for t in things) or [lex.none]
    return Question(text, "by_colour", answer, "several" if things else "absent")


def by_place(lex: Lexicon, r: Room, place: int) -> Question:
    q = lex.question
    text = f"{q['what']} {q['is']} {lex.place[place]} {q['you']}"
    things = [t for t in r.things if place_of(r.heading, t.place) == place]
    answer = sorted(lex.kind[t.kind] for t in things) or [lex.none]
    return Question(text, "by_place", answer, "several" if things else "absent")


def right(question: Question, said: list[str]) -> bool:
    """An answer is right where the words said are the answer's words, as a set."""
    return sorted(set(said)) == sorted(set(question.answer)) and len(said) == len(
        question.answer)
