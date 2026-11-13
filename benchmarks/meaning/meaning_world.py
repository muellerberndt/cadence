"""The meaning chamber's world (cadence#334): things the parrot sees as features, a teacher who
says words and asks questions, and the founder's nonce words.

Only the world and the teacher are designed. A thing is a kind, a colour and a place; the parrot
never receives a kind, a colour or a place as such: the eyes report each thing as tuned readings
of its hue, lightness, shape, size, direction and distance (``cadence.meaning.senses``), with the
spread a real instance has (no two balls alike). A word reaches the parrot as the code of its
sound. Kinds share shape dimensions (a ball, an apple, a drum and a clock are round), so an
attribute such as "round" is a direction in what the eyes report, across kinds.

Every word is a nonce word drawn for the founder (``benchmarks/words/world.nonce``, its own
salt), the question words included; the English names serve the reader alone.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadence.meaning.senses import Seen

HERE = Path(__file__).resolve().parent


def _words_world() -> Any:
    """The words chamber's world, loaded under a name of its own (its nonce forms)."""
    name = "words_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE.parent / "words" / "world.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def nonce(seed: int, need: int, *, salt: int) -> list[str]:
    """The words chamber's nonce forms: two or three consonant-vowel syllables."""
    return list(_words_world().nonce(seed, need, salt=salt))

SALT = 334
SHAPE_DIMS = ("round", "long", "hollow", "handle", "flat", "tall")
# kind: (shape prototype over SHAPE_DIMS, width in metres)
KINDS: dict[str, tuple[tuple[float, ...], float]] = {
    "ball": ((.95, .10, .10, .00, .10, .50), .12),
    "cup": ((.45, .20, .90, .90, .20, .50), .09),
    "box": ((.05, .30, .60, .00, .30, .50), .20),
    "book": ((.05, .40, .00, .00, .90, .20), .18),
    "duck": ((.55, .50, .00, .00, .20, .50), .15),
    "key": ((.10, .90, .00, .60, .70, .10), .07),
    "apple": ((.90, .10, .00, .30, .10, .50), .08),
    "lamp": ((.30, .60, .30, .00, .10, .95), .25),
    "kettle": ((.55, .30, .80, .90, .10, .60), .20),
    "drum": ((.85, .20, .70, .00, .30, .40), .30),
    "vase": ((.45, .50, .80, .00, .00, .90), .15),
    "clock": ((.95, .10, .00, .00, .80, .30), .22),
}
KIND_NAMES = tuple(KINDS)
TRAINED = 8  # kinds 0 to 7 are taught; 8 to 11 are new kinds of the later steps
# colour: (hue in degrees or None, saturation, lightness)
COLOURS: dict[str, tuple[float | None, float, float]] = {
    "red": (0.0, .90, .50), "yellow": (60.0, .90, .60), "green": (120.0, .80, .45),
    "blue": (230.0, .85, .45), "white": (None, .05, .95), "black": (None, .05, .08),
}
COLOUR_NAMES = tuple(COLOURS)
PLACE_NAMES = ("front", "right", "behind", "left")
NEAR = 0.9  # "next to you": nearer than this (metres)
ROUND = 0.8  # "round": a kind whose roundness prototype is this or more
QUESTION_WORDS = ("what", "is", "this", "where", "colour", "the", "you", "and", "next", "to")


@dataclass
class Lexicon:
    """A founder's nonce words: one for each kind, colour, place and attribute, and the
    question words."""

    kind: list[str]
    colour: list[str]
    place: list[str]
    attribute: dict[str, str]
    question: dict[str, str]

    def every(self) -> list[str]:
        return (self.kind + self.colour + self.place + list(self.attribute.values())
                + list(self.question.values()))

    def english(self) -> dict[str, str]:
        out = dict(zip(self.kind, KIND_NAMES, strict=True))
        out.update(zip(self.colour, COLOUR_NAMES, strict=True))
        out.update(zip(self.place, PLACE_NAMES, strict=True))
        out.update({w: n for n, w in self.attribute.items()})
        out.update({w: n for n, w in self.question.items()})
        return out


def lexicon(seed: int) -> Lexicon:
    forms = iter(nonce(seed, len(KINDS) + len(COLOURS) + len(PLACE_NAMES) + 1
                       + len(QUESTION_WORDS), salt=SALT))
    take = lambda n: [next(forms) for _ in range(n)]  # noqa: E731
    return Lexicon(kind=take(len(KINDS)), colour=take(len(COLOURS)),
                   place=take(len(PLACE_NAMES)), attribute={"round": take(1)[0]},
                   question=dict(zip(QUESTION_WORDS, take(len(QUESTION_WORDS)), strict=True)))


@dataclass
class Thing:
    kind: int
    colour: int
    place: np.ndarray  # metres on the room's plane, the parrot at the middle
    shape: np.ndarray
    width: float
    hue: float
    saturation: float
    lightness: float
    salient: bool = False


@dataclass
class Room:
    things: list[Thing]
    heading: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def thing(rng: np.random.Generator, kind: int, colour: int, place: np.ndarray) -> Thing:
    """An instance: the kind's shape and width and the colour's hue, saturation and lightness,
    each with the spread of a real instance."""
    proto, width = KINDS[KIND_NAMES[kind]]
    shape = np.clip(np.asarray(proto) + rng.normal(0.0, 0.07, len(proto)), 0.0, 1.0)
    hue_deg, sat, light = COLOURS[COLOUR_NAMES[colour]]
    hue = math.radians(float(rng.uniform(0, 360)) if hue_deg is None
                       else hue_deg + float(rng.normal(0.0, 8.0)))
    return Thing(kind, colour, np.asarray(place, dtype=np.float64), shape,
                 float(width * math.exp(rng.normal(0.0, 0.15))), hue,
                 float(np.clip(sat + rng.normal(0, .04), 0, 1)),
                 float(np.clip(light + rng.normal(0, .04), 0, 1)))


def free_place(rng: np.random.Generator, taken: list[np.ndarray], *,
               near: bool | None = None) -> np.ndarray:
    for _ in range(1000):
        r = rng.uniform(0.6, NEAR - 0.05) if near else (
            rng.uniform(NEAR + 0.25, 1.8) if near is False else rng.uniform(0.6, 1.8))
        a = rng.uniform(-math.pi, math.pi)
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if all(np.linalg.norm(p - q) >= 0.45 for q in taken):
            return p
    raise RuntimeError("no free place in the room")


def room(rng: np.random.Generator, kinds: list[int], count: int, *,
         colours: list[int] | None = None) -> Room:
    """``count`` things of distinct kinds from ``kinds`` in distinct colours (from ``colours``
    where given), at least 0.6 m from the parrot and 0.45 m apart."""
    chosen = [int(k) for k in rng.choice(kinds, size=count, replace=False)]
    palette = list(range(len(COLOURS))) if colours is None else list(colours)
    tints = [int(c) for c in rng.choice(palette, size=count, replace=False)]
    things: list[Thing] = []
    for k, c in zip(chosen, tints, strict=True):
        things.append(thing(rng, k, c, free_place(rng, [t.place for t in things])))
    return Room(things, heading=float(rng.uniform(-math.pi, math.pi)))


def direction(heading: float, point: np.ndarray) -> float:
    """The angle of a point from where the head points (positive to the left)."""
    return float(np.angle(np.exp(1j * (math.atan2(point[1], point[0]) - heading))))


def place_of(heading: float, point: np.ndarray) -> int:
    """0 in front, 1 to the right, 2 behind, 3 to the left: quarters of the turn."""
    a = direction(heading, point)
    return int(((-a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))


def seen(r: Room, rng: np.random.Generator) -> list[Seen]:
    """What the parrot holds of the room (the map holds things out of view as well), in an
    order with no meaning."""
    out = [Seen(hue=t.hue, saturation=t.saturation, lightness=t.lightness, shape=t.shape,
                width=t.width, direction=direction(r.heading, t.place),
                distance=float(np.linalg.norm(t.place)), salient=t.salient) for t in r.things]
    return [out[int(i)] for i in rng.permutation(len(out))]


def is_round(t: Thing) -> bool:
    return KINDS[KIND_NAMES[t.kind]][0][0] >= ROUND


def near(t: Thing) -> bool:
    return float(np.linalg.norm(t.place)) < NEAR
