"""The symbols chamber's world: things a parrot sees, and the words a trainer says.

**Things.** ``KINDS`` kinds, each a shape (three readings) and a surface (one reading), and two
colours a kind comes in (a hue). A thing seen is its kind, its colour and its place around the
body (a direction and a distance), each a population of tuned cells (``cadence.senses.Things``).

**Words.** Every kind's name, every colour's name and two questions ("what colour is this",
"what is this") are utterances: a word is a fixed pattern of log-mel energies (40 bands, 30 to
60 frames: a few blobs of energy in time and frequency, drawn once from the word's own seed),
and a take of it is that pattern said by a speaker: faster or slower, a band higher or lower,
louder or softer, with the room's noise. The parrot's ear hears a take as one code
(``cadence.senses.Ear``). The trainer knows what was said by the nearest word to a take.

Everything here is the world and the trainer. Nothing in it is part of the brain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cadence.senses import Ear, Things

BANDS = 40
KINDS = {  # shape (round, long, flat) and surface (soft to hard), and the two colours
    "cup": ((0.6, 0.3, 0.2), 0.9, ("white", "blue")),
    "apple": ((1.0, 0.1, 0.1), 0.3, ("red", "green")),
    "ball": ((0.95, 0.05, 0.05), 0.5, ("red", "blue")),
    "banana": ((0.2, 1.0, 0.2), 0.2, ("yellow", "green")),
    "block": ((0.1, 0.4, 0.5), 0.8, ("yellow", "white")),
    "duck": ((0.5, 0.6, 0.3), 0.6, ("yellow", "white")),
    "key": ((0.05, 0.8, 0.9), 1.0, ("yellow", "blue")),
    "pear": ((0.8, 0.5, 0.1), 0.35, ("green", "yellow")),
}
COLOURS = {"red": 0.0, "yellow": 0.17, "green": 0.33, "blue": 0.62, "white": 0.85}
QUESTIONS = ("what colour is this", "what is this")
WORDS = (*KINDS, *COLOURS, *QUESTIONS)


def things_sense() -> Things:
    """The eyes: tuned cells for a thing's hue, shape, surface, direction and distance."""
    return Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3, "surface": (0.0, 1.0),
                   "direction": (-1.0, 1.0), "distance": (0.0, 1.0)}, per=8)


@dataclass(frozen=True)
class Thing:
    kind: str
    colour: str
    direction: float
    distance: float

    def readings(self) -> dict[str, list]:
        shape, surface, _ = KINDS[self.kind]
        return {"hue": [COLOURS[self.colour]], "shape": [list(shape)], "surface": [surface],
                "direction": [self.direction], "distance": [self.distance]}


def a_thing(rng: np.random.Generator, kind: str | None = None, colour: str | None = None,
            ) -> Thing:
    """A thing at a place drawn around the body; its kind and colour drawn where not given."""
    kind = str(rng.choice(list(KINDS))) if kind is None else kind
    colour = str(rng.choice(KINDS[kind][2])) if colour is None else colour
    return Thing(kind, colour, float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.1, 0.9)))


class Words:
    """The words of the world as utterances, and a trainer's ear for what a take says."""

    def __init__(self, ear: Ear, seed: int = 0) -> None:
        self.ear = ear
        self.patterns: dict[str, np.ndarray] = {}
        for k, word in enumerate(WORDS):
            rng = np.random.default_rng((seed, 7, k))
            frames = int(rng.integers(30, 61))
            t, b = np.arange(frames)[:, None], np.arange(BANDS)[None, :]
            pattern = np.full((frames, BANDS), -4.0)
            for _ in range(int(rng.integers(3, 6))):
                at, band = rng.uniform(0, frames), rng.uniform(4, BANDS - 4)
                width_t, width_b = rng.uniform(3, 10), rng.uniform(2, 6)
                pattern += rng.uniform(2.0, 5.0) * np.exp(
                    -0.5 * ((t - at) / width_t) ** 2 - 0.5 * ((b - band) / width_b) ** 2)
            self.patterns[word] = pattern
        self.templates = np.concatenate([ear.code(p) for p in self.patterns.values()])

    def take(self, word: str, rng: np.random.Generator, noise: float = 0.3) -> np.ndarray:
        """One take of ``word`` by a speaker, as the parrot's ear codes it ``(1, width)``."""
        pattern = self.patterns[word]
        frames = max(8, int(round(len(pattern) * rng.uniform(0.8, 1.25))))
        at = np.linspace(0, len(pattern) - 1, frames)
        low = np.floor(at).astype(int)
        high = np.minimum(low + 1, len(pattern) - 1)
        w = (at - low)[:, None]
        said = pattern[low] * (1 - w) + pattern[high] * w
        said = np.roll(said, int(rng.integers(-1, 2)), axis=1) + rng.uniform(-1.0, 1.0)
        said = said + noise * rng.normal(size=said.shape)
        return self.ear.code(said)

    def heard(self, code: np.ndarray) -> str | None:
        """What a take (or a sound the parrot says) is heard as: the nearest word; None for
        silence."""
        code = np.asarray(code, dtype=np.float64).reshape(-1)
        if not np.any(code):
            return None
        return WORDS[int(np.argmax(self.templates @ code))]
