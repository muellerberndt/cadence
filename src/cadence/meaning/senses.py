"""The parrot's born senses for learning meaning (cadence#334): what the eyes report of a thing,
where it is around the body, and the code of a heard word.

Nothing here names a concept. A thing is reported as readings spread over tuned inputs (a
population code, as hue, orientation, place and head-direction cells tile their spaces):

- **hue**, a circle of tuned inputs, each reading weighted by the thing's saturation;
- **lightness**, a line from dark to light;
- **shape**, one line per shape dimension the eyes measure (how round, how long, how hollow, ...);
- **size**, a line over the logarithm of the thing's width;
- **direction**, a circle over the angle of the thing from where the head points;
- **distance**, a line over the logarithm of the distance;
- **salience**, one input: the thing moves or is held up (born salience, as the tectum's).

A scene is a fixed number of such slots, in an order with no meaning (the order the map lists its
things). A heard word reaches the brain as the code of its sound (``Forms``): a sparse code drawn
per form, a stand-in for the ear's compressed code of the word (#329), never a concept. Which
concept a word names, and which concepts there are, the brain learns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Circle:
    """``per`` inputs whose preferred angles tile the circle; an angle lights each input by
    ``exp(-0.5 (d / width)^2)``, ``d`` the wrapped distance to its preferred angle and ``width``
    the spacing times ``spread``."""

    def __init__(self, per: int, spread: float = 1.0):
        if int(per) < 3:
            raise ValueError("a circle takes three tuned inputs or more")
        self.per, self.spread = int(per), float(spread)
        self.centres = np.arange(self.per) * (2.0 * math.pi / self.per)

    def __call__(self, angle: float, weight: float = 1.0) -> np.ndarray:
        d = np.asarray(np.angle(np.exp(1j * (float(angle) - self.centres))), dtype=np.float64)
        width = self.spread * 2.0 * math.pi / self.per
        return np.asarray(float(weight) * np.exp(-0.5 * (d / width) ** 2), dtype=np.float64)


class Line:
    """``per`` inputs whose preferred values tile ``[low, high]``; a value lights each by a
    Gaussian of the spacing times ``spread`` (values outside are clipped to the ends)."""

    def __init__(self, per: int, low: float, high: float, spread: float = 1.0):
        if int(per) < 2 or not float(high) > float(low):
            raise ValueError("a line takes two tuned inputs or more over low < high")
        self.per, self.low, self.high = int(per), float(low), float(high)
        self.spread = float(spread)
        self.centres = np.linspace(self.low, self.high, self.per)

    def __call__(self, value: float) -> np.ndarray:
        v = min(max(float(value), self.low), self.high)
        width = self.spread * (self.high - self.low) / (self.per - 1)
        return np.exp(-0.5 * ((v - self.centres) / width) ** 2)


@dataclass
class Seen:
    """What the eyes report of one thing: its hue (radians), saturation and lightness (0 to 1),
    its shape (one value in 0 to 1 per shape dimension), its width (metres), its direction from
    where the head points (radians, positive to the left) and its distance (metres), and whether
    it stands out."""

    hue: float
    saturation: float
    lightness: float
    shape: np.ndarray
    width: float
    direction: float
    distance: float
    salient: bool = False


@dataclass
class Eyes:
    """The born sense of things: the tuned inputs of one slot and of a scene of ``slots``."""

    slots: int = 5
    hues: int = 12
    lights: int = 5
    shapes: int = 6  # shape dimensions the eyes measure
    per_shape: int = 5
    sizes: int = 4
    directions: int = 12
    distances: int = 4
    _hue: Circle = field(init=False, repr=False)
    _light: Line = field(init=False, repr=False)
    _shape: Line = field(init=False, repr=False)
    _size: Line = field(init=False, repr=False)
    _direction: Circle = field(init=False, repr=False)
    _distance: Line = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hue = Circle(self.hues)
        self._light = Line(self.lights, 0.0, 1.0)
        self._shape = Line(self.per_shape, 0.0, 1.0)
        self._size = Line(self.sizes, math.log(0.05), math.log(0.5))
        self._direction = Circle(self.directions)
        self._distance = Line(self.distances, math.log(0.4), math.log(2.0))

    @property
    def parts(self) -> dict[str, int]:
        """Each sense's inputs in one slot, in the slot's order."""
        return {"hue": self.hues, "lightness": self.lights, "shape": self.shapes * self.per_shape,
                "size": self.sizes, "direction": self.directions, "distance": self.distances,
                "salience": 1}

    @property
    def slot_width(self) -> int:
        return int(sum(self.parts.values()))

    @property
    def width(self) -> int:
        return self.slots * self.slot_width

    def slot(self, seen: Seen | None) -> np.ndarray:
        """One thing's tuned inputs (zeros for an empty slot)."""
        if seen is None:
            return np.zeros(self.slot_width)
        shape = np.asarray(seen.shape, dtype=np.float64).reshape(self.shapes)
        return np.concatenate([
            self._hue(seen.hue, seen.saturation),
            self._light(seen.lightness),
            np.concatenate([self._shape(v) for v in shape]),
            self._size(math.log(max(seen.width, 1e-3))),
            self._direction(seen.direction),
            self._distance(math.log(max(seen.distance, 1e-3))),
            [1.0 if seen.salient else 0.0],
        ])

    def scene(self, things: list[Seen]) -> np.ndarray:
        """The scene's inputs: each thing in a slot of its own, in the order given, empty slots
        after; a scene holds at most ``slots`` things."""
        if len(things) > self.slots:
            raise ValueError(f"a scene holds at most {self.slots} things")
        return np.concatenate([self.slot(t) for t in things]
                              + [self.slot(None)] * (self.slots - len(things)))


class Forms:
    """Heard forms: each word form a sparse code of ``width`` inputs with ``active`` of them on,
    drawn per founder when the form is first heard (a stand-in for the ear's code of its sound;
    the same form always gives the same code). The code says nothing of what the word names."""

    def __init__(self, width: int = 64, active: int = 5, seed: int = 0):
        if not 0 < int(active) <= int(width):
            raise ValueError("0 < active <= width")
        self.width, self.active = int(width), int(active)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 334, 1]))
        self.codes: dict[str, np.ndarray] = {}

    def __call__(self, form: str | None) -> np.ndarray:
        if form is None:
            return np.zeros(self.width)
        if form not in self.codes:
            code = np.zeros(self.width)
            code[self.rng.choice(self.width, size=self.active, replace=False)] = 1.0
            self.codes[form] = code / math.sqrt(self.active)
        return self.codes[form].copy()

    def heard_as(self, code: np.ndarray) -> str | None:
        """The form whose code is nearest ``code`` (the ear taking a sound for a word it
        knows), or None for silence."""
        if not np.any(code) or not self.codes:
            return None
        forms = list(self.codes)
        scores = np.array([float(self.codes[f] @ code) for f in forms])
        return forms[int(np.argmax(scores))]


def observation(eyes: Eyes, forms: Forms, things: list[Seen], heard: str | None) -> np.ndarray:
    """One moment's observation: the scene, then the code of the word heard (zeros: silence)."""
    return np.concatenate([eyes.scene(things), forms(heard)])


def width(eyes: Eyes, forms: Forms) -> int:
    return eyes.width + forms.width


def describe(eyes: Eyes, forms: Forms) -> dict[str, Any]:
    return {"slots": eyes.slots, "slot": eyes.parts, "forms": {"width": forms.width,
                                                                "active": forms.active}}
