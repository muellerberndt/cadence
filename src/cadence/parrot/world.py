"""The parrot's world: a table, a low perch, the things placed around it and the parrot's body.

The frame is the page's: metres, y up, the table top at y = 0, the origin under the perch bar's
middle. The parrot faces +z at rest and +x is its left, so an azimuth on the table is
``atan2(x, z)``, positive to the left.

- **The scene** (``Scene``) holds the things: each a kind, its base's place on the table, a yaw,
  a scale and the height of the clear stand it rests on. The eyes draw each kind as a stand-in of
  its real size and colour (``kinds.json``).
- **The body** (``Body``) holds the eleven joints the brain moves: targets in ``[-1, 1]`` over each
  joint's range, a servo that moves each joint toward its target at no more than its rate each
  moment, and the joint readings (``proprioception``). ``Body.eyes`` places the two eyes from the
  head's pose: the head turns about its pivot, and the eyes, 1.2 cm in front of it, move with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np

MOMENT = 0.1  # seconds a moment: the brain takes ten moments a second

# name, low, high, rest, rate a second (in the joint's own unit: degrees, or 0..1)
JOINTS: tuple[tuple[str, float, float, float, float], ...] = (
    ("head_yaw", -75.0, 75.0, 0.0, 360.0),
    ("head_pitch", -25.0, 35.0, 0.0, 240.0),
    ("head_roll", -20.0, 20.0, 0.0, 240.0),
    ("neck_pitch", -20.0, 35.0, 0.0, 120.0),
    ("beak_open", 0.0, 35.0, 0.0, 200.0),
    ("wing_l", 0.0, 1.0, 0.0, 3.0),
    ("wing_r", 0.0, 1.0, 0.0, 3.0),
    ("grip_l", 0.0, 1.0, 1.0, 4.0),
    ("grip_r", 0.0, 1.0, 1.0, 4.0),
    ("leg_l", 0.0, 1.0, 0.3, 2.0),
    ("leg_r", 0.0, 1.0, 0.3, 2.0),
)
NAMES = tuple(j[0] for j in JOINTS)
LOW = np.array([j[1] for j in JOINTS])
HIGH = np.array([j[2] for j in JOINTS])
REST = np.array([j[3] for j in JOINTS])
RATE = np.array([j[4] for j in JOINTS])

PERCH_TOP = 0.13  # the perch bar's top above the table
EYE_APART = 0.036  # metres between the eyes
EYE_FORWARD = 0.012  # the eyes' distance in front of the head's pivot
EYE_AXIS = math.radians(75.0)  # each eye's axis from the beak, to its side
EYE_FIELD = math.radians(87.5)  # each eye sees this far either side of its axis


def stand_ins() -> dict[str, Any]:
    """The kinds' stand-ins, the room and the light (``kinds.json``)."""
    text = resources.files("cadence.parrot").joinpath("kinds.json").read_text(encoding="utf-8")
    table: dict[str, Any] = json.loads(text)
    return table


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Thing:
    """A thing on the table: its ``kind`` (a name in ``kinds.json``), the middle of its base on
    the table ``(x, z)``, its ``yaw`` (radians), ``scale`` and the ``stand`` it rests on (m)."""

    id: str
    kind: str
    x: float
    z: float
    yaw: float = 0.0
    scale: float = 1.0
    stand: float = 0.0

    def half(self, table: dict[str, Any]) -> np.ndarray:
        """The stand-in's half-sizes along its own x, y and z, in metres."""
        return self.scale * np.asarray(table["kinds"][self.kind]["half"], dtype=np.float64)

    def center(self, table: dict[str, Any]) -> np.ndarray:
        """The middle of the stand-in, ``(x, y, z)``."""
        return np.array([self.x, self.stand + self.half(table)[1], self.z])

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "pos": [self.x, 0.0, self.z],
                "yaw": math.degrees(self.yaw), "scale": self.scale, "stand": self.stand}


@dataclass
class Scene:
    """The things around the perch, in the page's scene format (``parrot-scene/1``)."""

    things: list[Thing] = field(default_factory=list)

    @classmethod
    def from_json(cls, record: dict[str, Any] | str) -> Scene:
        data = json.loads(record) if isinstance(record, str) else record
        if data.get("schema") != "parrot-scene/1":
            raise ValueError("a scene is a parrot-scene/1 record")
        table = stand_ins()
        out = []
        for item in data.get("things", []):
            kind = str(item["kind"])
            if kind not in table["kinds"]:
                raise ValueError(f"no stand-in for the kind {kind!r}")
            pos = item["pos"]
            out.append(Thing(str(item["id"]), kind, float(pos[0]), float(pos[2]),
                             math.radians(float(item.get("yaw", 0.0))),
                             float(item.get("scale", 1.0)), float(item.get("stand", 0.0))))
        return cls(out)

    def to_json(self) -> dict[str, Any]:
        return {"schema": "parrot-scene/1",
                "perch": {"bar_y": PERCH_TOP, "length": 0.30, "radius": 0.011},
                "things": [t.to_json() for t in self.things]}


@dataclass
class Eyes:
    """Where the two eyes are and how the head is turned, for one pose: ``places`` ``(2, 3)``
    (left, right), ``rotation`` the head's ``(3, 3)`` rotation, ``pivot`` the head's pivot."""

    places: np.ndarray
    rotation: np.ndarray
    pivot: np.ndarray

    @property
    def middle(self) -> np.ndarray:
        """The point between the two eyes."""
        return np.asarray(self.places.mean(axis=0))


class Body:
    """The parrot's eleven joints (``JOINTS``): targets in, angles out.

    ``angles`` are in each joint's own unit (degrees, or 0 to 1). ``move(targets)`` takes
    ``(11,)`` targets in ``[-1, 1]`` over each range (NaN leaves a joint's target as it was) and
    moves every joint toward its target by at most its rate times ``MOMENT``."""

    def __init__(self) -> None:
        self.angles = REST.copy()
        self.targets = REST.copy()
        self.velocity = np.zeros(len(JOINTS))

    def reset(self) -> None:
        self.angles = REST.copy()
        self.targets = REST.copy()
        self.velocity = np.zeros(len(JOINTS))

    @staticmethod
    def scaled(values: np.ndarray) -> np.ndarray:
        """Joint values in their own units as ``[-1, 1]`` over each range."""
        return np.asarray(2.0 * (np.asarray(values) - LOW) / (HIGH - LOW) - 1.0)

    @staticmethod
    def unscaled(values: np.ndarray) -> np.ndarray:
        """``[-1, 1]`` over each range as the joints' own units."""
        return np.asarray(LOW + (np.clip(np.asarray(values), -1.0, 1.0) + 1.0) / 2.0 * (HIGH - LOW))

    def move(self, targets: np.ndarray | None = None) -> np.ndarray:
        """One moment of the servo; gives the angles."""
        if targets is not None:
            given = np.asarray(targets, dtype=np.float64).reshape(len(JOINTS))
            keep = np.isnan(given)
            self.targets = np.where(keep, self.targets, self.unscaled(np.where(keep, 0.0, given)))
        step = np.clip(self.targets - self.angles, -RATE * MOMENT, RATE * MOMENT)
        before = self.angles
        self.angles = np.clip(self.angles + step, LOW, HIGH)
        self.velocity = (self.angles - before) / MOMENT
        return self.angles.copy()

    def proprioception(self) -> np.ndarray:
        """``(22,)``: the angles in ``[-1, 1]``, then the velocities over each joint's rate."""
        return np.concatenate([self.scaled(self.angles), self.velocity / RATE])

    def joint(self, name: str) -> float:
        return float(self.angles[NAMES.index(name)])

    def set(self, name: str, value: float) -> None:
        """Put a joint at a value (its own unit) at once, its target with it: for the chamber."""
        k = NAMES.index(name)
        self.angles[k] = self.targets[k] = float(np.clip(value, LOW[k], HIGH[k]))

    def eyes(self, angles: np.ndarray | None = None) -> Eyes:
        """The eyes' places and the head's rotation for these angles (the body's own by
        default)."""
        a = self.angles if angles is None else np.asarray(angles, dtype=np.float64)
        crouch = 0.5 * (a[NAMES.index("leg_l")] + a[NAMES.index("leg_r")])
        hip = np.array([0.0, PERCH_TOP + 0.060 - 0.025 * crouch, 0.0])
        base = hip + np.array([0.0, 0.070, 0.0])
        lean = math.radians(a[NAMES.index("neck_pitch")])
        pivot = base + 0.050 * np.array([0.0, math.cos(lean), math.sin(lean)])
        rotation = (rotation_y(math.radians(a[NAMES.index("head_yaw")]))
                    @ rotation_x(-math.radians(a[NAMES.index("head_pitch")]))
                    @ rotation_z(math.radians(a[NAMES.index("head_roll")])))
        local = np.array([[EYE_APART / 2, 0.0, EYE_FORWARD], [-EYE_APART / 2, 0.0, EYE_FORWARD]])
        return Eyes(pivot[None, :] + local @ rotation.T, rotation, pivot)


def azimuth(points: np.ndarray) -> np.ndarray:
    """The azimuth of ``(..., 2)`` table places ``(x, z)`` seen from the origin, + to the left."""
    p = np.asarray(points, dtype=np.float64)
    return np.asarray(np.arctan2(p[..., 0], p[..., 1]))


__all__ = ["EYE_APART", "EYE_AXIS", "EYE_FIELD", "EYE_FORWARD", "HIGH", "JOINTS", "LOW",
           "MOMENT", "NAMES", "PERCH_TOP", "RATE", "REST", "Body", "Eyes", "Scene", "Thing",
           "azimuth", "rotation_x", "rotation_y", "rotation_z", "stand_ins"]
