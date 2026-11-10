"""The pixel chamber (cadence#284): a screen-dependent policy on rendered frames.

Every frame is rendered here from a scene, so the tests need no game data. A scene holds one
task object and up to two distractors on a lit background. The task object is one of three
kinds, told apart by shape alone:

- ``food``, a filled disc: approach it (the action toward its side);
- ``hazard``, a diagonal cross: avoid it (the action away from its side);
- ``target``, a ring: aim at it (the action toward it, ``FIRE`` once it is in the centre).

Three actions, ``LEFT`` (0), ``RIGHT`` (1) and ``FIRE`` (2). A scene draws one of eight
situations with equal chance: food left or right, hazard left or right, target left or right,
and target centre twice, so ``LEFT`` and ``RIGHT`` each take three eighths of the frames and
``FIRE`` a quarter. Food and the hazard on the same side take opposite actions, so a policy
must read the object's kind and its place.

Where and how the object appears is drawn per frame. Each factor has a trained range and a
held-out range that no training frame shows:

| Factor | Trained | Held out |
| --- | --- | --- |
| vertical place ``v`` (share of the height) | 0.20 to 0.38 and 0.62 to 0.80 | 0.45 to 0.55 |
| scale (of a radius of 0.11 of the shorter side) | 0.75 to 0.90 and 1.10 to 1.25 | 0.95 to 1.05 |
| lighting (background, object) | 0.05 to 0.25, 0.75 to 1.00 | 0.35 to 0.45, 0.55 to 0.65 |
| distractors | a horizontal bar, a filled square, in half the frames | a triangle, in every frame |

The horizontal place ``u`` is drawn within the side's zone (left 0.14 to 0.30, centre 0.44 to
0.56, right 0.70 to 0.86) in training and on every panel. Distractors are smaller (0.6 of the
radius at a scale drawn from the trained range) and never overlap the object. In colour every
shape takes a hue from six; in grey a frame holds the luminance alone. Every pixel carries
Gaussian noise of 0.03, clipped to [0, 1].

The panels, each a fixed draw per founder: ``held_out`` (fresh frames at the trained ranges),
``positions``, ``scales``, ``lighting`` and ``distractor`` (one factor at its held-out range,
the others trained). Invariance on a factor is its panel's accuracy within .05 of
``held_out``'s.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

LEFT, RIGHT, FIRE = 0, 1, 2
ACTIONS = 3
KINDS = ("food", "hazard", "target")
SITUATIONS = (
    ("food", "left"), ("food", "right"),
    ("hazard", "left"), ("hazard", "right"),
    ("target", "left"), ("target", "right"),
    ("target", "centre"), ("target", "centre"),
)
ZONES = {"left": (0.14, 0.30), "centre": (0.44, 0.56), "right": (0.70, 0.86)}
V_TRAINED = ((0.20, 0.38), (0.62, 0.80))
V_HELD = ((0.45, 0.55),)
SCALE_TRAINED = ((0.75, 0.90), (1.10, 1.25))
SCALE_HELD = ((0.95, 1.05),)
LIGHT_TRAINED = ((0.05, 0.25), (0.75, 1.00))
LIGHT_HELD = ((0.35, 0.45), (0.55, 0.65))
DISTRACTORS_TRAINED = ("bar", "square")
DISTRACTORS_HELD = ("triangle",)
RADIUS = 0.11  # of the shorter side, at scale 1
DISTRACTOR_RADIUS = 0.6  # of the object's radius at scale 1
NOISE = 0.03
HUES = np.array(
    [[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.3, 0.45, 1.0],
     [1.0, 1.0, 0.25], [0.25, 1.0, 1.0], [1.0, 0.3, 1.0]]
)
PANELS = ("held_out", "positions", "scales", "lighting", "distractor")
INVARIANCE = ("positions", "scales", "distractor")  # gated within .05 of held_out
SHAPES = {"grey84": (84, 84, 1), "colour84": (84, 84, 3), "grey160x120": (120, 160, 1)}
SEEDS = {"train": 28401, "panel": 28402, "noise": 28403}


@dataclass(frozen=True)
class Conditions:
    """The ranges a scene is drawn from: each factor trained or held out."""

    v: tuple[tuple[float, float], ...] = V_TRAINED
    scale: tuple[tuple[float, float], ...] = SCALE_TRAINED
    background: tuple[float, float] = LIGHT_TRAINED[0]
    luminance: tuple[float, float] = LIGHT_TRAINED[1]
    distractors: tuple[str, ...] = DISTRACTORS_TRAINED
    distractor_share: float = 0.5  # the share of frames with distractors
    blank: bool = False  # the blank-screen control: the background alone


TRAINED = Conditions()
PANEL_CONDITIONS = {
    "held_out": TRAINED,
    "positions": replace(TRAINED, v=V_HELD),
    "scales": replace(TRAINED, scale=SCALE_HELD),
    "lighting": replace(TRAINED, background=LIGHT_HELD[0], luminance=LIGHT_HELD[1]),
    "distractor": replace(TRAINED, distractors=DISTRACTORS_HELD, distractor_share=1.0),
}


@dataclass(frozen=True)
class Shape:
    kind: str
    u: float
    v: float
    scale: float
    luminance: float
    hue: int


@dataclass(frozen=True)
class Scene:
    situation: int  # index into SITUATIONS
    background: float
    item: Shape
    distractors: tuple[Shape, ...] = ()
    blank: bool = False

    @property
    def label(self) -> int:
        return label_of(*SITUATIONS[self.situation])


def label_of(kind: str, zone: str) -> int:
    """The rewarded action: approach food, avoid the hazard, aim at the target."""
    if zone == "centre":
        if kind != "target":
            raise ValueError("only the target appears in the centre")
        return FIRE
    toward = LEFT if zone == "left" else RIGHT
    away = RIGHT if zone == "left" else LEFT
    return away if kind == "hazard" else toward


def _uniform(rng: np.random.Generator, ranges: tuple[tuple[float, float], ...]) -> float:
    widths = np.array([b - a for a, b in ranges])
    pick = int(rng.choice(len(ranges), p=widths / widths.sum()))
    low, high = ranges[pick]
    return float(rng.uniform(low, high))


def _clear(u: float, v: float, radius: float, placed: list[tuple[float, float, float]],
           aspect: float) -> bool:
    """No overlap with any placed shape, with a margin of 0.03 of the height."""
    return all(np.hypot((u - pu) * aspect, v - pv) > radius + pr + 0.03 for pu, pv, pr in placed)


def draw_scene(rng: np.random.Generator, conditions: Conditions = TRAINED,
               aspect: float = 1.0) -> Scene:
    """One scene: the situation, the object's place, scale and lighting, and its distractors.
    ``aspect`` is width over height; radii are measured in shares of the height."""
    situation = int(rng.integers(len(SITUATIONS)))
    kind, zone = SITUATIONS[situation]
    background = float(rng.uniform(*conditions.background))
    item = Shape(kind, float(rng.uniform(*ZONES[zone])), _uniform(rng, conditions.v),
                 _uniform(rng, conditions.scale), float(rng.uniform(*conditions.luminance)),
                 int(rng.integers(len(HUES))))
    side = min(1.0, aspect)  # the shorter side in shares of the height
    placed = [(item.u, item.v, RADIUS * side * item.scale)]
    distractors: list[Shape] = []
    if conditions.distractor_share > 0 and rng.random() < conditions.distractor_share:
        count = 1 if conditions.distractor_share >= 1.0 else 1 + int(rng.integers(2))
        for _ in range(count):
            for _attempt in range(64):
                scale = _uniform(rng, SCALE_TRAINED)
                u, v = float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.15, 0.85))
                radius = RADIUS * side * DISTRACTOR_RADIUS * scale
                if _clear(u, v, radius, placed, aspect):
                    placed.append((u, v, radius))
                    distractors.append(Shape(
                        str(rng.choice(conditions.distractors)), u, v, scale,
                        float(rng.uniform(*conditions.luminance)) * float(rng.uniform(0.8, 1.0)),
                        int(rng.integers(len(HUES)))))
                    break
    return Scene(situation, background, item, tuple(distractors), conditions.blank)


def _mask(kind: str, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """The shape on coordinates in units of its radius (``dy`` grows downward)."""
    d2 = dx * dx + dy * dy
    if kind == "food":
        return d2 <= 1.0
    if kind == "target":
        return (d2 <= 1.0) & (d2 >= 0.55**2)
    if kind == "hazard":
        inside = (np.abs(dx) <= 0.85) & (np.abs(dy) <= 0.85)
        return inside & ((np.abs(dx - dy) <= 0.36) | (np.abs(dx + dy) <= 0.36))
    if kind == "bar":
        return (np.abs(dy) <= 0.3) & (np.abs(dx) <= 1.0)
    if kind == "square":
        return (np.abs(dx) <= 0.75) & (np.abs(dy) <= 0.75)
    if kind == "triangle":
        return (dy >= -0.85) & (dy <= 0.85) & (np.abs(dx) <= (dy + 0.85) / 1.7)
    raise ValueError(f"unknown shape {kind!r}")


def render(scene: Scene, shape: tuple[int, int, int], noise: np.random.Generator | None,
           ) -> np.ndarray:
    """The frame of a scene, ``(height, width, channels)`` in [0, 1]."""
    height, width, channels = shape
    frame = np.full((height, width, channels), scene.background)
    if not scene.blank:
        side = min(height, width)
        ys, xs = np.mgrid[0:height, 0:width].astype(float) + 0.5
        shapes = [(scene.item, 1.0), *((d, DISTRACTOR_RADIUS) for d in scene.distractors)]
        for item, size in shapes:
            radius = RADIUS * side * item.scale * size
            mask = _mask(item.kind, (xs - item.u * width) / radius,
                         (ys - item.v * height) / radius)
            if channels == 1:
                colour = np.array([item.luminance])
            else:
                colour = item.luminance * HUES[item.hue][:channels]
            frame[mask] = colour
    if noise is not None:
        frame = frame + noise.normal(0.0, NOISE, size=frame.shape)
    return np.clip(frame, 0.0, 1.0)


def founder_seed(kind: str, founder: int, salt: str = "") -> np.random.SeedSequence:
    digest = hashlib.sha256(f"{kind}:{founder}:{salt}".encode()).digest()
    return np.random.SeedSequence(
        [SEEDS[kind], int(founder), int.from_bytes(digest[:4], "little")]
    )


def frames(scenes: list[Scene], shape: tuple[int, int, int], noise: np.random.Generator | None,
           ) -> np.ndarray:
    """Rendered frames, ``(rows, height, width, channels)``."""
    return np.stack([render(scene, shape, noise) for scene in scenes])


def panel(founder: int, name: str, shape: tuple[int, int, int], size: int,
          ) -> tuple[np.ndarray, np.ndarray]:
    """The panel's frames and labels, a fixed draw per founder, panel and frame shape."""
    rng = np.random.default_rng(founder_seed("panel", founder, f"{name}:{shape}"))
    aspect = shape[1] / shape[0]
    scenes = [draw_scene(rng, PANEL_CONDITIONS[name], aspect) for _ in range(size)]
    return frames(scenes, shape, rng), np.array([s.label for s in scenes])


@dataclass
class Schedule:
    """The training stream of a life: batches of fresh frames at the trained ranges.
    ``control`` names a control's change to what the life sees: ``blank`` (the background
    alone, labels as drawn), ``shuffled`` (every frame's label the label of an independently
    drawn scene: the labels keep their shares and carry nothing of the frames). A permutation
    within a batch would keep a frame's own label among the batch's, so each frame would
    meet its own label about one time in eight more often than chance."""

    founder: int
    shape: tuple[int, int, int]
    rows: int = 8
    control: str | None = None
    rng: np.random.Generator = field(init=False)
    shuffle: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.control not in (None, "blank", "shuffled"):
            raise ValueError(f"unknown control {self.control!r}")
        self.rng = np.random.default_rng(founder_seed("train", self.founder))
        self.shuffle = np.random.default_rng(founder_seed("train", self.founder, "shuffle"))

    def batches(self, count: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        conditions = replace(TRAINED, blank=self.control == "blank")
        aspect = self.shape[1] / self.shape[0]
        for _ in range(count):
            scenes = [draw_scene(self.rng, conditions, aspect) for _ in range(self.rows)]
            labels = np.array([s.label for s in scenes])
            if self.control == "shuffled":  # the same frames, labels of other scenes
                labels = np.array([draw_scene(self.shuffle, conditions, aspect).label
                                   for _ in range(self.rows)])
            yield frames(scenes, self.shape, self.rng), labels


def flat(batch: np.ndarray) -> np.ndarray:
    """Frames as rows, ``(rows, height * width * channels)``, in the image's own order."""
    return batch.reshape(len(batch), -1)


def unit(batch: np.ndarray) -> np.ndarray:
    """Rows at unit length."""
    rows = flat(batch)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def pooled(batch: np.ndarray, size: int) -> np.ndarray:
    """The coarse fixed code: the mean of every ``size`` by ``size`` block of every channel."""
    rows, height, width, channels = batch.shape
    h, w = height // size, width // size
    cut = batch[:, : h * size, : w * size]
    return cut.reshape(rows, h, size, w, size, channels).mean(axis=(2, 4)).reshape(rows, -1)


def accuracy(answers: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(answers) == np.asarray(labels)))


def summary(readings: dict[str, float]) -> dict[str, Any]:
    """Invariance: every gated panel within .05 of ``held_out``."""
    gaps = {name: readings["held_out"] - readings[name] for name in INVARIANCE if name in readings}
    return {"gaps": gaps, "invariant": all(gap <= 0.05 for gap in gaps.values())}
