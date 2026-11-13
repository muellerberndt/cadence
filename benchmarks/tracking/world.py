"""The tracking chamber's world: a plane, an arm, things and a camera with a blind spot.

Everything lies in one plane, in metres and radians, one step every ``dt`` seconds. Plain NumPy:
``Geometry`` holds where everything stands, ``World`` holds what moves, and ``step`` advances the
world by one step and returns what a recorder writes down for it.

- **The arm** is a chain of links from a fixed base. Joint 0's angle is measured from the x axis
  and every further joint's from the link before it, counter-clockwise. A step's command is the
  joint increment asked for; the increment made differs from it by noise in proportion to its
  size, so an arm at rest stays where it is.
- **Things** are points: thing 0 is the hand, and every further thing has a place of its own.
  ``what`` says what each thing looks like; two things of one ``what`` look alike.
- **The camera** stands at a point and sees along a heading within a field of view and a range.
  A thing is in view where it lies in that field and the straight line from the camera to it
  crosses no occluding disc and no link of the arm: the discs' shadows, the arm's own shadow and
  the plane outside the field are the blind spot. In view, the camera reports the thing's
  position with Gaussian noise on each axis and what it looks like; out of view it reports
  nothing.
- **The ear** stands at a point and hears a thing that sounds, in view or out of it: the
  direction to the thing and its distance, each with noise.

Scenes: ``own`` (the hand is the arm's own end, moved by the commands), ``other`` (the hand is a
person's and moves by itself while the arm holds a pose) and ``room`` (several things). A scene is
one recording; ``save`` and ``load`` write and read one scene per file (``.npz`` or ``.json``),
and the chamber scores a loaded scene as it scores a generated one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "cadence-tracking-scene/1"
UNITS = {"length": "m", "time": "s", "angle": "rad"}
CASES = ("own", "other", "room")
KINDS = {
    "own": ("sweep", "stays"),
    "other": ("straight", "slow", "stop", "turn", "stays", "left", "visit", "pause", "hops",
              "slowhand"),
    "room": ("heard", "crossing", "demo"),
}
HAND = 0  # what a hand looks like; every thing that is no hand has a ``what`` of its own


@dataclass(frozen=True)
class Geometry:
    """Where the plane, the camera, the ear, the occluding discs and the arm stand."""

    plane: tuple[float, float, float, float] = (-1.0, 1.0, -0.2, 1.4)  # x and y, low and high
    camera: tuple[float, float] = (0.95, 1.35)
    heading: float = -2.443  # the camera's axis, from the x axis
    fov: float = 2.0944  # the whole angle of the field of view
    reach: float = 2.6  # how far the camera sees
    occluders: tuple[tuple[float, float, float], ...] = ((0.25, 0.85, 0.08),)  # x, y, radius
    ear: tuple[float, float] = (0.95, 1.35)
    base: tuple[float, float] = (0.0, 0.0)
    lengths: tuple[float, ...] = (0.45, 0.35)
    width: float = 0.06  # a link's thickness
    limits: tuple[tuple[float, float], ...] = ((0.35, 2.79), (-2.6, -0.3))

    @property
    def joints(self) -> int:
        return len(self.lengths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane": list(self.plane),
            "camera": {"position": list(self.camera), "heading": self.heading, "fov": self.fov,
                       "range": self.reach},
            "ear": list(self.ear),
            "occluders": [{"center": [x, y], "radius": r} for x, y, r in self.occluders],
            "arm": {"base": list(self.base), "lengths": list(self.lengths), "width": self.width,
                    "limits": [list(pair) for pair in self.limits]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Geometry:
        camera, arm = data["camera"], data["arm"]
        lengths = tuple(float(v) for v in arm["lengths"])
        limits = arm.get("limits") or [[-np.pi, np.pi]] * len(lengths)
        position = (float(camera["position"][0]), float(camera["position"][1]))
        ear = data.get("ear", position)
        x_low, x_high, y_low, y_high = (float(v) for v in data["plane"])
        return cls(
            plane=(x_low, x_high, y_low, y_high), camera=position,
            heading=float(camera.get("heading", 0.0)), fov=float(camera.get("fov", 2 * np.pi)),
            reach=float(camera.get("range", 1e9)),
            occluders=tuple((float(o["center"][0]), float(o["center"][1]), float(o["radius"]))
                            for o in data.get("occluders", [])),
            ear=(float(ear[0]), float(ear[1])),
            base=(float(arm["base"][0]), float(arm["base"][1])), lengths=lengths,
            width=float(arm.get("width", 0.0)),
            limits=tuple((float(lo), float(hi)) for lo, hi in limits),
        )


THREE_JOINTS = Geometry(lengths=(0.35, 0.28, 0.2), limits=((0.35, 2.79), (-2.2, -0.2), (-1.6, 1.6)))

# The lists of starting positions, each in view. ``own``: joint angles of the arm. ``other``:
# points near the plane's edge a person reaches from. The arm's poses while a person's hand moves.
OWN_STARTS: dict[int, tuple[tuple[float, ...], ...]] = {
    2: ((1.2, -1.2), (1.8, -1.1), (0.8, -0.6), (1.5, -2.0), (2.2, -0.7), (1.0, -1.8), (1.4, -0.4),
        (2.6, -0.5)),
    3: ((1.2, -1.0, 0.3), (1.6, -0.9, 0.4), (0.8, -0.5, 0.2), (1.5, -1.8, 0.8), (2.5, -0.8, 0.3),
        (1.0, -1.5, 0.5), (1.9, -1.6, -0.5), (2.0, -0.4, 0.9)),
}
OTHER_STARTS: tuple[tuple[float, float], ...] = (
    (-0.9, 0.35), (-0.9, 0.6), (-0.9, 0.85), (-0.85, 1.05), (-0.65, 1.3), (-0.4, 1.3), (0.1, 1.3),
    (0.4, 1.3), (0.8, 0.9), (0.75, 0.6), (0.8, 0.3), (0.75, -0.1),
)
POSES: dict[int, tuple[tuple[float, ...], ...]] = {
    2: ((1.3, -1.1), (2.0, -0.8), (0.9, -1.6), (1.6, -1.9)),
    3: ((1.3, -1.0, 0.2), (2.0, -0.7, -0.3), (0.9, -1.4, 0.5), (1.6, -1.6, 0.6)),
}


# -- geometry


def forward(geometry: Geometry, joints: np.ndarray) -> np.ndarray:
    """The arm's points for ``(..., joints)`` angles: ``(..., joints + 1, 2)``, the base first
    and the hand last."""
    angles = np.cumsum(np.asarray(joints, dtype=np.float64), axis=-1)
    links = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * np.asarray(geometry.lengths)[
        :, None
    ]
    points = np.cumsum(links, axis=-2) + np.asarray(geometry.base)
    base = np.broadcast_to(np.asarray(geometry.base), points[..., :1, :].shape)
    return np.concatenate([base, points], axis=-2)


def hand(geometry: Geometry, joints: np.ndarray) -> np.ndarray:
    """The position of the arm's end."""
    return np.asarray(forward(geometry, joints)[..., -1, :])


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points ``p`` to the segments ``a`` to ``b`` (all ``(..., 2)``)."""
    ab, ap = b - a, p - a
    along = np.sum(ap * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-12)
    nearest = a + np.clip(along, 0.0, 1.0)[..., None] * ab
    return np.asarray(np.linalg.norm(p - nearest, axis=-1))


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.asarray(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])


def _segments(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance between the segments ``a`` to ``b`` and ``c`` to ``d``: zero where they cross."""
    ends = np.minimum(
        np.minimum(_point_segment(a, c, d), _point_segment(b, c, d)),
        np.minimum(_point_segment(c, a, b), _point_segment(d, a, b)),
    )
    crossing = (np.sign(_cross(b - a, c - a)) != np.sign(_cross(b - a, d - a))) & (
        np.sign(_cross(d - c, a - c)) != np.sign(_cross(d - c, b - c))
    )
    return np.where(crossing, 0.0, ends)


def visible(
    geometry: Geometry, points: np.ndarray, joints: np.ndarray, own: int | None = None
) -> np.ndarray:
    """Whether the camera sees each of ``(..., things, 2)`` points with the arm at
    ``(..., joints)`` angles. Point ``own`` is the arm's own end: its link hides nothing within
    a link's width of it."""
    points = np.asarray(points, dtype=np.float64)
    camera = np.asarray(geometry.camera)
    ray = points - camera
    distance = np.linalg.norm(ray, axis=-1)
    bearing = np.arctan2(ray[..., 1], ray[..., 0]) - geometry.heading
    bearing = np.abs((bearing + np.pi) % (2 * np.pi) - np.pi)
    x_low, x_high, y_low, y_high = geometry.plane
    seen = (distance <= geometry.reach) & (bearing <= geometry.fov / 2)
    seen &= (points[..., 0] >= x_low) & (points[..., 0] <= x_high)
    seen &= (points[..., 1] >= y_low) & (points[..., 1] <= y_high)
    eye = np.broadcast_to(camera, points.shape)
    for x, y, radius in geometry.occluders:
        seen &= _point_segment(np.array([x, y]), eye, points) > radius
    chain = forward(geometry, joints)[..., None, :, :]  # one arm for every thing of a step
    last = geometry.joints - 1
    for k in range(geometry.joints):
        a, b = chain[..., k, :], chain[..., k + 1, :]
        clear = _segments(eye, points, np.broadcast_to(a, points.shape),
                          np.broadcast_to(b, points.shape)) > geometry.width / 2
        if own is not None and k == last:
            # the hand's own link, its length less a width at the hand, which the hand tops
            span = np.maximum(np.linalg.norm(b - a, axis=-1, keepdims=True), 1e-9)
            short = (a + (b - a) * np.clip(1.0 - geometry.width / span, 0.0, 1.0))[..., 0, :]
            clear[..., own] = _segments(eye[..., own, :], points[..., own, :], a[..., 0, :],
                                        short) > geometry.width / 2
        seen &= clear
    return np.asarray(seen, dtype=bool)


def polar(geometry: Geometry, points: np.ndarray) -> np.ndarray:
    """The direction and distance from the ear to each of ``(..., 2)`` points."""
    offset = np.asarray(points, dtype=np.float64) - np.asarray(geometry.ear)
    return np.stack([np.arctan2(offset[..., 1], offset[..., 0]),
                     np.linalg.norm(offset, axis=-1)], axis=-1)


def cartesian(geometry: Geometry, cue: np.ndarray) -> np.ndarray:
    """The point a ``(direction, distance)`` cue names."""
    cue = np.asarray(cue, dtype=np.float64)
    return np.asarray(geometry.ear) + cue[..., 1:2] * np.stack(
        [np.cos(cue[..., 0]), np.sin(cue[..., 0])], axis=-1
    )


# -- the world and its step


@dataclass
class World:
    """What moves: the arm's true joint angles and the things' true positions."""

    geometry: Geometry
    joints: np.ndarray  # (joints,)
    things: np.ndarray  # (things, 2); thing 0 is the hand
    own: bool = False  # the hand is the arm's own end
    camera_noise: float = 0.01  # metres on each axis of a report
    motor_noise: float = 0.03  # the share of a command the increment made differs by
    encoder_noise: float = 0.001  # radians on each joint reading
    cue_noise: tuple[float, float] = (0.04, 0.08)  # radians of direction, share of distance


def draws(rng: np.random.Generator, world: World) -> dict[str, np.ndarray]:
    """One step's standard normal draws, in a fixed order: motor, encoder, camera, cue."""
    j, k = len(world.joints), len(world.things)
    return {"motor": rng.standard_normal(j), "encoder": rng.standard_normal(j),
            "camera": rng.standard_normal((k, 2)), "cue": rng.standard_normal((k, 2))}


def step(
    world: World, command: np.ndarray, moves: np.ndarray, sounds: np.ndarray,
    noise: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Advance ``world`` by one step and return its record.

    ``command`` is the joint increment asked of the arm, ``moves`` each thing's own displacement
    (``(things, 2)``; the arm's own hand takes none) and ``sounds`` which things sound at this
    step. ``noise`` holds the step's standard normal draws (``draws``). The record: ``truth``,
    ``in_view``, ``report`` (NaN while out of view), ``cue`` (direction and distance from the
    ear, NaN while silent), ``joints`` (the readings) and ``commands``."""
    geometry = world.geometry
    world.joints = world.joints + command * (1.0 + world.motor_noise * noise["motor"])
    world.things = world.things + moves
    if world.own:
        world.things[0] = hand(geometry, world.joints)
    seen = visible(geometry, world.things, world.joints, 0 if world.own else None)
    report = world.things + world.camera_noise * noise["camera"]
    heard = polar(geometry, world.things)
    heard[:, 0] += world.cue_noise[0] * noise["cue"][:, 0]
    heard[:, 1] *= 1.0 + world.cue_noise[1] * noise["cue"][:, 1]
    return {
        "truth": world.things.copy(), "in_view": seen,
        "report": np.where(seen[:, None], report, np.nan),
        "cue": np.where(np.asarray(sounds, dtype=bool)[:, None], heard, np.nan),
        "joints": world.joints + world.encoder_noise * noise["encoder"],
        "commands": np.asarray(command, dtype=np.float64).copy(),
    }


@dataclass
class Scene:
    """One recording of ``steps`` steps and ``things`` things. ``report`` holds NaN where a
    thing is out of view and ``cue`` NaN where it is silent; ``commands[t]`` is the joint
    increment asked for between step ``t - 1`` and step ``t`` (zero at step 0), and ``joints[t]``
    and ``truth[t]`` are read at step ``t``. ``start`` is the list of starting positions, which
    every learner is given."""

    case: str
    report: np.ndarray  # (steps, things, 2)
    in_view: np.ndarray  # (steps, things) bool
    commands: np.ndarray  # (steps, joints)
    joints: np.ndarray  # (steps, joints)
    truth: np.ndarray  # (steps, things, 2)
    start: np.ndarray | None = None  # (things, 2); the truth at step 0 where omitted
    cue: np.ndarray | None = None  # (steps, things, 2) direction and distance; silent if omitted
    what: np.ndarray | None = None  # (things,) what each thing looks like; hands if omitted
    geometry: Geometry = field(default_factory=Geometry)
    dt: float = 1.0 / 30.0
    kind: str = ""
    event: int = -1  # the step at which the hand's movement changes, -1 for none
    camera_noise: float = float("nan")  # the report's standard deviation on each axis
    name: str = ""

    def __post_init__(self) -> None:
        self.truth = np.asarray(self.truth, dtype=np.float64)
        self.report = np.asarray(self.report, dtype=np.float64)
        self.in_view = np.asarray(self.in_view, dtype=bool)
        if self.truth.ndim == 2:  # one thing, the hand: rows of (x, y)
            self.truth = self.truth[:, None, :]
            self.report = self.report.reshape(len(self.truth), 1, 2)
            self.in_view = self.in_view.reshape(len(self.truth), 1)
        steps, things = self.truth.shape[:2]
        self.commands = np.asarray(self.commands, dtype=np.float64)
        self.joints = np.asarray(self.joints, dtype=np.float64)
        self.start = (self.truth[0].copy() if self.start is None
                      else np.asarray(self.start, dtype=np.float64).reshape(-1, 2))
        self.cue = (np.full((steps, things, 2), np.nan) if self.cue is None
                    else np.asarray(self.cue, dtype=np.float64).reshape(steps, -1, 2))
        self.what = (np.full(things, HAND, dtype=np.int64) if self.what is None
                     else np.asarray(self.what, dtype=np.int64).reshape(-1))
        if self.case not in CASES:
            raise ValueError(f"case must be one of {CASES}, got {self.case!r}")
        if steps == 0 or things == 0 or self.truth.shape != (steps, things, 2):
            raise ValueError("truth must hold one (x, y) row per step and thing")
        if self.report.shape != self.truth.shape or self.in_view.shape != (steps, things):
            raise ValueError("report and in_view must match truth's steps and things")
        if self.cue.shape != self.truth.shape or self.what.shape != (things,):
            raise ValueError("cue and what must match truth's steps and things")
        if self.commands.ndim != 2 or self.commands.shape != self.joints.shape or len(
            self.commands
        ) != steps:
            raise ValueError("commands and joints must hold one row of joint values per step")
        if self.start.shape != (things, 2) or not np.isfinite(self.start).all():
            raise ValueError("start must be one finite (x, y) position per thing")
        if not (np.isfinite(self.truth).all() and np.isfinite(self.commands).all()
                and np.isfinite(self.joints).all()):
            raise ValueError("truth, commands and joints must be finite on every step")
        if not np.isfinite(self.report[self.in_view]).all():
            raise ValueError("the report must be finite wherever a thing is in view")
        self.report = np.where(self.in_view[..., None], self.report, np.nan)

    @property
    def steps(self) -> int:
        return int(self.truth.shape[0])

    @property
    def things(self) -> int:
        return int(self.truth.shape[1])

    def unseen(self) -> np.ndarray:
        """Steps since each thing was last seen, ``(steps, things)``: 0 in view, 1 on the first
        step out of view. Step 0 counts as seen: the starting positions are given."""
        out = np.zeros((self.steps, self.things), dtype=np.int64)
        for t in range(1, self.steps):
            out[t] = np.where(self.in_view[t], 0, out[t - 1] + 1)
        return out


def record(
    world: World, commands: np.ndarray, moves: np.ndarray, sounds: np.ndarray,
    rng: np.random.Generator, *, case: str, what: np.ndarray | None = None, kind: str = "",
    event: int = -1, dt: float = 1.0 / 30.0, stepwise: bool = False,
) -> Scene:
    """Run ``world`` through a script and keep every step's record. Step 0 is the world as it
    stands (its command and moves are zero); ``commands[t]``, ``moves[t]`` and ``sounds[t]``
    lead to step ``t``. The draws are ``draws`` step by step; the arithmetic is ``step``'s over
    all steps at once, or with ``stepwise`` one ``step`` call a step (the same scene)."""
    commands, moves = commands.copy(), moves.copy()
    commands[0], moves[0] = 0.0, 0.0
    start = world.things.copy()
    if world.own:
        start[0] = hand(world.geometry, world.joints)
    noise = [draws(rng, world) for _ in range(len(commands))]
    if stepwise:
        rows = [step(world, commands[t], moves[t], sounds[t], noise[t])
                for t in range(len(commands))]
        held = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    else:
        held = _steps(world, commands, moves, sounds,
                      {name: np.stack([n[name] for n in noise]) for name in noise[0]})
    return Scene(case, held["report"], held["in_view"], held["commands"], held["joints"],
                 held["truth"], start, held["cue"], what, world.geometry, dt, kind, event,
                 world.camera_noise)


def _steps(
    world: World, commands: np.ndarray, moves: np.ndarray, sounds: np.ndarray,
    noise: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """``step`` over every step of a script at once: the same records, stacked."""
    geometry = world.geometry
    joints = world.joints + np.cumsum(commands * (1.0 + world.motor_noise * noise["motor"]), 0)
    things = world.things + np.cumsum(moves, axis=0)
    if world.own:
        things[:, 0] = hand(geometry, joints)
    world.joints, world.things = joints[-1].copy(), things[-1].copy()
    seen = visible(geometry, things, joints, 0 if world.own else None)
    report = things + world.camera_noise * noise["camera"]
    heard = polar(geometry, things)
    heard[..., 0] += world.cue_noise[0] * noise["cue"][..., 0]
    heard[..., 1] *= 1.0 + world.cue_noise[1] * noise["cue"][..., 1]
    return {
        "truth": things, "in_view": seen, "report": np.where(seen[..., None], report, np.nan),
        "cue": np.where(np.asarray(sounds, dtype=bool)[..., None], heard, np.nan),
        "joints": joints + world.encoder_noise * noise["encoder"], "commands": commands.copy(),
    }


# -- scripts: what the arm and the things do


def _smooth(fraction: np.ndarray | float) -> np.ndarray:
    """The minimum-jerk profile from 0 to 1."""
    f = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    return np.asarray(10 * f**3 - 15 * f**4 + 6 * f**5)


def _bounds(geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([lo for lo, _ in geometry.limits]),
            np.array([hi for _, hi in geometry.limits]))


def _path_commands(steps: int, plan: list[tuple[int, int, np.ndarray]], origin: np.ndarray
                   ) -> np.ndarray:
    """Joint increments of smooth moves: ``(first step, last step, target)`` in order."""
    commands = np.zeros((steps, len(origin)))
    at = np.asarray(origin, dtype=np.float64)
    for begin, end, target in plan:
        if begin >= steps - 1:
            break
        span = np.arange(begin, min(end, steps - 1) + 1)
        path = at + _smooth((span - begin) / (end - begin))[:, None] * (target - at)
        commands[span[1:]] = np.diff(path, axis=0)
        at = path[-1]
    return commands


def sweep(rng: np.random.Generator, geometry: Geometry, origin: np.ndarray, steps: int
          ) -> np.ndarray:
    """The arm's commands for moves between joint targets drawn within the limits, each within
    1.2 radians of the pose before it, over 25 to 60 steps, with rests of up to 25 steps."""
    low, high = _bounds(geometry)
    plan, t, at = [], int(rng.integers(0, 10)), np.asarray(origin, dtype=np.float64)
    while t < steps:
        length = int(rng.integers(25, 61))
        target = np.clip(at + rng.uniform(-1.2, 1.2, len(at)), low, high)
        plan.append((t, t + length, target))
        t += length + int(rng.integers(0, 26))
        at = target
    return _path_commands(steps, plan, origin)


def reach(
    rng: np.random.Generator, geometry: Geometry, start: np.ndarray, steps: int, kind: str,
    dt: float,
) -> tuple[np.ndarray, int]:
    """A person's hand: its displacement at every step and the step its movement changes.

    The hand leaves ``start`` toward a point of the plane's middle at 0.25 to 0.6 metres a
    second. At a step drawn between 15% and 70% of the scene a ``slow`` hand eases to 25% to 50%
    of its speed over 8 steps, a ``stop`` hand eases to rest over 8 steps and a ``turn`` hand
    turns by 40 to 110 degrees to either side over 10 steps; a ``straight`` hand keeps its line.
    A hand rests where its next step would leave the plane."""
    x_low, x_high, y_low, y_high = geometry.plane
    middle = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2])
    spread = np.array([x_high - x_low, y_high - y_low]) * 0.22
    aim = middle + rng.uniform(-1.0, 1.0, 2) * spread
    heading = float(np.arctan2(aim[1] - start[1], aim[0] - start[0]))
    speed = float(rng.uniform(0.25, 0.6)) * dt  # metres a step
    drawn = int(rng.integers(int(0.15 * steps), int(0.7 * steps)))
    factor = float(rng.uniform(0.25, 0.5))
    turn = float(rng.uniform(np.deg2rad(40), np.deg2rad(110))) * (1 if rng.random() < 0.5 else -1)
    event = -1 if kind == "straight" else drawn
    moves = np.zeros((steps, 2))
    at, margin, resting = np.asarray(start, dtype=np.float64).copy(), 0.02, False
    for t in range(1, steps):
        v, h = speed, heading
        if event >= 0 and t > event:
            if kind == "slow":
                v = speed * (1.0 - (1.0 - factor) * float(_smooth((t - event) / 8.0)))
            elif kind == "stop":
                v = speed * (1.0 - float(_smooth((t - event) / 8.0)))
            else:
                h = heading + turn * float(_smooth((t - event) / 10.0))
        move = v * np.array([np.cos(h), np.sin(h)])
        nxt = at + move
        inside = (x_low + margin <= nxt[0] <= x_high - margin) and (
            y_low + margin <= nxt[1] <= y_high - margin
        )
        resting = resting or not inside
        if not resting:
            moves[t], at = move, nxt
    return moves, event


ERRANDS = ("visit", "pause", "hops", "slowhand")


def errand(
    rng: np.random.Generator, geometry: Geometry, start: np.ndarray, pose: np.ndarray,
    steps: int, kind: str, dt: float,
) -> tuple[np.ndarray, int]:
    """What a visitor's or a worker's hand does: its displacement at every step and the step
    its movement first changes (-1 where the draw does not fit the view, to be drawn again).

    - ``visit``: a straight reach goes out of view, rests from a step drawn 2 to 20 steps after
      it is first hidden, stays for 30 to 120 steps and comes back along its line.
    - ``pause``: a straight reach rests for 12 to 40 steps from a step drawn in its first half,
      then goes on.
    - ``hops``: the hand moves for 4 to 10 steps and rests for 4 to 12, again and again.
    - ``slowhand``: a straight reach at 0.10 to 0.15 metres a second."""
    moves, _ = reach(rng, geometry, start, steps, "straight", dt)
    if kind == "slowhand":
        speed = float(np.linalg.norm(moves[1]))
        return moves * (float(rng.uniform(0.10, 0.15)) * dt / max(speed, 1e-12)), 0
    if kind == "pause":
        event = int(rng.integers(8, max(steps // 2, 9)))
        held = int(rng.integers(12, 41))
        out = np.zeros_like(moves)
        out[1 : event + 1] = moves[1 : event + 1]
        rest = moves[event + 1 : steps - held]
        out[event + 1 + held : event + 1 + held + len(rest)] = rest
        return out, event
    if kind == "hops":
        out, t, source = np.zeros_like(moves), 1, 1
        while t < steps and source < steps:
            go = int(rng.integers(4, 11))
            out[t : t + go] = moves[source : source + go][: max(steps - t, 0)]
            source += go
            t += go + int(rng.integers(4, 13))
        return out, 0
    path = start + np.cumsum(moves, axis=0)
    hidden = np.flatnonzero(~visible(geometry, path[:, None, :], np.tile(pose, (steps, 1)))[:, 0])
    hidden = hidden[hidden >= 10]
    if not len(hidden) or hidden[0] >= steps - 60:
        return moves, -1
    event = int(hidden[0]) + int(rng.integers(2, 21))
    back = event + 1 + int(rng.integers(30, 121))
    out = np.zeros_like(moves)
    out[1 : event + 1] = moves[1 : event + 1]
    span = min(event, max(steps - back, 0))
    out[back : back + span] = -moves[event - span + 1 : event + 1][::-1]
    return out, event


def _hidden_pose(rng: np.random.Generator, geometry: Geometry) -> np.ndarray:
    """Joint angles within the limits whose hand the camera does not see."""
    low, high = _bounds(geometry)
    for _ in range(10_000):
        pose = rng.uniform(low + 0.1, high - 0.1)
        if not visible(geometry, hand(geometry, pose)[None], pose, 0)[0]:
            return np.asarray(pose)
    raise ValueError("no pose of this arm hides its hand from this camera")


def blind_point(rng: np.random.Generator, geometry: Geometry, pose: np.ndarray,
                shadow: bool = True) -> np.ndarray:
    """A point of the plane the camera does not see with the arm at ``pose``; with ``shadow``
    one inside the camera's field, in the shadow of a disc or of the arm."""
    x_low, x_high, y_low, y_high = geometry.plane
    camera = np.asarray(geometry.camera)
    for _ in range(100_000):
        point = np.array([rng.uniform(x_low + 0.1, x_high - 0.1),
                          rng.uniform(y_low + 0.1, y_high - 0.1)])
        if visible(geometry, point[None], pose)[0]:
            continue
        ray = point - camera
        bearing = np.arctan2(ray[1], ray[0]) - geometry.heading
        inside = abs((bearing + np.pi) % (2 * np.pi) - np.pi) <= geometry.fov / 2 - 0.05
        if inside or not shadow:
            return point
    raise ValueError("this camera sees the whole plane")


def seen_point(rng: np.random.Generator, geometry: Geometry, pose: np.ndarray) -> np.ndarray:
    """A point of the plane the camera sees with the arm at ``pose``."""
    x_low, x_high, y_low, y_high = geometry.plane
    for _ in range(100_000):
        point = np.array([rng.uniform(x_low + 0.1, x_high - 0.1),
                          rng.uniform(y_low + 0.1, y_high - 0.1)])
        if visible(geometry, point[None], pose)[0]:
            return point
    raise ValueError("this camera sees nothing of the plane")


# -- scenes


def own_scene(
    rng: np.random.Generator, geometry: Geometry | None = None, starts: Any = None, *,
    steps: int = 150, kind: str = "sweep", hold: int = 1000, noise: float = 0.01,
) -> Scene:
    """The arm moves its own hand from one of ``starts`` (joint angles). ``sweep``: moves between
    joint targets with rests. ``stays``: one move into the blind spot, then ``hold`` steps at
    rest there."""
    geometry = geometry or Geometry()
    starts = starts or OWN_STARTS[geometry.joints]
    if kind not in KINDS["own"]:
        raise ValueError(f"an own-hand scene is one of {KINDS['own']}, got {kind!r}")
    origin = np.array(starts[int(rng.integers(len(starts)))], dtype=np.float64)
    if kind == "stays":
        steps = 60 + hold
        commands = _path_commands(steps, [(10, 50, _hidden_pose(rng, geometry))], origin)
    else:
        commands = sweep(rng, geometry, origin, steps)
    world = World(geometry, origin, hand(geometry, origin)[None].copy(), own=True,
                  camera_noise=noise)
    return record(world, commands, np.zeros((steps, 1, 2)), np.zeros((steps, 1), dtype=bool), rng,
                  case="own", kind=kind)


def other_scene(
    rng: np.random.Generator, geometry: Geometry | None = None, starts: Any = None,
    poses: Any = None, *, steps: int = 150, kind: str | None = None, hold: int = 1000,
    noise: float = 0.01, dt: float = 1.0 / 30.0,
) -> Scene:
    """A person's hand leaves one of ``starts`` while the arm holds one of ``poses`` (``reach``
    says how it moves; ``kind`` is drawn where omitted). ``stays``: a thing rests in view for 30
    steps, the arm then moves until its shadow holds the thing, and the thing rests there unseen
    for ``hold`` steps. ``left``: a hand on a straight reach rests from a step drawn 2 to 20
    steps after it first goes out of view, and stays there."""
    geometry = geometry or Geometry()
    starts = starts or OTHER_STARTS
    poses = poses or POSES[geometry.joints]
    kind = kind or str(rng.choice(KINDS["other"][:4]))
    if kind not in KINDS["other"]:
        raise ValueError(f"a person's-hand scene is one of {KINDS['other']}, got {kind!r}")
    pose = np.array(poses[int(rng.integers(len(poses)))], dtype=np.float64)
    if kind in ERRANDS:
        for _ in range(10_000):
            start = np.array(starts[int(rng.integers(len(starts)))], dtype=np.float64)
            moves, event = errand(rng, geometry, start, pose, steps, kind, dt)
            if event >= 0:
                world = World(geometry, pose, start[None].copy(), camera_noise=noise)
                return record(world, np.zeros((steps, geometry.joints)), moves[:, None, :],
                              np.zeros((steps, 1), dtype=bool), rng, case="other", kind=kind,
                              event=event, dt=dt)
        raise ValueError(f"no {kind} hand from these starts fits this camera's view")
    if kind == "left":
        # a hand that goes out of view and is left there: it rests from a step drawn 2 to 20
        # steps after it is first hidden (10 or more steps into the scene)
        for _ in range(10_000):
            start = np.array(starts[int(rng.integers(len(starts)))], dtype=np.float64)
            moves, _ = reach(rng, geometry, start, steps, "straight", dt)
            path = start + np.cumsum(moves, axis=0)
            hidden = np.flatnonzero(~visible(geometry, path[:, None, :],
                                             np.tile(pose, (steps, 1)))[:, 0])
            hidden = hidden[hidden >= 10]
            if len(hidden) and hidden[0] < steps - 30:
                event = int(hidden[0]) + int(rng.integers(2, 21))
                moves[event + 1 :] = 0.0
                world = World(geometry, pose, start[None].copy(), camera_noise=noise)
                return record(world, np.zeros((steps, geometry.joints)), moves[:, None, :],
                              np.zeros((steps, 1), dtype=bool), rng, case="other", kind=kind,
                              event=event, dt=dt)
        raise ValueError("no reach from these starts goes out of this camera's view")
    if kind == "stays":
        steps = 60 + hold
        second, point = _shading(rng, geometry, pose)
        commands = _path_commands(steps, [(30, 50, second)], pose)
        world = World(geometry, pose, point[None].copy(), camera_noise=noise)
        return record(world, commands, np.zeros((steps, 1, 2)), np.zeros((steps, 1), dtype=bool),
                      rng, case="other", kind=kind, dt=dt)
    start = np.array(starts[int(rng.integers(len(starts)))], dtype=np.float64)
    moves, event = reach(rng, geometry, start, steps, kind, dt)
    world = World(geometry, pose, start[None].copy(), camera_noise=noise)
    return record(world, np.zeros((steps, geometry.joints)), moves[:, None, :],
                  np.zeros((steps, 1), dtype=bool), rng, case="other", kind=kind, event=event,
                  dt=dt)


def _shading(rng: np.random.Generator, geometry: Geometry, first: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """A pose and a point: the camera sees the point with the arm at ``first`` and the arm at
    the pose shades it."""
    low, high = _bounds(geometry)
    camera = np.asarray(geometry.camera)
    for _ in range(100_000):
        second = rng.uniform(low + 0.1, high - 0.1)
        chain = forward(geometry, second)
        k = int(rng.integers(geometry.joints))
        on_link = chain[k] + rng.uniform(0.3, 0.8) * (chain[k + 1] - chain[k])
        away = (on_link - camera) / np.linalg.norm(on_link - camera)
        point = on_link + rng.uniform(0.08, 0.25) * away
        if visible(geometry, point[None], first)[0] and not visible(
            geometry, point[None], second
        )[0]:
            return np.asarray(second), np.asarray(point)
    raise ValueError("no pose of this arm shades a point this camera sees")


def room_scene(
    rng: np.random.Generator, geometry: Geometry | None = None, *, kind: str = "heard",
    steps: int = 150, noise: float = 0.01, dt: float = 1.0 / 30.0,
) -> Scene:
    """Several things, the arm at rest.

    ``heard``: thing 0 rests in the blind spot (its starting position is given), then, from a
    step drawn in the scene's second quarter, moves to another point of the blind spot over 15
    to 25 steps without coming into view, sounding at
    every step it moves and at the 3 steps after it; two further things rest in view and sound
    at times. ``crossing``: two hands that look alike cross inside the blind spot, both out
    of view, among six things at rest that each look like nothing else. ``demo``: a person's
    hand that reaches and comes back, two things at rest, and one that rests, is heard moving
    out of view and rests again."""
    geometry = geometry or Geometry()
    if kind not in KINDS["room"]:
        raise ValueError(f"a room scene is one of {KINDS['room']}, got {kind!r}")
    pose = np.array(POSES[geometry.joints][int(rng.integers(len(POSES[geometry.joints])))])
    if kind == "crossing":
        return _crossing(rng, geometry, pose, steps, noise, dt)
    count = 3 if kind == "heard" else 4
    things = np.zeros((count, 2))
    moves = np.zeros((steps, count, 2))
    sounds = np.zeros((steps, count), dtype=bool)
    what = np.arange(1, count + 1)
    mover = 0 if kind == "heard" else 3
    for _ in range(100_000):
        origin, target = (blind_point(rng, geometry, pose) for _ in range(2))
        hop = np.linalg.norm(target - origin)
        way = origin + np.linspace(0.0, 1.0, 12)[:, None] * (target - origin)
        if 0.2 <= hop <= 0.7 and not visible(geometry, way, pose).any():
            break
    else:
        raise ValueError("the blind spot holds no move of 0.2 to 0.7 metres")
    if steps < 60:
        raise ValueError("a room scene of this kind takes 60 steps or more")
    begin = int(rng.integers(int(0.25 * steps), int(0.5 * steps) + 1))
    length = int(rng.integers(15, 26))
    span = np.arange(begin, begin + length + 1)
    path = origin + _smooth((span - begin) / length)[:, None] * (target - origin)
    moves[span[1:], mover] = np.diff(path, axis=0)
    sounds[span[1] : span[-1] + 4, mover] = True
    things[mover] = origin
    event = begin
    for k in range(count):
        if k == mover:
            continue
        things[k] = seen_point(rng, geometry, pose)
        sounds[:, k] = rng.random(steps) < 0.3
    if kind == "demo":
        what[0] = HAND
        start = np.array(OTHER_STARTS[int(rng.integers(len(OTHER_STARTS)))])
        out, _ = reach(rng, geometry, start, steps // 2, "stop", dt)
        moves[: steps // 2, 0] = out
        moves[steps // 2 + 1 : steps // 2 + 1 + len(out) - 1, 0] = -out[:0:-1]
        things[0] = start
        sounds[:, 0] = False
    world = World(geometry, pose, things, camera_noise=noise)
    return record(world, np.zeros((steps, geometry.joints)), moves, sounds, rng, case="room",
                  what=what, kind=kind, event=event, dt=dt)


def _crossing(rng: np.random.Generator, geometry: Geometry, pose: np.ndarray, steps: int,
              noise: float, dt: float) -> Scene:
    x_low, x_high, y_low, y_high = geometry.plane
    for _ in range(100_000):
        meet = blind_point(rng, geometry, pose)
        at = int(rng.integers(int(0.3 * steps), int(0.6 * steps)))
        first = float(rng.uniform(-np.pi, np.pi))
        second = first + float(rng.uniform(np.deg2rad(40), np.deg2rad(140))) * (
            1 if rng.random() < 0.5 else -1
        )
        speeds = rng.uniform(0.25, 0.6, 2) * dt
        velocity = speeds[:, None] * np.array([[np.cos(first), np.sin(first)],
                                               [np.cos(second), np.sin(second)]])
        offsets = rng.integers(-3, 4, 2)  # each hand reaches the meeting point near step ``at``
        starts = meet - velocity * (at + offsets)[:, None]
        ends = starts + velocity * (steps - 1)
        inside = ((starts[:, 0] > x_low) & (starts[:, 0] < x_high) & (starts[:, 1] > y_low)
                  & (starts[:, 1] < y_high) & (ends[:, 0] > x_low) & (ends[:, 0] < x_high)
                  & (ends[:, 1] > y_low) & (ends[:, 1] < y_high))
        if not inside.all() or not visible(geometry, starts, pose).all():
            continue
        paths = starts[None] + np.arange(steps)[:, None, None] * velocity[None]
        near = slice(at - 4, at + 5)
        hidden = not visible(geometry, paths[near], np.tile(pose, (9, 1))).any()
        if hidden and visible(geometry, paths[-1], pose).all():
            break
    else:
        raise ValueError("no two lines cross out of view in this plane")
    rest = np.array([seen_point(rng, geometry, pose) if k < 4 else blind_point(rng, geometry, pose)
                     for k in range(6)])
    things = np.vstack([starts, rest])
    moves = np.zeros((steps, 8, 2))
    moves[1:, :2] = velocity[None]
    what = np.array([HAND, HAND, 1, 2, 3, 4, 5, 6])
    world = World(geometry, pose, things, camera_noise=noise)
    return record(world, np.zeros((steps, geometry.joints)), moves,
                  np.zeros((steps, 8), dtype=bool), rng, case="room", what=what, kind="crossing",
                  event=at, dt=dt)


def panel(scene: Scene) -> str:
    """Where a person's-hand scene's change of movement falls: ``straight`` for a straight scene
    with 8 or more steps in a row out of view, ``slow``, ``stop`` or ``turn`` where the change
    begins out of view and its 8 steps stay there, and ``""`` for every other scene. A scene
    whose hand comes to rest at the plane's edge while out of view is in no panel but ``stop``:
    that rest is a stop no kind names."""
    if scene.case != "other" or scene.kind not in ("straight", "slow", "stop", "turn"):
        return ""
    out = ~scene.in_view[:, 0]
    if scene.kind != "stop":
        still = np.flatnonzero(~np.diff(scene.truth[:, 0], axis=0).any(axis=1)) + 1
        if len(still) and out[still[0]]:
            return ""
    if scene.kind == "straight":
        run = longest = 0
        for flag in out[1:]:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return "straight" if longest >= 8 else ""
    e = scene.event
    return scene.kind if e >= 1 and e + 9 <= scene.steps and bool(out[e : e + 9].all()) else ""


def varied(rng: np.random.Generator, geometry: Geometry | None = None) -> Geometry:
    """``geometry`` with other occluding discs: one to three of them anywhere in the plane, a
    quarter of a metre or more from the camera, with radii of 0.05 to 0.15 metres."""
    from dataclasses import replace

    geometry = geometry or Geometry()
    x_low, x_high, y_low, y_high = geometry.plane
    discs: list[tuple[float, float, float]] = []
    while len(discs) < int(rng.integers(1, 4)):
        x, y = float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))
        if np.hypot(x - geometry.camera[0], y - geometry.camera[1]) >= 0.25:
            discs.append((x, y, float(rng.uniform(0.05, 0.15))))
    return replace(geometry, occluders=tuple(discs))


def darkened(scene: Scene, rng: np.random.Generator) -> Scene:
    """``scene`` with the camera off for a span of 20 to 100 steps: no report of any thing
    there. The span begins at step 10 or later."""
    from dataclasses import replace

    begin = int(rng.integers(10, max(scene.steps - 20, 11)))
    off = np.zeros(scene.steps, dtype=bool)
    off[begin : begin + int(rng.integers(20, 101))] = True
    return replace(scene, in_view=scene.in_view & ~off[:, None], report=scene.report.copy())


def scenes(
    case: str, seed: int, count: int = 0, *, geometry: Geometry | None = None,
    kind: str | None = None, quota: dict[str, int] | None = None, steps: int = 150,
    noise: float = 0.01, hold: int = 1000, vary: bool = False,
) -> list[Scene]:
    """``count`` scenes of ``case`` from one seed. ``quota`` (person's-hand scenes) asks for
    that many scenes of each panel in place of ``count``: scenes are drawn until every panel
    holds its number. With ``vary`` every scene has occluding discs of its own (``varied``; a
    scene whose hand starts out of view is drawn again) and one scene in three has the camera
    off for a span (``darkened``)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), CASES.index(case)]))
    out: list[Scene] = []
    if quota:
        held = {name: 0 for name in quota}
        while any(held[name] < quota[name] for name in quota):
            scene = other_scene(rng, geometry, steps=steps, noise=noise)
            name = panel(scene)
            if name in held and held[name] < quota[name]:
                held[name] += 1
                out.append(scene)
        return out

    def one(place: Geometry | None) -> Scene:
        if case == "own":
            return own_scene(rng, place, steps=steps, kind=kind or "sweep", hold=hold,
                             noise=noise)
        if case == "other":
            return other_scene(rng, place, steps=steps, kind=kind, hold=hold, noise=noise)
        return room_scene(rng, place, kind=kind or "heard", steps=steps, noise=noise)

    while len(out) < count:
        if not vary:
            out.append(one(geometry))
            continue
        scene = one(varied(rng, geometry))
        if scene.in_view[0, 0]:
            out.append(darkened(scene, rng) if rng.random() < 1 / 3 else scene)
    return out


# -- files

_ARRAYS = ("report", "in_view", "commands", "joints", "truth", "start", "cue", "what")


def to_dict(scene: Scene) -> dict[str, Any]:
    """The scene as a file holds it. A scene of one thing writes its hand's rows plainly:
    ``report`` and ``truth`` as ``(steps, 2)``, ``in_view`` as ``(steps,)``, ``start`` as
    ``(2,)``; a report out of view and a silent cue are ``null``."""
    assert scene.start is not None and scene.cue is not None and scene.what is not None
    one = scene.things == 1

    def rows(values: np.ndarray, present: np.ndarray) -> list[Any]:
        table = [[[float(v) for v in cell] if ok else None for cell, ok in zip(row, flags,
                                                                             strict=True)]
                 for row, flags in zip(values, present, strict=True)]
        return [row[0] for row in table] if one else table

    out: dict[str, Any] = {
        "format": FORMAT, "units": dict(UNITS), "dt": scene.dt, "case": scene.case,
        "kind": scene.kind, "event": scene.event, "name": scene.name,
        "geometry": scene.geometry.to_dict(),
        "start": scene.start[0].tolist() if one else scene.start.tolist(),
        "report": rows(scene.report, scene.in_view),
        "in_view": scene.in_view[:, 0].tolist() if one else scene.in_view.tolist(),
        "commands": scene.commands.tolist(), "joints": scene.joints.tolist(),
        "truth": scene.truth[:, 0].tolist() if one else scene.truth.tolist(),
    }
    sounding = np.isfinite(scene.cue).all(axis=-1)
    if sounding.any():
        out["cue"] = rows(scene.cue, sounding)
    if not one or scene.what[0] != HAND:
        out["what"] = scene.what.tolist()
    if np.isfinite(scene.camera_noise):
        out["camera_noise"] = scene.camera_noise
    return out


def _filled(rows: Any, steps: int, one: bool) -> np.ndarray:
    """Rows with ``null`` cells as an array with NaN in their place: ``(steps, things, 2)``. An
    array is taken as it is; ``one`` says a row is the hand's own cell."""
    if isinstance(rows, np.ndarray):
        return np.asarray(rows, dtype=np.float64).reshape(steps, -1, 2)
    table = [[row] for row in rows] if one else [list(row) for row in rows]
    return np.array([[[np.nan, np.nan] if cell is None else cell for cell in row]
                     for row in table], dtype=np.float64).reshape(steps, -1, 2)


def from_dict(data: dict[str, Any]) -> Scene:
    if data.get("format") != FORMAT:
        raise ValueError(f"a scene file states format {FORMAT!r}, got {data.get('format')!r}")
    if dict(data.get("units", UNITS)) != UNITS:
        raise ValueError(f"a scene is in {UNITS}, got {dict(data['units'])}")
    for name in ("case", "report", "in_view", "truth"):
        if name not in data:
            raise ValueError(f"a scene file holds {name!r}")
    truth = np.asarray(data["truth"], dtype=np.float64)
    steps, one = len(truth), truth.ndim == 2
    in_view = np.asarray(data["in_view"], dtype=bool).reshape(steps, -1)
    cue = data.get("cue")
    geometry = Geometry.from_dict(data["geometry"]) if "geometry" in data else Geometry()
    still = np.zeros((steps, geometry.joints))  # a scene without an arm's record: an arm at rest
    return Scene(
        case=str(data["case"]), report=_filled(data["report"], steps, one), in_view=in_view,
        commands=np.asarray(data.get("commands", still), dtype=np.float64),
        joints=np.asarray(data.get("joints", still), dtype=np.float64),
        truth=truth.reshape(steps, -1, 2), start=data.get("start"),
        cue=None if cue is None else _filled(cue, steps, one), what=data.get("what"),
        geometry=geometry, dt=float(data.get("dt", 1.0 / 30.0)),
        kind=str(data.get("kind", "")), event=int(data.get("event", -1)),
        camera_noise=float(data.get("camera_noise", np.nan)), name=str(data.get("name", "")),
    )


def save(scene: Scene, path: str | Path) -> None:
    """Write one scene: ``.json`` as text, any other suffix as a NumPy ``.npz`` whose arrays
    are the per-step fields (NaN where a ``.json`` holds ``null``) and whose ``meta`` entry is
    the JSON of every other field."""
    path = Path(path)
    data = to_dict(scene)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=1) + "\n")
        return
    one = scene.things == 1
    arrays = {}
    for name in _ARRAYS:
        if name not in data and name in ("cue", "what"):
            continue
        value = np.asarray(getattr(scene, name))
        arrays[name] = (value[:, 0] if one and name in ("report", "in_view", "truth", "cue")
                        else value[0] if one and name == "start" else value)
    meta = {k: v for k, v in data.items() if k not in _ARRAYS}
    with path.open("wb") as handle:
        np.savez_compressed(handle, meta=np.array(json.dumps(meta)), **arrays)


def load(path: str | Path) -> Scene:
    """Read one scene written by ``save`` or by a recorder that follows the README's format."""
    path = Path(path)
    if path.suffix == ".json":
        return from_dict(json.loads(path.read_text()))
    with np.load(path, allow_pickle=False) as held:
        data = json.loads(str(held["meta"]))
        for name in held.files:
            if name != "meta":
                data[name] = held[name]
    return from_dict(data)
