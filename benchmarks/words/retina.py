"""The words chamber seen through the parrot's two eyes (cadence#330).

The camera's report ("a ball at this place, red") is replaced by what the parrot's eyes see:
each eye's image of the room, a row of receptors across its field (``cadence.words.retina``).
The world draws those images; the brain finds the things in them, learns what they are and
what colour from what it sees, and places them on its map from each eye's ray and the head's
pose. Nothing tells the brain what or where a thing is.

**What a thing looks like.** The room is seen across, so a thing is its outline on the floor:
a ``disc``, an ``oval`` (its short axis, 0.3 of its width, toward the parrot), a ``board`` (a
thin flat face toward the parrot) or a ``dish`` (half a shallow ring, 0.15 of its width deep,
open toward the parrot), in one of three sizes. Each of the twelve kinds is one outline in one
size (``LOOKS``); the four new kinds are outlines and sizes the trained kinds have, in
combinations none of them has. A thing faces the room's middle (where the parrot sits) turned
by a few degrees (``TURN``), and is drawn in its colour, lit from the parrot's side: a surface
is brighter the more squarely its ray meets it. The walls are a warm grey.

**The rooms.** The life is the words chamber's script (``words.script``), with the rooms drawn
so that from the parrot's perch no thing hides any part of another (``clear``): a parrot that
cannot move from its perch would never see a hidden thing whole. A thing's turn is drawn for the
founder from its place. The parrot sees with two eyes (``cadence.words.retina.parrot_eyes``);
the camera of the reference arm reports every thing whose middle lies in either eye's field.

**Arms.** ``eyes``: the brain with the retina (``make_brain``); ``camera``: the words chamber's
brain fed by the camera's reports over the same fields (the reference). Both live the same
script and are asked the same questions.

**Readings.** R1 (things found: each thing wholly in an eye's field at a moment, found within
0.15 m; seen well enough to be recognised; figures that are no thing), R2 (each thing's place
on the map from the eyes against its place on the map from the camera, in view and out of view,
at each test room's first question), R3 (held-out views: a thing at a distance and a turn no
room holds, named by kind and colour), R4 (the words chamber's W1 to W6 for both arms).
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import words  # noqa: E402
import world  # noqa: E402

OUTLINES = ("disc", "oval", "board", "dish")
SIZES = (0.12, 0.19, 0.30)  # metres across
# each kind's outline and size: kinds 0 to 7 trained, 8 to 11 new
LOOKS = ((0, 0), (1, 1), (2, 2), (3, 0), (0, 1), (1, 2), (2, 0), (3, 1),
         (0, 2), (1, 0), (2, 1), (3, 2))
TURN = math.radians(15.0)  # a thing in a room is turned this far at most from facing the parrot
# R3's views: distances no room holds (nearer than any, and a band every room leaves out) and
# turns farther than any room's
HELD_OUT = {"near": (0.45, 0.6), "band": (1.25, 1.35), "turn": (math.radians(20.0),
                                                              math.radians(30.0))}
RGB = np.array([(0.85, 0.12, 0.10), (0.12, 0.25, 0.85), (0.10, 0.62, 0.18),
                (0.92, 0.82, 0.12), (0.93, 0.93, 0.93), (0.10, 0.10, 0.10)])
WALL = np.array((0.55, 0.52, 0.48))
NOISE = {"depth": 0.001, "colour": 0.01}  # receptor noise: a share of the distance, and absolute
GAP = math.radians(1.5)  # the least angle between two things' outlines seen from the perch
FIELD = math.radians(150.0)  # either eye sees this far either side of the beak, together
FOUND = 0.15  # a figure is a thing's where its place lies this near the thing's
MARGIN = 0.10  # R2: the map from the eyes holds a place this near the map from the camera's
LOOK = math.radians(3.0)  # R3: a thing looked at lies this near an eye's middle
LOOKS_AT = 6  # after looking around, the head turns to look at this many unknown things at most


# -- the look of a thing


def turn_of(seed: int, place: np.ndarray, held_out: bool = False) -> float:
    """A thing's turn from facing the parrot, drawn for the founder from its place."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), 330, 1, int(round(place[0] * 1e6)) % 2**31, int(round(place[1] * 1e6))
         % 2**31]))
    if held_out:
        low, high = HELD_OUT["turn"]
        return float(rng.choice([-1.0, 1.0]) * rng.uniform(low, high))
    return float(rng.uniform(-TURN, TURN))


def outline(kind: int, place: np.ndarray, turn: float) -> np.ndarray:
    """A thing's outline on the floor: ``(corners, 2)``, closed (the last corner joins the
    first). It faces the room's middle, turned by ``turn``."""
    shape, size = OUTLINES[LOOKS[kind][0]], SIZES[LOOKS[kind][1]]
    place = np.asarray(place, dtype=np.float64)
    toward = math.atan2(-place[1], -place[0]) + turn
    u = np.array([math.cos(toward), math.sin(toward)])  # toward the parrot
    v = np.array([-u[1], u[0]])
    half = size / 2
    if shape == "disc":
        a = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        local = np.stack([half * np.cos(a), half * np.sin(a)], axis=1)
    elif shape == "oval":
        a = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        local = np.stack([0.3 * size * np.cos(a), half * np.sin(a)], axis=1)
    elif shape == "board":
        local = np.array([[0.015, -half], [0.015, half], [-0.015, half], [-0.015, -half]])
    else:  # a dish: half a shallow ring, open toward the parrot
        t = 0.015
        a = np.linspace(math.pi / 2, 3 * math.pi / 2, 19)
        outer = np.stack([(0.15 * size + t) * np.cos(a), half * np.sin(a)], axis=1)
        inner = np.stack([0.15 * size * np.cos(a[::-1]), (half - t) * np.sin(a[::-1])], axis=1)
        local = np.concatenate([outer, inner])
    return np.asarray(place[None, :] + local[:, :1] * u[None, :] + local[:, 1:] * v[None, :])


def outlines(seed: int, r: world.Room) -> list[np.ndarray]:
    """Every thing's outline in a room."""
    return [outline(t.kind, t.place, float(r.extra.get("turns", {}).get(i, turn_of(seed, t.place))))
            for i, t in enumerate(r.things)]


def _span(points: np.ndarray, origin: np.ndarray, axis: float) -> tuple[float, float]:
    """The angles an outline covers seen from ``origin``, about ``axis`` (an outline never
    holds the origin)."""
    d = np.asarray(points) - origin
    a = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - axis)))
    return float(a.min()), float(a.max())


def clear(seed: int, r: world.Room) -> bool:
    """Whether no thing hides any part of another from the perch: their outlines lie
    ``GAP`` apart in angle, seen from the room's middle."""
    spans = []
    for t, shape in zip(r.things, outlines(seed, r), strict=True):
        axis = math.atan2(t.place[1], t.place[0])
        low, high = _span(shape, np.zeros(2), axis)
        spans.append((axis + low, axis + high))
    for i in range(len(spans)):
        for j in range(i):
            mid = math.atan2(math.sin(spans[i][0] - spans[j][0]), math.cos(spans[i][0]
                                                                           - spans[j][0]))
            a_low, a_high = spans[j][0] + mid, spans[j][0] + mid + (spans[i][1] - spans[i][0])
            if a_low < spans[j][1] + GAP and a_high > spans[j][0] - GAP:
                return False
    return True


# -- the eyes' images


def _segments(shapes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.concatenate([s for s in shapes]) if shapes else np.zeros((0, 2))
    ends = np.concatenate([np.roll(s, -1, axis=0) for s in shapes]) if shapes else np.zeros(
        (0, 2))
    owner = np.concatenate([np.full(len(s), i) for i, s in enumerate(shapes)]) if shapes else (
        np.zeros(0, dtype=int))
    return starts, ends, owner


def render(seed: int, r: world.Room, eyes: Any, at: Any, heading: float, *,
           rng: np.random.Generator | None = None, colours: Any = None,
           shapes: list[np.ndarray] | None = None) -> np.ndarray:
    """What each eye sees facing ``heading``: ``(eyes, receptors, 4)``, each receptor's red,
    green, blue and distance, the means over its rays, with the receptors' noise (``NOISE``,
    drawn from ``rng``; none without it)."""
    shapes = outlines(seed, r) if shapes is None else shapes
    tint = RGB[[t.colour for t in r.things]] if colours is None else np.asarray(colours)
    p, q, owner = _segments(shapes)
    e = q - p
    out = []
    at = np.asarray(at, dtype=np.float64).reshape(2)
    for eye in eyes:
        origin = eye.origin(at, heading)
        angles = (eye.facing(heading) + eye.rays).ravel()
        u = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (rays, 2)
        # the walls: the room's plane from the eye outward
        x_low, x_high, y_low, y_high = world.PLANE
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(u[:, 0] > 0, (x_high - origin[0]) / u[:, 0],
                          (x_low - origin[0]) / u[:, 0])
            ty = np.where(u[:, 1] > 0, (y_high - origin[1]) / u[:, 1],
                          (y_low - origin[1]) / u[:, 1])
        tx = np.where(np.isfinite(tx), tx, np.inf)
        ty = np.where(np.isfinite(ty), ty, np.inf)
        reach = np.minimum(tx, ty)
        normal = np.where((tx < ty)[:, None], np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        colour = np.broadcast_to(WALL, (len(u), 3)).copy()
        if len(p):
            denom = u[:, 0:1] * e[None, :, 1] - u[:, 1:2] * e[None, :, 0]  # (rays, segs)
            w = p[None, :, :] - origin[None, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / denom
                s = (w[..., 0] * u[:, None, 1] - w[..., 1] * u[:, None, 0]) / denom
            hit = (np.abs(denom) > 1e-12) & (t > 1e-9) & (s >= 0.0) & (s <= 1.0)
            t = np.where(hit, t, np.inf)
            nearest = np.argmin(t, axis=1)
            best = t[np.arange(len(u)), nearest]
            thing = best < reach
            reach = np.where(thing, best, reach)
            seg = e[nearest]
            seg_normal = np.stack([-seg[:, 1], seg[:, 0]], axis=1) / np.maximum(
                np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
            normal = np.where(thing[:, None], seg_normal, normal)
            colour = np.where(thing[:, None], tint[owner[nearest]], colour)
        shade = 0.55 + 0.45 * np.abs(np.sum(u * normal, axis=1))
        light = colour * shade[:, None]
        rays = eye.rays.shape[1]
        rgb = light.reshape(eye.receptors, rays, 3).mean(axis=1)
        depth = reach.reshape(eye.receptors, rays).mean(axis=1)
        if rng is not None:
            depth = depth * (1.0 + NOISE["depth"] * rng.standard_normal(len(depth)))
            rgb = np.clip(rgb + NOISE["colour"] * rng.standard_normal(rgb.shape), 0.0, 1.0)
        out.append(np.concatenate([rgb, depth[:, None]], axis=1))
    return np.stack(out)


def wholly_in_view(eyes: Any, at: Any, heading: float, shape: np.ndarray) -> bool:
    """Whether an outline lies wholly inside an eye's field."""
    for eye in eyes:
        low, high = _span(shape, eye.origin(at, heading), eye.facing(heading))
        if low >= -eye.field and high <= eye.field:
            return True
    return False


# -- the script, its rooms clear from the perch


def _held_out(place: np.ndarray) -> bool:
    """Whether a place lies in the band of distances every room leaves out (R3's)."""
    low, high = HELD_OUT["band"]
    return bool(low <= float(np.linalg.norm(place)) <= high)


@contextlib.contextmanager
def _clear_rooms(seed: int) -> Iterator[None]:
    """While the words chamber draws a script: every room is drawn again until it is clear
    from the perch (``clear``) and holds no thing in the band of distances R3 holds out, and a
    question's thing is in view where it lies in either eye's field."""
    room, free, view = world.room, words._free_place, world.VIEW

    def clear_room(rng: np.random.Generator, *args: Any, **kwargs: Any) -> world.Room:
        for _ in range(1000):
            r = room(rng, *args, **kwargs)
            if clear(seed, r) and not any(_held_out(t.place) for t in r.things):
                return r
        raise RuntimeError("no clear room in 1000 draws")

    def clear_place(rng: np.random.Generator, r: world.Room) -> np.ndarray:
        for _ in range(1000):
            p = free(rng, r)
            trial = world.Room(r.things + [world.Thing(0, 0, p)], heading=r.heading)
            # the thing to come may be of any look: the widest (a large disc) is checked
            trial.things[-1] = world.Thing(8, 0, p)
            if clear(seed, trial) and not _held_out(p):
                return p
        raise RuntimeError("no clear place in 1000 draws")

    world.room, words._free_place, world.VIEW = clear_room, clear_place, FIELD
    try:
        yield
    finally:
        world.room, words._free_place, world.VIEW = room, free, view


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """The words chamber's script for the founder, its rooms clear from the perch."""
    with _clear_rooms(seed):
        return words.script(seed, **given)


# -- the arms


def make_brain(seed: int, **settings: Any) -> Any:
    """The brain with eyes: a retina (two parrot eyes, its own categories of form and tint),
    and the words chamber's brain whose map holds the retina's categories."""
    from cadence.space.map import Map
    from cadence.words import Words
    from cadence.words.retina import Retina

    retina = Retina(**{k: v for k, v in settings.items() if k in RETINA_SETTINGS})
    held = Map(world.PLANE, world.CELLS, units=retina.kinds.capacity,
               values={"colour": retina.colours.capacity}, seed=int(seed))
    brain = Words(held, **{k: v for k, v in settings.items() if k not in RETINA_SETTINGS})
    brain.retina = retina  # type: ignore[attr-defined]
    return brain


RETINA_SETTINGS = ("eyes", "kinds", "colours", "edge", "least", "fuse", "near", "vigilance",
                   "rate", "size_units", "bend_units", "hue_units", "light_units")


def camera_seen(r: world.Room, heading: float) -> list[tuple[int, np.ndarray, dict[str, int]]]:
    """The reference camera: each thing whose middle lies in either eye's field."""
    return [(t.kind, t.place.copy(), {"colour": t.colour}) for t in r.things
            if world.in_view(heading, t.place)[0]]


class Sight:
    """A life's moments of sight for both arms, and R1's counts."""

    def __init__(self, seed: int, eyes_brain: Any, camera_brain: Any):
        self.seed = seed
        self.eyes, self.camera = eyes_brain, camera_brain
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 330, 2]))
        self.counts = {"in_view": 0, "found": 0, "recognised": 0, "figures": 0, "spurious": 0}
        # what each category was seen as: true kind by kind category, true colour by colour one
        self.kinds = np.zeros((world.KINDS, eyes_brain.retina.kinds.capacity), dtype=int)
        self.colours = np.zeros((world.COLOURS, eyes_brain.retina.colours.capacity), dtype=int)
        self.counting = False
        self.looks = 0  # the head's turns to look at a thing not recognised
        self.looked: list[list[float]] = []  # each room's look headings, in order

    def moment(self, r: world.Room, heading: float, *, camera: bool = True) -> None:
        if camera:
            with _view():
                self.camera.body.heading = heading
                self.camera.look(seen=camera_seen(r, heading), lit=world.camera(heading))
        brain = self.eyes
        brain.body.heading = heading
        retina = brain.retina
        shapes = outlines(self.seed, r)
        images = render(self.seed, r, retina.eyes, brain.body.at, heading, rng=self.rng,
                        shapes=shapes)
        reports, lit = retina.see(images, brain.body.at, heading, held=brain.map)
        brain.look(seen=reports, lit=lit)
        if self.counting:
            self.count(r, heading, shapes, retina.last)

    def count(self, r: world.Room, heading: float, shapes: list[np.ndarray],
              figures: list[Any]) -> None:
        places = np.array([t.place for t in r.things])
        for t, shape in zip(r.things, shapes, strict=True):
            if not wholly_in_view(self.eyes.retina.eyes, self.eyes.body.at, heading, shape):
                continue
            self.counts["in_view"] += 1
            near = [f for f in figures if np.linalg.norm(f.place - t.place) <= FOUND]
            self.counts["found"] += bool(near)
            self.counts["recognised"] += any(f.kind is not None for f in near)
            for f in near:
                if f.kind is not None and f.colour is not None:
                    self.kinds[t.kind, f.kind] += 1
                    self.colours[t.colour, f.colour] += 1
        self.counts["figures"] += len(figures)
        self.counts["spurious"] += sum(
            bool(np.linalg.norm(places - f.place, axis=1).min() > FOUND) for f in figures)

    def look_around(self, r: world.Room) -> None:
        """A new room: both arms look around in eight moments. The parrot with eyes then turns
        its head to look at each thing it found and did not recognise (``LOOKS_AT`` at most),
        one moment each, and both arms turn back to the room's heading."""
        self.eyes.map.reset()
        self.camera.map.reset()
        retina = self.eyes.retina
        unknown: list[np.ndarray] = []
        for i in range(8):
            self.moment(r, r.heading + i * math.pi / 4)
            unknown += retina.unknown
        heading = r.heading + 7 * math.pi / 4
        looked: list[np.ndarray] = []
        turns: list[float] = []
        for place in unknown:
            held = [e["place"] for e in self.eyes.map.things() if e["place"] is not None]
            if any(np.linalg.norm(place - q) <= retina.fuse for q in looked + held):
                continue
            if len(looked) == LOOKS_AT:
                break
            looked.append(place)
            heading = retina.look_at(place, self.eyes.body.at, heading)
            self.moment(r, heading, camera=False)
            turns.append(float(heading))
        self.looks += len(looked)
        self.looked.append(turns)
        self.moment(r, r.heading)

    def turn(self, r: world.Room, heading: float) -> None:
        r.heading = heading
        self.moment(r, heading)


@contextlib.contextmanager
def _view() -> Iterator[None]:
    view = world.VIEW
    world.VIEW = FIELD
    try:
        yield
    finally:
        world.VIEW = view


def places_held(brain: Any, camera: Any, r: world.Room) -> dict[str, list[float]]:
    """R2: for each thing, the distance between the place the map from the eyes holds nearest
    it and the place the map from the camera holds for its kind nearest it; by whether the
    thing is in either eye's field."""
    out: dict[str, list[float]] = {"in_view": [], "out_of_view": [], "eyes_true": [],
                                   "camera_true": []}
    held = [e for e in brain.map.things() if e["place"] is not None]
    with _view():
        for t in r.things:
            ours = min((e["place"] for e in held), key=lambda p: float(np.linalg.norm(
                p - t.place)), default=None)
            theirs = list(camera.map.where_is(t.kind))
            ref = min(theirs, key=lambda p: float(np.linalg.norm(p - t.place)), default=None)
            if ours is None or ref is None:
                apart = math.inf
            else:
                apart = float(np.linalg.norm(ours - ref))
            seen = bool(world.in_view(r.heading, t.place)[0])
            out["in_view" if seen else "out_of_view"].append(apart)
            out["eyes_true"].append(math.inf if ours is None else float(np.linalg.norm(
                ours - t.place)))
            out["camera_true"].append(math.inf if ref is None else float(np.linalg.norm(
                ref - t.place)))
    return out


def held_rate(brain: Any, r: world.Room) -> list[bool]:
    """For each thing out of view: whether the map holds a thing within 0.15 m of its place
    (the map from the eyes holds its own categories, so the place is read)."""
    out = []
    with _view():
        for t in r.things:
            if world.in_view(r.heading, t.place)[0]:
                continue
            held = [e for e in brain.map.things() if e["place"] is not None
                    and np.linalg.norm(e["place"] - t.place) <= 0.15]
            out.append(bool(held))
    return out


def held_out_views(seed: int, brain: Any, count: int = 96
                   ) -> tuple[dict[str, list[bool]], list[dict[str, Any]]]:
    """R3: ``count`` views of one trained thing at a distance no room holds (nearer than any,
    or in the band every room leaves out: ``HELD_OUT``) and turned farther than any room turns
    a thing, in a drawn colour, the parrot looking at it: the head turned so the thing lies
    within ``LOOK`` of an eye's middle. The map is emptied, the parrot sees the view for one
    moment (its categories take no lesson) and is asked what is at the thing's place around it
    (its kind's word) and what colour the thing is (its colour's word, asked by its kind's
    word). Gives the readings and a record of each view."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 330, 3]))
    lex = world.lexicon(seed)
    retina = brain.retina
    out: dict[str, list[bool]] = {"R3_kind": [], "R3_colour": [], "R3_both": [],
                                  "R3_found": []}
    views: list[dict[str, Any]] = []
    for i in range(count):
        kind = int(rng.integers(world.TRAINED))
        colour = int(rng.integers(world.COLOURS))
        low, high = HELD_OUT["near"] if i % 2 == 0 else HELD_OUT["band"]
        radius, angle = rng.uniform(low, high), rng.uniform(-math.pi, math.pi)
        place = np.array([radius * math.cos(angle), radius * math.sin(angle)])
        turn = turn_of(seed, place, held_out=True)
        eye = int(rng.integers(len(retina.eyes)))
        heading = angle - retina.eyes[eye].axis - rng.uniform(-LOOK, LOOK)
        r = world.Room([world.Thing(kind, colour, place)], heading=heading,
                       extra={"turns": {0: turn}})
        brain.map.reset()
        brain.body.heading = heading
        images = render(seed, r, retina.eyes, brain.body.at, heading, rng=rng)
        reports, lit = retina.see(images, brain.body.at, heading, learn=False)
        brain.look(seen=reports, lit=lit)
        near = [f for f in retina.last if np.linalg.norm(f.place - place) <= FOUND
                and f.kind is not None and f.colour is not None]
        out["R3_found"].append(bool(near))
        what = world.by_place(lex, r, world.place_of(heading, place))
        which = world.colour_of(lex, r, kind)
        out["R3_kind"].append(world.right(what, brain.ask(what.text)))
        out["R3_colour"].append(world.right(which, brain.ask(which.text)))
        out["R3_both"].append(out["R3_kind"][-1] and out["R3_colour"][-1])
        views.append({"kind": kind, "colour": colour, "radius": float(radius),
                      "turn": float(turn), "found": bool(near),
                      "category": near[0].kind if near else None,
                      "width": near[0].width if near else None,
                      "bend": near[0].bend if near else None,
                      "kind_right": out["R3_kind"][-1], "colour_right": out["R3_colour"][-1]})
    return out, views


def life(seed: int, *, record: Any = None, views: int = 96, **given: Any) -> dict[str, Any]:
    """One founder's life for both arms on the same script; the readings by gate, each arm's
    under its name. ``practice`` is the words chamber's (``"told"``, ``"reward"``: a pistachio
    for a right answer, ``"withheld"``)."""
    practice = str(given.pop("practice", "told"))
    if practice not in words.PRACTICE:
        raise ValueError(f"practice is one of {words.PRACTICE}")
    settings = {**words.DEFAULTS, **{k: v for k, v in given.items() if k in words.DEFAULTS}}
    brain_settings = {k: v for k, v in given.items() if k not in words.DEFAULTS}
    began = time.time()
    events = script(seed, **settings)
    eyes = make_brain(seed, **brain_settings)
    camera = words.make_brain(seed)
    sight = Sight(seed, eyes, camera)
    arms = {"eyes": eyes, "camera": camera}
    readings: dict[str, dict[str, list[bool]]] = {a: {} for a in arms}
    rewarded: dict[str, list[bool]] = {a: [] for a in arms}
    unheld: dict[str, dict[str, int]] = {a: {} for a in arms}  # namings of a place held empty
    held: dict[str, list[bool]] = {a: [] for a in arms}
    apart: dict[str, list[float]] = {}
    ablated: dict[str, Any] = {}
    r: world.Room | None = None
    held_for: world.Room | None = None
    seconds = {"script": time.time() - began}
    # the rooms whose things are asked about: their moments are R1's
    tested, room_of = set(), None
    for event in events:
        if event[0] == "room":
            room_of = event[1]
        elif event[0] == "ask":
            tested.add(id(room_of))
    began = time.time()
    for event in events:
        kind = event[0]
        if kind == "room":
            r = event[1]
            sight.counting = id(r) in tested  # R1 counts the moments of the test rooms
            sight.look_around(r)
        elif kind == "turn":
            assert r is not None
            sight.turn(r, event[1])
        elif kind == "name":
            for name, brain in arms.items():
                if math.isnan(brain.name(event[2], event[1])):  # the map holds nothing there
                    unheld[name][event[3]] = unheld[name].get(event[3], 0) + 1
        elif kind == "practice":
            q = event[1]
            for name, brain in arms.items():
                if practice == "told":
                    brain.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
                else:  # the answer is never told: a right one earns a pistachio
                    right = world.right(q, brain.ask(q.text, explore=True))
                    rewarded[name].append(right)
                    if practice == "reward":
                        brain.reward(1.0 if right else 0.0)
        else:
            gate, q = event[1], event[2]
            assert r is not None
            if held_for is not r:
                held_for = r
                for key, values in places_held(eyes, camera, r).items():
                    apart.setdefault(key, []).extend(values)
                if gate[:2] in ("W1", "W4", "W5"):
                    held["eyes"] += held_rate(eyes, r)
                    with _view():
                        held["camera"] += words.held_rate(camera, r)
            for name, brain in arms.items():
                said = brain.ask(q.text)
                ok = world.right(q, said)
                readings[name].setdefault(gate, []).append(ok)
                if gate.startswith("W1"):
                    if name not in ablated:
                        ablated[name] = words.ablate(brain)
                    readings[name].setdefault("W6_ablated", []).append(
                        world.right(q, ablated[name].ask(q.text)))
                if gate.startswith("W1") and q.about == "out_of_view":
                    readings[name].setdefault("W4_out_of_view", []).append(ok)
                if gate == "W5_by_place" and q.about == "several":  # a place that holds things
                    readings[name].setdefault("W5_by_place_held", []).append(ok)
                if record is not None:
                    record.append({"arm": name, "gate": gate, "question": q.text,
                                   "answer": q.answer, "said": said, "right": ok})
    seconds["life"] = time.time() - began
    began = time.time()
    r3, looked = held_out_views(seed, eyes, views)
    seconds["R3"] = time.time() - began
    out: dict[str, Any] = {"seed": int(seed), "seconds": seconds, "practice": practice}
    for name, brain in arms.items():
        got = words.readings_of(seed, settings, readings[name], held[name], brain, {})
        out[name] = {k: v for k, v in got.items() if k not in words.DEFAULTS and k != "seed"}
        if rewarded[name]:
            out[name]["pistachios"] = int(sum(rewarded[name]))
        out[name]["namings_unheld"] = unheld[name]
    retina = eyes.retina
    out["eyes"]["parameters_retina"] = retina.parameters()
    out["eyes"]["parameters"] = int(out["eyes"]["parameters"]) + retina.parameters()
    out["eyes"]["macs"] = int(out["eyes"]["macs"]) + int(
        retina.kinds.work["macs"] + retina.colours.work["macs"])
    out["eyes"]["categories"] = {"kinds": retina.kinds.used, "colours": retina.colours.used}
    english = world.lexicon(seed).english()
    out["eyes"]["said_for"] = {  # the word the brain says for each category, in English
        group: [english.get(eyes.lexicon.say((group, c)) or "", "") for c in range(layer.used)]
        for group, layer in (("kind", retina.kinds), ("colour", retina.colours))}
    c = sight.counts
    out["R1"] = {"found": c["found"] / max(c["in_view"], 1),
                 "recognised": c["recognised"] / max(c["in_view"], 1),
                 "spurious": c["spurious"] / max(c["figures"], 1), **c}
    # each category's share of views of its most frequent kind (colour): how pure it is
    for name, table in (("kinds", sight.kinds), ("colours", sight.colours)):
        seen = table.sum(axis=0)
        out["R1"][f"{name}_purity"] = float(table.max(axis=0).sum() / max(seen.sum(), 1))
        out["R1"][f"{name}_table"] = table.tolist()
    out["R2"] = {}
    for key, values in apart.items():
        v = np.array(values)
        if key in ("in_view", "out_of_view"):
            out["R2"][key] = float(np.mean(v <= MARGIN)) if len(v) else float("nan")
        out["R2"][f"{key}_median"] = float(np.median(v)) if len(v) else float("nan")
        out["R2"][f"{key}_n"] = len(v)
    out["R3"] = {k: float(np.mean(v)) for k, v in r3.items()}
    out["R3"]["n"] = len(r3["R3_kind"])
    out["R3"]["views"] = looked
    out["eyes"]["looks"] = sight.looks
    out["eyes"]["look_headings"] = sight.looked  # each room's looks: the controls see them too
    return out


if __name__ == "__main__":
    import json

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    told = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(life(seed, **told), indent=1, default=float))
