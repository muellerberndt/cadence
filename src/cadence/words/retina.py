"""Two retinas: a room seen as images across a parrot's two fields of view.

A parrot's eyes sit on the sides of its head. Each eye looks out sideways over a wide field,
the two fields overlap a little in front of the beak, and behind the head is a blind spot. The
parrot turns its head to look. Its room is flat and seen across, so each eye's image is one
row: a receptor a ray from the eye, reporting the colour and the distance of what the ray meets.

- **The eyes** (``Eye``, ``parrot_eyes``): receptors across each eye's field, sharp at the eye's
  centre (its fovea, pointing sideways) and coarse toward the edges. A receptor pools the light
  of the rays of its patch: its colour and its distance are the means over its patch, so a
  receptor on a thing's edge reads a mix of the thing and what lies behind it.
- **Finding things** (born, the eye's own wiring): a jump in distance between neighbouring
  receptors is an edge. A run of receptors between edges that lies nearer than the runs on both
  sides is a thing seen whole; nearer on one side only, a thing seen in part (another thing in
  front of it, or the field's edge). A receptor that mixes a thing and what lies behind it gives
  where the edge falls inside it, by how far its distance lies between the two.
- **Where a thing is** (born, as a body knows its own head): each edge is a point on the plane,
  from the eye's place, its ray and the distance the ray reached; a thing's width is the
  distance between its edges and its place their middle, or, for a thing that bulges toward the
  eye, the point behind its front by its width times its bend over four (a disc's middle). A
  thing both eyes see is one thing.
- **Looking** (the head's turn): a thing found but not recognised (seen too coarsely, or
  cut by a field's edge) is noted in ``unknown``; ``look_at`` gives the head's heading that
  brings a place to the middle of an eye, where the fovea sees it sharply.
- **What a thing is and its colour** (learned, ``Categories``): a whole thing's form (its width
  and how it bends: the curvature of the surface its receptors see, times its width, positive
  where it bulges toward the eye; a unit tuned to both at once) and its tint (hue and
  lightness) are population codes.
  Each settles a layer of category neurons whose synapses are the brain's lessons: a category
  holds the mean of what it was seen as, a thing that no category matches takes a new one, as a
  word enters a lexicon when first heard. Nothing tells the brain what a thing is.

``Retina.see`` turns the eyes' images into the map's reports ``(kind, place, {"colour": c})``
and the cells the eyes see (``lit``), for ``Words.look`` or ``Map.moment``: the kinds and the
colours are the retina's own categories, so the map is made with ``units=`` the kinds'
capacity and ``values={"colour": ...}`` the colours'.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SIZE = (0.06, 0.45)  # the widths a form's code tiles, in metres (a log scale)
BEND = (-2.5, 3.0)  # the bends a form's code tiles: curvature times width
LIGHT = 0.6  # a colour's lightness code, as a share of its hue code's length


class Eye:
    """One eye: ``receptors`` (odd) across ``field`` radians either side of its ``axis`` (from
    the head's heading, counter-clockwise), at ``offset`` metres from the head's middle (forward,
    left). The spacing of the receptors grows from ``fovea`` times the mean spacing at the axis to
    ``2 - fovea`` times it at the field's edges; each pools ``rays`` rays spread evenly over its
    patch. Angles are relative to the axis, ascending (receptor 0 at ``-field``)."""

    def __init__(self, *, axis: float = 0.0, offset: tuple[float, float] = (0.0, 0.0),
                 receptors: int = 321, field: float = math.radians(80.0),
                 fovea: float = 0.4, rays: int = 6):
        if int(receptors) < 3 or int(receptors) % 2 == 0 or not 0.0 < float(fovea) <= 1.0:
            raise ValueError("an eye takes an odd count of 3 receptors or more and a fovea in "
                             "(0, 1]")
        if not 0.0 < float(field) < math.pi or int(rays) < 1:
            raise ValueError("an eye's field is within (0, pi) either side; one ray or more")
        self.axis, self.offset = float(axis), (float(offset[0]), float(offset[1]))
        self.receptors, self.field, self.fovea = int(receptors), float(field), float(fovea)
        t = np.linspace(-1.0, 1.0, self.receptors + 1)
        self.edges = self.field * t * (self.fovea + (1.0 - self.fovea) * np.abs(t))
        self.widths = np.diff(self.edges)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        u = (np.arange(int(rays)) + 0.5) / int(rays)
        self.rays = self.edges[:-1, None] + u[None, :] * self.widths[:, None]

    def settings(self) -> dict[str, Any]:
        return {"axis": self.axis, "offset": list(self.offset), "receptors": self.receptors,
                "field": self.field, "fovea": self.fovea, "rays": int(self.rays.shape[1])}

    def origin(self, at: Any, heading: float) -> np.ndarray:
        """Where the eye is on the plane, for a head at ``at`` facing ``heading``."""
        c, s = math.cos(heading), math.sin(heading)
        f, left = self.offset
        return np.asarray(at, dtype=np.float64).reshape(2) + np.array(
            [f * c - left * s, f * s + left * c])

    def facing(self, heading: float) -> float:
        """The direction of the eye's axis on the plane, for a head facing ``heading``."""
        return float(heading) + self.axis


def parrot_eyes(receptors: int = 321, *, axis: float = math.radians(70.0),
                field: float = math.radians(80.0), apart: float = 0.04, fovea: float = 0.4,
                rays: int = 6) -> tuple[Eye, Eye]:
    """A parrot's two eyes, left and right: each looks out ``axis`` from the beak to its side
    over ``field`` either side, ``apart`` metres between them. At the founders each eye sees
    from 10 degrees across the beak's line to 150 degrees back: the fields overlap 20 degrees in
    front of the beak and leave 60 degrees behind the head unseen."""
    return (Eye(axis=axis, offset=(0.0, apart / 2), receptors=receptors, field=field,
                fovea=fovea, rays=rays),
            Eye(axis=-axis, offset=(0.0, -apart / 2), receptors=receptors, field=field,
                fovea=fovea, rays=rays))


@dataclass
class Figure:
    """A thing found in one eye's image: receptors ``low`` to ``high`` (a run between edges),
    its edges as points on the plane, its ``place`` (their middle) and ``width``; ``whole``
    where it lies nearer than what is on both sides of it, ``full`` its receptors that see
    only it. A whole thing seen well carries its ``form`` and ``tint`` codes and, once
    recognised, its ``kind`` and ``colour`` (the retina's categories)."""

    eye: int
    low: int
    high: int
    ends: np.ndarray  # (2, 2): the edge points, low angle first
    place: np.ndarray
    width: float
    whole: bool
    full: int
    rgb: np.ndarray
    bend: float | None = None
    form: np.ndarray | None = None
    tint: np.ndarray | None = None
    kind: int | None = None
    colour: int | None = None


class Categories:
    """Categories learned from what is seen, over ``features`` feature neurons: ``capacity``
    category neurons, each joined to every feature neuron by a reciprocal pair of synapses that
    starts at zero.

    Seeing drives the feature neurons with a code and the category neurons settle: the most
    active is the category seen. A lesson drives a category's neuron and nudges the feature
    neurons toward the code seen (the library's free and nudged phases, a quadratic nudge); a
    category's ``n``-th lesson closes ``1 / n`` of its error and no less than ``rate``, so its
    synapses hold the mean of what it was seen as. A code whose best category settles below
    ``vigilance`` takes the next free category: a category enters when a new sort of thing is
    first seen. One lesson's step closes about half its error, so a category enters imprinted:
    its first sight is taught ``imprint`` times and counts as one lesson."""

    def __init__(self, features: int, capacity: int, *, vigilance: float = 0.4,
                 rate: float = 0.05, steps: int = 6, level: float = 0.5, drive: float = 2.0,
                 threshold: float = 0.15, imprint: int = 4, backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        if int(features) < 1 or int(capacity) < 1 or not 0.0 < float(rate) <= 1.0:
            raise ValueError("categories take features and a capacity of 1 or more and a rate "
                             "in (0, 1]")
        self.features, self.capacity = int(features), int(capacity)
        self.vigilance, self.rate, self.steps = float(vigilance), float(rate), int(steps)
        self.level, self.drive, self.threshold = float(level), float(drive), float(threshold)
        self.imprint = max(int(imprint), 1)
        self.settings = {"features": self.features, "capacity": self.capacity,
                         "vigilance": self.vigilance, "rate": self.rate, "steps": self.steps,
                         "level": self.level, "drive": self.drive, "threshold": self.threshold,
                         "imprint": self.imprint}
        self.category = np.arange(self.capacity)
        self.feature = self.capacity + np.arange(self.features)
        self.neurons = n = self.capacity + self.features
        pairs = np.stack(np.meshgrid(self.category, self.feature, indexing="ij"),
                         axis=-1).reshape(-1, 2)
        self.connectome = Connectome.from_synapses(
            n, label="categories", pre=np.concatenate([pairs[:, 0], pairs[:, 1]]),
            post=np.concatenate([pairs[:, 1], pairs[:, 0]]),
            populations={"category": self.category.tolist(), "feature": self.feature.tolist()})
        self._model = learning_neuron_model(dt=1.0)
        # a category neuron's threshold: the weak drive of a code it shares little with leaves
        # it silent, so a settle does not spread from one category to another
        self._bias = np.zeros(n)
        self._bias[self.category] = -self.threshold
        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.zeros(self.connectome.synapses), bias=self._bias,
                           backend=backend)
        pre, post = np.asarray(self.connectome.pre), np.asarray(self.connectome.post)
        owner = np.where(np.isin(pre, self.category), pre, post)
        order = np.argsort(owner, kind="stable")
        self._touch = np.split(order, np.searchsorted(owner[order], self.category[1:]))
        self._learner: Any = None
        self.counts = np.zeros(self.capacity, dtype=np.int64)  # each category's lessons
        self.used = 0  # the categories that have entered
        self.work = {"settles": 0, "macs": 0}

    def _count(self, settles: int) -> None:
        self.work["settles"] += settles
        self.work["macs"] += settles * self.steps * int(self.connectome.synapses)

    def settle(self, codes: Any) -> np.ndarray:
        """The category neurons settled on ``(rows, features)`` codes: ``(rows, capacity)``."""
        given = np.asarray(codes, dtype=np.float64).reshape(-1, self.features)
        drive = np.zeros((len(given), self.neurons))
        drive[:, self.feature] = self.drive * given
        state = self.graph.settle_batch(drive, steps=self.steps, tolerance=None)
        self._count(len(given))
        return np.asarray(np.asarray(state.activation)[:, self.category], dtype=np.float64)

    def see(self, code: Any, *, learn: bool = True, enter: bool | None = None) -> int | None:
        """The category of a code: the most active of those that entered where it settles
        above ``vigilance``. Below it, with ``enter`` (``learn`` unless given) a new category
        enters for the code; without it the code is no category the layer knows (None). With
        ``learn`` the category takes the code as a lesson."""
        code = np.asarray(code, dtype=np.float64).reshape(self.features)
        enter = learn if enter is None else bool(enter)
        best: int | None = None
        if self.used:
            active = self.settle(code)[0, : self.used]
            best = int(np.argmax(active))
            if active[best] < self.vigilance:
                best = None
        if best is None:
            if not enter or self.used == self.capacity:
                return None
            best, self.used = self.used, self.used + 1
            for _ in range(self.imprint - 1):  # imprinted: the first sight taught again
                self.teach(best, code)
                self.counts[best] = 0
        if learn:
            self.teach(best, code)
        return best

    def teach(self, category: int, code: Any) -> float:
        """A lesson: ``category``'s neuron driven while ``code`` is seen. Gives the error before
        the lesson, as a share of the target's sum."""
        from ..learning import Learner, LearnerConfig

        if self._learner is None:
            self._learner = Learner(
                self.graph, list(self.feature),
                LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0,
                              free_steps=self.steps, nudged_steps=self.steps, tolerance=None),
                reciprocal=True, plastic_neurons=np.zeros(self.neurons, dtype=bool))
        learner = self._learner
        k = int(category)
        drive = np.zeros((1, self.neurons))
        drive[0, self.category[k]] = self.drive
        lit = np.tanh(drive[:, self.category] / 2.0)
        shared = float((lit.sum(axis=0) ** 2).sum())
        share = max(1.0 / (self.counts[k] + 1), self.rate)
        steps = np.zeros(self.connectome.synapses)
        steps[self._touch[k]] = 1.0
        learner.synapse_rate = steps
        learner.config = dataclasses.replace(
            learner.config, eta=share / (0.25 * max(shared, 1e-12)))
        target = np.zeros_like(drive)
        target[0, self.feature] = self.level * np.asarray(code, dtype=np.float64)
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        self.graph = learner.brain
        self.counts[k] += 1
        self._count(3)
        answer = np.asarray(free.activation)[0, self.feature]
        wanted = target[0, self.feature]
        return float(np.abs(answer - wanted).sum() / max(np.abs(wanted).sum(), 1e-12))

    def parameters(self) -> int:
        """Learned parameters: one efficacy a reciprocal pair."""
        return int(self.connectome.synapses // 2)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"efficacy": np.asarray(self.graph.efficacy, dtype=np.float64).copy(),
                "counts": self.counts.copy(), "used": np.array(self.used)}


def _tuned(value: float, low: float, high: float, per: int) -> np.ndarray:
    """A value as ``per`` Gaussian tuned inputs tiling ``(low, high)``, read at the nearer end
    outside it."""
    preferred = np.linspace(low, high, per)
    spacing = (high - low) / (per - 1)
    out = np.exp(-0.5 * ((float(np.clip(value, low, high)) - preferred) / spacing) ** 2)
    return np.asarray(np.where(out < 0.05, 0.0, out))


def _unit(code: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(code))
    return code / norm if norm > 0 else code


class Retina:
    """What a head's eyes see of its room, as the map's reports.

    ``eyes`` (two parrot eyes unless given); ``kinds`` and ``colours`` the capacities of the
    learned categories of form and of tint; ``edge`` the jump in distance between neighbouring
    receptors, as a share of the nearer, that is an edge; ``least`` the receptors that see a
    thing alone for its form to be read; ``fuse`` how near two eyes' things lie to be one;
    ``near`` how far short of a ray's reach a place lies to count as seen (``lit``);
    ``vigilance`` of each category layer, ``(form, tint)``; ``rate`` the categories' least
    share a lesson closes.

    ``find(images, at, heading)`` gives every thing found, ``see(images, at, heading)`` the
    reports of those recognised and the cells the eyes see. ``images`` is ``(eyes, receptors,
    4)``: each receptor's red, green, blue (0 to 1) and distance in metres."""

    def __init__(self, eyes: Any = None, *, kinds: int = 24, colours: int = 8,
                 edge: float = 0.06, least: int = 12, fuse: float = 0.12, near: float = 0.15,
                 vigilance: tuple[float, float] = (0.4, 0.4), rate: float = 0.05,
                 size_units: int = 12, bend_units: int = 14, hue_units: int = 12,
                 light_units: int = 8, backend: Any = "cpu"):
        self.eyes = tuple(eyes) if eyes is not None else parrot_eyes()
        self.edge, self.least = float(edge), int(least)
        self.fuse, self.near = float(fuse), float(near)
        self.size_units, self.bend_units = int(size_units), int(bend_units)
        self.hue_units, self.light_units = int(hue_units), int(light_units)
        form = self.size_units * self.bend_units
        tint = self.hue_units + self.light_units
        self.kinds = Categories(form, int(kinds), vigilance=float(vigilance[0]), rate=rate,
                                backend=backend)
        self.colours = Categories(tint, int(colours), vigilance=float(vigilance[1]), rate=rate,
                                  backend=backend)
        self.settings = {"eyes": [e.settings() for e in self.eyes], "kinds": int(kinds),
                         "colours": int(colours), "edge": self.edge, "least": self.least,
                         "fuse": self.fuse, "near": self.near,
                         "vigilance": [float(v) for v in vigilance], "rate": float(rate),
                         "size_units": self.size_units, "bend_units": self.bend_units,
                         "hue_units": self.hue_units, "light_units": self.light_units}
        self.last: list[Figure] = []
        self.unknown: list[np.ndarray] = []  # where things were found that were not recognised
        self.work = {"moments": 0, "figures": 0}

    # -- the codes of what a thing looks like

    def form_code(self, width: float, bend: float) -> np.ndarray:
        """A thing's form: a population code of units each tuned to a width (on a log scale)
        and a bend together, at unit length."""
        size = _tuned(math.log(max(width, 1e-3)), math.log(SIZE[0]), math.log(SIZE[1]),
                      self.size_units)
        shape = _tuned(float(bend), BEND[0], BEND[1], self.bend_units)
        return _unit(np.outer(size, shape).ravel())

    def tint_code(self, rgb: np.ndarray) -> np.ndarray:
        """A colour as hue (tuned around the hue circle, at the colour's saturation) and
        lightness (on a log scale, at ``LIGHT`` of the hue's weight), each a population code,
        at unit length together."""
        r, g, b = (float(v) for v in np.clip(rgb, 0.0, 1.0))
        top, low = max(r, g, b), min(r, g, b)
        chroma = top - low
        hue = math.atan2(math.sqrt(3.0) * (g - b), 2.0 * r - g - b)
        preferred = np.linspace(-math.pi, math.pi, self.hue_units, endpoint=False)
        hues = _unit(np.exp(8.0 * (np.cos(hue - preferred) - 1.0))) * min(chroma / 0.5, 1.0)
        light = _tuned(math.log(max((r + g + b) / 3.0, 1e-3)), math.log(0.04), 0.0,
                       self.light_units)
        return _unit(np.concatenate([hues if chroma > 0.05 else 0.0 * hues,
                                     LIGHT * _unit(light)]))

    # -- finding things in an image

    def _figures(self, k: int, image: np.ndarray, at: np.ndarray, heading: float
                 ) -> list[Figure]:
        eye = self.eyes[k]
        rgb, depth = image[:, :3], image[:, 3]
        count = len(depth)
        jump = np.abs(np.diff(depth)) / np.maximum(np.minimum(depth[:-1], depth[1:]), 1e-6)
        cut = np.flatnonzero(jump > self.edge)
        starts = np.concatenate([[0], cut + 1])
        stops = np.concatenate([cut, [count - 1]])
        runs = list(zip(starts.tolist(), stops.tolist(), strict=True))
        # a receptor alone between edges whose distance lies between its neighbours' is a mix
        mix = [False] * len(runs)
        for j, (a, b) in enumerate(runs):
            if a == b and 0 < j < len(runs) - 1:
                before, after = depth[runs[j - 1][1]], depth[runs[j + 1][0]]
                mix[j] = min(before, after) < depth[a] < max(before, after)
        kept = [j for j in range(len(runs)) if not mix[j]]
        origin = eye.origin(at, heading)
        facing = eye.facing(heading)
        out: list[Figure] = []
        for i, j in enumerate(kept):
            a, b = runs[j]
            sides = []
            for step, end, nb in ((-1, a, kept[i - 1] if i > 0 else None),
                                  (1, b, kept[i + 1] if i + 1 < len(kept) else None)):
                if nb is None:
                    sides.append(None)  # the field's edge
                    continue
                other = runs[nb][1] if step < 0 else runs[nb][0]
                sides.append(bool(depth[end] < depth[other]))
            if not any(s is True for s in sides):
                continue  # what lies behind: no thing
            whole = all(s is True for s in sides)
            # where each edge falls: inside a mix receptor, by its distance
            angles = []
            for step, end in ((-1, a), (1, b)):
                boundary = eye.edges[a] if step < 0 else eye.edges[b + 1]
                m = end + step
                nb_run = j + step
                if 0 <= nb_run < len(runs) and mix[nb_run] and 0 <= m < count:
                    far_run = runs[nb_run + step] if 0 <= nb_run + step < len(runs) else None
                    far = depth[far_run[1] if step < 0 else far_run[0]] if far_run else depth[m]
                    span = far - depth[end]
                    f = float(np.clip((far - depth[m]) / span, 0.0, 1.0)) if span > 0 else 0.0
                    boundary = boundary + step * f * eye.widths[m]
                angles.append(boundary)
            rays = facing + np.array(angles)
            ends = origin[None, :] + np.array([depth[a], depth[b]])[:, None] * np.stack(
                [np.cos(rays), np.sin(rays)], axis=-1)
            width = float(np.linalg.norm(ends[1] - ends[0]))
            inner = slice(a + 1, b) if b - a >= 2 else slice(a, b + 1)
            figure = Figure(eye=k, low=a, high=b, ends=ends, place=ends.mean(axis=0),
                            width=width, whole=whole, full=b - a + 1,
                            rgb=rgb[inner].mean(axis=0))
            if whole and figure.full >= self.least and width > 0:
                figure.bend = self._bend(eye, origin, facing, depth, a, b, ends)
                figure.form = self.form_code(width, figure.bend)
                figure.tint = self.tint_code(figure.rgb)
                if figure.bend > 0:
                    # a thing that bulges toward the eye reaches behind its front by its width
                    # times its bend over four (a disc's radius): its middle lies there
                    middle = 0.5 * (angles[0] + angles[1])
                    front = float(np.interp(middle, eye.centers[a: b + 1], depth[a: b + 1]))
                    back = width * min(figure.bend, BEND[1]) / 4.0
                    figure.place = origin + (front + back) * np.array(
                        [math.cos(facing + middle), math.sin(facing + middle)])
            out.append(figure)
        return out

    @staticmethod
    def _bend(eye: Eye, origin: np.ndarray, facing: float, depth: np.ndarray, a: int, b: int,
              ends: np.ndarray) -> float:
        """How a thing bends: its receptors' rays and distances as points on the plane, in the
        frame of the chord between its edges (across, and toward the eye), fitted by a
        parabola; its curvature times the width, positive where the middle bulges toward the
        eye. The receptors at its two ends are left out where three or more lie between."""
        span = np.arange(a + 1, b) if b - a >= 4 else np.arange(a, b + 1)
        rays = facing + eye.centers[span]
        points = origin[None, :] + depth[span, None] * np.stack([np.cos(rays), np.sin(rays)],
                                                                axis=1)
        middle = ends.mean(axis=0)
        across = ends[1] - ends[0]
        width = float(np.linalg.norm(across))
        across = across / width
        toward = origin - middle
        toward = toward - (toward @ across) * across
        toward = toward / max(float(np.linalg.norm(toward)), 1e-12)
        x = (points - middle) @ across
        y = (points - middle) @ toward
        if len(x) < 3:
            return 0.0
        c2 = float(np.polyfit(x, y, 2)[0])
        return -2.0 * c2 * width

    def find(self, images: Any, at: Any, heading: float) -> list[Figure]:
        """Every thing the eyes find, a thing both eyes see once (the one seen whole, or with
        more receptors)."""
        images = np.asarray(images, dtype=np.float64).reshape(len(self.eyes), -1, 4)
        at = np.asarray(at, dtype=np.float64).reshape(2)
        found = [f for k in range(len(self.eyes)) for f in self._figures(k, images[k], at,
                                                                          heading)]
        found.sort(key=lambda f: (not f.whole, -f.full))
        kept: list[Figure] = []
        for figure in found:
            if all(np.linalg.norm(figure.place - other.place) > self.fuse or (
                    other.eye == figure.eye) for other in kept):
                kept.append(figure)
        return kept

    def recognise(self, figures: list[Figure], *, learn: bool = True) -> list[Figure]:
        """The kind and colour of each thing seen whole and well: the categories settle on its
        form and tint (and take it as a lesson with ``learn``)."""
        for figure in figures:
            if figure.form is not None and figure.tint is not None:
                figure.kind = self.kinds.see(figure.form, learn=learn)
                figure.colour = self.colours.see(figure.tint, learn=learn)
        return figures

    def lit(self, images: Any, at: Any, heading: float) -> Any:
        """The places the eyes see: a function of ``(points, 2)`` places, true where a place
        lies in an eye's field and ``near`` short of how far that eye's ray toward it reached."""
        images = np.asarray(images, dtype=np.float64).reshape(len(self.eyes), -1, 4)
        at = np.asarray(at, dtype=np.float64).reshape(2)
        views = [(eye, eye.origin(at, heading), eye.facing(heading), images[k][:, 3])
                 for k, eye in enumerate(self.eyes)]

        def seen(points: np.ndarray) -> np.ndarray:
            p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            out = np.zeros(len(p), dtype=bool)
            for eye, origin, facing, depth in views:
                d = p - origin
                a = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - facing)))
                inside = np.abs(a) <= eye.field
                reach = np.interp(a, eye.centers, depth)
                out |= inside & (np.linalg.norm(d, axis=1) < reach - self.near)
            return out

        return seen

    def see(self, images: Any, at: Any, heading: float, *, learn: bool = True
            ) -> tuple[list[tuple[int, np.ndarray, dict[str, int]]], Any]:
        """One moment of sight: the reports of the things recognised, ``(kind, place,
        {"colour": colour})``, and the places the eyes see."""
        figures = self.recognise(self.find(images, at, heading), learn=learn)
        self.last = figures
        self.unknown += [f.place.copy() for f in figures if f.kind is None]
        self.work["moments"] += 1
        self.work["figures"] += len(figures)
        reports = [(int(f.kind), f.place.copy(), {"colour": int(f.colour)}) for f in figures
                   if f.kind is not None and f.colour is not None]
        return reports, self.lit(images, at, heading)

    def look_at(self, place: Any, at: Any, heading: float) -> float:
        """Where the head turns to look at a place: the heading that brings it to the middle of
        the eye that sees it nearer its middle from ``heading``."""
        d = np.asarray(place, dtype=np.float64).reshape(2) - np.asarray(at, dtype=np.float64)
        toward = math.atan2(float(d[1]), float(d[0]))

        def off(eye: Eye) -> float:
            return abs(math.remainder(toward - eye.facing(heading), 2 * math.pi))

        eye = min(self.eyes, key=off)
        return math.remainder(toward - eye.axis, 2 * math.pi)

    def parameters(self) -> int:
        """The retina's learned parameters: its two category layers' efficacies."""
        return self.kinds.parameters() + self.colours.parameters()


__all__ = ["BEND", "LIGHT", "SIZE", "Categories", "Eye", "Figure", "Retina", "parrot_eyes"]
