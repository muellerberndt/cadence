"""The two eyes' images: the scene as each eye sees it, one degree a pixel.

Each eye looks out to its side of the head: its axis 75 degrees from the beak, its field 87.5
degrees either side of the axis, so the two fields overlap 25 degrees in front of the beak and
leave 35 degrees behind the head unseen. The image of an eye is 175 columns by 100 rows, one
degree a pixel, in the head's frame: column 0 is the eye's rear edge and the high columns look
toward the beak (the two eyes mirror each other); row 0 is 50 degrees above the head's level and
row 99 is 50 degrees below it. A pixel is the mean of 2 x 2 rays over its patch.

The renderer casts rays: the table, the floor, the walls and the ceiling in their flat colours,
and the things as stand-ins of their kinds (``kinds.json``) lit by one light from above, with no
shadows; the room behind a pixel is read on the pixel's middle ray, the things on its 2 x 2
rays. The same file renders the eyes in the chamber and on the page, so the brain sees the same
images in both. The images carry no distance and no label; ``Sight.truth`` keeps, for the
chamber's readings only, which thing each pixel shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .world import EYE_AXIS, EYE_FIELD, Eyes, Scene, stand_ins

WIDTH, HEIGHT, SAMPLES = 175, 100, 2
TOP = 50.0  # degrees: the top row's upper edge above the head's level


def columns(eye: int, width: int = WIDTH) -> np.ndarray:
    """The head azimuth (radians, + left) at the middle of each column of an eye (0 left, 1
    right)."""
    rear = math.degrees(EYE_AXIS + EYE_FIELD)
    step = 2.0 * math.degrees(EYE_FIELD) / width
    middle = (np.arange(width) + 0.5) * step
    out = rear - middle if eye == 0 else -rear + middle
    return np.asarray(np.radians(out))


def rows(height: int = HEIGHT) -> np.ndarray:
    """The elevation (radians, + up) at the middle of each row."""
    step = 2.0 * TOP / height
    return np.asarray(np.radians(TOP - (np.arange(height) + 0.5) * step))


def directions(azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Unit directions in the head's frame (x left, y up, z beak) for broadcastable azimuths
    and elevations."""
    a, e = np.broadcast_arrays(np.asarray(azimuth, dtype=np.float64),
                               np.asarray(elevation, dtype=np.float64))
    return np.stack([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], axis=-1)


@dataclass
class Sight:
    """One moment's images: ``images`` ``(2, height, width, 3)`` uint8 (left, right) and the
    chamber's ``truth``: ``ids`` ``(2, height, width)`` the index of the thing each pixel's first
    ray meets (-1 none) and ``coverage`` ``(2, things)`` the pixels of each thing (its rays over
    the rays a pixel)."""

    images: np.ndarray
    ids: np.ndarray
    coverage: np.ndarray

    @property
    def left(self) -> np.ndarray:
        return np.asarray(self.images[0])

    @property
    def right(self) -> np.ndarray:
        return np.asarray(self.images[1])


class Renderer:
    """Casts the eyes' rays. ``width`` and ``height`` in pixels over the eye's field (175 by
    100 degrees), ``samples`` rays a pixel along each axis."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, samples: int = SAMPLES,
                 table: dict[str, Any] | None = None):
        self.width, self.height, self.samples = int(width), int(height), int(samples)
        self.table = stand_ins() if table is None else table
        s = self.samples
        a_step = 2.0 * EYE_FIELD / self.width
        e_step = math.radians(2.0 * TOP) / self.height
        offsets = (np.arange(s) + 0.5) / s - 0.5  # within a pixel, in pixels
        self._local = []
        self._centre = []  # each pixel's middle ray: the room behind is read once a pixel
        for eye in (0, 1):
            a = columns(eye, self.width)
            # a column's rays spread over its patch; the left eye's azimuth falls with the column
            sign = -1.0 if eye == 0 else 1.0
            sub_a = a[:, None] + sign * offsets[None, :] * a_step  # (width, s)
            sub_e = rows(self.height)[:, None] - offsets[None, :] * e_step  # (height, s)
            grid_a = sub_a[None, :, None, :]  # (1, width, 1, s)
            grid_e = sub_e[:, None, :, None]  # (height, 1, s, 1)
            d = directions(grid_a, grid_e)  # (height, width, s, s, 3)
            self._local.append(d.reshape(-1, 3))
            self._centre.append(directions(a[None, :], rows(self.height)[:, None]).reshape(-1, 3))
        room = self.table["room"]
        self._table_rect = (room["table"]["x"], room["table"]["z"])
        self._albedo = {name: np.asarray(room[name]["albedo"], dtype=np.float64)
                        for name in ("table", "floor", "walls", "ceiling")}
        self._floor, self._ceiling = float(room["floor"]["y"]), float(room["ceiling"]["y"])
        self._walls = float(room["walls"]["half"])
        light = self.table["light"]
        direction = np.asarray(light["direction"], dtype=np.float64)
        self._light = direction / np.linalg.norm(direction)
        self._ambient, self._diffuse = float(light["ambient"]), float(light["diffuse"])

    def _background(self, origin: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The room behind every ray: the colour and the distance it meets."""
        n = len(d)
        far = np.full(n, np.inf)
        colour = np.zeros((n, 3))
        dy = d[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            up = (self._ceiling - origin[1]) / dy
            down = (self._floor - origin[1]) / dy
            table = (0.0 - origin[1]) / dy
            wall_x = (np.sign(d[:, 0]) * self._walls - origin[0]) / d[:, 0]
            wall_z = (np.sign(d[:, 2]) * self._walls - origin[2]) / d[:, 2]
        wall = np.where(np.isfinite(wall_x) & (wall_x > 0), wall_x, np.inf)
        wall = np.minimum(wall, np.where(np.isfinite(wall_z) & (wall_z > 0), wall_z, np.inf))
        vertical = np.where(dy > 0, up, np.where(dy < 0, down, np.inf))
        vertical = np.where(np.isfinite(vertical) & (vertical > 0), vertical, np.inf)
        hit_wall = wall < vertical
        far = np.where(hit_wall, wall, vertical)
        colour[:] = np.where(hit_wall[:, None], self._albedo["walls"],
                             np.where((dy > 0)[:, None], self._albedo["ceiling"],
                                      self._albedo["floor"]))
        (x0, x1), (z0, z1) = self._table_rect
        on = (dy < 0) & np.isfinite(table) & (table > 0)
        tx = origin[0] + np.where(on, table, 0.0) * d[:, 0]
        tz = origin[2] + np.where(on, table, 0.0) * d[:, 2]
        on &= (tx >= x0) & (tx <= x1) & (tz >= z0) & (tz <= z1) & (table < far)
        far = np.where(on, table, far)
        colour[on] = self._albedo["table"]
        return colour, far

    def _hit(self, shape: str, o: np.ndarray, d: np.ndarray, half: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        """The nearest positive distance along each local ray to a stand-in centred at the
        origin (inf where it misses) and the local normal there."""
        n = len(d)
        t = np.full(n, np.inf)
        normal = np.zeros((n, 3))
        if shape in ("sphere", "ellipsoid"):
            oo, dd = o / half, d / half
            a = (dd * dd).sum(axis=1)
            b = (oo[None, :] * dd).sum(axis=1)
            c = float(oo @ oo) - 1.0
            disc = b * b - a * c
            ok = disc >= 0.0
            root = np.sqrt(np.where(ok, disc, 0.0))
            near = (-b - root) / a
            ok &= near > 0.0
            t = np.where(ok, near, np.inf)
            p = oo[None, :] + np.where(ok, near, 0.0)[:, None] * dd
            normal = p / half
        elif shape == "cylinder":
            ox, oz, dx, dz = o[0] / half[0], o[2] / half[2], d[:, 0] / half[0], d[:, 2] / half[2]
            a = dx * dx + dz * dz
            b = ox * dx + oz * dz
            c = ox * ox + oz * oz - 1.0
            disc = b * b - a * c
            ok = (disc >= 0.0) & (a > 0.0)
            root = np.sqrt(np.where(ok, disc, 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                side = np.where(ok, (-b - root) / np.where(a > 0.0, a, 1.0), np.inf)
            y = o[1] + np.where(np.isfinite(side), side, 0.0) * d[:, 1]
            side = np.where((side > 0.0) & (np.abs(y) <= half[1]), side, np.inf)
            with np.errstate(divide="ignore", invalid="ignore"):
                cap_top = (half[1] - o[1]) / d[:, 1]
                cap_bottom = (-half[1] - o[1]) / d[:, 1]
            caps = np.full(n, np.inf)
            cap_sign = np.zeros(n)
            for cap, sign in ((cap_top, 1.0), (cap_bottom, -1.0)):
                good = np.isfinite(cap) & (cap > 0.0)
                cx = o[0] + np.where(good, cap, 0.0) * d[:, 0]
                cz = o[2] + np.where(good, cap, 0.0) * d[:, 2]
                good &= (cx / half[0]) ** 2 + (cz / half[2]) ** 2 <= 1.0
                better = good & (cap < caps)
                caps = np.where(better, cap, caps)
                cap_sign = np.where(better, sign, cap_sign)
            t = np.minimum(side, caps)
            p = o[None, :] + np.where(np.isfinite(t), t, 0.0)[:, None] * d
            on_side = side <= caps
            normal = np.where(on_side[:, None],
                              np.stack([p[:, 0] / half[0] ** 2, np.zeros(n),
                                        p[:, 2] / half[2] ** 2], axis=1),
                              np.stack([np.zeros(n), cap_sign, np.zeros(n)], axis=1))
        elif shape == "box":
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-half[None, :] - o[None, :]) / d
                t2 = (half[None, :] - o[None, :]) / d
            low = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
            high = np.where(np.isnan(t1), np.inf, np.maximum(t1, t2))
            enter = low.max(axis=1)
            leave = high.min(axis=1)
            ok = (leave >= enter) & (enter > 0.0)
            t = np.where(ok, enter, np.inf)
            axis = low.argmax(axis=1)
            normal = np.zeros((n, 3))
            normal[np.arange(n), axis] = -np.sign(d[np.arange(n), axis])
        else:
            raise ValueError(f"no stand-in shape {shape!r}")
        return t, normal

    def render(self, scene: Scene, eyes: Eyes, *, cover: str | None = None) -> Sight:
        """Both eyes' images for the scene and the head's pose; ``cover`` ("left" or "right")
        gives that eye a dark image."""
        s2 = self.samples * self.samples
        things = scene.things
        images = np.zeros((2, self.height, self.width, 3), dtype=np.uint8)
        ids = np.full((2, self.height, self.width), -1, dtype=np.int16)
        coverage = np.zeros((2, len(things)))
        for eye in (0, 1):
            if cover == ("left", "right")[eye]:
                continue
            origin = eyes.places[eye]
            d = self._local[eye] @ eyes.rotation.T  # world directions
            colour, far = self._background(origin, self._centre[eye] @ eyes.rotation.T)
            colour, far = np.repeat(colour, s2, axis=0), np.repeat(far, s2)
            hit = np.full(len(d), -1, dtype=np.int64)
            for k, thing in enumerate(things):
                spec = self.table["kinds"][thing.kind]
                half = thing.half(self.table)
                center = thing.center(self.table)
                to = center - origin
                dist = float(np.linalg.norm(to))
                reach = float(np.linalg.norm(half))
                if dist <= reach:
                    continue  # the eye inside a thing's bounds sees nothing of it
                near = np.nonzero(d @ (to / dist) > math.sqrt(1.0 - (reach / dist) ** 2))[0]
                if len(near) == 0:
                    continue
                turn = np.array([[math.cos(thing.yaw), 0.0, math.sin(thing.yaw)],
                                 [0.0, 1.0, 0.0],
                                 [-math.sin(thing.yaw), 0.0, math.cos(thing.yaw)]])
                local_o = turn.T @ (origin - center)
                local_d = d[near] @ turn
                t, normal = self._hit(str(spec["shape"]), local_o, local_d, half)
                front = t < far[near]
                if not front.any():
                    continue
                rows_hit = near[front]
                world_n = normal[front] @ turn.T
                world_n /= np.linalg.norm(world_n, axis=1, keepdims=True)
                light = self._ambient + self._diffuse * np.maximum(world_n @ self._light, 0.0)
                albedo = np.asarray(spec["albedo"], dtype=np.float64)
                colour[rows_hit] = np.minimum(albedo[None, :] * light[:, None], 1.0)
                far[rows_hit] = t[front]
                hit[rows_hit] = k
            grid = colour.reshape(self.height, self.width, s2, 3).mean(axis=2)
            images[eye] = np.floor(grid * 255.0 + 0.5).astype(np.uint8)
            by_pixel = hit.reshape(self.height, self.width, s2)
            ids[eye] = by_pixel[:, :, 0]
            if len(things):
                counts = np.bincount(hit[hit >= 0], minlength=len(things))
                coverage[eye] = counts[: len(things)] / s2
        return Sight(images, ids, coverage)


_SHARED: dict[tuple[int, int, int], Renderer] = {}


def render(scene: Scene, eyes: Eyes, *, cover: str | None = None, width: int = WIDTH,
           height: int = HEIGHT, samples: int = SAMPLES) -> Sight:
    """Both eyes' images (``Renderer.render``), with one renderer kept a size."""
    key = (int(width), int(height), int(samples))
    if key not in _SHARED:
        _SHARED[key] = Renderer(width, height, samples)
    return _SHARED[key].render(scene, eyes, cover=cover)


__all__ = ["HEIGHT", "SAMPLES", "TOP", "WIDTH", "Renderer", "Sight", "columns", "directions",
           "render", "rows"]
