"""Early vision: the room learned as ground, things found as figures, and the two eyes compared.

- **Ground, learned.** The eye counts the colours it sees over its life. The colours that fill a
  large share of everything seen (the table, the walls, the floor, the ceiling) become the
  ground, and so do the mixtures of two such colours (a pixel on the edge between the table and
  a wall). A pixel's figure weight is how far its colour lies from the ground, from 0 to 1.
- **Figures.** The connected pieces of figure pixels, each with its weighted middle (the head's
  azimuth and elevation of it), its size, its thickness and its colour. A piece thinner than a
  stroke (an edge left between surfaces) is no figure.
- **Disparity units** (born wiring, the binocular cells of the visual Wulst): over the region
  both eyes see, each unit compares the left eye's edges of a figure with the right eye's edges
  shifted by its own disparity, and answers with how well they match. A thing nearer the eyes
  is seen farther to the right by the left eye and farther to the left by the right eye: its
  disparity is larger. The units carry no distance; how far a disparity is is learned
  (``cadence.parrot.distance``).

All of it is the eye's own wiring and the program beside it; nothing in it says what or where a
thing is or how far.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .eyes import HEIGHT, TOP, WIDTH, columns

OVERLAP = 25  # columns: the azimuths both eyes see, -12.5 to 12.5 degrees
MARGIN = 12  # columns beyond the overlap on either side kept in the binocular grid


def components(mask: np.ndarray) -> np.ndarray:
    """Labels of the 4-connected pieces of a ``(rows, columns)`` mask, numbered 0, 1, ... in the
    order of their first pixel, -1 outside the mask. The mask's runs along each row are joined
    where they touch a run of the row below."""
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    rows_s, starts = np.nonzero(step == 1)
    rows_e, ends = np.nonzero(step == -1)  # one past the run's last column
    out = np.full((h, w), -1, dtype=np.int64)
    count = len(starts)
    if count == 0:
        return out
    parent = np.arange(count)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    first = np.searchsorted(rows_s, np.arange(h + 1))
    for r in range(h - 1):
        a0, a1, b1 = first[r], first[r + 1], first[r + 2]
        if a0 == a1 or a1 == b1:
            continue
        sa, ea = starts[a0:a1], ends[a0:a1]
        sb, eb = starts[a1:b1], ends[a1:b1]
        touch = (sa[:, None] < eb[None, :]) & (sb[None, :] < ea[:, None])
        for i, j in zip(*np.nonzero(touch), strict=True):
            ri, rj = root(int(a0 + i)), root(int(a1 + j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([root(i) for i in range(count)])
    _, label = np.unique(roots, return_inverse=True)
    lengths = ends - starts
    cols = np.repeat(starts - np.repeat(np.cumsum(lengths) - lengths, 1), lengths) + np.arange(
        int(lengths.sum()))
    out[np.repeat(rows_s, lengths), cols] = np.repeat(label, lengths)
    return out


def _cover(edge: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> float:
    """How much of an edge pixel a thing covers, from its colour between the thing's colour
    (``inner``) and what lies behind it (``outer``), from 0 to 1."""
    span = inner - outer
    norm = float(span @ span)
    if norm < 1e-6:
        return 0.5
    return float(np.clip((edge - outer) @ span / norm, 0.0, 1.0))


def _blur(plane: np.ndarray) -> np.ndarray:
    """A plane blurred along its columns by (1/4, 1/2, 1/4)."""
    out = 0.5 * plane
    out[:, 1:] += 0.25 * plane[:, :-1]
    out[:, :-1] += 0.25 * plane[:, 1:]
    return out


@dataclass
class Figure:
    """A thing found in one eye's image (``eye`` 0 left, 1 right): its pixels' figure weight
    summed (``area``), its weighted middle in pixels (``col``, ``row``) and in the head's frame
    (``azimuth``, ``elevation``, radians), its extent in degrees, its ``thickness`` (area over its
    longer extent), its colour (``rgb``, the mean of its pixels well inside it) and whether it
    touches its image's side (``edge``). ``disparity`` is the disparity units' answer for a figure
    in the region both eyes see, ``pair`` the other eye's figure it was matched with and ``truth``
    the chamber's thing under it (-1 none; never read by the brain)."""

    eye: int
    label: int
    area: float
    col: float
    row: float
    azimuth: float
    elevation: float
    width: float
    height: float
    thickness: float
    rgb: np.ndarray
    edge: bool
    disparity: np.ndarray | None = None
    pair: int | None = None
    truth: int = -1


class Ground:
    """The colours of the room, learned from how much of everything seen they fill.

    ``bins`` levels a channel; a bin that holds ``share`` of all the pixels seen is ground. Bins
    of ground within 1.5 bins of each other are one colour of ground (their weighted mean)."""

    def __init__(self, *, bins: int = 16, share: float = 0.02, soft: float = 0.12):
        self.bins, self.share, self.soft = int(bins), float(share), float(soft)
        self.counts = np.zeros(self.bins ** 3)
        self.sums = np.zeros((self.bins ** 3, 3))
        self.seen = 0.0
        self._colours: np.ndarray | None = None

    def _bin(self, rgb: np.ndarray) -> np.ndarray:
        q = np.minimum((rgb * self.bins).astype(np.int64), self.bins - 1)
        return np.asarray((q[..., 0] * self.bins + q[..., 1]) * self.bins + q[..., 2])

    def learn(self, rgb: np.ndarray) -> None:
        """Count ``(..., 3)`` colours in [0, 1]."""
        flat = rgb.reshape(-1, 3)
        b = self._bin(flat)
        self.counts += np.bincount(b, minlength=len(self.counts))
        for c in range(3):
            self.sums[:, c] += np.bincount(b, weights=flat[:, c], minlength=len(self.counts))
        self.seen += len(flat)
        self._colours = None

    def colours(self) -> np.ndarray:
        """The colours of ground, ``(k, 3)``."""
        if self._colours is None:
            if self.seen == 0:
                self._colours = np.zeros((0, 3))
            else:
                held = np.nonzero(self.counts >= self.share * self.seen)[0]
                means = self.sums[held] / self.counts[held, None]
                order = np.argsort(-self.counts[held])
                merged: list[tuple[np.ndarray, float]] = []
                for k in order:
                    for m, (colour, weight) in enumerate(merged):
                        if np.abs(colour - means[k]).max() < 1.5 / self.bins:
                            total = weight + self.counts[held[k]]
                            merged[m] = ((colour * weight + means[k] * self.counts[held[k]])
                                         / total, total)
                            break
                    else:
                        merged.append((means[k], float(self.counts[held[k]])))
                self._colours = np.array([c for c, _ in merged]).reshape(-1, 3)
        return self._colours

    def weight(self, rgb: np.ndarray) -> np.ndarray:
        """Each pixel's figure weight: its colour's distance from the ground (a colour of it, or a
        mixture of two), over ``soft``, at most 1. Every pixel is figure while no ground is
        known."""
        colours = self.colours()
        flat = rgb.reshape(-1, 3)
        if len(colours) == 0:
            return np.ones(rgb.shape[:-1])
        near = np.full(len(flat), np.inf)
        for i in range(len(colours)):
            near = np.minimum(near, np.abs(flat - colours[i]).sum(axis=1))
        # a pixel of a ground colour itself is ground; the rest are read against the mixtures
        rest = np.nonzero(near > 0.02)[0]
        if len(rest):
            some = flat[rest]
            best = np.full(len(rest), np.inf)
            for i in range(len(colours)):
                best = np.minimum(best, np.linalg.norm(some - colours[i], axis=1))
                for j in range(i + 1, len(colours)):
                    a, b = colours[i], colours[j]
                    span = b - a
                    t = np.clip((some - a) @ span / max(float(span @ span), 1e-12), 0.0, 1.0)
                    best = np.minimum(best, np.linalg.norm(some - (a + t[:, None] * span),
                                                           axis=1))
            near[rest] = best
        near[near <= 0.02] = 0.0
        return np.asarray(np.minimum(near / self.soft, 1.0).reshape(rgb.shape[:-1]))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"counts": self.counts.copy(), "sums": self.sums.copy(),
                "seen": np.array(self.seen)}

    def take(self, arrays: dict[str, Any]) -> None:
        self.counts = np.asarray(arrays["counts"], dtype=np.float64).copy()
        self.sums = np.asarray(arrays["sums"], dtype=np.float64).copy()
        self.seen = float(arrays["seen"])
        self._colours = None


class Vision:
    """Both eyes' early vision: ``Ground``, the figures of each image and the disparity units.

    ``channels`` are the disparity units' preferred disparities (degrees); ``colour`` the
    colour difference over which a unit's match falls off; ``min_area``, ``min_thickness`` and
    ``max_area`` (pixels) bound a figure; ``match`` is the least answer of the best disparity
    unit that pairs a figure with the other eye's. ``dark`` is the brightness under which a pixel
    sees nothing (a covered eye). ``form`` "energy" answers with the edge correlation at each
    disparity; "edges" pairs the figures by it and then answers, as units tuned ``tuning``
    degrees wide, at the disparity where the thing's edges fall in the two eyes."""

    def __init__(self, *, channels: Any = None, colour: float = 0.12, min_area: float = 4.0,
                 min_thickness: float = 2.2, max_area: float = 3000.0, dark: float = 0.02,
                 match: float = 0.15, form: str = "edges", tuning: float = 0.5,
                 share: float = 0.02, soft: float = 0.12, width: int = WIDTH,
                 height: int = HEIGHT):
        self.channels = (np.arange(-2.0, 16.01, 0.5) if channels is None
                         else np.asarray(channels, dtype=np.float64))
        self.colour, self.dark, self.match = float(colour), float(dark), float(match)
        if form not in ("energy", "edges"):
            raise ValueError("the disparity units' form is 'energy' or 'edges'")
        self.form, self.tuning = form, float(tuning)
        self.min_area, self.min_thickness = float(min_area), float(min_thickness)
        self.max_area = float(max_area)
        self.width, self.height = int(width), int(height)
        self.ground = Ground(share=share, soft=soft)
        self._azimuth = [columns(0, self.width), columns(1, self.width)]
        self._deg = 2.0 * TOP / self.height  # degrees a row
        self._col_deg = float(np.degrees(abs(self._azimuth[0][1] - self._azimuth[0][0])))
        # the binocular grid: head azimuths -(12.5 + MARGIN) .. +(12.5 + MARGIN), one a degree
        self.grid = np.arange(-(OVERLAP / 2 + MARGIN) + 0.5, OVERLAP / 2 + MARGIN, 1.0)

    @property
    def units(self) -> int:
        """The disparity units: one a channel."""
        return len(self.channels)

    def _to_grid(self, eye: int, plane: np.ndarray) -> np.ndarray:
        """An eye's ``(rows, columns)`` plane on the binocular grid (azimuth ascending), zero
        where the eye does not see."""
        a = np.degrees(self._azimuth[eye])
        out = np.zeros((plane.shape[0], len(self.grid)))
        index = np.rint(a - self.grid[0]).astype(np.int64)
        keep = (index >= 0) & (index < len(self.grid))
        out[:, index[keep]] = plane[:, keep]
        return out

    def figures(self, eye: int, rgb: np.ndarray, weight: np.ndarray,
                ids: np.ndarray | None = None) -> tuple[list[Figure], np.ndarray]:
        """The figures of one eye's image and the figure of every pixel (-1 none)."""
        mask = weight > 0.25
        labels = components(mask)
        final = np.full(labels.shape, -1, dtype=np.int64)
        if not mask.any():
            return [], final
        keys, inverse = np.unique(labels[mask], return_inverse=True)
        count = len(keys)
        r, c = np.nonzero(mask)
        w = weight[mask]
        area = np.bincount(inverse, weights=w, minlength=count)
        col_sum = np.bincount(inverse, weights=w * c, minlength=count)
        row_sum = np.bincount(inverse, weights=w * r, minlength=count)
        c_low = np.full(count, self.width)
        c_high = np.full(count, -1)
        r_low = np.full(count, self.height)
        r_high = np.full(count, -1)
        np.minimum.at(c_low, inverse, c)
        np.maximum.at(c_high, inverse, c)
        np.minimum.at(r_low, inverse, r)
        np.maximum.at(r_high, inverse, r)
        width = (c_high - c_low + 1).astype(np.float64)
        height = (r_high - r_low + 1).astype(np.float64)
        thickness = area / np.maximum(width, height)
        keep = np.nonzero((area >= self.min_area) & (area <= self.max_area)
                          & (thickness >= self.min_thickness))[0]
        out: list[Figure] = []
        a0 = float(self._azimuth[eye][0])
        step = float(self._azimuth[eye][1] - self._azimuth[eye][0])
        colours = rgb[mask]
        under_all = None if ids is None else ids[mask]
        for k in keep:
            members = inverse == k
            col = float(col_sum[k] / area[k])
            row = float(row_sum[k] / area[k])
            inside = members & (w > 0.9)
            rgb_mean = colours[inside].mean(axis=0) if inside.any() else colours[members].mean(
                axis=0)
            truth = -1
            if under_all is not None:
                under = under_all[members]
                under = under[under >= 0]
                if len(under):
                    truth = int(np.bincount(under).argmax())
            final[r[members], c[members]] = len(out)
            out.append(Figure(eye, len(out), float(area[k]), col, row, a0 + col * step,
                              math.radians(TOP - (row + 0.5) * self._deg),
                              float(width[k]) * self._col_deg, float(height[k]) * self._deg,
                              float(thickness[k]), np.asarray(rgb_mean, dtype=np.float64),
                              bool(c_low[k] == 0 or c_high[k] == self.width - 1), truth=truth))
        return out, final

    def see(self, images: np.ndarray, *, learn: bool = True,
            ids: np.ndarray | None = None) -> list[Figure]:
        """Both eyes' figures for ``(2, rows, columns, 3)`` uint8 images; with ``learn`` the
        ground counts what was seen. A dark eye (covered) sees nothing and teaches nothing."""
        rgb = np.asarray(images, dtype=np.float64) / 255.0
        lit = [float(rgb[e].mean()) > self.dark for e in (0, 1)]
        if learn:
            for e in (0, 1):
                if lit[e]:
                    self.ground.learn(rgb[e])
        planes: list[tuple[np.ndarray, np.ndarray, list[Figure]]] = []
        for e in (0, 1):
            if not lit[e]:
                planes.append((np.zeros(rgb.shape[1:3]), np.full(rgb.shape[1:3], -1), []))
                continue
            weight = self.ground.weight(rgb[e])
            weight = np.where(rgb[e].max(axis=-1) > self.dark, weight, 0.0)
            found, labels = self.figures(e, rgb[e], weight, None if ids is None else ids[e])
            planes.append((weight, labels, found))
        if lit[0] and lit[1]:
            self._compare(rgb, planes)
        offset = len(planes[0][2])
        for f in planes[0][2]:
            if f.pair is not None:
                f.pair += offset  # a left figure's pair: an index into the list returned
        return planes[0][2] + planes[1][2]

    def _compare(self, rgb: np.ndarray, planes: list[tuple[np.ndarray, np.ndarray,
                                                           list[Figure]]]) -> None:
        """The disparity units over each left figure in the region both eyes see, and the
        right figure each is matched with."""
        (w_l, lab_l, figs_l), (w_r, lab_r, figs_r) = planes
        on_grid_l = self._to_grid(0, np.ones(lab_l.shape))[0] > 0
        on_grid_r = self._to_grid(1, np.ones(lab_r.shape))[0] > 0
        lab_l_grid = self._to_grid(0, lab_l.astype(np.float64) + 1.0) - 1.0
        lab_r_grid = self._to_grid(1, lab_r.astype(np.float64) + 1.0) - 1.0
        w_l_grid, w_r_grid = self._to_grid(0, w_l), self._to_grid(1, w_r)
        x = np.arange(len(self.grid), dtype=np.float64)
        for k, fig in enumerate(figs_l):
            rows_k = np.nonzero((lab_l_grid == k).any(axis=1))[0]
            if len(rows_k) == 0:
                continue
            band = slice(int(rows_k[0]), int(rows_k[-1]) + 1)
            own = (lab_l_grid[band] == k).astype(np.float64)
            left = np.where(own > 0, w_l_grid[band], 0.0)
            # the right figures of the left figure's colour, each whole (its edges as seen)
            similar = np.zeros(len(figs_r) + 1)
            for j, other in enumerate(figs_r):
                similar[j] = math.exp(-0.5 * (float(np.linalg.norm(other.rgb - fig.rgb))
                                              / self.colour) ** 2)
            right = w_r_grid[band] * similar[lab_r_grid[band].astype(np.int64)]
            # edges along azimuth, the left figure's own only (one column around it)
            grow = own.copy()
            grow[:, 1:] = np.maximum(grow[:, 1:], own[:, :-1])
            grow[:, :-1] = np.maximum(grow[:, :-1], own[:, 1:])
            g_l = np.zeros_like(left)
            g_l[:, 1:] = np.diff(left, axis=1)
            g_l *= grow
            g_l[:, ~on_grid_l] = 0.0
            g_r = np.zeros_like(right)
            g_r[:, 1:] = np.diff(right, axis=1)
            g_r[:, ~on_grid_r] = 0.0
            g_l, g_r = _blur(g_l), _blur(g_r)  # no pull toward whole-pixel shifts
            energy = float((g_l ** 2).sum())
            if energy <= 1e-9 or float((g_r ** 2).sum()) <= 1e-9:
                continue
            answer = np.zeros(len(self.channels))
            for i, delta in enumerate(self.channels):
                # the right edges at azimuth a + delta, sampled at every grid column a
                at = x + float(delta)
                low = np.floor(at).astype(np.int64)
                frac = at - low
                ok = (low >= 0) & (low + 1 < len(self.grid))
                shifted = np.zeros_like(g_r)
                shifted[:, ok] = ((1.0 - frac[ok]) * g_r[:, low[ok]]
                                  + frac[ok] * g_r[:, low[ok] + 1])
                answer[i] = float((g_l * shifted).sum())
            answer = np.maximum(answer / energy, 0.0)
            if answer.max() < self.match:
                continue
            fig.disparity = answer
            best = float(self.channels[int(answer.argmax())])
            # the right figure under the left figure's pixels shifted by the best disparity
            at = np.rint(x + best).astype(np.int64)
            ok = (at >= 0) & (at < len(self.grid))
            values = lab_r_grid[band][:, at[ok]][own[:, ok] > 0]
            under = values[values >= 0].astype(np.int64)
            if len(under):
                j = int(np.bincount(under).argmax())
                if self.form == "edges":
                    found = self._edges(rgb, lab_l == k, lab_r == j, on_grid_l, on_grid_r)
                    if found is None:
                        continue
                    strength = min(1.0, float(answer.max()))
                    answer = strength * np.exp(-0.5 * ((self.channels - found) / self.tuning) ** 2)
                    fig.disparity = answer
                fig.pair = j
                figs_r[j].pair = k
                figs_r[j].disparity = answer

    def _edges(self, rgb: np.ndarray, own_l: np.ndarray, own_r: np.ndarray, seen_l: np.ndarray,
               seen_r: np.ndarray) -> float | None:
        """The disparity of one thing from where its edges fall in each eye, row by row: the
        median over every edge both eyes see whole. An edge falls inside its pixel by how much
        of the pixel the thing covers, read from the pixel's colour between the colour just
        inside the edge and the colour just outside it. None where no edge is seen by both."""
        grids = []
        for eye, own in ((0, own_l), (1, own_r)):
            colour = np.stack([self._to_grid(eye, rgb[eye][..., c]) for c in range(3)], axis=-1)
            grids.append((self._to_grid(eye, own.astype(np.float64)) > 0.5, colour))
        seen = (seen_l, seen_r)
        found: list[float] = []
        for row in range(grids[0][0].shape[0]):
            ends = []
            for eye in (0, 1):
                mask, colour = grids[eye][0][row], grids[eye][1][row]
                inside = np.nonzero(mask)[0]
                if len(inside) < 2:
                    break
                lo, hi = int(inside[0]), int(inside[-1])
                low = high = math.nan
                if lo > 0 and seen[eye][lo - 1] and mask[lo + 1]:
                    low = lo + 0.5 - _cover(colour[lo], colour[lo + 1], colour[lo - 1])
                if hi + 1 < len(mask) and seen[eye][hi + 1] and mask[hi - 1]:
                    high = hi - 0.5 + _cover(colour[hi], colour[hi - 1], colour[hi + 1])
                ends.append((low, high))
            if len(ends) < 2:
                continue
            for side in (0, 1):
                d = ends[1][side] - ends[0][side]
                if np.isfinite(d):
                    found.append(float(d))
        return float(np.median(found)) if found else None

    def settings(self) -> dict[str, Any]:
        return {"channels": self.channels.tolist(), "colour": self.colour,
                "min_area": self.min_area, "min_thickness": self.min_thickness,
                "max_area": self.max_area, "dark": self.dark, "match": self.match,
                "form": self.form, "tuning": self.tuning,
                "share": self.ground.share, "soft": self.ground.soft}


__all__ = ["MARGIN", "OVERLAP", "Figure", "Ground", "Vision", "components"]
