"""Learned physics on the map: how things move, learned by watching them move.

The map (``cadence.space.map.Map``) holds where things are. With physics it also predicts where a
thing it cannot see has gone, by how it has seen things move. Four numbers are learned, and
nothing else about movement is given:

- ``rest``: the share of moving things that come to rest in a moment;
- ``leave``: the share that leave their line in a moment (they slow or turn);
- ``ahead``: how far along its old line a thing that left goes, as a share of its speed;
- ``aside``: how fast it strays from where that puts it, as a share of its speed.

**What the brain sees of motion.** While a thing is in view, the reports of it are a run, and a
line through the latest ``line`` of them (least squares, gains of 1/n) gives its place and its
velocity. The surprise of each report (its distance from where the line put it, against the
line's own error) corrects both. A surprise beyond ``surprise`` standard deviations at two moments
in a row ends the line: it begins anew at the latest reports. Whether the thing moves at all is
the evidence of the line against a thing at rest.

**Lessons.** Every ``every`` moments of a run whose line fits, the brain imagines the thing out
of view from there, and for ``horizon`` moments compares the imagined line with the report: how
far short of its line the thing fell (metres, along the line) and how far aside it went. Each
lesson is explained by the outcomes that fit it (it kept its line; it came to rest at moment j;
it left its line at moment j), weighted by how likely each is; the expected counts and moments
of those outcomes give the four numbers (the expectation-maximisation step, online). Lessons
are taken only away from the plane's edge: a line that would leave the plane says nothing about
motion. Before ``prior`` moments of lessons the brain acts on rest: a fresh brain answers as the
map does.

**Out of view.** A thing that goes out of view while moving is held in three parts: the core,
the things that kept their line, a narrow bump on the line as wide as the line's own error; the
trail, the things that came to rest, on the still cells; and the tail, the things that left
their line, a bump that goes ``ahead`` from where they left and widens ``aside``. Each moment a
share ``rest`` of the core and of the tail comes to rest where it is and a share ``leave`` of
the core joins the tail; the core moves by its velocity and rests at the plane's edge. Sight
takes away what it sees of each part: the cells it sees of the trail, and of each bump the
share of it the camera sees, the bump then centred on what is left. The parts keep their sum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# a thing's motion record: what the brain knows of how it moves, carried at its cells
EST, VEL, PREV = slice(0, 2), slice(2, 4), slice(4, 6)
COUNT, LOUD, SURPRISE, FIT, IDENT, HIDDEN = 6, 7, 8, 9, 10, 11
SLOTS = 4
SLOT = 12  # each slot: x0, y0, vx0, vy0, n0, k
CORE = SLOT + 6 * SLOTS  # x, y, vx, vy, px, pxv, pv, mass
TAIL = CORE + 8  # x, y, variance, mass
FIELDS = TAIL + 4


def gains(n: int) -> tuple[float, float]:
    """The line's gains on a surprise after ``n`` reports: least squares through them."""
    return 2.0 * (2 * n - 1) / (n * (n + 1)), 6.0 / (n * (n + 1))


def errors(n: int, noise: float) -> tuple[float, float, float]:
    """The line's own error, an axis, after ``n`` reports of error ``noise``: of its place
    (m^2), of its place with its velocity (m^2 a moment) and of its velocity (m^2 a moment^2).
    ``k`` moments on the line's place is off by ``place + 2 k with + k^2 velocity``."""
    s2 = noise * noise
    if n < 2:
        return s2, 0.0, 0.0
    if n == 2:
        return s2, s2, 2.0 * s2
    return (s2 * 2.0 * (2 * n - 1) / (n * (n + 1)), s2 * 6.0 / (n * (n + 1)),
            s2 * 12.0 / (n * (n * n - 1)))


def ahead_error(n: int, noise: float, k: Any) -> Any:
    """The line's error ``k`` moments on, an axis (m^2)."""
    px, pxv, pv = errors(n, noise)
    return px + 2.0 * k * pxv + k * k * pv


class Physics:
    """How things move, learned by watching them: ``rest``, ``leave``, ``ahead`` and ``aside``
    (the numbers acted on) and ``estimate`` (the running estimate the lessons refine).

    Settings: ``noise`` (the camera's error, an axis, m), ``line`` (the reports a line runs
    through), ``surprise`` (standard deviations that end a line), ``settled`` (the mean squared
    surprise under which a line fits), ``every`` and ``horizon`` (a lesson's launch and length,
    moments), ``reports`` (the reports a line needs before it teaches) and ``prior`` (moments of
    rest counted before any lesson). ``learning`` takes lessons; a map that shares a physics
    with another may leave it off."""

    def __init__(self, *, noise: float = 0.01, line: int = 30, surprise: float = 3.0,
                 settled: float = 1.5, every: int = 8, horizon: int = 30, reports: int = 10,
                 prior: float = 500.0, learning: bool = True,
                 arrays: dict[str, Any] | None = None):
        self.noise, self.line, self.surprise = float(noise), int(line), float(surprise)
        self.settled, self.every, self.horizon = float(settled), int(every), int(horizon)
        self.reports, self.prior, self.learning = int(reports), float(prior), bool(learning)
        if self.line < 3 or self.every < 1 or self.horizon < 1 or self.reports < 3:
            raise ValueError("physics takes a line of 3 reports or more, every and horizon of 1 "
                             "moment or more and 3 reports or more before a lesson")
        # rest, leave, exposure, ahead (num, den), aside (num, den), lessons, speed^2, starts
        self.stats = np.zeros(10)
        self.estimate = np.array([0.02, 0.02, 0.5, 0.5])  # rest, leave, ahead, aside
        if arrays is not None:
            self.stats = np.asarray(arrays["stats"], dtype=np.float64).reshape(10).copy()
            self.estimate = np.asarray(arrays["estimate"], dtype=np.float64).reshape(4).copy()
        self.fit()

    # -- what is acted on

    @property
    def lessons(self) -> int:
        return int(self.stats[7])

    @property
    def taught(self) -> bool:
        """Whether any lesson was taken: an untaught physics leaves the map as it is."""
        return self.stats[2] > 0

    def fit(self) -> None:
        """The numbers acted on, from the lessons so far: ``prior`` moments of a thing at rest
        are counted with the lessons, so a fresh brain acts on rest."""
        rest_ev, leave_ev, exposure, num_a, den_a, num_w, den_w, _, speed2, starts = self.stats
        if exposure > 0:
            self.estimate[0] = float(np.clip(rest_ev / exposure, 1e-5, 0.5))
            self.estimate[1] = float(np.clip(leave_ev / exposure, 1e-5, 0.5))
        if den_a > 0:
            self.estimate[2] = float(np.clip(num_a / den_a, 0.0, 1.0))
        if den_w > 0:
            self.estimate[3] = float(np.clip(np.sqrt(num_w / den_w), 0.02, 2.0))
        counted = exposure + self.prior
        self.rest = float((rest_ev + self.prior) / counted) if counted > 0 else 1.0
        self.leave = float(leave_ev / counted) if counted > 0 else 0.0
        self.ahead, self.aside = float(self.estimate[2]), float(self.estimate[3])
        self.speed = float(np.sqrt(speed2 / starts)) if starts > 0 else None

    def moving(self, velocity: np.ndarray, n: int) -> float:
        """The share of a thing that moves: the evidence of its line (``velocity`` through ``n``
        reports) against a thing at rest, with velocities as large as the things watched."""
        if self.speed is None or n < 3:
            return 0.0
        pv = errors(min(int(n), self.line), self.noise)[2]
        v0 = self.speed ** 2 / 2.0  # an axis
        lbf = 0.5 * float(velocity @ velocity) / pv * (v0 / (v0 + pv)) - np.log1p(v0 / pv)
        return float(1.0 / (1.0 + np.exp(-np.clip(lbf, -50.0, 50.0))))

    # -- a thing in view

    def see(self, record: np.ndarray | None, place: np.ndarray, ident: float,
            inside: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """A report of a thing at ``place``: its motion record carried on, or a new one (a thing
        first seen, or seen again after it was out of view: its run begins anew). The line takes
        the surprise; its lessons are taken (``watch``); every ``every`` reports of a line that
        fits a lesson is launched. Gives the record to write at the report's cells."""
        r = np.asarray(place, dtype=np.float64).reshape(2)
        rec = np.zeros(FIELDS) if record is None else np.array(record, dtype=np.float64)
        if record is None or rec[IDENT] <= 0:
            rec[IDENT] = ident
        if record is None or rec[HIDDEN] > 0 or rec[COUNT] < 1:  # a run begins
            keep = rec[IDENT]
            rec[:] = 0.0
            rec[IDENT], rec[COUNT], rec[EST], rec[PREV], rec[FIT] = keep, 1.0, r, r, 1.0
            return rec
        slots = rec[SLOT:CORE].reshape(SLOTS, 6)
        est = rec[EST] + rec[VEL]  # where the line puts it now
        active = slots[:, 4] > 0
        if active.any():  # the lessons launched before: their imagined line against the report
            slots[active, 5] += 1.0
            self.watch(slots[active, 0:2], slots[active, 2:4], slots[active, 4], r,
                       slots[active, 5], inside)
            slots[slots[:, 5] >= self.horizon] = 0.0
        count = int(rec[COUNT])
        n = min(count, self.line)
        s2 = self.noise ** 2
        z2 = float(((r - est) ** 2).sum()) / (2.0 * (s2 + ahead_error(n, self.noise, 1)))
        rec[LOUD] = rec[LOUD] + 1.0 if z2 > self.surprise ** 2 else 0.0
        rec[SURPRISE] += (z2 - rec[SURPRISE]) / 4.0
        if z2 <= 1.0:
            rec[FIT] = count
        if rec[LOUD] >= 2:  # the line ends: it begins anew at the report before and this one
            launched = count - slots[:, 5]
            slots[(slots[:, 4] > 0) & (launched > rec[FIT])] = 0.0  # taught from a stale line
            est, rec[VEL], count = rec[PREV].copy(), 0.0, 1
            rec[LOUD], rec[SURPRISE], rec[FIT] = 0.0, 0.0, 1.0
        count += 1
        a, b = gains(min(count, self.line))
        e = r - est
        rec[EST] = est + a * e
        rec[VEL] = rec[VEL] + b * e
        rec[COUNT], rec[PREV] = count, r
        if count >= 4 and count % self.every == 0 and rec[SURPRISE] <= self.settled:
            free = np.flatnonzero(slots[:, 4] <= 0)
            at = int(free[0]) if len(free) else int(np.argmax(slots[:, 5]))
            slots[at] = [*rec[EST], *rec[VEL], min(count, self.line), 0.0]
        return rec

    # -- lessons

    def watch(self, starts: np.ndarray, velocities: np.ndarray, counts: np.ndarray,
              place: np.ndarray, horizons: np.ndarray, inside: tuple[np.ndarray, np.ndarray]
              ) -> int:
        """Lessons: things whose lines through ``counts`` reports put them at ``starts`` with
        ``velocities`` were seen at ``place`` ``horizons`` moments later. ``inside`` is the
        plane's low and high corners less its margin: a lesson whose line leaves it, or whose
        place lies within 3 cm of its edge, is no lesson. Gives the lessons taken."""
        if not self.learning or len(starts) == 0:
            return 0
        x0 = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
        v0 = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        n = np.asarray(counts, dtype=np.float64).reshape(-1)
        k = np.asarray(horizons, dtype=np.float64).reshape(-1)
        r = np.asarray(place, dtype=np.float64).reshape(2)
        lo, hi = inside
        s2 = self.noise ** 2
        pv = s2 * 12.0 / (n * (n * n - 1))  # the line's velocity error, an axis
        speed = np.sqrt((v0 ** 2).sum(1))
        end = x0 + k[:, None] * v0
        ok = (n >= self.reports) & (speed >= 3.0 * np.sqrt(pv)) & (k >= 1) & (k <= self.horizon)
        ok &= ((end >= lo) & (end <= hi)).all(1) & ((x0 >= lo) & (x0 <= hi)).all(1)
        ok &= bool(((r >= lo + 0.03) & (r <= hi - 0.03)).all())
        if not ok.any():
            return 0
        x0, v0, n, k, pv, speed = x0[ok], v0[ok], n[ok], k[ok], pv[ok], speed[ok]
        u = v0 / speed[:, None]
        e = r - x0 - k[:, None] * v0
        short = -(e * u).sum(1)
        side = e[:, 0] * -u[:, 1] + e[:, 1] * u[:, 0]
        line = s2 + s2 * (2.0 * (2 * n - 1) / (n * (n + 1)) + 12.0 * k / (n * (n + 1))
                          + 12.0 * k * k / (n * (n * n - 1)))  # the report's and the line's error
        self._expect(k, short, side, speed, line)
        self.fit()
        return int(ok.sum())

    def _expect(self, k: np.ndarray, short: np.ndarray, side: np.ndarray, speed: np.ndarray,
                line: np.ndarray) -> None:
        """The expectation step over a batch of lessons, added to the sufficient statistics.

        Outcomes: kept its line (no shortfall, aside only the line's error); came to rest at
        moment j (short by the k - j + 1 moments it did not move); left its line at moment j
        (goes ``ahead`` of its speed, strays ``aside`` of its speed a moment, on each axis)."""
        h, g, a, w = (float(x) for x in self.estimate)
        stay = max(1.0 - h - g, 1e-9)
        J = np.arange(1, self.horizon + 1, dtype=np.float64)
        left = np.maximum(k[:, None] - J[None] + 1.0, 0.0)  # moments not moved after j
        valid = J[None] <= k[:, None]
        sp = speed[:, None]
        var0 = line[:, None]

        def lg(x: Any, m: Any, v: Any) -> Any:
            return -0.5 * (x - m) ** 2 / v - 0.5 * np.log(2.0 * np.pi * v)

        before = np.log(stay) * (J - 1.0)[None]  # kept its line until moment j
        lk = np.log(stay) * k + lg(short, 0.0, line) + lg(side, 0.0, line)
        lr = np.where(valid, np.log(h) + before + lg(short[:, None], left * sp, var0)
                      + lg(side[:, None], 0.0, var0), -np.inf)
        vt = var0 + (w * sp * left) ** 2
        ll = np.where(valid, np.log(g) + before + lg(short[:, None], (1.0 - a) * left * sp, vt)
                      + lg(side[:, None], 0.0, vt), -np.inf)
        top = np.maximum(lk, np.maximum(lr.max(1), ll.max(1)))
        pk, pr, pl = np.exp(lk - top), np.exp(lr - top[:, None]), np.exp(ll - top[:, None])
        z = pk + pr.sum(1) + pl.sum(1)
        pk, pr, pl = pk / z, pr / z[:, None], pl / z[:, None]
        exposure = (pk * k).sum() + ((pr + pl) * J[None]).sum()
        m = left * sp
        res = side[:, None] ** 2 + (short[:, None] - (1.0 - a) * m) ** 2 - 2.0 * var0
        self.stats += np.array([
            pr.sum(), pl.sum(), exposure,
            (pl * m * (m - short[:, None])).sum(), (pl * m * m).sum(),
            (pl * np.maximum(res, 0.0)).sum() / 2.0, (pl * m * m).sum(),
            len(k), (speed ** 2).sum(), len(k)])

    # -- files

    def settings(self) -> dict[str, Any]:
        return {"noise": self.noise, "line": self.line, "surprise": self.surprise,
                "settled": self.settled, "every": self.every, "horizon": self.horizon,
                "reports": self.reports, "prior": self.prior}

    def arrays(self) -> dict[str, np.ndarray]:
        """What a file keeps of the physics: its lessons' statistics and its estimate."""
        return {"stats": self.stats.copy(), "estimate": self.estimate.copy()}

    def describe(self) -> dict[str, Any]:
        """The numbers acted on and the lessons behind them."""
        return {"rest": self.rest, "leave": self.leave, "ahead": self.ahead,
                "aside": self.aside, "speed": self.speed, "lessons": self.lessons}


__all__ = ["CORE", "COUNT", "EST", "FIELDS", "HIDDEN", "IDENT", "SLOT", "SLOTS", "TAIL", "VEL",
           "Physics", "ahead_error", "errors", "gains"]
