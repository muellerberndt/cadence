"""The tracking chamber: where is each thing at this step, in view and out of view?

A learner is given a scene step by step: what the camera reports of the things in view (a
position and what the thing looks like, with no name), what the ear hears, the arm's commands
and joint readings, and at step 0 the list of starting positions. At every step it answers one
position per thing, and the answer is scored against the true position as a distance in the
plane. A scene read from a file is scored as a generated one is.

    python benchmarks/tracking/tracking.py life --arm code --case other --founder 0 --out DIR
    python benchmarks/tracking/tracking.py score --scene benchmarks/tracking/scenes/example.json
    python benchmarks/tracking/tracking.py make --case other --kind stop --seed 3 --out scene.npz

Arms. Controls that learn nothing: ``last_seen`` (each thing where it was last reported),
``kalman`` (a constant-velocity Kalman filter per thing, its two noise terms fitted on the
training scenes), ``reckoning`` (the arm's true kinematics on the commands from the last report;
it knows the arm's geometry, which no learner is given) and ``encoders`` (the true kinematics on
the joint readings). Floors: ``default`` (a default brain on the readings as they come) and
``code`` (a default brain on population-coded readings); each answers two motor slots of
``BINS`` tuned positions. The matched LSTM and transformer are in ``controls.py``.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as w  # noqa: E402

BINS = 32  # tuned positions an axis in a floor's answer
UNSEEN = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 10**9))  # steps since last seen
PANELS = ("straight", "slow", "stop", "turn")
RATE = 0.90  # the rate an answer's area is stated at
AREA = float(np.sqrt(-2.0 * np.log(1.0 - RATE)))  # that area's radius, in standard deviations
SEEDS = {"train": 0, "test": 1, "panels": 2, "stays": 3, "heard": 4, "crossing": 5}


# -- what a learner is given


def given(scene: w.Scene) -> dict[str, np.ndarray]:
    """A scene of one thing as a learner is given it: ``seen`` and ``report`` per step (zeros
    while out of view), ``commands`` and ``joints``. Step 0 holds the starting position where
    the hand starts out of view."""
    assert scene.start is not None
    seen = scene.in_view[:, 0].copy()
    report = np.where(seen[:, None], scene.report[:, 0], 0.0)
    if not seen[0]:
        seen[0], report[0] = True, scene.start[0]
    return {"seen": seen, "report": report, "commands": scene.commands, "joints": scene.joints}


def detections(scene: w.Scene, t: int) -> np.ndarray:
    """Step ``t``'s reports as the camera gives them: ``(reports, 3)`` rows of x, y and what the
    thing looks like, ordered by look and then by x, so a row's place says nothing of which
    thing it is. Step 0 lists every thing at its starting position."""
    assert scene.start is not None and scene.what is not None
    if t == 0:
        rows = np.column_stack([scene.start, scene.what])
    else:
        seen = scene.in_view[t]
        rows = np.column_stack([scene.report[t][seen], scene.what[seen]])
    return np.asarray(rows[np.lexsort((rows[:, 0], rows[:, 2]))])


def sounds(scene: w.Scene, t: int) -> np.ndarray:
    """Step ``t``'s cues: ``(cues, 3)`` rows of direction, distance and what the thing looks
    like."""
    assert scene.cue is not None and scene.what is not None
    heard = np.isfinite(scene.cue[t]).all(axis=-1)
    return np.column_stack([scene.cue[t][heard], scene.what[heard]])


def assign(places: np.ndarray, reports: np.ndarray) -> list[tuple[int, int]]:
    """Pair ``(members, 2)`` places with ``(reports, 2)`` reports at the least summed distance:
    ``(member, report)`` pairs, as many as the fewer of the two."""
    m, n = len(places), len(reports)
    if m == 0 or n == 0:
        return []
    cost = np.linalg.norm(places[:, None] - reports[None], axis=-1)
    if m <= n:
        best = min(itertools.permutations(range(n), m),
                   key=lambda p: sum(cost[i, j] for i, j in enumerate(p)))
        return list(enumerate(best))
    best = min(itertools.permutations(range(m), n),
               key=lambda p: sum(cost[i, j] for j, i in enumerate(p)))
    return [(i, j) for j, i in enumerate(best)]


# -- controls that learn nothing


class Still:
    """A thing where it was last reported."""

    def __init__(self, start: np.ndarray, settings: dict[str, float]):
        self.place = np.asarray(start, dtype=np.float64).copy()
        self.radius = 0.0

    def predict(self) -> None:
        pass

    def see(self, report: np.ndarray, noise: float | None = None) -> None:
        self.place = np.asarray(report, dtype=np.float64).copy()


class ConstantVelocity:
    """A constant-velocity Kalman filter of one thing: position and velocity on each axis,
    process noise ``q`` and report noise ``r`` (a standard deviation)."""

    def __init__(self, start: np.ndarray, settings: dict[str, float]):
        self.q, self.r = float(settings["q"]), float(settings["r"])
        self.state = np.array([[start[0], 0.0], [start[1], 0.0]])  # per axis: position, velocity
        self.cov = np.array([[self.r**2, 0.0], [0.0, 0.03**2]])  # one covariance for both axes
        self.step = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.noise = self.q * np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]])

    @property
    def place(self) -> np.ndarray:
        return np.asarray(self.state[:, 0].copy())

    @property
    def radius(self) -> float:
        return AREA * float(np.sqrt(self.cov[0, 0]))

    def predict(self) -> None:
        self.state = self.state @ self.step.T
        self.cov = self.step @ self.cov @ self.step.T + self.noise

    def see(self, report: np.ndarray, noise: float | None = None) -> None:
        r = self.r if noise is None else noise
        gain = self.cov[:, 0] / (self.cov[0, 0] + r**2)
        self.state = self.state + np.outer(np.asarray(report) - self.state[:, 0], gain)
        self.cov = self.cov - np.outer(gain, self.cov[0])


def follow(scene: w.Scene, make: Any, settings: dict[str, float] | None = None,
           hear: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Run one filter per thing over a scene: answers ``(steps, things, 2)`` and the radius of
    each answer's area at ``RATE``. Reports of things that look alike go to the nearer places
    (``assign``); with ``hear`` a cue is taken as a report at the cue's own noise."""
    assert scene.start is not None and scene.what is not None
    settings = settings or {}
    filters = [make(scene.start[k], settings) for k in range(scene.things)]
    answers = np.zeros((scene.steps, scene.things, 2))
    radius = np.zeros((scene.steps, scene.things))
    looks = {int(look): np.flatnonzero(scene.what == look) for look in np.unique(scene.what)}
    for t in range(scene.steps):
        if t:
            for f in filters:
                f.predict()
            rows = detections(scene, t)
            for look, members in looks.items():
                seen = rows[rows[:, 2] == look, :2]
                places = np.array([filters[k].place for k in members])
                for i, j in assign(places, seen):
                    filters[members[i]].see(seen[j])
            if hear:
                for direction, distance, look in sounds(scene, t):
                    members = looks[int(look)]
                    point = w.cartesian(scene.geometry, np.array([direction, distance]))
                    places = np.array([filters[k].place for k in members])
                    k = members[int(np.argmin(np.linalg.norm(places - point, axis=1)))]
                    filters[k].see(point, noise=float(settings.get("cue", 0.08)) * distance)
        answers[t] = [f.place for f in filters]
        radius[t] = [f.radius for f in filters]
    return answers, radius


def fit_kalman(training: list[w.Scene]) -> dict[str, float]:
    """The process and report noise at which the filter's out-of-view error over ``training``
    is least: a search over a grid, on the training scenes only."""
    best: tuple[float, dict[str, float]] | None = None
    for q in np.logspace(-10, -2, 17):
        for r in (0.005, 0.01, 0.02, 0.04):
            settings = {"q": float(q), "r": float(r)}
            total, count = 0.0, 0
            for scene in training:
                answers, _ = follow(scene, ConstantVelocity, settings)
                out = ~scene.in_view
                out[0] = False
                total += float(np.linalg.norm(answers - scene.truth, axis=-1)[out].sum())
                count += int(out.sum())
            mean = total / max(count, 1)
            if best is None or mean < best[0]:
                best = (mean, settings)
    assert best is not None
    return best[1]


def reckoning(scene: w.Scene, encoders: bool = False) -> np.ndarray:
    """The arm's own hand from its true kinematics: the commands summed onto the joint angles
    of the last report (two joints), or with ``encoders`` the joint readings themselves. It
    knows the arm's lengths, base and bend, which no learner is given."""
    geometry = scene.geometry
    if encoders:
        return np.asarray(w.hand(geometry, scene.joints)[:, None, :])
    if geometry.joints != 2:
        raise ValueError("reckoning from a report is written for an arm of two joints")
    feed = given(scene)
    first, second = geometry.lengths
    bend = -1.0 if geometry.limits[1][1] <= 0 else 1.0
    angles = np.zeros(2)
    answers = np.zeros((scene.steps, 1, 2))
    for t in range(scene.steps):
        if feed["seen"][t]:
            x, y = feed["report"][t] - np.asarray(geometry.base)
            cosine = np.clip((x * x + y * y - first**2 - second**2) / (2 * first * second), -1, 1)
            elbow = bend * float(np.arccos(cosine))
            angles = np.array([np.arctan2(y, x) - np.arctan2(second * np.sin(elbow),
                                                             first + second * np.cos(elbow)),
                               elbow])
            answers[t, 0] = feed["report"][t]
        else:
            angles = angles + scene.commands[t]
            answers[t, 0] = w.hand(geometry, angles)
    return answers


# -- scores


def errors(scene: w.Scene, answers: np.ndarray) -> np.ndarray:
    """The distance from each answer to the true position: ``(steps, things)``."""
    return np.asarray(np.linalg.norm(np.asarray(answers) - scene.truth, axis=-1))


def table(scenes: list[w.Scene], answers: list[np.ndarray],
          radius: list[np.ndarray] | None = None) -> dict[str, Any]:
    """The hand's error over ``scenes``: in view and out of view, by steps since it was last
    seen, by panel (the out-of-view steps of each panel's scenes, a mean and a standard error
    over scenes), and with ``radius`` the share of steps whose area holds the true position."""
    err = [errors(s, a)[:, 0] for s, a in zip(scenes, answers, strict=True)]
    seen = [s.in_view[:, 0] for s in scenes]
    since = [s.unseen()[:, 0] for s in scenes]
    scored = [np.arange(s.steps) > 0 for s in scenes]  # step 0 is given
    e, v, u, ok = (np.concatenate(x) for x in (err, seen, since, scored))
    out: dict[str, Any] = {
        "scenes": len(scenes), "steps": int(ok.sum()),
        "in_view": _mean(e[ok & v]), "out_of_view": _mean(e[ok & ~v]),
        "share_out_of_view": float((ok & ~v).sum() / max(ok.sum(), 1)),
        "camera": _mean(np.concatenate([
            np.linalg.norm(s.report[:, 0] - s.truth[:, 0], axis=-1)[s.in_view[:, 0]]
            for s in scenes])),
        "by_unseen": {f"{lo}-{hi}" if hi < 10**9 else f"{lo}+":
                      {"error": _mean(e[ok & (u >= lo) & (u <= hi)]),
                       "steps": int((ok & (u >= lo) & (u <= hi)).sum())}
                      for lo, hi in UNSEEN},
    }
    panels: dict[str, Any] = {}
    for name in PANELS:
        means = [float(x[~f & o].mean()) for s, x, f, o in zip(scenes, err, seen, scored,
                                                              strict=True)
                 if w.panel(s) == name and (~f & o).any()]
        if means:
            panels[name] = {"error": float(np.mean(means)), "scenes": len(means),
                            "se": float(np.std(means, ddof=1) / np.sqrt(len(means)))
                            if len(means) > 1 else None}
    if panels:
        out["panels"] = panels
    if radius is not None:
        inside = np.concatenate([x <= r[:, 0] for x, r in zip(err, radius, strict=True)])
        out["held"] = {"in_view": _mean(inside[ok & v]), "out_of_view": _mean(inside[ok & ~v]),
                       "radius_in_view": _mean(np.concatenate([r[:, 0] for r in radius])[ok & v]),
                       "radius_out_of_view": _mean(
                           np.concatenate([r[:, 0] for r in radius])[ok & ~v])}
    return out


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def stays(scenes: list[w.Scene], answers: list[np.ndarray]) -> dict[str, Any]:
    """A thing at rest out of view: its error while it was seen, at the step its rest out of
    view begins (the last step before its final 1,000) and over the last 100 of its steps, and
    how far the answer moved over those 1,000 steps."""
    seen, begins, late, moved = [], [], [], []
    for s, a in zip(scenes, answers, strict=True):
        e, v = errors(s, a)[:, 0], s.in_view[:, 0].copy()
        v[0] = False
        seen.append(float(e[v].mean()))
        begins.append(float(e[-1001]))
        late.append(float(e[-100:].mean()))
        moved.append(float(np.linalg.norm(a[-1, 0] - a[-1001, 0])))
    return {"scenes": len(scenes), "seen": float(np.mean(seen)),
            "rest_begins": float(np.mean(begins)), "unseen_end": float(np.mean(late)),
            "moved": float(np.mean(moved)),
            "unseen_steps": int(scenes[0].unseen()[-1, 0]) if scenes else 0}


def heard(scenes: list[w.Scene], answers: list[np.ndarray]) -> dict[str, Any]:
    """A thing heard moving out of view (the scene's ``event`` thing): its error over the steps
    after its last cue, beside its cues' own error (a cue's point against the true position)."""
    after, own, before = [], [], []
    for s, a in zip(scenes, answers, strict=True):
        assert s.cue is not None
        k = int(np.flatnonzero(np.isfinite(s.cue[..., 0]).sum(0) == np.isfinite(
            s.cue[..., 0]).sum(0).max())[0]) if s.kind != "heard" else 0
        sounding = np.flatnonzero(np.isfinite(s.cue[:, k, 0]))
        if not len(sounding) or sounding[-1] + 1 >= s.steps:
            continue
        e = errors(s, a)[:, k]
        after.append(float(e[sounding[-1] + 1 :].mean()))
        before.append(float(e[: sounding[0]].mean()))
        own.append(float(np.linalg.norm(
            w.cartesian(s.geometry, s.cue[sounding[-3:], k]) - s.truth[sounding[-3:], k],
            axis=-1).mean()))
    return {"scenes": len(after), "before": _mean(np.array(before)),
            "after": _mean(np.array(after)), "cue": _mean(np.array(own))}


def crossing(scenes: list[w.Scene], answers: list[np.ndarray]) -> dict[str, Any]:
    """Two hands that crossed out of view: the share of scenes whose last answers name each
    hand's own position (nearer in sum than the two exchanged), and the error of the things at
    rest."""
    kept, rest = [], []
    for s, a in zip(scenes, answers, strict=True):
        last, truth = a[-1], s.truth[-1]
        own = np.linalg.norm(last[0] - truth[0]) + np.linalg.norm(last[1] - truth[1])
        swapped = np.linalg.norm(last[0] - truth[1]) + np.linalg.norm(last[1] - truth[0])
        kept.append(bool(own < swapped))
        rest.append(float(errors(s, a)[1:, 2:].mean()))
    return {"scenes": len(scenes), "kept": _mean(np.array(kept, dtype=float)),
            "rest_error": _mean(np.array(rest))}


# -- the floors: a default brain asked the position at every step


class Tuned:
    """An answer as two motor slots of ``BINS`` tuned positions, one slot an axis."""

    def __init__(self, geometry: w.Geometry, bins: int = BINS):
        x_low, x_high, y_low, y_high = geometry.plane
        self.bins = bins
        self.centers = np.stack([np.linspace(x_low, x_high, bins),
                                 np.linspace(y_low, y_high, bins)])
        self.spacing = (self.centers[:, -1] - self.centers[:, 0]) / (bins - 1)

    def targets(self, places: np.ndarray) -> np.ndarray:
        """``(rows, 2)`` positions as ``(rows, 2 * bins)`` lesson targets: a peak one spacing
        wide around each coordinate, each slot summing to one."""
        offset = (places[:, :, None] - self.centers[None]) / self.spacing[None, :, None]
        peak = np.exp(-0.5 * offset**2)
        peak /= peak.sum(axis=-1, keepdims=True)
        return np.asarray(peak.reshape(len(places), 2 * self.bins))

    def read(self, motor: np.ndarray, temperature: float) -> np.ndarray:
        """The positions ``(rows, 2 * bins)`` motor activations name: each slot's mean under
        the softmax the lessons pull on."""
        s = motor.reshape(len(motor), 2, self.bins) / temperature
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return np.asarray((p * self.centers[None]).sum(axis=-1))


class Senses:
    """A step of a one-thing scene as a brain's observation. ``raw``: the report, the in-view
    flag, the commands and with ``joints`` the joint readings as they come. ``code``: each
    through a population code, a report out of view and a command of zero dark."""

    def __init__(self, geometry: w.Geometry, arm: str, joints: bool):
        from cadence import PopulationCode

        self.arm, self.joints, self.count = arm, joints, geometry.joints
        x_low, x_high, y_low, y_high = geometry.plane
        if arm == "code":
            self.place = PopulationCode([(x_low, x_high), (y_low, y_high)], per=16)
            self.command = PopulationCode([(-0.08, 0.08)] * self.count, per=9)
            self.angle = PopulationCode(list(geometry.limits), per=12)
            self.inputs = self.place.inputs + 1 + self.command.inputs + (
                self.angle.inputs if joints else 0)
        else:
            self.inputs = 3 + self.count * (2 if joints else 1)

    def __call__(self, report: np.ndarray, seen: np.ndarray, commands: np.ndarray,
                 joints: np.ndarray) -> np.ndarray:
        flag = seen.astype(np.float64)[:, None]
        if self.arm != "code":
            parts = [report * flag, flag, commands] + ([joints] if self.joints else [])
            return np.hstack(parts)
        moving = (commands != 0).any(axis=1, keepdims=True)
        parts = [self.place(report) * flag, flag, self.command(commands) * moving]
        if self.joints:
            parts.append(self.angle(joints))
        return np.hstack(parts)


class Floor:
    """A default brain asked the position at every step. ``lessons``: ``truth`` teaches every
    step of a training scene its true position; ``camera`` teaches the steps in view their
    report and no other step. With ``hide`` a ``camera`` floor leaves reports unread over spans
    it draws (a span begins at a step in view at rate 0.04 and lasts 20 steps on average) and
    is taught there by the report it did not read. ``settings`` are ``Brain.compose``'s: with
    ``efference_amplitude`` the brain holds the copy of its own last answer in a region of its
    own, a place held in the brain's carried activity."""

    def __init__(self, geometry: w.Geometry, arm: str, founder: int, *, joints: bool = False,
                 lessons: str = "truth", settings: dict[str, Any] | None = None,
                 hide: bool = False):
        from cadence import Brain

        self.senses = Senses(geometry, arm, joints)
        self.tuned = Tuned(geometry)
        self.lessons = lessons
        self.hide = bool(hide) and lessons == "camera"
        self.rng = np.random.default_rng(np.random.SeedSequence([int(founder), 77]))
        self.brain = Brain.compose(self.senses.inputs, 2 * BINS, slots=2, seed=founder,
                                   backend="cpu", **(settings or {}))
        self.synapses = int(self.brain.describe()["layout"]["synapses"])
        self.work = {"lesson_row_sweeps": 0.0, "act_sweeps_rows": 0.0, "moments": 0}

    def run(self, scenes: list[w.Scene], teach: bool, brain: Any = None) -> np.ndarray:
        """Take ``scenes`` of one length side by side, one row each: the answers
        ``(scenes, steps, 2)``. With ``teach`` every step's lesson comes before its answer."""
        brain = self.brain if brain is None else brain
        feeds = [given(s) for s in scenes]
        steps = scenes[0].steps
        answers = np.zeros((len(scenes), steps, 2))
        temperature = float(brain.learner.config.temperature)
        hidden = np.zeros((len(scenes), steps), dtype=bool)
        if teach and self.hide:
            for row, feed in enumerate(feeds):
                t = 1
                while t < steps:
                    if feed["seen"][t] and self.rng.random() < 0.04:
                        length = int(self.rng.geometric(1.0 / 20.0))
                        hidden[row, t : t + length] = True
                        t += length
                    t += 1
        brain.reset()
        for t in range(steps):
            shown = np.array([f["seen"][t] for f in feeds]) & ~hidden[:, t]
            x = self.senses(np.stack([f["report"][t] for f in feeds]), shown,
                            *(np.stack([f[name][t] for f in feeds])
                              for name in ("commands", "joints")))
            if teach:
                seen = np.array([f["seen"][t] for f in feeds])
                if self.lessons == "truth":
                    places, weights = np.stack([s.truth[t, 0] for s in scenes]), None
                else:
                    places = np.stack([f["report"][t] for f in feeds])
                    weights = seen.astype(np.float64)
                if weights is None or weights.any():
                    report = brain.lesson(x, targets=self.tuned.targets(places), weights=weights)
                    self.work["lesson_row_sweeps"] += float(report.get("total_row_sweeps", 0.0))
            brain.act(x, greedy=True)
            if teach:
                settled = brain.last_settlement or {}
                self.work["act_sweeps_rows"] += float(settled.get("steps", 0)) * len(scenes)
                self.work["moments"] += len(scenes)
            motor = brain.basal_ganglia.state.part(brain.motor_index)
            answers[:, t] = self.tuned.read(np.asarray(motor), temperature)
        return answers

    def train(self, scenes: list[w.Scene], rows: int) -> None:
        for begin in range(0, len(scenes), rows):
            self.run(scenes[begin : begin + rows], teach=True)

    def answer(self, scenes: list[w.Scene], rows: int = 32) -> list[np.ndarray]:
        """Answers of a frozen copy: nothing taught, scenes of one length side by side."""
        frozen = copy.deepcopy(self.brain)
        out: list[np.ndarray] = [np.zeros(0)] * len(scenes)
        by_length: dict[int, list[int]] = {}
        for i, s in enumerate(scenes):
            by_length.setdefault(s.steps, []).append(i)
        for members in by_length.values():
            for begin in range(0, len(members), rows):
                group = members[begin : begin + rows]
                got = self.run([scenes[i] for i in group], teach=False, brain=frozen)
                for i, a in zip(group, got, strict=True):
                    out[i] = a[:, None, :]
        return out

    def milliseconds(self, scene: w.Scene, steps: int = 200) -> float:
        """The median wall time of one answer on one row, in milliseconds."""
        frozen = copy.deepcopy(self.brain)
        feed = given(scene)
        frozen.reset()
        times = []
        for t in range(min(steps, scene.steps)):
            x = self.senses(*(feed[name][t][None] for name in ("report", "seen", "commands",
                                                               "joints")))
            began = time.perf_counter()
            frozen.act(x, greedy=True)
            times.append(time.perf_counter() - began)
        return float(np.median(times) * 1000.0)

    def macs(self) -> float:
        """Analytic multiply-accumulates of the training: every row sweep of a lesson and of an
        answer, each over every synapse once."""
        return (self.work["lesson_row_sweeps"] + self.work["act_sweeps_rows"]) * self.synapses


class Placed:
    """The space arm: ``cadence.space`` holds the hand's place, one row a scene. Its lessons
    hold only what the camera reports."""

    def __init__(self, geometry: w.Geometry, founder: int,
                 settings: dict[str, Any] | None = None):
        from cadence.space import Interface, Space

        settings = dict(settings or {})
        self.felt = bool(settings.pop("joints", False))  # the joint readings are given
        shape = {k: settings.pop(k) for k in list(settings) if k in Interface.__dataclass_fields__}
        shape = {k: tuple(v) if isinstance(v, list) else v for k, v in shape.items()}
        self.interface = Interface(plane=geometry.plane, joints=geometry.joints,
                                   ear=geometry.ear, limits=geometry.limits, **shape)
        self.space = Space(self.interface, seed=founder, backend="cpu", **settings)
        self.synapses = int(self.space.brain.describe()["layout"]["synapses"])

    def run(self, scenes: list[w.Scene], teach: bool, space: Any = None
            ) -> tuple[np.ndarray, np.ndarray]:
        """Scenes of one length side by side: answers ``(scenes, steps, 2)`` and each answer's
        radius at ``RATE``."""
        space = self.space if space is None else space
        feeds = [given(s) for s in scenes]
        steps = scenes[0].steps
        answers = np.zeros((len(scenes), steps, 2))
        radius = np.zeros((len(scenes), steps))
        space.begin(np.stack([f["report"][0] for f in feeds]),
                    own=np.array([s.case == "own" for s in scenes]))
        answers[:, 0], radius[:, 0] = space.where()
        for t in range(1, steps):
            seen, report, commands, joints = (np.stack([f[name][t] for f in feeds])
                                              for name in ("seen", "report", "commands",
                                                           "joints"))
            answers[:, t] = space.advance(seen, report, commands,
                                          joints=joints if self.felt else None, learn=teach)
            radius[:, t] = space.where()[1]
        return answers, radius

    def follow(self, scene: w.Scene, teach: bool = False, space: Any = None
               ) -> tuple[np.ndarray, np.ndarray]:
        """A scene of several things, one row a thing: answers ``(steps, things, 2)`` and each
        answer's radius. The reports carry no names: the space gives each to a row of its
        look, and each cue likewise."""
        assert scene.start is not None and scene.what is not None
        space = self.space if space is None else space
        answers = np.zeros((scene.steps, scene.things, 2))
        radius = np.zeros((scene.steps, scene.things))
        own = np.zeros(scene.things, dtype=bool)
        own[0] = scene.case == "own"
        space.begin(scene.start, own=own, looks=scene.what)
        answers[0], radius[0] = space.where()
        for t in range(1, scene.steps):
            space.predict(scene.commands[t], space.hear(sounds(scene, t)),
                          scene.joints[t] if self.felt else None, learn=teach)
            seen, reports = space.assign(detections(scene, t))
            answers[t] = space.observe(seen, reports, learn=teach)
            radius[t] = space.where()[1]
        return answers, radius

    def train(self, scenes: list[w.Scene], rows: int) -> None:
        """Scenes of one thing side by side, ``rows`` at a time; a scene of several things by
        itself, in the order given."""
        batch: list[w.Scene] = []
        for scene in scenes:
            if scene.things > 1:
                self.follow(scene, teach=True)
                continue
            batch.append(scene)
            if len(batch) == rows:
                self.run(batch, teach=True)
                batch = []
        if batch:
            self.run(batch, teach=True)

    def answer(self, scenes: list[w.Scene], rows: int = 32
               ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Answers of a frozen copy: nothing taught, scenes of one thing and one length side by
        side, a scene of several things by itself."""
        frozen = copy.deepcopy(self.space)
        answers: list[np.ndarray] = [np.zeros(0)] * len(scenes)
        radius: list[np.ndarray] = [np.zeros(0)] * len(scenes)
        by_length: dict[int, list[int]] = {}
        for i, s in enumerate(scenes):
            if s.things > 1:
                answers[i], radius[i] = self.follow(s, space=frozen)
            else:
                by_length.setdefault(s.steps, []).append(i)
        for members in by_length.values():
            for begin in range(0, len(members), rows):
                group = members[begin : begin + rows]
                got, wide = self.run([scenes[i] for i in group], teach=False, space=frozen)
                for i, a, r in zip(group, got, wide, strict=True):
                    answers[i], radius[i] = a[:, None, :], r[:, None]
        return answers, radius

    def milliseconds(self, scene: w.Scene, steps: int = 200) -> float:
        """The median wall time of one step on one row, in milliseconds."""
        frozen = copy.deepcopy(self.space)
        feed = given(scene)
        frozen.begin(feed["report"][:1], own=scene.case == "own")
        times = []
        for t in range(1, min(steps, scene.steps)):
            began = time.perf_counter()
            frozen.advance(feed["seen"][t : t + 1], feed["report"][t : t + 1],
                           feed["commands"][t : t + 1],
                           joints=feed["joints"][t : t + 1] if self.felt else None)
            times.append(time.perf_counter() - began)
        return float(np.median(times) * 1000.0)

    def macs(self) -> float:
        work = self.space.work
        return float((work["lesson_row_sweeps"] + work["act_sweeps_rows"]) * self.synapses)


# -- lives


def seed(founder: int, name: str) -> int:
    return 1_000_000 + 10 * int(founder) + SEEDS[name]


def test_scenes(case: str, founder: int, count: int = 200, quota: int = 40) -> list[w.Scene]:
    """A founder's held-out scenes of ``case``: ``count`` as they come and, for a person's hand,
    ``quota`` more of each panel."""
    out = w.scenes(case, seed(founder, "test"), count)
    if case == "other":
        out += w.scenes(case, seed(founder, "panels"), quota={name: quota for name in PANELS})
    return out


def curriculum(founder: int, count: int, vary: bool = True, left: float = 0.0
               ) -> list[w.Scene]:
    """A founder's training scenes of every kind, shuffled: of ``count``, 25% the arm's own
    hand and 35% a person's hand in the chamber's geometry, 5% and 20% of them with discs and
    dark spans of their own (``vary``; in the chamber's geometry without it), 10% a thing
    heard and 5% two hands that cross. ``left`` gives that share of the count, taken from the
    person's hand in equal parts, to hands that go out of view and are left there, two thirds
    of them among discs of their own."""
    shares = (("own", None, False, 0.25), ("other", None, False, 0.35 - left / 2),
              ("own", None, vary, 0.05), ("other", None, vary, 0.20 - left / 2),
              ("room", "heard", False, 0.10), ("room", "crossing", False, 0.05),
              ("other", "left", False, left / 3), ("other", "left", vary, 2 * left / 3))
    out: list[w.Scene] = []
    for index, (case, kind, varied, share) in enumerate(shares):
        out += w.scenes(case, seed(founder, "train") + 100 * (index + 1), round(share * count),
                        kind=kind, vary=varied)
    order = np.random.default_rng(seed(founder, "train")).permutation(len(out))
    return [out[i] for i in order]


def panels(founder: int, sizes: dict[str, int] | None = None) -> dict[str, list[w.Scene]]:
    """A founder's held-out scenes by panel: ``own`` and ``other`` (``test_scenes``),
    ``varied`` (a person's hand among discs and dark spans of the scene's own),
    ``stays_own`` and ``stays_other`` (the arm's own hand and another thing at rest out of view
    for 1,000 steps), ``heard`` and ``crossing``."""
    n = {"own": 200, "other": 200, "quota": 40, "varied": 100, "stays": 8, "heard": 60,
         "crossing": 100, **(sizes or {})}
    base = seed(founder, "test")
    return {
        "own": test_scenes("own", founder, n["own"]),
        "other": test_scenes("other", founder, n["other"], n["quota"]),
        "varied": w.scenes("other", base + 1000, n["varied"], vary=True),
        "stays_own": w.scenes("own", seed(founder, "stays"), n["stays"], kind="stays"),
        "stays_other": w.scenes("other", seed(founder, "stays"), n["stays"], kind="stays"),
        "heard": w.scenes("room", seed(founder, "heard"), n["heard"], kind="heard"),
        "crossing": w.scenes("room", seed(founder, "crossing"), n["crossing"], kind="crossing"),
    }


def readings(held_out: dict[str, list[w.Scene]], answers: dict[str, list[np.ndarray]],
             radius: dict[str, list[np.ndarray]] | None = None) -> dict[str, Any]:
    """Every panel's scores (``table``, ``stays``, ``heard``, ``crossing``)."""
    out: dict[str, Any] = {}
    for name, scenes in held_out.items():
        if name in ("own", "other", "varied"):
            out[name] = table(scenes, answers[name], None if radius is None else radius[name])
        else:
            score = {"stays_own": stays, "stays_other": stays, "heard": heard,
                     "crossing": crossing}[name]
            out[name] = score(scenes, answers[name])
    return out


def whole(spec: dict[str, Any]) -> dict[str, Any]:
    """One whole life: an arm taught on a founder's ``curriculum`` (``space``) or fitted on it
    (``kalman``) and read on every panel. ``spec``: ``arm``, ``founder``, ``train``, ``rows``,
    ``settings``, ``sizes`` (the panels' sizes), ``save``."""
    warnings.simplefilter("ignore")
    arm, founder = str(spec["arm"]), int(spec["founder"])
    count, rows = int(spec.get("train", 2000)), int(spec.get("rows", 8))
    started = time.monotonic()
    held_out = panels(founder, spec.get("sizes"))
    row: dict[str, Any] = {"arm": arm, "case": "all", "founder": founder, "train": count,
                           "rows": rows, "lessons": "camera", "tag": spec.get("tag"),
                           "settings": spec.get("settings")}
    answers: dict[str, list[np.ndarray]] = {}
    radius: dict[str, list[np.ndarray]] | None = None
    if arm == "space":
        settings = dict(spec.get("settings") or {})
        training = curriculum(founder, count, vary=bool(settings.pop("vary", True)),
                              left=float(settings.pop("left", 0.0)))
        placed = Placed(training[0].geometry, founder, settings)
        placed.train(training, rows)
        if spec.get("save"):  # the trained brain, before the panels are read
            placed.space.brain.save(str(spec["save"]))
            Path(str(spec["save"]) + ".json").write_text(json.dumps(
                {"founder": founder, "train": count, "noise": float(np.sqrt(placed.space.noise)),
                 "settings": spec.get("settings")}) + "\n")
        radius = {}
        for name, scenes in held_out.items():
            answers[name], radius[name] = placed.answer(scenes)
        for reading in ("peak", "seen"):  # the same brain's shift read otherwise
            placed.space.read = reading
            other = {name: placed.answer(scenes) for name, scenes in held_out.items()}
            row[reading] = readings(held_out, {k: a for k, (a, _) in other.items()},
                                    {k: r for k, (_, r) in other.items()})
        placed.space.read = "mean"
        row.update(parameters=int(placed.space.brain.parameters()), synapses=placed.synapses,
                   macs=placed.macs(), work=placed.space.work,
                   milliseconds=placed.milliseconds(held_out["other"][0]),
                   milliseconds_own=placed.milliseconds(held_out["own"][0]),
                   noise=float(np.sqrt(placed.space.noise)))
    elif arm in ("last_seen", "kalman"):
        fitted = {"own": {}, "other": {}}
        if arm == "kalman":
            fitted = {case: fit_kalman(w.scenes(case, seed(founder, "train"), 200))
                      for case in ("own", "other")}
            row["fitted"] = fitted
            radius = {}
        make = Still if arm == "last_seen" else ConstantVelocity
        for name, scenes in held_out.items():
            both = [follow(s, make, fitted["own" if s.case == "own" else "other"],
                           hear=name == "heard") for s in scenes]
            answers[name] = [a for a, _ in both]
            if radius is not None:
                radius[name] = [r for _, r in both]
        row.update(parameters=0 if arm == "last_seen" else 2, macs=0.0)
    else:
        raise ValueError(f"a whole life is of space, last_seen or kalman, got {arm!r}")
    row["tables"] = readings(held_out, answers, radius)
    row["table"] = row["tables"]["other"]
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: an arm on a case for a founder. ``spec``: ``arm``, ``case``, ``founder``,
    ``train`` (scenes), ``rows``, ``lessons``, ``joints``, ``test`` and ``quota`` (scenes)."""
    warnings.simplefilter("ignore")
    arm, case, founder = str(spec["arm"]), str(spec["case"]), int(spec["founder"])
    count, rows = int(spec.get("train", 1000)), int(spec.get("rows", 8))
    lessons, joints = str(spec.get("lessons", "truth")), bool(spec.get("joints", False))
    started = time.monotonic()
    training = w.scenes(case, seed(founder, "train"), count)
    if spec.get("vary"):  # half of the training scenes among discs and dark spans of their own
        varied = w.scenes(case, seed(founder, "train") + 50, count - count // 2, vary=True)
        mixed = training[: count // 2] + varied
        order = np.random.default_rng(seed(founder, "train")).permutation(len(mixed))
        training = [mixed[i] for i in order]
    held_out = test_scenes(case, founder, int(spec.get("test", 200)), int(spec.get("quota", 40)))
    recorded = w.load(spec["scene"]) if spec.get("scene") else None
    if recorded is not None:  # a scene file, answered by the same arm and scored by itself
        held_out = [*held_out, recorded]
    row: dict[str, Any] = {"arm": arm, "case": case, "founder": founder, "train": count,
                           "rows": rows, "lessons": lessons, "joints": joints,
                           "tag": spec.get("tag"), "vary": bool(spec.get("vary"))}
    radius: list[np.ndarray] | None = None
    if arm == "last_seen":
        answers = [follow(s, Still)[0] for s in held_out]
        row.update(parameters=0, macs=0.0)
    elif arm == "kalman":
        settings = fit_kalman(training[: int(spec.get("fit", 200))])
        both = [follow(s, ConstantVelocity, settings) for s in held_out]
        answers, radius = [a for a, _ in both], [r for _, r in both]
        row.update(parameters=2, macs=0.0, settings=settings)
    elif arm in ("reckoning", "encoders"):
        answers = [reckoning(s, encoders=arm == "encoders") for s in held_out]
        row.update(parameters=0, macs=0.0)
    elif arm in ("default", "code"):
        floor = Floor(training[0].geometry, "raw" if arm == "default" else "code", founder,
                      joints=joints, lessons=lessons, settings=spec.get("settings"),
                      hide=bool(spec.get("hide")))
        floor.train(training, rows)
        answers = floor.answer(held_out)
        row.update(parameters=int(floor.brain.parameters()), synapses=floor.synapses,
                   macs=floor.macs(), work=floor.work,
                   milliseconds=floor.milliseconds(held_out[0]))
        if spec.get("save"):
            floor.brain.save(str(spec["save"]))
    elif arm == "space":
        placed = Placed(training[0].geometry, founder, spec.get("settings"))
        placed.train(training, rows)
        answers, radius = placed.answer(held_out)
        for reading in ("peak", "seen"):  # the same brain's shift read otherwise
            placed.space.read = reading
            row[reading] = table(held_out, *placed.answer(held_out))
        placed.space.read = "mean"
        row.update(parameters=int(placed.space.brain.parameters()), synapses=placed.synapses,
                   macs=placed.macs(), work=placed.space.work,
                   milliseconds=placed.milliseconds(held_out[0]),
                   noise=float(np.sqrt(placed.space.noise)), settings=spec.get("settings"))
        if spec.get("save"):
            placed.space.brain.save(str(spec["save"]))
    elif arm.split("-")[0] in ("lstm", "transformer"):
        import controls

        answers, extra = controls.life(arm, training, held_out, founder, lessons=lessons,
                                       joints=joints, parameters=int(spec["parameters"]),
                                       rows=rows)
        row.update(extra)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if recorded is not None:
        held_out, answer = held_out[:-1], answers[-1]
        answers = answers[:-1]
        wide = None if radius is None else [radius[-1]]
        radius = None if radius is None else radius[:-1]
        row["scene"] = {"file": Path(str(spec["scene"])).name,
                        "table": table([recorded], [answer], wide),
                        "errors": [round(float(e), 5) for e in errors(recorded, answer)[:, 0]]}
    row["table"] = table(held_out, answers, radius)
    if arm in ("last_seen", "kalman"):
        began = time.perf_counter()
        follow(held_out[0], Still if arm == "last_seen" else ConstantVelocity,
               row.get("settings", {}))
        row["milliseconds"] = (time.perf_counter() - began) * 1000.0 / held_out[0].steps
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def show(rows: list[dict[str, Any]]) -> str:
    """Lives as a table: one line an arm and case, the mean over its founders, errors in
    centimetres."""
    groups: dict[tuple[str, str, str, Any], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["case"], r["arm"], r.get("lessons", ""), r.get("tag")), []).append(r)
        for reading in ("peak", "seen"):  # the same life's shift read otherwise
            if reading in r and "in_view" in r[reading]:
                groups.setdefault((r["case"], r["arm"], r.get("lessons", ""),
                                   f"{r.get('tag') or ''}:{reading}"), []).append(
                    {**r, "table": r[reading]})

    def cm(values: list[float | None]) -> str:
        kept = [v for v in values if v is not None]
        return f"{100 * float(np.mean(kept)):6.1f}" if kept else "     ."

    names = [f"{lo}-{hi}" if hi < 10**9 else f"{lo}+" for lo, hi in UNSEEN]
    lines = ["case  arm                 lessons  n  params      macs    ms  in view  out  | "
             + "  ".join(f"{n:>6}" for n in names) + " | " + "  ".join(f"{p:>8}" for p in PANELS)]
    for (case, arm, lessons, tag), lives in sorted(groups.items(), key=str):
        tables = [r["table"] for r in lives]
        line = (f"{case:5} {arm + ('/' + str(tag) if tag else ''):19} {lessons:7} {len(lives):2d} "
                f"{int(np.mean([r.get('parameters', 0) for r in lives])):7d} "
                f"{float(np.mean([r.get('macs', 0.0) for r in lives])):9.2e} "
                f"{float(np.mean([r.get('milliseconds', 0.0) for r in lives])):5.2f} "
                f"{cm([t['in_view'] for t in tables])} {cm([t['out_of_view'] for t in tables])} | "
                + "  ".join(cm([t["by_unseen"][n]["error"] for t in tables]) for n in names)
                + " | " + "  ".join(cm([t.get("panels", {}).get(p, {}).get("error")
                                        for t in tables]) + "  " for p in PANELS))
        lines.append(line)
    return "\n".join(lines)


def gates(rows: list[dict[str, Any]]) -> str:
    """The declared gates' readings over whole lives: for every ``space`` tag and each way its
    shift is read, the means over founders in centimetres beside the controls of the same
    founders, with the founder-paired standard error where a gate asks for one."""
    whole_rows = [r for r in rows if r.get("case") == "all"]
    by: dict[str, dict[int, dict[str, Any]]] = {}
    for r in whole_rows:
        name = r["arm"] + (f"/{r['tag']}" if r.get("tag") else "")
        by.setdefault(name, {})[int(r["founder"])] = r["tables"]
        for reading in ("peak", "seen"):
            if reading in r:
                by.setdefault(f"{name}:{reading}", {})[int(r["founder"])] = r[reading]
    single: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for r in rows:
        if r.get("case") in ("own", "other") and not r.get("tag"):
            single.setdefault((r["case"], r["arm"], r.get("lessons", "")), {})[
                int(r["founder"])] = r["table"]

    def mean(values: list[float | None], scale: float = 100.0) -> str:
        kept = [v for v in values if v is not None]
        return f"{scale * float(np.mean(kept)):.2f}" if kept else "."

    def paired(a: dict[int, float], b: dict[int, float]) -> str:
        both = sorted(set(a) & set(b))
        if len(both) < 2:
            return "."
        d = np.array([100.0 * (a[f] - b[f]) for f in both])
        return f"{d.mean():+.2f} ({d.std(ddof=1) / np.sqrt(len(d)):.2f})"

    def cell(tables: dict[int, dict[str, Any]], *path: str) -> dict[int, float]:
        out = {}
        for founder, t in tables.items():
            value: Any = t
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                out[founder] = float(value)
        return out

    lines: list[str] = []
    kalman = by.get("kalman", {})
    for name in sorted(by):
        t = by[name]
        lines.append(f"== {name} ({len(t)} founders)")
        for case in ("own", "other", "varied"):
            lines.append(
                f"  {case:6}: in view {mean(list(cell(t, case, 'in_view').values()))}"
                f" (camera {mean(list(cell(t, case, 'camera').values()))})"
                f"  out of view {mean(list(cell(t, case, 'out_of_view').values()))}"
                f"  held in/out {mean(list(cell(t, case, 'held', 'in_view').values()), 1.0)}"
                f"/{mean(list(cell(t, case, 'held', 'out_of_view').values()), 1.0)}")
        for name_panel in PANELS:
            mine = cell(t, "other", "panels", name_panel, "error")
            theirs = cell(kalman, "other", "panels", name_panel, "error")
            line = (f"  {name_panel:8}: {mean(list(mine.values()))}  Kalman "
                    f"{mean(list(theirs.values()))}  difference {paired(mine, theirs)}")
            for (case, arm, lessons), tables in sorted(single.items()):
                if case == "other" and "offline" in arm:
                    line += (f"  {arm.split('-')[0]}/{lessons} "
                             f"{mean(list(cell(tables, 'panels', name_panel, 'error').values()))}")
            lines.append(line)
        for stays_name in ("stays_own", "stays_other"):
            lines.append(
                f"  {stays_name}: moved {mean(list(cell(t, stays_name, 'moved').values()))}"
                f"  seen {mean(list(cell(t, stays_name, 'seen').values()))}"
                f"  rest begins {mean(list(cell(t, stays_name, 'rest_begins').values()))}"
                f"  unseen end {mean(list(cell(t, stays_name, 'unseen_end').values()))}")
        lines.append(f"  heard: after {mean(list(cell(t, 'heard', 'after').values()))}"
                     f"  cue {mean(list(cell(t, 'heard', 'cue').values()))}"
                     f"  before {mean(list(cell(t, 'heard', 'before').values()))}")
        lines.append(f"  crossing: kept {mean(list(cell(t, 'crossing', 'kept').values()), 1.0)}"
                     f"  rest error {mean(list(cell(t, 'crossing', 'rest_error').values()))}")
    for (case, arm, lessons), tables in sorted(single.items()):
        lines.append(f"== {case} {arm} {lessons} ({len(tables)} founders): in view "
                     f"{mean(list(cell(tables, 'in_view').values()))}  out of view "
                     f"{mean(list(cell(tables, 'out_of_view').values()))}")
    cost = [f"{r['arm']}/{r.get('tag') or ''} {r.get('parameters', 0):,} parameters, "
            f"{r.get('macs', 0.0):.2e} multiply-accumulates, {r.get('milliseconds', 0.0):.2f} ms"
            for r in whole_rows if int(r["founder"]) == min(int(x["founder"]) for x in whole_rows)]
    return "\n".join(lines + sorted(cost))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("life", help="one arm on one case for one founder")
    one.add_argument("--arm", required=True)
    one.add_argument("--case", choices=("own", "other", "all"), required=True)
    one.add_argument("--founder", type=int, required=True)
    one.add_argument("--train", type=int, default=1000)
    one.add_argument("--rows", type=int, default=8)
    one.add_argument("--lessons", choices=("truth", "camera"), default="truth")
    one.add_argument("--joints", action="store_true")
    one.add_argument("--hide", action="store_true")
    one.add_argument("--vary", action="store_true")
    one.add_argument("--parameters", type=int, default=0)
    one.add_argument("--test", type=int, default=200)
    one.add_argument("--quota", type=int, default=40)
    one.add_argument("--tag")
    one.add_argument("--save")
    one.add_argument("--settings", type=json.loads, help="an arm's settings, as JSON")
    one.add_argument("--scene", help="a scene file the arm also answers, scored by itself")
    one.add_argument("--out", type=Path, required=True)
    scored = commands.add_parser("score", help="score one scene file")
    scored.add_argument("--scene", type=Path, required=True)
    scored.add_argument("--arm", default="kalman", choices=("last_seen", "kalman", "reckoning",
                                                            "encoders"))
    scored.add_argument("--q", type=float, default=1e-5)
    scored.add_argument("--r", type=float, default=0.01)
    made = commands.add_parser("make", help="write one generated scene to a file")
    made.add_argument("--case", choices=w.CASES, required=True)
    made.add_argument("--kind")
    made.add_argument("--seed", type=int, default=0)
    made.add_argument("--steps", type=int, default=150)
    made.add_argument("--hold", type=int, default=1000)
    made.add_argument("--out", type=Path, required=True)
    shown = commands.add_parser("table", help="the lives of a directory as a table")
    shown.add_argument("directory", type=Path)
    read = commands.add_parser("gates", help="the declared gates' readings over a directory")
    read.add_argument("directory", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "life":
        spec = {k: v for k, v in vars(args).items() if k not in ("command", "out")}
        row = whole(spec) if args.case == "all" else life(spec)
        args.out.mkdir(parents=True, exist_ok=True)
        name = "-".join(str(x) for x in (args.case, args.arm, args.lessons, args.tag or "x",
                                         args.founder))
        (args.out / f"{name}.json").write_text(json.dumps(row, indent=1) + "\n")
        print(show([row]))
    elif args.command == "score":
        scene = w.load(args.scene)
        if args.arm in ("reckoning", "encoders"):
            answers, radius = reckoning(scene, encoders=args.arm == "encoders"), None
        else:
            answers, radius = follow(scene, Still if args.arm == "last_seen" else ConstantVelocity,
                                     {"q": args.q, "r": args.r})
        print(json.dumps(table([scene], [answers], None if radius is None else [radius]),
                         indent=1))
    elif args.command == "make":
        scene = w.scenes(args.case, args.seed, 1, kind=args.kind, steps=args.steps,
                         hold=args.hold)[0]
        w.save(scene, args.out)
        print(f"{args.out}: {scene.case} {scene.kind}, {scene.steps} steps, things: "
              f"{scene.things}, the hand's steps out of view: {int((~scene.in_view[:, 0]).sum())}")
    elif args.command == "gates":
        rows = [json.loads(p.read_text()) for d in args.directory for p in sorted(d.glob("*.json"))]
        print(gates(rows))
    else:
        rows = [json.loads(p.read_text()) for p in sorted(args.directory.glob("*.json"))]
        print(show(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
