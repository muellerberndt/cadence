"""Surprise with a place, and the one focus it draws (#337).

A brain carries its state from one moment into the next. What the moment's input does to that
state is the brain's surprise, unit by unit:

    r         = |the unit's state after the moment - the state carried into it|
    surprise  = log(r / (tolerance * usual + floor))   where that is positive, else 0
    usual    += adapt * (r - usual)  where r > usual,  else  recover * (r - usual)

``usual`` is each unit's learned size of ``r``: it rises fast when a moment exceeds it and falls
slowly, as tectal and forebrain neurons adapt to a repeated stimulus within a few repetitions
and recover over many moments (stimulus-specific adaptation). What the brain carries into a
moment is its learned equilibrium: a still scene, a sound that goes on, a thing on a map with
learned physics carried along its line. So a thing that keeps its velocity under learned physics
moves nothing, and a thing that starts from rest, a new thing, a stop, a turn or a sudden sound
moves the units that hold it. The law is ``Arousal.outcome``'s surprise, per unit: an error
against the error the unit is used to.

``Orienting`` turns that surprise into one focus. Every place of every sense competes in one
competition: the place of the largest surprise wins, the winner holds a share of its salience
from moment to moment, and a place the focus leaves is inhibited for a while before it can win
again (inhibition of return). The winner offers the orienting act (where to turn) and a top-down
gain around the attended place.

Born: the rule and its genes. Learned: what the brain carries into each moment (its weights,
its working trace, a map's physics) and each unit's ``usual``. Before anything is learned every
change surprises; a change that recurs becomes usual and stops drawing the focus.

Sources: global inhibition in the owl optic tectum lets the strongest stimulus win, whether
seen or heard (Mysore, Asadollahi and Knudsen 2010, 2011); tectal neurons adapt to a repeated
stimulus and answer a deviant (Reches and Gutfreund 2008); NCM's lasting adaptation to a
repeated song (Chew et al. 1995); Bayesian surprise draws human gaze (Itti and Baldi 2009);
inhibition of return (Posner and Cohen 1984).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Focus", "Orienting", "Surprise"]


def _gene(name: str, value: Any, *, low: float = 0.0, high: float = np.inf) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not np.isfinite(number) or number < low or number > high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return number


class Surprise:
    """Each unit's surprise at the moment: how far the moment moved it, against its usual.

    ``units`` counts the units. Genes: ``tolerance`` (a move within this many usual moves is no
    surprise), ``floor`` (nor is one within this size, in the units of ``r``), ``adapt`` (the
    share of an excess the usual takes at once) and ``recover`` (the share of a shortfall it
    gives back a moment). ``observe`` takes one moment's moves ``(streams, units)`` and gives
    the surprise per unit; ``last`` holds it and ``moment`` the largest per stream, the one
    number per stream that memory strength and a learning-rate gate read."""

    GENES = ("tolerance", "floor", "adapt", "recover")

    def __init__(self, units: int, *, tolerance: float = 2.0, floor: float = 0.05,
                 adapt: float = 0.5, recover: float = 0.01,
                 usual: np.ndarray | None = None) -> None:
        if isinstance(units, (bool, np.bool_)) or int(units) < 1:
            raise ValueError("a surprise takes one unit or more")
        self.size = int(units)
        self.tolerance = _gene("tolerance", tolerance)
        self.floor = _gene("floor", floor)
        self.adapt = _gene("adapt", adapt, high=1.0)
        self.recover = _gene("recover", recover, high=1.0)
        if self.floor <= 0.0 and self.tolerance <= 0.0:
            raise ValueError("a surprise needs a floor or a tolerance above zero")
        self.usual = np.zeros(self.size) if usual is None else np.asarray(
            usual, dtype=np.float64).reshape(self.size).copy()
        if not (np.isfinite(self.usual).all() and (self.usual >= 0.0).all()):
            raise ValueError("usual moves are finite and nonnegative")
        self.last = np.zeros((1, self.size))
        self.moments = 0

    @property
    def moment(self) -> np.ndarray:
        """The moment's surprise per stream: the largest of its units', ``(streams,)``."""
        return np.asarray(self.last.max(axis=1))

    def settings(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.GENES}

    def observe(self, moved: Any, *, learn: bool = True) -> np.ndarray:
        """One moment: ``moved`` is how far it moved each unit, ``(units,)`` or ``(streams,
        units)``, signs ignored. Gives the surprise per unit, measured against the usual as it
        stood before the moment; then the usual learns from the moment (the mean over streams),
        unless ``learn`` is False."""
        r = np.abs(np.asarray(moved, dtype=np.float64))
        one = r.ndim == 1
        r = np.atleast_2d(r)
        if r.shape[1] != self.size or not np.isfinite(r).all():
            raise ValueError(f"a moment's moves are finite, {self.size} a stream")
        unit = self.tolerance * self.usual + self.floor
        with np.errstate(divide="ignore"):
            out = np.where(r > unit, np.log(np.maximum(r, 1e-300) / np.maximum(unit, 1e-300)),
                           0.0)
        if learn:
            mean = r.mean(axis=0)
            rate = np.where(mean > self.usual, self.adapt, self.recover)
            self.usual = self.usual + rate * (mean - self.usual)
            self.moments += 1
        self.last = out
        return out[0] if one else out

    def to_dict(self) -> dict[str, Any]:
        return {"units": self.size, **self.settings(), "moments": int(self.moments)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], usual: Any) -> Surprise:
        """A surprise from ``to_dict`` and its usual moves."""
        try:
            out = cls(int(values["units"]), usual=usual,
                      **{name: values[name] for name in cls.GENES})
            moments = values.get("moments", 0)
            if isinstance(moments, bool) or not isinstance(moments, int) or moments < 0:
                raise ValueError("a surprise's moments are a count")
            out.moments = moments
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid saved surprise: {error}") from None
        return out


@dataclass
class Focus:
    """The one place attended: its ``place``, the ``sense`` and ``index`` of the place that won,
    its ``salience``, the moments it has been held (``held``, 1 at the moment it was taken) and
    whether it was taken at this moment (``turned``: the orienting act is new)."""

    place: np.ndarray
    sense: str
    index: int
    salience: float
    held: int
    turned: bool


class Orienting:
    """One focus among the places of every sense, drawn by surprise.

    Genes: ``hold`` (the share of the focus's salience it keeps a moment), ``least`` (the least
    salience that orients; with none above it there is no focus), ``reach`` (places this near
    the focus are its place; places this near a place left are inhibited with it), ``inhibit``
    (how much of its salience a place left loses at once, 1 for all of it) and ``release`` (the
    share of that inhibition it keeps a moment). ``gain`` is the top-down gain at the focus:
    ``gains`` gives ``1 + gain`` there, falling off over ``reach``.

    ``attend`` takes one moment's surprise by sense, each with the places of its units in one
    frame (points on a plane: a sound's direction as a point along it). The candidates compete
    in one competition: the focus keeps ``hold`` of its salience, renewed by the surprise at its
    place (and moves with it there, so it follows a thing); another place wins when its surprise
    exceeds that. A place left takes the inhibition of return."""

    GENES = ("hold", "least", "reach", "inhibit", "release", "gain")

    def __init__(self, *, hold: float = 0.8, least: float = 0.5, reach: float = 0.1,
                 inhibit: float = 1.0, release: float = 0.95, gain: float = 1.0) -> None:
        self.hold = _gene("hold", hold, high=1.0)
        self.least = _gene("least", least)
        self.reach = _gene("reach", reach)
        self.inhibit = _gene("inhibit", inhibit, high=1.0)
        self.release = _gene("release", release, high=1.0)
        self.gain = _gene("gain", gain)
        self.reset()

    def reset(self) -> None:
        """No focus and no place inhibited."""
        self.focus: Focus | None = None
        self.inhibited: list[tuple[np.ndarray, float]] = []
        self.moments = 0

    def settings(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.GENES}

    def _leave(self) -> None:
        """The place of the focus held until this moment is inhibited, if there was one."""
        if self.focus is not None and self.inhibit > 0.0:
            self.inhibited.append((self.focus.place.copy(), self.inhibit))

    def attend(self, surprise: Any, places: Any = None, *, sense: str = "place"
               ) -> Focus | None:
        """One moment of the competition. ``surprise`` is one sense's surprise per unit
        ``(units,)`` with ``places`` ``(units, d)``, or a mapping of sense to ``(surprise,
        places)`` for several senses at once. Gives the focus (also kept as ``focus``), or
        None where nothing reaches ``least``."""
        if isinstance(surprise, Mapping):
            given = [(str(name), *pair) for name, pair in surprise.items()]
        else:
            given = [(sense, surprise, places)]
        senses: list[str] = []
        indices: list[np.ndarray] = []
        values: list[np.ndarray] = []
        points: list[np.ndarray] = []
        for name, s, p in given:
            s = np.asarray(s, dtype=np.float64).reshape(-1)
            p = np.asarray(p, dtype=np.float64).reshape(len(s), -1)
            if not (np.isfinite(s).all() and np.isfinite(p).all()):
                raise ValueError("surprise and places are finite")
            senses += [name] * len(s)
            indices.append(np.arange(len(s)))
            values.append(np.maximum(s, 0.0))
            points.append(p)
        s = np.concatenate(values) if values else np.zeros(0)
        p = np.concatenate(points) if points else np.zeros((0, 2))
        index = np.concatenate(indices) if indices else np.zeros(0, dtype=np.intp)
        self.moments += 1
        # inhibition of return: a place left loses salience for a while
        kept: list[tuple[np.ndarray, float]] = []
        for at, weight in self.inhibited:
            if len(s):
                s = s * np.where(np.linalg.norm(p - at, axis=1) <= self.reach, 1.0 - weight, 1.0)
            if weight * self.release >= 0.01:
                kept.append((at, weight * self.release))
        self.inhibited = kept
        focus = self.focus
        held = 0.0
        near = np.zeros(len(s), dtype=bool)
        if focus is not None and len(s) and p.shape[1] == len(focus.place):
            near = np.linalg.norm(p - focus.place, axis=1) <= self.reach
        if focus is not None:
            held = self.hold * focus.salience
            if near.any() and s[near].max() > held:  # renewed where it is: it follows the place
                at = int(np.flatnonzero(near)[int(np.argmax(s[near]))])
                focus = Focus(p[at].copy(), senses[at], int(index[at]), float(s[at]),
                              focus.held + 1, False)
            else:
                focus = Focus(focus.place, focus.sense, focus.index, held, focus.held + 1,
                              False)
        others = np.where(near, 0.0, s)
        best = int(np.argmax(others)) if len(others) else -1
        if best >= 0 and others[best] > (focus.salience if focus is not None else 0.0):
            focus = Focus(p[best].copy(), senses[best], int(index[best]), float(others[best]),
                          1, True)
        if focus is not None and focus.salience < self.least:
            focus = None
        if focus is None or focus.turned:
            self._leave()  # the focus moves on or lets go: the place it leaves is inhibited
        self.focus = focus
        return focus

    def gains(self, places: Any) -> np.ndarray:
        """The top-down gain at each of ``places`` ``(n, d)``: ``1 + gain`` at the focus,
        falling off as a Gaussian of width ``reach``; one everywhere with no focus."""
        p = np.asarray(places, dtype=np.float64)
        p = p.reshape(len(p), -1)
        if self.focus is None or p.shape[1] != len(self.focus.place):
            return np.ones(len(p))
        d = np.linalg.norm(p - self.focus.place, axis=1)
        width = max(self.reach, 1e-12)
        return np.asarray(1.0 + self.gain * np.exp(-0.5 * (d / width) ** 2))

    def bearing(self, origin: Any) -> float | None:
        """The orienting act from ``origin`` (a point of the places' plane): the direction to the
        focus in radians, or None with no focus."""
        if self.focus is None:
            return None
        to = self.focus.place[:2] - np.asarray(origin, dtype=np.float64).reshape(-1)[:2]
        return float(np.arctan2(to[1], to[0]))
