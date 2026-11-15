"""One surprise signal, read from the settle's own residual, and the one focus it draws (#337).

A brain carries its settled state from one moment into the next, and each moment's settle starts
from it. Under the moment's drive that carried state is off its equations by a residual, which
the settle then repairs. The surprise of a unit is the size of its residual at the start of the
moment's settle, against the size the unit is used to:

    r         = |synaptic input + drive + bias - adaptation - v|   of the carried state, a unit
    surprise  = log(r / (tolerance * usual + floor))   where that is positive, else 0
    usual    += adapt * (r - usual)  where r > usual,  else  recover * (r - usual)

``r`` is the residual of the potential equation (``NeuralGraph.residual(drive, state,
per_unit=True)``). That is the equation a moment's drive enters and the one a settle's
qualification checks: the largest ``r`` over the units is the residual the settle starts from.
The adaptation equation (``activation - a``) holds for a carried state as it held when the state
settled, whatever the moment brings, so it reads nothing of the moment. For a state that settled
within the tolerance, under synapses unchanged since, ``r`` is the change of the unit's drive
since that settle, to within the tolerance: what the senses and the memories' reads bring that
the equilibrium carried in does not hold. A lesson taught between two moments adds what the
changed synapses move (an outcome is learned inside its own moment, whose act settles again
under the new synapses). A stream that begins from rest carries no equilibrium: its moment reads
no surprise and teaches the usual nothing.

``usual`` is each unit's learned size of ``r``: it rises fast when a moment exceeds it and falls
slowly, as tectal and forebrain neurons adapt to a repeated stimulus within a few repetitions
and recover over many moments (stimulus-specific adaptation). The law is ``Arousal.outcome``'s
surprise, per unit: an error against the error the unit is used to. The same law reads any size
a unit: a map of things gives each cell the move of the thing it holds from where the map
carried it (``Map(surprise=)``).

``Orienting`` turns that surprise into one focus. Every place of every sense competes in one
competition: the place of the largest surprise wins, the winner holds a share of its salience
from moment to moment, and a place the focus leaves is inhibited for a while before it can win
again (inhibition of return). The winner offers the orienting act (where to turn) and a top-down
gain around the attended place.

Born: the rule and its genes. Learned: what the brain carries into each moment (its weights, its
working trace, a map's physics) and each unit's ``usual``. Before a unit has a usual every change
of its drive surprises; a change that recurs becomes usual and stops drawing the focus.

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


def _count(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or (
        value < 0
    ):
        raise ValueError(f"{name} is a count")
    return int(value)


class Surprise:
    """Each unit's surprise at the moment: the size the moment gave it, against its usual size.

    ``units`` counts the units. Genes: ``tolerance`` (a size within this many usual sizes is no
    surprise), ``floor`` (nor is one within this size, above zero, in the units of the size),
    ``adapt`` (the share of an excess the usual takes at once) and ``recover`` (the share of a
    shortfall it gives back a moment). ``observe`` takes one moment's sizes ``(streams, units)``
    and gives the surprise per unit. ``last`` holds it, ``moment`` one number per stream,
    ``usual`` each unit's learned size and ``moments`` the moments observed.

    A brain composed with ``surprise`` holds one over its neurons (``Brain.surprise``): the size
    is each neuron's residual at the start of the moment's settle (the module's text)."""

    GENES = ("tolerance", "floor", "adapt", "recover")

    def __init__(self, units: int, *, tolerance: float = 2.0, floor: float = 0.05,
                 adapt: float = 0.5, recover: float = 0.01,
                 usual: np.ndarray | None = None) -> None:
        if isinstance(units, (bool, np.bool_)) or not isinstance(units, (int, np.integer)) or (
            units < 1
        ):
            raise ValueError("a surprise takes one unit or more")
        self.size = int(units)
        self.tolerance = _gene("tolerance", tolerance)
        self.floor = _gene("floor", floor)
        self.adapt = _gene("adapt", adapt, high=1.0)
        self.recover = _gene("recover", recover, high=1.0)
        if self.floor <= 0.0:
            raise ValueError("a surprise needs a floor above zero: a size no unit is surprised by")
        self.usual = self._fitted("usual", np.zeros(self.size) if usual is None else usual,
                                  (self.size,))
        self.last = np.zeros((1, self.size))
        self.moments = 0

    @staticmethod
    def _fitted(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
        """``value`` as an array of ``shape`` of finite values of zero or more, copied."""
        try:
            held = np.array(value, dtype=np.float64).reshape(shape)
        except (TypeError, ValueError):
            raise ValueError(f"a surprise's {name} does not fit its units") from None
        if not (np.isfinite(held).all() and (held >= 0.0).all()):
            raise ValueError(f"a surprise's {name} is finite and zero or more")
        return held

    @property
    def moment(self) -> np.ndarray:
        """The moment's surprise per stream, ``(streams,)``: the largest of its units', finite
        and zero or more. Zero: no unit's size exceeded ``tolerance`` times its usual size plus
        ``floor``, so the moment is within what every unit is used to (a stream that begins from
        rest reads zero as well). A value ``k`` above zero: the most surprised unit's size was
        ``exp(k)`` times that tolerated size; a unit with no usual yet under a size of 3 reads
        ``log(3 / floor)``, about 4.1 at the founder's floor."""
        return np.asarray(self.last.max(axis=1))

    def settings(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.GENES}

    def observe(self, sizes: Any, *, learn: bool = True, carried: Any = None) -> np.ndarray:
        """One moment: ``sizes`` is the size it gave each unit, ``(units,)`` or ``(streams,
        units)``, signs ignored. Gives the surprise per unit, measured against the usual as it
        stood before the moment; then the usual learns from the moment (the mean over the
        streams), unless ``learn`` is False. ``carried`` flags the streams that carried a state
        into the moment (every stream unless given): a stream that did not reads no surprise and
        teaches the usual nothing."""
        r = np.asarray(sizes, dtype=np.float64)
        one = r.ndim == 1
        r = np.abs(np.atleast_2d(r))
        if r.ndim != 2 or r.shape[1] != self.size or not len(r) or not np.isfinite(r).all():
            raise ValueError(f"a moment's sizes are finite, {self.size} a stream")
        rows = np.ones(len(r), dtype=bool) if carried is None else np.asarray(carried)
        if rows.shape != (len(r),) or rows.dtype != np.bool_:
            raise ValueError("carried flags one stream each")
        unit = self.tolerance * self.usual + self.floor
        out = np.log(np.maximum(r, unit)) - np.log(unit)  # zero up to the tolerated size
        out[~rows] = 0.0
        if learn and rows.any():
            mean = r[rows].mean(axis=0)
            rate = np.where(mean > self.usual, self.adapt, self.recover)
            self.usual = self.usual + rate * (mean - self.usual)
        self.moments += 1
        self.last = out
        return np.asarray(out[0] if one else out)

    def reset(self, streams: int = 1) -> None:
        """The streams begin afresh: ``last`` reads no surprise; what each unit is used to
        stays, as the weights do."""
        self.last = np.zeros((int(streams), self.size))

    def to_dict(self) -> dict[str, Any]:
        """The genes and the moments observed; ``arrays`` holds the rest."""
        return {**self.settings(), "moments": int(self.moments)}

    def arrays(self) -> dict[str, np.ndarray]:
        """What a file keeps of a surprise: each unit's ``usual`` and the latest ``last``."""
        return {"usual": self.usual.copy(), "last": self.last.copy()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], arrays: Mapping[str, Any]) -> Surprise:
        """A surprise from ``to_dict`` and ``arrays`` (``last`` may be left out)."""
        try:
            usual = np.asarray(arrays["usual"], dtype=np.float64)
            if usual.ndim != 1 or not usual.size:
                raise ValueError("a surprise's usual holds one size a unit")
            out = cls(int(usual.size), usual=usual,
                      **{name: values[name] for name in cls.GENES})
            out.moments = _count("a surprise's moments", values.get("moments", 0))
            if "last" in arrays:
                last = np.asarray(arrays["last"], dtype=np.float64)
                if last.ndim != 2 or not len(last):
                    raise ValueError("a surprise's last does not fit its units")
                out.last = cls._fitted("last", last, (len(last), out.size))
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
            self.inhibited = [*self.inhibited, (self.focus.place.copy(), self.inhibit)]

    def attend(self, surprise: Any, places: Any = None, *, sense: str = "place"
               ) -> Focus | None:
        """One moment of the competition. ``surprise`` is one sense's surprise per unit
        ``(units,)`` with ``places`` ``(units, d)``, or a mapping of sense to ``(surprise,
        places)`` for several senses at once, their places in one frame of ``d`` coordinates.
        Gives the focus (also kept as ``focus``), or None where nothing reaches ``least``.
        Refused before anything changes: values that are not finite, and places that do not
        fit their units or one another."""
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
            try:
                p = np.asarray(p, dtype=np.float64).reshape(len(s), -1)
            except (TypeError, ValueError):
                raise ValueError(f"the places of {name!r} hold one place a unit") from None
            if not (np.isfinite(s).all() and np.isfinite(p).all()):
                raise ValueError("surprise and places are finite")
            if not len(s):
                continue
            if points and p.shape[1] != points[0].shape[1]:
                raise ValueError("the senses' places share one frame: the same coordinates each")
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
            if len(s) and p.shape[1] == len(at):
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
                won = int(np.flatnonzero(near)[int(np.argmax(s[near]))])
                focus = Focus(p[won].copy(), senses[won], int(index[won]), float(s[won]),
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

    # -- what a file keeps

    def streams(self) -> tuple[Focus | None, list[tuple[np.ndarray, float]], int]:
        """The competition's state (the focus, the places inhibited and the moments attended),
        for ``resume``: every part is replaced, never written into, by a moment."""
        return self.focus, self.inhibited, self.moments

    def resume(self, held: tuple[Focus | None, list[tuple[np.ndarray, float]], int]) -> None:
        """Return the competition to the state ``streams`` held."""
        self.focus, self.inhibited, self.moments = held

    def parts(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """The competition as a record (its genes, its moments and the focus) and arrays (the
        focus's place and the places inhibited, each with its weight in the last column)."""
        record: dict[str, Any] = {**self.settings(), "moments": int(self.moments), "focus": None}
        arrays: dict[str, np.ndarray] = {}
        focus = self.focus
        if focus is not None:
            record["focus"] = {"sense": focus.sense, "index": int(focus.index),
                               "salience": float(focus.salience), "held": int(focus.held),
                               "turned": bool(focus.turned)}
            arrays["focus"] = focus.place.copy()
        if self.inhibited:
            arrays["inhibited"] = np.stack(
                [np.concatenate([at, [weight]]) for at, weight in self.inhibited])
        return record, arrays

    @classmethod
    def from_parts(cls, record: Mapping[str, Any], arrays: Mapping[str, Any]) -> Orienting:
        """A competition from ``parts``, with the focus it held."""
        try:
            out = cls(**{name: record[name] for name in cls.GENES})
            out.moments = _count("an orienting's moments", record.get("moments", 0))
            told = record.get("focus")
            if told is not None:
                place = np.array(arrays["focus"], dtype=np.float64).reshape(-1)
                if not place.size or not np.isfinite(place).all():
                    raise ValueError("a saved focus holds a finite place")
                out.focus = Focus(
                    place, str(told["sense"]), _count("a focus's index", told["index"]),
                    _gene("a focus's salience", told["salience"]),
                    _count("a focus's held moments", told["held"]), bool(told["turned"]),
                )
            if "inhibited" in arrays:
                rows = np.array(arrays["inhibited"], dtype=np.float64)
                if rows.ndim != 2 or rows.shape[1] < 2 or not np.isfinite(rows).all():
                    raise ValueError("saved inhibited places hold a place and a weight each")
                out.inhibited = [(row[:-1].copy(), float(row[-1])) for row in rows]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid saved orienting: {error}") from None
        return out
