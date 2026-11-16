"""``cadence.surprise.Orienting`` in the core: its genes, the focus it holds, the places it
inhibits (each with its weight) and the moments attended.

A focus's sense crosses as its code, its place in ``SENSES``. ``attend``'s candidates cross as
one frame: each one's surprise, its place and its sense's code, the senses in the order given.
The places inhibited cross flat (``inhibited.place``), with the coordinates of each
(``inhibited.dim``) and its weight (``inhibited.weight``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...surprise import Focus, Orienting

CLASS = Orienting
# the senses a focus is drawn from (``kernels/core/src/parts/surprise.rs``): ``attend``'s
# default sense, the sensory neurons with places, a held map's cells, the things a heard name
# makes salient (``cadence.generic.Brain._surprised``)
SENSES = ("place", "senses", "map", "named")


def _code(sense: str) -> int:
    if sense not in SENSES:
        raise ValueError(f"the core draws a focus from the senses {SENSES}, not {sense!r}")
    return SENSES.index(sense)


def _focus(focus: Focus | None) -> dict[str, Any]:
    if focus is None:
        return {"focused": np.bool_(False), "focus.place": np.zeros(0),
                "focus.sense": np.int64(-1), "focus.index": np.int64(-1),
                "focus.salience": np.float64(0.0), "focus.held": np.int64(0),
                "focus.turned": np.bool_(False)}
    return {"focused": np.bool_(True),
            "focus.place": np.array(focus.place, dtype=np.float64).reshape(-1),
            "focus.sense": np.int64(_code(focus.sense)), "focus.index": np.int64(focus.index),
            "focus.salience": np.float64(focus.salience), "focus.held": np.int64(focus.held),
            "focus.turned": np.bool_(focus.turned)}


def state_of(held: Orienting) -> dict[str, Any]:
    out: dict[str, Any] = {f"config.{g}": np.float64(getattr(held, g)) for g in Orienting.GENES}
    out["moments"] = np.int64(held.moments)
    out.update(_focus(held.focus))
    places = [np.array(at, dtype=np.float64).reshape(-1) for at, _ in held.inhibited]
    out["inhibited.place"] = np.concatenate(places) if places else np.zeros(0)
    out["inhibited.dim"] = np.array([len(at) for at in places], dtype=np.int64)
    out["inhibited.weight"] = np.array([w for _, w in held.inhibited], dtype=np.float64)
    return out


def restore(held: Orienting, state: dict[str, Any]) -> None:
    for g in Orienting.GENES:
        setattr(held, g, float(state[f"config.{g}"]))
    held.moments = int(state["moments"])
    held.focus = None
    if bool(state["focused"]):
        held.focus = Focus(np.array(state["focus.place"], dtype=np.float64).reshape(-1),
                           SENSES[int(state["focus.sense"])], int(state["focus.index"]),
                           float(state["focus.salience"]), int(state["focus.held"]),
                           bool(state["focus.turned"]))
    flat = np.asarray(state["inhibited.place"], dtype=np.float64)
    inhibited: list[tuple[np.ndarray, float]] = []
    at = 0
    for dim, weight in zip(state["inhibited.dim"], state["inhibited.weight"], strict=True):
        inhibited.append((flat[at:at + int(dim)].copy(), float(weight)))
        at += int(dim)
    held.inhibited = inhibited


def _attend(surprise: Any, places: Any = None, *, sense: str = "place") -> dict[str, Any]:
    if isinstance(surprise, Mapping):
        given = [(str(name), *pair) for name, pair in surprise.items()]
    else:
        given = [(sense, surprise, places)]
    values, points, codes = [], [], []
    for name, s, p in given:
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        values.append(s)
        points.append(np.asarray(p, dtype=np.float64).reshape(len(s), -1))
        codes.append(np.full(len(s), _code(name), dtype=np.int64))
    return {"surprise": np.concatenate(values) if values else np.zeros(0),
            "places": np.concatenate(points) if points else np.zeros((0, 2)),
            "sense": np.concatenate(codes) if codes else np.zeros(0, dtype=np.int64)}


# ``reset`` (no focus, nothing inhibited, no moments) is the core's as well; it is left out of
# the record because ``Orienting.__init__`` calls it on an object without its state yet
METHODS = {
    "attend": (_attend, _focus),
    "_leave": (lambda: {}, lambda value: {}),
}
