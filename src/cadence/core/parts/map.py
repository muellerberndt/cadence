"""``cadence.space.map.Map`` in the core: the map of things with its learned physics and its
surprise, the graph it settles on, and what a moment, a question by place and the inventory
read and write.

The state holds the map's settings and the arrays its rules fix (codes, cell middles, the bump's
grid and shares, the plain graph's arrays as ``compiled._graph_arrays`` makes them, ``flat``
by its non-zero entries), its carried state (``potential``, ``clock``, ``lost``, ``doubt``,
``reading``, what each report sensed and what the camera saw around it, the last identity
given, the work counted), with physics its settings, lessons and motion records, and with
surprise the cells' surprise as its own part holds it (``surprise.``, ``parts.surprise``).

``lit``, a function of places, crosses as the places it was asked about and its answers, call
by call (``calling`` records them): the core asks the same questions in the same order and is
refused where it asks another.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ... import compiled
from ...space.map import Map
from . import surprise as surprise_part

CLASS = Map
EMPTY = ("near", "clear", "keep")
SENSES = ("seen", "heard", "felt")
STATES = ("seen", "heard", "felt", "held", "near", "moving", "not known where")
GRAPH = ("flat", "pairs", "flat_apical", "apical_pairs", "starts", "pair_pre", "pair_post",
         "offset", "freezable", "bias", "dt", "slope", "threshold", "rest", "leak", "has_apical",
         "pool_of", "pool_index", "pool_starts", "pool_run", "fraction", "floor2", "tiny")
PHYSICS_REALS = ("noise", "surprise", "settled", "prior")
PHYSICS_COUNTS = ("line", "every", "horizon", "reports")


def _looked(held: Map) -> int:
    """The places the camera is asked about around one place (``Map._looks``)."""
    return 81 if held.margin else 1


def _graph(held: Map) -> dict[str, Any]:
    if held._plain.backend != "cpu":
        raise ValueError("the core settles a map's graph on the CPU")
    arrays = dict(zip(GRAPH, compiled._graph_arrays(held._plain), strict=True))
    flat = np.asarray(arrays.pop("flat"), dtype=np.float64)
    if arrays.pop("flat_apical") is not flat or arrays["has_apical"]:
        raise ValueError("the core settles a map's plain graph, with no apical compartment")
    at = np.flatnonzero(flat.view(np.uint64))  # every entry that is not +0.0
    out: dict[str, Any] = {"graph.flat.size": np.int64(flat.size),
                           "graph.flat.at": at.astype(np.int64),
                           "graph.flat.value": flat[at].copy()}
    for name, value in arrays.items():
        held_value = np.asarray(value)
        out[f"graph.{name}"] = held_value.copy() if held_value.ndim else held_value[()]
    out["graph.runs"] = compiled.runs_of(held._runs)
    return out


def state_of(held: Map) -> dict[str, Any]:
    nx, ny = held.cells
    looked = _looked(held)
    out: dict[str, Any] = {
        "plane": np.array(held.plane, dtype=np.float64),
        "cells": np.array([nx, ny], dtype=np.int64),
        "width": np.int64(held.width),
        "values": np.array([s.stop - s.start for s in held.values.values()], dtype=np.int64),
        "level": np.float64(held.level), "empty": np.int64(EMPTY.index(held.empty)),
        "reach": np.float64(held.reach), "step": np.float64(held.step),
        "stride": np.float64(held.stride), "margin": np.float64(held.margin),
        "floor": np.float64(held._floor), "trace": np.bool_(held.trace),
        "codes": held.codes.copy(), "centers": held.centers.copy(), "low": held._low.copy(),
        "size": held.size.copy(), "offsets": Map._OFFSETS.copy(),
        "density": Map._DENSITY.copy(),
        "potential": held.potential.copy(), "clock": held.clock.copy(),
        "lost": held.lost.copy(), "doubt": held.doubt.copy(), "reading": held.reading.copy(),
        "sensed.kind": np.array([k for k, _ in held._sensed], dtype=np.int64),
        "sensed.sense": np.array([SENSES.index(s) for _, s in held._sensed], dtype=np.int64),
        "views.place": np.array([p for p, _ in held._views], dtype=np.float64).reshape(-1, 2),
        "views.seen": np.array([f for _, f in held._views], dtype=bool).reshape(
            len(held._views), looked),
        "identity": np.float64(getattr(held, "_identity", 0.0)),
        "work": np.array([held.work["moments"], held.work["questions"]], dtype=np.int64),
    }
    ph = held.physics
    if ph is not None:
        out.update({f"physics.{k}": np.float64(getattr(ph, k)) for k in PHYSICS_REALS})
        out.update({f"physics.{k}": np.int64(getattr(ph, k)) for k in PHYSICS_COUNTS})
        out["physics.learning"] = np.bool_(ph.learning)
        out["physics.stats"] = ph.stats.copy()
        out["physics.estimate"] = ph.estimate.copy()
        out["motion"] = held.motion.copy()
    if held.surprise is not None:  # as the surprise's own part holds it
        out.update({f"surprise.{k}": v for k, v in surprise_part.state_of(held.surprise).items()})
    out.update(_graph(held))
    return out


def restore(held: Map, state: dict[str, Any]) -> None:
    """The carried state onto ``held`` (a map of the same settings): its settings and the arrays
    they fix stay as they are."""
    for name in ("potential", "clock", "lost", "doubt", "reading"):
        value = np.array(state[name], dtype=np.float64)
        if value.shape != getattr(held, name).shape:
            raise ValueError(f"the map's {name!r} does not fit its cells")
        setattr(held, name, value)
    held._sensed = [(int(k), SENSES[int(s)])
                    for k, s in zip(state["sensed.kind"], state["sensed.sense"], strict=True)]
    held._views = [(np.array(p, dtype=np.float64), np.array(f, dtype=bool))
                   for p, f in zip(state["views.place"], state["views.seen"], strict=True)]
    held._identity = float(state["identity"])
    held.work = {"moments": int(state["work"][0]), "questions": int(state["work"][1])}
    ph = held.physics
    if ph is not None:
        ph.learning = bool(state["physics.learning"])
        ph.stats = np.array(state["physics.stats"], dtype=np.float64).reshape(10)
        ph.estimate = np.array(state["physics.estimate"], dtype=np.float64).reshape(4)
        ph.fit()
        held.motion = np.array(state["motion"], dtype=np.float64).reshape(held.motion.shape)
    if held.surprise is not None:
        surprise_part.restore(held.surprise, {k[len("surprise."):]: v for k, v in state.items()
                                              if k.startswith("surprise.")})


class Camera:
    """``lit`` as it was asked: each call's places and the answers, in order."""

    def __init__(self, lit: Callable[[np.ndarray], Any]) -> None:
        self.lit = lit
        self.asked: list[np.ndarray] = []
        self.answers: list[np.ndarray] = []

    def __call__(self, points: np.ndarray) -> Any:
        places = np.array(points, dtype=np.float64).reshape(-1, 2)
        answer = self.lit(points)
        self.asked.append(places)
        self.answers.append(np.asarray(answer, dtype=bool).reshape(len(places)).copy())
        return answer

    def named(self) -> dict[str, Any]:
        bounds = np.zeros(len(self.asked) + 1, dtype=np.int64)
        bounds[1:] = np.cumsum([len(p) for p in self.asked])
        return {"lit.kind": np.int64(2),
                "lit.points": (np.concatenate(self.asked) if self.asked
                               else np.zeros((0, 2))),
                "lit.bounds": bounds,
                "lit.seen": (np.concatenate(self.answers) if self.answers
                             else np.zeros(0, dtype=bool))}


def reports(held: Map, seen: Any = (), heard: Any = (), felt: Any = ()) -> dict[str, Any]:
    """A moment's reports as the core reads them, in the order ``Map.moment`` takes them: each
    one's kind, place, sense, and the amounts it names on the value units (``named`` flags
    them)."""
    valued = held.width - held.units
    kinds, places, senses, amounts, named = [], [], [], [], []
    for sense, given in zip(SENSES, (seen, heard, felt), strict=True):
        for report in given:
            kind, place, *_ = report
            kinds.append(int(kind))
            places.append(np.asarray(place, dtype=np.float64).reshape(2))
            senses.append(SENSES.index(sense))
            row, flags = np.zeros(valued), np.zeros(valued, dtype=bool)
            for group, code in held._said(report).items():
                run = held.values[group]
                row[run.start - held.units:run.stop - held.units] = code
                flags[run.start - held.units:run.stop - held.units] = True
            amounts.append(row)
            named.append(flags)
    count = len(kinds)
    return {"reports.kind": np.array(kinds, dtype=np.int64),
            "reports.place": np.array(places, dtype=np.float64).reshape(count, 2),
            "reports.sense": np.array(senses, dtype=np.int64),
            "reports.values": np.array(amounts, dtype=np.float64).reshape(count, valued),
            "reports.named": np.array(named, dtype=bool).reshape(count, valued)}


def calling(held: Map, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> tuple[tuple[Any, ...], dict[str, Any], Callable[[], dict[str, Any]]]:
    """Before a call: ``lit`` wrapped so its questions and answers are kept, and what the call's
    arguments need of the map (its values' units); after it, those as named arguments."""
    if method != "moment":
        return args, kwargs, dict
    names = ("seen", "heard", "felt", "lit")
    given = {**dict(zip(names, args, strict=False)), **kwargs}
    told = reports(held, given.get("seen", ()), given.get("heard", ()), given.get("felt", ()))
    lit = given.get("lit")
    camera = Camera(lit) if callable(lit) else None
    if camera is not None:
        given["lit"] = camera
    if lit is None:
        told["lit.kind"] = np.int64(0)
    elif camera is None:
        told["lit.kind"] = np.int64(1)
        told["lit.flags"] = np.asarray(lit, dtype=bool).reshape(held.count).copy()

    def after() -> dict[str, Any]:
        return {**told, **(camera.named() if camera is not None else {})}

    return (), given, after


def _none(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {}


def _thing_at(place: Any = None, within: Any = None, *, cells: Any = None,
              body: Any = (0.0, 0.0), heading: float = 0.0) -> dict[str, Any]:
    out: dict[str, Any] = {"body": np.asarray(body, dtype=np.float64).reshape(2),
                           "heading": np.float64(heading)}
    if place is not None:
        out["place"] = np.asarray(place, dtype=np.float64).reshape(2)
    if within is not None:
        out["within"] = np.float64(within)
    if cells is not None:
        out["cells"] = np.asarray(cells, dtype=bool).ravel().copy()
    return out


def _thing(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"found": np.bool_(False)}
    values = [np.asarray(v, dtype=np.float64) for v in value["values"].values()]
    return {"found": np.bool_(True), "kind": np.int64(value["kind"]),
            "values": np.concatenate(values) if values else np.zeros(0),
            "place": np.asarray(value["place"], dtype=np.float64),
            "bearing": np.float64(value["bearing"]), "distance": np.float64(value["distance"])}


def _things(value: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(value)
    cells = [np.asarray(t["cells"], dtype=np.int64) for t in value]
    pieces = [len(p) for t in value for p in t["pieces"]]
    groups = max((len(t.get("values", {})) for t in value), default=0)
    out: dict[str, Any] = {
        "things.kind": np.array([t["kind"] for t in value], dtype=np.int64),
        "things.state": np.array([STATES.index(t["state"]) for t in value], dtype=np.int64),
        "things.since": np.array([-1 if t["since"] is None else t["since"] for t in value],
                                 dtype=np.int64),
        "things.place": np.array([np.full(2, np.nan) if t["place"] is None else t["place"]
                                  for t in value], dtype=np.float64).reshape(n, 2),
        "things.cells": np.concatenate(cells) if cells else np.zeros(0, dtype=np.int64),
        "things.cell_bounds": np.concatenate([[0], np.cumsum([len(c) for c in cells])]).astype(
            np.int64),
        "things.pieces": np.array([len(t["pieces"]) for t in value], dtype=np.int64),
        "things.piece_sizes": np.array(pieces, dtype=np.int64),
        "things.moving": np.array(["velocity" in t for t in value], dtype=bool),
        "things.velocity": np.array([t.get("velocity", np.zeros(2)) for t in value],
                                    dtype=np.float64).reshape(n, 2),
        "things.values": np.array([[-1 if v is None else v for v in t.get("values", {}).values()]
                                   for t in value], dtype=np.int64).reshape(n, groups),
    }
    return out


def _forecast(value: Map) -> dict[str, Any]:
    out = {name: getattr(value, name).copy()
           for name in ("potential", "clock", "lost", "doubt", "reading")}
    if value.physics is not None:
        out["motion"] = value.motion.copy()
    return out


METHODS = {
    "moment": (_none, lambda value: {}),
    "thing_at": (_thing_at, _thing),
    "things": (_none, _things),
    "sheet": (lambda kind=None: {} if kind is None else {"kind": np.int64(kind)},
              lambda value: {"sheet": np.asarray(value, dtype=np.float64)}),
    "forecast": (lambda moments=1: {"moments": np.int64(moments)}, _forecast),
    "moves": (lambda expected: {"expected": np.asarray(expected, dtype=np.float64).copy()},
              lambda value: {"moves": np.asarray(value, dtype=np.float64)}),
}
