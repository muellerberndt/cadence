"""``cadence.stream.Hearing`` in the core: the carried gains, one per route, their second moments
and count, and what a contrast reads of each route.

A route is a held memory's columns ``start:stop`` (``route.<k>.memory`` numbers the memory,
whose ``trace`` and ``amplitude`` cross as ``memory.<m>.*``), its gain's ``slot`` in the memory
(-1: the memory holds one gain), its ceiling ``bound``, its ``synapses`` in the graph and each
synapse's entry ``places`` of its ``(width, association)`` matrix (row ``place // association``,
column ``place % association``). The gains themselves are the memories' ``hearing``; they cross
here as ``gains``, read and set back as ``Hearing.gain`` and ``Hearing.set`` do without calling
them (a recorded call of either takes the state first). The neuron model's slope crosses as
``model.*``.

A method that takes a route is named by the route's place among the Hearing's routes; the place
of each route is noted when the Hearing's state is taken (``state_of``, which a recorded call
takes first). A contrast reads the graph's weights whole (``weights``), and whether its
efficacies rest on the host (``efficacy``), its synapses come in pairs (``parallel``) or a
device holds them (``device``). What a moment changes in place crosses as a copy."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...stream import Hearing

CLASS = Hearing
# the place of each route among its Hearing's routes, by the route's identity
_PLACES: dict[int, int] = {}


def _memories(held: Hearing) -> list[Any]:
    memories: list[Any] = []
    for route in held.routes:
        if not any(route["memory"] is m for m in memories):
            memories.append(route["memory"])
    return memories


def _gain(route: dict[str, Any]) -> float:
    memory, slot = route["memory"], route["slot"]
    return float(memory.hearing if slot is None else memory.hearing[slot])


def _set_gain(route: dict[str, Any], value: float) -> None:
    memory, slot = route["memory"], route["slot"]
    if slot is None:
        memory.hearing = value
    else:
        memory.hearing[slot] = value


def state_of(held: Hearing) -> dict[str, Any]:
    memories = _memories(held)
    model = held.model
    out: dict[str, Any] = {
        "config.rate": np.float64(held.rate),
        "compiled_moment": np.bool_(held.__dict__.get("compiled_moment", False)),
        "association": np.asarray(held.association, dtype=np.int64),
        "model.slope": np.float64(model.slope),
        "model.threshold": np.float64(model.threshold),
        "model.rest": np.float64(model.rest_emission),
        "model.leak": np.float64(model.leak),
        "second": np.array(held.second, dtype=np.float64),
        "count": np.int64(held.count),
        "gains": np.array([_gain(r) for r in held.routes], dtype=np.float64),
        "routes": np.int64(len(held.routes)),
        "memories": np.int64(len(memories)),
    }
    for m, memory in enumerate(memories):
        out[f"memory.{m}.trace"] = np.array(memory.trace, dtype=np.float64)
        out[f"memory.{m}.amplitude"] = np.float64(memory.amplitude)
    for k, route in enumerate(held.routes):
        _PLACES[id(route)] = k
        slot = route["slot"]
        out[f"route.{k}.memory"] = np.int64(
            next(m for m, memory in enumerate(memories) if route["memory"] is memory))
        out[f"route.{k}.slot"] = np.int64(-1 if slot is None else slot)
        for name in ("start", "stop", "width"):
            out[f"route.{k}.{name}"] = np.int64(route[name])
        out[f"route.{k}.bound"] = np.float64(route["bound"])
        out[f"route.{k}.synapses"] = np.asarray(route["synapses"], dtype=np.int64)
        out[f"route.{k}.places"] = np.asarray(route["places"], dtype=np.int64)
    return out


def restore(held: Hearing, state: dict[str, Any]) -> None:
    """The gains (set on their memories), their second moments and count, the rate and the
    compiled moment's flag; the routes and the memories' traces are the brain's and stay."""
    held.rate = float(state["config.rate"])
    held.compiled_moment = bool(state["compiled_moment"])
    held.second = np.array(state["second"], dtype=np.float64)
    held.count = int(state["count"])
    for route, gain in zip(held.routes, state["gains"].tolist(), strict=True):
        _set_gain(route, gain)


def _route(route: dict[str, Any]) -> dict[str, Any]:
    return {"route": np.int64(_PLACES[id(route)])}


def _set(route: dict[str, Any], value: float) -> dict[str, Any]:
    return {**_route(route), "value": np.float64(value)}


def _step(contrast: Any) -> dict[str, Any]:
    return {"contrast": np.array(contrast, dtype=np.float64)}


def _graph(graph: Any) -> dict[str, Any]:
    return {
        "weights": np.array(graph._weights, dtype=np.float64),
        "efficacy": np.bool_(graph._efficacy is not None),
        "parallel": np.bool_(bool(graph.layout.parallel_synapses)),
        "device": np.bool_(graph._mlx is not None and graph._efficacy is None),
    }


def _contrast(graph: Any, movement: Any, rows: Any, batch: int) -> dict[str, Any]:
    out = {"movement": np.array(movement, dtype=np.float64), "batch": np.int64(batch),
           **_graph(graph)}
    if rows is not None:
        out["rows"] = np.array(rows, dtype=np.int64)
    return out


def _contrast_rows(graph: Any, plus: Any, minus: Any, beta: float) -> dict[str, Any]:
    return {
        "plus": np.atleast_2d(np.array(plus.activation, dtype=np.float64)),
        "minus": np.atleast_2d(np.array(minus.activation, dtype=np.float64)),
        "beta": np.float64(beta), **_graph(graph),
        "kernel": np.bool_(graph._torch is not None or graph._mlx is not None),
    }


def _movement_on_device(graph: Any, plus: Any, minus: Any) -> dict[str, Any]:
    return {"kernel": np.bool_(graph._torch is not None or graph._mlx is not None)}


def _rows(value: Any) -> dict[str, Any]:
    return {"contrast": np.array(value, dtype=np.float64)}


METHODS = {
    "gain": (_route, lambda value: {"gain": np.float64(value)}),
    "set": (_set, lambda value: {}),
    "step": (_step, lambda value: {}),
    "_contrast": (_contrast, _rows),
    "contrast_rows": (_contrast_rows, _rows),
    "_movement_on_device": (_movement_on_device, lambda value: {}),
}
