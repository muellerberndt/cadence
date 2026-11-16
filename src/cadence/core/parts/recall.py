"""``cadence.stream.Recall`` in the core: the recall memory with its heads, as a Wild moment runs
it on the host. Its settings, the regions it reads and drives, every head's maps, each stream's
slots, the moment it addressed and the reads it held.

The core holds the memory Wild's animals carry: one event per slot (``window`` 1), no command
in a slot, no read of when, no gains on tokens, the cortex as it settled, no slot strengths, and
the host's path (a memory the compiled moment reads is refused). ``state_of`` refuses another.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...resident import _Moment
from ...stream import Recall

CLASS = Recall
FLOATS = ("amplitude", "temperature", "recency", "address", "address_reward", "position_reward",
          "position_rate", "slope")
INTEGERS = ("capacity", "heads", "position")
SWITCHES = ("next", "boundary", "anchor", "relative", "drive_gain")
KINDS = ("address", "position", "anchor", "relative", "slope", "grid")
PORTS = ("recall", "next")
MOMENT = _Moment._fields  # event, cortex, address, key, inputs
STREAMS = ("keys", "values", "nexts", "held", "previous", "ended", "told", "focus")


def _held(memory: Recall) -> None:
    """The memory is one the core holds (the module's text), or ``ValueError``."""
    beside = [name for name, given in (
        ("window", memory.window != 1), ("action", memory.action), ("when", memory.when),
        ("gain", bool(memory.gain)), ("tokens", memory.tokens is not None),
        ("cortex_norm", memory.cortex_norm), ("strength", memory.strength),
        ("a device", memory.__dict__.get("_device") is not None),
    ) if given]
    if memory.__dict__.get("compiled_moment", False) and memory.heads == 1:
        beside.append("the compiled moment")
    if beside:
        raise ValueError(f"the core holds the recall memory without {', '.join(beside)}")


def _maps(memory: Recall, kind: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return list(memory.head_maps if kind == "address" else getattr(memory, f"{kind}_maps"))


Synapses = tuple[np.ndarray, np.ndarray, np.ndarray]


def _ports(memory: Recall) -> list[tuple[str, np.ndarray, Synapses]]:
    """Every head's ports, head 0's first: their names, neurons and synapses into the
    association (indices, source and association coordinates)."""
    out: list[tuple[str, np.ndarray, Synapses]] = []
    for h, (neurons, synapses) in enumerate(zip(memory._heads, memory._head_synapses,
                                                strict=True)):
        for k, port in enumerate(PORTS[:2 if memory.next else 1]):
            out.append((f"{h}.{port}", neurons[k], synapses[k]))
    return out


def state_of(memory: Recall) -> dict[str, Any]:
    _held(memory)
    out: dict[str, Any] = {f"config.{name}": np.float64(getattr(memory, name)) for name in FLOATS}
    out.update({f"config.{name}": np.int64(getattr(memory, name)) for name in INTEGERS})
    out.update({f"config.{name}": np.bool_(getattr(memory, name)) for name in SWITCHES})
    out["config.grid"] = np.array(memory.grid, dtype=np.int64)
    out["hearing"] = np.float64(memory.hearing)
    out["association"] = np.asarray(memory.association, dtype=np.int64)
    for name, neurons, (index, rows, cols) in _ports(memory):
        out[f"ports.{name}"] = np.asarray(neurons, dtype=np.int64)
        out[f"synapses.{name}.index"] = np.asarray(index, dtype=np.int64)
        out[f"synapses.{name}.rows"] = np.asarray(rows, dtype=np.int64)
        out[f"synapses.{name}.cols"] = np.asarray(cols, dtype=np.int64)
    for kind in KINDS:
        for h, maps in enumerate(_maps(memory, kind)):
            for part, array in zip("xcb", maps, strict=True):
                out[f"maps.{kind}.{h}.{part}"] = np.asarray(array, dtype=np.float64)
    for name in STREAMS:
        out[name] = np.array(getattr(memory, name))  # a copy: a write moves slots in place
    out["writes"] = np.int64(memory.writes)
    out["trace"] = np.array(memory.trace, dtype=np.float64)
    out["private"] = np.bool_(memory._private)
    out["address_held"] = np.bool_(memory._address_held)
    if memory.last is not None:
        for name, array in zip(MOMENT, memory.last, strict=True):
            out[f"last.{name}"] = np.array(array, dtype=np.float64)
    if memory._attended is not None:
        out["attended"] = np.array(memory._attended, dtype=np.float64)
    reads = memory._head_reads
    if reads is not None:
        out["reads.writes"] = np.int64(reads[2])
        for h, address in enumerate(reads[0]):
            out[f"reads.address.{h}"] = np.array(address, dtype=np.float64)
        for h, attended in enumerate(reads[1]):
            if attended is not None:
                out[f"reads.attended.{h}"] = np.array(attended, dtype=np.float64)
    return out


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def restore(memory: Recall, state: dict[str, Any]) -> None:
    """The state onto the memory. The regions and synapses are the memory's anatomy, which a
    state does not rewire: another is refused (``ValueError``)."""
    _held(memory)
    if int(state["config.heads"]) != memory.heads or bool(state["config.next"]) != memory.next:
        raise ValueError("a recall state is restored onto the memory of its anatomy")
    for name, neurons, (index, rows, cols) in _ports(memory):
        for label, array in (("", neurons), (".index", index), (".rows", rows),
                             (".cols", cols)):
            given = state[f"ports.{name}" if not label else f"synapses.{name}{label}"]
            if not np.array_equal(given, array):
                raise ValueError("a recall state is restored onto the memory of its anatomy")
    if not np.array_equal(state["association"], memory.association):
        raise ValueError("a recall state is restored onto the memory of its anatomy")
    for name in FLOATS:
        setattr(memory, name, float(state[f"config.{name}"]))
    for name in INTEGERS:
        setattr(memory, name, int(state[f"config.{name}"]))
    for name in SWITCHES:
        setattr(memory, name, bool(state[f"config.{name}"]))
    memory.grid = tuple(int(m) for m in state["config.grid"])
    memory.hearing = float(state["hearing"])
    for kind in KINDS:
        maps = []
        h = 0
        while f"maps.{kind}.{h}.x" in state:
            maps.append(tuple(np.array(state[f"maps.{kind}.{h}.{part}"], dtype=np.float64)
                              for part in "xcb"))
            h += 1
        setattr(memory, "head_maps" if kind == "address" else f"{kind}_maps", maps)
    for name in STREAMS:
        setattr(memory, name, np.array(state[name]))
    memory.writes = int(state["writes"])
    memory.trace = np.array(state["trace"], dtype=np.float64)
    memory._private = bool(state["private"])
    memory._address_held = bool(state["address_held"])
    memory.last = (
        _Moment(*(np.array(state[f"last.{name}"], dtype=np.float64) for name in MOMENT))
        if "last.event" in state else None
    )
    memory._attended = (
        _read_only(np.array(state["attended"], dtype=np.float64)) if "attended" in state
        else None
    )
    memory._head_reads = None
    if "reads.writes" in state:
        heads = memory.heads
        memory._head_reads = (
            tuple(np.array(state[f"reads.address.{h}"], dtype=np.float64) for h in range(heads)),
            tuple(_read_only(np.array(state[f"reads.attended.{h}"], dtype=np.float64))
                  if f"reads.attended.{h}" in state else None for h in range(heads)),
            int(state["reads.writes"]),
        )
    memory._rooms = {}
    memory._contents, memory._content_keys = [], None


def _rows(activation: Any) -> np.ndarray:
    return np.atleast_2d(np.asarray(activation, dtype=np.float64))


def _stimulate(drive: np.ndarray, observations: np.ndarray) -> dict[str, Any]:
    return {"drive": np.asarray(drive, dtype=np.float64),
            "observations": np.asarray(observations, dtype=np.float64)}


def _drive(value: np.ndarray) -> dict[str, Any]:
    return {"drive": np.asarray(value, dtype=np.float64)}


def _witness(observations: np.ndarray, state: Any, command: Any = None,
             surprise: Any = None) -> dict[str, Any]:
    out = {"observations": np.asarray(observations, dtype=np.float64),
           "activation": _rows(state.activation)}
    if surprise is not None:  # a memory without strength refuses it
        out["surprise"] = np.asarray(surprise, dtype=np.float64)
    return out


def _contrast_rows(graph: Any, plus: Any, minus: Any) -> dict[str, Any]:
    return {"weights": np.asarray(graph._weights, dtype=np.float64),
            "plus": _rows(plus.activation), "minus": _rows(minus.activation)}


def _eligibility(value: np.ndarray | None) -> dict[str, Any]:
    return {} if value is None else {"rows": np.asarray(value, dtype=np.float64)}


def _propose(step: np.ndarray) -> dict[str, Any]:
    return {"step": np.asarray(step, dtype=np.float64)}


def _proposal(value: tuple[Any, ...] | None) -> dict[str, Any]:
    """A proposal as named arrays: every head's address maps, the step's norm, and with the
    position step every head's position maps (and anchored maps); empty for none."""
    if value is None:
        return {}
    if len(value) > 6:
        raise ValueError("the core's recall memory steps no gains")
    w_x, w_c, b, norm, position, anchor = value
    if isinstance(w_x, np.ndarray):  # one head
        w_x, w_c, b = (w_x,), (w_c,), (b,)
    out: dict[str, Any] = {}
    for h, maps in enumerate(zip(w_x, w_c, b, strict=True)):
        for part, array in zip("xcb", maps, strict=True):
            out[f"address.{h}.{part}"] = np.asarray(array, dtype=np.float64)
    out["norm"] = np.float64(norm)
    for kind, held in (("position", position), ("anchor", anchor)):
        for h, maps in enumerate(held or ()):
            for part, array in zip("xcb", maps, strict=True):
                out[f"{kind}.{h}.{part}"] = np.asarray(array, dtype=np.float64)
    return out


def _accept(proposal: tuple[Any, ...]) -> dict[str, Any]:
    return _proposal(proposal)


METHODS = {
    "stimulate": (_stimulate, _drive),
    "witness": (_witness, lambda value: {}),
    "contrast_rows": (_contrast_rows, _eligibility),
    "propose": (_propose, _proposal),
    "accept": (_accept, lambda value: {}),
}
