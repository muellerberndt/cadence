"""``cadence.memory.SynapticMemory`` in the core (a brain's ``hippocampus``, the place records):
its rule's settings, its ports, each stream's strengths and the shared records, the surprise
gate's noise scales, and the factor of the brain's maturity the gate reads (``maturity``, read
only: the maturity is a part of its own).

The core holds a delta store keyed on its pre neurons that does not fade; it refuses another
(a separator, a fade) and the compiled moment's read (``compiled_moment``, the spine's)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...memory import SynapticMemory

CLASS = SynapticMemory


def state_of(memory: SynapticMemory) -> dict[str, Any]:
    out: dict[str, Any] = {
        "config.decay": np.float64(memory.decay), "config.rate": np.float64(memory.rate),
        "config.amplitude": np.float64(memory.amplitude),
        "config.consolidation": np.float64(memory.consolidation),
        "config.normalize": np.bool_(memory.normalize), "config.replace": np.bool_(memory.replace),
        "config.delta": np.bool_(memory.rule == "delta"),
        "config.fade": np.float64(memory.fade), "config.fade_keep": np.float64(memory.fade_keep),
        "config.separator": np.bool_(memory.separator is not None),
        "config.compiled_moment": np.bool_(memory.__dict__.get("compiled_moment", False)),
        "pre": np.array(memory.pre, dtype=np.int64), "post": np.array(memory.post, dtype=np.int64),
        "strength": np.array(memory.strength, dtype=np.float64),
        "mass": np.array(memory.mass, dtype=np.float64), "writes": np.int64(memory.writes),
        "consolidated": np.array(memory.consolidated, dtype=np.float64),
    }
    if memory.gate_noise is not None:
        out["gate_noise"] = np.array(memory.gate_noise, dtype=np.float64)
    if memory.maturity is not None:
        out["maturity"] = np.float64(memory.maturity.factor())
    return out


def restore(memory: SynapticMemory, state: dict[str, Any]) -> None:
    """The strengths and the shared records onto a store of the same settings and ports
    (refused otherwise). The maturity's factor is the maturity's own and is not restored."""
    for name, value in state_of(memory).items():
        if name.startswith("config.") or name in ("pre", "post"):
            if not np.array_equal(np.asarray(state[name]), value):
                raise ValueError(f"{name}: a synaptic memory of other settings")
    shape = (memory.key_width, len(memory.post))
    strength = np.array(state["strength"], dtype=np.float64)
    consolidated = np.array(state["consolidated"], dtype=np.float64)
    if strength.ndim != 3 or strength.shape[1:] != shape or consolidated.shape != shape:
        raise ValueError("strength and consolidated hold the memory's ports")
    memory.leave_device()
    memory.strength, memory.consolidated = strength, consolidated
    memory.mass = np.array(state["mass"], dtype=np.float64)
    memory.writes = int(state["writes"])
    if ("gate_noise" in state) != (memory.gate_noise is not None):
        raise ValueError("gate_noise: a synaptic memory with another surprise gate")
    if "gate_noise" in state:
        memory.gate_noise = np.array(state["gate_noise"], dtype=np.float64)


def _observe(key: Any, value: Any, write: Any = None, *, salience: Any = None,
             value_mask: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"key": np.asarray(key, dtype=np.float64),
                           "value": np.asarray(value, dtype=np.float64)}
    if write is not None:
        out["write"] = np.asarray(write)
    if salience is not None:
        out["salience"] = np.asarray(salience, dtype=np.float64)
    if value_mask is not None:
        out["value_mask"] = np.asarray(value_mask)
    return out


def _stimulate(drive: Any, inplace: bool = False, key: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"drive": np.asarray(drive, dtype=np.float64)}
    if key is not None:
        out["key"] = np.asarray(key, dtype=np.float64)
    return out


def _stimulus(drive: Any, moment: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"drive": np.asarray(drive, dtype=np.float64)}
    if moment.place is not None:
        out["place"] = np.asarray(moment.place, dtype=np.float64)
    out["stores"] = np.bool_(moment.stores)
    return out


def _held(value: dict[str, Any]) -> dict[str, Any]:
    out = {"strength": np.asarray(value["strength"], dtype=np.float64),
           "mass": np.asarray(value["mass"], dtype=np.float64),
           "writes": np.int64(value["writes"]),
           "consolidated": np.asarray(value["consolidated"], dtype=np.float64)}
    if value.get("gate_noise") is not None:
        out["gate_noise"] = np.asarray(value["gate_noise"], dtype=np.float64)
    return out


METHODS = {
    "observe": (_observe, lambda value: {}),
    "recall": (lambda key: {"key": np.asarray(key, dtype=np.float64)},
               lambda value: {"read": np.asarray(value, dtype=np.float64)}),
    "stimulate": (_stimulate, lambda value: {"drive": np.asarray(value, dtype=np.float64)}),
    "stimulus": (_stimulus, lambda value: {"drive": np.asarray(value, dtype=np.float64)}),
    "held_values": (lambda: {}, _held),
    "reside": (lambda kernel: {}, lambda value: {}),
}
