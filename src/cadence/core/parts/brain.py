"""A composed brain (``cadence.generic.Brain``) in the core: each part it holds under its
attribute's name (``arousal.level``, ``recall.keys``, ``brain.efficacy``, ...), as that part's
module crosses it, the senses it declares and the spine's own state, and ``live``.

The senses cross as ``senses.<name>`` = ``[start, stop]`` in their order; a sense with fields
is followed by each field's ``senses.<name>/<field>``. The spine's settings and what a moment
leaves for the next cross under ``spine.``:

- ``spine.inputs``, ``spine.sensory``, ``spine.motor`` (the neurons the code drives and the
  motor neurons), ``spine.coded_cortex``, ``spine.coded_keys``, ``spine.surprise_private``,
  ``spine.hippocampus_maturity`` (whether the place records read the maturity's factor), and
  ``spine.map_value.<group>`` (each value group of the map by its place);
- ``spine.lived.*`` (the action ``live`` issued that awaits its outcome: observations, action,
  the value forecast, aroused, the brain's own best guess), present while that action's free
  state is the actor's; ``spine.moment.*`` (a sampled moment's observations and action),
  ``spine.prepared``, ``spine.surprise_read.*``, ``spine.surprise_seen``, ``spine.episode_told``;
- ``spine.last_arousal.*`` and ``spine.last_settlement.*``, the readings of the latest moment.

One copy of what two parts share: the carried gains are the memories' own (``trace.hearing``,
``history.hearing``, ``recall``'s ``hearing``), so the hearing crosses without ``gains`` and
without its memories' traces, and names each memory's attribute (``hearing.memory_of``, places
in ``Memories.ORDER``); the place records cross without the maturity's factor, which is the
maturity's.

The core holds the brain a Wild animal lives with: senses by name, a sparse code, the working
trace, the heard history, the recall memory, place records, bound records, surprise with its
focus, a map, arousal. ``state_of`` refuses a brain with another part by name."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from ...arousal import MODES
from ...generic import Brain
from ...memories import Memories
from . import part
from .graph import brain_state

CLASS = Brain
# attribute -> part module, in the order a state crosses
PARTS = {
    "arousal": "arousal", "sparse_code": "code", "working_memory": "trace",
    "efference": "efference", "history": "history", "recall": "recall",
    "hippocampus": "synaptic", "bound": "bound", "surprise": "surprise",
    "orienting": "orienting", "space": "map", "brain": "graph", "hearing": "hearing",
    "learner": "learner", "basal_ganglia": "actor",
}
MEMORIES = Memories.ORDER
OPERATIONS = ("act", "wait", "predict", "answer")
# what the core does not hold, by the attribute that holds it where it is present
UNHELD = ("learned_code", "fields", "_episodic", "message", "longterm", "_places", "_awaiting",
          "_forecast_read", "_judged")


def _senses(brain: Brain) -> dict[str, Any]:
    held = brain._senses
    if held is None or not held["declared"]:
        raise ValueError("the core holds a brain that declares its senses by name")
    if int(held["space"]):
        raise ValueError("the core holds a brain whose map is not one of its senses")
    out: dict[str, Any] = {}
    for name, columns in held["named"].items():
        if isinstance(columns, Mapping):
            spans = list(columns.values())
            out[f"senses.{name}"] = np.array([spans[0][0], spans[-1][1]], dtype=np.int64)
            for field, (first, last) in columns.items():
                out[f"senses.{name}/{field}"] = np.array([first, last], dtype=np.int64)
        else:
            out[f"senses.{name}"] = np.array(columns, dtype=np.int64)
    return out


def _unheld(brain: Brain) -> None:
    for name in UNHELD:
        if getattr(brain, name, None) is not None:
            raise ValueError(f"the core holds no {name.lstrip('_')}")
    if brain.sparse_code is None:
        raise ValueError("the core holds a brain with a sparse code")
    if brain.ports:
        raise ValueError("the core holds no ports")
    if brain._episode_moments or brain.learner.config.rehearsal:
        raise ValueError("the core holds no rehearsal")
    if brain.learner.config.compiled_moment:
        raise ValueError("the core holds the moment of the host's parts")
    if brain.arousal is None:
        raise ValueError("the core holds a brain with arousal")


def _hearing(brain: Brain, state: dict[str, Any]) -> dict[str, Any]:
    """The hearing's state without what its memories hold: the gains and the traces."""
    out = {k: v for k, v in state.items() if k != "gains" and not k.startswith("memory.")}
    held: list[Any] = []
    for route in brain.hearing.routes:
        if not any(route["memory"] is m for m in held):
            held.append(route["memory"])
    codes = []
    for memory in held:
        found = [k for k, name in enumerate(MEMORIES) if getattr(brain, name, None) is memory]
        if not found:
            raise ValueError("the core carries gains on the brain's own memories")
        codes.append(found[0])
    out["memory_of"] = np.array(codes, dtype=np.int64)
    return out


def _last_arousal(held: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": np.int64(MODES.index(held["mode"])),
        **{k: np.float64(held[k]) for k in ("level", "error", "surprise", "want")},
        "learned": np.bool_(held["learned"]), "recorded": np.bool_(held["recorded"]),
        "sweeps": np.int64(held["sweeps"]), "learning_sweeps": np.int64(held["learning_sweeps"]),
    }
    if held["temperature"] is not None:
        out["temperature"] = np.float64(held["temperature"])
    if held["temperatures"] is not None:
        out["temperatures"] = np.array(held["temperatures"], dtype=np.float64)
    if held["heated_slot"] is not None:
        out["heated_slot"] = np.int64(held["heated_slot"])
    return out


def _settlement(held: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "operation": np.int64(OPERATIONS.index(held["operation"])),
        "qualified": np.array(held["row_qualified"], dtype=bool),
        "residual": np.array(held["residual"], dtype=np.float64),
        **{k: np.int64(held[k]) for k in ("steps", "budget", "residual_checks",
                                           "damping_halvings", "stagnation_checks")},
        "tolerance": np.float64(held["tolerance"]),
    }
    for name in ("rescue_steps", "row_sweeps"):
        if name in held:
            out[name] = np.int64(held[name])
    return out


def state_of(brain: Brain) -> dict[str, Any]:
    _unheld(brain)
    out: dict[str, Any] = _senses(brain)
    out["spine.inputs"] = np.int64(brain.inputs)
    out["spine.sensory"] = np.asarray(brain.sensory_index, dtype=np.int64)
    out["spine.motor"] = np.asarray(brain.motor_index, dtype=np.int64)
    out["spine.coded_cortex"] = np.bool_(brain.coded_cortex)
    out["spine.coded_keys"] = np.bool_(brain._coded_keys)
    out["spine.surprise_private"] = np.bool_(brain._surprise_private)
    memory = brain.hippocampus
    out["spine.hippocampus_maturity"] = np.bool_(
        memory is not None and memory.maturity is not None)
    if memory is not None and memory.maturity is not None \
            and memory.maturity is not brain.basal_ganglia.maturity:
        raise ValueError("the place records read the actor's maturity")
    space = getattr(brain, "space", None)
    if space is not None:
        for k, name in enumerate(space.values):
            out[f"spine.map_value.{name}"] = np.int64(k)
    for attribute, name in PARTS.items():
        held = getattr(brain, attribute, None)
        if held is None:
            continue
        state = part(name).state_of(held)
        if attribute == "hearing":
            state = _hearing(brain, state)
        elif attribute == "hippocampus":
            state.pop("maturity", None)
        elif attribute == "learner":  # its graph is the brain's (``brain``)
            state = {k: v for k, v in state.items() if not k.startswith("graph.")}
        elif attribute == "basal_ganglia":  # its learner, memory and routes are the brain's
            if held.learner is not brain.learner or held.memory is not brain.recall \
                    or held.hearing is not brain.hearing:
                raise ValueError("the actor reaches the brain's own learner, recall and hearing")
            state = {k: v for k, v in state.items() if k != "hearing_routes"
                     and not k.startswith(("learner.", "memory.", "hearing."))}
        for key, value in state.items():
            out[f"{attribute}.{key}"] = value
    agent = brain.basal_ganglia
    lived = brain._lived
    if lived is not None and lived[4] is agent.state:
        x, action, value, aroused, _, own, place = lived
        if place is not None:
            raise ValueError("the core holds no learned code")
        out.update({"spine.lived.x": np.array(x, dtype=np.float64),
                    "spine.lived.action": np.array(action, dtype=np.int64),
                    "spine.lived.value": np.float64(value), "spine.lived.aroused": np.bool_(aroused),
                    "spine.lived.own": np.bool_(own)})
    if brain._moment is not None:
        out["spine.moment.x"] = np.array(brain._moment[0], dtype=np.float64)
        out["spine.moment.action"] = np.array(brain._moment[1], dtype=np.int64)
    if brain._moment_code is not None:
        raise ValueError("the core holds no learned code")
    if brain._prepared is not None:
        out["spine.prepared"] = np.array(brain._prepared, dtype=np.float64)
    read = brain._surprise_read
    if read is not None:
        out["spine.surprise_read.x"] = np.array(read[0], dtype=np.float64)
        out["spine.surprise_read.sizes"] = np.array(read[1], dtype=np.float64)
        out["spine.surprise_read.rows"] = np.array(read[2], dtype=bool)
    if brain._surprise_seen is not None:
        out["spine.surprise_seen"] = np.array(brain._surprise_seen, dtype=np.float64)
    out["spine.episode_told"] = np.array(brain._episode_told, dtype=np.int64)
    if brain._last_arousal is not None:
        out.update({f"spine.last_arousal.{k}": v
                    for k, v in _last_arousal(brain._last_arousal).items()})
    if brain._last_settlement is not None:
        out.update({f"spine.last_settlement.{k}": v
                    for k, v in _settlement(brain._last_settlement).items()})
    return out


def _heard(brain: Brain, inner: dict[str, Any]) -> dict[str, Any]:
    """The hearing's state as its part restores it: the gains as the memories, restored
    before, hold them."""
    out = {k: v for k, v in inner.items() if k != "memory_of"}
    out["gains"] = np.array([part("hearing")._gain(r) for r in brain.hearing.routes],
                            dtype=np.float64)
    return out


def _under(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    lead = prefix + "."
    return {k[len(lead):]: v for k, v in state.items() if k.startswith(lead)}


def restore(brain: Brain, state: dict[str, Any]) -> None:
    """Every part onto the brain's own, then the spine's state."""
    for attribute, name in PARTS.items():
        held = getattr(brain, attribute, None)
        inner = _under(state, attribute)
        if held is None or not inner:
            continue
        if attribute == "hearing":
            inner = _heard(brain, inner)
        elif attribute == "learner":
            inner = {**{f"graph.{k}": v for k, v in _under(state, "brain").items()}, **inner}
        elif attribute == "basal_ganglia":  # what the actor reaches, as the brain holds it
            reached = {f"learner.graph.{k}": v for k, v in _under(state, "brain").items()}
            reached.update({f"learner.{k}": v for k, v in _under(state, "learner").items()})
            reached.update({f"memory.{k}": v for k, v in _under(state, "recall").items()})
            reached.update({f"hearing.{k}": v
                            for k, v in _heard(brain, _under(state, "hearing")).items()})
            inner = {**reached, **inner}
        part(name).restore(held, inner)
    agent = brain.basal_ganglia
    if "spine.lived.x" in state:
        brain._lived = (np.array(state["spine.lived.x"], dtype=np.float64),
                        np.array(state["spine.lived.action"], dtype=np.int64),
                        float(state["spine.lived.value"]), bool(state["spine.lived.aroused"]),
                        agent.state, bool(state["spine.lived.own"]), None)
    else:
        brain._lived = None
    brain._moment = None if "spine.moment.x" not in state else (
        np.array(state["spine.moment.x"], dtype=np.float64),
        np.array(state["spine.moment.action"], dtype=np.int64))
    brain._prepared = (np.array(state["spine.prepared"], dtype=np.float64)
                       if "spine.prepared" in state else None)
    brain._surprise_read = None if "spine.surprise_read.x" not in state else (
        np.array(state["spine.surprise_read.x"], dtype=np.float64),
        np.array(state["spine.surprise_read.sizes"], dtype=np.float64),
        np.array(state["spine.surprise_read.rows"], dtype=bool))
    brain._surprise_seen = (np.array(state["spine.surprise_seen"], dtype=np.float64)
                            if "spine.surprise_seen" in state else None)
    brain._episode_told = np.array(state["spine.episode_told"], dtype=np.int64)
    held = _under(state, "spine.last_arousal")
    brain._last_arousal = None if not held else MappingProxyType({
        "mode": MODES[int(held["mode"])],
        **{k: float(held[k]) for k in ("level", "error", "surprise", "want")},
        "temperature": float(held["temperature"]) if "temperature" in held else None,
        "temperatures": tuple(float(t) for t in held["temperatures"])
        if "temperatures" in held else None,
        "heated_slot": int(held["heated_slot"]) if "heated_slot" in held else None,
        "learned": bool(held["learned"]), "recorded": bool(held["recorded"]),
        "sweeps": int(held["sweeps"]), "learning_sweeps": int(held["learning_sweeps"]),
    })
    held = _under(state, "spine.last_settlement")
    if not held:
        brain._last_settlement = None
    else:
        residual = tuple(float(r) for r in held["residual"])
        rows = tuple(bool(q) for q in held["qualified"])
        report: dict[str, Any] = {
            "operation": OPERATIONS[int(held["operation"])], "qualified": all(rows),
            "row_qualified": rows, "residual": residual, "max_residual": max(residual),
            "steps": int(held["steps"]), "budget": int(held["budget"]),
            "tolerance": float(held["tolerance"]),
            **{k: int(held[k]) for k in ("residual_checks", "damping_halvings",
                                         "stagnation_checks")},
        }
        for name in ("rescue_steps", "row_sweeps"):
            if name in held:
                report[name] = int(held[name])
        brain._last_settlement = MappingProxyType(report)


# -- the calls a replay checks

def whole_row(brain: Brain, observations: Mapping[str, Any]) -> np.ndarray:
    """The senses by name as one row of the brain's whole width, a silent sense zeros: the row
    the core's ``live`` takes."""
    row = np.zeros((1, brain.inputs))
    for name, value in observations.items():
        at = brain.senses[name]
        if isinstance(value, Mapping):
            for field, given in value.items():
                row[0, at[field]] = np.asarray(given, dtype=np.float64).reshape(-1)
        else:
            row[0, at] = np.asarray(value, dtype=np.float64).reshape(-1)
    return row


def _rows(observations: Any) -> dict[str, Any]:
    if isinstance(observations, Mapping):  # the whole row crosses (``calling``)
        return {}
    return {"observations": np.atleast_2d(np.asarray(observations, dtype=np.float64)),
            "mapping": np.bool_(False)}


def _live(observations: Any, *, reward: Any = None, done: Any = None, salience: Any = None,
          decision_id: Any = None) -> dict[str, Any]:
    out = _rows(observations)
    for name, value in (("reward", reward), ("salience", salience)):
        if value is not None:
            out[name] = np.asarray(value, dtype=np.float64).reshape(-1)
    if done is not None:
        out["done"] = np.asarray(done).reshape(-1)
    if decision_id is not None:
        out["decision_id"] = np.int64(decision_id)
    return out


def _act(observations: Any, *, greedy: bool = False, temperature: Any = None) -> dict[str, Any]:
    out = _rows(observations)
    out["greedy"] = np.bool_(greedy)
    if temperature is not None:
        out["temperature"] = np.asarray(temperature, dtype=np.float64).reshape(-1)
    return out


def _settled(x: Any) -> dict[str, Any]:
    return {"x": np.array(x, dtype=np.float64)}


def _choose(x: Any, drive: Any, free: Any, *, greedy: bool, temperature: Any = None,
            learn_code: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"x": np.array(x, dtype=np.float64),
                           "drive": np.array(drive, dtype=np.float64), **brain_state(free, "free"),
                           "greedy": np.bool_(greedy), "learn_code": np.bool_(learn_code)}
    if temperature is not None:
        out["temperature"] = np.asarray(temperature, dtype=np.float64).reshape(-1)
    return out


def _action(value: Any) -> dict[str, Any]:
    return {"action": np.asarray(value, dtype=np.int64)}


def _pair(value: Any) -> dict[str, Any]:
    drive, free = value
    return {"drive": np.array(drive, dtype=np.float64), **brain_state(free, "free")}


METHODS = {
    "live": (_live, _action),
    "act": (_act, _action),
    "say": (lambda observations: {}, lambda value: {"said": np.array(value, dtype=np.float64)}),
    "_observations": (lambda observations: _rows(observations),
                      lambda value: {"x": np.array(value, dtype=np.float64)}),
    "_settled": (_settled, _pair),
    "_surprised": (_settled, lambda value: {}),
    "_choose": (_choose, _action),
}


def calling(brain: Brain, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    """A call's arguments as the Python takes them, and what the core reads besides: the senses
    by name as the whole row (``observations``, ``mapping``), and for a part of a moment the
    row the bound records were given at it (``given.*``, ``Brain._bound_given``)."""
    named: dict[str, Any] = {}
    given = args[0] if args else kwargs.get("observations")
    if method in ("live", "act", "say", "_observations") and isinstance(given, Mapping):
        named["observations"] = whole_row(brain, given)
        named["mapping"] = np.bool_(True)
    if method in ("_surprised", "_choose") and brain._bound_given is not None:
        named["given.filled"] = np.array(brain._bound_given[0], dtype=np.float64)
        named["given.row"] = np.array(brain._bound_given[1], dtype=np.float64)
    return args, kwargs, lambda: named
