"""``cadence.learning.Learner`` in the core: its graph (``graph.*``, as ``cadence.core.parts.graph``
crosses it), its configuration (``config.<field>``; ``config.cross_entropy`` names the nudge, a
tolerance of None is absent), its outputs and their slots, which synapses and neurons are plastic,
the reciprocal pairing, the tie groups and per-synapse rates where it has them, its counts and
the optimizer's running moments. ``free`` settles the net's own answer, ``targets`` makes a
lesson's or an action's targets, ``apply`` takes a step (``_compiled_apply`` where the compiled
pass serves). What a moment changes in place crosses as a copy."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ...brain import BrainState
from ...learning import Learner, LearnerConfig
from . import graph as graph_part

CLASS = Learner
_MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")


def _config(config: LearnerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in LearnerConfig.__slots__:
        value = getattr(config, name)
        if name == "tolerance":
            if value is not None:
                out["config.tolerance"] = np.float64(value)
        elif name == "nudge":
            out["config.cross_entropy"] = np.bool_(value == "cross_entropy")
        elif isinstance(value, bool):
            out[f"config.{name}"] = np.bool_(value)
        elif isinstance(value, int):
            out[f"config.{name}"] = np.int64(value)
        else:
            out[f"config.{name}"] = np.float64(value)
    return out


def config_of(state: dict[str, Any]) -> LearnerConfig:
    fields: dict[str, Any] = {}
    for name in LearnerConfig.__slots__:
        if name == "tolerance":
            fields[name] = float(state["config.tolerance"]) if "config.tolerance" in state else None
        elif name == "nudge":
            fields[name] = "cross_entropy" if bool(state["config.cross_entropy"]) else "quadratic"
        else:
            value = np.asarray(state[f"config.{name}"]).reshape(())
            fields[name] = value.item()
    with warnings.catch_warnings():  # the config was made once; its warnings were given then
        warnings.simplefilter("ignore")
        return LearnerConfig(**fields)


def state_of(learner: Learner) -> dict[str, Any]:
    out = {f"graph.{k}": v for k, v in graph_part.state_of(learner.brain).items()}
    out.update(_config(learner.config))
    out["outputs"] = np.array(learner.output_index, dtype=np.int64)
    out["slot_sizes"] = np.array(learner.slot_sizes, dtype=np.int64)
    out["plastic_synapses"] = np.array(learner.plastic_synapses, dtype=np.bool_)
    out["plastic_neurons"] = np.array(learner.plastic_neurons, dtype=np.bool_)
    out["reciprocal"] = np.bool_(learner.reciprocal)
    if learner.tie_groups is not None:
        out["tie_groups"] = np.array(learner.tie_groups, dtype=np.int64)
    if learner.synapse_rate is not None:
        out["synapse_rate"] = np.array(learner.synapse_rate, dtype=np.float64)
    out["updates"] = np.int64(learner.updates)
    out["contrast_updates"] = np.int64(learner.contrast_updates)
    for name in _MOMENTS:
        out[name] = np.array(getattr(learner, name), dtype=np.float64)
    return out


def restore(learner: Learner, state: dict[str, Any]) -> None:
    """The graph's parameters, the configuration, the counts and the moments onto ``learner``,
    whose outputs, slots, masks, pairing, ties and rates are the state's (refused otherwise)."""
    same = (
        np.array_equal(learner.output_index, state["outputs"])
        and np.array_equal(learner.slot_sizes, state["slot_sizes"])
        and np.array_equal(np.asarray(learner.plastic_synapses), state["plastic_synapses"])
        and np.array_equal(np.asarray(learner.plastic_neurons), state["plastic_neurons"])
        and learner.reciprocal == bool(state["reciprocal"])
        and (learner.tie_groups is None) == ("tie_groups" not in state)
        and (learner.tie_groups is None
             or np.array_equal(np.asarray(learner.tie_groups), state["tie_groups"]))
        and (learner.synapse_rate is None) == ("synapse_rate" not in state)
    )
    if not same:
        raise ValueError("the state's outputs, masks and ties are not this learner's")
    graph = {k[6:]: v for k, v in state.items() if k.startswith("graph.")}
    graph_part.restore(learner.brain, graph)
    learner.config = config_of(state)
    if "synapse_rate" in state:
        learner.synapse_rate = np.array(state["synapse_rate"], dtype=np.float64)
    learner.updates = int(state["updates"])
    learner.contrast_updates = int(state["contrast_updates"])
    for name in _MOMENTS:
        setattr(learner, name, np.array(state[name], dtype=np.float64))


def _free(drive: Any, warm: BrainState | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"drive": np.array(drive, dtype=np.float64)}
    if warm is not None:
        out.update(graph_part.brain_state(warm, "warm"))
    return out


def _targets(labels: Any) -> dict[str, Any]:
    labels = np.array(labels)
    if labels.dtype.kind not in "iuf":
        labels = labels.astype(np.float64)  # what the core refuses as the Python does
    return {"labels": labels.astype(np.int64) if labels.dtype.kind in "iu" else labels}


def _apply(delta_scale: Any, delta_bias: Any) -> dict[str, Any]:
    return {"delta_scale": np.array(delta_scale, dtype=np.float64),
            "delta_bias": np.array(delta_bias, dtype=np.float64)}


def report(value: dict[str, float]) -> dict[str, Any]:
    return {f"report.{k}": np.float64(v) for k, v in value.items()}


METHODS = {
    "free": (_free, graph_part.brain_state),
    "targets": (_targets, lambda value: {"target": np.array(value, dtype=np.float64)}),
    "apply": (_apply, report),
}
