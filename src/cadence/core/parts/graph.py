"""``cadence.brain.Brain``, the neural graph, in the core: its parameters, its connectome's
synapses, its layout of blocks, its neuron rule and its apical compartment, and the settles,
solves and checks a moment makes on it (``settle_batch``, ``equilibrate``, ``residual``,
``apical_gain``).

A settled state (``BrainState``) crosses as ``state.v``, ``state.activation``,
``state.adaptation``, ``state.steps``, ``state.activity_change`` (where it has one) and
``state.published``: whether its activations are the ones a settle from it publishes (the
compiled kernels' record of the potentials and activations a state ended on, ``fused._publish``),
so a state the fast solve made publishes its own activations in the core as in the Python. A
nudge crosses as ``nudge.<field>``, a field left out where the nudge has none. A state's block
products only save work; the core makes them again, to the same bits.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from ...blocks import Layout
from ...brain import Apical, Brain, BrainState, Equilibrium, Nudge
from ...fused import sparse_offset
from ...neuron import Adaptation, NeuronModel

CLASS = Brain
_RECORD = 6  # the entries of a kept array's record (``fused._RECORD``)


def _apical_blocks_of(brain: Brain) -> Any:
    return None if brain.apical is None else (brain.apical, [], np.zeros(0), [])


def state_of(brain: Brain) -> dict[str, Any]:
    if brain.backend != "cpu":
        raise ValueError("the core holds a graph of the cpu backend")
    lay, c, m = brain.layout, brain.connectome, brain.neuron_model
    out: dict[str, Any] = {
        "efficacy": np.asarray(brain.efficacy, dtype=np.float64),
        "log_gain": np.asarray(brain.log_gain, dtype=np.float64),
        "bias": np.asarray(brain.bias, dtype=np.float64),
        "gain_pre": np.asarray(brain._gain_pre, dtype=np.float64),
        "pre": np.asarray(c.pre, dtype=np.int64),
        "post": np.asarray(c.post, dtype=np.int64),
        "dense_limit": np.int64(brain.dense_limit),
        "layout.starts": np.asarray(lay.starts, dtype=np.int64),
        "layout.pair_pre": np.asarray(lay.pair_pre, dtype=np.int64),
        "layout.pair_post": np.asarray(lay.pair_post, dtype=np.int64),
        "layout.offset": np.asarray(lay.offset, dtype=np.int64),
        "layout.edge_index": np.asarray(lay.edge_index, dtype=np.int64),
        "layout.sparse": sparse_offset(lay, _apical_blocks_of(brain)),
        "neuron.dt": np.float64(m.dt), "neuron.slope": np.float64(m.slope),
        "neuron.threshold": np.float64(m.threshold), "neuron.gain": np.float64(m.gain),
        "neuron.stimulus_amplitude": np.float64(m.stimulus_amplitude),
        "neuron.leak": np.float64(m.leak), "neuron.rest": np.float64(m.rest_emission),
    }
    if m.adaptation is not None:
        out["neuron.adaptation"] = np.array([m.adaptation.tau_steps, m.adaptation.strength])
    if brain.apical is not None:
        ap = brain.apical
        out["apical.synapses"] = np.asarray(ap.synapses, dtype=np.bool_)
        out["apical.groups"] = np.asarray(ap.groups, dtype=np.int64)
        out["apical.fraction"] = np.float64(ap.fraction)
        out["apical.floor"] = np.float64(ap.floor)
    return out


def restore(brain: Brain, state: dict[str, Any]) -> None:
    """The parameters, the rule and the compartment onto ``brain``, whose connectome and layout
    are the state's (refused otherwise)."""
    c, lay = brain.connectome, brain.layout
    if not (np.array_equal(state["pre"], c.pre) and np.array_equal(state["post"], c.post)):
        raise ValueError("the state's synapses are not this graph's")
    held = Layout(*(np.asarray(state[f"layout.{k}"], dtype=np.int64) for k in (
        "starts", "pair_pre", "pair_post", "offset", "edge_index")))
    if not all(np.array_equal(getattr(held, k), getattr(lay, k)) for k in (
            "starts", "pair_pre", "pair_post", "offset", "edge_index")):
        brain.layout = lay = held
    adaptation = None
    if "neuron.adaptation" in state:
        tau, strength = (float(x) for x in state["neuron.adaptation"])
        adaptation = Adaptation(tau_steps=tau, strength=strength)
    brain.neuron_model = NeuronModel(
        dt=float(state["neuron.dt"]), slope=float(state["neuron.slope"]),
        threshold=float(state["neuron.threshold"]), gain=float(state["neuron.gain"]),
        stimulus_amplitude=float(state["neuron.stimulus_amplitude"]), adaptation=adaptation,
        leak=float(state["neuron.leak"]))
    apical = None
    if "apical.synapses" in state:
        apical = brain.apical
        if apical is None or not (
            np.array_equal(apical.synapses, state["apical.synapses"])
            and np.array_equal(apical.groups, state["apical.groups"])
            and apical.fraction == float(state["apical.fraction"])
            and apical.floor == float(state["apical.floor"])
        ):
            apical = Apical(np.asarray(state["apical.synapses"], dtype=np.bool_),
                            np.asarray(state["apical.groups"], dtype=np.int64),
                            float(state["apical.fraction"]), float(state["apical.floor"]))
    brain.apical = apical
    brain.dense_limit = int(state["dense_limit"])
    brain._blocked = lay.size <= brain.dense_limit * brain.dense_limit
    brain._efficacy = np.array(state["efficacy"], dtype=np.float64)
    brain._log_gain = np.array(state["log_gain"], dtype=np.float64)
    brain._bias = np.array(state["bias"], dtype=np.float64)
    brain._gain_pre = np.array(state["gain_pre"], dtype=np.float64)
    brain._weights_host = brain._gain_pre * brain._efficacy
    brain._flat = None
    brain._csr = None
    brain.__dict__.pop("_apical_cache", None)


# -- a settled state, a nudge and a solve's report as named arrays

def published(state: BrainState) -> bool:
    """Whether ``state``'s activations are what a settle from it publishes: the kernels' record
    it keeps holds them, unmasked, at its potentials (``fused._publish``)."""
    products = state.__dict__.get("_products")
    if products is None:
        return False
    kept = products[3]
    head = len(kept) - _RECORD
    v = np.asarray(state.v, dtype=np.float64).ravel()
    s = np.asarray(state.activation, dtype=np.float64).ravel()
    count = v.size
    if head < 2 * count or kept[head] != 1.0 or kept[head + 1] != float(count):
        return False
    at = head - 2 * count
    return bool(np.array_equal(kept[at:at + count], v) and np.array_equal(kept[at + count:head], s))


def brain_state(state: BrainState, prefix: str = "state") -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}.v": np.array(state.v, dtype=np.float64),
        f"{prefix}.activation": np.array(state.activation, dtype=np.float64),
        f"{prefix}.adaptation": np.array(state.adaptation, dtype=np.float64),
        f"{prefix}.steps": np.int64(state.steps),
    }
    if state.activity_change is not None:
        out[f"{prefix}.activity_change"] = np.array(state.activity_change, dtype=np.float64)
    out[f"{prefix}.published"] = np.bool_(published(state))
    return out


def state_from(named: dict[str, Any], prefix: str, model: NeuronModel) -> BrainState:
    """The settled state ``brain_state`` crossed: a published one keeps the record the kernels
    read its activations by (``fused._publish``), under ``model``'s rule, and no products."""
    v = np.array(named[f"{prefix}.v"], dtype=np.float64)
    s = np.array(named[f"{prefix}.activation"], dtype=np.float64)
    change = named.get(f"{prefix}.activity_change")
    state = BrainState(v, s, np.array(named[f"{prefix}.adaptation"], dtype=np.float64),
                       int(np.asarray(named[f"{prefix}.steps"]).reshape(())),
                       activity_change=None if change is None else np.array(change, dtype=float))
    if bool(np.asarray(named[f"{prefix}.published"]).reshape(())):
        record = [1.0, float(v.size), model.slope, model.threshold, model.rest_emission, model.leak]
        kept = np.concatenate([v.ravel(), s.ravel(), record])
        gone = weakref.ref(np.zeros(1))  # no blocks: the products are made again
        state.__dict__["_products"] = (gone, gone, len(np.atleast_2d(v)), kept)
    return state


def nudge_arrays(nudge: Nudge, prefix: str = "nudge") -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}.target": np.array(nudge.target, dtype=np.float64),
        f"{prefix}.mask": np.array(nudge.mask, dtype=np.float64),
        f"{prefix}.beta": np.float64(nudge.beta),
    }
    if nudge.softmax_temperature is not None:
        out[f"{prefix}.softmax_temperature"] = np.array(nudge.softmax_temperature,
                                                         dtype=np.float64)
    if nudge.weight is not None:
        out[f"{prefix}.weight"] = np.array(nudge.weight, dtype=np.float64)
    if nudge.groups is not None:
        out[f"{prefix}.groups"] = np.array(nudge.groups, dtype=np.int64)
    if nudge.anchor is not None:
        out[f"{prefix}.anchor"] = np.array(nudge.anchor, dtype=np.float64)
        out[f"{prefix}.anchor_gain"] = np.array(nudge.anchor_gain, dtype=np.float64)
    if nudge.apical_gain is not None:
        out[f"{prefix}.apical_gain"] = np.array(nudge.apical_gain, dtype=np.float64)
    return out


def _common(drive: Any, state: BrainState | None, mask: Any,
            nudge: Nudge | None) -> dict[str, Any]:
    out: dict[str, Any] = {"drive": np.array(drive, dtype=np.float64)}
    if state is not None:
        out.update(brain_state(state))
    if mask is not None:
        out["mask"] = np.array(mask, dtype=np.float64)
    if nudge is not None:
        out.update(nudge_arrays(nudge))
    return out


def _settle_batch(drive: Any, *, steps: int = 60, state: BrainState | None = None,
                  mask: Any = None, trajectory: bool = False, nudge: Nudge | None = None,
                  tolerance: float | None = None) -> dict[str, Any]:
    out = _common(drive, state, mask, nudge)
    out["steps"] = np.int64(steps)
    out["trajectory"] = np.bool_(trajectory)
    if tolerance is not None:
        out["tolerance"] = np.float64(tolerance)
    return out


def _equilibrate(drive: Any, *, budget: int = 512, chunk: int = 32, tolerance: float = 1e-5,
                 state: BrainState | None = None, mask: Any = None, nudge: Nudge | None = None,
                 damping: int = 0, fast: bool = False) -> dict[str, Any]:
    out = _common(drive, state, mask, nudge)
    out.update(budget=np.int64(budget), chunk=np.int64(chunk), tolerance=np.float64(tolerance),
               damping=np.int64(damping), fast=np.bool_(fast))
    return out


def _residual(drive: Any, state: BrainState, *, nudge: Nudge | None = None, mask: Any = None,
              on_device: bool = True, own_precision: bool = False,
              per_unit: bool = False) -> dict[str, Any]:
    out = _common(drive, state, mask, nudge)
    out["per_unit"] = np.bool_(per_unit)
    return out


def _apical_gain(drive: Any, state: BrainState, *, own_precision: bool = False) -> dict[str, Any]:
    return _common(drive, state, None, None)


def equilibrium(value: Equilibrium) -> dict[str, Any]:
    out = brain_state(value.state)
    out.update({
        "residual": np.array(value.residual, dtype=np.float64),
        "steps": np.int64(value.steps), "tolerance": np.float64(value.tolerance),
        "residual_checks": np.int64(value.residual_checks),
        "damping_halvings": np.int64(value.damping_halvings),
        "stagnation_checks": np.int64(value.stagnation_checks),
    })
    if value.row_sweeps is not None:
        out["row_sweeps"] = np.int64(value.row_sweeps)
    return out


METHODS = {
    "settle_batch": (_settle_batch, brain_state),
    "equilibrate": (_equilibrate, equilibrium),
    "residual": (_residual, lambda value: {"residual": np.array(value, dtype=np.float64)}),
    "apical_gain": (_apical_gain, lambda value: {} if value is None else {
        "gain": np.array(value, dtype=np.float64)}),
}


def recording() -> None:
    """After the harness wraps ``settle_batch`` and ``residual``: a free solve still runs in one
    compiled call, as it does unrecorded (``Brain._solves_alone`` reads the class's own
    methods), so a recorded life takes the path an unrecorded one takes."""
    from ... import brain as module

    module._SETTLE_BATCH, module._RESIDUAL = Brain.settle_batch, Brain.residual
