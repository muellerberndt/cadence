"""``cadence.plasticity.ActorCritic`` in the core, with what its moments reach: the learner and its
graph (``learner.*``), the memory whose parameters reward trains (``memory.*``, as
``cadence.core.parts.recall`` crosses it), the carried routes' gains (``hearing.*``; the routes
whose gain is the memory's own as ``hearing_routes``) and the maturity (``maturity.*``), each
where the actor holds one.

The actor's own state: its configuration (``config.<field>``, ``hold`` and
``eligibility_steps`` absent where None), the critic's neurons, weights and bias, the generator
(``rng.*``, NumPy's PCG64 state), the count of updates, the eligibility traces, the salience, the
held drive, the free state of the moment (``free.*``, with ``free.current`` where the graph that
settled it is the learner's), the pending decision (``pending.plus.*``, ``pending.minus.*``,
``pending.value``) and the pending contrasts of the memory, the routes and the entropy. Its
methods: ``act``, ``learn``, ``fade``, ``probabilities`` and ``value``."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ...brain import BrainState
from ...plasticity import ActorCritic, ActorCriticConfig
from .. import generator_state
from . import graph as graph_part
from . import learner as learner_part
from . import part

CLASS = ActorCritic
_TRACES = ("trace", "trace_bias", "trace_critic", "trace_memory", "trace_hearing")
_FLOATS = ("gamma", "lam", "eta", "eta_bias", "eta_critic", "dopamine_cap", "entropy")


def _under(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    lead = prefix + "."
    return {k[len(lead):]: v for k, v in state.items() if k.startswith(lead)}


def state_of(actor: ActorCritic) -> dict[str, Any]:
    actor.leave_device()
    if actor._trace_device is not None or actor._trace_mlx is not None:
        raise ValueError("the core holds an actor whose traces rest on the host")
    out: dict[str, Any] = {
        f"learner.{k}": v for k, v in learner_part.state_of(actor.learner).items()}
    if actor.memory is not None:
        out.update({f"memory.{k}": v for k, v in part("recall").state_of(actor.memory).items()})
    if actor.hearing is not None:
        out.update({f"hearing.{k}": v
                    for k, v in part("hearing").state_of(actor.hearing).items()})
        if actor.memory is not None:  # the routes whose gain is the memory's own
            out["hearing_routes"] = np.array(
                [k for k, route in enumerate(actor.hearing.routes)
                 if route["memory"] is actor.memory], dtype=np.int64)
    if actor.maturity is not None:
        m = actor.maturity
        out.update({"maturity.scale": np.float64(m.scale),
                    "maturity.recovery": np.float64(m.recovery),
                    "maturity.use": np.float64(m.use)})
    c = actor.config
    out.update({f"config.{name}": np.float64(getattr(c, name)) for name in _FLOATS})
    if c.eligibility_steps is not None:
        out["config.eligibility_steps"] = np.int64(c.eligibility_steps)
    if c.hold is not None:
        out["config.hold"] = np.float64(c.hold)
    out["critic_index"] = np.array(actor.critic_index, dtype=np.int64)
    out["w_critic"] = np.array(actor.w_critic, dtype=np.float64)
    out["b_critic"] = np.float64(actor.b_critic)
    out.update({f"rng.{k}": v for k, v in generator_state(actor.rng).items()})
    out["updates"] = np.int64(actor.updates)
    for name in _TRACES:
        value = getattr(actor, name)
        if value is not None:
            out[name] = np.array(value, dtype=np.float64)
    if actor.salience is not None:
        out["salience"] = np.array(actor.salience, dtype=np.float64)
    if actor._drive is not None:
        out["drive"] = np.array(actor._drive, dtype=np.float64)
    if actor._free is not None:
        out.update(graph_part.brain_state(actor._free, "free"))
        out["free.current"] = np.bool_(actor._free_brain is actor.learner.brain)
    if actor._pending is not None:
        kind, plus, minus, value = actor._pending
        if kind != "states":
            raise ValueError("the core holds a decision kept as its two nudged states")
        out.update(graph_part.brain_state(plus, "pending.plus"))
        out.update(graph_part.brain_state(minus, "pending.minus"))
        out["pending.value"] = np.array(value, dtype=np.float64)
    if actor._pending_memory is not None:
        out["pending_memory"] = np.array(actor._pending_memory, dtype=np.float64)
    if actor._pending_hearing is not None:
        out["pending_hearing"] = np.array(actor._pending_hearing, dtype=np.float64)
    if actor._pending_entropy is not None:
        out["pending_entropy.surprise"] = np.array(actor._pending_entropy[0], dtype=np.float64)
        out["pending_entropy.entropy"] = np.array(actor._pending_entropy[1], dtype=np.float64)
    return out


def restore(actor: ActorCritic, state: dict[str, Any]) -> None:
    """Every part of the state onto ``actor`` and what it holds; the critic's neurons, the
    configuration and the parts it holds are the state's (refused otherwise)."""
    if not np.array_equal(actor.critic_index, state["critic_index"]):
        raise ValueError("the state's critic is not this actor's")
    learner_part.restore(actor.learner, _under(state, "learner"))
    if actor.memory is not None:
        part("recall").restore(actor.memory, _under(state, "memory"))
    if actor.hearing is not None:
        part("hearing").restore(actor.hearing, _under(state, "hearing"))
    if actor.maturity is not None:
        actor.maturity.use = float(state["maturity.use"])
    fields: dict[str, Any] = {name: float(state[f"config.{name}"]) for name in _FLOATS}
    fields["eligibility_steps"] = (int(state["config.eligibility_steps"])
                                   if "config.eligibility_steps" in state else None)
    fields["hold"] = float(state["config.hold"]) if "config.hold" in state else None
    with warnings.catch_warnings():  # the config was made once; its warnings were given then
        warnings.simplefilter("ignore")
        actor.config = ActorCriticConfig(**fields)
    actor.w_critic = np.array(state["w_critic"], dtype=np.float64)
    actor.b_critic = float(state["b_critic"])
    held = dict(actor.rng.bit_generator.state)
    s, inc = (np.asarray(state[f"rng.{k}"], dtype=np.uint64) for k in ("state", "inc"))
    held["state"] = {"state": (int(s[0]) << 64) | int(s[1]),
                     "inc": (int(inc[0]) << 64) | int(inc[1])}
    held["has_uint32"] = int(np.asarray(state["rng.has_uint32"]).reshape(()))
    held["uinteger"] = int(np.asarray(state["rng.uinteger"]).reshape(()))
    actor.rng.bit_generator.state = held
    actor.updates = int(state["updates"])
    for name in _TRACES:
        setattr(actor, name, np.array(state[name], dtype=np.float64) if name in state else None)
    actor.salience = np.array(state["salience"], dtype=np.float64) if "salience" in state else None
    actor._drive = np.array(state["drive"], dtype=np.float64) if "drive" in state else None
    model = actor.learner.brain.neuron_model
    if "free.v" in state:
        actor._free = graph_part.state_from(state, "free", model)
        actor._free_brain = actor.learner.brain if bool(state["free.current"]) else None
    else:
        actor._free = actor._free_brain = None
    actor._pending = None
    if "pending.value" in state:
        actor._pending = ("states", graph_part.state_from(state, "pending.plus", model),
                          graph_part.state_from(state, "pending.minus", model),
                          np.array(state["pending.value"], dtype=np.float64))
    for name in ("pending_memory", "pending_hearing"):
        setattr(actor, f"_{name}",
                np.array(state[name], dtype=np.float64) if name in state else None)
    actor._pending_entropy = None
    if "pending_entropy.surprise" in state:
        actor._pending_entropy = (np.array(state["pending_entropy.surprise"], dtype=np.float64),
                                  np.array(state["pending_entropy.entropy"], dtype=np.float64))


def _temperature(out: dict[str, Any], temperature: Any) -> None:
    if temperature is not None:
        held = np.array(temperature, dtype=np.float64)
        out["temperature"] = held
        out["temperature.each"] = np.bool_(held.ndim > 0)


def _act(drive: Any, greedy: bool = False, temperature: Any = None, *,
         bias: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"drive": np.array(drive, dtype=np.float64), "greedy": np.bool_(greedy)}
    _temperature(out, temperature)
    if bias is not None:
        out["bias"] = np.array(bias, dtype=np.float64)
    return out


def _learn(reward: Any, done: Any, next_drive: Any, bootstrap: Any = None, *,
           observed: Any = None, warm: BrainState | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"reward": np.array(reward, dtype=np.float64),
                           "done": np.array(done), "next_drive": np.array(next_drive, dtype=float)}
    if bootstrap is not None:
        out["bootstrap"] = np.array(bootstrap, dtype=np.float64)
    if observed is not None:
        out["observed"] = np.array(observed)
    if warm is not None:
        out.update(graph_part.brain_state(warm, "warm"))
    return out


def _fade(done: Any = None) -> dict[str, Any]:
    return {} if done is None else {"done": np.array(done)}


def _probabilities(state: BrainState, temperature: Any = None, *,
                   bias: Any = None) -> dict[str, Any]:
    out = graph_part.brain_state(state)
    _temperature(out, temperature)
    if bias is not None:
        out["bias"] = np.array(bias, dtype=np.float64)
    return out


METHODS = {
    "act": (_act, lambda value: {"action": np.array(value, dtype=np.int64)}),
    "learn": (_learn, learner_part.report),
    "fade": (_fade, lambda value: {}),
    "probabilities": (_probabilities, lambda value: {"p": np.array(value, dtype=np.float64)}),
    "value": (lambda state: graph_part.brain_state(state),
              lambda value: {"value": np.array(value, dtype=np.float64)}),
}
