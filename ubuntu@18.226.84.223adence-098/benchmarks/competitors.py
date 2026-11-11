"""Conventional online learners as competitor arms of the chambers.

An online recurrent actor-critic with the same information as the brain: the chamber's
observation enters a recurrent cell (an Elman cell, ``elman``, or a gated recurrent unit,
``gru``) whose state carries across moments; a softmax policy and a linear value read the
state. Learning is online at every moment, TD(lambda) eligibility traces over every weight
with the gradient taken through the current moment's hidden state only (the previous state
held fixed), so nothing is stored or replayed and no batch is formed. One moment's
gradient of the executed action's log-probability and of the value are each bounded in
norm by ``clip``, and so is the TD error; accumulated traces and the weights themselves are
not bounded by this guard. The ``elman`` cell is the port of the key-door chamber's
``recurrent`` arm (``benchmarks/keydoor/key_door.py``); the ``gru`` cell is the same
actor-critic with gates, its gradients written by hand and checked against finite
differences by the chambers' tests. NumPy only, outside the library and its requirements.

Work is counted: parameter updates, the forward and backward passes, the trace decays and
accumulations and the weight updates as multiply-accumulates (an analytic count, one per
parameter touched), the guard's activations and the parameter count. A chamber loads this
file by path (``load``) and records the learner's settings and ledger in its receipt.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

CELLS = ("elman", "gru")
SETTINGS = ("hidden", "alpha", "alpha_value", "gamma", "lam", "temperature", "clip")


def check_settings(settings: dict[str, Any]) -> None:
    """Refuse a learner setting outside its declared range (the key-door protocol's rule)."""
    hidden = settings["hidden"]
    if type(hidden) is not int or hidden <= 0:
        raise ValueError("the competitor's hidden size must be a positive integer")
    for name in SETTINGS[1:]:
        value = float(settings[name])
        if (
            not np.isfinite(value)
            or value < 0
            or (name in ("temperature", "clip") and value == 0)
            or (name in ("gamma", "lam") and value > 1)
        ):
            raise ValueError(f"invalid competitor {name}")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    return probs / probs.sum()


class ElmanCell:
    """``h = tanh(Wx x + Wh h_prev + b)``."""

    name = "elman"

    def __init__(self, inputs: int, hidden: int, rng: np.random.Generator) -> None:
        self.inputs, self.hidden = inputs, hidden
        self.Wx = rng.normal(0.0, 0.3, (hidden, inputs))
        self.Wh = rng.normal(0.0, 0.3, (hidden, hidden)) / np.sqrt(hidden)
        self.b = np.zeros(hidden)

    def parameters(self) -> dict[str, np.ndarray]:
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        h = np.tanh(self.Wx @ x + self.Wh @ h_prev + self.b)
        return h, {"x": x, "h_prev": h_prev, "h": h}

    def backward(self, g: np.ndarray, cache: dict[str, Any]) -> dict[str, np.ndarray]:
        """The gradient of ``g . h`` with respect to the cell's weights, ``h_prev`` fixed."""
        gz = g * (1.0 - cache["h"] ** 2)
        return {"Wx": np.outer(gz, cache["x"]), "Wh": np.outer(gz, cache["h_prev"]), "b": gz}

    @property
    def forward_macs(self) -> int:
        return self.hidden * (self.inputs + self.hidden)

    @property
    def backward_macs(self) -> int:
        return self.hidden * (self.inputs + self.hidden)


class GRUCell:
    """``z = s(Wz x + Uz h_prev + bz)``, ``r = s(Wr x + Ur h_prev + br)``,
    ``n = tanh(Wn x + Un (r * h_prev) + bn)``, ``h = (1 - z) * n + z * h_prev``."""

    name = "gru"

    def __init__(self, inputs: int, hidden: int, rng: np.random.Generator) -> None:
        self.inputs, self.hidden = inputs, hidden
        self.weights: dict[str, np.ndarray] = {}
        for gate in "zrn":
            self.weights["W" + gate] = rng.normal(0.0, 0.3, (hidden, inputs))
            self.weights["U" + gate] = rng.normal(0.0, 0.3, (hidden, hidden)) / np.sqrt(hidden)
            self.weights["b" + gate] = np.zeros(hidden)

    def parameters(self) -> dict[str, np.ndarray]:
        return self.weights

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        w = self.weights
        z = 1.0 / (1.0 + np.exp(-(w["Wz"] @ x + w["Uz"] @ h_prev + w["bz"])))
        r = 1.0 / (1.0 + np.exp(-(w["Wr"] @ x + w["Ur"] @ h_prev + w["br"])))
        reset = r * h_prev
        n = np.tanh(w["Wn"] @ x + w["Un"] @ reset + w["bn"])
        h = (1.0 - z) * n + z * h_prev
        return h, {"x": x, "h_prev": h_prev, "z": z, "r": r, "reset": reset, "n": n}

    def backward(self, g: np.ndarray, cache: dict[str, Any]) -> dict[str, np.ndarray]:
        """The gradient of ``g . h`` with respect to the cell's weights, ``h_prev`` fixed."""
        x, h_prev, z, r, n = (cache[k] for k in ("x", "h_prev", "z", "r", "n"))
        gn = g * (1.0 - z) * (1.0 - n * n)
        gz = g * (h_prev - n) * z * (1.0 - z)
        gr = (self.weights["Un"].T @ gn) * h_prev * r * (1.0 - r)
        return {
            "Wn": np.outer(gn, x),
            "Un": np.outer(gn, cache["reset"]),
            "bn": gn,
            "Wz": np.outer(gz, x),
            "Uz": np.outer(gz, h_prev),
            "bz": gz,
            "Wr": np.outer(gr, x),
            "Ur": np.outer(gr, h_prev),
            "br": gr,
        }

    @property
    def forward_macs(self) -> int:
        return 3 * self.hidden * (self.inputs + self.hidden)

    @property
    def backward_macs(self) -> int:
        return 3 * self.hidden * (self.inputs + self.hidden) + self.hidden * self.hidden


def make_cell(name: str, inputs: int, hidden: int, rng: np.random.Generator) -> Any:
    if name == "elman":
        return ElmanCell(inputs, hidden, rng)
    if name == "gru":
        return GRUCell(inputs, hidden, rng)
    raise ValueError(f"unknown cell {name!r}")


class ActorCritic:
    """One online recurrent actor-critic life. ``act(x, reward)`` learns from the reward of
    the preceding action, then answers the observation ``x``; ``reward`` is ``None`` at the
    first moment. ``done`` ends an attempt where the chamber marks one (the traces clear and
    the state resets); a chamber whose brain receives no such mark passes none. ``frozen``
    answers greedily without learning. ``probs`` is the policy that chose the last action,
    ``last_delta`` the TD error of the last update (``None`` when none happened), ``work``
    the ledger. ``clone`` gives a copy for probes and ``step`` advances it without
    learning."""

    def __init__(
        self,
        inputs: int,
        actions: int,
        seed: int,
        *,
        cell: str = "elman",
        hidden: int = 16,
        alpha: float = 0.02,
        alpha_value: float = 0.04,
        gamma: float = 0.95,
        lam: float = 0.8,
        temperature: float = 1.0,
        clip: float = 5.0,
    ) -> None:
        check_settings(
            {
                "hidden": hidden,
                "alpha": alpha,
                "alpha_value": alpha_value,
                "gamma": gamma,
                "lam": lam,
                "temperature": temperature,
                "clip": clip,
            }
        )
        self.inputs, self.actions, self.hidden = inputs, actions, hidden
        self.rng = np.random.default_rng(seed)
        self.cell = make_cell(cell, inputs, hidden, self.rng)
        self.alpha, self.alpha_value, self.gamma, self.lam = alpha, alpha_value, gamma, lam
        self.temperature, self.clip = temperature, clip
        self.Wa = np.zeros((actions, hidden))
        self.ba = np.zeros(actions)
        self.wv = np.zeros(hidden)
        self.bv = np.zeros(1)
        self.h = np.zeros(hidden)
        self.frozen = False
        self.memo: float | None = None
        self.probs = np.full(actions, 1.0 / actions)
        self.last_delta: float | None = None
        self.actor_trace = {k: np.zeros_like(v) for k, v in self._actor().items()}
        self.value_trace = {k: np.zeros_like(v) for k, v in self._value().items()}
        self.work = {
            "updates": 0,
            "forward_passes": 0,
            "clipped": 0,
            "value_clipped": 0,
            "delta_clipped": 0,
            "multiply_accumulates": 0,
        }

    # ------------------------------------------------------------------------ parameters

    def _actor(self) -> dict[str, np.ndarray]:
        return {"Wa": self.Wa, "ba": self.ba, **self.cell.parameters()}

    def _value(self) -> dict[str, np.ndarray]:
        return {"wv": self.wv, "bv": self.bv, **self.cell.parameters()}

    def parameters(self) -> dict[str, np.ndarray]:
        readouts = {"Wa": self.Wa, "ba": self.ba, "wv": self.wv, "bv": self.bv}
        return {**readouts, **self.cell.parameters()}

    @property
    def parameter_count(self) -> int:
        return int(sum(v.size for v in self.parameters().values()))

    def describe(self) -> dict[str, Any]:
        return {
            "cell": self.cell.name,
            "inputs": self.inputs,
            "actions": self.actions,
            "hidden": self.hidden,
            "parameters": self.parameter_count,
            "alpha": self.alpha,
            "alpha_value": self.alpha_value,
            "gamma": self.gamma,
            "lam": self.lam,
            "temperature": self.temperature,
            "clip": self.clip,
        }

    def ledger(self) -> dict[str, int]:
        return {**self.work, "parameters": self.parameter_count, "hidden": self.hidden}

    # --------------------------------------------------------------------------- forward

    def evaluate(
        self, x: np.ndarray, h_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        """The hidden state, policy and value of ``x`` after ``h_prev``; no state changes."""
        h, cache = self.cell.forward(np.asarray(x, dtype=float).reshape(-1), h_prev)
        probs = softmax((self.Wa @ h + self.ba) / self.temperature)
        value = float(self.wv @ h + self.bv[0])
        self.work["forward_passes"] += 1
        self.work["multiply_accumulates"] += self.cell.forward_macs + self.hidden * (
            self.actions + 1
        )
        return h, probs, value, cache

    def policy(self, x: np.ndarray, h_prev: np.ndarray | None = None) -> np.ndarray:
        """The policy's probabilities for ``x`` from ``h_prev`` (the cleared state when
        absent) without changing anything."""
        state = np.zeros(self.hidden) if h_prev is None else h_prev
        return self.evaluate(x, state)[1]

    def clone(self, *, cleared: bool = True) -> ActorCritic:
        """A copy for a probe; ``cleared`` starts it from a zero hidden state."""
        twin = copy.deepcopy(self)
        if cleared:
            twin.h = np.zeros(self.hidden)
        return twin

    def step(self, x: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Answer ``x`` and carry the hidden state, without learning: the greedy action, or
        a sample of the policy when ``rng`` is given. For probes on a clone."""
        h, probs, _, _ = self.evaluate(x, self.h)
        self.h = h
        self.probs = probs
        if rng is None:
            return int(np.argmax(probs))
        return self._sample(probs, rng)

    @staticmethod
    def _sample(probs: np.ndarray, rng: np.random.Generator) -> int:
        return int(min(np.searchsorted(np.cumsum(probs), rng.random()), len(probs) - 1))

    # ---------------------------------------------------------------------------- living

    def act(self, x: np.ndarray, reward: float | None, done: bool = False) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        h_prev = self.h
        h, probs, value, cache = self.evaluate(x, h_prev)
        self.last_delta = None
        if reward is not None and self.memo is not None and not self.frozen:
            target = float(reward) + (0.0 if done else self.gamma * value)
            raw = target - self.memo
            delta = float(np.clip(raw, -self.clip, self.clip))
            self.work["delta_clipped"] += int(delta != raw)
            self.Wa += self.alpha * delta * self.actor_trace["Wa"]
            self.ba += self.alpha * delta * self.actor_trace["ba"]
            self.wv += self.alpha_value * delta * self.value_trace["wv"]
            self.bv += self.alpha_value * delta * self.value_trace["bv"]
            for name, shared in self.cell.parameters().items():
                shared += self.alpha * delta * self.actor_trace[name]
                shared += self.alpha_value * delta * self.value_trace[name]
            self.work["updates"] += 1
            self.work["multiply_accumulates"] += 2 * self.parameter_count
            self.last_delta = delta
            if done:
                for trace in (self.actor_trace, self.value_trace):
                    for v in trace.values():
                        v[:] = 0.0
                h_prev = np.zeros(self.hidden)
            # Bootstrap from the preceding weights, then choose and differentiate one
            # coherent policy after learning (the key-door chamber's correction).
            h, probs, value, cache = self.evaluate(x, h_prev)
        elif done:
            h_prev = np.zeros(self.hidden)
            h, probs, value, cache = self.evaluate(x, h_prev)
        if self.frozen:
            action = int(np.argmax(probs))
        else:
            action = self._sample(probs, self.rng)
        # The eligibility of this decision: decayed traces plus the gradient of the executed
        # action's log-probability and of the value through h, each bounded in norm.
        onehot = np.zeros(self.actions)
        onehot[action] = 1.0
        dlogits = (onehot - probs) / self.temperature
        actor = {"Wa": np.outer(dlogits, h), "ba": dlogits}
        actor.update(self.cell.backward(self.Wa.T @ dlogits, cache))
        critic = {"wv": h.copy(), "bv": np.ones(1)}
        critic.update(self.cell.backward(self.wv, cache))
        self.work["multiply_accumulates"] += 2 * self.cell.backward_macs + 2 * self.hidden * (
            self.actions + 1
        )
        scale = min(1.0, self.clip / self._norm(actor))
        self.work["clipped"] += int(scale < 1.0)
        value_scale = min(1.0, self.clip / self._norm(critic))
        self.work["value_clipped"] += int(value_scale < 1.0)
        decay = self.gamma * self.lam
        for trace, gradient, factor in (
            (self.actor_trace, actor, scale),
            (self.value_trace, critic, value_scale),
        ):
            for name, v in trace.items():
                v *= decay
                v += factor * gradient[name]
        self.work["multiply_accumulates"] += 4 * self.parameter_count
        self.h = h
        self.memo = value
        self.probs = probs
        return action

    @staticmethod
    def _norm(gradient: dict[str, np.ndarray]) -> float:
        return float(np.sqrt(sum(float(np.sum(v * v)) for v in gradient.values()))) or 1e-300


def load() -> Any:
    """This module, for a chamber that lives beside it and loads it by path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve()
    spec = importlib.util.spec_from_file_location("competitors", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
