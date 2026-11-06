"""The brain's own graph trained by exact gradients: #183's same-graph reference on the chamber.

The attention ablations put the recall memory's structure on the transformer's training. This
file puts the transformer's training on the recall memory's structure: the recipe brain's graph,
the same sentences and the same exposure, with the local learning replaced in two rungs.

The moment. The association and motor regions settle on fixed sources (sensory, the working
trace's prefrontal, recall, next):

    v_a = W_sa s(x) + W_pa s(3 c) + W_ra s(3 sum p V) + W_na s(3 sum p N) + W_am s(v_m) + b_a
    v_m = W_am' s(v_a) + b_m

``s`` the rectified sigmoid, ``c`` the association of the moment before and ``p`` the held read,
addressed before the settle from the event ``x`` and ``c``. The cost is ``T`` times the
cross-entropy of the motor activity at the temperature ``T`` of the lesson's nudge.

- ``recipe``: the chamber's local contrast. ``deep`` is the recipe at the documented point for a
  brain with an apical compartment (``apical=0.25, teaching="deep"``), ``deepna`` the teaching
  point alone. ``observed``, ``obsdeep`` and ``obsdeepna`` are those lives with the exact gradient
  read beside each lesson's contrast and never applied: each projection's cosine and scale, the
  regions' movement between the nudged phases and the phases' sweeps. With the compartment the
  gradient is the normalized fixed point's, the gain's dependence on the state included.
- ``G1n``, ``G1a`` (per moment): each lesson's contrast is replaced by the exact implicit gradient
  at that moment's free fixed point, everything else as the recipe: the held read, the trace, the
  surprise rule and the address map's least-mean-squares lesson, which still reads the nudged
  phases. ``G1n`` takes the brain's normalized step with its momentum; ``G1a`` takes Adam at the
  online control's rate, the gradient norm clipped at 1 as the control's is. ``G1d`` is ``G1n`` on
  the ``deep`` brain, at its teaching point's step.
- ``G2a``, ``G2s``, ``G2n`` (through time): the same graph with the gradient carried across the
  moments of a sentence: through the trace and the address, into the address map and every
  position map, which no longer take their lesson. One step per batch over every word of the
  batch, as the controls take (``G2s``: over the words the brain would say wrong, the surprise
  rule's). ``G2a`` and ``G2s`` take Adam, ``G2n`` the normalized step.
- Which path carries the credit, each ``G2s`` with paths cut: ``G2t`` carries it across moments
  through the trace alone, ``G2r`` through the read's address alone, ``G2m`` not across moments
  at all (the address and position maps still take the moment's own gradient); ``G2c`` carries it
  through the trace with the address and position maps frozen at birth, and ``G2o`` has neither.
  ``G2nt``, ``G2nr``, ``G2nm``, ``G2nc`` and ``G2no`` are the same cuts of ``G2n``. A cut path's
  gradient is truncated by construction and is no gradient of the sentence's cost.
- ``Ln``, ``Lb``, ``Ld``, ``Lda`` (no gradient anywhere): the recipe, the normalized point with
  whole nudged phases (``whole`` without the form), the deep teaching point alone and the
  ``deep`` brain, each on its local contrast, with the address lesson's step in the form of the
  gradient. The lesson's own readback of the association's movement through the recall and next
  synapses scores each slot, ``g``; the address steps along ``sum_i p_i (g_i - mean g) K_i`` and the
  position maps along ``p_i (g_i - mean g)``, in place of ``K* - q`` toward the slots a sharp
  softmax names. At the temperature of the lesson's scores this is the first order of its target,
  without the pull of the address onto the keys, and as the nudge shrinks on a settled moment it
  is the descent of the cost through the read (a test reads the cosine).
- ``Fn``, ``Fn2``, ``Fd`` (no gradient anywhere): the learned-keys lane's rule on the address map,
  the lesson's own sharp target ``p*`` with the error ``sum_i (p*_i - p_i) K_i`` in place of
  ``K* - q``: one head, two heads, and one head at the deep teaching point alone. ``FdL`` is
  ``Fd`` through the library's ``recall_address_contrast``, to the bit (a test holds it). ``Ln``
  is its first order for a soft target, and ``Fn3``, ``Fn10``, ``Fn100`` and ``Lf`` lie between
  them: the rule at the lesson's temperature 0.3, 1 and 10, its step scaled so that the first
  order stays the same, and that first order on the lesson's own scores (``Ln`` holds each
  source's slope).
  ``Lf3``, ``Lf1`` and ``Fcap`` are that first order at a gain of 3 and 1 and with its target kept
  a distribution, and ``Fu3``, ``Fu10`` the soft targets unscaled: they separate the target's
  sharpness from the size of its step.
  ``FnT``, ``FnC``, ``FnR`` and ``FnX`` live with what crosses moments cut from birth, at every
  lesson and act: the working trace, the cortex the address reads, the previous read's focus, and
  the three. ``FdT``, ``FdC``, ``FdR`` and ``FdX`` are the same cuts of ``Fd``.
- The step's granularity, at the normalized step over the wrong rows alone: ``G2ns`` takes one
  step per batch, ``G2np`` one per moment, as a lesson does, each moment's words said wrong at the
  parameters of that moment; ``G2nms`` and ``G2nmp`` are the same without credit across moments,
  and ``G2nop`` is ``G2nmp`` with the maps frozen at birth. ``G1n`` is ``G2nmp`` with the maps on
  the address lesson in place of their gradient, up to the free phase's start.

G2 runs its forward in float64 torch, each settle solved to rounding and then differentiated
implicitly (an adjoint solve at the fixed point), and its parameters are written back into a
``Brain`` to be read: every panel is the library's own settle, act and scorer. A test holds the
two forwards together. Both rungs' gradients are checked against central finite differences of
the fixed points (``check``) before a row is read.

``--settled`` audits a life: sampled acts and lessons' free phases are settled on from where the
composed tolerance stopped them to a residual of 1e-10, and the distance and the answer's change
are read through the life, with the panels at both tolerances.

The source neurons' biases stay at zero: the local contrast never moves them (their activity
does not change between phases), so the same parameters learn. This file changes nothing in the
library. It is an independent reference and never part of a brain's answer path.

    python benchmarks/grammar/gradients.py --check --founders 0 --out DIR/check.json
    python benchmarks/grammar/gradients.py --arms recipe G1n G1a --founders 0 1 2 \\
        --out DIR/gradients.jsonl --workers 3
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import grammar as gr
import numpy as np

from cadence.brain import _APICAL_TINY

DEEP = {"apical": 0.25, "teaching": "deep"}
WHOLE = {"nudged_fixed": True, "nudged_steps": 32}  # the normalized point, whole nudged phases
ARMS: dict[str, dict[str, Any]] = {
    "recipe": {},
    "deep": {"brain": DEEP},
    "deepna": {"brain": {"teaching": "deep"}},
    "observed": {"rung": 1, "step": "observe"},
    "obsdeep": {"rung": 1, "step": "observe", "brain": DEEP},
    "obsdeepna": {"rung": 1, "step": "observe", "brain": {"teaching": "deep"}},
    "G1n": {"rung": 1, "step": "normalized"},
    "G1a": {"rung": 1, "step": "adam"},
    "G1d": {"rung": 1, "step": "normalized", "brain": DEEP},
    "G2a": {"rung": 2, "step": "adam"},
    "G2s": {"rung": 2, "step": "adam", "surprise": True},
    "G2n": {"rung": 2, "step": "normalized"},
    "Fn": {"named_credit": 0.1},
    "Fn2": {"named_credit": 0.1, "brain": {"recall_heads": 2}},
    "Fd": {"named_credit": 0.1, "brain": {"teaching": "deep"}},
    "Fd2": {"named_credit": 0.1, "brain": {"teaching": "deep", "recall_heads": 2}},
    "FdL": {"brain": {"teaching": "deep", "recall_address_contrast": True}},
    "Fn3": {"named_credit": 0.3},
    "Fn10": {"named_credit": 1.0},
    "Fn100": {"named_credit": 10.0},
    "Lf": {"read_credit": True, "no_slope": True},
    "Lf3": {"read_credit": True, "no_slope": True, "gain": 3.0},
    "Lf1": {"read_credit": True, "no_slope": True, "gain": 1.0},
    "Fu3": {"named_credit": 0.3, "unscaled": True},
    "Fu10": {"named_credit": 1.0, "unscaled": True},
    "Fcap": {"read_credit": True, "no_slope": True, "capped": True},
    "FnT": {"named_credit": 0.1, "cut": ["trace"]},
    "FnC": {"named_credit": 0.1, "cut": ["cortex"]},
    "FnR": {"named_credit": 0.1, "cut": ["focus"]},
    "FnX": {"named_credit": 0.1, "cut": ["trace", "cortex", "focus"]},
    "FdT": {"named_credit": 0.1, "brain": {"teaching": "deep"}, "cut": ["trace"]},
    "FdC": {"named_credit": 0.1, "brain": {"teaching": "deep"}, "cut": ["cortex"]},
    "FdR": {"named_credit": 0.1, "brain": {"teaching": "deep"}, "cut": ["focus"]},
    "FdX": {"named_credit": 0.1, "brain": {"teaching": "deep"},
            "cut": ["trace", "cortex", "focus"]},
    "whole": {"config": WHOLE},
    "Lb": {"read_credit": True, "config": WHOLE},
    "Ln": {"read_credit": True},
    "Ld": {"read_credit": True, "brain": {"teaching": "deep"}},
    "Lda": {"read_credit": True, "brain": DEEP},
    "G2t": {"rung": 2, "step": "adam", "surprise": True, "carry_address": False},
    "G2r": {"rung": 2, "step": "adam", "surprise": True, "carry_trace": False},
    "G2m": {"rung": 2, "step": "adam", "surprise": True, "carry_address": False,
            "carry_trace": False},
    "G2c": {"rung": 2, "step": "adam", "surprise": True, "frozen_maps": True,
            "carry_address": False},
    "G2o": {"rung": 2, "step": "adam", "surprise": True, "frozen_maps": True,
            "carry_address": False, "carry_trace": False},
    "G2nt": {"rung": 2, "step": "normalized", "carry_address": False},
    "G2nr": {"rung": 2, "step": "normalized", "carry_trace": False},
    "G2nm": {"rung": 2, "step": "normalized", "carry_address": False, "carry_trace": False},
    "G2nc": {"rung": 2, "step": "normalized", "frozen_maps": True, "carry_address": False},
    "G2no": {"rung": 2, "step": "normalized", "frozen_maps": True, "carry_address": False,
             "carry_trace": False},
    "G2ns": {"rung": 2, "step": "normalized", "surprise": True},
    "G2np": {"rung": 2, "step": "normalized", "per_moment": True},
    "G2nms": {"rung": 2, "step": "normalized", "surprise": True, "carry_address": False,
              "carry_trace": False},
    "G2nmp": {"rung": 2, "step": "normalized", "per_moment": True, "carry_address": False,
              "carry_trace": False},
    "G2nop": {"rung": 2, "step": "normalized", "per_moment": True, "frozen_maps": True,
              "carry_address": False, "carry_trace": False},
}
SHARE = {"learning_surprise_share": True}
for _width in (64, 80, 81):  # the composition without the trace's region, not silent but absent
    ARMS[f"FdA{_width}"] = {**ARMS["Fd"], "module": _width, "absent": True}
    ARMS[f"FdA{_width}f"] = {**ARMS["Fd"], "module": _width, "absent": True, "fan_in": True}
for _width in (128, 256):  # wider association, on the composed steps, fan-in steps, and the share
    for _name in ("Fd", "FdT"):
        _arm = ARMS[_name]
        ARMS[f"{_name}{_width}"] = {**_arm, "module": _width}
        ARMS[f"{_name}{_width}f"] = {**_arm, "module": _width, "fan_in": True}
        ARMS[f"{_name}{_width}s"] = {**_arm, "module": _width, "fan_in": True,
                                     "brain": {**_arm["brain"], **SHARE}}
RATES = {"online": 0.003, "offline": 0.001}  # the controls'
CLIP, PATIENCE, EPOCHS = 1.0, 3, 30  # the controls'
ADAM = (0.9, 0.999, 1e-8)  # torch's defaults, the controls'
TOLERANCE = 1e-11  # the polish of a fixed point
REPLICA = 1e-13  # a replica's relative change of its newborn's efficacies: last-bit arithmetic
OBSERVE_EVERY = 16  # the lessons between two readings of the contrast against the gradient


# ------------------------------------------------------------------------- one moment (G1)


def _partition(brain: Any) -> tuple[np.ndarray, np.ndarray]:
    """The neurons with inbound synapses, which settle, and the sources."""
    graph = brain.connectome
    free = np.unique(graph.post)
    return free, np.setdiff1d(np.arange(graph.n), free)


class Settling:
    """The settling neurons' equations at a brain's parameters, a row per stream. ``v = b + g u``:
    ``b`` the basal input (the sources, the bias and the basal synapses), ``u`` the apical input
    and ``g`` the normalization's gain over the pool (no compartment: ``v = b``)."""

    def __init__(self, brain: Any, state: Any) -> None:
        graph = brain.connectome
        self.brain, self.model = brain, brain.neuron_model
        self.free, sources = _partition(brain)
        self.sources = sources
        weights = np.zeros((graph.n, graph.n))
        weights[graph.pre, graph.post] = brain.weights
        self.apical = brain.apical
        if self.apical is None:
            self.w_b = weights[np.ix_(self.free, self.free)]
            self.w_u = None
        else:
            raw = np.zeros((graph.n, graph.n))
            chosen = np.asarray(self.apical.synapses)
            raw[graph.pre[chosen], graph.post[chosen]] = np.asarray(brain.weights)[chosen]
            if raw[sources].any():
                raise ValueError("a source reaches the compartment")
            self.w_b = (weights - raw)[np.ix_(self.free, self.free)]
            self.w_u = raw[np.ix_(self.free, self.free)]
            where = {int(n): i for i, n in enumerate(self.free)}
            self.pools = [np.array([where[int(n)] for n in np.arange(graph.n)[members]])
                          for members in self.apical.members]
        self.field = (np.atleast_2d(state.activation)[:, sources]
                      @ weights[np.ix_(sources, self.free)] + brain.bias[self.free])

    def gain(self, b: np.ndarray, u: np.ndarray, pool: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A pool's gain per row and its derivatives by each member's basal and apical input."""
        fraction, floor2 = self.apical.fraction, self.apical.floor ** 2
        b_m, u_m = b[:, pool], u[:, pool]
        rb2 = (b_m * b_m).mean(axis=1)
        total = (u_m * u_m).mean(axis=1) + floor2 * rb2 + _APICAL_TINY
        g = fraction * np.sqrt(rb2) / np.sqrt(total)
        n = len(pool)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_basal = np.where(rb2 > 0, 1.0 / rb2 - floor2 / total, 0.0)
        return g, (g * per_basal)[:, None] * b_m / n, -(g / total)[:, None] * u_m / n

    def residual(self, v: np.ndarray) -> np.ndarray:
        s = self.model.activation(v)
        b = s @ self.w_b + self.field
        if self.w_u is None:
            return np.asarray(v - b)
        u = s @ self.w_u
        out = b.copy()
        for pool in self.pools:
            g, _, _ = self.gain(b, u, pool)
            out[:, pool] += g[:, None] * u[:, pool]
        return np.asarray(v - out)

    def linear(self, v: np.ndarray) -> dict[str, Any]:
        """The Jacobian ``J`` of the residual at ``v`` and what the adjoint needs of it."""
        s, d = self.model.activation(v), self.model.slope_at(v)
        rows, size = v.shape
        identity = np.eye(size)[None]
        if self.w_u is None:
            return {"s": s, "d": d, "J": identity - self.w_b.T[None] * d[:, None, :]}
        b, u = s @ self.w_b + self.field, s @ self.w_u
        gains = np.zeros((rows, size))
        by_basal, by_apical = np.zeros((rows, size)), np.zeros((rows, size))
        carried = np.zeros((rows, size))
        for pool in self.pools:
            g, g_b, g_u = self.gain(b, u, pool)
            gains[:, pool] = g[:, None]
            by_basal[:, pool], by_apical[:, pool] = g_b, g_u
            carried[:, pool] = u[:, pool]
        response = (self.w_b.T[None] + gains[:, :, None] * self.w_u.T[None]
                    + carried[:, :, None] * (by_basal[:, None, :] @ self.w_b.T[None]
                                             + by_apical[:, None, :] @ self.w_u.T[None]))
        return {"s": s, "d": d, "J": identity - response * d[:, None, :], "gains": gains,
                "by_basal": by_basal, "by_apical": by_apical, "carried": carried}


def polish(system: Settling, state: Any) -> tuple[np.ndarray, dict[str, float]]:
    """The free neurons' potentials at the fixed point nearest a settled ``state``, by Newton's
    method on its own Jacobian, each row's step halved until its residual falls."""
    v = np.array(np.atleast_2d(state.v)[:, system.free], dtype=float)
    g = system.residual(v)
    error = np.abs(g).max(axis=1)
    start = float(error.max())
    steps = 0
    while steps < 40 and error.max() > TOLERANCE:
        move = np.linalg.solve(system.linear(v)["J"], -g[..., None])[..., 0]
        scale = np.ones(len(v))
        for _ in range(8):
            candidate = v + scale[:, None] * move
            g_new = system.residual(candidate)
            error_new = np.abs(g_new).max(axis=1)
            worse = (error_new >= error) & (error > TOLERANCE)
            if not worse.any():
                break
            scale = np.where(worse, scale / 2, scale)
        keep = error_new < error
        v = np.where(keep[:, None], candidate, v)
        g = np.where(keep[:, None], g_new, g)
        error = np.where(keep, error_new, error)
        steps += 1
        if not keep.any():
            break
    # across the activation's kink a row's Newton direction can have no descending step at all;
    # the settle's own contraction still holds there, so it finishes what Newton left
    dt = float(system.model.dt)
    sweeps = 0
    while sweeps < 8192 and error.max() > TOLERANCE:
        v = v - dt * g
        g = system.residual(v)
        error = np.abs(g).max(axis=1)
        sweeps += 1
    return v, {"polish_steps": float(steps), "polish_sweeps": float(sweeps),
               "residual_before": start, "residual": float(error.max()),
               "unpolished": float((error > 1e-8).sum())}


def exact_gradient(learner: Any, state: Any, labels: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """The gradient of ``T`` times the cross-entropy at the free fixed point, the mean over the
    rows: per synapse with respect to its efficacy (a reciprocal pair's one efficacy, both of its
    synapses holding the sum), per neuron with respect to its bias (zero at a source)."""
    brain = learner.brain
    graph = brain.connectome
    system = Settling(brain, state)
    free = system.free
    v, report = polish(system, state)
    parts = system.linear(v)
    s = np.array(np.atleast_2d(state.activation), dtype=float)
    s[:, free] = parts["s"]
    z = s[:, learner.output_index] / learner.config.temperature
    z -= z.max(axis=1, keepdims=True)
    probability = np.exp(z)
    probability /= probability.sum(axis=1, keepdims=True)
    probability[np.arange(len(labels)), labels] -= 1.0
    cost = np.zeros_like(s)
    cost[:, learner.output_index] = probability
    adjoint = np.linalg.solve(np.swapaxes(parts["J"], 1, 2),
                              (parts["d"] * cost[:, free])[..., None])[..., 0]
    by_basal = np.zeros_like(s)  # what the cost gains per unit of a neuron's basal input
    if system.w_u is None:
        by_basal[:, free] = adjoint
        grad = (s[:, graph.pre] * by_basal[:, graph.post]).mean(axis=0)
    else:
        total = (adjoint * parts["carried"]).sum(axis=1, keepdims=True)
        by_apical = np.zeros_like(s)
        by_basal[:, free] = adjoint + total * parts["by_basal"]
        by_apical[:, free] = parts["gains"] * adjoint + total * parts["by_apical"]
        inputs = np.where(np.asarray(system.apical.synapses)[None, :],
                          by_apical[:, graph.post], by_basal[:, graph.post])
        grad = (s[:, graph.pre] * inputs).mean(axis=0)
    grad = brain._gain_pre * grad
    paired = learner.reverse >= 0
    grad[paired] += grad[learner.reverse[paired]].copy()
    return grad, by_basal.mean(axis=0), report


def summarize_readings(readings: list[dict[str, float]]) -> dict[str, Any]:
    """The contrast against the gradient over a life: every lesson read, its first and its last
    quarter."""
    quarter = max(1, len(readings) // 4)
    keys = readings[0].keys()
    return {"lessons_read": len(readings),
            **{part: {key: round(float(np.mean([r[key] for r in kept])), 5) for key in keys}
               for part, kept in (("all", readings), ("first", readings[:quarter]),
                                  ("last", readings[-quarter:]))}}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 0 else 0.0


def install_exact(brain: Any, step: str, rate: float = RATES["online"]) -> None:
    """The brain's learner, its contrast replaced by the exact gradient (``normalized``), its
    whole step (``adam``), or with the gradient only read beside the contrast (``observe``)."""
    base = type(brain.learner)

    class Exact(base):  # type: ignore[misc, valid-type]
        def step(self, drive: Any, labels: Any, warm: Any = None, weight: Any = None, *,
                 answer: Any = None) -> Any:
            labels = self._labels(labels)
            self.exact_labels = (labels[self._differs(np.asarray(answer), labels)]
                                 if self.config.surprise and answer is not None else labels)
            return super().step(drive, labels, warm, weight, answer=answer)

        def _note(self, report: dict[str, float]) -> None:
            note = self.exact_notes
            note["gradients"] += 1
            note["polish_steps"] += report["polish_steps"]
            note["polish_sweeps"] += report["polish_sweeps"]
            note["unpolished_rows"] += report["unpolished"]
            note["worst_residual"] = max(note["worst_residual"], report["residual"])

        def contrast(self, free: Any, nudged: Any, opposite: Any = None) -> Any:
            local = super().contrast(free, nudged, opposite)
            grad, grad_bias, report = exact_gradient(self, free, self.exact_labels)
            self._note(report)
            if step == "observe":
                note = self.exact_notes
                if note["gradients"] % OBSERVE_EVERY == 1:
                    note["readings"].append(self._against(local, grad, grad_bias, free, nudged,
                                                          opposite))
                return local
            bias = -grad_bias
            bias[self.exact_sources] = 0.0
            return -grad / self.brain._gain_pre, bias

        def _against(self, local: Any, grad: np.ndarray, grad_bias: np.ndarray, free: Any,
                     nudged: Any, opposite: Any) -> dict[str, float]:
            """The lesson's contrast against the descent per projection, and how far the regions
            moved between its phases."""
            descent = -grad / self.brain._gain_pre
            out: dict[str, float] = {"rows": float(len(self.exact_labels)),
                                     "all": cosine(local[0], descent)}
            for name, index in self.exact_families.items():
                out[name] = cosine(local[0][index], descent[index])
                out["scale_" + name] = float(
                    np.linalg.norm(local[0][index]) / (np.linalg.norm(descent[index]) or 1.0))
            minus = free if opposite is None else opposite
            moved = np.abs(np.atleast_2d(nudged.activation) - np.atleast_2d(minus.activation))
            for name, region in self.exact_regions.items():
                out["bias_" + name] = cosine(local[1][region], -grad_bias[region])
                out["moved_" + name] = float(moved[:, region].mean())
            out["nudged_steps"] = float(nudged.steps)
            return out

        def update(self, free: Any, nudged: Any, opposite: Any = None, *,
                   share: float = 1.0) -> dict[str, float]:
            if step != "adam":
                return dict(super().update(free, nudged, opposite, share=share))
            grad, grad_bias, report = exact_gradient(self, free, self.exact_labels)
            self._note(report)
            grad_bias[self.exact_sources] = 0.0
            scale = 1.0 / self.config.temperature  # the gradient of the cross-entropy itself
            grad, grad_bias = grad * scale, grad_bias * scale
            independent = (self.reverse < 0) | (np.arange(len(grad)) < self.reverse)
            norm = math.sqrt(float(grad[independent] @ grad[independent])
                             + float(grad_bias @ grad_bias))
            if norm > CLIP:
                grad, grad_bias = grad * CLIP / norm, grad_bias * CLIP / norm
            beta1, beta2, eps = ADAM
            adam = self.exact_adam
            adam["t"] += 1
            correction1 = 1.0 - beta1 ** adam["t"]
            correction2 = 1.0 - beta2 ** adam["t"]
            moves = []
            for name, g in (("e", grad), ("b", grad_bias)):
                adam["m" + name] = beta1 * adam["m" + name] + (1 - beta1) * g
                adam["s" + name] = beta2 * adam["s" + name] + (1 - beta2) * g * g
                moves.append(-rate * (adam["m" + name] / correction1)
                             / (np.sqrt(adam["s" + name] / correction2) + eps))
            out = self.apply(moves[0], moves[1])
            self.contrast_updates += 1
            return dict(out)

    learner = brain.learner
    learner.__class__ = Exact
    free, sources = _partition(learner.brain)
    learner.exact_sources = sources
    learner.exact_notes = {"gradients": 0, "polish_steps": 0.0, "polish_sweeps": 0.0,
                           "unpolished_rows": 0.0, "worst_residual": 0.0, "readings": []}
    graph = learner.brain.connectome
    regions = {name: np.asarray(ids) for name, ids in graph.populations.items()}
    learner.exact_regions = {name: regions[name] for name in ("association", "motor")}
    learner.exact_families = {
        name: np.flatnonzero(np.isin(graph.pre, regions[a]) & np.isin(graph.post, regions[b]))
        for name, a, b in (("sensory", "sensory", "association"),
                           ("prefrontal", "prefrontal", "association"),
                           ("recall", "recall", "association"), ("next", "next", "association"),
                           ("motor", "association", "motor"))}
    size, n = learner.brain.connectome.synapses, learner.brain.connectome.n
    learner.exact_adam = {"t": 0, "me": np.zeros(size), "se": np.zeros(size),
                          "mb": np.zeros(n), "sb": np.zeros(n)}


# ------------------------------------------------------------- the read's credit, locally


def install_read_credit(brain: Any, no_slope: bool = False, gain_of: float | None = None,
                        capped: bool = False) -> None:
    """The address lesson in the gradient's form, on the lesson's own readback (``Ln``): one head,
    every other step of the lesson as it is. Reads the local contrast's movement only. With
    ``no_slope`` the scores are the lesson's own, each source's slope left out; ``gain_of`` is
    the gain on the centred scores in place of the lesson's ``1 / T``; ``capped`` keeps the target a
    distribution, each slot's move within what it can move toward zero or one, as a sharp target's
    is by construction."""
    recall = brain.recall
    if recall.heads != 1:
        raise ValueError("the read's credit is written for one head")
    model_slope = brain.learner.brain.neuron_model.slope_at

    def slope(drive: np.ndarray) -> Any:
        return 1.0 if no_slope else model_slope(drive)

    base = type(recall)

    class Credited(base):  # type: ignore[misc, valid-type]
        def _row_steps(self, moment: Any, rows: Any, d_value: Any, d_next: Any, head: int = 0,
                       read: Any = None) -> Any:
            if len(rows) == len(self.keys):
                values, nexts, keys, held = self.values, self.nexts, self.keys, self.held
            else:
                values, nexts, keys = self.values[rows], self.nexts[rows], self.keys[rows]
                held = self.held[rows]
            masked = not held.all()
            x, c, q, key = (a[rows] for a in (moment.inputs, moment.cortex, moment.address,
                                              moment.key))
            streams = np.asarray(rows)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                attention = self._attention(q, keys, held if masked else None,
                                            self.position_term(head, x, c, streams))
                attention -= attention.max(axis=1, keepdims=True)
                attention -= np.log(np.exp(attention).sum(axis=1, keepdims=True))
                p = np.exp(attention)
                # each source's own slope at the drive it was held at: what it passes on
                held_value = self.amplitude * np.einsum("bm,bmc->bc", p, values)
                held_next = self.amplitude * np.einsum("bm,bmc->bc", p, nexts)
                by_value = d_value * slope(held_value)
                by_next = d_next * slope(held_next)
                score = (np.einsum("bc,bmc->bm", by_value, values)
                         + np.einsum("bc,bmc->bm", by_next, nexts))
                norm = np.sqrt((by_value * by_value).sum(axis=1) + (by_next * by_next).sum(axis=1))
                moving = norm > 1e-12
                if masked:
                    moving &= held.any(axis=1)
                if not moving.any():
                    return None
                g = score[moving] / norm[moving, None]
                p, attention = p[moving], attention[moving]
                centered = p * (g - (p * g).sum(axis=1, keepdims=True))
                gain = 1.0 / self.address_temperature if gain_of is None else gain_of
                if capped:  # a move that would leave a slot below 0 or above 1 is scaled back
                    move = gain * centered
                    room = np.where(move > 0, (1.0 - p) / np.maximum(move, 1e-300),
                                    p / np.maximum(-move, 1e-300))
                    scale = np.minimum(room.min(axis=1, keepdims=True), 1.0)
                    centered = centered * scale
                target = p + gain * centered
                x, c, key = x[moving], c[moving], key[moving]
                error = gain * np.einsum("bm,bmk->bk", centered, keys[moving])
                energy = np.einsum("bi,bi->b", x, x) + np.einsum("bi,bi->b", c, c) + 1.0
                step = error * np.minimum(1.0, 1.0 / (self.address * energy))[:, None]
            return moving, x, c, key, error, step, target, attention

    recall.__class__ = Credited


def install_named_credit(brain: Any, temperature: float = 0.1, unscaled: bool = False) -> None:
    """The learned-keys lane's rule on the address map: the lesson's own target ``p*``, softmax of
    the slots' scores at ``temperature`` over the prior attention, and the error ``sum_i (p*_i -
    p_i) K_i`` in place of ``K* - q``; the position maps' target is ``p*``. Above the lesson's
    temperature 0.1 the step is scaled by the ratio, so that the first order in the scores stays the
    one of 0.1: from the rule at 0.1 toward its first-order form (``Ln``); ``unscaled`` leaves the
    step as it falls, a soft target at the rule's own bound."""
    recall = brain.recall
    base = type(recall)
    ratio = 1.0 if unscaled else float(temperature) / float(base.address_temperature)

    class Named(base):  # type: ignore[misc, valid-type]
        address_temperature = float(temperature)

        def _row_steps(self, moment: Any, rows: Any, d_value: Any, d_next: Any, head: int = 0,
                       read: Any = None) -> Any:
            found = super()._row_steps(moment, rows, d_value, d_next, head, read)
            if found is None:
                return None
            moving, x, c, key, _, _, named, attention = found
            everyone = len(rows) == len(self.keys)
            keys = (self.keys if everyone else self.keys[rows])[moving]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                prior = np.where(np.isfinite(attention), np.exp(attention), 0.0)
                centered = ratio * (named - prior)
                error = np.einsum("bm,bmk->bk", centered, keys)
                energy = np.einsum("bi,bi->b", x, x) + np.einsum("bi,bi->b", c, c) + 1.0
                step = error * np.minimum(1.0, 1.0 / (self.address * energy))[:, None]
            return moving, x, c, key, error, step, prior + centered, attention

    recall.__class__ = Named


def install_cut(brain: Any, cut: list[str]) -> None:
    """A brain that lives with what crosses moments cut, before every lesson and act: ``trace`` the
    working trace, ``cortex`` the cortex of the moment before as the address reads it, ``focus`` the
    previous read's focus. A copy for a reading keeps the cut."""
    base = type(brain)
    parts = tuple(cut)

    def clear(self: Any) -> None:
        if "trace" in parts:
            self.working_memory.trace = np.zeros_like(self.working_memory.trace)
        if "cortex" in parts:
            self.recall.previous = np.zeros_like(self.recall.previous)
        if "focus" in parts:
            self.recall.focus = np.full_like(self.recall.focus, -1)

    class Cut(base):  # type: ignore[misc, valid-type]
        def act(self, observations: Any, **options: Any) -> Any:
            clear(self)
            return super().act(observations, **options)

        def lesson(self, observations: Any, labels: Any) -> Any:
            clear(self)
            return super().lesson(observations, labels)

    brain.__class__ = Cut


def fan_in_rates(wide: Any, narrow: Any) -> np.ndarray:
    """Each synapse's step scaled to the largest fan-in of its neuron's population in ``narrow``,
    the same composition at a smaller width; capped at 1 (docs/tune.md, wide brains)."""
    c, n = wide.connectome, narrow.connectome
    fan = np.bincount(c.post, minlength=c.n).astype(float)
    fan_narrow = np.bincount(n.post, minlength=n.n).astype(float)
    reference = fan.copy()
    for name, neurons in c.populations.items():
        index = np.asarray(neurons, dtype=np.int64)
        if name in n.populations and len(index):
            top = fan_narrow[np.asarray(n.populations[name], dtype=np.int64)].max(initial=0.0)
            if top > 0:
                reference[index] = np.minimum(reference[index], top)
    rates: np.ndarray = np.minimum(1.0, reference[c.post] / np.maximum(fan[c.post], 1.0))
    return rates


def arm_width(settings: dict[str, Any], spec: dict[str, Any]) -> int:
    """A life's association width: the specification's, else the arm's, else the founder's 64."""
    return int(spec.get("module", settings.get("module", 64)))


@contextlib.contextmanager
def without_region(name: str) -> Iterator[None]:
    """``Brain.compose`` within builds no ``name`` region and none of its projections; the library
    has no such option. The draws after the region's shift, so a brain built here is a new newborn,
    not the composed one less some synapses."""
    from cadence import generic

    base = generic._layout

    def layout(*args: Any, **options: Any) -> Any:
        genome, found = base(*args, **options)
        regions = tuple(r for r in genome.regions if r.name != name)
        projections = tuple(p for p in genome.projections if name not in (p.pre, p.post))
        return dataclasses.replace(genome, regions=regions, projections=projections), found

    generic._layout = layout  # type: ignore[attr-defined]
    try:
        yield
    finally:
        generic._layout = base  # type: ignore[attr-defined]


def newborn(settings: dict[str, Any], init_seed: int, spec: dict[str, Any]) -> Any:
    """An arm's newborn at its width, on the steps it names."""
    brain_settings = settings.get("brain")

    def composed(module: int) -> Any:
        if settings.get("absent"):
            with without_region("prefrontal"):
                return gr.make_brain("recipe", init_seed, settings=brain_settings, module=module)
        return gr.make_brain("recipe", init_seed, settings=brain_settings, module=module)

    brain = composed(arm_width(settings, spec))
    if settings.get("fan_in"):
        brain.learner.synapse_rate = fan_in_rates(brain, composed(64))
    return brain


def install_arm(brain: Any, settings: dict[str, Any]) -> None:
    """A brain's lessons as an arm that keeps the local contrast has them."""
    if "config" in settings:
        brain.learner.config = dataclasses.replace(brain.learner.config, **settings["config"])
    if settings.get("read_credit"):
        install_read_credit(brain, no_slope=bool(settings.get("no_slope")),
                            gain_of=settings.get("gain"), capped=bool(settings.get("capped")))
    if settings.get("named_credit"):
        install_named_credit(brain, float(settings["named_credit"]),
                             unscaled=bool(settings.get("unscaled")))
    if settings.get("cut"):
        install_cut(brain, list(settings["cut"]))


# --------------------------------------------------------------------- how settled it is


class SettledAudit:
    """How far the composed tolerance leaves a state from its fixed point. Read-only: each
    sampled state is settled on from where it stopped, on a copy, to a residual of 1e-10, and
    the rows' distances and answers are kept. The life itself is unchanged (a test holds it to
    the bit)."""

    def __init__(self, brain: Any, batches: int, bins: int = 8) -> None:
        graph = brain.connectome
        self.motor = np.asarray(graph.populations["motor"])
        self.association = np.asarray(graph.populations["association"])
        self.active = False
        self.bin = 0
        self.size = max(1, batches // bins)
        self.bins = bins
        self.rows: dict[str, dict[str, list[list[float]]]] = {}
        self.unqualified: dict[str, int] = {}

    def at(self, batch: int, sample_every: int) -> None:
        self.active = batch % sample_every == 0
        self.bin = min(batch // self.size, self.bins - 1)

    def read(self, kind: str, graph_brain: Any, drive: np.ndarray, state: Any) -> None:
        from cadence.brain import _copy_state

        phase = graph_brain.equilibrate(
            drive, state=_copy_state(state), budget=40000, chunk=64, tolerance=1e-10, damping=3)
        stopped = np.atleast_2d(state.activation)
        truth = np.atleast_2d(phase.state.activation)
        gap = np.abs(truth - stopped)
        residual = graph_brain.residual(drive, state)
        qualified = np.asarray(phase.qualified, dtype=bool)
        self.unqualified[kind] = self.unqualified.get(kind, 0) + int((~qualified).sum())
        keep = self.rows.setdefault(kind, {})
        key = str(self.bin)
        row = keep.setdefault(key, [])
        first, second = stopped[:, self.motor].argmax(axis=1), truth[:, self.motor].argmax(axis=1)
        for i in range(len(stopped)):
            if qualified[i]:
                row.append([float(gap[i, self.motor].max()), float(gap[i, self.association].max()),
                            float(gap[i, self.association].mean()), float(first[i] != second[i]),
                            float(residual[i])])

    def summary(self) -> dict[str, Any]:
        def describe(rows: list[list[float]]) -> dict[str, float]:
            a = np.array(rows)
            motor, association = a[:, 0], a[:, 1]
            quantiles = (50, 90, 99, 99.9)
            return {
                "rows": float(len(a)),
                "motor_mean": float(motor.mean()), "association_mean": float(association.mean()),
                **{f"motor_p{q}": float(np.percentile(motor, q)) for q in quantiles},
                **{f"association_p{q}": float(np.percentile(association, q)) for q in quantiles},
                "motor_max": float(motor.max()), "association_max": float(association.max()),
                **{f"motor_over_{t}": float((motor > t).mean()) for t in (0.01, 0.05, 0.1, 0.2)},
                "answer_changed": float(a[:, 3].mean()),
                "residual_mean": float(a[:, 4].mean()), "residual_max": float(a[:, 4].max()),
            }

        out: dict[str, Any] = {"unqualified": self.unqualified, "bins": self.bins}
        for kind, by_bin in self.rows.items():
            everything = [row for rows in by_bin.values() for row in rows]
            out[kind] = {"life": describe(everything),
                         "by_bin": {key: describe(rows) for key, rows in sorted(
                             by_bin.items(), key=lambda item: int(item[0]))}}
        return out


def install_audit(brain: Any, audit: SettledAudit) -> None:
    """The audit on a brain's acts and its lessons' free phases, until ``remove_audit``."""
    base = type(brain.learner)

    class Audited(base):  # type: ignore[misc, valid-type]
        def free(self, drive: Any, warm: Any = None) -> Any:
            state = super().free(drive, warm)
            if audit.active:
                audit.read("lesson_free", self.brain, np.asarray(drive), state)
            return state

    brain.learner.__class__ = Audited
    original = brain.act

    def act(observations: Any, **options: Any) -> Any:
        out = original(observations, **options)
        if audit.active:
            agent = brain.basal_ganglia
            audit.read("act", brain.learner.brain, np.asarray(agent._drive), agent.state)
        return out

    brain.act = act


def remove_audit(brain: Any) -> None:
    """The brain's acts and lessons as they were: a fork for a reading holds no audit."""
    del brain.__dict__["act"]
    brain.learner.__class__ = type(brain.learner).__mro__[1]


def run_settled(spec: dict[str, Any]) -> dict[str, Any]:
    """A life with the audit on every ``sample_every``-th batch, and its panels read at the
    composed tolerance and at the fixed point."""
    import warnings

    warnings.simplefilter("ignore")
    arm, founder = spec["arm"], int(spec["founder"])
    settings = ARMS[arm]
    n = int(spec.get("batches", 4000))
    every = int(spec.get("sample_every", 40))
    init_seed = int(spec.get("init_seed", founder))
    started = time.monotonic()
    brain = newborn(settings, init_seed, spec)
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    audit = SettledAudit(brain, n)
    install_audit(brain, audit)
    work = gr.new_work()
    for b in range(n):
        audit.at(b, every)
        gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True, work=work)
    remove_audit(brain)
    work["seconds"] = round(time.monotonic() - started, 1)
    names = tuple(spec.get("panels") or gr.G1_PANELS)
    composed = _read(brain, tests, names)
    tight = _read(tightened(brain), tests, names)
    words_changed: dict[str, float] = {}
    for name in names:
        fork, fork_tight = copy.deepcopy(brain), tightened(brain)
        same = total = 0
        for batch in tests[name]:
            for a, b_ in zip(gr.speak(fork, batch, teach=False),
                             gr.speak(fork_tight, batch, teach=False), strict=True):
                same += int(a == b_)
                total += 1
        words_changed[name] = 1.0 - same / max(total, 1)
    parameters = int(brain.parameters())
    return {
        "row": arm, "settings": settings, "mode": "online", "founder": founder,
        "init_seed": init_seed, "replica": 0, "width": arm_width(settings, spec),
        "parameters": parameters, "parameters_total": parameters, "phases": [["g1", n]],
        "sample_every": every, "audit": audit.summary(), "sentences_changed": words_changed,
        "notes": {"panels_tight": {name: {key: value.get(key) for key in (
            "exact", "both", "agree", "verb", "next_word")} for name, value in tight.items()}},
        "refused_reads": 0, "train_refused": work["refused"],
        "readings": [{"phase": 0, "grammar": "g1", "batches": n, "epochs": 1, "loss_last": None,
                      "work": work, "panels": composed}],
        "seconds": round(time.monotonic() - started, 1),
    }


# ------------------------------------------------------------------------ through time (G2)


class Unrolled:
    """The recipe brain's graph as one differentiable function of its learned parameters over
    the moments of a batch: float64 torch, an implicit layer at each settle."""

    MAPS = ("qx", "qc", "qb", "posx", "posc", "posb", "ancx", "ancc", "ancb", "relx", "relc",
            "relb", "slox", "sloc", "slob")

    def __init__(self, brain: Any, *, frozen_maps: bool = False, carry_trace: bool = True,
                 carry_address: bool = True) -> None:
        import torch

        self.torch = torch
        self.brain = brain
        self.carry_trace, self.carry_address = carry_trace, carry_address
        graph, recall = brain.connectome, brain.recall
        if recall.heads != 1 or recall.window != 1 or recall.action or recall.null is not None:
            raise ValueError("the unrolled graph is the recipe's: one head, window 1, no null")
        model = brain.learner.brain.neuron_model
        self.leak = float(model.leak)
        self.temperature = float(brain.learner.config.temperature)
        self.nudge_temperature = self.temperature
        self.recall_temperature = float(recall.temperature)
        self.recall_amplitude = float(recall.amplitude)
        self.trace_amplitude = float(brain.working_memory.amplitude)
        self.stimulus = float(model.stimulus_amplitude)
        self.buckets = int(recall.position_buckets)
        self.position = float(recall.position)
        self.relative = int(recall.RELATIVE)
        self.slope = float(recall.SLOPE)
        self.cap = float(brain.learner.config.scale_cap)
        pops = graph.populations
        self.pops = pops
        self.index: dict[str, np.ndarray] = {}
        efficacy = brain.learner.brain.efficacy
        tensors: dict[str, Any] = {}
        for name, a, b in (("sa", "sensory", "association"), ("pa", "prefrontal", "association"),
                           ("ra", "recall", "association"), ("na", "next", "association"),
                           ("am", "association", "motor"), ("ma", "motor", "association")):
            index = self._block(graph, np.asarray(pops[a]), np.asarray(pops[b]))
            self.index[name] = index
            if name != "ma":
                tensors[name] = efficacy[index]
        if not np.array_equal(efficacy[self.index["ma"]].T, efficacy[self.index["am"]]):
            raise ValueError("the association and motor synapses are one reciprocal efficacy")
        bias = brain.learner.brain.bias
        tensors["ba"] = bias[np.asarray(pops["association"])]
        tensors["bm"] = bias[np.asarray(pops["motor"])]
        for key, maps in (("q", recall.head_maps[0]), ("pos", recall.position_maps[0]),
                          ("anc", recall.anchor_maps[0]), ("rel", recall.relative_maps[0]),
                          ("slo", recall.slope_maps[0])):
            for part, array in zip(("x", "c", "b"), maps, strict=True):
                tensors[key + part] = array
        self.params = {name: torch.tensor(np.array(value, dtype=float), dtype=torch.float64,
                                          requires_grad=True)
                       for name, value in tensors.items()}
        self.synapse_params = ("sa", "pa", "ra", "na", "am")
        if frozen_maps:
            for name in self.MAPS:
                self.params[name].requires_grad_(False)

    @staticmethod
    def _block(graph: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The synapse index of each ``(a, b)`` pair of neurons; every pair is a synapse."""
        lookup = {(int(p), int(q)): i for i, (p, q) in enumerate(zip(graph.pre, graph.post,
                                                                     strict=True))}
        return np.array([[lookup[int(p), int(q)] for q in b] for p in a], dtype=np.int64)

    def parameters(self) -> list[Any]:
        return [p for p in self.params.values() if p.requires_grad]

    def count(self) -> int:
        return int(sum(p.numel() for p in self.params.values()))

    def write_back(self, brain: Any | None = None) -> Any:
        """Set a brain's efficacies, biases and maps to these parameters; returns the brain."""
        brain = self.brain if brain is None else brain
        learner, recall = brain.learner, brain.recall
        p = {name: value.detach().numpy().copy() for name, value in self.params.items()}
        efficacy = np.array(learner.brain.efficacy, dtype=float)
        for name in self.synapse_params:
            efficacy[self.index[name]] = p[name]
        efficacy[self.index["ma"]] = p["am"].T
        bias = np.array(learner.brain.bias, dtype=float)
        bias[np.asarray(self.pops["association"])] = p["ba"]
        bias[np.asarray(self.pops["motor"])] = p["bm"]
        learner.brain = learner.brain.with_parameters(efficacy=efficacy, bias=bias)
        recall._set_map(0, p["qx"], p["qc"], p["qb"])
        recall.position_maps = [(p["posx"], p["posc"], p["posb"])]
        recall.anchor_maps = [(p["ancx"], p["ancc"], p["ancb"])]
        recall.relative_maps = [(p["relx"], p["relc"], p["relb"])]
        recall.slope_maps = [(p["slox"], p["sloc"], p["slob"])]
        return brain

    def act(self, v: Any) -> Any:
        t = self.torch.tanh(v / 2)
        return self.torch.where(v > 0, t, self.leak * t)

    def slope_at(self, v: Any) -> Any:
        t = self.torch.tanh(v / 2)
        d = 0.5 * (1.0 - t * t)
        return self.torch.where(v > 0, d, self.leak * d)

    def _bucket(self, ages: np.ndarray) -> np.ndarray:
        exact = self.position
        out = np.where(ages < exact, ages,
                       exact + np.floor(np.log2(np.maximum(ages, 1.0) / exact)))
        return np.minimum(out, self.buckets - 1).astype(np.int64)

    def _fixed_point(self, field: Any, before: Any, a: Any, bm: Any, v_a: Any, v_m: Any
                     ) -> tuple[Any, Any, int, float, bool]:
        """The association and motor potentials at the fixed point the library's act reaches: the
        settle's synchronous sweeps from the stream's last state, the first one hearing the
        sources of the moment before, a half step where they do not converge (as the act's
        rescue), then polished by Newton's method on the motor block's Schur complement. The
        settle has several attractors in places, and the path selects among them."""
        torch = self.torch
        v_a, v_m = before + self.act(v_m) @ a.T, self.act(v_a) @ a + bm
        sweeps = 1
        for step in (1.0, 0.5):
            for _ in range(512):
                new_a = field + self.act(v_m) @ a.T
                new_m = self.act(v_a) @ a + bm
                change = max(float((new_a - v_a).abs().max()), float((new_m - v_m).abs().max()))
                v_a, v_m = v_a + step * (new_a - v_a), v_m + step * (new_m - v_m)
                sweeps += 1
                if change < 1e-3:
                    break
            if change < 1e-3:
                break
        identity = torch.eye(a.shape[1], dtype=torch.float64)

        def residuals(u_a: Any, u_m: Any) -> tuple[Any, Any, Any]:
            g_a = u_a - field - self.act(u_m) @ a.T
            g_m = u_m - self.act(u_a) @ a - bm
            return g_a, g_m, torch.maximum(g_a.abs().amax(dim=1), g_m.abs().amax(dim=1))

        g_a, g_m, error = residuals(v_a, v_m)
        for _ in range(40):
            if float(error.max()) < TOLERANCE:
                break
            d_a, d_m = self.slope_at(v_a), self.slope_at(v_m)
            # (I - A' D_a A D_m) delta_m = -g_m - A' D_a g_a
            aa = (a.T[None] * d_a[:, None, :]) @ a
            schur = identity[None] - aa * d_m[:, None, :]
            right = -g_m - (d_a * g_a) @ a
            delta_m = torch.linalg.solve(schur, right[..., None])[..., 0]
            delta_a = -g_a + (d_m * delta_m) @ a.T
            # the activation's slope jumps tenfold at zero: a full step can cycle across the
            # kink, so each row's step halves until its residual falls
            scale = torch.ones(len(v_a), 1, dtype=torch.float64)
            for _ in range(12):
                new_a, new_m = v_a + scale * delta_a, v_m + scale * delta_m
                new_g_a, new_g_m, new_error = residuals(new_a, new_m)
                worse = (new_error >= error) & (error >= TOLERANCE)
                if not worse.any():
                    break
                scale = torch.where(worse[:, None], scale / 2, scale)
            keep = (new_error < error)[:, None]
            v_a, v_m = torch.where(keep, new_a, v_a), torch.where(keep, new_m, v_m)
            g_a, g_m = torch.where(keep, new_g_a, g_a), torch.where(keep, new_g_m, g_m)
            error = torch.minimum(new_error, error)
            if not keep.any():
                break
        # across the kink a row's Newton direction can have no descending step at all; the
        # settle's own contraction still holds there, so it finishes what Newton left
        for _ in range(4096):
            if float(error.max()) < TOLERANCE:
                break
            new_a = field + self.act(v_m) @ a.T
            new_m = self.act(v_a) @ a + bm
            v_a, v_m = new_a, new_m
            sweeps += 1
            _, _, error = residuals(v_a, v_m)
        return v_a, v_m, sweeps, float(error.max()), change >= 1e-3

    def forward(self, xs: Any, labels: Any, *, surprise: bool = False, record: bool = False
                ) -> tuple[Any, dict[str, Any]]:
        """The batch's moments ``(moments, rows, inputs)``, each row's word to be said or -1
        ``(moments, rows)``. Returns the mean cross-entropy over the taught words at the nudge's
        temperature and what the forward saw."""
        torch = self.torch
        p = self.params
        moments, rows, width = xs.shape
        keys = torch.zeros(rows, moments + 1, width, dtype=torch.float64)
        keys[:, 1:] = xs.transpose(0, 1)
        nexts = torch.zeros(rows, moments + 1, width, dtype=torch.float64)
        nexts[:, :-1] = xs.transpose(0, 1)
        a = p["am"]
        # the settle's parameters without their graph, for the solve and the adjoint
        a_held = a.detach()
        c = torch.zeros(rows, p["ba"].numel(), dtype=torch.float64)
        c_trace = c
        v_a = torch.zeros_like(c)
        v_m = torch.zeros(rows, p["bm"].numel(), dtype=torch.float64)
        before = p["ba"].detach().expand(rows, -1)  # the sources at rest
        focus = np.full(rows, -1, dtype=np.int64)
        total = torch.zeros((), dtype=torch.float64)
        taught = 0
        wrong = 0
        seen: dict[str, Any] = {"read": [], "association": [], "motor": [], "sweeps": 0,
                                "residual": 0.0, "focus": [], "unsettled": 0}
        for t in range(moments):
            x = xs[t]
            slots = t + 1
            ages = np.arange(slots - 1, -1, -1, dtype=float)
            q = x + x @ p["qx"] + c @ p["qc"] + p["qb"]
            logits = torch.einsum("bk,bmk->bm", q, keys[:, :slots]) / self.recall_temperature
            term = x @ p["posx"] + c @ p["posc"] + p["posb"]
            logits = logits + term[:, torch.from_numpy(self._bucket(ages))]
            places = float(t) - 1.0 - ages
            anchored = np.where(places < 0, self.buckets, self._bucket(np.maximum(places, 0.0)))
            term = x @ p["ancx"] + c @ p["ancc"] + p["ancb"]
            logits = logits + term[:, torch.from_numpy(anchored.astype(np.int64))]
            offsets = np.clip(ages[None, :] - focus[:, None].astype(float), -self.relative,
                              self.relative) + self.relative
            relative = np.where(focus[:, None] < 0, 2 * self.relative + 1, offsets)
            term = x @ p["relx"] + c @ p["relc"] + p["relb"]
            logits = logits + torch.gather(term, 1, torch.from_numpy(relative.astype(np.int64)))
            lean = x @ p["slox"] + c @ p["sloc"] + p["slob"]
            logits = logits + lean[:, None] * torch.from_numpy(ages / self.slope)[None, :]
            read = torch.softmax(logits, dim=1)
            recall_drive = self.recall_amplitude * torch.einsum(
                "bm,bmc->bc", read, keys[:, :slots])
            next_nexts = nexts[:, :slots].clone()
            next_nexts[:, slots - 1] = 0.0  # the latest slot's successor is this moment
            next_drive = self.recall_amplitude * torch.einsum("bm,bmc->bc", read, next_nexts)
            field = (self.act(self.stimulus * x) @ p["sa"]
                     + self.act(self.trace_amplitude * c_trace) @ p["pa"]
                     + self.act(recall_drive) @ p["ra"]
                     + self.act(next_drive) @ p["na"] + p["ba"])
            with torch.no_grad():
                v_a_star, v_m_star, sweeps, error, unsettled = self._fixed_point(
                    field.detach(), before, a_held, p["bm"].detach(), v_a, v_m)
                d_a, d_m = self.slope_at(v_a_star), self.slope_at(v_m_star)
            seen["sweeps"] += sweeps
            seen["unsettled"] += int(unsettled)
            seen["residual"] = max(seen["residual"], error)
            v_a_out = field + self.act(v_m_star) @ a.T
            v_m_out = self.act(v_a_star) @ a + p["bm"]
            joined = torch.cat([v_a_out, v_m_out], dim=1)
            if joined.requires_grad:
                joined.register_hook(self._adjoint(a_held, d_a, d_m))
            units = p["ba"].numel()
            v_a_now, v_m_now = joined[:, :units], joined[:, units:]
            s_a, s_m = self.act(v_a_now), self.act(v_m_now)
            y = labels[t]
            if (y >= 0).any():
                mask = y >= 0
                if surprise:
                    mask = mask & (s_m.detach().argmax(dim=1) != y)
                count = int((y >= 0).sum())
                taught += count
                wrong += int(((s_m.detach().argmax(dim=1) != y) & (y >= 0)).sum())
                if mask.any():
                    total = total + torch.nn.functional.cross_entropy(
                        s_m[mask] / self.nudge_temperature, y[mask], reduction="sum")
            if record:
                seen["read"].append(read.detach().numpy().copy())
                seen["association"].append(s_a.detach().numpy().copy())
                seen["motor"].append(s_m.detach().numpy().copy())
            # the slot this read attended most, one moment older once this moment is written;
            # the first moment of a batch found no slots to read, and leaves no focus
            if t:
                at = read.detach().argmax(dim=1).numpy()
                focus = ages[at].astype(np.int64) + 1
            seen["focus"].append(focus.copy())
            c = s_a if self.carry_address else s_a.detach()
            c_trace = s_a if self.carry_trace else s_a.detach()
            v_a, v_m = v_a_star, v_m_star
            before = field.detach()
        seen["taught"], seen["wrong"] = taught, wrong
        return total / max(taught, 1), seen

    def _adjoint(self, a: Any, d_a: Any, d_m: Any) -> Any:
        """The hook on a settle's fixed point: the incoming gradient ``w`` becomes the adjoint
        ``y``, ``J' y = w`` with ``J = [[I, -A D_m], [-A' D_a, I]]``."""
        torch = self.torch
        units = a.shape[0]
        identity = torch.eye(a.shape[1], dtype=torch.float64)

        def hook(g: Any) -> Any:
            w_a, w_m = g[:, :units], g[:, units:]
            aa = (a.T[None] * d_a[:, None, :]) @ a
            schur = identity[None] - d_m[:, :, None] * aa
            right = w_m + d_m * (w_a @ a)
            y_m = torch.linalg.solve(schur, right[..., None])[..., 0]
            y_a = w_a + d_a * (y_m @ a.T)
            return torch.cat([y_a, y_m], dim=1)

        return hook

    def clamp(self) -> None:
        with self.torch.no_grad():
            for name in self.synapse_params:
                self.params[name].clamp_(-self.cap, self.cap)


def teach_by_moment(unrolled: Unrolled, stepper: Stepper, xs: Any, ys: Any) -> dict[str, Any]:
    """One normalized or Adam step per moment that has a wrong word, as lessons step: the moments
    in order, each one's words said wrong at the parameters of that moment, read from the start of
    the sentence at those parameters. Returns the steps and the words."""
    torch = unrolled.torch
    moments = xs.shape[0]
    taught = int((ys >= 0).sum())
    t, steps, wrong_words, sweeps, losses = 0, 0, 0, 0, []
    while t < moments:
        with torch.no_grad():
            _, seen = unrolled.forward(xs, ys, record=True)
        sweeps += seen["sweeps"]
        said = [torch.from_numpy(seen["motor"][u].argmax(axis=1)) for u in range(moments)]
        wrong = [(said[u] != ys[u]) & (ys[u] >= 0) for u in range(moments)]
        later = [u for u in range(t, moments) if wrong[u].any()]
        if not later:
            break
        t = later[0]
        labels = torch.full_like(ys[: t + 1], -1)
        labels[t][wrong[t]] = ys[t][wrong[t]]
        loss, seen = unrolled.forward(xs[: t + 1], labels)
        sweeps += seen["sweeps"]
        stepper(loss)
        steps += 1
        wrong_words += int(wrong[t].sum())
        losses.append(float(loss))
        t += 1
    return {"steps": steps, "wrong": wrong_words, "taught": taught, "sweeps": sweeps,
            "loss": float(np.mean(losses)) if losses else 0.0, "residual": seen["residual"],
            "unsettled": seen["unsettled"]}


def encode(rows: list[gr.Row]) -> tuple[Any, Any]:
    """The chamber's moments and words for a batch, ``(moments, rows, inputs)`` and ``(moments,
    rows)`` with -1 where nothing is taught, teacher-forced as its lessons are."""
    import torch

    xs, ys = [], []
    for r in rows:
        moments = [*r.pre, gr.concept("GO")] + [gr.spoken(w) for w in r.target[:-1]]
        xs.append(np.stack(moments))
        ys.append([-1] * len(r.pre) + list(r.target))
    return (torch.from_numpy(np.stack(xs).astype(np.float64)).transpose(0, 1),
            torch.from_numpy(np.array(ys, dtype=np.int64)).T)


class Stepper:
    """One step per batch on the unrolled graph: Adam as the controls take it, or the normalized
    step the brain's learner takes, on the gradient of ``T`` times the cross-entropy."""

    def __init__(self, unrolled: Unrolled, kind: str, rate: float) -> None:
        import torch

        self.torch, self.unrolled, self.kind, self.rate = torch, unrolled, kind, rate
        if kind == "adam":
            self.opt = torch.optim.Adam(unrolled.parameters(), lr=rate)
        else:
            config = unrolled.brain.learner.config
            self.rho, self.momentum = config.normalize, config.momentum
            self.floor, self.eta, self.eta_bias = (config.normalize_floor, config.eta,
                                                   config.eta_bias)
            self.count = 0
            self.velocity = {n: torch.zeros_like(p) for n, p in unrolled.params.items()}
            self.second = {n: torch.zeros_like(p) for n, p in unrolled.params.items()}

    def __call__(self, loss: Any) -> None:
        torch = self.torch
        params = self.unrolled.parameters()
        for p in params:
            p.grad = None
        loss.backward()
        if self.kind == "adam":
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            self.opt.step()
        else:
            self.count += 1
            with torch.no_grad():
                for name, p in self.unrolled.params.items():
                    if p.grad is None:
                        continue
                    g = p.grad * self.unrolled.temperature
                    self.velocity[name] = (self.momentum * self.velocity[name]
                                           + (1 - self.momentum) * g)
                    self.second[name] = self.rho * self.second[name] + (1 - self.rho) * g * g
                    rms = (self.second[name] / (1 - self.rho ** self.count)).sqrt() + self.floor
                    move = self.velocity[name] / (1 - self.momentum ** self.count) / rms
                    eta = self.eta_bias if name in ("ba", "bm") else self.eta
                    p -= eta * move
        self.unrolled.clamp()


# ------------------------------------------------------------------------------ the checks


def _settled_drive(brain: Any, rows: list[gr.Row]) -> tuple[np.ndarray, np.ndarray]:
    """The chamber's life up to a batch's first word, the moment's drive and the word's labels:
    the moment a lesson at ``GO`` reads."""
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    x = np.tile(gr.concept("GO"), (len(rows), 1))
    return brain.stimulus(x), np.array([r.target[0] for r in rows])


def _cost_at(learner: Any, drive: np.ndarray, labels: np.ndarray, brain: Any, warm: Any
             ) -> float:
    """``T`` times the cross-entropy at the fixed point of ``brain``, by the library's settle."""
    phase = brain.equilibrate(drive, state=warm, budget=200000, chunk=64, tolerance=1e-12)
    if not np.all(phase.qualified):
        raise ArithmeticError(f"unqualified fixed point: {phase.residual.max()}")
    z = phase.state.activation[:, learner.output_index] / learner.config.temperature
    z -= z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-learner.config.temperature * log_p[np.arange(len(labels)), labels].mean())


def _family_coordinates(brain: Any, rng: np.random.Generator,
                        grad: np.ndarray | None = None) -> list[tuple[str, int]]:
    graph = brain.connectome
    pops = {name: np.asarray(ids) for name, ids in graph.populations.items()}
    out: list[tuple[str, int]] = []
    for family, a, b in (("sensory", "sensory", "association"),
                         ("prefrontal", "prefrontal", "association"),
                         ("recall", "recall", "association"), ("next", "next", "association"),
                         ("motor", "association", "motor")):
        members = np.flatnonzero(np.isin(graph.pre, pops[a]) & np.isin(graph.post, pops[b]))
        out += [(family, int(i)) for i in rng.choice(members, 3, replace=False)]
        if grad is not None:
            out.append((family, int(members[np.abs(grad[members]).argmax()])))
    out += [("bias", int(i)) for i in rng.choice(pops["association"], 3, replace=False)]
    out += [("bias", int(i)) for i in rng.choice(pops["motor"], 3, replace=False)]
    return out


def check_moment(brain: Any, rows: list[gr.Row], seed: int = 0, h: float = 1e-6
                 ) -> dict[str, Any]:
    """G1's gradient against central finite differences of the library's own fixed points, and
    against the local contrast as its nudge shrinks."""
    learner = brain.learner
    drive, labels = _settled_drive(brain, rows)
    graph_brain = learner.brain
    free = graph_brain.equilibrate(drive, budget=200000, chunk=64, tolerance=1e-12).state
    grad, grad_bias, report = exact_gradient(learner, free, labels)
    rng = np.random.default_rng(seed)
    entries = []
    for family, i in _family_coordinates(graph_brain, rng, grad):
        values = []
        for sign in (-1.0, 1.0):
            if family == "bias":
                bias = graph_brain.bias.copy()
                bias[i] += sign * h
                candidate = graph_brain.with_parameters(bias=bias)
            else:
                efficacy = graph_brain.efficacy.copy()
                efficacy[i] += sign * h
                if learner.reverse[i] >= 0:
                    efficacy[learner.reverse[i]] += sign * h
                candidate = graph_brain.with_parameters(efficacy=efficacy)
            values.append(_cost_at(learner, drive, labels, candidate, free))
        finite = (values[1] - values[0]) / (2 * h)
        exact = float(grad_bias[i] if family == "bias" else grad[i])
        entries.append({"family": family, "coordinate": i, "finite": finite, "exact": exact,
                        "error": abs(finite - exact)})
    limits = []
    target = learner.targets(labels)
    for beta in (0.1, 0.01, 0.001):
        plus = graph_brain.equilibrate(drive, state=free, budget=200000, chunk=64, tolerance=1e-12,
                                       nudge=learner.nudge_for(target, beta)).state
        minus = graph_brain.equilibrate(drive, state=free, budget=200000, chunk=64,
                                        tolerance=1e-12,
                                        nudge=learner.nudge_for(target, -beta)).state
        a_plus, a_minus = plus.activation[:, graph_brain.connectome.pre], minus.activation[
            :, graph_brain.connectome.pre]
        b_plus, b_minus = plus.activation[:, graph_brain.connectome.post], minus.activation[
            :, graph_brain.connectome.post]
        local = ((a_plus * b_plus - a_minus * b_minus).mean(axis=0) / (2 * beta))
        limits.append({"beta": beta, "cosine": cosine(local, -grad / graph_brain._gain_pre),
                       "ratio": float(np.linalg.norm(local) / np.linalg.norm(grad))})
    worst = max(float(e["error"]) for e in entries)  # type: ignore[arg-type]
    scale = max(abs(float(e["exact"])) for e in entries)  # type: ignore[arg-type]
    return {"entries": entries, "worst_error": worst, "largest_gradient": scale,
            "polish": report, "contrast_at_a_full_settle": limits}


def check_time(unrolled: Unrolled, rows: list[gr.Row], seed: int = 0, h: float = 1e-6
               ) -> dict[str, Any]:
    """G2's gradient against central finite differences of the whole batch's cost, per family of
    parameters, the read's address and position maps among them."""
    torch = unrolled.torch
    xs, ys = encode(rows)
    for p in unrolled.parameters():
        p.grad = None
    loss, seen = unrolled.forward(xs, ys)
    loss.backward()
    rng = np.random.default_rng(seed)
    entries: list[dict[str, Any]] = []
    for name, param in unrolled.params.items():
        if param.grad is None:
            continue
        flat = param.grad.reshape(-1).numpy()
        picks = rng.choice(flat.size, min(3, flat.size), replace=False).tolist()
        picks.append(int(np.abs(flat).argmax()))
        for i in picks:
            values = []
            flipped = False
            for sign in (-1.0, 1.0):
                with torch.no_grad():
                    param.reshape(-1)[i] += sign * h
                    cost, moved = unrolled.forward(xs, ys)
                    values.append(float(cost))
                    param.reshape(-1)[i] -= sign * h
                flipped = flipped or not all(
                    np.array_equal(a, b) for a, b in zip(seen["focus"], moved["focus"],
                                                         strict=True))
            finite = (values[1] - values[0]) / (2 * h)
            entries.append({"family": name, "coordinate": i, "finite": finite,
                            "exact": float(flat[i]), "error": abs(finite - float(flat[i])),
                            "focus_flipped": flipped})
    for p in unrolled.parameters():
        p.grad = None
    smooth = [e for e in entries if not e["focus_flipped"]]
    return {"entries": entries, "loss": float(loss),
            "worst_error": max(e["error"] for e in smooth),
            "largest_gradient": max(abs(e["exact"]) for e in smooth),
            "worst_relative": max(e["error"] / max(abs(e["exact"]), 1e-3) for e in smooth),
            "coordinates": len(entries), "focus_flips": len(entries) - len(smooth),
            "worst_error_at_a_flip": max((e["error"] for e in entries if e["focus_flipped"]),
                                         default=0.0),
            "sweeps": seen["sweeps"], "residual": seen["residual"]}


def run_check(spec: dict[str, Any]) -> dict[str, Any]:
    """Both rungs' checks on the newborn and after ``teach`` recipe batches, with rows of every
    family."""
    import torch

    torch.set_num_threads(1)
    founder = int(spec["founder"])
    split = gr.Split.of(founder)
    teach = int(spec.get("teach", 300))
    tests = gr.panels(founder, 1)
    out: dict[str, Any] = {"founder": founder, "teach": teach, "torch": torch.__version__,
                           "moments": {}, "time": {}}
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    brain = gr.make_brain("recipe", founder)
    for stage in ("newborn", "taught"):
        if stage == "taught":
            for _ in range(teach):
                gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True)
        for panel in ("clause", "dyck", "path"):
            rows = tests[panel][0][:8]
            fork = copy.deepcopy(brain)
            out["moments"][f"{stage}/{panel}"] = check_moment(fork, rows)
            fork = copy.deepcopy(brain)
            out["time"][f"{stage}/{panel}"] = check_time(Unrolled(fork), rows)
    out["agreement"] = {panel: agreement(brain, tests[panel][0][:8])
                        for panel in ("clause", "dyck", "path")}
    out["agreement_tight"] = {panel: agreement(tightened(brain), tests[panel][0][:8])
                              for panel in ("clause", "dyck", "path")}
    deep = gr.make_brain("recipe", founder, settings=DEEP)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    out["moments_deep"] = {}
    for stage in ("newborn", "taught"):
        if stage == "taught":
            for _ in range(teach):
                gr.speak(deep, gr.train_batch(rng, split, "g1"), teach=True)
        for panel in ("clause", "dyck", "path"):
            out["moments_deep"][f"{stage}/{panel}"] = check_moment(copy.deepcopy(deep),
                                                                   tests[panel][0][:8])
    out["moments_deep_worst"] = max(r["worst_error"] for r in out["moments_deep"].values())
    for rung in ("moments", "time"):
        out[rung + "_worst"] = max(r["worst_error"] for r in out[rung].values())
    out["time_flips"] = sum(r["focus_flips"] for r in out["time"].values())
    return out


def tightened(brain: Any) -> Any:
    """A copy of a brain that settles to the fixed point itself: its acts at a residual of 1e-10,
    where the composed 0.003 can stop in a slow region far from it."""
    fork = copy.deepcopy(brain)
    fork.learner.config = dataclasses.replace(fork.learner.config, tolerance=1e-10,
                                              free_steps=20000)
    return fork


def agreement(brain: Any, rows: list[gr.Row]) -> dict[str, float]:
    """The unrolled forward against the library's on the same moments, teacher-forced: the read,
    the association and the motor activity, and the word."""
    fork = copy.deepcopy(brain)
    unrolled = Unrolled(fork)
    xs, ys = encode(rows)
    with unrolled.torch.no_grad():
        _, seen = unrolled.forward(xs, ys, record=True)
    fork.reset()
    worst = {"read": 0.0, "association": 0.0, "motor": 0.0}
    same = total = 0
    population = fork.connectome.populations
    for t in range(xs.shape[0]):
        action = fork.act(xs[t].numpy(), greedy=True)
        state = fork.basal_ganglia.state
        heads = fork.recall.attended_heads
        if heads is not None and heads[0] is not None:  # the first moment found no slots
            read = heads[0]
            worst["read"] = max(worst["read"], float(np.abs(read - seen["read"][t]).max()))
        s = np.atleast_2d(state.activation)
        for name, region in (("association", "association"), ("motor", "motor")):
            gap = np.abs(s[:, np.asarray(population[region])] - seen[name][t]).max()
            worst[name] = max(worst[name], float(gap))
        said = seen["motor"][t].argmax(axis=1)
        same += int((said == np.asarray(action).reshape(-1)).sum())
        total += len(said)
    return {**worst, "words_equal": same / max(total, 1)}


# -------------------------------------------------------------------------------- the lives


def _read(brain: Any, tests: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: gr.read_panel(brain, tests[name], twins=name in gr.TWINS) for name in names}


def verb_heads(brain: Any, founder: int, batches: int = 17) -> dict[str, Any]:
    """The verb, teacher-forced, per distractor count on fresh trained clauses (seventeen batches
    of sixteen, as the value path's lane reads it): what is said, and where each head of the read
    lands: its mass on the verb's meaning moment, the agent's meaning moment, the agent's word as
    said, the distractors' meaning moments and words as said, and its effective slots; with two
    heads, how much of their mass they share and how often their top slots are one."""
    test = np.random.default_rng(np.random.SeedSequence([founder, 4217]))
    split = gr.Split.of(founder)
    out: dict[str, Any] = {}
    for length in (0, 1, 2):
        tally = {key: 0 for key in ("n", "said", "other_number", "other_verb", "near", "noun",
                                    "other")}
        heads: dict[int, dict[str, list[float]]] = {}
        shared: list[float] = []
        same_top: list[float] = []
        for _ in range(batches):
            rows = [gr.trained_clause(test, split, length, False) for _ in range(gr.BATCH)]
            fork = copy.deepcopy(brain)
            fork.reset()
            pre = len(rows[0].pre)
            for t in range(pre):
                fork.act(np.stack([r.pre[t] for r in rows]), greedy=True)
            expected = [gr.sources(r) for r in rows]
            for j in range(len(rows[0].target)):
                x = (np.tile(gr.concept("GO"), (len(rows), 1)) if j == 0
                     else np.stack([gr.spoken(r.target[j - 1]) for r in rows]))
                action = np.asarray(fork.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
                if not gr._kind(rows[0], j).endswith("_verb"):
                    continue
                reads = [p for p in fork.recall.attended_heads if p is not None]
                for i, r in enumerate(rows):
                    word, said = int(r.target[j]), int(action[i])
                    tally["n"] += 1
                    tally["said"] += int(said == word)
                    if said != word:
                        kind = ("other_number" if 8 <= said <= 11 and (said - 8) // 2 ==
                                (word - 8) // 2 else "other_verb" if 8 <= said <= 11
                                else "near" if said == gr.NEAR else "noun" if said < 8
                                else "other")
                        tally[kind] += 1
                    # the boundary slot, the heard moments, GO, then each word said
                    verb, agent = int(expected[i][j]) + 1, int(expected[i][0]) + 1
                    mods = [int(expected[i][k]) + 1 for k in range(2, j, 2)]
                    words = [pre + 2 + k for k in range(2, j, 2)]
                    for h, p in enumerate(reads):
                        row = p[i]
                        positive = row[row > 0]
                        size = len(row)
                        cells = {
                            "verb": float(row[verb]), "agent": float(row[agent]),
                            "agent_said": float(row[pre + 2]) if pre + 2 < size else 0.0,
                            "mods": float(sum(row[m] for m in mods)),
                            "mods_said": float(sum(row[w] for w in words if w < size)),
                            "top": float(row.max()),
                            "slots": float(np.exp(-(positive * np.log(positive)).sum())),
                        }
                        keep = heads.setdefault(h, {})
                        for name, value in cells.items():
                            keep.setdefault(name, []).append(value)
                    if len(reads) == 2:
                        shared.append(float(np.minimum(reads[0][i], reads[1][i]).sum()))
                        same_top.append(float(np.argmax(reads[0][i]) == np.argmax(reads[1][i])))
        reading: dict[str, Any] = dict(tally)
        reading["heads"] = {str(h): {name: round(float(np.mean(values)), 4)
                                     for name, values in cells.items()}
                            for h, cells in sorted(heads.items())}
        if shared:
            reading.update(shared=round(float(np.mean(shared)), 4),
                           same_top=round(float(np.mean(same_top)), 4))
        out[f"L{length}"] = reading
    return out


def replicate(brain: Any, replica: int) -> None:
    """A replica of a newborn: its efficacies changed at the last bits, tied pairs alike, so that
    only arithmetic separates the lives (replica 0 is the newborn itself)."""
    if not replica:
        return
    learner = brain.learner
    rng = np.random.default_rng(np.random.SeedSequence([int(replica), 260]))
    noise = rng.uniform(-1.0, 1.0, learner.brain.connectome.synapses)
    index = np.flatnonzero(learner.reverse >= 0)
    noise[index] = noise[np.minimum(index, learner.reverse[index])]  # a pair is one efficacy
    learner.brain = learner.brain.with_parameters(
        efficacy=learner.brain.efficacy * (1.0 + REPLICA * noise))


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``arm``, ``founder``, ``mode``, optional ``init_seed`` (the brain's seed, on the
    founder's split, lessons and panels), ``replica``, ``batches``, and for a brain's lessons
    ``epochs``."""
    import warnings

    import torch

    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    arm, founder = spec["arm"], int(spec["founder"])
    mode = spec.get("mode", "online")
    settings = ARMS[arm]
    n = int(spec.get("batches", 4000))
    init_seed = int(spec.get("init_seed", founder))
    started = time.monotonic()
    brain = newborn(settings, init_seed, spec)
    replica = int(spec.get("replica", 0))
    replicate(brain, replica)
    install_arm(brain, settings)
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    batches = [gr.train_batch(rng, split, "g1") for _ in range(n)]
    order = np.random.default_rng(np.random.SeedSequence([founder, 99, 0]))
    rate = float(spec.get("rate") or RATES[mode])
    work = gr.new_work()
    epochs = 1
    last = None
    notes: dict[str, Any] = {}
    rung = settings.get("rung", 0)
    if rung == 2:
        unrolled = Unrolled(brain, **{key: settings[key] for key in (
            "frozen_maps", "carry_trace", "carry_address") if key in settings})
        stepper = Stepper(unrolled, settings["step"], rate)
        surprise = bool(settings.get("surprise"))
        encoded = [encode(b) for b in batches]

        def teach(index: int) -> float:
            if settings.get("per_moment"):
                went = teach_by_moment(unrolled, stepper, *encoded[index])
                notes["sweeps"] = notes.get("sweeps", 0) + went["sweeps"]
                notes["unsettled"] = notes.get("unsettled", 0) + went["unsettled"]
                notes["worst_residual"] = max(notes.get("worst_residual", 0.0), went["residual"])
                notes["taught"] = notes.get("taught", 0) + went["taught"]
                notes["wrong"] = notes.get("wrong", 0) + went["wrong"]
                notes["updates"] = notes.get("updates", 0) + went["steps"]
                return float(went["loss"])
            loss, seen = unrolled.forward(*encoded[index], surprise=surprise)
            notes["sweeps"] = notes.get("sweeps", 0) + seen["sweeps"]
            notes["unsettled"] = notes.get("unsettled", 0) + seen["unsettled"]
            notes["worst_residual"] = max(notes.get("worst_residual", 0.0), seen["residual"])
            notes["taught"] = notes.get("taught", 0) + seen["taught"]
            notes["wrong"] = notes.get("wrong", 0) + seen["wrong"]
            if seen["wrong"] or not surprise:
                stepper(loss)
                notes["updates"] = notes.get("updates", 0) + 1
            return float(loss)

        if mode == "online":
            losses = [teach(i) for i in range(n)]
            last = float(np.mean(losses[-50:]))
        else:
            best, stale, epochs = math.inf, 0, 0
            while epochs < EPOCHS and stale < PATIENCE:
                epoch = [teach(int(i)) for i in order.permutation(n)]
                epochs += 1
                mean = float(np.mean(epoch))
                stale = 0 if mean < best - 1e-4 else stale + 1
                best = min(best, mean)
                last = mean
        unrolled.write_back()
        notes["agreement"] = {panel: agreement(brain, tests[panel][0][:8])
                              for panel in ("clause", "dyck", "path") if panel in tests}
        notes["agreement_tight"] = {panel: agreement(tightened(brain), tests[panel][0][:8])
                                    for panel in ("clause", "dyck", "path") if panel in tests}
    else:
        if rung == 1:
            install_exact(brain, settings["step"], rate)
        rounds = int(spec.get("epochs", 1)) if mode == "offline" else 1
        if rounds == 0:  # the controls' rule: passes until the epoch's error stops falling
            names = tuple(spec.get("panels") or gr.G1_PANELS)
            best, stale, epochs = math.inf, 0, 0
            history: list[dict[str, Any]] = []
            while epochs < EPOCHS and stale < PATIENCE:
                before, began = dict(work), time.monotonic()
                for i in order.permutation(n):
                    gr.speak(brain, batches[int(i)], teach=True, work=work)
                epochs += 1
                passed = {key: work[key] - before[key] for key in work}
                # rows answered wrong, the only loss the lessons see
                last = 1.0 - passed["agreed_rows"] / max(passed["rows"], 1)
                stale = 0 if last < best - 1e-4 else stale + 1
                best = min(best, last)
                history.append({"epoch": epochs, "error": last, "work": passed,
                                "seconds": round(time.monotonic() - began, 1),
                                "panels": _read(copy.deepcopy(brain), tests, names)})
                if spec.get("epoch_log"):
                    with Path(spec["epoch_log"]).open("a") as sink:
                        sink.write(json.dumps({"row": arm, "founder": founder,
                                               "init_seed": init_seed, **history[-1]},
                                              default=float) + "\n")
            notes["epochs"] = history
        else:
            for _ in range(rounds):
                sequence = range(n) if mode == "online" else order.permutation(n)
                for i in sequence:
                    gr.speak(brain, batches[int(i)], teach=True, work=work)
            epochs = rounds
        if rung == 1:
            kept = brain.learner.exact_notes
            notes = {k: kept[k] for k in ("gradients", "polish_steps", "polish_sweeps",
                                          "unpolished_rows", "worst_residual")}
            if kept["readings"]:
                notes["observed"] = summarize_readings(kept["readings"])
    work["seconds"] = round(time.monotonic() - started, 1)
    names = tuple(spec.get("panels") or gr.G1_PANELS)
    panels = _read(brain, tests, names)
    contrast = (settings.get("brain") or {}).get("recall_address_contrast")
    if (settings.get("named_credit") or settings.get("read_credit") or contrast or arm == "recipe"
            or spec.get("locate")) and "path" in tests and "path" in names:
        located = {name: gr.locate(brain, tests[name]) for name in ("path", "clause")}
        notes["located"] = {name: {key: located[name][key] for key in sorted(located[name])
                                   if key.startswith(("hop", "verb", "path"))}
                            for name in located}
    if rung == 2:
        notes["panels_tight"] = {
            name: {key: reading.get(key) for key in ("exact", "both", "agree", "verb", "next_word")}
            for name, reading in _read(tightened(brain), tests, names).items()}
    refused = gr.new_work()
    fork = copy.deepcopy(brain)
    for batch in tests[names[0]]:
        gr.speak(fork, batch, teach=False, work=refused)
    if spec.get("verb_heads"):
        notes["verb_heads"] = verb_heads(brain, founder, int(spec.get("verb_batches", 17)))
    parameters = int(brain.parameters())
    c = brain.connectome
    notes["synapses"] = len(c.pre)
    notes["trace_synapses"] = int(np.isin(c.pre, c.populations.get("prefrontal", ())).sum())
    return {
        "row": arm, "settings": settings, "mode": mode, "founder": founder, "init_seed": init_seed,
        "replica": replica, "width": arm_width(settings, spec), "parameters": parameters,
        "parameters_total": parameters, "brain_parameters": parameters, "rate": rate,
        "phases": [["g1", n]], "torch": torch.__version__, "notes": notes,
        "refused_reads": refused["exhausted"], "train_refused": work["refused"],
        "readings": [{"phase": 0, "grammar": "g1", "batches": n, "epochs": epochs,
                      "loss_last": last, "work": work, "panels": panels}],
        "seconds": round(time.monotonic() - started, 1),
    }


def run_crossing(spec: dict[str, Any]) -> dict[str, Any]:
    """An arm's life on the crossing panel of the ablations (#228's non-additive addressing): the
    same batches and test as the rows' and the recipe brain's, read by the library."""
    import warnings

    import torch

    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    gr._module("controls", gr.HERE / "controls.py")
    ab = gr._module("ablations", gr.HERE / "ablations.py")
    arm, founder = spec["arm"], int(spec["founder"])
    settings = ARMS[arm]
    n = int(spec.get("batches", 30000))
    init_seed = int(spec.get("init_seed", founder))
    started = time.monotonic()
    brain = newborn(settings, init_seed, spec)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"], 228]))
    rate = float(spec.get("rate") or RATES["online"])
    work = gr.new_work()
    notes: dict[str, Any] = {}
    rung = settings.get("rung", 0)
    if rung == 2:
        unrolled = Unrolled(brain, **{key: settings[key] for key in (
            "frozen_maps", "carry_trace", "carry_address") if key in settings})
        stepper = Stepper(unrolled, settings["step"], rate)
        for _ in range(n):
            xs, ys = encode(ab.crossing(rng))
            if settings.get("per_moment"):
                went = teach_by_moment(unrolled, stepper, xs, ys)
                notes["updates"] = notes.get("updates", 0) + went["steps"]
                continue
            loss, seen = unrolled.forward(xs, ys, surprise=bool(settings.get("surprise")))
            notes["unsettled"] = notes.get("unsettled", 0) + seen["unsettled"]
            if seen["wrong"] or not settings.get("surprise"):
                stepper(loss)
                notes["updates"] = notes.get("updates", 0) + 1
        unrolled.write_back()
    else:
        if rung == 1:
            install_exact(brain, settings["step"], rate)
        for _ in range(n):
            gr.speak(brain, ab.crossing(rng), teach=True, work=work)
    work["seconds"] = round(time.monotonic() - started, 1)
    test = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["test"], 228]))
    batches = [ab.crossing(test) for _ in range(int(spec.get("panel_batches", 4)))]
    panel = gr.read_panel(brain, batches)
    pairs: dict[str, list[float]] = {}
    fork = copy.deepcopy(brain)
    for batch in batches:
        for r, said in zip(batch, gr.speak(fork, batch, teach=False), strict=True):
            pairs.setdefault(f"c{r.meta['c']}k{r.meta['k']}", []).append(float(said == r.target))
    panel["conditions"] = {key: round(float(np.mean(v)), 4) for key, v in sorted(pairs.items())}
    parameters = int(brain.parameters())
    return {
        "row": arm, "settings": settings, "mode": "crossing", "founder": founder,
        "init_seed": init_seed, "replica": 0, "width": arm_width(settings, spec),
        "parameters": parameters, "parameters_total": parameters, "phases": [["crossing", n]],
        "notes": notes, "refused_reads": 0, "train_refused": work["refused"],
        "readings": [{"phase": 0, "grammar": "crossing", "batches": n, "epochs": 1,
                      "loss_last": None, "work": work, "panels": {"crossing": panel}}],
        "seconds": round(time.monotonic() - started, 1),
    }


# ----------------------------------------------------------------- #228's crossing, keyed

NAMES, VALUES, DIRECT = (0, 1), (2, 3, 6, 7), 3  # dog, cat; fox owl foxes owls; a third by name


@dataclass
class Ask:
    """One stream of #228's crossing, the rows of the coupled lane's ``keyed`` to the bit (a check
    in the lane's evidence): two records in an order drawn per stream, so that a position names no
    fact, then the context, at once the question, and the value of the fact ``c xor k``."""

    pre: list[np.ndarray]
    question: np.ndarray
    answer: int
    c: int
    k: int  # the question's bit; -1 for a fact asked by its name
    fact: int
    values: tuple[int, int]
    first: int  # the fact witnessed first


def keyed(rng: np.random.Generator, n: int = gr.BATCH, direct: float = 0.0) -> list[Ask]:
    rows = []
    for _ in range(n):
        c, k = int(rng.integers(2)), int(rng.integers(2))
        drawn = rng.permutation(VALUES)[:2]
        values = (int(drawn[0]), int(drawn[1]))
        first = int(rng.integers(2))
        order = (first, 1 - first)
        pre = [gr.heard(word) for fact in order for word in (NAMES[fact], values[fact])]
        pre.append(gr.heard(gr.OPENS[c]))
        if rng.random() < direct:
            fact = int(rng.integers(2))
            rows.append(Ask(pre, gr.heard(NAMES[fact]), values[fact], c, -1, fact, values, first))
        else:
            fact = c ^ k
            question = gr.heard(gr.MATCH[gr.OPENS[k]])
            rows.append(Ask(pre, question, values[fact], c, k, fact, values, first))
    return rows


def keyed_moments(rows: list[Ask], timing: str) -> tuple[Any, Any]:
    """The moments and words of a batch: ``own``, the answer at the question's own moment, #228's
    form; ``go``, the answer and ``end`` after a ``GO`` moment, where a conjunction can form in the
    cortex first."""
    import torch

    xs, ys = [], []
    for r in rows:
        if timing == "own":
            moments = [*r.pre, r.question]
            labels = [-1] * len(r.pre) + [r.answer]
        else:
            moments = [*r.pre, r.question, gr.concept("GO"), gr.spoken(r.answer)]
            labels = [-1] * (len(r.pre) + 1) + [r.answer, gr.END]
        xs.append(np.stack(moments))
        ys.append(labels)
    return (torch.from_numpy(np.stack(xs).astype(np.float64)).transpose(0, 1),
            torch.from_numpy(np.array(ys, dtype=np.int64)).T)


def read_keyed(brain: Any, rows: list[Ask], timing: str, control: str | None = None
               ) -> dict[str, Any]:
    """The rows answered on a fork of ``brain``, by the library: right, the other fact's value
    instead, and the read's attention where the answer is read and at the context. ``relevant``
    and ``other`` are the mass on the records' names as the coupled lane's reading gives them
    (a name's next port holds its value), ``*_record`` the mass on a record's name and value,
    ``slots`` the read's effective slot count. The controls act after the context is heard:
    ``erase_relevant``, ``erase_other``, ``memory_off`` zero slots, ``trace_off`` the working
    trace, ``cortex_off`` the cortex the address reads, ``both_off`` the two, each before the
    question. ``mass`` holds, by context and question, the mass on fact 0's name and value, fact
    1's name and value, and the context's slot."""
    fork = copy.deepcopy(brain)
    recall = fork.recall
    n = len(rows)
    fork.reset()
    at_context = None
    for t in range(len(rows[0].pre)):
        fork.act(np.stack([r.pre[t] for r in rows]), greedy=True)
        if t == len(rows[0].pre) - 1:
            at_context = recall.attended_heads[0].copy()
    start = 1 if recall.boundary else 0

    def names_of(r: Ask) -> tuple[int, int]:
        return (start, start + 2) if r.first == 0 else (start + 2, start)

    if control in ("erase_relevant", "erase_other", "memory_off"):
        keys, values, nexts = recall.keys.copy(), recall.values.copy(), recall.nexts.copy()
        for i, r in enumerate(rows):
            if control == "memory_off":
                slots = list(range(recall.slots))
            else:
                name = names_of(r)[r.fact if control == "erase_relevant" else 1 - r.fact]
                slots = [name, name + 1]
            for slot in slots:
                keys[i, slot], values[i, slot], nexts[i, slot] = 0.0, 0.0, 0.0
        recall.keys, recall.values, recall.nexts = keys, values, nexts
        recall._contents, recall._content_keys = [], None
    if control in ("trace_off", "both_off"):
        fork.working_memory.trace = np.zeros_like(fork.working_memory.trace)
    if control in ("cortex_off", "both_off"):
        recall.previous = np.zeros_like(recall.previous)
    question = np.stack([r.question for r in rows])
    if timing == "own":
        answer = np.asarray(fork.act(question, greedy=True)).reshape(n, -1)[:, 0]
    else:
        fork.act(question, greedy=True)
        answer = np.asarray(fork.act(np.tile(gr.concept("GO"), (n, 1)), greedy=True)
                            ).reshape(n, -1)[:, 0]
    read = recall.attended_heads[0]
    out: dict[str, list[float]] = {key: [] for key in (
        "right", "other_value", "relevant", "other", "relevant_record", "other_record", "slots",
        "context_relevant", "context_other", "context_first_by_c")}
    pairs: dict[str, list[float]] = {}
    mass_by: dict[str, list[list[float]]] = {}
    for i, r in enumerate(rows):
        right = bool(answer[i] == r.answer)
        out["right"].append(float(right))
        out["other_value"].append(float(answer[i] == r.values[1 - r.fact]))
        names = names_of(r)
        mass = read[i, list(names)]
        record = read[i, list(names)] + read[i, [names[0] + 1, names[1] + 1]]
        out["relevant"].append(float(mass[r.fact]))
        out["other"].append(float(mass[1 - r.fact]))
        out["relevant_record"].append(float(record[r.fact]))
        out["other_record"].append(float(record[1 - r.fact]))
        positive = read[i][read[i] > 0]
        out["slots"].append(float(np.exp(-(positive * np.log(positive)).sum())))
        if at_context is not None:
            before = at_context[i, list(names)] + at_context[i, [names[0] + 1, names[1] + 1]]
            out["context_relevant"].append(float(before[r.fact]))
            out["context_other"].append(float(before[1 - r.fact]))
            out["context_first_by_c"].append(float(before[r.c]))  # the record named by c alone
        pairs.setdefault(f"c{r.c}k{r.k}", []).append(float(right))
        # by context and question, the mass on each fact's name and value, by the fact's identity
        # and whichever came first: the XOR's pattern here is selection; a pattern that follows
        # the question or the context alone, with the answer still right, is the cortex decoding
        mass_by.setdefault(f"c{r.c}k{r.k}", []).append(
            [float(read[i, names[0]]), float(read[i, names[0] + 1]), float(read[i, names[1]]),
             float(read[i, names[1] + 1]), float(read[i, start + 4])])
    summary: dict[str, Any] = {key: round(float(np.mean(v)), 4) for key, v in out.items() if v}
    summary["conditions"] = {key: round(float(np.mean(v)), 4) for key, v in sorted(pairs.items())}
    summary["mass"] = {key: [round(float(x), 3) for x in np.mean(v, axis=0)]
                       for key, v in sorted(mass_by.items())}
    return summary


def teach_keyed(brain: Any, rows: list[Ask], timing: str) -> None:
    """One batch by the brain's own lessons: a lesson before the greedy act on each taught moment,
    as the chamber's life orders them."""
    n = len(rows)
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    question = np.stack([r.question for r in rows])
    labels = np.array([r.answer for r in rows])
    if timing == "own":
        brain.lesson(question, labels)
        brain.act(question, greedy=True)
        return
    brain.act(question, greedy=True)
    go = np.tile(gr.concept("GO"), (n, 1))
    brain.lesson(go, labels)
    brain.act(go, greedy=True)
    said = np.stack([gr.spoken(int(word)) for word in labels])
    brain.lesson(said, np.full(n, gr.END))
    brain.act(said, greedy=True)


def run_keyed(spec: dict[str, Any]) -> dict[str, Any]:
    """An arm's life on #228's keyed crossing at a ``timing``, on the coupled lane's seeds and its
    taught mix (a third of the rows ask a fact by its name)."""
    import warnings

    import torch

    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    arm, founder, timing = spec["arm"], int(spec["founder"]), spec.get("timing", "own")
    settings = ARMS[arm]
    n = int(spec.get("batches", 3000))
    init_seed = int(spec.get("init_seed", founder))
    started = time.monotonic()
    brain = newborn(settings, init_seed, spec)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"], 228, 1]))
    test = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["test"], 228, 1]))
    panel_batches = int(spec.get("panel_batches", 8))
    batches = [keyed(test) for _ in range(panel_batches)]
    direct = [keyed(test, direct=1.0) for _ in range(max(panel_batches // 4, 1))]
    rate = float(spec.get("rate") or RATES["online"])
    notes: dict[str, Any] = {}
    rung = settings.get("rung", 0)
    if rung == 2:
        unrolled = Unrolled(brain, **{key: settings[key] for key in (
            "frozen_maps", "carry_trace", "carry_address") if key in settings})
        stepper = Stepper(unrolled, settings["step"], rate)
        for _ in range(n):
            xs, ys = keyed_moments(keyed(rng, direct=1.0 / DIRECT), timing)
            if settings.get("per_moment"):
                went = teach_by_moment(unrolled, stepper, xs, ys)
                notes["updates"] = notes.get("updates", 0) + went["steps"]
                continue
            loss, seen = unrolled.forward(xs, ys, surprise=bool(settings.get("surprise")))
            notes["unsettled"] = notes.get("unsettled", 0) + seen["unsettled"]
            if seen["wrong"] or not settings.get("surprise"):
                stepper(loss)
                notes["updates"] = notes.get("updates", 0) + 1
        unrolled.write_back()
    else:
        install_arm(brain, settings)
        if rung == 1:
            install_exact(brain, settings["step"], rate)
        for _ in range(n):
            teach_keyed(brain, keyed(rng, direct=1.0 / DIRECT), timing)

    def panel(rows_: list[list[Ask]], control: str | None) -> dict[str, Any]:
        reads = [read_keyed(brain, rows, timing, control) for rows in rows_]
        keys = [key for key in reads[0] if key not in ("conditions", "mass")]
        out: dict[str, Any] = {key: round(float(np.mean([r[key] for r in reads])), 4)
                               for key in keys}
        conditions: dict[str, list[float]] = {}
        masses: dict[str, list[list[float]]] = {}
        for r in reads:
            for key, value in r["conditions"].items():
                conditions.setdefault(key, []).append(value)
            for key, row in r["mass"].items():
                masses.setdefault(key, []).append(row)
        out["conditions"] = {key: round(float(np.mean(v)), 4)
                             for key, v in sorted(conditions.items())}
        out["mass"] = {key: [round(float(x), 3) for x in np.mean(v, axis=0)]
                       for key, v in sorted(masses.items())}
        return out

    readings = {"intact": panel(batches, None), "direct": panel(direct, None)}
    for control in ("erase_relevant", "erase_other", "memory_off", "trace_off", "cortex_off",
                    "both_off"):
        readings[control] = panel(batches, control)
    return {
        "row": arm, "settings": settings, "mode": "keyed_" + timing, "timing": timing,
        "founder": founder, "init_seed": init_seed, "batches": n, "notes": notes,
        "readings": readings, "seconds": round(time.monotonic() - started, 1),
    }


def pending(specs: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    """The specifications whose lives ``out`` does not hold yet."""

    def key(life: dict[str, Any]) -> tuple[Any, ...]:
        founder = int(life["founder"])
        return (life.get("row", life.get("arm")), life.get("mode"), founder,
                int(life.get("init_seed", founder)), int(life.get("replica", 0)))

    done = {key(json.loads(line)) for line in out.read_text().splitlines()
            if line.strip()} if out.exists() else set()
    return [spec for spec in specs if key(spec) not in done]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arms", nargs="*", default=["G1n", "G1a"], choices=list(ARMS))
    parser.add_argument("--modes", nargs="*", default=["online"], choices=["online", "offline"])
    parser.add_argument("--founders", nargs="*", type=int, default=list(gr.FOUNDERS))
    parser.add_argument("--init-seeds", nargs="*", type=int, default=[],
                        help="the brain's seeds per founder, -1 for the founder's own")
    parser.add_argument("--replicas", nargs="*", type=int, default=[0],
                        help="last-bit replicas of each life, 0 for the newborn itself")
    parser.add_argument("--batches", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=1,
                        help="a brain's replays, offline; 0 for the controls' stopping rule, the "
                             "panels read at every epoch")
    parser.add_argument("--epoch-log", type=Path, default=None,
                        help="each epoch's reading as it falls, one line per epoch")
    parser.add_argument("--verb-heads", action="store_true",
                        help="each head's read at the verb, per distractor count, at a life's end")
    parser.add_argument("--rate", type=float, default=0.0, help="Adam's (default: the controls')")
    parser.add_argument("--check", action="store_true", help="the finite-difference checks")
    parser.add_argument("--settled", action="store_true",
                        help="the audit of how settled the composed tolerance leaves a state")
    parser.add_argument("--crossing", action="store_true",
                        help="the crossing panel of the ablations, at --batches")
    parser.add_argument("--keyed", nargs="*", default=[], choices=["own", "go"],
                        help="#228's keyed crossing at these timings, at --batches")
    parser.add_argument("--resume", action="store_true",
                        help="skip the lives the results file already holds")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        results = [run_check({"founder": f}) for f in args.founders]
        args.out.write_text(json.dumps(results, indent=1))
        return 0
    seeds: list[dict[str, Any]] = [{} if s < 0 else {"init_seed": s}
                                   for s in args.init_seeds] or [{}]
    specs = [{"arm": arm, "founder": f, "mode": mode, "batches": args.batches, "replica": replica,
              "epochs": args.epochs, "rate": args.rate, "verb_heads": args.verb_heads,
              "epoch_log": str(args.epoch_log) if args.epoch_log else None, **seed}
             for seed in seeds for replica in args.replicas for arm in args.arms
             for mode in args.modes for f in args.founders]
    runner: Any = run_crossing if args.crossing else run_settled if args.settled else run_life
    if args.keyed:
        runner = run_keyed
        specs = [{**spec, "timing": timing, "mode": "keyed_" + timing}
                 for timing in args.keyed for spec in specs]
    if args.resume:
        specs = pending(specs, args.out)
    if not specs:
        return 0
    with args.out.open("a") as sink:
        lives: Any = map(runner, specs)
        if args.workers > 1:
            pool = get_context("spawn").Pool(min(args.workers, len(specs)))
            lives = pool.imap_unordered(runner, specs, chunksize=1)
        for life in lives:
            sink.write(json.dumps(life, default=float) + "\n")
            sink.flush()
            if args.keyed:
                r = life["readings"]["intact"]
                print(f"{life['row']} {life['mode']} f{life['founder']}/{life['init_seed']} right "
                      f"{r['right']:.2f} relevant {r['relevant']:.2f} other {r['other']:.2f} "
                      f"slots {r['slots']:.1f} {life['seconds']}s", flush=True)
                continue
            panels = life["readings"][0]["panels"]
            said = " ".join(f"{k} {panels[k]['exact']:.2f}" for k in
                            ("clause", "clause_held", "dyck", "path", "crossing") if k in panels)
            print(f"{life['row']} {life['mode']} f{life['founder']}/{life['init_seed']}"
                  f"/r{life['replica']} {said}"
                  f" refused {life['refused_reads']} {life['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
