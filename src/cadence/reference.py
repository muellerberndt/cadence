"""The owner-by-owner reference engine and its message ledger.

Everything in ``Settlement`` is a vectorized scatter. This module is the
slow, literal version: owners are rows, a transport delivers one message
per declared overlap into an inbox, and every owner repairs from its own
row and its inbox slice alone. A ledger counts deliveries. Comparing a
backend against this engine is how a lane certifies that it settles by
patch-net dynamics only: if the fast path computed anything the owners
could not, the two would part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rules import GradedRule
from .settle import Settlement
from .wiring import Wiring

__all__ = ["Ledger", "settle_owner_by_owner", "conformance"]


@dataclass
class Ledger:
    declared_overlaps: int
    steps: int = 0
    deliveries: int = 0
    undeclared: int = 0

    @property
    def clean(self) -> bool:
        return self.undeclared == 0 and self.deliveries == self.steps * self.declared_overlaps

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_overlaps": self.declared_overlaps,
            "steps": self.steps,
            "deliveries": self.deliveries,
            "undeclared_deliveries": self.undeclared,
            "clean": self.clean,
        }


def settle_owner_by_owner(
    wiring: Wiring,
    rule: GradedRule,
    clamp: np.ndarray,
    *,
    steps: int,
    log_gain: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    edge_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, Ledger]:
    """Activation trajectory ``(steps, n)`` and the ledger; no owner reads a global state vector."""
    n = wiring.n
    lg = np.zeros(n) if log_gain is None else np.asarray(log_gain, float)
    b = np.zeros(n) if bias is None else np.asarray(bias, float)
    scale = wiring.sign if edge_scale is None else np.asarray(edge_scale, float)
    weight = rule.gain * wiring.count * scale * np.exp(lg[wiring.pre])
    starts = np.searchsorted(wiring.post, np.arange(n), side="left")
    stops = np.searchsorted(wiring.post, np.arange(n), side="right")
    ledger = Ledger(wiring.edges)
    adapt = rule.adaptation
    cell = np.zeros(n)
    adaptation = np.zeros(n)
    published = np.zeros(n)
    trajectory = np.zeros((steps, n))
    for t in range(steps):
        messages = published[wiring.pre] * weight  # the transport: one delivery per overlap
        ledger.steps += 1
        ledger.deliveries += len(messages)
        new_cell = np.empty_like(cell)
        for owner in range(n):  # each owner reads its own row and its inbox slice only
            inbox = messages[starts[owner] : stops[owner]]
            total = float(inbox.sum()) + float(clamp[owner]) + float(b[owner])
            if adapt is not None:
                total -= adapt.strength * adaptation[owner]
            new_cell[owner] = cell[owner] + rule.dt * (-cell[owner] + total)
        cell = new_cell
        published = rule.activation(cell)
        if adapt is not None:
            adaptation += (published - adaptation) / adapt.tau_steps
        trajectory[t] = published
    return trajectory, ledger


def conformance(engine: Settlement, clamp: Any, *, steps: int = 60) -> dict[str, Any]:
    """Compare ``engine`` against the owner-by-owner reference on the same wiring and clamp."""
    drive = engine.clamp_vector(clamp)
    state = engine.settle(drive, steps=steps, trajectory=True)
    assert state.trajectory is not None
    reference, ledger = settle_owner_by_owner(
        engine.wiring,
        engine.rule,
        drive,
        steps=steps,
        log_gain=engine.log_gain,
        bias=engine.bias,
        edge_scale=engine.edge_scale,
    )
    deviation = float(np.abs(reference - state.trajectory).max())
    return {
        "backend": engine.backend,
        "steps": steps,
        "max_abs_deviation": deviation,
        "ledger": ledger.to_dict(),
        "final_active": int((state.activation >= 0.5).sum()),
    }
