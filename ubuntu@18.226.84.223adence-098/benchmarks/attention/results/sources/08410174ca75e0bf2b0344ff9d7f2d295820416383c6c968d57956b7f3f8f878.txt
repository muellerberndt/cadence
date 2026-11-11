"""The neuron-by-neuron reference and its transmission ledger.

Everything in ``Brain`` is a vectorized scatter. This module is the
slow, literal version: neurons are rows, a transport carries one transmission
per declared synapse into a synaptic input, and every neuron updates from its own
row and its synaptic input slice alone. A ledger counts transmissions. Comparing a
backend against this reference checks that it settles by neuron-local dynamics
only: if the fast path computed anything the neurons could not, the two would
diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .brain import Brain
from .connectome import Connectome
from .neuron import NeuronModel

__all__ = ["Ledger", "settle_neuron_by_neuron", "conformance"]


@dataclass
class Ledger:
    declared_synapses: int
    steps: int = 0
    transmissions: int = 0
    undeclared: int = 0

    @property
    def clean(self) -> bool:
        return self.undeclared == 0 and self.transmissions == self.steps * self.declared_synapses

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_synapses": self.declared_synapses,
            "steps": self.steps,
            "transmissions": self.transmissions,
            "undeclared_transmissions": self.undeclared,
            "clean": self.clean,
        }


def settle_neuron_by_neuron(
    connectome: Connectome,
    neuron_model: NeuronModel,
    stimulus: np.ndarray,
    *,
    steps: int,
    log_gain: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    efficacy: np.ndarray | None = None,
) -> tuple[np.ndarray, Ledger]:
    """Activation trajectory ``(steps, n)`` and the ledger; no neuron reads a global state."""
    n = connectome.n
    lg = np.zeros(n) if log_gain is None else np.asarray(log_gain, float)
    b = np.zeros(n) if bias is None else np.asarray(bias, float)
    scale = connectome.sign if efficacy is None else np.asarray(efficacy, float)
    weight = neuron_model.gain * connectome.count * scale * np.exp(lg[connectome.pre])
    starts = np.searchsorted(connectome.post, np.arange(n), side="left")
    stops = np.searchsorted(connectome.post, np.arange(n), side="right")
    ledger = Ledger(connectome.synapses)
    adapt = neuron_model.adaptation
    cell = np.zeros(n)
    adaptation = np.zeros(n)
    published = np.zeros(n)
    trajectory = np.zeros((steps, n))
    for t in range(steps):
        messages = published[connectome.pre] * weight  # the transport: one transmission per synapse
        ledger.steps += 1
        ledger.transmissions += len(messages)
        new_cell = np.empty_like(cell)
        for neuron in range(n):  # each neuron reads its own row and its synaptic input slice only
            synaptic_input = messages[starts[neuron] : stops[neuron]]
            total = float(synaptic_input.sum()) + float(stimulus[neuron]) + float(b[neuron])
            if adapt is not None:
                total -= adapt.strength * adaptation[neuron]
            new_cell[neuron] = cell[neuron] + neuron_model.dt * (-cell[neuron] + total)
        cell = new_cell
        published = neuron_model.activation(cell)
        if adapt is not None:
            adaptation += (published - adaptation) / adapt.tau_steps
        trajectory[t] = published
    return trajectory, ledger


def conformance(brain: Brain, stimulus: Any, *, steps: int = 60) -> dict[str, Any]:
    """Compare ``brain`` with the neuron-by-neuron reference on the same stimulus."""
    drive = brain.stimulus_vector(stimulus)
    state = brain.settle(drive, steps=steps, trajectory=True)
    assert state.trajectory is not None
    reference, ledger = settle_neuron_by_neuron(
        brain.connectome,
        brain.neuron_model,
        drive,
        steps=steps,
        log_gain=brain.log_gain,
        bias=brain.bias,
        efficacy=brain.efficacy,
    )
    deviation = float(np.abs(reference - state.trajectory).max(initial=0))
    return {
        "backend": brain.backend,
        "steps": steps,
        "max_abs_deviation": deviation,
        "ledger": ledger.to_dict(),
        "final_active": int((state.activation >= 0.5).sum()),
    }
