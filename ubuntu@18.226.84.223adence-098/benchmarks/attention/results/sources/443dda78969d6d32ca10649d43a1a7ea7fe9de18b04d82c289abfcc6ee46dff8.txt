"""The neuron model: what one neuron does with what arrives over its synapses.

Every neuron of a ``Brain`` runs this graded rate model. A neuron
holds a potential ``v`` and publishes an activation ``s(v)``; each step it
moves ``v`` toward the sum of its synaptic input, its stimulus, and its bias:

    v <- v + dt * ( -v + synaptic input + stimulus + bias - strength * a )
    synaptic input = sum over inbound synapses of  gain * count * sign * exp(log_gain[pre]) * s[pre]
    s(v) = rectified sigmoid, exactly zero at rest

With ``leak`` above zero a neuron below rest publishes a small negative
activation, ``-leak`` at most, so the learning rule can tell a
silent neuron from a dead one; rest stays an exact zero. ``NeuronModel``
defaults ``leak`` to zero; ``learning_neuron_model``, which ``Brain`` uses, sets
0.1. ``a`` is an optional adaptation variable, one per neuron, following its own
activation with a slow time constant. It is what lets a connectome with mutual
inhibition produce rhythm instead of a fixed point. It is off by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

__all__ = ["Adaptation", "NeuronModel"]


@dataclass(frozen=True, slots=True)
class Adaptation:
    """Spike-frequency adaptation: ``a <- a + (s - a) / tau_steps``; drive loses ``strength * a``.

    Neuron-local: a neuron reads only its own ``a``.
    """

    tau_steps: float = 50.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.tau_steps, self.strength]).all():
            raise ValueError("adaptation parameters must be finite")
        if self.tau_steps <= 0 or self.strength < 0:
            raise ValueError("tau_steps must be positive and strength nonnegative")


@dataclass(frozen=True, slots=True)
class NeuronModel:
    dt: float = 0.2
    slope: float = 4.0
    threshold: float = 1.5
    gain: float = 0.02
    stimulus_amplitude: float = 3.0
    adaptation: Adaptation | None = None
    leak: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(
            [self.dt, self.slope, self.threshold, self.gain, self.stimulus_amplitude, self.leak]
        ).all():
            raise ValueError("rule parameters must be finite")
        if not 0 < self.dt <= 1:
            raise ValueError("dt must lie in (0, 1]")
        if self.slope <= 0 or self.gain <= 0:
            raise ValueError("slope and gain must be positive")
        if not 0 <= self.leak <= 1:
            raise ValueError("leak must lie in [0, 1]")
        with np.errstate(over="ignore"):
            rest = self.rest_emission
        if not 0 < rest < 1:
            raise ValueError(
                "slope and threshold saturate the resting sigmoid; reduce their product's magnitude"
            )

    @property
    def rest_emission(self) -> float:
        """Activation the raw sigmoid would emit at v = 0; subtracted so rest emits nothing."""
        return float(1.0 / (1.0 + np.exp(self.slope * self.threshold)))

    def _validate_precision(self, dtype: str) -> None:
        """Reject a rule whose rebasing constants fail in a backend's precision."""
        rest = self.rest_emission
        values = [self.dt, self.slope, self.threshold, rest,
                  1.0 / (1.0 - rest), self.leak / rest]
        if self.adaptation is not None:
            values.extend([self.adaptation.tau_steps, self.adaptation.strength,
                           1.0 / self.adaptation.tau_steps])
        with np.errstate(over="ignore", under="ignore"):
            cast = np.asarray(values, dtype=dtype)
        if not np.isfinite(cast).all() or cast[0] <= 0 or cast[1] <= 0 or not 0 < cast[3] < 1:
            raise ValueError("neuron rule is not representable at runtime precision")

    def activation(self, v: np.ndarray) -> np.ndarray:
        rest = self.rest_emission
        r = np.exp((-self.slope) * (v - self.threshold))
        r += 1.0
        np.reciprocal(r, out=r)
        r -= rest  # zero at rest, in the array's own precision
        if self.leak == 0.0:
            np.maximum(r, 0.0, out=r)
            r *= 1.0 / (1.0 - rest)
            return r
        return np.where(r > 0.0, r * (1.0 / (1.0 - rest)), r * (self.leak / rest))

    def slope_at(self, v: np.ndarray) -> np.ndarray:
        """Derivative of the activation, for reports on how responsive neurons are."""
        rest = self.rest_emission
        sig = 1.0 / (1.0 + np.exp(-self.slope * (v - self.threshold)))
        d = self.slope * sig * (1.0 - sig)
        return np.where(sig > rest, d / (1.0 - rest), self.leak * d / rest)

    def replace(self, **changes: Any) -> NeuronModel:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["adaptation"] = None if self.adaptation is None else asdict(self.adaptation)
        return out
