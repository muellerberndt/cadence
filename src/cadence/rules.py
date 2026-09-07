"""Owner rules: what one owner does with what arrives over its overlaps.

The graded rule is the one every connectome lane so far has used. An owner
holds a potential ``v`` and publishes an activation ``s(v)``; each step it
moves ``v`` toward the sum of its inbox, its clamp, and its bias:

    v <- v + dt * ( -v + inbox + clamp + bias - strength * a )
    inbox = sum over inbound overlaps of  gain * count * sign * exp(log_gain[pre]) * s[pre]
    s(v) = rectified sigmoid, exactly zero at rest

With ``leak`` above zero an owner below rest publishes a small negative
activation, ``-leak`` at most, so the rule that learns can still tell a
silent owner from a dead one; rest stays an exact zero. Connectome lanes
leave it at zero. ``a`` is an optional adaptation variable, one per owner, following its own
activation with a slow time constant. It is what lets a wiring with mutual
inhibition produce rhythm instead of a fixed point. It is off by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = ["Adaptation", "GradedRule"]


@dataclass(frozen=True, slots=True)
class Adaptation:
    """Spike-frequency adaptation: ``a <- a + (s - a) / tau_steps``; drive loses ``strength * a``.

    Owner-local: an owner reads only its own ``a``.
    """

    tau_steps: float = 50.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.tau_steps <= 0 or self.strength < 0:
            raise ValueError("tau_steps must be positive and strength nonnegative")


@dataclass(frozen=True, slots=True)
class GradedRule:
    dt: float = 0.2
    slope: float = 4.0
    threshold: float = 1.5
    gain: float = 0.02
    clamp_amplitude: float = 3.0
    adaptation: Adaptation | None = None
    leak: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.dt <= 1:
            raise ValueError("dt must lie in (0, 1]")
        if self.slope <= 0 or self.gain <= 0:
            raise ValueError("slope and gain must be positive")
        if not 0 <= self.leak <= 1:
            raise ValueError("leak must lie in [0, 1]")

    @property
    def rest_emission(self) -> float:
        """Activation the raw sigmoid would emit at v = 0; subtracted so rest emits nothing."""
        return float(1.0 / (1.0 + np.exp(self.slope * self.threshold)))

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
        """Derivative of the activation, for reports on how responsive owners are."""
        rest = self.rest_emission
        sig = 1.0 / (1.0 + np.exp(-self.slope * (v - self.threshold)))
        d = self.slope * sig * (1.0 - sig)
        return np.where(sig > rest, d / (1.0 - rest), self.leak * d / rest)

    def replace(self, **changes: Any) -> GradedRule:
        return GradedRule(**{**asdict(self), **changes})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["adaptation"] = None if self.adaptation is None else asdict(self.adaptation)
        return out
