"""The settlement engine.

One ``Settlement`` holds a wiring and a rule and runs the owner rule for a
number of steps from rest, or from a given state, under a clamp. Two
backends do the same arithmetic:

* ``"cpu"``: NumPy in float64. The transport is one ``bincount`` per step,
  which is a scatter of every overlap's message into its owner's inbox.
  This is the receipt-grade backend: deterministic, exact to rounding, and
  the one the owner-by-owner reference engine is compared against.
* ``"torch"``: the same scatter with ``index_add_`` on whatever device
  torch offers, CUDA in float64, Apple silicon in float32 (MPS has no
  float64), CPU in float64. Use it for interactive work and for very large
  wirings, and keep a conformance check against ``"cpu"``; the fly brain
  has owners on knife edges where float32 flips a bistable readout.

Per-owner parameters, a ``log_gain`` on each owner's outgoing overlaps and a
``bias``, default to zero. They are how a lane declares a stimulus rate
(fold ``log(rate)`` into the clamped owners' log-gain) or a region's gain
(fold ``log(region_gain / gain)`` into that region's owners).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .rules import GradedRule
from .wiring import Wiring

__all__ = ["Settlement", "SettledState", "available_backends", "Backend"]

Backend = Literal["cpu", "torch"]


def available_backends() -> dict[str, str]:
    """Backends importable here, with the device each would use."""
    out = {"cpu": "numpy float64"}
    try:
        import torch

        if torch.cuda.is_available():
            out["torch"] = "cuda float64"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            out["torch"] = "mps float32"
        else:
            out["torch"] = "cpu float64"
    except ImportError:
        pass
    return out


@dataclass(frozen=True)
class SettledState:
    """What the net came to rest in: potentials, activations, adaptation, optional trajectory."""

    v: np.ndarray
    activation: np.ndarray
    adaptation: np.ndarray
    steps: int
    trajectory: np.ndarray | None = None  # (steps, n) activations when requested

    def mean(self, members: Sequence[int]) -> float:
        return float(self.activation[list(members)].mean()) if len(members) else 0.0

    def fraction_active(self, members: Sequence[int] | None = None, level: float = 0.5) -> float:
        values = self.activation if members is None else self.activation[list(members)]
        return float((values >= level).mean()) if len(values) else 0.0

    def active(self, level: float = 0.5) -> int:
        return int((self.activation >= level).sum())


class Settlement:
    def __init__(
        self,
        wiring: Wiring,
        rule: GradedRule,
        *,
        backend: Backend = "cpu",
        log_gain: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        device: str | None = None,
    ) -> None:
        self.wiring = wiring
        self.rule = rule
        self.backend: Backend = backend
        self.log_gain = np.zeros(wiring.n) if log_gain is None else np.asarray(log_gain, float)
        self.bias = np.zeros(wiring.n) if bias is None else np.asarray(bias, float)
        if self.log_gain.shape != (wiring.n,) or self.bias.shape != (wiring.n,):
            raise ValueError("log_gain and bias must have one entry per owner")
        self._weights: np.ndarray = (
            rule.gain * wiring.count * wiring.sign * np.exp(self.log_gain[wiring.pre])
        )
        self._torch: Any = None
        if backend == "torch":
            self._torch = _TorchKernel(wiring, self._weights, self.bias, rule, device)
        elif backend != "cpu":
            raise ValueError(f"unknown backend {backend!r}")

    # -- parameters

    def with_parameters(
        self, *, log_gain: np.ndarray | None = None, bias: np.ndarray | None = None
    ) -> Settlement:
        return Settlement(
            self.wiring,
            self.rule,
            backend=self.backend,
            log_gain=self.log_gain if log_gain is None else log_gain,
            bias=self.bias if bias is None else bias,
        )

    @property
    def weights(self) -> np.ndarray:
        """Effective drive per unit presynaptic activation, one per overlap."""
        return self._weights

    # -- clamps

    def clamp_vector(
        self, clamp: Mapping[int, float] | Sequence[int] | np.ndarray | None
    ) -> np.ndarray:
        """A dense drive vector, a list of owners at full amplitude, or a {owner: level} map."""
        out = np.zeros(self.wiring.n)
        if clamp is None:
            return out
        if isinstance(clamp, Mapping):
            for owner, level in clamp.items():
                out[int(owner)] = max(out[int(owner)], self.rule.clamp_amplitude * float(level))
            return out
        array = np.asarray(clamp)
        if (
            array.dtype.kind in "iu"
            and array.ndim == 1
            and (array.shape[0] != self.wiring.n or array.max(initial=0) > 1)
        ):
            out[array] = self.rule.clamp_amplitude
            return out
        if array.shape == (self.wiring.n,):
            return array.astype(float)
        out[array.astype(np.int64)] = self.rule.clamp_amplitude
        return out

    # -- settlement

    def settle(
        self,
        clamp: Mapping[int, float] | Sequence[int] | np.ndarray | None = None,
        *,
        steps: int = 60,
        state: SettledState | None = None,
        mask: np.ndarray | None = None,
        trajectory: bool = False,
    ) -> SettledState:
        """Run the owner rule for ``steps`` steps from rest, or from ``state``.

        ``mask`` zeros ablated owners. The trajectory, when requested, holds
        the activation after every step.
        """
        drive = self.clamp_vector(clamp)
        keep = np.ones(self.wiring.n) if mask is None else np.asarray(mask, float)
        v = np.zeros(self.wiring.n) if state is None else state.v.copy()
        a = np.zeros(self.wiring.n) if state is None else state.adaptation.copy()
        if self.backend == "torch":
            v, a, s, traj = self._torch.run(v, a, drive, keep, steps, trajectory)
        else:
            v, a, s, traj = self._run_numpy(v, a, drive, keep, steps, trajectory)
        return SettledState(v=v, activation=s, adaptation=a, steps=steps, trajectory=traj)

    def _run_numpy(
        self,
        v: np.ndarray,
        a: np.ndarray,
        drive: np.ndarray,
        keep: np.ndarray,
        steps: int,
        want: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        rule, w = self.rule, self._weights
        pre, post, n = self.wiring.pre, self.wiring.post, self.wiring.n
        adapt = rule.adaptation
        traj = np.zeros((steps, n)) if want else None
        s = rule.activation(v) * keep
        for t in range(steps):
            inbox = np.bincount(post, weights=w * s[pre], minlength=n)  # transport, one scatter
            total = inbox + drive + self.bias
            if adapt is not None:
                total = total - adapt.strength * a
            v = (v + rule.dt * (-v + total)) * keep  # owner-local repair
            s = rule.activation(v) * keep
            if adapt is not None:
                a = a + (s - a) / adapt.tau_steps
            if traj is not None:
                traj[t] = s
        return v, a, s, traj

    def readings(self, state: SettledState, names: Sequence[str]) -> dict[str, dict[str, float]]:
        """Mean and fraction-active of named sets."""
        return {
            name: {
                "mean": state.mean(self.wiring.sets[name]),
                "fraction": state.fraction_active(self.wiring.sets[name]),
            }
            for name in names
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "wiring": self.wiring.summary(),
            "rule": self.rule.to_dict(),
            "backend": self.backend,
            "log_gain_nonzero": int((self.log_gain != 0).sum()),
            "bias_nonzero": int((self.bias != 0).sum()),
        }


class _TorchKernel:
    """Gather-scatter settlement on a torch device; float32 on MPS, float64 elsewhere."""

    def __init__(
        self,
        wiring: Wiring,
        weights: np.ndarray,
        bias: np.ndarray,
        rule: GradedRule,
        device: str | None,
    ) -> None:
        import torch

        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "mps" else torch.float64
        self.pre = torch.from_numpy(wiring.pre).to(self.device)
        self.post = torch.from_numpy(wiring.post).to(self.device)
        self.w = torch.from_numpy(weights).to(self.device, self.dtype)
        self.bias = torch.from_numpy(bias).to(self.device, self.dtype)
        self.rule = rule
        self.n = wiring.n

    def _activation(self, v: Any) -> Any:
        torch, rule = self.torch, self.rule
        rest = torch.tensor(rule.rest_emission, dtype=self.dtype, device=self.device)
        s = torch.relu(torch.sigmoid(rule.slope * (v - rule.threshold)) - rest)
        return s / (1.0 - rest)

    def run(
        self,
        v0: np.ndarray,
        a0: np.ndarray,
        drive: np.ndarray,
        keep: np.ndarray,
        steps: int,
        want: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        torch, rule = self.torch, self.rule

        def to(x: np.ndarray) -> Any:
            return torch.from_numpy(np.asarray(x, float)).to(self.device, self.dtype)

        v, a, d, k = to(v0), to(a0), to(drive), to(keep)
        adapt = rule.adaptation
        traj = []
        with torch.no_grad():
            s = self._activation(v) * k
            for _ in range(steps):
                inbox = torch.zeros(self.n, dtype=self.dtype, device=self.device).index_add_(
                    0, self.post, self.w * s[self.pre]
                )
                total = inbox + d + self.bias
                if adapt is not None:
                    total = total - adapt.strength * a
                v = (v + rule.dt * (-v + total)) * k
                s = self._activation(v) * k
                if adapt is not None:
                    a = a + (s - a) / adapt.tau_steps
                if want:
                    traj.append(s.detach().cpu().double().numpy())
        return (
            v.cpu().double().numpy(),
            a.cpu().double().numpy(),
            s.cpu().double().numpy(),
            np.stack(traj) if want else None,
        )
