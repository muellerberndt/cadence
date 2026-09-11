"""The settlement engine.

One ``Settlement`` holds a wiring and a rule and runs the owner rule for a
number of steps from rest, or from a given state, under a clamp. Two
backends do the same arithmetic:

* ``"cpu"``: NumPy in float64. The transport is one segmented sum per
  step, a scatter of every overlap's message into its owner's inbox; for
  wirings whose dense blocks fit (at most ``dense_limit`` squared entries,
  see ``cadence.blocks``) the same sum is done as block matrix products,
  reusing the product of every range of owners that did not move since
  the previous step, which is far faster when the interpreter overhead of
  the scatter would dominate and lets a settlement step cost what its
  moving owners cost rather than what the whole net does. This is the
  receipt-grade backend: deterministic, exact to rounding, and the one the
  owner-by-owner reference engine is compared against.
* ``"torch"``: the same scatter with ``index_add_`` on whatever device
  torch offers, CUDA in float64, Apple silicon in float32 (MPS has no
  float64), CPU in float64. Use it for interactive work and for very large
  wirings, and keep a conformance check against ``"cpu"``; the fly brain
  has owners on knife edges where float32 flips a bistable readout.

Three kinds of parameter sit on a settlement, all defaulting to the wiring:
``edge_scale``, one signed number per overlap (the wiring's sign unless a
learner has moved it); ``log_gain``, one per owner on its outgoing
overlaps, how a lane declares a stimulus rate or a region's gain; and
``bias``, one per owner. The effective drive of overlap ``e`` per unit
presynaptic activation is ``gain * count[e] * edge_scale[e] *
exp(log_gain[pre[e]])``.

Settlements run on a batch of clamps at once; ``settle`` is the one-clamp
convenience. A ``Nudge`` adds a drive that pulls chosen owners toward a
target activation, which is how the learning rule's second phase enters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .blocks import BlockTransport, Layout
from .blocks import layout as make_layout
from .rules import GradedRule
from .wiring import Wiring

__all__ = ["Settlement", "SettledState", "Nudge", "available_backends", "Backend"]

Backend = Literal["cpu", "torch"]


def _fused_available() -> bool:
    import os

    if os.environ.get("CADENCE_FUSED", "1") == "0":
        return False
    try:
        from .fused import available
    except ImportError:
        return False
    return available()


_FUSED = (
    _fused_available()
)  # the compiled dense kernel, identical arithmetic; CADENCE_FUSED=0 forces the NumPy loop


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
class Nudge:
    """Extra drive ``beta * (target - s)`` on the owners where ``mask`` is one.

    With ``softmax_temperature`` set, the owners under the mask compete: the
    drive is ``beta * (target - softmax(s / T))`` over that group, a
    cross-entropy nudge, so pushing one owner up pushes the others down only
    as much as their share. ``weight``, one number per batch row, scales the
    nudge row by row; a negative weight pushes away from the target. That is
    how a reward enters: the target is the action taken and the weight its
    advantage.
    """

    target: np.ndarray  # (batch, n) or (n,)
    mask: np.ndarray  # (n,)
    beta: float
    softmax_temperature: float | None = None
    weight: np.ndarray | None = None  # (batch,)
    groups: np.ndarray | None = None  # (n,) group id per owner, -1 for none: one softmax per group

    def drive(self, s: np.ndarray) -> np.ndarray:
        if self.softmax_temperature is None:
            out = np.asarray(self.beta * (self.target - s) * self.mask, dtype=float)
        else:
            out = np.zeros_like(s)
            full_target = np.broadcast_to(self.target, s.shape)
            if self.groups is None:
                members = [np.flatnonzero(self.mask > 0)]
            else:
                ids = np.asarray(self.groups)
                members = [
                    np.flatnonzero((ids == g) & (self.mask > 0)) for g in np.unique(ids[ids >= 0])
                ]
            for group in members:
                z = s[:, group] / self.softmax_temperature
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p = p / p.sum(axis=1, keepdims=True)
                out[:, group] = self.beta * (full_target[:, group] - p)
        if self.weight is not None:
            out = out * np.asarray(self.weight, dtype=float)[:, None]
        return out


@dataclass(frozen=True)
class SettledState:
    """What the net came to rest in: potentials, activations, adaptation, optional trajectory.

    From ``settle`` every array is ``(n,)`` and ``trajectory`` is ``(steps, n)``;
    from ``settle_batch`` they carry a leading batch axis.
    """

    v: np.ndarray
    activation: np.ndarray
    adaptation: np.ndarray
    steps: int
    trajectory: np.ndarray | None = None

    @property
    def batched(self) -> bool:
        return self.v.ndim == 2

    def row(self, i: int = 0) -> np.ndarray:
        return self.activation[i] if self.batched else self.activation

    def mean(self, members: Sequence[int], i: int = 0) -> float:
        values = self.row(i)
        return float(values[list(members)].mean()) if len(members) else 0.0

    def fraction_active(
        self, members: Sequence[int] | None = None, level: float = 0.5, i: int = 0
    ) -> float:
        values = self.row(i) if members is None else self.row(i)[list(members)]
        return float((values >= level).mean()) if len(values) else 0.0

    def active(self, level: float = 0.5, i: int = 0) -> int:
        return int((self.row(i) >= level).sum())


class Settlement:
    def __init__(
        self,
        wiring: Wiring,
        rule: GradedRule,
        *,
        backend: Backend = "cpu",
        edge_scale: np.ndarray | None = None,
        log_gain: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        device: str | None = None,
        dense_limit: int = 2048,
        layout: Layout | None = None,
        precision: str | None = None,
    ) -> None:
        self.wiring = wiring
        self.rule = rule
        self.backend: Backend = backend
        self.dense_limit = dense_limit
        self.precision = precision  # torch only: "float64" or "float32"; default by device
        self.layout: Layout = make_layout(wiring) if layout is None else layout
        self.edge_scale = (
            wiring.sign.copy() if edge_scale is None else np.asarray(edge_scale, float).copy()
        )
        self.log_gain = np.zeros(wiring.n) if log_gain is None else np.asarray(log_gain, float)
        self.bias = np.zeros(wiring.n) if bias is None else np.asarray(bias, float)
        if self.edge_scale.shape != (wiring.edges,):
            raise ValueError("edge_scale must have one entry per overlap")
        if self.log_gain.shape != (wiring.n,) or self.bias.shape != (wiring.n,):
            raise ValueError("log_gain and bias must have one entry per owner")
        self._weights: np.ndarray = (
            rule.gain * wiring.count * self.edge_scale * np.exp(self.log_gain)[wiring.pre]
        )
        # Segment boundaries of the (post-sorted) overlap arrays, for the segmented sum.
        post = wiring.post
        if wiring.edges:
            change = np.flatnonzero(np.diff(post)) + 1
            self._starts = np.concatenate([[0], change]).astype(np.int64)
            self._owners_with_inbox = post[self._starts]
        else:
            self._starts = np.zeros(0, np.int64)
            self._owners_with_inbox = np.zeros(0, np.int64)
        # The block transport: dense blocks between owner ranges, when they fit.
        blocked = self.layout.size <= dense_limit * dense_limit
        self._blocks: np.ndarray | None = self.layout.flat(self._weights) if blocked else None
        self._torch: Any = None
        if backend == "torch":
            self._torch = _TorchKernel(
                wiring,
                self._weights,
                self.bias,
                rule,
                device,
                self.layout if blocked else None,
                precision,
            )
        elif backend != "cpu":
            raise ValueError(f"unknown backend {backend!r}")

    # -- parameters

    def with_parameters(
        self,
        *,
        edge_scale: np.ndarray | None = None,
        log_gain: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> Settlement:
        """A new settlement on the same wiring with some parameters replaced."""
        return Settlement(
            self.wiring,
            self.rule,
            backend=self.backend,
            edge_scale=self.edge_scale if edge_scale is None else edge_scale,
            log_gain=self.log_gain if log_gain is None else log_gain,
            bias=self.bias if bias is None else bias,
            dense_limit=self.dense_limit,
            layout=self.layout,
            precision=self.precision,
        )

    @property
    def weights(self) -> np.ndarray:
        """Effective drive per unit presynaptic activation, one per overlap."""
        return self._weights

    def dense(self) -> np.ndarray:
        """The overlap matrix ``W[pre, post]``: the inbox of a batch ``s`` is ``s @ W``.

        Built on demand; this is what a page that settles the net in a browser embeds.
        """
        dense = np.zeros((self.wiring.n, self.wiring.n))
        np.add.at(dense, (self.wiring.pre, self.wiring.post), self._weights)
        return dense

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

    def clamp_levels(self, levels: np.ndarray) -> np.ndarray:
        """Dense drive from per-owner levels in [0, 1], batched or not."""
        return np.asarray(levels, float) * self.rule.clamp_amplitude

    # -- settlement

    def settle(
        self,
        clamp: Mapping[int, float] | Sequence[int] | np.ndarray | None = None,
        *,
        steps: int = 60,
        state: SettledState | None = None,
        mask: np.ndarray | None = None,
        trajectory: bool = False,
        nudge: Nudge | None = None,
        tolerance: float | None = None,
    ) -> SettledState:
        """Run the owner rule for ``steps`` steps from rest, or from ``state``, under one clamp.

        ``mask`` zeros ablated owners. The trajectory, when requested, holds
        the activation after every step. With ``tolerance`` set the run stops
        early once no owner's activation moved more than that in a step;
        ``steps`` is then the most it will run, and the state reports how
        many steps it took.
        """
        drive = self.clamp_vector(clamp)[None, :]
        out = self.settle_batch(
            drive,
            steps=steps,
            state=state,
            mask=mask,
            trajectory=trajectory,
            nudge=nudge,
            tolerance=tolerance,
        )
        return SettledState(
            v=out.v[0],
            activation=out.activation[0],
            adaptation=out.adaptation[0],
            steps=out.steps,
            trajectory=None if out.trajectory is None else out.trajectory[:, 0, :],
        )

    def settle_batch(
        self,
        drive: np.ndarray,
        *,
        steps: int = 60,
        state: SettledState | None = None,
        mask: np.ndarray | None = None,
        trajectory: bool = False,
        nudge: Nudge | None = None,
        tolerance: float | None = None,
    ) -> SettledState:
        """Settle a batch of dense drives ``(batch, n)`` at once; see ``settle``."""
        drive = np.asarray(drive, float)
        if drive.ndim == 1:
            drive = drive[None, :]
        batch, n = drive.shape
        if n != self.wiring.n:
            raise ValueError("drive must have one column per owner")
        keep = np.ones(n) if mask is None else np.asarray(mask, float)
        v = np.zeros((batch, n)) if state is None else np.atleast_2d(state.v).copy()
        a = np.zeros((batch, n)) if state is None else np.atleast_2d(state.adaptation).copy()
        if v.shape != (batch, n):
            raise ValueError("state batch does not match the drive batch")
        if self.backend == "torch":
            v, a, s, traj, taken = self._torch.run(
                v, a, drive, keep, steps, trajectory, nudge, tolerance
            )
        elif self._blocks is not None and not trajectory and _FUSED:
            from .fused import fused_settle

            s, taken = fused_settle(
                v,
                a,
                drive,
                self.bias,
                self.layout,
                self._blocks,
                keep,
                self.rule,
                nudge,
                steps,
                tolerance,
                activation=None if state is None else np.atleast_2d(state.activation),
            )
            traj = None
        else:
            v, a, s, traj, taken = self._run_numpy(
                v, a, drive, keep, steps, trajectory, nudge, tolerance
            )
        return SettledState(v=v, activation=s, adaptation=a, steps=taken, trajectory=traj)

    def _inbox(self, s: np.ndarray, blocks: BlockTransport | None = None) -> np.ndarray:
        """Transport: one segmented sum of every overlap's message into its owner's inbox."""
        if blocks is not None:
            return blocks.inbox(s)
        inbox = np.zeros_like(s)
        if self.wiring.edges:
            messages = s[:, self.wiring.pre] * self._weights
            inbox[:, self._owners_with_inbox] = np.add.reduceat(messages, self._starts, axis=1)
        return inbox

    def _run_numpy(
        self,
        v: np.ndarray,
        a: np.ndarray,
        drive: np.ndarray,
        keep: np.ndarray,
        steps: int,
        want: bool,
        nudge: Nudge | None,
        tolerance: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
        rule = self.rule
        adapt = rule.adaptation
        traj = np.zeros((steps, *v.shape)) if want else None
        masked = bool((keep != 1.0).any())
        standing = drive + self.bias  # the part of every owner's drive that does not move
        blocks = None if self._blocks is None else BlockTransport(self.layout, self._blocks)
        s = rule.activation(v)
        if masked:
            s *= keep
        taken = 0
        for t in range(steps):
            total = self._inbox(s, blocks)
            total += standing
            if adapt is not None:
                total -= adapt.strength * a
            if nudge is not None:
                total += nudge.drive(s)
            total -= v
            total *= rule.dt
            v = v + total  # owner-local repair: v <- v + dt (-v + total)
            if masked:
                v *= keep
            previous = s
            s = rule.activation(v)
            if masked:
                s *= keep
            if adapt is not None:
                a = a + (s - a) / adapt.tau_steps
            if traj is not None:
                traj[t] = s
            taken = t + 1
            if tolerance is not None and float(np.abs(s - previous).max()) < tolerance:
                break
        if traj is not None:
            traj = traj[:taken]
        return v, a, s, traj, taken

    def readings(
        self, state: SettledState, names: Sequence[str], i: int = 0
    ) -> dict[str, dict[str, float]]:
        """Mean and fraction-active of named sets, for batch row ``i``."""
        return {
            name: {
                "mean": state.mean(self.wiring.sets[name], i),
                "fraction": state.fraction_active(self.wiring.sets[name], i=i),
            }
            for name in names
        }

    def _is_dense(self) -> bool:
        return self._blocks is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "wiring": self.wiring.summary(),
            "rule": self.rule.to_dict(),
            "backend": self.backend,
            "transport": "dense" if self._is_dense() else "segmented",
            "layout": self.layout.to_dict(),
            "edge_scale_changed": int((self.edge_scale != self.wiring.sign).sum()),
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
        layout: Layout | None = None,
        precision: str | None = None,
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
        if precision is None:
            self.dtype = torch.float32 if self.device.type == "mps" else torch.float64
        elif precision in ("float32", "float64"):
            self.dtype = torch.float32 if precision == "float32" else torch.float64
        else:
            raise ValueError("precision must be 'float32' or 'float64'")
        self.pre = torch.from_numpy(wiring.pre).to(self.device)
        self.post = torch.from_numpy(wiring.post).to(self.device)
        self.w = torch.from_numpy(weights).to(self.device, self.dtype)
        self.bias = torch.from_numpy(bias).to(self.device, self.dtype)
        self.rule = rule
        self.n = wiring.n
        self.layout = layout
        self.blocks: list[Any] = []
        if layout is not None:  # block products per step instead of a gather and a scatter
            flat = layout.flat(weights)
            self.blocks = [
                torch.from_numpy(np.ascontiguousarray(block)).to(self.device, self.dtype)
                for block in layout.blocks(flat)
            ]

    def _inbox(self, s: Any, previous: Any, cache: list[Any]) -> Any:
        """The block transport on the device, reusing the products of still ranges."""
        torch, lay = self.torch, self.layout
        assert lay is not None
        out = torch.zeros_like(s)
        moved = [True] * lay.ranges
        if previous is not None:
            for r in lay.sources():
                a0, a1 = int(lay.starts[r]), int(lay.starts[r + 1])
                moved[r] = not torch.equal(s[:, a0:a1], previous[:, a0:a1])
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            if cache[k] is None or moved[int(lay.pair_pre[k])]:
                cache[k] = s[:, a0:a1] @ self.blocks[k]
            out[:, b0:b1] += cache[k]
        return out

    def _activation(self, v: Any) -> Any:
        torch, rule = self.torch, self.rule
        rest = torch.tensor(rule.rest_emission, dtype=self.dtype, device=self.device)
        r = torch.sigmoid(rule.slope * (v - rule.threshold)) - rest
        s = torch.relu(r) / (1.0 - rest)
        if rule.leak:
            s = s + rule.leak * torch.clamp(r, max=0.0) / rest
        return s

    def run(
        self,
        v0: np.ndarray,
        a0: np.ndarray,
        drive: np.ndarray,
        keep: np.ndarray,
        steps: int,
        want: bool,
        nudge: Nudge | None,
        tolerance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
        torch, rule = self.torch, self.rule

        def to(x: np.ndarray) -> Any:
            return torch.from_numpy(np.array(x, dtype=float)).to(self.device, self.dtype)

        v, a, d, k = to(v0), to(a0), to(drive), to(keep)
        batch = v.shape[0]
        adapt = rule.adaptation
        target = mask = weight = None
        group: Any = None
        if nudge is not None:
            target = to(np.broadcast_to(nudge.target, (batch, self.n)))
            mask = to(nudge.mask)
            group = torch.from_numpy(np.flatnonzero(nudge.mask > 0)).to(self.device)
            if nudge.weight is not None:
                weight = to(np.asarray(nudge.weight, dtype=float))[:, None]
        traj = []
        taken = 0
        cache: list[Any] = [None] * (0 if self.layout is None else self.layout.pairs)
        previous = None
        with torch.no_grad():
            s = self._activation(v) * k
            for t in range(steps):
                if self.layout is not None:
                    inbox = self._inbox(s, previous, cache)
                else:
                    inbox = torch.zeros(batch, self.n, dtype=self.dtype, device=self.device)
                    inbox = inbox.index_add_(1, self.post, s[:, self.pre] * self.w)
                total = inbox + d + self.bias
                if adapt is not None:
                    total = total - adapt.strength * a
                if nudge is not None:
                    if nudge.softmax_temperature is None:
                        push = nudge.beta * (target - s) * mask
                    else:
                        assert group is not None and target is not None
                        p = torch.softmax(s[:, group] / nudge.softmax_temperature, dim=1)
                        push = torch.zeros_like(s)
                        push[:, group] = nudge.beta * (target[:, group] - p)
                    if weight is not None:
                        push = push * weight
                    total = total + push
                v = (v + rule.dt * (-v + total)) * k
                previous = s
                s = self._activation(v) * k
                if adapt is not None:
                    a = a + (s - a) / adapt.tau_steps
                if want:
                    traj.append(s.detach().cpu().double().numpy())
                taken = t + 1
                if tolerance is not None and float((s - previous).abs().max()) < tolerance:
                    break
        return (
            v.cpu().double().numpy(),
            a.cpu().double().numpy(),
            s.cpu().double().numpy(),
            np.stack(traj) if want else None,
            taken,
        )
