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

Backend = Literal["cpu", "torch", "mlx"]


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
    try:
        import mlx.core as mx

        out["mlx"] = f"{mx.default_device().type.name} float32"
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


def _softmax_groups(nudge: Nudge) -> list[np.ndarray]:
    """The masked owners as one index array per softmax group (one group without ``groups``)."""
    masked = np.flatnonzero(np.asarray(nudge.mask) > 0)
    if nudge.groups is None:
        return [masked.astype(np.int64)]
    ids = np.asarray(nudge.groups)[masked]
    return [masked[ids == g].astype(np.int64) for g in np.unique(ids[ids >= 0])]


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
    repair: np.ndarray | None = None  # per row: the total movement of the published activations
    device: Any = (
        None  # the same state on an accelerator, so a continuation or a contrast stays there
    )

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
        self._mlx: Any = None
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
        elif backend == "mlx":
            if not blocked:
                raise ValueError("the mlx backend needs a wiring whose blocks fit dense_limit")
            self._mlx = _MlxKernel(wiring, self._weights, self.bias, rule, self.layout)
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
            repair=None if out.repair is None else out.repair[0],
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
        if state is None:
            v, a = np.zeros((batch, n)), np.zeros((batch, n))
        elif self.backend in ("torch", "mlx"):  # the kernels never write the host arrays
            v, a = np.atleast_2d(state.v), np.atleast_2d(state.adaptation)
        else:
            v, a = np.atleast_2d(state.v).copy(), np.atleast_2d(state.adaptation).copy()
        if v.shape != (batch, n):
            raise ValueError("state batch does not match the drive batch")
        handle = None
        if self.backend in ("torch", "mlx"):
            kernel = self._torch if self.backend == "torch" else self._mlx
            v, a, s, traj, taken, repair, handle = kernel.run(
                v, a, drive, keep, steps, trajectory, nudge, tolerance, state
            )
        elif self._blocks is not None and not trajectory and _FUSED:
            from .fused import fused_settle

            s, taken, repair = fused_settle(
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
            v, a, s, traj, taken, repair = self._run_numpy(
                v, a, drive, keep, steps, trajectory, nudge, tolerance
            )
        return SettledState(
            v=v,
            activation=s,
            adaptation=a,
            steps=taken,
            trajectory=traj,
            repair=repair,
            device=handle,
        )

    def contrast_on_device(
        self, plus: SettledState, minus: SettledState
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``sum_b s+ s+ - s- s-`` per overlap and ``sum_b (s+ - s-)`` per owner, on the device.

        Returns ``None`` when either state does not carry this engine's device
        handle; the learner then reads the activations on the host.
        """
        kernel = self._torch if self.backend == "torch" else self._mlx
        if kernel is None or plus.device is None or minus.device is None:
            return None
        if plus.device.get("kernel") != self.backend or minus.device.get("kernel") != self.backend:
            return None
        out: tuple[np.ndarray, np.ndarray] = kernel.contrast(plus.device["s"], minus.device["s"])
        return out

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray]:
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
        repair = np.zeros(len(v))
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
            movement = np.abs(s - previous)
            repair += movement.sum(axis=1)
            if tolerance is not None and float(movement.max()) < tolerance:
                break
        if traj is not None:
            traj = traj[:taken]
        return v, a, s, traj, taken, repair

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
        self.backend_name = "torch"
        self._rest: Any = None
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
        else:  # per-overlap arrays serve only the gather-scatter path
            self.pre = torch.from_numpy(wiring.pre).to(self.device)
            self.post = torch.from_numpy(wiring.post).to(self.device)
            self.w = torch.from_numpy(weights).to(self.device, self.dtype)

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
        if self._rest is None:  # one scalar on the device, made once
            self._rest = torch.tensor(rule.rest_emission, dtype=self.dtype, device=self.device)
        rest = self._rest
        r = torch.sigmoid(rule.slope * (v - rule.threshold)) - rest
        s = torch.relu(r) / (1.0 - rest)
        if rule.leak:
            s = s + rule.leak * torch.clamp(r, max=0.0) / rest
        return s

    def contrast(self, s_plus: Any, s_minus: Any) -> tuple[np.ndarray, np.ndarray]:
        """The block Gram contrast on the device; only the per-overlap result comes back."""
        torch, lay = self.torch, self.layout
        assert lay is not None
        flat = torch.zeros(lay.size, dtype=self.dtype, device=self.device)
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            a_plus, a_minus = s_plus[:, a0:a1], s_minus[:, a0:a1]
            if torch.equal(a_plus, a_minus):
                block = a_plus.T @ (s_plus[:, b0:b1] - s_minus[:, b0:b1])
            else:
                block = a_plus.T @ s_plus[:, b0:b1] - a_minus.T @ s_minus[:, b0:b1]
            flat[lay.offset[k] : lay.offset[k + 1]] = block.reshape(-1)
        index = torch.from_numpy(lay.edge_index).to(self.device)
        edges = flat[index].cpu().double().numpy()
        owners = (s_plus - s_minus).sum(dim=0).cpu().double().numpy()
        return edges, owners

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
        state: SettledState | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray, Any]:
        torch, rule = self.torch, self.rule

        def to(x: np.ndarray) -> Any:  # no host copy when already contiguous float64
            y = np.ascontiguousarray(x, dtype=float)
            return torch.from_numpy(y).to(self.device, self.dtype)

        handle = state.device if state is not None and state.device is not None else None
        if handle is not None and handle.get("owner") is self:
            v, a = handle["v"].clone(), handle["a"].clone()  # the state never left the device
        else:
            v, a = to(v0), to(a0)
        d, k = to(drive), to(keep)
        batch = v.shape[0]
        adapt = rule.adaptation
        target = mask = weight = None
        groups: list[Any] = []
        if nudge is not None:
            target = to(np.broadcast_to(nudge.target, (batch, self.n)))
            mask = to(nudge.mask)
            groups = [
                torch.from_numpy(members).to(self.device) for members in _softmax_groups(nudge)
            ]
            if nudge.weight is not None:
                weight = to(np.asarray(nudge.weight, dtype=float))[:, None]
        traj = []
        taken = 0
        repair = torch.zeros(batch, dtype=self.dtype, device=self.device)
        cache: list[Any] = [None] * (0 if self.layout is None else self.layout.pairs)
        previous = None
        with torch.no_grad():
            if handle is not None and handle.get("owner") is self:
                s = handle["s"] * k
            else:
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
                    else:  # one softmax per group of competing owners
                        assert target is not None
                        push = torch.zeros_like(s)
                        for members in groups:
                            p = torch.softmax(s[:, members] / nudge.softmax_temperature, dim=1)
                            push[:, members] = nudge.beta * (target[:, members] - p)
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
                movement = (s - previous).abs()
                repair = repair + movement.sum(dim=1)
                if tolerance is not None and float(movement.max()) < tolerance:
                    break
        return (
            v.cpu().double().numpy(),
            a.cpu().double().numpy(),
            s.cpu().double().numpy(),
            np.stack(traj) if want else None,
            taken,
            repair.cpu().double().numpy(),
            {"kernel": self.backend_name, "owner": self, "v": v, "a": a, "s": s},
        )


class _MlxKernel:
    """The block settlement on Apple silicon through MLX: float32, unified memory, lazy graphs."""

    def __init__(
        self,
        wiring: Wiring,
        weights: np.ndarray,
        bias: np.ndarray,
        rule: GradedRule,
        layout: Layout,
    ) -> None:
        import mlx.core as mx

        self.mx = mx
        self.backend_name = "mlx"
        self.rule = rule
        self.n = wiring.n
        self.layout = layout
        self.bias = mx.array(bias.astype(np.float32))
        flat = layout.flat(weights)
        self.blocks = [
            mx.array(np.ascontiguousarray(b, dtype=np.float32)) for b in layout.blocks(flat)
        ]
        self.by_post: list[list[int]] = [[] for _ in range(layout.ranges)]
        for k in range(layout.pairs):
            self.by_post[int(layout.pair_post[k])].append(k)
        self.sources = layout.sources()
        self.edge_index = mx.array(layout.edge_index)

    def _activation(self, v: Any) -> Any:
        mx, rule = self.mx, self.rule
        rest = rule.rest_emission
        r = mx.sigmoid(rule.slope * (v - rule.threshold)) - rest
        s = mx.maximum(r, 0.0) / (1.0 - rest)
        if rule.leak:
            s = s + rule.leak * mx.minimum(r, 0.0) / rest
        return s

    def _inbox(self, s: Any, previous: Any, cache: list[Any]) -> Any:
        mx, lay = self.mx, self.layout
        moved = [True] * lay.ranges
        if previous is not None:
            for r in self.sources:
                a0, a1 = int(lay.starts[r]), int(lay.starts[r + 1])
                moved[r] = not bool(mx.array_equal(s[:, a0:a1], previous[:, a0:a1]).item())
        parts = []
        batch = s.shape[0]
        for r in range(lay.ranges):
            b0, b1 = int(lay.starts[r]), int(lay.starts[r + 1])
            total = None
            for k in self.by_post[r]:
                a0, a1, _, _ = lay.bounds(k)
                if cache[k] is None or moved[int(lay.pair_pre[k])]:
                    cache[k] = s[:, a0:a1] @ self.blocks[k]
                total = cache[k] if total is None else total + cache[k]
            parts.append(mx.zeros((batch, b1 - b0), dtype=mx.float32) if total is None else total)
        return mx.concatenate(parts, axis=1)

    def contrast(self, s_plus: Any, s_minus: Any) -> tuple[np.ndarray, np.ndarray]:
        """The block Gram contrast on the device; only the per-overlap result comes back."""
        mx, lay = self.mx, self.layout
        pieces = []
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            a_plus, a_minus = s_plus[:, a0:a1], s_minus[:, a0:a1]
            if bool(mx.array_equal(a_plus, a_minus).item()):
                block = a_plus.T @ (s_plus[:, b0:b1] - s_minus[:, b0:b1])
            else:
                block = a_plus.T @ s_plus[:, b0:b1] - a_minus.T @ s_minus[:, b0:b1]
            pieces.append(block.reshape(-1))
        flat = mx.concatenate(pieces) if pieces else mx.zeros((0,), dtype=mx.float32)
        edges = np.array(flat[self.edge_index], dtype=np.float64)
        owners = np.array((s_plus - s_minus).sum(axis=0), dtype=np.float64)
        return edges, owners

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
        state: SettledState | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray, Any]:
        mx, rule = self.mx, self.rule

        def to(x: np.ndarray) -> Any:
            return mx.array(np.asarray(x, dtype=np.float32))

        handle = state.device if state is not None and state.device is not None else None
        if handle is not None and handle.get("owner") is self:
            v, a, s = handle["v"], handle["a"], handle["s"]
        else:
            v, a = to(v0), to(a0)
            s = None
        d, k = to(drive), to(keep)
        batch = v.shape[0]
        masked = bool((keep != 1.0).any())
        adapt = rule.adaptation
        target = mask = weight = None
        groups: list[Any] = []
        if nudge is not None:
            target = to(np.broadcast_to(nudge.target, (batch, self.n)))
            mask = to(nudge.mask)
            groups = [mx.array(members) for members in _softmax_groups(nudge)]
            if nudge.weight is not None:
                weight = to(np.asarray(nudge.weight, dtype=float))[:, None]
        traj = []
        taken = 0
        repair = mx.zeros((batch,), dtype=mx.float32)
        cache: list[Any] = [None] * self.layout.pairs
        previous = None
        if s is None:
            s = self._activation(v)
        if masked:
            s = s * k
        standing = d + self.bias
        for t in range(steps):
            inbox = self._inbox(s, previous, cache)
            total = inbox + standing
            if adapt is not None:
                total = total - adapt.strength * a
            if nudge is not None:
                assert target is not None and mask is not None
                if nudge.softmax_temperature is None:
                    push = nudge.beta * (target - s) * mask
                else:  # one softmax per group of competing owners
                    push = mx.zeros_like(s)
                    for members in groups:
                        p = mx.softmax(s[:, members] / nudge.softmax_temperature, axis=1)
                        push[:, members] = nudge.beta * (target[:, members] - p)
                if weight is not None:
                    push = push * weight
                total = total + push
            v = v + rule.dt * (-v + total)
            if masked:
                v = v * k
            previous = s
            s = self._activation(v)
            if masked:
                s = s * k
            if adapt is not None:
                a = a + (s - a) / adapt.tau_steps
            movement = mx.abs(s - previous)
            repair = repair + movement.sum(axis=1)
            mx.eval(v, s, a, repair)  # one graph per step; the tolerance needs the number
            if want:
                traj.append(np.array(s, dtype=np.float64))
            taken = t + 1
            if tolerance is not None and float(movement.max().item()) < tolerance:
                break
        return (
            np.array(v, dtype=np.float64),
            np.array(a, dtype=np.float64),
            np.array(s, dtype=np.float64),
            np.stack(traj) if want else None,
            taken,
            np.array(repair, dtype=np.float64),
            {"kernel": "mlx", "owner": self, "v": v, "a": a, "s": s},
        )
