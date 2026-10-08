"""The brain: neurons and synapses settling under a stimulus.

One ``Brain`` holds a connectome and a neuron model and runs the neuron update for a
number of steps from rest, or from a given state, under a stimulus. Two
backends do the same arithmetic:

* ``"cpu"``: NumPy in float64. The transport is one segmented sum per
  step, a scatter of every synapse's message into its neuron's synaptic input; for
  connectomes whose dense blocks fit (at most ``dense_limit`` squared entries,
  see ``cadence.blocks``) the same sum is done as block matrix products,
  reusing the product of every range of neurons that did not move since
  the previous step, which is far faster when the interpreter overhead of
  the scatter would dominate and lets a settling step cost what its
  moving neurons cost rather than what the whole net does. This is the
  receipt-grade backend: deterministic, exact to rounding, and the one the
  neuron-by-neuron reference is compared against.
* ``"torch"``: the same scatter with ``index_add_`` on whatever device
  torch offers, CUDA in float64, Apple silicon in float32 (MPS has no
  float64), CPU in float64. Use it for interactive work and for very large
  connectomes, and keep a conformance check against ``"cpu"``; the fly brain
  has neurons on knife edges where float32 flips a bistable readout.

Three kinds of parameter sit on a brain, all defaulting to the connectome:
``efficacy``, one signed number per synapse (the connectome's sign unless a
learner has moved it); ``log_gain``, one per neuron on its outgoing
synapses, how a lane declares a stimulus rate or a region's gain; and
``bias``, one per neuron. The effective drive of synapse ``e`` per unit
presynaptic activation is ``gain * count[e] * efficacy[e] *
exp(log_gain[pre[e]])``.

Settling runs on a batch of stimuli at once; ``settle`` is the one-stimulus
convenience. A ``Nudge`` adds a drive that pulls chosen neurons toward a
target activation, which is how the learning rule's second phase enters.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .blocks import BlockTransport, Layout
from .blocks import layout as make_layout
from .connectome import Connectome, _neuron_indices
from .neuron import NeuronModel
from .recording import _Capture, _capture

__all__ = [
    "Brain", "BrainState", "Equilibrium", "RefinementReport", "Nudge",
    "available_backends", "Backend",
]

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
    """Extra drive ``beta * (target - s)`` on the neurons where ``mask`` is one.

    With ``softmax_temperature`` set, the neurons under the mask compete: the
    drive is ``beta * (target - softmax(s / T))`` over that group, a
    cross-entropy nudge, so pushing one neuron up pushes the others down only
    as much as their share. ``weight``, one number per batch row, scales the
    nudge row by row; a negative weight pushes away from the target. That is
    how a reward enters: the target is the action taken and the weight its
    advantage. A temperature vector has one entry per neuron and must be constant
    within each competing group, allowing different motor slots to explore at
    different temperatures in the same equilibrium.

    Optional ``anchor`` and ``anchor_gain`` add an independent quadratic
    boundary drive ``anchor_gain * (anchor - s)``. This term is held fixed
    during a solve and is not scaled by beta, target mask or row weight.
    Positive and negative target phases therefore share the same boundary,
    including when their combined quadratic coefficients cancel.
    """

    target: np.ndarray  # (batch, n) or (n,)
    mask: np.ndarray  # (n,)
    beta: float
    softmax_temperature: float | np.ndarray | None = None
    weight: np.ndarray | None = None  # (batch,)
    groups: np.ndarray | None = None  # (n,) group id per neuron, -1 for none: one softmax per group
    # An independent quadratic boundary: anchor_gain * (anchor - s).
    # It is not multiplied by the target nudge's beta or per-row weight.
    anchor: np.ndarray | None = None  # (batch, n), (1, n), or (n,)
    anchor_gain: np.ndarray | None = None  # nonnegative (n,)

    def __post_init__(self) -> None:
        mask, target = np.asarray(self.mask), np.asarray(self.target)
        if mask.ndim != 1:
            raise ValueError("nudge mask must have one entry per neuron")
        if target.ndim not in (1, 2) or target.shape[-1] != mask.size:
            raise ValueError(
                "nudge target must have one entry per neuron, optionally per batch row"
            )
        if not np.isfinite(self.beta):
            raise ValueError("nudge beta must be finite")
        if not np.isfinite(target).all() or not np.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("nudge target and mask must be finite; mask must be nonnegative")
        object.__setattr__(self, "target", target.astype(float, copy=True))
        object.__setattr__(self, "mask", mask.astype(float, copy=True))
        temperature = self.softmax_temperature
        if temperature is not None:
            temperatures = np.asarray(temperature, dtype=float)
            if (
                temperatures.ndim > 1
                or (temperatures.ndim == 1 and temperatures.shape != mask.shape)
                or not np.isfinite(temperatures).all()
                or (temperatures <= 0).any()
            ):
                raise ValueError(
                    "softmax_temperature must be finite and positive, scalar or per neuron"
                )
            if temperatures.ndim == 1:
                temperatures = temperatures.copy()
                temperatures.flags.writeable = False
                object.__setattr__(self, "softmax_temperature", temperatures)
            else:
                object.__setattr__(self, "softmax_temperature", float(temperatures))
        if self.weight is not None and np.asarray(self.weight).ndim != 1:
            raise ValueError("nudge weight must have one entry per batch row")
        if self.groups is not None and np.asarray(self.groups).shape != mask.shape:
            raise ValueError("nudge groups must have one entry per neuron")
        if self.groups is not None:
            groups = np.asarray(self.groups)
            if not np.issubdtype(groups.dtype, np.integer) or (groups < -1).any():
                raise ValueError("nudge groups must be integer IDs, -1 for ungrouped neurons")
            object.__setattr__(self, "groups", groups.copy())
        if isinstance(self.softmax_temperature, np.ndarray):
            for members in _softmax_groups(self):
                if not np.all(
                    self.softmax_temperature[members] == self.softmax_temperature[members[0]]
                ):
                    raise ValueError("softmax_temperature must be constant within each group")
        if self.weight is not None:
            weight = np.asarray(self.weight, dtype=float)
            if not np.isfinite(weight).all():
                raise ValueError("nudge weight must be finite")
            object.__setattr__(self, "weight", weight.copy())
        if (self.anchor is None) != (self.anchor_gain is None):
            raise ValueError("anchor and anchor_gain must be supplied together")
        if self.anchor is not None:
            anchor, gain = np.asarray(self.anchor, float), np.asarray(self.anchor_gain, float)
            if anchor.ndim not in (1, 2) or anchor.shape[-1] != mask.size:
                raise ValueError("anchor must have one entry per neuron, optionally per batch row")
            if gain.shape != mask.shape or not np.isfinite(gain).all() or (gain < 0).any():
                raise ValueError("anchor_gain must be a finite nonnegative vector over neurons")
            if not np.isfinite(anchor).all():
                raise ValueError("anchor must be finite")
            object.__setattr__(self, "anchor", anchor.copy())
            object.__setattr__(self, "anchor_gain", gain.copy())

    def drive(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        single = s.ndim == 1
        if s.ndim not in (1, 2) or s.shape[-1] != len(self.mask):
            raise ValueError("activation must have one entry per neuron, optionally per batch row")
        s = np.atleast_2d(s)
        full_target = np.broadcast_to(self.target, s.shape)
        if self.softmax_temperature is None:
            out = np.asarray(self.beta * (full_target - s) * self.mask, dtype=float)
        else:
            out = np.zeros_like(s)
            for group in _softmax_groups(self):
                temperature = _group_temperature(self, group)
                with np.errstate(over="ignore", invalid="ignore"):
                    z = s[:, group] / temperature
                    if not np.isfinite(z).all():
                        logits = s[:, group]
                        z = (logits - logits.max(axis=1, keepdims=True)) / temperature
                    else:
                        z -= z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                out[:, group] = self.beta * (full_target[:, group] - p) * self.mask[group]
        if self.weight is not None:
            if np.asarray(self.weight).shape != (len(s),):
                raise ValueError("nudge weight must have one entry per batch row")
            out = out * np.asarray(self.weight, dtype=float)[:, None]
        if self.anchor is not None:
            out += self.anchor_gain * (np.broadcast_to(self.anchor, s.shape) - s)
        return out[0] if single else out


def _softmax_groups(nudge: Nudge) -> list[np.ndarray]:
    """The masked neurons as one index array per softmax group (one group without ``groups``)."""
    masked = np.flatnonzero(np.asarray(nudge.mask) > 0)
    if nudge.groups is None:
        return [masked.astype(np.int64)] if masked.size else []
    ids = np.asarray(nudge.groups)[masked]
    return [masked[ids == g].astype(np.int64) for g in np.unique(ids[ids >= 0])]


def _group_temperature(nudge: Nudge, members: np.ndarray) -> float:
    temperature = nudge.softmax_temperature
    assert temperature is not None
    return float(temperature[members[0]]) if isinstance(temperature, np.ndarray) else temperature


@dataclass(frozen=True)
class BrainState:
    """State after settling: potentials, activations, adaptation, trajectory.

    A step cap or activation movement tolerance need not imply equilibrium;
    use ``Brain.residual`` to check the remaining neuron-equation error.

    From ``settle`` every array is ``(n,)`` and ``trajectory`` is ``(steps, n)``;
    from ``settle_batch`` they carry a leading batch axis.
    """

    v: np.ndarray
    activation: np.ndarray
    adaptation: np.ndarray
    steps: int
    trajectory: np.ndarray | None = None
    # per row: the total movement of the published activations while settling
    activity_change: np.ndarray | None = None
    device: Any = (
        None  # the same state on an accelerator, so a continuation or a contrast stays there
    )

    @property
    def batched(self) -> bool:
        handle = self.__dict__.get("device")
        if self.__dict__.get("v") is None and handle is not None and "s" in handle:
            return len(handle["s"].shape) == 2  # the shape alone; no fetch from the device
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


@dataclass(frozen=True)
class RefinementReport:
    """Per-row diagnostics from optional CPU energy refinement.

    ``local_residual`` is the equation error before refinement. ``steps`` counts
    accepted Newton steps, separately from local sweeps; it does not count dense
    curvature evaluations or line-search trials. ``min_curvature`` is the
    smallest eigenvalue of the reduced activity-space energy Hessian at the
    returned state. Combined with a small full residual, a positive value is a
    numerical local-stability check, not a certified global minimum, uniqueness
    theorem or gradient-branch guarantee between free and nudged phases.

    ``status`` is ``local`` or ``refined`` on a qualified row; any other value
    explains why refinement did not qualify it. Failed rows must not be acted on
    merely because a finite state was returned.
    """

    local_residual: np.ndarray
    steps: np.ndarray
    min_curvature: np.ndarray
    status: tuple[str, ...]


@dataclass(frozen=True)
class Equilibrium:
    """A bounded solve and its measured equation error, one entry per batch row.

    ``steps`` counts local sweeps in this call. ``state.steps`` is the last chunk's count.
    Convergence here means a small residual, not uniqueness, stability or task quality.
    Optional refinement diagnostics add a local-curvature admission check;
    ``qualified`` combines it with the full residual without changing ``converged``.
    """

    state: BrainState
    residual: np.ndarray
    steps: int
    tolerance: float
    refinement: RefinementReport | None = None
    residual_checks: int = 0  # additional transport evaluations, outside the local sweeps
    damping_halvings: int = 0  # integration-step halvings used by this bounded solve
    stagnation_checks: int = 0  # complete-state comparisons, with no synaptic transport

    @property
    def converged(self) -> np.ndarray:
        return np.asarray(self.residual <= self.tolerance)

    @property
    def qualified(self) -> np.ndarray:
        """Rows meeting the chosen solver's checks (residual-only for local solves)."""
        if self.refinement is None:
            return self.converged
        report = self.refinement
        return np.asarray(
            self.converged
            & np.isfinite(report.min_curvature)
            & (report.min_curvature > 0)
            & np.isin(report.status, ("local", "refined"))
        )


def _lazy(name: str, key: str) -> property:
    """A host array of a settled state that is fetched from the device on first use.

    The accelerator kernels leave the state on the device and hand back a fetch; a
    continuation, a contrast on the device and the step count never need the host copy,
    so it is made only when something reads it.
    """

    def get(self: BrainState) -> np.ndarray:
        value = self.__dict__.get(name)
        if value is None:
            handle = self.__dict__.get("device")
            if handle is None or "fetch" not in handle:
                raise AttributeError(name)
            value = handle["fetch"](key)
            self.__dict__[name] = value
        return np.asarray(value)

    def put(self: BrainState, value: np.ndarray | None) -> None:
        self.__dict__[name] = value

    return property(get, put)


for _name, _key in (("v", "v"), ("activation", "s"), ("adaptation", "a")):
    setattr(BrainState, _name, _lazy(_name, _key))


class Brain:
    def __init__(
        self,
        connectome: Connectome,
        neuron_model: NeuronModel,
        *,
        backend: Backend = "cpu",
        efficacy: np.ndarray | None = None,
        log_gain: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        device: str | None = None,
        dense_limit: int = 2048,
        layout: Layout | None = None,
        precision: str | None = None,
    ) -> None:
        self.connectome = connectome
        self.neuron_model = neuron_model
        self.backend: Backend = backend
        self.dense_limit = dense_limit
        self.precision = precision  # torch only: "float64" or "float32"; default by device
        self.layout: Layout = make_layout(connectome) if layout is None else layout
        # The parameters live on the host, or on the torch device once a learner has moved
        # them there; the host arrays are then fetched when something reads them.
        self._efficacy: np.ndarray | None = (
            connectome.sign.copy() if efficacy is None else np.asarray(efficacy, float).copy()
        )
        self._log_gain = np.zeros(connectome.n) if log_gain is None else np.array(log_gain, float)
        self._bias: np.ndarray | None = (
            np.zeros(connectome.n) if bias is None else np.array(bias, float)
        )
        assert self._efficacy is not None and self._bias is not None
        if self._efficacy.shape != (connectome.synapses,):
            raise ValueError("efficacy must have one entry per synapse")
        if self.log_gain.shape != (connectome.n,) or self._bias.shape != (connectome.n,):
            raise ValueError("log_gain and bias must have one entry per neuron")
        if not all(np.isfinite(x).all() for x in (self._efficacy, self.log_gain, self._bias)):
            raise ValueError("brain parameters must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            self._gain_pre: np.ndarray = (
                neuron_model.gain * connectome.count * np.exp(self.log_gain)[connectome.pre]
            )
        self._weights_host: np.ndarray | None = self._gain_pre * self._efficacy
        if not np.isfinite(self._weights_host).all():
            raise ValueError("effective synaptic weights must be finite; reduce gain or efficacy")
        # Segment boundaries of the (post-sorted) synapse arrays, for the segmented sum.
        # kept on the connectome: one pass per connectome
        segments = connectome.__dict__.get("_segments")
        if segments is None:
            post = connectome.post
            if connectome.synapses:
                change = np.flatnonzero(np.diff(post)) + 1
                starts = np.concatenate([[0], change]).astype(np.int64)
                segments = (starts, post[starts])
            else:
                segments = (np.zeros(0, np.int64), np.zeros(0, np.int64))
            connectome.__dict__["_segments"] = segments
        self._starts: np.ndarray = segments[0]
        self._neurons_with_synaptic_input: np.ndarray = segments[1]
        # The block transport: dense blocks between neuron ranges, when they fit.
        blocked = self.layout.size <= dense_limit * dense_limit
        self._blocked = blocked
        self._flat: np.ndarray | None = None  # the host blocks, made on first use (cpu paths)
        self._csr: tuple[np.ndarray, Any] | None = None  # optional sparse matrix and its weights
        self._torch: Any = None
        self._mlx: Any = None
        if backend == "torch":  # the kernel scatters the weights into its blocks on the device
            self._torch = _TorchKernel(
                connectome,
                self._efficacy,
                self._gain_pre,
                self._bias,
                neuron_model,
                device,
                self.layout if blocked else None,
                precision,
            )
        elif backend == "mlx":
            if not blocked:
                raise ValueError("the mlx backend needs a connectome whose blocks fit dense_limit")
            self._mlx = _MlxKernel(connectome, self._weights, self.bias, neuron_model, self.layout)
        elif backend != "cpu":
            raise ValueError(f"unknown backend {backend!r}")

    @property
    def _blocks(self) -> np.ndarray | None:
        """The flat host blocks when the connectome is blocked; scattered once, on first use."""
        if not self._blocked:
            return None
        if self._flat is None:
            self._flat = self.layout.flat(self._weights)
        return self._flat

    # -- parameters

    @property
    def efficacy(self) -> np.ndarray:
        """Read-only parameters; replace via the setter or ``with_parameters``."""
        if self._efficacy is None:
            assert self._torch is not None
            self._efficacy = self._torch.host_scale()
        result = self._efficacy.view()
        result.setflags(write=False)
        return result

    @efficacy.setter
    def efficacy(self, value: np.ndarray) -> None:
        updated = self.with_parameters(efficacy=value)
        self.__dict__.update(updated.__dict__)

    @property
    def bias(self) -> np.ndarray:
        """Read-only biases; replace via the setter or ``with_parameters``."""
        if self._bias is None:
            assert self._torch is not None
            self._bias = self._torch.host_bias()
        result = self._bias.view()
        result.setflags(write=False)
        return result

    @bias.setter
    def bias(self, value: np.ndarray) -> None:
        updated = self.with_parameters(bias=value)
        self.__dict__.update(updated.__dict__)

    @property
    def log_gain(self) -> np.ndarray:
        """Read-only gains; explicit replacement rebuilds every derived weight."""
        result = self._log_gain.view()
        result.setflags(write=False)
        return result

    @log_gain.setter
    def log_gain(self, value: np.ndarray) -> None:
        updated = self.with_parameters(log_gain=value)
        self.__dict__.update(updated.__dict__)

    @property
    def _weights(self) -> np.ndarray:
        if self._weights_host is None:
            self._weights_host = self._gain_pre * self.efficacy
        return self._weights_host

    def _with_device_parameters(self, scale: Any, bias: Any) -> Brain:
        """The brain with new parameters that stay on the torch device.

        The kernel is shared and updated in place, so a state settled on it continues on it;
        the brain this was called on is superseded (its host copies, if any, are stale).
        """
        assert self._torch is not None
        self._torch.set_parameters(scale, bias)
        new = copy.copy(self)
        new._efficacy = None
        new._bias = None
        new._weights_host = None
        new._flat = None
        new._csr = None
        return new

    def with_parameters(
        self,
        *,
        efficacy: np.ndarray | None = None,
        log_gain: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> Brain:
        """A new brain on the same connectome with some parameters replaced.

        On the host, new efficacies and biases alone make a copy whose derived arrays
        (the weights, the blocks) are remade on first use, not a rebuilt brain.
        """
        if self.backend == "cpu" and log_gain is None:
            new = copy.copy(self)
            if efficacy is not None:
                scale = np.asarray(efficacy, float)
                if scale.shape != (self.connectome.synapses,) or not np.isfinite(scale).all():
                    raise ValueError("efficacy must have one finite entry per synapse")
                with np.errstate(over="ignore", invalid="ignore"):
                    weights = self._gain_pre * scale
                if not np.isfinite(weights).all():
                    raise ValueError(
                        "effective synaptic weights must be finite; reduce gain or efficacy"
                    )
                new._efficacy = scale.copy()
                new._weights_host = weights
                new._flat = None
                new._csr = None
            if bias is not None:
                b = np.asarray(bias, float)
                if b.shape != (self.connectome.n,) or not np.isfinite(b).all():
                    raise ValueError("bias must have one finite entry per neuron")
                new._bias = b.copy()
            return new
        return Brain(
            self.connectome,
            self.neuron_model,
            backend=self.backend,
            efficacy=self.efficacy if efficacy is None else efficacy,
            log_gain=self.log_gain if log_gain is None else log_gain,
            bias=self.bias if bias is None else bias,
            device=str(self._torch.device) if self._torch is not None else None,
            dense_limit=self.dense_limit,
            layout=self.layout,
            precision=self.precision,
        )

    @property
    def weights(self) -> np.ndarray:
        """Effective drive per unit presynaptic activation, one per synapse."""
        result = self._weights.view()
        result.setflags(write=False)
        return result

    def dense(self) -> np.ndarray:
        """The synapse matrix ``W[pre, post]``: the synaptic input of a batch ``s`` is ``s @ W``.

        Built on demand; this is what a page that settles the net in a browser embeds.
        """
        dense = np.zeros((self.connectome.n, self.connectome.n))
        np.add.at(dense, (self.connectome.pre, self.connectome.post), self._weights)
        return dense

    # -- stimuli

    def stimulus_vector(
        self, stimulus: Mapping[int, float] | Sequence[int] | np.ndarray | None
    ) -> np.ndarray:
        """A dense drive vector, a list of neurons at full amplitude, or a {neuron: level} map."""
        out = np.zeros(self.connectome.n)
        if stimulus is None:
            return out
        if isinstance(stimulus, Mapping):
            for neuron, level in stimulus.items():
                amplitude = self.neuron_model.stimulus_amplitude
                index = _neuron_indices([neuron], self.connectome.n, "stimulus")[0]
                if not np.isfinite(level):
                    raise ValueError("stimulus levels must be finite")
                out[index] = amplitude * float(level)
            return out
        array = np.asarray(stimulus)
        if (
            array.dtype.kind in "iu"
            and array.ndim == 1
            and (array.shape[0] != self.connectome.n or array.max(initial=0) > 1)
        ):
            out[_neuron_indices(array, self.connectome.n, "stimulus")] = (
                self.neuron_model.stimulus_amplitude
            )
            return out
        if array.shape == (self.connectome.n,):
            return array.astype(float)
        out[_neuron_indices(array, self.connectome.n, "stimulus")] = (
            self.neuron_model.stimulus_amplitude
        )
        return out

    def stimulus_levels(self, levels: np.ndarray) -> np.ndarray:
        """Scale dense per-neuron levels by stimulus amplitude; signed levels are allowed."""
        levels = np.asarray(levels, float)
        if not np.isfinite(levels).all():
            raise ValueError("stimulus levels must be finite")
        return levels * self.neuron_model.stimulus_amplitude

    # -- settling

    def settle(
        self,
        stimulus: Mapping[int, float] | Sequence[int] | np.ndarray | None = None,
        *,
        steps: int = 60,
        state: BrainState | None = None,
        mask: np.ndarray | None = None,
        trajectory: bool = False,
        nudge: Nudge | None = None,
        tolerance: float | None = None,
    ) -> BrainState:
        """Run the neuron model for ``steps`` steps from rest or from ``state``, under one stimulus.

        ``mask`` zeros ablated neurons. The trajectory, when requested, holds
        the activation after every step. With ``tolerance`` set the run stops
        early once no neuron's activation moved more than that in a step;
        ``steps`` is then the most it will run, and the state reports how
        many steps it took. ``None`` or zero disables early stopping without
        reading a device scalar at every iteration.
        """
        drive = self.stimulus_vector(stimulus)[None, :]
        out = self.settle_batch(
            drive,
            steps=steps,
            state=state,
            mask=mask,
            trajectory=trajectory,
            nudge=nudge,
            tolerance=tolerance,
        )
        return BrainState(
            v=out.v[0],
            activation=out.activation[0],
            adaptation=out.adaptation[0],
            steps=out.steps,
            trajectory=None if out.trajectory is None else out.trajectory[:, 0, :],
            activity_change=None if out.activity_change is None else out.activity_change[0],
        )

    def equilibrate(
        self,
        drive: np.ndarray,
        *,
        budget: int = 512,
        chunk: int = 32,
        tolerance: float = 1e-5,
        state: BrainState | None = None,
        mask: np.ndarray | None = None,
        nudge: Nudge | None = None,
        damping: int = 0,
        first_halving: int = 0,
    ) -> Equilibrium:
        """Continue a joint state until its equations hold or the exact step budget expires.

        Unlike ``settle_batch(tolerance=...)``, this tests the potential and adaptation
        equations, including the same mask and nudge. Checks cost one extra transport
        per chunk. Resident float64 Torch states are checked on their device. All rows
        advance together; a row's small residual does not freeze it.
        ``drive`` is dense, with one column per neuron, as in ``settle_batch``.
        Optional ``damping`` reserves equal portions of the one sweep budget
        for successively halved integration steps. Two consecutive repeated
        complete-state checkpoints at rounding precision end an unqualified
        attempt early, leaving its unused sweeps for the remaining attempts.
        Every attempt checks the original equations; this changes the numerical
        method, not the model. Repetition never qualifies an answer.
        ``first_halving`` skips the first attempts of that same schedule: the solve
        starts at ``dt * 2**-first_halving`` and keeps the smallest step
        ``dt * 2**-damping``, splitting the budget over the remaining attempts.
        """
        for name, value, minimum in (
            ("budget", budget, 0), ("chunk", chunk, 1), ("damping", damping, 0),
            ("first_halving", first_halving, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if first_halving > damping:
            raise ValueError("first_halving cannot exceed damping")
        if damping:
            attempts = min(damping + 1 - first_halving, max(1, budget))
            minimum_rule = self.neuron_model.replace(
                dt=float(np.ldexp(self.neuron_model.dt, -(first_halving + attempts - 1)))
            )
            dtype = "float32" if (self._mlx is not None or (
                self._torch is not None and self._torch.dtype == self._torch.torch.float32
            )) else "float64"
            minimum_rule._validate_precision(dtype)
            used = checks = stagnation_checks = 0
            current_state = state
            for attempt in range(attempts):
                halving = first_halving + attempt
                portion = (budget - used + attempts - attempt - 1) // (attempts - attempt)
                candidate = self if halving == 0 else Brain(
                    self.connectome,
                    self.neuron_model.replace(dt=float(np.ldexp(self.neuron_model.dt, -halving))),
                    backend=self.backend,
                    efficacy=self.efficacy,
                    log_gain=self.log_gain,
                    bias=self.bias,
                    device=str(self._torch.device) if self._torch is not None else None,
                    dense_limit=self.dense_limit,
                    layout=self.layout,
                    precision=self.precision,
                )
                phase = candidate._equilibrate_local(
                    drive, state=current_state, budget=portion, chunk=chunk,
                    tolerance=tolerance, mask=mask, nudge=nudge,
                    stop_if_stalled=attempt < attempts - 1,
                )
                used += phase.steps
                checks += phase.residual_checks
                stagnation_checks += phase.stagnation_checks
                error = phase.residual
                if halving:
                    error = self.residual(drive, phase.state, mask=mask, nudge=nudge)
                    checks += 1
                result = Equilibrium(
                    phase.state, error, used, tolerance,
                    residual_checks=checks, damping_halvings=halving,
                    stagnation_checks=stagnation_checks,
                )
                if np.all(result.qualified) or used >= budget:
                    return result
                if not all(np.isfinite(value).all() for value in (
                    phase.state.v, phase.state.activation, phase.state.adaptation
                )):
                    return result
                current_state = phase.state
            return result
        return self._equilibrate_local(
            drive, budget=budget, chunk=chunk, tolerance=tolerance,
            state=state, mask=mask, nudge=nudge,
        )

    def _repeated_state(self, previous: BrainState, current: BrainState) -> bool:
        """Compare all equation state on its backend, without a transport or host copy."""
        keys = ("v", "s", "a")
        old, new = previous.device, current.device
        if self._torch is not None and old is not None and new is not None and (
            old.get("holder") is self._torch and new.get("holder") is self._torch
        ):
            torch = self._torch.torch
            rounding = 8 * torch.finfo(self._torch.dtype).eps
            unchanged = [
                ((new[key] - old[key]).abs() <= rounding * (
                    1 + torch.maximum(new[key].abs(), old[key].abs())
                )).all()
                for key in keys
            ]
            return bool(torch.stack(unchanged).all().item())
        if self._mlx is not None and old is not None and new is not None and (
            old.get("holder") is self._mlx and new.get("holder") is self._mlx
        ):
            mx = self._mlx.mx
            rounding = 8 * np.finfo(np.float32).eps
            unchanged = [
                mx.all(mx.abs(new[key] - old[key]) <= rounding * (
                    1 + mx.maximum(mx.abs(new[key]), mx.abs(old[key]))
                ))
                for key in keys
            ]
            return bool(mx.all(mx.stack(unchanged)).item())
        rounding = 8 * np.finfo(np.float64).eps
        with np.errstate(over="ignore", invalid="ignore"):
            return all(
                np.all(np.abs(after - before) <= rounding * (
                    1 + np.maximum(np.abs(after), np.abs(before))
                ))
                for before, after in zip(
                    (previous.v, previous.activation, previous.adaptation),
                    (current.v, current.activation, current.adaptation), strict=True,
                )
            )

    def _equilibrate_local(
        self,
        drive: np.ndarray,
        *,
        budget: int,
        chunk: int,
        tolerance: float,
        state: BrainState | None,
        mask: np.ndarray | None,
        nudge: Nudge | None,
        stop_if_stalled: bool = False,
    ) -> Equilibrium:
        """Run a validated local attempt; only a subsequent damping attempt may skip a stall."""
        current = self.settle_batch(drive, steps=0, state=state, mask=mask, nudge=nudge)
        error = self.residual(drive, current, mask=mask, nudge=nudge)
        used, checks, stagnation_checks, repeats = 0, 1, 0, 0
        while used < budget and not np.all(error <= tolerance):
            if not np.isfinite(error).all() and any(
                not np.isfinite(value).all()
                for value in (current.v, current.activation, current.adaptation)
            ):
                # A divergent numerical state is not a valid warm start for
                # another chunk. Keep its failed diagnostic, not a new exception.
                break
            previous = current
            current = self.settle_batch(
                drive, steps=min(chunk, budget - used), state=current, mask=mask, nudge=nudge
            )
            used += current.steps
            error = self.residual(drive, current, mask=mask, nudge=nudge)
            checks += 1
            if stop_if_stalled and used < budget and not np.all(error <= tolerance):
                stagnation_checks += 1
                repeats = repeats + 1 if self._repeated_state(previous, current) else 0
                if repeats >= 2:
                    break
        return Equilibrium(
            current, error, used, tolerance, residual_checks=checks,
            stagnation_checks=stagnation_checks,
        )

    def settle_batch(
        self,
        drive: np.ndarray,
        *,
        steps: int = 60,
        state: BrainState | None = None,
        mask: np.ndarray | None = None,
        trajectory: bool = False,
        nudge: Nudge | None = None,
        tolerance: float | None = None,
    ) -> BrainState:
        """Settle a batch of dense drives ``(batch, n)`` at once; see ``settle``.

        ``mask`` has shape ``(n,)`` or ``(1, n)`` for shared ablations, or
        ``(batch, n)`` for a separate mask on each row.
        """
        drive = np.asarray(drive, float)
        if drive.ndim == 1:
            drive = drive[None, :]
        if drive.ndim != 2 or drive.shape[0] == 0:
            raise ValueError("drive must have shape (batch, neurons) with at least one row")
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerance must be finite and nonnegative")
        # Movement is nonnegative, so it can never be strictly below zero.
        # Avoid an otherwise useless accelerator-to-host synchronization per step.
        if tolerance == 0:
            tolerance = None
        batch, n = drive.shape
        if n != self.connectome.n:
            raise ValueError("drive must have one column per neuron")
        if n == 0 or not np.isfinite(drive).all():
            raise ValueError("drive must be finite and the brain must contain neurons")
        keep = np.ones(n) if mask is None else np.asarray(mask, float)
        if keep.shape not in ((n,), (1, n), (batch, n)):
            raise ValueError("mask must have one entry per neuron, optionally per batch row")
        if not np.isfinite(keep).all() or ((keep < 0) | (keep > 1)).any():
            raise ValueError("mask must contain finite fractions in [0, 1]")
        if nudge is not None:
            if np.asarray(nudge.mask).shape != (n,):
                raise ValueError("nudge mask must have one entry per neuron")
            if np.asarray(nudge.target).shape not in ((n,), (1, n), (batch, n)):
                raise ValueError("nudge target must match the drive's neurons and batch size")
            if nudge.weight is not None and np.asarray(nudge.weight).shape != (batch,):
                raise ValueError("nudge weight must have one entry per batch row")
            if nudge.anchor is not None and np.asarray(nudge.anchor).shape not in (
                (n,),
                (1, n),
                (batch, n),
            ):
                raise ValueError("nudge anchor must match the drive's neurons and batch size")
        on_kernel = (
            state is not None
            and state.device is not None
            and self.backend in ("torch", "mlx")
            and state.device.get("holder")
            is (self._torch if self.backend == "torch" else self._mlx)
        )
        v: Any
        a: Any
        if state is None:
            v, a = np.zeros((batch, n)), np.zeros((batch, n))
        elif on_kernel:  # the state never left the device: no host array is read
            assert state is not None and state.device is not None
            v = a = None
            if tuple(state.device["s"].shape) != (batch, n):
                raise ValueError("state batch does not match the drive batch")
        elif self.backend in ("torch", "mlx"):  # the kernels never write the host arrays
            v, a = np.atleast_2d(state.v), np.atleast_2d(state.adaptation)
        else:
            # CPU kernels update in place; integer or float32 warm states must not
            # truncate potentials or silently lower the documented float64 precision.
            v = np.array(state.v, dtype=np.float64, copy=True, ndmin=2)
            a = np.array(state.adaptation, dtype=np.float64, copy=True, ndmin=2)
        if v is not None and (v.shape != (batch, n) or a.shape != (batch, n)):
            raise ValueError("state batch does not match the drive batch")
        if v is not None and (not np.isfinite(v).all() or not np.isfinite(a).all()):
            raise ValueError("warm potentials and adaptation must be finite")
        capture = _capture()
        handle = None
        if self.backend in ("torch", "mlx"):
            kernel = self._torch if self.backend == "torch" else self._mlx
            v, a, s, traj, taken, activity_change, handle = kernel.run(
                v, a, drive, keep, steps, trajectory, nudge, tolerance, state, capture
            )
        elif self._blocks is not None and not trajectory and capture is None and _FUSED:
            from .fused import fused_settle

            s, taken, activity_change = fused_settle(
                v,
                a,
                drive,
                self.bias,
                self.layout,
                self._blocks,
                keep,
                self.neuron_model,
                nudge,
                steps,
                tolerance,
            )
            traj = None
        else:
            v, a, s, traj, taken, activity_change = self._run_numpy(
                v, a, drive, keep, steps, trajectory, nudge, tolerance, capture
            )
        result = BrainState(
            v=v,
            activation=s,
            adaptation=a,
            steps=taken,
            trajectory=traj,
            activity_change=activity_change,
            device=handle,
        )
        if capture is not None:
            capture.finish(self, drive, keep, nudge, result)
        return result

    def contrast_on_device(
        self, plus: BrainState, minus: BrainState
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``sum_b s+ s+ - s- s-`` per synapse and ``sum_b (s+ - s-)`` per neuron, on the device.

        Returns ``None`` when either state does not carry this brain's device
        handle; the learner then reads the activations on the host.
        """
        kernel = self._torch if self.backend == "torch" else self._mlx
        if kernel is None or kernel.layout is None or plus.device is None or minus.device is None:
            return None
        if plus.device.get("holder") is not kernel or minus.device.get("holder") is not kernel:
            return None
        out: tuple[np.ndarray, np.ndarray] = kernel.contrast(plus.device["s"], minus.device["s"])
        return out

    def residual(
        self,
        drive: np.ndarray,
        state: BrainState,
        *,
        nudge: Nudge | None = None,
        mask: np.ndarray | None = None,
        on_device: bool = True,
    ) -> np.ndarray:
        """Maximum absolute fixed-point equation residual per row, without settling.

        For unmasked neurons the potential equation is ``synaptic input + drive + bias
        - adaptation_strength * a + nudge - v = 0``. When adaptation is enabled,
        ``activation(v) - a = 0`` must hold too. Neither error is reduced by a
        small integration step or a long adaptation time constant. A mask uses
        the actual potential projection of ``settle_batch``; ablated neurons
        must have zero potential, and their adaptation must decay to zero.

        A small residual certifies these equations at this state, not stability,
        uniqueness, or convergence from another start. Activation-movement
        tolerance and reaching a step cap do not provide this certificate.
        The calculation performs one transport. An unread resident float64 Torch state
        is checked on its device, returning only one scalar per row. Other states use
        the float64 CPU reference, including float32 accelerator states; ``on_device=False``
        forces that reference. No state or parameter is changed.
        A single state returns a length-one array; nonfinite errors return infinity.
        """
        d = np.asarray(drive, dtype=float)
        if d.ndim == 1:
            d = d[None, :]
        if d.ndim != 2 or d.shape[1] != self.connectome.n:
            raise ValueError("drive must have one column per neuron")
        keep = np.ones(self.connectome.n) if mask is None else np.asarray(mask, dtype=float)
        if keep.shape not in ((self.connectome.n,), (1, self.connectome.n), d.shape):
            raise ValueError("mask must have one entry per neuron, optionally per batch row")
        handle, kernel = state.device, self._torch
        if (
            on_device
            and kernel is not None
            and kernel.dtype == kernel.torch.float64
            and handle is not None
            and handle.get("holder") is kernel
            and state.__dict__.get("v") is None
            and state.__dict__.get("adaptation") is None
        ):
            if tuple(handle["v"].shape) != d.shape or tuple(handle["a"].shape) != d.shape:
                raise ValueError("state batch does not match the drive batch")
            return np.asarray(kernel.residual(d, keep, nudge, handle))
        v = np.atleast_2d(state.v)
        a = np.atleast_2d(state.adaptation)
        if v.shape != d.shape or a.shape != d.shape:
            raise ValueError("state batch does not match the drive batch")
        if self._blocks is not None and _FUSED:
            # the same equations in one compiled pass: a checked settle on the CPU then
            # costs one transport per chunk, not one NumPy operation per term
            from .fused import fused_residual

            return fused_residual(
                v, a, d, self.bias, self.layout, self._blocks, keep, self.neuron_model, nudge
            )
        s = self.neuron_model.activation(v) * keep
        blocks = None if self._blocks is None else BlockTransport(self.layout, self._blocks)
        error = self._synaptic_input(s, blocks) + d + self.bias - v
        adapt = self.neuron_model.adaptation
        if adapt is not None:
            error -= adapt.strength * a
        if nudge is not None:
            error += nudge.drive(s)
        # (projected_next_v - v) / dt, written without cancellation for keep=1.
        error = keep * error + (keep - 1.0) * v / self.neuron_model.dt
        result = np.max(np.abs(error), axis=1, initial=0.0)
        if adapt is not None:
            result = np.maximum(result, np.max(np.abs(s - a), axis=1, initial=0.0))
        return np.where(np.isfinite(result), result, np.inf)

    def _synaptic_input(self, s: np.ndarray, blocks: BlockTransport | None = None) -> np.ndarray:
        """Transport through fitting blocks, optional CSR, or the NumPy segmented sum."""
        if blocks is not None:
            return blocks.synaptic_input(s)
        if self.connectome.synapses:
            weights = self._weights
            if self._csr is None or self._csr[0] is not weights:
                from .sparse import transport

                self._csr = weights, transport(self.connectome, weights)
            if self._csr[1] is not None:
                return np.asarray(self._csr[1].dot(s.T).T)
        synaptic_input = np.zeros_like(s)
        if self.connectome.synapses:
            messages = s[:, self.connectome.pre] * self._weights
            synaptic_input[:, self._neurons_with_synaptic_input] = np.add.reduceat(
                messages, self._starts, axis=1
            )
        return synaptic_input

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
        capture: _Capture | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray]:
        neuron_model = self.neuron_model
        adapt = neuron_model.adaptation
        traj = np.zeros((steps, *v.shape)) if want else None
        masked = bool((keep != 1.0).any())
        standing = drive + self.bias  # the part of every neuron's drive that does not move
        blocks = None if self._blocks is None else BlockTransport(self.layout, self._blocks)
        s = neuron_model.activation(v)
        if masked:
            s *= keep
        if capture is not None:
            capture.step(v, s, a)
        taken = 0
        activity_change = np.zeros(len(v))
        for t in range(steps):
            total = self._synaptic_input(s, blocks)
            total += standing
            if adapt is not None:
                total -= adapt.strength * a
            if nudge is not None:
                total += nudge.drive(s)
            total -= v
            total *= neuron_model.dt
            v += total  # neuron-local update: v <- v + dt (-v + total)
            if masked:
                v *= keep
            previous = s
            s = neuron_model.activation(v)
            if masked:
                s *= keep
            if adapt is not None:
                a += (s - a) / adapt.tau_steps
            if capture is not None:
                capture.step(v, s, a)
            if traj is not None:
                traj[t] = s
            taken = t + 1
            movement = np.abs(s - previous)
            activity_change += movement.sum(axis=1)
            if tolerance is not None and float(movement.max()) < tolerance:
                break
        if traj is not None:
            traj = traj[:taken]
        return v, a, s, traj, taken, activity_change

    def readings(
        self, state: BrainState, names: Sequence[str], i: int = 0, level: float = 0.5
    ) -> dict[str, dict[str, float]]:
        """Mean and fraction-active (at ``level``) of named sets, for batch row ``i``."""
        return {
            name: {
                "mean": state.mean(self.connectome.populations[name], i),
                "fraction": state.fraction_active(self.connectome.populations[name], level, i),
            }
            for name in names
        }

    def _is_dense(self) -> bool:
        return self._blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "connectome": self.connectome.summary(),
            "neuron_model": self.neuron_model.to_dict(),
            "backend": self.backend,
            "transport": "dense" if self._is_dense() else "segmented",
            "sparse_kernel": (
                None
                if self._csr is None
                else "scipy_csr"
                if self._csr[1] is not None
                else "numpy_segmented"
            ),
            "layout": self.layout.to_dict(),
            "efficacy_changed": int((self.efficacy != self.connectome.sign).sum()),
            "log_gain_nonzero": int((self.log_gain != 0).sum()),
            "bias_nonzero": int((self.bias != 0).sum()),
        }


class _TorchKernel:
    """Gather-scatter settling on a torch device; float32 on MPS, float64 elsewhere."""

    def __init__(
        self,
        connectome: Connectome,
        efficacy: np.ndarray,
        gain_pre: np.ndarray,
        bias: np.ndarray,
        neuron_model: NeuronModel,
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
        if self.device.type == "mps" and self.dtype == torch.float64:
            raise ValueError("MPS has no float64; use precision='float32' or another device")
        neuron_model._validate_precision("float32" if self.dtype == torch.float32 else "float64")
        # The parameters, kept on the device in float64 (float32 on MPS, which has no float64)
        # so a learner can move them there without a host round trip per update.
        self.param_dtype = torch.float32 if self.device.type == "mps" else torch.float64
        self.gain_pre = torch.from_numpy(np.ascontiguousarray(gain_pre)).to(
            self.device, self.param_dtype
        )
        self.neuron_model = neuron_model
        self.n = connectome.n
        self.layout = layout
        self.sources = [] if layout is None else layout.sources()
        self._connectome = connectome  # for the per-row contrast's edge index, made on first use
        self._row_index: Any = None
        self.blocks: list[Any] = []
        self.index: Any = None  # the layout's edge index on the device, shared across rebuilds
        if layout is not None:  # block products per step instead of a gather and a scatter
            self.index = self._device_index(layout)
            self.flat = torch.zeros(layout.size, dtype=self.dtype, device=self.device)
            self.blocks = []
            for k in range(layout.pairs):
                a0, a1, b0, b1 = layout.bounds(k)
                block = self.flat[int(layout.offset[k]) : int(layout.offset[k + 1])]
                self.blocks.append(block.view(a1 - a0, b1 - b0))
        else:  # per-synapse arrays serve only the gather-scatter path
            self.pre = torch.from_numpy(connectome.pre.copy()).to(self.device)
            self.post = torch.from_numpy(connectome.post.copy()).to(self.device)
        self.set_parameters(
            torch.from_numpy(np.ascontiguousarray(efficacy)).to(self.device, self.param_dtype),
            torch.from_numpy(np.ascontiguousarray(bias)).to(self.device, self.param_dtype),
        )

    def set_parameters(self, scale: Any, bias: Any) -> None:
        """New parameters, as device tensors: the block weights follow, in place."""
        torch = self.torch
        if tuple(scale.shape) != tuple(self.gain_pre.shape) or tuple(bias.shape) != (self.n,):
            raise ValueError("device parameters must match synapse and neuron shapes")
        weights = (self.gain_pre * scale).to(self.dtype)
        live_bias = bias.to(self.dtype)
        if not bool(torch.stack([
            torch.isfinite(value).all() for value in (scale, bias, weights, live_bias)
        ]).all()):
            raise ValueError(
                "device parameters and effective weights must be finite at runtime precision"
            )
        flat = None
        if self.layout is not None and self.layout.parallel_synapses:
            flat = torch.zeros_like(self.flat)
            flat.index_add_(0, self.index, weights)
            if not bool(torch.isfinite(flat).all()):
                raise ValueError("summed effective synaptic weights must be finite")
        # Validation precedes every write to the shared live kernel and blocks.
        self.scale = scale
        self.bias_param = bias
        if self.layout is not None:
            if flat is not None:
                self.flat.copy_(flat)
            else:
                self.flat[self.index] = weights
        else:
            self.w = weights
        self.bias = live_bias

    def host_scale(self) -> np.ndarray:
        return np.asarray(self.scale.cpu().double().numpy())

    def host_bias(self) -> np.ndarray:
        return np.asarray(self.bias_param.cpu().double().numpy())

    def _device_index(self, layout: Layout) -> Any:
        """The layout's edge index on this device, kept on the layout (one upload per device)."""
        held = layout.__dict__.setdefault("_device_index", {})
        key = str(self.device)
        if key not in held:
            held[key] = self.torch.from_numpy(layout.edge_index).to(self.device)
        return held[key]

    def _synaptic_input(self, s: Any, previous: Any, cache: list[Any]) -> Any:
        """The block transport on the device, reusing the products of still ranges."""
        torch, lay = self.torch, self.layout
        assert lay is not None
        out = torch.zeros_like(s)
        moved = [True] * lay.ranges
        # torch.equal returns a host bool. On accelerators its synchronization costs
        # more than these source products on small settling graphs. Recompute there;
        # CPU keeps the exact cache, with no change to the neuron equations.
        if previous is not None and self.device.type == "cpu":
            for r in self.sources:
                a0, a1 = int(lay.starts[r]), int(lay.starts[r + 1])
                moved[r] = not torch.equal(s[:, a0:a1], previous[:, a0:a1])
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            if self.device.type == "cuda":
                # Accumulate directly: avoid a temporary product and a separate
                # addition kernel per block. Keep the declared block order.
                out[:, b0:b1].addmm_(s[:, a0:a1], self.blocks[k])
                continue
            if cache[k] is None or moved[int(lay.pair_pre[k])]:
                cache[k] = s[:, a0:a1] @ self.blocks[k]
            out[:, b0:b1] += cache[k]
        return out

    def _activation(self, v: Any) -> Any:
        torch, neuron_model = self.torch, self.neuron_model
        if self._rest is None:  # one scalar on the device, made once
            self._rest = torch.tensor(
                neuron_model.rest_emission, dtype=self.dtype, device=self.device
            )
        rest = self._rest
        r = torch.sigmoid(neuron_model.slope * (v - neuron_model.threshold)) - rest
        # One leaky-rectifier primitive replaces separate positive/negative arrays.
        # This is the same piecewise-linear map, including exact silence at rest.
        rest_value = neuron_model.rest_emission
        negative_slope = neuron_model.leak * (1.0 - rest_value) / rest_value
        return torch.nn.functional.leaky_relu(r, negative_slope=negative_slope) / (1.0 - rest)

    def _nudge_tensors(
        self, nudge: Nudge | None, batch: int
    ) -> tuple[Any, Any, list[Any], Any, Any, Any]:
        """The nudge's target, mask, softmax groups and row weights as device tensors."""
        torch = self.torch
        if nudge is None:
            return None, None, [], None, None, None

        def to(x: np.ndarray) -> Any:
            y = np.ascontiguousarray(x, dtype=float)
            if not y.flags.writeable:
                y = y.copy()
            return torch.from_numpy(y).to(self.device, self.dtype)

        target = to(np.broadcast_to(nudge.target, (batch, self.n)))
        mask = to(nudge.mask)
        groups = [
            (torch.from_numpy(members).to(self.device), _group_temperature(nudge, members))
            for members in _softmax_groups(nudge)
        ] if nudge.softmax_temperature is not None else []
        weight = None
        if nudge.weight is not None:
            weight = to(np.asarray(nudge.weight, dtype=float))[:, None]
        anchor = gain = None
        if nudge.anchor is not None:
            anchor = to(np.broadcast_to(nudge.anchor, (batch, self.n)))
            gain = to(np.asarray(nudge.anchor_gain))
        return target, mask, groups, weight, anchor, gain

    def _push(
        self,
        s: Any,
        nudge: Nudge,
        target: Any,
        mask: Any,
        groups: list[Any],
        weight: Any,
        anchor: Any,
        anchor_gain: Any,
    ) -> Any:
        """The nudge drive at activation ``s``: ``Nudge.drive`` on the device."""
        torch = self.torch
        if nudge.softmax_temperature is None:
            push = nudge.beta * (target - s) * mask
        else:  # one softmax per group of competing neurons
            push = torch.zeros_like(s)
            for members, temperature in groups:
                logits = s[:, members]
                tiny = torch.finfo(logits.dtype).tiny
                if temperature < tiny:
                    # Divide in representable pieces: zero remains zero, negative
                    # overflow remains a legitimate zero-probability logit.
                    logits = logits - logits.max(dim=1, keepdim=True).values
                    while temperature < tiny:
                        logits = logits / tiny
                        temperature /= tiny
                p = torch.softmax(logits / temperature, dim=1)
                push[:, members] = nudge.beta * (target[:, members] - p) * mask[members]
        if weight is not None:
            push = push * weight
        if anchor is not None:
            push = push + anchor_gain * (anchor - s)
        return push

    def residual(
        self, drive: np.ndarray, keep: np.ndarray, nudge: Nudge | None, handle: dict[str, Any]
    ) -> np.ndarray:
        """``Brain.residual`` for a state on this device; the same equations, one row each."""
        torch, neuron_model = self.torch, self.neuron_model

        def to(x: np.ndarray) -> Any:
            return torch.from_numpy(np.ascontiguousarray(x, dtype=float)).to(
                self.device, self.dtype
            )

        v, a = handle["v"], handle["a"]
        batch = v.shape[0]
        d, k = to(drive), to(keep)
        target, mask, groups, weight, anchor, anchor_gain = self._nudge_tensors(nudge, batch)
        with torch.no_grad():
            s = self._activation(v) * k
            if self.layout is not None:
                synaptic_input = self._synaptic_input(s, None, [None] * self.layout.pairs)
            else:
                zeros = torch.zeros(batch, self.n, dtype=self.dtype, device=self.device)
                synaptic_input = zeros.index_add_(1, self.post, s[:, self.pre] * self.w)
            error = synaptic_input + d + self.bias - v
            adapt = neuron_model.adaptation
            if adapt is not None:
                error = error - adapt.strength * a
            if nudge is not None:
                error = error + self._push(
                    s, nudge, target, mask, groups, weight, anchor, anchor_gain
                )
            error = k * error + (k - 1.0) * v / neuron_model.dt
            result = error.abs().amax(dim=1)
            if adapt is not None:
                result = torch.maximum(result, (s - a).abs().amax(dim=1))
        out = result.cpu().double().numpy()
        return np.where(np.isfinite(out), out, np.inf)

    def keep_rows(self, handle: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
        """The state with every row where ``keep`` is false returned to rest, still on the device.

        A stream whose episode ended starts its next life from rest; the other rows continue
        untouched, and none of them comes to the host for it.
        """
        torch = self.torch
        k = torch.from_numpy(np.ascontiguousarray(keep, dtype=float)).to(self.device, self.dtype)
        k = k[:, None]
        tensors = {key: handle[key] * k for key in ("v", "a", "s")}

        def fetch(key: str) -> np.ndarray:
            return np.asarray(tensors[key].cpu().double().numpy())

        return {"kernel": self.backend_name, "holder": self, "fetch": fetch, **tensors}

    def contrast_tensors(self, s_plus: Any, s_minus: Any) -> tuple[Any, Any]:
        """The block Gram contrast on the device, per synapse and per neuron, as tensors."""
        torch, lay = self.torch, self.layout
        assert lay is not None
        flat = torch.zeros(lay.size, dtype=self.dtype, device=self.device)
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            a_plus, a_minus = s_plus[:, a0:a1], s_minus[:, a0:a1]
            b_plus, b_minus = s_plus[:, b0:b1], s_minus[:, b0:b1]
            # Difference the phases before the products: subtracting two large Gram
            # matrices can erase a small contrast, especially in float32. The identity
            # A+^T(B+ - B-) + (A+ - A-)^T B- uses the same two matrix products.
            block = a_plus.T @ (b_plus - b_minus)
            # Keep the CPU shortcut; a GPU never needs a device-to-host boolean.
            if self.device.type != "cpu" or not torch.equal(a_plus, a_minus):
                block = block + (a_plus - a_minus).T @ b_minus
            flat[lay.offset[k] : lay.offset[k + 1]] = block.reshape(-1)
        edges = flat[self.index].to(self.param_dtype)
        neurons = (s_plus - s_minus).sum(dim=0).to(self.param_dtype)
        return edges, neurons

    def contrast_rows(self, s_plus: Any, s_minus: Any) -> tuple[Any, Any]:
        """The contrast per row, ``(batch, edges)`` and ``(batch, neurons)``, as tensors: each
        stream's own product of the two phases, for an eligibility trace kept per stream."""
        torch = self.torch
        if self._row_index is None:
            self._row_index = (
                torch.from_numpy(self._connectome.pre.copy()).to(self.device),
                torch.from_numpy(self._connectome.post.copy()).to(self.device),
            )
        pre, post = self._row_index
        a_plus, a_minus = s_plus[:, pre], s_minus[:, pre]
        b_plus, b_minus = s_plus[:, post], s_minus[:, post]
        edges = a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus
        edges = edges.to(self.param_dtype)
        neurons = (s_plus - s_minus).to(self.param_dtype)
        return edges, neurons

    def contrast(self, s_plus: Any, s_minus: Any) -> tuple[np.ndarray, np.ndarray]:
        """The block Gram contrast on the device; only the per-synapse result comes back."""
        edges, neurons = self.contrast_tensors(s_plus, s_minus)
        return edges.cpu().double().numpy(), neurons.cpu().double().numpy()

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
        state: BrainState | None = None,
        capture: _Capture | None = None,
    ) -> tuple[Any, Any, Any, np.ndarray | None, int, np.ndarray, Any]:
        torch, neuron_model = self.torch, self.neuron_model

        def to(x: np.ndarray) -> Any:  # no host copy when already contiguous float64
            y = np.ascontiguousarray(x, dtype=float)
            if not y.flags.writeable:  # a broadcast view: torch wants a writable buffer
                y = y.copy()
            return torch.from_numpy(y).to(self.device, self.dtype)

        handle = state.device if state is not None and state.device is not None else None
        if handle is not None and handle.get("holder") is self:
            v, a = handle["v"].clone(), handle["a"].clone()  # the state never left the device
        else:
            v, a = to(v0), to(a0)
        d, k = to(drive), to(keep)
        batch = v.shape[0]
        adapt = neuron_model.adaptation
        target, mask, groups, weight, anchor, anchor_gain = self._nudge_tensors(nudge, batch)
        traj = []
        taken = 0
        activity_change = torch.zeros(batch, dtype=self.dtype, device=self.device)
        cache: list[Any] = [None] * (0 if self.layout is None else self.layout.pairs)
        previous = None
        with torch.no_grad():
            s = self._activation(v) * k
            if capture is not None:
                capture.step(v.cpu().numpy(), s.cpu().numpy(), a.cpu().numpy())
            for t in range(steps):
                if self.layout is not None:
                    synaptic_input = self._synaptic_input(s, previous, cache)
                else:
                    zeros = torch.zeros(batch, self.n, dtype=self.dtype, device=self.device)
                    synaptic_input = zeros.index_add_(1, self.post, s[:, self.pre] * self.w)
                total = synaptic_input + d + self.bias
                if adapt is not None:
                    total -= adapt.strength * a
                if nudge is not None:
                    total += self._push(s, nudge, target, mask, groups, weight, anchor, anchor_gain)
                v = (v + neuron_model.dt * (-v + total)) * k
                previous = s
                s = self._activation(v) * k
                if adapt is not None:
                    a = a + (s - a) / adapt.tau_steps
                if capture is not None:
                    capture.step(v.cpu().numpy(), s.cpu().numpy(), a.cpu().numpy())
                if want:
                    traj.append(s.detach().cpu().double().numpy())
                taken = t + 1
                movement = (s - previous).abs()
                activity_change = activity_change + movement.sum(dim=1)
                if tolerance is not None and float(movement.max()) < tolerance:
                    break
        tensors = {"v": v, "a": a, "s": s}

        def fetch(key: str) -> np.ndarray:  # the host copy, made when something reads it
            return np.asarray(tensors[key].cpu().double().numpy())

        # fetch closes over the tensors, never over the handle: a cycle there would hold the
        # device memory of every settled state until a garbage-collection pass
        held = {"kernel": self.backend_name, "holder": self, "fetch": fetch, **tensors}
        return (
            None,
            None,
            None,
            (np.stack(traj) if traj else np.empty((0, batch, self.n))) if want else None,
            taken,
            activity_change.cpu().double().numpy(),
            held,
        )


class _MlxKernel:
    """Block settling on Apple silicon through MLX: float32, unified memory, lazy graphs."""

    def __init__(
        self,
        connectome: Connectome,
        weights: np.ndarray,
        bias: np.ndarray,
        neuron_model: NeuronModel,
        layout: Layout,
    ) -> None:
        import mlx.core as mx

        neuron_model._validate_precision("float32")
        self.mx = mx
        self.backend_name = "mlx"
        self.neuron_model = neuron_model
        self.n = connectome.n
        self.layout = layout
        with np.errstate(over="ignore", invalid="ignore"):
            live_bias = bias.astype(np.float32)
            flat = layout.flat(weights).astype(np.float32)
        if not np.isfinite(live_bias).all() or not np.isfinite(flat).all():
            raise ValueError(
                "device parameters and effective weights must be finite at runtime precision"
            )
        self.bias = mx.array(live_bias)
        self.blocks = [
            mx.array(np.ascontiguousarray(b, dtype=np.float32)) for b in layout.blocks(flat)
        ]
        self.by_post: list[list[int]] = [[] for _ in range(layout.ranges)]
        for k in range(layout.pairs):
            self.by_post[int(layout.pair_post[k])].append(k)
        self.sources = layout.sources()
        self.edge_index = mx.array(layout.edge_index)

    def _activation(self, v: Any) -> Any:
        mx, neuron_model = self.mx, self.neuron_model
        rest = neuron_model.rest_emission
        r = mx.sigmoid(neuron_model.slope * (v - neuron_model.threshold)) - rest
        s = mx.maximum(r, 0.0) / (1.0 - rest)
        if neuron_model.leak:
            s = s + neuron_model.leak * mx.minimum(r, 0.0) / rest
        return s

    def _synaptic_input(self, s: Any, previous: Any, cache: list[Any]) -> Any:
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
        """The block Gram contrast on the device; only the per-synapse result comes back."""
        mx, lay = self.mx, self.layout
        pieces = []
        for k in range(lay.pairs):
            a0, a1, b0, b1 = lay.bounds(k)
            a_plus, a_minus = s_plus[:, a0:a1], s_minus[:, a0:a1]
            b_plus, b_minus = s_plus[:, b0:b1], s_minus[:, b0:b1]
            block = a_plus.T @ (b_plus - b_minus) + (a_plus - a_minus).T @ b_minus
            pieces.append(block.reshape(-1))
        flat = mx.concatenate(pieces) if pieces else mx.zeros((0,), dtype=mx.float32)
        edges = np.array(flat[self.edge_index], dtype=np.float64)
        neurons = np.array((s_plus - s_minus).sum(axis=0), dtype=np.float64)
        return edges, neurons

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
        state: BrainState | None = None,
        capture: _Capture | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray, Any]:
        mx, neuron_model = self.mx, self.neuron_model

        def to(x: np.ndarray) -> Any:
            return mx.array(np.asarray(x, dtype=np.float32))

        handle = state.device if state is not None and state.device is not None else None
        if handle is not None and handle.get("holder") is self:
            v, a = handle["v"], handle["a"]
        else:
            v, a = to(v0), to(a0)
        d, k = to(drive), to(keep)
        batch = v.shape[0]
        masked = bool((keep != 1.0).any())
        adapt = neuron_model.adaptation
        target = mask = weight = anchor = anchor_gain = None
        groups: list[Any] = []
        if nudge is not None:
            target = to(np.broadcast_to(nudge.target, (batch, self.n)))
            mask = to(nudge.mask)
            groups = [
                (mx.array(members), _group_temperature(nudge, members))
                for members in _softmax_groups(nudge)
            ] if nudge.softmax_temperature is not None else []
            if nudge.weight is not None:
                weight = to(np.asarray(nudge.weight, dtype=float))[:, None]
            if nudge.anchor is not None:
                anchor = to(np.broadcast_to(nudge.anchor, (batch, self.n)))
                anchor_gain = to(np.asarray(nudge.anchor_gain))
        traj = []
        taken = 0
        activity_change = mx.zeros((batch,), dtype=mx.float32)
        cache: list[Any] = [None] * self.layout.pairs
        previous = None
        s = self._activation(v)
        if masked:
            s = s * k
        standing = d + self.bias
        if capture is not None:
            capture.step(np.array(v), np.array(s), np.array(a))
        for t in range(steps):
            synaptic_input = self._synaptic_input(s, previous, cache)
            total = synaptic_input + standing
            if adapt is not None:
                total = total - adapt.strength * a
            if nudge is not None:
                assert target is not None and mask is not None
                if nudge.softmax_temperature is None:
                    push = nudge.beta * (target - s) * mask
                else:  # one softmax per group of competing neurons
                    push = mx.zeros_like(s)
                    for members, temperature in groups:
                        logits = s[:, members]
                        tiny = float(np.finfo(np.float32).tiny)
                        if temperature < tiny:
                            logits = logits - mx.max(logits, axis=1, keepdims=True)
                            while temperature < tiny:
                                logits = logits / tiny
                                temperature /= tiny
                        p = mx.softmax(logits / temperature, axis=1)
                        push[:, members] = nudge.beta * (target[:, members] - p) * mask[members]
                if weight is not None:
                    push = push * weight
                if anchor is not None:
                    push = push + anchor_gain * (anchor - s)
                total = total + push
            v = v + neuron_model.dt * (-v + total)
            if masked:
                v = v * k
            previous = s
            s = self._activation(v)
            if masked:
                s = s * k
            if adapt is not None:
                a = a + (s - a) / adapt.tau_steps
            movement = mx.abs(s - previous)
            activity_change = activity_change + movement.sum(axis=1)
            mx.eval(v, s, a, activity_change)  # one graph per step; the tolerance needs the number
            if capture is not None:
                capture.step(np.array(v), np.array(s), np.array(a))
            if want:
                traj.append(np.array(s, dtype=np.float64))
            taken = t + 1
            if tolerance is not None and float(movement.max().item()) < tolerance:
                break
        return (
            np.array(v, dtype=np.float64),
            np.array(a, dtype=np.float64),
            np.array(s, dtype=np.float64),
            (np.stack(traj) if traj else np.empty((0, batch, self.n))) if want else None,
            taken,
            np.array(activity_change, dtype=np.float64),
            {"kernel": "mlx", "holder": self, "v": v, "a": a, "s": s},
        )
