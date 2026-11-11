"""The brain: neurons and synapses settling under a stimulus.

One ``Brain`` holds a connectome and a neuron model and runs the neuron update for a
number of steps from rest, or from a given state, under a stimulus. Two
backends do the same arithmetic:

* ``"cpu"``: NumPy in float64. The transport is one segmented sum per
  step, a scatter of every synapse's message into its neuron's synaptic input, or a
  SciPy CSR product when SciPy is installed (``cadence.sparse``). For
  connectomes whose dense blocks fit (at most ``dense_limit`` squared entries,
  see ``cadence.blocks``) the same sum is done as block matrix products,
  reusing the products of every source range whose activation did not change since
  the previous step. Blocks are far faster when the interpreter overhead of
  the scatter would dominate. This is the receipt-grade backend:
  deterministic, exact to rounding, and the one the neuron-by-neuron
  reference is compared against.
* ``"torch"``: the same block products when the blocks fit, else a scatter with
  ``index_add_``, on whatever device torch offers; by default CUDA in float64,
  Apple silicon in float32 (MPS has no float64) and CPU in float64. Use it for
  interactive work and for very large connectomes, and keep a conformance
  check against ``"cpu"``; the fly brain has neurons on knife edges where
  float32 flips a bistable readout.

Three kinds of parameter sit on a brain, all defaulting to the connectome:
``efficacy``, one signed number per synapse (the connectome's sign unless a
learner has moved it); ``log_gain``, one per neuron on its outgoing
synapses, how a caller declares a stimulus rate or a region's gain; and
``bias``, one per neuron. The effective drive of synapse ``e`` per unit
presynaptic activation is ``gain * count[e] * efficacy[e] *
exp(log_gain[pre[e]])``.

Settling runs on a batch of stimuli at once; ``settle`` is the one-stimulus
convenience. A ``Nudge`` adds a drive that pulls chosen neurons toward a
target activation, which is how the learning rule's second phase enters.
"""

from __future__ import annotations

import copy
import os
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from .blocks import BlockTransport, Layout
from .blocks import layout as make_layout
from .connectome import Connectome, _neuron_indices
from .neuron import NeuronModel
from .recording import _Capture, _capture

__all__ = [
    "Apical", "Brain", "BrainState", "Equilibrium", "Nudge",
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
    # A graph with an apical compartment: the gain each pool's apical input passes at, one per
    # row and pool, held through the nudged settle (the free phase's, ``Brain.apical_gain``)
    apical_gain: np.ndarray | None = None  # nonnegative (batch, pools)

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
        if self.apical_gain is not None:
            held = np.asarray(self.apical_gain, dtype=float)
            if held.ndim != 2 or not np.isfinite(held).all() or (held < 0).any():
                raise ValueError("apical_gain must be finite and nonnegative, one per row and pool")
            object.__setattr__(self, "apical_gain", held.copy())

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


# -- the apical compartment: top-down input normalized to the bottom-up input it joins
#
# A cortical pyramidal neuron hears the area below it on its basal dendrites and the areas
# above it on its apical tuft, and the tuft's effect on the cell is gain controlled: it
# modulates and adds to what the basal input drives rather than replacing it (Larkum 2013;
# divisive normalization, Carandini and Heeger 2012). In a brain whose regions are joined by
# reciprocal synapse pairs, each region's top-down half is the other region's bottom-up half
# run backwards, and a plain sum lets the loop through the pair take a region over as the
# weights grow: the settle slows near the loop's critical gain, and the region carries an
# echo of the answer instead of a representation of its input.
#
# ``Apical`` names the synapses that land on the apical compartment and the pools of neurons
# that normalize them. In every sweep of a settle, for each row and each pool, it reads every
# member's apical input ``u`` (the summed input of its apical synapses) and its basal input
# ``b`` (everything else the neuron receives: the other synapses, the stimulus, the bias, a
# nudge), and the neuron receives ``b`` plus
#
#     fraction * rms(b) * u / sqrt(rms(u)^2 + (floor * rms(b))^2),
#
# the root mean squares taken over the pool. The top-down direction is kept; its magnitude
# is at most ``fraction`` times the pool's basal drive whatever the weights grow to, and a
# weak top-down input passes linearly with gain ``fraction / floor``. The fixed point is the
# settle's: a nudge on the output moves the regions above and their apical input follows, so
# every synapse's two-phase contrast is read as usual.

# The founder semi-saturation of the normalization, in units of the pool's basal RMS: a
# top-down input as large as the basal input passes at 1/sqrt(2) of its saturated share.
APICAL_FLOOR = 1.0
_APICAL_TINY = 1e-24  # under the square root: a pool with no input at all receives none


@dataclass(frozen=True, eq=False)
class Apical:
    """The synapses that land on the apical compartment, and how each pool normalizes them.

    ``synapses`` is a bool per synapse; ``groups`` an integer per neuron naming its pool
    (``-1``: no pool, so it may receive no apical synapse). ``fraction`` bounds the apical
    drive's RMS over a pool to that share of the pool's basal RMS; ``floor`` is the
    normalization's semi-saturation constant in units of the basal RMS.
    """

    synapses: np.ndarray
    groups: np.ndarray
    fraction: float
    floor: float = APICAL_FLOOR
    members: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        synapses = np.asarray(self.synapses)
        groups = np.asarray(self.groups)
        if synapses.ndim != 1 or synapses.dtype != np.bool_:
            raise ValueError("apical synapses must be a boolean vector, one entry per synapse")
        if (
            groups.ndim != 1
            or not np.issubdtype(groups.dtype, np.integer)
            or (groups < -1).any()
            or (groups > np.iinfo(np.int64).max).any()
        ):
            raise ValueError("apical groups must be an integer per neuron, -1 for none")
        for name in ("fraction", "floor"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"apical {name} must be a real number")
        if not np.isfinite(self.fraction) or self.fraction <= 0:
            raise ValueError("apical fraction must be finite and positive")
        if not np.isfinite(self.floor) or self.floor <= 0:
            raise ValueError("apical floor must be finite and positive")
        synapses = synapses.copy()
        groups = groups.astype(np.int64, copy=True)
        synapses.setflags(write=False)
        groups.setflags(write=False)
        object.__setattr__(self, "synapses", synapses)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "fraction", float(self.fraction))
        object.__setattr__(self, "floor", float(self.floor))
        members: list[Any] = []
        for pool in np.unique(groups[groups >= 0]):
            index = np.flatnonzero(groups == pool)
            if index[-1] - index[0] + 1 == len(index):  # one contiguous run: a view, no gather
                members.append(slice(int(index[0]), int(index[-1]) + 1))
            else:
                members.append(index)
        object.__setattr__(self, "members", tuple(members))

    def check(self, synapses: int, neurons: int, post: np.ndarray) -> None:
        """Refuse a spec that does not fit a connectome of these sizes and post neurons."""
        if self.synapses.shape != (synapses,) or self.groups.shape != (neurons,):
            raise ValueError("apical synapses and groups must match the connectome")
        if self.synapses.any() and (self.groups[np.asarray(post)[self.synapses]] < 0).any():
            raise ValueError("every apical synapse must land on a neuron of a pool")

    def gain(self, basal: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """The normalization's gain ``(batch, pools)``: ``fraction * rms(b) / sqrt(rms(u)^2 +
        (floor * rms(b))^2)`` per row and pool."""
        fraction, floor2 = self.fraction, self.floor * self.floor
        out = np.empty((len(basal), len(self.members)))
        for k, members in enumerate(self.members):
            b = basal[:, members]
            u = raw[:, members]
            basal2 = np.mean(b * b, axis=1)
            raw2 = np.mean(u * u, axis=1)
            out[:, k] = fraction * np.sqrt(basal2) / np.sqrt(raw2 + floor2 * basal2 + _APICAL_TINY)
        return out

    def drive(
        self, basal: np.ndarray, raw: np.ndarray, held: np.ndarray | None = None
    ) -> np.ndarray:
        """The apical drive ``(batch, n)``: the raw apical input at the normalization's gain,
        or at the ``held`` gain ``(batch, pools)`` of a nudged settle."""
        if held is not None and np.shape(held) != (len(raw), len(self.members)):
            raise ValueError("a nudge's apical_gain needs one entry per row and pool")
        gain = self.gain(basal, raw) if held is None else held
        out = np.zeros_like(raw)
        for k, members in enumerate(self.members):
            out[:, members] = gain[:, k : k + 1] * raw[:, members]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "synapses": int(self.synapses.sum()),
            "pools": len(self.members),
            "fraction": self.fraction,
            "floor": self.floor,
        }


# the blocks that hold basal and apical synapses, per compartment: fixed by the compartment and
# the layout, so found once for every brain that shares both (each lesson makes a new brain)
_APICAL_BLOCKS: weakref.WeakKeyDictionary[Apical, tuple[Layout, list[int], list[int]]] = (
    weakref.WeakKeyDictionary()
)


def _apical_blocks(apical: Apical, layout: Layout) -> tuple[list[int], list[int]]:
    """The blocks of ``layout`` that hold basal synapses and those that hold apical ones, in
    block order."""
    held = _APICAL_BLOCKS.get(apical)
    if held is None or held[0] is not layout:
        block = np.asarray(layout.block_of_edge)
        mask = apical.synapses
        basal = np.bincount(block[~mask], minlength=layout.pairs) > 0
        top_down = np.bincount(block[mask], minlength=layout.pairs) > 0
        held = (layout, [int(k) for k in np.flatnonzero(basal)],
                [int(k) for k in np.flatnonzero(top_down)])
        _APICAL_BLOCKS[apical] = held
    return held[1], held[2]


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
    discarded_steps: int = 0  # CUDA tolerance lookahead beyond the returned state

    @property
    def attempted_steps(self) -> int:
        """Executed sweeps of this settle, including discarded CUDA tolerance lookahead."""
        return self.steps + self.discarded_steps

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


def _copy_state(state: BrainState | None) -> BrainState | None:
    if state is None:
        return None
    return BrainState(
        state.v.copy(),
        state.activation.copy(),
        state.adaptation.copy(),
        state.steps,
        activity_change=None if state.activity_change is None else state.activity_change.copy(),
        discarded_steps=state.discarded_steps,
    )


@dataclass(frozen=True)
class Equilibrium:
    """A bounded solve and its measured equation error, one entry per batch row.

    ``steps`` counts local sweeps in this call. ``state.steps`` is the last chunk's count.
    Convergence here means a small residual; ``Brain.residual`` says what that certifies.
    ``qualified`` names the same rows as ``converged``.
    """

    state: BrainState
    residual: np.ndarray
    steps: int
    tolerance: float
    residual_checks: int = 0  # additional transport evaluations, outside the local sweeps
    damping_halvings: int = 0  # absolute final halving index of this bounded solve
    stagnation_checks: int = 0  # complete-state comparisons, with no synaptic transport
    initial_damping: int = 0  # initial halving index, relative to the unchanged live model

    @property
    def relative_damping_halvings(self) -> int:
        """Further halvings after the explicitly selected initial numerical step."""
        return self.damping_halvings - self.initial_damping

    @property
    def converged(self) -> np.ndarray:
        return np.asarray(self.residual <= self.tolerance)

    @property
    def qualified(self) -> np.ndarray:
        """Rows meeting the solver's check: the full residual within ``tolerance``."""
        return self.converged


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
        apical: Apical | None = None,
    ) -> None:
        self.connectome = connectome
        self.neuron_model = neuron_model
        self.backend: Backend = backend
        # the synapses that land on an apical compartment, normalized per pool (``Apical``);
        # None sums every synapse alike
        if apical is not None:
            if not isinstance(apical, Apical):
                raise ValueError("apical must be an Apical or None")
            apical.check(connectome.synapses, connectome.n, connectome.post)
            if backend == "mlx":
                raise ValueError("the mlx backend does not settle an apical compartment")
        self.apical = apical
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
                apical,
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

        On the ``"cpu"`` backend, new efficacies and biases alone make a shallow copy with
        the new weights, whose blocks are remade on first use. A new ``log_gain``, or
        another backend, builds a new brain.
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
            apical=self.apical,
        )

    @property
    def weights(self) -> np.ndarray:
        """Effective drive per unit presynaptic activation, one per synapse."""
        result = self._weights.view()
        result.setflags(write=False)
        return result

    def dense(self) -> np.ndarray:
        """The synapse matrix ``W[pre, post]``: the synaptic input of a batch ``s`` is ``s @ W``.

        Built on demand, for exporting the net to another runtime such as a browser page. A
        graph with an apical compartment is refused: its top-down input is normalized per
        region, which the matrix does not carry.
        """
        if self.apical is not None:
            raise ValueError(
                "dense() is the plain synaptic sum; a graph with an apical compartment "
                "normalizes its top-down input, which the matrix does not carry"
            )
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
            discarded_steps=out.discarded_steps,
        )

    def _with_dt(self, dt: float) -> Brain:
        """A numerical copy on current parameters; boundary subclasses must override.

        This does not change the live model. A subclass adding equation state
        must carry that state into its numerical candidates too.
        """
        return Brain(
            self.connectome, self.neuron_model.replace(dt=dt), backend=self.backend,
            efficacy=self.efficacy, log_gain=self.log_gain, bias=self.bias,
            device=str(self._torch.device) if self._torch is not None else None,
            dense_limit=self.dense_limit, layout=self.layout, precision=self.precision,
            apical=self.apical,
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
        initial_damping: int = 0,
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
        Every attempt checks the original equations; a halved step changes only the
        numerical method. Repetition never qualifies an answer.

        ``initial_damping`` optionally skips initial attempts, without changing
        the live model or the shared budget. It lies in ``0..damping``; nonzero
        values currently require CPU float64. Results report the absolute final
        halving index and the additional ``relative_damping_halvings`` separately.
        """
        for name, value, minimum in (
            ("budget", budget, 0), ("chunk", chunk, 1), ("damping", damping, 0),
            ("initial_damping", initial_damping, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if initial_damping > damping:
            raise ValueError("initial_damping must lie in 0..damping")
        budget, chunk, damping, initial_damping = map(
            int, (budget, chunk, damping, initial_damping))
        if initial_damping and (self.backend != "cpu" or self.precision not in (None, "float64")):
            raise ValueError("nonzero initial_damping requires CPU float64")
        if (initial_damping and type(self) is not Brain
                and type(self)._with_dt is Brain._with_dt):
            raise ValueError("a boundary graph must preserve its numerical copies")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if damping:
            attempts = min(damping - initial_damping + 1, max(1, budget))
            minimum_rule = self.neuron_model.replace(
                dt=float(np.ldexp(self.neuron_model.dt, -(initial_damping + attempts - 1)))
            )
            dtype = "float32" if (self._mlx is not None or (
                self._torch is not None and self._torch.dtype == self._torch.torch.float32
            )) else "float64"
            minimum_rule._validate_precision(dtype)
            used = checks = stagnation_checks = 0
            current_state = state
            for relative in range(attempts):
                halving = initial_damping + relative
                portion = (budget - used + attempts - relative - 1) // (attempts - relative)
                candidate = self if halving == 0 else self._with_dt(
                    float(np.ldexp(self.neuron_model.dt, -halving)))
                if initial_damping and type(candidate) is not type(self):
                    raise ValueError("numerical copy changed the boundary graph type")
                phase = candidate._equilibrate_local(
                    drive, state=current_state, budget=portion, chunk=chunk,
                    tolerance=tolerance, mask=mask, nudge=nudge,
                    stop_if_stalled=relative < attempts - 1,
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
                    initial_damping=initial_damping,
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
                # another chunk. Return its failed diagnostic instead of raising.
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
        elif not trajectory and capture is None and _FUSED and self._fused_blocks() is not None:
            from .fused import fused_settle

            flat, apical = cast(tuple[np.ndarray, Any], self._fused_blocks())
            s, taken, activity_change = fused_settle(
                v,
                a,
                drive,
                self.bias,
                self.layout,
                flat,
                keep,
                self.neuron_model,
                nudge,
                steps,
                tolerance,
                apical=apical,
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
            discarded_steps=0 if handle is None else int(handle.get("discarded_steps", 0)),
        )
        if capture is not None:
            capture.finish(self, drive, keep, nudge, result)
        return result

    def contrast_on_device(
        self, plus: BrainState, minus: BrainState
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``sum_b s+ s+ - s- s-`` per synapse and ``sum_b (s+ - s-)`` per neuron, on the device.

        Returns ``None`` when the brain has no device kernel or either state does not carry
        this brain's device handle; the learner then reads the activations on the host.
        """
        kernel = self._torch if self.backend == "torch" else self._mlx
        if kernel is None or plus.device is None or minus.device is None:
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

        A small residual certifies these equations at this state only; it says nothing
        about stability, uniqueness or convergence from another start. Activation-movement
        tolerance and reaching a step cap do not provide this certificate.
        The calculation performs one transport. An unread resident Torch state is checked
        on its device in float64, returning only one scalar per row (a float32 state by a
        float64 copy of the kernel, the same float64 reference). Other states, MPS and MLX
        states included, use the float64 CPU reference; ``on_device=False`` forces that
        reference. No state or parameter is changed.
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
            and kernel.float64_reference
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
        if _FUSED and self._fused_blocks() is not None:
            # the same equations in one compiled pass: a checked settle on the CPU then
            # costs one transport per chunk, not one NumPy operation per term
            from .fused import fused_residual

            flat, apical = cast(tuple[np.ndarray, Any], self._fused_blocks())
            return fused_residual(
                v, a, d, self.bias, self.layout, flat, keep, self.neuron_model, nudge,
                apical=apical,
            )
        s = self.neuron_model.activation(v) * keep
        adapt = self.neuron_model.adaptation
        if self.apical is None:
            blocks = None if self._blocks is None else BlockTransport(self.layout, self._blocks)
            error = self._synaptic_input(s, blocks) + d + self.bias - v
            if adapt is not None:
                error -= adapt.strength * a
            if nudge is not None:
                error += nudge.drive(s)
        else:
            basal_blocks, apical_blocks = self._apical_transports()
            error, raw = self._split_input(s, basal_blocks, apical_blocks)
            error += d + self.bias
            if adapt is not None:
                error -= adapt.strength * a
            if nudge is not None:
                error += nudge.drive(s)
            error += self.apical.drive(error, raw, None if nudge is None else nudge.apical_gain)
            error -= v
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

    def _apical_parts(self) -> tuple[Any, ...]:
        """The basal and the apical weights apart, as blocks when the connectome is blocked:
        ``(weights, basal flat, apical flat, basal blocks, apical blocks, basal weights,
        apical weights)``, made once per set of weights."""
        apical = self.apical
        assert apical is not None
        weights = self._weights
        parts = self.__dict__.get("_apical_cache")
        if parts is not None and parts[0] is weights:
            return cast(tuple[Any, ...], parts)
        mask = apical.synapses
        basal_weights = np.where(mask, 0.0, weights)
        apical_weights = np.where(mask, weights, 0.0)
        flat_basal = flat_apical = None
        basal_pairs: list[int] = []
        apical_pairs: list[int] = []
        if self._blocked:
            lay = self.layout
            flat_basal = lay.flat(basal_weights)
            flat_apical = lay.flat(apical_weights)
            basal_pairs, apical_pairs = _apical_blocks(apical, lay)
        parts = (weights, flat_basal, flat_apical, basal_pairs, apical_pairs,
                 basal_weights, apical_weights)
        self.__dict__["_apical_cache"] = parts
        return parts

    def apical_gain(self, drive: np.ndarray, state: BrainState) -> np.ndarray | None:
        """The apical compartment's gain ``(batch, pools)`` at ``state`` under ``drive``: what a
        nudged settle from it holds (``Nudge.apical_gain``); None without the compartment."""
        apical = self.apical
        if apical is None:
            return None
        d = np.atleast_2d(np.asarray(drive, dtype=float))
        handle, kernel = state.device, self._torch
        if (
            kernel is not None
            and kernel.float64_reference
            and handle is not None
            and handle.get("holder") is kernel
            and state.__dict__.get("activation") is None
            and state.__dict__.get("adaptation") is None
        ):  # the state never left the device: the gain is read there, one row per pool
            if tuple(handle["s"].shape) != d.shape:
                raise ValueError("state batch does not match the drive batch")
            gain: np.ndarray = kernel.apical_gain(d, handle)
            return gain
        s = np.atleast_2d(np.asarray(state.activation, dtype=float))
        basal, raw = self._split_input(s, *self._apical_transports())
        basal += d + self.bias
        adapt = self.neuron_model.adaptation
        if adapt is not None:
            basal -= adapt.strength * np.atleast_2d(np.asarray(state.adaptation, dtype=float))
        return apical.gain(basal, raw)

    def _fused_blocks(self) -> tuple[np.ndarray, Any] | None:
        """The flat blocks the fused kernels read and, with an apical compartment, its blocks
        apart (``fused.ApicalBlocks``): the flat array then holds the basal synapses alone.
        None when the connectome is not blocked, or a subclass withholds its blocks."""
        if self.apical is None:
            flat = self._blocks
            return None if flat is None else (flat, None)
        if not self._blocked:
            return None
        _, flat_basal, flat_apical, basal_pairs, apical_pairs, _, _ = self._apical_parts()
        return flat_basal, (self.apical, basal_pairs, flat_apical, apical_pairs)

    def _apical_transports(self) -> tuple[BlockTransport | None, BlockTransport | None]:
        """One settle's transports of the basal and the apical blocks; None when segmented."""
        _, flat_basal, flat_apical, basal_pairs, apical_pairs, _, _ = self._apical_parts()
        if flat_basal is None:
            return None, None
        return (
            BlockTransport(self.layout, flat_basal, basal_pairs),
            BlockTransport(self.layout, flat_apical, apical_pairs),
        )

    def _split_input(
        self,
        s: np.ndarray,
        basal_blocks: BlockTransport | None,
        apical_blocks: BlockTransport | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The synaptic input of the basal synapses and of the apical synapses, apart."""
        if basal_blocks is not None and apical_blocks is not None:
            return basal_blocks.synaptic_input(s), apical_blocks.synaptic_input(s)
        _, _, _, _, _, basal_weights, apical_weights = self._apical_parts()
        basal, raw = np.zeros_like(s), np.zeros_like(s)
        if self.connectome.synapses:
            messages = s[:, self.connectome.pre]
            target = self._neurons_with_synaptic_input
            basal[:, target] = np.add.reduceat(messages * basal_weights, self._starts, axis=1)
            raw[:, target] = np.add.reduceat(messages * apical_weights, self._starts, axis=1)
        return basal, raw

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
        apical = self.apical
        blocks = basal_blocks = apical_blocks = None
        held = None if nudge is None else nudge.apical_gain
        if apical is None:
            blocks = None if self._blocks is None else BlockTransport(self.layout, self._blocks)
        else:
            basal_blocks, apical_blocks = self._apical_transports()
            if held is not None and held.shape != (len(v), len(apical.members)):
                raise ValueError("a nudge's apical_gain needs one entry per row and pool")
        s = neuron_model.activation(v)
        if masked:
            s *= keep
        if capture is not None:
            capture.step(v, s, a)
        taken = 0
        activity_change = np.zeros(len(v))
        for t in range(steps):
            raw = None
            if apical is None:
                total = self._synaptic_input(s, blocks)
            else:
                total, raw = self._split_input(s, basal_blocks, apical_blocks)
            total += standing
            if adapt is not None:
                total -= adapt.strength * a
            if nudge is not None:
                total += nudge.drive(s)
            if raw is not None:  # the apical compartment, normalized to the basal input
                assert apical is not None
                total += apical.drive(total, raw, held)
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
            **({} if self.apical is None else {"apical": self.apical.to_dict()}),
            "efficacy_changed": int((self.efficacy != self.connectome.sign).sum()),
            "log_gain_nonzero": int((self.log_gain != 0).sum()),
            "bias_nonzero": int((self.bias != 0).sum()),
        }


# -- the torch backend's settle: one transport plan, preallocated sweep buffers, CUDA graphs
#
# ``_TorchKernel`` keeps a settle's state on its device in neuron-major layout, ``(neurons,
# batch)``, so that a neuron's row of every batch element is contiguous. Three pieces serve it:
#
# * ``_TransportPlan``: the synaptic input of a sweep as few, large products. Rows of the synapse
#   matrix (one per postsynaptic neuron, a second copy for the apical compartment) that read the
#   same presynaptic neurons, such as the feature maps of one receptive field, form a group; the
#   groups of one shape are one batched product over a gather of their presynaptic rows (on CUDA
#   the shapes of several small groups are one Triton launch, below). Nothing is padded and every
#   output row is written by one product, so the sum is deterministic and every operation can be
#   captured in a CUDA graph. A graph whose rows share few presynaptic neurons falls back to the
#   dense blocks of ``cadence.blocks`` (one product per block, also deterministic) when they fit,
#   and to a CSR sparse product otherwise (cuSPARSE on CUDA: fast, but neither deterministic nor
#   capturable, so such a settle runs eagerly).
# * ``_Staging``: pinned host buffers for the host-to-device copies of drives and masks, so an
#   upload never waits for the sweeps already queued on the device.
# * ``_Workspace``: the buffers of one batch size, two state slots, the flags of the movement
#   test, and the CUDA graphs captured on them.
#
# None of it changes what a neuron reads or how it updates.

# A grouped plan pays when every gathered presynaptic row serves at least this many synapses and
# the products stay few; otherwise blocks or CSR (measured on an A10G: see docs/backends.md).
_MIN_REUSE = 2.0
_MAX_BUCKETS = 64
_SMALL_GATHER = 4096  # a gather this small costs less than a sparse product's setup
# Bound on the elements of one temporary in the chunked contrast products.
_CHUNK = 1 << 25
# Pinned staging pays for copies at least this large; smaller ones go straight from NumPy.
_PINNED_BYTES = 1 << 20


def _graphs_enabled() -> bool:
    """CUDA graph replay of sweeps; ``CADENCE_CUDA_GRAPHS=0`` runs every sweep eagerly."""
    return os.environ.get("CADENCE_CUDA_GRAPHS", "1") != "0"


def _transport_override() -> str | None:
    """``CADENCE_TORCH_TRANSPORT`` forces ``grouped``, ``blocks`` or ``sparse``; unset is auto."""
    value = os.environ.get("CADENCE_TORCH_TRANSPORT", "").strip().lower()
    if value in ("", "auto"):
        return None
    if value not in ("grouped", "blocks", "sparse"):
        raise ValueError("CADENCE_TORCH_TRANSPORT must be auto, grouped, blocks or sparse")
    return value


def _row_groups(
    entry_row: np.ndarray, entry_col: np.ndarray, rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive nonempty rows whose column lists are identical, as groups.

    ``entry_row`` and ``entry_col`` are the matrix entries sorted by row, then column. Returns the
    row starts over the entries and, per group, its first row, its row count, its column count
    and the index of its first entry.
    """
    starts = np.searchsorted(entry_row, np.arange(rows + 1)).astype(np.int64)
    length = np.diff(starts)
    nonempty = np.flatnonzero(length)
    empty = np.zeros(0, dtype=np.int64)
    if not len(nonempty):
        return starts, empty, empty, empty, empty
    prev, cur = nonempty[:-1], nonempty[1:]
    candidate = np.flatnonzero((cur == prev + 1) & (length[cur] == length[prev]))
    same = np.zeros(len(cur), dtype=bool)
    if len(candidate):
        width = length[cur[candidate]]
        first = np.cumsum(width) - width
        offs = np.repeat(starts[cur[candidate]], width) + (
            np.arange(int(width.sum())) - np.repeat(first, width)
        )
        equal = entry_col[offs] == entry_col[offs - np.repeat(width, width)]
        same[candidate] = np.logical_and.reduceat(equal, first)
    begins = np.concatenate([[True], ~same])
    group_first = nonempty[begins]
    group_rows = np.bincount(np.cumsum(begins) - 1).astype(np.int64)
    group_cols = length[group_first]
    group_entry = starts[group_first]
    return starts, group_first, group_rows, group_cols, group_entry


class _Bucket:
    """Groups of one shape ``(rows, cols)``: one batched product per sweep."""

    def __init__(
        self,
        torch: Any,
        device: Any,
        count: int,
        rows: int,
        cols: int,
        out_rows: np.ndarray,
        in_cols: np.ndarray,
        offset: int,
        n: int,
    ) -> None:
        self.count, self.rows, self.cols, self.offset = count, rows, cols, offset
        self.size = count * rows * cols
        # output rows: a contiguous run is written in place, otherwise copied by index
        r0 = int(out_rows[0])
        self.row_range = (
            (r0, len(out_rows))
            if np.array_equal(out_rows, np.arange(r0, r0 + len(out_rows)))
            else None
        )
        self.row_index = (
            None if self.row_range is not None else torch.from_numpy(out_rows).to(device)
        )
        # the state rows these output rows belong to (an apical row reads its neuron's state)
        post = out_rows % n
        p0 = int(post[0])
        self.post_range = (
            (p0, len(post)) if np.array_equal(post, np.arange(p0, p0 + len(post))) else None
        )
        self.post_index = (
            None if self.post_range is not None else torch.from_numpy(post).to(device)
        )
        c0 = int(in_cols[0])
        self.col_range = (
            (c0, len(in_cols))
            if np.array_equal(in_cols, np.arange(c0, c0 + len(in_cols)))
            else None
        )
        self.col_index = (
            None if self.col_range is not None else torch.from_numpy(in_cols).to(device)
        )


def _take(x: Any, span: tuple[int, int] | None, index: Any) -> Any:
    """Rows of ``x``: a view of a contiguous span, else a gather."""
    if span is not None:
        return x.narrow(0, span[0], span[1])
    return x.index_select(0, index)


class _PlanStructure:
    """What a transport plan knows of its graph alone (no weights): its kind, the row groups
    and their device index tensors, made once per connectome, compartment, layout and device."""

    def __init__(
        self,
        torch: Any,
        pre: np.ndarray,
        post: np.ndarray,
        n: int,
        apical: np.ndarray | None,
        device: Any,
        layout: Any = None,
        kind: str | None = None,
    ) -> None:
        synapses = len(pre)
        row = np.asarray(post, dtype=np.int64)
        if apical is not None:
            row = row + n * np.asarray(apical, dtype=np.int64)
        self.rows = 2 * n if apical is not None else n
        order = np.lexsort((pre, row))
        r_s, c_s = row[order], np.asarray(pre, dtype=np.int64)[order]
        first = np.ones(synapses, dtype=bool)
        if synapses:
            first[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
        entry_sorted = np.cumsum(first) - 1
        entry_row, entry_col = r_s[first], c_s[first]
        self.entries = len(entry_row)
        self.parallel = self.entries != synapses  # distinct synapses sharing one entry
        entry_of_synapse = np.empty(synapses, dtype=np.int64)
        entry_of_synapse[order] = entry_sorted
        starts, g_first, g_rows, g_cols, g_entry = _row_groups(entry_row, entry_col, self.rows)
        shapes: dict[tuple[int, int], list[int]] = {}
        for g in range(len(g_first)):
            shapes.setdefault((int(g_rows[g]), int(g_cols[g])), []).append(g)
        gathered = int(g_cols.sum())
        blocked = layout is not None
        if kind is None:
            if synapses == 0 or (
                len(shapes) <= _MAX_BUCKETS
                and (self.entries >= _MIN_REUSE * gathered or gathered <= _SMALL_GATHER)
            ):
                kind = "grouped"
            elif blocked:
                kind = "blocks"
            else:
                kind = "sparse"
        if kind == "blocks" and not blocked:
            kind = "sparse"  # the blocks do not fit dense_limit
        if kind == "sparse" and device.type not in ("cuda", "cpu"):
            kind = "blocks" if blocked else "grouped"  # no sparse tensors there (MPS)
        self.kind = kind
        self.groups = len(g_first)
        self.buckets: list[_Bucket] = []
        self.pairs: list[tuple[int, int, int, int, int, int]] = []  # blocks: rows and bounds
        bucket_cols: list[np.ndarray] = []
        bucket_rows: list[np.ndarray] = []
        if kind == "grouped":
            positions = []
            offset = 0
            for (rows_, cols_), members in shapes.items():
                out_rows = np.concatenate(
                    [np.arange(g_first[g], g_first[g] + rows_) for g in members]
                )
                in_cols = np.concatenate(
                    [entry_col[g_entry[g] : g_entry[g] + cols_] for g in members]
                )
                bucket = _Bucket(
                    torch, device, len(members), rows_, cols_, out_rows, in_cols, offset, n
                )
                self.buckets.append(bucket)
                bucket_cols.append(in_cols)
                bucket_rows.append(out_rows)
                # the bucket's weights (count, rows, cols): entry of group g, row i, column j
                for g in members:
                    r0 = int(g_first[g])
                    positions.append(
                        (starts[r0 : r0 + rows_, None] + np.arange(cols_)[None, :]).ravel()
                    )
                offset += bucket.size
            entry_at = np.concatenate(positions) if positions else np.zeros(0, np.int64)
            where = np.empty(self.entries, dtype=np.int64)
            where[entry_at] = np.arange(self.entries)
            position = where[entry_of_synapse]
            self.size = self.entries
        elif kind == "blocks":
            lay = layout
            of_edge = np.asarray(lay.block_of_edge)
            in_apical = (
                np.zeros(synapses, bool) if apical is None else np.asarray(apical, dtype=bool)
            )
            # basal blocks in the first copy of the flat array, apical blocks in the second
            for space, chosen in ((0, ~in_apical), (1, in_apical)):
                for k in sorted(int(k) for k in np.unique(of_edge[chosen])):
                    a0, a1, b0, b1 = lay.bounds(k)
                    start = space * lay.size + int(lay.offset[k])
                    self.pairs.append((space * n + b0, b1 - b0, a0, a1 - a0, start, k))
            position = np.asarray(lay.edge_index, dtype=np.int64) + lay.size * in_apical
            self.size = lay.size * (2 if apical is not None else 1)
        else:
            position = entry_of_synapse
            self.size = self.entries
            self.crow = torch.from_numpy(starts).to(device)
            self.col = torch.from_numpy(entry_col.copy()).to(device)
            self.entry_row = torch.from_numpy(entry_row % n).to(device)
        self.position = torch.from_numpy(position).to(device)
        # On CUDA the buckets of several small groups are one Triton launch; a bucket of one
        # group is one dense product, which cuBLAS does well.
        self.small: Any = None
        self.large = list(range(len(self.buckets)))
        several = [k for k, bucket in enumerate(self.buckets) if bucket.count > 1]
        if kind == "grouped" and device.type == "cuda" and several:
            if _triton_transport() is not None:
                self.small = _SmallGroups(
                    torch, device, [self.buckets[k] for k in several],
                    [self.buckets[k].offset for k in several],
                    [bucket_cols[k] for k in several], [bucket_rows[k] for k in several],
                )
                self.large = [k for k in self.large if k not in set(several)]


def _plan_structure(
    torch: Any, connectome: Connectome, apical: Apical | None, device: Any, layout: Any,
    kind: str | None,
) -> _PlanStructure:
    """The connectome's plan structure for this compartment, layout and device, made once: a
    brain rebuilt on new parameters (``with_parameters``, a damped copy) reuses it."""
    held = connectome.__dict__.setdefault("_torch_plans", {})
    key = (str(device), kind, id(layout), id(apical), _triton_transport() is not None)
    entry = held.get(key)
    if entry is None or entry[1] is not layout or entry[2] is not apical:
        structure = _PlanStructure(
            torch, connectome.pre, connectome.post, connectome.n,
            None if apical is None else apical.synapses, device, layout, kind,
        )
        if len(held) >= 8:
            held.clear()
        entry = held[key] = (structure, layout, apical)  # the objects keep their ids unique
    return cast(_PlanStructure, entry[0])


class _TransportPlan:
    """The synaptic input of a sweep on a torch device, and the per-synapse Gram of two phases.

    Rows are postsynaptic neurons, ``rows = n`` without an apical compartment and ``2 n`` with
    one (rows ``n + i`` sum the apical synapses onto neuron ``i``). ``forward`` maps a
    neuron-major activation ``(n, batch)`` to the inputs ``(rows, batch)``. ``values`` holds the
    effective weights in the plan's own order; ``position`` maps each synapse to its entry.
    """

    def __init__(self, torch: Any, structure: _PlanStructure, n: int, device: Any,
                 dtype: Any) -> None:
        self.torch, self.n, self.device, self.dtype = torch, n, device, dtype
        self.structure = structure
        self.kind, self.rows, self.entries = structure.kind, structure.rows, structure.entries
        self.parallel, self.groups = structure.parallel, structure.groups
        self.buckets, self.pairs = structure.buckets, structure.pairs
        self.small, self.large = structure.small, structure.large
        self.position = structure.position
        if dtype not in (torch.float32, torch.float64):
            self.small = None
            self.large = list(range(len(self.buckets)))
        self.values = torch.zeros(structure.size, dtype=dtype, device=device)
        if self.kind == "sparse":
            self.csr = torch.sparse_csr_tensor(
                structure.crow, structure.col, self.values, (self.rows, n), device=device,
                check_invariants=False,
            )
            self.entry_row, self.entry_col = structure.entry_row, structure.col
        self.weights: list[Any] = []
        for bucket in self.buckets:
            self.weights.append(
                self.values[bucket.offset : bucket.offset + bucket.size].view(
                    bucket.count, bucket.rows, bucket.cols
                )
            )
        for _rows, width, _a0, height, start, _k in self.pairs:
            self.weights.append(self.values[start : start + height * width].view(height, width))

    @property
    def deterministic(self) -> bool:
        """Every sum in a fixed order: grouped and block products, or any CPU product."""
        return self.kind != "sparse" or self.device.type == "cpu"

    @property
    def capturable(self) -> bool:
        """Every operation of the transport can be captured in a CUDA graph."""
        return self.kind != "sparse"

    def scatter(self, weights: Any) -> Any:
        """The plan's values for per-synapse ``weights``, summing synapses that share an entry."""
        out = self.torch.zeros_like(self.values)
        if self.parallel:
            out.index_add_(0, self.position, weights)
        else:
            out[self.position] = weights
        return out

    def forward(self, s: Any, out: Any) -> Any:
        """The inputs ``(rows, batch)`` of a neuron-major activation ``s``, into ``out``.

        ``out`` must hold zeros in the rows no synapse writes; grouped products write every
        other row in full, block products accumulate into zeroed rows.
        """
        torch = self.torch
        batch = s.shape[1]
        if self.kind == "grouped":
            if self.small is not None:
                with torch.cuda.device(self.device):
                    self.small(self.values, s, out)
            for k in self.large:
                bucket, w = self.buckets[k], self.weights[k]
                x = _take(s, bucket.col_range, bucket.col_index)
                x = x.view(bucket.count, bucket.cols, batch)
                if bucket.row_range is not None:
                    target = out.narrow(0, bucket.row_range[0], bucket.row_range[1])
                    torch.bmm(w, x, out=target.view(bucket.count, bucket.rows, batch))
                else:
                    y = torch.bmm(w, x)
                    out.index_copy_(0, bucket.row_index, y.view(-1, batch))
            return out
        if self.kind == "blocks":
            out.zero_()
            for (r0, width, a0, height, _start, _k), w in zip(self.pairs, self.weights,
                                                              strict=True):
                out.narrow(0, r0, width).addmm_(w.T, s.narrow(0, a0, height))
            return out
        return torch.sparse.mm(self.csr, s)

    def gram(self, left: Any, right: Any, out: Any = None) -> Any:
        """Per entry ``(post, pre)`` the sum over ``k`` of ``left[post, k] * right[pre, k]``.

        ``left`` and ``right`` are neuron-major ``(n, K)``; the result is in the plan's value
        order, so ``result[position]`` is each synapse's term. With ``out`` the sums are added
        to it (a contrast taken over chunks of the batch).
        """
        torch = self.torch
        width = left.shape[1]
        add = out is not None
        if out is None:
            out = torch.zeros(self.values.shape[0], dtype=left.dtype, device=left.device)
        if self.kind == "grouped":
            for bucket in self.buckets:
                per = max(1, _CHUNK // max(1, (bucket.rows + bucket.cols) * width))
                for g0 in range(0, bucket.count, per):
                    g1 = min(bucket.count, g0 + per)
                    count = g1 - g0
                    if bucket.post_range is not None:
                        lhs = left.narrow(
                            0, bucket.post_range[0] + g0 * bucket.rows, count * bucket.rows
                        )
                    else:
                        assert bucket.post_index is not None
                        lhs = left.index_select(
                            0, bucket.post_index[g0 * bucket.rows : g1 * bucket.rows]
                        )
                    if bucket.col_range is not None:
                        rhs = right.narrow(
                            0, bucket.col_range[0] + g0 * bucket.cols, count * bucket.cols
                        )
                    else:
                        assert bucket.col_index is not None
                        rhs = right.index_select(
                            0, bucket.col_index[g0 * bucket.cols : g1 * bucket.cols]
                        )
                    start = bucket.offset + g0 * bucket.rows * bucket.cols
                    target = out[start : start + count * bucket.rows * bucket.cols].view(
                        count, bucket.rows, bucket.cols
                    )
                    a = lhs.reshape(count, bucket.rows, width)
                    b = rhs.reshape(count, bucket.cols, width).transpose(1, 2)
                    if add:
                        target.baddbmm_(a, b)
                    else:
                        torch.bmm(a, b, out=target)
            return out
        if self.kind == "blocks":
            for r0, width_, a0, height, start, _k in self.pairs:
                b0 = r0 % self.n
                block = right.narrow(0, a0, height) @ left.narrow(0, b0, width_).T
                if add:
                    out[start : start + height * width_] += block.reshape(-1)
                else:
                    out[start : start + height * width_] = block.reshape(-1)
            return out
        per = max(1, _CHUNK // max(1, width))
        for e0 in range(0, self.entries, per):
            e1 = min(self.entries, e0 + per)
            lhs = left.index_select(0, self.entry_row[e0:e1])
            rhs = right.index_select(0, self.entry_col[e0:e1])
            if add:
                out[e0:e1] += (lhs * rhs).sum(dim=1)
            else:
                out[e0:e1] = (lhs * rhs).sum(dim=1)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "entries": int(self.values.shape[0]),
            "groups": self.groups,
            "products": (len(self.large) + (self.small is not None))
            if self.kind == "grouped" else (len(self.pairs) if self.kind == "blocks" else 1),
            "triton_programs": 0 if self.small is None else self.small.programs,
            "deterministic": self.deterministic,
        }


class _Staging:
    """Host-to-device copies through a ring of pinned buffers, never waiting on queued sweeps.

    A copy from pageable memory makes the host wait until the device stream reaches it; a
    pinned buffer lets the copy queue behind the sweeps already launched. A buffer is reused
    only after its previous copy completed (its event), ``slots`` copies later.
    """

    def __init__(self, torch: Any, device: Any, slots: int = 4) -> None:
        self.torch, self.device = torch, device
        self.cuda = device.type == "cuda"
        self.slots = slots
        self.buffers: list[Any] = [None] * slots
        self.events: list[Any] = [None] * slots
        self.next = 0

    def upload(self, array: np.ndarray, dtype: Any) -> Any:
        torch = self.torch
        host = np.ascontiguousarray(array, dtype=np.float64)
        if not self.cuda or host.nbytes < _PINNED_BYTES:
            if not host.flags.writeable:  # a broadcast view: torch wants a writable buffer
                host = host.copy()
            source = torch.from_numpy(host)
            if self.cuda:  # a pageable copy is staged at once and does not wait for the stream
                return source.to(self.device, non_blocking=True).to(dtype)
            return source.to(self.device, dtype)
        k = self.next
        self.next = (k + 1) % self.slots
        if self.events[k] is not None:
            self.events[k].synchronize()
        buffer = self.buffers[k]
        if buffer is None or buffer.numel() < host.size:
            self.buffers[k] = buffer = torch.empty(
                max(host.size, 0 if buffer is None else 2 * buffer.numel()),
                dtype=torch.float64,
                pin_memory=True,
            )
        pinned = buffer[: host.size].view(host.shape)
        pinned.numpy()[...] = host
        out = pinned.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.events[k] = event
        return out if dtype == torch.float64 else out.to(dtype)


class _Workspace:
    """The buffers of one batch size: two state slots, the work arrays, flags and graphs."""

    def __init__(self, torch: Any, device: Any, dtype: Any, n: int, rows: int, batch: int,
                 adapt: bool, light: bool = False) -> None:
        def zeros(*shape: int) -> Any:
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.batch = batch
        self.out = zeros(rows, batch)  # the transport's output; rows no synapse writes stay zero
        # a light workspace (residual checks only) holds no state slots
        state = 0 if light else n
        self.v = [zeros(state, batch), zeros(state, batch)]
        self.s = [zeros(state, batch), zeros(state, batch)]
        self.a = [zeros(state, batch), zeros(state, batch)] if adapt else None
        self.change = [zeros(batch), zeros(batch)]
        self.total = zeros(state, batch)
        self.standing = zeros(state, batch)
        self.keep_shared: Any = None  # the mask buffers, (n, 1) and (n, batch), made on use
        self.keep_rows: Any = None
        self.held: Any = None  # (n, batch): a nudge's held apical gain spread over its pools
        self.part: Any = None  # the fused update's movement sums per row tile, and its peaks
        self.peak: Any = None
        self.flag = [torch.zeros((), dtype=torch.bool, device=device) for _ in range(2)]
        self.cuda = device.type == "cuda"
        self.flag_host = (
            torch.zeros(2, dtype=torch.bool, pin_memory=True) if self.cuda else None
        )
        self.events = [torch.cuda.Event() for _ in range(2)] if self.cuda else None
        self.nudges: dict[Any, Any] = {}  # per nudge signature: its buffers
        self.graphs: OrderedDict[Any, Any] = OrderedDict()
        self.seen: dict[Any, int] = {}  # eager runs per graph key, before a capture
        self.configs: dict[Any, Any] = {}  # per graph key: its sweep configuration
        self.executed_steps = 0

    def clear_captures(self) -> None:
        """Drop configurations and every graph/reference that can own their nudge buffers."""
        self.graphs.clear()
        self.seen.clear()
        self.configs.clear()



class _Workspaces:
    """A few workspaces, least recently used dropped first, with the graphs captured on them."""

    def __init__(self, limit: int = 4) -> None:
        self.limit = limit
        self.held: OrderedDict[int, _Workspace] = OrderedDict()

    def get(self, batch: int, make: Any) -> _Workspace:
        ws = self.held.get(batch)
        if ws is None:
            while len(self.held) >= self.limit:
                self.held.popitem(last=False)
            ws = cast(_Workspace, make())
            self.held[batch] = ws
        else:
            self.held.move_to_end(batch)
        return ws

# -- the small row groups on CUDA: one Triton launch
#
# Each program of the kernel computes a tile of ``ROWS`` output rows of one row group over
# ``BATCH`` columns of the batch: for every presynaptic neuron of the group, in order, it loads
# that neuron's batch row and the group's weight column and accumulates their outer product.
# Every output element is summed by one program in a fixed order, so the result is deterministic,
# and the launch can be captured in a CUDA graph. The cuBLAS batched products it replaces lose
# most of their tile to groups of eight or sixteen rows, and their float64 kernels run far below
# the device's rate; a single launch over every small group also replaces three operations per
# group shape (gather, product, scatter) with one.
#
# Triton is optional and CUDA-only: its language module is bound to ``tl`` and the kernel
# compiled on first use (``_triton_transport``); ``CADENCE_TRITON=0`` keeps every group on
# cuBLAS products.

_TRITON_ROWS = 8  # output rows per program: the smallest group height of the visual anatomies
tl: Any = None  # triton.language, once bound
_TRITON_KERNEL: Any = None


def _small_group_tiles(  # type: ignore[no-untyped-def]  # Triton reads the annotations
    w_ptr, s_ptr, out_ptr, meta_ptr, col_ptr, row_ptr, batch,
    ROWS: tl.constexpr, BATCH: tl.constexpr, DOUBLE: tl.constexpr,
):
    p = tl.program_id(0)
    tile = tl.program_id(1)
    w_off = tl.load(meta_ptr + p * 5)
    cols = tl.load(meta_ptr + p * 5 + 1)
    rows = tl.load(meta_ptr + p * 5 + 2)
    c_off = tl.load(meta_ptr + p * 5 + 3)
    r_off = tl.load(meta_ptr + p * 5 + 4)
    offs_b = tile * BATCH + tl.arange(0, BATCH)
    in_batch = offs_b < batch
    offs_r = tl.arange(0, ROWS)
    in_rows = offs_r < rows
    if DOUBLE:
        acc = tl.zeros((ROWS, BATCH), dtype=tl.float64)
    else:
        acc = tl.zeros((ROWS, BATCH), dtype=tl.float32)
    for c in range(0, cols):
        col = tl.load(col_ptr + c_off + c)
        x = tl.load(s_ptr + col * batch + offs_b, mask=in_batch, other=0.0)
        w = tl.load(w_ptr + w_off + offs_r * cols + c, mask=in_rows, other=0.0)
        acc += w[:, None] * x[None, :]
    row = tl.load(row_ptr + r_off + offs_r, mask=in_rows, other=0)
    tl.store(
        out_ptr + row[:, None] * batch + offs_b[None, :], acc,
        mask=in_rows[:, None] & in_batch[None, :],
    )


def _triton_transport() -> Any:
    """The compiled small-group kernel, or None without Triton or under ``CADENCE_TRITON=0``."""
    global tl, _TRITON_KERNEL
    if os.environ.get("CADENCE_TRITON", "1") == "0":
        return None
    if _TRITON_KERNEL is None:
        try:
            import triton
            import triton.language as language
        except ImportError:
            return None
        tl = language
        _TRITON_KERNEL = triton.jit(_small_group_tiles)
    return _TRITON_KERNEL


_UPDATE_KERNEL: Any = None
_UPDATE_ROWS = 32  # neurons per program of the fused update


def _update_tiles(  # type: ignore[no-untyped-def]  # Triton reads the annotations
    total_ptr, v_ptr, s_ptr, a_ptr, keep_ptr, v_out, s_out, a_out, part_ptr, peak_ptr, c_ptr,
    n, batch,
    ROWS: tl.constexpr, BATCH: tl.constexpr, KEEP: tl.constexpr, ADAPT: tl.constexpr,
    LEAK: tl.constexpr,
):
    # the neuron update of a sweep after its drive ``total``, as ``_TorchKernel._update`` does
    # it one operation at a time; the constants come from a device array in the state's dtype
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rows = rb * ROWS + tl.arange(0, ROWS)
    cols = cb * BATCH + tl.arange(0, BATCH)
    in_rows = rows < n
    inside = in_rows[:, None] & (cols < batch)[None, :]
    at = rows[:, None] * batch + cols[None, :]
    dt = tl.load(c_ptr)
    threshold = tl.load(c_ptr + 1)
    slope = tl.load(c_ptr + 2)
    rest = tl.load(c_ptr + 3)
    positive = tl.load(c_ptr + 4)
    negative = tl.load(c_ptr + 5)
    tau = tl.load(c_ptr + 6)
    v = tl.load(v_ptr + at, mask=inside, other=0.0)
    step = tl.load(total_ptr + at, mask=inside, other=0.0) - v
    step = step * dt
    v_next = v + step
    if KEEP == 1:
        keep = tl.load(keep_ptr + rows, mask=in_rows, other=1.0)[:, None]
        v_next = v_next * keep
    if KEEP == 2:
        keep = tl.load(keep_ptr + at, mask=inside, other=1.0)
        v_next = v_next * keep
    x = (v_next - threshold) * slope
    r = 1.0 / (1.0 + tl.exp(-x)) - rest
    if LEAK:
        r = tl.where(r > 0.0, r, r * negative)
    else:
        r = tl.maximum(r, 0.0)
    s_next = r * positive
    if KEEP != 0:
        s_next = s_next * keep
    tl.store(v_out + at, v_next, mask=inside)
    tl.store(s_out + at, s_next, mask=inside)
    if ADAPT:
        a = tl.load(a_ptr + at, mask=inside, other=0.0)
        tl.store(a_out + at, (s_next - a) / tau + a, mask=inside)
    movement = tl.where(inside, tl.abs(s_next - tl.load(s_ptr + at, mask=inside, other=0.0)), 0.0)
    tl.store(part_ptr + rb * batch + cols, tl.sum(movement, axis=0), mask=cols < batch)
    # a NaN movement never meets a tolerance: count it as infinite in the peak
    peak = tl.max(tl.max(tl.where(movement != movement, float("inf"), movement), axis=1), axis=0)
    tl.store(peak_ptr + rb * tl.num_programs(1) + cb, peak)


def _triton_update() -> Any:
    """The compiled fused update, or None without Triton or under ``CADENCE_TRITON=0``."""
    global _UPDATE_KERNEL
    if _triton_transport() is None:  # binds ``tl``
        return None
    if _UPDATE_KERNEL is None:
        import triton

        _UPDATE_KERNEL = triton.jit(_update_tiles)
    return _UPDATE_KERNEL


class _SmallGroups:
    """The programs of one launch: per program its weights, columns and output rows."""

    def __init__(self, torch: Any, device: Any, buckets: list[Any], weights_offset: list[int],
                 cols: list[np.ndarray], rows: list[np.ndarray]) -> None:
        meta = []
        c_base = r_base = 0
        for bucket, offset, c, r in zip(buckets, weights_offset, cols, rows, strict=True):
            for g in range(bucket.count):
                for r0 in range(0, bucket.rows, _TRITON_ROWS):
                    meta.append((
                        offset + (g * bucket.rows + r0) * bucket.cols,
                        bucket.cols,
                        min(_TRITON_ROWS, bucket.rows - r0),
                        c_base + g * bucket.cols,
                        r_base + g * bucket.rows + r0,
                    ))
            c_base += len(c)
            r_base += len(r)
        self.programs = len(meta)
        self.meta = torch.tensor(np.asarray(meta, dtype=np.int64).reshape(-1), device=device)
        self.cols = torch.from_numpy(np.concatenate(cols).astype(np.int64)).to(device)
        self.rows = torch.from_numpy(np.concatenate(rows).astype(np.int64)).to(device)
        self.device = device

    def __call__(self, values: Any, s: Any, out: Any) -> None:
        batch = s.shape[1]
        block = 64 if batch >= 64 else max(16, 1 << max(0, batch - 1).bit_length())
        grid = (self.programs, -(-batch // block))
        _triton_transport()[grid](
            values, s, out, self.meta, self.cols, self.rows, batch,
            ROWS=_TRITON_ROWS, BATCH=block, DOUBLE=values.element_size() == 8,
        )


def _host_copy(x: Any) -> np.ndarray:
    """A float64 host array that shares no memory with the tensor (a CPU tensor's ``numpy``
    would: the workspace buffers are overwritten by the next sweep)."""
    return np.array(x.detach().cpu().numpy(), dtype=np.float64, copy=True)


class _SweepConfig:
    """What one settle's sweeps read besides the state: the buffers and the graph key."""

    __slots__ = ("key", "masked", "keep", "nudge", "held", "tolerance")

    def __init__(
        self, key: Any, masked: bool, keep: Any, nudge: Any, held: bool, tolerance: float | None
    ) -> None:
        self.key, self.masked, self.keep = key, masked, keep
        self.nudge, self.held, self.tolerance = nudge, held, tolerance


class _NudgeBuffers:
    """A nudge's arrays on the device, restricted to the neurons it can push (``U``)."""

    def __init__(
        self, torch: Any, device: Any, dtype: Any, spec: Any, batch: int, weight: bool,
        anchor: bool,
    ) -> None:
        u, u_range, u_index, groups, tiled = spec
        m = len(u)
        self.m, self.u_range, self.u_index, self.groups = m, u_range, u_index, groups
        self.temperatures = (
            None if groups is None else torch.zeros(len(groups), 1, 1, dtype=dtype, device=device)
        )
        self.tiled = None if tiled is None else (
            *tiled[:3], self.temperatures, tiled[3]
        )  # equal contiguous groups: one softmax over a (groups, size) view
        self.target = torch.zeros(m, batch, dtype=dtype, device=device)
        self.mask = torch.zeros(m, 1, dtype=dtype, device=device)
        self.beta = torch.zeros((), dtype=dtype, device=device)
        self.push = torch.zeros(m, batch, dtype=dtype, device=device)
        self.weight = torch.zeros(1, batch, dtype=dtype, device=device) if weight else None
        self.anchor = torch.zeros(m, batch, dtype=dtype, device=device) if anchor else None
        self.gain = torch.zeros(m, 1, dtype=dtype, device=device) if anchor else None


class _TorchKernel:
    """Settling on a torch device: one transport plan, neuron-major state, CUDA graph replay.

    The state of a settle lives on the device as ``(neurons, batch)``; the synaptic input of a
    sweep is the ``_TransportPlan`` above, a few batched products of the
    row groups that share their presynaptic neurons. On CUDA the sweeps of a settle are replayed
    from captured graphs (``CADENCE_CUDA_GRAPHS=0`` runs them eagerly) and the movement test of
    a tolerance is read one sweep behind, so the host never waits on the device between sweeps;
    the reported state and step count are those of the first sweep that met the tolerance.

    On CUDA with Triton the products of the small row groups and the neuron update of a sweep
    are one launch each (``CADENCE_TRITON=0`` keeps cuBLAS products and one torch operation at a
    time). The default precision is float32 on MPS and float64 elsewhere; ``precision``
    overrides it.
    """

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
        apical: Apical | None = None,
        *,
        reference: bool = False,
    ) -> None:
        import torch

        self.torch = torch
        self.backend_name = "torch"
        # a float64 copy of a float32 kernel that only checks residuals and reads apical gains:
        # its workspaces hold the transport's output alone
        self._reference_only = reference
        self._reference: Any = None
        self._version = 0  # parameter replacements so far, for the reference copy
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
        self._connectome = connectome  # for the per-row contrast's edge index, made on first use
        self._row_index: Any = None
        self.apical = apical
        if apical is not None:
            # the pools as an indicator matrix (neurons x pools): a pool's sum of squares and
            # the spread of its gain to its members are one product each
            indicator = np.zeros((connectome.n, len(apical.members)))
            for k, members in enumerate(apical.members):
                indicator[members, k] = 1.0
            self.pool_matrix = torch.from_numpy(indicator).to(self.device, self.dtype)
            self.pool_rows = self.pool_matrix.T.contiguous()
            self.pool_count = torch.from_numpy(
                np.maximum(indicator.sum(axis=0), 1.0)[:, None]
            ).to(self.device, self.dtype)
        self.plan = _TransportPlan(
            torch,
            _plan_structure(torch, connectome, apical, self.device, layout,
                            _transport_override()),
            connectome.n,
            self.device,
            self.dtype,
        )
        self.flat = self.plan.values  # the live effective weights, in the plan's order
        self._staging = _Staging(torch, self.device)
        self._workspaces = _Workspaces()
        self._nudge_specs: dict[Any, Any] = {}
        self._graph_pool: Any = None
        self._capture_stream: Any = None
        self._graphs_failed = False
        self.graph_stats: dict[str, Any] = {"captured": 0, "replays": 0, "failed": None}
        rest = neuron_model.rest_emission
        self._positive = 1.0 / (1.0 - rest)
        self._negative = neuron_model.leak * (1.0 - rest) / rest  # leaky slope before _positive
        # On CUDA with Triton the neuron update of a sweep is one fused launch (CADENCE_TRITON=0
        # keeps the operation-by-operation update); its constants in the state's dtype
        self._fused: Any = None
        if self.device.type == "cuda" and not reference:
            self._fused = _triton_update()
        adapt = neuron_model.adaptation
        self._constants = torch.tensor(
            [neuron_model.dt, neuron_model.threshold, neuron_model.slope, rest, self._positive,
             self._negative, 1.0 if adapt is None else adapt.tau_steps],
            dtype=self.dtype, device=self.device,
        )
        self.set_parameters(
            torch.from_numpy(np.ascontiguousarray(efficacy)).to(self.device, self.param_dtype),
            torch.from_numpy(np.ascontiguousarray(bias)).to(self.device, self.param_dtype),
        )

    # -- parameters

    def set_parameters(self, scale: Any, bias: Any) -> None:
        """New parameters, as device tensors: the transport weights follow, in place."""
        torch = self.torch
        if tuple(scale.shape) != tuple(self.gain_pre.shape) or tuple(bias.shape) != (self.n,):
            raise ValueError("device parameters must match synapse and neuron shapes")
        weights = (self.gain_pre * scale).to(self.dtype)
        live_bias = bias.to(self.dtype)
        values = self.plan.scatter(weights)
        checked = [scale, bias, weights, live_bias] + ([values] if self.plan.parallel else [])
        if not bool(torch.stack([torch.isfinite(value).all() for value in checked]).all()):
            if self.plan.parallel and bool(
                torch.stack([torch.isfinite(v).all() for v in checked[:4]]).all()
            ):
                raise ValueError("summed effective synaptic weights must be finite")
            raise ValueError(
                "device parameters and effective weights must be finite at runtime precision"
            )
        # Validation precedes every write to the shared live kernel and its weights.
        self.scale = scale
        self.bias_param = bias
        self.plan.values.copy_(values)
        self.bias = live_bias
        self._version += 1

    @property
    def float64_reference(self) -> bool:
        """Whether residuals and apical gains are read on this device in float64 (MPS has none)."""
        return bool(self.dtype == self.torch.float64 or self.device.type != "mps")

    def _float64(self) -> _TorchKernel:
        """This kernel, or for a float32 one its float64 copy on the same parameters: what the
        host's float64 reference computes, read on the device."""
        if self.dtype == self.torch.float64:
            return self
        ref = self._reference
        if ref is None:
            ref = self._reference = _TorchKernel(
                self._connectome, np.zeros(self._connectome.synapses),
                np.zeros(self._connectome.synapses), np.zeros(self.n), self.neuron_model,
                str(self.device), self.layout, "float64",
                self.apical, reference=True,
            )
            ref._version = -1
        if ref._version != self._version:
            ref.set_parameters_from(self)
        return cast(_TorchKernel, ref)

    def set_parameters_from(self, other: _TorchKernel) -> None:
        """The parameters of ``other`` (same connectome), converted to this kernel's precision."""
        self.gain_pre = other.gain_pre
        self.set_parameters(other.scale, other.bias_param)
        self._version = other._version

    def host_scale(self) -> np.ndarray:
        return np.asarray(self.scale.cpu().double().numpy())

    def host_bias(self) -> np.ndarray:
        return np.asarray(self.bias_param.cpu().double().numpy())

    # -- the equations, neuron-major

    def _synaptic_input(self, s: Any, out: Any) -> Any:
        """The transport of a neuron-major activation: ``(rows, batch)``, basal rows first."""
        return self.plan.forward(s, out)

    def _activation_into(self, v: Any, out: Any) -> Any:
        """``NeuronModel.activation`` of ``v`` into ``out``, one in-place operation at a time."""
        torch, neuron_model = self.torch, self.neuron_model
        torch.sub(v, neuron_model.threshold, out=out)
        out.mul_(neuron_model.slope)
        out.sigmoid_()
        out.sub_(neuron_model.rest_emission)  # zero at rest
        # One leaky rectifier for the piecewise-linear map, exact silence at rest.
        if neuron_model.leak == 0.0:
            out.clamp_(min=0.0)
        else:
            torch.nn.functional.leaky_relu_(out, self._negative)
        out.mul_(self._positive)
        return out

    def _activation(self, v: Any) -> Any:
        return self._activation_into(v, self.torch.empty_like(v))

    def _apical_spread(self, basal: Any, raw: Any) -> Any:
        """The normalization's gain at every neuron ``(n, batch)``: ``Apical.gain`` per pool."""
        apical = self.apical
        assert apical is not None
        gain = self._apical_gain(basal, raw)
        return self.torch.mm(self.pool_matrix, gain)

    def _apical_gain(self, basal: Any, raw: Any) -> Any:
        """``Apical.gain`` on the device, ``(pools, batch)``: pool mean squares, one product."""
        torch, apical = self.torch, self.apical
        assert apical is not None
        basal2 = torch.mm(self.pool_rows, basal * basal) / self.pool_count
        raw2 = torch.mm(self.pool_rows, raw * raw) / self.pool_count
        spread = (raw2 + (apical.floor * apical.floor) * basal2 + _APICAL_TINY).sqrt()
        return apical.fraction * basal2.sqrt() / spread

    def _push(self, total: Any, s: Any, buffers: _NudgeBuffers) -> None:
        """Add ``Nudge.drive`` at neuron-major ``s`` into ``total``, on the neurons it reaches."""
        torch = self.torch
        s_u = _take(s, buffers.u_range, buffers.u_index)
        push = buffers.push
        if buffers.groups is None:
            torch.sub(buffers.target, s_u, out=push)
            push.mul_(buffers.beta)
            push.mul_(buffers.mask)
        elif buffers.tiled is not None:  # equal groups side by side: one batched softmax
            start, count, size, temperatures, covers = buffers.tiled
            batch = s_u.shape[1]
            if not covers:
                push.zero_()
            logits = s_u.narrow(0, start, count * size).view(count, size, batch)
            p = torch.softmax(logits / temperatures, dim=1)
            value = buffers.target.narrow(0, start, count * size).view(count, size, batch) - p
            value.mul_(buffers.beta)
            value.mul_(buffers.mask.narrow(0, start, count * size).view(count, size, 1))
            push.narrow(0, start, count * size).copy_(value.view(count * size, batch))
        else:  # one softmax per group of competing neurons
            push.zero_()
            assert buffers.temperatures is not None
            for k, (span, index, pieces, _members) in enumerate(buffers.groups):
                logits = _take(s_u, span, index)
                if pieces:
                    # Divide in representable pieces: zero remains zero, negative
                    # overflow remains a legitimate zero-probability logit.
                    logits = logits - logits.amax(dim=0, keepdim=True)
                    for _ in range(pieces):
                        logits = logits / torch.finfo(logits.dtype).tiny
                p = torch.softmax(logits / buffers.temperatures[k, 0, 0], dim=0)
                value = _take(buffers.target, span, index) - p
                value.mul_(buffers.beta)
                value.mul_(_take(buffers.mask, span, index))
                if span is not None:
                    push.narrow(0, span[0], span[1]).copy_(value)
                else:
                    push.index_copy_(0, index, value)
        if buffers.weight is not None:
            push.mul_(buffers.weight)
        if buffers.anchor is not None:
            push.addcmul_(buffers.gain, buffers.anchor - s_u)
        if buffers.u_range is not None:
            total.narrow(0, buffers.u_range[0], buffers.u_range[1]).add_(push)
        else:
            total.index_add_(0, buffers.u_index, push)

    def _sweep(self, ws: Any, cfg: _SweepConfig, i: int, j: int) -> None:
        """One sweep from slot ``i`` into slot ``j``: no host synchronization, no new buffer."""
        synaptic = self._synaptic_input(ws.s[i], ws.out)
        self._update(ws, cfg, i, j, synaptic)
        if cfg.tolerance is not None and ws.cuda:
            ws.flag_host[j].copy_(ws.flag[j], non_blocking=True)

    def _update(self, ws: Any, cfg: _SweepConfig, i: int, j: int, synaptic: Any) -> None:
        """The neuron-local part of a sweep, from the synaptic input into slot ``j``."""
        torch, neuron_model, n = self.torch, self.neuron_model, self.n
        v, s = ws.v[i], ws.s[i]
        total = ws.total
        torch.add(synaptic.narrow(0, 0, n), ws.standing, out=total)
        adapt = neuron_model.adaptation
        if adapt is not None:
            total.sub_(ws.a[i], alpha=adapt.strength)
        if cfg.nudge is not None:
            self._push(total, s, cfg.nudge)
        if self.apical is not None:  # the apical compartment, normalized to the basal input
            raw = synaptic.narrow(0, n, n)
            total.addcmul_(ws.held if cfg.held else self._apical_spread(total, raw), raw)
        if self._fused is not None:
            self._fused_update(ws, cfg, i, j)
            return
        total.sub_(v)
        total.mul_(neuron_model.dt)
        v_next, s_next = ws.v[j], ws.s[j]
        torch.add(v, total, out=v_next)  # neuron-local update: v <- v + dt (-v + total)
        if cfg.masked:
            v_next.mul_(cfg.keep)
        self._activation_into(v_next, s_next)
        if cfg.masked:
            s_next.mul_(cfg.keep)
        if adapt is not None:
            a, a_next = ws.a[i], ws.a[j]
            torch.sub(s_next, a, out=a_next)
            a_next.div_(adapt.tau_steps)
            a_next.add_(a)
        movement = total  # the drive is spent: its buffer takes the movement
        torch.sub(s_next, s, out=movement)
        movement.abs_()
        torch.add(ws.change[i], movement.sum(dim=0), out=ws.change[j])
        if cfg.tolerance is not None:
            torch.lt(movement.amax(), cfg.tolerance, out=ws.flag[j])

    def _fused_update(self, ws: Any, cfg: _SweepConfig, i: int, j: int) -> None:
        """The neuron update from ``ws.total`` in one Triton launch (CUDA): potentials,
        activations, adaptation and per-tile movement sums and peaks, then their reductions."""
        torch, n, batch = self.torch, self.n, ws.batch
        block = 64 if batch >= 64 else max(16, 1 << max(0, batch - 1).bit_length())
        grid = (-(-n // _UPDATE_ROWS), -(-batch // block))
        if ws.part is None:
            ws.part = torch.zeros(grid[0], batch, dtype=self.dtype, device=self.device)
            ws.peak = torch.zeros(grid[0] * grid[1], dtype=self.dtype, device=self.device)
        keep_mode = 0 if not cfg.masked else (1 if cfg.keep.shape[1] == 1 else 2)
        a_in = ws.a[i] if ws.a is not None else ws.total
        a_out = ws.a[j] if ws.a is not None else ws.total
        with torch.cuda.device(self.device):
            self._fused[grid](
                ws.total, ws.v[i], ws.s[i], a_in, cfg.keep if cfg.masked else ws.total,
                ws.v[j], ws.s[j], a_out, ws.part, ws.peak, self._constants, n, batch,
                ROWS=_UPDATE_ROWS, BATCH=block, KEEP=keep_mode, ADAPT=ws.a is not None,
                LEAK=self.neuron_model.leak != 0.0,
            )
        torch.add(ws.change[i], ws.part.sum(dim=0), out=ws.change[j])
        if cfg.tolerance is not None:
            torch.lt(ws.peak.amax(), cfg.tolerance, out=ws.flag[j])

    # -- device arrays of a settle

    def _workspace(self, batch: int) -> Any:
        rows = 0 if self.plan.kind == "sparse" else self.plan.rows
        return self._workspaces.get(
            batch,
            lambda: _Workspace(
                self.torch, self.device, self.dtype, self.n, rows, batch,
                self.neuron_model.adaptation is not None, self._reference_only,
            ),
        )

    def _upload(self, x: np.ndarray) -> Any:
        return self._staging.upload(x, self.dtype)

    def _nudge_spec(self, nudge: Nudge) -> tuple[Any, Any]:
        """The nudge's topology and arithmetic branches, independent of temperature values.

        Temperatures are mutable device inputs. Only a change in the number of divisions
        needed for a subnormal temperature changes the captured operation sequence.
        """
        torch = self.torch
        mask = np.asarray(nudge.mask)
        reach = mask > 0
        if nudge.anchor is not None:
            reach = reach | (np.asarray(nudge.anchor_gain) > 0)
        temperature = nudge.softmax_temperature
        members = None if temperature is None else _softmax_groups(nudge)
        tiny = float(np.finfo(np.float32 if self.dtype == torch.float32 else np.float64).tiny)
        pieces = []
        for group in members or ():
            value, count = _group_temperature(nudge, group), 0
            while value < tiny:
                value /= tiny
                count += 1
            pieces.append(count)
        key = (
            reach.tobytes(),
            None if nudge.groups is None else np.asarray(nudge.groups).tobytes(),
            None if members is None else tuple(pieces),
            (mask > 0).tobytes() if nudge.anchor is not None else None,
        )
        held = self._nudge_specs.get(key)
        if held is None:
            u = np.flatnonzero(reach).astype(np.int64)
            u_range = (
                (int(u[0]), len(u))
                if len(u) and np.array_equal(u, np.arange(u[0], u[0] + len(u)))
                else None
            )
            u_index = None if u_range is not None or not len(u) else (
                torch.from_numpy(u).to(self.device)
            )
            groups = None
            if members is not None:
                groups = []
                for group, count in zip(members, pieces, strict=True):
                    where = np.searchsorted(u, group).astype(np.int64)
                    span = (
                        (int(where[0]), len(where))
                        if np.array_equal(where, np.arange(where[0], where[0] + len(where)))
                        else None
                    )
                    index = None if span is not None else torch.from_numpy(where).to(self.device)
                    groups.append((span, index, count, group))
            spec = (u, u_range, u_index, groups, self._tiled(groups, len(u)))
            if len(self._nudge_specs) > 64:
                self._nudge_specs.clear()
            held = self._nudge_specs[key] = (key, spec)  # the key determines the spec
        return held

    def _tiled(self, groups: list[Any] | None, reach: int) -> Any:
        """Softmax groups that are equal, contiguous and side by side, as one batched view:
        ``(start, groups, size, covers the reach)``; else None."""
        if not groups or any(span is None or pieces for span, _, pieces, _ in groups):
            return None
        size = groups[0][0][1]
        start = groups[0][0][0]
        for k, (span, _, _pieces, _) in enumerate(groups):
            if span != (start + k * size, size):
                return None
        return start, len(groups), size, start == 0 and len(groups) * size == reach

    def _nudge_buffers(self, ws: Any, nudge: Nudge | None, batch: int) -> Any:
        """The nudge's buffers in ``ws``, filled for this settle; None without a reach."""
        if nudge is None:
            return None
        signature, spec = self._nudge_spec(nudge)
        u = spec[0]
        if not len(u):
            return None
        weight, anchor = nudge.weight is not None, nudge.anchor is not None
        full = (signature, weight, anchor)
        buffers = ws.nudges.get(full)
        if buffers is None:
            if len(ws.nudges) >= 16:
                ws.clear_captures()
                ws.nudges.clear()
            buffers = ws.nudges[full] = _NudgeBuffers(
                self.torch, self.device, self.dtype, spec, batch, weight, anchor
            )
        if buffers.groups is not None:
            temperatures = []
            tiny = float(self.torch.finfo(self.dtype).tiny)
            for _, _, pieces, group in buffers.groups:
                value = _group_temperature(nudge, group)
                for _ in range(pieces):
                    value /= tiny
                temperatures.append(value)
            buffers.temperatures.copy_(self._upload(np.asarray(temperatures)).view(-1, 1, 1))
        n = self.n
        buffers.target.copy_(self._upload(np.broadcast_to(nudge.target, (batch, n))[:, u]).T)
        buffers.mask.copy_(self._upload(np.asarray(nudge.mask)[u])[:, None])
        buffers.beta.fill_(float(nudge.beta))
        if weight:
            buffers.weight.copy_(self._upload(np.asarray(nudge.weight, dtype=float))[None, :])
        if anchor:
            assert nudge.anchor is not None
            buffers.anchor.copy_(self._upload(np.broadcast_to(nudge.anchor, (batch, n))[:, u]).T)
            buffers.gain.copy_(self._upload(np.asarray(nudge.anchor_gain)[u])[:, None])
        return buffers

    def _held_into(self, out: Any, nudge: Nudge | None, batch: int) -> bool:
        """A nudge's held apical gains, spread over their pools into ``out``; False without."""
        if nudge is None or nudge.apical_gain is None or self.apical is None:
            return False
        if nudge.apical_gain.shape != (batch, len(self.apical.members)):
            raise ValueError("a nudge's apical_gain needs one entry per row and pool")
        held = self._upload(nudge.apical_gain)
        self.torch.mm(self.pool_matrix, held.T, out=out)
        return True

    def _keep(self, ws: Any, keep: np.ndarray, batch: int) -> tuple[str | None, Any]:
        """The mask as a neuron-major buffer of ``ws``: ``(n, 1)`` shared or ``(n, batch)``."""
        keep = np.asarray(keep, dtype=float)
        if not bool((keep != 1.0).any()):
            return None, None
        torch = self.torch
        if keep.ndim == 1 or keep.shape[0] == 1:
            if ws.keep_shared is None:
                ws.keep_shared = torch.ones(self.n, 1, dtype=self.dtype, device=self.device)
            ws.keep_shared.copy_(self._upload(keep.reshape(-1))[:, None])
            return "shared", ws.keep_shared
        if ws.keep_rows is None:
            ws.keep_rows = torch.ones(self.n, batch, dtype=self.dtype, device=self.device)
        ws.keep_rows.copy_(self._upload(keep).T)
        return "rows", ws.keep_rows

    def _configure(
        self, ws: Any, drive: np.ndarray, keep: np.ndarray, nudge: Nudge | None,
        tolerance: float | None, batch: int,
    ) -> _SweepConfig:
        """Fill ``ws`` with what this settle's sweeps read and name the graph they replay."""
        d = self._upload(drive)
        ws.standing.copy_(d.T)
        ws.standing.add_(self.bias[:, None])  # the part of every neuron's drive that never moves
        kind, keep_buffer = self._keep(ws, keep, batch)
        buffers = self._nudge_buffers(ws, nudge, batch)
        held = False
        if self.apical is not None and nudge is not None and nudge.apical_gain is not None:
            if ws.held is None:
                ws.held = self.torch.zeros(self.n, batch, dtype=self.dtype, device=self.device)
            held = self._held_into(ws.held, nudge, batch)
        key = (
            kind,
            None if buffers is None else id(buffers),
            held,
            tolerance,
        )
        # one configuration object per key, reused by every settle of that key
        cfg = ws.configs.get(key)
        if cfg is None:
            if len(ws.configs) >= 32:
                ws.clear_captures()
            cfg = ws.configs[key] = _SweepConfig(
                key, kind is not None, keep_buffer, buffers, held, tolerance
            )
        return cast(_SweepConfig, cfg)

    # -- graphs

    def _graphs_on(self, sync_each: bool) -> bool:
        return (
            self.device.type == "cuda"
            and not sync_each
            and not self._graphs_failed
            and self.plan.capturable
            and "_synaptic_input" not in self.__dict__  # an override must run every sweep
            and _graphs_enabled()
        )

    def _graph(self, ws: Any, cfg: _SweepConfig, parity: int, length: int) -> Any:
        """The captured graph of ``length`` sweeps from slot ``parity``, or None to run eagerly.

        A configuration runs eagerly from both slots first (the kernels' lazy setup), then is
        captured; a graph whose capture fails turns graphs off for this kernel."""
        torch = self.torch
        key = (cfg.key, parity, length)
        graph = ws.graphs.get(key)
        if graph is not None:
            ws.graphs.move_to_end(key)
            return graph
        if self._graphs_failed or ws.seen.get(cfg.key, 0) < 2:
            return None  # not yet run eagerly from both slots
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
            self._capture_stream = torch.cuda.Stream(self.device)
        stream, current = self._capture_stream, torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        try:  # the capture runs on this kernel's device, whichever device is current
            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                # relaxed: the Triton launcher's driver queries must not invalidate the capture
                graph.capture_begin(pool=self._graph_pool, capture_error_mode="relaxed")
                try:
                    for k in range(length):
                        self._sweep(ws, cfg, (parity + k) % 2, (parity + k + 1) % 2)
                finally:
                    graph.capture_end()
        except Exception as error:
            self._graphs_failed = True
            self.graph_stats["failed"] = repr(error)[:200]
            current.wait_stream(stream)
            torch.cuda.synchronize(self.device)
            return None
        current.wait_stream(stream)
        while len(ws.graphs) >= 24:
            ws.graphs.popitem(last=False)
        ws.graphs[key] = graph
        self.graph_stats["captured"] += 1
        return graph

    def _advance(self, ws: Any, cfg: _SweepConfig, t: int, length: int, graphs: bool) -> None:
        """Sweeps ``t`` to ``t + length - 1``: a graph replay or eager sweeps; on CUDA the
        event of the last one's slot is recorded for the movement test."""
        graph = self._graph(ws, cfg, t % 2, length) if graphs else None
        ws.executed_steps += length
        if graph is not None:
            with self.torch.cuda.device(self.device):
                graph.replay()
            self.graph_stats["replays"] += 1
        else:
            for k in range(length):
                self._sweep(ws, cfg, (t + k) % 2, (t + k + 1) % 2)
            ws.seen[cfg.key] = ws.seen.get(cfg.key, 0) + 1
        if ws.cuda:
            ws.events[(t + length) % 2].record(self.torch.cuda.current_stream(self.device))

    def _met(self, ws: Any, t: int) -> bool:
        """Whether sweep ``t`` moved no activation by the tolerance (its flag, read on host)."""
        j = (t + 1) % 2
        if ws.cuda:
            ws.events[j].synchronize()
            return bool(ws.flag_host[j].item())
        return bool(ws.flag[j].item())

    def _loop(
        self, ws: Any, cfg: _SweepConfig, steps: int, traj: list[np.ndarray] | None,
        capture: _Capture | None, a_in: Any,
    ) -> int:
        """Run up to ``steps`` sweeps from slot 0; returns the sweeps taken (the final slot is
        their parity)."""
        sync_each = traj is not None or capture is not None
        ws.executed_steps = 0
        graphs = self._graphs_on(sync_each)
        if sync_each:
            for t in range(steps):
                self._sweep(ws, cfg, t % 2, (t + 1) % 2)
                ws.executed_steps += 1
                j = (t + 1) % 2
                if capture is not None:
                    capture.step(
                        _host_copy(ws.v[j].T), _host_copy(ws.s[j].T),
                        _host_copy(a_in if ws.a is None else ws.a[j].T),
                    )
                if traj is not None:
                    traj.append(_host_copy(ws.s[j].T))
                if cfg.tolerance is not None and bool(ws.flag[j].item()):
                    return t + 1
            return steps
        if cfg.tolerance is None:
            t = 0
            while t < steps:
                length = min(8, steps - t) if graphs else steps - t
                self._advance(ws, cfg, t, length, graphs)
                t += length
            return steps
        if not ws.cuda:
            # CPU and MPS have no CUDA host-copy overlap to hide. Check before starting
            # another sweep, so the returned count also counts all attempted work.
            for t in range(steps):
                self._advance(ws, cfg, t, 1, False)
                if self._met(ws, t):
                    return t + 1
            return steps
        # The movement test one sweep behind: sweep t + 1 is queued before sweep t's flag is
        # read, and writes the other slot, so a met tolerance at t keeps its state intact.
        self._advance(ws, cfg, 0, 1, graphs)
        for t in range(steps):
            if t + 1 < steps:
                self._advance(ws, cfg, t + 1, 1, graphs)
            if self._met(ws, t):
                return t + 1
        return steps

    # -- the kernel's entry points

    def residual(
        self, drive: np.ndarray, keep: np.ndarray, nudge: Nudge | None, handle: dict[str, Any]
    ) -> np.ndarray:
        """``Brain.residual`` for a state on this device; the same equations, one row each.

        A float32 state is checked in float64 by the kernel's float64 copy, as the host's
        float64 reference would check it."""
        reference = self._float64()
        if reference is not self:
            return reference.residual(
                drive, keep, nudge, {"v": handle["v"].double(), "a": handle["a"].double()}
            )
        torch, neuron_model, n = self.torch, self.neuron_model, self.n
        v, a = handle["v"], handle["a"]
        batch = v.shape[0]
        adapt = neuron_model.adaptation
        with torch.no_grad():
            ws = self._workspace(batch)

            def scratch(buffer: Any) -> Any:  # a workspace buffer between settles, or a new one
                if buffer is not None and tuple(buffer.shape) == (n, batch):
                    return buffer
                return torch.empty(n, batch, dtype=self.dtype, device=self.device)

            d = self._upload(drive).T
            k = np.asarray(keep, dtype=float)
            kt = self._upload(k.reshape(-1))[:, None] if k.ndim == 1 or k.shape[0] == 1 else (
                self._upload(k).T
            )
            vt = scratch(ws.v[0])
            vt.copy_(v.T)
            s = self._activation_into(vt, scratch(ws.s[0]))
            s.mul_(kt)
            at = None
            if adapt is not None:
                at = scratch(ws.v[1])
                at.copy_(a.T)
            synaptic = self._synaptic_input(s, ws.out)
            buffers = self._nudge_buffers(ws, nudge, batch)
            error = scratch(ws.total)
            if self.apical is None:
                torch.add(synaptic, d, out=error)
                error.add_(self.bias[:, None])
                error.sub_(vt)
                if adapt is not None and at is not None:
                    error.sub_(at, alpha=adapt.strength)
                if buffers is not None:
                    self._push(error, s, buffers)
            else:
                torch.add(d, self.bias[:, None], out=error)
                error.add_(synaptic.narrow(0, 0, n))
                if adapt is not None and at is not None:
                    error.sub_(at, alpha=adapt.strength)
                if buffers is not None:
                    self._push(error, s, buffers)
                raw = synaptic.narrow(0, n, n)
                held = scratch(ws.standing)
                if not self._held_into(held, nudge, batch):
                    held = self._apical_spread(error, raw)
                error.addcmul_(held, raw)
                error.sub_(vt)
            error.mul_(kt)  # (projected_next_v - v) / dt, written without cancellation
            projection = scratch(ws.s[1])
            torch.mul(vt, kt - 1.0, out=projection)
            projection.div_(neuron_model.dt)
            error.add_(projection)
            result = error.abs_().amax(dim=0)
            if at is not None:
                torch.sub(s, at, out=projection)
                result = torch.maximum(result, projection.abs_().amax(dim=0))
        out = result.cpu().double().numpy()
        return np.where(np.isfinite(out), out, np.inf)

    def apical_gain(self, drive: np.ndarray, handle: dict[str, Any]) -> np.ndarray:
        """``Brain.apical_gain`` for a state on this device: ``(batch, pools)`` on the host,
        in float64 (a float32 state's by the kernel's float64 copy)."""
        reference = self._float64()
        if reference is not self:
            return reference.apical_gain(
                drive, {"s": handle["s"].double(), "a": handle["a"].double()}
            )
        torch, n = self.torch, self.n
        s, a = handle["s"], handle["a"]
        batch = s.shape[0]
        adapt = self.neuron_model.adaptation
        with torch.no_grad():
            ws = self._workspace(batch)
            synaptic = self._synaptic_input(s.T.contiguous(), ws.out)
            basal = synaptic.narrow(0, 0, n) + (self._upload(drive).T + self.bias[:, None])
            if adapt is not None:
                basal = basal - adapt.strength * a.T
            gain = self._apical_gain(basal, synaptic.narrow(0, n, n))
        return np.asarray(gain.T.cpu().double().numpy())

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
        """The Gram contrast on the device, per synapse and per neuron, as tensors.

        The phases are differenced before the products: subtracting two large Gram matrices
        can erase a small contrast, especially in float32. ``A+^T (B+ - B-) + (A+ - A-)^T B-``
        is one product per row group over both terms, in the transport plan's groups.
        """
        torch, n = self.torch, self.n
        batch = s_plus.shape[0]
        # rows per chunk of the batch: the two neuron-major operands stay near 256 MB each
        step = max(1, min(batch, (1 << 24) // max(1, 2 * n)))
        values = None
        for b0 in range(0, batch, step):
            plus, minus = s_plus[b0 : b0 + step], s_minus[b0 : b0 + step]
            m = plus.shape[0]
            diff = plus - minus
            left = torch.empty(n, 2 * m, dtype=diff.dtype, device=diff.device)  # postsynaptic
            right = torch.empty(n, 2 * m, dtype=diff.dtype, device=diff.device)  # presynaptic
            left[:, :m].copy_(diff.T)
            left[:, m:].copy_(minus.T)
            right[:, :m].copy_(plus.T)
            right[:, m:].copy_(diff.T)
            values = self.plan.gram(left, right, values)
        assert values is not None
        edges = values[self.plan.position].to(self.param_dtype)
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
        """The Gram contrast on the device; the per-synapse and per-neuron sums come back."""
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
        torch = self.torch
        batch = drive.shape[0]
        handle = state.device if state is not None and state.device is not None else None
        traj: list[np.ndarray] | None = [] if want else None
        with torch.no_grad():
            ws = self._workspace(batch)
            cfg = self._configure(ws, drive, keep, nudge, tolerance, batch)
            if handle is not None and handle.get("holder") is self:
                a_in = handle["a"]  # the state never left the device
                ws.v[0].copy_(handle["v"].T)
            else:
                a_in = self._upload(a0)
                ws.v[0].copy_(self._upload(v0).T)
            if ws.a is not None:
                ws.a[0].copy_(a_in.T)
            self._activation_into(ws.v[0], ws.s[0])
            if cfg.masked:
                ws.s[0].mul_(cfg.keep)
            ws.change[0].zero_()
            if capture is not None:
                capture.step(
                    _host_copy(ws.v[0].T), _host_copy(ws.s[0].T),
                    _host_copy(a_in if ws.a is None else ws.a[0].T),
                )
            ws.executed_steps = 0
            taken = self._loop(ws, cfg, steps, traj, capture, a_in) if steps else 0
            j = taken % 2
            contiguous = torch.contiguous_format
            tensors = {
                "v": ws.v[j].T.clone(memory_format=contiguous),
                "a": (a_in if ws.a is None else ws.a[j].T).clone(memory_format=contiguous),
                "s": ws.s[j].T.clone(memory_format=contiguous),
            }
            activity_change = _host_copy(ws.change[j])

        def fetch(key: str) -> np.ndarray:  # the host copy, made when something reads it
            return np.asarray(tensors[key].cpu().double().numpy())

        # fetch closes over the tensors, never over the handle: a cycle there would hold the
        # device memory of every settled state until a garbage-collection pass
        held = {"kernel": self.backend_name, "holder": self, "fetch": fetch,
                "discarded_steps": ws.executed_steps - taken, **tensors}
        return (
            None,
            None,
            None,
            (np.stack(traj) if traj else np.empty((0, batch, self.n))) if want else None,
            taken,
            activity_change,
            held,
        )

    def describe(self) -> dict[str, Any]:
        """The transport plan and the replay mode, for reports."""
        return {
            **self.plan.describe(),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "fused_update": self._fused is not None,
            "graphs": self._graphs_on(False) if _graphs_enabled() else False,
            **{"graph_" + key: value for key, value in self.graph_stats.items()},
        }


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
        """The block Gram contrast on the device; the per-synapse and per-neuron sums come back."""
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
