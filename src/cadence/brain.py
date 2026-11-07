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
import functools
import itertools
import os
import sys
import threading
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


def _resident(array: Any) -> bool:
    """Whether an array rests on a torch device, not on the host."""
    return hasattr(array, "detach")


def _rows_finite(x: np.ndarray) -> bool:
    """Whether a settle's ``(batch, neurons)`` float64 array is finite."""
    if _FUSED:
        from .fused import rows_finite

        return bool(rows_finite(x))
    return bool(np.isfinite(x).all())


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
    device, _ = _mlx_probe()
    if device is not None:
        out["mlx"] = f"{device} float32"
    return out


_MLX_PROBE: tuple[str | None, str] | None = None
_MLX_LOCK = threading.Lock()
_MLX_MISSING = "mlx is not installed"
# the child: this process's import path, isolated from its working directory and environment;
# it leaves without the interpreter's shutdown once it has written the device
_MLX_CHILD = (
    "import os, sys; sys.path[:] = {path!r}; import mlx.core as mx; "
    "sys.stdout.write(mx.default_device().type.name + '\\n'); sys.stdout.flush(); os._exit(0)"
)


def _mlx_probe() -> tuple[str | None, str]:
    """The device MLX would use, or None with the reason.

    An MLX already imported here answers here. Otherwise a child interpreter imports it, once
    per process: an MLX whose discovery aborts the interpreter (an unsupported Metal or system
    configuration) ends the child, and this process reads MLX as unavailable. The child sees this
    process's import path and nothing else. A frozen application, whose executable is itself,
    imports MLX here, as releases before 0.92.1 did."""
    import importlib.util

    imported = sys.modules.get("mlx.core")
    if imported is not None:
        try:
            return str(imported.default_device().type.name), ""
        except Exception as exc:  # noqa: BLE001 - any failure reads as unavailable
            return None, f"mlx failed here: {exc}"
    try:
        installed = importlib.util.find_spec("mlx") is not None
    except (ImportError, ValueError):
        installed = False
    if not installed:  # not kept: an MLX installed later in this process is found
        return None, _MLX_MISSING
    global _MLX_PROBE
    with _MLX_LOCK:
        if _MLX_PROBE is None:
            _MLX_PROBE = _mlx_discover()
        return _MLX_PROBE


def _mlx_discover() -> tuple[str | None, str]:
    """MLX's device, asked of a child interpreter, or here in a frozen application."""
    import subprocess

    if getattr(sys, "frozen", False) or not sys.executable:
        try:
            import mlx.core as mx

            return str(mx.default_device().type.name), ""
        except Exception as exc:  # noqa: BLE001 - any failure reads as unavailable
            return None, f"mlx failed here: {exc}"
    code = _MLX_CHILD.format(path=[entry for entry in sys.path if isinstance(entry, str)])
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, timeout=300, check=False
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, f"mlx discovery could not run: {exc}"
    lines = done.stdout.decode("utf-8", "replace").strip().splitlines()
    if done.returncode == 0 and lines:
        return lines[-1], ""
    tail = (done.stderr.decode("utf-8", "replace").strip().splitlines() or ["no message"])[-1]
    return None, f"mlx discovery failed (exit {done.returncode}): {tail}"


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

    A state the compiled CPU kernels settled or checked keeps the block products of its
    activations, so a check of it or a settle from it computes only the products whose
    activations differ (the same bits); ``copy`` and ``pickle`` leave them out.
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

    def part(self, index: np.ndarray) -> np.ndarray:
        """The activation at the neurons ``index`` on the host, ``(batch, neurons)``. Of a state
        resting on a torch device only they come to the host, once each."""
        handle = self.__dict__.get("device")
        if handle is None or self.__dict__.get("activation") is not None or "s" not in handle:
            return np.asarray(self.activation[:, index])
        s = handle["s"]
        if not _resident(s):
            return np.asarray(self.activation[:, index])
        held = self.__dict__.setdefault("_parts", {})
        part = held.get(id(index))
        if part is None:
            part = held[id(index)] = _host_copy(s[:, handle["holder"].index_of(index)])
        return np.asarray(part)

    def __getstate__(self) -> dict[str, Any]:
        held = dict(self.__dict__)
        held.pop("_products", None)  # the kernels' products stay with this object
        held.pop("_parts", None)
        return held


def _check_ladder(neuron_model: NeuronModel, halvings: int, dtype: str) -> None:
    """Refuse a damping ladder whose smallest step's rule is not a valid rule representable at
    the backend's precision. A rule that passes is remembered by value (the check is a function
    of the rule's fields alone); one that fails is refused at every call, and so is one whose
    fields are arrays, which cannot be remembered."""
    try:
        hash(neuron_model)
    except TypeError:
        _ladder_rule(neuron_model, halvings, dtype)
    else:
        _checked_ladder(neuron_model, halvings, dtype)


def _ladder_rule(neuron_model: NeuronModel, halvings: int, dtype: str) -> None:
    neuron_model.replace(dt=float(np.ldexp(neuron_model.dt, -halvings)))._validate_precision(dtype)


_checked_ladder = functools.lru_cache(maxsize=64)(_ladder_rule)


def _copy_state(state: BrainState | None) -> BrainState | None:
    if state is None:
        return None
    handle = state.__dict__.get("device")
    if handle is not None and "s" in handle and _resident(handle["s"]):
        return state  # no solve writes a state on a torch device
    copied = BrainState(
        state.v.copy(),
        state.activation.copy(),
        state.adaptation.copy(),
        state.steps,
        activity_change=None if state.activity_change is None else state.activity_change.copy(),
        discarded_steps=state.discarded_steps,
    )
    _keep_products(copied, state.__dict__.get("_products"))  # the same activations
    return copied


def _keep_products(state: BrainState, products: Any) -> None:
    """Keep the compiled kernels' block products with ``state`` (``fused.Products``); a check
    of the state or a settle from it reads them where they hold for its activations."""
    if products is not None:
        state.__dict__["_products"] = products


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
    residual_checks: int = 0  # equation checks, each a transport outside the local sweeps
    damping_halvings: int = 0  # final halving index of this bounded solve
    stagnation_checks: int = 0  # complete-state comparisons, with no synaptic transport
    row_sweeps: int | None = None  # with ``fast``: the sweeps summed over rows, which stop apart

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


def _lazy_change(self: BrainState) -> np.ndarray | None:
    """``activity_change``, summed from what the launches made on first use on MLX."""
    value: np.ndarray | None = self.__dict__.get("activity_change")
    if value is None:
        handle = self.__dict__.get("device")
        if isinstance(handle, _MlxState):
            value = handle._fetch("change")
            self.__dict__["activity_change"] = value
    return value


def _put_change(self: BrainState, value: np.ndarray | None) -> None:
    self.__dict__["activity_change"] = value


BrainState.activity_change = property(_lazy_change, _put_change)  # type: ignore[assignment]


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
            self._mlx = _MlxKernel(
                connectome, self._weights, self.bias, neuron_model, self.layout, apical
            )
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
            kernel = self._torch if self._mlx is None else self._mlx
            assert kernel is not None
            self._efficacy = kernel.host_scale()
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

    def _with_mlx_parameters(self, kernel: _MlxKernel, bias: np.ndarray) -> Brain:
        """The brain under a lesson's parameters on the MLX device: ``kernel`` holds them, with
        ``bias``, the host's array of its biases; the host reads the efficacies when it asks."""
        new = copy.copy(self)
        new._mlx = kernel
        new._efficacy = None
        new._bias = bias
        new._weights_host = None
        new._flat = None
        new._csr = None
        return new

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

    def _with_update(
        self,
        efficacy: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        finite: tuple[bool, bool],
    ) -> Brain:
        """``with_parameters`` for a learner's compiled update: ``efficacy``, its effective
        ``weights`` (the product ``with_parameters`` makes, made by the update's pass) and
        ``bias`` are fresh arrays of the right shape that the update made; ``finite`` says
        whether the efficacy and the weights are finite, as that pass found them. They are taken
        as they are; on the ``"mlx"`` backend the new brain gets a kernel of its own whose
        blocks are made from these weights as construction makes them (``_MlxKernel``), and
        on ``"torch"`` ``with_parameters`` builds the new brain. The same brain, and the same
        refusals, as ``with_parameters`` gives."""
        if not finite[0]:
            raise ValueError("efficacy must have one finite entry per synapse")
        if not finite[1]:
            raise ValueError("effective synaptic weights must be finite; reduce gain or efficacy")
        if not np.isfinite(bias).all():
            raise ValueError("bias must have one finite entry per neuron")
        if self.backend == "torch":
            return self.with_parameters(efficacy=efficacy, bias=bias)
        new = copy.copy(self)
        if self._mlx is not None:  # before anything is replaced: a refusal changes nothing
            new._mlx = self._mlx.with_parameters(weights, bias)
        new._efficacy = efficacy
        new._weights_host = weights
        new._flat = None
        new._csr = None
        new._bias = bias
        return new

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
        single = BrainState(
            v=out.v[0],
            activation=out.activation[0],
            adaptation=out.adaptation[0],
            steps=out.steps,
            trajectory=None if out.trajectory is None else out.trajectory[:, 0, :],
            activity_change=None if out.activity_change is None else out.activity_change[0],
            discarded_steps=out.discarded_steps,
        )
        _keep_products(single, out.__dict__.get("_products"))  # the same row
        return single

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
        fast: bool = False,
    ) -> Equilibrium:
        """Continue a joint state until its equations hold or the exact step budget expires.

        Unlike ``settle_batch(tolerance=...)``, this tests the potential and adaptation
        equations, including the same mask and nudge. Checks cost one extra transport
        per chunk (on the compiled CPU path, the block products of the ranges the chunk's last
        sweep moved). Resident float64 Torch states are checked on their device. All rows
        advance together; a row's small residual does not freeze it.
        ``drive`` is dense, with one column per neuron, as in ``settle_batch``.
        Optional ``damping`` reserves equal portions of the one sweep budget
        for successively halved integration steps. Two consecutive repeated
        complete-state checkpoints at rounding precision end an unqualified
        attempt early, leaving its unused sweeps for the remaining attempts.
        Every attempt checks the original equations; a halved step changes only the
        numerical method. Repetition never qualifies an answer. Results report the final
        halving index (``damping_halvings``). ``fast``, for a free solve on the compiled CPU path
        at ``dt`` 1: a row stops at the check it qualifies, ``chunk`` is at most 16 and the
        activation is made in arithmetic that vectorizes, within about 1e-14 of the rule's.
        The same equations and tolerance; the bits are another solve's. ``row_sweeps`` reports
        the sweeps over the rows, and ``steps`` those of the slowest row.
        """
        for name, value, minimum in (
            ("budget", budget, 0), ("chunk", chunk, 1), ("damping", damping, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        budget, chunk, damping = map(int, (budget, chunk, damping))
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if damping:
            attempts = min(damping + 1, max(1, budget))
            dtype = "float32" if (self._mlx is not None or (
                self._torch is not None and self._torch.dtype == self._torch.torch.float32
            )) else "float64"
            _check_ladder(self.neuron_model, attempts - 1, dtype)
            used = checks = stagnation_checks = 0
            current_state = state
            for halving in range(attempts):
                portion = (budget - used + attempts - halving - 1) // (attempts - halving)
                candidate = self if halving == 0 else self._with_dt(
                    float(np.ldexp(self.neuron_model.dt, -halving)))
                phase = candidate._equilibrate_local(
                    drive, state=current_state, budget=portion, chunk=chunk,
                    tolerance=tolerance, mask=mask, nudge=nudge,
                    stop_if_stalled=halving < attempts - 1,
                )
                used += phase.steps
                checks += phase.residual_checks
                stagnation_checks += phase.stagnation_checks
                result = phase  # at the full step the attempt's report is the solve's
                if halving:
                    error = self.residual(drive, phase.state, mask=mask, nudge=nudge)
                    checks += 1
                    result = Equilibrium(
                        phase.state, error, used, tolerance,
                        residual_checks=checks, damping_halvings=halving,
                        stagnation_checks=stagnation_checks,
                    )
                if result.qualified.all() or used >= budget:
                    return result
                if not all(np.isfinite(value).all() for value in (
                    phase.state.v, phase.state.activation, phase.state.adaptation
                )):
                    return result
                current_state = phase.state
            return result
        return self._equilibrate_local(
            drive, budget=budget, chunk=chunk, tolerance=tolerance,
            state=state, mask=mask, nudge=nudge, fast=fast,
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
            old.get("holder") is self._mlx.root and new.get("holder") is self._mlx.root
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
        fast: bool = False,
    ) -> Equilibrium:
        """Run a validated local attempt; only a subsequent damping attempt may skip a stall."""
        current = self.settle_batch(drive, steps=0, state=state, mask=mask, nudge=nudge)
        if mask is None and nudge is None and not stop_if_stalled and self._solves_compiled():
            solved = self._solve_compiled(drive, current, budget, chunk, tolerance, fast)
            if solved is not None:
                return solved
        if mask is None and not stop_if_stalled and self._mlx is not None and self._solves_alone():
            kernel: _MlxKernel = self._mlx
            solved = kernel.solve(np.atleast_2d(drive), current, nudge, budget, chunk, tolerance)
            if solved is not None:
                return solved
        # below the launch's floor the float64 reference decides, also of a launched state
        reference: dict[str, bool] = (
            {"on_device": False}
            if self._mlx is not None and tolerance < _MLX_CHECK_FLOOR else {}
        )
        error = self.residual(drive, current, mask=mask, nudge=nudge, **reference)
        used, checks, stagnation_checks, repeats = 0, 1, 0, 0
        while used < budget and not (error <= tolerance).all():
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
            error = self.residual(drive, current, mask=mask, nudge=nudge, **reference)
            checks += 1
            if stop_if_stalled and used < budget and not (error <= tolerance).all():
                stagnation_checks += 1
                repeats = repeats + 1 if self._repeated_state(previous, current) else 0
                if repeats >= 2:
                    break
        return Equilibrium(
            current, error, used, tolerance, residual_checks=checks,
            stagnation_checks=stagnation_checks,
        )

    def _solves_compiled(self) -> bool:
        """Whether a free solve can run in one compiled call: no compartment, no recording, on
        the compiled CPU path. A wrapped ``settle_batch`` or ``residual`` sees every launch, and
        a subclass that withholds its blocks settles its own way."""
        return (
            _FUSED
            and self.backend == "cpu"
            and self.apical is None
            and self._solves_alone()
            and self._fused_blocks() is not None
        )

    def _solves_alone(self) -> bool:
        """Whether nothing has to see a solve's every settle and check: no wrapped ``settle_batch``
        or ``residual``, no recording."""
        return (
            type(self).settle_batch is _SETTLE_BATCH
            and type(self).residual is _RESIDUAL
            and "settle_batch" not in self.__dict__
            and "residual" not in self.__dict__
            and _capture() is None
        )

    def _solve_compiled(
        self, drive: np.ndarray, current: BrainState, budget: int, chunk: int, tolerance: float,
        fast: bool = False,
    ) -> Equilibrium | None:
        """``_equilibrate_local``'s solve from the published state ``current`` in one compiled
        call (``fused.fused_solve``): the same sweeps and checks, their counts and every bit;
        None where only the launches serve. ``fast`` (``equilibrate``) where it applies."""
        from .fused import fused_fast_solve, fused_solve

        compiled = self._fused_blocks()
        assert compiled is not None
        if fast:
            quick = fused_fast_solve(
                current.v, current.adaptation, current.activation, np.atleast_2d(drive), self.bias,
                self.layout, compiled[0], self.neuron_model, budget, min(chunk, 16), tolerance,
            )
            if quick is not None:
                error, used, checks, taken, row_sweeps, products = quick
                result = BrainState(
                    v=current.v, activation=current.activation, adaptation=current.adaptation,
                    steps=taken,  # the fast solve keeps no running movement
                )
                _keep_products(result, products)
                return Equilibrium(result, error, used, tolerance, residual_checks=checks,
                                   row_sweeps=row_sweeps)
        solved = fused_solve(
            current.v, current.adaptation, current.activation, np.atleast_2d(drive), self.bias,
            self.layout, compiled[0], self.neuron_model, budget, chunk, tolerance,
            current.__dict__.get("_products"),
        )
        if solved is None:
            return None
        error, used, checks, taken, change, products = solved
        result = BrainState(
            v=current.v, activation=current.activation, adaptation=current.adaptation,
            steps=taken, activity_change=change,
        )
        _keep_products(result, products)
        return Equilibrium(result, error, used, tolerance, residual_checks=checks)

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
        resident = _resident(drive)
        if not resident:
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
        if n == 0 or not (resident or _rows_finite(drive)):
            raise ValueError("drive must be finite and the brain must contain neurons")
        if mask is not None:
            keep = np.asarray(mask, float)
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
            is (self._torch if self.backend == "torch" else self._mlx.root)
        )
        v: Any
        a: Any
        if state is None:
            # rest needs no upload on a torch device
            v = a = None if self.backend == "torch" else np.zeros((batch, n))
            if self.backend != "torch":
                a = np.zeros((batch, n))
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
        if v is not None and not (
            (_rows_finite(v) and _rows_finite(a)) if self.backend == "cpu"
            else (np.isfinite(v).all() and np.isfinite(a).all())
        ):
            raise ValueError("warm potentials and adaptation must be finite")
        capture = _capture()
        if capture is not None and resident:
            drive = _host_copy(drive)
        handle = products = None
        if self.backend in ("torch", "mlx"):
            kernel = self._torch if self.backend == "torch" else self._mlx
            v, a, s, traj, taken, activity_change, handle = kernel.run(
                v, a, drive, np.ones(n) if mask is None else keep, steps, trajectory, nudge,
                tolerance, state, capture
            )
        elif (
            not trajectory and capture is None and _FUSED
            and (compiled := self._fused_blocks()) is not None
        ):
            from .fused import fused_settle

            flat, apical = compiled
            s, taken, activity_change, products = fused_settle(
                v,
                a,
                drive,
                self.bias,
                self.layout,
                flat,
                None if mask is None else keep,  # the kernel reads no mask where none is given
                self.neuron_model,
                nudge,
                steps,
                tolerance,
                apical=apical,
                products=None if state is None else state.__dict__.get("_products"),
            )
            traj = None
        else:
            v, a, s, traj, taken, activity_change = self._run_numpy(
                v, a, drive, np.ones(n) if mask is None else keep, steps, trajectory, nudge,
                tolerance, capture
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
        _keep_products(result, products)
        if capture is not None:
            capture.finish(self, drive, np.ones(n) if mask is None else keep, nudge, result)
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
        holder = kernel if self._mlx is None else kernel.root
        if plus.device.get("holder") is not holder or minus.device.get("holder") is not holder:
            return None
        out: tuple[np.ndarray, np.ndarray] = kernel.contrast(plus.device["s"], minus.device["s"])
        return out

    def _small_contrast(self, plus: BrainState) -> bool:
        """Whether a contrast costs the host less than the synchronization its launch takes: on
        MLX, few rows of small blocks whose activations the host reads anyway."""
        if self._mlx is None or self._mlx.plan is None or not isinstance(plus.device, _MlxState):
            return False
        rows = int(plus.device["vm"].shape[0])
        return rows * self.layout.size < _MLX_CONTRAST_ENTRIES

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
        The calculation performs one transport; on the compiled CPU path it reads the block
        products the state keeps where its activations are the ones they were made from, and
        the state keeps the check's products for a settle from it. An unread resident Torch
        state is checked on its device in float64, returning only one scalar per row (a
        float32 state by a float64 copy of the kernel, the same float64 reference). Other
        states, MPS and other MLX states included, use the float64 CPU reference;
        ``on_device=False`` forces that reference. No state value or parameter is changed.
        A state an MLX launch made under this drive, nudge and these parameters reads the launch's
        own check, in float32, with no launch of its own. A single state returns a length-one array;
        nonfinite errors return infinity.
        """
        d = drive if _resident(drive) else np.asarray(drive, dtype=float)
        if d.ndim == 1:
            d = d[None, :]
        if d.ndim != 2 or d.shape[1] != self.connectome.n:
            raise ValueError("drive must have one column per neuron")
        keep = None if mask is None else np.asarray(mask, dtype=float)
        if keep is not None and keep.shape not in (
            (self.connectome.n,), (1, self.connectome.n), d.shape
        ):
            raise ValueError("mask must have one entry per neuron, optionally per batch row")
        handle, kernel = state.device, self._torch
        if on_device and keep is None and isinstance(handle, _MlxState):
            checked = handle.get("checked")  # the launch that made the state read its equations
            if (
                checked is not None and checked[0] is d and checked[1] is nudge
                and self._mlx is not None and handle.get("version") == self._mlx.version
            ):
                return np.array(checked[2])
        if (
            on_device
            and kernel is not None
            and kernel.float64_reference
            and handle is not None
            and handle.get("holder") is kernel
            and state.__dict__.get("v") is None
            and state.__dict__.get("adaptation") is None
        ):
            if tuple(handle["v"].shape) != tuple(d.shape) or (
                tuple(handle["a"].shape) != tuple(d.shape)
            ):
                raise ValueError("state batch does not match the drive batch")
            return np.asarray(kernel.residual(
                d, np.ones(self.connectome.n) if keep is None else keep, nudge, handle
            ))
        if _resident(d):
            d = _host_copy(d)
        v = np.atleast_2d(state.v)
        a = np.atleast_2d(state.adaptation)
        if v.shape != d.shape or a.shape != d.shape:
            raise ValueError("state batch does not match the drive batch")
        if _FUSED and (compiled := self._fused_blocks()) is not None:
            # the same equations in one compiled pass: a checked settle on the CPU then costs
            # the products of the ranges that moved since the state's were made, not one NumPy
            # operation per term
            from .fused import fused_residual

            flat, apical = compiled
            error, products = fused_residual(
                v, a, d, self.bias, self.layout, flat, keep,
                self.neuron_model, nudge, apical=apical,
                products=state.__dict__.get("_products"),
            )
            _keep_products(state, products)  # every one current, for a settle from it
            return error
        if keep is None:
            keep = np.ones(self.connectome.n)
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
        d = drive if _resident(drive) else np.atleast_2d(np.asarray(drive, dtype=float))
        if self._mlx is not None and self._mlx.plan is not None:
            held: np.ndarray | None = self._mlx.gains_at(d, state)
            return held
        handle, kernel = state.device, self._torch
        if (
            kernel is not None
            and kernel.float64_reference
            and handle is not None
            and handle.get("holder") is kernel
            and state.__dict__.get("activation") is None
            and state.__dict__.get("adaptation") is None
        ):  # the state never left the device: the gain is read there, one row per pool
            if tuple(handle["s"].shape) != tuple(d.shape):
                raise ValueError("state batch does not match the drive batch")
            gain: np.ndarray = kernel.apical_gain(d, handle)
            return gain
        if _resident(d):
            d = _host_copy(d)
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


# The launches as the class defines them: a solve in one call stands in for these and no other.
_SETTLE_BATCH, _RESIDUAL = Brain.settle_batch, Brain.residual


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
        self._observed: tuple[np.ndarray, Any] | None = None
        self._indices: dict[int, tuple[np.ndarray, Any]] = {}
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

    def __deepcopy__(self, memo: dict[int, Any]) -> _TorchKernel:
        """A kernel of its own on the same device: a copy of a brain moves its own parameters, and
        the workspaces and captured graphs are made again."""
        twin = _TorchKernel(
            self._connectome, self.host_scale(), _host_copy(self.gain_pre), self.host_bias(),
            self.neuron_model, str(self.device), self.layout,
            "float32" if self.dtype == self.torch.float32 else "float64", self.apical,
            reference=self._reference_only,
        )
        memo[id(self)] = twin
        return twin

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

    def _drive_here(self, drive: Any) -> Any:
        """A settle's drive on the device, in the kernel's floats: a resident one as it rests."""
        return drive.to(self.dtype) if _resident(drive) else self._upload(drive)

    def observed(self, x: np.ndarray) -> Any:
        """Observations on the device in the parameters' floats, uploaded once however many of a
        moment's memories read the same array."""
        held = self._observed
        if held is None or held[0] is not x:
            held = self._observed = (x, self._staging.upload(x, self.param_dtype))
        return held[1]

    def index_of(self, index: np.ndarray) -> Any:
        """Neurons as an index of the device's arrays: a slice where they are a contiguous
        range, made once per array of them."""
        held = self._indices.get(id(index))
        if held is None or held[0] is not index:
            if len(self._indices) > 64:
                self._indices.clear()
            first = int(index[0]) if len(index) else 0
            contiguous = np.array_equal(index, np.arange(first, first + len(index)))
            on_device = (
                slice(first, first + len(index)) if contiguous
                else self.torch.from_numpy(np.array(index, dtype=np.int64)).to(self.device)
            )
            held = self._indices[id(index)] = (index, on_device)
        return held[1]

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
        d = self._drive_here(drive)
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

            d = self._drive_here(drive).T
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
            basal = synaptic.narrow(0, 0, n) + (self._drive_here(drive).T + self.bias[:, None])
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
            elif v0 is None:  # from rest
                a_in = torch.zeros((batch, self.n), dtype=self.dtype, device=self.device)
                ws.v[0].zero_()
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


# The most sweeps one lazy MLX graph holds before it is evaluated: a settle without a movement
# tolerance is evaluated once at its end, and a long one this often, which bounds its memory.
_MLX_GRAPH_STEPS = 64
# The most sweeps one launch runs under a movement tolerance: a bit per sweep of a 32-bit word.
_MLX_BOUNDED_STEPS = 32
# The most settling neurons one launch holds in threadgroup memory (four arrays, 32 KB in all),
# and with an apical compartment (six, each a multiple of 16 bytes).
_MLX_LAUNCH_NEURONS = 2048
_MLX_COMPARTMENT_NEURONS = 1360
# A source range's gap to its drive that a settle's sweeps no longer read, under another step than
# dt = 1: its products are made per sweep until then. The floats of one lazy part of them.
_MLX_SOURCE_ROUNDING = 1e-7
_MLX_RAMP_FLOATS = 1 << 26
# Threads in a row's threadgroup: Metal's limit on Apple GPUs.
_MLX_LAUNCH_THREADS = 1024
# Inside a solve the launch's check reads float32: its rounding is about 1e-6 at potentials near
# ten. A tolerance below this floor is checked by the float64 reference on the host.
_MLX_CHECK_FLOOR = 1e-4
# The smallest temperature of a launch's softmax: an activation is at most one, so its exponential
# stays within float32 for any sum of neurons.
_MLX_LEAST_TEMPERATURE = 0.02
# Block entries from which a lesson's update costs the device less than the host: below it the
# update stays on the host, in float64.
_MLX_UPDATE_ENTRIES = 1 << 17
# Entries of a recall memory's address map from which its product and its step cost the device
# less than the host. Smaller maps stay on the host, in float64.
_MLX_MAP_ENTRIES = 1 << 20
# Rows times block entries from which the device's contrast costs less than the host's.
_MLX_CONTRAST_ENTRIES = 1 << 20

# A chunk of sweeps of the settling neurons in one Metal launch, with the equations' residual of
# its first and last state. One threadgroup per row, one thread per settling neuron (or PER of
# them), so the rows and neurons run in parallel with no host synchronization between sweeps. The
# activations live in threadgroup memory, read four at a time: a GPU thread is many times slower
# per load from device memory than the SIMD group it runs in. A source range's products reach the
# launch summed in ``first`` (one array per sweep of the ``ramp``, where the range is still moving)
# and ``fixed`` (every later sweep's). With an apical compartment a neuron's basal and top-down
# inputs are read apart and join at its pool's gain: the free phase's from the pool's mean squares
# after a barrier, a nudged settle's held.
_MLX_LAUNCH = """
    constexpr int CAP = {cap}, PER = {per};
    threadgroup float tg_a[CAP];
    threadgroup float tg_b[CAP];
    threadgroup float tg_p[CAP];
    threadgroup float tg_q[CAP];
    {top}
    int row = (int)threadgroup_position_in_grid.x, me = (int)thread_index_in_threadgroup;
    int m = plan[0], blocks = plan[1], threads = plan[2], pools = plan[3];
    int sweeps = control[0], ramp = control[5], stride = control[6], compartment = control[7];
    bool nudging = control[1] != 0, shares = control[2] != 0, bounded = control[3] != 0;
    float row_weight = control[4] != 0 ? weight[row] : 1.0f;
    float slope = scales[0], threshold = scales[1], rest = scales[2], up = scales[3];
    float down = scales[4], beta = scales[5], tolerance = scales[6], dt = scales[7];
    float fraction = scales[8], floor2 = scales[9], tiny = scales[10];
    int low = me * PER, high = min(m, low + PER), at = row * m;
    float held[PER], below[PER], above[PER];
    threadgroup float* now = tg_a;
    threadgroup float* next = tg_b;
    threadgroup float* heat = tg_p;
    threadgroup float* next_heat = tg_q;
    for (int j = low; j < high; j++) {{
        held[j - low] = v0[at + j];
        now[j] = s0[at + j];
        if (shares && mask[j] != 0.0f) heat[j] = exp(now[j] * over[j]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float moved = 0.0f, worst_first = 0.0f;
    uint moving = 0u;
    for (int t = 0;; t++) {{
        for (int j = low; j < high; j++) {{
            float total = t < ramp ? first[t * stride + at + j] : fixed[at + j];
            float top = 0.0f;
            for (int k = 0; k < blocks; k++) {{
                int e = 4 + 6 * k;
                int a0 = plan[e], a1 = plan[e + 1], b0 = plan[e + 2];
                if (j < b0 || j >= plan[e + 3]) continue;
                int offset = plan[e + 4] + (j - b0) * (a1 - a0) - a0;
                int i = a0;
                float4 sum = 0.0f;
                for (; i + 4 <= a1; i += 4) {{
                    int c = offset + i;
                    sum += float4(now[i], now[i + 1], now[i + 2], now[i + 3])
                        * float4(w[c], w[c + 1], w[c + 2], w[c + 3]);
                }}
                float product = sum.x + sum.y + sum.z + sum.w;
                for (; i < a1; i++) product += now[i] * w[offset + i];
                if (plan[e + 5] != 0) top += product;
                else total += product;
            }}
            if (nudging && mask[j] != 0.0f) {{
                float share = now[j];
                if (shares) {{  // the softmax of the neuron's group, a range of settling neurons
                    int i = group_low[j], g1 = group_high[j];
                    float4 sum = 0.0f;
                    for (; i + 4 <= g1; i += 4) {{
                        sum += float4(heat[i], heat[i + 1], heat[i + 2], heat[i + 3]);
                    }}
                    float all = sum.x + sum.y + sum.z + sum.w;
                    for (; i < g1; i++) all += heat[i];
                    share = heat[j] / all;
                }}
                total += beta * (target[at + j] - share) * mask[j] * row_weight;
            }}
            below[j - low] = total;
            above[j - low] = top;
            if (compartment == 1) {{
                tg_x[j] = total;
                tg_u[j] = top;
            }}
        }}
        if (compartment == 1) threadgroup_barrier(mem_flags::mem_threadgroup);
        float worst = 0.0f, gain = 0.0f;
        bool broken = false;
        int seen = -1;
        for (int j = low; j < high; j++) {{
            float total = below[j - low];
            int pool = compartment != 0 ? pool_of[j] : -1;
            if (pool >= 0) {{  // the apical input at its pool's gain, read once per pool
                if (pool != seen) {{
                    seen = pool;
                    if (compartment == 2) gain = given[row * pools + pool];
                    else {{
                        int i = pool_low[j], g1 = pool_high[j];
                        float count = float(g1 - i);
                        float4 sum_x = 0.0f, sum_u = 0.0f;
                        for (; i + 4 <= g1; i += 4) {{
                            float4 x = float4(tg_x[i], tg_x[i + 1], tg_x[i + 2], tg_x[i + 3]);
                            float4 u = float4(tg_u[i], tg_u[i + 1], tg_u[i + 2], tg_u[i + 3]);
                            sum_x += x * x;
                            sum_u += u * u;
                        }}
                        float x2 = sum_x.x + sum_x.y + sum_x.z + sum_x.w;
                        float u2 = sum_u.x + sum_u.y + sum_u.z + sum_u.w;
                        for (; i < g1; i++) {{
                            x2 += tg_x[i] * tg_x[i];
                            u2 += tg_u[i] * tg_u[i];
                        }}
                        x2 /= count;
                        u2 /= count;
                        gain = fraction * sqrt(x2) / sqrt(u2 + floor2 * x2 + tiny);
                    }}
                }}
                total += gain * above[j - low];
                if (t == sweeps && j == pool_low[j]) out_g[row * pools + pool] = gain;
            }}
            float v = held[j - low];
            float error = abs(total - v);
            worst = max(worst, error);
            broken = broken || !(error < INFINITY);
            if (t < sweeps) {{
                v = v + dt * (total - v);
                held[j - low] = v;
                float r = 1.0f / (1.0f + exp(-slope * (v - threshold))) - rest;
                float a = r > 0.0f ? r * up : r * down;
                float step = abs(a - now[j]);
                moved += step;
                if (bounded && !(step < tolerance)) moving |= 1u << t;
                next[j] = a;
                if (shares && mask[j] != 0.0f) next_heat[j] = exp(a * over[j]);
            }}
        }}
        if (broken) worst = INFINITY;
        if (t == 0) worst_first = worst;
        if (t == sweeps) {{
            out_e[(row * threads + me) * 2] = worst_first;
            out_e[(row * threads + me) * 2 + 1] = worst;
            break;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* swap = now;
        now = next;
        next = swap;
        swap = heat;
        heat = next_heat;
        next_heat = swap;
    }}
    for (int j = low; j < high; j++) {{
        out_v[at + j] = held[j - low];
        out_s[at + j] = now[j];
    }}
    out_c[row * threads + me] = moved;
    out_m[row * threads + me] = moving;
"""


class _MlxMatrix:
    """A large matrix of a memory on the MLX device, in float32, that reads as a host array in
    float64: ``np.asarray`` and the array operations it serves read it whole, once. What a moment
    does to it (``term``, ``stepped``) stays on the device. The array is never written; a step
    makes another."""

    __slots__ = ("mx", "array", "flag", "_host")

    def __deepcopy__(self, memo: dict[int, Any]) -> _MlxMatrix:
        return self  # never written: a copy shares it

    def __init__(self, mx: Any, array: Any, flag: Any = None) -> None:
        self.mx = mx
        self.array = array
        self.flag = flag  # with a stepped matrix: whether it is finite, read with its first use
        self._host: np.ndarray | None = None

    @classmethod
    def of(cls, mx: Any, host: np.ndarray) -> _MlxMatrix:
        return cls(mx, mx.array(np.ascontiguousarray(host, dtype=np.float32)))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def size(self) -> int:
        return int(self.array.size)

    @property
    def ndim(self) -> int:
        return int(self.array.ndim)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(np.float64)

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, index: Any) -> Any:
        return np.asarray(self)[index]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        if self._host is None:
            self._host = np.array(self.array, dtype=np.float64)
            self._host.setflags(write=False)
        return np.array(self._host, dtype=dtype) if dtype is not None or copy else self._host

    def _kept(self) -> list[Any]:
        return [] if self.flag is None else [self.flag]

    def term(self, x: np.ndarray, other: _MlxMatrix, c: np.ndarray) -> np.ndarray:
        """``x @ self + c @ other`` as a host array, in one launch with the two matrices' checks."""
        mx = self.mx
        out = (
            mx.array(np.asarray(x, dtype=np.float32)) @ self.array
            + mx.array(np.asarray(c, dtype=np.float32)) @ other.array
        )
        mx.eval(out, self.array, other.array, self._kept(), other._kept())
        return np.asarray(np.array(out, copy=False), dtype=np.float64)

    def stepped(
        self, other: _MlxMatrix, x: np.ndarray, c: np.ndarray, step: np.ndarray, rate: float,
    ) -> tuple[_MlxMatrix, _MlxMatrix]:
        """The two matrices after a least-mean-squares ``step``: ``x.T @ step`` and ``c.T @ step``
        at ``rate`` over the rows, added. Not launched: the term that reads them launches them."""
        mx = self.mx
        s = mx.array(np.asarray(step, dtype=np.float32))
        share = rate / len(x)
        ours = self.array + share * (mx.array(np.asarray(x, dtype=np.float32)).T @ s)
        theirs = other.array + share * (mx.array(np.asarray(c, dtype=np.float32)).T @ s)
        return (
            _MlxMatrix(mx, ours, mx.isfinite(mx.sum(ours))),
            _MlxMatrix(mx, theirs, mx.isfinite(mx.sum(theirs))),
        )

    def finite(self) -> bool:
        """Whether every entry is finite: a sum over them is, or overflows and is refused too."""
        if self.flag is None:
            self.flag = self.mx.isfinite(self.mx.sum(self.array))
        return bool(self.flag.item())


def _launch_error(errors: Any, which: int) -> np.ndarray:
    """Each row's largest residual at a launch's first (0) or last (1) state, from every thread's;
    infinity where one is not finite."""
    held = np.array(errors, copy=False)[:, which::2]
    return np.asarray(held.max(axis=1), dtype=np.float64)


class _MlxState(dict):  # type: ignore[type-arg]
    """The handle of a settle launched on MLX: the settling neurons' arrays and each source
    range's. The whole ``(batch, n)`` arrays (``"v"``, ``"s"``, ``"a"``) are made when first read
    on the device, and the host's (``"fetch"``) without another launch: most moments read neither
    all of them nor any of them twice."""

    __slots__ = ()

    def __deepcopy__(self, memo: dict[int, Any]) -> _MlxState:
        return self  # never written: a copy shares it

    def __missing__(self, key: str) -> Any:
        if key == "fetch":  # not kept: the handle holds no reference to itself
            return self._fetch
        if key in ("v", "s", "a"):
            value = self["plan"].whole(self, key)
            self[key] = value
            return value
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in ("fetch", "v", "s", "a") or dict.__contains__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def _fetch(self, key: str) -> np.ndarray:
        result: np.ndarray = self["plan"].fetch(self, key)
        return result


class _MlxPlan:
    """What a launch reads of a layout and a neuron rule, made once per kernel family: the settling
    ranges (those that receive synapses) and the source ranges. Without adaptation a source range's
    potential follows its drive in closed form, ``goal + (1 - dt) ** t * gap``: under ``dt = 1`` it
    stands there from the first sweep and its products are made once a settle, and under another
    step once per sweep until the gap is below rounding. Only the settling neurons sweep."""

    def __init__(
        self, mx: Any, layout: Layout, neuron_model: NeuronModel, n: int, apical: Apical | None,
    ) -> None:
        self.mx = mx
        self.n = n
        self.rho = 1.0 - float(neuron_model.dt)
        self.settling = sorted({int(r) for r in layout.pair_post})
        self.sources = [r for r in range(layout.ranges) if r not in self.settling]
        self.spans = {
            r: (int(layout.starts[r]), int(layout.starts[r + 1])) for r in range(layout.ranges)
        }
        self.place: dict[int, int] = {}
        at = 0
        for r in self.settling:
            self.place[r] = at
            at += self.spans[r][1] - self.spans[r][0]
        self.m = at
        self.pre = [int(r) for r in layout.pair_pre]
        self.post = [int(r) for r in layout.pair_post]
        self.inner = [k for k, r in enumerate(self.pre) if r in self.place]
        self.outer = [k for k, r in enumerate(self.pre) if r not in self.place]
        self.columns = np.concatenate(
            [np.arange(*self.spans[r]) for r in self.settling]
        ) if self.settling else np.zeros(0, np.int64)
        self.where = np.full(n, -1, dtype=np.int64)
        self.where[self.columns] = np.arange(self.m)
        self.pools = 0 if apical is None else len(apical.members)
        self.takes_compartment = True
        tops: set[int] = set()
        if apical is not None:
            basal, up = _apical_blocks(apical, layout)
            tops = set(up)
            self.takes_compartment = (
                not (set(basal) & tops)  # no block holds synapses of both
                and not (tops & set(self.outer))
                and self.m <= _MLX_COMPARTMENT_NEURONS
            )
        self.per = max(1, -(-self.m // _MLX_LAUNCH_THREADS))
        self.threads = -(-self.m // self.per) if self.m else 0
        table = [self.m, len(self.inner), self.threads, self.pools]
        entries = 0
        for k in self.inner:
            a0, a1, b0, b1 = layout.bounds(k)
            p, q = self.place[self.pre[k]], self.place[self.post[k]]
            table += [p, p + a1 - a0, q, q + b1 - b0, entries, int(k in tops)]
            entries += (a1 - a0) * (b1 - b0)
        rest = neuron_model.rest_emission
        self.constants = (
            neuron_model.slope, neuron_model.threshold, rest, 1.0 / (1.0 - rest),
            neuron_model.leak / rest,
        )
        self.compartment_constants = (
            neuron_model.dt, 0.0 if apical is None else apical.fraction,
            0.0 if apical is None else apical.floor * apical.floor, _APICAL_TINY,
        )
        self.at_rest = float(neuron_model.activation(np.zeros(1))[0])
        cap = 64  # a power of two at least m: one pipeline per size in a process
        while cap < self.m:
            cap *= 2
        if apical is not None:
            cap = min(cap, _MLX_COMPARTMENT_NEURONS)
        # without a compartment no array of its own: its names are never read
        top = (
            "threadgroup float tg_x[CAP]; threadgroup float tg_u[CAP];" if apical is not None
            else "threadgroup float* tg_x = tg_p; threadgroup float* tg_u = tg_q;"
        )
        self.launch = mx.fast.metal_kernel(
            name=f"cadence_settle_{cap}_{self.per}_{int(apical is not None)}",
            input_names=[
                "v0", "s0", "first", "fixed", "w", "plan", "scales", "control", "target", "mask",
                "weight", "group_low", "group_high", "over", "pool_of", "pool_low", "pool_high",
                "given",
            ],
            output_names=["out_v", "out_s", "out_e", "out_c", "out_m", "out_g"],
            source=_MLX_LAUNCH.format(cap=cap, per=self.per, top=top),
            atomic_outputs=False,
            ensure_row_contiguous=True,
        )
        self.activate = mx.compile(self._activation_of(mx, neuron_model))
        self.under_drive = mx.compile(self._under_drive)
        self.ramp = mx.compile(self._ramp)
        self.none = mx.zeros((8,), dtype=mx.float32)  # an input the launch does not read
        self.none_int = mx.zeros((8,), dtype=mx.int32)
        self.pool_of, self.pool_low, self.pool_high = self.none_int, self.none_int, self.none_int
        if apical is not None:
            of, low, high = (np.full(self.m, -1, dtype=np.int32) for _ in range(3))
            for pool, member in enumerate(apical.members):
                places = self.where[np.arange(n)[member]]
                if (places < 0).any() or (np.diff(places) != 1).any():
                    self.takes_compartment = False  # a pool beyond one range of settling neurons
                    break
                of[places], low[places], high[places] = pool, places[0], places[-1] + 1
            self.pool_of, self.pool_low = mx.array(of), mx.array(low)
            self.pool_high = mx.array(high)
        self.plan_array = mx.array(np.array(table + [0] * max(0, 8 - len(table)), dtype=np.int32))
        self._controls: dict[tuple[int, ...], Any] = {}
        self._scales: dict[tuple[float, float], Any] = {}
        self._nudges: list[tuple[Any, Any, Any, Any]] = []

    @staticmethod
    def of(
        mx: Any, layout: Layout, neuron_model: NeuronModel, n: int, apical: Apical | None,
    ) -> _MlxPlan | None:
        """The plan of a layout, or None where the launch does not apply: adaptation, a settling
        set beyond threadgroup memory or below the inputs' least size (MLX declares an input of
        fewer than eight values in another address space), or a compartment whose blocks or pools
        the launch does not hold. ``CADENCE_MLX_LAUNCH=0`` keeps one MLX operation at a time."""
        if os.environ.get("CADENCE_MLX_LAUNCH", "1") == "0":
            return None
        if neuron_model.adaptation is not None or not layout.pairs:
            return None
        plan = _MlxPlan(mx, layout, neuron_model, n, apical)
        fits = 8 <= plan.m <= _MLX_LAUNCH_NEURONS and plan.takes_compartment
        return plan if fits else None

    @staticmethod
    def _activation_of(mx: Any, neuron_model: NeuronModel) -> Any:
        rest = neuron_model.rest_emission
        slope, threshold = neuron_model.slope, neuron_model.threshold
        up, down = 1.0 / (1.0 - rest), neuron_model.leak / rest

        def activate(v: Any) -> Any:
            r = mx.sigmoid(slope * (v - threshold)) - rest
            return mx.where(r > 0.0, r * up, r * down)

        return activate

    def _under_drive(
        self, potentials: list[Any], blocks: list[Any], standing: Any,
    ) -> tuple[Any, ...]:
        """A new drive's work before its sweeps, compiled: the source ranges' activations, their
        products summed per settling neuron (``blocks``, the source blocks), and ``fixed``, the
        standing input at every sweep after the first."""
        mx = self.mx
        sources = [self.activate(p) for p in potentials]
        of = dict(zip(self.sources, sources, strict=True))
        totals: dict[int, Any] = {}
        for block, k in zip(blocks, self.outer, strict=True):
            product = of[self.pre[k]] @ block
            post = self.post[k]
            totals[post] = product + totals[post] if post in totals else product
        rows = standing.shape[0]
        parts = [
            totals[r] if r in totals
            else mx.zeros((rows, self.spans[r][1] - self.spans[r][0]), dtype=mx.float32)
            for r in self.settling
        ]
        outer = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        return sources, outer, standing + outer

    def _ramp(
        self, goals: list[Any], gaps: list[Any], powers: Any, blocks: list[Any], starts: list[Any],
        rows: Any,
    ) -> tuple[Any, ...]:
        """The source ranges along their way to the drive, under another step than ``dt = 1``:
        at the ``powers`` of ``1 - dt``, ``(sweeps, 1, 1)``, their activations, their products
        summed per settling neuron, ``(sweeps, rows, m)``, the activations at the last, the
        movement per sweep and row from ``starts``, the activations before the first, and each
        sweep's largest movement."""
        mx = self.mx
        count = powers.shape[0]
        made = [self.activate(goal + powers * gap) for goal, gap in zip(goals, gaps, strict=True)]
        of = dict(zip(self.sources, made, strict=True))
        totals: dict[int, Any] = {}
        for block, k in zip(blocks, self.outer, strict=True):
            product = of[self.pre[k]] @ block
            post = self.post[k]
            totals[post] = product + totals[post] if post in totals else product
        parts = [
            totals[r] if r in totals
            else mx.zeros(
                (count, rows.shape[0], self.spans[r][1] - self.spans[r][0]), dtype=mx.float32
            )
            for r in self.settling
        ]
        outer = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
        path = mx.zeros((count, rows.shape[0]), dtype=mx.float32)
        peaks = mx.zeros((count,), dtype=mx.float32)
        for at, begin in zip(made, starts, strict=True):
            before = begin[None] if count == 1 else mx.concatenate([begin[None], at[:-1]], axis=0)
            step = mx.abs(at - before)
            path = path + step.sum(axis=2)
            peaks = mx.maximum(peaks, step.max(axis=(1, 2)))
        return outer, [at[-1] for at in made], path, peaks

    def relaxing(self, worst: float) -> int:
        """The sweeps until a source range's largest gap ``worst`` is below rounding."""
        if self.rho == 0.0 or worst <= _MLX_SOURCE_ROUNDING:
            return 1
        return int(np.ceil(np.log(_MLX_SOURCE_ROUNDING / worst) / np.log(self.rho)))

    def controls(
        self, sweeps: int, nudging: bool, shares: bool, bounded: bool, weighted: bool, ramp: int,
        stride: int, compartment: int,
    ) -> Any:
        key = (
            sweeps, int(nudging), int(shares), int(bounded), int(weighted), ramp, stride,
            compartment,
        )
        held = self._controls.get(key)
        if held is None:
            held = self.mx.array(np.array(key, dtype=np.int32))
            if len(self._controls) >= 256:
                self._controls.clear()
            self._controls[key] = held
        return held

    def scales(self, beta: float, tolerance: float) -> Any:
        key = (beta, tolerance)
        held = self._scales.get(key)
        if held is None:
            held = self.mx.array(np.array(
                [*self.constants, beta, tolerance, *self.compartment_constants, 0.0],
                dtype=np.float32,
            ))
            if len(self._scales) >= 16:
                self._scales.clear()
            self._scales[key] = held
        return held

    def nudge_tables(self, nudge: Nudge) -> tuple[Any, Any, Any, Any] | None:
        """The launch's arrays of ``nudge``, kept for the nudges that share its mask, groups and
        temperature: ``(mask, group_low, group_high, over)``. None where the launch does not apply:
        an anchor, a neuron under the mask that does not settle, a group that is not a range of the
        settling neurons, or a temperature whose exponentials float32 cannot hold without the
        host's ladder."""
        if nudge.anchor is not None:
            return None
        heat = np.asarray(nudge.softmax_temperature, dtype=float)
        whole = np.asarray(nudge.mask, dtype=float)
        groups = None if nudge.groups is None else np.asarray(nudge.groups)
        for held_mask, held_groups, held_heat, tables in self._nudges:
            if (
                np.array_equal(held_heat, heat, equal_nan=True) and np.array_equal(held_mask, whole)
                and (held_groups is None) == (groups is None)
                and (groups is None or np.array_equal(held_groups, groups))
            ):
                return cast("tuple[Any, Any, Any, Any] | None", tables)
        tables = self._nudge_arrays(nudge)
        del self._nudges[:-3]
        self._nudges.append((
            whole.copy(), None if groups is None else groups.copy(), heat.copy(), tables,
        ))
        return tables

    def _nudge_arrays(self, nudge: Nudge) -> tuple[Any, Any, Any, Any] | None:
        mx = self.mx
        whole = np.asarray(nudge.mask, dtype=float)
        if (self.where[np.flatnonzero(whole)] < 0).any():
            return None
        mask = np.zeros(self.m, dtype=np.float32)
        if nudge.softmax_temperature is None:
            mask[self.where[np.flatnonzero(whole)]] = whole[whole != 0]
            return mx.array(mask), self.none_int, self.none_int, self.none
        low, high = np.zeros(self.m, dtype=np.int32), np.zeros(self.m, dtype=np.int32)
        over = np.zeros(self.m, dtype=np.float32)
        for group in _softmax_groups(nudge):
            temperature = _group_temperature(nudge, group)
            places = self.where[group]
            if not temperature >= _MLX_LEAST_TEMPERATURE or (
                np.diff(places) != 1
            ).any():
                return None
            mask[places] = whole[group]
            low[places], high[places] = places[0], places[-1] + 1
            over[places] = 1.0 / temperature
        return mx.array(mask), mx.array(low), mx.array(high), mx.array(over)

    def whole(self, handle: _MlxState, key: str) -> Any:
        """The ``(batch, n)`` array of a handle on the device, in neuron order."""
        mx = self.mx
        rows = int(handle["vm"].shape[0])
        if key == "a":
            return mx.array(np.asarray(handle["a_host"], dtype=np.float32))
        part, sources = (handle["sm"], handle["sources"]) if key == "s" else (
            handle["vm"], handle["potentials"],
        )
        pieces = []
        for r in range(len(self.spans)):
            if r in self.place:
                n0, n1 = self.spans[r]
                pieces.append(part[:, self.place[r] : self.place[r] + n1 - n0])
            else:
                pieces.append(mx.array(sources[r]) if key == "v" else sources[r])
        return mx.concatenate(pieces, axis=1).reshape(rows, self.n)

    def fetch(self, handle: _MlxState, key: str) -> np.ndarray:
        """A host array of a handle, in float64, read from the arrays the launch made."""
        if key == "a":
            return np.array(handle["a_host"], dtype=np.float64)
        if key == "change":
            return self._change(handle)
        part, sources = (handle["sm"], handle["sources"]) if key == "s" else (
            handle["vm"], handle["potentials"],
        )
        whole = np.array(part, copy=False)
        out = np.empty((len(whole), self.n))
        for r, (n0, n1) in self.spans.items():
            if r in self.place:
                p = self.place[r]
                out[:, n0:n1] = whole[:, p : p + n1 - n0]
            else:
                out[:, n0:n1] = np.array(sources[r], copy=False)
        return out

    def _change(self, handle: _MlxState) -> np.ndarray:
        """The settle's total movement per row: the launches' sums, and with a first sweep the
        movement of the source ranges onto their drive."""
        rows = int(handle["vm"].shape[0])
        total = np.zeros(rows)
        for moved in handle["moved"]:
            total += np.array(moved, copy=False).sum(axis=1)
        for before, path, count in handle.get("source_moves", ()):
            if path is not None:  # a ramp: the movement per sweep, of the sweeps taken
                total += np.array(path, copy=False)[:count].sum(axis=0)
            elif before is not None:  # dt = 1: one sweep moved them onto their drive
                for r, after in handle["sources"].items():
                    step = np.array(after, copy=False) - np.array(before[r], copy=False)
                    total += np.abs(step).sum(axis=1)
        return total


_MLX_VERSIONS = itertools.count()  # each set of a kernel's parameters: the products a state keeps


class _MlxKernel:
    """Block settling on Apple silicon through MLX: float32, unified memory, lazy graphs."""

    def __init__(
        self,
        connectome: Connectome,
        weights: np.ndarray,
        bias: np.ndarray,
        neuron_model: NeuronModel,
        layout: Layout,
        apical: Apical | None = None,
    ) -> None:
        device, reason = _mlx_probe()
        if device is None:  # an MLX that aborts on import would take this interpreter down
            missing = ModuleNotFoundError if reason == _MLX_MISSING else ImportError
            raise missing(f"the mlx backend is unavailable here: {reason}", name="mlx")
        import mlx.core as mx

        neuron_model._validate_precision("float32")
        self.mx = mx
        self.backend_name = "mlx"
        self.neuron_model = neuron_model
        self.n = connectome.n
        self.layout = layout
        self.apical = apical
        self.root = self  # with_parameters' kernels share it: a state continues on any of them
        self.plain = os.environ.get("CADENCE_MLX_LAUNCH", "1") == "0"
        self.sweeps: dict[tuple[Any, ...], Any] = {}  # the compiled sweeps, shared with those too
        self.read_blocks: dict[int, tuple[np.ndarray, Any]] = {}  # a read's synapses, their blocks
        self.lesson_steps: dict[tuple[bool, bool, bool], Any] = {}  # the optimizer's, compiled
        self.reward_steps: dict[bool, Any] = {}  # a reward's eligibility and step, compiled
        self._activate = self._activation if self.plain else mx.compile(self._activation)
        self.plan = _MlxPlan.of(mx, layout, neuron_model, connectome.n, apical)
        # the blocks of each postsynaptic range, basal and apical apart
        basal, up = (
            (range(layout.pairs), ()) if apical is None else _apical_blocks(apical, layout)
        )
        self.mixed = bool(set(basal) & set(up))  # a block holds synapses of both: masked copies
        self._take(weights, bias)
        self.by_post: list[list[int]] = [[] for _ in range(layout.ranges)]
        self.by_post_up: list[list[int]] = [[] for _ in range(layout.ranges)]
        for k in basal:
            self.by_post[int(layout.pair_post[k])].append(k)
        for k in up:
            self.by_post_up[int(layout.pair_post[k])].append(k)
        self.pool_members = [] if apical is None else [
            mx.array(np.arange(connectome.n)[member]) for member in apical.members
        ]
        self.edge_index = mx.array(layout.edge_index)

    def _take(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """The bias and the weight blocks on the device, made from the host ``weights`` and
        ``bias``; refused where float32 cannot hold them."""
        mx, layout = self.mx, self.layout
        self.scale: list[Any] | None = None  # the efficacies by block, once a lesson moved them
        with np.errstate(over="ignore", invalid="ignore"):
            live_bias = bias.astype(np.float32)
            flat = layout.flat(weights).astype(np.float32)
        if not np.isfinite(live_bias).all() or not np.isfinite(flat).all():
            raise ValueError(
                "device parameters and effective weights must be finite at runtime precision"
            )
        self.bias = mx.array(live_bias)
        hosted = layout.blocks(flat)
        self.blocks = [mx.array(np.ascontiguousarray(b, dtype=np.float32)) for b in hosted]
        self.up_blocks = self.blocks
        if self.mixed:  # the synapses of each side in a block of its own
            assert self.apical is not None
            mask = self.apical.synapses
            self.blocks, self.up_blocks = (
                [mx.array(np.ascontiguousarray(b, dtype=np.float32)) for b in layout.blocks(
                    layout.flat(np.where(side, weights, 0.0)).astype(np.float32)
                )]
                for side in (~mask, mask)
            )
        plan = self.plan
        if plan is not None:  # the launch reads a column per settling neuron
            self.version = next(_MLX_VERSIONS)
            self.w = mx.array(np.concatenate(
                [hosted[k].T.ravel() for k in plan.inner] + [np.zeros(8, dtype=np.float32)]
            ))
            self.bias_host = live_bias

    def __deepcopy__(self, memo: dict[int, Any]) -> _MlxKernel:
        return self  # a kernel's arrays are never written: a new set of parameters is a new kernel

    def with_parameters(self, weights: np.ndarray, bias: np.ndarray) -> _MlxKernel:
        """A kernel of its own on new parameters: its bias and blocks made from the host
        ``weights`` and ``bias`` as construction makes them, the layout's tables shared. This
        kernel keeps its own blocks, so a brain holding it still settles under its weights."""
        new = copy.copy(self)
        new._take(weights, bias)
        return new

    def _activation(self, v: Any) -> Any:
        mx, neuron_model = self.mx, self.neuron_model
        rest = neuron_model.rest_emission
        r = mx.sigmoid(neuron_model.slope * (v - neuron_model.threshold)) - rest
        s = mx.maximum(r, 0.0) / (1.0 - rest)
        if neuron_model.leak:
            s = s + neuron_model.leak * mx.minimum(r, 0.0) / rest
        return s

    def _synaptic_input(self, s: Any, blocks: Sequence[Any], by_post: Sequence[list[int]]) -> Any:
        """Every block's product at every step: the device makes a product in less time than a
        host synchronization would take to find its range unchanged."""
        mx, lay = self.mx, self.layout
        parts = []
        batch = s.shape[0]
        for r in range(lay.ranges):
            b0, b1 = int(lay.starts[r]), int(lay.starts[r + 1])
            total = None
            for k in by_post[r]:
                a0, a1, _, _ = lay.bounds(k)
                product = s[:, a0:a1] @ blocks[k]
                total = product if total is None else total + product
            parts.append(mx.zeros((batch, b1 - b0), dtype=mx.float32) if total is None else total)
        return mx.concatenate(parts, axis=1)

    def _sweep(
        self, peak: bool, v: Any, s: Any, a: Any, change: Any, standing: Any, blocks: Sequence[Any],
        kept: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        """One sweep of every neuron, the neuron rule's own order. ``kept`` holds what this sweep's
        kind reads: ``k`` with a mask; ``beta``, ``target`` and ``mask`` with a nudge, ``groups``
        with its softmax (members and temperature), ``weight`` and ``anchor`` with ``anchor_gain``
        where it has them; ``up`` with an apical compartment, its blocks, and ``held`` the gains a
        nudged settle holds. With ``peak`` the step's largest movement comes with it."""
        mx, neuron_model = self.mx, self.neuron_model
        adapt = neuron_model.adaptation
        total = self._synaptic_input(s, blocks, self.by_post) + standing
        if adapt is not None:
            total = total - adapt.strength * a
        k, target, groups = kept.get("k"), kept.get("target"), kept.get("groups")
        if target is not None:
            beta, mask = kept["beta"], kept["mask"]
            if groups is None:
                push = beta * (target - s) * mask
            else:  # one softmax per group of competing neurons
                push = mx.zeros_like(s)
                for members, temperature in groups:
                    logits = s[:, members]
                    tiny = float(np.finfo(np.float32).tiny)
                    if isinstance(temperature, float) and temperature < tiny:
                        logits = logits - mx.max(logits, axis=1, keepdims=True)
                        while temperature < tiny:
                            logits = logits / tiny
                            temperature /= tiny
                    p = mx.softmax(logits / temperature, axis=1)
                    push[:, members] = beta * (target[:, members] - p) * mask[members]
            if "weight" in kept:
                push = push * kept["weight"]
            if "anchor" in kept:
                push = push + kept["anchor_gain"] * (kept["anchor"] - s)
            total = total + push
        if "up" in kept:  # the top-down input, normalized to the basal input it joins
            raw = self._synaptic_input(s, kept["up"], self.by_post_up)
            total = total + self._apical_drive(total, raw, kept.get("held"))
        v = v + neuron_model.dt * (-v + total)
        if k is not None:
            v = v * k
        previous = s
        s = self._activation(v)
        if k is not None:
            s = s * k
        if adapt is not None:
            a = a + (s - a) / adapt.tau_steps
        movement = mx.abs(s - previous)
        change = change + movement.sum(axis=1)
        if peak:  # the stop reads the step's largest movement, made with it
            return v, s, a, change, movement.max()
        return v, s, a, change

    def _apical_drive(self, basal: Any, raw: Any, held: Any) -> Any:
        """``Apical.drive`` on the device: every pool's gain from its mean squares, per row, or
        the ``held`` gains ``(batch, pools)`` of a nudged settle."""
        mx, apical = self.mx, self.apical
        assert apical is not None
        floor2 = apical.floor * apical.floor
        gain = mx.zeros_like(raw)
        for k, members in enumerate(self.pool_members):
            if held is None:
                b, u = basal[:, members], raw[:, members]
                b2, u2 = mx.mean(b * b, axis=1), mx.mean(u * u, axis=1)
                one = apical.fraction * mx.sqrt(b2) / mx.sqrt(u2 + floor2 * b2 + _APICAL_TINY)
            else:
                one = held[:, k]
            gain[:, members] = one[:, None]
        return gain * raw

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
        mx = self.mx
        plan = self.plan
        if plan is not None and not want and capture is None and (keep == 1.0).all():
            tables = None if nudge is None else plan.nudge_tables(nudge)
            if nudge is None or tables is not None:
                return self._launched(v0, a0, drive, steps, nudge, tables, tolerance, state)

        def to(x: np.ndarray) -> Any:
            return mx.array(np.asarray(x, dtype=np.float32))

        handle = state.device if state is not None and state.device is not None else None
        if handle is not None and handle.get("holder") is self.root:
            v, a = handle["v"], handle["a"]
        else:
            v, a = to(v0), to(a0)
        d, k = to(drive), to(keep)
        batch = v.shape[0]
        masked = bool((keep != 1.0).any())
        target = mask = weight = anchor = anchor_gain = None
        groups: list[Any] = []
        if nudge is not None:
            target = to(np.broadcast_to(nudge.target, (batch, self.n)))
            mask = to(nudge.mask)
            groups = [
                (mx.array(members), self._heat(_group_temperature(nudge, members)))
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
        s = self._activate(v)
        if masked:
            s = s * k
        standing = d + self.bias
        if capture is not None:
            capture.step(np.array(v), np.array(s), np.array(a))
        bounded = tolerance is not None
        kept: dict[str, Any] = {"k": k} if masked else {}
        if nudge is not None:
            kept.update(
                beta=mx.array(float(nudge.beta), dtype=mx.float32), target=target, mask=mask,
            )
            if nudge.softmax_temperature is not None:
                kept["groups"] = groups
            if weight is not None:
                kept["weight"] = weight
            if anchor is not None:
                kept.update(anchor=anchor, anchor_gain=anchor_gain)
            if self.apical is not None and nudge.apical_gain is not None:
                kept["held"] = to(np.asarray(nudge.apical_gain, dtype=float))
        if self.apical is not None:
            kept["up"] = self.up_blocks
        sweep = self._sweep_of(bounded, kept)
        for t in range(steps):
            out = sweep(v, s, a, activity_change, standing, self.blocks, kept)
            v, s, a, activity_change = out[:4]
            if bounded:
                mx.eval(*out)
            elif (t + 1) % _MLX_GRAPH_STEPS == 0:
                mx.eval(v, s, a, activity_change)
            if capture is not None:
                capture.step(np.array(v), np.array(s), np.array(a))
            if want:
                traj.append(np.array(s, dtype=np.float64))
            taken = t + 1
            if bounded and tolerance is not None and float(out[4].item()) < tolerance:
                break
        return (
            np.array(v, dtype=np.float64),
            np.array(a, dtype=np.float64),
            np.array(s, dtype=np.float64),
            (np.stack(traj) if traj else np.empty((0, batch, self.n))) if want else None,
            taken,
            np.array(activity_change, dtype=np.float64),
            {"kernel": "mlx", "holder": self.root, "v": v, "a": a, "s": s},
        )

    # -- a lesson's update on the device: the same arithmetic per block entry, in float32

    def update_tables(
        self, gain_pre: np.ndarray, paired: np.ndarray, reverse: np.ndarray,
    ) -> dict[str, Any] | None:
        """What the update on the device reads of the synapses, or None where it does not apply:
        too few entries, an entry without one synapse, a gain that varies within a block, or a tie
        that is not one block's entries with another's, transposed. Each block's gain and the
        block it is tied to."""
        lay = self.layout
        if lay.size < _MLX_UPDATE_ENTRIES or lay.parallel_synapses or self.mixed:
            return None
        block = np.asarray(lay.block_of_edge)
        entries = np.diff(lay.offset)
        if len(block) != lay.size or (np.bincount(block, minlength=lay.pairs) != entries).any():
            return None
        low, high = np.full(lay.pairs, np.inf), np.full(lay.pairs, -np.inf)
        np.minimum.at(low, block, gain_pre)
        np.maximum.at(high, block, gain_pre)
        if (low != high).any():
            return None
        ties: dict[int, int] = {}
        if len(paired):
            at = np.asarray(lay.edge_index)
            k, back = block[paired], block[reverse]
            widths = np.array([lay.bounds(j)[3] - lay.bounds(j)[2] for j in range(lay.pairs)])
            i, j = np.divmod(at[paired] - lay.offset[k], widths[k])
            i2, j2 = np.divmod(at[reverse] - lay.offset[back], widths[back])
            whole = np.bincount(k, minlength=lay.pairs)
            if (i != j2).any() or (j != i2).any() or ((whole != 0) & (whole != entries)).any():
                return None
            for one, other in zip(*np.unique(np.stack([k, back]), axis=1), strict=True):
                if ties.setdefault(int(one), int(other)) != int(other):
                    return None
        return {
            "gains": [float(g) for g in low], "entries": [int(e) for e in entries], "ties": ties,
            "synapses": len(block),
        }

    def to_blocks(self, edges: np.ndarray) -> list[Any]:
        """Per-synapse host values as device arrays by block; an entry without a synapse reads 0."""
        flat = self.layout.flat(np.asarray(edges, dtype=float)).astype(np.float32)
        return [self.mx.array(np.ascontiguousarray(part)) for part in self.layout.blocks(flat)]

    def to_edges(self, blocks: Sequence[Any]) -> np.ndarray:
        """Device arrays by block as the host's per-synapse values, in float64."""
        flat = np.concatenate([np.array(b, copy=False).ravel() for b in blocks])
        return np.asarray(flat[self.layout.edge_index], dtype=np.float64)

    def host_scale(self) -> np.ndarray:
        """The efficacies a lesson on the device left, as the host's per-synapse array."""
        assert self.scale is not None
        return self.to_edges(self.scale)

    def read_back(
        self, groups: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
        movement: np.ndarray,
    ) -> list[np.ndarray] | None:
        """``movement @ W.T`` per group of synapses ``(synapses, rows, cols, width)`` read as
        ``W[rows, cols]``, on the blocks, one launch for all; None where a group is not whole rows
        of blocks in their order."""
        mx = self.mx
        parts = [self._read_blocks(group) for group in groups]
        if any(part is None for part in parts):
            return None
        m = mx.array(np.asarray(movement, dtype=np.float32)).T
        products = [
            [self.blocks[k][i0:i1] @ m for k, _, i0, i1 in part or ()] for part in parts
        ]
        mx.eval(products)
        out = []
        for group, part, made in zip(groups, parts, products, strict=True):
            row = np.empty((len(movement), group[3]))
            for (_, at, i0, i1), product in zip(part or (), made, strict=True):
                row[:, at : at + i1 - i0] = np.array(product, copy=False).T
            out.append(row)
        return out

    def _read_blocks(
        self, group: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    ) -> list[tuple[int, int, int, int]] | None:
        """The block rows of a group of synapses, ``(block, the group's row they start at, first
        row, row past the last)``, kept with the synapses; None where they are not whole rows of
        blocks in their order."""
        synapses, rows, cols, _ = group
        lay, known = self.layout, self.root.read_blocks
        held = known.get(id(synapses))
        if held is not None:
            return cast("list[tuple[int, int, int, int]] | None", held[1])
        block = np.asarray(lay.block_of_edge)[synapses]
        rows, cols = np.asarray(rows), np.asarray(cols)
        at = np.asarray(lay.edge_index)[synapses]
        parts: list[tuple[int, int, int, int]] | None = []
        for k in (int(one) for one in np.unique(block)):
            mine = block == k
            a0, a1, b0, b1 = lay.bounds(k)
            width = b1 - b0
            row_in_block, col_in_block = np.divmod(at[mine] - lay.offset[k], width)
            start, i0 = int(rows[mine].min()), int(row_in_block.min())
            i1 = int(row_in_block.max()) + 1
            if (
                int(mine.sum()) != (i1 - i0) * width
                or (row_in_block - i0 != rows[mine] - start).any()
                or (col_in_block != cols[mine]).any()
            ):
                parts = None
                break
            assert parts is not None
            parts.append((k, start, i0, i1))
        known[id(synapses)] = (synapses, parts)  # the synapses kept: their id stays theirs
        return parts

    def _ranges_of(self, handle: Mapping[str, Any]) -> Any:
        """A state's activations of a neuron range, ``(rows, width)``, wherever it keeps them."""
        plan = self.plan
        if plan is not None and isinstance(handle, _MlxState):
            sm, sources = handle["sm"], handle["sources"]

            def part(r: int) -> Any:
                if r in plan.place:
                    n0, n1 = plan.spans[r]
                    return sm[:, plan.place[r] : plan.place[r] + n1 - n0]
                return sources[r]

            return part
        s = handle["s"]
        starts = self.layout.starts
        return lambda r: s[:, int(starts[r]) : int(starts[r + 1])]

    def _lesson_step_of(self, momentum: bool, normalize: bool, relative: bool) -> Any:
        """One block's step of the optimizer, compiled once per kind and shape: the arithmetic of
        ``fused.optimizer_step`` per entry, fused into few passes over the block. Its numbers are
        ``norm, m, 1 - m, m's correction, rho, 1 - rho, rho's correction, eta, floor, share, tiny,
        entries``."""
        held = self.root.lesson_steps.get((momentum, normalize, relative))
        if held is not None:
            return held
        mx = self.mx

        def lesson_step(raw: Any, velocity: Any, second: Any, numbers: Sequence[Any]) -> Any:
            norm, m, keep_m, m_c, rho, keep_rho, rho_c, eta, floor, share, tiny, entries = numbers
            raw = raw / norm
            term = raw
            if momentum:
                velocity = m * velocity + keep_m * raw
                term = velocity / m_c
            if normalize:
                second = rho * second + keep_rho * (raw * raw)
                root = mx.sqrt(second / rho_c)
                if relative:
                    term = term / (root + share * mx.sqrt(mx.sum(root * root) / entries) + tiny)
                else:
                    term = term / (root + floor)
            return eta * term, velocity, second

        held = mx.compile(lesson_step)
        self.root.lesson_steps[(momentum, normalize, relative)] = held
        return held

    def taught(
        self, plus: Mapping[str, Any], minus: Mapping[str, Any], norm: float,
        moments: dict[str, Any], tables: dict[str, Any], numbers: Sequence[float], cap: float,
        momentum: bool, normalize: bool, relative: bool,
    ) -> tuple[_MlxKernel, dict[str, Any], Any, float]:
        """A lesson's update of the synapses, all of it on the device: the block Gram contrast of
        the two phases over ``norm``, the optimizer, the ties, the bound and the new blocks. One
        launch at its end. Returns the new kernel (without its bias, the host's business), the
        moments, the neurons' raw contrast as a host array and the mean step's size; a value that
        is not finite raises ``ValueError`` with nothing changed."""
        mx, lay = self.mx, self.layout
        scale = self.scale if self.scale is not None else moments["scale"]
        plus_of, minus_of = self._ranges_of(plus), self._ranges_of(minus)
        shared = plus.get("sources") if plus.get("sources") is minus.get("sources") else None
        plan = self.plan
        constants = [mx.array(float(x), dtype=mx.float32) for x in (norm, *numbers)]
        lesson_step = self._lesson_step_of(momentum, normalize, relative)
        steps: list[Any] = []
        velocity, second = list(moments["velocity"]), list(moments["second_moment"])
        # each range's movement and each postsynaptic Gram factor, once a range
        moved = {r: plus_of(r) - minus_of(r) for r in {int(x) for x in lay.pair_post}}
        for k in range(lay.pairs):
            pre, post = int(lay.pair_pre[k]), int(lay.pair_post[k])
            a_plus, b_minus = plus_of(pre), minus_of(post)
            raw = a_plus.T @ moved[post]
            if shared is None or plan is None or pre in plan.place:
                raw = raw + (a_plus - minus_of(pre)).T @ b_minus
            entries = mx.array(float(tables["entries"][k]), dtype=mx.float32)
            step, velocity[k], second[k] = lesson_step(
                raw, velocity[k], second[k], [*constants, entries]
            )
            steps.append(step)
        new, total = self._moved_by(steps, scale, tables, cap)
        ok = mx.all(mx.stack([mx.isfinite(mx.sum(x)) for x in (*velocity, *second)]))
        neurons = self._movement_of_neurons(plus, minus, plus_of, minus_of)
        mx.eval(new.blocks, new.scale, velocity, second, total, ok, neurons, new.w if plan else [])
        if not bool(ok.item()) or not np.isfinite(float(total.item())):
            raise ValueError("phase contrasts and optimizer steps must remain finite")
        size = float(total.item()) / max(tables["synapses"], 1)
        kept = {"holder": self.root, "velocity": velocity, "second_moment": second}
        return new, kept, np.array(neurons, dtype=np.float64), size

    def _moved_by(
        self, steps: list[Any], scale: Sequence[Any], tables: dict[str, Any], cap: float,
    ) -> tuple[_MlxKernel, Any]:
        """The kernel after steps by block, not yet evaluated, and the sum of the steps' sizes:
        the ties, the bound and the new blocks. ``steps`` is written with the ties' means."""
        mx, plan = self.mx, self.plan
        for k, other in tables["ties"].items():  # one synapse, one weight: the mean of the two
            if k <= other:
                mean = 0.5 * (steps[k] + steps[other].T)
                steps[k], steps[other] = mean, mean.T if k != other else mean
        new_scale = [mx.clip(e + step, -cap, cap) for e, step in zip(scale, steps, strict=True)]
        new_blocks = [
            e if tables["gains"][k] == 1.0 else tables["gains"][k] * e
            for k, e in enumerate(new_scale)
        ]
        total = mx.sum(mx.stack([mx.sum(mx.abs(step)) for step in steps]))
        new = copy.copy(self)
        new.blocks = new_blocks
        new.up_blocks = new_blocks
        new.scale = new_scale
        if plan is not None:
            pieces = [new_blocks[k].T.reshape(-1) for k in plan.inner]
            new.w = mx.concatenate([*pieces, mx.zeros((8,), dtype=mx.float32)])
            new.version = next(_MLX_VERSIONS)
        return new, total

    # -- a reward's update on the device: every stream's eligibility, by block

    def _reward_step_of(self, moves_pre: bool) -> Any:
        """One block's eligibility and its dopamine's step, compiled once per kind and shape:
        ``e <- decay e + contrast / span`` per row, the contrast a row's own
        ``a+ (b+ - b-) + (a+ - a-) b-``, and ``eta`` times the mean over rows of ``d e``. A block
        whose presynaptic range stands in both phases has no second term. Each row's mean size
        comes with them."""
        held = self.root.reward_steps.get(moves_pre)
        if held is not None:
            return held
        mx = self.mx

        def step(trace: Any, a_plus: Any, a_minus: Any, moved: Any, b_minus: Any,
                 numbers: Sequence[Any]) -> tuple[Any, Any, Any]:
            decay, span, d, eta = numbers
            contrast = a_plus[:, :, None] * moved[:, None, :]
            if moves_pre:
                contrast = contrast + (a_plus - a_minus)[:, :, None] * b_minus[:, None, :]
            new = decay * trace + contrast / span
            return new, eta * (d * new).mean(axis=0), mx.abs(new).mean(axis=(1, 2))

        held = self.mx.compile(step)
        self.root.reward_steps[moves_pre] = held
        return held

    def rewarded(
        self, plus: Mapping[str, Any], minus: Mapping[str, Any], trace: Sequence[Any] | None,
        decay: np.ndarray, span: float, delta: np.ndarray, eta: float, scale: Sequence[Any],
        tables: dict[str, Any], cap: float,
    ) -> tuple[_MlxKernel, list[Any], np.ndarray, float]:
        """A reward's update of the synapses, all of it on the device: each row's eligibility by
        block, ``decay`` its own factor, the dopamine ``delta``'s mean step at ``eta``, the ties,
        the bound and the new blocks. One launch at its end. Returns the new kernel (without its
        bias, the host's business), the eligibility, each row's mean size of it and the mean step's
        size; a value that is not finite raises ``ValueError`` with nothing changed."""
        mx, lay = self.mx, self.layout
        rows = len(delta)
        plus_of, minus_of = self._ranges_of(plus), self._ranges_of(minus)
        shared = plus.get("sources") if plus.get("sources") is minus.get("sources") else None
        plan = self.plan
        numbers = [
            mx.array(np.asarray(decay, dtype=np.float32))[:, None, None],
            mx.array(float(span), dtype=mx.float32),
            mx.array(np.asarray(delta, dtype=np.float32))[:, None, None],
            mx.array(float(eta), dtype=mx.float32),
        ]
        moved = {r: plus_of(r) - minus_of(r) for r in {int(x) for x in lay.pair_post}}
        new_trace, steps, sizes = [], [], []
        for k in range(lay.pairs):
            pre, post = int(lay.pair_pre[k]), int(lay.pair_post[k])
            a0, a1, b0, b1 = lay.bounds(k)
            held = (
                mx.zeros((rows, a1 - a0, b1 - b0), dtype=mx.float32) if trace is None else trace[k]
            )
            moves_pre = shared is None or plan is None or pre in plan.place
            one, step, size = self._reward_step_of(moves_pre)(
                held, plus_of(pre), minus_of(pre), moved[post], minus_of(post), numbers,
            )
            new_trace.append(one)
            steps.append(step)
            sizes.append((a1 - a0) * (b1 - b0) * size)
        new, total = self._moved_by(steps, scale, tables, cap)
        each = mx.sum(mx.stack(sizes), axis=0)
        ok = mx.isfinite(total)
        mx.eval(new.blocks, new.scale, new_trace, each, total, ok, new.w if plan else [])
        if not bool(ok.item()):
            raise ValueError("steps must be finite vectors, one per synapse and one per neuron")
        return (
            new, new_trace, np.array(each, dtype=np.float64) / max(tables["synapses"], 1),
            float(total.item()) / max(tables["synapses"], 1),
        )

    def rows_to_blocks(self, edges: np.ndarray) -> list[Any]:
        """A per-row, per-synapse host array as device arrays by block, ``(rows, a, b)``."""
        mx, lay = self.mx, self.layout
        rows = [lay.blocks(lay.flat(row).astype(np.float32)) for row in np.asarray(edges, float)]
        return [
            mx.array(np.stack([np.ascontiguousarray(row[k]) for row in rows]))
            for k in range(lay.pairs)
        ]

    def rows_to_edges(self, blocks: Sequence[Any]) -> np.ndarray:
        """Device arrays by block, ``(rows, a, b)``, as the host's per-row, per-synapse array."""
        rows = int(blocks[0].shape[0]) if blocks else 0
        flat = np.concatenate([np.array(b, copy=False).reshape(rows, -1) for b in blocks], axis=1)
        return np.asarray(flat[:, self.layout.edge_index], dtype=np.float64)

    def _movement_of_neurons(self, plus: Any, minus: Any, plus_of: Any, minus_of: Any) -> Any:
        """Each neuron's activation movement between two phases, summed over the rows, ``(n,)``."""
        mx, lay = self.mx, self.layout
        pieces = [(plus_of(r) - minus_of(r)).sum(axis=0) for r in range(lay.ranges)]
        return mx.concatenate(pieces)

    def with_bias(self, bias: np.ndarray) -> None:
        """This kernel's bias, from the host's array."""
        live = bias.astype(np.float32)
        self.bias = self.mx.array(live)
        self.bias_host = live

    def _heat(self, temperature: float) -> Any:
        """A softmax's temperature as a compiled sweep reads it, an input a new value does not
        trace again; a temperature below float32's least normal stays a number, for its ladder."""
        if temperature < float(np.finfo(np.float32).tiny):
            return float(temperature)
        return self.mx.array(float(temperature), dtype=self.mx.float32)

    def _sweep_of(self, peak: bool, kept: Mapping[str, Any]) -> Any:
        """The sweep, compiled once per kind and shape: the neuron update fuses into one operation
        and a replay costs a fraction of building its graph. ``CADENCE_MLX_LAUNCH=0``, and a
        temperature below float32's least normal, keep one operation at a time."""
        def sweep(*inputs: Any) -> tuple[Any, ...]:
            return self._sweep(peak, *inputs)

        groups = kept.get("groups", ())
        if self.plain or any(isinstance(temperature, float) for _, temperature in groups):
            return sweep
        kind = (peak, *sorted(kept))
        held = self.sweeps.get(kind)
        if held is None:
            held = self.mx.compile(sweep)
            self.sweeps[kind] = held
        return held

    # -- the launch: a chunk of sweeps in one Metal dispatch

    def _start(
        self, v0: np.ndarray | None, a0: np.ndarray | None, state: BrainState | None, rows: int,
    ) -> _MlxState:
        """A settle's start on the device: the state's own handle where it rests here; else its
        arrays cut by range, uploaded where they stand on the host."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        handle = None if state is None else state.device
        if isinstance(handle, _MlxState) and handle.get("holder") is self.root:
            return handle
        spans = plan.spans
        if handle is not None and handle.get("holder") is self.root:  # the operations' whole arrays
            v, s = handle["v"], handle["s"]
            parts = [(v[:, slice(*spans[r])], s[:, slice(*spans[r])]) for r in plan.settling]
            vm = mx.concatenate([p[0] for p in parts], axis=1)
            sm = mx.concatenate([p[1] for p in parts], axis=1)
            whole = np.array(v, copy=False)
            potentials = {r: whole[:, slice(*spans[r])] for r in plan.sources}
            sources = {r: s[:, slice(*spans[r])] for r in plan.sources}
            a_host = np.array(handle["a"], dtype=np.float64)
            rest = False
        else:
            assert v0 is not None and a0 is not None
            rest = state is None
            vm = mx.array(np.asarray(v0[:, plan.columns], dtype=np.float32))
            potentials = {
                r: np.asarray(v0[:, slice(*spans[r])], dtype=np.float32) for r in plan.sources
            }
            if rest:
                sm = mx.array(np.full((rows, plan.m), plan.at_rest, dtype=np.float32))
                sources = {
                    r: mx.array(np.full(
                        (rows, spans[r][1] - spans[r][0]), plan.at_rest, dtype=np.float32
                    ))
                    for r in plan.sources
                }
            else:
                sm = plan.activate(vm)
                sources = {r: plan.activate(mx.array(p)) for r, p in potentials.items()}
            a_host = a0
        return _MlxState(
            kernel="mlx", holder=self.root, plan=plan, vm=vm, sm=sm, sources=sources,
            potentials=potentials, a_host=a_host, moved=[], source_moves=[], rest=rest,
        )

    def _products(self, sources: Mapping[int, Any], rows: int) -> Any:
        """Every source range's products, summed per settling neuron, ``(rows, m)``; None without
        a source block."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        if not plan.outer:
            return None
        totals: dict[int, Any] = {}
        for k in plan.outer:
            product = sources[plan.pre[k]] @ self.blocks[k]
            post = plan.post[k]
            totals[post] = product + totals[post] if post in totals else product
        parts = []
        for r in plan.settling:
            part = totals.get(r)
            if part is None:
                width = plan.spans[r][1] - plan.spans[r][0]
                part = mx.array(np.zeros((rows, width), dtype=np.float32))
            parts.append(part)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)

    def _driven(self, start: _MlxState, drive: np.ndarray, sweeps: int) -> dict[str, Any]:
        """What the sweeps of a launch of ``sweeps`` from ``start`` under ``drive`` read: ``first``,
        the settling neurons' standing input at each of the first ``ramp`` sweeps, ``fixed``, at
        every later one, and the source ranges' ``goals``, ``gaps``, activations and products at
        the ramp's end. ``needed`` counts the sweeps until the sources stand at their drive. A
        state that settled there under this drive and these parameters already holds them."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        held = start.get("drive")
        rows = len(drive)
        same = bool(start.get("relaxed")) and np.array_equal(held, drive)
        bias = self.bias_host
        if same:
            fixed, outer = start.get("fixed"), start.get("outer")
            if start.get("version") != self.version:  # the sources stand, under new parameters
                settling = np.asarray(drive[:, plan.columns], dtype=np.float32) + bias[plan.columns]
                outer = self._products(start["sources"], rows)
                fixed = mx.array(settling) if outer is None else mx.array(settling) + outer
            return {"first": fixed, "fixed": fixed, "ramp": 1, "needed": 1, "made": [],
                    "sources": start["sources"], "goals": start["goals"], "gaps": None,
                    "outer": outer, "before": None, "path": None, "peaks": None}
        settling = np.asarray(drive[:, plan.columns], dtype=np.float32) + bias[plan.columns]
        standing = mx.array(settling)
        goals = {
            r: np.asarray(drive[:, slice(*plan.spans[r])], dtype=np.float32)
            + bias[slice(*plan.spans[r])]
            for r in plan.sources
        }
        gaps = {r: start["potentials"][r] - goals[r] for r in plan.sources}
        kept = None
        if not (start.get("rest") and plan.at_rest == 0.0):
            kept = start.get("outer") if start.get("version") == self.version else None
            if kept is None:
                kept = self._products(start["sources"], rows)
        if plan.rho != 0.0:
            return self._ramped(start, goals, gaps, kept, standing, sweeps)
        if not plan.outer:  # nothing the sources reach: their activations and no product
            sources = {r: plan.activate(mx.array(p)) for r, p in goals.items()}
            outer, fixed = None, standing
        else:
            activated, outer, fixed = plan.under_drive(
                [mx.array(goals[r]) for r in plan.sources],
                [self.blocks[k] for k in plan.outer], standing,
            )
            sources = dict(zip(plan.sources, activated, strict=True))
        first = standing if kept is None else standing + kept  # None: the products of silence
        return {"first": first, "fixed": fixed, "ramp": 1, "needed": 1, "made": sources,
                "sources": sources, "goals": goals, "gaps": gaps, "outer": outer,
                "before": start["sources"], "path": None, "peaks": None}

    def _ramped(
        self, start: _MlxState, goals: dict[int, np.ndarray], gaps: dict[int, np.ndarray],
        kept: Any, standing: Any, sweeps: int,
    ) -> dict[str, Any]:
        """``_driven`` under another step than ``dt = 1``: the source ranges still move, so every
        sweep of the ramp reads its own products, made in parts of a bounded size."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        worst = max((float(np.abs(g).max(initial=0.0)) for g in gaps.values()), default=0.0)
        needed = plan.relaxing(worst)
        ramp = max(1, min(needed, sweeps))
        powers = plan.rho ** np.arange(1, ramp + 1, dtype=np.float64)
        rows = int(standing.shape[0])
        widest = max((g.shape[1] for g in gaps.values()), default=1)
        each = max(1, _MLX_RAMP_FLOATS // (rows * widest))
        goal = [mx.array(goals[r]) for r in plan.sources]
        gap = [mx.array(gaps[r]) for r in plan.sources]
        blocks = [self.blocks[k] for k in plan.outer]
        at = [start["sources"][r] for r in plan.sources]
        stacks, paths, peaks = [], [], []
        for begin in range(0, ramp, each):
            part = mx.array(powers[begin : begin + each].astype(np.float32))[:, None, None]
            stack, at, path, peak = plan.ramp(goal, gap, part, blocks, at, standing)
            stacks.append(stack)
            paths.append(path)
            peaks.append(peak)
        stack = stacks[0] if len(stacks) == 1 else mx.concatenate(stacks, axis=0)
        path = paths[0] if len(paths) == 1 else mx.concatenate(paths, axis=0)
        peak = peaks[0] if len(peaks) == 1 else mx.concatenate(peaks, axis=0)
        outer = stack[-1]
        begun = mx.zeros_like(outer) if kept is None else kept  # None: the products of silence
        first = standing + begun if ramp == 1 else (
            standing + mx.concatenate([begun[None], stack[:-1]], axis=0)
        )
        sources = dict(zip(plan.sources, at, strict=True))
        return {"first": first, "fixed": standing + outer, "ramp": ramp, "needed": needed,
                "made": [sources, path, peak], "sources": sources, "goals": goals, "gaps": gaps,
                "outer": outer, "before": None, "path": path, "peaks": peak}

    def _nudged_inputs(self, nudge: Nudge, tables: Any, rows: int) -> dict[str, Any]:
        mx, plan = self.mx, self.plan
        assert plan is not None
        mask, group_low, group_high, over = tables
        target = np.broadcast_to(nudge.target, (rows, self.n))[:, plan.columns]
        weight = plan.none
        if nudge.weight is not None:
            values = np.asarray(nudge.weight, dtype=np.float32)
            weight = mx.array(np.pad(values, (0, max(0, 8 - len(values)))))
        given = None
        if plan.pools and nudge.apical_gain is not None:
            gains = np.asarray(nudge.apical_gain, dtype=np.float32).ravel()
            given = mx.array(np.pad(gains, (0, max(0, 8 - len(gains)))))
        return {
            "target": mx.array(np.asarray(target, dtype=np.float32)), "mask": mask,
            "weight": weight,
            "group_low": group_low, "group_high": group_high, "over": over,
            "beta": float(nudge.beta), "shares": nudge.softmax_temperature is not None,
            "weighted": nudge.weight is not None, "given": given,
        }

    def _launch(
        self, vm: Any, sm: Any, driven: dict[str, Any], sweeps: int, nudged: dict[str, Any] | None,
        tolerance: float | None,
    ) -> Any:
        """One launch of ``sweeps`` sweeps: ``(v, s, errors, moved, moving, gains)``; ``errors``
        holds each thread's largest residual at the launch's first and last state, ``moving`` a
        bit per sweep at which one of a thread's neurons moved at least ``tolerance`` and ``gains``
        each pool's at the last state."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        rows, m, threads = int(vm.shape[0]), plan.m, plan.threads
        bounded = tolerance is not None
        limit = 0.0 if tolerance is None else float(tolerance)
        given = None if nudged is None else nudged["given"]
        compartment = 0 if not plan.pools else (1 if given is None else 2)
        stride = rows * m
        if nudged is None:
            scales = plan.scales(0.0, limit)
            control = plan.controls(
                sweeps, False, False, bounded, False, driven["ramp"], stride, compartment,
            )
            target = mask = weight = over = plan.none
            group_low = group_high = plan.none_int
        else:
            scales = plan.scales(nudged["beta"], limit)
            control = plan.controls(
                sweeps, True, nudged["shares"], bounded, nudged["weighted"], driven["ramp"], stride,
                compartment,
            )
            target, mask, weight = nudged["target"], nudged["mask"], nudged["weight"]
            group_low, group_high, over = nudged["group_low"], nudged["group_high"], nudged["over"]
        return plan.launch(
            inputs=[vm, sm, driven["first"], driven["fixed"], self.w, plan.plan_array, scales,
                    control, target, mask, weight, group_low, group_high, over, plan.pool_of,
                    plan.pool_low, plan.pool_high, plan.none if given is None else given],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[
                (rows, m), (rows, m), (rows, threads * 2), (rows, threads), (rows, threads),
                (rows, max(plan.pools, 1)),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.uint32, mx.float32],
        )

    def _settled_state(
        self, start: _MlxState, driven: dict[str, Any], drive: np.ndarray, out: Any, taken: int,
        moved: list[Any], moves: list[Any], nudge: Nudge | None, discarded: int = 0,
    ) -> _MlxState:
        """The state a launch of ``taken`` sweeps leaves: its settling neurons, its source ranges
        along their way, and the check of its equations, theirs among them."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        gaps, goals = driven["gaps"], driven["goals"]
        sources, outer = driven["sources"], driven["outer"]
        potentials = start["potentials"]
        error = _launch_error(out[2], 1)
        if gaps is not None:
            share = plan.rho**taken
            potentials = goals if share == 0.0 else {
                r: goals[r] + np.float32(share) * gaps[r] for r in plan.sources
            }
            if taken < driven["ramp"]:  # a stop inside the ramp: the sources where it left them
                sources = {r: plan.activate(mx.array(p)) for r, p in potentials.items()}
                outer = None
            if share != 0.0:
                error = np.maximum(error, self._source_gap(driven, taken))
        return _MlxState(
            kernel="mlx", holder=self.root, plan=plan, vm=out[0], sm=out[1], sources=sources,
            potentials=potentials, goals=goals, a_host=start["a_host"], moved=moved,
            source_moves=moves, drive=np.array(drive, copy=True), version=self.version,
            fixed=driven["fixed"], outer=outer, relaxed=taken >= driven["needed"],
            checked=(drive, nudge, error), discarded_steps=discarded,
            gains=out[5] if nudge is None and plan.pools else None,
        )

    def _launched(
        self, v0: np.ndarray | None, a0: np.ndarray | None, drive: np.ndarray, steps: int,
        nudge: Nudge | None, tables: Any, tolerance: float | None, state: BrainState | None,
    ) -> tuple[Any, Any, Any, Any, int, Any, Any]:
        """``run`` in launches: the same sweeps with one synchronization per launch. Under a
        movement tolerance a launch reads which sweeps moved a neuron; a state that stops inside
        the launch is made again from the launch's start, the sweeps it stops at and no more, so
        the steps and the state are those of the tolerance read at every sweep."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        rows = len(drive)
        current = self._start(v0, a0, state, rows)
        if steps == 0:
            return None, None, None, None, 0, None, _MlxState(current, moved=[], source_moves=[])
        nudged = None if nudge is None else self._nudged_inputs(nudge, tables, rows)
        limit = _MLX_GRAPH_STEPS if tolerance is None else _MLX_BOUNDED_STEPS
        taken, discarded = 0, 0
        moved: list[Any] = []
        moves: list[Any] = []
        while taken < steps:
            sweeps = min(limit, steps - taken)
            driven = self._driven(current, drive, sweeps)
            out = self._launch(current["vm"], current["sm"], driven, sweeps, nudged, tolerance)
            mx.eval(out, driven["made"])
            stopped = False
            if tolerance is not None:
                word = int(np.bitwise_or.reduce(np.array(out[4], copy=False), axis=None))
                word |= self._sources_moved(driven, sweeps, tolerance)
                stop = next((t for t in range(sweeps) if not word >> t & 1), None)
                if stop is not None:
                    stopped = True
                    if stop + 1 < sweeps:  # it went past its stop: to the stop, from its start
                        discarded += sweeps
                        sweeps = stop + 1
                        out = self._launch(
                            current["vm"], current["sm"], driven, sweeps, nudged, None
                        )
                        mx.eval(out)
            taken += sweeps
            moved.append(out[3])
            moves.append((driven["before"], driven["path"], sweeps))
            current = self._settled_state(
                current, driven, drive, out, sweeps, moved, moves, nudge, discarded,
            )
            if stopped:
                break
        return None, None, None, None, taken, None, current

    def _sources_moved(self, driven: dict[str, Any], sweeps: int, tolerance: float) -> int:
        """A bit per sweep at which a source range moved at least ``tolerance``."""
        word = 0
        if driven["peaks"] is not None:  # a ramp: read per sweep
            peaks = np.array(driven["peaks"], copy=False)
            for t in range(min(sweeps, len(peaks))):
                if not peaks[t] < tolerance:
                    word |= 1 << t
        elif driven["before"] is not None and any(
            not np.abs(
                np.array(after, copy=False) - np.array(driven["before"][r], copy=False)
            ).max(initial=0.0) < tolerance
            for r, after in driven["sources"].items()
        ):
            word = 1  # dt = 1: they moved at the first sweep
        return word

    def solve(
        self, drive: np.ndarray, current: BrainState, nudge: Nudge | None, budget: int, chunk: int,
        tolerance: float,
    ) -> Equilibrium | None:
        """``_equilibrate_local``'s solve from the published state ``current``: a launch per chunk,
        each ending with the equations' residual on the device, in float32, within rounding of the
        float64 reference at tolerances from ``_MLX_CHECK_FLOOR``. The first launch checks the
        start too, so a solve that needs one chunk synchronizes once; a start within tolerance
        keeps it, with the sources' equations read after all. None where the launch does not
        apply."""
        mx, plan = self.mx, self.plan
        start = current.device
        if plan is None or not isinstance(start, _MlxState) or not tolerance >= _MLX_CHECK_FLOOR:
            return None
        if chunk > _MLX_GRAPH_STEPS:  # one launch holds a bounded time of the device
            return None
        tables = None if nudge is None else plan.nudge_tables(nudge)
        rows = len(drive)
        nudged = None if nudge is None else self._nudged_inputs(nudge, tables, rows)
        sweeps = min(chunk, budget)
        driven = self._driven(start, drive, sweeps)
        out = self._launch(start["vm"], start["sm"], driven, sweeps, nudged, None)
        mx.eval(out, driven["made"])
        first = _launch_error(out[2], 0)
        if sweeps == 0 or (first <= tolerance).all():
            first = np.maximum(first, self._source_gap(driven, 0))
            if sweeps == 0 or (first <= tolerance).all():
                return Equilibrium(current, first, 0, tolerance, residual_checks=1)
        used, checks = sweeps, 2
        moves = [(driven["before"], driven["path"], sweeps)]
        handle = self._settled_state(start, driven, drive, out, sweeps, [out[3]], moves, nudge)
        error = handle["checked"][2]
        while used < budget and not (error <= tolerance).all() and np.isfinite(error).all():
            sweeps = min(chunk, budget - used)
            driven = self._driven(handle, drive, sweeps)
            out = self._launch(handle["vm"], handle["sm"], driven, sweeps, nudged, None)
            mx.eval(out, driven["made"])
            used, checks = used + sweeps, checks + 1
            moves = [(driven["before"], driven["path"], sweeps)]
            handle = self._settled_state(handle, driven, drive, out, sweeps, [out[3]], moves, nudge)
            error = handle["checked"][2]
        unread: Any = None
        return Equilibrium(
            BrainState(unread, unread, unread, sweeps, device=handle), error, used, tolerance,
            residual_checks=checks,
        )

    def gains_at(self, drive: np.ndarray, state: BrainState) -> np.ndarray | None:
        """``Brain.apical_gain``: the gains the launch that made ``state`` read at it under this
        drive, or a launch of no sweeps that reads them. None without a compartment."""
        mx, plan = self.mx, self.plan
        assert plan is not None
        if not plan.pools:
            return None
        handle = state.device
        rows = len(drive)
        if (
            isinstance(handle, _MlxState) and handle.get("holder") is self.root
            and handle.get("gains") is not None and handle.get("version") == self.version
            and np.array_equal(handle.get("drive"), drive)
        ):
            return np.array(handle["gains"], dtype=np.float64)
        on_device = handle is not None and handle.get("holder") is self.root
        v0 = None if on_device else np.atleast_2d(state.v)
        a0 = None if on_device else np.atleast_2d(state.adaptation)
        start = self._start(v0, a0, state, rows)
        if int(start["vm"].shape[0]) != rows:
            raise ValueError("state batch does not match the drive batch")
        driven = self._driven(start, drive, 0)
        out = self._launch(start["vm"], start["sm"], driven, 0, None, None)
        mx.eval(out)
        return np.array(out[5], dtype=np.float64)

    def _source_gap(self, driven: dict[str, Any], taken: int) -> np.ndarray:
        """Each row's largest residual among the source neurons after ``taken`` sweeps from the
        launch's start: ``drive + bias - v``, on the host, with no launch of its own."""
        plan = self.plan
        assert plan is not None
        gaps = driven["gaps"]
        if not gaps:
            return np.zeros(int(driven["fixed"].shape[0]))
        worst = np.zeros(int(next(iter(gaps.values())).shape[0]))
        for gap in gaps.values():
            worst = np.maximum(worst, np.abs(gap).max(axis=1, initial=0.0))
        return np.asarray(plan.rho**taken * worst)
