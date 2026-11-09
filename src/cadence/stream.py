"""Carried state as a stimulus: a brain that carries its last equilibrium into the next one.

If settling has a unique attracting fixed point, fully settling the next input
erases the starting state. Multiple attractors and interrupted settling can retain
history, but are not a reliable addressed store. Here a range of *context* neurons,
one per hidden neuron, is stimulated to a leaky trace of the hidden neurons' own activation at
the previous inputs:

    c <- decay * c + (1 - decay) * h

so the state reverberates and fades over about ``1 / (1 - decay)`` inputs. The context
neurons hear nothing (they are a source range of the block transport, so their product is
computed once per settling run) and their synapses into the hidden neurons learn under the
same free/nudged rule as every other synapse. Credit does not flow back through time: the
rule sees the trace as a stimulus and does not differentiate through it.
``Trace`` keeps the trace for a batch of streams.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field, fields
from math import gcd
from threading import Lock
from types import MappingProxyType
from typing import Any, ClassVar, cast

import numpy as np

from ._configuration import real
from .brain import _MLX_MAP_ENTRIES, BrainState, _MlxMatrix, _resident
from .brain import Brain as NeuralGraph
from .connectome import Connectome
from .learning import LearnedState
from .resident import (
    _host_array,
    _Moment,
    _on_host,
    _RecallOnDevice,
    _set_columns,
    _set_silent,
    _StoreOnDevice,
    _stream_rows,
    _TraceOnDevice,
    arrays_of,
    route_contrast,
)

__all__ = ["Hearing", "Trace", "FastSynapses", "History", "Recall", "columns"]


# Large products and copies may share at most 32 workers in one pool per process.
_THREADED = 1 << 18
_RECALL_WORKERS = 32
_POOL: tuple[int, ThreadPoolExecutor] | None = None
_POOL_LOCK = Lock()
# The tokens of the recall operations in progress (``Recall._scope``).
_OPERATIONS: set[object] = set()


def _after_recall_fork() -> None:
    # A child cannot submit work to threads that existed only in its parent.
    global _POOL, _POOL_LOCK
    _POOL = None
    _POOL_LOCK = Lock()
    # nor is an operation another thread had in progress still running in the child
    _OPERATIONS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_recall_fork)


def _recall_threads() -> int:
    """Requested recall workers; 1 (the default) dispatches no threaded work."""
    try:
        return max(1, int(os.environ.get("CADENCE_RECALL_THREADS", "1")))
    except ValueError:
        return 1


def _recall_pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or _POOL[0] != os.getpid():
            _POOL = (os.getpid(), ThreadPoolExecutor(
                _RECALL_WORKERS, thread_name_prefix="cadence-recall"
            ))
        return _POOL[1]


def _by_rows(task: Callable[[int, int], Any], rows: int, size: int) -> list[Any] | None:
    """Run ordered blocks of streams, or return None for the ordinary serial operation.

    Changing the requested count reuses one bounded process pool. A forked child creates
    its own pool; it never waits for its parent's vanished workers or inherited lock.
    """
    if size < _THREADED:
        return None
    threads = min(_recall_threads(), rows, _RECALL_WORKERS)
    if threads <= 1:
        return None
    bounds = np.linspace(0, rows, threads + 1).astype(int)
    return list(_recall_pool().map(
        lambda i: task(int(bounds[i]), int(bounds[i + 1])), range(threads)
    ))


def _einsum(spec: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The original contraction, optionally split over independent canonical stream rows.

    Noncontiguous expert operands keep their original NumPy traversal and summation order.
    """
    if not (a.flags.c_contiguous and b.flags.c_contiguous):
        return np.asarray(np.einsum(spec, a, b))
    parts = _by_rows(
        lambda start, stop: np.einsum(spec, a[start:stop], b[start:stop]),
        len(a), max(a.size, b.size),
    )
    return np.einsum(spec, a, b) if parts is None else np.concatenate(parts)


def _compiled_terms() -> bool:
    """Whether the compiled passes of ``cadence.fused`` are in use (the same bits either way)."""
    try:
        from .fused import compiled
    except ImportError:  # pragma: no cover
        return False
    return compiled()


def _route_matrix(
    graph: Any, synapses: np.ndarray, places: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """The effective weights of a route's ``synapses`` as a matrix of ``shape``, each at its entry
    of ``places`` (raveled), one synapse per entry; the one pass of ``cadence.fused`` where it is
    in use (the same bits either way)."""
    weights = graph._weights
    if _compiled_terms():
        from .fused import gather, scatter

        return scatter(shape[0] * shape[1], places, gather(weights, synapses)).reshape(shape)
    out = np.zeros(shape[0] * shape[1])
    out[places] = weights[synapses]
    return out.reshape(shape)


def columns(index: np.ndarray) -> slice | np.ndarray:
    """A slice when the neurons are one contiguous range, else the index array.

    Reading or writing the columns of a wide batch through a slice is a strided pass;
    through an index array it is a gather that comes back Fortran-ordered (and a scatter
    on write), which costs tens of times more on a batch of thousands of rows.
    """
    if len(index) and np.array_equal(index, np.arange(index[0], index[0] + len(index))):
        return slice(int(index[0]), int(index[0]) + len(index))
    return np.asarray(index, dtype=np.int64)


def _left(p: np.ndarray) -> np.ndarray:
    """The share of each slot a head's attention ``p`` leaves to the heads after it."""
    return np.clip(1.0 - p, 0.0, 1.0)


def _share(logits: np.ndarray, left: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """A later head's attention among competing heads: a slot can be read in full once per
    moment, so the head takes of each slot at most the share ``left`` the heads before it left.
    Its own attention (the softmax of ``logits`` over the slots something is left of) keeps its
    shape where the share allows, ``w = min(left, lam p)``, and ``lam`` gives the head its
    ``mass`` over the slots, or every share left where that is less: what a taken slot cannot
    give goes to the slots the head prefers next. A row whose slots the heads before took whole
    reads nothing.

    The slots taken whole leave before the shares are taken, and the arithmetic is in logits:
    the slots are capped in the order of ``log(left) - logit``, and a slot past the capped ones
    takes its fraction of the attention left, ``exp(logit - logsumexp(logits from it on))``,
    which lies in [0, 1] however far apart the logits are."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        z = np.where(left > 0, logits, -np.inf)
        open_ = np.isfinite(z)
        mass = np.minimum(mass, np.where(open_, left, 0.0).sum(axis=1))
        ratio = np.where(open_, np.log(np.where(open_, left, 1.0)) - np.where(open_, z, 0.0),
                         np.inf)
        order = np.argsort(ratio, axis=1, kind="stable")
        lefts = np.take_along_axis(left, order, axis=1)
        zs = np.take_along_axis(z, order, axis=1)
        # with the k slots of least ratio capped at their shares, the rest share what mass
        # remains in proportion to their attention; the first k whose next slot that share does
        # not cap is the one
        capped = np.cumsum(lefts, axis=1) - lefts
        rest = np.logaddexp.accumulate(zs[:, ::-1], axis=1)[:, ::-1]  # log attention from k on
        live = np.isfinite(rest)
        unit = np.where(live, np.exp(zs - np.where(live, rest, 0.0)), 0.0)
        fits = live & ((mass[:, None] - capped) * unit <= lefts)
        found = fits.any(axis=1)
        rows = np.arange(len(z))
        first = np.where(found, np.argmax(fits, axis=1), z.shape[1])
        k = np.minimum(first, z.shape[1] - 1)
        spread = np.where(found, mass - capped[rows, k], 0.0)
        log_rest = np.where(found, rest[rows, k], 0.0)
        tail = np.arange(z.shape[1])[None, :] >= first[:, None]
        shares = np.where(tail, spread[:, None] * np.exp(zs - log_rest[:, None]), lefts)
        shares = np.where(np.isfinite(zs), np.clip(shares, 0.0, lefts), 0.0)
        out = np.empty_like(shares)
        np.put_along_axis(out, order, shares, axis=1)
        return out


@dataclass
class Trace:
    """A quantity decaying across moments, kept per stream, entering the next settling run
    as a stimulus: the memory of the moment before.

    Over a range's activation it is the trace of that range, the echo of the interpretation;
    of the input neurons it is an afterimage of the picture itself. ``last`` holds the
    source read of the latest moment and ``cold`` the streams that have not been read yet.
    """

    connectome: Connectome
    decay: float = 0.5
    amplitude: float = 1.0  # the stimulus amplitude of the neuron model
    source: str = "hidden"  # the range traced
    # the range the trace enters as a stimulus, one neuron per source neuron
    target: str = "context"
    # the learned multiplier on the amplitude (``Hearing``), 1 the working value
    hearing: float = 1.0
    trace: np.ndarray = field(init=False)
    last: np.ndarray = field(init=False)
    cold: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._device: _TraceOnDevice | None = None  # what rests on a device (``reside``)
        if not 0 <= self.decay < 1:
            raise ValueError("decay lies in [0, 1)")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.hidden = np.asarray(self.connectome.populations[self.source], dtype=np.int64)
        if len(self.glow) != len(self.hidden):
            raise ValueError(f"one {self.target} neuron per {self.source} neuron")
        self._glow_columns = columns(self.glow)
        self._hidden_columns = columns(self.hidden)
        self.reset(0)

    def __getattr__(self, name: str) -> Any:
        """``trace`` or ``last`` resting on the device, read on the host: both come back."""
        if self.__dict__.get("_device") is not None and name in ("trace", "last"):
            self.leave_device()
            return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getstate__(self) -> dict[str, Any]:
        self.leave_device()
        return self.__dict__

    def reside(self, kernel: Any) -> None:
        """``trace`` and ``last`` on the device of a brain's ``kernel``, read and written there
        each moment."""
        if type(self) is not Trace or kernel is None:
            self.leave_device()
            return
        if self._device is not None:
            if "trace" not in self.__dict__ and "last" not in self.__dict__:
                return
            self.leave_device()
        self._device = _TraceOnDevice(self, kernel)

    def leave_device(self) -> None:
        """Back on the host; what was written there since (``_restore_traces``) stays."""
        held = self.__dict__.get("_device")
        if held is not None:
            for name in ("trace", "last"):
                if name not in self.__dict__:
                    self.__dict__[name] = _host_array(getattr(held, name))
            self._device = None

    def _resident_path(self) -> _TraceOnDevice | None:
        """What rests on the device, None once an array was written on the host: that brings the
        other back first, and the host's path reads them."""
        if self._device is not None and ("trace" in self.__dict__ or "last" in self.__dict__):
            self.leave_device()
        return self._device

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Fresh for every stream, or for ``rows`` only (streams that begin again)."""
        held = self._resident_path()
        if held is not None:
            held.reset(batch, rows)
            return
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.hidden)))
            self.last = np.zeros((batch, len(self.hidden)))
            self.cold = np.ones(batch, dtype=bool)
        else:
            self.trace[rows] = 0.0
            self.last[rows] = 0.0
            self.cold[rows] = True

    def keep(self, rows: np.ndarray) -> None:
        """Keep the traces of ``rows`` only (streams that ended are dropped)."""
        self.trace, self.last, self.cold = self.trace[rows], self.last[rows], self.cold[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the trace into a drive copy; another batch size reads zero context."""
        if self._device is not None and _resident(drive):
            return cast(np.ndarray, self._device.stimulate(drive))
        if _resident(drive):  # a resident drive reads what the host holds
            if len(self.trace) == len(drive):
                _set_columns(
                    drive, self._glow_columns, self.amplitude * self.hearing * self.trace
                )
            else:
                _set_silent(drive, self._glow_columns)
            return drive
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = (
            self.amplitude * self.hearing * self.trace if len(self.trace) == len(out) else 0.0
        )
        return out

    def update(self, state: BrainState) -> None:
        """After the free phase: the trace decays toward the source neurons' activation."""
        held = self._resident_path()
        if held is not None:
            held.update(state)
            return
        h = self._source_activation(state)
        if len(self.trace) != len(h):
            self.reset(len(h))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * h
        self.last = h.copy()  # own the previous moment even if the caller reuses state storage
        self.cold[:] = False

    def _source_activation(self, state: BrainState) -> np.ndarray:
        """The source range's activation as a host array. A state that rests on a torch device
        is sliced there and only the slice comes to the host: for an afterimage of a retina
        of thousands of neurons that is the trace's whole cost."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            cols = self._hidden_columns
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                import torch

                part = s[:, torch.as_tensor(np.asarray(cols), device=s.device)]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        return np.ascontiguousarray(np.atleast_2d(state.activation)[:, self._hidden_columns])

    def ringing(self, floor: float = 0.1) -> np.ndarray:
        """Each source neuron's trace magnitude over the row's mean magnitude, plus ``floor``;
        one for every other neuron: a salience for the eligibility of the synapses
        out of them, so that what is still ringing is what a signal writes through
        (``ActorCritic.salience``)."""
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and nonnegative")
        out = np.ones((len(self.trace), self.connectome.n))
        magnitude = np.abs(self.trace)
        mean = magnitude.mean(axis=1, keepdims=True) + 1e-9
        out[:, self._hidden_columns] = floor + magnitude / mean
        return out

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "decay": self.decay,
            "amplitude": self.amplitude,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The trace read into a drive copy (``stimulate``)."""
        return self.stimulate(drive)

    def branch(self) -> Trace:
        """A private copy for an imagined branch, its arrays its own."""
        branch = copy(self)
        for name in ("trace", "last", "cold"):
            setattr(branch, name, getattr(self, name).copy())
        return branch

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps."""
        return {name: getattr(self, name) for name in ("trace", "last", "cold")}


@dataclass
class Efference(Trace):
    """The corollary discharge: a Trace written from the command the brain issued.

    One ``efference`` neuron per motor neuron carries the fading one-hot of the actions the
    stream executed, and the next moment reads it through the plastic projection from the
    ``efference`` range into the association region. The working trace copies the settled
    state before the decision, so under identical observations it carries the executed
    action only through whatever margin the motor competition left, and nothing of a
    sampled choice; this trace carries the command itself. The write is fixed, as the
    working trace's is; what the brain does with its own last command is learned.
    """

    source: str = "motor"
    target: str = "efference"

    def update(self, state: BrainState) -> None:
        raise TypeError("the efference copy is written by the issued command; use issue")

    def issue(self, command: np.ndarray) -> None:
        """After the decision: the trace decays toward the ``(batch, motor)`` command, one
        row per stream with ones at the motor neurons of the executed action."""
        command = np.asarray(command, dtype=float)
        if command.ndim != 2 or command.shape[1] != len(self.hidden):
            raise ValueError("a command is one row of motor values per stream")
        if not np.isfinite(command).all():
            raise ValueError("a command must be finite")
        if len(self.trace) != len(command):
            self.reset(len(command))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * command
        self.last = command.copy()
        self.cold[:] = False


@dataclass
class History:
    """The observations as heard, carried at several timescales in a region of their own.

    One block of ``target`` neurons per decay, each block one neuron per ``source`` neuron.
    After each act the block of decay ``d`` moves toward the observation the brain acted on,
    ``h <- d * h + (1 - d) * x``: the values the ``source`` neurons were stimulated with,
    before any settling. The next moment reads ``amplitude * h`` as the stimulus of the
    block's own neurons, so at a decision the history holds the past and the sensory drive
    holds the present. A fast block keeps the last moments apart, a slow one keeps an
    average of many. The write is fixed, as the working trace's is; the association region
    reads the blocks through a plastic projection, so what the past means is learned. This
    is the pattern of the efference copy applied to the senses: in a closed loop the brain
    hears its own output as its next observation.
    """

    connectome: Connectome
    decays: tuple[float, ...]
    amplitude: float = 3.0
    source: str = "sensory"  # the range heard
    target: str = "history"  # one block of source width per decay
    # the learned multiplier per block (``Hearing``); None holds the working amplitude
    hearing: np.ndarray | None = None
    trace: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        decays = tuple(self.decays)
        if not decays or any(
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, float, np.integer, np.floating))
            or not 0 <= d < 1
            for d in decays
        ):
            raise ValueError("decays are one or more real values in [0, 1)")
        self.decays = tuple(float(d) for d in decays)
        if (
            isinstance(self.amplitude, (bool, np.bool_))
            or not isinstance(self.amplitude, (int, float, np.integer, np.floating))
            or not np.isfinite(self.amplitude)
            or self.amplitude < 0
        ):
            raise ValueError("amplitude must be a finite nonnegative real scalar")
        self.amplitude = float(self.amplitude)
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.width = len(self.connectome.populations[self.source])
        if len(self.glow) != len(self.decays) * self.width:
            raise ValueError(f"one block of {self.source} width per decay in {self.target}")
        self._glow_columns = columns(self.glow)
        # the decay of every history neuron: block by block
        self._decay = np.repeat(np.asarray(self.decays), self.width)
        self.reset(0)

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Silence for every stream, or for ``rows`` only (streams that begin again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.glow)))
        else:
            self.trace[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        """Keep the histories of ``rows`` only (streams that ended are dropped)."""
        self.trace = self.trace[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the history into a drive copy; another batch size reads silence."""
        amplitude: Any = self.amplitude
        if self.hearing is not None:
            amplitude = amplitude * np.repeat(self.hearing, self.width)
        if _resident(drive):  # a resident drive reads what the host holds
            if len(self.trace) == len(drive):
                _set_columns(drive, self._glow_columns, amplitude * self.trace)
            else:
                _set_silent(drive, self._glow_columns)
            return drive
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = amplitude * self.trace if len(self.trace) == len(out) else 0.0
        return out

    def hear(self, observations: np.ndarray) -> None:
        """After the act: every block decays toward the ``(batch, source)`` observations the
        streams acted on; another batch size begins the streams in silence first."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.width:
            raise ValueError(f"observations are one row of {self.width} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        if len(self.trace) != len(x):
            self.reset(len(x))
        heard = np.tile(x, (1, len(self.decays)))
        self.trace = self._decay * self.trace + (1.0 - self._decay) * heard

    def to_dict(self) -> dict[str, Any]:
        return {
            "decays": list(self.decays),
            "amplitude": self.amplitude,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The history read into a drive copy (``stimulate``)."""
        return self.stimulate(drive)

    def branch(self) -> History:
        """A private copy for an imagined branch, its trace its own."""
        branch = copy(self)
        branch.trace = self.trace.copy()
        return branch

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps."""
        return {"trace": self.trace}


@dataclass
class Message:
    """What a stream heard in its episode, held as one state: phased unions of a sparse code.

    Every heard moment is given a fixed sparse code: ``units`` code neurons, each reading
    ``fan_in`` of the observation's ON and OFF channels with positive weights
    (``PatternSeparator`` with ``cover``), the ``winners`` largest sums set to one. A unit wins
    where it reads several of the moment's active inputs at once, so the code binds what a
    moment holds together (a noun with its number and its role). The ``t``-th moment of a
    stream's episode joins the union of phase ``t mod phases``. The state is ``phases`` blocks
    of ``units`` each: what a block holds is present or absent, neighbouring moments lie in
    neighbouring blocks, and nothing fades while the episode lasts.

    ``state`` is the held message, ``(streams, phases * units)``, zeros and ones. ``key()`` is
    the union over the phases, ``(streams, units)``: the same for one set of moments heard in
    any order, and close for episodes that share moments, which is what a memory of encoded
    meanings addresses a record by. The write is fixed, as the working trace's and the heard
    history's are, and holds no learned parameter; what a reader makes of the message is
    learned.

    The class stands alone: it holds no neurons and reads no brain. ``hear`` takes the
    observations one moment at a time, one row per stream; ``reset`` begins streams afresh.
    """

    inputs: int
    phases: int = 4
    units: int = 70
    winners: int = 5
    fan_in: int = 16
    seed: int = 0
    trace: np.ndarray = field(init=False, repr=False)
    told: np.ndarray = field(init=False, repr=False)
    separator: PatternSeparator = field(init=False, repr=False)

    COVER = 2  # code neurons that read each channel at least
    STREAM = 286  # the code's own stream of the seed

    def __post_init__(self) -> None:
        for name in ("inputs", "phases", "units", "winners", "fan_in"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.winners > self.units:
            raise ValueError("winners must not exceed the units")
        channels = 2 * int(self.inputs)
        cover = min(self.COVER, int(self.units))
        # every channel read by ``cover`` units: a fan-in too small for that is raised to it
        fan_in = min(channels, max(int(self.fan_in), -(-cover * channels // int(self.units))))
        self.separator = PatternSeparator(
            int(self.inputs), int(self.units), int(self.winners),
            seed=int(np.random.SeedSequence([int(self.seed), self.STREAM]).generate_state(1)[0]),
            fan_in=fan_in, cover=cover,
        )
        self.reset(0)

    @property
    def width(self) -> int:
        """The neurons the state takes: ``phases * units``."""
        return int(self.phases) * int(self.units)

    @property
    def state(self) -> np.ndarray:
        """The held message, ``(streams, phases * units)``."""
        return self.trace

    def code(self, observations: np.ndarray) -> np.ndarray:
        """The sparse code of ``(streams, inputs)`` observations: one at each row's winners,
        ``(streams, units)``. A silent observation codes to zero."""
        return np.asarray((self.separator.code(observations) > 0.0).astype(float))

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """An empty message for every stream, or for ``rows`` only (streams whose episode
        begins again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, self.width))
            self.told = np.zeros(batch, dtype=np.int64)
        else:
            self.trace[rows] = 0.0
            self.told[rows] = 0

    def keep(self, rows: np.ndarray) -> None:
        """Keep the messages of ``rows`` only (streams that ended are dropped)."""
        self.trace, self.told = self.trace[rows], self.told[rows]

    def hear(self, observations: np.ndarray) -> None:
        """One heard moment per stream joins its phase's union; another batch size begins the
        streams afresh first."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs:
            raise ValueError(f"observations are one row of {self.inputs} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        if len(self.trace) != len(x):
            self.reset(len(x))
        code = self.code(x)
        held = self.trace.reshape(len(x), int(self.phases), int(self.units))
        rows = np.arange(len(x))
        phase = self.told % int(self.phases)
        held[rows, phase] = np.maximum(held[rows, phase], code)
        self.told = self.told + 1

    def key(self) -> np.ndarray:
        """The union over the phases, ``(streams, units)``: what the episode held, whatever the
        order it was heard in."""
        held = self.trace.reshape(len(self.trace), int(self.phases), int(self.units))
        return np.asarray(held.max(axis=1))

    def branch(self) -> Message:
        """A private copy, its arrays its own."""
        branch = copy(self)
        branch.trace, branch.told = self.trace.copy(), self.told.copy()
        return branch

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps: the code is drawn again from the seed."""
        return {"trace": self.trace, "told": self.told}

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": int(self.inputs),
            "phases": int(self.phases),
            "units": int(self.units),
            "winners": int(self.winners),
            "fan_in": int(self.separator.fan_in),
            "seed": int(self.seed),
        }


@dataclass
class HeldMessage(Message):
    """A ``Message`` on a region of a brain's own neurons (``Brain.compose(message=units)``).

    The region ``target`` holds one neuron per unit of the message, block after block. After
    each moment the brain hears, the moment's winners in the block whose turn it is latch on
    and stay on until the stream's episode ends: the write is fixed, as the working trace's and
    the heard history's are. The association reads the region through plastic synapses at
    ``amplitude`` times what it holds, and with ``carried_gain`` one learned gain per block
    (``hearing``) sets how loudly.

    With a ``port`` region the brain also reads its message one block at a time
    (``message_read``). The present moment addresses the blocks,

        p = softmax((x W_x + c W_c + w) . block / temperature + x T_x + c T_c + t)

    ``x`` the moment as heard and ``c`` the association's settled activity of the moment
    before, and the port holds ``read_amplitude * sum_k p_k block_k``: the settle answers on
    the block the address chose. Every map starts at zero, so a newborn reads the blocks'
    mean. After a lesson the blocks are scored against the association's movement between
    the nudged phases, read back through the port's synapses, and the maps step toward the
    blocks the lesson names, the recall memory's address lesson on the message's own blocks.
    """

    connectome: Connectome | None = None
    amplitude: float = 1.5
    read_amplitude: float = 3.0
    read_rate: float = 0.01
    read_temperature: float = 0.25
    target: str = "message"
    port: str = "message_read"
    hearing: np.ndarray = field(init=False, repr=False)
    previous: np.ndarray = field(init=False, repr=False)

    SCORE_TEMPERATURE = 0.1  # of the blocks' scores in the read's lesson, the recall memory's

    def __post_init__(self) -> None:
        connectome = self.connectome
        if connectome is None:
            raise ValueError("a held message needs the brain's connectome")
        for name in ("amplitude", "read_amplitude", "read_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"message {name} must be a finite nonnegative real scalar")
            setattr(self, name, float(value))
        if not np.isfinite(self.read_temperature) or self.read_temperature <= 0:
            raise ValueError("message read_temperature must be positive")
        populations = connectome.populations
        self.glow = np.asarray(populations[self.target], dtype=np.int64)
        self.association = np.asarray(populations["association"], dtype=np.int64)
        self.reads = self.port in populations
        super().__post_init__()
        if len(self.glow) != self.width:
            raise ValueError(
                f"the {self.target} population holds {len(self.glow)} neurons, where the "
                f"message holds {self.width} (phases * units)"
            )
        self._glow_columns = columns(self.glow)
        self.hearing = np.ones(int(self.phases))
        self.last: tuple[np.ndarray, ...] | None = None  # the latest read, for its lesson
        self.maps: list[_Maps] = []
        if self.reads:
            port = np.asarray(populations[self.port], dtype=np.int64)
            if len(port) != self.units:
                raise ValueError(
                    f"the {self.port} population holds {len(port)} neurons, where a block of "
                    f"the message holds {self.units}"
                )
            self._port_columns = columns(port)
            pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
            source = -np.ones(connectome.n, dtype=np.int64)
            source[port] = np.arange(len(port))
            cortex = -np.ones(connectome.n, dtype=np.int64)
            cortex[self.association] = np.arange(len(self.association))
            synapses = np.flatnonzero((source[pre] >= 0) & (cortex[post] >= 0))
            self._port_synapses = (synapses, source[pre[synapses]], cortex[post[synapses]])
            inputs, wide = int(self.inputs), len(self.association)
            # the content address over a block's units, then the turn over the blocks
            self.maps = [
                (np.zeros((inputs, outputs)), np.zeros((wide, outputs)), np.zeros(outputs))
                for outputs in (int(self.units), int(self.phases))
            ]

    # -- the streams

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """An empty message for every stream, or for ``rows`` only."""
        wide = len(self.__dict__.get("association", ()))
        whole = rows is None or len(self.trace) != batch
        super().reset(batch, rows)
        if whole:
            self.previous = np.zeros((batch, wide))
        else:
            self.previous[rows] = 0.0
        self.last = None

    def keep(self, rows: np.ndarray) -> None:
        super().keep(rows)
        self.previous = self.previous[rows]
        self.last = None

    def hear(self, observations: np.ndarray, cortex: np.ndarray | None = None) -> None:
        """One heard moment per stream joins its block; ``cortex`` is the association's settled
        activity at that moment, which the next moment's read addresses with."""
        super().hear(observations)
        if cortex is not None and np.shape(cortex) == self.previous.shape:
            self.previous = np.array(cortex, dtype=float)
        else:
            self.previous = np.zeros((len(self.trace), len(self.association)))

    def streams(self) -> dict[str, np.ndarray]:
        """What the streams hold, the learned maps apart."""
        return {"trace": self.trace, "told": self.told, "previous": self.previous}

    # -- the read

    def _heard_of(self, moment: Any) -> np.ndarray | None:
        """The moment as the message hears it: the observations, or their code where that is
        what the cortex reads."""
        for x in (getattr(moment, "observations", None), getattr(moment, "code", None)):
            if x is not None and np.ndim(x) == 2 and np.shape(x)[1] == self.inputs:
                return np.asarray(x, dtype=float)
        return None

    def attention(self, x: np.ndarray, c: np.ndarray, blocks: np.ndarray) -> np.ndarray:
        """``p`` over each stream's blocks for the moments ``x`` and the cortex ``c``."""
        content, turn = self.maps
        logits = _einsum("bu,bku->bk", _term(content, x, c), blocks) / self.read_temperature
        logits = logits + _term(turn, x, c)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return np.asarray(p / p.sum(axis=1, keepdims=True))

    def stimulate(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The message read into a drive copy, and with a port the block the moment addresses;
        another batch size reads silence."""
        out = np.array(_on_host(drive), dtype=float)
        present = len(self.trace) == len(out)
        loud = self.amplitude * np.repeat(self.hearing, int(self.units))
        out[:, self._glow_columns] = loud * self.trace if present else 0.0
        if self.reads:
            self.last = None
            x = self._heard_of(moment) if present else None
            if x is None or len(x) != len(out):
                out[:, self._port_columns] = 0.0
            else:
                blocks = self.trace.reshape(len(out), int(self.phases), int(self.units))
                c = self.previous
                p = self.attention(x, c, blocks)
                out[:, self._port_columns] = self.read_amplitude * _einsum("bk,bku->bu", p, blocks)
                self.last = (x, c, p, blocks.copy())
        return out

    def stimulus(self, drive: np.ndarray, moment: Any = None) -> np.ndarray:
        """The memory contract's read (``stimulate``)."""
        return self.stimulate(drive, moment)

    def learn(self, graph: Any, learned: Any) -> dict[str, float]:
        """The read's lesson after a lesson whose phases are ``learned``, at its taught rows:
        each block is scored against the association's movement read back through the port's
        synapses, and the maps step toward the blocks the scores name. Returns how many rows
        moved (``message_rows``) and how many a refused step left as they were."""
        quiet = {"message_rows": 0.0, "message_refused": 0.0}
        read = self.last
        if (
            not self.reads
            or not self.read_rate
            or read is None
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return quiet
        x, c, p, blocks = read
        taught = learned.taught
        rows = (
            np.arange(len(x)) if taught is None
            else np.flatnonzero(np.asarray(taught, dtype=bool))
        )
        plus = np.atleast_2d(learned.nudged.activation)
        if not len(rows) or len(plus) != len(x):
            return quiet
        minus = np.atleast_2d(
            learned.free.activation if learned.opposite is None else learned.opposite.activation
        )
        movement = (plus - minus)[:, self.association][rows]
        synapses, sources, targets = self._port_synapses
        kernel = getattr(graph, "_mlx", None)
        back = None
        if kernel is not None and graph._efficacy is None:  # the synapses rest on the device
            found = kernel.read_back([(synapses, sources, targets, int(self.units))], movement)
            back = None if found is None else np.asarray(found[0])
        if back is None:
            w = np.zeros((int(self.units), len(self.association)))
            np.add.at(w, (sources, targets), np.asarray(graph._weights)[synapses])
            back = movement @ w.T
        x, c, p, blocks = x[rows], c[rows], p[rows], blocks[rows]
        norm = np.sqrt(np.sum(back * back, axis=1))
        moving = norm > 1e-12
        if not moving.any():
            return quiet
        x, c, p, blocks, back, norm = (a[moving] for a in (x, c, p, blocks, back, norm))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            score = _einsum("bu,bku->bk", back, blocks) / norm[:, None]
            logits = score / self.SCORE_TEMPERATURE + np.log(np.maximum(p, 1e-300))
            logits -= logits.max(axis=1, keepdims=True)
            named = np.exp(logits)
            named /= named.sum(axis=1, keepdims=True)
            turn = named - p
            content = _einsum("bk,bku->bu", turn, blocks)
            # each row's step bounded by its own energy, as the recall memory's is
            energy = np.sum(x * x, axis=1) + np.sum(c * c, axis=1) + 1.0
            bound = np.minimum(1.0, 1.0 / (self.read_rate * energy))[:, None]
            stepped = [
                _lms(maps, step * bound, self.read_rate, x, c)
                for maps, step in zip(self.maps, (content, turn), strict=True)
            ]
        if not all(_finite(a) for maps in stepped for a in maps):
            return {"message_rows": 0.0, "message_refused": float(len(x))}
        self.maps = stepped
        return {"message_rows": float(len(x)), "message_refused": 0.0}

    # -- the memory contract

    def branch(self) -> HeldMessage:
        """A private copy for an imagined branch: its streams its own, its maps shared."""
        branch = copy(self)
        branch.trace, branch.told = self.trace.copy(), self.told.copy()
        branch.previous = self.previous.copy()
        branch.last = None
        return branch

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps: the streams and the read's maps."""
        out = dict(self.streams())
        for kind, maps in zip(("content", "turn"), self.maps, strict=False):
            for name, value in zip(("x", "c", "b"), maps, strict=True):
                out[f"{kind}_{name}"] = np.asarray(value)
        return out

    def to_dict(self) -> dict[str, Any]:
        out = {
            **super().to_dict(),
            "amplitude": self.amplitude,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }
        if self.reads:
            out.update(read=True, read_amplitude=self.read_amplitude, read_rate=self.read_rate,
                       read_temperature=self.read_temperature)
        return out


# A head's map of the moment ``(M_x, M_c, m)``, read as ``x M_x + c M_c + m`` from the moment's
# inputs ``x`` and the cortex ``c`` of the moment before: the address map, the position table and
# the maps beside it.
_Maps = tuple[np.ndarray, np.ndarray, np.ndarray]


def _addressed(key: np.ndarray, maps: _Maps, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The address ``key + x M_x + c M_c + m``, on the device where the map rests there."""
    m_x, m_c, m = maps
    if isinstance(m_x, _MlxMatrix):
        return np.asarray(key + m_x.term(x, cast(_MlxMatrix, m_c), c) + m)
    return np.asarray(key + x @ m_x + c @ m_c + m)


def _finite(a: Any) -> bool:
    """Whether every entry of a map or an array is finite, without fetching a map on the device."""
    return a.finite() if isinstance(a, _MlxMatrix) else bool(np.isfinite(a).all())


def _term(maps: _Maps, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """A map of the moment's term at the inputs ``x`` and the cortex ``c``."""
    m_x, m_c, m = maps
    return np.asarray(x @ m_x + c @ m_c + m)


def _lms(maps: _Maps, step: np.ndarray, rate: float, x: np.ndarray, c: np.ndarray) -> _Maps:
    """A map of the moment after the least-mean-squares ``step`` at ``rate``: each row's step on
    its inputs ``z = [x, c, 1]``, averaged over the rows. Unchecked: a caller checks what it
    keeps."""
    m_x, m_c, m = maps
    if isinstance(m_x, _MlxMatrix):
        on_x, on_c = m_x.stepped(cast(_MlxMatrix, m_c), x, c, step, rate)
        return cast(np.ndarray, on_x), cast(np.ndarray, on_c), m + rate * step.mean(axis=0)
    s_x = rate * x.T @ step
    s_x /= len(x)
    s_x += m_x
    s_c = rate * c.T @ step
    s_c /= len(x)
    s_c += m_c
    return s_x, s_c, m + rate * step.mean(axis=0)


# A BLAS dot sums in lanes, so what a dot of two lists of numbers rounds to depends on where the
# zeros lie: a term keeps its lane where the zeros before it keep their count modulo this, and
# the kernels make the same choices above this length.
# A reward eligibility laid out in full past this many entries (1 GiB) goes to a checkpoint as the
# actor keeps it: at the speech size and 256 streams the full layout is 34 GiB of zeros and 1 GiB.
_SAVED_IN_FULL = 1 << 27
_DOT_LANES = 64
_DOT_LONG = 1 << 17


def _laid_out_norm(pieces: Sequence[np.ndarray | int]) -> float:
    """``np.linalg.norm`` of the vector the ``pieces`` join, an array or the count of a run of
    zeros, to its bits, without the zeros of the long runs."""
    total = sum(p if isinstance(p, int) else p.size for p in pieces)
    runs = [p for p in pieces if isinstance(p, int)]
    if total <= _DOT_LONG or not runs:
        short = [np.zeros(p) if isinstance(p, int) else p.ravel() for p in pieces]
    else:
        spare = sum(max(run - run % _DOT_LANES - _DOT_LANES, 0) for run in runs)
        floor = max(_DOT_LONG - (total - spare), 0)
        floor += -floor % _DOT_LANES  # a whole number of lanes
        short = []
        for piece in pieces:
            if isinstance(piece, int):
                kept = piece if piece < 2 * _DOT_LANES else piece % _DOT_LANES + _DOT_LANES
                short.append(np.zeros(kept + (min(floor, piece - kept))))
                floor -= min(floor, piece - kept)
            else:
                short.append(piece.ravel())
    return float(np.linalg.norm(np.concatenate(short)))


def _per_bucket(gradient: np.ndarray, buckets: np.ndarray, size: int) -> np.ndarray:
    """Each row's ``gradient`` over the slots summed into ``size`` buckets, slot ``i`` into its
    bucket ``buckets[..., i]`` (one row of buckets for every row, or a row each)."""
    out = np.zeros((len(gradient), size))
    np.add.at(out, (np.arange(len(gradient))[:, None], buckets), gradient)
    return out


def _of(state: np.ndarray, streams: int) -> np.ndarray:
    """A per-stream ``state`` where it holds a row for each of ``streams``, else zeros of its
    rows' shape: streams of another batch begin afresh."""
    return state if len(state) == streams else np.zeros((streams, *state.shape[1:]))


def _with_head(maps: list[_Maps], head: int, new: _Maps) -> list[_Maps]:
    """``maps`` with head ``head``'s replaced by ``new``, a new list: an imagined branch may hold
    the old one."""
    return [new if h == head else kept for h, kept in enumerate(maps)]


class Hearing:
    """How strongly the association hears each carried region: one learned gain per route,
    starting at the working value and taught by the settle's contrast.

    A route is a held region and its one-way synapses into the association: the working trace,
    the efference copy, every block of the heard history and, with ``recall_drive_gain``, every
    port of the recall memory's read as one route. Its
    stimulus is the gain times what it holds, so the energy's derivative for the gain is local
    to the route: ``c = sum_j f'(v_j) a t_j sum_i W_ij m_i``, ``v_j`` the region's held
    potential, ``a`` the route's amplitude, ``t_j`` what it holds and ``m_i`` the association's
    movement between the nudged phases over their span. A lesson steps each gain on the mean of
    ``c`` over its taught rows; a sampled act keeps ``c`` per row in an eligibility of its own,
    stepped by the dopamine as every synapse is (``ActorCritic``). The step is normalized by the
    gain's own running second moment, as a synapse's is at the normalized teaching point, at
    ``rate``. A gain stays within ``[0, BOUND]``, or the lower ceiling its route was added with.
    """

    RHO = 0.99
    FLOOR = 1e-9
    BOUND = 8.0

    def __init__(self, rate: float, connectome: Connectome, association: np.ndarray,
                 model: Any) -> None:
        self.rate = float(rate)
        self.connectome = connectome
        self.association = np.asarray(association, dtype=np.int64)
        self.model = model
        self.routes: list[dict[str, Any]] = []
        self.second = np.zeros(0)
        self.count = 0

    def add(self, name: str, memory: Any, slot: int | None, start: int, stop: int,
            neurons: np.ndarray, bound: float = BOUND) -> None:
        """A route: ``memory.trace[:, start:stop]`` held on ``neurons`` at the memory's
        amplitude; its gain is ``memory.hearing``, at ``slot`` when the memory holds one per
        block, and goes no higher than ``bound``."""
        pre, post = np.asarray(self.connectome.pre), np.asarray(self.connectome.post)
        local_source = -np.ones(self.connectome.n, dtype=np.int64)
        local_source[neurons] = np.arange(len(neurons))
        local_cortex = -np.ones(self.connectome.n, dtype=np.int64)
        local_cortex[self.association] = np.arange(len(self.association))
        synapses = np.flatnonzero((local_source[pre] >= 0) & (local_cortex[post] >= 0))
        self.routes.append({
            "name": name, "memory": memory, "slot": slot, "start": start, "stop": stop,
            "width": len(neurons), "synapses": synapses, "bound": float(bound),
            "rows": local_source[pre[synapses]], "cols": local_cortex[post[synapses]],
            "places": local_source[pre[synapses]] * len(self.association)
            + local_cortex[post[synapses]],
        })
        self.second = np.zeros(len(self.routes))

    def gain(self, route: dict[str, Any]) -> float:
        memory, slot = route["memory"], route["slot"]
        return float(memory.hearing if slot is None else memory.hearing[slot])

    def gains(self) -> dict[str, float]:
        """Every route's gain, by name."""
        return {route["name"]: self.gain(route) for route in self.routes}

    def set(self, route: dict[str, Any], value: float) -> None:
        memory, slot = route["memory"], route["slot"]
        if slot is None:
            memory.hearing = value
        else:
            memory.hearing[slot] = value

    def _contrast_on_device(
        self, graph: Any, movement: Any, rows: np.ndarray | None, batch: int
    ) -> np.ndarray:
        """``_contrast`` of a movement on the device: the routes' synapses are read there, and one
        number per row and route comes back."""
        kernel = graph._torch if graph._torch is not None else graph._mlx
        return route_contrast(
            arrays_of(kernel), graph, self.routes, self.gain, self.model, len(self.association),
            movement, rows, batch,
        )

    def _movement_on_device(self, graph: Any, plus: Any, minus: Any) -> Any:
        """The association's movement between two phases on the device where both rest, in the
        maps' floats as the host's copies are; None otherwise."""
        kernel = graph._torch if graph._torch is not None else graph._mlx
        if kernel is None or not any(
            getattr(route["memory"], "_device", None) is not None for route in self.routes
        ):
            return None
        xp = arrays_of(kernel)
        if not (xp.rests_here(plus) and xp.rests_here(minus)):
            return None
        place = xp.neurons(self.association, columns(self.association))
        return xp.part(plus, place) - xp.part(minus, place)

    def _contrast(
        self, graph: Any, movement: np.ndarray, rows: np.ndarray | None, batch: int
    ) -> np.ndarray:
        """``c`` per row and route: ``movement`` is the association's over the phases' span, at
        ``rows`` of ``batch`` streams; a memory of another batch reads silence."""
        out = np.zeros((len(movement), len(self.routes)))
        backs = None
        kernel = graph._mlx
        if kernel is not None and graph._efficacy is None:  # the synapses rest on the device
            backs = kernel.read_back([
                (route["synapses"], route["rows"], route["cols"], route["width"])
                for route in self.routes
            ], movement)
        for k, route in enumerate(self.routes):
            memory = route["memory"]
            if len(memory.trace) != batch:
                continue
            trace = memory.trace if rows is None else memory.trace[rows]
            held = trace[:, route["start"]:route["stop"]]
            amplitude = float(memory.amplitude)
            v = amplitude * self.gain(route) * held
            if backs is None:
                if graph.layout.parallel_synapses:  # the last of a pair holds the entry
                    w = np.zeros((route["width"], len(self.association)))
                    w[route["rows"], route["cols"]] = graph._weights[route["synapses"]]
                else:
                    w = _route_matrix(
                        graph, route["synapses"], route["places"],
                        (route["width"], len(self.association)),
                    )
                back = movement @ w.T
            else:
                back = backs[k]
            out[:, k] = np.einsum("bj,bj->b", self.model.slope_at(v) * amplitude * held, back)
        return out

    def step(self, contrast: np.ndarray) -> None:
        """Every route's gain on its own contrast, over the root of its own second moment."""
        self.count += 1
        self.second = self.RHO * self.second + (1.0 - self.RHO) * contrast**2
        correction = 1.0 - self.RHO**self.count
        moved = self.rate * contrast / (np.sqrt(self.second / correction) + self.FLOOR)
        for route, change in zip(self.routes, moved, strict=True):
            self.set(route, float(np.clip(self.gain(route) + change, 0.0, route["bound"])))

    def learn(self, graph: Any, learned: Any, beta: float) -> None:
        """A lesson's step, at its taught rows."""
        if (
            not self.routes
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return
        taught = learned.taught
        rows = None if taught is None else np.flatnonzero(np.asarray(taught, dtype=bool))
        if rows is not None and not len(rows):
            return
        minus_state = learned.free if learned.opposite is None else learned.opposite
        moved = self._movement_on_device(graph, learned.nudged, minus_state)
        if moved is not None:
            span = beta if learned.opposite is None else 2.0 * beta
            batch = len(moved) if rows is None else len(np.asarray(taught))
            contrast = self._contrast_on_device(graph, moved / span, rows, batch)
            self.step(contrast.mean(axis=0))
            return
        plus = np.atleast_2d(learned.nudged.activation)
        one_sided = learned.opposite is None
        minus = np.atleast_2d(learned.free.activation if one_sided else learned.opposite.activation)
        span = beta if one_sided else 2.0 * beta
        movement = (plus - minus)[:, self.association] / span
        batch = len(movement) if rows is None else len(np.asarray(taught))
        self.step(self._contrast(graph, movement, rows, batch).mean(axis=0))

    def contrast_rows(self, graph: Any, plus: BrainState, minus: BrainState,
                      beta: float) -> np.ndarray:
        """A sampled act's ``c`` per row and route, for its eligibility."""
        moved = self._movement_on_device(graph, plus, minus)
        if moved is not None:
            return self._contrast_on_device(graph, moved / (2.0 * beta), None, len(moved))
        movement = (
            np.atleast_2d(plus.activation) - np.atleast_2d(minus.activation)
        )[:, self.association] / (2.0 * beta)
        return self._contrast(graph, movement, None, len(movement))


@dataclass
class Recall:
    """The index memory, a held-input recall: each stream's recent moments as slots, the
    address of a moment computed before its settle and the recalled drive held through it.

    Each moment the brain accepts (an act, a moment ``wait`` senses) is written after its settle
    as one slot per stream (``witness``): its key the event ``x`` as given (a brain gives its
    observation code), its value the event, and the slot before learns it as its next. Before each
    settle the moment's address ``q = x + x W_x + c W_c + b`` (``c`` the settled association
    activity of the moment before) weighs the slots once,

        p_i = softmax_i(q . K_i / temperature - recency * age_i + position terms)

    and ``stimulate`` holds ``amplitude * sum_i p_i V_i`` on the ``target`` region and, with
    ``next``, ``amplitude * sum_i p_i N_i`` on the ``next`` region through the settle, as a
    sensory drive is held: the settle answers on the mixture and does not revise it, so a second
    recall takes a second moment. Their one-way synapses into the association region learn by the
    settle's own contrast. ``learn`` is the address lesson, a rule beside the graph's contrast the
    brain runs after teacher lessons: it reads the association's movement between the nudged
    phases back through the read ports, scores each slot's value and next against it, weighs the
    scores with the read's attention and steps the maps toward the slots the lesson names.
    ``capacity`` bounds the slots per stream, the oldest dropped first; ``reset`` empties streams;
    ``end`` ends their episodes and keeps their slots.

    The options, each a field below, zero at birth where it adds a map, so a newborn reads as
    without it, and described and saved only where it differs from its founder (``OPTIONS``;
    ``CONSTRUCTION_OPTIONS`` shape the memory): ``boundary`` (a silent slot a stream begins
    with), ``window`` (an ordered key of the last k events), ``heads`` (several read ports over
    the one bank, reading in turn), ``position`` (a learned term per age bucket, ``position_rate``
    its lesson's rate), ``anchor`` (the same by a slot's place in its episode), ``relative`` (by
    its age relative to the previous read's focus), ``slope`` (a lean by age at every length),
    ``grid`` (past the table a term per residue of each of a few coprime numbers), ``when`` (the
    read also delivers when its slots happened), ``action`` (each slot carries the command
    executed), ``address_reward`` and ``position_reward`` (reward trains the maps through
    the actor's eligibility, ``contrast_rows``, ``propose``, ``accept``), ``tokens`` (which
    actions are tokens a slot can hold), ``gain`` (a learned gain per port on those tokens),
    ``copy`` (plastic synapses from the read ports into the motor region) and ``drive_gain``
    (the read's drive is one more route of ``Hearing``: one learned gain on every port, at most 1).

    A retired setting (``RETIRED``) is refused at construction, under
    ``Brain.retune`` and on load with one sentence naming the release that retired it and what
    the memory does instead (``removed``); a saved memory loads where it carries the one value the
    table keeps for it. docs/memory.md holds each option's law and its readings.
    """

    connectome: Connectome
    amplitude: float = 3.0
    temperature: float = 0.25
    recency: float = 0.25
    capacity: int = 64
    next: bool = True
    address: float = 0.01
    # the options beside the founder memory (``OPTIONS``)
    address_reward: float = 0.0
    position_reward: float = 0.0
    window: int = 1
    heads: int = 1
    source: str = "sensory"  # the observations' range
    target: str = "recall"  # the recall region; the next region is ``next``
    # the width of the observations the memory is given, when it is not the source range's: a
    # brain whose memories key on a code of the observation passes the code
    inputs: int = 0
    writes: int = 0  # slots written over the memory's life, one per stream and moment
    position: int = 32
    position_rate: float = 0.1
    boundary: bool = False
    copy: float = 0.0
    tokens: tuple[int, ...] | None = None
    action: bool = False
    # several heads read in turn, each from the share of every slot the heads before it left
    # every head's position term also reads a slot's place in its episode (``anchor_maps``)
    anchor: bool = False
    # every head's position term also reads a slot's age relative to the previous read's focus
    relative: bool = False
    # above 0: every head's position term also leans by age (``slope_maps``), taught at this rate
    slope: float = 0.0
    # every head's read also delivers its slots' age features on a region of its own
    when: bool = False
    # above 0, with ``tokens``: one learned scalar per read port on the tokens' identity into the
    # motor region's drive (``gains``), taught at this rate by the motor's movement
    gain: float = 0.0
    # past the exact table every head's position term also reads each slot's age modulo each of
    # these coprime numbers (``grid_maps``), and the slope takes no step
    grid: tuple[int, ...] = ()
    # the motor neurons a slot's command covers with ``action``: a brain passes its motor width
    actions: int = 0
    # the read's drive is one more carried route (``Hearing``): ``hearing`` multiplies the amplitude
    # on every port and ``trace`` keeps what the ports held at the latest read, at amplitude one
    drive_gain: bool = False
    hearing: float = 1.0
    keys: np.ndarray = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    nexts: np.ndarray = field(init=False, repr=False)
    held: np.ndarray = field(init=False, repr=False)
    previous: np.ndarray = field(init=False, repr=False)

    # the constants of the memory: settings of 0.92.1 that every reading took at one value,
    # retired as genes in 0.93.0 (``RETIRED`` refuses the names and keeps the saved values)
    heard = 1.0  # the present event's weight in its key and its address
    address_temperature = 0.1  # the temperature of the slots' scores in the address lesson
    window_decay = 1.0  # the weight of the window's events per position back
    # the settings ``Brain.retune`` may change in a life, and those that shape the memory
    RUNTIME = ("amplitude", "temperature", "recency", "address")
    CONSTRUCTION = ("capacity", "next")
    # the keys a memory always describes and saves, in the order 0.92.1 wrote them: the
    # founders above and the constants that were its genes, at their one value
    SAVED = ("amplitude", "temperature", "heard", "recency", "address", "address_temperature",
             "capacity", "next")
    # options beside the founder memory: a memory describes and saves an option only where it
    # differs from its founder, and a saved memory without one takes its founder; each is
    # runtime except the window, the heads, the position term, the copy pathway and the
    # executed command, which shape the memory or the anatomy
    OPTIONS = (
        "address_reward", "position_reward", "window", "heads", "position",
        "position_rate", "boundary", "copy", "tokens", "action", "anchor",
        "relative", "slope", "when", "gain", "grid", "drive_gain",
    )
    CONSTRUCTION_OPTIONS = ("window", "heads", "position", "copy", "tokens", "action", "anchor",
                            "relative", "slope", "when", "gain", "grid", "drive_gain")
    # what ``Brain.compose(..., recall=True)`` gives a new brain's memory over the founders above,
    # ordinary settings a caller's value overrides. The founder stays what a saved memory without
    # the setting means, so a saved brain keeps the values it was saved with
    # above this many observation-code coordinates ``Brain.compose`` holds a memory only by name:
    # the address map grows with the square of the width (140 MB at 4,096, 549 MB at 8,192)
    WIDE = 4096
    COMPOSED: ClassVar[dict[str, Any]] = {
        "recency": 0.0, "boundary": True, "anchor": True, "slope": 0.1, "position_rate": 0.01,
        "relative": True, "position_reward": 4.0, "grid": (7, 8, 9, 11), "capacity": 128,
        "drive_gain": True,
    }
    RELATIVE = 8  # the offsets from the previous read's focus with a bucket each, either side
    # the position lesson's rate a unit of dopamine steps the position maps at, times
    # ``position_reward``: the founder's ``position_rate``, whatever rate the lessons take
    REWARD_RATE = 0.1
    SLOPE = 8.0  # the age at which the slope's lean reaches one unit of the slope
    # the gains' step (``gain``): a row's rate is bounded by this over the energy of its reads, so
    # a row taught alone moves its own drive on a motor neuron by at most this many times the
    # neuron's movement between the nudged phases
    GAIN_BOUND = 5.0
    # the kinds of a head's maps beside its address map (``map_names``), in the order a
    # checkpoint keeps them, each kept with its option, and the per-stream state each reads,
    # kept before its maps: the reads' focus for the relative maps, the moments told for the
    # anchored ones
    _KINDS = (("position", None), ("slope", None), ("relative", "focus"), ("anchor", "told"),
              ("grid", None))
    # every stream's state: its slots, the cortex of the moment before, whether its latest slot
    # closed an episode, the window's events, the command executed at the moment before, the
    # moments told in its episode and the age of the slot its previous read attended most
    _STATE = ("keys", "values", "nexts", "held", "previous", "ended", "past", "acted", "told",
              "focus")
    # the retired settings, by name: the release that retired each, the one value a saved
    # memory loads at (the founder's, or the value its law keeps) and what the memory does
    # instead; ``removed`` makes the one sentence that refuses the name wherever it is given
    RETIRED = MappingProxyType({
        "address_step": ("0.93.0", "toward", "the address lesson steps on the named keys less "
                         "the attended ones, what a file at 'toward' saved"),
        "heard": ("0.93.0", heard, "the present event weighs 1.0 in its key and its address, "
                  "the one weight ever read"),
        "address_temperature": ("0.93.0", address_temperature, "the address lesson weighs "
                                "the slots' scores at the temperature 0.1, the one value "
                                "ever read"),
        "window_decay": ("0.93.0", window_decay, "the window's earlier events enter the key "
                         "at their full weight (1.0), the one value ever read"),
        "compete": ("0.95.0", True, "several heads read in turn, each taking of a slot at most "
                    "the share the heads before it left; cadence 0.94.0 takes it"),
        "null": ("0.95.0", None, "every moment's read takes its whole mass from the slots held; "
                 "cadence 0.94.0 takes it"),
    })

    @classmethod
    def founders(cls) -> dict[str, Any]:
        """The founder value of every setting the memory always describes: the genes of
        ``Brain.compose(..., recall=True)``; ``options()`` gives the options beside them."""
        defaults = {item.name: item.default for item in fields(cls)}
        return {name: defaults[name] for name in (*cls.RUNTIME, *cls.CONSTRUCTION)}

    @classmethod
    def composed(cls, named: Mapping[str, Any]) -> dict[str, Any]:
        """What a new memory takes (``COMPOSED``) beside the settings a caller ``named``:
        ``recall_position=0`` builds no position term, so what reads beside it stays out,
        ``recall_position_rate=0.0`` holds the maps still, so the reward's step on them does, and
        ``carried_gain=0.0`` learns no gain, so the read's route does."""
        out = dict(cls.COMPOSED)
        if named.get("recall_position") == 0:
            for name in ("position_rate", "anchor", "relative", "slope", "grid",
                         "position_reward"):
                out.pop(name, None)
        elif named.get("recall_position_rate") == 0:
            out.pop("position_reward", None)
        if named.get("carried_gain") == 0:
            out.pop("drive_gain", None)
        return out

    @classmethod
    def options(cls) -> dict[str, Any]:
        """The founder value of every option (``OPTIONS``), a gene ``recall_<option>`` of
        ``Brain.compose(..., recall=True)`` as well, which a memory describes and saves only
        where it differs from its founder."""
        defaults = {item.name: item.default for item in fields(cls)}
        return {name: defaults[name] for name in cls.OPTIONS}

    @staticmethod
    def removed(name: str) -> str:
        """The one sentence that refuses a retired setting, ``recall_<name>`` of ``RETIRED``:
        the release that retired it and what the memory does instead."""
        release, _, instead = Recall.RETIRED[name]
        return f"recall_{name} was retired in {release}: {instead}"

    @staticmethod
    def setting(name: str, value: Any) -> Any:
        """One setting, validated and converted; ``name`` is a field of the memory. A
        retired setting is refused by name (``removed``)."""
        if name in Recall.RETIRED:
            raise ValueError(Recall.removed(name))
        if name in (
            "amplitude", "recency", "address", "address_reward", "position_reward",
            "position_rate", "copy", "slope", "gain",
        ):
            return real(name, value, low=0)
        if name == "temperature":
            number = real(name, value, low=0)
            if number <= 0:
                raise ValueError(f"{name} must be positive")
            return number
        if name in (
            "next", "boundary", "action", "anchor", "relative", "when", "drive_gain",
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            return value
        if name in ("capacity", "window", "heads", "position"):
            low = 0 if name == "position" else 1  # no position term at 0
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < low
            ):
                raise ValueError(
                    f"{name} must be an integer of at least 1" if low else
                    "position must be a nonnegative integer: the ages with a bucket each, 0 for "
                    "no position term"
                )
            return int(value)
        if name == "grid":
            if isinstance(value, (str, bytes, Mapping)) or not isinstance(
                value, (Sequence, np.ndarray)
            ):
                raise ValueError("grid is a sequence of coprime integers of at least 2")
            moduli = []
            for item in value:
                if (
                    isinstance(item, (bool, np.bool_))
                    or not isinstance(item, (int, np.integer))
                    or item < 2
                ):
                    raise ValueError("grid is a sequence of coprime integers of at least 2")
                moduli.append(int(item))
            if any(gcd(a, b) > 1 for i, a in enumerate(moduli) for b in moduli[i + 1:]):
                raise ValueError("grid is a sequence of coprime integers of at least 2")
            return tuple(moduli)
        if name == "tokens":
            if value is None:
                return None
            if isinstance(value, (str, bytes, Mapping)) or not isinstance(
                value, (Sequence, np.ndarray)
            ):
                raise ValueError(
                    "tokens is a sequence of observation coordinates, one per motor neuron "
                    "(-1 where none)"
                )
            out = []
            for item in value:
                if (
                    isinstance(item, (bool, np.bool_))
                    or not isinstance(item, (int, np.integer))
                    or item < -1
                ):
                    raise ValueError(
                        "tokens are integer observation coordinates, -1 where a motor neuron "
                        "has none"
                    )
                out.append(int(item))
            return tuple(out)
        raise TypeError(f"unknown recall setting: {name!r}")

    @staticmethod
    def position_count(position: int, capacity: int) -> int:
        """The age buckets of a position term: one per age below ``position``, then one per
        doubling up to ``capacity``; 0 without a position term."""
        if not position:
            return 0
        if capacity <= position:
            return int(capacity)
        return int(position + np.floor(np.log2((capacity - 1) / position)) + 1)

    @staticmethod
    def ports(heads: int, next: bool = True) -> tuple[tuple[str, ...], ...]:
        """The regions of each head's read port: ``("recall", "next")`` for head 0 and
        ``("recall_h", "next_h")`` for head h (without ``next``, the recall region alone)."""
        names = ("recall", "next") if next else ("recall",)
        return tuple(
            names if h == 0 else tuple(f"{name}_{h}" for name in names) for h in range(heads)
        )

    @staticmethod
    def action_ports(heads: int, next: bool = True) -> tuple[tuple[str, ...], ...]:
        """The regions that hold each head's read of the command with ``action``: one per
        region of ``ports``, named ``<region>_action``."""
        return tuple(
            tuple(f"{name}_action" for name in port) for port in Recall.ports(heads, next)
        )

    def _port_neurons(self, name: str, width: int) -> np.ndarray:
        """The neurons of a read port region in value order: the region's ``width`` event
        neurons and, with ``action``, its command region's neurons after them."""
        populations = self.connectome.populations
        neurons = np.asarray(populations[name], dtype=np.int64)
        if len(neurons) != width:
            raise ValueError(
                f"the {name} region needs one neuron per event coordinate ({width})"
            )
        if not self.action:
            return neurons
        command = np.asarray(populations[f"{name}_action"], dtype=np.int64)
        if len(command) != self.actions:
            raise ValueError(
                f"the {name}_action region needs one neuron per motor neuron ({self.actions})"
            )
        return np.concatenate([neurons, command])

    def copy_synapses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The copy pathway's synapses: every synapse from a head's read port neuron into the
        motor region, their indices, and for each the value coordinate it reads and the motor
        neuron it drives (the motor population's order)."""
        connectome = self.connectome
        motor = np.asarray(connectome.populations.get("motor", ()), dtype=np.int64)
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        value = -np.ones(connectome.n, dtype=np.int64)
        for recall, following in self._heads:
            value[recall] = np.arange(len(recall))
            value[following] = np.arange(len(following))
        local_motor = -np.ones(connectome.n, dtype=np.int64)
        local_motor[motor] = np.arange(len(motor))
        synapses = np.flatnonzero((value[pre] >= 0) & (local_motor[post] >= 0))
        return synapses, value[pre[synapses]], local_motor[post[synapses]]

    def _check_copy(self) -> None:
        """With ``copy`` every read port reaches the motor region; ``tokens`` names one
        coordinate per motor neuron within the event."""
        motor = self.connectome.populations.get("motor", ())
        if self.tokens is not None and len(motor) and len(self.tokens) != len(motor):
            raise ValueError(
                f"tokens name one observation coordinate per motor neuron ({len(motor)})"
            )
        if not self.copy:
            return
        if not len(motor):
            raise ValueError("a copy pathway needs a population named 'motor'")
        synapses, _, _ = self.copy_synapses()
        pre = np.asarray(self.connectome.pre)[synapses]
        for recall, following in self._heads:
            for region in (recall, following):
                if len(region) and not np.isin(region[: self.key_width], pre).any():
                    raise ValueError(
                        "copy needs synapses from every read port into the motor region; "
                        "Brain.compose adds them"
                    )

    def _check_gain(self) -> None:
        """With ``gain`` every read port reaches the motor region through its gain on the declared
        tokens, whose coordinates lie in the value."""
        if not self.gain:
            return
        if self.tokens is None:
            raise ValueError("gain multiplies the declared tokens' identity: it needs tokens")
        if not len(self.motor):
            raise ValueError("gain needs a population named 'motor'")
        offset = self.key_width - len(self.observed)
        if offset < 0 or max(self.tokens) + offset >= self.key_width:
            raise ValueError(
                "gain needs the observation's own coordinates in the recall values: a brain "
                "without a code, or a learned code joined to an observation no wider than it"
            )

    @staticmethod
    def _rewarded(position_reward: float, position: int, position_rate: float) -> None:
        """Reward trains the position maps (``position_reward`` above 0) only where they exist
        and a lesson's step moves them: refused at construction and by ``Brain.retune``."""
        if position_reward and not position:
            raise ValueError("position_reward trains the position maps: it needs position")
        if position_reward and not position_rate:
            raise ValueError(
                "position_reward steps the position maps by a lesson's step: it needs "
                "position_rate above 0"
            )

    def __post_init__(self) -> None:
        self._device: _RecallOnDevice | None = None  # what rests on a device (``reside``)
        self._actor: Any = None  # the actor that keeps the read's eligibility
        for name in (*self.RUNTIME, *self.CONSTRUCTION, *self.OPTIONS):
            setattr(self, name, self.setting(name, getattr(self, name)))
        populations = self.connectome.populations
        if self.target != "recall" and self.heads > 1:
            raise ValueError("a recall memory of several heads reads into the recall regions")
        regions = (
            self.source, "association", self.target, *(("next",) if self.next else ()),
            *(name for port in self.ports(self.heads, self.next)[1:] for name in port),
        )
        if self.action and self.target != "recall":
            raise ValueError("a recall memory that keeps the command reads into the recall regions")
        if self.action:
            regions = (*regions, *(
                name for port in self.action_ports(self.heads, self.next) for name in port
            ))
        if self.when and self.target != "recall":
            raise ValueError("a recall memory that delivers when reads into the recall regions")
        if self.when:
            regions = (*regions, *self.when_regions(self.heads))
        for name in regions:
            if name not in populations:
                raise ValueError(f"a recall memory needs a population named {name!r}")
        if (
            isinstance(self.actions, (bool, np.bool_))
            or not isinstance(self.actions, (int, np.integer))
            or self.actions < 0
        ):
            raise ValueError("actions must be a nonnegative integer: the motor neurons")
        self.actions = int(self.actions)
        if self.action and not self.actions:
            raise ValueError("action needs the motor width a slot's command covers (actions)")
        # the command coordinates of a slot, of its key and of the maps' inputs
        self.command_width = self.actions if self.action else 0
        self.observed: np.ndarray = np.asarray(populations[self.source], dtype=np.int64)
        self.association: np.ndarray = np.asarray(populations["association"], dtype=np.int64)
        if (
            isinstance(self.inputs, (bool, np.bool_))
            or not isinstance(self.inputs, (int, np.integer))
            or self.inputs < 0
        ):
            raise ValueError("inputs must be a nonnegative integer; 0 reads the source range")
        self.inputs = int(self.inputs)
        self.key_width = self.inputs or len(self.observed)
        # the values are the events, with ``action`` followed by the command
        self.value_width = self.key_width + self.command_width
        width = self.key_width
        self.recall_neurons: np.ndarray = self._port_neurons(self.target, width)
        self.next_neurons: np.ndarray = (
            self._port_neurons("next", width) if self.next else np.zeros(0, np.int64)
        )
        self.address_width = self.window * width + self.command_width  # the window's k events
        # the maps read the event, and with ``action`` the command of the moment before
        self.map_width = width + self.command_width
        # every head's address map ``(w_x, w_c, b)``, head 0's first
        self.head_maps = self._zero_maps(self.heads, self.address_width)
        self._association_columns = columns(self.association)
        # every head's port: its recall and next neurons, their columns and their synapses into
        # the association; head 0's are the ones above
        self._heads: list[tuple[np.ndarray, np.ndarray]] = [
            (self.recall_neurons, self.next_neurons)
        ]
        for port in self.ports(self.heads, self.next)[1:]:
            recall = self._port_neurons(port[0], width)
            following = self._port_neurons(port[1], width) if self.next else np.zeros(0, np.int64)
            self._heads.append((recall, following))
        self._head_columns = [(columns(r), columns(n)) for r, n in self._heads]
        self._head_synapses = [(self._synapses(r), self._synapses(n)) for r, n in self._heads]
        # with ``when`` every head's when region: its neurons, columns and synapses into the
        # association
        self._whens: list[tuple[np.ndarray, Any, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        for name in self.when_regions(self.heads) if self.when else ():
            neurons = np.asarray(populations[name], dtype=np.int64)
            if len(neurons) != self.when_width:
                raise ValueError(
                    f"the {name} region needs one neuron per feature of when ({self.when_width})"
                )
            self._whens.append((neurons, columns(neurons), self._synapses(neurons)))
        # every neuron the read drives, the route of ``drive_gain``, and what they held at the
        # latest read (another batch's reads silence)
        self.port_neurons: np.ndarray = np.concatenate(
            [part for head in self._heads for part in head]
            + [neurons for neurons, _, _ in self._whens]
        )
        self._port_columns = columns(self.port_neurons)
        self.trace: np.ndarray = np.zeros((0, len(self.port_neurons)))
        self._rewarded(self.position_reward, self.position, self.position_rate)
        # whether a contrast has written the address columns of the reward eligibility
        self._address_held = False
        # every head's position maps ``(P_x, P_c, p)``, head 0's first; none without ``position``
        count = self.position_buckets
        self.position_maps = self._zero_maps(self.heads if count else 0, count)
        for name, verb in (("anchor", "reads"), ("relative", "reads"), ("slope", "leans"),
                           ("grid", "reads")):
            if getattr(self, name) and not count:
                raise ValueError(f"{name} {verb} beside the position term: it needs position")
        # every head's anchored maps ``(E_x, E_c, e)``, one column per anchored bucket and one
        # for the slots written before the episode began; none without ``anchor``
        self.anchor_maps = self._zero_maps(self.heads if self.anchor else 0, count + 1)
        # every head's relative maps ``(R_x, R_c, r)``, one column per offset and one for a
        # stream with no read before; none without ``relative``
        width_r = 2 * self.RELATIVE + 2
        self.relative_maps = self._zero_maps(self.heads if self.relative else 0, width_r)
        # every head's slope maps ``(S_x, S_c, s)``, one number each; none without ``slope``
        self.slope_maps = self._zero_maps(self.heads if self.slope else 0)
        # every head's grid maps ``(G_x, G_c, g)``, one column per residue of each modulus; none
        # without ``grid``
        self.grid_maps = self._zero_maps(self.heads if self.grid else 0, sum(self.grid))
        self._check_copy()
        # the motor region, which the gains drive (``gain``); empty where the anatomy has none
        self.motor: np.ndarray = np.asarray(populations.get("motor", ()), dtype=np.int64)
        self._motor_columns = columns(self.motor)
        self._check_gain()
        # with ``gain`` one scalar per read port in ``ports`` order, and the token coordinates in
        # the value and the motor neurons they drive
        self.gains: np.ndarray = np.zeros(
            self.heads * (2 if self.next else 1) if self.gain else 0
        )
        pairs = [(token, j) for j, token in enumerate(self.tokens or ()) if token >= 0]
        offset = self.key_width - len(self.observed)
        self._token_values = np.array([offset + t for t, _ in pairs], dtype=np.int64)
        self._token_motors = np.array([j for _, j in pairs], dtype=np.int64)
        self.last: _Moment | None = None
        self._attended: np.ndarray | None = None  # the attention of the latest read
        # every head's address and attention of the latest read (head 0's are ``last.address``
        # and ``attended``), and the writes the bank had then
        self._head_reads: (
            tuple[tuple[np.ndarray, ...], tuple[np.ndarray | None, ...], int] | None
        ) = None
        self._private = False  # an imagined branch reads the slots and writes none
        # the latest content scores per address over the present slots (``_content``)
        self._contents: list[tuple[np.ndarray, np.ndarray]] = []
        self._content_keys: np.ndarray | None = None
        self._scope_token: object | None = None  # the token of the operation in progress
        self._device_columns: Any = None  # the association's columns on a torch device
        # per slot array: the wider array it is a view of, the view, its first slot and its end
        self._rooms: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
        self.reset(0)

    # -- what rests on a torch device

    def __getattr__(self, name: str) -> Any:
        """An array that rests on the device (``_RecallOnDevice.OWNED``), read on the host: every
        array comes back, and the next moment places them there again."""
        if self.__dict__.get("_device") is not None and name in _RecallOnDevice.OWNED:
            self.leave_device()
            return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getstate__(self) -> dict[str, Any]:
        self.leave_device()
        return self.__dict__

    @property
    def resides(self) -> bool:
        """Whether this memory's options have the device's path (``_RecallOnDevice``)."""
        return (
            self.heads == 1 and self.window == 1 and not self.action and not self.when
            and not self.gain and self.tokens is None and self.target == "recall"
        )

    def reside(self, kernel: Any) -> None:
        """The slots, the maps and the reads on the device of a brain's ``kernel``. An array
        written on the host since brings the others back first."""
        if kernel is None:
            self.leave_device()
            return
        held = self._device
        if held is not None:
            if any(name in self.__dict__ for name in _RecallOnDevice.OWNED):
                self.leave_device()
            else:
                held.kernel = kernel
                return
        if self.resides and not self._private:
            self._device = _RecallOnDevice(self, kernel)

    def _resident_path(self) -> _RecallOnDevice | None:
        """What rests on the device, None once an array was written on the host: that brings the
        others back first, and the host's path reads them."""
        held = self._device
        if held is not None and any(name in self.__dict__ for name in _RecallOnDevice.OWNED):
            self.leave_device()
            return None
        return held

    def leave_device(self) -> None:
        """Every array back on the host, the actor's eligibility of the read with them; what was
        written on the host since stays."""
        held = self.__dict__.get("_device")
        if held is None:
            return
        written = {
            name: self.__dict__.pop(name) for name in _RecallOnDevice.OWNED
            if name in self.__dict__
        }
        held.give_back()
        self.__dict__.update(written)
        self._device = None
        actor = self._actor
        if actor is not None:
            for name in ("trace_memory", "_pending_memory"):
                value = getattr(actor, name)
                if value is not None and not isinstance(value, np.ndarray):
                    setattr(actor, name, _host_array(value))

    def joined_eligibility(self, trace: Any, contrast: Any, decay: float) -> Any:
        """A resident memory's eligibility after an outcome: ``e <- decay e + contrast``. A new
        tensor, so a refused outcome keeps the old one."""
        held = self._device
        assert held is not None
        if isinstance(contrast, np.ndarray):
            contrast = held.xp.here(contrast)
        if isinstance(trace, np.ndarray):
            trace = held.xp.here(trace)
        if trace is None or trace.shape != contrast.shape:
            return contrast
        return decay * trace + contrast

    # -- layout and per-stream state

    @property
    def neurons(self) -> int:
        """The neurons the memory drives: every head's recall region, and next region with
        ``next``, and with ``when`` its when region."""
        return int(
            sum(len(r) + len(n) for r, n in self._heads)
            + sum(len(neurons) for neurons, _, _ in self._whens)
        )

    @property
    def when_width(self) -> int:
        """The features of when a head's read delivers with ``when``: two of each slot's age,
        and with ``anchor`` two of its place in the episode."""
        return 4 if self.anchor else 2

    @staticmethod
    def when_regions(heads: int) -> tuple[str, ...]:
        """The region of each head's read of when with ``when``: ``recall_when`` for head 0
        and ``recall_h_when`` for head h."""
        return tuple(f"{port[0]}_when" for port in Recall.ports(heads, False))

    def when_features(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Each slot's features of when for each stream, or for the streams ``rows``,
        ``(rows, slots, when_width)``: ``log(1 + age) / log(1 + SLOPE)`` and ``exp(-age /
        SLOPE)``, and with ``anchor`` the same of its place in the episode (``told - 1 -
        age``), zero for a slot written before the episode began."""
        streams = self.streams if rows is None else len(rows)
        ages = self.ages()
        out = np.broadcast_to(self._when(ages)[None], (streams, len(ages), 2))
        if not self.anchor:
            return np.asarray(out)
        places = self._places(rows)
        anchored = np.where((places >= 0)[..., None], self._when(np.maximum(places, 0.0)), 0.0)
        return np.concatenate([out, anchored], axis=2)

    def _when(self, ages: np.ndarray) -> np.ndarray:
        """The two features of when of ``ages`` (or places in an episode), on a last axis:
        ``log(1 + age) / log(1 + SLOPE)`` and ``exp(-age / SLOPE)``."""
        return np.stack(
            [np.log1p(ages) / np.log1p(self.SLOPE), np.exp(-ages / self.SLOPE)], axis=-1
        )

    def _places(self, rows: np.ndarray | None) -> np.ndarray:
        """Each slot's place in its stream's current episode, ``told - 1 - age``, for each
        stream or for the streams ``rows``; below zero for a slot written before it began."""
        told = self.told if rows is None else self.told[rows]
        return np.asarray(told[:, None] - 1.0 - self.ages()[None, :])

    @property
    def streams(self) -> int:
        held = self._device
        return int(len(self.keys)) if held is None else held.streams

    @property
    def slots(self) -> int:
        """The slots every stream holds (a stream that began since holds fewer, ``held``)."""
        held = self._device
        return int(self.keys.shape[1]) if held is None else held.slots

    def ages(self) -> np.ndarray:
        """Moments since each slot was written: ``slots - 1`` for the oldest, 0 the latest."""
        return np.arange(self.slots - 1, -1, -1, dtype=float)

    def _whole_ages(self) -> np.ndarray:
        """``ages`` as whole numbers."""
        return np.arange(self.slots - 1, -1, -1, dtype=np.int64)

    def _age_buckets(self) -> np.ndarray:
        """Every whole age's bucket up to ``capacity - 1``, ``_bucket`` of each, made once for
        the memory's position and capacity; an older age lies in the last bucket with it."""
        held = self.__dict__.get("_age_bucket_table")
        if held is None or held[0] != (self.position, self.capacity):
            table = self._bucket(np.arange(self.capacity, dtype=float)).astype(np.int64)
            held = ((self.position, self.capacity), table)
            self.__dict__["_age_bucket_table"] = held
        return np.asarray(held[1])

    def position_map(self, head: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Head ``head``'s position maps ``(P_x, P_c, p)``; refused without a position term."""
        if not self.position_maps:
            raise ValueError("this memory has no position term (position=0)")
        if not 0 <= head < self.heads:
            raise ValueError(f"head must lie in [0, {self.heads})")
        return self.position_maps[head]

    @property
    def position_buckets(self) -> int:
        """The age buckets of the position term (``position_count``); 0 without one."""
        return self.position_count(self.position, self.capacity)

    def buckets(self) -> np.ndarray:
        """Each slot's age bucket: its age below ``position``, then ``position`` plus the
        doublings of its age over ``position``, the last bucket holding every older age."""
        if not self.position:
            return np.zeros(self.slots, dtype=np.int64)
        return np.asarray(self._age_buckets()[self._whole_ages()])

    def _bucket(self, ages: np.ndarray) -> np.ndarray:
        """``ages`` (or places in an episode) in the position term's buckets: below
        ``position`` one each, then one per doubling, the last holding every older one."""
        exact = float(self.position)
        out = np.where(ages < exact, ages, exact + np.floor(np.log2(np.maximum(ages, 1.0) / exact)))
        return np.asarray(np.minimum(out, self.position_buckets - 1))

    def anchored_buckets(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Each slot's anchored bucket for each stream, or for the streams ``rows``, ``(rows,
        slots)``: its place in the stream's current episode (``told - 1 - age``) bucketed as
        an age is, and ``position_buckets`` for a slot written before the episode began."""
        told = self.told if rows is None else self.told[rows]
        places = told[:, None] - 1 - self._whole_ages()[None, :]
        table = self._age_buckets()
        out = table[np.minimum(np.maximum(places, 0), len(table) - 1)]
        return np.where(places < 0, self.position_buckets, out).astype(np.int64)

    def relative_buckets(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Each slot's relative bucket for each stream, or for the streams ``rows``, ``(rows,
        slots)``: its age less the age of the previous read's focus, clipped to
        ``[-RELATIVE, RELATIVE]`` and shifted to start at zero; ``2 * RELATIVE + 1`` for a
        stream with no read before."""
        focus = self.focus if rows is None else self.focus[rows]
        offsets = self._whole_ages()[None, :] - focus[:, None]
        out = np.minimum(np.maximum(offsets, -self.RELATIVE), self.RELATIVE) + self.RELATIVE
        return np.where(focus[:, None] < 0, 2 * self.RELATIVE + 1, out).astype(np.int64)

    def grid_buckets(self) -> np.ndarray:
        """Each slot's residue under each modulus of ``grid`` as a column of the grid maps,
        ``(moduli, slots)``: the columns of a modulus follow those of the moduli before it."""
        ages = self.ages().astype(np.int64)
        starts = np.cumsum([0, *self.grid[:-1]])
        return np.stack([
            start + ages % modulus for start, modulus in zip(starts, self.grid, strict=True)
        ])

    def position_term(
        self, head: int, event: np.ndarray, cortex: np.ndarray, rows: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Head ``head``'s position term at every slot for the moments ``event`` (the event
        heard; with ``action`` the event and the command executed at the moment before, the
        maps' inputs) and ``cortex`` (the association of the moment before), ``(rows,
        slots)``; None without a position term or without slots. With ``anchor`` the term over
        each slot's place in the episode joins it; ``rows`` names the streams of the moments
        (every stream, in order, without)."""
        if not self.position_maps or not self.slots:
            return None
        term = _term(self.position_maps[head], event, cortex)[:, self.buckets()]
        # the terms over each stream's episode and previous read need the moments' streams; a
        # batch of other streams reads silence (``weights``) and gets the table and the slope
        if rows is not None or len(event) == self.streams:
            anchored = relative = None
            if self.anchor_maps:
                anchored = _term(self.anchor_maps[head], event, cortex)
            if self.relative_maps:
                relative = _term(self.relative_maps[head], event, cortex)
            if _compiled_terms():
                from .fused import add_episode_terms

                add_episode_terms(
                    term, anchored, relative,
                    self.told if rows is None else self.told[rows],
                    self.focus if rows is None else self.focus[rows],
                    self._age_buckets(), self.position_buckets, self.RELATIVE,
                )
            else:
                each = np.arange(len(event))[:, None]
                if anchored is not None:
                    term = term + anchored[each, self.anchored_buckets(rows)]
                if relative is not None:
                    term = term + relative[each, self.relative_buckets(rows)]
        if self.slope_maps:
            lean = _term(self.slope_maps[head], event, cortex)
            term = term + lean[:, None] * (self.ages() / self.SLOPE)[None, :]
        if self.grid_maps:
            grid = _term(self.grid_maps[head], event, cortex)[:, self.grid_buckets()].sum(axis=1)
            term = term + grid * (self.ages() >= self.position)
        return np.asarray(term)

    @property
    def attended_heads(self) -> tuple[np.ndarray | None, ...] | None:
        """Every head's attention ``p`` of the latest read over each stream's slots, head 0's
        first, each ``(streams, slots)`` as that read used it (recency included) and before
        the moment was written, read-only; None before a read, when the read found no slots,
        and once the streams changed."""
        if self._attended is None or self._head_reads is None:
            if self._device is not None and self._attended is not None:
                return (self._device.xp.host(self._attended),)
            return None if self._attended is None else (self._attended,)
        return self._head_reads[1]

    def head_map(self, head: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Head ``head``'s address map ``(w_x, w_c, b)``."""
        if not 0 <= head < self.heads:
            raise ValueError(f"head must lie in [0, {self.heads})")
        return self.head_maps[head]

    def _set_map(self, head: int, w_x: np.ndarray, w_c: np.ndarray, b: np.ndarray) -> None:
        self.head_maps = _with_head(self.head_maps, head, (w_x, w_c, b))

    def _zero_maps(self, heads: int, *columns: int) -> list[_Maps]:
        """``heads`` maps of the moment at zero, each onto ``columns`` outputs (one number
        without)."""
        return [
            (
                np.zeros((self.map_width, *columns)),
                np.zeros((len(self.association), *columns)),
                np.zeros(columns or (1,)),
            )
            for _ in range(heads)
        ]

    def _empty(self, batch: int) -> None:
        self._contents = []
        self._content_keys = None
        # with ``boundary`` every stream begins at its boundary slot, a silent moment
        slots = 1 if self._bounded() and batch else 0
        self.keys = np.zeros((batch, slots, self.address_width))
        self.values = np.zeros((batch, slots, self.value_width))
        self.nexts = np.zeros((batch, slots, self.value_width))
        self.held = np.ones((batch, slots), dtype=bool)
        self.previous = np.zeros((batch, len(self.association)))
        # per stream with ``action``: the command it executed at the moment before
        self.acted = np.zeros((batch, self.command_width))
        # per stream: its latest slot closed an episode (``end``) and learns no successor
        self.ended = np.zeros(batch, dtype=bool)
        # per stream: the moments written in its current episode (``anchor``)
        self.told = np.zeros(batch, dtype=np.int64)
        # per stream: the age of the slot its previous read attended most (``relative``); -1
        # before any read
        self.focus = np.full(batch, -1, dtype=np.int64)
        # the window's previous events per stream, the latest first; none: zeros
        self.past = np.zeros((0, self.window - 1, self.key_width))
        self._unread()

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Empty every stream, or the streams of ``rows`` only (streams that begin again): their
        slots, the cortex of the moment before and the window's events. The address map is
        learned and stays."""
        held_device = self._resident_path()
        if held_device is not None:
            held_device.reset(batch, rows)
            return
        if rows is not None:
            rows = _stream_rows(rows, batch)
        if rows is None or self.streams != batch:
            self._empty(batch)
            return
        chosen = np.zeros(batch, dtype=bool)
        chosen[rows] = True
        if chosen.all():
            self._empty(batch)
            return
        if not chosen.any():
            return
        keep = ~chosen
        # new arrays, never writes into the old ones: a caller may hold them to restore
        held = self.held & keep[:, None]
        keys = np.where(keep[:, None, None], self.keys, 0.0)
        values = np.where(keep[:, None, None], self.values, 0.0)
        nexts = np.where(keep[:, None, None], self.nexts, 0.0)
        if self._bounded():
            # the streams that begin again hold their boundary in the latest slot, the moment
            # the other streams wrote last
            if not held.shape[1]:
                keys, values, nexts = (
                    np.zeros((batch, 1, a.shape[2])) for a in (keys, values, nexts)
                )
                held = np.zeros((batch, 1), dtype=bool)
            held[chosen, -1] = True
        # slots no stream holds any more are dropped from the oldest end
        lead = held.any(axis=0)
        first = int(np.argmax(lead)) if lead.any() else held.shape[1]
        self.keys, self.values, self.nexts, self.held = (
            a[:, first:].copy() for a in (keys, values, nexts, held)
        )
        self._fresh(keep)
        self.ended = self.ended & keep
        self._unread()

    def _fresh(self, keep: np.ndarray) -> None:
        """The moment before begins afresh for the streams ``keep`` does not mark: the cortex
        of the moment before, the window's events and the command executed at it; and so does
        the episode: the next moment is its first and follows no read of it."""
        batch = len(keep)
        self.previous = np.where(keep[:, None], self.previous, 0.0)
        if len(self.acted) == batch:
            self.acted = np.where(keep[:, None], self.acted, 0.0)
        if len(self.past) == batch:
            self.past = np.where(keep[:, None, None], self.past, 0.0)
        self.told = np.where(keep, self.told, 0)
        self.focus = np.where(keep, self.focus, -1)

    def _unread(self) -> None:
        """No read is held: the next moment is addressed afresh."""
        self.last = None
        self._attended = None
        self._head_reads = None

    def end(self, batch: int, rows: np.ndarray | None = None) -> None:
        """The streams of ``rows`` (every stream without) end an episode and keep their
        slots: the latest slot of each learns no successor from the stream's next written
        moment, and the moment before (the cortex of the moment before and the window's
        events) begins afresh. The
        address map is learned and stays. Another batch size empties the streams, as
        ``reset`` does."""
        if rows is not None:
            rows = _stream_rows(rows, batch)
        if self.streams != batch:
            self._empty(batch)
            return
        chosen = np.zeros(batch, dtype=bool)
        chosen[slice(None) if rows is None else rows] = True
        if not chosen.any():
            return
        # new arrays, never writes into the old ones: a caller may hold them to restore
        if self._bounded() and chosen.all():
            # every stream begins its next episode at a boundary slot, which learns the first
            # event; the closed episode's latest slot learns no successor
            silent = self._silent()
            if not silent.all():
                if silent.any():  # a silent latest slot gives way to the new one, never stays
                    held = self.held.copy()
                    held[silent, -1] = False
                    self.held = held
                self._append(np.zeros((batch, self.address_width)), np.ones(batch, dtype=bool))
            self.ended = np.zeros(batch, dtype=bool)
        else:
            self.ended = self.ended | chosen
        self._fresh(~chosen)
        self._unread()

    def _bounded(self) -> bool:
        """Whether streams that begin hold a boundary slot: ``boundary`` on, and room for the
        boundary beside a moment (a memory of one slot holds none)."""
        return bool(self.boundary) and self.capacity > 1

    def _silent(self) -> np.ndarray:
        """Each stream whose latest slot it holds is silent and has no successor yet: a
        boundary, or a silent moment, which serves as one."""
        if not self.slots:
            return np.zeros(self.streams, dtype=bool)
        return np.asarray(
            self.held[:, -1]
            & ~self.keys[:, -1].any(axis=1)
            & ~self.values[:, -1].any(axis=1)
            & ~self.nexts[:, -1].any(axis=1)
        )

    def keep(self, rows: np.ndarray) -> None:
        """Keep the memories of ``rows`` only (streams that ended are dropped)."""
        rows = _stream_rows(rows, self.streams)
        for name in self._STATE:
            state = getattr(self, name)
            if len(state):  # the window's events and the command may hold no stream
                setattr(self, name, state[rows])
        self._unread()

    def snapshot(self) -> dict[str, Any]:
        """The per-stream state, to restore after a refused operation. The next write starts
        new arrays and changes none of these."""
        held = self._resident_path()
        if held is not None:
            return held.snapshot()
        self._rooms = {}
        return {name: getattr(self, name) for name in self._STATE}

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Return to a ``snapshot``; the next moment is addressed afresh. A snapshot of the host's
        arrays stays on the host, whatever the operation since placed on the device."""
        held = self._resident_path()
        if held is not None and not isinstance(snapshot["keys"], np.ndarray):
            held.restore(snapshot)
            return
        self.leave_device()
        for name, value in snapshot.items():
            if not isinstance(value, np.ndarray):
                # a device snapshot comes home as ``leave_device`` brings its arrays: the episodes'
                # counts and the reads' focus as integers
                value = _host_array(value)
                if name in ("told", "focus"):
                    value = value.astype(np.int64)
            setattr(self, name, value)
        self._unread()

    def reading(self) -> tuple[Any, ...]:
        """The latest read as ``witness``, ``learn`` and ``contrast_rows`` take it: the moment
        it addressed (``last``, None once the moment is written) and the attention it gave the
        slots (head 0's). ``restore``, ``end`` and ``reset`` address the next moment
        afresh; a read taken on the state they leave is held through them with ``hold``. With
        several heads every head's address and attention follow."""
        if self.heads > 1:
            return self.last, self._attended, self._head_reads
        return self.last, self._attended

    def hold(self, reading: tuple[Any, ...]) -> None:
        """Hold a ``reading`` as the latest read: one taken on the state the streams are in.
        A forecast at an episode's end reads the streams as the end leaves them before the
        end is committed; once it is, that read is the moment's, so the act it decides keeps
        the read's eligibility and ``attended_heads`` describes it."""
        last, attended = reading[0], reading[1]
        if self._device is None and attended is not None and not isinstance(attended, np.ndarray):
            attended = _host_array(attended)  # a read on the device of arrays that have left it
            attended.flags.writeable = False
            if last is not None:
                last = _Moment(last.event, *(_host_array(a) for a in last[1:]))
        self.last, self._attended = last, attended
        self._head_reads = reading[2] if len(reading) > 2 else None

    # -- the MLX device

    def place(self, kernel: Any) -> None:
        """A brain on the MLX device keeps a large address map there (``_MLX_MAP_ENTRIES``): its
        product is most of a moment's read and its step most of a lesson's address lesson. The
        slots and the smaller maps stay on the host."""
        if kernel is None or kernel.plain or self.__dict__.get("_device") is not None:
            return
        first = self.head_maps[0][0]
        if isinstance(first, _MlxMatrix) or first.size < _MLX_MAP_ENTRIES:
            return
        of = _MlxMatrix.of
        self.head_maps = [
            (
                cast(np.ndarray, of(kernel.mx, w_x)),
                cast(np.ndarray, of(kernel.mx, w_c)),
                b,
            )
            for w_x, w_c, b in self.head_maps
        ]

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any) -> np.ndarray:
        """The moment's address read into a drive copy (``stimulate`` with its code)."""
        return self.stimulate(drive, moment.code)

    def branch(self) -> Recall:
        """A private copy for an imagined branch: it reads the live slots and writes none; the
        cortex of the moment before, the window's events and the command follow the branch."""
        self.leave_device()
        branch = copy(self)
        branch.previous = self.previous.copy()
        branch.past = self.past.copy()
        branch.acted = self.acted.copy()
        branch._unread()
        branch._private = True
        branch._rooms = {}
        branch._contents = []
        branch._content_keys = None
        branch._scope_token = None
        return branch

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps: the slots, the address map, the streams whose latest
        slot closed an episode (while one does) and the window's previous events."""
        out = {
            name: getattr(self, name) for name in ("keys", "values", "nexts", "held", "previous")
        }
        for h, maps in enumerate(self.head_maps):  # every head's address map, head 0's first
            held = (np.asarray(maps[0]), np.asarray(maps[1]), maps[2])  # the host's, once read
            out.update(zip(self.map_names("address", h), held, strict=True))
        if self.ended.any():
            out["ended"] = self.ended
        if self.window > 1:
            out["past"] = self.past
        if self.action:  # the command each stream executed at the moment before
            out["acted"] = self.acted
        for kind, state in self._KINDS:  # each kind saved only with its option
            kept = getattr(self, f"{kind}_maps")
            if kept and state:
                out[state] = getattr(self, state)
            for h, maps in enumerate(kept):
                out.update(zip(self.map_names(kind, h), maps, strict=True))
        if self.gain:  # the gains, one per read port, saved only with the option
            out["gains"] = self.gains
        return out

    @staticmethod
    def map_names(kind: str, head: int) -> tuple[str, str, str]:
        """The checkpoint names of head ``head``'s three maps of one ``kind``: ``"address"``
        for ``(w_x, w_c, b)``, ``"position"`` for ``(P_x, P_c, p)``, ``"anchor"`` for ``(E_x,
        E_c, e)``, ``"relative"`` for ``(R_x, R_c, r)``, ``"slope"`` for ``(S_x, S_c, s)`` and
        ``"grid"`` for ``(G_x, G_c, g)``;
        head 0's names are bare and head ``h``'s carry ``_h``."""
        suffix = "" if head == 0 else f"_{head}"
        if kind == "address":
            return f"w_x{suffix}", f"w_c{suffix}", f"b{suffix}"
        if kind not in dict(Recall._KINDS):
            raise ValueError(f"unknown map kind: {kind!r}")
        return f"{kind}_x{suffix}", f"{kind}_c{suffix}", f"{kind}_b{suffix}"

    # -- the moment's address and read

    def _port(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.key_width:
            raise ValueError(f"observations are one row of {self.key_width} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        return x

    def _moment(self, event: np.ndarray) -> _Moment:
        """The address of the present moment of each stream, from the cortex of the moment
        before."""
        x = np.asarray(event, dtype=float)
        n = len(x)
        c = _of(self.previous, n)
        key = self.heard * x
        if self.window > 1:
            # the stream's previous events in order, each on coordinates of its own
            past = _of(self.past, n)
            scale = self.heard * self.window_decay ** np.arange(1, self.window)
            key = np.concatenate([key, (scale[None, :, None] * past).reshape(n, -1)], axis=1)
        inputs = x
        if self.action:
            # the command coordinates of the key are the slot's own command, written after the
            # settle; before it the address holds zero there, and the maps read the command
            # the stream executed at the moment before
            acted = _of(self.acted, n)
            key = np.concatenate([key, np.zeros((n, self.command_width))], axis=1)
            inputs = np.concatenate([x, acted], axis=1)
        q = _addressed(key, self.head_maps[0], inputs, c)
        return _Moment(x, c, q, key, inputs)

    def _attention(
        self,
        address: np.ndarray,
        keys: np.ndarray,
        held: np.ndarray | None,
        position: np.ndarray | None = None,
    ) -> np.ndarray:
        """The logits of the attention over slots: ``address . key / temperature`` less
        ``recency`` times each slot's age, plus a head's ``position`` term (``(rows, slots)``,
        ``_position``), ``-inf`` at a slot its stream does not hold."""
        logits = self._content(address, keys)
        if self.recency:
            logits = logits - np.asarray(self.recency * self.ages())[None, :]
        if position is not None:
            logits = logits + position
        return logits if held is None else np.where(held, logits, -np.inf)

    @contextmanager
    def _scope(self) -> Iterator[None]:
        """One public operation of the brain (``act``, ``lesson``, ``step``, ``live``,
        ``wait``). No caller runs between its reads, and the memory replaces its bank rather
        than writing into it, so the content products the operation reuses are bound to the
        bank array itself instead of a bitwise snapshot of it; they leave with the operation.
        A nested operation shares the outermost one's; a copy taken during one holds its
        token, which stops binding when the operation ends."""
        if self._scope_token in _OPERATIONS:
            yield
            return
        token = object()
        self._scope_token = token
        _OPERATIONS.add(token)
        self._contents, self._content_keys = [], None
        try:
            yield
        finally:
            _OPERATIONS.discard(token)
            self._scope_token = None
            self._contents, self._content_keys = [], None

    def _content(self, address: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Content logits with reusable raw dot products and the current temperature.

        Public expert key arrays remain writable. Between operations a bitwise snapshot
        detects in-place changes, including signed zero; object identity alone cannot
        establish unchanged contents. Within an operation (``_scope``) the bank is bound
        by identity, which a bank of many streams and slots needs: its snapshot and
        comparisons cost more than the products they save. At most four addresses per head
        and one bank snapshot are retained.
        """
        if (keys is not self.keys or address.dtype != np.float64
                or keys.dtype != np.float64 or not address.flags.c_contiguous):
            return np.asarray(_einsum("bk,bmk->bm", address, keys) / self.temperature)
        if self._scope_token is not None and self._scope_token in _OPERATIONS:
            if self._content_keys is not keys:
                self._content_keys = keys
                self._contents = []
        else:
            key_bits = keys.view(np.uint64)
            if (self._content_keys is None or self._content_keys.dtype != np.uint64
                    or not np.array_equal(self._content_keys, key_bits)):
                self._content_keys = key_bits.copy()
                self._contents = []
        bits = address.view(np.uint64)
        for scored, product in self._contents:
            if np.array_equal(scored, bits):
                return np.asarray(product / self.temperature)
        product = _einsum("bk,bmk->bm", address, keys)
        if np.isfinite(product).all():
            if len(self._contents) >= 4 * self.heads:
                self._contents.pop(0)
            self._contents.append((bits.copy(), product))
        return np.asarray(product / self.temperature)

    def weights(self, address: np.ndarray, position: np.ndarray | None = None) -> np.ndarray | None:
        """``p`` over each stream's slots for the ``(batch, key)`` addresses, or None when the
        streams hold no slot or are another batch. ``position`` is a head's position term at
        every slot (``position_term``); without it the attention is the content's and the
        recency's alone, which is what the read used only where the memory has no position
        term."""
        if not self.slots or self.streams != len(address):
            return None
        if self.held.all():
            logits = self._attention(address, self.keys, None, position)
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            return np.asarray(p / p.sum(axis=1, keepdims=True))
        # a stream that began since holds fewer slots; one that holds none reads silence
        logits = self._attention(address, self.keys, self.held, position)
        empty = ~self.held.any(axis=1)
        top = np.where(empty, 0.0, logits.max(axis=1))
        p = np.exp(logits - top[:, None])
        total = p.sum(axis=1, keepdims=True)
        return np.asarray(p / np.where(total > 0, total, 1.0))

    def stimulate(self, drive: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Address the present moment of every stream and read its slots into a drive copy:
        ``amplitude * sum_i p_i V_i`` on the recall region and, with ``next``,
        ``amplitude * sum_i p_i N_i`` on the next region. ``observations`` are ``(batch,
        inputs)``. Streams without slots, or of another batch, read silence. The moment is
        kept for its write and its lesson, and the attention the read used as
        ``attended_heads``. With several heads each head reads the slots with its own address
        into its own port."""
        if self._device is not None:
            return cast(np.ndarray, self._device.stimulate(drive, observations))
        moment = self._moment(self._port(observations))
        self.last = moment
        out = np.array(_on_host(drive), dtype=float)
        addresses, attended = self._reads(moment)
        whens = self.when_features() if self._whens and self.slots else None
        amplitude = 1.0 if self.drive_gain else self.amplitude
        for h, p in enumerate(attended):
            if p is not None:
                p.flags.writeable = False
                recall_columns, next_columns = self._head_columns[h]
                out[:, recall_columns] = amplitude * _einsum("bm,bmc->bc", p, self.values)
                if self.next:
                    out[:, next_columns] = amplitude * _einsum("bm,bmc->bc", p, self.nexts)
                if whens is not None:
                    out[:, self._whens[h][1]] = amplitude * _einsum("bm,bmw->bw", p, whens)
        if self.drive_gain:  # what the ports hold, kept for the route's lesson, at the gain
            self.trace = out[:, self._port_columns].copy()
            out[:, self._port_columns] = self.amplitude * self.hearing * self.trace
        if self.gain:  # every port's read of a token, through the port's gain, into the motor
            out[:, self._motor_columns] += self._gain_drive(out)
        self._attended = attended[0]  # the attention this read uses, kept past the write
        # with several heads, every head's address and attention, and the writes the bank had
        self._head_reads = (
            (tuple(addresses), tuple(attended), self.writes) if self.heads > 1 else None
        )
        return out

    def _reads(self, moment: _Moment) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """Every head's address and attention at ``moment``."""
        addresses = [moment.address] + [self._address(moment, h) for h in range(1, self.heads)]
        return addresses, self._read(addresses, moment)

    # -- the gains (``gain``)

    def _gain_drive(self, drive: np.ndarray) -> np.ndarray:
        """The gains' drive on the motor region, ``(batch, motor)``: every port's read of a token
        as ``drive`` holds it on the port's columns, through the port's gain, summed over the
        ports."""
        total = np.zeros((len(drive), len(self.motor)))
        ports = [
            part for recall_columns, next_columns in self._head_columns
            for part in ((recall_columns, next_columns) if self.next else (recall_columns,))
        ]
        for k, port in enumerate(ports):
            total[:, self._token_motors] += self.gains[k] * drive[:, port][:, self._token_values]
        return total

    def _held_attention(self, moment: _Moment) -> list[np.ndarray | None]:
        """Every head's attention over the slots held now, as the latest read used it
        (``stimulate``, head 0's ``_attended`` and the other heads' kept reads while the bank
        holds what they read), else computed now."""
        if self.heads == 1:
            return [self._attended]
        reads = self._head_reads
        if reads is not None and reads[2] == self.writes and len(reads[1]) == self.heads:
            return list(reads[1])
        return self._reads(moment)[1]

    def _port_reads(self, attended: Sequence[np.ndarray | None]) -> list[np.ndarray | None]:
        """Every port's read in ``ports`` order, ``(batch, value)`` each: head ``h``'s
        attention ``attended[h]`` over the slots' values and, with ``next``, their nexts, at
        ``amplitude``, as ``stimulate`` holds it on the port; None where the head read
        nothing."""
        reads: list[np.ndarray | None] = []
        for p in attended:
            if p is None or p.shape[1] != self.slots:
                reads.extend([None] * (2 if self.next else 1))
                continue
            amplitude = self.amplitude * self.hearing
            reads.append(amplitude * _einsum("bm,bmc->bc", p, self.values))
            if self.next:
                reads.append(amplitude * _einsum("bm,bmc->bc", p, self.nexts))
        return reads

    def _motor_movement(self, plus: BrainState, minus: BrainState) -> np.ndarray:
        """The motor region's movement between two phases, ``(batch, motor)``."""
        s_plus = np.atleast_2d(np.asarray(plus.activation, dtype=float))
        s_minus = np.atleast_2d(np.asarray(minus.activation, dtype=float))
        return np.asarray((s_plus - s_minus)[:, self.motor], dtype=float)

    def _gain_rates(self, reads: Sequence[np.ndarray | None]) -> np.ndarray:
        """Each row's rate of the gains' step, ``min(gain, GAIN_BOUND / E)``, ``E`` the energy of
        the row's reads over every port: the drive on a motor neuron reads them all together, so a
        row taught alone moves its own drive by at most ``GAIN_BOUND`` times the neuron's
        movement. A row that read nothing keeps the rate; its step is zero."""
        energy = np.zeros(self.streams)
        for r in reads:
            if r is not None:
                energy = energy + np.einsum("bi,bi->b", r, r)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.minimum(self.gain, self.GAIN_BOUND / energy)

    def _gain_step(self, read: np.ndarray, movement: np.ndarray) -> np.ndarray:
        """Per row, the sum over the tokens of a port's read of the token times its motor
        neuron's movement: the contrast of the gain that multiplies every token's identity."""
        return np.asarray(np.einsum(
            "bt,bt->b", read[:, self._token_values], movement[:, self._token_motors]
        ))

    def learn_gain(self, learned: LearnedState) -> dict[str, float]:
        """The gains' lesson after a teacher lesson whose phases are ``learned``, at its taught
        rows: each port's gain takes the mean over the rows of the row's rate (``_gain_rates``)
        times ``_gain_step`` on the port's read (``_port_reads``, what the port held at the moment
        the lesson's drive addressed) and the motor region's movement between the nudged phases. A
        step that is not finite is refused whole. Returns ``gain_rows``, the rows that moved, and
        ``gain_refused``, the rows a refused step left as they were; nothing without the option,
        and both zero for a lesson without a nudged phase or a moment."""
        if not self.gain:
            return {}
        quiet = {"gain_rows": 0.0, "gain_refused": 0.0}
        moment = self.last
        if (
            moment is None
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return quiet
        taught = learned.taught
        rows = (
            np.arange(len(moment.address))
            if taught is None
            else np.flatnonzero(np.asarray(taught, dtype=bool))
        )
        if not self.slots or not len(rows) or self.streams != len(moment.address):
            return quiet
        movement = self._motor_movement(
            learned.nudged, learned.free if learned.opposite is None else learned.opposite
        )
        reads = self._port_reads(self._held_attention(moment))
        with np.errstate(over="ignore", invalid="ignore"):
            rates = self._gain_rates(reads)[rows]
            gains = self.gains + np.array([
                0.0 if r is None else np.mean(rates * self._gain_step(r[rows], movement))
                for r in reads
            ])
        if not np.isfinite(gains).all():
            return {"gain_rows": 0.0, "gain_refused": float(len(rows))}
        self.gains = gains
        return {"gain_rows": float(len(rows)), "gain_refused": 0.0}

    def _gain_rows(
        self, out: np.ndarray, moment: _Moment, plus: BrainState, minus: BrainState
    ) -> None:
        """The gains' eligibility of a sampled act, written into the last columns of ``out``
        (``contrast_rows``' array) on the array's first row: per row and port, the row's rate
        (``_gain_rates``) times ``_gain_step`` on the act's two nudged phases, the step the lesson
        would take were the act's action the label; a row whose step is not finite keeps none."""
        start = out.shape[2] - len(self.gains)
        movement = self._motor_movement(plus, minus)
        reads = self._port_reads(self._held_attention(moment))
        rate = self._gain_rates(reads)
        for k, r in enumerate(reads):
            if r is None:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                step = rate * self._gain_step(r, movement)
            kept = np.isfinite(step)
            out[kept, 0, start + k] = step[kept]

    def _read(self, addresses: Sequence[np.ndarray], moment: _Moment) -> list[np.ndarray | None]:
        """Every head's attention over the slots held now, head ``h`` at ``addresses[h]`` with
        its position term at ``moment``: the first head's own (``weights``), and each after it
        taking of every slot at most the share the heads before it left (``_share``)."""
        attended: list[np.ndarray | None] = []
        left: np.ndarray | None = None  # each slot's share the heads before have left
        for h, address in enumerate(addresses):
            position = self.position_term(h, moment.inputs, moment.cortex)
            p = self.weights(address, position)
            if p is not None and self.heads > 1:
                if left is not None:
                    held = None if self.held.all() else self.held
                    logits = self._attention(address, self.keys, held, position)
                    p = _share(logits, left, p.sum(axis=1))
                left = _left(p) if left is None else left * _left(p)
            attended.append(p)
        return attended

    def _lesson_reads(self, moment: _Moment) -> list[np.ndarray | None]:
        """For each head's lesson, the attention it read the moment with (None for head 0, whose
        prior is its own attention): the reads ``stimulate`` kept while the bank holds what they
        read, else the read computed here, before any lesson steps a map."""
        if self.heads == 1:
            return [None]
        reads = self._head_reads
        if reads is not None and reads[2] == self.writes and all(r is not None for r in reads[1]):
            attended = list(reads[1])
        else:
            attended = self._reads(moment)[1]
        return [None, *attended[1:]]

    def _address(self, moment: _Moment, head: int) -> np.ndarray:
        """Head ``head``'s address of a moment: its key and head 0's inputs through the head's
        own map."""
        if head == 0:
            return moment.address
        return _addressed(moment.key, self.head_maps[head], moment.inputs, moment.cortex)

    def _head_address(self, moment: _Moment, head: int) -> np.ndarray:
        """Head ``head``'s address of the moment the latest read addressed: the one it read
        with (kept by ``stimulate``), or computed from its map."""
        reads = self._head_reads
        if head and reads is not None and len(reads[0]) == self.heads:
            return reads[0][head]
        return self._address(moment, head)

    # -- the write after the settle

    def activity(self, state: BrainState) -> np.ndarray:
        """The association region's settled activation as a host array, ``(batch,
        association)``. A state on a torch device is sliced there, and only the slice comes
        to the host."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        cols = self._association_columns
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                array = np.asarray(cols)
                key = (str(s.device), array.dtype.str, array.shape, array.tobytes())
                cached = self._device_columns
                if cached is None or cached[0] != key:
                    import torch

                    index = torch.as_tensor(np.asarray(cols), device=s.device)
                    self._device_columns = (key, index)
                else:
                    index = cached[1]
                part = s[:, index]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        settled = np.atleast_2d(np.asarray(state.activation, dtype=float))
        return np.asarray(settled[:, cols], dtype=float).copy()

    def witness(
        self, observations: np.ndarray, state: BrainState, command: np.ndarray | None = None
    ) -> None:
        """After the settle: the ``(batch, inputs)`` observations the streams acted on join
        their slots under the moment's key, the slot before learns them as its next, and the
        settled cortex of ``state`` and the window move on to this moment. With ``action``
        ``command`` is the ``(batch, actions)`` command each stream executed at this moment
        (None for a moment without one, which carries zeros); it joins the slot's key and
        value and becomes the command of the moment before. Another batch size begins the
        streams with empty memories first. An imagined branch moves the cortex, the window
        and the command only."""
        if self._device is not None:
            self._device.witness(observations, state, command)
            return
        x = self._port(observations)
        moment = self.last
        if moment is None or moment.event.shape != x.shape or not np.array_equal(moment.event, x):
            moment = self._moment(x)
        cortex = self.activity(state)
        if len(cortex) != len(x):
            raise ValueError("the settled state needs one row per stream")
        key, value = moment.key, moment.event
        if self.action:
            done = np.zeros((len(x), self.command_width))
            if command is not None:
                done = np.asarray(command, dtype=float)
                if done.shape != (len(x), self.command_width) or not np.isfinite(done).all():
                    raise ValueError(
                        f"a command is one finite row of {self.command_width} values per stream"
                    )
            key = np.concatenate([key[:, : -self.command_width], done], axis=1)
            value = np.concatenate([value, done], axis=1)
        if not self._private:
            self._observe(key, value)
            self.writes += len(x)
        if self.action:  # after the write, which begins streams of another batch afresh
            self.acted = done.copy()
        self.previous = cortex
        if self.window > 1:  # this moment enters the window, the oldest leaves it
            past = _of(self.past, len(x))
            self.past = np.concatenate([moment.event[:, None, :], past[:, :-1]], axis=1)
        self.last = None

    def _observe(self, key: np.ndarray, value: np.ndarray) -> None:
        """Write one moment per stream: the latest slot learns what followed it, a new slot
        holds this moment; beyond ``capacity`` the oldest slot is dropped. A stream that began
        since the others (``reset`` of its rows) holds fewer slots: its latest held slot,
        wherever it lies, learns this moment as its next. A stream whose latest slot closed
        an episode (``end``) teaches it nothing, and its new slot is open again."""
        if len(key) != self.streams:
            self._empty(len(key))
        successor: tuple[Any, ...] | None = None
        if self.slots:
            if self.held.all() and not self.ended.any():
                successor = (slice(None), value)
            else:
                # each stream's latest held slot learns what followed it, unless it closed an
                # episode
                held = self.held
                rows = held.any(axis=1) & ~self.ended
                latest = held.shape[1] - 1 - np.argmax(held[:, ::-1], axis=1)
                index = np.flatnonzero(rows)
                successor = (index, latest[index], value[index])
        if self.ended.any():  # every stream's new slot is open to its successor
            self.ended = np.zeros(len(key), dtype=bool)
        # the slot this moment's read attended most, one moment older once this moment is
        # written; without a read the previous focus ages too
        attended = self._attended
        if (
            attended is not None
            and attended.shape == (len(key), self.slots)
            and self.slots
        ):
            read = np.argmax(attended, axis=1)
            seen = attended.max(axis=1) > 0
            focus = np.where(seen, self.ages()[read].astype(np.int64) + 1, -1)
        else:
            focus = np.where(self.focus >= 0, self.focus + 1, -1)
        self._append(key, np.ones(len(key), dtype=bool), value, successor)
        self.told = self.told + 1
        self.focus = focus

    def _longer(
        self, name: str, old: np.ndarray, drop: int, new: np.ndarray | None
    ) -> np.ndarray:
        """The slots ``old`` less the ``drop`` oldest, with a new latest slot of ``new`` (zero
        without one). The slots are a view into a wider array, in the order every read sums
        them, and the new slot is written in the room behind them, so a write costs one slot
        and not the bank; when the room runs out the slots move into a new array of twice their
        number, at most twice the capacity. An array the memory did not make itself starts one."""
        held = self._rooms.get(name)
        if (
            held is not None and held[1] is old and old.base is held[0]
            and held[3] < held[0].shape[1]
        ):
            room, first, stop = held[0], held[2] + drop, held[3]
        else:
            kept = old.shape[1] - drop
            size = min(2 * self.capacity, max(16, 2 * kept + 2))
            room = np.empty((old.shape[0], size, *old.shape[2:]), dtype=old.dtype)

            def move(start: int, end: int) -> None:
                room[start:end, :kept] = old[start:end, drop:]

            if _by_rows(move, len(old), old.size) is None:
                move(0, len(old))
            first, stop = 0, kept
        if new is None:
            room[:, stop] = 0
        else:
            room[:, stop] = new
        out = room[:, first : stop + 1]
        self._rooms[name] = (room, out, first, stop + 1)
        return out

    def _append(
        self,
        key: np.ndarray,
        held: np.ndarray,
        value: np.ndarray | None = None,
        successor: tuple[Any, ...] | None = None,
    ) -> None:
        """A new latest slot for every stream, of ``key`` and ``value`` (zero without one) and
        no successor yet, held by the streams ``held`` marks; beyond ``capacity`` the oldest
        slot is dropped. ``successor`` is what the slots before learn as their next
        (``_observe``): ``(slice(None), value)`` for every stream's latest slot, or ``(rows,
        slots, values)``. The arrays are new views of the slots (``_longer``): a caller that holds
        the old ones to restore them says so (``snapshot``)."""
        if value is None:
            value = np.zeros((len(key), self.value_width))
        kept = self.keys.shape[1]
        drop = max(kept + 1 - self.capacity, 0)
        nexts = self._longer("nexts", self.nexts, drop, None)
        if successor is not None and kept:
            if len(successor) == 2:
                if kept - drop:  # the latest slot survives the drop
                    nexts[:, kept - 1 - drop] = successor[1]
            else:
                rows, slots, values = successor
                keep = slots >= drop
                nexts[rows[keep], slots[keep] - drop] = values[keep]
        self.keys = self._longer("keys", self.keys, drop, key)
        self.values = self._longer("values", self.values, drop, value)
        self.nexts = nexts
        self.held = self._longer("held", self.held, drop, held)
        self._contents = []
        self._content_keys = None

    # -- the address lesson

    def _synapses(self, region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The synapses from ``region`` into the association region: their indices, and the
        source and association coordinate of each."""
        connectome = self.connectome
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        local_source = -np.ones(connectome.n, dtype=np.int64)
        local_source[region] = np.arange(len(region))
        local_cortex = -np.ones(connectome.n, dtype=np.int64)
        local_cortex[self.association] = np.arange(len(self.association))
        synapses = np.flatnonzero((local_source[pre] >= 0) & (local_cortex[post] >= 0))
        return synapses, local_source[pre[synapses]], local_cortex[post[synapses]]

    def readback(
        self, graph: NeuralGraph, region: str, movement: np.ndarray, head: int = 0
    ) -> np.ndarray:
        """``movement @ W.T`` for the one-way synapses of the ``"recall"`` or ``"next"``
        region of head ``head``'s port into the association region: per source neuron, the
        sum over its synapses of effective weight times the movement of its association
        neuron."""
        recall_synapses, next_synapses = self._head_synapses[head]
        synapses, rows, cols = recall_synapses if region == "recall" else next_synapses
        width = len(self._heads[head][0 if region == "recall" else 1])
        if graph.layout.parallel_synapses:
            w = np.zeros((width, len(self.association)))
            np.add.at(w, (rows, cols), graph._weights[synapses])
        else:  # one synapse per entry: the sum with zero that np.add.at would make, in one pass
            w = _route_matrix(
                graph, synapses, rows * len(self.association) + cols, (width, len(self.association))
            )
            w += 0.0
        return np.asarray(movement @ w.T)

    def _read_backs(
        self, graph: NeuralGraph, movement: np.ndarray, head: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Head ``head``'s two ``readback`` reads: on the blocks where the synapses rest on the
        MLX device, in one launch."""
        kernel = graph._mlx
        if kernel is not None and graph._efficacy is None:
            ports = [
                (*synapses, len(neurons)) for synapses, neurons in zip(
                    self._head_synapses[head], self._heads[head], strict=True
                )
            ]
            read = kernel.read_back(ports if self.next else ports[:1], movement)
            if read is not None:
                return read[0], read[1] if self.next else None
        d_value = self.readback(graph, "recall", movement, head)
        d_next = self.readback(graph, "next", movement, head) if self.next else None
        return d_value, d_next

    def learn(
        self, graph: NeuralGraph, learned: LearnedState, content: np.ndarray | None = None
    ) -> dict[str, float]:
        """The address lesson after a lesson whose phases are ``learned``, at its taught rows,
        through the synapses of ``graph`` (the learner's brain after the lesson's update).
        ``content`` (a boolean per row, from the brain with ``tokens``) keeps the rows whose
        answer is a token a slot can hold: the lesson takes its target from the slots'
        contents, and an answer that is no token (yes or no, a count) names none of them.
        Reads the moment the lesson's drive addressed (``stimulate``); returns how many rows
        moved, the mean absolute error the step corrects before it (``address_error``, the
        addresses' error ``K* - q``), and how many rows a refused step left as they were
        (``address_refused``). With several heads each head steps its own map; the
        rows and refusals are summed over the heads (a row counts once per head that moved
        it) and the error is the mean over the heads that moved. With a position term the
        report also counts the rows whose position maps moved (``position_rows``) and those a
        refused position step left as they were (``position_refused``), summed over the
        heads."""
        if self._device is not None:
            return self._device.learn(graph, learned)
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        if self.position_maps:
            quiet.update(position_rows=0.0, position_refused=0.0)
        moment = self.last
        positions = bool(self.position_maps) and (self.position_rate > 0 or self.slope > 0)
        if (
            not (self.address or positions)
            or moment is None
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return quiet
        taught = learned.taught
        rows = (
            np.arange(len(moment.address))
            if taught is None
            else np.flatnonzero(np.asarray(taught, dtype=bool))
        )
        if content is not None:
            rows = rows[np.asarray(content, dtype=bool)[rows]]
        if not self.slots or not len(rows) or self.streams != len(moment.address):
            return quiet
        plus = np.atleast_2d(learned.nudged.activation)
        minus = np.atleast_2d(
            learned.free.activation if learned.opposite is None else learned.opposite.activation
        )
        movement = (plus - minus)[:, self.association]
        reports = []
        moved_positions = refused_positions = 0
        # the attention every competing head read with, taken before any head's lesson steps
        reads = self._lesson_reads(moment)
        for h in range(self.heads):
            d_value, d_next = self._read_backs(graph, movement, h)
            head_moment = moment._replace(address=self._head_address(moment, h))
            # one reading of the rows for both steps: the address step changes neither the
            # address the read used nor this head's position term
            found = self._row_steps(head_moment, rows, d_value, d_next, h, reads[h])
            if found is None:  # no row moves (a competing head may have had nothing to read)
                continue
            if self.address:
                reports.append(self._step(head_moment, rows, d_value, d_next, head=h, found=found))
            if positions:
                rows_moved, rows_refused = self._position_step(rows, h, found)
                moved_positions += rows_moved
                refused_positions += rows_refused
        # every head's rows and refusals summed, the error averaged over the heads that moved
        moved = [r["address_error"] for r in reports if r["address_rows"]]
        report = {
            "address_rows": float(sum(r["address_rows"] for r in reports)),
            "address_error": float(np.mean(moved)) if moved else 0.0,
            "address_refused": float(sum(r["address_refused"] for r in reports)),
        }
        if self.position_maps:
            report["position_rows"] = float(moved_positions)
            report["position_refused"] = float(refused_positions)
        return report

    def _position_step(
        self, rows: np.ndarray, head: int, found: tuple[np.ndarray, ...]
    ) -> tuple[int, int]:
        """The step of head ``head``'s position maps toward the slots the lesson's movement
        names (``p*`` of the address lesson; ``found`` is the reading of the streams ``rows``,
        ``_row_steps``): each moving row's terms take the cross-entropy gradient ``g_b =
        sum_{i in b} (p*_i - p_i)``, and the maps the least-mean-squares step on ``z = [x, c,
        1]`` at ``position_rate``, each row's rate bounded by its own energy; the anchored and
        relative maps take the same gradient over their buckets, and the slope maps along each
        slot's age at ``slope``. Returns the rows that moved and the rows a refused step left
        as they were: a step that is not finite is refused whole and moves none."""
        moving, x, c, _, _, _, target, prior = found
        rate = self.position_rate
        count = self.position_buckets
        streams = np.asarray(rows)[moving]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # a slot its stream does not hold has neither attention nor a target
            gradient = np.where(np.isfinite(prior), target - np.exp(prior), 0.0)
            energy = np.einsum("bi,bi->b", x, x) + np.einsum("bi,bi->b", c, c) + 1.0
            bound = np.minimum(1.0, 1.0 / (rate * energy))[:, None]
            step = _per_bucket(gradient, self.buckets()[None, :], count) * bound
            stepped = {"position_maps": _lms(self.position_maps[head], step, rate, x, c)}
            if self.anchor_maps:  # the same gradient over each slot's place in the episode
                step = _per_bucket(gradient, self.anchored_buckets(streams), count + 1) * bound
                stepped["anchor_maps"] = _lms(self.anchor_maps[head], step, rate, x, c)
            if self.relative_maps:  # the same gradient over each slot's offset from the focus
                width = 2 * self.RELATIVE + 2
                step = _per_bucket(gradient, self.relative_buckets(streams), width) * bound
                stepped["relative_maps"] = _lms(self.relative_maps[head], step, rate, x, c)
            # the same gradient, read along each slot's age; with ``grid`` only within the table
            if self.slope_maps and not (self.grid_maps and self.slots > self.position):
                lean = gradient @ (self.ages() / self.SLOPE)
                lean = lean * np.minimum(1.0, 1.0 / (self.slope * energy))
                stepped["slope_maps"] = _lms(self.slope_maps[head], lean, self.slope, x, c)
            if self.grid_maps:  # past the table, the gradient over its peak, on the residues
                attended = np.where(np.isfinite(prior), np.exp(prior), 0.0)
                peak = np.maximum(
                    target.max(axis=1, keepdims=True), attended.max(axis=1, keepdims=True)
                )
                far = np.where(peak > 0, gradient / peak, gradient)
                far = far * (self.ages() >= self.position)
                width = sum(self.grid)
                step = sum(
                    _per_bucket(far, residues[None, :], width) for residues in self.grid_buckets()
                ) * bound
                stepped["grid_maps"] = _lms(self.grid_maps[head], step, rate, x, c)
        # every map of the step or none: a step that is not finite is refused whole
        if not all(np.isfinite(a).all() for maps in stepped.values() for a in maps):
            return 0, int(len(x))
        for name, maps in stepped.items():
            setattr(self, name, _with_head(getattr(self, name), head, maps))
        return int(len(x)), 0

    def _step(
        self,
        moment: _Moment,
        rows: np.ndarray,
        d_value: np.ndarray,
        d_next: np.ndarray | None,
        head: int = 0,
        found: tuple[np.ndarray, ...] | None = None,
    ) -> dict[str, float]:
        """The least-mean-squares step of the address map toward the keys of the slots the
        scores name, on each row's inputs ``z = [x, c, 1]``. A row's rate is ``address``
        bounded by the row's own energy, ``r = min(address, 1 / |z|^2)``: a row taught alone
        moves its own address toward its target, at most onto it. The rows' steps are
        averaged, and each row's step moves the addresses of the rows whose inputs overlap
        its own, so in a batch a row can pass its target. The batch error does not grow: the
        rows' squared distances to their targets, each weighted by the row's rate ``r``, sum
        to no more after the step than before (with equal weights where no row reaches the
        bound). The candidate map and the candidate addresses of the rows are checked before
        the step is kept; one that is not finite refuses the step, the map stays as it was
        and the report counts the refused rows. ``head`` names the map that steps
        (``moment.address`` is that head's address); ``found`` is the rows' reading
        (``_row_steps``) when the caller has it."""
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        if found is None:
            # the shares a competing head read are its prior, as in ``learn``
            found = self._row_steps(
                moment, rows, d_value, d_next, head, self._lesson_reads(moment)[head]
            )
        if found is None:
            return quiet
        _, x, c, key, error, step, _, _ = found
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            w_x, w_c, b = _lms(self.head_map(head), step, self.address, x, c)
            address = _addressed(key, (w_x, w_c, b), x, c)
        if not all(_finite(a) for a in (w_x, w_c, b, address)):
            return {**quiet, "address_refused": float(len(x))}
        self._set_map(head, w_x, w_c, b)
        return {
            "address_rows": float(len(x)),
            "address_error": float(np.abs(error).mean()),
            "address_refused": 0.0,
        }

    def _row_steps(
        self,
        moment: _Moment,
        rows: np.ndarray,
        d_value: np.ndarray,
        d_next: np.ndarray | None,
        head: int = 0,
        read: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...] | None:
        """Each row's step of the address lesson before the average over rows: the rows of
        ``rows`` that move (a movement and a held slot), and at those rows the event ``x``,
        the cortex ``c``, the key, the error (the named keys less the attended ones, zero where the
        attention is what the lesson names), the error bounded by the row's own energy, ``error *
        min(1, 1 / (address |z|^2))``, the attention ``p*`` the lesson names and the log of the
        prior attention the address gave (head ``head``'s position term in it). Among competing
        heads ``read`` is the attention the head read with over every stream's slots
        (``_lesson_reads``), and it is the prior; a row it read nothing at does not move. None when
        no row moves. The arithmetic is unchecked: a caller checks what it keeps."""
        if len(rows) == len(self.keys):  # every stream, in order: the slots themselves
            values, nexts, keys, held = self.values, self.nexts, self.keys, self.held
        else:
            values, nexts, keys = self.values[rows], self.nexts[rows], self.keys[rows]
            held = self.held[rows]
        masked = not held.all()
        score = _einsum("bc,bmc->bm", d_value, values)
        norm = np.sum(d_value * d_value, axis=1)
        if d_next is not None:
            score += _einsum("bc,bmc->bm", d_next, nexts)
            norm += np.sum(d_next * d_next, axis=1)
        norm = np.sqrt(norm)
        moving = norm > 1e-12
        if masked:
            moving &= held.any(axis=1)  # a stream that began since and holds no slot
        if not moving.any():
            return None
        x, c, q, key = (
            a[rows][moving] for a in (moment.inputs, moment.cortex, moment.address, moment.key)
        )
        everyone = bool(moving.all())
        keys = keys if everyone else keys[moving]
        rate = self.address
        # finite inputs can still overflow the arithmetic below: the check before the step is
        # kept refuses what is not finite
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = score[moving] / norm[moving, None] / self.address_temperature
            if masked:
                held = held if everyone else held[moving]
                logits = np.where(held, logits, -np.inf)
            # among the moments the address attended to: the attention is the prior
            if read is None:
                streams = np.asarray(rows)[moving]
                attention = self._attention(
                    q, keys, held if masked else None,
                    self.position_term(head, x, c, streams),
                )
            else:  # a competing head's: the shares it read
                attention = np.log(read[rows][moving])
                alive = np.isfinite(attention).any(axis=1)
                if not alive.any():
                    return None
                if not alive.all():  # the heads before took this row's slots whole
                    moving[np.flatnonzero(moving)[~alive]] = False
                    x, c, q, key, keys, logits, attention = (
                        a[alive] for a in (x, c, q, key, keys, logits, attention)
                    )
                    if masked:
                        held = held[alive]
            attention -= attention.max(axis=1, keepdims=True)
            attention -= np.log(np.exp(attention).sum(axis=1, keepdims=True))
            logits = logits + attention
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            # the gradient of the content logits toward the named slots: the named keys less
            # the attended ones, zero where the attention is what the lesson names
            gradient = p - np.where(np.isfinite(attention), np.exp(attention), 0.0)
            error = _einsum("bm,bmk->bk", gradient, keys)
            # each row's gain on its own address, rate * |z|^2, is at most one: a row taught
            # alone moves its address by its error at most, and a row within the bound keeps the
            # rate as it is
            energy = np.einsum("bi,bi->b", x, x) + np.einsum("bi,bi->b", c, c) + 1.0
            step = error * np.minimum(1.0, 1.0 / (rate * energy))[:, None]
        return moving, x, c, key, error, step, p, attention

    # -- reward trains the address map

    def contrast_rows(
        self, graph: NeuralGraph, plus: BrainState, minus: BrainState
    ) -> np.ndarray | None:
        """The read's eligibility from a sampled act, one ``(inputs, kept columns)`` array per
        row, or None when neither ``address_reward`` nor ``position_reward`` nor ``gain`` trains
        the read: the event's rows of the maps and the columns that can step (``reward_blocks``;
        ``full_eligibility`` lays it out over the rows of ``w_x``, ``w_c`` and ``b`` and over
        every column, with several heads one block of ``address`` columns per head, head 0's
        first, then with ``position_reward`` the position and anchored buckets, and with ``gain``
        one column per port, ``_gain_rows``): the step the address and position lessons would
        take at each row,
        were the act's action a lesson's label, before the average over rows. The
        association's movement between the act's two nudged phases, ``plus`` less ``minus``,
        is read back through the recall and next synapses of ``graph`` and scored against the
        slots of the moment the act read (``last``, before its write); a row contributes
        ``min(address, 1 / |x|^2)`` times the outer product of ``[x, 0, 0]``, the event's rows
        of ``z = [x, c, 1]``, and its error ``K* - q`` (the position rows alike at
        ``position_rate``): the cortex and bias rows learn from lessons alone.
        A row that does not move, or whose contribution is not finite, keeps none."""
        if self._device is not None:
            return cast("np.ndarray | None", self._device.contrast_rows(graph, plus, minus))
        positions = bool(self.position_reward) and bool(self.position_maps)
        if not (self.address_reward or positions or self.gain):
            return None
        movement = self.activity(plus) - self.activity(minus)
        batch, width, columns_ = len(movement), self.map_width, self.address_width
        count = self.position_buckets
        if self.address_reward:
            self._address_held = True
        out = np.zeros((batch, width, sum(block[1] for block in self.reward_blocks)))
        moment = self.last
        if moment is None or not self.slots or self.streams != batch or len(moment.key) != batch:
            return out
        if self.gain:  # the gains' step, in their own columns
            self._gain_rows(out, moment, plus, minus)
        if not (self.address_reward or positions):
            return out
        reads = self._lesson_reads(moment)
        for h in range(self.heads):
            d_value, d_next = self._read_backs(graph, movement, h)
            head_moment = moment._replace(address=self._head_address(moment, h))
            found = self._row_steps(head_moment, np.arange(batch), d_value, d_next, h, reads[h])
            if found is None:
                continue
            moving, x, _, _, error, _, target, prior = found
            index = np.flatnonzero(moving)
            # the event's rows of ``z = [x, c, 1]`` step, each at a rate bounded by the
            # event's energy; the cortex and bias rows learn from lessons alone
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                energy = np.einsum("bi,bi->b", x, x)
                if self.address_reward:
                    step = error * np.minimum(1.0, 1.0 / (self.address * energy))[:, None]
                    rows = self.address * np.einsum("bi,bk->bik", x, step)
                    kept = np.isfinite(rows).all(axis=(1, 2))
                    out[index[kept], :, h * columns_:(h + 1) * columns_] = rows[kept]
            if not positions:
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                # the position lesson's step at each row, as ``_position_step`` takes it, on
                # the event's rows
                gradient = np.where(np.isfinite(prior), target - np.exp(prior), 0.0)
                # at the founder's lesson rate, whatever rate the lessons take: a slow table
                # taught by lessons does not slow what reward teaches
                rate = self.REWARD_RATE
                bound = np.minimum(1.0, 1.0 / (rate * energy))[:, None]
                blocks = [(self.buckets()[None, :], count)]
                if self.anchor_maps:
                    blocks.append((self.anchored_buckets(index), count + 1))
                start = self.heads * columns_ if self._address_held else 0
                for buckets, size in blocks:
                    terms = _per_bucket(gradient, buckets, size)
                    rows = rate * np.einsum("bi,bk->bik", x, terms * bound)
                    kept = np.isfinite(rows).all(axis=(1, 2))
                    out[index[kept], :, start + h * size:start + (h + 1) * size] = rows[kept]
                    start += self.heads * size
        return out

    @property
    def reward_blocks(self) -> list[tuple[int, int]]:
        """The columns of the reward eligibility that can hold anything, as ``(start, width)`` of
        ``reward_columns``: the address's once ``address_reward`` wrote them, with
        ``position_reward`` the position's and the anchored ones, and with ``gain`` the gains'.
        The other columns stay zero, as do the cortex and bias rows, so the actor keeps these
        alone, over the event's rows (``full_eligibility``)."""
        columns_ = self.heads * self.address_width
        blocks = [(0, columns_)] if self._address_held or self.address_reward else []
        start = columns_
        if self.position_reward and self.position_buckets:
            size = self.heads * self.position_buckets
            blocks.append((start, size))
            start += size
            if self.anchor:
                size = self.heads * (self.position_buckets + 1)
                blocks.append((start, size))
                start += size
        if len(self.gains):
            blocks.append((start, len(self.gains)))
        return blocks

    def full_eligibility(self, kept: np.ndarray) -> np.ndarray:
        """A reward eligibility laid out as a checkpoint keeps it, ``(batch, inputs + association
        + 1, reward_columns)``: the rows of ``w_x``, ``w_c`` and ``b``, every column."""
        if not isinstance(kept, np.ndarray):
            kept = _host_array(kept)
        out = np.zeros((len(kept), self.map_width + len(self.association) + 1, self.reward_columns))
        at = 0
        for start, width in self.reward_blocks:
            out[:, : self.map_width, start : start + width] = kept[:, :, at : at + width]
            at += width
        return out

    def kept_eligibility(self, full: np.ndarray) -> np.ndarray:
        """What the actor keeps of a reward eligibility laid out in full, a row's or a batch's."""
        parts = [
            full[..., : self.map_width, start : start + width]
            for start, width in self.reward_blocks
        ]
        return np.concatenate(parts, axis=-1)

    def reblocked_eligibility(
        self, kept: np.ndarray, before: list[tuple[int, int]]
    ) -> np.ndarray:
        """What the actor keeps once the columns that can step changed from the blocks ``before``:
        a column held before keeps its numbers, and a new one holds nothing. No full layout: at
        the speech size and 256 streams that is 34 GiB."""
        after = self.reward_blocks
        if after == before:
            return kept
        held, at = {}, 0
        for start, width in before:
            held[start, width] = at
            at += width
        out = np.zeros((*kept.shape[:-1], sum(width for _, width in after)))
        at = 0
        for start, width in after:
            source = held.get((start, width))
            if source is not None:
                out[..., at : at + width] = kept[..., source : source + width]
            at += width
        return out

    def saved_eligibility(self, name: str, kept: Any) -> dict[str, np.ndarray]:
        """A reward eligibility as a checkpoint holds it: laid out in full while that stays small,
        and past ``_SAVED_IN_FULL`` entries what the actor keeps, under ``name + "_kept"``."""
        whole = len(kept) * (self.map_width + len(self.association) + 1) * self.reward_columns
        if whole <= _SAVED_IN_FULL:
            return {name: self.full_eligibility(kept)}
        return {name + "_kept": kept if isinstance(kept, np.ndarray) else _host_array(kept)}

    def took_eligibility(self, *saved: np.ndarray) -> None:
        """A checkpoint's reward eligibilities, laid out in full, are about to be kept: the
        address columns count as written where they hold anything."""
        if any(full[..., : self.heads * self.address_width].any() for full in saved):
            self._address_held = True

    def took_kept_eligibility(self, *saved: np.ndarray) -> None:
        """The same of those a checkpoint holds as the actor keeps them: every column there means
        the address columns were written."""
        if any(kept.shape[-1] == self.reward_columns for kept in saved):
            self._address_held = True

    @property
    def reward_columns(self) -> int:
        """The columns of the reward eligibility (``contrast_rows``): every head's address
        columns, and with ``position_reward`` every head's position buckets, then with
        ``anchor`` every head's anchored buckets, and last with ``gain`` one column per read
        port."""
        columns_ = self.heads * self.address_width
        if self.position_reward and self.position_buckets:
            columns_ += self.heads * self.position_buckets
            if self.anchor:
                columns_ += self.heads * (self.position_buckets + 1)
        return columns_ + len(self.gains)

    def propose(self, step: np.ndarray) -> tuple[Any, ...] | None:
        """The address map after a reward's step, not installed: ``step`` is the rows'
        eligibility weighted by the dopamine and averaged (``contrast_rows``' shape), taken at
        ``address_reward``. Returns ``w_x``, ``w_c``, ``b`` and the norm of the step (with
        several heads, each of the three a tuple of every head's, head 0's first), then with
        ``position_reward`` every head's position maps and, with ``anchor``, its anchored maps
        (None without), with ``gain`` last the gains (a further item, none without), or None when
        nothing reward trains is there; a step whose map is not finite is refused
        (``ValueError``)."""
        if self._device is not None:
            return self._device.propose(step)
        positions = bool(self.position_reward) and bool(self.position_maps)
        if not (self.address_reward or positions or self.gain):
            return None
        step = np.asarray(step, dtype=float)
        width, columns_ = self.map_width, self.address_width
        shape = (width + len(self.association) + 1, self.reward_columns)
        if step.shape == shape:
            step = self.kept_eligibility(step)
        elif step.shape != (width, sum(block[1] for block in self.reward_blocks)):
            raise ValueError(f"an address map's step is a {shape} array")
        gains: np.ndarray | None = None
        moved_gains: list[np.ndarray] = []
        if self.gain:  # the gains, stepped on their own columns of the first row
            moved_gains = [step[0, step.shape[1] - len(self.gains):]]
            with np.errstate(over="ignore", invalid="ignore"):
                gains = self.gains + moved_gains[0]
        split = self.heads * columns_ if self._address_held or self.address_reward else 0
        rest = len(self.association) + 1  # the cortex and bias rows, which hold nothing
        full_split = self.heads * columns_
        # a whole row's zeros: every part stepped, in the order of the columns
        parts: list[np.ndarray | int] = []
        maps = self.head_maps
        with np.errstate(over="ignore", invalid="ignore"):
            if self.address_reward:
                moved = self.address_reward * step[:, :split]
                maps = [
                    (kept[0] + moved[:, h * columns_:(h + 1) * columns_], kept[1], kept[2])
                    for h, kept in enumerate(self.head_maps)
                ]
                parts += [moved, rest * full_split]
            else:
                parts.append((width + rest) * full_split)
            stepped: list[np.ndarray] = [] if not self.address_reward else [moved]
            position_maps = anchor_maps = None
            if positions:
                count, start = self.position_buckets, split
                kinds = [(self.position_maps, count)]
                if self.anchor_maps:
                    kinds.append((self.anchor_maps, count + 1))
                found_maps = []
                for kept_maps, size in kinds:
                    part = self.position_reward * step[:, start:start + self.heads * size]
                    stepped.append(part)
                    parts += [part, rest * self.heads * size]
                    found_maps.append([
                        (kept[0] + part[:, h * size:(h + 1) * size], kept[1], kept[2])
                        for h, kept in enumerate(kept_maps)
                    ])
                    start += self.heads * size
                position_maps = found_maps[0]
                anchor_maps = found_maps[1] if len(found_maps) > 1 else None
        parts += moved_gains
        stepped += moved_gains
        every = [*(head[0] for head in maps), *stepped] if self.address_reward else [*stepped]
        if gains is not None:
            every.append(gains)
        for found in (position_maps, anchor_maps):
            every += [head[0] for head in found or ()]
        if not all(np.isfinite(a).all() for a in every):
            raise ValueError("the address map's reward step must be finite")
        # one norm over every part stepped, as of the whole eligibility: with the address map
        # alone, 0.92.1's own
        norm = _laid_out_norm(parts)
        proposal: tuple[Any, ...] = (
            (maps[0][0], maps[0][1], maps[0][2], norm, position_maps, anchor_maps)
            if self.heads == 1
            else (
                tuple(m[0] for m in maps), tuple(m[1] for m in maps), tuple(m[2] for m in maps),
                norm, position_maps, anchor_maps,
            )
        )
        return proposal if gains is None else (*proposal, gains)

    def accept(self, proposal: tuple[Any, ...]) -> None:
        """Install a step ``propose`` made: the address maps, with ``position_reward`` the
        position maps (and the anchored ones), and with ``gain`` the gains."""
        if self._device is not None:
            self._device.accept(proposal)
            return
        heads = [proposal[:3]] if self.heads == 1 else zip(*proposal[:3], strict=True)
        self.head_maps = [(w_x, w_c, b) for w_x, w_c, b in heads]
        if len(proposal) > 4 and proposal[4] is not None:
            self.position_maps = list(proposal[4])
        if len(proposal) > 5 and proposal[5] is not None:
            self.anchor_maps = list(proposal[5])
        if len(proposal) > 6:
            self.gains = proposal[6]

    def to_dict(self) -> dict[str, Any]:
        """The memory's settings and counts: the keys always written (``SAVED``), the ports,
        and each option where it differs from its founder."""
        out = {
            **{name: getattr(self, name) for name in self.SAVED},
            "source": self.source,
            "target": self.target,
            "neurons": self.neurons,
            "writes": int(self.writes),
        }
        if self.inputs:  # only a memory keyed on a code of the observation records its width
            out["inputs"] = self.inputs
        # an option at its founder is left out of the description
        founders = self.options()
        out.update(
            {name: getattr(self, name) for name in self.OPTIONS
             if getattr(self, name) != founders[name]}
        )
        if self.heads > 1:
            # always written with several heads: a file saved before heads competed holds none,
            # and its heads read alike, a law this version refuses
            out["compete"] = True
        for name in ("tokens", "grid"):  # JSON-safe: a list, as a checkpoint keeps it
            if name in out:
                out[name] = list(out[name])
        if self.drive_gain:  # the learned multiplier on the amplitude, read here and in the gains
            out["hearing"] = self.hearing
        return out


@dataclass
class PatternSeparator:
    """Pattern separation: expand a key into a wider random code and keep the strongest winners.

    ``inputs`` key neurons project through a fixed random matrix onto ``expansion`` code
    neurons; the ``winners`` largest positive entries stay and the rest are zero. Expansion
    can reduce overlap between correlated keys; it does not guarantee separate codes.
    Codes with disjoint winners do not interfere: a delta write changes another key's read
    in proportion to their dot product. ``center`` > 0 keeps a running mean of the keys
    seen by ``observe`` (forgetting factor ``center``) and subtracts it first. Updating that
    mean can move the code of a previously stored key.

    ``fan_in`` above zero makes the projection sparse: each code neuron reads ``fan_in``
    distinct inputs chosen at random from the seed, with random weights of variance
    ``1 / fan_in``. ``index`` and ``weight``, ``(expansion, fan_in)`` each, hold them in place
    of ``projection``, so the separator's storage and work follow the expansion and the
    fan-in, whatever the number of inputs. The founder ``fan_in=0`` keeps the dense
    ``projection``; ``index`` and ``weight`` are then None, and ``projection`` is None for a
    sparse separator.

    ``cover`` above zero (with a fan-in) reads every input in both signs, through two
    rectified channels: ON ``max(x, 0)``, channel ``i`` of input ``i``, and OFF
    ``max(-x, 0)``, channel ``inputs + i`` (``channels`` is ``2 * inputs``, and ``index``
    numbers channels). Each channel is given ``cover`` distinct code neurons first, each
    code neuron's remaining fan-in is then filled with distinct channels at random, and the
    weights are positive, uniform in ``(0, 1]`` over ``sqrt(fan_in)``. Without ``center``, a
    single input of either sign, the others zero, drives each reader of its channel above
    zero, so it never codes to zero, and a nonnegative key leaves every OFF channel silent.
    It needs ``fan_in`` within the channels, ``cover * 2 * inputs`` within
    ``expansion * fan_in`` and ``cover`` within ``expansion``. The founder ``cover=0`` draws
    the signed sparse weights over the inputs independently, as above.

    The projection is drawn once from the seed and never learns. A brain's ``code=units``
    observation code and its episodic separators are such fixed random codes; a brain's learned
    code is the region ``learned_code=units`` adds (``cadence.code.LearnedCode``).
    """

    inputs: int
    expansion: int
    winners: int
    seed: int = 0
    center: float = 0.0
    fan_in: int = 0
    cover: int = 0
    projection: np.ndarray | None = field(init=False, repr=False)
    index: np.ndarray | None = field(init=False, repr=False)
    weight: np.ndarray | None = field(init=False, repr=False)
    mean: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("inputs", "expansion", "winners"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        fan_in = self.fan_in
        for name in ("fan_in", "cover"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if fan_in > self.channels:
            raise ValueError(
                "fan_in must not exceed the channels, two per input"
                if self.cover
                else "fan_in must not exceed the inputs"
            )
        if self.winners > self.expansion:
            raise ValueError("winners must not exceed the expansion")
        if not 0 <= self.center < 1:
            raise ValueError("center lies in [0, 1)")
        if self.cover and (
            not fan_in
            or self.cover > self.expansion
            or self.cover * self.channels > self.expansion * fan_in
        ):
            raise ValueError(
                "cover needs a fan-in, at most the expansion, and cover * 2 * inputs (the "
                "channels) within expansion * fan_in"
            )
        rng = np.random.default_rng(self.seed)
        self.projection = self.index = self.weight = None
        if fan_in and self.cover:
            self.index = self._covering(rng)
            self.weight = (1.0 - rng.random((self.expansion, fan_in))) / np.sqrt(fan_in)
        elif fan_in:
            # each code neuron's inputs, distinct and in order, then their weights
            self.index = np.sort(
                np.stack([rng.choice(self.inputs, fan_in, replace=False)
                          for _ in range(self.expansion)]),
                axis=1,
            ).astype(np.int64)
            self.weight = rng.standard_normal((self.expansion, fan_in)) / np.sqrt(fan_in)
        else:
            self.projection = (
                rng.standard_normal((self.inputs, self.expansion)) / np.sqrt(self.inputs)
            )
        self.mean = np.zeros(self.inputs)

    @property
    def channels(self) -> int:
        """The values a code neuron's fan-in is drawn from: the inputs, or with ``cover`` the
        ON and the OFF channel of each input, ``2 * inputs``."""
        return 2 * int(self.inputs) if self.cover else int(self.inputs)

    def _covering(self, rng: np.random.Generator) -> np.ndarray:
        """Each code neuron's channels, distinct and in order, every channel among those of
        ``cover`` distinct code neurons: the readers of the channels are dealt first, from the
        places of every neuron's fan-in in a random order, and each neuron's places left over
        take distinct other channels at random."""
        channels, units, fan_in, cover = self.channels, self.expansion, self.fan_in, self.cover
        places = rng.permutation(np.repeat(np.arange(units), fan_in))
        readers = places[: cover * channels].reshape(channels, cover).copy()
        spare = places[cover * channels:].copy()
        ordered = np.sort(readers, axis=1)
        for row in np.flatnonzero((ordered[:, 1:] == ordered[:, :-1]).any(axis=1)):
            # a neuron dealt twice to one channel trades the place for another neuron's
            given = readers[row]
            for k in range(1, cover):
                for _ in range(64 * (channels + len(spare))):
                    if given[k] not in given[:k]:
                        break
                    open_ = np.flatnonzero(~np.isin(spare, given))
                    if len(open_):
                        s = open_[rng.integers(len(open_))]
                        given[k], spare[s] = spare[s], given[k]
                        continue
                    other, place = rng.integers(channels), rng.integers(cover)
                    if readers[other, place] not in given and given[k] not in readers[other]:
                        given[k], readers[other, place] = readers[other, place], given[k]
                else:
                    raise ValueError("the channels could not be covered")
        owner = readers.ravel()
        order = np.argsort(owner, kind="stable")  # each neuron's given channels, in order
        read = np.repeat(np.arange(channels), cover)[order]
        bounds = np.searchsorted(owner[order], np.arange(units + 1))
        index = np.empty((units, fan_in), dtype=np.int64)
        for unit in range(units):
            given = read[bounds[unit] : bounds[unit + 1]]
            # the remaining places: distinct channels the neuron does not read yet, by rank
            ranks = rng.choice(channels - len(given), fan_in - len(given), replace=False)
            extra = ranks + np.searchsorted(given - np.arange(len(given)), ranks, side="right")
            index[unit] = np.sort(np.concatenate([given, extra]))
        return index

    def habituate(self, keys: np.ndarray) -> None:
        """Initialize the running mean from ``keys`` before storing records.

        With positive ``center``, later learning calls continue updating this mean;
        initialization does not freeze the codes used to address stored records.
        """
        x = np.asarray(keys, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all() or not len(x):
            raise ValueError(f"keys must be a finite nonempty (batch, {self.inputs}) array")
        self.mean = (x / len(x)).sum(axis=0)

    def code(self, key: np.ndarray, learn: bool = False) -> np.ndarray:
        """The sparse code of ``(batch, inputs)`` keys: ``(batch, expansion)``, ``winners``
        nonzero entries per row at most. ``learn`` updates the running mean (writes only).
        With ``cover`` each key is read through its ON and its OFF channels."""
        x = np.asarray(key, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"key must be a finite (batch, {self.inputs}) array")
        mean = self.mean
        if self.center > 0.0:
            if learn and len(x):
                mean = self.center * mean + (1.0 - self.center) * (x / len(x)).sum(axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                x = x - mean
        with np.errstate(over="ignore", invalid="ignore"):
            if self.fan_in:
                assert self.index is not None and self.weight is not None
                if self.cover:  # the ON channels, then the OFF channels
                    x = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=1)
                drive = np.maximum(np.einsum("bek,ek->be", x[:, self.index], self.weight), 0.0)
            else:
                assert self.projection is not None
                drive = np.maximum(x @ self.projection, 0.0)
        if not np.isfinite(mean).all() or not np.isfinite(drive).all():
            raise ValueError("pattern separation overflowed; scale the input keys")
        if learn:
            self.mean = mean
        if self.winners >= self.expansion:
            return np.asarray(drive)
        keep = np.argpartition(-drive, self.winners - 1, axis=1)[:, : self.winners]
        out: np.ndarray = np.zeros_like(drive)
        np.put_along_axis(out, keep, np.take_along_axis(drive, keep, axis=1), axis=1)
        return np.asarray(out)

    def arrays(self) -> dict[str, np.ndarray]:
        """The separator's state by name, as a checkpoint holds it: ``projection`` and ``mean``,
        or with a fan-in ``index``, ``weight`` and ``mean``."""
        if self.fan_in:
            assert self.index is not None and self.weight is not None
            return {"index": self.index, "weight": self.weight, "mean": self.mean}
        assert self.projection is not None
        return {"projection": self.projection, "mean": self.mean}

    def to_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "inputs": self.inputs,
            "expansion": self.expansion,
            "winners": self.winners,
            "seed": self.seed,
            "center": self.center,
        }
        if self.fan_in:  # a dense separator leaves fan_in out of its description
            out["fan_in"] = int(self.fan_in)
        if self.cover:  # and a separator without cover leaves cover out
            out["cover"] = int(self.cover)
        return out


@dataclass
class FastSynapses:
    """A bounded associative memory between two ranges, one matrix per stream.

    ``rule="hebb"`` (the default) adds an outer product.
    ``rule="delta"`` reads before writing: for a unit key ``k``,
    ``M += rate * outer(k, value - k @ M)``. Repeated evidence then stops changing
    a correct prediction. A rate-one write corrects this key's unscaled prediction; other
    nonorthogonal keys can interfere. This is the delta/LMS rule. Its capacity is bounded,
    and its addressing is fixed: the key itself, or the separator's code of it.

    ``observe``/``recall`` operate directly on key/value ports. ``update``/``read``
    adapt the same operations to settled states and full drives. Key normalisation
    reads the whole key vector; a synapse uses its key coordinate and the post neuron's
    prediction error. Strengths decay once per observation/update, never on reads.
    """

    pre: np.ndarray
    post: np.ndarray
    decay: float = 1.0
    rate: float = 1.0
    amplitude: float = 1.0
    normalize: bool = False  # unit keys and cue, the read divided by the decayed count of writes
    replace: bool = False  # a write clears what its active pre neurons held (a slot; one-hot keys)
    rule: str = "hebb"  # "delta": unit keys, residual writes, no count-averaged read
    separator: PatternSeparator | None = None  # pattern separation of the keys before use
    strength: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)  # (batch,) the decayed count of writes, for ``normalize``
    writes: int = 0

    def __post_init__(self) -> None:
        self._device: _StoreOnDevice | None = None  # what rests on a device (``reside``)
        for name in ("pre", "post"):
            raw = np.asarray(getattr(self, name))
            if (
                raw.dtype.kind not in "iuf"
                or not np.isfinite(raw).all()
                or (raw != np.floor(raw)).any()
            ):
                raise ValueError(f"{name} must contain integer neuron indices")
        self.pre = np.asarray(self.pre, dtype=np.int64)
        self.post = np.asarray(self.post, dtype=np.int64)
        for name, neurons in (("pre", self.pre), ("post", self.post)):
            if neurons.ndim != 1 or not len(neurons) or np.any(neurons < 0):
                raise ValueError(f"{name} must be a nonempty vector of neuron indices")
            if len(np.unique(neurons)) != len(neurons):
                raise ValueError(f"{name} neuron indices must be unique")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay lies in [0, 1]")
        if not np.isfinite(self.rate) or self.rate < 0 or not np.isfinite(self.amplitude):
            raise ValueError("rate must be finite and nonnegative; amplitude must be finite")
        if self.rule not in ("hebb", "delta"):
            raise ValueError("rule must be 'hebb' or 'delta'")
        if self.rule == "delta":
            if self.rate > 1:
                raise ValueError("delta rate lies in [0, 1]")
            if self.normalize or self.replace:
                raise ValueError(
                    "delta already normalises keys; normalize/replace are Hebbian modes"
                )
        if self.separator is not None and self.separator.inputs != len(self.pre):
            raise ValueError("the separator's inputs must equal the number of pre neurons")
        self._pre_columns = columns(self.pre)
        self._post_columns = columns(self.post)
        self.strength = np.zeros((0, self.key_width, len(self.post)))
        self.mass = np.zeros(0)

    def __getattr__(self, name: str) -> Any:
        """An array resting on the device, read on the host: they all come back."""
        if self.__dict__.get("_device") is not None and name in _StoreOnDevice.OWNED:
            self.leave_device()
            return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getstate__(self) -> dict[str, Any]:
        self.leave_device()
        return self.__dict__

    def reside(self, kernel: Any) -> None:
        """The strengths on the device of a brain's ``kernel``; only a consolidating store has
        the device's path."""
        self.leave_device()

    def leave_device(self) -> None:
        held = self.__dict__.get("_device")
        if held is not None:
            held.give_back()
            self._device = None

    def held_values(self) -> dict[str, Any]:
        """What a refused outcome restores (``take_values``): a write replaces the arrays."""
        held: Any = self._device or self
        return {"strength": held.strength, "mass": held.mass, "writes": self.writes}

    def take_values(self, values: Mapping[str, Any]) -> None:
        """Return to ``held_values``."""
        held = self._device
        for name, value in values.items():
            if held is not None and name in _StoreOnDevice.OWNED:
                setattr(held, name, value if _resident(value) else held.xp.here(value))
            else:
                setattr(self, name, _on_host(value))

    @property
    def key_width(self) -> int:
        """Rows of the strength matrix: the code width with a separator, else the pre count."""
        return self.separator.expansion if self.separator is not None else int(len(self.pre))

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Clear all streams, or only the selected streams at episode boundaries."""
        if rows is None or len(self.strength) != batch:
            self.strength = np.zeros((batch, self.key_width, len(self.post)))
            self.mass = np.zeros(batch)
        else:
            self.strength[rows] = 0.0
            self.mass[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        self.strength = self.strength[rows]
        self.mass = self.mass[rows]

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        # The Hebbian normalisation, with a small-norm floor.
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))

    @staticmethod
    def _delta_unit(x: np.ndarray) -> np.ndarray:
        # Scale first so even very small/large finite keys have a well-defined direction.
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
        scaled = x / np.where(scale > 0, scale, 1.0)
        norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
        return np.asarray(scaled / np.where(norm > 0, norm, 1.0))

    @staticmethod
    def _port(values: np.ndarray, width: int, name: str) -> np.ndarray:
        out = np.asarray(values, dtype=float)
        if out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all():
            raise ValueError(f"{name} must be a finite (batch, {width}) array")
        return out

    def recall(self, key: np.ndarray) -> np.ndarray:
        """Read ``(batch, pre)`` keys as ``(batch, post)`` values without changing memory.

        A different batch size reads the baseline without erasing live streams. Delta uses
        unit keys and does not divide by the number of writes. ``amplitude``
        scales the returned drive, not the residual used when learning a value.
        """
        cue = self._port(key, len(self.pre), "key")
        same_batch = len(self.strength) == len(cue)
        strength = self.strength if same_batch else self._baseline()[None, :, :]
        if self.separator is not None:
            cue = self.separator.code(cue)
        if self.rule == "delta":
            cue = self._delta_unit(cue)
        elif self.normalize:
            cue = self._unit(cue)
        out = (cue[:, None, :] @ strength)[:, 0, :]
        if self.normalize:
            mass = self.mass if same_batch else np.zeros(len(cue))
            out = out / np.maximum(mass, 1e-12)[:, None]
        return np.asarray(self.amplitude * out)

    def _baseline(self) -> np.ndarray:
        """The unobserved stream's readback, without changing any stored stream."""
        return np.zeros((self.key_width, len(self.post)))

    def observe(self, key: np.ndarray, value: np.ndarray, write: np.ndarray | None = None) -> None:
        """Decay once, then write selected rows from the key and observed-value ports.

        Inputs are ``(batch, pre)`` and ``(batch, post)``. ``write=None`` writes
        every row; a boolean mask gates individual streams. A zero key cannot
        write a delta association. Validation happens before any state change.
        For prediction tasks call ``recall`` before revealing the new value.
        """
        key = self._port(key, len(self.pre), "key")
        value = self._port(value, len(self.post), "value")
        batch = len(key)
        if len(value) != batch:
            raise ValueError("key and value batches must match")
        gate = np.ones(batch, dtype=bool) if write is None else np.asarray(write)
        if gate.shape != (batch,) or gate.dtype != np.bool_:
            raise ValueError("write must be a boolean vector with one entry per stream")
        same_batch = len(self.strength) == batch
        strength = (
            self.strength * self.decay
            if same_batch
            else np.zeros((batch, self.key_width, len(self.post)))
        )
        mass = self.mass * self.decay if same_batch else np.zeros(batch)
        if self.rule == "delta":
            gate = gate & np.any(key != 0.0, axis=1)
        rows = np.flatnonzero(gate)
        if not len(rows):
            self.strength, self.mass = strength, mass
            return
        a, b = key[rows], value[rows]
        previous_mean = None if self.separator is None else self.separator.mean
        if self.separator is not None:
            a = self.separator.code(a, learn=True)
        if self.rule == "delta":
            a = self._delta_unit(a)
        elif self.normalize:
            a = self._unit(a)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.rule == "delta":
                b = b - (a[:, None, :] @ strength[rows])[:, 0, :]
            elif self.replace:
                strength[rows] *= (a <= 0.0)[:, :, None]
            strength[rows] += self.rate * a[:, :, None] * b[:, None, :]
            mass[rows] += self.rate
        if not np.isfinite(strength).all() or not np.isfinite(mass).all():
            if self.separator is not None and previous_mean is not None:
                self.separator.mean = previous_mean
            raise ValueError("synaptic update overflowed; scale the observed keys and values")
        self.strength, self.mass = strength, mass
        self.writes += len(rows)

    def read(self, drive: np.ndarray) -> np.ndarray:
        """The post neurons' drive from the pre range's stimulus in ``drive``: ``(batch, post)``.

        Normalized Hebbian mode uses unit keys and cue, then divides the read by the
        decayed count of writes. Cosine weights can be signed, so this need not be a
        convex average. ``rate`` and ``amplitude`` also scale the returned values.
        """
        return self.recall(drive[:, self._pre_columns])

    def stimulate(
        self, drive: np.ndarray, inplace: bool = False, key: np.ndarray | None = None
    ) -> np.ndarray:
        """Add the read to the post columns of ``drive`` (a copy, unless ``inplace``): the read
        of the pre columns of ``drive``, or with ``key`` the read of that ``(batch, pre)``
        key, for a memory keyed on a code of the observation rather than on neurons."""
        if self._device is not None and _resident(drive):
            return cast(np.ndarray, self._device.stimulate(drive, key))
        drive = _on_host(drive)
        out = drive if inplace else np.array(drive, dtype=float)
        out[:, self._post_columns] += self.read(out) if key is None else self.recall(key)
        return out

    # -- the memory contract (``cadence.memories``)

    def stimulus(self, drive: np.ndarray, moment: Any) -> np.ndarray:
        """The read added to a drive copy: under the moment's ``place`` key, or of the drive
        itself without one; nothing where the stores do not read into the drive."""
        if not moment.stores:
            return drive
        if moment.place is None:  # an instrument may wrap the drive's own read
            return self.stimulate(drive)
        return self.stimulate(drive, key=moment.place)

    def branch(self) -> FastSynapses:
        """An imagined branch reads the records and writes none: the memory itself."""
        return self

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays a checkpoint keeps: the ports, strengths, masses and the separator's
        tables."""
        out = {name: getattr(self, name) for name in ("pre", "post", "strength", "mass")}
        if self.separator is not None:
            out.update({"separator/" + k: v for k, v in self.separator.arrays().items()})
        return out

    def update(
        self,
        state: BrainState,
        write: np.ndarray | None = None,
        post: np.ndarray | None = None,
    ) -> None:
        """After settling: strengths fade once; the rows in ``write`` store the settled pre
        activity as the key and the post activity as the value (``observe``). With
        ``write=None`` no row writes.

        ``post`` replaces the post neurons' activations for the write, ``(batch, post)``: what
        actually followed (the next symbols read) rather than what the net settled on.
        """
        s = np.atleast_2d(state.activation)
        self.observe(
            s[:, self._pre_columns],
            s[:, self._post_columns] if post is None else post,
            np.zeros(len(s), dtype=bool) if write is None else write,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre": int(len(self.pre)),
            "post": int(len(self.post)),
            "decay": self.decay,
            "rate": self.rate,
            "amplitude": self.amplitude,
            "normalize": self.normalize,
            "replace": self.replace,
            "rule": self.rule,
            "separator": None if self.separator is None else self.separator.to_dict(),
            "writes": self.writes,
        }
