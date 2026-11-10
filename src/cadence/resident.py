"""A brain's memories on its device: one form, two devices.

The recall memory, the working trace and the consolidating store have a host path in
``cadence.stream`` and ``cadence.memory``, in NumPy, and this device path: the same equations,
written once against a small array namespace that a torch kernel (``TorchArrays``) and an MLX kernel
(``MlxArrays``) each fill (``arrays_of``). What the device path reads directly is what both array
types have: slicing, ``a[key] = v``, ``@``, ``.T``, arithmetic, ``.sum(axis=, keepdims=)``,
``.mean(axis=)``, ``.any(axis=)``, ``.all()``, ``.abs()``, ``.shape`` and ``len``.

MLX is lazy: a moment's reads join the launch that settles on them, and ``read`` is the one place
the device path waits for numbers. A slot is written in place in unified memory (``write_row``), one
slot's cost where a functional update would copy the bank. Torch runs as it is written.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np

from .brain import BrainState, _host_copy, _mlx_probe, _MlxState, _resident

__all__: list[str] = []

# The memories of a moment of ``streams * width * (width + _RESIDENT_ROW)`` from this many rest on
# CUDA: below it each operation's launch costs more than the host's arithmetic (docs/backends.md,
# read on an A10G from 12 to 1,024 inputs and 1 to 256 streams).
_RESIDENT_WORK = 3_000_000
_RESIDENT_ROW = 1024
# The same on MLX, read on an Apple M4 from 12 to 3,920 inputs and 1 to 1,024 streams: at 35 inputs
# and 256 streams, 128 and 64, 1,024 and 4 the device's lesson and step come to the host's.
_MLX_RESIDENT_WORK = 8_000_000
_MLX_RESIDENT_ROW = 1024


def _memory_on_device(kernel: Any, streams: int, width: int) -> bool:
    """Whether a brain's memories rest on its device at ``streams`` of keys of ``width``. On CUDA
    and on MLX from the size that pays for it. ``CADENCE_TORCH_MEMORY=1`` places them at any size
    on any torch device but MPS (float32 alone), ``CADENCE_MLX_MEMORY=1`` on MLX at any size, and
    ``=0`` keeps them on the host."""
    if getattr(kernel, "lives", 0):  # lives keep their critic and memories on their device
        return True
    if hasattr(kernel, "mx"):
        choice = os.environ.get("CADENCE_MLX_MEMORY", "")
        if choice == "0" or kernel.plain:
            return False
        return choice == "1" or (
            streams * width * (width + _MLX_RESIDENT_ROW) >= _MLX_RESIDENT_WORK
        )
    choice = os.environ.get("CADENCE_TORCH_MEMORY", "")
    if choice == "0" or kernel.device.type == "mps":
        return False
    if choice == "1":
        return True
    return (
        kernel.device.type == "cuda"
        and streams * width * (width + _RESIDENT_ROW) >= _RESIDENT_WORK
    )


# -- the backend a composed brain takes when none is named (docs/backends.md, the default backend)

# ``rows * synapses`` from which a device passes the CPU at its default on a composed brain, over
# a moment's ``act``, ``lesson`` and ``step``: in float32 on an A10G, under a compartment, whose
# settle costs the CPU more, and on an Apple M4.
_AUTO_CUDA_WORK = 800_000
_AUTO_CUDA_APICAL = 110_000
_AUTO_MLX_WORK = 800_000
_AUTO_MLX_APICAL = 500_000
# A launch on MLX takes this many settling neurons, and past them a brain runs its compiled sweep,
# which passes one CPU core from about this many rows (one deep anatomy, between 16 and 64).
_MLX_LAUNCH_NEURONS = (8, 2048)
_MLX_LAUNCH_APICAL = 1360
_MLX_SWEEP_ROWS = 32
# Reasoned, not read at the rule's size. Float64 named on CUDA: that device does float64 at a small
# part of its float32 rate. Memories that stay on the host (``Brain._memories_stay_on_host``): a
# moment crosses. A card the table has not read and that has no calibration: a stated margin.
_AUTO_CUDA_FLOAT64 = 4
_AUTO_HOST_MEMORY = 4
_AUTO_UNSEEN = 2
_AUTO_READ_CARDS = ("A10G",)
# A card the table has not read is calibrated once (``_card_calibration``) where a brain's work
# reaches this fraction of the table's crossing: the chamber's default brain is timed on the card
# and on one core at these rows, and where the two cross, in ``rows * synapses``, scales the
# table's crossings by its ratio to the A10G's crossing read by the same calibration.
_CALIBRATION_FROM = 0.25
_CALIBRATION_ROWS = (16, 64, 256)
# The A10G's crossing on the chamber's brain, by the calibration's interpolation of the default
# backend's table (docs/backends.md): 43.7 rows of 13,120 synapses.
_CALIBRATION_A10G = 573_568.0
_CALIBRATIONS: dict[str, dict[str, Any]] = {}  # by card, this process's
_CALIBRATING = False  # a calibration in progress: its own brains name their backends


class _Candidate(NamedTuple):
    """A place a brain can run, and the rule's reason."""

    backend: str
    device: str | None
    precision: str | None
    why: str


def _cuda_card() -> str | None:
    """The CUDA card here, None without one. Torch is imported only when the rule asks."""
    if sys.platform == "darwin" or importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch

        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except Exception:  # a broken install is a machine without the device
        return None


def _mlx_present() -> bool:
    if sys.platform != "darwin" or importlib.util.find_spec("mlx") is None:
        return False
    return _mlx_probe()[0] is not None


def _fastest_backends(
    rows: int, synapses: int, *, settling: int | None = None, apical: bool = False,
    precision: str | None = None, host_memory: bool = False, learned_code: bool = False,
) -> list[_Candidate]:
    """Where a brain of ``synapses`` runs fastest at ``rows``, the fastest first; the compiled CPU
    path ends the list. MPS and other torch devices are not on it: MPS reads slower than MLX and
    the CPU, and the others have no reading. No package is imported below a device's work."""
    work = rows * synapses
    margin = _AUTO_HOST_MEMORY if host_memory else 1
    said = f"{rows} rows of {synapses} synapses"
    out: list[_Candidate] = []
    low, high = _MLX_LAUNCH_NEURONS
    launched = settling is None or low <= settling <= (_MLX_LAUNCH_APICAL if apical else high)
    mlx_work = (_AUTO_MLX_APICAL if apical else _AUTO_MLX_WORK) * margin
    if (
        precision != "float64"
        and (work >= mlx_work if launched else rows >= _MLX_SWEEP_ROWS)
        and _mlx_present()
    ):
        out.append(_Candidate("mlx", None, None, f"{said} are past where MLX passes one CPU core"))
    floats = precision or "float32"
    needed = (_AUTO_CUDA_APICAL if apical else _AUTO_CUDA_WORK) * margin
    if floats == "float64":
        needed *= _AUTO_CUDA_FLOAT64
    if not learned_code and work >= needed * _CALIBRATION_FROM:
        card = _cuda_card()
        if card is not None and not any(name in card for name in _AUTO_READ_CARDS):
            calibration = _card_calibration(card)
            needed *= _AUTO_UNSEEN if calibration is None else calibration["scale"]
        if card is not None and work >= needed:
            out.append(_Candidate(
                "torch", "cuda", floats, f"{said} are past where {card} passes one CPU core"
            ))
    why = "the next fastest" if out else f"one CPU core is the fastest here for {said}"
    out.append(_Candidate("cpu", None, None, why))
    return out


def _crossing(rows: Sequence[int], ratios: Sequence[float]) -> float:
    """The rows at which the device's time over the core's crosses one, by straight lines in the
    logarithms of both between the readings, and past them along the nearest two."""
    x, y = np.log(np.asarray(rows, dtype=float)), np.log(np.asarray(ratios, dtype=float))
    for k in range(len(x) - 1):
        if y[k] >= 0.0 >= y[k + 1] and y[k] != y[k + 1]:
            return float(np.exp(x[k] + (x[k + 1] - x[k]) * y[k] / (y[k] - y[k + 1])))
    k = 0 if y[0] < 0.0 else len(x) - 2
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    if slope >= 0.0:  # the device does not gain on the core with rows: it never passes it
        return float("inf") if y[0] >= 0.0 else float(rows[0])
    return float(np.exp(x[k] - y[k] / slope))


def calibrate_card(rows: Sequence[int] = _CALIBRATION_ROWS) -> dict[str, Any]:
    """Time the chamber's default brain, ``Brain.compose(35, 18)``, on this machine's CUDA card in
    float32 and on one CPU core, at each of ``rows``: a greedy ``act``, a ``lesson`` and a ``step``
    that learns from the outcome of the step before it, two of each after one to warm. Returns the
    card, the device's time over the core's by rows (the geometric mean of the three), the rows
    where they cross, that crossing in ``rows * synapses`` (``work``) and the seconds it took."""
    import time
    import warnings

    from .generic import Brain

    card = _cuda_card()
    if card is None:
        raise RuntimeError("no CUDA card to calibrate")
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    ratios = []
    for count in rows:
        x = rng.random((count, 35)) * (rng.random((count, 35)) < 0.3)
        y = rng.integers(0, 18, count)
        seconds = []
        for on_card in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                brain = Brain.compose(
                    35, 18, seed=0, backend="torch", device="cuda", precision="float32"
                ) if on_card else Brain.compose(35, 18, seed=0, backend="cpu")
                times: list[list[float]] = [[], [], []]
                for t in range(3):
                    for k in range(3):
                        if k == 2:  # a step that learns: the outcome of a sampled action
                            brain.step(x)
                        start = time.perf_counter()
                        if k == 0:
                            brain.act(x, greedy=True)
                        elif k == 1:
                            brain.lesson(x, y)
                        else:
                            brain.step(x, reward=np.ones(count))
                        if t:
                            times[k].append(time.perf_counter() - start)
            seconds.append(np.exp(np.mean([np.log(min(kind)) for kind in times])))
        ratios.append(float(seconds[1] / seconds[0]))
    crossing = _crossing(rows, ratios)
    return {
        "card": card, "rows": list(rows), "ratios": ratios, "crossing": crossing,
        "work": crossing * 13_120, "seconds": time.perf_counter() - started,
    }


def _card_calibration(card: str) -> dict[str, Any] | None:
    """The calibration of ``card``: this process's, the cache's (``CADENCE_PLAN_CACHE``), or made
    here once and kept in both, with ``scale``, the factor on the table's crossings. None where
    ``CADENCE_CALIBRATE=0`` or the calibration fails, and the stated margin holds."""
    held = _CALIBRATIONS.get(card)
    if held is not None:
        return held
    if os.environ.get("CADENCE_CALIBRATE", "") == "0":
        return None
    from .brain import _store_plan, _stored_plan

    name = f"card {card}"
    held = _stored_plan(name)
    if held is None or "work" not in held:
        global _CALIBRATING
        if _CALIBRATING:
            return None
        _CALIBRATING = True
        try:
            held = {"name": name, **calibrate_card()}
        except Exception:  # a calibration that fails is the stated margin's
            return None
        finally:
            _CALIBRATING = False
        _store_plan(held)
    held = {**held, "scale": float(held["work"]) / _CALIBRATION_A10G}
    _CALIBRATIONS[card] = held
    return held


def _backend_of_the_run() -> _Candidate | None:
    """``CADENCE_BACKEND`` names a backend for a whole run: ``cpu``, ``mlx``, ``torch`` or
    ``torch:<device>``, and with ``:float32`` or ``:float64`` after a device its precision."""
    named = os.environ.get("CADENCE_BACKEND", "")
    if not named:
        return None
    backend, device, precision = (*named.split(":"), None, None)[:3]
    if backend not in ("cpu", "torch", "mlx"):
        raise ValueError(f"CADENCE_BACKEND names {named!r}: cpu, torch[:device[:precision]] or mlx")
    return _Candidate(backend, device, precision, "CADENCE_BACKEND names it for the run")


def _host_array(array: Any) -> np.ndarray:
    """A host copy of a resident array, whichever its device."""
    return _host_copy(array) if type(array).__module__ == "mlx.core" else np.array(
        array.detach().cpu().numpy()
    )


def _on_host(drive: Any) -> Any:
    """A drive a memory without a device path reads and writes: the moment's continues on the
    host from it."""
    return _host_array(drive) if _resident(drive) else drive


def _stream_rows(rows: np.ndarray, batch: int) -> np.ndarray:
    """Validate stream indices or a boolean mask before changing any stream state."""
    selected = np.asarray(rows)
    if (
        selected.ndim != 1
        or (selected.size and selected.dtype.kind not in "biu")
        or (selected.dtype.kind == "b" and selected.shape != (batch,))
        or (selected.dtype.kind == "u" and (selected >= batch).any())
    ):
        raise ValueError("rows must be stream indices or a boolean mask matching the batch")
    if not selected.size and selected.dtype.kind != "b":
        return np.zeros(0, dtype=np.int64)
    try:
        return np.asarray(np.arange(batch, dtype=np.int64)[selected], dtype=np.int64)
    except IndexError as exc:
        raise ValueError(
            "rows must be stream indices or a boolean mask matching the batch"
        ) from exc


class _Moment(NamedTuple):
    """The present moment of every stream as the recall memory addresses it."""

    event: np.ndarray  # the observations as given, to recognise the moment at its write
    cortex: np.ndarray  # the settled association activity of the moment before
    address: np.ndarray  # the address q, computed before the settle
    key: np.ndarray  # the fixed part of the address, the slot's key once it is written
    # what the address and position maps read of the moment: the event, and with ``action``
    # the command the stream executed at the moment before
    inputs: np.ndarray


# -- the namespace, per device


class TorchArrays:
    """The array namespace of a torch kernel: floats in the brain's precision, on its device."""

    inf = float("inf")

    def __init__(self, kernel: Any) -> None:
        self.torch = kernel.torch
        self.device, self.dtype = kernel.device, kernel.dtype
        self.staging = kernel._staging
        self.placed: dict[int, tuple[Any, ...]] = {}

    # the arrays' way there and back

    def here(self, array: Any) -> Any:
        return self.torch.from_numpy(np.array(array, dtype=np.float64)).to(self.device, self.dtype)

    def long(self, array: Any) -> Any:
        return self.torch.from_numpy(np.array(array, dtype=np.int64)).to(self.device)

    def flags(self, array: Any) -> Any:
        return self.torch.from_numpy(np.array(array, dtype=bool)).to(self.device)

    def upload(self, array: np.ndarray) -> Any:
        return self.staging.upload(array, self.dtype)

    def floats(self, array: Any) -> Any:
        return array.to(self.dtype)

    def host(self, array: Any) -> np.ndarray:
        return np.array(array.detach().cpu().numpy(), dtype=np.float64)

    def host_long(self, array: Any) -> np.ndarray:
        return np.array(array.cpu().numpy(), dtype=np.int64)

    def host_flags(self, array: Any) -> np.ndarray:
        return np.asarray(array.cpu().numpy())

    def read(self, *scalars: Any, also: Sequence[Any] = ()) -> np.ndarray:
        """One read of several scalars; ``also`` is what MLX makes in the same launch."""
        return np.asarray(self.torch.stack([s.to(self.dtype) for s in scalars]).cpu().numpy())

    def settle(self, *arrays: Any) -> None:
        """MLX makes these now, so a graph does not grow; torch has made them."""

    # new arrays

    def zeros(self, shape: Any) -> Any:
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def zeros_long(self, shape: Any) -> Any:
        return self.torch.zeros(shape, dtype=self.torch.int64, device=self.device)

    def full_long(self, shape: Any, value: int) -> Any:
        return self.torch.full(shape, value, dtype=self.torch.int64, device=self.device)

    def ones_like(self, array: Any) -> Any:
        return self.torch.ones_like(array)

    def columns(self, index: Any) -> Any:
        """Columns of a ``(rows, neurons)`` array as both sides of an assignment read them."""
        return index

    def neurons(self, index: np.ndarray, columns: Any) -> Any:
        """The place of neurons in a state, for ``part``, made once per array of them."""
        if isinstance(columns, slice):
            return columns
        held = self.placed.get(id(index))
        if held is None:
            held = self.placed[id(index)] = (index, self.long(index))
        return held[1]

    # arithmetic

    def bmm(self, a: Any, b: Any) -> Any:
        return self.torch.bmm(a, b)

    def where(self, condition: Any, a: Any, b: Any) -> Any:
        return self.torch.where(condition, a, b)

    def exp(self, x: Any) -> Any:
        return self.torch.exp(x)

    def log(self, x: Any) -> Any:
        return self.torch.log(x)

    def sqrt(self, x: Any) -> Any:
        return self.torch.sqrt(x)

    def isfinite(self, x: Any) -> Any:
        return self.torch.isfinite(x)

    def minimum(self, a: Any, b: Any) -> Any:
        return self.torch.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.torch.maximum(a, b)

    def amax(self, x: Any, axis: int, keepdims: bool = False) -> Any:
        return x.amax(dim=axis, keepdim=keepdims)

    def argmax(self, x: Any, axis: int) -> Any:
        return x.argmax(dim=axis)

    def clamp(self, x: Any, low: int, high: int) -> Any:
        return x.clamp(min=low, max=high)

    def gather(self, x: Any, index: Any) -> Any:
        """Take along the second axis."""
        return x.gather(1, index)

    def per_bucket(self, gradient: Any, buckets: Any, size: int) -> Any:
        """Each row's ``gradient`` summed into ``size`` buckets; ``buckets`` is ``(1, slots)`` or
        ``(rows, slots)``. Through a product and not an atomic add: on a GPU an atomic add takes its
        floats in arrival order, and a life would not repeat to the bit."""
        ones = self.torch.nn.functional.one_hot(buckets, size).to(gradient.dtype)
        return (gradient[:, None, :] @ ones)[:, 0, :]

    def cat(self, parts: Sequence[Any], axis: int) -> Any:
        return self.torch.cat(list(parts), dim=axis)

    def stack(self, parts: Sequence[Any], axis: int = 0) -> Any:
        return self.torch.stack(list(parts), dim=axis)

    def values(self, activity: Any, weights: Any, bias: float) -> np.ndarray:
        """``activity @ weights + bias`` on the host, one number per row."""
        return self.host(activity @ weights + bias)

    def all_finite(self, *arrays: Any) -> Any:
        torch = self.torch
        return torch.stack([torch.isfinite(a).all() for a in arrays]).all()

    def add_outer(self, m: Any, a: Any, b: Any, rate: float, count: int) -> Any:
        """``m + rate * (a @ b) / count``."""
        return rate * (a @ b) / count + m

    def unit_norm(self, x: Any) -> Any:
        """The norm along the last axis, kept."""
        return self.torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    def rows_of(self, array: Any, rows: int) -> Any:
        """A ``(a, b)`` array as ``rows`` rows of itself."""
        return array[None, :, :].expand(rows, -1, -1)

    # the slots

    def room(self, old: Any, drop: int, size: int) -> Any:
        """A wider array holding ``old``'s slots less the ``drop`` oldest, first."""
        kept = old.shape[1] - drop
        room = self.torch.empty(
            (old.shape[0], size, old.shape[2]), dtype=old.dtype, device=self.device
        )
        room[:, :kept] = old[:, drop:]
        return room

    def write_row(self, room: Any, at: int, value: Any) -> None:
        """One slot of every stream; ``value`` None zeroes it."""
        room[:, at] = 0.0 if value is None else value

    def write_cells(self, room: Any, rows: np.ndarray, at: np.ndarray, values: Any) -> None:
        room[self.long(rows), self.long(at)] = values

    def part(self, state: BrainState, place: Any) -> Any:
        """A state's activation at ``place``, in the maps' floats."""
        handle = getattr(state, "device", None)
        s = handle.get("s") if isinstance(handle, dict) else None
        if s is None or not hasattr(s, "device") or s.device.type != self.device.type:
            s = self.here(np.atleast_2d(np.asarray(state.activation, dtype=float)))
        return s[:, place].to(self.dtype)

    def rests_here(self, state: BrainState) -> bool:
        handle = state.device
        return (
            handle is not None and "s" in handle and handle["s"].device.type == self.device.type
        )

    def read_back(
        self, graph: Any, ports: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
        movement: Any, width: int,
    ) -> list[Any]:
        """``movement @ W.T`` per port ``(synapses, rows, cols, width)``, read as
        ``W[rows, cols]``, through the effective synapses of ``graph``."""
        kernel = graph._torch
        out = []
        for port in ports:
            placed = self.placed.get(id(port[0]))
            if placed is None:
                placed = self.placed[id(port[0])] = (
                    port[0], self.long(port[0]), self.long(port[1]), self.long(port[2]),
                )
            _, synapses, on_rows, on_cols = placed
            values = (kernel.gain_pre[synapses] * kernel.scale[synapses]).to(movement.dtype)
            w = self.torch.zeros((port[3], width), dtype=movement.dtype, device=self.device)
            if graph.layout.parallel_synapses:
                w.view(-1).index_add_(0, on_rows * w.shape[1] + on_cols, values)
            else:
                w[on_rows, on_cols] = values
            out.append(movement @ w.T)
        return out


class MlxArrays:
    """The array namespace of an MLX kernel: float32, in unified memory, lazy until ``read``."""

    inf = float("inf")

    def __init__(self, kernel: Any) -> None:
        self.mx = kernel.mx
        self.float = kernel.mx.float32
        self.long_type = kernel.mx.int32
        self.placed: dict[int, tuple[Any, ...]] = {}
        # one-hot rows of a bucket count, read with a matrix product: a scatter's sums on the GPU
        # arrive in no fixed order, and a run reads the same bits again
        self.eyes: dict[int, Any] = {}

    # the arrays' way there and back

    def here(self, array: Any) -> Any:
        return self.mx.array(np.asarray(array, dtype=np.float32))

    def long(self, array: Any) -> Any:
        return self.mx.array(np.asarray(array, dtype=np.int32))

    def flags(self, array: Any) -> Any:
        return self.mx.array(np.asarray(array, dtype=bool))

    def upload(self, array: np.ndarray) -> Any:
        return self.here(array)

    def floats(self, array: Any) -> Any:
        return array.astype(self.float)

    def host(self, array: Any) -> np.ndarray:
        return np.array(array, dtype=np.float64)

    def host_long(self, array: Any) -> np.ndarray:
        return np.array(array, dtype=np.int64)

    def host_flags(self, array: Any) -> np.ndarray:
        return np.array(array, dtype=bool)

    def read(self, *scalars: Any, also: Sequence[Any] = ()) -> np.ndarray:
        """One launch: the scalars and ``also``, what they were checked for."""
        self.mx.eval(*scalars, *also)
        return np.array([float(s.item()) for s in scalars])

    def settle(self, *arrays: Any) -> None:
        self.mx.eval(*arrays)

    # new arrays

    def zeros(self, shape: Any) -> Any:
        return self.mx.zeros(shape, dtype=self.float)

    def zeros_long(self, shape: Any) -> Any:
        return self.mx.zeros(shape, dtype=self.long_type)

    def full_long(self, shape: Any, value: int) -> Any:
        return self.mx.full(shape, value, dtype=self.long_type)

    def ones_like(self, array: Any) -> Any:
        return self.mx.ones_like(array)

    def columns(self, index: Any) -> Any:
        """Columns of a ``(rows, neurons)`` array as both sides of an assignment read them."""
        return index if isinstance(index, slice) else self.long(index)

    def neurons(self, index: np.ndarray, columns: Any) -> Any:
        """The place of neurons in a state, for ``part``, made once per array of them: the neurons
        on the host, and their columns in a state of whole arrays."""
        held = self.placed.get(id(index))
        if held is None:
            held = self.placed[id(index)] = (index, np.asarray(index), self.columns(columns))
        return held[1:]

    # arithmetic

    def bmm(self, a: Any, b: Any) -> Any:
        return a @ b

    def where(self, condition: Any, a: Any, b: Any) -> Any:
        return self.mx.where(condition, a, b)

    def exp(self, x: Any) -> Any:
        return self.mx.exp(x)

    def log(self, x: Any) -> Any:
        return self.mx.log(x)

    def sqrt(self, x: Any) -> Any:
        return self.mx.sqrt(x)

    def isfinite(self, x: Any) -> Any:
        return self.mx.isfinite(x)

    def minimum(self, a: Any, b: Any) -> Any:
        return self.mx.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.mx.maximum(a, b)

    def amax(self, x: Any, axis: int, keepdims: bool = False) -> Any:
        return self.mx.max(x, axis=axis, keepdims=keepdims)

    def argmax(self, x: Any, axis: int) -> Any:
        return self.mx.argmax(x, axis=axis).astype(self.long_type)

    def clamp(self, x: Any, low: int, high: int) -> Any:
        return self.mx.clip(x, low, high)

    def gather(self, x: Any, index: Any) -> Any:
        """Take along the second axis."""
        return self.mx.take_along_axis(x, index, axis=1)

    def _eye(self, size: int) -> Any:
        held = self.eyes.get(size)
        if held is None:
            held = self.eyes[size] = self.mx.eye(size, dtype=self.float)
        return held

    def per_bucket(self, gradient: Any, buckets: Any, size: int) -> Any:
        """Each row's ``gradient`` summed into ``size`` buckets, in slot order; ``buckets`` is
        ``(1, slots)`` or ``(rows, slots)``."""
        return (gradient[:, None, :] @ self._eye(size)[buckets])[:, 0, :]

    def cat(self, parts: Sequence[Any], axis: int) -> Any:
        return self.mx.concatenate(list(parts), axis=axis)

    def stack(self, parts: Sequence[Any], axis: int = 0) -> Any:
        return self.mx.stack(list(parts), axis=axis)

    def values(self, activity: Any, weights: Any, bias: float) -> np.ndarray:
        """``activity @ weights + bias`` on the host, one number per row: a short product of
        what a launch made, read in unified memory, costs less than one synchronization."""
        product = np.array(activity, copy=False) @ np.array(weights, copy=False)
        return np.asarray(product + bias, dtype=np.float64)

    def all_finite(self, *arrays: Any) -> Any:
        """Whether every entry is finite, read off each array's sum: a fifth of the cost of a
        look at every entry on a map of millions, and it refuses only a sum that overflows."""
        mx = self.mx
        return mx.isfinite(mx.sum(mx.stack([mx.sum(a) for a in arrays])))

    def add_outer(self, m: Any, a: Any, b: Any, rate: float, count: int) -> Any:
        """``m + rate * (a @ b) / count``, the product written into the sum: a third of the cost
        of the two operations on a map of millions. ``addmm`` reads an ``alpha`` of zero as one."""
        if not rate:
            return m
        return self.mx.addmm(m, a, b, alpha=rate / count, beta=1.0)

    def unit_norm(self, x: Any) -> Any:
        """The norm along the last axis, kept."""
        return self.mx.sqrt((x * x).sum(axis=-1, keepdims=True))

    def rows_of(self, array: Any, rows: int) -> Any:
        """A ``(a, b)`` array as ``rows`` rows of itself."""
        return self.mx.broadcast_to(array[None, :, :], (rows, *array.shape))

    # the slots

    def room(self, old: Any, drop: int, size: int) -> Any:
        """A wider array holding ``old``'s slots less the ``drop`` oldest, first."""
        kept = old.shape[1] - drop
        room = self.mx.zeros((old.shape[0], size, old.shape[2]), dtype=old.dtype)
        self.mx.eval(room)
        if kept:
            np.array(room, copy=False)[:, :kept] = np.array(old, copy=False)[:, drop:]
        return room

    def write_row(self, room: Any, at: int, value: Any) -> None:
        """One slot of every stream, in place in unified memory: nothing reads this room's slot
        ``at`` yet, or what read it has been made; ``value`` None zeroes it."""
        np.array(room, copy=False)[:, at] = 0.0 if value is None else np.array(value, copy=False)

    def write_cells(self, room: Any, rows: np.ndarray, at: np.ndarray, values: Any) -> None:
        np.array(room, copy=False)[rows, at] = np.array(values, copy=False)

    def part(self, state: BrainState, place: Any) -> Any:
        """A state's activation at ``place``, in the maps' floats."""
        neurons, columns = place
        handle = getattr(state, "device", None)
        if isinstance(handle, _MlxState):
            plan = handle["plan"]
            where = plan.where[neurons]
            if len(where) and where[0] >= 0 and (np.diff(where) == 1).all():
                return handle["sm"][:, int(where[0]) : int(where[-1]) + 1]
            return handle["s"][:, columns]
        if isinstance(handle, dict) and "s" in handle:
            return handle["s"][:, columns]
        return self.here(np.atleast_2d(np.asarray(state.activation, dtype=float))[:, neurons])

    def rests_here(self, state: BrainState) -> bool:
        return isinstance(state.device, _MlxState) or (
            isinstance(state.device, dict) and "s" in state.device
        )

    def read_back(
        self, graph: Any, ports: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
        movement: Any, width: int,
    ) -> list[Any]:
        """``movement @ W.T`` per port ``(synapses, rows, cols, width)``, read as
        ``W[rows, cols]``, on the blocks where a port is whole rows of them in their order, else
        through the host's efficacies."""
        mx, kernel = self.mx, graph._mlx
        out = []
        m = movement.T
        for port in ports:
            parts = kernel._read_blocks(port)
            if parts is None:
                w = np.zeros((port[3], width), dtype=np.float32)
                values = graph._gain_pre[port[0]] * graph.efficacy[port[0]]
                if graph.layout.parallel_synapses:
                    np.add.at(w, (port[1], port[2]), values)
                else:
                    w[port[1], port[2]] = values
                out.append(movement @ mx.array(w).T)
                continue
            made = [
                (kernel.blocks[k][i0:i1] @ m).T
                for k, _, i0, i1 in sorted(parts, key=lambda part: part[1])
            ]
            out.append(made[0] if len(made) == 1 else mx.concatenate(made, axis=1))
        return out


def arrays_of(kernel: Any) -> TorchArrays | MlxArrays:
    """The array namespace of a kernel, one per kernel family."""
    root = getattr(kernel, "root", kernel)
    held = root.__dict__.get("_arrays")
    if held is None:
        held = root.__dict__["_arrays"] = (
            MlxArrays(kernel) if hasattr(kernel, "mx") else TorchArrays(kernel)
        )
    return cast("TorchArrays | MlxArrays", held)


def _forgotten(array: Any, done: np.ndarray) -> Any:
    """A per-row array whose ``done`` rows hold nothing: in place, but on MLX, whose arrays a
    boolean mask cannot index."""
    if type(array).__module__ == "mlx.core":
        import mlx.core as mx

        keep = mx.array(~np.asarray(done)).astype(array.dtype)
        return array * keep.reshape(-1, *([1] * (array.ndim - 1)))
    array[done] = 0.0
    return array


def _set_columns(drive: Any, columns: Any, values: np.ndarray) -> None:
    """What the host holds, into a resident drive."""
    if type(drive).__module__ == "mlx.core":
        import mlx.core as mx

        index = columns if isinstance(columns, slice) else mx.array(
            np.asarray(columns, dtype=np.int32)
        )
        drive[:, index] = mx.array(np.asarray(values, dtype=np.float32))
    else:
        drive[:, columns] = drive.new_tensor(values)


def _set_silent(drive: Any, columns: Any) -> None:
    """Silence on columns of a resident drive."""
    if type(drive).__module__ == "mlx.core":
        import mlx.core as mx

        index = columns if isinstance(columns, slice) else mx.array(
            np.asarray(columns, dtype=np.int32)
        )
        drive[:, index] = 0.0
    else:
        drive[:, columns] = 0.0


# -- the working trace


class _TraceOnDevice:
    """A working trace's ``trace`` and ``last`` on the device of its brain (``Trace``'s equations);
    what no device path reads brings them back (``Trace.__getattr__``)."""

    def __init__(self, owner: Any, kernel: Any) -> None:
        xp = arrays_of(kernel)
        self.owner, self.kernel, self.xp = owner, kernel, xp
        held = owner.__dict__
        self.trace = xp.here(held.pop("trace"))
        self.last = xp.here(held.pop("last"))
        self.glow = xp.columns(owner._glow_columns)
        self.source = xp.neurons(owner.hidden, kernel.index_of(owner.hidden))

    def reset(self, batch: int, rows: np.ndarray | None) -> None:
        owner, xp = self.owner, self.xp
        if rows is None or len(self.trace) != batch:
            width = (batch, len(owner.hidden))
            self.trace = xp.zeros(width)
            self.last = xp.zeros(width)
            owner.cold = np.ones(batch, dtype=bool)
            return
        keep = np.ones(batch)
        keep[rows] = 0.0
        kept = xp.here(keep)[:, None]
        self.trace = self.trace * kept
        self.last = self.last * kept
        owner.cold[rows] = True

    def stimulate(self, drive: Any) -> Any:
        owner = self.owner
        drive[:, self.glow] = (
            owner.amplitude * owner.hearing * self.trace if len(self.trace) == len(drive) else 0.0
        )
        return drive

    def update(self, state: BrainState) -> None:
        owner = self.owner
        h = self.xp.part(state, self.source)
        if len(self.trace) != len(h):
            self.reset(len(h), None)
        self.trace = owner.decay * self.trace + (1.0 - owner.decay) * h
        self.last = h
        owner.cold[:] = False


# -- the recall memory

_MAP_KINDS = (
    "head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps", "grid_maps",
)


class _RecallOnDevice:
    """A recall memory's slots, maps and reads on the device of its brain.

    The same equations as ``Recall``'s host methods, in the maps' floats (float64 wherever a torch
    device has it, float32 on MLX), for a memory without several heads, a window, the command, the
    read of when, the gains or declared tokens (``Recall.resides``). What rests here rests only
    here: the memory's attribute of each name in ``OWNED`` is absent while it does, and reading
    one brings every array back to the host (``Recall.__getattr__``), so what no device path
    takes (a checkpoint, an imagined branch, an expert's edit) runs on the host's arrays, and the
    next moment places them here again. ``held`` and ``ended`` stay the host's, which decides
    without a read of the device. A slot's key and value are one array while they are equal,
    as they are without a window or the command.
    """

    OWNED = (
        "keys", "values", "nexts", "previous", "told", "focus", "trace", "head_maps",
        "position_maps", "anchor_maps", "relative_maps", "slope_maps", "grid_maps",
    )
    keys: Any
    values: Any
    nexts: Any
    previous: Any
    told: Any
    focus: Any
    trace: Any
    head_maps: list[Any]
    position_maps: list[Any]
    anchor_maps: list[Any]
    relative_maps: list[Any]
    slope_maps: list[Any]
    grid_maps: list[Any]

    def __init__(self, recall: Any, kernel: Any) -> None:
        xp = arrays_of(kernel)
        self.recall, self.kernel, self.xp = recall, kernel, xp
        self.rooms: dict[str, tuple[Any, Any, int, int]] = {}
        self.columns = xp.neurons(recall.association, recall._association_columns)
        capacity = int(recall.capacity)
        # every age up to the capacity, the oldest first: a stream's slots are their last ones
        self.whole_ages = xp.long(np.arange(capacity - 1, -1, -1, dtype=np.int64))
        self.ages_ = xp.floats(self.whole_ages)
        self.table = xp.long(
            recall._age_buckets() if recall.position else np.zeros(capacity, dtype=np.int64)
        )
        self.grid: Any = None
        if recall.grid:
            ages = np.arange(capacity - 1, -1, -1, dtype=np.int64)
            starts = np.cumsum([0, *recall.grid[:-1]])
            self.grid = xp.long(np.stack([
                start + ages % modulus
                for start, modulus in zip(starts, recall.grid, strict=True)
            ]))
        self.ports: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        for name, (synapses, rows, cols), width in (
            ("recall", recall._head_synapses[0][0], len(recall.recall_neurons)),
            ("next", recall._head_synapses[0][1], len(recall.next_neurons)),
        ):
            self.ports[name] = (synapses, rows, cols, width)
        self.port_columns = xp.columns(recall._port_columns)
        self.unread = 0  # the writes since a read made what they add to
        self.adopt()

    # -- the arrays' way here and back

    def adopt(self) -> None:
        """Every array the memory holds comes here, and leaves the host."""
        recall, xp = self.recall, self.xp
        here = xp.here
        held = recall.__dict__
        keys, values = held.pop("keys"), held.pop("values")
        self.keys = here(keys)
        self.values = self.keys if np.array_equal(keys, values) else here(values)
        self.nexts = here(held.pop("nexts"))
        self.previous = here(held.pop("previous"))
        self.told = xp.long(held.pop("told"))
        self.focus = xp.long(held.pop("focus"))
        self.trace = here(held.pop("trace"))
        for name in _MAP_KINDS:
            setattr(self, name, [tuple(here(a) for a in maps) for maps in held.pop(name)])
        attended = recall._attended
        if attended is not None:
            recall._attended = here(attended)
        recall.last = None if recall.last is None else self._moment_here(recall.last)
        self.rooms = {}
        recall._rooms = {k: v for k, v in recall._rooms.items() if k == "held"}

    def _moment_here(self, moment: _Moment) -> _Moment:
        here = self.xp.here
        return _Moment(moment.event, here(moment.cortex), here(moment.address), here(moment.key),
                       here(moment.inputs))

    def give_back(self) -> None:
        """Every array returns to the host, as the host's methods read and write them."""
        recall, xp = self.recall, self.xp
        host = xp.host
        held = recall.__dict__
        held["keys"] = host(self.keys)
        held["values"] = host(self.values)
        held["nexts"] = host(self.nexts)
        held["previous"] = host(self.previous)
        held["told"] = xp.host_long(self.told)
        held["focus"] = xp.host_long(self.focus)
        held["trace"] = host(self.trace)
        for name in _MAP_KINDS:
            held[name] = [tuple(host(a) for a in maps) for maps in getattr(self, name)]
        if recall._attended is not None:
            attended = host(recall._attended)
            attended.flags.writeable = False
            recall._attended = attended
        if recall.last is not None:
            moment = recall.last
            recall.last = _Moment(moment.event, host(moment.cortex), host(moment.address),
                                  host(moment.key), host(moment.inputs))

    # -- the slots' count

    @property
    def streams(self) -> int:
        return int(self.keys.shape[0])

    @property
    def slots(self) -> int:
        return int(self.keys.shape[1])

    def _whole_ages(self) -> Any:
        return self.whole_ages[self.whole_ages.shape[0] - self.slots:]

    def _ages(self) -> Any:
        return self.ages_[self.ages_.shape[0] - self.slots:]

    # -- the moment's address and read

    def _moment(self, x: np.ndarray) -> _Moment:
        """``Recall._moment``: ``event`` stays the host's, to recognise the moment at its write."""
        recall, xp = self.recall, self.xp
        inputs = self.kernel.observed(x)
        n = len(x)
        c = self.previous if len(self.previous) == n else xp.zeros((n, len(recall.association)))
        key = recall.heard * inputs
        w_x, w_c, b = self.head_maps[0]
        return _Moment(x, c, key + inputs @ w_x + c @ w_c + b, key, inputs)

    def _term(self, maps: Any, x: Any, c: Any) -> Any:
        m_x, m_c, m = maps
        return x @ m_x + c @ m_c + m

    def _anchored(self, told: Any) -> Any:
        """``Recall.anchored_buckets`` of the streams whose moments ``told`` holds."""
        recall, xp = self.recall, self.xp
        places = told[:, None] - 1 - self._whole_ages()[None, :]
        out = self.table[xp.clamp(places, 0, self.table.shape[0] - 1)]
        return xp.where(places < 0, recall.position_buckets, out)

    def _relative(self, focus: Any) -> Any:
        """``Recall.relative_buckets`` of the streams whose reads' focus is ``focus``."""
        reach = self.recall.RELATIVE
        offsets = self._whole_ages()[None, :] - focus[:, None]
        out = self.xp.clamp(offsets, -reach, reach) + reach
        return self.xp.where(focus[:, None] < 0, 2 * reach + 1, out)

    def _position(self, x: Any, c: Any, told: Any, focus: Any) -> Any:
        """``Recall.position_term`` of head 0, at every slot."""
        recall, xp = self.recall, self.xp
        if not self.position_maps or not self.slots:
            return None
        whole = self._whole_ages()
        term = self._term(self.position_maps[0], x, c)[:, self.table[whole]]
        if self.anchor_maps:
            term = term + xp.gather(self._term(self.anchor_maps[0], x, c), self._anchored(told))
        if self.relative_maps:
            term = term + xp.gather(self._term(self.relative_maps[0], x, c), self._relative(focus))
        ages = self._ages()
        if self.slope_maps:
            lean = self._term(self.slope_maps[0], x, c)
            term = term + lean[:, None] * (ages / recall.SLOPE)[None]
        if self.grid_maps:
            grid = self.grid[:, self.grid.shape[1] - self.slots:]
            terms = self._term(self.grid_maps[0], x, c)[:, grid].sum(axis=1)
            term = term + terms * (ages >= recall.position)
        return term

    def _logits(self, address: Any, keys: Any, held: Any, position: Any) -> Any:
        """``Recall._attention``: ``held`` None where every stream holds every slot."""
        recall, xp = self.recall, self.xp
        logits = xp.bmm(keys, address[:, :, None])[:, :, 0] / recall.temperature
        if recall.recency:
            logits = logits - recall.recency * self._ages()[None, :]
        if position is not None:
            logits = logits + position
        return logits if held is None else xp.where(held, logits, -xp.inf)

    def _attention(self, moment: _Moment) -> Any:
        """``Recall.weights`` at the moment's address, with its position term; None without
        slots or of another batch."""
        recall, xp = self.recall, self.xp
        if not self.slots or self.streams != len(moment.address):
            return None
        position = self._position(moment.inputs, moment.cortex, self.told, self.focus)
        if recall.held.all():
            logits = self._logits(moment.address, self.keys, None, position)
            p = xp.exp(logits - xp.amax(logits, 1, keepdims=True))
            return p / p.sum(axis=1, keepdims=True)
        held = xp.flags(recall.held)
        logits = self._logits(moment.address, self.keys, held, position)
        top = xp.where(~held.any(axis=1), 0.0, xp.amax(logits, 1))
        p = xp.exp(logits - top[:, None])
        total = p.sum(axis=1, keepdims=True)
        return p / xp.where(total > 0, total, 1.0)

    def stimulate(self, drive: Any, observations: np.ndarray) -> Any:
        """``Recall.stimulate``."""
        recall, xp = self.recall, self.xp
        moment = self._moment(recall._port(observations))
        recall.last = moment
        p = self._attention(moment)
        columns_ = self.port_columns
        resident = _resident(drive)
        out: Any = drive if resident else np.array(drive, dtype=float)
        if p is None:
            held = out[:, columns_] if resident else xp.here(out[:, recall._port_columns])
        else:
            amplitude = 1.0 if recall.drive_gain else recall.amplitude
            read = p[:, None, :]
            parts = [amplitude * xp.bmm(read, self.values)[:, 0]]
            if recall.next:
                parts.append(amplitude * xp.bmm(read, self.nexts)[:, 0])
            held = xp.cat(parts, 1)
        if recall.drive_gain:
            self.trace = held
            held = recall.amplitude * recall.hearing * held
        if resident:
            out[:, columns_] = held
        else:
            out[:, recall._port_columns] = xp.host(held)
        recall._attended = p
        recall._head_reads = None
        return out

    # -- the write after the settle

    def witness(
        self, observations: np.ndarray, state: BrainState, command: np.ndarray | None = None
    ) -> None:
        """``Recall.witness``."""
        recall, xp = self.recall, self.xp
        x = recall._port(observations)
        moment = recall.last
        if moment is None or moment.event.shape != x.shape or not (
            moment.event is x or np.array_equal(moment.event, x)
        ):
            moment = self._moment(x)
        cortex = xp.part(state, self.columns)
        if len(cortex) != len(x):
            raise ValueError("the settled state needs one row per stream")
        self._observe(moment.key, moment.inputs)
        recall.writes += len(x)
        self.previous = cortex
        recall.last = None
        self.unread += 1
        if self.unread >= 64:  # what no read of a moment makes: no graph grows without end
            xp.settle(self.told, self.focus, self.previous)
            self.unread = 0

    def _observe(self, key: Any, value: Any) -> None:
        """``Recall._observe``."""
        recall, xp = self.recall, self.xp
        if len(key) != self.streams:
            self.empty(len(key))
        successor: tuple[Any, ...] | None = None
        if self.slots:
            if recall.held.all() and not recall.ended.any():
                successor = (slice(None), value)
            else:
                held = recall.held
                rows = held.any(axis=1) & ~recall.ended
                latest = held.shape[1] - 1 - np.argmax(held[:, ::-1], axis=1)
                index = np.flatnonzero(rows)
                successor = (index, latest[index], value[xp.long(index)])
        if recall.ended.any():
            recall.ended = np.zeros(len(key), dtype=bool)
        attended: Any = recall._attended
        if attended is not None and tuple(attended.shape) == (len(key), self.slots) and self.slots:
            read = xp.argmax(attended, 1)
            focus = xp.where(xp.amax(attended, 1) > 0, self._whole_ages()[read] + 1, -1)
        else:
            focus = xp.where(self.focus >= 0, self.focus + 1, -1)
        self._append(key, np.ones(len(key), dtype=bool), value, successor)
        self.told = self.told + 1
        self.focus = focus

    def _longer(self, name: str, old: Any, drop: int, new: Any) -> Any:
        """``Recall._longer``: the slots a view into a wider array, a write one slot of it."""
        xp = self.xp
        held = self.rooms.get(name)
        if held is not None and held[1] is old and held[3] < held[0].shape[1]:
            room, first, stop = held[0], held[2] + drop, held[3]
        else:
            kept = old.shape[1] - drop
            size = min(2 * self.recall.capacity, max(16, 2 * kept + 2))
            room = xp.room(old, drop, size)
            first, stop = 0, kept
        xp.write_row(room, stop, new)
        out = room[:, first : stop + 1]
        self.rooms[name] = (room, out, first, stop + 1)
        return out

    def _append(
        self, key: Any, held: np.ndarray, value: Any = None,
        successor: tuple[Any, ...] | None = None,
    ) -> None:
        """``Recall._append``; the host keeps ``held``."""
        recall, xp = self.recall, self.xp
        kept = self.keys.shape[1]
        drop = max(kept + 1 - recall.capacity, 0)
        shared = self.values is self.keys
        nexts = self._longer("nexts", self.nexts, drop, None)
        if successor is not None and kept:
            room, _, first, _ = self.rooms["nexts"]
            if len(successor) == 2:
                if kept - drop:
                    xp.write_row(room, first + kept - 1 - drop, successor[1])
            else:
                rows, slots, values = successor
                keep = slots >= drop
                if keep.any():
                    xp.write_cells(
                        room, rows[keep], first + slots[keep] - drop,
                        values[xp.long(np.flatnonzero(keep))],
                    )
        self.keys = self._longer("keys", self.keys, drop, key)
        if shared:
            self.values = self.keys
        else:
            self.values = self._longer("values", self.values, drop, value)
        self.nexts = nexts
        recall.held = recall._longer("held", recall.held, drop, held)

    def empty(self, batch: int) -> None:
        """``Recall._empty``."""
        recall, xp = self.recall, self.xp
        slots = 1 if recall._bounded() and batch else 0
        self.keys = xp.zeros((batch, slots, recall.address_width))
        self.values = self.keys
        self.nexts = xp.zeros((batch, slots, recall.value_width))
        self.previous = xp.zeros((batch, len(recall.association)))
        self.told = xp.zeros_long(batch)
        self.focus = xp.full_long((batch,), -1)
        recall.held = np.ones((batch, slots), dtype=bool)
        recall.ended = np.zeros(batch, dtype=bool)
        recall.acted = np.zeros((batch, recall.command_width))
        recall.past = np.zeros((0, recall.window - 1, recall.key_width))
        self.rooms = {}
        recall._unread()

    def reset(self, batch: int, rows: np.ndarray | None) -> None:
        """``Recall.reset``."""
        recall, xp = self.recall, self.xp
        if rows is not None:
            rows = _stream_rows(rows, batch)
        if rows is None or self.streams != batch:
            self.empty(batch)
            return
        chosen = np.zeros(batch, dtype=bool)
        chosen[rows] = True
        if chosen.all():
            self.empty(batch)
            return
        if not chosen.any():
            return
        keep = ~chosen
        here = xp.flags(keep)
        shared = self.values is self.keys
        held = recall.held & keep[:, None]
        keys = xp.where(here[:, None, None], self.keys, 0.0)
        values = keys if shared else xp.where(here[:, None, None], self.values, 0.0)
        nexts = xp.where(here[:, None, None], self.nexts, 0.0)
        if recall._bounded():
            if not held.shape[1]:
                keys = xp.zeros((batch, 1, keys.shape[2]))
                values = keys if shared else xp.zeros((batch, 1, values.shape[2]))
                nexts = xp.zeros((batch, 1, nexts.shape[2]))
                held = np.zeros((batch, 1), dtype=bool)
            held[chosen, -1] = True
        lead = held.any(axis=0)
        first = int(np.argmax(lead)) if lead.any() else held.shape[1]
        self.keys = keys[:, first:]
        self.values = self.keys if shared else values[:, first:]
        self.nexts = nexts[:, first:]
        recall.held = held[:, first:].copy()
        self.rooms = {}
        self.previous = xp.where(here[:, None], self.previous, 0.0)
        self.told = xp.where(here, self.told, 0)
        self.focus = xp.where(here, self.focus, -1)
        recall.ended = recall.ended & keep
        recall._unread()

    def snapshot(self) -> dict[str, Any]:
        """``Recall.snapshot``: the next write starts new arrays and changes none of these."""
        recall = self.recall
        self.rooms = {}
        recall._rooms = {}
        return {
            "keys": self.keys, "values": self.values, "nexts": self.nexts, "held": recall.held,
            "previous": self.previous, "ended": recall.ended, "past": recall.past,
            "acted": recall.acted, "told": self.told, "focus": self.focus,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """``Recall.restore``."""
        recall = self.recall
        for name in ("keys", "values", "nexts", "previous", "told", "focus"):
            setattr(self, name, snapshot[name])
        for name in ("held", "ended", "past", "acted"):
            recall.__dict__[name] = snapshot[name]
        self.rooms = {}
        recall._unread()

    # -- the address lesson

    def _readbacks(self, graph: Any, movement: Any) -> tuple[Any, Any]:
        """``Recall.readback`` through the recall and next ports."""
        recall = self.recall
        ports = [self.ports["recall"]]
        if recall.next:
            ports.append(self.ports["next"])
        backs = self.xp.read_back(graph, ports, movement, len(recall.association))
        return backs[0], backs[1] if recall.next else None

    def _row_steps(
        self, moment: _Moment, rows: np.ndarray, d_value: Any, d_next: Any,
        moved: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """``Recall._row_steps`` of head 0, every row of ``rows`` taken to move, or those ``moved``
        marks: ``moving`` first in what it returns, for the caller to read with what it checks,
        so a lesson waits once for the device."""
        recall, xp = self.recall, self.xp
        whole = len(rows) == self.streams
        values, nexts, keys = self.values, self.nexts, self.keys
        told, focus = self.told, self.focus
        q, x, c, key = moment.address, moment.inputs, moment.cortex, moment.key
        held = recall.held
        # each row's slope rule (``Recall.within_table``): its own slots against the table
        own = (
            held[rows].sum(axis=1) <= recall.position if self.grid_maps
            else np.ones(len(rows), dtype=bool)
        )
        if not whole:
            index = xp.long(rows)
            shared = values is keys
            values, nexts, keys = values[index], nexts[index], keys[index]
            if shared:
                values = keys
            told, focus = told[index], focus[index]
            q, x, c, key = q[index], x[index], c[index], key[index]
            held = held[rows]
        masked = not held.all()
        score = xp.bmm(values, d_value[:, :, None])[:, :, 0]
        norm = (d_value * d_value).sum(axis=1)
        if d_next is not None:
            score = score + xp.bmm(nexts, d_next[:, :, None])[:, :, 0]
            norm = norm + (d_next * d_next).sum(axis=1)
        norm = xp.sqrt(norm)
        moving = norm > 1e-12
        if masked:
            moving = moving & xp.flags(held.any(axis=1))
        if moved is not None and not moved.all():
            index = xp.long(np.flatnonzero(moved))
            keys, score, norm, told, focus = (
                a[index] for a in (keys, score, norm, told, focus)
            )
            q, x, c, key = q[index], x[index], c[index], key[index]
            held = held[moved]
            own = own[moved]
        logits = score / norm[:, None] / recall.address_temperature
        held_t = None
        if masked:
            held_t = xp.flags(held)
            logits = xp.where(held_t, logits, -xp.inf)
        attention = self._logits(q, keys, held_t, self._position(x, c, told, focus))
        attention = attention - xp.amax(attention, 1, keepdims=True)
        attention = attention - xp.log(xp.exp(attention).sum(axis=1, keepdims=True))
        logits = logits + attention
        p = xp.exp(logits - xp.amax(logits, 1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        finite = xp.isfinite(attention)
        gradient = p - xp.where(finite, xp.exp(attention), 0.0)
        error = xp.bmm(gradient[:, None, :], keys)[:, 0]
        energy = (x * x).sum(axis=1) + (c * c).sum(axis=1) + 1.0
        bound = xp.minimum(xp.ones_like(energy), 1.0 / (recall.address * energy))
        step = error * bound[:, None]
        return moving, x, c, key, error, step, p, attention, told, focus, own

    def learn(self, graph: Any, learned: Any) -> dict[str, float]:
        """``Recall.learn``: the address step and the position step checked in one read."""
        recall, xp = self.recall, self.xp
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        if self.position_maps:
            quiet.update(position_rows=0.0, position_refused=0.0)
        moment = recall.last
        positions = bool(self.position_maps) and (recall.position_rate > 0 or recall.slope > 0)
        if (
            not (recall.address or positions)
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
        if not self.slots or not len(rows) or self.streams != len(moment.address):
            return quiet
        minus = learned.free if learned.opposite is None else learned.opposite
        movement = self._movement(learned.nudged, minus)
        d_value, d_next = self._readbacks(graph, movement)
        found = self._row_steps(moment, rows, d_value, d_next)
        steps, read = self._checked_steps(found, positions)
        if not read[0]:  # not every row moves: which do, and those alone
            moved = xp.host_flags(found[0])
            if not moved.any():
                return dict(quiet)
            found = self._row_steps(moment, rows, d_value, d_next, moved)
            steps, read = self._checked_steps(found, positions)
        return self._install_steps(quiet, found, steps, read, positions)

    def _checked_steps(
        self, found: tuple[Any, ...], positions: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """The steps of the maps at ``found``, not installed, and one read of their checks:
        whether every row moves, whether the address step is finite and its mean error, whether
        the position step is finite."""
        recall, xp = self.recall, self.xp
        _, x, c, key, error, step = found[:6]
        steps: dict[str, Any] = {}
        scalars = [found[0].all()]
        also: list[Any] = []
        if recall.address:
            w_x, w_c, b = self._lms(self.head_maps[0], step, recall.address, x, c)
            address = key + x @ w_x + c @ w_c + b
            steps["head_maps"] = (w_x, w_c, b)
            scalars += [xp.all_finite(w_x, w_c, b, address), error.abs().mean()]
            also += [w_x, w_c, b]
        if positions:
            stepped = self._position_step(found)
            steps["positions"] = stepped
            arrays = [a for maps in stepped.values() for a in maps]
            scalars.append(xp.all_finite(*arrays))
            also += arrays
        return steps, xp.read(*scalars, also=also)

    def _install_steps(
        self, quiet: dict[str, float], found: tuple[Any, ...], steps: dict[str, Any],
        read: np.ndarray, positions: bool,
    ) -> dict[str, float]:
        """What ``_checked_steps`` read: each step kept or refused whole, and the report."""
        recall = self.recall
        report = dict(quiet)
        rows = float(len(found[1]))
        at = 1
        if recall.address:
            if read[at]:
                self.head_maps = [steps["head_maps"]]
                report["address_rows"] = rows
                report["address_error"] = float(read[at + 1])
            else:
                report["address_refused"] = rows
            at += 2
        if positions:
            if read[at]:
                for name, maps in steps["positions"].items():
                    setattr(self, name, [maps])
                report["position_rows"] = rows
            else:
                report["position_refused"] = rows
        return report

    def _movement(self, plus: BrainState, minus: BrainState) -> Any:
        """The association's movement between two phases, in the maps' floats: differenced after
        the cast, as the host's copies are."""
        part = self.xp.part
        return part(plus, self.columns) - part(minus, self.columns)

    def _lms(self, maps: Any, step: Any, rate: float, x: Any, c: Any) -> Any:
        """``_lms``: the same arithmetic on the device."""
        m_x, m_c, m = maps
        count = len(x)
        add_outer = self.xp.add_outer
        return (
            add_outer(m_x, x.T, step, rate, count),
            add_outer(m_c, c.T, step, rate, count),
            m + rate * step.mean(axis=0),
        )

    def _position_step(self, found: tuple[Any, ...]) -> dict[str, Any]:
        """``Recall._position_step``: the maps it would keep, by name."""
        recall, xp = self.recall, self.xp
        _, x, c, _, _, _, target, prior, told, focus, within = found
        rate = recall.position_rate
        count = recall.position_buckets
        whole = self._whole_ages()
        ages = self._ages()
        finite = xp.isfinite(prior)
        attended = xp.where(finite, xp.exp(prior), 0.0)
        gradient = xp.where(finite, target - attended, 0.0)
        energy = (x * x).sum(axis=1) + (c * c).sum(axis=1) + 1.0
        one = xp.ones_like(energy)
        bound = xp.minimum(one, 1.0 / (rate * energy))[:, None]
        step = xp.per_bucket(gradient, self.table[whole][None, :], count) * bound
        stepped = {"position_maps": self._lms(self.position_maps[0], step, rate, x, c)}
        if self.anchor_maps:
            step = xp.per_bucket(gradient, self._anchored(told), count + 1) * bound
            stepped["anchor_maps"] = self._lms(self.anchor_maps[0], step, rate, x, c)
        if self.relative_maps:
            width = 2 * recall.RELATIVE + 2
            step = xp.per_bucket(gradient, self._relative(focus), width) * bound
            stepped["relative_maps"] = self._lms(self.relative_maps[0], step, rate, x, c)
        if self.slope_maps and within.any():  # the streams whose own slots fit the table
            lean = gradient @ (ages / recall.SLOPE)
            lean = lean * xp.minimum(one, 1.0 / (recall.slope * energy))
            if not within.all():
                lean = xp.where(xp.flags(within), lean, 0.0 * lean)
            stepped["slope_maps"] = self._lms(self.slope_maps[0], lean, recall.slope, x, c)
        if self.grid_maps:
            peak = xp.maximum(
                xp.amax(target, 1, keepdims=True), xp.amax(attended, 1, keepdims=True)
            )
            far = xp.where(peak > 0, gradient / peak, gradient)
            far = far * (ages >= recall.position)
            width = sum(recall.grid)
            grid = self.grid[:, self.grid.shape[1] - self.slots:]
            total = xp.zeros((far.shape[0], width))
            for residues in grid:
                total = total + xp.per_bucket(far, residues[None, :], width)
            stepped["grid_maps"] = self._lms(self.grid_maps[0], total * bound, rate, x, c)
        return stepped

    # -- reward trains the read

    def contrast_rows(self, graph: Any, plus: BrainState, minus: BrainState) -> Any:
        """``Recall.contrast_rows``: ``(batch, inputs, columns)``, the columns ``reward_blocks``
        names. Made before the slot this moment writes, so a write reads nothing unmade."""
        recall, xp = self.recall, self.xp
        positions = bool(recall.position_reward) and bool(self.position_maps)
        if not (recall.address_reward or positions):
            return None
        movement = self._movement(plus, minus)
        batch = len(movement)
        if recall.address_reward:
            recall._address_held = True
        columns_ = sum(b[1] for b in recall.reward_blocks)
        width = recall.map_width
        moment = recall.last
        if moment is None or not self.slots or self.streams != batch or len(moment.key) != batch:
            return xp.zeros((batch, width, columns_))
        d_value, d_next = self._readbacks(graph, movement)
        moving, x, _, _, error, _, target, prior, told, _, _ = self._row_steps(
            moment, np.arange(batch), d_value, d_next
        )
        energy = (x * x).sum(axis=1)
        one = xp.ones_like(energy)

        def join(block: Any) -> Any:
            rows = x[:, :, None] * block[:, None, :]
            keep = moving & xp.isfinite(rows).all(axis=2).all(axis=1)
            return xp.where(keep[:, None, None], rows, 0.0)

        parts = []
        if recall.address_reward:
            step = error * xp.minimum(one, 1.0 / (recall.address * energy))[:, None]
            parts.append(join(recall.address * step))
        elif recall._address_held:
            parts.append(xp.zeros((batch, width, recall.address_width)))
        if positions:
            rate = recall.REWARD_RATE
            gradient = xp.where(xp.isfinite(prior), target - xp.exp(prior), 0.0)
            bound = xp.minimum(one, 1.0 / (rate * energy))[:, None]
            count = recall.position_buckets
            whole = self._whole_ages()
            terms = xp.per_bucket(gradient, self.table[whole][None, :], count)
            parts.append(join(rate * terms * bound))
            if self.anchor_maps:
                terms = xp.per_bucket(gradient, self._anchored(told), count + 1)
                parts.append(join(rate * terms * bound))
        out = parts[0] if len(parts) == 1 else xp.cat(parts, 2)
        xp.settle(out)
        return out

    def weighted(self, trace: Any, delta: Any) -> Any:
        """The rows' eligibility weighted by the dopamine ``delta`` (on the host or on the device)
        and averaged."""
        xp = self.xp
        d = xp.floats(delta) if isinstance(delta, type(trace)) else xp.here(
            np.asarray(delta, dtype=float)
        )
        return (d[:, None, None] * trace).mean(axis=0)

    def propose(self, step: Any) -> tuple[Any, ...] | None:
        """``Recall.propose`` of a step on the event's rows and the blocks that step."""
        unread = self.proposal_unread(step)
        if unread is None:
            return None
        xp = self.xp
        read = xp.read(unread[3], unread[4], also=unread[5])
        if not read[0]:
            raise ValueError("the address map's reward step must be finite")
        return (*unread[:3], float(read[1]), *unread[6:])

    def proposal_unread(self, step: Any) -> tuple[Any, ...] | None:
        """``propose`` with nothing read: the maps, a flag of their and the step's finiteness, the
        step's norm, what MLX makes with them, then the position and anchored maps."""
        recall, xp = self.recall, self.xp
        positions = bool(recall.position_reward) and bool(self.position_maps)
        if not (recall.address_reward or positions):
            return None
        w_x, w_c, b = self.head_maps[0]
        stepped, changed = [], []
        start = recall.address_width if recall._address_held or recall.address_reward else 0
        if recall.address_reward:
            moved = recall.address_reward * step[:, :recall.address_width]
            w_x = w_x + moved
            stepped.append(moved)
            changed.append(w_x)
        position_maps = anchor_maps = None
        if positions:
            count = recall.position_buckets
            part = recall.position_reward * step[:, start : start + count]
            on_x, on_c, on_b = self.position_maps[0]
            position_maps = [(on_x + part, on_c, on_b)]
            stepped.append(part)
            changed.append(position_maps[0][0])
            start += count
            if self.anchor_maps:
                part = recall.position_reward * step[:, start : start + count + 1]
                on_x, on_c, on_b = self.anchor_maps[0]
                anchor_maps = [(on_x + part, on_c, on_b)]
                stepped.append(part)
                changed.append(anchor_maps[0][0])
        every = [*changed, *stepped]
        norm = xp.sqrt(sum((part * part).sum() for part in stepped))
        return (w_x, w_c, b, xp.all_finite(*every), norm, every, position_maps, anchor_maps)

    def accept(self, proposal: tuple[Any, ...]) -> None:
        """``Recall.accept``."""
        self.head_maps = [proposal[:3]]
        if proposal[4] is not None:
            self.position_maps = list(proposal[4])
        if proposal[5] is not None:
            self.anchor_maps = list(proposal[5])


# -- the consolidating store


class _StoreOnDevice:
    """A consolidating store's strengths on the device of its brain (``SynapticMemory``'s
    equations, without a separator or a surprise gate): its read at every moment, its write at
    every outcome. What no device path reads brings them back (``FastSynapses.__getattr__``)."""

    OWNED = ("strength", "mass", "consolidated")
    strength: Any
    mass: Any
    consolidated: Any

    def __init__(self, owner: Any, kernel: Any) -> None:
        xp = arrays_of(kernel)
        self.owner, self.kernel, self.xp = owner, kernel, xp
        held = owner.__dict__
        for name in self.OWNED:
            setattr(self, name, xp.here(held.pop(name)))
        self.pre = xp.columns(kernel.index_of(owner.pre))
        self.post = xp.columns(owner._post_columns)

    def give_back(self) -> None:
        for name in self.OWNED:
            self.owner.__dict__[name] = self.xp.host(getattr(self, name))

    def _unit(self, x: Any) -> Any:
        """``FastSynapses._delta_unit``."""
        xp = self.xp
        scale = xp.amax(x.abs(), -1, keepdims=True)
        scaled = x / xp.where(scale > 0, scale, 1.0)
        norm = xp.unit_norm(scaled)
        return scaled / xp.where(norm > 0, norm, 1.0)

    def stimulate(self, drive: Any, key: np.ndarray | None) -> Any:
        """``FastSynapses.stimulate`` of the memory's read into a resident drive."""
        owner, xp = self.owner, self.xp
        if key is None:
            cue = xp.floats(drive[:, self.pre])
        else:
            cue = self.kernel.observed(owner._port(key, len(owner.pre), "key"))
        same_batch = len(self.strength) == len(cue)
        strength = self.strength if same_batch else self.consolidated[None, :, :]
        cue = self._unit(cue)
        read = (cue[:, None, :] @ strength)[:, 0, :]
        drive[:, self.post] = drive[:, self.post] + owner.amplitude * read
        return drive

    def observe(
        self, key: np.ndarray, value: np.ndarray, gate: np.ndarray, salience: np.ndarray,
        observed: np.ndarray,
    ) -> None:
        """``SynapticMemory.observe`` without a surprise gate: validated on the host."""
        owner, xp = self.owner, self.xp
        batch = len(key)
        same_batch = len(self.strength) == batch
        consolidated = self.consolidated
        previous = self.strength if same_batch else xp.rows_of(consolidated, batch)
        strength = consolidated + owner.decay * (previous - consolidated)
        mass = self.mass * owner.decay if same_batch else xp.zeros(batch)
        rows = np.flatnonzero(gate)
        if not len(rows):
            self.strength, self.mass = strength, mass
            return
        cue = self._unit(xp.upload(key[rows]))
        value_t = xp.upload(value[rows])
        mask = xp.upload(observed[rows].astype(np.float64))
        rate = owner.consolidation + np.minimum(1.0, owner.consolidation * salience[rows])
        rate = xp.upload(np.minimum(1.0, rate))
        error = (value_t - cue @ consolidated) * mask
        change = ((rate[:, None] * cue).T @ error) / len(rows)
        strength = strength + change
        index = xp.long(rows)
        prediction = (cue[:, None, :] @ strength[index])[:, 0, :]
        correction = (value_t - prediction) * mask
        fast = owner.rate * cue[:, :, None] * correction[:, None, :]
        new = consolidated + change
        strength[index] = strength[index] + fast
        mass[index] = mass[index] + owner.rate
        read = xp.read(xp.all_finite(strength, new), also=(strength, new, mass))
        if not read[0]:
            raise ValueError("synaptic update overflowed; scale the observed values")
        self.consolidated, self.strength, self.mass = new, strength, mass
        owner.writes += len(rows)


# -- the critic


class _CriticOnDevice:
    """A critic's weights and eligibility on the device of its brain (``ActorCritic``'s equations):
    a value per row comes to the host at each moment and the dopamine goes there, one number per
    row each. The bias is the host's, one number. A moment's eligibility is a new array, so a
    refused outcome keeps the old one. What no device path reads brings them back
    (``ActorCritic.__getattr__``)."""

    OWNED = ("w_critic", "trace_critic")

    def __init__(self, owner: Any, kernel: Any) -> None:
        xp = arrays_of(kernel)
        self.owner, self.kernel, self.xp = owner, kernel, xp
        held = owner.__dict__
        self.weights = xp.here(held.pop("w_critic"))
        trace = held.pop("trace_critic")
        self.trace = None if trace is None else xp.here(trace)
        self.source = xp.neurons(owner.critic_index, kernel.index_of(owner.critic_index))
        self.faded = 0  # the moments since an outcome made the eligibility

    def give_back(self) -> None:
        held = self.owner.__dict__
        held["w_critic"] = self.xp.host(self.weights)
        held["trace_critic"] = None if self.trace is None else self.xp.host(self.trace)

    def value(self, state: BrainState) -> np.ndarray:
        """``ActorCritic.value``."""
        xp = self.xp
        return xp.values(xp.part(state, self.source), self.weights, self.owner.b_critic)

    def add(self, state: BrainState, batch: int, decay: float) -> None:
        """``e <- decay e + (activity, 1)``."""
        xp = self.xp
        trace = self.trace
        if trace is None or len(trace) != batch:
            trace = xp.zeros((batch, len(self.owner.critic_index) + 1))
        trace = decay * trace
        trace[:, :-1] = trace[:, :-1] + xp.part(state, self.source)
        trace[:, -1] = trace[:, -1] + 1.0
        self.trace = trace  # made by the outcome's candidate, which reads it

    def value_here(self, state: BrainState) -> Any:
        """``ActorCritic.value`` left on the device, in the maps' floats."""
        xp = self.xp
        return xp.part(state, self.source) @ self.weights + self.owner.b_critic

    def unread(self, delta: Any) -> tuple[Any, Any, Any]:
        """``candidate`` with nothing read: the weights, a flag of their finiteness and the bias's
        step, on the device, for a caller that reads them with the rest of an outcome."""
        trace = self.trace
        assert trace is not None
        trace = trace / (1.0 + (trace * trace).sum(axis=1, keepdims=True))
        xp = self.xp
        d = xp.here(delta) if isinstance(delta, np.ndarray) else xp.floats(delta)
        step = self.owner.config.eta_critic * (d[:, None] * trace).mean(axis=0)
        weights = self.weights + step[:-1]
        return weights, xp.all_finite(weights), step[-1]

    def candidate(self, delta: np.ndarray) -> tuple[Any, float]:
        """``ActorCritic._critic_candidate``: one read, of the checks and the bias's step."""
        owner, xp = self.owner, self.xp
        trace = self.trace
        assert trace is not None
        trace = trace / (1.0 + (trace * trace).sum(axis=1, keepdims=True))
        step = owner.config.eta_critic * (xp.here(delta)[:, None] * trace).mean(axis=0)
        weights = self.weights + step[:-1]
        read = xp.read(xp.all_finite(weights), step[-1], also=(weights,))
        bias = owner.b_critic + float(read[1])
        if not read[0] or not np.isfinite(bias):
            raise ValueError("critic parameters must remain finite")
        return weights, bias

    def forget(self, done: np.ndarray) -> None:
        if self.trace is not None:
            self.trace = _forgotten(self.trace, done)

    def fade(self, decay: float, ended: np.ndarray | None) -> None:
        """``ActorCritic.fade``: ``ended`` rows forget. What no outcome reads is made now and then,
        so no graph grows without end."""
        if self.trace is None:
            return
        self.trace = decay * self.trace
        if ended is not None and ended.any():
            self.forget(ended)
        self.faded += 1
        if self.faded >= 64:
            self.xp.settle(self.trace)
            self.faded = 0


# -- the carried routes' contrast


def route_contrast(
    xp: Any, graph: Any, routes: Sequence[Mapping[str, Any]], gain: Any, model: Any, width: int,
    movement: Any, rows: np.ndarray | None, batch: int,
) -> np.ndarray:
    """``Hearing._contrast`` of a movement on the device: the routes' synapses are read there, and
    one number per row and route comes back."""
    out = []
    index = None if rows is None else xp.long(rows)
    ports = [
        (route["synapses"], route["rows"], route["cols"], route["width"]) for route in routes
    ]
    backs = xp.read_back(graph, ports, movement, width)
    for route, back in zip(routes, backs, strict=True):
        memory = route["memory"]
        resident = getattr(memory, "_device", None)
        trace = resident.trace if resident is not None else memory.trace
        if isinstance(trace, np.ndarray):
            trace = xp.here(trace)
        if len(trace) != batch:
            out.append(xp.zeros(len(movement)))
            continue
        held = (trace if index is None else trace[index])[:, route["start"] : route["stop"]]
        amplitude = float(memory.amplitude)
        v = amplitude * gain(route) * held
        rest = model.rest_emission
        sig = 1.0 / (1.0 + xp.exp(-model.slope * (v - model.threshold)))
        d = model.slope * sig * (1.0 - sig)
        slope = xp.where(sig > rest, d / (1.0 - rest), model.leak * d / rest)
        out.append((slope * amplitude * held * back).sum(axis=1))
    return cast(np.ndarray, xp.host(xp.stack(out, axis=1)))
