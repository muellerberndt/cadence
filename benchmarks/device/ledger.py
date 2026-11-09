"""The ledger of a moment: every crossing between the host and the device, with its bytes and time.

A moment of ``live``, ``step``, ``lesson`` or ``act`` on a composed brain runs some of its work on a
device and some on the host. Each time an array goes one way or the other, and each time the host
waits for the device, the moment pays for it. This script lists those crossings by the place in the
library that made them, per moment, beside the seconds of the moment itself.

Two brains: ``chamber``, the default ``Brain.compose(35, 18)`` (251 neurons, 13,120 synapses), and
``speech``, ``Brain.compose(3920, 37, modules=(512,))`` (12,821 neurons, 6.3 million synapses)::

    python benchmarks/device/ledger.py --backend mlx --brains chamber speech --rows 1 16
    python benchmarks/device/ledger.py --backend torch:cuda --rows 1 16 256 --out ledger.jsonl
    python benchmarks/device/ledger.py --backend cpu --rows 1 16
    python benchmarks/device/ledger.py --table ledger.jsonl

Each reading is two passes over the same moments of one brain. The first, untouched, gives the
seconds of a moment (the median of ``--moments`` after ``--warm`` moments). The second counts:
it wraps the ways an array crosses (on torch a tensor's ``cpu``, ``numpy``, ``item``, ``tolist``,
``to``, ``copy_`` and its conversions to a Python number, ``torch.as_tensor`` and ``torch.tensor``
onto a device; on MLX ``mx.array`` of a host array, NumPy's ``array``, ``asarray`` and
``ascontiguousarray`` of an MLX array, an array's ``item`` and ``tolist``, and ``mx.eval``) and
names each by the innermost function of the library on the stack. A crossing's time is the time
of its call, which includes the wait for the device's queued work. The counting pass is slower,
so its seconds are not the moment's. The script uses the public API only.

This is a benchmark of a source checkout: ``PYTHONPATH=src`` from the repository's root.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

BRAINS: dict[str, tuple[int, int, dict[str, Any]]] = {
    "chamber": (35, 18, {}),
    "speech": (3920, 37, {"modules": (512,)}),
}
SOURCE = str(Path(cd.__file__).resolve().parent)


def _site() -> str:
    """The innermost frame of the library on the stack: ``module.function``."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if name.startswith(SOURCE):
            module = Path(name).stem
            return f"{module}.{getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)}"
        frame = frame.f_back
    return "outside"


class Ledger:
    """Crossings by site: count, bytes and seconds, in each direction."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0, 0.0])
        self.active = False

    def add(self, way: str, nbytes: int, seconds: float) -> None:
        if not self.active:
            return
        row = self.rows[(way, _site())]
        row[0] += 1
        row[1] += nbytes
        row[2] += seconds

    def timed(self, way: str, nbytes: Callable[[Any], int], call: Callable[..., Any]) -> Any:
        ledger = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not ledger.active:
                return call(*args, **kwargs)
            start = time.perf_counter()
            out = call(*args, **kwargs)
            ledger.add(way, nbytes((args, kwargs, out)), time.perf_counter() - start)
            return out

        return wrapped


def _nbytes(value: Any) -> int:
    try:
        return int(np.asarray(value).nbytes) if not hasattr(value, "nbytes") else int(value.nbytes)
    except Exception:
        return 0


@contextmanager
def _torch_counting(ledger: Ledger) -> Iterator[None]:
    import torch

    Tensor = torch.Tensor
    saved: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, new: Any) -> None:
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, new)

    def on_device(t: Any) -> bool:
        return isinstance(t, Tensor) and t.device.type != "cpu"

    def tensor_bytes(t: Any) -> int:
        return int(t.numel() * t.element_size())

    def to_host(name: str) -> None:
        original = getattr(Tensor, name)

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not ledger.active or not on_device(self):
                return original(self, *args, **kwargs)
            start = time.perf_counter()
            out = original(self, *args, **kwargs)
            ledger.add("to host", tensor_bytes(self), time.perf_counter() - start)
            return out

        patch(Tensor, name, wrapped)

    for name in ("cpu", "numpy", "item", "tolist", "__bool__", "__float__", "__int__",
                 "__index__"):
        to_host(name)

    original_to = Tensor.to

    def to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not ledger.active:
            return original_to(self, *args, **kwargs)
        start = time.perf_counter()
        out = original_to(self, *args, **kwargs)
        if isinstance(out, Tensor) and out.device.type != self.device.type:
            way = "to host" if out.device.type == "cpu" else "to device"
            ledger.add(way, tensor_bytes(self), time.perf_counter() - start)
        return out

    patch(Tensor, "to", to)
    original_copy = Tensor.copy_

    def copy_(self: Any, src: Any, *args: Any, **kwargs: Any) -> Any:
        if not ledger.active or not isinstance(src, Tensor) or src.device.type == self.device.type:
            return original_copy(self, src, *args, **kwargs)
        start = time.perf_counter()
        out = original_copy(self, src, *args, **kwargs)
        way = "to host" if self.device.type == "cpu" else "to device"
        ledger.add(way, tensor_bytes(src), time.perf_counter() - start)
        return out

    patch(Tensor, "copy_", copy_)

    def made(name: str) -> None:
        original = getattr(torch, name)

        def wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
            device = kwargs.get("device")
            if not ledger.active or device is None or torch.device(device).type == "cpu" or (
                isinstance(data, Tensor)
            ):
                return original(data, *args, **kwargs)
            start = time.perf_counter()
            out = original(data, *args, **kwargs)
            ledger.add("to device", tensor_bytes(out), time.perf_counter() - start)
            return out

        patch(torch, name, wrapped)

    made("as_tensor")
    made("tensor")
    for module in (getattr(torch, "cuda", None), getattr(torch, "mps", None)):
        if module is not None and hasattr(module, "synchronize"):
            patch(module, "synchronize", ledger.timed("wait", lambda _: 0, module.synchronize))
    try:
        yield
    finally:
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)


class _ArrayType(type):
    """``mx.array`` in the counting pass: made arrays are counted, ``isinstance`` is MLX's."""

    real: Any = None
    ledger: Ledger | None = None

    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, cls.real)

    def __call__(cls, data: Any, *args: Any, **kwargs: Any) -> Any:
        ledger = cls.ledger
        if ledger is None or not ledger.active or isinstance(data, cls.real):
            return cls.real(data, *args, **kwargs)
        start = time.perf_counter()
        out = cls.real(data, *args, **kwargs)
        ledger.add("to device", int(out.nbytes), time.perf_counter() - start)
        return out


@contextmanager
def _mlx_counting(ledger: Ledger) -> Iterator[None]:
    import mlx.core as mx

    real = mx.array
    saved: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, new: Any) -> None:
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, new)

    proxy = _ArrayType("array", (), {})
    proxy.real, proxy.ledger = real, ledger
    patch(mx, "array", proxy)
    patch(mx, "eval", ledger.timed("wait", lambda _: 0, mx.eval))
    for name in ("array", "asarray", "ascontiguousarray"):
        original = getattr(np, name)

        def wrapped(obj: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if not ledger.active or not isinstance(obj, real):
                return _original(obj, *args, **kwargs)
            start = time.perf_counter()
            out = _original(obj, *args, **kwargs)
            ledger.add("to host", int(obj.nbytes), time.perf_counter() - start)
            return out

        patch(np, name, wrapped)
    calls: dict[int, float] = {}

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "c_call" and getattr(arg, "__name__", "") in ("item", "tolist"):
            if isinstance(getattr(arg, "__self__", None), real):
                calls[id(arg)] = time.perf_counter()
        elif event in ("c_return", "c_exception") and id(arg) in calls:
            start = calls.pop(id(arg))
            ledger.add("to host", int(arg.__self__.nbytes), time.perf_counter() - start)

    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)


def _counting(backend: str, ledger: Ledger) -> Any:
    if backend == "mlx":
        return _mlx_counting(ledger)
    if backend.startswith("torch"):
        return _torch_counting(ledger)

    @contextmanager
    def nothing() -> Iterator[None]:
        yield

    return nothing()


def _place(backend: str) -> dict[str, Any]:
    if backend.startswith("torch"):
        parts = backend.split(":")
        place: dict[str, Any] = {"backend": "torch"}
        if len(parts) > 1:
            place["device"] = parts[1]
        if len(parts) > 2:
            place["precision"] = parts[2]
        return place
    return {"backend": backend}


def _sync(brain: cd.Brain) -> None:
    kernel = brain.brain._torch
    if kernel is not None and kernel.device.type == "cuda":
        kernel.torch.cuda.synchronize()


def _brain(name: str, backend: str, seed: int, arousal: bool) -> cd.Brain:
    inputs, motor, options = BRAINS[name]
    if arousal:
        options = {**options, "arousal": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(inputs, motor, seed=seed, **options, **_place(backend))


def _moments(
    brain: cd.Brain, kind: str, rows: int, count: int, seed: int
) -> Iterator[Callable[[], Any]]:
    """``count`` moments of ``kind``, each a call with its observations drawn here."""
    inputs, motor = brain.inputs, len(brain.motor_index)
    rng = np.random.default_rng(seed)

    def observed() -> np.ndarray:
        return rng.random((rows, inputs)) * (rng.random((rows, inputs)) < 0.3)

    for t in range(count):
        x = observed()
        if kind == "live":
            reward = np.array([float(rng.random() < 0.3)]) if t else None
            yield lambda x=x, r=reward: brain.live(x, **({} if r is None else {"reward": r}))
        elif kind == "step":
            reward = (rng.random(rows) < 0.3).astype(float) if t else None
            yield lambda x=x, r=reward: brain.step(x, **({} if r is None else {"reward": r}))
        elif kind == "lesson":
            labels = rng.integers(0, motor, rows)
            yield lambda x=x, y=labels: brain.lesson(x, y)
        elif kind == "act":
            yield lambda x=x: brain.act(x, greedy=True)
        else:
            raise ValueError(kind)


def read(name: str, backend: str, kind: str, rows: int, warm: int, count: int, seed: int,
         ledger: bool) -> dict[str, Any]:
    """One brain, one kind of moment, one batch: seconds per moment and the crossings."""
    if kind == "live" and rows != 1:
        raise ValueError("live follows one stream")
    out: dict[str, Any] = {"brain": name, "backend": backend, "kind": kind, "rows": rows}
    brain = _brain(name, backend, seed, kind == "live")
    seconds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t, call in enumerate(_moments(brain, kind, rows, warm + count, seed)):
            start = time.perf_counter()
            call()
            _sync(brain)
            if t >= warm:
                seconds.append(time.perf_counter() - start)
    out["placement"] = {k: brain.placement[k] for k in ("backend", "device", "precision")}
    out["synapses"] = int(brain.connectome.synapses)
    out["seconds"] = float(np.median(seconds))
    out["seconds_mean"] = float(np.mean(seconds))
    if not ledger:
        return out
    book = Ledger()
    with warnings.catch_warnings(), _counting(backend, book):
        warnings.simplefilter("ignore")
        for t, call in enumerate(_moments(brain, kind, rows, count, seed + 1)):
            book.active = True
            call()
            _sync(brain)
            book.active = False
    out["crossings"] = sorted(
        (
            {"way": way, "site": site, "count": row[0] / count, "bytes": row[1] / count,
             "seconds": row[2] / count}
            for (way, site), row in book.rows.items()
        ),
        key=lambda r: -r["seconds"],
    )
    return out


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "machine": platform.machine(), "system": platform.system(),
        "python": sys.version.split()[0], "cadence": getattr(cd, "__version__", "?"),
        "backends": cd.available_backends(),
    }
    try:
        env["load_average"] = os.getloadavg()
    except OSError:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["card"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def table(paths: list[str]) -> None:
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if "brain" not in row:
                continue
            cross = row.get("crossings", [])
            count = sum(c["count"] for c in cross if c["way"] != "wait")
            waits = sum(c["count"] for c in cross if c["way"] == "wait")
            nbytes = sum(c["bytes"] for c in cross)
            spent = sum(c["seconds"] for c in cross)
            print(f"{row['brain']:8} {row['backend']:12} {row['kind']:7} {row['rows']:>4}  "
                  f"{row['seconds'] * 1e3:9.2f} ms  crossings {count:7.1f}  waits {waits:5.1f}  "
                  f"{nbytes / 1e3:10.1f} kB  {spent * 1e3:8.2f} ms in them")
            for c in cross[:12]:
                print(f"      {c['way']:9} {c['site']:60} {c['count']:7.2f} x "
                      f"{c['bytes'] / 1e3:10.2f} kB {c['seconds'] * 1e3:8.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="cpu",
                        help="cpu, mlx, torch, torch:cuda, torch:mps, torch:cuda:float64")
    parser.add_argument("--brains", nargs="+", default=["chamber", "speech"], choices=BRAINS)
    parser.add_argument("--kinds", nargs="+", default=["live", "step", "lesson"],
                        choices=["live", "step", "lesson", "act"])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--moments", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ledger", action="store_true", help="seconds only")
    parser.add_argument("--out")
    parser.add_argument("--table", nargs="+")
    args = parser.parse_args()
    if args.table:
        table(args.table)
        return
    sink = open(args.out, "a") if args.out else None
    env = environment()
    print(json.dumps({"environment": env}), file=sink or sys.stdout, flush=True)
    for name in args.brains:
        for kind in args.kinds:
            for rows in args.rows:
                if kind == "live" and rows != 1:
                    continue
                row = read(name, args.backend, kind, rows, args.warm, args.moments, args.seed,
                           not args.no_ledger)
                print(json.dumps(row), file=sink or sys.stdout, flush=True)
                if sink:
                    print(f"{name} {kind} {rows}: {row['seconds'] * 1e3:.2f} ms", flush=True)
    if sink:
        sink.close()


if __name__ == "__main__":
    main()
