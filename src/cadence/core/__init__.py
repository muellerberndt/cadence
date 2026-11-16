"""The native core (``kernels/core``): a brain's whole moment in Rust, its state kept there
between moments, the library's Python the reference.

A brain's state, a moment's inputs and its replies cross as named arrays (``pack`` and
``unpack``, the format of ``kernels/core/src/state.rs``). ``Core`` loads the shared library
(``CADENCE_CORE`` names it; by default the one ``cargo build --release`` makes in
``kernels/core``) and calls it through ctypes. A part's state follows the attributes its Python
reads and writes in a moment; each part's ``state_of`` and ``restore`` keep the crossing
exact, so an export read back and exported again is the same bytes.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Core", "pack", "unpack"]

MAGIC = b"CAD1"
_KINDS = {1: np.dtype("<f8"), 2: np.dtype("<i8"), 3: np.dtype("bool"), 4: np.dtype("<u8")}
_CODES = {"f": 1, "i": 2, "b": 3, "u": 4}


def pack(named: Mapping[str, Any]) -> bytes:
    """Named arrays as the core reads them: float64, int64, bool or uint64, in their order."""
    parts = [MAGIC, struct.pack("<I", len(named))]
    for name, value in named.items():
        array = np.asarray(value)
        code = _CODES.get(array.dtype.kind)
        if code is None:
            raise TypeError(f"{name}: the core takes float64, int64, bool or uint64, "
                            f"not {array.dtype}")
        array = np.ascontiguousarray(array, dtype=_KINDS[code])
        encoded = name.encode()
        head = struct.pack("<H", len(encoded)) + encoded + struct.pack(
            "<BB", code, array.ndim) + b"".join(struct.pack("<Q", d) for d in array.shape)
        data = array.tobytes()
        pad = (8 - (len(head) + len(data)) % 8) % 8
        parts.extend([head, data, b"\0" * pad])
    return b"".join(parts)


def unpack(data: bytes) -> dict[str, np.ndarray]:
    """The named arrays ``pack`` wrote, each its own copy."""
    view = memoryview(data)
    if bytes(view[:4]) != MAGIC:
        raise ValueError("not the core's named arrays")
    (count,) = struct.unpack_from("<I", view, 4)
    at, out = 8, {}
    for _ in range(count):
        start = at
        (length,) = struct.unpack_from("<H", view, at)
        name = bytes(view[at + 2:at + 2 + length]).decode()
        at += 2 + length
        code, ndim = struct.unpack_from("<BB", view, at)
        at += 2
        shape = struct.unpack_from(f"<{ndim}Q", view, at)
        at += 8 * ndim
        kind = _KINDS[code]
        size = int(np.prod(shape, dtype=np.int64)) * kind.itemsize
        out[name] = np.frombuffer(view[at:at + size], dtype=kind).reshape(shape).copy()
        at += size
        at += (8 - (at - start) % 8) % 8
    return out


def _default_library() -> Path:
    root = Path(__file__).resolve().parents[3] / "kernels" / "core" / "target" / "release"
    name = {"darwin": "libcadence_core.dylib", "win32": "cadence_core.dll"}.get(
        sys.platform, "libcadence_core.so")
    return root / name


class Core:
    """The shared library, loaded once."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        path = path or os.environ.get("CADENCE_CORE") or _default_library()
        lib: Any = ctypes.CDLL(str(path))
        lib.core_version.restype = ctypes.c_uint32
        lib.core_release.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.core_call.restype = ctypes.c_int32
        lib.core_call.argtypes = [ctypes.c_char_p, ctypes.c_size_t] * 2 + [
            ctypes.c_char_p, ctypes.c_size_t] * 2 + [ctypes.POINTER(ctypes.c_void_p),
                                                     ctypes.POINTER(ctypes.c_size_t)]
        lib.core_probe.restype = ctypes.c_int32
        lib.core_probe.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                                   ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_size_t)]
        self.lib = lib
        self.version = int(lib.core_version())

    def _taken(self, call: Any, *args: Any) -> bytes:
        at, length = ctypes.c_void_p(), ctypes.c_size_t()
        code = int(call(*args, ctypes.byref(at), ctypes.byref(length)))
        if code < 0:
            raise RuntimeError(f"cadence core: the call refused ({code})")
        try:
            return ctypes.string_at(at, length.value) if length.value else b""
        finally:
            self.lib.core_release(at, length)

    def call(self, part: str, method: str, state: Mapping[str, Any],
             args: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """One call of a part's method in the core on ``state``: the state after it and what
        it returned, each as named arrays."""
        p, m = part.encode(), method.encode()
        s, a = pack(state), pack(args)
        both = unpack(self._taken(self.lib.core_call, p, len(p), m, len(m), s, len(s), a,
                                  len(a)))
        after = {k[6:]: v for k, v in both.items() if k.startswith("state:")}
        out = {k[4:]: v for k, v in both.items() if k.startswith("out:")}
        return after, out

    def probe(self, name: str, **named: Any) -> dict[str, np.ndarray]:
        """One of the core's primitives on named arrays (the checks against NumPy)."""
        data, encoded = pack(named), name.encode()
        return unpack(self._taken(self.lib.core_probe, encoded, len(encoded), data, len(data)))


def generator_state(rng: np.random.Generator) -> dict[str, Any]:
    """A NumPy ``PCG64`` generator's state as named arrays (the 128-bit state and increment as
    two uint64 each, high first)."""
    held = rng.bit_generator.state
    if held["bit_generator"] != "PCG64":
        raise ValueError("the core draws as PCG64 draws")
    s, inc = int(held["state"]["state"]), int(held["state"]["inc"])
    mask = (1 << 64) - 1
    return {"state": np.array([s >> 64, s & mask], dtype=np.uint64),
            "inc": np.array([inc >> 64, inc & mask], dtype=np.uint64),
            "has_uint32": np.int64(held["has_uint32"]), "uinteger": np.int64(held["uinteger"])}
