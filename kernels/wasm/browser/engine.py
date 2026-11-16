"""Where a script of this folder runs and on which engine: under Pyodide (``browser/run.py``)
the bundle sits at /w, natively the checkout's source is used.

``choose(name)`` before ``cadence`` is imported: ``wasm`` (the kernel under Pyodide), ``fused``
(Numba natively), ``numpy`` (the NumPy loop, ``CADENCE_FUSED=0``)."""
from __future__ import annotations

import os
import sys

BROWSER = sys.platform == "emscripten"
ROOT = "/w" if BROWSER else os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
KERNEL = "/w/cadence_kernel.wasm"


def choose(name: str) -> str:
    if name == "numpy":
        os.environ["CADENCE_FUSED"] = "0"
    elif name == "wasm":
        if not BROWSER:
            raise SystemExit("the wasm engine runs under Pyodide: browser/run.py")
        os.environ["CADENCE_WASM"] = KERNEL
    elif name != "fused":
        raise SystemExit(f"unknown engine {name}")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    return name


def fit_32_bit_numpy() -> None:
    """Pyodide's NumPy is 32-bit: ``np.repeat`` with int64 counts refuses the cast (a gap of the
    library under Pyodide, outside the kernels); the learning module gets a ``repeat`` that
    casts the counts."""
    import types

    import numpy as np

    if np.dtype(np.intp).itemsize >= 8:
        return
    import cadence.learning as learning

    shim = types.ModuleType("numpy")
    shim.__dict__.update(np.__dict__)
    shim.repeat = lambda a, repeats, axis=None: np.repeat(  # type: ignore[attr-defined]
        a, np.asarray(repeats).astype(np.intp), axis=axis)
    learning.np = shim  # type: ignore[attr-defined]
