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


def finish(ok: bool) -> None:
    """The script's verdict: its exit status natively; under Pyodide a failure raises, which the
    page reports as the script's error."""
    print("PASS" if ok else "FAIL", flush=True)
    if BROWSER:
        if not ok:
            raise RuntimeError("the check failed")
        return
    sys.exit(0 if ok else 1)
