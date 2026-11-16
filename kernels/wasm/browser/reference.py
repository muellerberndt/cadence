"""Every compiled call, checked against the kernels' own Python.

The kernels of cadence/fused.py are Python; run without Numba (a stand-in ``numba`` whose
``njit`` returns the function) they are the reference arithmetic. Under Pyodide each call the
library makes into the WebAssembly kernels is made again on copies through that reference, and
every output and every array written in place is compared to the bit.

Imported by shadow.py and direct.py: after it, every call the library makes into the kernels
of ``cadence.fused`` is made again through the reference, and ``report()`` says what differed.
"""
import importlib.util
import sys
import types

import engine

engine.choose("wasm")

import numpy as np  # noqa: E402

from cadence import _wasm, fused  # noqa: E402

assert _wasm.available(), "the WebAssembly kernels did not load"
engine.fit_32_bit_numpy()
EVERY = 1

fake = types.ModuleType("numba")


def njit(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f


fake.njit, fake.prange = njit, range
fake.config = types.SimpleNamespace(NUMBA_NUM_THREADS=1)
fake.get_num_threads = lambda: 1
sys.modules["numba"] = fake
spec = importlib.util.spec_from_file_location("cadence._fused_reference", fused.__file__)
ref = importlib.util.module_from_spec(spec)
ref.__package__ = "cadence"
spec.loader.exec_module(ref)
del sys.modules["numba"]
ref._index = int  # Numba's unsigned index arithmetic, as Python integers

WRAPPERS = ["fused_settle", "fused_residual", "fused_solve", "fused_fast_solve", "trace_step",
            "scatter", "gather", "relative_floor", "optimizer_step", "tied_step",
            "add_episode_terms", "clear_choice", "rows_finite"]
checked = {name: 0 for name in WRAPPERS}
called = {name: 0 for name in WRAPPERS}
failures = []


def same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if x.dtype.kind == "f":
            return bool(np.array_equal(x.view(np.uint64 if x.itemsize == 8 else np.uint32),
                                       y.view(np.uint64 if y.itemsize == 8 else np.uint32)))
        return bool(np.array_equal(x, y))
    if isinstance(x, tuple) and isinstance(y, tuple):
        if len(x) == 4 and callable(x[0]) and isinstance(x[3], np.ndarray):  # products
            return x[2] == y[2] and same(x[3], y[3])
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if x is None or y is None:
        return x is y
    if isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating)):
        return np.float64(x).view(np.uint64) == np.float64(y).view(np.uint64)
    return x == y


def copied(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple) and len(value) == 4 and callable(value[0]):
        return (value[0], value[1], value[2], value[3].copy())  # products: the kept array
    return value


def shadow(name):
    live, reference = getattr(fused, name), getattr(ref, name)

    def call(*args, **kwargs):
        called[name] += 1
        if called[name] % globals()["EVERY"]:
            return live(*args, **kwargs)
        theirs = [copied(a) for a in args]
        their_kw = {k: copied(v) for k, v in kwargs.items()}
        expected = reference(*theirs, **their_kw)
        got = live(*args, **kwargs)
        checked[name] += 1
        ok = same(got, expected)
        for k, (a, b) in enumerate(zip(args, theirs)):
            if isinstance(a, np.ndarray) and not same(a, b):
                ok = False
                failures.append(f"{name}: argument {k} differs after the call")
        if not ok:
            failures.append(f"{name} call {called[name]}: outputs differ")
        return got

    setattr(fused, name, call)


for name in WRAPPERS:
    shadow(name)



def report():
    print("checked", {k: v for k, v in checked.items() if v})
    print("failures", len(failures))
    for f in failures[:20]:
        print("  ", f)
    print("PASS" if not failures else "FAIL")
