"""The compiled kernels of ``cadence.fused`` in WebAssembly, for the library under Pyodide.

In a browser the library runs in Pyodide, where Numba is not available. The kernels of
``cadence/fused.py`` are built from ``kernels/wasm`` into one module, ``cadence_kernel.wasm``, a
release artefact. A page opts in by placing the file in Pyodide's file system and naming it
before ``cadence`` is imported::

    pyodide.FS.writeFile('/home/pyodide/cadence_kernel.wasm', bytes);
    pyodide.runPython(
      "import os; os.environ['CADENCE_WASM'] = '/home/pyodide/cadence_kernel.wasm'");

The library then takes its compiled path as it does with Numba: ``cadence.fused`` holds a
function of the same name and arguments for each kernel, which hands its arrays to the module,
and ``cadence.compiled`` the kernels of the parts around the settle. ``CADENCE_FUSED=0`` keeps
the NumPy loop, as everywhere.

The module runs in Pyodide's own memory: it imports Pyodide's memory, its stack pointer,
``malloc``, ``free`` and the C library's ``exp`` (the one NumPy calls). An array is handed over as
its object's address, ``id(array)``, and the module reads the data pointer, shape, flags and type
from NumPy's array object, so nothing is copied; an array it does not take (another type, not
C-contiguous) is refused before anything is written, and the call is made again with copies of
the module's types. The products sum in the order NumPy's matrix product without BLAS sums
(Pyodide's NumPy has none), so every kernel that keeps the NumPy loop's bits in ``fused.py``
keeps the bits of the NumPy loop under Pyodide.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["available", "load"]

# the version of the kernels' argument lists this file speaks (``kernel_version``)
VERSION = 2

_EXPORTS: Any = None


def available() -> bool:
    """Whether the module is loaded."""
    return _EXPORTS is not None


def _instantiate(data: bytes) -> Any:
    """The module's exports, instantiated in Pyodide's memory."""
    import pyodide_js  # type: ignore[import-not-found]
    from pyodide.code import run_js  # type: ignore[import-not-found]
    from pyodide.ffi import to_js  # type: ignore[import-not-found]

    make = run_js(
        """(bytes, M) => {
          const module = new WebAssembly.Module(bytes);
          const env = { memory: M.memory, __stack_pointer: M.___stack_pointer,
                        host_exp: M._exp, host_tanh: M._tanh, host_malloc: M._malloc,
                        host_free: M._free };
          return new WebAssembly.Instance(module, { env }).exports;
        }"""
    )
    return make(to_js(data), pyodide_js._module)


def _agrees(exports: Any) -> str | None:
    """Why the module cannot read this NumPy's arrays, or None where it reads them."""
    if int(exports.kernel_version()) != VERSION:
        return "the kernel file is another version"
    numbers = {np.float64: 12, np.int64: 9, np.bool_: 0}
    for kind, number in numbers.items():
        if np.dtype(kind).num != number:
            return f"NumPy's type number of {np.dtype(kind)} is not {number}"
    if np.dtype(np.intp).itemsize != 4 or np.dtype(np.intp).num not in (5, 7):
        return "NumPy's index type is not 32 bits"
    for kind, number in numbers.items():
        for shape in ((3,), (2, 5)):
            x: np.ndarray = np.zeros(shape, dtype=kind)
            pointer = x.__array_interface__["data"][0]
            rows, columns = (shape[0], 1) if len(shape) == 1 else shape
            if int(exports.array_data(id(x), number)) != pointer:
                return "the module does not read NumPy's array objects"
            if int(exports.array_shape(id(x))) != rows * 65536 + columns:
                return "the module does not read NumPy's shapes"
        view: np.ndarray = np.zeros((4, 4), dtype=kind)[:, 1:3]
        if int(exports.array_data(id(view), number)) != 0:
            return "the module does not refuse a strided view"
    return None


def load(source: str | bytes | None = None) -> bool:
    """Load the kernel module (a path in Pyodide's file system, or its bytes; by default the
    path ``CADENCE_WASM`` names) and give ``cadence.fused`` its kernels. Returns whether they are
    in use; outside Pyodide, without a file, or with ``CADENCE_FUSED=0`` nothing is loaded."""
    global _EXPORTS
    if _EXPORTS is not None:
        return True
    if sys.platform != "emscripten" or os.environ.get("CADENCE_FUSED", "1") == "0":
        return False
    if source is None:
        source = os.environ.get("CADENCE_WASM")
        if not source:
            return False
    try:
        if isinstance(source, str):
            with open(source, "rb") as f:
                source = f.read()
        exports = _instantiate(bytes(source))
        why = _agrees(exports)
    except Exception as error:  # a page without the module keeps the NumPy loop
        warnings.warn(f"cadence: the WebAssembly kernels are not in use ({error})",
                      RuntimeWarning, stacklevel=2)
        return False
    if why is not None:
        warnings.warn(f"cadence: the WebAssembly kernels are not in use ({why})",
                      RuntimeWarning, stacklevel=2)
        return False
    _EXPORTS = exports
    from . import brain, fused
    from . import compiled as parts

    install(vars(fused))
    for name in _PARTS:
        setattr(parts, "_" + name, _caller(name))
    brain._FUSED = brain._fused_available()
    return True


# -- the calls: each export takes its arrays as object addresses ------------------------------
#
# A kind per argument: ``d`` a float64 array, ``w`` a float64 array the kernel writes, ``q`` an
# int64 array, ``i`` an index array (int64 or int32), ``?`` a bool array, ``f`` a float, ``n`` an
# integer, ``b`` a flag. A refused call (a negative code) is made again with each array as the
# module takes it, the written ones copied back.

_ARRAYS = {"d": np.float64, "w": np.float64, "q": np.int64, "i": np.int64, "?": np.bool_}

# the kernels of ``cadence.compiled``, each held there under its name with an underscore
_PARTS = ("map_settle", "bound_recall", "map_pieces", "map_moves", "map_write_values",
          "map_report")

_KINDS = {
    "settle_rows": "wwwddqdqqqqq?dbfffffbffbddqnfddddbqqqqdffdbnfbdbww",
    "settle_lives": "wwwddqdqqqqq?dbfffffbffbddqnfddddbqqqqdffdbnfbdbww",
    "residual_rows": "wwwddqdqqqqqdbfffffbfbddqnfddddbqqqqdffdbwwbbw",
    "solve_rows": "wwwddqdqqqqq?dfffffffddqnfddddqqqqdnnfwwbbbdffwwwq",
    "fast_solve": "wwwdqqqqffffnnffwqw",
    "publish_rows": "dwwbffff",
    "kept_length": "qqqqnn",
    "kept_length_all": "qqnn",
    "rows_finite": "d",
    "trace_step": "wwfddiifdww",
    "place": "idw",
    "pick": "difw",
    "group_power": "diww",
    "divisor": "didffw",
    "plain": "dddffffffbbffwww",
    "moments": "dddffffffbwwww",
    "step": "ddidfffw",
    "tied": "diddfwwwq",
    "add_episode_terms": "wddbbiiinnnn",
    "clear_choice": "dfq",
    "map_settle": "dqdqqqqq?dfffffbqqqqdffd?bdbddd?b?b?bdbqnw",
    "bound_recall": "qddd?b?bdbnnd?qnffnfffwq",
    "map_pieces": "?nnqq",
    "map_moves": "ddddfffnnw",
    "map_write_values": "nqqdd?d?wdffnn",
    "map_report": "nffdddddbnnnffffffffnnbd??w?wqq",
}


def _again(export: Callable[..., int], name: str, kinds: str, args: tuple[Any, ...],
           code: int) -> int:
    """``export`` once more with every array as the module takes it, the written ones copied
    back; a second refusal is a shape the kernel does not take."""
    held: list[tuple[np.ndarray, np.ndarray]] = []
    alive: list[np.ndarray] = []
    calls: list[int | float] = []
    for kind, value in zip(kinds, args, strict=True):
        if kind in _ARRAYS:
            array: np.ndarray = np.ascontiguousarray(value, dtype=_ARRAYS[kind])
            if kind == "w" and array is not value:
                held.append((value, array))
            alive.append(array)
            calls.append(id(array))
        elif kind == "f":
            calls.append(float(value))
        elif kind == "n":
            calls.append(int(value))
        else:
            calls.append(1 if value else 0)
    again = int(export(*calls))
    if again < 0:
        raise RuntimeError(f"cadence: the WebAssembly kernel {name} refused argument "
                           f"{-1 - again} (first refusal {-1 - code})")
    for original, copy in held:
        original[...] = copy
    return again


def _caller(name: str) -> Callable[..., int]:
    """A function of the export's arguments that hands each array over as its address."""
    kinds = _KINDS[name]
    params = [f"a{k}" for k in range(len(kinds))]
    args = []
    for k, kind in enumerate(kinds):
        if kind in _ARRAYS:
            args.append(f"_id(a{k})")
        elif kind == "f":
            args.append(f"float(a{k})")
        elif kind == "n":
            args.append(f"int(a{k})")
        else:
            args.append(f"(1 if a{k} else 0)")
    source = (
        f"def {name}({', '.join(params)}):\n"
        f"    code = _export({', '.join(args)})\n"
        f"    if code < 0:\n"
        f"        code = _again(_export, {name!r}, {kinds!r}, ({', '.join(params)},), code)\n"
        f"    return code\n"
    )
    scope = {"_id": id, "_export": getattr(_EXPORTS, name), "_again": _again}
    exec(source, scope)
    return scope[name]  # type: ignore[no-any-return]


def install(namespace: dict[str, Any]) -> None:
    """The kernels of ``cadence.fused`` under their names and arguments, made by the module."""
    call = {name: _caller(name) for name in _KINDS}
    settle_rows, residual_rows, solve_rows = (
        call["settle_rows"], call["residual_rows"], call["solve_rows"])
    fast_solve, publish_rows, kept_length = (
        call["fast_solve"], call["publish_rows"], call["kept_length"])
    kept_length_all, finite = call["kept_length_all"], call["rows_finite"]

    def rows_finite(x: np.ndarray) -> bool:
        return finite(x) == 1

    def _standing(drive: np.ndarray, bias: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.add(drive, bias)
        return out

    def _publish(v: np.ndarray, s: np.ndarray, kept: np.ndarray, has_kept: bool, slope: float,
                 threshold: float, rest: float, leak: float) -> bool:
        return publish_rows(v, s, kept, has_kept, slope, threshold, rest, leak) == 1

    def _kernel(v: np.ndarray, a: np.ndarray, s: np.ndarray, standing: np.ndarray,
                flat: np.ndarray, pairs: np.ndarray, flat_apical: np.ndarray,
                apical_pairs: np.ndarray, starts: np.ndarray, *rest: Any,
                on_threads: bool = False) -> tuple[int, np.ndarray, np.ndarray]:
        batch, n = v.shape
        change = np.zeros(batch)
        pair_post = rest[1]
        kept = np.zeros(kept_length(starts, pair_post, pairs, apical_pairs, batch, n))
        taken = settle_rows(v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts,
                            *rest, change, kept)
        return taken, change, kept

    def _residual_kernel(v: np.ndarray, a: np.ndarray, s: np.ndarray, standing: np.ndarray,
                         flat: np.ndarray, pairs: np.ndarray, flat_apical: np.ndarray,
                         apical_pairs: np.ndarray, starts: np.ndarray, *rest: Any
                         ) -> np.ndarray:
        given, in_place = rest[-3], rest[-1]
        kept: np.ndarray
        if in_place:
            kept = given
        else:
            batch, n = v.shape
            kept = np.zeros(kept_length(starts, rest[1], pairs, apical_pairs, batch, n))
        residual_rows(v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, *rest,
                      kept)
        return kept

    def _solve_rows(v: np.ndarray, a: np.ndarray, s: np.ndarray, standing: np.ndarray,
                    flat: np.ndarray, pairs: np.ndarray, flat_apical: np.ndarray,
                    apical_pairs: np.ndarray, starts: np.ndarray, *rest: Any
                    ) -> tuple[int, int, int, np.ndarray, np.ndarray]:
        batch, n = v.shape
        length = kept_length(starts, rest[1], pairs, apical_pairs, batch, n)
        change, first, second = np.zeros(batch), np.zeros(length), np.zeros(length)
        report = np.zeros(4, dtype=np.int64)
        solve_rows(v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, *rest,
                   change, first, second, report)
        given = rest[-7]  # (given, has_given, in_place, has_apical, fraction, floor2, tiny)
        kept = (given, first, second)[int(report[3])]
        return int(report[0]), int(report[1]), int(report[2]), change, kept

    def _fast_solve(v: np.ndarray, s: np.ndarray, standing: np.ndarray, flat: np.ndarray,
                    starts: np.ndarray, pair_pre: np.ndarray, pair_post: np.ndarray,
                    offset: np.ndarray, slope: float, threshold: float, rest: float,
                    leak: float, budget: int, chunk: int, tolerance: float, settled: float,
                    out: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
        batch, n = v.shape
        kept = np.zeros(kept_length_all(starts, pair_post, batch, n))
        report = np.zeros(4, dtype=np.int64)
        fast_solve(v, s, standing, flat, starts, pair_pre, pair_post, offset, slope, threshold,
                   rest, leak, budget, chunk, tolerance, settled, out, report, kept)
        return int(report[0]), int(report[1]), int(report[2]), int(report[3]), kept

    group_power, tied, choose = call["group_power"], call["tied"], call["clear_choice"]

    def _group_power(values: np.ndarray, group: np.ndarray, size: int
                     ) -> tuple[np.ndarray, np.ndarray]:
        power, count = np.zeros(size), np.zeros(size)
        group_power(values, group, power, count)
        return power, count

    def _tied(delta: np.ndarray, partner: np.ndarray, efficacy: np.ndarray,
              gain_pre: np.ndarray, cap: float, out_delta: np.ndarray, out_scale: np.ndarray,
              out_weight: np.ndarray) -> tuple[int, int, int]:
        report = np.zeros(3, dtype=np.int64)
        tied(delta, partner, efficacy, gain_pre, cap, out_delta, out_scale, out_weight, report)
        return int(report[0]), int(report[1]), int(report[2])

    def _clear_choice(s: np.ndarray, temperature: float) -> tuple[np.ndarray, bool]:
        choice = np.empty(len(s), dtype=np.int64)
        clear = choose(s, temperature, choice)
        return choice, clear == 1

    kernels: dict[str, Any] = {
        "rows_finite": rows_finite, "_standing": _standing, "_publish": _publish,
        "_kernel": _kernel, "_residual_kernel": _residual_kernel, "_solve_rows": _solve_rows,
        "_fast_solve": _fast_solve, "_trace_step": call["trace_step"], "_place": call["place"],
        "_pick": call["pick"], "_group_power": _group_power, "_divisor": call["divisor"],
        "_plain": call["plain"], "_moments": call["moments"], "_step": call["step"],
        "_tied": _tied, "_add_episode_terms": call["add_episode_terms"],
        "_clear_choice": _clear_choice,
    }
    for name in ("_kernel", "_residual_kernel", "_place", "_pick", "_divisor", "_plain",
                 "_moments", "_step", "_tied"):
        kernels[name + "_parallel"] = kernels[name]  # one thread: the serial build
    kernels["_sweeps"], kernels["_check"] = kernels["_kernel"], kernels["_residual_kernel"]
    namespace.update(kernels)
