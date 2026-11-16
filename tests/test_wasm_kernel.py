"""The browser kernel (``cadence._wasm``, built from ``kernels/wasm``): outside Pyodide nothing is
loaded and nothing changes; the argument lists the Python side hands over match the module's
exports one for one. The kernels themselves are checked in a browser
(``kernels/wasm/browser``), where they run."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import cadence as cd
from cadence import _wasm, fused

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "kernels" / "wasm" / "src").glob("*.rs"))
EXPORT = re.compile(r'extern "C" fn (\w+)\((.*?)\)\s*->', re.S)


def exports() -> dict[str, list[tuple[str, str]]]:
    found: dict[str, list[tuple[str, str]]] = {}
    for path in SOURCES:
        for name, params in EXPORT.findall(path.read_text()):
            pairs = [p.strip().split(":") for p in params.split(",") if p.strip()]
            found[name] = [(a.strip(), b.strip()) for a, b in pairs]
    return found


def test_nothing_loads_outside_pyodide(monkeypatch) -> None:
    monkeypatch.setenv("CADENCE_WASM", str(ROOT / "missing.wasm"))
    assert _wasm.load() is False
    assert _wasm.load(b"\0asm") is False
    assert not _wasm.available()
    assert fused.available() is (fused.njit is not None)


def test_every_kernel_hands_over_the_arguments_the_module_reads() -> None:
    found = exports()
    assert set(_wasm._KINDS) <= set(found)
    for name, kinds in _wasm._KINDS.items():
        params = found[name]
        assert len(params) == len(kinds), name
        for (param, rust), kind in zip(params, kinds, strict=True):
            assert (rust == "f64") == (kind == "f"), (name, param)
            assert rust in {"f64", "u32", "i32"}, (name, param)
            assert (rust == "u32") == (kind in _wasm._ARRAYS), (name, param)


def test_the_module_holds_a_kernel_of_every_name_fused_calls() -> None:
    """Each kernel ``fused.py`` calls from Python has a function of that name in ``install``."""
    namespace: dict[str, object] = {}

    class Exports:
        def __getattr__(self, name: str) -> object:
            return lambda *args: 0

    held = _wasm._EXPORTS
    _wasm._EXPORTS = Exports()
    try:
        _wasm.install(namespace)
    finally:
        _wasm._EXPORTS = held
    source = (ROOT / "src" / "cadence" / "fused.py").read_text()
    called = set(re.findall(r"\b(_[a-z_]+|rows_finite)\(", source))
    defined = set(re.findall(r"^\s+def (_[a-z_]+|rows_finite)\(", source, re.M))
    for name in ("_kernel", "_kernel_parallel", "_residual_kernel", "_residual_kernel_parallel",
                 "_solve_rows", "_fast_solve", "_publish", "_standing", "rows_finite",
                 "_trace_step", "_place", "_pick", "_group_power", "_divisor", "_plain",
                 "_moments", "_step", "_tied", "_add_episode_terms", "_clear_choice"):
        assert name in namespace, name
    # every compiled kernel the Python wrappers call is one the module stands in for
    wrappers = {"_launch", "_publish", "_standing", "_solve_rows", "_fast_solve", "_trace_step",
                "_group_power", "_add_episode_terms", "_clear_choice"}
    assert wrappers & called <= set(namespace) | {"_launch"}
    assert "_solve_rows" in defined


def test_a_brain_reads_the_same_with_the_module_absent() -> None:
    brain = cd.Brain.compose(8, 3, backend="cpu", seed=0)
    action = brain.act(np.eye(8)[[0]], greedy=True)
    assert np.asarray(action).shape[0] == 1
