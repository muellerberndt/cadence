"""The map of things' settle in one compiled call.

``Map._settle`` makes a drive, a mask and a start from the map's carried state and settles the
map's graph for one or two sweeps through ``Brain.settle_batch``. At the map's size the layers
of that call cost several times the sweeps. ``_map_settle`` makes the same drive, mask and start
in the same order and runs the same sweeps (``cadence.fused``'s settle kernel), to the bit; the
Python of ``Map._settle`` stays the reference, and ``CADENCE_FUSED=0`` or a trace takes it.
Numba compiles the kernel; under Pyodide the WebAssembly module (``cadence._wasm``) holds it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import fused

__all__ = ["available", "map_settle"]

# the kernel's code for a drive or a start that is not finite, which ``settle_batch`` refuses
NOT_FINITE = -2000
_NO_FLAGS = np.zeros(1, dtype=bool)
_NO_VALUES = np.zeros(1)
_RUNS = ("reference", "held", "map", "where", "what", "clock", "lost", "doubt")

_map_settle: Any = None


def available() -> bool:
    """Whether the map's settle runs compiled: the compiled passes are in use and the kernel is
    here."""
    return _map_settle is not None and fused.compiled()


if fused.njit is not None:

    @fused._compiled
    def _numba_map_settle(
        flat: np.ndarray, pairs: np.ndarray, flat_apical: np.ndarray, apical_pairs: np.ndarray,
        starts: np.ndarray, pair_pre: np.ndarray, pair_post: np.ndarray, offset: np.ndarray,
        freezable: np.ndarray, bias: np.ndarray, dt: float, slope: float, threshold: float,
        rest: float, leak: float, has_apical: bool, pool_of: np.ndarray,
        pool_index: np.ndarray, pool_starts: np.ndarray, pool_run: np.ndarray, fraction: float,
        floor2: float, tiny: float, potential: np.ndarray, carried: np.ndarray,
        has_carried: bool, write: np.ndarray, has_write: bool, clock: np.ndarray,
        lost: np.ndarray, doubt: np.ndarray, fresh: np.ndarray, has_fresh: bool,
        kept: np.ndarray, has_kept: bool, sure: np.ndarray, has_sure: bool, cue: np.ndarray,
        has_cue: bool, runs: np.ndarray, sweeps: int, out: np.ndarray,
    ) -> int:
        """``Map._settle`` without a trace: its drive, mask and start, then ``fused._kernel``.
        ``runs`` holds the start and stop of each run of ``_RUNS``; ``out`` (1, n) receives the
        settled potentials. Returns the sweeps, or ``NOT_FINITE``."""
        n = starts[starts.shape[0] - 1]
        cells, width = potential.shape
        size = cells * width
        pot = potential.ravel()
        drive = np.zeros((1, n))
        mask = np.ones((1, n))
        r0, h0, m0, m1 = runs[0], runs[2], runs[4], runs[5]
        c0, c1, l0, d0 = runs[10], runs[11], runs[12], runs[14]
        held = np.empty(size)
        for i in range(size):
            held[i] = pot[i] * (1.0 if carried[i] else 0.0) if has_carried else pot[i]
        if has_cue:
            for k in range(width):
                drive[0, r0 + k] = cue[k]
            for i in range(size):
                drive[0, h0 + i] = held[i]
        else:
            for i in range(size):
                drive[0, m0 + i] = held[i] + write[i] if has_write else held[i]
            if has_carried:
                for i in range(size):
                    alive = (carried[i] or write[i] != 0.0) if has_write else carried[i]
                    mask[0, m0 + i] = 1.0 if alive else 0.0
        if has_kept:
            for c in range(cells):
                on = 1.0 if kept[c] else 0.0
                for u in range(width):
                    mask[0, m0 + c * width + u] *= on
        for c in range(cells):
            drive[0, c0 + c] = clock[c]
            drive[0, d0 + c] = doubt[c]
        for u in range(lost.shape[0]):
            drive[0, l0 + u] = lost[u]
        if has_fresh:
            for c in range(cells):
                mask[0, c0 + c] = 0.0 if fresh[c] else 1.0
        if has_sure:
            for c in range(cells):
                mask[0, d0 + c] = 0.0 if sure[c] else 1.0
        good = True
        any_v = False
        masked = False
        for i in range(n):
            x = drive[0, i]
            good = good and x - x == 0.0
            if c0 <= i < c1:
                x += 1.0
            if has_cue and m0 <= i < m1:
                x = 0.0
            x *= mask[0, i]
            out[0, i] = x
            good = good and x - x == 0.0
            any_v = any_v or x != 0.0
            masked = masked or mask[0, i] != 1.0
        if not good:
            return NOT_FINITE
        s = np.empty((1, n))
        if not any_v:
            at_rest = fused._act_scalar(0.0, slope, threshold, rest, leak)
            for i in range(n):
                s[0, i] = at_rest
        else:
            for i in range(n):
                s[0, i] = fused._act_scalar(out[0, i], slope, threshold, rest, leak)
        if masked:
            for i in range(n):
                s[0, i] *= mask[0, i]
        standing = np.empty((1, n))
        for i in range(n):
            standing[0, i] = drive[0, i] + bias[i]
        taken, _, _ = fused._kernel(
            out, np.zeros((1, n)), s, standing, flat, pairs, flat_apical, apical_pairs, starts,
            pair_pre, pair_post, offset, freezable, mask, masked, dt, slope, threshold, rest,
            leak, False, 0.0, 1.0, False, np.zeros((1, 1)), np.zeros(n),
            np.full(n, -1, dtype=np.int64), 0, 0.0, np.zeros(n), np.ones(1), np.zeros((1, 1)),
            np.zeros(n), has_apical, pool_of, pool_index, pool_starts, pool_run, fraction,
            floor2, tiny, np.zeros((1, 1)), False, sweeps, 0.0, False, np.zeros(1), False,
        )
        return int(taken)

    _map_settle = _numba_map_settle


def _graph_arrays(graph: Any) -> tuple[Any, ...]:
    """The kernel's arguments a map's graph fixes, made once per graph."""
    held = graph.__dict__.get("_map_kernel")
    blocks = graph._fused_blocks()
    if held is not None and held[0] is blocks[0]:
        return held[1]  # type: ignore[no-any-return]
    flat, apical = blocks
    lay, n = graph.layout, graph.connectome.n
    pairs, flat_apical, apical_pairs, has_apical, pool_of, pool_index, pool_starts, pool_run, \
        fraction, floor2, tiny, _, _ = fused._compartment(apical, lay, flat, None, 1, n)
    dt, slope, threshold, rest, leak, _, _ = fused._rule(graph.neuron_model)
    arrays = (flat, pairs, flat_apical, apical_pairs, lay.starts, lay.pair_pre, lay.pair_post,
              lay.offset, ~fused._fed_ranges(lay), np.ascontiguousarray(graph.bias, dtype=float),
              dt, slope, threshold, rest, leak, bool(has_apical), pool_of, pool_index,
              pool_starts, pool_run, fraction, floor2, tiny)
    graph.__dict__["_map_kernel"] = (flat, arrays)
    return arrays


def runs_of(runs: dict[str, slice]) -> np.ndarray | None:
    """The kernel's ``runs``: each run's start and stop; None where a run is not a slice."""
    out: list[int] = []
    for name in _RUNS:
        run = runs[name]
        if not isinstance(run, slice) or run.step not in (None, 1):
            return None
        out.extend((int(run.start), int(run.stop)))
    return np.array(out, dtype=np.int64)


def map_settle(graph: Any, runs: np.ndarray, potential: np.ndarray, clock: np.ndarray,
               lost: np.ndarray, doubt: np.ndarray, write: np.ndarray | None,
               carried: np.ndarray | None, fresh: np.ndarray | None, kept: np.ndarray | None,
               sure: np.ndarray | None, cue: np.ndarray | None, sweeps: int) -> np.ndarray:
    """The settled potentials of ``Map._settle``, every neuron's, in one compiled call."""
    def flags(x: np.ndarray | None) -> np.ndarray:
        return _NO_FLAGS if x is None else np.ascontiguousarray(x, dtype=bool).reshape(-1)

    def values(x: np.ndarray | None) -> np.ndarray:
        return _NO_VALUES if x is None else np.ascontiguousarray(x, dtype=float).reshape(-1)

    out = np.empty((1, graph.connectome.n))
    taken = _map_settle(
        *_graph_arrays(graph), np.ascontiguousarray(potential, dtype=float), flags(carried),
        carried is not None, values(write), write is not None, values(clock), values(lost),
        values(doubt), flags(fresh), fresh is not None, flags(kept), kept is not None,
        flags(sure), sure is not None, values(cue), cue is not None, runs, int(sweeps), out,
    )
    if taken == NOT_FINITE:
        raise ValueError("drive must be finite and the brain must contain neurons")
    settled: np.ndarray = out[0]
    return settled
