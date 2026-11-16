"""The compiled kernels of the brain's parts around the settle, each one call of the Python it
stands for, to the bit: the map of things' settle and the bound records' read.

The map. ``Map._settle`` makes a drive, a mask and a start from the map's carried state and
settles the map's graph for one or two sweeps through ``Brain.settle_batch``. At the map's size
the layers of that call cost several times the sweeps. ``_map_settle`` makes the same drive,
mask and start in the same order and runs the same sweeps (``cadence.fused``'s settle kernel),
to the bit; the Python of ``Map._settle`` stays the reference, and ``CADENCE_FUSED=0`` or a
trace takes it.

The bound records. ``BoundRecords.recall`` reads the records a cue reaches (``_taking``) and the
fields they hold alike (``_agreed``) in a few NumPy calls a field and a record. ``_bound_recall``
makes both in one call, summing as NumPy sums under Pyodide, where no BLAS runs: a dot product
and a vector's norm in order, a row's sum pairwise. Natively NumPy's dot products run in BLAS,
so there the compiled read agrees with the Python within rounding.

Numba compiles the kernels; under Pyodide the WebAssembly module (``cadence._wasm``) holds them.
A kernel the Python of its part does not reach stays unused.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import fused

__all__ = ["available", "bound_recall", "map_settle"]

# the kernel's code for a drive or a start that is not finite, which ``settle_batch`` refuses
NOT_FINITE = -2000
_NO_FLAGS = np.zeros(1, dtype=bool)
_NO_VALUES = np.zeros(1)
_RUNS = ("reference", "held", "map", "where", "what", "clock", "lost", "doubt")

_map_settle: Any = None
_bound_recall: Any = None


def available(kernel: str = "_map_settle") -> bool:
    """Whether a part's kernel runs: the compiled passes are in use and the kernel is here."""
    return globals()[kernel] is not None and fused.compiled()


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


# -- the bound records' read

if fused.njit is not None:

    @fused._inlined
    def _block_sum(a: np.ndarray, lo: int, n: int) -> float:
        """NumPy's sum of ``a[lo:lo + n]`` for ``n <= 128``: under eight in order from -0.0,
        else eight running sums combined in pairs, then the remainder."""
        if n < 8:
            run = -0.0
            for i in range(lo, lo + n):
                run += a[i]
            return run
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        stop = n - n % 8
        while i < stop:
            j = lo + i
            r0 += a[j]
            r1 += a[j + 1]
            r2 += a[j + 2]
            r3 += a[j + 3]
            r4 += a[j + 4]
            r5 += a[j + 5]
            r6 += a[j + 6]
            r7 += a[j + 7]
            i += 8
        res: float = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res

    @fused._compiled
    def _np_sum(a: np.ndarray, n: int) -> float:
        """NumPy's sum of ``a[:n]`` (``pairwise_sum``): runs of more than 128 split at half
        their length rounded down to a multiple of eight, walked without recursion as
        ``fused._square_sum`` walks them."""
        if n <= 128:
            return _block_sum(a, 0, n)
        frames = np.empty((3, 64), dtype=np.int64)
        left = np.empty(64)
        sp = 0
        frames[0, 0], frames[1, 0], frames[2, 0] = 0, n, 0
        while True:
            m = frames[1, sp]
            if m > 128 and frames[2, sp] == 0:
                half = m // 2
                half -= half % 8
                frames[2, sp] = 1
                sp += 1
                frames[0, sp], frames[1, sp], frames[2, sp] = frames[0, sp - 1], half, 0
                continue
            value = _block_sum(a, frames[0, sp], m)
            while True:
                sp -= 1
                if sp < 0:
                    return value
                if frames[2, sp] == 1:
                    left[sp] = value
                    frames[2, sp] = 2
                    half = frames[1, sp] // 2
                    half -= half % 8
                    sp += 1
                    frames[0, sp] = frames[0, sp - 1] + half
                    frames[1, sp] = frames[1, sp - 1] - half
                    frames[2, sp] = 0
                    break
                value = left[sp] + value

    @fused._inlined
    def _np_dot(a: np.ndarray, b: np.ndarray, n: int) -> float:
        """NumPy's dot product of two vectors without BLAS: in order from 0.0."""
        s = 0.0
        for i in range(n):
            s += a[i] * b[i]
        return s

    @fused._inlined
    def _unit_into(a: np.ndarray, n: int, out: np.ndarray) -> None:
        norm = np.sqrt(_np_dot(a, a, n))
        for i in range(n):
            out[i] = a[i] / norm if norm > 1e-12 else a[i]

    @fused._compiled
    def _numba_bound_recall(
        slots: np.ndarray, keys: np.ndarray, code: np.ndarray, deep: np.ndarray,
        kept: np.ndarray, has_keep: bool, gone: np.ndarray, has_forget: bool,
        values: np.ndarray, has_voice: bool, v0: int, v1: int, paid: np.ndarray,
        refused: np.ndarray, fields: np.ndarray, voice_field: int, known_at: float,
        match_at: float, heads: int, near: float, agree: float, most: float, out: np.ndarray,
        present: np.ndarray,
    ) -> int:
        """``BoundRecords._agreed(_taking(code), row)`` (``bound_rs``'s twin): each field's mean
        into ``out`` at its columns, a one into ``present``; returns the fields that came
        back."""
        kw = keys.shape[1]
        inputs = values.shape[1]
        out[:] = 0.0
        present[:] = 0
        k = slots.shape[0]
        code_norm = np.sqrt(_np_dot(code, code, kw))
        taken = np.empty(k, dtype=np.int64)
        weights = np.empty(k)
        squares = np.empty(max(kw, 1))
        count = 0
        for q in range(k):
            s = slots[q]
            known = deep[s] >= known_at
            if has_keep:
                known = known or kept[s]
            if has_voice:
                said = False
                for i in range(v0, v1):
                    said = said or values[s, i] != 0.0
                if said:
                    known = paid[s] > 0.0 or refused[s]
            if has_forget:
                known = known and not gone[s]
            if not known:
                continue
            for i in range(kw):
                squares[i] = keys[s, i] * keys[s, i]
            norm = np.sqrt(_np_sum(squares, kw)) * code_norm
            dot = _np_dot(keys[s], code, kw)
            cosine = dot / (norm if norm > 1e-12 else 1e-12) if norm > 0.0 else 0.0
            if cosine >= match_at:
                taken[count] = s
                weights[count] = cosine
                count += 1
        if count == 0:
            return 0
        holding = np.empty(count, dtype=np.int64)
        group_of = np.empty(count, dtype=np.int64)
        firsts = np.empty(count, dtype=np.int64)
        picked = np.empty(count)
        ua = np.empty(max(inputs, 1))
        ub = np.empty(max(inputs, 1))
        mean = np.empty(max(inputs, 1))
        back = 0
        for f in range(fields.shape[0]):
            if f == voice_field:
                continue
            first, last = fields[f, 0], fields[f, 1]
            width = last - first
            held = 0
            for q in range(count):
                hold = False
                for i in range(first, last):
                    hold = hold or values[taken[q], i] != 0.0
                if hold:
                    holding[held] = q
                    held += 1
            if held == 0:
                continue
            for a in range(1, held):  # the nearest first: a stable sort by falling cosine
                x = holding[a]
                b = a
                while b > 0 and -weights[holding[b - 1]] > -weights[x]:
                    holding[b] = holding[b - 1]
                    b -= 1
                holding[b] = x
            held = min(held, heads)
            top = near * weights[holding[0]]
            n_kept = 0
            for a in range(held):
                if weights[holding[a]] >= top:
                    holding[n_kept] = holding[a]
                    n_kept += 1
            held = n_kept
            groups = 0
            for a in range(held):
                _unit_into(values[taken[holding[a]], first:last], width, ua)
                joined = -1
                for g in range(groups):
                    _unit_into(values[taken[firsts[g]], first:last], width, ub)
                    if _np_dot(ua, ub, width) >= agree:
                        joined = g
                        break
                if joined < 0:
                    firsts[groups] = holding[a]
                    joined = groups
                    groups += 1
                group_of[a] = joined
            best = 0
            best_mass = -np.inf
            for g in range(groups):
                m = 0
                for a in range(held):
                    if group_of[a] == g:
                        picked[m] = weights[holding[a]]
                        m += 1
                mass = _np_sum(picked, m)
                if g == 0 or mass > best_mass:
                    best_mass = mass
                    best = g
            for a in range(held):
                picked[a] = weights[holding[a]]
            if not best_mass > most * _np_sum(picked, held):
                continue
            m = 0
            for a in range(held):
                if group_of[a] == best:
                    row = taken[holding[a]]
                    if width == 1:
                        picked[m] = values[row, first]
                    elif m == 0:
                        for i in range(width):
                            mean[i] = values[row, first + i]
                    else:
                        for i in range(width):
                            mean[i] += values[row, first + i]
                    m += 1
            if width == 1:
                mean[0] = _np_sum(picked, m)
            for i in range(width):
                out[first + i] = mean[i] / m
            present[f] = 1
            back += 1
        return back

    _bound_recall = _numba_bound_recall


def bound_recall(records: Any, code: np.ndarray) -> dict[str, np.ndarray]:
    """``records._agreed(records._taking(code), row)`` in one compiled call: each field the
    records a read of ``code`` reaches hold alike, as its mean."""
    store = records.store
    if not store.count or not np.any(code):
        return {}
    code = np.ascontiguousarray(code, dtype=np.float64)
    slots = np.asarray(store._candidates(code)[0], dtype=np.int64)
    if not len(slots):
        return {}
    table = records.__dict__.get("_kernel_fields")
    if table is None:
        names = list(records.fields)
        voice = records._voice_field() if records.voice is not None else None
        spans = np.array([records.fields[f] for f in names], dtype=np.int64).reshape(-1, 2)
        vf = names.index(voice) if voice is not None else -1
        v0, v1 = records.fields[voice] if voice is not None else (0, 0)
        table = records.__dict__["_kernel_fields"] = (names, spans, vf, int(v0), int(v1))
    names, spans, vf, v0, v1 = table
    out = np.empty(records.inputs)
    present = np.empty(len(names), dtype=np.int64)
    back = _bound_recall(
        slots, store.keys, code, store.deep, store.kept, bool(store.keep), store.gone,
        bool(store.forget), store.values, records.voice is not None, v0, v1, records.paid,
        records.refused, spans, vf, records.KNOWN, records.MATCH, records.HEADS, records.NEAR,
        records.AGREE, records.MOST, out, present,
    )
    if back < 0:
        raise RuntimeError(f"cadence: the bound records' kernel refused its arrays ({back})")
    return {name: out[span[0]:span[1]].copy() for name, span, on in zip(names, spans, present,
                                                                         strict=True) if on}
