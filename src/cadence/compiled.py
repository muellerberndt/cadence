"""The compiled kernels of the brain's parts around the settle, each one call of the Python it
stands for, to the bit: the map of things' settle, pieces, moves, written values and a report's
claims and writes, and the bound records' read.

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

import math
from typing import Any

import numpy as np

from . import fused

__all__ = ["available", "bound_recall", "map_moves", "map_pieces", "map_report", "map_settle",
           "map_write_values"]

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
              fused.sparse_offset(lay, apical), ~fused._fed_ranges(lay),
              np.ascontiguousarray(graph.bias, dtype=float), dt, slope, threshold, rest, leak,
              bool(has_apical), pool_of, pool_index, pool_starts, pool_run, fraction, floor2,
              tiny)
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


# -- the map's pieces: connected cells, in the order a Python set pops their seeds

_map_pieces: Any = None

if fused.njit is not None:

    @fused._compiled
    def _set_table(keys: np.ndarray, n: int) -> np.ndarray:
        """The table of CPython's set built from the ascending small ints ``keys[:n]`` (their
        hash is the int): each key at its slot, -1 elsewhere (``set_add_entry``, its linear
        probes and perturbation; ``set_table_resize`` with ``set_insert_clean``)."""
        size = 8
        table = np.full(size, -1, dtype=np.int64)
        fill = 0
        used = 0
        for q in range(n):
            key = keys[q]
            mask = size - 1
            i = key & mask
            perturb = key
            placed = False
            while not placed:
                probes = 9 if i + 9 <= mask else 0
                e = i
                for _ in range(probes + 1):
                    if table[e] == -1:
                        table[e] = key
                        fill += 1
                        used += 1
                        placed = True
                        break
                    e += 1
                if not placed:
                    perturb >>= 5
                    i = (i * 5 + 1 + perturb) & mask
            if fill * 5 >= mask * 3:
                grown = 8
                while grown <= used * 4:
                    grown <<= 1
                new = np.full(grown, -1, dtype=np.int64)
                newmask = grown - 1
                for s in range(size):
                    k = table[s]
                    if k == -1:
                        continue
                    i = k & newmask
                    perturb = k
                    while True:
                        if new[i] == -1:
                            new[i] = k
                            break
                        done = False
                        if i + 9 <= newmask:
                            for j in range(1, 10):
                                if new[i + j] == -1:
                                    new[i + j] = k
                                    done = True
                                    break
                        if done:
                            break
                        perturb >>= 5
                        i = (i * 5 + 1 + perturb) & newmask
                table = new
                size = grown
                fill = used
        return table

    @fused._compiled
    def _numba_map_pieces(active: np.ndarray, nx: int, ny: int, cells: np.ndarray,
                          bounds: np.ndarray) -> int:
        """``Map._pieces``: cells that touch, corners too, each piece ascending, the pieces in
        the order a Python set of the active cells pops their seeds; into ``cells`` piece after
        piece, ``bounds`` where each begins. Returns the number of pieces."""
        count = nx * ny
        keys = np.flatnonzero(active).astype(np.int64)
        n = keys.shape[0]
        bounds[0] = 0
        if n == 0:
            return 0
        label = np.full(count, -1, dtype=np.int64)
        stack = np.empty(n, dtype=np.int64)
        pieces = 0
        for q in range(n):
            seed = keys[q]
            if label[seed] >= 0:
                continue
            label[seed] = pieces
            top = 1
            stack[0] = seed
            while top > 0:
                top -= 1
                cell = stack[top]
                i, j = cell // ny, cell % ny
                for a in range(i - 1, i + 2):
                    for b in range(j - 1, j + 2):
                        if 0 <= a < nx and 0 <= b < ny:
                            other = a * ny + b
                            if active[other] and label[other] < 0:
                                label[other] = pieces
                                stack[top] = other
                                top += 1
            pieces += 1
        start = np.zeros(pieces + 1, dtype=np.int64)
        for q in range(n):
            start[label[keys[q]] + 1] += 1
        for p in range(pieces):
            start[p + 1] += start[p]
        fill = start[:pieces].copy()
        grouped = np.empty(n, dtype=np.int64)
        for q in range(n):
            p = label[keys[q]]
            grouped[fill[p]] = keys[q]
            fill[p] += 1
        table = _set_table(keys, n)
        size = table.shape[0]
        taken = np.zeros(pieces, dtype=np.bool_)
        finger = 0
        at = 0
        for out in range(pieces):
            s = finger & (size - 1)
            while table[s] == -1 or taken[label[table[s]]]:
                s = (s + 1) & (size - 1)
            p = label[table[s]]
            taken[p] = True
            finger = s + 1
            for q in range(start[p], start[p + 1]):
                cells[at] = grouped[q]
                at += 1
            bounds[out + 1] = at
        return pieces

    _map_pieces = _numba_map_pieces


def map_pieces(active: np.ndarray, nx: int, ny: int) -> list[np.ndarray]:
    """``Map._pieces(active)`` in one compiled call: each piece's cells (NumPy's index type)."""
    flags = np.ascontiguousarray(active, dtype=bool).reshape(-1)
    count = int(nx) * int(ny)
    cells = np.empty(count, dtype=np.int64)
    bounds = np.empty(count + 1, dtype=np.int64)
    found = int(_map_pieces(flags, int(nx), int(ny), cells, bounds))
    if found < 0:
        raise RuntimeError(f"cadence: the map's pieces kernel refused its arrays ({found})")
    held = cells[: bounds[found]].astype(np.intp)
    return [held[bounds[p]:bounds[p + 1]] for p in range(found)]


# -- the map's moves: how far a moment moved each thing

_map_moves: Any = None

if fused.njit is not None:

    @fused._compiled
    def _numba_map_moves(potential: np.ndarray, expected: np.ndarray, codes: np.ndarray,
                         centers: np.ndarray, floor: float, stride: float, step: float, nx: int,
                         ny: int, out: np.ndarray) -> int:
        """``Map.moves(expected)`` into ``out`` (the module's ``map_moves``, in its order)."""
        count, width = potential.shape
        kinds, units = codes.shape
        after = np.maximum(np.tanh(potential / 2.0), 0.0)
        before = np.maximum(np.tanh(expected / 2.0), 0.0)
        held_after = np.zeros(units, dtype=np.bool_)
        held_before = np.zeros(units, dtype=np.bool_)
        for u in range(units):
            for c in range(count):
                held_after[u] = held_after[u] or after[c, u] > floor
                held_before[u] = held_before[u] or before[c, u] > floor
        out[:] = 0.0
        own = np.empty(count)
        cells = np.empty((2, count), dtype=np.int64)
        bounds = np.empty((2, count + 1), dtype=np.int64)
        middles = np.empty((2, count, 2))
        picked = np.empty(max(count, 1))
        found = np.zeros(2, dtype=np.int64)
        free = np.zeros(count, dtype=np.bool_)
        for kind in range(kinds):
            present_after = True
            present_before = True
            for u in range(units):
                if codes[kind, u] > 0.0:
                    present_after = present_after and held_after[u]
                    present_before = present_before and held_before[u]
            if not (present_after or present_before):
                continue
            for side in range(2):
                sheet = after if side == 0 else before
                for c in range(count):
                    least = np.inf
                    first = True
                    for u in range(units):
                        if codes[kind, u] != 0.0:
                            x = sheet[c, u]
                            if first or x < least or x != x:
                                least = x
                                first = False
                    own[c] = least
                found[side] = _numba_map_pieces(own > floor, nx, ny, cells[side], bounds[side])
                for p in range(found[side]):
                    lo, hi = bounds[side, p], bounds[side, p + 1]
                    mx = 0.0
                    my = 0.0
                    for q in range(lo, hi):
                        c = cells[side, q]
                        mx += own[c] * centers[c, 0]
                        my += own[c] * centers[c, 1]
                        picked[q - lo] = own[c]
                    total = _np_sum(picked, hi - lo)
                    middles[side, p, 0] = mx / total
                    middles[side, p, 1] = my / total
            for j in range(found[1]):
                free[j] = True
            for p in range(found[0]):
                best = np.inf
                nearest = -1
                for j in range(found[1]):
                    if not free[j]:
                        continue
                    dx = middles[0, p, 0] - middles[1, j, 0]
                    dy = middles[0, p, 1] - middles[1, j, 1]
                    gap = np.sqrt(0.0 + dx * dx + dy * dy)
                    if nearest < 0 or gap < best:
                        best = gap
                        nearest = j
                if nearest >= 0 and best <= stride:
                    free[nearest] = False
                    moved = best / step
                    for q in range(bounds[1, nearest], bounds[1, nearest + 1]):
                        c = cells[1, q]
                        out[c] = np.maximum(out[c], moved)
                else:
                    moved = 1.0
                for q in range(bounds[0, p], bounds[0, p + 1]):
                    c = cells[0, q]
                    out[c] = np.maximum(out[c], moved)
            for j in range(found[1]):
                if free[j]:
                    for q in range(bounds[1, j], bounds[1, j + 1]):
                        c = cells[1, q]
                        out[c] = np.maximum(out[c], 1.0)
        return 0

    _map_moves = _numba_map_moves


def map_moves(held: Any, expected: np.ndarray) -> np.ndarray:
    """``Map.moves(expected)`` in one compiled call."""
    nx, ny = held.cells
    out = np.empty(held.count)
    code = int(_map_moves(np.ascontiguousarray(held.potential, dtype=np.float64),
                          np.ascontiguousarray(expected, dtype=np.float64),
                          np.ascontiguousarray(held.codes, dtype=np.float64),
                          np.ascontiguousarray(held.centers, dtype=np.float64),
                          float(held._floor), float(held.stride), float(held.step), int(nx),
                          int(ny), out))
    if code < 0:
        raise RuntimeError(f"cadence: the map's moves kernel refused its arrays ({code})")
    return out


# -- the map's values a report writes

_map_write_values: Any = None

if fused.njit is not None:

    @fused._inlined
    def _of_kind_at(activity: np.ndarray, codes: np.ndarray, kind: int, c: int) -> float:
        least = np.inf
        first = True
        for u in range(codes.shape[1]):
            if codes[kind, u] != 0.0:
                x = activity[c, u]
                if first or x < least or x != x:
                    least = x
                    first = False
        return least

    @fused._compiled
    def _numba_map_write_values(
        kind: int, near_cells: np.ndarray, near_bounds: np.ndarray, weights: np.ndarray,
        named: np.ndarray, named_mask: np.ndarray, activity: np.ndarray, claimed: np.ndarray,
        write: np.ndarray, codes: np.ndarray, level: float, floor: float, nx: int, ny: int,
    ) -> int:
        """``Map._write_values`` in place on ``claimed`` and ``write`` (the module's
        ``map_write_values``, in its order)."""
        count = nx * ny
        width = activity.shape[1]
        kinds, units = codes.shape
        valued = width - units
        pieces = near_bounds.shape[0] - 1
        mark = np.zeros(count, dtype=np.bool_)
        around = np.zeros(count, dtype=np.bool_)
        taken = np.empty(count * (pieces + 1) + 8, dtype=np.int64)
        taken_n = 0
        for p in range(pieces):
            mark[:] = False
            around[:] = False
            for q in range(near_bounds[p], near_bounds[p + 1]):
                cell = near_cells[q]
                mark[cell] = True
                i, j = cell // ny, cell % ny
                for a in range(i - 1, i + 2):
                    for b in range(j - 1, j + 2):
                        around[min(max(a, 0), nx - 1) * ny + min(max(b, 0), ny - 1)] = True
            for c in range(count):
                if around[c]:
                    holds = False
                    for u in range(units):
                        if codes[kind, u] != 0.0:
                            holds = holds or activity[c, u] > 0.0
                    if holds:
                        mark[c] = True
            for c in range(count):
                if mark[c]:
                    taken[taken_n] = c
                    taken_n += 1
        others = np.zeros(count, dtype=np.bool_)
        for other in range(kinds):
            if other == kind:
                continue
            present = True
            for u in range(units):
                if codes[other, u] > 0.0:
                    seen = False
                    for c in range(count):
                        seen = seen or activity[c, u] > floor
                    present = present and seen
            if present:
                for c in range(count):
                    if _of_kind_at(activity, codes, other, c) > floor:
                        others[c] = True
        held = weights > 0.0
        for q in range(taken_n):
            held[taken[q]] = True
        for c in range(count):
            if others[c]:
                held[c] = False
        amounts = np.zeros(valued)
        if taken_n > 0:
            sums = np.empty(valued + 1)
            for q in range(taken_n):
                c = taken[q]
                own = _of_kind_at(activity, codes, kind, c)
                if q == 0:
                    sums[0] = own
                    for v in range(valued):
                        sums[v + 1] = activity[c, units + v]
                else:
                    sums[0] += own
                    for v in range(valued):
                        sums[v + 1] += activity[c, units + v]
            if sums[0] > floor:
                for v in range(valued):
                    x = sums[v + 1] / sums[0]
                    if x == x:
                        x = x if x > 0.0 else 0.0
                        x = x if x < 1.0 else 1.0
                    amounts[v] = x
        for v in range(valued):
            if named_mask[v]:
                amounts[v] = named[v]
        for c in range(count):
            if held[c] or weights[c] > 0.0:
                for v in range(valued):
                    claimed[c, units + v] = True
        some = False
        for v in range(valued):
            some = some or amounts[v] != 0.0
        if some:
            square = 0.0
            for c in range(count):
                square += weights[c] * weights[c]
            norm = np.sqrt(square)
            for c in range(count):
                a = level * weights[c] / norm
                for v in range(valued):
                    write[c, units + v] += a * amounts[v]
        return 0

    _map_write_values = _numba_map_write_values


def map_write_values(held: Any, kind: int, near: list[np.ndarray], weights: np.ndarray,
                     named: dict[str, np.ndarray], activity: np.ndarray, claimed: np.ndarray,
                     write: np.ndarray) -> None:
    """``Map._write_values`` in one compiled call, in place on ``claimed`` and ``write``."""
    valued = held.width - held.units
    given, mask = np.zeros(valued), np.zeros(valued, dtype=bool)
    for group, amounts in named.items():
        run = held.values[group]
        given[run.start - held.units:run.stop - held.units] = amounts
        mask[run.start - held.units:run.stop - held.units] = True
    bounds = np.zeros(len(near) + 1, dtype=np.int64)
    bounds[1:] = np.cumsum([len(p) for p in near]) if near else bounds[1:]
    cells = (np.concatenate(near).astype(np.int64) if near else np.zeros(0, dtype=np.int64))
    nx, ny = held.cells
    status = int(_map_write_values(
        int(kind), cells, bounds, np.ascontiguousarray(weights, dtype=np.float64), given, mask,
        np.ascontiguousarray(activity, dtype=np.float64), claimed, write,
        np.ascontiguousarray(held.codes, dtype=np.float64), float(held.level),
        float(held._floor), int(nx), int(ny)))
    if status < 0:
        raise RuntimeError(f"cadence: the map's values kernel refused its arrays ({status})")


# -- a report of the map's moment: its pieces, claims and writes

_map_report: Any = None

if fused.njit is not None:

    @fused._compiled
    def _numba_map_report(
        kind: int, px: float, py: float, codes: np.ndarray, centers: np.ndarray,
        activity: np.ndarray, doubt: np.ndarray, motion: np.ndarray, has_physics: bool,
        hidden: int, ident: int, fields: int, lowx: float, lowy: float, sizex: float,
        sizey: float, floor: float, step: float, stride: float, level: float, nx: int, ny: int,
        has_values: bool, named: np.ndarray, named_mask: np.ndarray, claimed: np.ndarray,
        write: np.ndarray, fresh: np.ndarray, lost: np.ndarray, near_cells: np.ndarray,
        near_bounds: np.ndarray,
    ) -> int:
        """One report of ``Map.moment`` but its line, in place (the module's ``map_report``,
        in its order). Returns how many pieces it takes, into ``near_cells`` and
        ``near_bounds``."""
        count = nx * ny
        kinds, units = codes.shape
        first_unit = -1
        for u in range(units):
            if codes[kind, u] != 0.0:
                first_unit = u
                break
        apart = np.empty(count)
        for c in range(count):
            dx = centers[c, 0] - px
            dy = centers[c, 1] - py
            apart[c] = np.sqrt(-0.0 + dx * dx + dy * dy)
        own = np.empty(count, dtype=np.bool_)
        for c in range(count):
            own[c] = _of_kind_at(activity, codes, kind, c) > floor
        cells = np.empty(count, dtype=np.int64)
        bounds = np.empty(count + 1, dtype=np.int64)
        found = _numba_map_pieces(own, nx, ny, cells, bounds)
        nearest = np.empty(max(found, 1))
        for p in range(found):
            least = apart[cells[bounds[p]]]
            for q in range(bounds[p] + 1, bounds[p + 1]):
                x = apart[cells[q]]
                if x < least or x != x:
                    least = x
            nearest[p] = least
        near = np.empty(max(found, 1), dtype=np.int64)
        n_near = 0
        for p in range(found):
            if nearest[p] <= step:
                near[n_near] = p
                n_near += 1
        if n_near == 0:
            best = -1
            for p in range(found):
                if nearest[p] <= stride and (best < 0 or nearest[p] < nearest[best]):
                    best = p
            if best >= 0:
                near[0] = best
                n_near = 1
        doubtful = np.empty(max(found, 1), dtype=np.int64)
        gathered = np.empty(count)
        n_doubt = 0
        for p in range(found):
            lo, hi = bounds[p], bounds[p + 1]
            for q in range(lo, hi):
                gathered[q - lo] = doubt[cells[q]]
            if _np_sum(gathered, hi - lo) / (hi - lo) > 0.5:
                doubtful[n_doubt] = p
                n_doubt += 1
        if has_physics:
            theirs = np.empty(count)
            at_hidden = first_unit * fields + hidden
            at_ident = first_unit * fields + ident
            for c in range(count):
                theirs[c] = motion[c, at_ident] if motion[c, at_hidden] > 0.0 else 0.0
            mine = np.empty(count)
            n_mine = 0
            for k in range(n_near if n_near > 0 else count):
                lo = bounds[near[k]] if n_near > 0 else k
                hi = bounds[near[k] + 1] if n_near > 0 else k + 1
                for q in range(lo, hi):
                    x = theirs[cells[q]] if n_near > 0 else theirs[q]
                    if x > 0.0:
                        seen = False
                        for m in range(n_mine):
                            if mine[m] == x:
                                seen = True
                                break
                        if not seen:
                            mine[n_mine] = x
                            n_mine += 1
            kept = 0
            for k in range(n_doubt):
                p = doubtful[k]
                some = False
                carried = False
                for q in range(bounds[p], bounds[p + 1]):
                    x = theirs[cells[q]]
                    some = some or x != 0.0
                    for m in range(n_mine):
                        if mine[m] == x:
                            carried = True
                if not some or carried:
                    doubtful[kept] = p
                    kept += 1
            n_doubt = kept
            for c in range(count):
                for m in range(n_mine):
                    if mine[m] == theirs[c]:
                        for u in range(units):
                            if codes[kind, u] != 0.0:
                                claimed[c, u] = True
                        break
        taken = n_near + n_doubt
        at = 0
        near_bounds[0] = 0
        for k in range(taken):
            p = near[k] if k < n_near else doubtful[k - n_near]
            for q in range(bounds[p], bounds[p + 1]):
                near_cells[at] = cells[q]
                at += 1
            near_bounds[k + 1] = at
        if taken > 0:
            others = np.zeros(count, dtype=np.bool_)
            for other in range(kinds):
                if other == kind:
                    continue
                present = True
                for u in range(units):
                    if codes[other, u] > 0.0:
                        any_ = False
                        for c in range(count):
                            any_ = any_ or activity[c, u] > floor
                        present = present and any_
                if present:
                    for c in range(count):
                        if _of_kind_at(activity, codes, other, c) > floor:
                            others[c] = True
            around = np.empty(count, dtype=np.bool_)
            for k in range(taken):
                around[:] = False
                for q in range(near_bounds[k], near_bounds[k + 1]):
                    cell = near_cells[q]
                    for u in range(units):
                        if codes[kind, u] != 0.0:
                            claimed[cell, u] = True
                    i, j = cell // ny, cell % ny
                    for a in range(i - 1, i + 2):
                        for b in range(j - 1, j + 2):
                            around[min(max(a, 0), nx - 1) * ny + min(max(b, 0), ny - 1)] = True
                for c in range(count):
                    if not around[c] or others[c]:
                        continue
                    holds = False
                    for u in range(units):
                        if codes[kind, u] != 0.0:
                            holds = holds or activity[c, u] > 0.0
                    if holds:
                        for u in range(units):
                            if codes[kind, u] != 0.0:
                                claimed[c, u] = True
        weights = np.zeros(count)
        ax = (px - lowx) / sizex - 0.5
        ax = ax if ax > 0.0 else 0.0
        ax = ax if ax < nx - 1.0 else nx - 1.0
        ay = (py - lowy) / sizey - 0.5
        ay = ay if ay > 0.0 else 0.0
        ay = ay if ay < ny - 1.0 else ny - 1.0
        ix = min(int(ax), nx - 2)
        iy = min(int(ay), ny - 2)
        fx, fy = ax - ix, ay - iy
        cell = ix * ny + iy
        weights[cell] = 1.0 * (1.0 - fx) * (1.0 - fy)
        weights[cell + ny] = 1.0 * fx * (1.0 - fy)
        weights[cell + 1] = 1.0 * (1.0 - fx) * fy
        weights[cell + ny + 1] = 1.0 * fx * fy
        square = 0.0
        for c in range(count):
            square += weights[c] * weights[c]
        norm = np.sqrt(square)
        for c in range(count):
            w = weights[c]
            add = level * w / norm
            for u in range(units):
                if codes[kind, u] != 0.0:
                    if w > 0.0:
                        claimed[c, u] = True
                    write[c, u] += add
        if has_values:
            _numba_map_write_values(kind, near_cells[:at], near_bounds[:taken + 1], weights,
                                    named, named_mask, activity, claimed, write, codes, level,
                                    floor, nx, ny)
        for c in range(count):
            if weights[c] > 0.0:
                fresh[c] = True
        for u in range(units):
            if codes[kind, u] != 0.0:
                lost[u] = 0.0
        return taken

    _map_report = _numba_map_report

_NO_MOTION = np.zeros((1, 1))


def map_report(held: Any, kind: int, place: np.ndarray, named: dict[str, np.ndarray],
               activity: np.ndarray, claimed: np.ndarray, write: np.ndarray,
               fresh: np.ndarray, lost: np.ndarray) -> list[np.ndarray] | None:
    """One report of ``Map.moment`` in one compiled call, in place on ``claimed``, ``write``,
    ``fresh`` and ``lost``: the pieces it takes (``near``, for its line), or None where the
    Python takes the report (a place that is not finite, a sheet narrower than two cells, a
    kind with no units)."""
    px, py = float(place[0]), float(place[1])
    nx, ny = held.cells
    if not (math.isfinite(px) and math.isfinite(py)) or nx < 2 or ny < 2 or not (
            held.codes[kind] != 0).any():
        return None
    count = held.count
    room = held.__dict__.get("_report_room")
    if room is None or len(room[0]) != 2 * count:
        room = held.__dict__["_report_room"] = (np.empty(2 * count, dtype=np.int64),
                                                np.empty(2 * count + 1, dtype=np.int64))
    cells, bounds = room
    valued = held.width - held.units
    given, mask = np.zeros(valued), np.zeros(valued, dtype=bool)
    for group, amounts in named.items():
        run = held.values[group]
        given[run.start - held.units:run.stop - held.units] = amounts
        mask[run.start - held.units:run.stop - held.units] = True
    physics = held.physics is not None
    if physics:
        from .space.physics import FIELDS, HIDDEN, IDENT

        motion = held.motion.reshape(count, -1)
    else:
        motion, FIELDS, HIDDEN, IDENT = _NO_MOTION, 1, 0, 0
    found = int(_map_report(
        int(kind), px, py, np.ascontiguousarray(held.codes, dtype=np.float64),
        np.ascontiguousarray(held.centers, dtype=np.float64), activity,
        np.ascontiguousarray(held.doubt, dtype=np.float64),
        np.ascontiguousarray(motion, dtype=np.float64), physics, HIDDEN, IDENT, FIELDS,
        float(held._low[0]), float(held._low[1]), float(held.size[0]), float(held.size[1]),
        float(held._floor), float(held.step), float(held.stride), float(held.level), int(nx),
        int(ny), bool(held.values), given, mask, claimed, write, fresh, lost, cells, bounds))
    if found < 0:
        raise RuntimeError(f"cadence: the map's report kernel refused its arrays ({found})")
    taken = cells[: bounds[found]].astype(np.intp)
    return [taken[bounds[p]:bounds[p + 1]] for p in range(found)]
