"""A fused settling kernel for dense connectomes on the CPU.

The reference loop in ``settle`` does one NumPy operation per term per step
and pays interpreter and memory traffic for each. This kernel does the same
arithmetic, in the same order and the same float64, in one compiled loop:
transport as the block products of ``cadence.blocks`` (a block whose source
range did not move since the previous step is reused), then every neuron's
update, activation, adaptation, and nudge in place. ``Brain`` uses it on the CPU backend
when Numba is installed, the connectome's blocks fit ``dense_limit``, and no trajectory or
settlement recording is requested (``CADENCE_FUSED=0`` forces the NumPy loop); the
conformance check and ``settle_neuron_by_neuron`` are the reference it is
measured against. Nothing a neuron could not see enters: the kernel reads
the same synaptic input, stimulus, bias, and nudge the reference reads.

Each sweep computes the block products for the whole batch, then settles the rows: every
row's update reads only that row, so with more than one row the rows run on Numba's threads
(``NUMBA_NUM_THREADS``, ``numba.set_num_threads``), each with exactly the arithmetic of the
serial build, and the batch-wide stopping rule is read after the sweep. A graph with an
apical compartment sums its basal and its apical blocks apart and normalizes the apical
input per row and pool, in the order and summation the NumPy loop uses.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

try:
    import numba

    _numba: Any = numba
    njit: Any = numba.njit
    prange: Any = numba.prange
except ImportError:  # pragma: no cover
    _numba = njit = None
    prange = range

if TYPE_CHECKING:
    from .blocks import Layout
    from .brain import Apical, Nudge
    from .neuron import NeuronModel


__all__ = ["available", "fused_residual", "fused_settle"]

_Kernel = TypeVar("_Kernel", bound=Callable[..., Any])


def _compiled(kernel: _Kernel) -> _Kernel:
    """``numba.njit(cache=True)``, keeping the kernel's annotated signature for the checker."""
    assert njit is not None
    return cast(_Kernel, njit(cache=True)(kernel))


def _rows_builds(kernel: _Kernel) -> tuple[_Kernel, _Kernel]:
    """One source compiled twice: serially, and with its ``prange`` over rows on Numba's
    threads. ``prange`` is ``range`` in the serial build. The parallel build compiles a copy
    under a name of its own, because Numba's cache files a function by name and not by its
    parallel flag."""
    assert njit is not None
    name = kernel.__name__ + "_parallel"
    copy = types.FunctionType(
        kernel.__code__, kernel.__globals__, name, kernel.__defaults__, kernel.__closure__
    )
    copy.__qualname__ = kernel.__qualname__ + "_parallel"
    serial = njit(cache=True)(kernel)
    parallel = njit(cache=True, parallel=True)(copy)
    return cast(_Kernel, serial), cast(_Kernel, parallel)


def _inlined(kernel: _Kernel) -> _Kernel:
    """``numba.njit(inline="always")``: a per-neuron helper merged into its caller, so no call
    and no array reference counting is paid per neuron."""
    assert njit is not None
    return cast(_Kernel, njit(inline="always")(kernel))


def available() -> bool:
    return njit is not None


def compiled() -> bool:
    """Whether the compiled passes are in use: Numba is installed and ``CADENCE_FUSED`` is not
    ``0`` (``Brain``'s switch for the fused kernels, read at call time)."""
    from . import brain

    return njit is not None and bool(brain._FUSED)


# A process forked from one whose GNU OpenMP threads had started cannot start them again
# (the runtime terminates the child); such a child settles its rows on the serial build.
_FORKED_FROM_OPENMP = False


def _after_fork() -> None:  # pragma: no cover - runs in a forked child
    global _FORKED_FROM_OPENMP
    try:
        layer = _numba.threading_layer()
    except ValueError:  # no parallel region ran before the fork
        return
    if layer == "omp":
        _FORKED_FROM_OPENMP = True


if njit is not None and hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


# A sweep of at least this many neurons over all rows runs its rows on Numba's threads; a
# smaller one runs serially, since handing a sweep to the threads costs tens of microseconds.
_PARALLEL_WORK = 1 << 16


def _threads() -> bool:
    """Whether Numba runs more than one thread here. ``NUMBA_NUM_THREADS=1`` never starts them."""
    if _FORKED_FROM_OPENMP or int(_numba.config.NUMBA_NUM_THREADS) < 2:
        return False
    return bool(_numba.get_num_threads() > 1)


def _in_parallel(batch: int, neurons: int) -> bool:
    """Whether ``batch`` rows of ``neurons`` settle on the parallel build: more than one row, a
    sweep of at least ``_PARALLEL_WORK`` neurons, more than one thread. Both builds give the
    same bits."""
    return batch >= 2 and batch * neurons >= _PARALLEL_WORK and _threads()


def _launch(
    serial: Callable[..., Any], parallel: Callable[..., Any], batch: int, neurons: int,
    *args: Any,
) -> Any:
    """Run a kernel's serial or parallel build on ``batch`` rows of ``neurons``."""
    return (parallel if _in_parallel(batch, neurons) else serial)(*args)


if njit is not None:

    @_compiled
    def _activation(
        v: np.ndarray,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        out: np.ndarray,
    ) -> None:
        n = v.shape[0]
        scale_up = 1.0 / (1.0 - rest)
        scale_down = leak / rest if rest > 0.0 else 0.0
        for i in range(n):
            r = 1.0 / (1.0 + np.exp(-slope * (v[i] - threshold))) - rest
            if r > 0.0:
                out[i] = r * scale_up
            elif leak == 0.0:
                out[i] = 0.0
            else:
                out[i] = r * scale_down

    @_compiled
    def _act_scalar(
        x: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
    ) -> float:
        r: float = 1.0 / (1.0 + np.exp(-slope * (x - threshold))) - rest
        if r > 0.0:
            return r * (1.0 / (1.0 - rest))
        if leak == 0.0:
            return 0.0
        return r * (leak / rest)

    @_compiled
    def _softmax_groups(
        s: np.ndarray, group: np.ndarray, gid: np.ndarray, ngroups: int,
        temperatures: np.ndarray, p: np.ndarray, zmax: np.ndarray, total: np.ndarray,
    ) -> None:
        for g in range(ngroups):
            zmax[g] = -np.inf
            total[g] = 0.0
        finite = True
        for k in range(group.shape[0]):
            i = group[k]
            z = s[i] / temperatures[i]
            p[k] = z
            finite = finite and np.isfinite(z)
            if z > zmax[gid[i]]:
                zmax[gid[i]] = z
        if not finite:
            # A subnormal temperature may overflow raw logits. Center first only
            # in that case, preserving the ordinary scalar path's arithmetic.
            for g in range(ngroups):
                zmax[g] = -np.inf
            for k in range(group.shape[0]):
                i = group[k]
                if s[i] > zmax[gid[i]]:
                    zmax[gid[i]] = s[i]
            for k in range(group.shape[0]):
                i = group[k]
                p[k] = (s[i] - zmax[gid[i]]) / temperatures[i]
            for g in range(ngroups):
                zmax[g] = 0.0
        for k in range(group.shape[0]):
            g = gid[group[k]]
            p[k] = np.exp(p[k] - zmax[g])
            total[g] += p[k]
        for k in range(group.shape[0]):
            p[k] /= total[gid[group[k]]]


    # -- the block transport, in two halves: the products for the whole batch, then a row's sum

    @_compiled
    def _product_cache(
        batch: int, pairs: np.ndarray, pair_post: np.ndarray, starts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Room for one cached product per block of ``pairs`` (``cache``, its offsets) and the
        ranges a block lands on."""
        cache_offset = np.zeros(pairs.shape[0] + 1, dtype=np.int64)
        receives = np.zeros(starts.shape[0] - 1, dtype=np.bool_)
        for q in range(pairs.shape[0]):
            b = pair_post[pairs[q]]
            cache_offset[q + 1] = cache_offset[q] + batch * (starts[b + 1] - starts[b])
            receives[b] = True
        return np.zeros(cache_offset[pairs.shape[0]]), cache_offset, receives

    @_compiled
    def _task_order(
        pairs: np.ndarray, apical_pairs: np.ndarray, pair_pre: np.ndarray, pair_post: np.ndarray,
        starts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The block products as tasks, largest first (a task below ``len(pairs)`` is a basal
        block, the rest apical ones), and the ranges some block reads."""
        count = pairs.shape[0]
        size = np.empty(count + apical_pairs.shape[0])
        feeds = np.zeros(starts.shape[0] - 1, dtype=np.bool_)
        for q in range(size.shape[0]):
            k = pairs[q] if q < count else apical_pairs[q - count]
            a, b = pair_pre[k], pair_post[k]
            size[q] = -float((starts[a + 1] - starts[a]) * (starts[b + 1] - starts[b]))
            feeds[a] = True
        return np.argsort(size, kind="mergesort"), feeds

    @_compiled
    def _due(
        first: bool,
        order: np.ndarray,
        pairs: np.ndarray,
        apical_pairs: np.ndarray,
        pair_pre: np.ndarray,
        moved_range: np.ndarray,
        todo: np.ndarray,
    ) -> int:
        """The products this sweep recomputes into ``todo``, largest first: every block on the
        first sweep, afterwards each block whose source range moved; returns their count."""
        count = pairs.shape[0]
        due = 0
        for j in range(order.shape[0]):
            q = order[j]
            k = pairs[q] if q < count else apical_pairs[q - count]
            if first or moved_range[pair_pre[k]]:
                todo[due] = q
                due += 1
        return due

    @_compiled
    def _copy_columns(
        b: int, s: np.ndarray, columns: np.ndarray, starts: np.ndarray, feeds: np.ndarray
    ) -> None:
        """Row ``b``'s activations into ``columns``, the batch's activations range by range:
        each range a block reads as one contiguous ``(batch, width)`` matrix at ``batch *
        starts[r]``, the operand of its products."""
        batch = s.shape[0]
        for r in range(feeds.shape[0]):
            if feeds[r]:
                a0 = starts[r]
                na = starts[r + 1] - a0
                base = batch * a0 + b * na
                for i in range(na):
                    columns[base + i] = s[b, a0 + i]

    @_compiled
    def _block_product(
        columns: np.ndarray,
        flat: np.ndarray,
        k: int,
        pair_pre: np.ndarray,
        pair_post: np.ndarray,
        starts: np.ndarray,
        offset: np.ndarray,
        cache: np.ndarray,
        c0: int,
    ) -> None:
        """Block ``k``'s product for the whole batch into its cache slot at ``c0``: one BLAS
        call on the whole batch, the same call whichever thread makes it."""
        a, b = pair_pre[k], pair_post[k]
        a0, a1 = starts[a], starts[a + 1]
        na, nb = a1 - a0, starts[b + 1] - starts[b]
        batch = columns.shape[0] // starts[starts.shape[0] - 1]
        source = columns[batch * a0 : batch * a1].reshape(batch, na)
        block = flat[offset[k] : offset[k + 1]].reshape(na, nb)
        np.dot(source, block, cache[c0 : c0 + batch * nb].reshape(batch, nb))

    @_compiled
    def _gather(
        b: int,
        out: np.ndarray,
        cache: np.ndarray,
        cache_offset: np.ndarray,
        pairs: np.ndarray,
        pair_post: np.ndarray,
        starts: np.ndarray,
        receives: np.ndarray,
    ) -> None:
        """Row ``b`` of the synaptic input of ``pairs``: zero where a block lands, then every
        block's cached product added in block order."""
        for r in range(receives.shape[0]):
            if receives[r]:
                for i in range(starts[r], starts[r + 1]):
                    out[b, i] = 0.0
        for q in range(pairs.shape[0]):
            k = pairs[q]
            b0 = starts[pair_post[k]]
            nb = starts[pair_post[k] + 1] - b0
            c0 = cache_offset[q] + b * nb
            for j in range(nb):
                out[b, b0 + j] += cache[c0 + j]

    # -- one neuron's step, and the apical compartment's terms

    @_inlined
    def _square_block(x: np.ndarray, lo: int, count: int) -> float:
        """``sum(x[lo:lo + count] ** 2)`` for ``count <= 128``, as NumPy's pairwise sum adds a
        block: eight running sums, combined in pairs, then the remainder."""
        res: float = 0.0
        if count < 8:
            for i in range(lo, lo + count):
                y = x[i]
                res += y * y
            return res
        y = x[lo]
        r0 = y * y
        y = x[lo + 1]
        r1 = y * y
        y = x[lo + 2]
        r2 = y * y
        y = x[lo + 3]
        r3 = y * y
        y = x[lo + 4]
        r4 = y * y
        y = x[lo + 5]
        r5 = y * y
        y = x[lo + 6]
        r6 = y * y
        y = x[lo + 7]
        r7 = y * y
        i = 8
        stop = count - count % 8
        while i < stop:
            j = lo + i
            y = x[j]
            r0 += y * y
            y = x[j + 1]
            r1 += y * y
            y = x[j + 2]
            r2 += y * y
            y = x[j + 3]
            r3 += y * y
            y = x[j + 4]
            r4 += y * y
            y = x[j + 5]
            r5 += y * y
            y = x[j + 6]
            r6 += y * y
            y = x[j + 7]
            r7 += y * y
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < count:
            y = x[lo + i]
            res += y * y
            i += 1
        return res

    @_compiled
    def _square_sum(
        x: np.ndarray, lo: int, count: int, frames: np.ndarray, left: np.ndarray
    ) -> float:
        """``sum(x[lo:lo + count] ** 2)`` in NumPy's pairwise order, what ``np.mean`` over a
        pool's row adds up, to the bit. NumPy splits a run of more than 128 at half its
        length rounded down to a multiple of eight and adds the halves' sums; this walks the
        same tree without recursion. ``frames`` (3, 64) and ``left`` (64,) are scratch: each
        split halves the run, so 64 levels cover any length."""
        if count <= 128:
            return _square_block(x, lo, count)
        sp = 0
        frames[0, 0] = lo
        frames[1, 0] = count
        frames[2, 0] = 0  # 0: not split yet, 1: left half pending, 2: right half pending
        while True:
            m = frames[1, sp]
            if m > 128 and frames[2, sp] == 0:  # split: the left half first
                half = m // 2
                half -= half % 8
                frames[2, sp] = 1
                sp += 1
                frames[0, sp] = frames[0, sp - 1]
                frames[1, sp] = half
                frames[2, sp] = 0
                continue
            value = _square_block(x, frames[0, sp], m)
            while True:  # hand the finished sum up the tree
                sp -= 1
                if sp < 0:
                    return value
                if frames[2, sp] == 1:  # it was the left half: keep it, go right
                    left[sp] = value
                    frames[2, sp] = 2
                    half = frames[1, sp] // 2
                    half -= half % 8
                    sp += 1
                    frames[0, sp] = frames[0, sp - 1] + half
                    frames[1, sp] = frames[1, sp - 1] - half
                    frames[2, sp] = 0
                    break
                value = left[sp] + value  # it was the right half: the node is done

    @_compiled
    def _pool_gains(
        b: int,
        basal: np.ndarray,
        raw: np.ndarray,
        pool_index: np.ndarray,
        pool_starts: np.ndarray,
        pool_run: np.ndarray,
        fraction: float,
        floor2: float,
        tiny: float,
        held: np.ndarray,
        has_held: bool,
        gain: np.ndarray,
        values: np.ndarray,
        frames: np.ndarray,
        left: np.ndarray,
    ) -> None:
        """Row ``b``'s gain per pool, as ``Apical.gain``: ``fraction * rms(basal) /
        sqrt(rms(raw)^2 + floor^2 rms(basal)^2)``, or the held gain of a nudged settle. A pool
        that is one run of neurons (``pool_run``: its first neuron, else -1) is summed in
        place; another is gathered into ``values`` first."""
        for k in range(pool_starts.shape[0] - 1):
            if has_held:
                gain[k] = held[b, k]
                continue
            lo = pool_starts[k]
            m = pool_starts[k + 1] - lo
            first = pool_run[k]
            if first >= 0:
                basal2 = _square_sum(basal, first, m, frames, left) / m
                raw2 = _square_sum(raw, first, m, frames, left) / m
            else:
                for j in range(m):
                    values[j] = basal[pool_index[lo + j]]
                basal2 = _square_sum(values, 0, m, frames, left) / m
                for j in range(m):
                    values[j] = raw[pool_index[lo + j]]
                raw2 = _square_sum(values, 0, m, frames, left) / m
            gain[k] = fraction * np.sqrt(basal2) / np.sqrt(raw2 + floor2 * basal2 + tiny)

    # -- one sweep of one row

    @_compiled
    def _sweep_row(
        b: int,
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        starts: np.ndarray,
        frozen: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        has_adapt: bool,
        adapt_strength: float,
        adapt_tau: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        group: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        position: np.ndarray,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        p: np.ndarray,
        zmax: np.ndarray,
        total: np.ndarray,
        moved_range: np.ndarray,
        moved_potential: np.ndarray,
        activity_change: np.ndarray,
    ) -> float:
        """Every neuron of row ``b`` takes one step; returns the row's largest movement and
        marks the ranges whose potential (``moved_potential``) and activation moved."""
        ranges = starts.shape[0] - 1
        moved = 0.0
        # the row's running total and range flags live in registers and are written once: rows
        # on other threads write the neighbouring entries
        change = activity_change[b]
        # the nudge on this row, from the activations before the step: one softmax per group
        if has_nudge and softmax_t[0] > 0.0:
            _softmax_groups(s[b], group, gid, ngroups, softmax_t, p, zmax, total)
        for r in range(ranges):
            range_moved = False
            potential_moved = False
            for i in range(starts[r], starts[r] if frozen[r] else starts[r + 1]):
                tot = synaptic_input[b, i] + standing[b, i]
                if has_adapt:
                    tot -= adapt_strength * a[b, i]
                if has_nudge and nmask[i] > 0.0 and softmax_t[0] <= 0.0:
                    tot += beta * weight[b] * nmask[i] * (target[b, i] - s[b, i])
                if anchor_gain[i] > 0.0:
                    tot += anchor_gain[i] * (anchor[b, i] - s[b, i])
                tot -= v[b, i]
                vn = v[b, i] + dt * tot
                if has_nudge and softmax_t[0] > 0.0 and nmask[i] > 0.0:
                    vn += dt * beta * weight[b] * (target[b, i] - p[position[i]]) * nmask[i]
                if masked:
                    vn *= keep[b, i]
                if vn != v[b, i]:  # a neuron whose potential did not move publishes what it did
                    potential_moved = True
                    v[b, i] = vn
                    sn = _act_scalar(vn, slope, threshold, rest, leak)
                    if masked:
                        sn *= keep[b, i]
                    d = abs(sn - s[b, i])
                    change += d
                    if d > 0.0:
                        range_moved = True
                        if d > moved:
                            moved = d
                    s[b, i] = sn
                if has_adapt:
                    a[b, i] = a[b, i] + (s[b, i] - a[b, i]) / adapt_tau
            moved_range[r] = range_moved
            moved_potential[r] = potential_moved
        activity_change[b] = change
        return moved

    @_compiled
    def _sweep_row_apical(
        b: int,
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        raw: np.ndarray,
        starts: np.ndarray,
        frozen: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        has_adapt: bool,
        adapt_strength: float,
        adapt_tau: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        group: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        position: np.ndarray,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        p: np.ndarray,
        zmax: np.ndarray,
        total: np.ndarray,
        pool_of: np.ndarray,
        pool_index: np.ndarray,
        pool_starts: np.ndarray,
        pool_run: np.ndarray,
        fraction: float,
        floor2: float,
        tiny: float,
        held: np.ndarray,
        has_held: bool,
        basal: np.ndarray,
        gain: np.ndarray,
        values: np.ndarray,
        frames: np.ndarray,
        left: np.ndarray,
        moved_range: np.ndarray,
        moved_potential: np.ndarray,
        activity_change: np.ndarray,
    ) -> float:
        """``_sweep_row`` for a graph with an apical compartment, in the NumPy loop's order:
        every neuron's basal input (synapses, stimulus and bias, adaptation, nudge), the pools'
        gains from it, then each neuron's step on its basal input plus its pool's gain times
        its apical input. ``basal`` (n,), ``gain`` (pools,), ``values``, ``frames`` and ``left``
        are the row's scratch."""
        n = v.shape[1]
        ranges = starts.shape[0] - 1
        moved = 0.0
        change = activity_change[b]
        softmax = has_nudge and softmax_t[0] > 0.0
        if softmax:
            _softmax_groups(s[b], group, gid, ngroups, softmax_t, p, zmax, total)
        for i in range(n):  # a still range's neurons count in their pool's RMS too
            x = synaptic_input[b, i] + standing[b, i]
            if has_adapt:
                x -= adapt_strength * a[b, i]
            if has_nudge:  # summed as ``Nudge.drive`` sums it: beta (target - s) mask, the
                # row's weight, then the anchor's pull (``p`` in place of ``s`` under a softmax)
                nd = 0.0
                pulled = False
                if nmask[i] > 0.0:
                    if softmax:
                        nd = beta * (target[b, i] - p[position[i]]) * nmask[i]
                    else:
                        nd = beta * (target[b, i] - s[b, i]) * nmask[i]
                    nd = nd * weight[b]
                    pulled = True
                if anchor_gain[i] > 0.0:
                    nd = nd + anchor_gain[i] * (anchor[b, i] - s[b, i])
                    pulled = True
                if pulled:
                    x = x + nd
            basal[i] = x
        _pool_gains(b, basal, raw[b], pool_index, pool_starts, pool_run, fraction, floor2, tiny,
                    held, has_held, gain, values, frames, left)
        for r in range(ranges):
            range_moved = False
            potential_moved = False
            for i in range(starts[r], starts[r] if frozen[r] else starts[r + 1]):
                x = basal[i]
                k = pool_of[i]
                if k >= 0:
                    x = x + gain[k] * raw[b, i]
                vn = v[b, i] + (x - v[b, i]) * dt
                if masked:
                    vn *= keep[b, i]
                if vn != v[b, i]:  # a neuron whose potential did not move publishes what it did
                    potential_moved = True
                    v[b, i] = vn
                    sn = _act_scalar(vn, slope, threshold, rest, leak)
                    if masked:
                        sn *= keep[b, i]
                    d = abs(sn - s[b, i])
                    change += d
                    if d > 0.0:
                        range_moved = True
                        if d > moved:
                            moved = d
                    s[b, i] = sn
                if has_adapt:
                    a[b, i] = a[b, i] + (s[b, i] - a[b, i]) / adapt_tau
            moved_range[r] = range_moved
            moved_potential[r] = potential_moved
        activity_change[b] = change
        return moved

    @_compiled
    def _close_sweep(
        t: int,
        row_moved: np.ndarray,
        row_range: np.ndarray,
        row_potential: np.ndarray,
        moved_range: np.ndarray,
        freezable: np.ndarray,
        frozen: np.ndarray,
    ) -> float:
        """The batch's reading of a sweep: the largest movement of any row, the ranges whose
        activation moved in any row, and the still ranges with nothing arriving, which stay
        still from now on."""
        batch, ranges = row_range.shape
        moved = 0.0
        for b in range(batch):
            if row_moved[b] > moved:
                moved = row_moved[b]
        for r in range(ranges):
            range_moved = False
            potential_moved = False
            for b in range(batch):
                range_moved = range_moved or row_range[b, r]
                potential_moved = potential_moved or row_potential[b, r]
            moved_range[r] = range_moved
            if freezable[r] and not potential_moved and t > 0:
                frozen[r] = True
        return moved

    def _settle_rows(
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        flat: np.ndarray,
        pairs: np.ndarray,
        flat_apical: np.ndarray,
        apical_pairs: np.ndarray,
        starts: np.ndarray,
        pair_pre: np.ndarray,
        pair_post: np.ndarray,
        offset: np.ndarray,
        freezable: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        has_adapt: bool,
        adapt_strength: float,
        adapt_tau: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        has_apical: bool,
        pool_of: np.ndarray,
        pool_index: np.ndarray,
        pool_starts: np.ndarray,
        pool_run: np.ndarray,
        fraction: float,
        floor2: float,
        tiny: float,
        held: np.ndarray,
        has_held: bool,
        steps: int,
        tolerance: float,
        use_tolerance: bool,
    ) -> tuple[int, np.ndarray]:
        """The settle: per sweep the block products of the batch, then every row's step (the
        rows in parallel in the parallel build), then the batch-wide reading and stop."""
        batch, n = v.shape
        ranges = starts.shape[0] - 1
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)  # an output neuron's place in the softmax group
        for k in range(group.shape[0]):
            position[group[k]] = k
        cache, cache_offset, receives = _product_cache(batch, pairs, pair_post, starts)
        cache_up, cache_up_offset, receives_up = _product_cache(batch, apical_pairs, pair_post,
                                                                starts)
        order, feeds = _task_order(pairs, apical_pairs, pair_pre, pair_post, starts)
        todo = np.empty(order.shape[0], dtype=np.int64)
        basal_count = pairs.shape[0]
        columns = np.empty(batch * n)
        for b in prange(batch):
            _copy_columns(b, s, columns, starts, feeds)
        synaptic_input = np.zeros((batch, n))
        raw = np.zeros((batch, n if has_apical else 1))
        basal = np.empty((batch, n if has_apical else 1))
        gain = np.empty((batch, max(pool_starts.shape[0] - 1, 1)))
        values = np.empty((batch, n if has_apical else 1))
        frames = np.empty((batch, 3, 64), dtype=np.int64)
        left = np.empty((batch, 64))
        p = np.empty((batch, group.shape[0]))
        zmax = np.empty((batch, max(ngroups, 1)))
        total = np.empty((batch, max(ngroups, 1)))
        activity_change = np.zeros(batch)
        row_moved = np.zeros(batch)
        row_range = np.zeros((batch, ranges), dtype=np.bool_)
        row_potential = np.zeros((batch, ranges), dtype=np.bool_)
        moved_range = np.ones(ranges, dtype=np.bool_)  # activation changed in the last step
        frozen = np.zeros(ranges, dtype=np.bool_)  # a range that can never change again
        taken = 0
        for t in range(steps):
            # the block products of the batch, distinct blocks on distinct threads
            due = _due(t == 0, order, pairs, apical_pairs, pair_pre, moved_range, todo)
            for j in prange(due):
                q = todo[j]
                if q < basal_count:
                    _block_product(columns, flat, pairs[q], pair_pre, pair_post, starts, offset,
                                   cache, cache_offset[q])
                else:
                    _block_product(columns, flat_apical, apical_pairs[q - basal_count], pair_pre,
                                   pair_post, starts, offset, cache_up,
                                   cache_up_offset[q - basal_count])
            # every row's step, the rows on distinct threads
            for b in prange(batch):
                _gather(b, synaptic_input, cache, cache_offset, pairs, pair_post, starts, receives)
                if has_apical:
                    _gather(b, raw, cache_up, cache_up_offset, apical_pairs, pair_post, starts,
                            receives_up)
                    row_moved[b] = _sweep_row_apical(
                        b, v, a, s, standing, synaptic_input, raw, starts, frozen, keep, masked,
                        dt, slope, threshold, rest, leak, has_adapt, adapt_strength, adapt_tau,
                        has_nudge, target, nmask, group, gid, ngroups, position, beta, softmax_t,
                        weight, anchor, anchor_gain, p[b], zmax[b], total[b], pool_of,
                        pool_index, pool_starts, pool_run, fraction, floor2, tiny, held, has_held,
                        basal[b], gain[b], values[b], frames[b], left[b], row_range[b],
                        row_potential[b], activity_change,
                    )
                else:
                    row_moved[b] = _sweep_row(
                        b, v, a, s, standing, synaptic_input, starts, frozen, keep, masked, dt,
                        slope, threshold, rest, leak, has_adapt, adapt_strength, adapt_tau,
                        has_nudge, target, nmask, group, gid, ngroups, position, beta, softmax_t,
                        weight, anchor, anchor_gain, p[b], zmax[b], total[b], row_range[b],
                        row_potential[b], activity_change,
                    )
                _copy_columns(b, s, columns, starts, feeds)  # what the next products read
            moved = _close_sweep(t, row_moved, row_range, row_potential, moved_range, freezable,
                                 frozen)
            taken = t + 1
            if use_tolerance and moved < tolerance:
                break
        return taken, activity_change

    _kernel, _kernel_parallel = _rows_builds(_settle_rows)

    # -- the residual: one transport, then the equations' error per row

    @_compiled
    def _residual_row(
        b: int,
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        has_adapt: bool,
        adapt_strength: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        group: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        position: np.ndarray,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        p: np.ndarray,
        zmax: np.ndarray,
        total: np.ndarray,
    ) -> float:
        """Row ``b``'s largest equation error at the published activations ``s``: the terms the
        settle adds, without a step; infinity where it is not finite."""
        n = v.shape[1]
        if has_nudge and softmax_t[0] > 0.0:
            _softmax_groups(s[b], group, gid, ngroups, softmax_t, p, zmax, total)
        worst = 0.0
        finite = True
        for i in range(n):
            tot = synaptic_input[b, i] + standing[b, i] - v[b, i]
            if has_adapt:
                tot -= adapt_strength * a[b, i]
            if anchor_gain[i] > 0.0:
                tot += anchor_gain[i] * (anchor[b, i] - s[b, i])
            if has_nudge and nmask[i] > 0.0:
                if softmax_t[0] > 0.0:
                    tot += beta * weight[b] * (target[b, i] - p[position[i]]) * nmask[i]
                else:
                    tot += beta * weight[b] * nmask[i] * (target[b, i] - s[b, i])
            k_ = keep[b, i] if masked else 1.0
            err = abs(k_ * tot + (k_ - 1.0) * v[b, i] / dt)
            if has_adapt:
                gap = abs(s[b, i] - a[b, i])
                if gap > err:
                    err = gap
            if not (err <= 1e300):
                finite = False
            elif err > worst:
                worst = err
        return worst if finite else np.inf

    @_compiled
    def _residual_row_apical(
        b: int,
        v: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        raw: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        has_adapt: bool,
        adapt_strength: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        group: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        position: np.ndarray,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        p: np.ndarray,
        zmax: np.ndarray,
        total: np.ndarray,
        pool_of: np.ndarray,
        pool_index: np.ndarray,
        pool_starts: np.ndarray,
        pool_run: np.ndarray,
        fraction: float,
        floor2: float,
        tiny: float,
        held: np.ndarray,
        has_held: bool,
        basal: np.ndarray,
        gain: np.ndarray,
        values: np.ndarray,
        frames: np.ndarray,
        left: np.ndarray,
    ) -> float:
        """``_residual_row`` with an apical compartment, in the NumPy residual's order."""
        n = v.shape[1]
        softmax = has_nudge and softmax_t[0] > 0.0
        if softmax:
            _softmax_groups(s[b], group, gid, ngroups, softmax_t, p, zmax, total)
        for i in range(n):
            x = synaptic_input[b, i] + standing[b, i]
            if has_adapt:
                x -= adapt_strength * a[b, i]
            if has_nudge:  # summed as ``Nudge.drive`` sums it: beta (target - s) mask, the
                # row's weight, then the anchor's pull (``p`` in place of ``s`` under a softmax)
                nd = 0.0
                pulled = False
                if nmask[i] > 0.0:
                    if softmax:
                        nd = beta * (target[b, i] - p[position[i]]) * nmask[i]
                    else:
                        nd = beta * (target[b, i] - s[b, i]) * nmask[i]
                    nd = nd * weight[b]
                    pulled = True
                if anchor_gain[i] > 0.0:
                    nd = nd + anchor_gain[i] * (anchor[b, i] - s[b, i])
                    pulled = True
                if pulled:
                    x = x + nd
            basal[i] = x
        _pool_gains(b, basal, raw[b], pool_index, pool_starts, pool_run, fraction, floor2, tiny,
                    held, has_held, gain, values, frames, left)
        worst = 0.0
        finite = True
        for i in range(n):
            x = basal[i]
            k = pool_of[i]
            if k >= 0:
                x = x + gain[k] * raw[b, i]
            x = x - v[b, i]
            k_ = keep[b, i] if masked else 1.0
            err = abs(k_ * x + (k_ - 1.0) * v[b, i] / dt)
            if has_adapt:
                gap = abs(s[b, i] - a[b, i])
                if gap > err:
                    err = gap
            if not (err <= 1e300):
                finite = False
            elif err > worst:
                worst = err
        return worst if finite else np.inf

    def _residual_rows(
        v: np.ndarray,
        a: np.ndarray,
        standing: np.ndarray,
        flat: np.ndarray,
        pairs: np.ndarray,
        flat_apical: np.ndarray,
        apical_pairs: np.ndarray,
        starts: np.ndarray,
        pair_pre: np.ndarray,
        pair_post: np.ndarray,
        offset: np.ndarray,
        keep: np.ndarray,
        masked: bool,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        has_adapt: bool,
        adapt_strength: float,
        has_nudge: bool,
        target: np.ndarray,
        nmask: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        has_apical: bool,
        pool_of: np.ndarray,
        pool_index: np.ndarray,
        pool_starts: np.ndarray,
        pool_run: np.ndarray,
        fraction: float,
        floor2: float,
        tiny: float,
        held: np.ndarray,
        has_held: bool,
        out: np.ndarray,
    ) -> None:
        """The fixed-point equation error of every row, in one compiled call: the activations
        of ``v`` under the mask, one block transport, then each row's error into ``out``."""
        batch, n = v.shape
        s = np.empty((batch, n))
        for b in prange(batch):
            _activation(v[b], slope, threshold, rest, leak, s[b])
            if masked:
                for i in range(n):
                    s[b, i] *= keep[b, i]
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)
        for k in range(group.shape[0]):
            position[group[k]] = k
        cache, cache_offset, receives = _product_cache(batch, pairs, pair_post, starts)
        cache_up, cache_up_offset, receives_up = _product_cache(batch, apical_pairs, pair_post,
                                                                starts)
        order, feeds = _task_order(pairs, apical_pairs, pair_pre, pair_post, starts)
        basal_count = pairs.shape[0]
        columns = np.empty(batch * n)
        for b in prange(batch):
            _copy_columns(b, s, columns, starts, feeds)
        for j in prange(order.shape[0]):
            q = order[j]
            if q < basal_count:
                _block_product(columns, flat, pairs[q], pair_pre, pair_post, starts, offset, cache,
                               cache_offset[q])
            else:
                _block_product(columns, flat_apical, apical_pairs[q - basal_count], pair_pre,
                               pair_post, starts, offset, cache_up,
                               cache_up_offset[q - basal_count])
        synaptic_input = np.zeros((batch, n))
        raw = np.zeros((batch, n if has_apical else 1))
        basal = np.empty((batch, n if has_apical else 1))
        gain = np.empty((batch, max(pool_starts.shape[0] - 1, 1)))
        values = np.empty((batch, n if has_apical else 1))
        frames = np.empty((batch, 3, 64), dtype=np.int64)
        left = np.empty((batch, 64))
        p = np.empty((batch, group.shape[0]))
        zmax = np.empty((batch, max(ngroups, 1)))
        total = np.empty((batch, max(ngroups, 1)))
        for b in prange(batch):
            _gather(b, synaptic_input, cache, cache_offset, pairs, pair_post, starts, receives)
            if has_apical:
                _gather(b, raw, cache_up, cache_up_offset, apical_pairs, pair_post, starts,
                        receives_up)
                out[b] = _residual_row_apical(
                    b, v, a, s, standing, synaptic_input, raw, keep, masked, dt, has_adapt,
                    adapt_strength, has_nudge, target, nmask, group, gid, ngroups, position,
                    beta, softmax_t, weight, anchor, anchor_gain, p[b], zmax[b], total[b],
                    pool_of, pool_index, pool_starts, pool_run, fraction, floor2, tiny, held,
                    has_held, basal[b], gain[b], values[b], frames[b], left[b],
                )
            else:
                out[b] = _residual_row(
                    b, v, a, s, standing, synaptic_input, keep, masked, dt, has_adapt,
                    adapt_strength, has_nudge, target, nmask, group, gid, ngroups, position,
                    beta, softmax_t, weight, anchor, anchor_gain, p[b], zmax[b], total[b],
                )

    _residual_kernel, _residual_kernel_parallel = _rows_builds(_residual_rows)


def _nudge_arrays(
    nudge: Nudge | None, batch: int, n: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """The nudge as the dense arrays the kernels read: target, mask, group id per neuron
    (0..ngroups-1, -1 for none), the group count, beta, the per-neuron softmax temperatures (0 for a
    plain nudge) and the per-row weight. A softmax nudge drives only grouped neurons."""
    if nudge is None:
        return (
            np.zeros((1, 1)),
            np.zeros(n),
            np.full(n, -1, dtype=np.int64),
            0,
            0.0,
            np.zeros(n),
            np.ones(batch),
            np.zeros((1, 1)),
            np.zeros(n),
        )
    target = np.ascontiguousarray(np.broadcast_to(nudge.target, (batch, n)).astype(float))
    nmask = np.asarray(nudge.mask, float)
    softmax_t = (
        np.zeros(n) if nudge.softmax_temperature is None
        else np.full(n, nudge.softmax_temperature)
    )
    weight = np.ones(batch) if nudge.weight is None else np.asarray(nudge.weight, float)
    beta = float(nudge.beta)
    if nudge.groups is None:
        gid = np.where(nmask > 0, 0, -1).astype(np.int64)
        ngroups = 1
    else:
        raw = np.asarray(nudge.groups, dtype=np.int64)
        ids = np.unique(raw[raw >= 0])
        gid = np.full(n, -1, dtype=np.int64)
        for k, g in enumerate(ids):
            gid[raw == g] = k
        ngroups = len(ids)
    if softmax_t[0] > 0:
        nmask = np.where(gid >= 0, nmask, 0.0)
    anchor, anchor_gain = np.zeros((1, 1)), np.zeros(n)
    if nudge.anchor is not None:
        anchor = np.ascontiguousarray(np.broadcast_to(nudge.anchor, (batch, n)), dtype=float)
        anchor_gain = np.ascontiguousarray(nudge.anchor_gain, dtype=float)
    return target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain




# ``(compartment, basal blocks, apical flat blocks, apical blocks)`` of a graph with an apical
# compartment; the ``flat`` beside it then holds the basal synapses alone
ApicalBlocks = tuple["Apical", Sequence[int], np.ndarray, Sequence[int]]


def _compartment(
    apical: ApicalBlocks | None, layout: Layout, flat: np.ndarray, nudge: Nudge | None,
    batch: int, n: int,
) -> tuple[Any, ...]:
    """The kernels' arguments for the blocks and the apical compartment: ``(pairs,
    flat_apical, apical_pairs, has_apical, pool_of, pool_index, pool_starts, pool_run,
    fraction, floor2, tiny, held, has_held)``. Without a compartment every block is basal."""
    if apical is None:
        return (
            np.arange(layout.pairs, dtype=np.int64), flat, np.zeros(0, dtype=np.int64), False,
            np.full(n, -1, dtype=np.int64), np.zeros(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), 1.0, 1.0, 0.0,
            np.zeros((1, 1)), False,
        )
    from .brain import _APICAL_TINY

    compartment, basal_pairs, flat_apical, apical_pairs = apical
    pools = len(compartment.members)
    held = None if nudge is None else nudge.apical_gain
    if held is not None and held.shape != (batch, pools):
        raise ValueError("a nudge's apical_gain needs one entry per row and pool")
    index = np.arange(n, dtype=np.int64)
    members = [index[m] for m in compartment.members]  # each pool in the order NumPy reads it
    pool_of = np.full(n, -1, dtype=np.int64)
    for k, part in enumerate(members):
        pool_of[part] = k
    pool_index = np.concatenate(members) if members else np.zeros(0, dtype=np.int64)
    pool_starts = np.zeros(pools + 1, dtype=np.int64)
    pool_starts[1:] = np.cumsum([len(part) for part in members])
    pool_run = np.array(  # a pool that is one run of neurons is summed in place
        [part[0] if part[-1] - part[0] + 1 == len(part) else -1 for part in members],
        dtype=np.int64,
    )
    return (
        np.asarray(basal_pairs, dtype=np.int64), np.ascontiguousarray(flat_apical, dtype=float),
        np.asarray(apical_pairs, dtype=np.int64), True, pool_of, pool_index, pool_starts, pool_run,
        compartment.fraction, compartment.floor * compartment.floor, _APICAL_TINY,
        np.zeros((1, 1)) if held is None else np.ascontiguousarray(held, dtype=float),
        held is not None,
    )


def fused_settle(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray,
    neuron_model: NeuronModel,
    nudge: Nudge | None,
    steps: int,
    tolerance: float | None,
    activation: np.ndarray | None = None,
    apical: ApicalBlocks | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Run the fused kernel in place on ``v`` and ``a``; returns ``(s, taken, activity_change)``.

    ``activation`` is accepted and ignored. Publication is recomputed
    from ``v`` under the current mask, since cached readback may use an old mask.
    ``apical`` settles a graph with an apical compartment (``ApicalBlocks``); ``flat`` then
    holds its basal synapses. With more than one row and more than one Numba thread the rows
    settle in parallel, to the same bits.
    """
    assert njit is not None
    batch, n = v.shape
    standing = np.ascontiguousarray(drive + bias)
    masked = bool((keep != 1.0).any())
    # A broadcast view preserves shared masks without allocating a batch-sized copy.
    keep = np.broadcast_to(np.asarray(keep, float), (batch, n))
    if not v.any():  # from rest every neuron publishes the same thing
        row = np.empty(n)
        m = neuron_model
        _activation(np.zeros(n), m.slope, m.threshold, m.rest_emission, m.leak, row)
        s = np.empty((batch, n))
        s[:] = row
    else:
        s = np.empty_like(v)
        for b in range(batch):
            m = neuron_model
            _activation(v[b], m.slope, m.threshold, m.rest_emission, m.leak, s[b])
    if masked:
        s *= keep
    adapt = neuron_model.adaptation
    has_adapt = adapt is not None
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        nudge, batch, n
    )
    lay = layout
    has_synaptic_input = np.zeros(lay.ranges, dtype=np.bool_)
    has_synaptic_input[lay.pair_post] = True
    freezable = ~has_synaptic_input  # nothing arrives: once still, still for good
    if has_adapt:
        freezable = np.zeros(lay.ranges, dtype=np.bool_)
    elif nudge is not None:
        nudged = (
            np.searchsorted(
                lay.starts, np.flatnonzero((nmask > 0) | (anchor_gain > 0)), side="right"
            )
            - 1
        )
        freezable[nudged] = False
    pairs, flat_apical, apical_pairs, *compartment = _compartment(
        apical, lay, flat, nudge, batch, n
    )
    taken, activity_change = _launch(
        _kernel,
        _kernel_parallel,
        batch,
        n,
        v,
        a,
        s,
        standing,
        flat,
        pairs,
        flat_apical,
        apical_pairs,
        lay.starts,
        lay.pair_pre,
        lay.pair_post,
        lay.offset,
        freezable,
        np.asarray(keep, float),
        masked,
        neuron_model.dt,
        neuron_model.slope,
        neuron_model.threshold,
        neuron_model.rest_emission,
        neuron_model.leak,
        has_adapt,
        0.0 if adapt is None else adapt.strength,
        1.0 if adapt is None else adapt.tau_steps,
        nudge is not None,
        target,
        nmask,
        gid,
        ngroups,
        beta,
        softmax_t,
        weight,
        anchor,
        anchor_gain,
        *compartment,
        int(steps),
        float(tolerance) if tolerance is not None else 0.0,
        tolerance is not None,
    )
    return s, taken, activity_change


def fused_residual(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray,
    neuron_model: NeuronModel,
    nudge: Nudge | None,
    apical: ApicalBlocks | None = None,
) -> np.ndarray:
    """``Brain.residual`` for a host state of a blocked connectome, in one compiled call.

    The same equations as the NumPy path: the activation of ``v`` under the mask, one block
    transport, the potential equation with adaptation, nudge and apical compartment
    (``apical``, as in ``fused_settle``), the mask's projection, and the adaptation
    equation; the maximum absolute error per row, infinity where it is not finite. Nothing
    is changed.
    """
    assert njit is not None
    batch, n = v.shape
    m = neuron_model
    keep = np.broadcast_to(np.asarray(keep, float), (batch, n))
    masked = bool((keep != 1.0).any())
    adapt = m.adaptation
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        nudge, batch, n
    )
    pairs, flat_apical, apical_pairs, *compartment = _compartment(
        apical, layout, flat, nudge, batch, n
    )
    out = np.empty(batch)
    _launch(
        _residual_kernel,
        _residual_kernel_parallel,
        batch,
        n,
        np.ascontiguousarray(v, dtype=float),
        np.ascontiguousarray(a, dtype=float),
        np.ascontiguousarray(drive + bias),
        flat,
        pairs,
        flat_apical,
        apical_pairs,
        layout.starts,
        layout.pair_pre,
        layout.pair_post,
        layout.offset,
        np.ascontiguousarray(keep, dtype=float),
        masked,
        m.dt,
        m.slope,
        m.threshold,
        m.rest_emission,
        m.leak,
        adapt is not None,
        0.0 if adapt is None else adapt.strength,
        nudge is not None,
        target,
        nmask,
        gid,
        ngroups,
        beta,
        softmax_t,
        weight,
        anchor,
        anchor_gain,
        *compartment,
        out,
    )
    return out


if njit is not None:

    @_compiled
    def _trace_step(
        trace: np.ndarray,
        trace_bias: np.ndarray,
        decay: float,
        s_plus: np.ndarray,
        s_minus: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        span: float,
        delta: np.ndarray,
        step_scale: np.ndarray,
        step_bias: np.ndarray,
    ) -> None:
        """One pass per row: contrast from the two phases, trace decay and accumulation, the
        dopamine-weighted sum into the step. ``trace`` is (batch, edges) and ``trace_bias``
        (batch, n)."""
        batch, edges = trace.shape
        n = trace_bias.shape[1]
        for e in range(edges):
            step_scale[e] = 0.0
        for i in range(n):
            step_bias[i] = 0.0
        for b in range(batch):
            d = delta[b] / batch
            for e in range(edges):
                a_plus, a_minus = s_plus[b, pre[e]], s_minus[b, pre[e]]
                b_plus, b_minus = s_plus[b, post[e]], s_minus[b, post[e]]
                c = (a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus) / span
                t = decay * trace[b, e] + c
                trace[b, e] = t
                step_scale[e] += d * t
            for i in range(n):
                c = (s_plus[b, i] - s_minus[b, i]) / span
                t = decay * trace_bias[b, i] + c
                trace_bias[b, i] = t
                step_bias[i] += d * t


def trace_step(
    trace: np.ndarray,
    trace_bias: np.ndarray,
    decay: float,
    s_plus: np.ndarray,
    s_minus: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    span: float,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused three-factor step: returns ``(step_scale, step_bias)``; traces are updated in place."""
    assert njit is not None
    step_scale = np.empty(trace.shape[1])
    step_bias = np.empty(trace_bias.shape[1])
    _trace_step(
        trace,
        trace_bias,
        float(decay),
        s_plus,
        s_minus,
        pre,
        post,
        float(span),
        np.ascontiguousarray(delta, dtype=np.float64),
        step_scale,
        step_bias,
    )
    return step_scale, step_bias


if njit is not None:

    @_compiled
    def _contrast_mean(
        s_plus: np.ndarray,
        s_minus: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        span: float,
        out_edges: np.ndarray,
        out_neurons: np.ndarray,
    ) -> None:
        """Batch-mean contrast per synapse and per neuron, one pass, no (batch, edges) temporary."""
        batch, n = s_plus.shape
        edges = pre.shape[0]
        for e in range(edges):
            out_edges[e] = 0.0
        for i in range(n):
            out_neurons[i] = 0.0
        for b in range(batch):
            for e in range(edges):
                a_plus, a_minus = s_plus[b, pre[e]], s_minus[b, pre[e]]
                b_plus, b_minus = s_plus[b, post[e]], s_minus[b, post[e]]
                out_edges[e] += a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus
            for i in range(n):
                out_neurons[i] += s_plus[b, i] - s_minus[b, i]
        scale = 1.0 / (batch * span)
        for e in range(edges):
            out_edges[e] *= scale
        for i in range(n):
            out_neurons[i] *= scale


def contrast_mean(
    s_plus: np.ndarray, s_minus: np.ndarray, pre: np.ndarray, post: np.ndarray, span: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fused ``Learner.contrast``: ``(per_synapse, per_neuron)`` batch means divided by ``span``."""
    assert njit is not None
    out_edges = np.empty(pre.shape[0])
    out_neurons = np.empty(s_plus.shape[1])
    _contrast_mean(
        np.ascontiguousarray(s_plus),
        np.ascontiguousarray(s_minus),
        pre,
        post,
        float(span),
        out_edges,
        out_neurons,
    )
    return out_edges, out_neurons


# -- per-synapse passes of a lesson: the same element arithmetic as NumPy, one compiled pass

# A pass this long runs on Numba's threads (when it runs more than one); a shorter one serially.
_PARALLEL_ELEMENTS = 1 << 19


def _elements_in_parallel(count: int) -> bool:
    return count >= _PARALLEL_ELEMENTS and _threads()


if njit is not None:

    def _scatter_entries(index: np.ndarray, values: np.ndarray, out: np.ndarray) -> None:
        for e in prange(index.shape[0]):
            out[index[e]] = values[e]

    _place, _place_parallel = _rows_builds(_scatter_entries)

    def _gather_entries(
        values: np.ndarray, index: np.ndarray, divisor: float, out: np.ndarray
    ) -> None:
        for e in prange(index.shape[0]):
            out[e] = values[index[e]] / divisor

    _pick, _pick_parallel = _rows_builds(_gather_entries)

    @_compiled
    def _group_power(
        values: np.ndarray, group: np.ndarray, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each group's sum of squares, added in index order as ``np.bincount`` adds, and its
        member count (at least one)."""
        power = np.zeros(size)
        count = np.zeros(size)
        for e in range(values.shape[0]):
            g = group[e]
            power[g] += values[e] * values[e]
            count[g] += 1.0
        for g in range(size):
            if count[g] < 1.0:
                count[g] = 1.0
        return power, count

    def _floor_entries(
        values: np.ndarray, group: np.ndarray, root: np.ndarray, share: float, tiny: float,
        out: np.ndarray,
    ) -> None:
        for e in prange(values.shape[0]):
            out[e] = values[e] + share * root[group[e]] + tiny

    _divisor, _divisor_parallel = _rows_builds(_floor_entries)


def scatter(size: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``out = zeros(size); out[index] = values`` for distinct indices, in one compiled pass."""
    assert njit is not None
    out = np.zeros(size)
    count = len(index)
    kernel = _place_parallel if _elements_in_parallel(count) else _place
    kernel(index, np.ascontiguousarray(values, dtype=float), out)
    return out


def gather(values: np.ndarray, index: np.ndarray, divisor: float = 1.0) -> np.ndarray:
    """``values[index] / divisor`` in one compiled pass."""
    assert njit is not None
    out = np.empty(len(index))
    kernel = _pick_parallel if _elements_in_parallel(len(index)) else _pick
    kernel(np.ascontiguousarray(values, dtype=float), index, float(divisor), out)
    return out


def relative_floor(
    values: np.ndarray, group: np.ndarray, size: int, share: float, tiny: float
) -> np.ndarray:
    """``values + share * sqrt(power / count)[group] + tiny``, ``power`` and ``count`` each
    group's ``np.bincount`` of ``values ** 2`` and of members (at least one): the divisor of a
    projection-relative floor, to the bit."""
    assert njit is not None
    values = np.ascontiguousarray(values, dtype=float)
    power, count = _group_power(values, group, int(size))
    root = np.sqrt(power / count)
    out = np.empty(len(values))
    kernel = _divisor_parallel if _elements_in_parallel(len(values)) else _divisor
    kernel(values, group, root, float(share), float(tiny), out)
    return out


# -- the optimizer step of a lesson: moments, normalization and the tied, bounded step

if njit is not None:

    def _moment_entries(
        raw: np.ndarray,
        velocity: np.ndarray,
        second: np.ndarray,
        momentum: float,
        momentum_rest: float,
        momentum_correction: float,
        rho: float,
        rho_rest: float,
        rho_correction: float,
        has_momentum: bool,
        has_normalize: bool,
        new_velocity: np.ndarray,
        new_second: np.ndarray,
        term: np.ndarray,
        root: np.ndarray,
    ) -> None:
        for e in prange(raw.shape[0]):
            x = raw[e]
            t = x
            if has_momentum:
                moving = momentum * velocity[e] + momentum_rest * x
                new_velocity[e] = moving
                t = moving / momentum_correction
            if has_normalize:
                held = rho * second[e] + rho_rest * (x * x)
                new_second[e] = held
                root[e] = np.sqrt(held / rho_correction)
            term[e] = t

    _moments, _moments_parallel = _rows_builds(_moment_entries)

    def _step_entries(
        term: np.ndarray,
        root: np.ndarray,
        group: np.ndarray,
        group_root: np.ndarray,
        eta: float,
        mode: int,
        floor: float,
        share: float,
        tiny: float,
        out: np.ndarray,
    ) -> None:
        for e in prange(term.shape[0]):
            if mode == 0:  # no normalization
                out[e] = eta * term[e]
            elif mode == 1:  # the absolute floor
                out[e] = eta * (term[e] / (root[e] + floor))
            else:  # the projection-relative floor
                out[e] = eta * (term[e] / (root[e] + share * group_root[group[e]] + tiny))

    _step, _step_parallel = _rows_builds(_step_entries)

    def _tied_entries(
        delta: np.ndarray,
        partner: np.ndarray,
        efficacy: np.ndarray,
        cap: float,
        out_delta: np.ndarray,
        out_scale: np.ndarray,
    ) -> None:
        for e in prange(delta.shape[0]):
            k = partner[e]
            d = 0.5 * (delta[e] + delta[k]) if k >= 0 else delta[e]
            out_delta[e] = d
            out_scale[e] = min(max(efficacy[e] + d, -cap), cap)

    _tied, _tied_parallel = _rows_builds(_tied_entries)


def optimizer_step(
    raw: np.ndarray,
    velocity: np.ndarray,
    second: np.ndarray,
    momentum: float,
    momentum_correction: float,
    rho: float,
    rho_correction: float,
    eta: float,
    floor: float,
    relative: float,
    tiny: float,
    group: np.ndarray,
    groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Learner._update``'s per-synapse arithmetic in two compiled passes, to the bit: the
    momentum (above zero), the second moment and its normalization (``rho`` above zero; the
    absolute ``floor``, or the projection-relative floor of share ``relative`` over ``group``),
    and the rate. Returns ``(velocity, second moment, step)``; the moments a setting leaves off
    come back as given."""
    assert njit is not None
    count = len(raw)
    raw = np.ascontiguousarray(raw, dtype=float)
    has_momentum, has_normalize = momentum > 0, rho > 0
    new_velocity = np.empty(count) if has_momentum else velocity
    new_second = np.empty(count) if has_normalize else second
    term, root = np.empty(count), np.empty(count if has_normalize else 1)
    parallel = _elements_in_parallel(count)
    (_moments_parallel if parallel else _moments)(
        raw, np.ascontiguousarray(velocity, dtype=float), np.ascontiguousarray(second, dtype=float),
        float(momentum), float(1 - momentum), float(momentum_correction), float(rho),
        float(1 - rho), float(rho_correction), has_momentum, has_normalize, new_velocity,
        new_second, term, root,
    )
    mode = 0 if not has_normalize else (2 if relative else 1)
    group_root = np.zeros(1)
    if mode == 2:
        power, members = _group_power(root, group, int(groups))
        group_root = np.sqrt(power / members)
    step = np.empty(count)
    (_step_parallel if parallel else _step)(
        term, root, group, group_root, float(eta), mode, float(floor), float(relative),
        float(tiny), step,
    )
    return new_velocity, new_second, step


def tied_step(
    delta: np.ndarray, partner: np.ndarray, efficacy: np.ndarray, cap: float
) -> tuple[np.ndarray, np.ndarray]:
    """``Learner.apply``'s per-synapse arithmetic for an all-plastic learner without per-synapse
    rates or groups, to the bit: each tied pair moves by the mean of its two steps (``partner``,
    -1 for an untied synapse), and the new efficacy is bounded by ``cap``. Returns ``(step,
    efficacy)``."""
    assert njit is not None
    count = len(delta)
    out_delta, out_scale = np.empty(count), np.empty(count)
    (_tied_parallel if _elements_in_parallel(count) else _tied)(
        np.ascontiguousarray(delta, dtype=float), partner,
        np.ascontiguousarray(efficacy, dtype=float), float(cap), out_delta, out_scale,
    )
    return out_delta, out_scale
