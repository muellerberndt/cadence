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

The helpers of a row's sweep are inlined into the kernels: a compiled call pays atomic reference
counts per array argument, and at this rate a call per row costs as much as the row's arithmetic.
Inlined helpers still count their arrays, so the rows of a free sweep run in a kernel that counts
none (``_free_rows``). Index arithmetic is unsigned, which lets the loops vectorize (``_index``). A
free solve runs its checks and its chunks in one compiled call (``fused_solve``), and a state keeps
what it published (``_keep_published``), so a launch from it makes no activation it already holds.

Every kernel above keeps the NumPy loop's bits. ``_fast_solve``, under ``learning_fast_settle``, is
another solve of the same equations and does not: rows that stop apart and an activation that
vectorizes.
"""

from __future__ import annotations

import functools
import os
import types
import weakref
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


def _vectorized(kernel: _Kernel) -> _Kernel:
    """``numba.njit(cache=True, error_model="numpy")``: Python's error model branches on every
    divisor, which keeps a loop with a division from vectorizing."""
    assert njit is not None
    return cast(_Kernel, njit(cache=True, error_model="numpy")(kernel))


def _rows_builds(kernel: _Kernel, *, divides: bool = False) -> tuple[_Kernel, _Kernel]:
    """One source compiled twice: serially, and with its ``prange`` over rows on Numba's
    threads. ``prange`` is ``range`` in the serial build. The parallel build compiles a copy
    under a name of its own, because Numba's cache files a function by name and not by its
    parallel flag. ``divides``: a per-synapse pass with a division takes NumPy's error model,
    as ``_vectorized``, and a zero divisor gives what NumPy's form gives."""
    assert njit is not None
    errors = {"error_model": "numpy"} if divides else {}
    name = kernel.__name__ + "_parallel"
    on_threads = (True,) if kernel.__defaults__ else None  # a kernel's last parameter says which
    copy = types.FunctionType(
        kernel.__code__, kernel.__globals__, name, on_threads, kernel.__closure__
    )
    copy.__qualname__ = kernel.__qualname__ + "_parallel"
    serial = njit(cache=True, **errors)(kernel)
    parallel = njit(cache=True, parallel=True, **errors)(copy)
    return cast(_Kernel, serial), cast(_Kernel, parallel)


def _inlined(kernel: _Kernel) -> _Kernel:
    """``numba.njit(inline="always")``: a per-neuron or per-row helper merged into its caller, so
    no call is paid per neuron or per row. Each array of a compiled call costs two atomic
    operations, and an inlined helper's arrays cost them too where the compiler cannot drop the
    pair (``_unreferenced``)."""
    assert njit is not None
    return cast(_Kernel, njit(inline="always")(kernel))


def _unreferenced(kernel: _Kernel) -> _Kernel:
    """``numba.njit(cache=True, _nrt=False)``: a kernel that allocates nothing and counts no
    references, whatever is inlined into it. A helper inlined under the runtime still costs atomic
    operations per array, and at a row's rate they cost about as much as its exponentials."""
    assert njit is not None
    return cast(_Kernel, njit(cache=True, _nrt=False)(kernel))


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


# A fast solve settles its rows past the tolerance that qualifies them, as a solve that checks
# every 32 sweeps does: a row stops within this, and a row that does not get there stops at the
# tolerance once this many sweeps are spent.
_FAST_SETTLED = 1e-5
_FAST_PATIENCE = 96

# The entries of the record at the end of a state's kept array (``_keep_published``)
_RECORD = 6

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

    @_inlined
    def _index(x: int) -> Any:
        """A count or an offset as the unsigned index it is."""
        return np.uint64(x)

    @_compiled
    def rows_finite(x: np.ndarray) -> bool:
        """Whether every entry of the ``(rows, columns)`` array ``x`` is finite: a finite number
        less itself is zero, anything else NaN. A loop that vectorizes, without the array of
        flags ``np.isfinite(x).all()`` makes."""
        finite = True
        width = _index(x.shape[1])
        for b in range(x.shape[0]):
            for i in range(width):
                d = x[b, i]
                finite = finite & (d - d == 0.0)
        return finite

    @_compiled
    def _standing(drive: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """``drive + bias``, a new C-ordered array."""
        batch, n = drive.shape
        out = np.empty((batch, n))
        width = _index(n)
        for b in range(batch):
            for i in range(width):
                out[b, i] = drive[b, i] + bias[i]
        return out

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """The offsets of one cached product per block of ``pairs`` and the ranges a block
        lands on."""
        cache_offset = np.zeros(pairs.shape[0] + 1, dtype=np.int64)
        receives = np.zeros(starts.shape[0] - 1, dtype=np.bool_)
        for q in range(pairs.shape[0]):
            b = pair_post[pairs[q]]
            cache_offset[q + 1] = cache_offset[q] + batch * (starts[b + 1] - starts[b])
            receives[b] = True
        return cache_offset, receives

    @_compiled
    def _kept_room(
        columns: int, size: int, size_up: int, ranges: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One array for what a state keeps, and its parts as views: the activations range by
        range (``columns`` entries), the products of the basal blocks (``size``) and of the
        apical blocks (``size_up``), one flag per range, 1 where the products the range feeds
        are stale, then what ``_keep_published`` writes."""
        kept = np.empty(columns + size + size_up + ranges + 2 * columns + _RECORD)
        up, flags = columns + size, columns + size + size_up
        return kept, kept[:columns], kept[columns:up], kept[up:flags], kept[flags:flags + ranges]

    @_compiled
    def _keep_published(
        v: np.ndarray, s: np.ndarray, kept: np.ndarray, masked: bool, slope: float,
        threshold: float, rest: float, leak: float,
    ) -> None:
        """The potentials and the activations a launch ends on at the end of ``kept``, then
        the record ``_publish`` reads them by: whether they hold (a mask's activations are not
        the rule's), their count, and the activation rule."""
        batch, n = v.shape
        count = _index(batch) * _index(n)
        head = _index(kept.shape[0]) - _index(_RECORD)
        kept[head] = 0.0 if masked else 1.0
        kept[head + _index(1)] = float(batch * n)
        kept[head + _index(2)] = slope
        kept[head + _index(3)] = threshold
        kept[head + _index(4)] = rest
        kept[head + _index(5)] = leak
        if masked:
            return
        at = head - _index(2) * count
        width = _index(n)
        for b in range(batch):
            for i in range(width):
                kept[at + i] = v[b, i]
            for i in range(width):
                kept[at + count + i] = s[b, i]
            at += width

    @_compiled
    def _publish(
        v: np.ndarray, s: np.ndarray, kept: np.ndarray, has_kept: bool, slope: float,
        threshold: float, rest: float, leak: float,
    ) -> bool:
        """The activations of the potentials ``v`` into ``s``: read from ``kept``
        (``_keep_published``) where they are the ones it holds under the same rule, which
        returns True, else each neuron's. Equal potentials have equal activations; a NaN never
        equals. From rest every neuron publishes the same thing."""
        batch, n = v.shape
        head = kept.shape[0] - _RECORD
        same = (
            has_kept and kept[head] == 1.0 and kept[head + 1] == float(batch * n)
            and kept[head + 2] == slope and kept[head + 3] == threshold
            and kept[head + 4] == rest and kept[head + 5] == leak
        )
        count = _index(batch) * _index(n)
        width = _index(n)
        if same:
            at = _index(head) - _index(2) * count
            b = 0
            while same and b < batch:
                for i in range(width):
                    same = same and v[b, i] == kept[at + i]
                at += width
                b += 1
        if same:
            at = _index(head) - count
            for b in range(batch):
                for i in range(width):
                    s[b, i] = kept[at + i]
                at += width
            return True
        if not v.any():
            row = np.empty(n)
            _activation(np.zeros(n), slope, threshold, rest, leak, row)
            for b in range(batch):
                for i in range(width):
                    s[b, i] = row[i]
        else:
            for b in range(batch):
                _activation(v[b], slope, threshold, rest, leak, s[b])
        return False

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
        order = np.arange(size.shape[0])
        for q in range(1, size.shape[0]):  # stable; a sort of NumPy's costs a second to compile
            task, at = order[q], q
            while at > 0 and size[order[at - 1]] > size[task]:
                order[at] = order[at - 1]
                at -= 1
            order[at] = task
        return order, feeds

    @_inlined
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
        first sweep (``first``), otherwise each block whose source range moved; returns their
        count."""
        count = pairs.shape[0]
        due = 0
        for j in range(order.shape[0]):
            q = order[j]
            k = pairs[q] if q < count else apical_pairs[q - count]
            if first or moved_range[pair_pre[k]]:
                todo[due] = q
                due += 1
        return due

    @_inlined
    def _copy_columns(
        b: int, s: np.ndarray, columns: np.ndarray, starts: np.ndarray, feeds: np.ndarray
    ) -> None:
        """Row ``b``'s activations into ``columns``, the batch's activations range by range:
        each range ``feeds`` names as one contiguous ``(batch, width)`` matrix at ``batch *
        starts[r]``, the operand of its products."""
        batch = s.shape[0]
        for r in range(feeds.shape[0]):
            if feeds[r]:
                a0 = _index(starts[r])
                na = _index(starts[r + 1]) - a0
                base = _index(batch) * a0 + _index(b) * na
                for i in range(na):
                    columns[base + i] = s[b, a0 + i]

    @_compiled
    def _reusable(
        columns: np.ndarray,
        kept: np.ndarray,
        given: np.ndarray,
        starts: np.ndarray,
        feeds: np.ndarray,
        moved_range: np.ndarray,
    ) -> None:
        """Take the given products (``given``, laid out as ``kept``) that hold for ``columns``:
        a range some block reads keeps the products it feeds where they are not stale and its
        activations equal the given ones in every row (a NaN never does); every other range is
        marked moved, so its products are made anew. Equal activations give the same products:
        what a check or a sweep reads from them is the same to the bit."""
        batch = columns.shape[0] // starts[starts.shape[0] - 1]
        ranges = feeds.shape[0]
        flags = given.shape[0] - _RECORD - 2 * columns.shape[0] - ranges
        for r in range(ranges):
            held = feeds[r] and given[flags + r] == 0.0
            i, stop = batch * starts[r], batch * starts[r + 1]
            while held and i < stop:
                held = columns[i] == given[i]
                i += 1
            moved_range[r] = not held
        for i in range(columns.shape[0], flags):  # the products; those not held are remade
            kept[i] = given[i]

    @_inlined
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

    @_inlined
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
        block's cached product added in block order. Indices that cannot be negative are
        unsigned here and in ``_copy_columns``: no wraparound is computed per entry, so the
        loops vectorize, with the same sums."""
        for r in range(receives.shape[0]):
            if receives[r]:
                for i in range(_index(starts[r]), _index(starts[r + 1])):
                    out[b, i] = 0.0
        for q in range(pairs.shape[0]):
            k = pairs[q]
            b0 = _index(starts[pair_post[k]])
            nb = _index(starts[pair_post[k] + 1]) - b0
            c0 = _index(cache_offset[q]) + _index(b) * nb
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

    @_inlined
    def _sweep_range(
        b: int,
        r: int,
        v: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        ahead: np.ndarray,
        starts: np.ndarray,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        change: float,
        moved: float,
    ) -> tuple[float, float, bool, bool]:
        """Range ``r`` of row ``b`` takes a step with no nudge, adaptation or mask on it: the next
        potentials into ``ahead`` in one loop that vectorizes, then each neuron that moved
        publishes its activation. The same arithmetic as ``_sweep_row``'s per-neuron loop.
        Returns the row's running movement, its largest, and whether the range's activation and
        potential moved."""
        range_moved = False
        potential_moved = False
        first, stop = _index(starts[r]), _index(starts[r + 1])
        for i in range(first, stop):
            x = v[b, i]
            ahead[b, i] = x + dt * (synaptic_input[b, i] + standing[b, i] - x)
        for i in range(first, stop):
            vn = ahead[b, i]
            if vn != v[b, i]:  # a neuron whose potential did not move publishes what it did
                potential_moved = True
                v[b, i] = vn
                sn = _act_scalar(vn, slope, threshold, rest, leak)
                d = abs(sn - s[b, i])
                change += d
                if d > 0.0:
                    range_moved = True
                    if d > moved:
                        moved = d
                s[b, i] = sn
        return change, moved, range_moved, potential_moved

    @_inlined
    def _sweep_row_plain(
        b: int,
        v: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        ahead: np.ndarray,
        starts: np.ndarray,
        frozen: np.ndarray,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        row_range: np.ndarray,
        row_potential: np.ndarray,
        activity_change: np.ndarray,
    ) -> float:
        """``_sweep_row`` with no nudge, no adaptation and no mask, the free settle."""
        ranges = starts.shape[0] - 1
        moved = 0.0
        change = activity_change[b]
        for r in range(ranges):
            range_moved = False
            potential_moved = False
            if not frozen[r]:
                change, moved, range_moved, potential_moved = _sweep_range(
                    b, r, v, s, standing, synaptic_input, ahead, starts, dt, slope, threshold,
                    rest, leak, change, moved,
                )
            row_range[b, r] = range_moved
            row_potential[b, r] = potential_moved
        activity_change[b] = change
        return moved

    @_inlined
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
        rule: tuple[float, float, float, float, float],
        has_adapt: bool,
        adapt: tuple[float, float],
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
        scratch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        row_range: np.ndarray,
        row_potential: np.ndarray,
        activity_change: np.ndarray,
    ) -> float:
        """Every neuron of row ``b`` takes one step; returns the row's largest movement and
        marks the ranges whose potential (``row_potential``) and activation (``row_range``)
        moved. ``rule`` is ``(dt, slope, threshold, rest, leak)``, ``adapt`` ``(strength, time
        constant)`` and ``scratch`` every row's softmax arrays ``(p, zmax, total)``, the next
        potentials ``ahead`` and ``quiet``, the ranges nothing but synapses and the drive reach
        (``_sweep_range``): tuples keep the call within what Numba can inline."""
        dt, slope, threshold, rest, leak = rule
        adapt_strength, adapt_tau = adapt
        p, zmax, total, ahead, quiet = scratch
        ranges = starts.shape[0] - 1
        moved = 0.0
        # the row's running total and range flags live in registers and are written once: rows
        # on other threads write the neighbouring entries
        change = activity_change[b]
        # the nudge on this row, from the activations before the step: one softmax per group
        if has_nudge and softmax_t[0] > 0.0:
            _softmax_groups(s[b], group, gid, ngroups, softmax_t, p[b], zmax[b], total[b])
        for r in range(ranges):
            range_moved = False
            potential_moved = False
            if quiet[r] and not frozen[r]:
                change, moved, range_moved, potential_moved = _sweep_range(
                    b, r, v, s, standing, synaptic_input, ahead, starts, dt, slope, threshold,
                    rest, leak, change, moved,
                )
            for i in range(starts[r], starts[r] if frozen[r] or quiet[r] else starts[r + 1]):
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
                    vn += dt * beta * weight[b] * (target[b, i] - p[b, position[i]]) * nmask[i]
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
            row_range[b, r] = range_moved
            row_potential[b, r] = potential_moved
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

    @_inlined
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

    @_unreferenced
    def _free_rows(
        first: int,
        stop: int,
        v: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        synaptic_input: np.ndarray,
        ahead: np.ndarray,
        cache: np.ndarray,
        cache_offset: np.ndarray,
        pairs: np.ndarray,
        pair_post: np.ndarray,
        starts: np.ndarray,
        receives: np.ndarray,
        frozen: np.ndarray,
        moving: np.ndarray,
        columns: np.ndarray,
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        row_moved: np.ndarray,
        row_range: np.ndarray,
        row_potential: np.ndarray,
        activity_change: np.ndarray,
    ) -> None:
        """The rows ``first`` to ``stop`` of a free sweep: no nudge, adaptation, mask or
        compartment. Each row's input, its step and its activations into ``columns``."""
        for b in range(first, stop):
            _gather(b, synaptic_input, cache, cache_offset, pairs, pair_post, starts, receives)
            row_moved[b] = _sweep_row_plain(
                b, v, s, standing, synaptic_input, ahead, starts, frozen, dt, slope, threshold,
                rest, leak, row_range, row_potential, activity_change,
            )
            _copy_columns(b, s, columns, starts, moving)

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
        given: np.ndarray,
        has_given: bool,
        on_threads: bool = False,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """The settle: per sweep the block products of the batch, then every row's step (the
        rows in parallel in the parallel build, ``on_threads``), then the batch-wide reading and
        stop. The first sweep takes the given products (``has_given``: what the starting state
        keeps, laid out as ``_kept_room``) that hold for its activations. Returns the sweeps taken,
        the activity change, and what the last state keeps: its activations, the cached
        products, and the ranges whose products its last sweep left stale."""
        batch, n = v.shape
        ranges = starts.shape[0] - 1
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)  # an output neuron's place in the softmax group
        for k in range(group.shape[0]):
            position[group[k]] = k
        cache_offset, receives = _product_cache(batch, pairs, pair_post, starts)
        cache_up_offset, receives_up = _product_cache(batch, apical_pairs, pair_post, starts)
        kept, columns, cache, cache_up, stale = _kept_room(
            batch * n, int(cache_offset[-1]), int(cache_up_offset[-1]), ranges
        )
        order, feeds = _task_order(pairs, apical_pairs, pair_pre, pair_post, starts)
        todo = np.empty(order.shape[0], dtype=np.int64)
        basal_count = pairs.shape[0]
        for b in prange(batch):
            _copy_columns(b, s, columns, starts, feeds)
        moved_range = np.ones(ranges, dtype=np.bool_)  # activation changed in the last step
        if has_given:  # the first sweep makes only the products the given ones do not hold for
            _reusable(columns, kept, given, starts, feeds, moved_range)
        synaptic_input = np.zeros((batch, n))
        ahead = np.empty((batch, n))
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
        frozen = np.zeros(ranges, dtype=np.bool_)  # a range that can never change again
        moving = np.empty(ranges, dtype=np.bool_)  # a range some block reads that can still change
        plain = not (has_nudge or has_adapt or masked or has_apical)
        # the ranges with no neuron a nudge or an anchor reaches, under no adaptation and no mask
        quiet = np.zeros(ranges, dtype=np.bool_)
        if not (has_adapt or masked or plain):
            for r in range(ranges):
                quiet[r] = True
                for i in range(starts[r], starts[r + 1]):
                    if nmask[i] > 0.0 or anchor_gain[i] > 0.0:
                        quiet[r] = False
        rule = (dt, slope, threshold, rest, leak)
        adapt = (adapt_strength, adapt_tau)
        scratch = (p, zmax, total, ahead, quiet)
        taken = 0
        for t in range(steps):
            # the block products of the batch, distinct blocks on distinct threads
            due = _due(t == 0 and not has_given, order, pairs, apical_pairs, pair_pre,
                       moved_range, todo)
            for j in prange(due):
                q = todo[j]
                if q < basal_count:
                    _block_product(columns, flat, pairs[q], pair_pre, pair_post, starts, offset,
                                   cache, cache_offset[q])
                else:
                    _block_product(columns, flat_apical, apical_pairs[q - basal_count], pair_pre,
                                   pair_post, starts, offset, cache_up,
                                   cache_up_offset[q - basal_count])
            for r in range(ranges):
                moving[r] = feeds[r] and not frozen[r]
            # every row's step, the rows on distinct threads
            if plain and not on_threads:
                _free_rows(
                    0, batch, v, s, standing, synaptic_input, ahead, cache, cache_offset, pairs,
                    pair_post, starts, receives, frozen, moving, columns, dt, slope, threshold,
                    rest, leak, row_moved, row_range, row_potential, activity_change,
                )
            elif plain:
                for b in prange(batch):
                    _free_rows(
                        b, b + 1, v, s, standing, synaptic_input, ahead, cache, cache_offset,
                        pairs, pair_post, starts, receives, frozen, moving, columns, dt, slope,
                        threshold, rest, leak, row_moved, row_range, row_potential,
                        activity_change,
                    )
            else:
                for b in prange(batch):
                    _gather(b, synaptic_input, cache, cache_offset, pairs, pair_post, starts,
                            receives)
                    if has_apical:
                        _gather(b, raw, cache_up, cache_up_offset, apical_pairs, pair_post, starts,
                                receives_up)
                        row_moved[b] = _sweep_row_apical(
                            b, v, a, s, standing, synaptic_input, raw, starts, frozen, keep, masked,
                            dt, slope, threshold, rest, leak, has_adapt, adapt_strength, adapt_tau,
                            has_nudge, target, nmask, group, gid, ngroups, position, beta,
                            softmax_t, weight, anchor, anchor_gain, p[b], zmax[b], total[b],
                            pool_of, pool_index, pool_starts, pool_run, fraction, floor2, tiny,
                            held, has_held, basal[b], gain[b], values[b], frames[b], left[b],
                            row_range[b], row_potential[b], activity_change,
                        )
                    else:
                        row_moved[b] = _sweep_row(
                            b, v, a, s, standing, synaptic_input, starts, frozen, keep, masked,
                            rule, has_adapt, adapt, has_nudge, target, nmask, group, gid, ngroups,
                            position, beta, softmax_t, weight, anchor, anchor_gain, scratch,
                            row_range, row_potential, activity_change,
                        )
                    _copy_columns(b, s, columns, starts, moving)  # what the next products read
            moved = _close_sweep(t, row_moved, row_range, row_potential, moved_range, freezable,
                                 frozen)
            taken = t + 1
            if use_tolerance and moved < tolerance:
                break
        for r in range(ranges):
            stale[r] = 1.0 if moved_range[r] else 0.0
        _keep_published(v, s, kept, masked, slope, threshold, rest, leak)
        return taken, activity_change, kept

    _kernel, _kernel_parallel = _rows_builds(_settle_rows)

    # -- the residual: one transport (the products kept where they hold), then each row's error

    @_inlined
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
        settle adds, without a step; infinity where it is not finite. ``p``, ``zmax`` and
        ``total`` are every row's softmax scratch."""
        n = v.shape[1]
        if has_nudge and softmax_t[0] > 0.0:
            _softmax_groups(s[b], group, gid, ngroups, softmax_t, p[b], zmax[b], total[b])
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
                    tot += beta * weight[b] * (target[b, i] - p[b, position[i]]) * nmask[i]
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
        given: np.ndarray,
        has_given: bool,
        in_place: bool,
    ) -> np.ndarray:
        """The fixed-point equation error of every row, in one compiled call: one block
        transport at the activations ``s`` of ``v`` under the mask, then each row's error into
        ``out``. The transport takes the given products (``has_given``: what the state keeps,
        laid out as ``_kept_room``) that hold for its activations and makes the rest; with
        ``in_place`` the given array holds these very activations, and its stale products are
        made in it, which leaves it true for every state that holds it. Returns what the state
        keeps after the check: its activations and its products, every one current."""
        batch, n = v.shape
        ranges = starts.shape[0] - 1
        group = np.flatnonzero(nmask > 0.0)
        position = np.full(n, -1, dtype=np.int64)
        for k in range(group.shape[0]):
            position[group[k]] = k
        cache_offset, receives = _product_cache(batch, pairs, pair_post, starts)
        cache_up_offset, receives_up = _product_cache(batch, apical_pairs, pair_post, starts)
        order, feeds = _task_order(pairs, apical_pairs, pair_pre, pair_post, starts)
        basal_count = pairs.shape[0]
        moved_range = np.ones(ranges, dtype=np.bool_)
        if in_place:
            size = int(cache_offset[-1])
            kept = given
            up, flags = batch * n + size, batch * n + size + int(cache_up_offset[-1])
            cache, cache_up = kept[batch * n:up], kept[up:flags]
            stale = kept[flags:flags + ranges]
            columns = kept[:batch * n]
            for r in range(ranges):
                moved_range[r] = not (feeds[r] and stale[r] == 0.0)
        else:
            kept, columns, cache, cache_up, stale = _kept_room(
                batch * n, int(cache_offset[-1]), int(cache_up_offset[-1]), ranges
            )
            for b in prange(batch):
                _copy_columns(b, s, columns, starts, feeds)
            if has_given:
                _reusable(columns, kept, given, starts, feeds, moved_range)
            _keep_published(v, s, kept, masked, slope, threshold, rest, leak)
        stale[:] = 0.0
        for j in prange(order.shape[0]):
            q = order[j]
            if q < basal_count:
                if moved_range[pair_pre[pairs[q]]]:
                    _block_product(columns, flat, pairs[q], pair_pre, pair_post, starts, offset,
                                   cache, cache_offset[q])
            elif moved_range[pair_pre[apical_pairs[q - basal_count]]]:
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
                    beta, softmax_t, weight, anchor, anchor_gain, p, zmax, total,
                )
        return kept

    _residual_kernel, _residual_kernel_parallel = _rows_builds(_residual_rows)
    # the serial builds as a solve calls them
    _sweeps, _check = _kernel, _residual_kernel

    @_compiled
    def _solve_rows(
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
        dt: float,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        strength: float,
        tau: float,
        target: np.ndarray,
        nmask: np.ndarray,
        gid: np.ndarray,
        ngroups: int,
        beta: float,
        softmax_t: np.ndarray,
        weight: np.ndarray,
        anchor: np.ndarray,
        anchor_gain: np.ndarray,
        pool_of: np.ndarray,
        pool_index: np.ndarray,
        pool_starts: np.ndarray,
        pool_run: np.ndarray,
        held: np.ndarray,
        budget: int,
        chunk: int,
        tolerance: float,
        out: np.ndarray,
        given: np.ndarray,
        has_given: bool,
        in_place: bool,
    ) -> tuple[int, int, int, np.ndarray, np.ndarray]:
        """A free solve of a state whose activations ``s`` are published, in one compiled call:
        the check, then chunks of sweeps and their checks while a row is over ``tolerance``
        and sweeps are left, as ``Brain._equilibrate_local`` makes them one launch each, in place
        on ``v``, ``a`` and ``s``. ``out`` ends on each row's error. Returns the sweeps, the
        checks, the last chunk's sweeps and activity change, and what the state keeps."""
        batch = v.shape[0]
        # variables and not the constants: to the compiler a constant argument is a type of its
        # own, and the kernels would compile again for the calls a launch makes
        off, on = batch < 0, batch >= 0
        kept = _check(
            v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre,
            pair_post, offset, keep, off, dt, slope, threshold, rest, leak, off, strength,
            off, target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain,
            off, pool_of, pool_index, pool_starts, pool_run, 1.0, 1.0, 0.0, held, off, out,
            given, has_given, in_place,
        )
        used, checks, taken = 0, 1, 0
        change = np.zeros(batch)
        while used < budget:
            qualified = True
            finite = True
            for b in range(batch):
                qualified = qualified and out[b] <= tolerance
                finite = finite and out[b] <= 1.7976931348623157e308
            if qualified:
                break
            if not finite and not (rows_finite(v) and rows_finite(s) and rows_finite(a)):
                break  # a divergent state is no warm start for another chunk
            taken, change, kept = _sweeps(
                v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre,
                pair_post, offset, freezable, keep, off, dt, slope, threshold, rest, leak,
                off, strength, tau, off, target, nmask, gid, ngroups, beta, softmax_t,
                weight, anchor, anchor_gain, off, pool_of, pool_index, pool_starts, pool_run,
                1.0, 1.0, 0.0, held, off, min(chunk, budget - used), 0.0, off, kept, on,
            )
            used += taken
            kept = _check(
                v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre,
                pair_post, offset, keep, off, dt, slope, threshold, rest, leak, off, strength,
                off, target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain,
                off, pool_of, pool_index, pool_starts, pool_run, 1.0, 1.0, 0.0, held, off,
                out, kept, on, on,
            )
            checks += 1
        return used, checks, taken, change, kept


if njit is not None:

    @_inlined
    def _act_plain(
        x: float, slope: float, threshold: float, rest: float, up: float, down: float
    ) -> float:
        """The activation in plain arithmetic a loop can vectorize: the exponential as a power
        series at 1/256 of the argument, squared eight times, the argument held within +-36,
        where the sigmoid has saturated. Within about 1e-14 of the rule's."""
        z = -slope * (x - threshold)
        z = 36.0 if z > 36.0 else (-36.0 if z < -36.0 else z)
        y = z * 0.00390625
        e = 1.0 + y * (1.0 + y * (0.5 + y * (0.16666666666666666 + y * (
            0.041666666666666664 + y * (0.008333333333333333 + y * (
                0.001388888888888889 + y * 0.0001984126984126984))))))
        e = e * e
        e = e * e
        e = e * e
        e = e * e
        e = e * e
        e = e * e
        e = e * e
        e = e * e
        q = 1.0 / (1.0 + e) - rest
        return q * up if q > 0.0 else q * down

    @_inlined
    def _row_home(
        home: int, b: int, batch: int, work_v: np.ndarray, v: np.ndarray, s: np.ndarray,
        columns: np.ndarray, starts: np.ndarray,
    ) -> None:
        """Working row ``b`` of ``_fast_solve`` into row ``home`` of the state."""
        width = _index(starts[starts.shape[0] - 1])
        for i in range(width):
            v[home, i] = work_v[b, i]
        for r in range(starts.shape[0] - 1):
            a0 = _index(starts[r])
            wr = _index(starts[r + 1]) - a0
            at = _index(batch) * a0 + _index(b) * wr
            for i in range(wr):
                s[home, a0 + i] = columns[at + i]

    @_vectorized
    def _fast_solve(
        v: np.ndarray,
        s: np.ndarray,
        standing: np.ndarray,
        flat: np.ndarray,
        starts: np.ndarray,
        pair_pre: np.ndarray,
        pair_post: np.ndarray,
        offset: np.ndarray,
        slope: float,
        threshold: float,
        rest: float,
        leak: float,
        budget: int,
        chunk: int,
        tolerance: float,
        settled: float,
        out: np.ndarray,
    ) -> tuple[int, int, int, int, np.ndarray]:
        """A free solve at dt = 1 and no mask, in the ordinary order: every range reads the sweep
        before. A row stops at the check it is within ``settled``, or within ``tolerance`` once
        ``_FAST_PATIENCE`` sweeps are spent. Rows that stop leave
        the working arrays, so the block products cover the rows still settling. ``v``, ``s``
        and ``out`` end on every row's state and error. Returns the sweeps, the checks, the last
        chunk's sweeps, the sweeps over the rows and what the state keeps: its publication,
        every product stale."""
        batch, n = v.shape
        ranges = starts.shape[0] - 1
        pairs = pair_pre.shape[0]
        up = 1.0 / (1.0 - rest)
        down = leak / rest if rest > 0.0 else 0.0
        fed = np.zeros(ranges, dtype=np.bool_)
        widest = 1
        for q in range(pairs):
            fed[pair_post[q]] = True
            widest = max(widest, starts[pair_post[q] + 1] - starts[pair_post[q]])
        total = np.empty((batch, n))
        work_v = np.empty((batch, n))
        work_standing = np.empty((batch, n))
        columns = np.empty(batch * n)
        scratch = np.empty(batch * widest)
        rows = np.arange(batch)
        m = batch
        un = _index(n)
        for b in range(batch):
            for i in range(un):
                work_v[b, i] = v[b, i]
                work_standing[b, i] = standing[b, i]
                total[b, i] = 0.0
            for r in range(ranges):
                a0 = _index(starts[r])
                a1 = _index(starts[r + 1])
                at = _index(batch) * a0 + _index(b) * (a1 - a0)
                if fed[r]:
                    for i in range(a1 - a0):
                        columns[at + i] = s[b, a0 + i]
                else:
                    for i in range(a1 - a0):
                        x = work_standing[b, a0 + i]
                        work_v[b, a0 + i] = x
                        columns[at + i] = _act_plain(x, slope, threshold, rest, up, down)
        for q in range(pairs):
            a, c = pair_pre[q], pair_post[q]
            if not fed[a]:
                a0, a1, b0, b1 = starts[a], starts[a + 1], starts[c], starts[c + 1]
                nb = b1 - b0
                product = scratch[: batch * nb].reshape(batch, nb)
                np.dot(columns[batch * a0 : batch * a1].reshape(batch, a1 - a0),
                       flat[offset[q] : offset[q + 1]].reshape(a1 - a0, nb), product)
                ub0, unb = _index(b0), _index(nb)
                for b in range(batch):
                    for i in range(unb):
                        total[b, ub0 + i] += product[b, i]
        used, checks, taken, row_sweeps = 0, 0, 0, 0
        while True:
            # the check at the rows still settling: the rows' errors, each row that meets the
            # tolerance leaves
            for b in range(m):
                out[rows[b]] = 0.0
            for r in range(ranges):
                if not fed[r]:
                    continue
                b0, b1 = starts[r], starts[r + 1]
                nb = b1 - b0
                ub0, unb = _index(b0), _index(nb)
                check = np.empty((m, nb))
                for b in range(m):
                    for i in range(unb):
                        check[b, i] = (
                            total[b, ub0 + i] + work_standing[b, ub0 + i] - work_v[b, ub0 + i]
                        )
                for q in range(pairs):
                    a = pair_pre[q]
                    if pair_post[q] == r and fed[a]:
                        a0, a1 = starts[a], starts[a + 1]
                        product = scratch[: m * nb].reshape(m, nb)
                        source = columns[batch * a0 : batch * a0 + m * (a1 - a0)]
                        np.dot(source.reshape(m, a1 - a0),
                               flat[offset[q] : offset[q + 1]].reshape(a1 - a0, nb), product)
                        for b in range(m):
                            for i in range(unb):
                                check[b, i] += product[b, i]
                for b in range(m):
                    worst = out[rows[b]]
                    for i in range(unb):
                        d = abs(check[b, i])
                        if d > worst or d != d:
                            worst = d
                    out[rows[b]] = worst
            checks += 1
            b = 0
            while b < m:
                error = out[rows[b]]
                if (
                    error <= settled
                    or (error <= tolerance and used >= _FAST_PATIENCE)
                    or not (error <= 1e300)
                ):
                    # this row is done: its state goes home, the last active row takes its place
                    home = rows[b]
                    if not (out[home] <= 1e300):
                        out[home] = np.inf
                    _row_home(home, b, batch, work_v, v, s, columns, starts)
                    m -= 1
                    if b != m:
                        for r in range(ranges):
                            a0 = _index(starts[r])
                            wr = _index(starts[r + 1]) - a0
                            at = _index(batch) * a0 + _index(b) * wr
                            at2 = _index(batch) * a0 + _index(m) * wr
                            for i in range(wr):
                                columns[at + i] = columns[at2 + i]
                        for i in range(un):
                            work_v[b, i] = work_v[m, i]
                            work_standing[b, i] = work_standing[m, i]
                            total[b, i] = total[m, i]
                        rows[b] = rows[m]
                else:
                    b += 1
            if m == 0 or used >= budget:
                break
            taken = min(chunk, budget - used)
            for _ in range(taken):
                for r in range(ranges):
                    if not fed[r]:
                        continue
                    b0, b1 = starts[r], starts[r + 1]
                    nb = b1 - b0
                    ub0, unb = _index(b0), _index(nb)
                    for b in range(m):
                        for i in range(unb):
                            work_v[b, ub0 + i] = total[b, ub0 + i] + work_standing[b, ub0 + i]
                    for q in range(pairs):
                        a = pair_pre[q]
                        if pair_post[q] == r and fed[a]:
                            a0, a1 = starts[a], starts[a + 1]
                            product = scratch[: m * nb].reshape(m, nb)
                            source = columns[batch * a0 : batch * a0 + m * (a1 - a0)]
                            np.dot(source.reshape(m, a1 - a0),
                                   flat[offset[q] : offset[q + 1]].reshape(a1 - a0, nb), product)
                            for b in range(m):
                                for i in range(unb):
                                    work_v[b, ub0 + i] += product[b, i]
                for r in range(ranges):  # the next potentials are in; now the activations
                    if not fed[r]:
                        continue
                    b0, b1 = starts[r], starts[r + 1]
                    nb = b1 - b0
                    ub0, unb = _index(b0), _index(nb)
                    base = _index(batch) * ub0
                    for b in range(m):
                        at = base + _index(b) * unb
                        for i in range(unb):
                            columns[at + i] = _act_plain(
                                work_v[b, ub0 + i], slope, threshold, rest, up, down
                            )
            used += taken
            row_sweeps += taken * m
        for b in range(m):  # the budget ran out: the rows still settling go home as they are
            _row_home(rows[b], b, batch, work_v, v, s, columns, starts)
        # what the state keeps: its publication, every product stale
        feeds = np.zeros(ranges, dtype=np.bool_)
        size = 0
        for q in range(pairs):
            feeds[pair_pre[q]] = True
            size += batch * (starts[pair_post[q] + 1] - starts[pair_post[q]])
        kept, kept_columns, _, _, stale = _kept_room(batch * n, size, 0, ranges)
        for b in range(batch):
            _copy_columns(b, s, kept_columns, starts, feeds)
        for r in range(ranges):
            stale[r] = 1.0
        _keep_published(v, s, kept, False, slope, threshold, rest, leak)
        return used, checks, taken, row_sweeps, kept


@functools.lru_cache(maxsize=32)
def _no_nudge(
    batch: int, n: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """The arrays of a settle without a nudge, made once per batch size and width: the kernels
    read them and never write them, so one set serves every such call."""
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


def _nudge_arrays(
    nudge: Nudge | None, batch: int, n: int
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """The nudge as the dense arrays the kernels read: target, mask, group id per neuron
    (0..ngroups-1, -1 for none), the group count, beta, the per-neuron softmax temperatures (0 for a
    plain nudge) and the per-row weight. A softmax nudge drives only grouped neurons. A nudge
    keeps its arrays for the calls that read it again (a phase's settle and its check)."""
    if nudge is None:
        return _no_nudge(batch, n)
    held = nudge.__dict__.get("_kernel_arrays")
    if held is not None and held[0] == batch:
        return cast(tuple[Any, ...], held[1])
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
    arrays = target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain
    nudge.__dict__["_kernel_arrays"] = (batch, arrays)
    return arrays


# the mask argument of a settle or check without a mask: the kernels read a mask only where
# ``masked`` is set, so one unread array stands in for every such call
_NO_MASK = np.ones((1, 1))


def _fed_ranges(layout: Layout) -> np.ndarray:
    """The ranges some block lands on, found once per layout."""
    fed = layout.__dict__.get("_fed_ranges")
    if fed is None:
        fed = np.zeros(layout.ranges, dtype=np.bool_)
        fed[layout.pair_post] = True
        layout.__dict__["_fed_ranges"] = fed
    return cast(np.ndarray, fed)


# ``(compartment, basal blocks, apical flat blocks, apical blocks)`` of a graph with an apical
# compartment; the ``flat`` beside it then holds the basal synapses alone
ApicalBlocks = tuple["Apical", Sequence[int], np.ndarray, Sequence[int]]


def _compartment(
    apical: ApicalBlocks | None, layout: Layout, flat: np.ndarray, nudge: Nudge | None,
    batch: int, n: int,
) -> tuple[Any, ...]:
    """The kernels' arguments for the blocks and the apical compartment: ``(pairs,
    flat_apical, apical_pairs, has_apical, pool_of, pool_index, pool_starts, pool_run,
    fraction, floor2, tiny, held, has_held)``. Without a compartment every block is basal. The
    arrays a layout or a compartment determines are made once and kept on it: the kernels read
    them and never write them."""
    if apical is None:
        plain = layout.__dict__.get("_plain_compartment")
        if plain is None or plain[0] != n:
            plain = (
                n, np.arange(layout.pairs, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.full(n, -1, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((1, 1)),
            )
            layout.__dict__["_plain_compartment"] = plain
        _, pairs, apical_pairs, pool_of, pool_index, pool_starts, pool_run, no_held = plain
        return (
            pairs, flat, apical_pairs, False, pool_of, pool_index, pool_starts, pool_run,
            1.0, 1.0, 0.0, no_held, False,
        )
    from .brain import _APICAL_TINY

    compartment, basal_pairs, flat_apical, apical_pairs = apical
    pools = len(compartment.members)
    held = None if nudge is None else nudge.apical_gain
    if held is not None and held.shape != (batch, pools):
        raise ValueError("a nudge's apical_gain needs one entry per row and pool")
    tables = compartment.__dict__.get("_kernel_pools")
    if tables is None or tables[0] != n:
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
        tables = (n, pool_of, pool_index, pool_starts, pool_run)
        compartment.__dict__["_kernel_pools"] = tables
    _, pool_of, pool_index, pool_starts, pool_run = tables
    return (
        np.asarray(basal_pairs, dtype=np.int64), np.ascontiguousarray(flat_apical, dtype=float),
        np.asarray(apical_pairs, dtype=np.int64), True, pool_of, pool_index, pool_starts, pool_run,
        compartment.fraction, compartment.floor * compartment.floor, _APICAL_TINY,
        np.zeros((1, 1)) if held is None else np.ascontiguousarray(held, dtype=float),
        held is not None,
    )


# What a settled or checked state keeps of its transport (``BrainState``), so a check of it or
# a settle from it makes only the block products of the ranges whose activations differ from
# the ones they were made from, and publishes its potentials' activations by reading them:
# ``(blocks, apical blocks, rows, kept)``, weak references to the flat blocks the products were
# made with (they keep no blocks alive), the batch, and the kernels' array of the activations
# range by range, the products, the stale ranges, the potentials and activations it ended on
# (``_kept_room``)
Products = tuple[Any, Any, int, np.ndarray]

# the given products of a launch without any: the kernels read them only where ``has_given``
# is set
_NO_PRODUCTS = (np.zeros(1), False)


def _given(
    products: Products | None, flat: np.ndarray, flat_apical: np.ndarray, batch: int
) -> tuple[np.ndarray, bool]:
    """The kernels' given products: those kept, when made with these blocks for as many rows."""
    if products is None:
        return _NO_PRODUCTS
    made, made_up, rows, kept = products
    if rows != batch or made() is not flat:
        return _NO_PRODUCTS
    if (flat if made_up is made else made_up()) is not flat_apical:
        return _NO_PRODUCTS
    return kept, True


def _held(products: Products | None) -> tuple[np.ndarray, bool]:
    """What a state keeps of its publication, whichever blocks its products were made with."""
    return _NO_PRODUCTS if products is None else (products[3], True)


def _rule(neuron_model: NeuronModel) -> tuple[Any, ...]:
    """The neuron rule as the kernels read it, read once per call (the rest emission is an
    exponential): ``(dt, slope, threshold, rest emission, leak, adaptation strength,
    adaptation time constant)``."""
    m, adapt = neuron_model, neuron_model.adaptation
    return (
        m.dt, m.slope, m.threshold, m.rest_emission, m.leak,
        0.0 if adapt is None else adapt.strength, 1.0 if adapt is None else adapt.tau_steps,
    )


def _kept(flat: np.ndarray, flat_apical: np.ndarray, batch: int, kept: np.ndarray) -> Products:
    made = weakref.ref(flat)
    return made, made if flat_apical is flat else weakref.ref(flat_apical), batch, kept


def fused_settle(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray | None,
    neuron_model: NeuronModel,
    nudge: Nudge | None,
    steps: int,
    tolerance: float | None,
    activation: np.ndarray | None = None,
    apical: ApicalBlocks | None = None,
    products: Products | None = None,
) -> tuple[np.ndarray, int, np.ndarray, Products | None]:
    """Run the fused kernel in place on ``v`` and ``a``; returns ``(s, taken, activity_change,
    products)``.

    ``activation`` is accepted and ignored. Publication is recomputed
    from ``v`` under the current mask, since cached readback may use an old mask; ``keep`` is
    None without a mask. A settle of no sweeps publishes the activations and launches nothing.
    ``apical`` settles a graph with an apical compartment (``ApicalBlocks``); ``flat`` then
    holds its basal synapses. With more than one row and more than one Numba thread the rows
    settle in parallel, to the same bits. ``products`` are those the starting state keeps:
    the first sweep reads them where they hold; the products returned are the last state's
    (at no sweeps, the given ones).
    """
    assert njit is not None
    batch, n = v.shape
    standing = _standing(np.ascontiguousarray(drive), bias)
    if keep is None:
        masked = False
        keep = _NO_MASK
    else:
        # one (batch, n) array of the kernel's type, a shared mask broadcast
        keep = np.ascontiguousarray(np.broadcast_to(np.asarray(keep, float), (batch, n)))
        masked = bool((keep != 1.0).any())
    dt, slope, threshold, rest, leak, strength, tau = _rule(neuron_model)
    s = np.empty_like(v)  # the state's own memory order
    _publish(v, s, *_held(products), slope, threshold, rest, leak)
    if masked:
        s *= keep
    if steps == 0:  # no sweep: the activations are published and nothing else is read
        if apical is not None:  # a held gain's shape is checked at no sweeps too
            _compartment(apical, layout, flat, nudge, batch, n)
        return s, 0, np.zeros(batch), products
    has_adapt = neuron_model.adaptation is not None
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        nudge, batch, n
    )
    lay = layout
    freezable = ~_fed_ranges(lay)  # nothing arrives: once still, still for good
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
    taken, activity_change, kept = _launch(
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
        keep,
        masked,
        dt,
        slope,
        threshold,
        rest,
        leak,
        has_adapt,
        strength,
        tau,
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
        *_given(products, flat, flat_apical, batch),
    )
    return s, taken, activity_change, _kept(flat, flat_apical, batch, kept)


def fused_solve(
    v: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    neuron_model: NeuronModel,
    budget: int,
    chunk: int,
    tolerance: float,
    products: Products | None,
) -> tuple[np.ndarray, int, int, int, np.ndarray, Products] | None:
    """A free solve without a mask, a nudge or a compartment in one compiled call, in place on a
    C-ordered state of published activations ``s``: what ``Brain._equilibrate_local`` makes of the
    state in launches of one settle or one check. Returns each row's error, the sweeps, the
    checks, the last chunk's sweeps and activity change and the products; None for adaptation,
    rows that settle on Numba's threads, or a state in another memory order, which the launches
    serve."""
    assert njit is not None
    batch, n = v.shape
    if (
        neuron_model.adaptation is not None
        or _in_parallel(batch, n)
        or not (v.flags.c_contiguous and a.flags.c_contiguous)
    ):
        return None
    dt, slope, threshold, rest, leak, strength, tau = _rule(neuron_model)
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        None, batch, n
    )
    compartment = _compartment(None, layout, flat, None, batch, n)
    pairs, flat_apical, apical_pairs = compartment[:3]
    pool_of, pool_index, pool_starts, pool_run = compartment[4:8]
    held = compartment[11]
    freezable = ~_fed_ranges(layout)
    # the check's activations are the state's own: published under no mask, they are the rule's
    given, has_given = _given(products, flat, flat_apical, batch)
    in_place = has_given and _publish(v, s, *_held(products), slope, threshold, rest, leak)
    out = np.empty(batch)
    used, checks, taken, change, kept = _solve_rows(
        v, a, s, _standing(np.ascontiguousarray(drive), bias), flat, pairs, flat_apical,
        apical_pairs, layout.starts, layout.pair_pre, layout.pair_post, layout.offset, freezable,
        _NO_MASK, dt, slope, threshold, rest, leak, strength, tau, target, nmask, gid, ngroups,
        beta, softmax_t, weight, anchor, anchor_gain, pool_of, pool_index, pool_starts, pool_run,
        held, budget, chunk, float(tolerance), out, given, has_given, in_place,
    )
    return out, used, checks, taken, change, _kept(flat, flat_apical, batch, kept)


def fused_fast_solve(
    v: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    neuron_model: NeuronModel,
    budget: int,
    chunk: int,
    tolerance: float,
) -> tuple[np.ndarray, int, int, int, int, Products] | None:
    """``Brain.equilibrate``'s ``fast`` solve in place on a C-ordered state of published activations
    (``_fast_solve``). Returns each row's error, the sweeps, the checks, the last chunk's sweeps,
    the sweeps over the rows and what the state keeps; None under adaptation or another ``dt``,
    which take the ordinary solve."""
    assert njit is not None
    batch, n = v.shape
    if (
        neuron_model.adaptation is not None
        or neuron_model.dt != 1.0
        or not (v.flags.c_contiguous and s.flags.c_contiguous)
    ):
        return None
    _, slope, threshold, rest, leak, _, _ = _rule(neuron_model)
    out = np.empty(batch)
    used, checks, taken, row_sweeps, kept = _fast_solve(
        v, s, _standing(np.ascontiguousarray(drive), bias), flat, layout.starts, layout.pair_pre,
        layout.pair_post, layout.offset, slope, threshold, rest, leak, budget, chunk,
        float(tolerance), min(float(tolerance), _FAST_SETTLED), out,
    )
    return out, used, checks, taken, row_sweeps, _kept(flat, flat, batch, kept)


def fused_residual(
    v: np.ndarray,
    a: np.ndarray,
    drive: np.ndarray,
    bias: np.ndarray,
    layout: Layout,
    flat: np.ndarray,
    keep: np.ndarray | None,
    neuron_model: NeuronModel,
    nudge: Nudge | None,
    apical: ApicalBlocks | None = None,
    products: Products | None = None,
) -> tuple[np.ndarray, Products]:
    """``Brain.residual`` for a host state of a blocked connectome, in one compiled call;
    returns the error and the state's products.

    The same equations as the NumPy path: the activation of ``v`` under the mask (``keep``,
    None without one), one block transport, the potential equation with adaptation, nudge and
    apical compartment (``apical``, as in ``fused_settle``), the mask's projection, and the
    adaptation equation; the maximum absolute error per row, infinity where it is not finite.
    The transport reads the products the state keeps (``products``) where they hold and makes
    the rest; the products returned are all current. Nothing is changed.
    """
    assert njit is not None
    batch, n = v.shape
    dt, slope, threshold, rest, leak, strength, _ = _rule(neuron_model)
    if keep is None:
        masked = False
        keep = _NO_MASK
    else:
        keep = np.ascontiguousarray(np.broadcast_to(np.asarray(keep, float), (batch, n)))
        masked = bool((keep != 1.0).any())
    target, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = _nudge_arrays(
        nudge, batch, n
    )
    pairs, flat_apical, apical_pairs, *compartment = _compartment(
        apical, layout, flat, nudge, batch, n
    )
    out = np.empty(batch)
    v = np.ascontiguousarray(v, dtype=float)
    s = np.empty_like(v)
    held = _publish(v, s, *_held(products), slope, threshold, rest, leak)
    if masked:
        s *= keep
    given, has_given = _given(products, flat, flat_apical, batch)
    kept = _launch(
        _residual_kernel,
        _residual_kernel_parallel,
        batch,
        n,
        v,
        np.ascontiguousarray(a, dtype=float),
        s,
        _standing(np.ascontiguousarray(drive), bias),
        flat,
        pairs,
        flat_apical,
        apical_pairs,
        layout.starts,
        layout.pair_pre,
        layout.pair_post,
        layout.offset,
        keep,
        masked,
        dt,
        slope,
        threshold,
        rest,
        leak,
        neuron_model.adaptation is not None,
        strength,
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
        given,
        has_given,
        held and has_given and not masked,
    )
    return out, _kept(flat, flat_apical, batch, kept)


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

    _pick, _pick_parallel = _rows_builds(_gather_entries, divides=True)

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

    def _plain_entries(
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
        eta: float,
        floor: float,
        new_velocity: np.ndarray,
        new_second: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Each synapse's moments and step in one pass: the momentum, the second moment and the
        absolute floor where they are set, then the rate."""
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
                out[e] = eta * (t / (np.sqrt(held / rho_correction) + floor))
            else:
                out[e] = eta * t

    _plain, _plain_parallel = _rows_builds(_plain_entries, divides=True)

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
        new_velocity: np.ndarray,
        new_second: np.ndarray,
        term: np.ndarray,
        root: np.ndarray,
    ) -> None:
        """The first pass of the projection-relative floor: each synapse's moments, the term
        its step divides and the root of its second moment."""
        for e in prange(raw.shape[0]):
            x = raw[e]
            t = x
            if has_momentum:
                moving = momentum * velocity[e] + momentum_rest * x
                new_velocity[e] = moving
                t = moving / momentum_correction
            held = rho * second[e] + rho_rest * (x * x)
            new_second[e] = held
            root[e] = np.sqrt(held / rho_correction)
            term[e] = t

    _moments, _moments_parallel = _rows_builds(_moment_entries, divides=True)

    def _step_entries(
        term: np.ndarray,
        root: np.ndarray,
        group: np.ndarray,
        group_root: np.ndarray,
        eta: float,
        share: float,
        tiny: float,
        out: np.ndarray,
    ) -> None:
        """The second pass: each step over its root plus its projection's share."""
        for e in prange(term.shape[0]):
            out[e] = eta * (term[e] / (root[e] + share * group_root[group[e]] + tiny))

    _step, _step_parallel = _rows_builds(_step_entries, divides=True)

    def _tied_entries(
        delta: np.ndarray,
        partner: np.ndarray,
        efficacy: np.ndarray,
        gain_pre: np.ndarray,
        cap: float,
        out_delta: np.ndarray,
        out_scale: np.ndarray,
        out_weight: np.ndarray,
    ) -> tuple[int, int, int]:
        """Each synapse's tied step, bounded efficacy and effective weight; returns how many
        of each are not finite (``x <= big and x >= -big`` fails for an infinity and a NaN)."""
        bad_delta = 0
        bad_scale = 0
        bad_weight = 0
        big = 1.7976931348623157e308
        for e in prange(delta.shape[0]):
            k = partner[e]
            d = 0.5 * (delta[e] + delta[k]) if k >= 0 else delta[e]
            out_delta[e] = d
            s = min(max(efficacy[e] + d, -cap), cap)
            out_scale[e] = s
            w = gain_pre[e] * s
            out_weight[e] = w
            if not (d <= big and d >= -big):
                bad_delta += 1
            if not (s <= big and s >= -big):
                bad_scale += 1
            if not (w <= big and w >= -big):
                bad_weight += 1
        return bad_delta, bad_scale, bad_weight

    _tied, _tied_parallel = _rows_builds(_tied_entries)


if njit is not None:

    @_compiled
    def _add_episode_terms(
        term: np.ndarray,
        anchored: np.ndarray,
        relative: np.ndarray,
        has_anchored: bool,
        has_relative: bool,
        told: np.ndarray,
        focus: np.ndarray,
        table: np.ndarray,
        last_bucket: int,
        outside: int,
        reach: int,
        none: int,
    ) -> None:
        """The anchored and relative terms of a read added to ``term`` ``(rows, slots)`` in place,
        anchored first: each slot's place in its stream's episode through ``table``, ``last_bucket``
        for an older place and ``outside`` for a place before the episode; each slot's age less
        the stream's focus, within ``reach``, ``none`` for a stream with no read before."""
        rows, slots = term.shape
        for b in range(rows):
            if has_anchored:
                t = told[b]
                for m in range(slots):
                    place = t - 1 - (slots - 1 - m)
                    if place < 0:
                        bucket = outside
                    else:
                        bucket = table[place] if place <= last_bucket else table[last_bucket]
                    term[b, m] += anchored[b, bucket]
            if has_relative:
                f = focus[b]
                for m in range(slots):
                    if f < 0:
                        bucket = none
                    else:
                        offset = slots - 1 - m - f
                        offset = min(offset, reach)
                        bucket = max(offset, -reach) + reach
                    term[b, m] += relative[b, bucket]


def add_episode_terms(
    term: np.ndarray,
    anchored: np.ndarray | None,
    relative: np.ndarray | None,
    told: np.ndarray,
    focus: np.ndarray,
    table: np.ndarray,
    outside: int,
    reach: int,
) -> None:
    """The recall read's anchored and relative position terms added to ``term``, a new array
    (``_add_episode_terms``)."""
    assert njit is not None
    _add_episode_terms(
        term, _NO_TERM if anchored is None else anchored,
        _NO_TERM if relative is None else relative, anchored is not None, relative is not None,
        told, focus, table, len(table) - 1, outside, reach, 2 * reach + 1,
    )


# the term of a read without it: the kernel reads it only where told to
_NO_TERM = np.zeros((1, 1))


if njit is not None:

    @_compiled
    def _clear_choice(s: np.ndarray, temperature: float) -> tuple[np.ndarray, bool]:
        """Each row's largest ``s / temperature`` and whether it stands clear of the others in
        every row, finite and by more than 1e-9: its softmax is then strictly the largest too."""
        rows, width = s.shape
        choice = np.empty(rows, dtype=np.int64)
        clear = True
        for b in range(rows):
            best = -np.inf
            second = -np.inf
            at = 0
            for i in range(width):
                z = s[b, i] / temperature
                if not abs(z) < np.inf:
                    clear = False
                if z > best:
                    second = best
                    best = z
                    at = i
                elif z > second:
                    second = z
            choice[b] = at
            if not best - second > 1e-9:
                clear = False
        return choice, clear


def clear_choice(s: np.ndarray, temperature: float) -> np.ndarray | None:
    """The greedy choice of one slot's activations ``(rows, width)`` where every row's largest
    stands clear of the rest, which is ``np.argmax`` of their softmax; None where one does not."""
    assert njit is not None
    choice, clear = _clear_choice(np.ascontiguousarray(s, dtype=float), float(temperature))
    return choice if clear else None


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
    """``Learner._update``'s per-synapse arithmetic in compiled passes, to the bit: the
    momentum (above zero), the second moment and its normalization (``rho`` above zero; the
    absolute ``floor``, or the projection-relative floor of share ``relative`` over ``group``),
    and the rate. Returns ``(velocity, second moment, step)``; the moments a setting leaves off
    come back as given. One pass without the relative floor; with it, a second after every
    projection's mean square is summed."""
    assert njit is not None
    count = len(raw)
    raw = np.ascontiguousarray(raw, dtype=float)
    has_momentum, has_normalize = momentum > 0, rho > 0
    new_velocity = np.empty(count) if has_momentum else velocity
    new_second = np.empty(count) if has_normalize else second
    parallel = _elements_in_parallel(count)
    if not (has_normalize and relative):
        step = np.empty(count)
        (_plain_parallel if parallel else _plain)(
            raw, np.ascontiguousarray(velocity, dtype=float),
            np.ascontiguousarray(second, dtype=float), float(momentum), float(1 - momentum),
            float(momentum_correction), float(rho), float(1 - rho), float(rho_correction),
            has_momentum, has_normalize, float(eta), float(floor), new_velocity, new_second,
            step,
        )
        return new_velocity, new_second, step
    term, root = np.empty(count), np.empty(count)
    (_moments_parallel if parallel else _moments)(
        raw, np.ascontiguousarray(velocity, dtype=float), np.ascontiguousarray(second, dtype=float),
        float(momentum), float(1 - momentum), float(momentum_correction), float(rho),
        float(1 - rho), float(rho_correction), has_momentum, new_velocity, new_second, term, root,
    )
    power, members = _group_power(root, group, int(groups))
    step = np.empty(count)
    (_step_parallel if parallel else _step)(
        term, root, group, np.sqrt(power / members), float(eta), float(relative), float(tiny),
        step,
    )
    return new_velocity, new_second, step


def tied_step(
    delta: np.ndarray, partner: np.ndarray, efficacy: np.ndarray, gain_pre: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    """``Learner.apply``'s per-synapse arithmetic for an all-plastic learner without per-synapse
    rates or groups, to the bit: each tied pair moves by the mean of its two steps (``partner``,
    -1 for an untied synapse), the new efficacy is bounded by ``cap``, and the effective weight
    is ``gain_pre`` times it (the product ``Brain`` takes). Returns ``(step, efficacy, weights,
    finite)``, ``finite`` saying whether every step, every efficacy and every weight is finite,
    as ``np.isfinite(...).all()`` would."""
    assert njit is not None
    count = len(delta)
    out_delta, out_scale, out_weight = np.empty(count), np.empty(count), np.empty(count)
    bad = (_tied_parallel if _elements_in_parallel(count) else _tied)(
        np.ascontiguousarray(delta, dtype=float), partner,
        np.ascontiguousarray(efficacy, dtype=float), np.ascontiguousarray(gain_pre, dtype=float),
        float(cap), out_delta, out_scale, out_weight,
    )
    return out_delta, out_scale, out_weight, (bad[0] == 0, bad[1] == 0, bad[2] == 0)
