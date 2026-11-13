"""The table of pairs that grows with its records (``LongTermMemory(grow=True)``, cadence#310).

The address of pairs files a record under every pair of its most driven units. The table as
``pairs`` builds it is sized to the capacity; this one starts at ``BASE`` cells and grows a cell
at a time, by linear hashing, so its bytes follow the records held and no write rebuilds it.

**A pair's place.** A pair of units ``(low, high)`` is mixed into 64 bits. Its low 32 bits name
the cell and its high bits the tag, which lies above the slot in an entry as in the fixed table.
The table holds ``cells + split`` cells, ``cells`` a power of two times ``BASE``: a pair's cell is
its low bits under ``cells - 1``, and where that cell is below ``split`` (it has split in this
round) its low bits under ``2 * cells - 1``.

**A split.** Cell ``split`` gives the entries whose next low bit is one to the new cell
``cells + split``. An entry keeps its slot and its tag, not its pair, so the split reads the
slot's filed units (``filed``, one row a record) and walks their pairs for the one that lies in
this cell under this tag. When ``split`` reaches ``cells`` the round ends and ``cells`` doubles.

Every function has two forms that make the same table: a loop that Numba compiles where it is
installed, and a NumPy form a record at a time. ``CADENCE_FUSED=0`` forces the NumPy forms.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import numba

    _njit: Any = numba.njit
    _prange: Any = numba.prange
except ImportError:  # pragma: no cover
    _njit = None
    _prange = range

MIX = 0x9E3779B97F4A7C15
MIX2 = 0xD6E8FEB86659FD93
EMPTY = 0xFFFFFFFF
BASE = 64  # the cells of an empty table

COMPILED = _njit is not None and os.environ.get("CADENCE_FUSED", "1") != "0"


def threads() -> int:
    """The threads the compiled loops run on."""
    return int(numba.get_num_threads()) if _njit is not None else 1


def mixed(low: np.ndarray, high: np.ndarray, expansion: int) -> np.ndarray:
    """The 64 mixed bits of each pair of units ``(low, high)``, ``low < high``."""
    with np.errstate(over="ignore"):
        z = (low.astype(np.uint64) * np.uint64(expansion) + high.astype(np.uint64)) \
            * np.uint64(MIX)
        z ^= z >> np.uint64(32)
        z *= np.uint64(MIX2)
        z ^= z >> np.uint64(32)
    return np.asarray(z)


def place(z: np.ndarray, bits: int, cells: int, split: int) -> tuple[np.ndarray, np.ndarray]:
    """The cell and the tag (in an entry's high bits) of mixed pairs ``z`` in a table of
    ``cells + split`` cells."""
    low = z & np.uint64(0xFFFFFFFF)
    cell = low & np.uint64(cells - 1)
    again = cell < np.uint64(split)
    cell[again] = low[again] & np.uint64(2 * cells - 1)
    tag = ((z >> np.uint64(32)) & np.uint64((1 << (32 - bits)) - 1)) << np.uint64(bits)
    return cell.astype(np.int64), tag.astype(np.uint32)


def shape(target: int) -> tuple[int, int]:
    """The ``cells`` and ``split`` of a table of ``target`` cells."""
    cells = BASE
    while 2 * cells <= target:
        cells *= 2
    return cells, max(target - cells, 0)


# -- the compiled forms


def _count(units: Any, flat: Any, depth: Any, expansion: Any, bits: Any, capacity: Any,
           hits: Any, touched: Any, ats: Any, tags: Any, held: Any, cells: Any,
           split: Any) -> int:
    """The coarse stage over the growing table, as ``longterm._count_pairs`` over the fixed
    one: every pair of the cue's sorted ``units`` reads its cell, and each slot found under the
    pair's tag counts one in ``hits`` (to 255); the slots found are ``touched[:n]``."""
    mask = np.uint64(cells - 1)
    mask2 = np.uint64(2 * cells - 1)
    edge = np.uint64(split)
    tagmask = np.uint64((1 << (32 - bits)) - 1)
    mix, mix2 = np.uint64(MIX), np.uint64(MIX2)
    s32, up, low32 = np.uint64(32), np.uint64(bits), np.uint64(0xFFFFFFFF)
    wide = np.uint64(expansion)
    block = len(ats)
    n = 0
    k = 0
    seen = np.uint32(0)
    last = len(units) - 1
    for i in range(last):
        base = np.uint64(units[i]) * wide
        for j in range(i + 1, len(units)):
            z = (base + np.uint64(units[j])) * mix
            z ^= z >> s32
            z *= mix2
            z ^= z >> s32
            low = z & low32
            cell = low & mask
            if cell < edge:
                cell = low & mask2
            at = np.int64(cell) * depth
            ats[k] = at
            tags[k] = np.uint32(((z >> s32) & tagmask) << up)
            seen ^= flat[at]
            k += 1
            if k == block or (i == last - 1 and j == last):
                m = 0
                for q in range(k):
                    at, tag = ats[q], tags[q]
                    for d in range(depth):
                        slot = flat[at + d] ^ tag
                        held[m] = slot
                        m += slot < capacity
                for q in range(m):
                    slot = held[q]
                    count = hits[slot]
                    if count == 0:
                        touched[n] = slot
                        n += 1
                    if count < 255:
                        hits[slot] = count + 1
                k = 0
    tags[0] = seen  # the first pass's reads are kept: they fetch the rows
    return n


def _count_rows(rows: Any, tags: Any, capacity: Any, hits: Any, touched: Any) -> int:
    """The same count over cells' rows already gathered, ``rows[q]`` the row of pair ``q``."""
    n = 0
    depth = rows.shape[1]
    for q in range(rows.shape[0]):
        tag = tags[q]
        for d in range(depth):
            slot = rows[q, d] ^ tag
            if slot < capacity:
                count = hits[slot]
                if count == 0:
                    touched[n] = slot
                    n += 1
                if count < 255:
                    hits[slot] = count + 1
    return n


def _split(flat: Any, depth: Any, filed: Any, expansion: Any, bits: Any, cells: Any,
           split: Any, count: Any, old: Any) -> tuple[Any, Any, int]:
    """Split ``count`` cells in turn and return the table's ``cells`` and ``split`` after them,
    with the entries no pair of their slot's filed units placed (dropped)."""
    mix, mix2 = np.uint64(MIX), np.uint64(MIX2)
    s32, up, low32 = np.uint64(32), np.uint64(bits), np.uint64(0xFFFFFFFF)
    wide = np.uint64(expansion)
    slotmask = np.uint32((1 << bits) - 1)
    tagmask = np.uint64((1 << (32 - bits)) - 1)
    empty = np.uint32(0xFFFFFFFF)
    one = np.uint64(1)
    m = filed.shape[1]
    big, p = np.uint64(cells), np.uint64(split)
    lost = 0
    for _ in range(count):
        lo = np.int64(p) * depth
        hi = np.int64(big + p) * depth
        f = 0
        for d in range(depth):
            e = flat[lo + d]
            if e != empty:
                old[f] = e
                f += 1
            flat[lo + d] = empty
            flat[hi + d] = empty
        nl = 0
        nh = 0
        mask = big - one
        for a in range(f):
            e = old[a]
            if e == empty:  # placed with an equal entry before it
                continue
            slot = np.int64(e & slotmask)
            for i in range(m - 1):
                ui = filed[slot, i]
                if ui < 0:
                    break
                base = np.uint64(ui) * wide
                for j in range(i + 1, m):
                    uj = filed[slot, j]
                    if uj < 0:
                        break
                    z = (base + np.uint64(uj)) * mix
                    z ^= z >> s32
                    z *= mix2
                    z ^= z >> s32
                    low = z & low32
                    if (low & mask) != p:
                        continue
                    if np.uint32((((z >> s32) & tagmask) << up) | np.uint64(slot)) != e:
                        continue
                    for b in range(a, f):
                        if old[b] == e:
                            old[b] = empty
                            if low & big:
                                flat[hi + nh] = e
                                nh += 1
                            else:
                                flat[lo + nl] = e
                                nl += 1
                            break
            for b in range(a, f):
                if old[b] == e:
                    old[b] = empty
                    lost += 1
        p += one
        if p == big:
            big = big * np.uint64(2)
            p = np.uint64(0)
    return np.int64(big), np.int64(p), lost


def _file(filed: Any, first: Any, flat: Any, depth: Any, expansion: Any, bits: Any,
          cells: Any, split: Any, lo: Any, hi: Any) -> tuple[int, int]:
    """Enter the pairs of the records ``first, first + 1, ...`` (``filed`` their rows of units)
    whose cells lie in ``[lo, hi)``; a full cell takes no more. The entries entered and
    dropped."""
    mix, mix2 = np.uint64(MIX), np.uint64(MIX2)
    s32, up, low32 = np.uint64(32), np.uint64(bits), np.uint64(0xFFFFFFFF)
    wide = np.uint64(expansion)
    tagmask = np.uint64((1 << (32 - bits)) - 1)
    empty = np.uint32(0xFFFFFFFF)
    mask = np.uint64(cells - 1)
    mask2 = np.uint64(2 * cells - 1)
    edge = np.uint64(split)
    m = filed.shape[1]
    entered = 0
    dropped = 0
    for r in range(filed.shape[0]):
        slot = np.uint64(first + r)
        for i in range(m - 1):
            ui = filed[r, i]
            if ui < 0:
                break
            base = np.uint64(ui) * wide
            for j in range(i + 1, m):
                uj = filed[r, j]
                if uj < 0:
                    break
                z = (base + np.uint64(uj)) * mix
                z ^= z >> s32
                z *= mix2
                z ^= z >> s32
                low = z & low32
                cell = low & mask
                if cell < edge:
                    cell = low & mask2
                c = np.int64(cell)
                if c < lo or c >= hi:
                    continue
                at = c * depth
                e = np.uint32((((z >> s32) & tagmask) << up) | slot)
                placed = False
                for d in range(depth):
                    if flat[at + d] == empty:
                        flat[at + d] = e
                        placed = True
                        break
                if placed:
                    entered += 1
                else:
                    dropped += 1
    return entered, dropped


def _file_ranges(filed: Any, first: Any, flat: Any, depth: Any, expansion: Any, bits: Any,
                 cells: Any, split: Any, edges: Any, out: Any) -> None:
    """``_file`` over the ranges of cells ``[edges[t], edges[t + 1])``, one thread a range: no
    two threads write one cell. ``out[t]`` takes the range's entries entered and dropped."""
    for t in _prange(len(edges) - 1):
        entered, dropped = _FILE(filed, first, flat, depth, expansion, bits, cells, split,
                                 edges[t], edges[t + 1])
        out[t, 0] = entered
        out[t, 1] = dropped


def _unfile(row: Any, slot: Any, flat: Any, depth: Any, expansion: Any, bits: Any,
            cells: Any, split: Any) -> int:
    """The record in ``slot`` (``row`` its filed units) leaves its pairs' cells: each cell's
    last entry takes its place. The entries removed."""
    mix, mix2 = np.uint64(MIX), np.uint64(MIX2)
    s32, up, low32 = np.uint64(32), np.uint64(bits), np.uint64(0xFFFFFFFF)
    wide = np.uint64(expansion)
    tagmask = np.uint64((1 << (32 - bits)) - 1)
    empty = np.uint32(0xFFFFFFFF)
    mask = np.uint64(cells - 1)
    mask2 = np.uint64(2 * cells - 1)
    edge = np.uint64(split)
    m = len(row)
    removed = 0
    for i in range(m - 1):
        ui = row[i]
        if ui < 0:
            break
        base = np.uint64(ui) * wide
        for j in range(i + 1, m):
            uj = row[j]
            if uj < 0:
                break
            z = (base + np.uint64(uj)) * mix
            z ^= z >> s32
            z *= mix2
            z ^= z >> s32
            low = z & low32
            cell = low & mask
            if cell < edge:
                cell = low & mask2
            at = np.int64(cell) * depth
            e = np.uint32((((z >> s32) & tagmask) << up) | np.uint64(slot))
            found = -1
            held = 0
            for d in range(depth):
                v = flat[at + d]
                if v == empty:
                    break
                held += 1
                if found < 0 and v == e:
                    found = d
            if found >= 0:
                flat[at + found] = flat[at + held - 1]
                flat[at + held - 1] = empty
                removed += 1
    return removed


def _weakest(best: Any, unit: Any, k: Any) -> int:
    """The place of the weakest of ``k`` kept units (the higher unit among equal drives)."""
    least = 0
    for q in range(1, k):
        if best[q] < best[least] or (best[q] == best[least] and unit[q] > unit[least]):
            least = q
    return least


def _top(drive: Any, mean: Any, sums: Any, k: Any, out: Any) -> None:
    """Each row's ``k`` most driven units, those driven above zero, in ascending order of
    unit and then -1: ``out[r]`` for ``drive[r]`` less ``mean[r] * sums`` (a key's projection
    less its mean times the units' summed weights, as the address of pairs centres a key).
    Among equal drives the lower unit wins."""
    rows, wide = drive.shape
    for r in _prange(rows):
        best = np.empty(k, dtype=drive.dtype)
        unit = np.empty(k, dtype=np.int64)
        n = 0
        least = 0  # the place of the weakest kept unit once k are kept (the higher among equals)
        centre = mean[r]
        u = 0
        while u < wide and n < k:  # the first k units driven above zero
            v = drive[r, u] - centre * sums[u]
            if v > 0:
                best[n] = v
                unit[n] = u
                n += 1
            u += 1
        if n == k:
            least = _WEAKEST(best, unit, k)
        floor = best[least]  # once k are kept every unit kept is driven above zero, so a unit
        while u < wide:      # that passes the weakest passes zero: one test a unit
            v = drive[r, u] - centre * sums[u]
            if v > floor:
                best[least] = v
                unit[least] = u
                least = _WEAKEST(best, unit, k)
                floor = best[least]
            u += 1
        kept = np.sort(unit[:n])
        for q in range(k):
            out[r, q] = kept[q] if q < n else -1


def _top_few(keys: Any, projection: Any, sums: Any, k: Any, out: Any) -> None:
    """``_top`` for keys of few active coordinates, with the drive made here: each key drives
    the units by its active coordinates' weights alone (rows of ``projection``), less its mean
    times the units' summed weights ``sums``, as the address of pairs centres a key."""
    rows, width = keys.shape
    wide = projection.shape[1]
    for r in _prange(rows):
        mean = keys[r].mean()
        drive = np.empty(wide, dtype=projection.dtype)
        for u in range(wide):
            drive[u] = -mean * sums[u]
        for c in range(width):
            v = keys[r, c]
            if v != 0:
                for u in range(wide):
                    drive[u] += v * projection[c, u]
        best = np.empty(k, dtype=projection.dtype)
        unit = np.empty(k, dtype=np.int64)
        n = 0
        least = 0
        u = 0
        while u < wide and n < k:
            v = drive[u]
            if v > 0:
                best[n] = v
                unit[n] = u
                n += 1
            u += 1
        if n == k:
            least = _WEAKEST(best, unit, k)
        floor = best[least]
        while u < wide:
            v = drive[u]
            if v > floor:
                best[least] = v
                unit[least] = u
                least = _WEAKEST(best, unit, k)
                floor = best[least]
            u += 1
        kept = np.sort(unit[:n])
        for q in range(k):
            out[r, q] = kept[q] if q < n else -1


def _rows(array: Any, rows: Any, out: Any) -> None:
    """``out[i] = array[rows[i]]`` for rows of a table. Compiled, it runs without the
    interpreter's lock, so several threads fetch rows, and wait for the disk, side by side."""
    for i in range(len(rows)):
        out[i] = array[rows[i]]


_COUNT: Any = None
_COUNT_ROWS: Any = None
_ROWS: Any = None
_SPLIT: Any = None
_FILE: Any = _file
_FILE_RANGES: Any = None
_UNFILE: Any = None
_TOP: Any = None
_TOP_FEW: Any = None
_WEAKEST: Any = _weakest
if COMPILED:
    _WEAKEST = _njit(cache=True)(_weakest)
    _COUNT = _njit(cache=True)(_count)
    _COUNT_ROWS = _njit(cache=True)(_count_rows)
    _SPLIT = _njit(cache=True)(_split)
    _FILE = _njit(cache=True)(_file)
    _FILE_RANGES = _njit(cache=True, parallel=True)(_file_ranges)
    _UNFILE = _njit(cache=True)(_unfile)
    _TOP = _njit(cache=True, parallel=True)(_top)
    _TOP_FEW = _njit(cache=True, parallel=True)(_top_few)
    _ROWS = _njit(cache=True, nogil=True)(_rows)


# -- the NumPy forms, a record at a time


def pairs_of(row: np.ndarray, expansion: int) -> np.ndarray:
    """The mixed bits of every pair of a record's filed units ``row`` (ascending, then -1), in
    the order the compiled loops walk them."""
    units = np.asarray(row, dtype=np.int64)
    units = units[units >= 0]
    if len(units) < 2:
        return np.zeros(0, dtype=np.uint64)
    low, high = np.triu_indices(len(units), 1)
    return mixed(units[low], units[high], expansion)


def split_cells(flat: np.ndarray, depth: int, filed: np.ndarray, expansion: int, bits: int,
                cells: int, split: int, count: int) -> tuple[int, int, int]:
    """``_split`` through NumPy: the same table."""
    lost = 0
    slotmask = (1 << bits) - 1
    table = flat.reshape(-1, depth)
    for _ in range(int(count)):
        old = table[split][table[split] != EMPTY].copy()
        table[split] = EMPTY
        table[cells + split] = EMPTY
        low_row: list[int] = []
        high_row: list[int] = []
        left = old.tolist()
        for a in range(len(old)):
            e = left[a]
            if e == EMPTY:
                continue
            slot = e & slotmask
            z = pairs_of(filed[slot], expansion)
            low = z & np.uint64(0xFFFFFFFF)
            here = (low & np.uint64(cells - 1)) == np.uint64(split)
            tag = ((z >> np.uint64(32)) & np.uint64((1 << (32 - bits)) - 1)) << np.uint64(bits)
            same = (tag | np.uint64(slot)).astype(np.uint32) == np.uint32(e)
            for at in np.flatnonzero(here & same).tolist():
                for b in range(a, len(left)):
                    if left[b] == e:
                        left[b] = EMPTY
                        (high_row if int(low[at]) & cells else low_row).append(e)
                        break
            for b in range(a, len(left)):
                if left[b] == e:
                    left[b] = EMPTY
                    lost += 1
        table[split, : len(low_row)] = low_row
        table[cells + split, : len(high_row)] = high_row
        split += 1
        if split == cells:
            cells, split = 2 * cells, 0
    return cells, split, lost


def file_rows(filed: np.ndarray, first: int, flat: np.ndarray, depth: int, expansion: int,
              bits: int, cells: int, split: int) -> tuple[int, int]:
    """``_file`` through NumPy over every cell: the same table."""
    table = flat.reshape(-1, depth)
    entered = dropped = 0
    for r in range(len(filed)):
        z = pairs_of(filed[r], expansion)
        cell, tag = place(z, bits, cells, split)
        entries = (tag | np.uint32(first + r)).tolist()
        for c, e in zip(cell.tolist(), entries, strict=True):
            row = table[c]
            free = np.flatnonzero(row == EMPTY)
            if len(free):
                row[free[0]] = e
                entered += 1
            else:
                dropped += 1
    return entered, dropped


def unfile_row(row: np.ndarray, slot: int, flat: np.ndarray, depth: int, expansion: int,
               bits: int, cells: int, split: int) -> int:
    """``_unfile`` through NumPy: the same table."""
    table = flat.reshape(-1, depth)
    z = pairs_of(row, expansion)
    cell, tag = place(z, bits, cells, split)
    removed = 0
    for c, e in zip(cell.tolist(), (tag | np.uint32(slot)).tolist(), strict=True):
        line = table[c]
        held = int(np.count_nonzero(line != EMPTY))
        at = np.flatnonzero(line[:held] == e)
        if len(at):
            line[at[0]] = line[held - 1]
            line[held - 1] = EMPTY
            removed += 1
    return removed


def top_units(drive: np.ndarray, k: int) -> np.ndarray:
    """``_top`` through NumPy: each row's ``k`` most driven units above zero, ascending, then
    -1; among equal drives the lower unit wins."""
    rows, wide = drive.shape
    out = np.full((rows, k), -1, dtype=np.int32)
    # a stable order by descending drive keeps the lower unit among equals
    order = np.argsort(-drive, axis=1, kind="stable")[:, :k]
    kept = np.take_along_axis(drive, order, axis=1) > 0
    for r in range(rows):
        units = np.sort(order[r][kept[r]])
        out[r, : len(units)] = units
    return out
