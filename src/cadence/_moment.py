"""The compiled moment (``learning_compiled_moment``, opt-in): the work around the solve kernels
of a moment on the CPU in compiled passes.

A moment's settle runs in the kernels of ``cadence.fused``. Around it the recall memory reads its
slots, the brain chooses, the memory writes the moment and a lesson steps the read's maps. On the
CPU each of these is many small NumPy operations at a few rows and slots, and at that size the
interpreter's cost per operation is most of the moment. Here each is one compiled pass of the same
equations: the read (the address, every position term, the attention and the reads of the values
and the nexts) and the address lesson (the scores, the prior, the attention it names and the
least-mean-squares steps of the address and position maps). The sums run in another order than
NumPy's and BLAS's, so the bits differ from the ordinary path, which stays the default.

The kernels take the memory's arrays as they are and allocate their outputs; a caller checks
what it keeps, as the ordinary path does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fused import _compiled, njit

__all__ = ["available", "recall_read", "address_lesson"]


def available() -> bool:
    """Whether the compiled moment can run here: Numba is installed and the compiled passes of
    ``cadence.fused`` are in use."""
    from .fused import compiled

    return compiled()


if njit is not None:

    @_compiled
    def _map_row(
        x: np.ndarray, c: np.ndarray, b: int, w_x: np.ndarray, w_c: np.ndarray, m: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """``out = x[b] @ w_x + c[b] @ w_c + m`` for one row: a map of the moment's term."""
        width = out.shape[0]
        for j in range(width):
            out[j] = 0.0
        for k in range(x.shape[1]):
            v = x[b, k]
            if v != 0.0:
                for j in range(width):
                    out[j] += v * w_x[k, j]
        for a in range(c.shape[1]):
            v = c[b, a]
            if v != 0.0:
                for j in range(width):
                    out[j] += v * w_c[a, j]
        for j in range(width):
            out[j] += m[j]

    @_compiled
    def _lean_row(
        x: np.ndarray, c: np.ndarray, b: int, s_x: np.ndarray, s_c: np.ndarray, s: np.ndarray
    ) -> float:
        """The slope map's one number for one row: ``x[b] @ s_x + c[b] @ s_c + s``."""
        total = 0.0
        for k in range(x.shape[1]):
            total += x[b, k] * s_x[k]
        for a in range(c.shape[1]):
            total += c[b, a] * s_c[a]
        return total + s[0]

    @_compiled
    def _position_row(
        x: np.ndarray, c: np.ndarray, b: int, slots: int,
        has_position: bool, p_x: np.ndarray, p_c: np.ndarray, p: np.ndarray,
        buckets: np.ndarray,
        has_anchor: bool, e_x: np.ndarray, e_c: np.ndarray, e: np.ndarray,
        told: np.ndarray, table: np.ndarray, outside: int,
        has_relative: bool, r_x: np.ndarray, r_c: np.ndarray, r: np.ndarray,
        focus: np.ndarray, reach: int,
        has_slope: bool, s_x: np.ndarray, s_c: np.ndarray, s: np.ndarray, slope_unit: float,
        has_grid: bool, g_x: np.ndarray, g_c: np.ndarray, g: np.ndarray, residues: np.ndarray,
        position: int,
        scratch: np.ndarray, term: np.ndarray,
    ) -> None:
        """Row ``b``'s position term at every slot into ``term`` (``Recall.position_term``): the
        table over the age buckets, the anchored term over each slot's place in the episode, the
        relative term over its offset from the previous read's focus, the slope along its age
        and the grid's residues past the table."""
        for m in range(slots):
            term[m] = 0.0
        if not has_position:
            return
        _map_row(x, c, b, p_x, p_c, p, scratch[: p.shape[0]])
        for m in range(slots):
            term[m] = scratch[buckets[m]]
        last = table.shape[0] - 1
        if has_anchor:
            _map_row(x, c, b, e_x, e_c, e, scratch[: e.shape[0]])
            t = told[b]
            for m in range(slots):
                place = t - 1 - (slots - 1 - m)
                if place < 0:
                    bucket = outside
                else:
                    bucket = table[place] if place <= last else table[last]
                term[m] += scratch[bucket]
        if has_relative:
            _map_row(x, c, b, r_x, r_c, r, scratch[: r.shape[0]])
            f = focus[b]
            for m in range(slots):
                if f < 0:
                    bucket = 2 * reach + 1
                else:
                    offset = min(slots - 1 - m - f, reach)
                    bucket = max(offset, -reach) + reach
                term[m] += scratch[bucket]
        if has_slope:
            lean = _lean_row(x, c, b, s_x, s_c, s)
            for m in range(slots):
                term[m] += lean * ((slots - 1 - m) / slope_unit)
        if has_grid:
            _map_row(x, c, b, g_x, g_c, g, scratch[: g.shape[0]])
            for m in range(slots):
                if slots - 1 - m >= position:
                    total = 0.0
                    for i in range(residues.shape[0]):
                        total += scratch[residues[i, m]]
                    term[m] += total

    @_compiled
    def _read_kernel(
        x: np.ndarray, c: np.ndarray, heard: float,
        w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray,
        keys: np.ndarray, values: np.ndarray, nexts: np.ndarray, has_next: bool,
        held: np.ndarray, temperature: float, recency: float,
        has_position: bool, p_x: np.ndarray, p_c: np.ndarray, p: np.ndarray,
        buckets: np.ndarray,
        has_anchor: bool, e_x: np.ndarray, e_c: np.ndarray, e: np.ndarray,
        told: np.ndarray, table: np.ndarray, outside: int,
        has_relative: bool, r_x: np.ndarray, r_c: np.ndarray, r: np.ndarray,
        focus: np.ndarray, reach: int,
        has_slope: bool, s_x: np.ndarray, s_c: np.ndarray, s: np.ndarray, slope_unit: float,
        has_grid: bool, g_x: np.ndarray, g_c: np.ndarray, g: np.ndarray, residues: np.ndarray,
        position: int, widest: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every row's key, address, attention over its slots and reads of the values and the
        nexts (``Recall.stimulate`` with one head)."""
        rows, width = x.shape
        slots = keys.shape[1]
        address_width = w.shape[0]
        value_width = values.shape[2]
        key = np.empty((rows, width))
        q = np.empty((rows, address_width))
        attention = np.empty((rows, slots))
        read = np.zeros((rows, value_width))
        following = np.zeros((rows, value_width if has_next else 0))
        scratch = np.empty(max(widest, address_width, 1))
        term = np.empty(slots)
        for b in range(rows):
            for k in range(width):
                key[b, k] = heard * x[b, k]
            _map_row(x, c, b, w_x, w_c, w, scratch[:address_width])
            for j in range(address_width):
                q[b, j] = key[b, j] + scratch[j]
            _position_row(
                x, c, b, slots, has_position, p_x, p_c, p, buckets, has_anchor, e_x, e_c, e,
                told, table, outside, has_relative, r_x, r_c, r, focus, reach, has_slope, s_x,
                s_c, s, slope_unit, has_grid, g_x, g_c, g, residues, position, scratch, term,
            )
            top = -np.inf
            for m in range(slots):
                if held[b, m]:
                    dot = 0.0
                    for j in range(address_width):
                        dot += q[b, j] * keys[b, m, j]
                    z = dot / temperature
                    if recency != 0.0:
                        z -= recency * (slots - 1 - m)
                    z += term[m]
                    if z > top:
                        top = z
                    attention[b, m] = z
                else:
                    attention[b, m] = -np.inf
            if top == -np.inf:  # a stream that holds no slot reads silence
                for m in range(slots):
                    attention[b, m] = 0.0
                continue
            total = 0.0
            for m in range(slots):
                v = np.exp(attention[b, m] - top)
                attention[b, m] = v
                total += v
            for m in range(slots):
                a = attention[b, m] / total
                attention[b, m] = a
                if a != 0.0:
                    for j in range(value_width):
                        read[b, j] += a * values[b, m, j]
                    if has_next:
                        for j in range(value_width):
                            following[b, j] += a * nexts[b, m, j]
        return key, q, attention, read, following


_NONE2 = np.zeros((1, 1))
_NONE1 = np.zeros(1)
_NONE_INT2 = np.zeros((1, 1), dtype=np.int64)


def _maps(maps: list[Any], head: int) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """A head's maps as the kernels take them, or placeholders without them."""
    if not maps:
        return False, _NONE2, _NONE2, _NONE1
    m_x, m_c, m = maps[head]
    return True, m_x, m_c, m


def _slope(maps: list[Any], head: int) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    if not maps:
        return False, _NONE1, _NONE1, _NONE1
    s_x, s_c, s = maps[head]
    return True, s_x, s_c, np.atleast_1d(s)


def position_arguments(recall: Any, head: int) -> tuple[Any, ...]:
    """The position maps and tables of ``recall``'s head ``head`` in the kernels' order."""
    has_position, p_x, p_c, p = _maps(recall.position_maps, head)
    has_anchor, e_x, e_c, e = _maps(recall.anchor_maps, head)
    has_relative, r_x, r_c, r = _maps(recall.relative_maps, head)
    has_slope, s_x, s_c, s = _slope(recall.slope_maps, head)
    has_grid, g_x, g_c, g = _maps(recall.grid_maps, head)
    slots = recall.slots
    buckets = recall.buckets() if has_position else np.zeros(slots, dtype=np.int64)
    table = recall._age_buckets() if has_position else np.zeros(1, dtype=np.int64)
    residues = recall.grid_buckets() if has_grid else _NONE_INT2
    widest = max(len(p), len(e), len(r), len(g))
    return (
        has_position, p_x, p_c, p, np.ascontiguousarray(buckets, dtype=np.int64),
        has_anchor, e_x, e_c, e, np.asarray(recall.told, dtype=np.int64),
        np.ascontiguousarray(table, dtype=np.int64), int(recall.position_buckets),
        has_relative, r_x, r_c, r, np.asarray(recall.focus, dtype=np.int64),
        int(recall.RELATIVE),
        has_slope, s_x, s_c, s, float(recall.SLOPE),
        has_grid, g_x, g_c, g, np.ascontiguousarray(residues, dtype=np.int64),
        int(recall.position), widest,
    )


def recall_read(
    recall: Any, x: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(key, address, attention, read of the values, read of the nexts)`` of head 0 at the
    moments ``x`` with the cortex ``c`` of the moment before."""
    w_x, w_c, w = recall.head_maps[0]
    return _read_kernel(  # type: ignore[no-any-return]
        x, c, float(recall.heard), w_x, w_c, w, recall.keys, recall.values,
        recall.nexts if recall.next else recall.values, bool(recall.next), recall.held,
        float(recall.temperature), float(recall.recency), *position_arguments(recall, 0),
    )


# -- the address lesson

if njit is not None:

    _numeric = njit(cache=True, error_model="numpy")

    @_numeric
    def _readback(
        movement: np.ndarray, weights: np.ndarray, synapses: np.ndarray, sources: np.ndarray,
        targets: np.ndarray, width: int,
    ) -> np.ndarray:
        """``movement @ W.T`` for a port's synapses into the association (``Recall.readback``):
        per source neuron, its synapses' weights times their association neuron's movement."""
        rows = movement.shape[0]
        out = np.zeros((rows, width))
        for b in range(rows):
            for e in range(synapses.shape[0]):
                out[b, sources[e]] += weights[synapses[e]] * movement[b, targets[e]]
        return out

    @_numeric
    def _lms_into(
        w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray, step: np.ndarray, rate: float,
        x: np.ndarray, c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A map of the moment after the least-mean-squares ``step`` ``(rows, columns)`` at
        ``rate`` on each row's inputs ``[x, c, 1]``, averaged over the rows (``_lms``)."""
        rows, columns = step.shape
        s_x = np.zeros((x.shape[1], columns))
        s_c = np.zeros((c.shape[1], columns))
        s = np.zeros(columns)
        for b in range(rows):
            for k in range(x.shape[1]):
                v = x[b, k]
                if v != 0.0:
                    for j in range(columns):
                        s_x[k, j] += v * step[b, j]
            for a in range(c.shape[1]):
                v = c[b, a]
                if v != 0.0:
                    for j in range(columns):
                        s_c[a, j] += v * step[b, j]
            for j in range(columns):
                s[j] += step[b, j]
        new_x = np.empty_like(s_x)
        new_c = np.empty_like(s_c)
        new = np.empty(columns)
        for k in range(x.shape[1]):
            for j in range(columns):
                new_x[k, j] = w_x[k, j] + rate * s_x[k, j] / rows
        for a in range(c.shape[1]):
            for j in range(columns):
                new_c[a, j] = w_c[a, j] + rate * s_c[a, j] / rows
        for j in range(columns):
            new[j] = w[j] + rate * (s[j] / rows)
        return new_x, new_c, new

    @_numeric
    def _finite(a: np.ndarray) -> bool:
        for v in a.ravel():
            if not (v - v == 0.0):
                return False
        return True

    @_numeric
    def _lesson_kernel(
        rows: np.ndarray, movement: np.ndarray, weights: np.ndarray,
        value_synapses: np.ndarray, value_sources: np.ndarray, value_targets: np.ndarray,
        next_synapses: np.ndarray, next_sources: np.ndarray, next_targets: np.ndarray,
        has_next: bool, values: np.ndarray, nexts: np.ndarray, keys: np.ndarray,
        held: np.ndarray, masked: bool, inputs: np.ndarray, cortex: np.ndarray,
        address: np.ndarray, key: np.ndarray,
        address_temperature: float, temperature: float, recency: float,
        has_position: bool, p_x: np.ndarray, p_c: np.ndarray, p: np.ndarray,
        buckets: np.ndarray,
        has_anchor: bool, e_x: np.ndarray, e_c: np.ndarray, e: np.ndarray,
        told: np.ndarray, table: np.ndarray, outside: int,
        has_relative: bool, r_x: np.ndarray, r_c: np.ndarray, r: np.ndarray,
        focus: np.ndarray, reach: int,
        has_slope: bool, s_x: np.ndarray, s_c: np.ndarray, s: np.ndarray, slope_unit: float,
        has_grid: bool, g_x: np.ndarray, g_c: np.ndarray, g: np.ndarray, residues: np.ndarray,
        position: int, widest: int,
        w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray, address_rate: float,
        position_rate: float, slope_rate: float,
    ) -> tuple[Any, ...]:
        """The address lesson of one head (``Recall.learn``): the rows' scores of the slots
        against the lesson's movement, the attention the lesson names, the prior and the steps of
        the address and the position maps. Returns ``(moved, error, address maps, address
        finite, position maps, position finite)``; ``moved`` counts the rows that move and the
        maps are new arrays."""
        slots = keys.shape[1]
        key_width = keys.shape[2]
        value_width = values.shape[2]
        d_value = _readback(movement, weights, value_synapses, value_sources, value_targets,
                            value_width)
        d_next = (
            _readback(movement, weights, next_synapses, next_sources, next_targets, value_width)
            if has_next else d_value
        )
        count = rows.shape[0]
        moving = np.zeros(count, dtype=np.bool_)
        norms = np.zeros(count)
        for i in range(count):
            total = 0.0
            for j in range(value_width):
                total += d_value[i, j] * d_value[i, j]
            if has_next:
                for j in range(value_width):
                    total += d_next[i, j] * d_next[i, j]
            norms[i] = np.sqrt(total)
            moving[i] = norms[i] > 1e-12
            if masked and moving[i]:
                anything = False
                for m in range(slots):
                    anything = anything or held[rows[i], m]
                moving[i] = anything
        moved = 0
        for i in range(count):
            if moving[i]:
                moved += 1
        x = np.empty((moved, inputs.shape[1]))
        c = np.empty((moved, cortex.shape[1]))
        error = np.zeros((moved, key_width))
        step = np.empty((moved, key_width))
        gradient = np.zeros((moved, slots))
        target = np.zeros((moved, slots))
        prior = np.empty((moved, slots))
        energy = np.empty(moved)
        scratch = np.empty(max(widest, 1))
        term = np.empty(slots)
        logits = np.empty(slots)
        k = 0
        for i in range(count):
            if not moving[i]:
                continue
            b = rows[i]
            for j in range(inputs.shape[1]):
                x[k, j] = inputs[b, j]
            for j in range(cortex.shape[1]):
                c[k, j] = cortex[b, j]
            _position_row(
                inputs, cortex, b, slots, has_position, p_x, p_c, p, buckets, has_anchor, e_x,
                e_c, e, told, table, outside, has_relative, r_x, r_c, r, focus, reach, has_slope,
                s_x, s_c, s, slope_unit, has_grid, g_x, g_c, g, residues, position, scratch, term,
            )
            top = -np.inf
            best = -np.inf
            for m in range(slots):
                if masked and not held[b, m]:
                    prior[k, m] = -np.inf
                    logits[m] = -np.inf
                    continue
                score = 0.0
                for j in range(value_width):
                    score += d_value[i, j] * values[b, m, j]
                if has_next:
                    for j in range(value_width):
                        score += d_next[i, j] * nexts[b, m, j]
                logits[m] = score / norms[i] / address_temperature
                dot = 0.0
                for j in range(key_width):
                    dot += address[b, j] * keys[b, m, j]
                z = dot / temperature
                if recency != 0.0:
                    z -= recency * (slots - 1 - m)
                z += term[m]
                prior[k, m] = z
                if z > top:
                    top = z
            total = 0.0
            for m in range(slots):
                prior[k, m] -= top
                total += np.exp(prior[k, m])
            shift = np.log(total)
            for m in range(slots):
                prior[k, m] -= shift
                logits[m] += prior[k, m]
                if logits[m] > best:
                    best = logits[m]
            total = 0.0
            for m in range(slots):
                v = np.exp(logits[m] - best)
                target[k, m] = v
                total += v
            for m in range(slots):
                target[k, m] /= total
                attended = np.exp(prior[k, m]) if prior[k, m] > -np.inf else 0.0
                gradient[k, m] = target[k, m] - attended
                if gradient[k, m] != 0.0:
                    for j in range(key_width):
                        error[k, j] += gradient[k, m] * keys[b, m, j]
            total = 1.0
            for j in range(x.shape[1]):
                total += x[k, j] * x[k, j]
            for j in range(c.shape[1]):
                total += c[k, j] * c[k, j]
            energy[k] = total
            bound = min(1.0, 1.0 / (address_rate * total))
            for j in range(key_width):
                step[k, j] = error[k, j] * bound
            k += 1
        absolute = 0.0
        for k in range(moved):
            for j in range(key_width):
                absolute += abs(error[k, j])
        mean_error = absolute / max(moved * key_width, 1)
        # the address step, its candidate addresses checked with it
        a_x, a_c, a = _lms_into(w_x, w_c, w, step, address_rate, x, c)
        address_ok = _finite(a_x) and _finite(a_c) and _finite(a)
        if address_ok:
            scratch_q = np.empty(a.shape[0])
            k = 0
            for i in range(count):
                if not moving[i]:
                    continue
                _map_row(x, c, k, a_x, a_c, a, scratch_q)
                for j in range(a.shape[0]):
                    if not ((key[rows[i], j] + scratch_q[j]) - (key[rows[i], j] + scratch_q[j])
                            == 0.0):
                        address_ok = False
                k += 1
        # the position steps: the same gradient over the buckets of the table, the anchored and
        # relative maps, the slope along the ages and the grid's residues past the table
        position_ok = True
        n_p = (p_x, p_c, p)
        n_e = (e_x, e_c, e)
        n_r = (r_x, r_c, r)
        n_s = (s_x, s_c, s)
        n_g = (g_x, g_c, g)
        if has_position:
            bounds = np.empty(moved)
            for k in range(moved):
                bounds[k] = min(1.0, 1.0 / (position_rate * energy[k]))
            buckets_count = p.shape[0]
            step_p = np.zeros((moved, buckets_count))
            for k in range(moved):
                for m in range(slots):
                    step_p[k, buckets[m]] += gradient[k, m]
                for j in range(buckets_count):
                    step_p[k, j] *= bounds[k]
            n_p = _lms_into(p_x, p_c, p, step_p, position_rate, x, c)
            last = table.shape[0] - 1
            if has_anchor:
                step_e = np.zeros((moved, e.shape[0]))
                k = 0
                for i in range(count):
                    if not moving[i]:
                        continue
                    t = told[rows[i]]
                    for m in range(slots):
                        place = t - 1 - (slots - 1 - m)
                        if place < 0:
                            bucket = outside
                        else:
                            bucket = table[place] if place <= last else table[last]
                        step_e[k, bucket] += gradient[k, m]
                    for j in range(e.shape[0]):
                        step_e[k, j] *= bounds[k]
                    k += 1
                n_e = _lms_into(e_x, e_c, e, step_e, position_rate, x, c)
            if has_relative:
                step_r = np.zeros((moved, r.shape[0]))
                k = 0
                for i in range(count):
                    if not moving[i]:
                        continue
                    f = focus[rows[i]]
                    for m in range(slots):
                        if f < 0:
                            bucket = 2 * reach + 1
                        else:
                            offset = min(slots - 1 - m - f, reach)
                            bucket = max(offset, -reach) + reach
                        step_r[k, bucket] += gradient[k, m]
                    for j in range(r.shape[0]):
                        step_r[k, j] *= bounds[k]
                    k += 1
                n_r = _lms_into(r_x, r_c, r, step_r, position_rate, x, c)
            if has_slope and not (has_grid and slots > position):
                lean = np.zeros((moved, 1))
                for k in range(moved):
                    total = 0.0
                    for m in range(slots):
                        total += gradient[k, m] * ((slots - 1 - m) / slope_unit)
                    lean[k, 0] = total * min(1.0, 1.0 / (slope_rate * energy[k]))
                l_x, l_c, l_b = _lms_into(
                    s_x.reshape((s_x.shape[0], 1)), s_c.reshape((s_c.shape[0], 1)), s, lean,
                    slope_rate, x, c,
                )
                n_s = (l_x.reshape(s_x.shape[0]), l_c.reshape(s_c.shape[0]), l_b)
            if has_grid:
                width = g.shape[0]
                step_g = np.zeros((moved, width))
                for k in range(moved):
                    peak = 0.0
                    for m in range(slots):
                        attended = np.exp(prior[k, m]) if prior[k, m] > -np.inf else 0.0
                        peak = max(peak, target[k, m], attended)
                    for m in range(slots):
                        if slots - 1 - m < position:
                            continue
                        far = gradient[k, m] / peak if peak > 0 else gradient[k, m]
                        for i in range(residues.shape[0]):
                            step_g[k, residues[i, m]] += far
                    for j in range(width):
                        step_g[k, j] *= bounds[k]
                n_g = _lms_into(g_x, g_c, g, step_g, position_rate, x, c)
            position_ok = (
                _finite(n_p[0]) and _finite(n_p[1]) and _finite(n_p[2])
                and _finite(n_e[0]) and _finite(n_e[1]) and _finite(n_e[2])
                and _finite(n_r[0]) and _finite(n_r[1]) and _finite(n_r[2])
                and _finite(n_s[0]) and _finite(n_s[1]) and _finite(n_s[2])
                and _finite(n_g[0]) and _finite(n_g[1]) and _finite(n_g[2])
            )
        return (
            moved, mean_error, a_x, a_c, a, address_ok, n_p, n_e, n_r, n_s, n_g, position_ok,
        )


def address_lesson(
    recall: Any, graph: Any, rows: np.ndarray, movement: np.ndarray, positions: bool
) -> tuple[Any, ...]:
    """The kernel's address lesson of head 0 (``_lesson_kernel``) for ``recall`` at the streams
    ``rows`` with the association's ``movement``, through the synapses of ``graph``."""
    (v_syn, v_src, v_dst), (n_syn, n_src, n_dst) = recall._head_synapses[0]
    moment = recall.last
    held = recall.held
    w_x, w_c, w = recall.head_maps[0]
    return _lesson_kernel(  # type: ignore[no-any-return]
        np.asarray(rows, dtype=np.int64), np.ascontiguousarray(movement, dtype=float),
        np.ascontiguousarray(graph._weights, dtype=float), v_syn, v_src, v_dst, n_syn, n_src,
        n_dst, bool(recall.next), recall.values,
        recall.nexts if recall.next else recall.values, recall.keys, held,
        not bool(held.all()), moment.inputs, moment.cortex, moment.address, moment.key,
        float(recall.address_temperature), float(recall.temperature), float(recall.recency),
        *position_arguments(recall, 0), w_x, w_c, w, float(recall.address),
        float(recall.position_rate) if positions else 0.0, float(recall.slope),
    )


# -- the place store's read

if njit is not None:

    @_numeric
    def _store_kernel(
        cue: np.ndarray, strength: np.ndarray, mass: np.ndarray, normalize: bool,
        amplitude: float,
    ) -> np.ndarray:
        """``FastSynapses.recall`` of the delta rule: each row's cue scaled by its largest entry
        and set to unit length, read through the row's strengths, over the decayed count of
        writes with ``normalize``, at ``amplitude``."""
        rows, width = cue.shape
        post = strength.shape[2]
        out = np.zeros((rows, post))
        unit = np.empty(width)
        for b in range(rows):
            scale = 0.0
            for k in range(width):
                scale = max(scale, abs(cue[b, k]))
            if scale <= 0.0:
                scale = 1.0
            norm = 0.0
            for k in range(width):
                unit[k] = cue[b, k] / scale
                norm += unit[k] * unit[k]
            norm = np.sqrt(norm)
            if norm <= 0.0:
                norm = 1.0
            for k in range(width):
                v = unit[k] / norm
                if v != 0.0:
                    for j in range(post):
                        out[b, j] += v * strength[b, k, j]
            divisor = max(mass[b], 1e-12) if normalize else 1.0
            for j in range(post):
                out[b, j] = amplitude * (out[b, j] / divisor)
        return out


def store_read(store: Any, cue: np.ndarray) -> np.ndarray:
    """``FastSynapses.recall`` of a delta-rule store without a separator at its own batch."""
    return _store_kernel(  # type: ignore[no-any-return]
        cue, store.strength, store.mass, bool(store.normalize), float(store.amplitude)
    )
