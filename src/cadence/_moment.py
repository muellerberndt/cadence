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

from .fused import _compiled, _inlined, _vectorized, compiled, njit

__all__ = ["available", "recall_read", "address_lesson"]


def available() -> bool:
    """Whether the compiled moment can run here: Numba is installed and the compiled passes of
    ``cadence.fused`` are in use."""
    return compiled()


if njit is not None:

    @_inlined
    def _term(
        z: np.ndarray, b: int, age: int, columns: np.ndarray, numbers: np.ndarray,
        slope_unit: float, told: int, focus: int, table: np.ndarray, grid: np.ndarray,
    ) -> float:
        """A slot's position term from row ``b`` of the packed products ``z``
        (``Recall.position_term``): the table over its age bucket, the anchored term over its
        place in the episode, the relative term over its offset from the previous read's focus,
        the slope along its age and past the table the grid's residues."""
        if not numbers[0]:
            return 0.0
        last = table.shape[0] - 1
        term = z[b, columns[1] + table[min(age, last)]]
        if numbers[1]:
            place = told - 1 - age
            term += z[b, columns[2] + (numbers[7] if place < 0 else table[min(place, last)])]
        if numbers[2]:
            reach = numbers[6]
            if focus < 0:
                bucket = 2 * reach + 1
            else:
                bucket = max(min(age - focus, reach), -reach) + reach
            term += z[b, columns[3] + bucket]
        if numbers[3]:
            term += z[b, columns[4]] * (age / slope_unit)
        if numbers[4] and age >= numbers[5]:
            total = 0.0
            start = 0
            for i in range(grid.shape[0]):
                total += z[b, columns[5] + start + age % grid[i]]
                start += grid[i]
            term += total
        return float(term)

    @_vectorized
    def _products(
        x: np.ndarray, c: np.ndarray, w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """``x w_x + c w_c + w``: every row's inputs through every packed column."""
        z: np.ndarray = np.dot(x, w_x) + np.dot(c, w_c)
        for b in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[b, j] += w[j]
        return z

    @_vectorized
    def _packed_read(
        x: np.ndarray, c: np.ndarray, w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray,
        columns: np.ndarray, numbers: np.ndarray, scalars: np.ndarray, keys: np.ndarray,
        values: np.ndarray, nexts: np.ndarray, has_next: bool, held: np.ndarray,
        told: np.ndarray, focus: np.ndarray, table: np.ndarray, grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every row's key, address, attention over its slots and reads of the values and the
        nexts (``Recall.stimulate`` with one head), the maps packed (``packed_maps``): one
        product of the rows' inputs with every column, then each slot's logit."""
        rows, width = x.shape
        slots = keys.shape[1]
        address_width = columns[1]
        value_width = values.shape[2]
        heard, temperature, recency, slope_unit = scalars[0], scalars[1], scalars[2], scalars[3]
        z = _products(x, c, w_x, w_c, w)
        key = np.empty((rows, width))
        q = np.empty((rows, address_width))
        attention = np.empty((rows, slots))
        read = np.zeros((rows, value_width))
        following = np.zeros((rows, value_width if has_next else 0))
        for b in range(rows):
            for k in range(width):
                key[b, k] = heard * x[b, k]
            for j in range(address_width):
                q[b, j] = key[b, j] + z[b, j]
            top = -np.inf
            for m in range(slots):
                if not held[b, m]:
                    attention[b, m] = -np.inf
                    continue
                age = slots - 1 - m
                dot = 0.0
                for j in range(address_width):
                    dot += q[b, j] * keys[b, m, j]
                logit = dot / temperature
                if recency != 0.0:
                    logit -= recency * age
                logit += _term(z, b, age, columns, numbers, slope_unit, told[b], focus[b], table,
                               grid)
                attention[b, m] = logit
                top = max(top, logit)
            if top == -np.inf:  # a stream that holds no slot reads silence
                for m in range(slots):
                    attention[b, m] = 0.0
                continue
            total = 0.0
            for m in range(slots):
                p = np.exp(attention[b, m] - top)
                attention[b, m] = p
                total += p
            for m in range(slots):
                p = attention[b, m] / total
                attention[b, m] = p
                if p != 0.0:
                    for j in range(value_width):
                        read[b, j] += p * values[b, m, j]
                    if has_next:
                        for j in range(value_width):
                            following[b, j] += p * nexts[b, m, j]
        return key, q, attention, read, following

    @_vectorized
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

    @_vectorized
    def _finite(a: np.ndarray) -> bool:
        for v in a.ravel():
            if not (v - v == 0.0):
                return False
        return True

    @_vectorized
    def _lesson_kernel(
        rows: np.ndarray, movement: np.ndarray, weights: np.ndarray,
        value_synapses: np.ndarray, value_sources: np.ndarray, value_targets: np.ndarray,
        next_synapses: np.ndarray, next_sources: np.ndarray, next_targets: np.ndarray,
        has_next: bool, values: np.ndarray, nexts: np.ndarray, keys: np.ndarray,
        held: np.ndarray, masked: bool, inputs: np.ndarray, cortex: np.ndarray,
        address: np.ndarray, key: np.ndarray, w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray,
        columns: np.ndarray, numbers: np.ndarray, scalars: np.ndarray, told: np.ndarray,
        focus: np.ndarray, table: np.ndarray, grid: np.ndarray, positions: bool,
    ) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        """The address lesson of head 0 (``Recall.learn``), the maps packed (``packed_maps``):
        the scores of each moving row's slots against the lesson's movement, the prior its
        address gave, the attention the lesson names, and one least-mean-squares step of every
        packed column: the address columns at the address rate, the table, anchored, relative
        and grid columns at the position rate and the slope column at its own. Returns the rows
        that moved, the address error, the new packed maps and whether the address step and the
        position step are finite (the candidate addresses with the address step)."""
        slots = keys.shape[1]
        key_width = keys.shape[2]
        value_width = values.shape[2]
        temperature, recency, slope_unit = scalars[1], scalars[2], scalars[3]
        address_temperature, address_rate = scalars[4], scalars[5]
        position_rate, slope_rate = scalars[6], scalars[7]
        d_value = _readback(movement, weights, value_synapses, value_sources, value_targets,
                            value_width)
        d_next = (
            _readback(movement, weights, next_synapses, next_sources, next_targets, value_width)
            if has_next else d_value
        )
        count = rows.shape[0]
        moving = np.zeros(count, dtype=np.bool_)
        norms = np.zeros(count)
        moved = 0
        for i in range(count):
            total = 0.0
            for j in range(value_width):
                total += d_value[i, j] * d_value[i, j]
            if has_next:
                for j in range(value_width):
                    total += d_next[i, j] * d_next[i, j]
            norms[i] = np.sqrt(total)
            go = norms[i] > 1e-12
            if masked and go:
                go = False
                for m in range(slots):
                    go = go or held[rows[i], m]
            moving[i] = go
            moved += go
        width = columns[6]
        if moved == 0:
            return 0, 0.0, w_x, w_c, w, False, False
        streams = np.empty(moved, dtype=np.int64)
        x = np.empty((moved, inputs.shape[1]))
        c = np.empty((moved, cortex.shape[1]))
        k = 0
        for i in range(count):
            if moving[i]:
                streams[k] = i
                x[k] = inputs[rows[i]]
                c[k] = cortex[rows[i]]
                k += 1
        z = _products(x, c, w_x, w_c, w)
        step = np.zeros((moved, width))  # every packed column's step, rate aside
        prior = np.empty(slots)
        logits = np.empty(slots)
        absolute = 0.0
        stepped_slope = numbers[3] and not (numbers[4] and slots > numbers[5])
        reach = numbers[6]
        last = table.shape[0] - 1
        for k in range(moved):
            i = streams[k]
            b = rows[i]
            top = -np.inf
            for m in range(slots):
                if masked and not held[b, m]:
                    prior[m] = -np.inf
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
                logit = dot / temperature
                if recency != 0.0:
                    logit -= recency * (slots - 1 - m)
                logit += _term(z, k, slots - 1 - m, columns, numbers, slope_unit, told[b],
                               focus[b], table, grid)
                prior[m] = logit
                top = max(top, logit)
            total = 0.0
            for m in range(slots):
                prior[m] -= top
                total += np.exp(prior[m])
            shift = np.log(total)
            best = -np.inf
            for m in range(slots):
                prior[m] -= shift
                logits[m] += prior[m]
                best = max(best, logits[m])
            total = 0.0
            for m in range(slots):
                logits[m] = np.exp(logits[m] - best)
                total += logits[m]
            energy = 1.0
            for j in range(x.shape[1]):
                energy += x[k, j] * x[k, j]
            for j in range(c.shape[1]):
                energy += c[k, j] * c[k, j]
            bound = min(1.0, 1.0 / (address_rate * energy))
            near = min(1.0, 1.0 / (position_rate * energy))
            peak = 0.0
            for m in range(slots):
                logits[m] /= total  # the attention the lesson names
                peak = max(peak, logits[m], np.exp(prior[m]) if prior[m] > -np.inf else 0.0)
            lean = 0.0
            for m in range(slots):
                attended = np.exp(prior[m]) if prior[m] > -np.inf else 0.0
                gradient = logits[m] - attended
                if gradient == 0.0:
                    continue
                for j in range(key_width):
                    step[k, j] += gradient * keys[b, m, j]  # the error, bounded below
                if not numbers[0]:
                    continue
                age = slots - 1 - m
                step[k, columns[1] + table[min(age, last)]] += gradient * near
                if numbers[1]:
                    place = told[b] - 1 - age
                    bucket = numbers[7] if place < 0 else table[min(place, last)]
                    step[k, columns[2] + bucket] += gradient * near
                if numbers[2]:
                    if focus[b] < 0:
                        bucket = 2 * reach + 1
                    else:
                        bucket = max(min(age - focus[b], reach), -reach) + reach
                    step[k, columns[3] + bucket] += gradient * near
                lean += gradient * (age / slope_unit)
                if numbers[4] and age >= numbers[5]:
                    far = gradient / peak if peak > 0 else gradient
                    start = 0
                    for i2 in range(grid.shape[0]):
                        step[k, columns[5] + start + age % grid[i2]] += far * near
                        start += grid[i2]
            for j in range(key_width):
                absolute += abs(step[k, j])
                step[k, j] *= bound
            if stepped_slope:
                step[k, columns[4]] = lean * min(1.0, 1.0 / (slope_rate * energy))
        mean_error = absolute / (moved * key_width)
        # one least-mean-squares step of every column, each at its part's rate, averaged
        s_x = np.dot(x.T.copy(), step)
        s_c = np.dot(c.T.copy(), step)
        rates = np.empty(width)
        for j in range(width):
            rates[j] = position_rate if positions else 0.0
        for j in range(columns[1]):
            rates[j] = address_rate
        if numbers[3]:
            rates[columns[4]] = slope_rate if (positions and stepped_slope) else 0.0
        new_x = np.empty_like(w_x)
        new_c = np.empty_like(w_c)
        new = np.empty_like(w)
        for j in range(width):
            mean = 0.0
            for k in range(moved):
                mean += step[k, j]
            new[j] = w[j] + rates[j] * (mean / moved)
        for a in range(w_x.shape[0]):
            for j in range(width):
                new_x[a, j] = w_x[a, j] + rates[j] * s_x[a, j] / moved
        for a in range(w_c.shape[0]):
            for j in range(width):
                new_c[a, j] = w_c[a, j] + rates[j] * s_c[a, j] / moved
        address_ok = (
            _finite(new_x[:, : columns[1]]) and _finite(new_c[:, : columns[1]])
            and _finite(new[: columns[1]])
        )
        if address_ok:  # the candidate addresses of the moving rows
            candidate = _products(x, c, new_x[:, : columns[1]].copy(),
                                  new_c[:, : columns[1]].copy(), new[: columns[1]].copy())
            for k in range(moved):
                b = rows[streams[k]]
                for j in range(columns[1]):
                    v = key[b, j] + candidate[k, j]
                    if not (v - v == 0.0):
                        address_ok = False
        position_ok = (
            _finite(new_x[:, columns[1]:]) and _finite(new_c[:, columns[1]:])
            and _finite(new[columns[1]:])
        )
        return moved, mean_error, new_x, new_c, new, address_ok, position_ok


_NONE2 = np.zeros((1, 1))


def _settings(recall: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The read's settings as the kernels take them: which position terms it has and their
    sizes, its scalars, the age buckets' table and the grid's moduli; made again where one
    changes."""
    key = (
        bool(recall.position_maps), bool(recall.anchor_maps), bool(recall.relative_maps),
        bool(recall.slope_maps), bool(recall.grid_maps), recall.position, recall.capacity,
        recall.heard, recall.temperature, recall.recency, recall.address_temperature,
        recall.address, recall.position_rate, recall.slope, tuple(recall.grid or ()),
    )
    held = recall.__dict__.get("_kernel_settings")
    if held is not None and held[0] == key:
        return held[1]  # type: ignore[no-any-return]
    numbers = np.array([
        *key[:5], int(recall.position), int(recall.RELATIVE), int(recall.position_buckets),
    ], dtype=np.int64)
    scalars = np.array([
        float(recall.heard), float(recall.temperature), float(recall.recency), float(recall.SLOPE),
        float(recall.address_temperature), float(recall.address), float(recall.position_rate),
        float(recall.slope),
    ])
    table = recall._age_buckets() if recall.position_maps else _NO_TABLE
    grid = np.asarray(recall.grid if recall.grid_maps else (1,), dtype=np.int64)
    settings = (numbers, scalars, np.ascontiguousarray(table, dtype=np.int64), grid)
    recall.__dict__["_kernel_settings"] = (key, settings)
    return settings


def recall_read(
    recall: Any, x: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(key, address, attention, read of the values, read of the nexts)`` of head 0 at the
    moments ``x`` with the cortex ``c`` of the moment before (``_packed_read``)."""
    w_x, w_c, w, columns = packed_maps(recall)
    numbers, scalars, table, grid = _settings(recall)
    if numbers[4] and recall.slots <= recall.position:  # no slot reaches the grid's columns
        w_x, w_c, w = recall.__dict__["_packed"][3]
        numbers = numbers.copy()
        numbers[4] = 0
    return _packed_read(
        np.ascontiguousarray(x), np.ascontiguousarray(c), w_x, w_c, w, columns, numbers,
        scalars, recall.keys, recall.values, recall.nexts if recall.next else recall.values,
        bool(recall.next), recall.held, recall.told, recall.focus, table, grid,
    )


_NO_TABLE = np.zeros(1, dtype=np.int64)
# the maps of a read in their order in the packed columns: the address map, then the position term's
PACKED = ("position_maps", "anchor_maps", "relative_maps", "slope_maps", "grid_maps")


def packed_maps(recall: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Head 0's address map and position maps as one map ``(P_x, P_c, p)`` of every column in
    turn (the address, the table, the anchored, the relative, the slope and the grid columns) and
    where each part starts (``columns``, with the end last), made again only where a map is
    replaced: a lesson replaces the maps it steps and never writes into them."""
    parts = [recall.head_maps[0]] + [
        getattr(recall, name)[0] if getattr(recall, name) else None for name in PACKED
    ]
    identity = tuple(id(a) for part in parts if part is not None for a in part)
    held = recall.__dict__.get("_packed")
    if held is not None and held[0] == identity:
        return held[1]  # type: ignore[no-any-return]
    blocks_x, blocks_c, blocks, columns = [], [], [], [0]
    for part in parts:
        if part is None:
            columns.append(columns[-1])
            continue
        m_x, m_c, m = part
        m_x, m_c = np.asarray(m_x, dtype=float), np.asarray(m_c, dtype=float)
        if m_x.ndim == 1:  # the slope maps: one column
            m_x, m_c = m_x[:, None], m_c[:, None]
        blocks_x.append(m_x)
        blocks_c.append(m_c)
        blocks.append(np.atleast_1d(np.asarray(m, dtype=float)))
        columns.append(columns[-1] + m_x.shape[1])
    packed = (
        np.ascontiguousarray(np.concatenate(blocks_x, axis=1)),
        np.ascontiguousarray(np.concatenate(blocks_c, axis=1)),
        np.concatenate(blocks), np.asarray(columns, dtype=np.int64),
    )
    end = columns[-2]  # without the grid's columns, which only slots past the table read
    within = (
        np.ascontiguousarray(packed[0][:, :end]), np.ascontiguousarray(packed[1][:, :end]),
        packed[2][:end].copy(),
    )
    # the arrays are held with their identities, so an identity is never a freed array's
    recall.__dict__["_packed"] = (identity, packed, parts, within)
    return packed


# -- the address lesson


def address_lesson(
    recall: Any, graph: Any, rows: np.ndarray, movement: np.ndarray, positions: bool
) -> tuple[Any, ...]:
    """The kernel's address lesson of head 0 (``_lesson_kernel``) for ``recall`` at the streams
    ``rows`` with the association's ``movement``, through the synapses of ``graph``."""
    (v_syn, v_src, v_dst), (n_syn, n_src, n_dst) = recall._head_synapses[0]
    moment = recall.last
    held = recall.held
    w_x, w_c, w, columns = packed_maps(recall)
    numbers, scalars, table, grid = _settings(recall)
    return _lesson_kernel(
        np.asarray(rows, dtype=np.int64), np.ascontiguousarray(movement, dtype=float),
        np.ascontiguousarray(graph._weights, dtype=float), v_syn, v_src, v_dst, n_syn, n_src,
        n_dst, bool(recall.next), recall.values,
        recall.nexts if recall.next else recall.values, recall.keys, held,
        not bool(held.all()), moment.inputs, moment.cortex, moment.address, moment.key,
        w_x, w_c, w, columns, numbers, scalars, recall.told, recall.focus, table, grid,
        bool(positions),
    )


# -- the place store's read

if njit is not None:

    @_vectorized
    def _store_kernel(
        cue: np.ndarray, strength: np.ndarray, mass: np.ndarray, shared: bool, normalize: bool,
        amplitude: float,
    ) -> np.ndarray:
        """``FastSynapses.recall`` of the delta rule: each row's cue scaled by its largest entry
        and set to unit length, read through the row's strengths (``shared``: one baseline for
        every row, with no writes), over the decayed count of writes with ``normalize``, at
        ``amplitude``."""
        rows, width = cue.shape
        post = strength.shape[2]
        out = np.zeros((rows, post))
        unit = np.empty(width)
        for b in range(rows):
            held = 0 if shared else b
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
                        out[b, j] += v * strength[held, k, j]
            divisor = max(0.0 if shared else mass[b], 1e-12) if normalize else 1.0
            for j in range(post):
                out[b, j] = amplitude * (out[b, j] / divisor)
        return out


def store_read(store: Any, cue: np.ndarray) -> np.ndarray:
    """``FastSynapses.recall`` of a delta-rule store without a separator: at its own batch the
    streams' strengths, at another the baseline every new stream reads (``_baseline``)."""
    shared = len(store.strength) != len(cue)
    strength = store._baseline()[None, :, :] if shared else store.strength
    return _store_kernel(
        cue, np.ascontiguousarray(strength, dtype=float), store.mass, shared,
        bool(store.normalize), float(store.amplitude),
    )


# -- the carried routes' gains

if njit is not None:

    @_vectorized
    def _route_kernel(
        movement: np.ndarray, weights: np.ndarray, synapses: np.ndarray, sources: np.ndarray,
        targets: np.ndarray, held: np.ndarray, scale: float, amplitude: float, slope: float,
        threshold: float, rest: float, leak: float,
    ) -> np.ndarray:
        """One route's ``c`` per row (``Hearing._contrast``): the movement read back through the
        route's synapses, weighed by the slope of the activation at the held potential
        ``scale * held`` times ``amplitude * held``."""
        rows, width = held.shape
        out = np.zeros(rows)
        back = np.empty(width)
        for b in range(rows):
            for j in range(width):
                back[j] = 0.0
            for e in range(synapses.shape[0]):
                back[sources[e]] += weights[synapses[e]] * movement[b, targets[e]]
            total = 0.0
            for j in range(width):
                v = scale * held[b, j]
                sig = 1.0 / (1.0 + np.exp(-slope * (v - threshold)))
                d = slope * sig * (1.0 - sig)
                d = d / (1.0 - rest) if sig > rest else leak * d / rest
                total += d * amplitude * held[b, j] * back[j]
            out[b] = total
        return out


def route_contrast(
    graph: Any, route: dict[str, Any], movement: np.ndarray, held: np.ndarray, scale: float,
    amplitude: float, model: Any,
) -> np.ndarray:
    """``_route_kernel`` for one route of ``Hearing`` through the synapses of ``graph``."""
    return _route_kernel(
        np.ascontiguousarray(movement, dtype=float),
        np.ascontiguousarray(graph._weights, dtype=float), route["synapses"], route["rows"],
        route["cols"], np.ascontiguousarray(held, dtype=float), float(scale), float(amplitude),
        float(model.slope), float(model.threshold), float(model.rest_emission),
        float(model.leak),
    )


# -- the recall memory's write

if njit is not None:

    @_compiled
    def _write_kernel(
        key: np.ndarray, value: np.ndarray, held: np.ndarray, ended: np.ndarray,
        attended: np.ndarray, has_attended: bool, focus: np.ndarray,
        keys_room: np.ndarray, values_room: np.ndarray, nexts_room: np.ndarray,
        held_room: np.ndarray, first: int, drop: int, stop: int,
    ) -> np.ndarray:
        """One moment per stream written into the rooms behind the slots (``Recall._observe``):
        each stream's latest held slot that closed no episode learns ``value`` as its next where
        it survives the drop, the new slot at column ``stop`` holds ``key`` and ``value`` and no
        next, and the focus moves to the slot this moment's read attended most. Returns the
        new focus."""
        rows, slots = held.shape
        out = np.empty(rows, dtype=np.int64)
        for b in range(rows):
            for j in range(nexts_room.shape[2]):
                nexts_room[b, stop, j] = 0.0
            if not ended[b]:
                latest = -1
                for m in range(slots):
                    if held[b, m]:
                        latest = m
                if latest >= drop:
                    for j in range(value.shape[1]):
                        nexts_room[b, first + latest, j] = value[b, j]
            for j in range(key.shape[1]):
                keys_room[b, stop, j] = key[b, j]
            for j in range(value.shape[1]):
                values_room[b, stop, j] = value[b, j]
            held_room[b, stop] = True
            if has_attended:
                best = attended[b, 0]
                at = 0
                for m in range(1, slots):
                    if attended[b, m] > best:
                        best = attended[b, m]
                        at = m
                out[b] = slots - 1 - at + 1 if best > 0 else -1
            else:
                out[b] = focus[b] + 1 if focus[b] >= 0 else -1
        return out


def recall_write(recall: Any, key: np.ndarray, value: np.ndarray) -> bool:
    """``Recall._observe`` at the bank's own batch where every slot array has room behind it:
    the write in one compiled pass, the same bits. False where it does not apply (the ordinary
    write then runs)."""
    if len(key) != recall.streams:
        return False
    rooms = recall._rooms
    names = ("keys", "values", "nexts", "held")
    held_rooms = [rooms.get(name) for name in names]
    for name, room in zip(names, held_rooms, strict=True):
        old = getattr(recall, name)
        if room is None or room[1] is not old or old.base is not room[0] or (
            room[3] >= room[0].shape[1]
        ):
            return False
    if len({(room[2], room[3]) for room in held_rooms if room is not None}) != 1:
        return False
    kept = recall.keys.shape[1]
    drop = max(kept + 1 - recall.capacity, 0)
    first, stop = held_rooms[0][2], held_rooms[0][3]
    attended = recall._attended
    has_attended = (
        attended is not None and attended.shape == (len(key), kept) and kept > 0
    )
    focus = _write_kernel(
        key, value, recall.held, recall.ended, attended if has_attended else _NONE2,
        has_attended, recall.focus, held_rooms[0][0], held_rooms[1][0],
        held_rooms[2][0], held_rooms[3][0], first, drop, stop,
    )
    if recall.ended.any():
        recall.ended = np.zeros(len(key), dtype=bool)
    new_first = first + drop
    for name, room in zip(names, held_rooms, strict=True):
        assert room is not None
        view = room[0][:, new_first : stop + 1]
        rooms[name] = (room[0], view, new_first, stop + 1)
        setattr(recall, name, view)
    recall._contents = []
    recall._content_keys = None
    recall.told = recall.told + 1
    recall.focus = focus
    return True
