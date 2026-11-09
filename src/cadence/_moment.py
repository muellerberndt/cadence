"""The compiled moment (``learning_compiled_moment``, opt-in): the work around the solve kernels
of a moment on the CPU in compiled passes.

A moment's settle runs in the kernels of ``cadence.fused``. Around it the recall memory reads its
slots, the brain chooses, the memory writes the moment and a lesson steps the read's maps. On the
CPU each of these is many small NumPy operations at a few rows and slots, and at that size the cost
of a call is most of the moment. Here each is one compiled pass of the same equations:

- the read (``recall_read``): the address and every position term of every row through one product
  of the inputs with the maps packed side by side (``packed_maps``, made again only where a lesson
  replaces a map), then each slot's logit, the attention and the reads of the values and nexts;
- the address lesson (``address_lesson``): the scores of the slots against the lesson's movement,
  the prior, the attention the lesson names and one least-mean-squares step of every packed column;
- the place store's read (``store_read``) and the carried gains' contrast (``route_contrast``);
- the write (``recall_write``) into the room behind the slots, with the ordinary write's bits.

The sums of the read, the lesson and the contrasts run in another order than NumPy's and BLAS's, so
those bits differ from the ordinary path, which stays the default. The kernels allocate their
outputs; a caller checks what it keeps, as the ordinary path does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fused import _compiled, _inlined, _vectorized, compiled, njit

if njit is not None:
    from .fused import _index

__all__ = [
    "address_lesson", "available", "recall_read", "recall_write", "route_contrast", "store_read",
]


# The compiled moment's shortcuts that keep its bits: a read of what the latest read read is its
# drive, and an act and a lesson call their passes without the general path's layers. False runs
# the compiled passes through the general path (the tests compare the two to the bit).
SHORTCUTS = True
# An act, and a lesson's judgment, each as one compiled call (``one_call``), with the bits of the
# shortcuts above. False runs them as the passes they are made of (the tests compare the two).
ONE_CALL = True


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
        if numbers[0] == 0:
            return 0.0
        last = table.shape[0] - 1
        row = _index(b)
        term = z[row, _index(columns[1] + table[min(age, last)])]
        if numbers[1] != 0:
            place = told - 1 - age
            bucket = numbers[7] if place < 0 else table[min(place, last)]
            term += z[row, _index(columns[2] + bucket)]
        if numbers[2] != 0:
            reach = numbers[6]
            if focus < 0:
                bucket = 2 * reach + 1
            else:
                bucket = max(min(age - focus, reach), -reach) + reach
            term += z[row, _index(columns[3] + bucket)]
        if numbers[3] != 0:
            term += z[row, _index(columns[4])] * (age / slope_unit)
        if numbers[4] != 0 and age >= numbers[5]:
            total = 0.0
            start = 0
            for i in range(grid.shape[0]):
                total += z[row, _index(columns[5] + start + age % grid[i])]
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
        values: np.ndarray, nexts: np.ndarray, has_next: bool, held: np.ndarray, first: int,
        slots: int, told: np.ndarray, focus: np.ndarray, table: np.ndarray, grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every row's key, address, attention over its slots and reads of the values and the
        nexts (``Recall.stimulate`` with one head), the maps packed (``packed_maps``): one
        product of the rows' inputs with every column, then each slot's logit. The slots are
        ``slots`` entries of the arrays from ``first`` on (the room behind a memory's slots).
        A slot's dot product (four running sums, which one sum cannot overlap) and its position
        term (``_term``'s arithmetic) are written in place: a helper's arrays cost their
        reference counts per slot."""
        rows, width = x.shape
        address_width = columns[1]
        value_width = values.shape[2]
        heard, temperature, recency, slope_unit = scalars[0], scalars[1], scalars[2], scalars[3]
        has_position, has_anchor = numbers[0] != 0, numbers[1] != 0
        has_relative, has_slope, has_grid = numbers[2] != 0, numbers[3] != 0, numbers[4] != 0
        position, reach, outside = numbers[5], numbers[6], numbers[7]
        c2, c3, c4, c5 = columns[2], columns[3], columns[4], columns[5]
        last = table.shape[0] - 1
        moduli = grid.shape[0]
        z = _products(x, c, w_x, w_c, w)
        key = np.empty((rows, width))
        q = np.empty((rows, address_width))
        attention = np.empty((rows, slots))
        read = np.zeros((rows, value_width))
        following = np.zeros((rows, value_width if has_next else 0))
        # per slot, the same in every row: its table column and its age over the slope's unit
        column_of = np.zeros(slots, dtype=np.int64)
        lean_of = np.empty(slots)
        for m in range(slots):
            age = slots - 1 - m
            if has_position:
                column_of[m] = columns[1] + table[min(age, last)]
            lean_of[m] = age / slope_unit
        quads = address_width - address_width % 4
        for b in range(rows):
            for k in range(width):
                key[b, k] = heard * x[b, k]
            for j in range(address_width):
                q[b, j] = key[b, j] + z[b, j]
            episode, focused = told[b], focus[b]
            top = -np.inf
            for m in range(slots):
                at = first + m
                if not held[b, at]:
                    attention[b, m] = -np.inf
                    continue
                age = slots - 1 - m
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                j = 0
                while j < quads:
                    s0 += q[b, j] * keys[b, at, j]
                    s1 += q[b, j + 1] * keys[b, at, j + 1]
                    s2 += q[b, j + 2] * keys[b, at, j + 2]
                    s3 += q[b, j + 3] * keys[b, at, j + 3]
                    j += 4
                while j < address_width:
                    s0 += q[b, j] * keys[b, at, j]
                    j += 1
                logit = ((s0 + s1) + (s2 + s3)) / temperature
                if recency != 0.0:
                    logit -= recency * age
                term = 0.0
                if has_position:
                    term = z[b, column_of[m]]
                    if has_anchor:
                        place = episode - 1 - age
                        if place < 0:
                            term += z[b, c2 + outside]
                        else:
                            term += z[b, c2 + table[min(place, last)]]
                    if has_relative:
                        if focused < 0:
                            term += z[b, c3 + 2 * reach + 1]
                        else:
                            term += z[b, c3 + max(min(age - focused, reach), -reach) + reach]
                    if has_slope:
                        term += z[b, c4] * lean_of[m]
                    if has_grid and age >= position:
                        total = 0.0
                        start = 0
                        for i in range(moduli):
                            total += z[b, c5 + start + age % grid[i]]
                            start += grid[i]
                        term += total
                logit += term
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
                    row, at = _index(b), _index(first + m)
                    for j in range(_index(value_width)):
                        read[row, j] += p * values[row, at, j]
                    if has_next:
                        for j in range(_index(value_width)):
                            following[row, j] += p * nexts[row, at, j]
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
    within = numbers.copy()
    within[4] = 0  # the numbers of a read whose slots do not reach the grid's columns
    recall.__dict__["_kernel_settings"] = (key, settings, within)
    return settings


def _behind(recall: Any, name: str) -> tuple[np.ndarray, int]:
    """The slot array ``name`` as the room it is a view of and its first slot there
    (``Recall._longer``), or the array itself, C-ordered, from slot 0."""
    array = getattr(recall, name)
    held = recall._rooms.get(name)
    if held is not None and held[1] is array:
        return held[0], int(held[2])
    return np.ascontiguousarray(array), 0


def recall_read(
    recall: Any, x: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(key, address, attention, read of the values, read of the nexts)`` of head 0 at the
    moments ``x`` with the cortex ``c`` of the moment before (``_packed_read``)."""
    w_x, w_c, w, columns = packed_maps(recall)
    numbers, scalars, table, grid = _settings(recall)
    slots = recall.slots
    if numbers[4] and slots <= recall.position:  # no slot reaches the grid's columns
        w_x, w_c, w = recall.__dict__["_packed"][3]
        numbers = numbers.copy()
        numbers[4] = 0
    keys, first = _behind(recall, "keys")
    values, at_values = _behind(recall, "values")
    nexts, at_nexts = _behind(recall, "nexts") if recall.next else (values, at_values)
    held, at_held = _behind(recall, "held")
    if not first == at_values == at_nexts == at_held:  # rooms of another history: every array
        keys, values, held = (np.ascontiguousarray(a) for a in (recall.keys, recall.values,
                                                                 recall.held))
        nexts = np.ascontiguousarray(recall.nexts) if recall.next else values
        first = 0
    return _packed_read(
        np.ascontiguousarray(x), np.ascontiguousarray(c), w_x, w_c, w, columns, numbers,
        scalars, keys, values, nexts, bool(recall.next), held, first, slots, recall.told,
        recall.focus, table, grid,
    )


_NO_TABLE = np.zeros(1, dtype=np.int64)
# the maps of a read in their order in the packed columns: the address map, then the position term's
PACKED = ("position_maps", "anchor_maps", "relative_maps", "slope_maps", "grid_maps")


def packed_parts(recall: Any) -> tuple[Any, ...]:
    """Head 0's address map and position maps, a tuple each (None for a map the memory does not
    hold), in their order in the packed columns."""
    position, anchor, relative = recall.position_maps, recall.anchor_maps, recall.relative_maps
    slope, grid = recall.slope_maps, recall.grid_maps
    return (
        recall.head_maps[0], position[0] if position else None, anchor[0] if anchor else None,
        relative[0] if relative else None, slope[0] if slope else None,
        grid[0] if grid else None,
    )


def packed_key(recall: Any) -> tuple[int, ...]:
    """The identity of head 0's maps: a lesson replaces a map's whole tuple."""
    return tuple(id(part) for part in packed_parts(recall))


def packed_maps(recall: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Head 0's address map and position maps as one map ``(P_x, P_c, p)`` of every column in
    turn (the address, the table, the anchored, the relative, the slope and the grid columns) and
    where each part starts (``columns``, with the end last), made again only where a map is
    replaced: a lesson replaces the maps it steps and never writes into them."""
    parts = packed_parts(recall)
    held = recall.__dict__.get("_packed")
    if held is not None:
        made = held[2]  # the maps the packed ones were made from, held, so compared by identity
        if (
            parts[0] is made[0] and parts[1] is made[1] and parts[2] is made[2]
            and parts[3] is made[3] and parts[4] is made[4] and parts[5] is made[5]
        ):
            return held[1]  # type: ignore[no-any-return]
    identity = None
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
    # the slots as arrays of one kind, whether or not they are views of the room behind them:
    # the kernel compiles once
    held = np.ascontiguousarray(recall.held)
    values = np.ascontiguousarray(recall.values)
    w_x, w_c, w, columns = packed_maps(recall)
    numbers, scalars, table, grid = _settings(recall)
    return _lesson_kernel(
        np.asarray(rows, dtype=np.int64), np.ascontiguousarray(movement, dtype=float),
        np.ascontiguousarray(graph._weights, dtype=float), v_syn, v_src, v_dst, n_syn, n_src,
        n_dst, bool(recall.next), values,
        np.ascontiguousarray(recall.nexts) if recall.next else values,
        np.ascontiguousarray(recall.keys), held,
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


# -- the moment's drive in one pass

if njit is not None:

    @_vectorized
    def _drive_kernel(
        x: np.ndarray, c: np.ndarray, w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray,
        columns: np.ndarray, numbers: np.ndarray, scalars: np.ndarray, keys: np.ndarray,
        values: np.ndarray, nexts: np.ndarray, has_next: bool, held: np.ndarray, first: int,
        slots: int, told: np.ndarray, focus: np.ndarray, table: np.ndarray, grid: np.ndarray,
        neurons: int, spans: np.ndarray, gains: np.ndarray, flags: np.ndarray,
        trace: np.ndarray, strength: np.ndarray, mass: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The drive of a moment in the order ``Memories.stimulus`` reads the memories: the
        sensory part, the working trace, the recall read (``_packed_read``) at its gain and the
        place store's read of the sensory part (``_store_kernel``). ``spans`` are the column
        ranges (sensory, trace, recall, next, ports, the store's key and value), ``gains`` the
        scales (the stimulus amplitude, the trace's, the read's, the ports' and the store's
        amplitude) and ``flags`` which parts there are (a trace of this batch, the read's drive
        gain, a store, a store read at its baseline, a normalized store). Returns the drive,
        the read's key, address and attention, and what the ports held at the read's gain."""
        rows = x.shape[0]
        drive = np.zeros((rows, neurons))
        for b in range(rows):
            for k in range(spans[1] - spans[0]):
                drive[b, spans[0] + k] = x[b, k] * gains[0]
        if flags[0]:
            for b in range(rows):
                for k in range(spans[3] - spans[2]):
                    drive[b, spans[2] + k] = gains[1] * trace[b, k]
        ports = np.zeros((rows, spans[7] - spans[6]) if flags[1] else (rows, 0))
        if flags[5]:
            key, q, attention, read, following = _packed_read(
                x, c, w_x, w_c, w, columns, numbers, scalars, keys, values, nexts, has_next,
                held, first, slots, told, focus, table, grid,
            )
            value_width = read.shape[1]
            for b in range(rows):
                for j in range(value_width):
                    drive[b, spans[4] + j] = gains[2] * read[b, j]
                if has_next:
                    for j in range(value_width):
                        drive[b, spans[5] + j] = gains[2] * following[b, j]
            if flags[1]:
                for b in range(rows):
                    for j in range(spans[7] - spans[6]):
                        ports[b, j] = drive[b, spans[6] + j]
                        drive[b, spans[6] + j] = gains[3] * ports[b, j]
        else:  # streams that begin: no slot to read, the ports held silent
            key = np.empty((0, 0))
            q = np.empty((0, 0))
            attention = np.empty((0, 0))
        if flags[2]:
            cue = np.empty((rows, spans[9] - spans[8]))
            for b in range(rows):
                for k in range(spans[9] - spans[8]):
                    cue[b, k] = drive[b, spans[8] + k]
            stored = _store_kernel(cue, strength, mass, flags[3] != 0, flags[4] != 0, gains[4])
            for b in range(rows):
                for j in range(spans[11] - spans[10]):
                    drive[b, spans[10] + j] += stored[b, j]
        return drive, key, q, attention, ports


# -- the free solve of a moment

_RULES: dict[int, tuple[Any, tuple[float, float, float, float]]] = {}


def _rule_of(model: Any) -> tuple[float, float, float, float]:
    """The neuron rule as the solve kernels read it (``fused._rule``), made once per model."""
    held = _RULES.get(id(model))
    if held is None or held[0] is not model:
        held = (model, (model.slope, model.threshold, model.rest_emission, model.leak))
        if len(_RULES) > 64:
            _RULES.clear()
        _RULES[id(model)] = held
    return held[1]


def free_solve(graph: Any, drive: np.ndarray, state: Any, budget: int, tolerance: float) -> Any:
    """``graph.equilibrate(drive, state=state, budget=budget, tolerance=tolerance, fast=True)``
    of a host state on the compiled path, in the kernels that path calls (the activations of the
    warm potentials, ``fused._fast_solve``): the same state, report and bits; None where the
    solve takes another path (a compartment, adaptation, another step, a wrapped settle)."""
    from . import fused
    from .brain import BrainState, Equilibrium, _keep_products

    model = graph.neuron_model
    if (
        model.adaptation is not None or model.dt != 1.0 or graph.backend != "cpu"
        or not graph._solves_compiled() or budget < 0 or drive.ndim != 2
        or drive.dtype != np.float64 or not drive.flags.c_contiguous
    ):
        return None
    batch, n = drive.shape
    if n != graph.connectome.n or n == 0 or not fused.rows_finite(drive):
        return None  # the general path refuses it
    if state is None:
        v, a, products = np.zeros((batch, n)), np.zeros((batch, n)), None
    else:
        v = np.array(state.v, dtype=np.float64, copy=True, ndmin=2)
        a = np.array(state.adaptation, dtype=np.float64, copy=True, ndmin=2)
        if v.shape != (batch, n) or a.shape != (batch, n) or not (
            fused.rows_finite(v) and fused.rows_finite(a)
        ):
            return None
        products = state.__dict__.get("_products")
    flat = graph._fused_blocks()[0]
    slope, threshold, rest, leak = _rule_of(model)
    s = np.empty_like(v)
    fused._publish(v, s, *fused._held(products), slope, threshold, rest, leak)
    layout = graph.layout
    out = np.empty(batch)
    used, checks, taken, row_sweeps, kept = fused._fast_solve(
        v, s, fused._standing(drive, graph.bias), flat, layout.starts, layout.pair_pre,
        layout.pair_post, layout.offset, slope, threshold, rest, leak, budget, 16,
        float(tolerance), min(float(tolerance), fused._FAST_SETTLED), out,
    )
    result = BrainState(v=v, activation=s, adaptation=a, steps=taken)
    _keep_products(result, fused._kept(flat, flat, batch, kept))
    return Equilibrium(result, out, used, tolerance, residual_checks=checks, row_sweeps=row_sweeps)


_NO_ROWS = np.zeros((1, 1))
_NO_HELD = np.zeros((1, 1), dtype=np.bool_)
_NO_STORE = np.zeros((1, 1, 1))
_NO_MASS = np.zeros(1)


def drive_spans(brain: Any) -> np.ndarray | None:
    """The column ranges of a brain's moment drive (``_drive_kernel``): the sensory region, the
    working trace's, the recall read's ports and the place store's key and value; None where
    one is not a contiguous range. Made once per brain and its memories."""
    recall, trace, store = brain.recall, brain.working_memory, brain.hippocampus
    held = brain.__dict__.get("_drive_spans")
    if (
        held is not None and held[0] is recall and held[1] is trace and held[2] is store
        and held[3] is brain.sensory_index
    ):
        return held[4]  # type: ignore[no-any-return]
    parts = (
        brain._sensory_columns, None if trace is None else trace._glow_columns,
        *recall._head_columns[0], recall._port_columns,
        None if store is None else store._pre_columns,
        None if store is None else store._post_columns,
    )
    spans: np.ndarray | None = np.zeros(12, dtype=np.int64)
    for k, part in enumerate(parts):
        if part is None:
            continue
        if not isinstance(part, slice) or spans is None:
            spans = None
            continue
        if k in (2, 3):  # the recall and next ports: their first column
            spans[4 + (k - 2)] = part.start
        else:
            at = {0: 0, 1: 2, 4: 6, 5: 8, 6: 10}[k]
            spans[at], spans[at + 1] = part.start, part.stop
    brain.__dict__["_drive_spans"] = (recall, trace, store, brain.sensory_index, spans)
    return spans


def moment_drive(brain: Any, x: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """The drive of a moment of the compiled moment in one pass (``_drive_kernel``), the recall
    memory left holding its read as ``Recall.stimulate`` leaves it: the same drive and memory to
    the bit as the read of each memory in turn."""
    from .resident import _Moment
    from .stream import _of

    recall, trace, store = brain.recall, brain.working_memory, brain.hippocampus
    rows = len(x)
    c = _of(recall.previous, rows)
    w_x, w_c, w, columns = packed_maps(recall)
    numbers, scalars, table, grid = _settings(recall)
    slots = recall.slots
    if numbers[4] and slots <= recall.position:  # no slot reaches the grid's columns
        w_x, w_c, w = recall.__dict__["_packed"][3]
        numbers = numbers.copy()
        numbers[4] = 0
    keys, first = _behind(recall, "keys")
    values, at_values = _behind(recall, "values")
    nexts, at_nexts = _behind(recall, "nexts") if recall.next else (values, at_values)
    held, at_held = _behind(recall, "held")
    if not first == at_values == at_nexts == at_held:
        keys, values, held = (np.ascontiguousarray(a) for a in (recall.keys, recall.values,
                                                                 recall.held))
        nexts = np.ascontiguousarray(recall.nexts) if recall.next else values
        first = 0
    has_trace = trace is not None and len(trace.trace) == rows
    shared = store is not None and len(store.strength) != rows
    strength = (
        _NO_STORE if store is None
        else np.ascontiguousarray(store._baseline()[None, :, :]) if shared else store.strength
    )
    gains = np.array([
        brain.brain.neuron_model.stimulus_amplitude,
        trace.amplitude * trace.hearing if has_trace else 0.0,
        1.0 if recall.drive_gain else recall.amplitude,
        recall.amplitude * recall.hearing,
        0.0 if store is None else float(store.amplitude),
    ])
    flags = np.array([
        has_trace, recall.drive_gain, store is not None, shared,
        store is not None and bool(store.normalize), True, has_trace,
    ], dtype=np.int64)
    drive, key, q, attention, ports = _drive_kernel(
        x, np.ascontiguousarray(c), w_x, w_c, w, columns, numbers, scalars, keys, values, nexts,
        bool(recall.next), held, first, slots, recall.told, recall.focus, table, grid,
        brain.connectome.n, spans, gains, flags, trace.trace if has_trace else _NO_ROWS,
        strength, _NO_MASS if store is None else store.mass,
    )
    recall.last = _Moment(x, c, q, key, x)
    attention.flags.writeable = False
    recall._attended = attention
    recall._head_reads = None
    if recall.drive_gain:
        recall.trace = ports
    return drive


# -- the act's choice and write in one pass

if njit is not None:
    from .fused import _clear_choice

    @_compiled
    def _after_kernel(
        s: np.ndarray, spans: np.ndarray, temperature: float, decay: float, trace: np.ndarray,
        has_trace: bool, write: bool, key: np.ndarray, value: np.ndarray, ended: np.ndarray,
        attended: np.ndarray, has_attended: bool, focus: np.ndarray, keys_room: np.ndarray,
        values_room: np.ndarray, nexts_room: np.ndarray, held_room: np.ndarray, first: int,
        slots: int, drop: int, stop: int,
    ) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What a greedy act does after its solve, from the settled activations ``s``: the
        choice of the motor span where every row's largest stands clear (``_clear_choice``), the
        working trace's step toward the source span (``Trace.update``), the cortex the recall
        memory carries (the association span) and, with ``write``, the moment's write
        (``_write_kernel``). ``spans`` are the motor, the source and the association ranges.
        Returns the action, whether it stands clear, the trace, the source activation, the
        cortex and the focus."""
        rows = s.shape[0]
        motor = np.empty((rows, spans[1] - spans[0]))
        for b in range(rows):
            for j in range(spans[1] - spans[0]):
                motor[b, j] = s[b, spans[0] + j]
        action, clear = _clear_choice(motor, temperature)
        source = np.empty((rows, spans[3] - spans[2]))
        for b in range(rows):
            for j in range(spans[3] - spans[2]):
                source[b, j] = s[b, spans[2] + j]
        stepped = np.empty_like(source)
        if has_trace:
            for b in range(rows):
                for j in range(source.shape[1]):
                    stepped[b, j] = decay * trace[b, j] + (1.0 - decay) * source[b, j]
        cortex = np.empty((rows, spans[5] - spans[4]))
        for b in range(rows):
            for j in range(spans[5] - spans[4]):
                cortex[b, j] = s[b, spans[4] + j]
        moved = focus.copy()
        if write:
            moved = _write_kernel(
                key, value, held_room[:, first:first + slots], ended, attended, has_attended,
                focus, keys_room, values_room, nexts_room, held_room, first, drop, stop,
            )
        return action, clear, stepped, source, cortex, moved


def _rooms(recall: Any, rows: int) -> tuple[Any, ...] | None:
    """Where the next write goes (``recall_write``): the rooms behind the slots, the first slot,
    the slots dropped and the column of the new slot; None where the ordinary write runs."""
    if rows != recall.streams:
        return None
    names = ("keys", "values", "nexts", "held")
    held_rooms = [recall._rooms.get(name) for name in names]
    for name, room in zip(names, held_rooms, strict=True):
        old = getattr(recall, name)
        if room is None or room[1] is not old or old.base is not room[0] or (
            room[3] >= room[0].shape[1]
        ):
            return None
    if len({(room[2], room[3]) for room in held_rooms if room is not None}) != 1:
        return None
    kept = recall.keys.shape[1]
    return (*held_rooms, max(kept + 1 - recall.capacity, 0))


def act_after(brain: Any, x: np.ndarray, free: Any, spans: np.ndarray) -> tuple[np.ndarray, bool]:
    """A greedy act's choice, its working trace's step and its write after the solve, in one pass
    (``_after_kernel``): the same action, trace and slots to the bit as the actor's greedy act,
    ``Trace.update`` and ``Recall.witness`` in turn. Returns the action and whether it stands
    clear (where it does not, the caller takes the softmax's argmax)."""
    recall, trace = brain.recall, brain.working_memory
    s = free.activation
    rows = len(s)
    moment = recall.last
    rooms = _rooms(recall, rows) if moment is not None and moment.event is x else None
    has_trace = trace is not None and len(trace.trace) == rows
    if trace is not None and not has_trace:
        trace.reset(rows)  # as ``Trace.update`` begins streams of another batch
        has_trace = True
    attended = recall._attended
    has_attended = (
        attended is not None and attended.shape == (rows, recall.slots) and recall.slots > 0
    )
    if rooms is None:
        write_args: tuple[Any, ...] = (
            False, _NONE2, _NONE2, recall.ended, _NONE2, False, recall.focus, _NO_STORE,
            _NO_STORE, _NO_STORE, np.zeros((1, 1), dtype=np.bool_), 0, 0, 0, 0,
        )
    else:
        keys, values, nexts, held, drop = rooms
        first, stop = keys[2], keys[3]
        write_args = (
            True, moment.key, moment.event, recall.ended,
            attended if has_attended else _NONE2, has_attended, recall.focus, keys[0],
            values[0], nexts[0], held[0], first, recall.slots, drop, stop,
        )
    action, clear, stepped, source, cortex, focus = _after_kernel(
        s, spans, float(brain.learner.config.temperature),
        0.0 if trace is None else float(trace.decay),
        trace.trace if trace is not None else _NONE2, has_trace, *write_args,
    )
    if trace is not None:
        trace.trace = stepped
        trace.last = source
        trace.cold[:] = False
    if rooms is None:
        recall.witness(x, free, None)
        return action, clear
    if recall.ended.any():
        recall.ended = np.zeros(rows, dtype=bool)
    new_first = first + drop
    for name, room in zip(("keys", "values", "nexts", "held"), rooms[:4], strict=True):
        view = room[0][:, new_first : stop + 1]
        recall._rooms[name] = (room[0], view, new_first, stop + 1)
        setattr(recall, name, view)
    recall._contents = []
    recall._content_keys = None
    recall.told = recall.told + 1
    recall.focus = focus
    recall.writes += rows
    if recall.cortex_norm:  # as ``Recall.witness``: the maps read it at unit norm
        norm = np.linalg.norm(cortex, axis=1, keepdims=True)
        cortex = np.where(norm > 0, cortex / np.where(norm > 0, norm, 1.0), cortex)
    recall.previous = cortex
    recall.last = None
    return action, clear


def after_spans(brain: Any) -> np.ndarray | None:
    """The motor, the working trace's source and the association ranges of a brain, for
    ``act_after``; None where one is not a contiguous range. Made once per brain."""
    from .stream import columns as contiguous

    trace = brain.working_memory
    output = brain.learner.output_index
    held = brain.__dict__.get("_after_spans")
    if (
        held is not None and held[0] is output and held[1] is trace
        and held[2] is brain.recall
    ):
        return held[3]  # type: ignore[no-any-return]
    parts: tuple[Any, ...] = (
        contiguous(np.asarray(output, dtype=np.int64)),
        trace._hidden_columns if trace is not None else slice(0, 0),
        brain.recall._association_columns,
    )
    spans: np.ndarray | None = None
    if all(isinstance(part, slice) for part in parts):
        spans = np.array([v for part in parts for v in (part.start, part.stop)], dtype=np.int64)
    brain.__dict__["_after_spans"] = (output, trace, brain.recall, spans)
    return spans


def written(recall: Any, rooms: tuple[Any, ...], rows: int, focus: np.ndarray,
            cortex: np.ndarray) -> None:
    """The memory after a compiled write into ``rooms`` (``_rooms``), as ``Recall.witness``
    leaves it: the slots one longer, the episode and the focus moved, the cortex carried."""
    first, stop, drop = rooms[0][2], rooms[0][3], rooms[4]
    if recall.ended.any():
        recall.ended = np.zeros(rows, dtype=bool)
    new_first = first + drop
    for name, room in zip(("keys", "values", "nexts", "held"), rooms[:4], strict=True):
        view = room[0][:, new_first : stop + 1]
        recall._rooms[name] = (room[0], view, new_first, stop + 1)
        setattr(recall, name, view)
    recall._contents = []
    recall._content_keys = None
    recall.told = recall.told + 1
    recall.focus = focus
    recall.writes += rows
    if recall.cortex_norm:  # as ``Recall.witness``: the maps read it at unit norm
        norm = np.linalg.norm(cortex, axis=1, keepdims=True)
        cortex = np.where(norm > 0, cortex / np.where(norm > 0, norm, 1.0), cortex)
    recall.previous = cortex
    recall.last = None


# -- an act, and a lesson's judgment, in one call

if njit is not None:
    from .fused import _fast_solve, _publish, _standing, rows_finite

    @_compiled
    def _moment_kernel(
        x: np.ndarray, c: np.ndarray, w_x: np.ndarray, w_c: np.ndarray, w: np.ndarray,
        columns: np.ndarray, numbers: np.ndarray, scalars: np.ndarray, keys: np.ndarray,
        values: np.ndarray, nexts: np.ndarray, has_next: bool, held: np.ndarray, first: int,
        slots: int, told: np.ndarray, focus: np.ndarray, table: np.ndarray, grid: np.ndarray,
        neurons: int, spans: np.ndarray, gains: np.ndarray, flags: np.ndarray,
        trace: np.ndarray, strength: np.ndarray, mass: np.ndarray, warm: np.ndarray,
        warm_a: np.ndarray, has_warm: bool, kept_in: np.ndarray, has_kept: bool,
        bias: np.ndarray, flat: np.ndarray, starts: np.ndarray, pair_pre: np.ndarray,
        pair_post: np.ndarray, offset: np.ndarray, rule: np.ndarray, budget: int, chunk: int,
        tolerance: float, settled: float, mode: int, after: np.ndarray, temperature: float,
        decay: float, write: bool, ended: np.ndarray, drop: int, stop: int,
    ) -> tuple[Any, ...]:
        """A moment in one call, each part the kernel the passes in turn call: the drive
        (``_drive_kernel``), the free solve from the warm state (the activations of its
        potentials, ``fused._fast_solve``) and, where every row qualifies, for an act (``mode``
        0) its choice, trace step and write (``_after_kernel``), for a lesson's judgment
        (``mode`` 1) each row's most active motor neuron. ``status`` is 0 where the moment is
        whole, 1 where the drive or the warm state is not finite and 2 where a row does not
        qualify; nothing of the memory is written then."""
        rows = x.shape[0]
        drive, key, q, attention, ports = _drive_kernel(
            x, c, w_x, w_c, w, columns, numbers, scalars, keys, values, nexts, has_next, held,
            first, slots, told, focus, table, grid, neurons, spans, gains, flags, trace,
            strength, mass,
        )
        v = np.zeros((rows, neurons))
        a = np.zeros((rows, neurons))
        s = np.empty((rows, neurons))
        out = np.empty(rows)
        used = 0
        checks = 0
        taken = 0
        row_sweeps = 0
        kept = kept_in
        status = 0
        if not rows_finite(drive):
            status = 1
        if status == 0 and has_warm:
            if rows_finite(warm) and rows_finite(warm_a):
                for b in range(rows):
                    for i in range(neurons):
                        v[b, i] = warm[b, i]
                        a[b, i] = warm_a[b, i]
            else:
                status = 1
        if status == 0:
            _publish(v, s, kept_in, has_kept, rule[0], rule[1], rule[2], rule[3])
            used, checks, taken, row_sweeps, kept = _fast_solve(
                v, s, _standing(drive, bias), flat, starts, pair_pre, pair_post, offset,
                rule[0], rule[1], rule[2], rule[3], budget, chunk, tolerance, settled, out,
            )
            for b in range(rows):
                if not out[b] <= tolerance:
                    status = 2
        action = np.zeros(rows, dtype=np.int64)
        clear = True
        stepped = np.empty((0, 0))
        source = np.empty((0, 0))
        cortex = np.empty((0, 0))
        moved = focus
        if status == 0 and mode == 0:
            action, clear, stepped, source, cortex, moved = _after_kernel(
                s, after, temperature, decay, trace, flags[6] != 0, write, key, x, ended,
                attention, slots > 0, focus, keys, values, nexts, held, first, slots, drop,
                stop,
            )
        if status == 0 and mode == 1:
            start, width = after[0], after[1] - after[0]
            for b in range(rows):
                best = s[b, start]
                at = 0
                for j in range(1, width):
                    if s[b, start + j] > best:
                        best = s[b, start + j]
                        at = j
                action[b] = at
        return (
            status, drive, key, q, attention, ports, v, a, s, out, used, checks, taken,
            row_sweeps, kept, action, clear, stepped, source, cortex, moved,
        )


_RULE_ARRAYS: dict[int, tuple[Any, np.ndarray]] = {}


def _rule_array(model: Any) -> np.ndarray:
    """``_rule_of`` as the array the one-call kernel reads, made once per model."""
    held = _RULE_ARRAYS.get(id(model))
    if held is None or held[0] is not model:
        held = (model, np.array(_rule_of(model), dtype=float))
        if len(_RULE_ARRAYS) > 64:
            _RULE_ARRAYS.clear()
        _RULE_ARRAYS[id(model)] = held
    return held[1]


def motor_spans(brain: Any) -> np.ndarray | None:
    """The motor region's range as the one-call kernel's judgment reads it (``after`` of
    ``_moment_kernel``); None where it is not one contiguous range. Made once per brain."""
    from .stream import columns as contiguous

    index = brain.motor_index
    held = brain.__dict__.get("_motor_spans")
    if held is not None and held[0] is index:
        return held[1]  # type: ignore[no-any-return]
    part: Any = contiguous(np.asarray(index, dtype=np.int64))
    spans = (
        np.array([part.start, part.stop, 0, 0, 0, 0], dtype=np.int64)
        if isinstance(part, slice) else None
    )
    brain.__dict__["_motor_spans"] = (index, spans)
    return spans


def one_call(
    brain: Any, x: np.ndarray, state: Any, mode: int, spans: np.ndarray, after: np.ndarray
) -> tuple[Any, ...] | None:
    """A moment of ``brain`` at the observations ``x`` from the warm ``state`` in one compiled
    call (``_moment_kernel``): for an act (``mode`` 0) with ``after`` from ``after_spans``, for
    a lesson's judgment (``mode`` 1) with ``after`` from ``motor_spans``. Returns the cortex of
    the moment before, the rooms a write went into (None without a compiled write), the blocks
    the solve read and the kernel's values; None where an array is not the kernel's kind (the
    caller takes the passes in turn). The caller installs what the kernel returns."""
    from . import fused
    from .stream import _of

    recall, trace, store = brain.recall, brain.working_memory, brain.hippocampus
    graph = brain.brain
    rows = len(x)
    n = graph.connectome.n
    if not (x.flags.c_contiguous and x.flags.writeable):
        return None
    if state is None:
        warm = warm_a = _NO_ROWS
        has_warm = False
        kept_in, has_kept = fused._NO_PRODUCTS
    else:
        warm, warm_a = state.v, state.adaptation
        for array in (warm, warm_a):
            if (
                type(array) is not np.ndarray or array.shape != (rows, n)
                or array.dtype != np.float64 or not array.flags.c_contiguous
                or not array.flags.writeable
            ):
                return None
        has_warm = True
        kept_in, has_kept = fused._held(state.__dict__.get("_products"))
    c = np.ascontiguousarray(_of(recall.previous, rows))
    w_x, w_c, w, columns = packed_maps(recall)
    numbers, scalars, table, grid = _settings(recall)
    slots = recall.slots
    begun = bool(slots) and recall.streams == rows
    if numbers[4] and slots <= recall.position:  # no slot reaches the grid's columns
        w_x, w_c, w = recall.__dict__["_packed"][3]
        numbers = recall.__dict__["_kernel_settings"][2]
    rooms = _rooms(recall, rows) if mode == 0 and begun else None
    told, focus, ended = recall.told, recall.focus, recall.ended
    if rooms is not None:
        keys, values, nexts, held = rooms[0][0], rooms[1][0], rooms[2][0], rooms[3][0]
        first, stop, drop = rooms[0][2], rooms[0][3], rooms[4]
    elif not begun:  # streams that begin read silence, and the general write begins them
        stop = drop = first = slots = 0
        keys = values = nexts = _NO_STORE
        held = _NO_HELD
        told = focus = np.zeros(rows, dtype=np.int64)
        ended = np.zeros(rows, dtype=np.bool_)
    else:
        stop = drop = 0
        keys, first = _behind(recall, "keys")
        values, at_values = _behind(recall, "values")
        nexts, at_nexts = _behind(recall, "nexts") if recall.next else (values, at_values)
        held, at_held = _behind(recall, "held")
        if not first == at_values == at_nexts == at_held:
            keys, values, held = (
                np.ascontiguousarray(a) for a in (recall.keys, recall.values, recall.held)
            )
            nexts = np.ascontiguousarray(recall.nexts) if recall.next else values
            first = 0
    steps = trace is not None  # a trace of another batch reads silence and begins at zero
    has_trace = steps and len(trace.trace) == rows
    traced = _NO_ROWS
    if steps:
        traced = trace.trace if has_trace else np.zeros((rows, trace.trace.shape[1]))
    shared = store is not None and len(store.strength) != rows
    if store is None:
        strength, mass = _NO_STORE, _NO_MASS
    elif shared:
        strength, mass = np.ascontiguousarray(store._baseline()[None, :, :]), store.mass
    else:
        strength, mass = store.strength, store.mass
    scales = (
        graph.neuron_model.stimulus_amplitude,
        trace.amplitude * trace.hearing if has_trace else 0.0,
        1.0 if recall.drive_gain else recall.amplitude,
        recall.amplitude * recall.hearing,
        0.0 if store is None else float(store.amplitude),
        has_trace, bool(recall.drive_gain), store is not None, shared,
        store is not None and bool(store.normalize), begun, steps,
    )
    made = brain.__dict__.get("_moment_scales")
    if made is None or made[0] != scales:
        made = (scales, np.array(scales[:5], dtype=float), np.array(scales[5:], dtype=np.int64))
        brain.__dict__["_moment_scales"] = made
    cfg = brain.learner.config
    budget = cfg.free_steps
    tolerance = float(cfg.tolerance)
    flat = graph._fused_blocks()[0]
    layout = graph.layout
    out = _moment_kernel(
        x, c, w_x, w_c, w, columns, numbers, scalars, keys, values, nexts, bool(recall.next),
        held, first, slots, told, focus, table, grid, n, spans, made[1], made[2],
        traced, strength, mass, warm, warm_a, has_warm,
        kept_in, has_kept, graph.bias, flat, layout.starts, layout.pair_pre, layout.pair_post,
        layout.offset, _rule_array(graph.neuron_model),
        budget if budget < 2 else (budget + 1) // 2, 16,  # the fast solve's chunk
        tolerance, min(tolerance, fused._FAST_SETTLED), mode, after,
        float(cfg.temperature), 0.0 if trace is None else float(trace.decay),
        rooms is not None, ended, drop, stop,
    )
    return c, rooms, flat, out, begun


def solved_state(
    flat: np.ndarray, rows: int, v: np.ndarray, a: np.ndarray, s: np.ndarray, taken: int,
    kept: np.ndarray,
) -> Any:
    """The state a one-call solve ends on, with what it keeps, as ``free_solve`` returns it."""
    from . import fused
    from .brain import BrainState, _keep_products

    result = BrainState(v=v, activation=s, adaptation=a, steps=taken)
    _keep_products(result, fused._kept(flat, flat, rows, kept))
    return result


# -- a taught lesson's phases and their checks in one call

if njit is not None:
    from .fused import _kernel, _residual_kernel

    @_compiled
    def _phases_kernel(
        warm_v: np.ndarray, warm_a: np.ndarray, kept_in: np.ndarray, has_kept: bool,
        has_given: bool, drive: np.ndarray, bias: np.ndarray, flat: np.ndarray,
        pairs: np.ndarray, apical_pairs: np.ndarray, starts: np.ndarray, pair_pre: np.ndarray,
        pair_post: np.ndarray, offset: np.ndarray, freezable: np.ndarray,
        freezable_nudged: np.ndarray, keep: np.ndarray, rule: np.ndarray, target0: np.ndarray,
        nmask0: np.ndarray, gid0: np.ndarray, softmax0: np.ndarray, weight0: np.ndarray,
        anchor0: np.ndarray, gain0: np.ndarray, target: np.ndarray, nmask: np.ndarray,
        gid: np.ndarray, ngroups: int, beta: float, softmax_t: np.ndarray, weight: np.ndarray,
        anchor: np.ndarray, anchor_gain: np.ndarray, pool_of: np.ndarray,
        pool_index: np.ndarray, pool_starts: np.ndarray, pool_run: np.ndarray,
        no_held: np.ndarray, free_steps: int, nudged_steps: int, tolerance: float,
        stops: bool, nudged_tolerance: float, nudged_stops: bool,
    ) -> tuple[Any, ...]:
        """A finite lesson's three phases and their checks, each the kernel ``Brain.settle_batch``
        and ``Brain.residual`` launch, in their order: the free phase from the warm state and its
        check, the phases nudged toward and away from the target from the free state, and their
        checks. A phase starts from a copy of its state's potentials, publishes them from what
        the state keeps and reads its products; a check reads and renews them."""
        dt, slope, threshold, rest, leak = rule[0], rule[1], rule[2], rule[3], rule[4]
        strength, tau = rule[5], rule[6]
        rows = drive.shape[0]
        v0 = warm_v.copy()
        a0 = warm_a.copy()
        s0 = np.empty_like(v0)
        _publish(v0, s0, kept_in, has_kept, slope, threshold, rest, leak)
        taken0, change0, kept0 = _kernel(
            v0, a0, s0, _standing(drive, bias), flat, pairs, flat, apical_pairs, starts,
            pair_pre, pair_post, offset, freezable, keep, False, dt, slope, threshold, rest,
            leak, False, strength, tau, False, target0, nmask0, gid0, 0, 0.0, softmax0, weight0,
            anchor0, gain0, False, pool_of, pool_index, pool_starts, pool_run, 1.0, 1.0, 0.0,
            no_held, False, free_steps, tolerance, stops, kept_in, has_given,
        )
        read = np.empty_like(v0)
        published = _publish(v0, read, kept0, True, slope, threshold, rest, leak)
        out0 = np.empty(rows)
        kept0 = _residual_kernel(
            v0, a0, read, _standing(drive, bias), flat, pairs, flat, apical_pairs, starts,
            pair_pre, pair_post, offset, keep, False, dt, slope, threshold, rest, leak, False,
            strength, False, target0, nmask0, gid0, 0, 0.0, softmax0, weight0, anchor0, gain0,
            False, pool_of, pool_index, pool_starts, pool_run, 1.0, 1.0, 0.0, no_held, False,
            out0, kept0, True, published,
        )
        v1 = v0.copy()
        a1 = a0.copy()
        s1 = np.empty_like(v1)
        _publish(v1, s1, kept0, True, slope, threshold, rest, leak)
        taken1, change1, kept1 = _kernel(
            v1, a1, s1, _standing(drive, bias), flat, pairs, flat, apical_pairs, starts,
            pair_pre, pair_post, offset, freezable_nudged, keep, False, dt, slope, threshold,
            rest, leak, False, strength, tau, True, target, nmask, gid, ngroups, beta,
            softmax_t, weight, anchor, anchor_gain, False, pool_of, pool_index, pool_starts,
            pool_run, 1.0, 1.0, 0.0, no_held, False, nudged_steps, nudged_tolerance,
            nudged_stops, kept0, True,
        )
        v2 = v0.copy()
        a2 = a0.copy()
        s2 = np.empty_like(v2)
        _publish(v2, s2, kept0, True, slope, threshold, rest, leak)
        taken2, change2, kept2 = _kernel(
            v2, a2, s2, _standing(drive, bias), flat, pairs, flat, apical_pairs, starts,
            pair_pre, pair_post, offset, freezable_nudged, keep, False, dt, slope, threshold,
            rest, leak, False, strength, tau, True, target, nmask, gid, ngroups, -beta,
            softmax_t, weight, anchor, anchor_gain, False, pool_of, pool_index, pool_starts,
            pool_run, 1.0, 1.0, 0.0, no_held, False, nudged_steps, nudged_tolerance,
            nudged_stops, kept0, True,
        )
        read = np.empty_like(v1)
        published = _publish(v1, read, kept1, True, slope, threshold, rest, leak)
        out1 = np.empty(rows)
        kept1 = _residual_kernel(
            v1, a1, read, _standing(drive, bias), flat, pairs, flat, apical_pairs, starts,
            pair_pre, pair_post, offset, keep, False, dt, slope, threshold, rest, leak, False,
            strength, True, target, nmask, gid, ngroups, beta, softmax_t, weight, anchor,
            anchor_gain, False, pool_of, pool_index, pool_starts, pool_run, 1.0, 1.0, 0.0,
            no_held, False, out1, kept1, True, published,
        )
        read = np.empty_like(v2)
        published = _publish(v2, read, kept2, True, slope, threshold, rest, leak)
        out2 = np.empty(rows)
        kept2 = _residual_kernel(
            v2, a2, read, _standing(drive, bias), flat, pairs, flat, apical_pairs, starts,
            pair_pre, pair_post, offset, keep, False, dt, slope, threshold, rest, leak, False,
            strength, True, target, nmask, gid, ngroups, -beta, softmax_t, weight, anchor,
            anchor_gain, False, pool_of, pool_index, pool_starts, pool_run, 1.0, 1.0, 0.0,
            no_held, False, out2, kept2, True, published,
        )
        return (
            v0, a0, s0, taken0, change0, kept0, out0, v1, a1, s1, taken1, change1, kept1, out1,
            v2, a2, s2, taken2, change2, kept2, out2,
        )


def lesson_phases(learner: Any, drive: Any, warm: Any, target: np.ndarray) -> Any:
    """The free phase, the two nudged phases and their checks of a finite lesson of ``learner``
    at the rows of ``drive`` from the ``warm`` state toward ``target``, in one compiled call
    (``_phases_kernel``): ``(free, nudged, opposite, checks)``, the states with what they keep
    and each phase's residual, to the bit as ``Learner.free``, ``Learner._nudged`` and
    ``Brain.residual`` make them in turn; None where a phase takes another path (a device, a
    compartment, adaptation, a recording, a state that is not finite), which the caller takes."""
    from . import fused
    from .brain import BrainState, _keep_products

    cfg, graph = learner.config, learner.brain
    model = graph.neuron_model
    if (
        graph.backend != "cpu" or graph.apical is not None or model.adaptation is not None
        or not graph._solves_alone() or cfg.free_steps < 1 or cfg.nudged_steps < 1
        or cfg.qualified or type(drive) is not np.ndarray or drive.ndim != 2
        or drive.dtype != np.float64 or not drive.flags.c_contiguous
    ):
        return None
    blocks = graph._fused_blocks()
    rows, n = drive.shape
    if blocks is None or blocks[1] is not None or n != graph.connectome.n or not rows:
        return None
    flat = blocks[0]
    if not fused.rows_finite(drive):
        return None
    if warm is None:
        warm_v = warm_a = np.zeros((rows, n))
        products = None
    else:
        if warm.device is not None:
            return None
        warm_v, warm_a = warm.v, warm.adaptation
        for array in (warm_v, warm_a):
            if (
                type(array) is not np.ndarray or array.shape != (rows, n)
                or array.dtype != np.float64 or not fused.rows_finite(array)
            ):
                return None
        warm_v, warm_a = np.ascontiguousarray(warm_v), np.ascontiguousarray(warm_a)
        products = warm.__dict__.get("_products")
    kept_in, has_kept = fused._held(products)
    has_given = fused._given(products, flat, flat, rows)[1]
    layout = graph.layout
    pull = learner.nudge_for(target, cfg.beta, None, None)
    target_k, nmask, gid, ngroups, beta, softmax_t, weight, anchor, anchor_gain = (
        fused._nudge_arrays(pull, rows, n)
    )
    target0, nmask0, gid0, _, _, softmax0, weight0, anchor0, gain0 = fused._no_nudge(rows, n)
    freezable = ~fused._fed_ranges(layout)
    freezable_nudged = freezable.copy()
    freezable_nudged[
        np.searchsorted(
            layout.starts, np.flatnonzero((nmask > 0) | (anchor_gain > 0)), side="right"
        ) - 1
    ] = False
    pairs, _, apical_pairs, _, pool_of, pool_index, pool_starts, pool_run, _, _, _, no_held, _ = (
        fused._compartment(None, layout, flat, None, rows, n)
    )
    tolerance = cfg.tolerance if cfg.tolerance else None  # a tolerance of zero never stops
    nudged_tolerance = None if cfg.nudged_fixed else tolerance
    out = _phases_kernel(
        warm_v, warm_a, kept_in, has_kept, has_given, drive, graph.bias, flat, pairs,
        apical_pairs, layout.starts, layout.pair_pre, layout.pair_post, layout.offset,
        freezable, freezable_nudged, fused._NO_MASK, np.array(fused._rule(model), dtype=float),
        target0, nmask0, gid0, softmax0, weight0, anchor0, gain0, target_k, nmask, gid,
        int(ngroups), float(beta), softmax_t, weight, anchor, anchor_gain, pool_of, pool_index,
        pool_starts, pool_run, no_held, int(cfg.free_steps), int(cfg.nudged_steps),
        float(tolerance) if tolerance is not None else 0.0, tolerance is not None,
        float(nudged_tolerance) if nudged_tolerance is not None else 0.0,
        nudged_tolerance is not None,
    )
    states, checks = [], []
    for k in range(3):
        v, a, s, taken, change, kept, residual = out[7 * k : 7 * k + 7]
        state = BrainState(
            v=v, activation=s, adaptation=a, steps=int(taken), activity_change=change,
        )
        _keep_products(state, fused._kept(flat, flat, rows, kept))
        states.append(state)
        checks.append(residual)
    return states[0], states[1], states[2], checks
