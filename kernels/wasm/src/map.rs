//! The map of things' settle, `_map_settle` of `cadence/compiled.py`: `Map._settle`'s
//! drive, mask and start from the map's carried state, then the sweeps of `_settle_rows`, in
//! one call and in the order the Python makes them.

use crate::settle::{act_scalar, settle, Net, RECORD};
use crate::{finite, refused, room, view, Scratch, View, BOOL, FLOAT64, INT64, NO_ROOM};

macro_rules! take {
    ($id:expr, $t:ty, $num:expr, $k:expr) => {
        match view::<$t>($id, $num) {
            Some(x) => x,
            None => return refused($k),
        }
    };
}

/// A drive or a start that is not finite: `Brain.settle_batch` refuses it.
pub(crate) const NOT_FINITE: i32 = -2000;

/// `Map._settle` without a trace. `runs` (int64, 16) holds the start and stop of `reference`,
/// `held`, `map`, `where`, `what`, `clock`, `lost` and `doubt`; `out` (1, n) receives every
/// neuron's settled potential. Returns the sweeps taken.
#[no_mangle]
pub unsafe extern "C" fn map_settle(
    flat: usize, pairs: usize, flat_apical: usize, apical_pairs: usize, starts: usize, pair_pre: usize,
    pair_post: usize, offset: usize, freezable: usize, bias: usize, dt: f64, slope: f64,
    threshold: f64, rest: f64, leak: f64, has_apical: i32, pool_of: usize, pool_index: usize,
    pool_starts: usize, pool_run: usize, fraction: f64, floor2: f64, tiny: f64, potential: usize,
    carried: usize, has_carried: i32, write: usize, has_write: i32, clock: usize, lost: usize,
    doubt: usize, fresh: usize, has_fresh: i32, kept: usize, has_kept: i32, sure: usize,
    has_sure: i32, cue: usize, has_cue: i32, runs: usize, sweeps: i32, out: usize,
) -> i32 {
    let graph = MapGraph {
        flat: take!(flat, f64, FLOAT64, 0),
        pairs: take!(pairs, i64, INT64, 1),
        flat_apical: take!(flat_apical, f64, FLOAT64, 2),
        apical_pairs: take!(apical_pairs, i64, INT64, 3),
        starts: take!(starts, i64, INT64, 4),
        pair_pre: take!(pair_pre, i64, INT64, 5),
        pair_post: take!(pair_post, i64, INT64, 6),
        offset: take!(offset, i64, INT64, 7),
        freezable: take!(freezable, u8, BOOL, 8),
        bias: take!(bias, f64, FLOAT64, 9),
        dt, slope, threshold, rest, leak, has_apical,
        pool_of: take!(pool_of, i64, INT64, 16),
        pool_index: take!(pool_index, i64, INT64, 17),
        pool_starts: take!(pool_starts, i64, INT64, 18),
        pool_run: take!(pool_run, i64, INT64, 19),
        fraction, floor2, tiny,
    };
    let potential = take!(potential, f64, FLOAT64, 23);
    let carried = take!(carried, u8, BOOL, 24);
    let write = take!(write, f64, FLOAT64, 26);
    let clock = take!(clock, f64, FLOAT64, 28);
    let lost = take!(lost, f64, FLOAT64, 29);
    let doubt = take!(doubt, f64, FLOAT64, 30);
    let fresh = take!(fresh, u8, BOOL, 31);
    let kept = take!(kept, u8, BOOL, 33);
    let sure = take!(sure, u8, BOOL, 35);
    let cue = take!(cue, f64, FLOAT64, 37);
    let runs = take!(runs, i64, INT64, 39);
    let out = take!(out, f64, FLOAT64, 41);
    settle_map(&graph, potential, carried, has_carried, write, has_write, clock, lost, doubt, fresh,
               has_fresh, kept, has_kept, sure, has_sure, cue, has_cue, runs, sweeps, out)
}

/// The arguments of `map_settle` a map's graph fixes (`compiled._graph_arrays`).
#[derive(Clone, Copy)]
pub(crate) struct MapGraph {
    pub flat: View<f64>,
    pub pairs: View<i64>,
    pub flat_apical: View<f64>,
    pub apical_pairs: View<i64>,
    pub starts: View<i64>,
    pub pair_pre: View<i64>,
    pub pair_post: View<i64>,
    pub offset: View<i64>,
    pub freezable: View<u8>,
    pub bias: View<f64>,
    pub dt: f64,
    pub slope: f64,
    pub threshold: f64,
    pub rest: f64,
    pub leak: f64,
    pub has_apical: i32,
    pub pool_of: View<i64>,
    pub pool_index: View<i64>,
    pub pool_starts: View<i64>,
    pub pool_run: View<i64>,
    pub fraction: f64,
    pub floor2: f64,
    pub tiny: f64,
}

/// `map_settle` on views: the map's graph, its carried state and the moment's arrays.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn settle_map(
    graph: &MapGraph, potential: View<f64>, carried: View<u8>, has_carried: i32, write: View<f64>,
    has_write: i32, clock: View<f64>, lost: View<f64>, doubt: View<f64>, fresh: View<u8>,
    has_fresh: i32, kept: View<u8>, has_kept: i32, sure: View<u8>, has_sure: i32, cue: View<f64>,
    has_cue: i32, runs: View<i64>, sweeps: i32, out: View<f64>,
) -> i32 {
    let MapGraph {
        flat, pairs, flat_apical, apical_pairs, starts, pair_pre, pair_post, offset, freezable,
        bias, dt, slope, threshold, rest, leak, has_apical, pool_of, pool_index, pool_starts,
        pool_run, fraction, floor2, tiny,
    } = *graph;
    let ranges = if starts.len > 0 { starts.len - 1 } else { 0 };
    if ranges == 0 || runs.len != 16 {
        return refused(39);
    }
    let n = starts.get(ranges) as usize;
    let run = |k: usize| (runs.get(2 * k) as usize, runs.get(2 * k + 1) as usize);
    let (reference, held_run, sheet, _, _, clock_run, lost_run, doubt_run) =
        (run(0), run(1), run(2), run(3), run(4), run(5), run(6), run(7));
    let cells = potential.n0;
    let width = potential.n1;
    let size = cells * width;
    if out.len != n || bias.len != n || freezable.len != ranges || sheet.1 - sheet.0 != size
        || held_run.1 - held_run.0 != size || reference.1 - reference.0 != width
        || clock_run.1 - clock_run.0 != cells || doubt_run.1 - doubt_run.0 != cells
        || lost_run.1 - lost_run.0 != lost.len || clock.len != cells || doubt.len != cells
        || (has_carried != 0 && carried.len != size) || (has_write != 0 && write.len != size)
        || (has_fresh != 0 && fresh.len != cells) || (has_kept != 0 && kept.len != cells)
        || (has_sure != 0 && sure.len != cells) || (has_cue != 0 && cue.len != width)
        || [reference, held_run, sheet, clock_run, lost_run, doubt_run].iter().any(|r| r.1 > n)
    {
        return refused(41);
    }
    for q in 0..pair_pre.len.max(pairs.len) {
        if q < pairs.len && pairs.get(q) as usize >= pair_pre.len {
            return refused(1);
        }
    }
    let mut kept_len = 3 * n + ranges + RECORD;
    for list in [pairs, apical_pairs] {
        for q in 0..list.len {
            let b = pair_post.get(list.get(q) as usize) as usize;
            kept_len += (starts.get(b + 1) - starts.get(b)) as usize;
        }
    }
    let mut scratch = match Scratch::new(
        room::<f64>(n) * 9 + room::<i64>(n) + room::<f64>(kept_len) + 64,
    ) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    let drive = scratch.take::<f64>(n);
    let mask = scratch.take::<f64>(n);
    let a = scratch.take::<f64>(n);
    let s = scratch.take::<f64>(n);
    let standing = scratch.take::<f64>(n);
    let nmask = scratch.take::<f64>(n);
    let softmax_t = scratch.take::<f64>(n);
    let anchor_gain = scratch.take::<f64>(n);
    let gid = scratch.take::<i64>(n);
    let small = scratch.take::<f64>(8); // target, anchor, held, weight, change, given
    let kept_room = scratch.take::<f64>(kept_len);
    for i in 0..n {
        *mask.add(i) = 1.0;
        *gid.add(i) = -1;
    }
    *small.add(3) = 1.0; // the one row's weight
    // the drive and the mask, as the Python makes them
    let held = |i: usize| -> f64 {
        let p = potential.get(i);
        if has_carried != 0 {
            p * if carried.get(i) != 0 { 1.0 } else { 0.0 }
        } else {
            p
        }
    };
    if has_cue != 0 {
        for k in 0..width {
            *drive.add(reference.0 + k) = cue.get(k);
        }
        for i in 0..size {
            *drive.add(held_run.0 + i) = held(i);
        }
    } else {
        for i in 0..size {
            *drive.add(sheet.0 + i) = if has_write != 0 { held(i) + write.get(i) } else { held(i) };
        }
        if has_carried != 0 {
            for i in 0..size {
                let alive = if has_write != 0 {
                    carried.get(i) != 0 || write.get(i) != 0.0
                } else {
                    carried.get(i) != 0
                };
                *mask.add(sheet.0 + i) = if alive { 1.0 } else { 0.0 };
            }
        }
    }
    if has_kept != 0 {
        for c in 0..cells {
            let k = if kept.get(c) != 0 { 1.0 } else { 0.0 };
            for u in 0..width {
                *mask.add(sheet.0 + c * width + u) *= k;
            }
        }
    }
    for c in 0..cells {
        *drive.add(clock_run.0 + c) = clock.get(c);
        *drive.add(doubt_run.0 + c) = doubt.get(c);
    }
    for u in 0..lost.len {
        *drive.add(lost_run.0 + u) = lost.get(u);
    }
    if has_fresh != 0 {
        for c in 0..cells {
            *mask.add(clock_run.0 + c) = if fresh.get(c) != 0 { 0.0 } else { 1.0 };
        }
    }
    if has_sure != 0 {
        for c in 0..cells {
            *mask.add(doubt_run.0 + c) = if sure.get(c) != 0 { 0.0 } else { 1.0 };
        }
    }
    // the start: the drive, a moment on for the clock, the sheet at rest under a cue, masked
    let mut all_finite = true;
    let mut any = false;
    let mut masked = false;
    for i in 0..n {
        let mut x = *drive.add(i);
        all_finite &= finite(x);
        if i >= clock_run.0 && i < clock_run.1 {
            x += 1.0;
        }
        if has_cue != 0 && i >= sheet.0 && i < sheet.1 {
            x = 0.0;
        }
        x *= *mask.add(i);
        out.set(i, x);
        all_finite &= finite(x);
        any |= x != 0.0;
        masked |= *mask.add(i) != 1.0;
    }
    if !all_finite {
        return NOT_FINITE;
    }
    // the activations it publishes (`_publish` with nothing kept), under the mask
    if !any {
        let at_rest = act_scalar(0.0, slope, threshold, rest, leak);
        for i in 0..n {
            *s.add(i) = at_rest;
        }
    } else {
        for i in 0..n {
            *s.add(i) = act_scalar(out.get(i), slope, threshold, rest, leak);
        }
    }
    if masked {
        for i in 0..n {
            *s.add(i) *= *mask.add(i);
        }
    }
    for i in 0..n {
        *standing.add(i) = *drive.add(i) + bias.get(i);
    }
    let row = |p: *mut f64| View { p, n0: 1, n1: n, len: n };
    let one = |p: *mut f64| View { p, n0: 1, n1: 1, len: 1 };
    let net = Net {
        batch: 1, n, ranges, v: row(out.p), a: row(a), s: row(s), standing: row(standing),
        flat, pairs, flat_apical, apical_pairs, starts, pair_pre, pair_post, offset,
        keep: row(mask), masked, dt, slope, threshold, rest, leak, has_adapt: false,
        strength: 0.0, tau: 1.0, has_nudge: false, target: one(small),
        nmask: View { p: nmask, n0: n, n1: 1, len: n },
        gid: View { p: gid, n0: n, n1: 1, len: n }, ngroups: 0, beta: 0.0,
        softmax_t: View { p: softmax_t, n0: n, n1: 1, len: n },
        weight: View { p: small.add(3), n0: 1, n1: 1, len: 1 }, anchor: one(small.add(1)),
        anchor_gain: View { p: anchor_gain, n0: n, n1: 1, len: n }, has_apical: has_apical != 0,
        pool_of, pool_index, pool_starts, pool_run, fraction, floor2, tiny,
        held: one(small.add(2)), has_held: false, per_life: false,
    };
    if net.offset.len < net.pair_pre.len + 1 {
        return refused(7);
    }
    let change = View { p: small.add(4), n0: 1, n1: 1, len: 1 };
    let kept_view = View { p: kept_room, n0: kept_len, n1: 1, len: kept_len };
    let given = View { p: small.add(5), n0: 1, n1: 1, len: 1 };
    settle(&net, freezable, sweeps.max(0) as usize, 0.0, false, given, false, change, kept_view)
}

// -- the map's pieces and moves (`Map._pieces`, `Map.moves`)

const EMPTY: i64 = -1;

/// The slot each key takes in CPython's set built from the ascending `keys` (a set of small
/// non-negative ints, whose hash is the int): `table` (room for `16 * keys + 32` slots: a grown
/// table holds at most eight times its keys, and the old one is copied past it) receives
/// the keys at their slots, EMPTY elsewhere; returns the table's size. CPython's `set_add_entry`
/// with its linear probes and perturbation, and `set_table_resize` with `set_insert_clean`.
unsafe fn set_table(keys: *const i64, n: usize, table: *mut i64) -> usize {
    const PROBES: usize = 9;
    let mut size = 8usize;
    for s in 0..size {
        *table.add(s) = EMPTY;
    }
    let mut fill = 0usize;
    let mut used = 0usize;
    for q in 0..n {
        let key = *keys.add(q);
        let mask = size - 1;
        let mut i = key as usize & mask;
        let mut perturb = key as usize;
        'place: loop {
            let probes = if i + PROBES <= mask { PROBES } else { 0 };
            let mut e = i;
            for _ in 0..=probes {
                if *table.add(e) == EMPTY {
                    *table.add(e) = key;
                    fill += 1;
                    used += 1;
                    break 'place;
                }
                e += 1;
            }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
        if fill * 5 >= mask * 3 {
            // the table grows to the least power of two above four times what it holds; its
            // keys go over in the old table's order
            let mut grown = 8usize;
            while grown <= used * 4 {
                grown <<= 1;
            }
            let old = table.add(grown); // the old keys, copied past the new table's room
            for s in 0..size {
                *old.add(s) = *table.add(s);
            }
            for s in 0..grown {
                *table.add(s) = EMPTY;
            }
            let newmask = grown - 1;
            for s in 0..size {
                let k = *old.add(s);
                if k == EMPTY {
                    continue;
                }
                let mut i = k as usize & newmask;
                let mut perturb = k as usize;
                'clean: loop {
                    if *table.add(i) == EMPTY {
                        *table.add(i) = k;
                        break 'clean;
                    }
                    if i + PROBES <= newmask {
                        for j in 1..=PROBES {
                            if *table.add(i + j) == EMPTY {
                                *table.add(i + j) = k;
                                break 'clean;
                            }
                        }
                    }
                    perturb >>= 5;
                    i = (i * 5 + 1 + perturb) & newmask;
                }
            }
            size = grown;
            fill = used;
        }
    }
    size
}

/// The pieces of `active` (a flag a cell of an `nx` by `ny` grid) as `Map._pieces` finds them:
/// cells that touch, corners too, each piece's cells ascending, the pieces in the order a
/// Python set of the active cells pops their seeds. `cells` (count) receives the pieces one
/// after another and `bounds` (count + 1) where each begins; returns the number of pieces.
pub(crate) unsafe fn pieces(
    active: *const u8, nx: usize, ny: usize, cells: *mut i64, bounds: *mut i64,
    scratch: &mut Scratch,
) -> usize {
    let count = nx * ny;
    let keys = scratch.take::<i64>(count.max(1));
    let mut n = 0usize;
    for c in 0..count {
        if *active.add(c) != 0 {
            *keys.add(n) = c as i64;
            n += 1;
        }
    }
    *bounds = 0;
    if n == 0 {
        return 0;
    }
    let label = scratch.take::<i64>(count);
    let stack = scratch.take::<i64>(n);
    for c in 0..count {
        *label.add(c) = -1;
    }
    let mut pieces = 0usize;
    for q in 0..n {
        let seed = *keys.add(q);
        if *label.add(seed as usize) >= 0 {
            continue;
        }
        *label.add(seed as usize) = pieces as i64;
        let mut top = 1usize;
        *stack = seed;
        while top > 0 {
            top -= 1;
            let cell = *stack.add(top) as usize;
            let (i, j) = (cell.checked_div(ny).unwrap_or(0) as i64, cell.checked_rem(ny).unwrap_or(0) as i64);
            for a in i - 1..=i + 1 {
                for b in j - 1..=j + 1 {
                    if a < 0 || a >= nx as i64 || b < 0 || b >= ny as i64 {
                        continue;
                    }
                    let other = (a * ny as i64 + b) as usize;
                    if *active.add(other) != 0 && *label.add(other) < 0 {
                        *label.add(other) = pieces as i64;
                        *stack.add(top) = other as i64;
                        top += 1;
                    }
                }
            }
        }
        pieces += 1;
    }
    // each piece's cells ascending, at its label's place
    let start = scratch.take::<i64>(pieces + 1);
    for p in 0..=pieces {
        *start.add(p) = 0;
    }
    for q in 0..n {
        *start.add(*label.add(*keys.add(q) as usize) as usize + 1) += 1;
    }
    for p in 0..pieces {
        *start.add(p + 1) += *start.add(p);
    }
    let fill = scratch.take::<i64>(pieces);
    let grouped = scratch.take::<i64>(n);
    for p in 0..pieces {
        *fill.add(p) = *start.add(p);
    }
    for q in 0..n {
        let c = *keys.add(q);
        let p = *label.add(c as usize) as usize;
        *grouped.add(*fill.add(p) as usize) = c;
        *fill.add(p) += 1;
    }
    // the seeds in the order the set pops them: from its finger, the next slot that holds a
    // cell of a piece not yet taken
    let table = scratch.take::<i64>(16 * n + 32);
    let size = set_table(keys, n, table);
    let taken = scratch.take::<u8>(pieces);
    let mut finger = 0usize;
    let mut at = 0usize;
    for out in 0..pieces {
        let mut s = finger & (size - 1);
        loop {
            let k = *table.add(s);
            if k != EMPTY && *taken.add(*label.add(k as usize) as usize) == 0 {
                break;
            }
            s = (s + 1) & (size - 1);
        }
        let p = *label.add(*table.add(s) as usize) as usize;
        *taken.add(p) = 1;
        finger = s + 1;
        for q in *start.add(p) as usize..*start.add(p + 1) as usize {
            *cells.add(at) = *grouped.add(q);
            at += 1;
        }
        *bounds.add(out + 1) = at as i64;
    }
    pieces
}

pub(crate) fn pieces_room(count: usize) -> usize {
    room::<i64>(count.max(1)) * 4 + room::<i64>(count) * 2 + room::<i64>(16 * count + 32)
        + room::<i64>(count + 1) * 2 + room::<u8>(count) + 64
}

/// `Map._pieces(active)` (see `pieces`): `cells` and `bounds` int64, `count` and `count + 1`
/// long. Returns the number of pieces.
#[no_mangle]
pub unsafe extern "C" fn map_pieces(active: usize, nx: i32, ny: i32, cells: usize, bounds: usize) -> i32 {
    let active = take!(active, u8, BOOL, 0);
    let cells = take!(cells, i64, INT64, 3);
    let bounds = take!(bounds, i64, INT64, 4);
    let (nx, ny) = (nx.max(0) as usize, ny.max(0) as usize);
    let count = nx * ny;
    if active.len != count || cells.len < count || bounds.len < count + 1 {
        return refused(0);
    }
    let mut scratch = match Scratch::new(pieces_room(count)) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    pieces(active.p, nx, ny, cells.p, bounds.p, &mut scratch) as i32
}

/// `np.maximum(a, b)`: a NaN on either side is the answer.
#[inline(always)]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a != a || b != b {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// The activity above rest of potentials: `max(tanh(v / 2), 0)` (`Map._activity`).
#[inline(always)]
fn activity(v: f64) -> f64 {
    let t = crate::tanh(v / 2.0);
    if t != t {
        t
    } else if t >= 0.0 {
        t
    } else {
        0.0
    }
}

/// `Map.moves(expected)`: how far the moment moved each thing, on its cells, into `out`
/// (count). `potential` and `expected` (count, width) are the sheet after and before, `codes`
/// (kinds, units) the kinds' codes (a positive entry a unit of the code), `centers` (count, 2)
/// the cells' middles. In `moves`' order: the kinds present before or after, ascending; each
/// kind's pieces after and before (`_pieces`), their middles (the activity's weighted mean);
/// each piece after takes the nearest piece before within `stride` (its move that gap over
/// `step`) or appears (1); a piece before none took is gone (1).
#[no_mangle]
pub unsafe extern "C" fn map_moves(
    potential: usize, expected: usize, codes: usize, centers: usize, floor: f64, stride: f64,
    step: f64, nx: i32, ny: i32, out: usize,
) -> i32 {
    let potential = take!(potential, f64, FLOAT64, 0);
    let expected = take!(expected, f64, FLOAT64, 1);
    let codes = take!(codes, f64, FLOAT64, 2);
    let centers = take!(centers, f64, FLOAT64, 3);
    let out = take!(out, f64, FLOAT64, 9);
    moves(potential, expected, codes, centers, floor, stride, step, nx, ny, out)
}

/// `map_moves` on views.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn moves(
    potential: View<f64>, expected: View<f64>, codes: View<f64>, centers: View<f64>, floor: f64,
    stride: f64, step: f64, nx: i32, ny: i32, out: View<f64>,
) -> i32 {
    let (nx, ny) = (nx.max(0) as usize, ny.max(0) as usize);
    let count = nx * ny;
    let width = potential.n1;
    let (kinds, units) = (codes.n0, codes.n1);
    if potential.n0 != count || expected.n0 != count || expected.n1 != width || units > width
        || centers.n0 != count || centers.n1 != 2 || out.len != count
    {
        return refused(0);
    }
    let mut scratch = match Scratch::new(
        room::<f64>(count * width) * 2 + room::<f64>(count) * 2 + room::<u8>(count)
            + room::<u8>(units.max(1)) * 2 + room::<i64>(count) * 4 + room::<f64>(count * 2) * 2
            + room::<u8>(count) + room::<f64>(count.max(1)) + pieces_room(count) * 2 + 64,
    ) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    let after = scratch.take::<f64>(count * width);
    let before = scratch.take::<f64>(count * width);
    for i in 0..count * width {
        *after.add(i) = activity(potential.get(i));
        *before.add(i) = activity(expected.get(i));
    }
    let own = scratch.take::<f64>(count);
    let flags = scratch.take::<u8>(count);
    let held_after = scratch.take::<u8>(units.max(1));
    let held_before = scratch.take::<u8>(units.max(1));
    for (held, sheet) in [(held_after, after), (held_before, before)] {
        for u in 0..units {
            let mut any = false;
            for c in 0..count {
                any |= *sheet.add(c * width + u) > floor;
            }
            *held.add(u) = any as u8;
        }
    }
    let cells = [scratch.take::<i64>(count), scratch.take::<i64>(count)];
    let bounds = [scratch.take::<i64>(count + 1), scratch.take::<i64>(count + 1)];
    let middles = [scratch.take::<f64>(count * 2), scratch.take::<f64>(count * 2)];
    let free = scratch.take::<u8>(count);
    let picked = scratch.take::<f64>(count.max(1));
    for c in 0..count {
        out.set(c, 0.0);
    }
    for kind in 0..kinds {
        let present = |held: *mut u8| {
            let mut all = true;
            for u in 0..units {
                if codes.at(kind, u) > 0.0 {
                    all &= *held.add(u) != 0;
                }
            }
            all
        };
        if !(present(held_after) || present(held_before)) {
            continue;
        }
        let mut found = [0usize; 2];
        for (side, sheet) in [after, before].iter().enumerate() {
            // the kind's own activity: the least of its units at each cell (`_of_kind`)
            for c in 0..count {
                let mut least = f64::INFINITY;
                let mut first = true;
                for u in 0..units {
                    if codes.at(kind, u) != 0.0 {
                        let x = *sheet.add(c * width + u);
                        if first || x < least || x != x {
                            least = x;
                            first = false;
                        }
                    }
                }
                *own.add(c) = least;
                *flags.add(c) = (least > floor) as u8;
            }
            let mut room = match Scratch::new(pieces_room(count)) {
                Some(x) => x,
                None => return NO_ROOM,
            };
            let n = pieces(flags, nx, ny, cells[side], bounds[side], &mut room);
            found[side] = n;
            for p in 0..n {
                let (lo, hi) = (*bounds[side].add(p) as usize, *bounds[side].add(p + 1) as usize);
                let mut mx = 0.0f64;
                let mut my = 0.0f64;
                for q in lo..hi {
                    let c = *cells[side].add(q) as usize;
                    mx += *own.add(c) * centers.at(c, 0);
                    my += *own.add(c) * centers.at(c, 1);
                    *picked.add(q - lo) = *own.add(c);
                }
                let total = crate::bound::np_sum(picked, hi - lo);
                *middles[side].add(2 * p) = mx / total;
                *middles[side].add(2 * p + 1) = my / total;
            }
        }
        for j in 0..found[1] {
            *free.add(j) = 1;
        }
        for p in 0..found[0] {
            let (ax, ay) = (*middles[0].add(2 * p), *middles[0].add(2 * p + 1));
            let mut best = f64::INFINITY;
            let mut nearest = usize::MAX;
            for j in 0..found[1] {
                if *free.add(j) == 0 {
                    continue;
                }
                let dx = ax - *middles[1].add(2 * j);
                let dy = ay - *middles[1].add(2 * j + 1);
                let gap = crate::sqrt(0.0 + dx * dx + dy * dy);
                if nearest == usize::MAX || gap < best {
                    best = gap;
                    nearest = j;
                }
            }
            let moved;
            if nearest != usize::MAX && best <= stride {
                *free.add(nearest) = 0;
                moved = best / step;
                let (lo, hi) = (*bounds[1].add(nearest) as usize, *bounds[1].add(nearest + 1) as usize);
                for q in lo..hi {
                    let c = *cells[1].add(q) as usize;
                    out.set(c, np_maximum(out.get(c), moved));
                }
            } else {
                moved = 1.0;
            }
            let (lo, hi) = (*bounds[0].add(p) as usize, *bounds[0].add(p + 1) as usize);
            for q in lo..hi {
                let c = *cells[0].add(q) as usize;
                out.set(c, np_maximum(out.get(c), moved));
            }
        }
        for j in 0..found[1] {
            if *free.add(j) != 0 {
                let (lo, hi) = (*bounds[1].add(j) as usize, *bounds[1].add(j + 1) as usize);
                for q in lo..hi {
                    let c = *cells[1].add(q) as usize;
                    out.set(c, np_maximum(out.get(c), 1.0));
                }
            }
        }
    }
    0
}

/// `np.clip(x, 0, 1)`: the larger of x and 0 (a NaN kept), then the smaller of that and 1.
#[inline(always)]
fn clip01(x: f64) -> f64 {
    if x != x {
        return x;
    }
    let m = if x > 0.0 { x } else { 0.0 };
    if m < 1.0 {
        m
    } else {
        1.0
    }
}

/// The least of a kind's units' activity at cell `c` (`Map._of_kind`).
#[inline(always)]
unsafe fn of_kind(activity: View<f64>, codes: View<f64>, kind: usize, c: usize) -> f64 {
    let mut least = f64::INFINITY;
    let mut first = true;
    for u in 0..codes.n1 {
        if codes.at(kind, u) != 0.0 {
            let x = activity.at(c, u);
            if first || x < least || x != x {
                least = x;
                first = false;
            }
        }
    }
    least
}

/// `Map._write_values`: a report's values on its cells, claimed, and written at its kind's
/// weights. `near_cells` and `near_bounds` (int64) hold the pieces the report takes, one after
/// another; `named` and `named_mask` (value units) the values it names; `claimed` (bool) and
/// `write` (count, width) are changed in place. In `_write_values`' order: each piece and the
/// cells around it its kind holds (`_around`), the cells held where no other kind is, the
/// amounts the pieces held (`_amounts_of`: its kind's activity and every value unit summed
/// over their cells in order), the named values in their place, the claims, the write.
#[no_mangle]
pub unsafe extern "C" fn map_write_values(
    kind: i32, near_cells: usize, near_bounds: usize, weights: usize, named: usize, named_mask: usize,
    activity: usize, claimed: usize, write: usize, codes: usize, level: f64, floor: f64, nx: i32,
    ny: i32,
) -> i32 {
    let near_cells = take!(near_cells, i64, INT64, 1);
    let near_bounds = take!(near_bounds, i64, INT64, 2);
    let weights = take!(weights, f64, FLOAT64, 3);
    let named = take!(named, f64, FLOAT64, 4);
    let named_mask = take!(named_mask, u8, BOOL, 5);
    let activity = take!(activity, f64, FLOAT64, 6);
    let claimed = take!(claimed, u8, BOOL, 7);
    let write = take!(write, f64, FLOAT64, 8);
    let codes = take!(codes, f64, FLOAT64, 9);
    let (nx, ny) = (nx.max(0) as usize, ny.max(0) as usize);
    let count = nx * ny;
    let width = activity.n1;
    let units = codes.n1;
    let kinds = codes.n0;
    let kind = kind.max(0) as usize;
    let valued = width.saturating_sub(units);
    let pieces = if near_bounds.len > 0 { near_bounds.len - 1 } else { 0 };
    if activity.n0 != count || claimed.n0 != count || claimed.n1 != width || write.n0 != count
        || write.n1 != width || weights.len != count || named.len != valued
        || named_mask.len != valued || kind >= kinds || units > width
        || (pieces > 0 && near_bounds.get(pieces) as usize > near_cells.len)
    {
        return refused(0);
    }
    for q in 0..near_cells.len.min(if pieces > 0 { near_bounds.get(pieces) as usize } else { 0 }) {
        if near_cells.get(q) as usize >= count {
            return refused(1);
        }
    }
    let total_near = if pieces > 0 { near_bounds.get(pieces) as usize } else { 0 };
    let mut scratch = match Scratch::new(write_values_room(count, pieces, total_near, valued)) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    write_values(kind, near_cells.p, near_bounds.p, pieces, weights.p, named.p, named_mask.p,
                 activity, claimed, write, codes, level, floor, nx, ny, &mut scratch);
    0
}

/// Bytes `write_values` takes in a `Scratch`.
fn write_values_room(count: usize, pieces: usize, total_near: usize, valued: usize) -> usize {
    room::<u8>(count) * 4 + room::<i64>(count * (pieces + 1) + total_near * 9 + 8)
        + room::<f64>(valued + 1) * 2 + 64
}

/// The cells where a kind other than `kind` holds a thing (`Map._other_kinds`): of the kinds
/// with every unit of their code active somewhere, those whose least unit is above `floor`.
unsafe fn other_kinds(activity: View<f64>, codes: View<f64>, kind: usize, floor: f64,
                      count: usize, others: *mut u8) {
    let (kinds, units) = (codes.n0, codes.n1);
    for c in 0..count {
        *others.add(c) = 0;
    }
    for other in 0..kinds {
        if other == kind {
            continue;
        }
        let mut present = true;
        for u in 0..units {
            if codes.at(other, u) > 0.0 {
                let mut any = false;
                for c in 0..count {
                    any |= activity.at(c, u) > floor;
                }
                present &= any;
            }
        }
        if !present {
            continue;
        }
        for c in 0..count {
            if of_kind(activity, codes, other, c) > floor {
                *others.add(c) = 1;
            }
        }
    }
}

/// `Map._write_values` on arrays the caller has checked (see `map_write_values`).
#[allow(clippy::too_many_arguments)]
unsafe fn write_values(
    kind: usize, near_cells: *const i64, near_bounds: *const i64, pieces: usize,
    weights: *const f64, named: *const f64, named_mask: *const u8, activity: View<f64>,
    claimed: View<u8>, write: View<f64>, codes: View<f64>, level: f64, floor: f64, nx: usize,
    ny: usize, scratch: &mut Scratch,
) {
    let count = nx * ny;
    let units = codes.n1;
    let valued = activity.n1.saturating_sub(units);
    let total_near = if pieces > 0 { *near_bounds.add(pieces) as usize } else { 0 };
    let mark = scratch.take::<u8>(count);
    let around_flag = scratch.take::<u8>(count);
    let others = scratch.take::<u8>(count);
    let held = scratch.take::<u8>(count);
    let taken_all = scratch.take::<i64>(count * (pieces + 1) + total_near * 9 + 8);
    let sums = scratch.take::<f64>(valued + 1);
    let amounts = scratch.take::<f64>(valued.max(1));
    // the pieces it takes, each with the cells around it its kind holds, in order
    let mut taken_n = 0usize;
    for p in 0..pieces {
        let (lo, hi) = (*near_bounds.add(p) as usize, *near_bounds.add(p + 1) as usize);
        for c in 0..count {
            *around_flag.add(c) = 0;
            *mark.add(c) = 0;
        }
        for q in lo..hi {
            let cell = *near_cells.add(q) as usize;
            *mark.add(cell) = 1;
            let (i, j) = (cell.checked_div(ny).unwrap_or(0) as i64, cell.checked_rem(ny).unwrap_or(0) as i64);
            for a in i - 1..=i + 1 {
                for b in j - 1..=j + 1 {
                    // the neighbour clipped to the grid, as ``_around`` clips it
                    let a = if a < 0 { 0 } else if a > nx as i64 - 1 { nx - 1 } else { a as usize };
                    let b = if b < 0 { 0 } else if b > ny as i64 - 1 { ny - 1 } else { b as usize };
                    *around_flag.add(a * ny + b) = 1;
                }
            }
        }
        for c in 0..count {
            if *around_flag.add(c) != 0 {
                let mut holds = false;
                for u in 0..units {
                    if codes.at(kind, u) != 0.0 {
                        holds |= activity.at(c, u) > 0.0;
                    }
                }
                if holds {
                    *mark.add(c) = 1;
                }
            }
        }
        for c in 0..count {
            if *mark.add(c) != 0 {
                *taken_all.add(taken_n) = c as i64;
                taken_n += 1;
            }
        }
    }
    other_kinds(activity, codes, kind, floor, count, others);
    // the cells held: the report's own cells and those it takes, where no other kind is
    for c in 0..count {
        *held.add(c) = 0;
    }
    for c in 0..count {
        if *weights.add(c) > 0.0 {
            *held.add(c) = 1;
        }
    }
    for q in 0..taken_n {
        *held.add(*taken_all.add(q) as usize) = 1;
    }
    for c in 0..count {
        if *others.add(c) != 0 {
            *held.add(c) = 0;
        }
    }
    // the amounts the pieces it takes held on every value unit
    for v in 0..valued {
        *amounts.add(v) = 0.0;
    }
    if taken_n > 0 {
        for q in 0..taken_n {
            let c = *taken_all.add(q) as usize;
            let own = of_kind(activity, codes, kind, c);
            if q == 0 {
                *sums = own;
                for v in 0..valued {
                    *sums.add(v + 1) = activity.at(c, units + v);
                }
            } else {
                *sums += own;
                for v in 0..valued {
                    *sums.add(v + 1) += activity.at(c, units + v);
                }
            }
        }
        if *sums > floor {
            for v in 0..valued {
                *amounts.add(v) = clip01(*sums.add(v + 1) / *sums);
            }
        }
    }
    for v in 0..valued {
        if *named_mask.add(v) != 0 {
            *amounts.add(v) = *named.add(v);
        }
    }
    // the claims: the value units of the cells held and of its own cells
    for c in 0..count {
        if *held.add(c) != 0 || *weights.add(c) > 0.0 {
            for v in 0..valued {
                claimed.put(c, units + v, 1);
            }
        }
    }
    let mut any = false;
    for v in 0..valued {
        any |= *amounts.add(v) != 0.0;
    }
    if any {
        let mut square = 0.0f64;
        for c in 0..count {
            square += *weights.add(c) * *weights.add(c);
        }
        let norm = crate::sqrt(square);
        for c in 0..count {
            let a = level * *weights.add(c) / norm;
            for v in 0..valued {
                write.put(c, units + v, write.at(c, units + v) + a * *amounts.add(v));
            }
        }
    }
}


/// `np.min` of the values at `cells`: the least, a NaN where one is.
#[inline(always)]
unsafe fn least_at(values: *const f64, cells: *const i64, lo: usize, hi: usize) -> f64 {
    let mut least = *values.add(*cells.add(lo) as usize);
    for q in lo + 1..hi {
        let x = *values.add(*cells.add(q) as usize);
        if x < least || x != x {
            least = x;
        }
    }
    least
}

/// One report of `Map.moment`, the work of its loop over reports but the motion record's
/// line (`_line`, which reads none of what this writes): the pieces of the report's kind
/// (`_pieces`), those within `step` of its place or else the nearest within `stride`, the
/// pieces in doubt (with physics, those that carry none of the things the near pieces carry,
/// or carry one of them, and the cells that carry one of them are claimed), the claims of each
/// piece taken and of the cells around it its kind alone holds, the place's weights on cells
/// (`weights`), their claim and their write, the values (`_write_values`), the fresh cells and
/// the kind found. `place` is (`px`, `py`); `motion` (count, width * fields) the records;
/// `claimed`, `write`, `fresh` and `lost` change in place. The pieces taken go into
/// `near_cells` and `near_bounds` (int64, 2 count and 2 count + 1), in `moment`'s order.
/// Returns how many.
#[no_mangle]
pub unsafe extern "C" fn map_report(
    kind: i32, px: f64, py: f64, codes: usize, centers: usize, activity: usize, doubt: usize,
    motion: usize, has_physics: i32, hidden: i32, ident: i32, fields: i32, lowx: f64, lowy: f64,
    sizex: f64, sizey: f64, floor: f64, step: f64, stride: f64, level: f64, nx: i32, ny: i32,
    has_values: i32, named: usize, named_mask: usize, claimed: usize, write: usize, fresh: usize,
    lost: usize, near_cells: usize, near_bounds: usize,
) -> i32 {
    let codes = take!(codes, f64, FLOAT64, 3);
    let centers = take!(centers, f64, FLOAT64, 4);
    let activity = take!(activity, f64, FLOAT64, 5);
    let doubt = take!(doubt, f64, FLOAT64, 6);
    let motion = take!(motion, f64, FLOAT64, 7);
    let named = take!(named, f64, FLOAT64, 23);
    let named_mask = take!(named_mask, u8, BOOL, 24);
    let claimed = take!(claimed, u8, BOOL, 25);
    let write = take!(write, f64, FLOAT64, 26);
    let fresh = take!(fresh, u8, BOOL, 27);
    let lost = take!(lost, f64, FLOAT64, 28);
    let near_cells = take!(near_cells, i64, INT64, 29);
    let near_bounds = take!(near_bounds, i64, INT64, 30);
    report(kind, px, py, codes, centers, activity, doubt, motion, has_physics, hidden, ident, fields,
           lowx, lowy, sizex, sizey, floor, step, stride, level, nx, ny, has_values, named,
           named_mask, claimed, write, fresh, lost, near_cells, near_bounds)
}

/// `map_report` on views.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn report(
    kind: i32, px: f64, py: f64, codes: View<f64>, centers: View<f64>, activity: View<f64>,
    doubt: View<f64>, motion: View<f64>, has_physics: i32, hidden: i32, ident: i32, fields: i32,
    lowx: f64, lowy: f64, sizex: f64, sizey: f64, floor: f64, step: f64, stride: f64, level: f64,
    nx: i32, ny: i32, has_values: i32, named: View<f64>, named_mask: View<u8>, claimed: View<u8>,
    write: View<f64>, fresh: View<u8>, lost: View<f64>, near_cells: View<i64>,
    near_bounds: View<i64>,
) -> i32 {
    let (nx, ny) = (nx.max(0) as usize, ny.max(0) as usize);
    let count = nx * ny;
    let width = activity.n1;
    let (kinds, units) = (codes.n0, codes.n1);
    let valued = width.saturating_sub(units);
    let physics = has_physics != 0;
    let values = has_values != 0;
    let fields = fields.max(0) as usize;
    let (hidden, ident) = (hidden.max(0) as usize, ident.max(0) as usize);
    if kind < 0 || kind as usize >= kinds || nx < 2 || ny < 2 || units > width
        || activity.n0 != count || centers.n0 != count || centers.n1 != 2 || doubt.len != count
        || claimed.n0 != count || claimed.n1 != width || write.n0 != count || write.n1 != width
        || fresh.len != count || lost.len < units || near_cells.len < 2 * count
        || near_bounds.len < 2 * count + 1
        || (values && (named.len != valued || named_mask.len != valued))
        || (physics && (motion.n0 != count || hidden >= fields || ident >= fields
                        || motion.n1 != width * fields))
    {
        return refused(0);
    }
    if !finite(px) || !finite(py) {
        return refused(1);
    }
    let kind = kind as usize;
    let mut first_unit = units;
    for u in 0..units {
        if codes.at(kind, u) != 0.0 {
            first_unit = u;
            break;
        }
    }
    if first_unit == units {
        return refused(0); // a kind with no units: `_of_kind` has nothing to read
    }
    let mut scratch = match Scratch::new(
        pieces_room(count) + room::<u8>(count) * 4 + room::<i64>(count) * 3
            + room::<i64>(count + 1) + room::<f64>(count) * 6 + 64,
    ) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    // how far each cell's middle lies from the place (`np.linalg.norm(axis=1)`)
    let apart = scratch.take::<f64>(count);
    for c in 0..count {
        let dx = centers.at(c, 0) - px;
        let dy = centers.at(c, 1) - py;
        *apart.add(c) = crate::sqrt(-0.0 + dx * dx + dy * dy);
    }
    // the pieces of the kind's activity
    let own = scratch.take::<u8>(count);
    for c in 0..count {
        *own.add(c) = (of_kind(activity, codes, kind, c) > floor) as u8;
    }
    let cells = scratch.take::<i64>(count);
    let bounds = scratch.take::<i64>(count + 1);
    let found = pieces(own, nx, ny, cells, bounds, &mut scratch);
    let nearest = scratch.take::<f64>(count);
    for p in 0..found {
        *nearest.add(p) = least_at(apart, cells, *bounds.add(p) as usize,
                                   *bounds.add(p + 1) as usize);
    }
    // the pieces within a step, or the nearest within a stride (the first of equals)
    let near = scratch.take::<i64>(count);
    let mut n_near = 0usize;
    for p in 0..found {
        if *nearest.add(p) <= step {
            *near.add(n_near) = p as i64;
            n_near += 1;
        }
    }
    if n_near == 0 {
        let mut best = -1i64;
        for p in 0..found {
            if *nearest.add(p) <= stride && (best < 0 || *nearest.add(p) < *nearest.add(best as usize)) {
                best = p as i64;
            }
        }
        if best >= 0 {
            *near = best;
            n_near = 1;
        }
    }
    // the pieces in doubt: the mean of their doubt above a half
    let doubtful = scratch.take::<i64>(count);
    let gathered = scratch.take::<f64>(count);
    let mut n_doubt = 0usize;
    for p in 0..found {
        let (lo, hi) = (*bounds.add(p) as usize, *bounds.add(p + 1) as usize);
        for q in lo..hi {
            *gathered.add(q - lo) = doubt.get(*cells.add(q) as usize);
        }
        if crate::bound::np_sum(gathered, hi - lo) / ((hi - lo) as f64) > 0.5 {
            *doubtful.add(n_doubt) = p as i64;
            n_doubt += 1;
        }
    }
    if physics {
        // the thing the physics carries at each cell on the kind's first unit, and those the
        // near pieces carry (every cell's where none is near)
        let theirs = scratch.take::<f64>(count);
        let at_hidden = first_unit * fields + hidden;
        let at_ident = first_unit * fields + ident;
        for c in 0..count {
            *theirs.add(c) = if motion.at(c, at_hidden) > 0.0 { motion.at(c, at_ident) } else { 0.0 };
        }
        let mine = scratch.take::<f64>(count);
        let mut n_mine = 0usize;
        let mut add = |x: f64| {
            if x > 0.0 {
                for m in 0..n_mine {
                    if *mine.add(m) == x {
                        return;
                    }
                }
                *mine.add(n_mine) = x;
                n_mine += 1;
            }
        };
        if n_near > 0 {
            for k in 0..n_near {
                let p = *near.add(k) as usize;
                for q in *bounds.add(p) as usize..*bounds.add(p + 1) as usize {
                    add(*theirs.add(*cells.add(q) as usize));
                }
            }
        } else {
            for c in 0..count {
                add(*theirs.add(c));
            }
        }
        let is_mine = |x: f64| -> bool {
            for m in 0..n_mine {
                if *mine.add(m) == x {
                    return true;
                }
            }
            false
        };
        let mut kept = 0usize;
        for k in 0..n_doubt {
            let p = *doubtful.add(k) as usize;
            let (mut any, mut carried) = (false, false);
            for q in *bounds.add(p) as usize..*bounds.add(p + 1) as usize {
                let x = *theirs.add(*cells.add(q) as usize);
                any |= x != 0.0;
                carried |= is_mine(x);
            }
            if !any || carried {
                *doubtful.add(kept) = p as i64;
                kept += 1;
            }
        }
        n_doubt = kept;
        for c in 0..count {
            if is_mine(*theirs.add(c)) {
                for u in 0..units {
                    if codes.at(kind, u) != 0.0 {
                        claimed.put(c, u, 1);
                    }
                }
            }
        }
    }
    // the pieces taken, near then in doubt
    let taken = n_near + n_doubt;
    let mut at = 0usize;
    *near_bounds.p = 0;
    for k in 0..taken {
        let p = if k < n_near { *near.add(k) } else { *doubtful.add(k - n_near) } as usize;
        for q in *bounds.add(p) as usize..*bounds.add(p + 1) as usize {
            near_cells.set(at, *cells.add(q));
            at += 1;
        }
        near_bounds.set(k + 1, at as i64);
    }
    // each piece's cells and the cells around them its kind alone holds
    if taken > 0 {
        let others = scratch.take::<u8>(count);
        other_kinds(activity, codes, kind, floor, count, others);
        let around = scratch.take::<u8>(count);
        for k in 0..taken {
            for c in 0..count {
                *around.add(c) = 0;
            }
            for q in near_bounds.get(k) as usize..near_bounds.get(k + 1) as usize {
                let cell = near_cells.get(q) as usize;
                for u in 0..units {
                    if codes.at(kind, u) != 0.0 {
                        claimed.put(cell, u, 1);
                    }
                }
                let (i, j) = (cell.checked_div(ny).unwrap_or(0) as i64,
                              cell.checked_rem(ny).unwrap_or(0) as i64);
                for a in i - 1..=i + 1 {
                    for b in j - 1..=j + 1 {
                        let a = if a < 0 { 0 } else if a > nx as i64 - 1 { nx - 1 } else { a as usize };
                        let b = if b < 0 { 0 } else if b > ny as i64 - 1 { ny - 1 } else { b as usize };
                        *around.add(a * ny + b) = 1;
                    }
                }
            }
            for c in 0..count {
                if *around.add(c) == 0 || *others.add(c) != 0 {
                    continue;
                }
                let mut holds = false;
                for u in 0..units {
                    if codes.at(kind, u) != 0.0 {
                        holds |= activity.at(c, u) > 0.0;
                    }
                }
                if holds {
                    for u in 0..units {
                        if codes.at(kind, u) != 0.0 {
                            claimed.put(c, u, 1);
                        }
                    }
                }
            }
        }
    }
    // the place as weights on the four cells around it (`_spread` of one place)
    let weights = scratch.take::<f64>(count);
    let ax = {
        let x = (px - lowx) / sizex - 0.5;
        let x = if x > 0.0 { x } else { 0.0 };
        if x < nx as f64 - 1.0 { x } else { nx as f64 - 1.0 }
    };
    let ay = {
        let y = (py - lowy) / sizey - 0.5;
        let y = if y > 0.0 { y } else { 0.0 };
        if y < ny as f64 - 1.0 { y } else { ny as f64 - 1.0 }
    };
    let ix = (ax as i64).min(nx as i64 - 2);
    let iy = (ay as i64).min(ny as i64 - 2);
    let (fx, fy) = (ax - ix as f64, ay - iy as f64);
    let cell = (ix * ny as i64 + iy) as usize;
    *weights.add(cell) = 1.0 * (1.0 - fx) * (1.0 - fy);
    *weights.add(cell + ny) = 1.0 * fx * (1.0 - fy);
    *weights.add(cell + 1) = 1.0 * (1.0 - fx) * fy;
    *weights.add(cell + ny + 1) = 1.0 * fx * fy;
    let mut square = 0.0f64;
    for c in 0..count {
        square += *weights.add(c) * *weights.add(c);
    }
    let norm = crate::sqrt(square);
    for c in 0..count {
        let w = *weights.add(c);
        let add = level * w / norm;
        for u in 0..units {
            if codes.at(kind, u) != 0.0 {
                if w > 0.0 {
                    claimed.put(c, u, 1);
                }
                write.put(c, u, write.at(c, u) + add);
            }
        }
    }
    if values {
        let mut more = match Scratch::new(write_values_room(count, taken, at, valued)) {
            Some(x) => x,
            None => return NO_ROOM,
        };
        write_values(kind, near_cells.p, near_bounds.p, taken, weights, named.p, named_mask.p,
                     activity, claimed, write, codes, level, floor, nx, ny, &mut more);
    }
    for c in 0..count {
        if *weights.add(c) > 0.0 {
            fresh.set(c, 1);
        }
    }
    for u in 0..units {
        if codes.at(kind, u) != 0.0 {
            lost.set(u, 0.0);
        }
    }
    taken as i32
}
