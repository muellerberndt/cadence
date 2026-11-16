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
    flat: u32, pairs: u32, flat_apical: u32, apical_pairs: u32, starts: u32, pair_pre: u32,
    pair_post: u32, offset: u32, freezable: u32, bias: u32, dt: f64, slope: f64,
    threshold: f64, rest: f64, leak: f64, has_apical: i32, pool_of: u32, pool_index: u32,
    pool_starts: u32, pool_run: u32, fraction: f64, floor2: f64, tiny: f64, potential: u32,
    carried: u32, has_carried: i32, write: u32, has_write: i32, clock: u32, lost: u32,
    doubt: u32, fresh: u32, has_fresh: i32, kept: u32, has_kept: i32, sure: u32,
    has_sure: i32, cue: u32, has_cue: i32, runs: u32, sweeps: i32, out: u32,
) -> i32 {
    let flat = take!(flat, f64, FLOAT64, 0);
    let pairs = take!(pairs, i64, INT64, 1);
    let flat_apical = take!(flat_apical, f64, FLOAT64, 2);
    let apical_pairs = take!(apical_pairs, i64, INT64, 3);
    let starts = take!(starts, i64, INT64, 4);
    let pair_pre = take!(pair_pre, i64, INT64, 5);
    let pair_post = take!(pair_post, i64, INT64, 6);
    let offset = take!(offset, i64, INT64, 7);
    let freezable = take!(freezable, u8, BOOL, 8);
    let bias = take!(bias, f64, FLOAT64, 9);
    let pool_of = take!(pool_of, i64, INT64, 16);
    let pool_index = take!(pool_index, i64, INT64, 17);
    let pool_starts = take!(pool_starts, i64, INT64, 18);
    let pool_run = take!(pool_run, i64, INT64, 19);
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
        held: one(small.add(2)), has_held: false,
    };
    let change = View { p: small.add(4), n0: 1, n1: 1, len: 1 };
    let kept_view = View { p: kept_room, n0: kept_len, n1: 1, len: kept_len };
    let given = View { p: small.add(5), n0: 1, n1: 1, len: 1 };
    settle(&net, freezable, sweeps.max(0) as usize, 0.0, false, given, false, change, kept_view)
}
