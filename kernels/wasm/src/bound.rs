//! The bound records' read, `_bound_recall` of `cadence/compiled.py`: `BoundRecords._taking`
//! and `_agreed` of `cadence/bound.py` in one call, in the order NumPy sums them under Pyodide
//! (a dot product and a vector's norm in order, a row's sum pairwise).

use crate::{refused, room, sqrt, view, Scratch, View, BOOL, FLOAT64, INT64, NO_ROOM};

macro_rules! take {
    ($id:expr, $t:ty, $num:expr, $k:expr) => {
        match view::<$t>($id, $num) {
            Some(x) => x,
            None => return refused($k),
        }
    };
}

/// NumPy's sum of a contiguous run (`pairwise_sum`): under eight in order from -0.0, up to 128
/// with eight running sums, longer runs halved at a multiple of eight.
pub(crate) unsafe fn np_sum(a: *const f64, n: usize) -> f64 {
    if n < 8 {
        let mut res = -0.0f64;
        for i in 0..n {
            res += *a.add(i);
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        for (j, slot) in r.iter_mut().enumerate() {
            *slot = *a.add(j);
        }
        let mut i = 8usize;
        let stop = n - n % 8;
        while i < stop {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += *a.add(i + j);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += *a.add(i);
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    np_sum(a, half) + np_sum(a.add(half), n - half)
}

/// NumPy's dot product of two vectors without BLAS: in order from 0.0.
#[inline(always)]
pub(crate) unsafe fn np_dot(a: *const f64, b: *const f64, n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        s += *a.add(i) * *b.add(i);
    }
    s
}

/// `bound._unit`: a vector over its norm where the norm exceeds 1e-12, into `out`.
#[inline(always)]
unsafe fn unit(a: *const f64, n: usize, out: *mut f64) {
    let norm = sqrt(np_dot(a, a, n));
    for i in 0..n {
        *out.add(i) = if norm > 1e-12 { *a.add(i) / norm } else { *a.add(i) };
    }
}

/// `BoundRecords._agreed(_taking(code), row)`: the records the store's candidates `slots` hold
/// that a read of `code` reaches, then each field they hold alike, its mean into `out` at its
/// columns and a one into `present`. Returns the fields that came back.
#[no_mangle]
pub unsafe extern "C" fn bound_recall(
    slots: usize, keys: usize, code: usize, deep: usize, kept: usize, has_keep: i32, gone: usize,
    has_forget: i32, values: usize, has_voice: i32, v0: i32, v1: i32, paid: usize, refused_: usize,
    fields: usize, voice_field: i32, known_at: f64, match_at: f64, heads: i32, near: f64,
    agree: f64, most: f64, out: usize, present: usize,
) -> i32 {
    let slots = take!(slots, i64, INT64, 0);
    let keys = take!(keys, f64, FLOAT64, 1);
    let code = take!(code, f64, FLOAT64, 2);
    let deep = take!(deep, f64, FLOAT64, 3);
    let kept = take!(kept, u8, BOOL, 4);
    let gone = take!(gone, u8, BOOL, 6);
    let values = take!(values, f64, FLOAT64, 8);
    let paid = take!(paid, f64, FLOAT64, 12);
    let refused_v = take!(refused_, u8, BOOL, 13);
    let fields = take!(fields, i64, INT64, 14);
    let out = take!(out, f64, FLOAT64, 22);
    let present = take!(present, i64, INT64, 23);
    recall(slots, keys, code, deep, kept, has_keep, gone, has_forget, values, has_voice, v0, v1,
           paid, refused_v, fields, voice_field, known_at, match_at, heads, near, agree, most,
           out, present)
}

/// `bound_recall` on views of the arrays: the read itself, for a caller that holds the arrays
/// (the native core) as for one that hands NumPy's objects over.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn recall(
    slots: View<i64>, keys: View<f64>, code: View<f64>, deep: View<f64>, kept: View<u8>,
    has_keep: i32, gone: View<u8>, has_forget: i32, values: View<f64>, has_voice: i32, v0: i32,
    v1: i32, paid: View<f64>, refused_v: View<u8>, fields: View<i64>, voice_field: i32,
    known_at: f64, match_at: f64, heads: i32, near: f64, agree: f64, most: f64, out: View<f64>,
    present: View<i64>,
) -> i32 {
    let cap = keys.n0;
    let kw = keys.n1;
    let inputs = values.n1;
    let nf = fields.n0;
    let (v0, v1) = (v0.max(0) as usize, v1.max(0) as usize);
    if code.len != kw || values.n0 != cap || deep.len < cap || kept.len < cap || gone.len < cap
        || paid.len < cap || refused_v.len < cap || fields.n1 != 2 || out.len != inputs
        || present.len < nf || (has_voice != 0 && (v1 > inputs || v0 > v1))
    {
        return refused(22);
    }
    for q in 0..slots.len {
        let s = slots.get(q);
        if s < 0 || s as usize >= cap {
            return refused(0);
        }
    }
    for f in 0..nf {
        let (a, b) = (fields.at(f, 0), fields.at(f, 1));
        if a < 0 || b < a || b as usize > inputs {
            return refused(14);
        }
    }
    for i in 0..inputs {
        out.set(i, 0.0);
    }
    for f in 0..nf {
        present.set(f, 0);
    }
    let k = slots.len;
    let mut scratch = match Scratch::new(
        room::<f64>(k) * 3 + room::<usize>(k) * 4 + room::<f64>(inputs.max(1)) * 3
            + room::<f64>(kw.max(1)) + 64,
    ) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    // `_taking`: the known slots and their cosines to the cue, those at MATCH or above
    let code_norm = sqrt(np_dot(code.p, code.p, kw));
    let taken = scratch.take::<usize>(k);
    let weights = scratch.take::<f64>(k);
    let squares = scratch.take::<f64>(kw.max(1));
    let mut count = 0usize;
    for q in 0..k {
        let s = slots.get(q) as usize;
        let mut known = deep.get(s) >= known_at;
        if has_keep != 0 {
            known |= kept.get(s) != 0;
        }
        if has_voice != 0 {
            let mut said = false;
            for i in v0..v1 {
                said |= values.at(s, i) != 0.0;
            }
            if said {
                known = paid.get(s) > 0.0 || refused_v.get(s) != 0;
            }
        }
        if has_forget != 0 {
            known &= gone.get(s) == 0;
        }
        if !known {
            continue;
        }
        let key = keys.p.add(s * kw);
        for i in 0..kw {
            *squares.add(i) = *key.add(i) * *key.add(i);
        }
        let norm = sqrt(np_sum(squares, kw)) * code_norm;
        let dot = np_dot(key, code.p, kw);
        let floor = if norm > 1e-12 { norm } else { 1e-12 };
        let cosine = if norm > 0.0 { dot / floor } else { 0.0 };
        if cosine >= match_at {
            *taken.add(count) = s;
            *weights.add(count) = cosine;
            count += 1;
        }
    }
    if count == 0 {
        return 0;
    }
    // `_agreed`: each field the reached records hold alike
    let holding = scratch.take::<usize>(count);
    let group_of = scratch.take::<usize>(count);
    let firsts = scratch.take::<usize>(count);
    let picked = scratch.take::<f64>(count);
    let ua = scratch.take::<f64>(inputs.max(1));
    let ub = scratch.take::<f64>(inputs.max(1));
    let mean = scratch.take::<f64>(inputs.max(1));
    let heads = heads.max(0) as usize;
    let mut back = 0i32;
    for f in 0..nf {
        if f as i32 == voice_field {
            continue;
        }
        let (first, last) = (fields.at(f, 0) as usize, fields.at(f, 1) as usize);
        let width = last - first;
        let part = |q: usize| values.p.add(*taken.add(q) * inputs + first);
        let mut held = 0usize;
        for q in 0..count {
            let mut any = false;
            for i in 0..width {
                any |= *part(q).add(i) != 0.0;
            }
            if any {
                *holding.add(held) = q;
                held += 1;
            }
        }
        if held == 0 {
            continue;
        }
        // the nearest first (a stable sort by falling cosine), HEADS of them
        for a in 1..held {
            let x = *holding.add(a);
            let mut b = a;
            while b > 0 && -*weights.add(*holding.add(b - 1)) > -*weights.add(x) {
                *holding.add(b) = *holding.add(b - 1);
                b -= 1;
            }
            *holding.add(b) = x;
        }
        if held > heads {
            held = heads;
        }
        let top = near * *weights.add(*holding);
        let mut kept_n = 0usize;
        for a in 0..held {
            let q = *holding.add(a);
            if *weights.add(q) >= top {
                *holding.add(kept_n) = q;
                kept_n += 1;
            }
        }
        held = kept_n;
        // groups held alike: a record joins the first group whose first it agrees with
        let mut groups = 0usize;
        for a in 0..held {
            let q = *holding.add(a);
            unit(part(q), width, ua);
            let mut joined = usize::MAX;
            for g in 0..groups {
                unit(part(*firsts.add(g)), width, ub);
                if np_dot(ua, ub, width) >= agree {
                    joined = g;
                    break;
                }
            }
            if joined == usize::MAX {
                *firsts.add(groups) = q;
                joined = groups;
                groups += 1;
            }
            *group_of.add(a) = joined;
        }
        // each group's mass (its cosines summed as NumPy sums them); the heaviest first
        let mut best = 0usize;
        let mut best_mass = f64::NEG_INFINITY;
        for g in 0..groups {
            let mut m = 0usize;
            for a in 0..held {
                if *group_of.add(a) == g {
                    *picked.add(m) = *weights.add(*holding.add(a));
                    m += 1;
                }
            }
            let mass = np_sum(picked, m);
            if g == 0 || mass > best_mass {
                best_mass = mass;
                best = g;
            }
        }
        for a in 0..held {
            *picked.add(a) = *weights.add(*holding.add(a));
        }
        let all = np_sum(picked, held);
        if !(best_mass > most * all) {
            continue;
        }
        // the group's mean: its rows summed in order (a field one wide, pairwise), over its size
        let mut m = 0usize;
        for a in 0..held {
            if *group_of.add(a) == best {
                let row = part(*holding.add(a));
                if width == 1 {
                    *picked.add(m) = *row;
                } else if m == 0 {
                    for i in 0..width {
                        *mean.add(i) = *row.add(i);
                    }
                } else {
                    for i in 0..width {
                        *mean.add(i) += *row.add(i);
                    }
                }
                m += 1;
            }
        }
        if width == 1 {
            *mean = np_sum(picked, m);
        }
        for i in 0..width {
            out.set(first + i, *mean.add(i) / m as f64);
        }
        present.set(f, 1);
        back += 1;
    }
    back
}
