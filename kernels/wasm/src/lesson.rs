//! A lesson's per-synapse passes and the small kernels around a moment: `_trace_step`,
//! `_place`, `_pick`, `_group_power`, `_divisor`, `_plain`, `_moments`, `_step`, `_tied`,
//! `_add_episode_terms` and `_clear_choice` of `cadence/fused.py`, in its order.

use crate::{
    fabs, index, pymax, pymin, refused, sqrt, view, Index, View, FLOAT64, INT64,
};

macro_rules! take {
    ($id:expr, $t:ty, $num:expr, $k:expr) => {
        match view::<$t>($id, $num) {
            Some(x) => x,
            None => return refused($k),
        }
    };
}

macro_rules! index_of {
    ($id:expr, $k:expr) => {
        match index($id) {
            Some(x) => x,
            None => return refused($k),
        }
    };
}

/// `_trace_step`: one pass per row of the two phases' contrast, the traces' decay and
/// accumulation and the dopamine-weighted sum into the step.
#[no_mangle]
pub unsafe extern "C" fn trace_step(
    trace: usize, trace_bias: usize, decay: f64, s_plus: usize, s_minus: usize, pre: usize, post: usize,
    span: f64, delta: usize, step_scale: usize, step_bias: usize,
) -> i32 {
    let trace = take!(trace, f64, FLOAT64, 0);
    let trace_bias = take!(trace_bias, f64, FLOAT64, 1);
    let s_plus = take!(s_plus, f64, FLOAT64, 3);
    let s_minus = take!(s_minus, f64, FLOAT64, 4);
    let pre = index_of!(pre, 5);
    let post = index_of!(post, 6);
    let delta = take!(delta, f64, FLOAT64, 8);
    let step_scale = take!(step_scale, f64, FLOAT64, 9);
    let step_bias = take!(step_bias, f64, FLOAT64, 10);
    let (batch, edges) = (trace.n0, trace.n1);
    let n = trace_bias.n1;
    if pre.len() != edges || post.len() != edges || s_plus.n0 != batch || s_plus.n1 != n
        || s_minus.n0 != batch || s_minus.n1 != n || trace_bias.n0 != batch
        || delta.len < batch || step_scale.len != edges || step_bias.len != n
    {
        return refused(0);
    }
    for e in 0..edges {
        let (i, j) = (pre.get(e), post.get(e));
        if i < 0 || j < 0 || i as usize >= n || j as usize >= n {
            return refused(5);
        }
    }
    for e in 0..edges {
        step_scale.set(e, 0.0);
    }
    for i in 0..n {
        step_bias.set(i, 0.0);
    }
    for b in 0..batch {
        let d = delta.get(b) / batch as f64;
        for e in 0..edges {
            let (i, j) = (pre.get(e) as usize, post.get(e) as usize);
            let (a_plus, a_minus) = (s_plus.at(b, i), s_minus.at(b, i));
            let (b_plus, b_minus) = (s_plus.at(b, j), s_minus.at(b, j));
            let c = (a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus) / span;
            let t = decay * trace.at(b, e) + c;
            trace.put(b, e, t);
            step_scale.set(e, step_scale.get(e) + d * t);
        }
        for i in 0..n {
            let c = (s_plus.at(b, i) - s_minus.at(b, i)) / span;
            let t = decay * trace_bias.at(b, i) + c;
            trace_bias.put(b, i, t);
            step_bias.set(i, step_bias.get(i) + d * t);
        }
    }
    0
}

unsafe fn inside(ix: Index, size: usize) -> bool {
    for e in 0..ix.len() {
        let k = ix.get(e);
        if k < 0 || k as usize >= size {
            return false;
        }
    }
    true
}

/// `_place`: `out[index] = values`.
#[no_mangle]
pub unsafe extern "C" fn place(index: usize, values: usize, out: usize) -> i32 {
    let ix = index_of!(index, 0);
    let values = take!(values, f64, FLOAT64, 1);
    let out = take!(out, f64, FLOAT64, 2);
    if values.len < ix.len() || !inside(ix, out.len) {
        return refused(0);
    }
    for e in 0..ix.len() {
        out.set(ix.get(e) as usize, values.get(e));
    }
    0
}

/// `_pick`: `out = values[index] / divisor`.
#[no_mangle]
pub unsafe extern "C" fn pick(values: usize, index: usize, divisor: f64, out: usize) -> i32 {
    let values = take!(values, f64, FLOAT64, 0);
    let ix = index_of!(index, 1);
    let out = take!(out, f64, FLOAT64, 3);
    if out.len < ix.len() || !inside(ix, values.len) {
        return refused(1);
    }
    for e in 0..ix.len() {
        out.set(e, values.get(ix.get(e) as usize) / divisor);
    }
    0
}

/// `_group_power`: each group's sum of squares in index order and its member count (at least
/// one), into `power` and `count` (zeros, `size` long).
#[no_mangle]
pub unsafe extern "C" fn group_power(values: usize, group: usize, power: usize, count: usize) -> i32 {
    let values = take!(values, f64, FLOAT64, 0);
    let group = index_of!(group, 1);
    let power = take!(power, f64, FLOAT64, 2);
    let count = take!(count, f64, FLOAT64, 3);
    if group.len() < values.len || count.len != power.len || !inside(group, power.len) {
        return refused(1);
    }
    for e in 0..values.len {
        let g = group.get(e) as usize;
        power.set(g, power.get(g) + values.get(e) * values.get(e));
        count.set(g, count.get(g) + 1.0);
    }
    for g in 0..count.len {
        if count.get(g) < 1.0 {
            count.set(g, 1.0);
        }
    }
    0
}

/// `_divisor`: `out = values + share * root[group] + tiny`.
#[no_mangle]
pub unsafe extern "C" fn divisor(
    values: usize, group: usize, root: usize, share: f64, tiny: f64, out: usize,
) -> i32 {
    let values = take!(values, f64, FLOAT64, 0);
    let group = index_of!(group, 1);
    let root = take!(root, f64, FLOAT64, 2);
    let out = take!(out, f64, FLOAT64, 5);
    if group.len() < values.len || out.len < values.len || !inside(group, root.len) {
        return refused(1);
    }
    for e in 0..values.len {
        out.set(e, values.get(e) + share * root.get(group.get(e) as usize) + tiny);
    }
    0
}

/// `_plain`: each synapse's moments and step in one pass.
#[no_mangle]
pub unsafe extern "C" fn plain(
    raw: usize, velocity: usize, second: usize, momentum: f64, momentum_rest: f64,
    momentum_correction: f64, rho: f64, rho_rest: f64, rho_correction: f64, has_momentum: i32,
    has_normalize: i32, eta: f64, floor: f64, new_velocity: usize, new_second: usize, out: usize,
) -> i32 {
    let raw = take!(raw, f64, FLOAT64, 0);
    let velocity = take!(velocity, f64, FLOAT64, 1);
    let second = take!(second, f64, FLOAT64, 2);
    let new_velocity = take!(new_velocity, f64, FLOAT64, 13);
    let new_second = take!(new_second, f64, FLOAT64, 14);
    let out = take!(out, f64, FLOAT64, 15);
    let count = raw.len;
    if out.len < count
        || (has_momentum != 0 && (velocity.len < count || new_velocity.len < count))
        || (has_normalize != 0 && (second.len < count || new_second.len < count))
    {
        return refused(0);
    }
    for e in 0..count {
        let x = raw.get(e);
        let mut t = x;
        if has_momentum != 0 {
            let moving = momentum * velocity.get(e) + momentum_rest * x;
            new_velocity.set(e, moving);
            t = moving / momentum_correction;
        }
        if has_normalize != 0 {
            let held = rho * second.get(e) + rho_rest * (x * x);
            new_second.set(e, held);
            out.set(e, eta * (t / (sqrt(held / rho_correction) + floor)));
        } else {
            out.set(e, eta * t);
        }
    }
    0
}

/// `_moments`: the first pass of the projection-relative floor.
#[no_mangle]
pub unsafe extern "C" fn moments(
    raw: usize, velocity: usize, second: usize, momentum: f64, momentum_rest: f64,
    momentum_correction: f64, rho: f64, rho_rest: f64, rho_correction: f64, has_momentum: i32,
    new_velocity: usize, new_second: usize, term: usize, root: usize,
) -> i32 {
    let raw = take!(raw, f64, FLOAT64, 0);
    let velocity = take!(velocity, f64, FLOAT64, 1);
    let second = take!(second, f64, FLOAT64, 2);
    let new_velocity = take!(new_velocity, f64, FLOAT64, 10);
    let new_second = take!(new_second, f64, FLOAT64, 11);
    let term = take!(term, f64, FLOAT64, 12);
    let root = take!(root, f64, FLOAT64, 13);
    let count = raw.len;
    if second.len < count || new_second.len < count || term.len < count || root.len < count
        || (has_momentum != 0 && (velocity.len < count || new_velocity.len < count))
    {
        return refused(0);
    }
    for e in 0..count {
        let x = raw.get(e);
        let mut t = x;
        if has_momentum != 0 {
            let moving = momentum * velocity.get(e) + momentum_rest * x;
            new_velocity.set(e, moving);
            t = moving / momentum_correction;
        }
        let held = rho * second.get(e) + rho_rest * (x * x);
        new_second.set(e, held);
        root.set(e, sqrt(held / rho_correction));
        term.set(e, t);
    }
    0
}

/// `_step`: each step over its root plus its projection's share.
#[no_mangle]
pub unsafe extern "C" fn step(
    term: usize, root: usize, group: usize, group_root: usize, eta: f64, share: f64, tiny: f64,
    out: usize,
) -> i32 {
    let term = take!(term, f64, FLOAT64, 0);
    let root = take!(root, f64, FLOAT64, 1);
    let group = index_of!(group, 2);
    let group_root = take!(group_root, f64, FLOAT64, 3);
    let out = take!(out, f64, FLOAT64, 7);
    let count = term.len;
    if root.len < count || group.len() < count || out.len < count
        || !inside(group, group_root.len)
    {
        return refused(2);
    }
    for e in 0..count {
        let g = group.get(e) as usize;
        out.set(e, eta * (term.get(e) / (root.get(e) + share * group_root.get(g) + tiny)));
    }
    0
}

/// `_tied`: each synapse's tied step, bounded efficacy and effective weight; `report`
/// (int64, 3) receives how many of each are not finite.
#[no_mangle]
pub unsafe extern "C" fn tied(
    delta: usize, partner: usize, efficacy: usize, gain_pre: usize, cap: f64, out_delta: usize,
    out_scale: usize, out_weight: usize, report: usize,
) -> i32 {
    let delta = take!(delta, f64, FLOAT64, 0);
    let partner = index_of!(partner, 1);
    let efficacy = take!(efficacy, f64, FLOAT64, 2);
    let gain_pre = take!(gain_pre, f64, FLOAT64, 3);
    let out_delta = take!(out_delta, f64, FLOAT64, 5);
    let out_scale = take!(out_scale, f64, FLOAT64, 6);
    let out_weight = take!(out_weight, f64, FLOAT64, 7);
    let report = take!(report, i64, INT64, 8);
    let count = delta.len;
    if partner.len() < count || efficacy.len < count || gain_pre.len < count
        || out_delta.len < count || out_scale.len < count || out_weight.len < count
        || report.len < 3
    {
        return refused(1);
    }
    for e in 0..count {
        if partner.get(e) >= count as i64 {
            return refused(1);
        }
    }
    let (mut bad_delta, mut bad_scale, mut bad_weight) = (0i64, 0i64, 0i64);
    let big = 1.7976931348623157e308f64;
    for e in 0..count {
        let k = partner.get(e);
        let d = if k >= 0 { 0.5 * (delta.get(e) + delta.get(k as usize)) } else { delta.get(e) };
        out_delta.set(e, d);
        let s = pymin(pymax(efficacy.get(e) + d, -cap), cap);
        out_scale.set(e, s);
        let w = gain_pre.get(e) * s;
        out_weight.set(e, w);
        if !(d <= big && d >= -big) {
            bad_delta += 1;
        }
        if !(s <= big && s >= -big) {
            bad_scale += 1;
        }
        if !(w <= big && w >= -big) {
            bad_weight += 1;
        }
    }
    report.set(0, bad_delta);
    report.set(1, bad_scale);
    report.set(2, bad_weight);
    0
}

/// `_add_episode_terms`: a read's anchored and relative position terms added to `term`.
#[no_mangle]
pub unsafe extern "C" fn add_episode_terms(
    term: usize, anchored: usize, relative: usize, has_anchored: i32, has_relative: i32, told: usize,
    focus: usize, table: usize, last_bucket: i32, outside: i32, reach: i32, none: i32,
) -> i32 {
    let term = take!(term, f64, FLOAT64, 0);
    let anchored = take!(anchored, f64, FLOAT64, 1);
    let relative = take!(relative, f64, FLOAT64, 2);
    let told = index_of!(told, 5);
    let focus = index_of!(focus, 6);
    let table = index_of!(table, 7);
    let (rows, slots) = (term.n0, term.n1);
    let (last, outside, reach, none) =
        (last_bucket as i64, outside as i64, reach as i64, none as i64);
    if (has_anchored != 0 && (told.len() < rows || anchored.n0 < rows
        || last < 0 || last as usize >= table.len()))
        || (has_relative != 0 && (focus.len() < rows || relative.n0 < rows))
    {
        return refused(5);
    }
    // every bucket first, so a refusal leaves `term` as it was
    for pass in 0..2 {
        for b in 0..rows {
            if has_anchored != 0 {
                let t = told.get(b);
                for m in 0..slots {
                    let place = t - 1 - (slots as i64 - 1 - m as i64);
                    let bucket = if place < 0 {
                        outside
                    } else if place <= last {
                        table.get(place as usize)
                    } else {
                        table.get(last as usize)
                    };
                    if pass == 0 {
                        if bucket < 0 || bucket as usize >= anchored.n1 {
                            return refused(1);
                        }
                    } else {
                        term.put(b, m, term.at(b, m) + anchored.at(b, bucket as usize));
                    }
                }
            }
            if has_relative != 0 {
                let f = focus.get(b);
                for m in 0..slots {
                    let bucket = if f < 0 {
                        none
                    } else {
                        let mut offset = slots as i64 - 1 - m as i64 - f;
                        if reach < offset {
                            offset = reach;
                        }
                        (if -reach > offset { -reach } else { offset }) + reach
                    };
                    if pass == 0 {
                        if bucket < 0 || bucket as usize >= relative.n1 {
                            return refused(2);
                        }
                    } else {
                        term.put(b, m, term.at(b, m) + relative.at(b, bucket as usize));
                    }
                }
            }
        }
    }
    0
}

/// `_clear_choice`: each row's largest `s / temperature` into `choice` (int64); returns 1 where
/// it stands clear of the others in every row, finite and by more than 1e-9.
#[no_mangle]
pub unsafe extern "C" fn clear_choice(s: usize, temperature: f64, choice: usize) -> i32 {
    let s = take!(s, f64, FLOAT64, 0);
    let choice: View<i64> = take!(choice, i64, INT64, 2);
    let (rows, width) = (s.n0, s.n1);
    if choice.len < rows {
        return refused(2);
    }
    let mut clear = true;
    for b in 0..rows {
        let mut best = f64::NEG_INFINITY;
        let mut second = f64::NEG_INFINITY;
        let mut at = 0i64;
        for i in 0..width {
            let z = s.at(b, i) / temperature;
            if !(fabs(z) < f64::INFINITY) {
                clear = false;
            }
            if z > best {
                second = best;
                best = z;
                at = i as i64;
            } else if z > second {
                second = z;
            }
        }
        choice.set(b, at);
        if !(best - second > 1e-9) {
            clear = false;
        }
    }
    clear as i32
}
