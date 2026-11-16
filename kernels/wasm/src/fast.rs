//! The fast free solve, `_fast_solve` of `cadence/fused.py`: dt = 1, no mask, rows that stop
//! apart and an activation in plain arithmetic.

use crate::settle::{product, RECORD};
use crate::{fabs, refused, room, view, Scratch, View, FLOAT64, INT64, NO_ROOM};

macro_rules! take {
    ($id:expr, $t:ty, $num:expr, $k:expr) => {
        match view::<$t>($id, $num) {
            Some(x) => x,
            None => return refused($k),
        }
    };
}

/// A fast solve's rows stop within this once `tolerance` qualifies them (`_FAST_PATIENCE`).
const PATIENCE: usize = 96;

/// `_act_plain`: the activation with the exponential as a power series at 1/256 of the
/// argument, squared eight times, the argument held within +-36.
#[inline(always)]
fn act_plain(x: f64, slope: f64, threshold: f64, rest: f64, up: f64, down: f64) -> f64 {
    let mut z = -slope * (x - threshold);
    z = if z > 36.0 {
        36.0
    } else if z < -36.0 {
        -36.0
    } else {
        z
    };
    let y = z * 0.00390625;
    let mut e = 1.0
        + y * (1.0
            + y * (0.5
                + y * (0.16666666666666666
                    + y * (0.041666666666666664
                        + y * (0.008333333333333333
                            + y * (0.001388888888888889 + y * 0.0001984126984126984))))));
    e = e * e;
    e = e * e;
    e = e * e;
    e = e * e;
    e = e * e;
    e = e * e;
    e = e * e;
    e = e * e;
    let q = 1.0 / (1.0 + e) - rest;
    if q > 0.0 {
        q * up
    } else {
        q * down
    }
}

/// The kept length of a launch over every block of a layout (`_fast_solve`'s `_kept_room`).
#[no_mangle]
pub unsafe extern "C" fn kept_length_all(starts: usize, pair_post: usize, batch: i32, n: i32) -> i32 {
    let starts = take!(starts, i64, INT64, 0);
    let pair_post = take!(pair_post, i64, INT64, 1);
    let (batch, n) = (batch.max(0) as usize, n.max(0) as usize);
    let ranges = if starts.len > 0 { starts.len - 1 } else { 0 };
    let mut size = 0usize;
    for q in 0..pair_post.len {
        let b = pair_post.get(q) as usize;
        size += batch * (starts.get(b + 1) - starts.get(b)) as usize;
    }
    (3 * batch * n + size + ranges + RECORD) as i32
}

/// `_fast_solve`; `report` (int64, 4) receives the sweeps, the checks, the last chunk's sweeps
/// and the sweeps over the rows, `kept` (zeros, `kept_length_all` long) what the state keeps.
#[no_mangle]
pub unsafe extern "C" fn fast_solve(
    v: usize, s: usize, standing: usize, flat: usize, starts: usize, pair_pre: usize, pair_post: usize,
    offset: usize, slope: f64, threshold: f64, rest: f64, leak: f64, budget: i32, chunk: i32,
    tolerance: f64, settled: f64, out: usize, report: usize, kept: usize,
) -> i32 {
    let v = take!(v, f64, FLOAT64, 0);
    let s = take!(s, f64, FLOAT64, 1);
    let standing = take!(standing, f64, FLOAT64, 2);
    let flat = take!(flat, f64, FLOAT64, 3);
    let starts = take!(starts, i64, INT64, 4);
    let pair_pre = take!(pair_pre, i64, INT64, 5);
    let pair_post = take!(pair_post, i64, INT64, 6);
    let offset = take!(offset, i64, INT64, 7);
    let out = take!(out, f64, FLOAT64, 16);
    let report = take!(report, i64, INT64, 17);
    let kept = take!(kept, f64, FLOAT64, 18);
    let (batch, n) = (v.n0, v.n1);
    let ranges = if starts.len > 0 { starts.len - 1 } else { 0 };
    let pairs = pair_pre.len;
    if s.n0 != batch || s.n1 != n || standing.n0 != batch || standing.n1 != n {
        return refused(1);
    }
    if ranges == 0 || starts.get(ranges) as usize != n || offset.len != pairs + 1
        || pair_post.len != pairs || offset.get(pairs) as usize > flat.len
    {
        return refused(4);
    }
    if out.len != batch || report.len < 4 {
        return refused(16);
    }
    let start = |r: usize| starts.get(r) as usize;
    let mut size = 0usize;
    for q in 0..pairs {
        let b = pair_post.get(q) as usize;
        size += batch * (start(b + 1) - start(b));
    }
    let bn = batch * n;
    if kept.len != 3 * bn + size + ranges + RECORD {
        return refused(18);
    }
    let up = 1.0 / (1.0 - rest);
    let down = if rest > 0.0 { leak / rest } else { 0.0 };
    let mut scratch = match Scratch::new(
        room::<u8>(ranges) * 2 + room::<f64>(bn) * 5 + room::<f64>(batch * n.max(1))
            + room::<usize>(batch)
            + room::<u32>(n),
    ) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    let fed = scratch.take::<u8>(ranges);
    let mut widest = 1usize;
    for q in 0..pairs {
        let c = pair_post.get(q) as usize;
        *fed.add(c) = 1;
        let w = start(c + 1) - start(c);
        if w > widest {
            widest = w;
        }
    }
    let total = scratch.take::<f64>(bn);
    let work_v = scratch.take::<f64>(bn);
    let work_standing = scratch.take::<f64>(bn);
    let columns = scratch.take::<f64>(bn);
    let check = scratch.take::<f64>(bn);
    let product_room = scratch.take::<f64>(batch * n.max(1));
    let rows = scratch.take::<usize>(batch);
    let keep = scratch.take::<u32>(n);
    let _ = widest;
    let mut m = batch;
    for b in 0..batch {
        *rows.add(b) = b;
        for i in 0..n {
            *work_v.add(b * n + i) = v.at(b, i);
            *work_standing.add(b * n + i) = standing.at(b, i);
            *total.add(b * n + i) = 0.0;
        }
        for r in 0..ranges {
            let (a0, a1) = (start(r), start(r + 1));
            let at = batch * a0 + b * (a1 - a0);
            if *fed.add(r) != 0 {
                for i in 0..a1 - a0 {
                    *columns.add(at + i) = s.at(b, a0 + i);
                }
            } else {
                for i in 0..a1 - a0 {
                    let x = *work_standing.add(b * n + a0 + i);
                    *work_v.add(b * n + a0 + i) = x;
                    *columns.add(at + i) = act_plain(x, slope, threshold, rest, up, down);
                }
            }
        }
    }
    // the products of the ranges no block reaches, made once
    for q in 0..pairs {
        let (a, c) = (pair_pre.get(q) as usize, pair_post.get(q) as usize);
        if *fed.add(a) == 0 {
            let (a0, a1, b0, b1) = (start(a), start(a + 1), start(c), start(c + 1));
            let nb = b1 - b0;
            product(columns.add(batch * a0), flat.p.add(offset.get(q) as usize), product_room,
                    batch, a1 - a0, nb, keep);
            for b in 0..batch {
                for i in 0..nb {
                    *total.add(b * n + b0 + i) += *product_room.add(b * nb + i);
                }
            }
        }
    }
    let (mut used, mut checks, mut taken, mut row_sweeps) = (0usize, 0usize, 0usize, 0usize);
    let budget = budget.max(0) as usize;
    let chunk = chunk.max(0) as usize;
    let home = |h: usize, b: usize| {
        for i in 0..n {
            v.put(h, i, *work_v.add(b * n + i));
        }
        for r in 0..ranges {
            let a0 = start(r);
            let wr = start(r + 1) - a0;
            let at = batch * a0 + b * wr;
            for i in 0..wr {
                s.put(h, a0 + i, *columns.add(at + i));
            }
        }
    };
    loop {
        // the check at the rows still settling
        for b in 0..m {
            out.set(*rows.add(b), 0.0);
        }
        for r in 0..ranges {
            if *fed.add(r) == 0 {
                continue;
            }
            let (b0, b1) = (start(r), start(r + 1));
            let nb = b1 - b0;
            for b in 0..m {
                for i in 0..nb {
                    let k = b * n + b0 + i;
                    *check.add(b * nb + i) = *total.add(k) + *work_standing.add(k) - *work_v.add(k);
                }
            }
            for q in 0..pairs {
                let a = pair_pre.get(q) as usize;
                if pair_post.get(q) as usize == r && *fed.add(a) != 0 {
                    let (a0, a1) = (start(a), start(a + 1));
                    product(columns.add(batch * a0), flat.p.add(offset.get(q) as usize),
                            product_room, m, a1 - a0, nb, keep);
                    for k in 0..m * nb {
                        *check.add(k) += *product_room.add(k);
                    }
                }
            }
            for b in 0..m {
                let mut worst = out.get(*rows.add(b));
                for i in 0..nb {
                    let d = fabs(*check.add(b * nb + i));
                    if d > worst || d != d {
                        worst = d;
                    }
                }
                out.set(*rows.add(b), worst);
            }
        }
        checks += 1;
        let mut b = 0usize;
        while b < m {
            let error = out.get(*rows.add(b));
            if error <= settled || (error <= tolerance && used >= PATIENCE) || !(error <= 1e300) {
                // this row is done: its state goes home, the last active row takes its place
                let h = *rows.add(b);
                if !(out.get(h) <= 1e300) {
                    out.set(h, f64::INFINITY);
                }
                home(h, b);
                m -= 1;
                if b != m {
                    for r in 0..ranges {
                        let a0 = start(r);
                        let wr = start(r + 1) - a0;
                        let at = batch * a0 + b * wr;
                        let at2 = batch * a0 + m * wr;
                        for i in 0..wr {
                            *columns.add(at + i) = *columns.add(at2 + i);
                        }
                    }
                    for i in 0..n {
                        *work_v.add(b * n + i) = *work_v.add(m * n + i);
                        *work_standing.add(b * n + i) = *work_standing.add(m * n + i);
                        *total.add(b * n + i) = *total.add(m * n + i);
                    }
                    *rows.add(b) = *rows.add(m);
                }
            } else {
                b += 1;
            }
        }
        if m == 0 || used >= budget {
            break;
        }
        taken = if chunk < budget - used { chunk } else { budget - used };
        for _ in 0..taken {
            for r in 0..ranges {
                if *fed.add(r) == 0 {
                    continue;
                }
                let (b0, b1) = (start(r), start(r + 1));
                let nb = b1 - b0;
                for b in 0..m {
                    for i in 0..nb {
                        let k = b * n + b0 + i;
                        *work_v.add(k) = *total.add(k) + *work_standing.add(k);
                    }
                }
                for q in 0..pairs {
                    let a = pair_pre.get(q) as usize;
                    if pair_post.get(q) as usize == r && *fed.add(a) != 0 {
                        let (a0, a1) = (start(a), start(a + 1));
                        product(columns.add(batch * a0), flat.p.add(offset.get(q) as usize),
                                product_room, m, a1 - a0, nb, keep);
                        for b in 0..m {
                            for i in 0..nb {
                                *work_v.add(b * n + b0 + i) += *product_room.add(b * nb + i);
                            }
                        }
                    }
                }
            }
            for r in 0..ranges {
                if *fed.add(r) == 0 {
                    continue;
                }
                let (b0, b1) = (start(r), start(r + 1));
                let nb = b1 - b0;
                let base = batch * b0;
                for b in 0..m {
                    let at = base + b * nb;
                    for i in 0..nb {
                        *columns.add(at + i) =
                            act_plain(*work_v.add(b * n + b0 + i), slope, threshold, rest, up,
                                      down);
                    }
                }
            }
        }
        used += taken;
        row_sweeps += taken * m;
    }
    for b in 0..m {
        home(*rows.add(b), b);
    }
    // what the state keeps: its publication, every product stale
    let feeds = scratch_feeds(&mut scratch, ranges, pair_pre);
    for b in 0..batch {
        for r in 0..ranges {
            if *feeds.add(r) != 0 {
                let a0 = start(r);
                let na = start(r + 1) - a0;
                let base = batch * a0 + b * na;
                for i in 0..na {
                    kept.set(base + i, s.at(b, a0 + i));
                }
            }
        }
    }
    let stale = bn + size;
    for r in 0..ranges {
        kept.set(stale + r, 1.0);
    }
    let head = kept.len - RECORD;
    kept.set(head, 1.0);
    kept.set(head + 1, bn as f64);
    kept.set(head + 2, slope);
    kept.set(head + 3, threshold);
    kept.set(head + 4, rest);
    kept.set(head + 5, leak);
    let mut at = head - 2 * bn;
    for b in 0..batch {
        for i in 0..n {
            kept.set(at + i, v.at(b, i));
        }
        for i in 0..n {
            kept.set(at + bn + i, s.at(b, i));
        }
        at += n;
    }
    report.set(0, used as i64);
    report.set(1, checks as i64);
    report.set(2, taken as i64);
    report.set(3, row_sweeps as i64);
    0
}

unsafe fn scratch_feeds(scratch: &mut Scratch, ranges: usize, pair_pre: View<i64>) -> *mut u8 {
    let feeds = scratch.take::<u8>(ranges);
    for q in 0..pair_pre.len {
        *feeds.add(pair_pre.get(q) as usize) = 1;
    }
    feeds
}
