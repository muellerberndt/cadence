//! The settle, the residual and the free solve: `_settle_rows`, `_residual_rows`,
//! `_solve_rows`, `_publish` and `rows_finite` of `cadence/fused.py`, in its order.

use core::arch::wasm32::*;

use crate::{
    exp, fabs, finite, refused, room, sqrt, view, Scratch, View, BOOL, FLOAT64, INT64, NO_ROOM,
};

/// The entries of the record at the end of a state's kept array (`_keep_published`)
pub(crate) const RECORD: usize = 6;

macro_rules! take {
    ($id:expr, $t:ty, $num:expr, $k:expr) => {
        match view::<$t>($id, $num) {
            Some(x) => x,
            None => return refused($k),
        }
    };
}

// -- the neuron rule

#[inline(always)]
pub(crate) fn act_scalar(x: f64, slope: f64, threshold: f64, rest: f64, leak: f64) -> f64 {
    let r = 1.0 / (1.0 + exp(-slope * (x - threshold))) - rest;
    if r > 0.0 {
        return r * (1.0 / (1.0 - rest));
    }
    if leak == 0.0 {
        return 0.0;
    }
    r * (leak / rest)
}

/// `_softmax_groups`: the softmax of each nudged group of row `s` into `p`.
unsafe fn softmax_groups(
    s: *const f64, group: *const usize, groups: usize, gid: View<i64>, ngroups: usize,
    temperatures: View<f64>, p: *mut f64, zmax: *mut f64, total: *mut f64,
) {
    for g in 0..ngroups {
        *zmax.add(g) = f64::NEG_INFINITY;
        *total.add(g) = 0.0;
    }
    let mut all_finite = true;
    for k in 0..groups {
        let i = *group.add(k);
        let z = *s.add(i) / temperatures.get(i);
        *p.add(k) = z;
        all_finite = all_finite && finite(z);
        let g = gid.get(i) as usize;
        if z > *zmax.add(g) {
            *zmax.add(g) = z;
        }
    }
    if !all_finite {
        for g in 0..ngroups {
            *zmax.add(g) = f64::NEG_INFINITY;
        }
        for k in 0..groups {
            let i = *group.add(k);
            let g = gid.get(i) as usize;
            if *s.add(i) > *zmax.add(g) {
                *zmax.add(g) = *s.add(i);
            }
        }
        for k in 0..groups {
            let i = *group.add(k);
            let g = gid.get(i) as usize;
            *p.add(k) = (*s.add(i) - *zmax.add(g)) / temperatures.get(i);
        }
        for g in 0..ngroups {
            *zmax.add(g) = 0.0;
        }
    }
    for k in 0..groups {
        let g = gid.get(*group.add(k)) as usize;
        *p.add(k) = exp(*p.add(k) - *zmax.add(g));
        *total.add(g) += *p.add(k);
    }
    for k in 0..groups {
        let g = gid.get(*group.add(k)) as usize;
        *p.add(k) /= *total.add(g);
    }
}

// -- the block transport

/// `out[b, j] = sum_k src[b * na + k] * w[k * nb + j]` for the first `m` rows, each sum from
/// 0.0 in the order of `k`: NumPy's matrix product without BLAS, two outputs to a lane. A source
/// entry of exactly zero adds a zero to every sum it reaches, which leaves a sum begun at 0.0
/// as it was (it is never -0.0), so its row of the block is skipped: the weights of a graph are
/// finite (``Brain`` refuses others). `keep` holds room for `na` indices.
pub(crate) unsafe fn product(
    src: *const f64, w: *const f64, out: *mut f64, m: usize, na: usize, nb: usize,
    keep: *mut u32,
) {
    for b in 0..m {
        let x = src.add(b * na);
        let o = out.add(b * nb);
        let mut count = 0usize;
        for k in 0..na {
            if *x.add(k) != 0.0 {
                *keep.add(count) = k as u32;
                count += 1;
            }
        }
        let mut j = 0usize;
        while j + 8 <= nb {
            let mut c0 = f64x2_splat(0.0);
            let mut c1 = f64x2_splat(0.0);
            let mut c2 = f64x2_splat(0.0);
            let mut c3 = f64x2_splat(0.0);
            for q in 0..count {
                let k = *keep.add(q) as usize;
                let a = f64x2_splat(*x.add(k));
                let row = w.add(k * nb + j);
                c0 = f64x2_add(c0, f64x2_mul(a, v128_load(row as *const v128)));
                c1 = f64x2_add(c1, f64x2_mul(a, v128_load(row.add(2) as *const v128)));
                c2 = f64x2_add(c2, f64x2_mul(a, v128_load(row.add(4) as *const v128)));
                c3 = f64x2_add(c3, f64x2_mul(a, v128_load(row.add(6) as *const v128)));
            }
            v128_store(o.add(j) as *mut v128, c0);
            v128_store(o.add(j + 2) as *mut v128, c1);
            v128_store(o.add(j + 4) as *mut v128, c2);
            v128_store(o.add(j + 6) as *mut v128, c3);
            j += 8;
        }
        while j + 2 <= nb {
            let mut c0 = f64x2_splat(0.0);
            for q in 0..count {
                let k = *keep.add(q) as usize;
                let a = f64x2_splat(*x.add(k));
                c0 = f64x2_add(c0, f64x2_mul(a, v128_load(w.add(k * nb + j) as *const v128)));
            }
            v128_store(o.add(j) as *mut v128, c0);
            j += 2;
        }
        if j < nb {
            let mut c = 0.0f64;
            for q in 0..count {
                let k = *keep.add(q) as usize;
                c += *x.add(k) * *w.add(k * nb + j);
            }
            *o.add(j) = c;
        }
    }
}

/// What every launch of a graph reads: the state, the blocks, the rule, the nudge and the
/// compartment.
#[derive(Clone, Copy)]
pub(crate) struct Net {
    pub batch: usize,
    pub n: usize,
    pub ranges: usize,
    pub v: View<f64>,
    pub a: View<f64>,
    pub s: View<f64>,
    pub standing: View<f64>,
    pub flat: View<f64>,
    pub pairs: View<i64>,
    pub flat_apical: View<f64>,
    pub apical_pairs: View<i64>,
    pub starts: View<i64>,
    pub pair_pre: View<i64>,
    pub pair_post: View<i64>,
    pub offset: View<i64>,
    pub keep: View<f64>,
    pub masked: bool,
    pub dt: f64,
    pub slope: f64,
    pub threshold: f64,
    pub rest: f64,
    pub leak: f64,
    pub has_adapt: bool,
    pub strength: f64,
    pub tau: f64,
    pub has_nudge: bool,
    pub target: View<f64>,
    pub nmask: View<f64>,
    pub gid: View<i64>,
    pub ngroups: usize,
    pub beta: f64,
    pub softmax_t: View<f64>,
    pub weight: View<f64>,
    pub anchor: View<f64>,
    pub anchor_gain: View<f64>,
    pub has_apical: bool,
    pub pool_of: View<i64>,
    pub pool_index: View<i64>,
    pub pool_starts: View<i64>,
    pub pool_run: View<i64>,
    pub fraction: f64,
    pub floor2: f64,
    pub tiny: f64,
    pub held: View<f64>,
    pub has_held: bool,
    /// each row a life of its own: `flat` and `flat_apical` hold one row of blocks a life
    pub per_life: bool,
}

impl Net {
    #[inline(always)]
    unsafe fn start(&self, r: usize) -> usize {
        self.starts.get(r) as usize
    }
    fn pools(&self) -> usize {
        if self.pool_starts.len > 0 {
            self.pool_starts.len - 1
        } else {
            0
        }
    }
    /// Entries of the products of `pairs` over the batch.
    unsafe fn product_size(&self, pairs: View<i64>) -> usize {
        let mut size = 0usize;
        for q in 0..pairs.len {
            let b = self.pair_post.get(pairs.get(q) as usize) as usize;
            size += self.batch * (self.start(b + 1) - self.start(b));
        }
        size
    }
    /// The length of a kept array (`_kept_room`).
    pub unsafe fn kept_len(&self) -> usize {
        let bn = self.batch * self.n;
        bn + self.product_size(self.pairs) + self.product_size(self.apical_pairs) + self.ranges
            + 2 * bn
            + RECORD
    }
    /// Whether the softmax nudge applies.
    #[inline(always)]
    unsafe fn softmax(&self) -> bool {
        self.has_nudge && self.softmax_t.get(0) > 0.0
    }
}

/// The offsets of one cached product per block of `pairs`, and the ranges a block lands on
/// (`_product_cache`).
unsafe fn product_cache(net: &Net, pairs: View<i64>, offset: *mut usize, receives: *mut u8) {
    *offset = 0;
    for q in 0..pairs.len {
        let b = net.pair_post.get(pairs.get(q) as usize) as usize;
        *offset.add(q + 1) = *offset.add(q) + net.batch * (net.start(b + 1) - net.start(b));
        *receives.add(b) = 1;
    }
}

/// Row `b`'s activations into `columns`, range by range, for the ranges `feeds` names.
#[inline(always)]
unsafe fn copy_columns(net: &Net, b: usize, s: View<f64>, columns: *mut f64, feeds: *const u8) {
    let batch = net.batch;
    for r in 0..net.ranges {
        if *feeds.add(r) != 0 {
            let a0 = net.start(r);
            let na = net.start(r + 1) - a0;
            let base = batch * a0 + b * na;
            let row = s.p.add(b * net.n + a0);
            for i in 0..na {
                *columns.add(base + i) = *row.add(i);
            }
        }
    }
}

/// `_reusable`: take the given products that hold for `columns`.
unsafe fn reusable(
    net: &Net, columns: *const f64, columns_len: usize, kept: *mut f64, given: View<f64>,
    feeds: *const u8, moved_range: *mut u8,
) {
    let batch = net.batch;
    let ranges = net.ranges;
    let flags = given.len - RECORD - 2 * columns_len - ranges;
    for r in 0..ranges {
        let mut held = *feeds.add(r) != 0 && given.get(flags + r) == 0.0;
        let mut i = batch * net.start(r);
        let stop = batch * net.start(r + 1);
        while held && i < stop {
            held = *columns.add(i) == given.get(i);
            i += 1;
        }
        *moved_range.add(r) = if held { 0 } else { 1 };
    }
    for i in columns_len..flags {
        *kept.add(i) = given.get(i);
    }
}

/// Block `k`'s product for the whole batch into `cache` at `c0`.
#[inline(always)]
unsafe fn block_product(
    net: &Net, columns: *const f64, flat: View<f64>, k: usize, cache: *mut f64, c0: usize,
    keep: *mut u32,
) {
    let a = net.pair_pre.get(k) as usize;
    let b = net.pair_post.get(k) as usize;
    let a0 = net.start(a);
    let na = net.start(a + 1) - a0;
    let nb = net.start(b + 1) - net.start(b);
    let batch = net.batch;
    if net.per_life {  // each row through its own life's block
        for row in 0..batch {
            product(
                columns.add(batch * a0 + row * na),
                flat.p.add(row * flat.n1 + net.offset.get(k) as usize),
                cache.add(c0 + row * nb),
                1,
                na,
                nb,
                keep,
            );
        }
        return;
    }
    product(
        columns.add(batch * a0),
        flat.p.add(net.offset.get(k) as usize),
        cache.add(c0),
        batch,
        na,
        nb,
        keep,
    );
}

/// Row `b` of the synaptic input of `pairs` (`_gather`): zero where a block lands, then every
/// block's cached product added in block order.
#[inline(always)]
unsafe fn gather(
    net: &Net, b: usize, out: *mut f64, cache: *const f64, cache_offset: *const usize,
    pairs: View<i64>, receives: *const u8,
) {
    let row = out.add(b * net.n);
    for r in 0..net.ranges {
        if *receives.add(r) != 0 {
            for i in net.start(r)..net.start(r + 1) {
                *row.add(i) = 0.0;
            }
        }
    }
    for q in 0..pairs.len {
        let post = net.pair_post.get(pairs.get(q) as usize) as usize;
        let b0 = net.start(post);
        let nb = net.start(post + 1) - b0;
        let c = cache.add(*cache_offset.add(q) + b * nb);
        let o = row.add(b0);
        for j in 0..nb {
            *o.add(j) += *c.add(j);
        }
    }
}

// -- the apical compartment's terms

/// `sum(x[lo:lo + count] ** 2)` for `count <= 128`, as NumPy's pairwise sum adds a block.
unsafe fn square_block(x: *const f64, lo: usize, count: usize) -> f64 {
    let mut res = 0.0f64;
    if count < 8 {
        for i in lo..lo + count {
            let y = *x.add(i);
            res += y * y;
        }
        return res;
    }
    let mut r = [0.0f64; 8];
    for (j, slot) in r.iter_mut().enumerate() {
        let y = *x.add(lo + j);
        *slot = y * y;
    }
    let mut i = 8usize;
    let stop = count - count % 8;
    while i < stop {
        for (j, slot) in r.iter_mut().enumerate() {
            let y = *x.add(lo + i + j);
            *slot += y * y;
        }
        i += 8;
    }
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while i < count {
        let y = *x.add(lo + i);
        res += y * y;
        i += 1;
    }
    res
}

/// `sum(x[lo:lo + count] ** 2)` in NumPy's pairwise order: a run of more than 128 splits at
/// half its length rounded down to a multiple of eight.
unsafe fn square_sum(x: *const f64, lo: usize, count: usize) -> f64 {
    if count <= 128 {
        return square_block(x, lo, count);
    }
    let mut half = count / 2;
    half -= half % 8;
    square_sum(x, lo, half) + square_sum(x, lo + half, count - half)
}

/// Row `b`'s gain per pool (`_pool_gains`).
unsafe fn pool_gains(
    net: &Net, b: usize, basal: *const f64, raw: *const f64, gain: *mut f64, values: *mut f64,
) {
    let pools = net.pools();
    for k in 0..pools {
        if net.has_held {
            *gain.add(k) = net.held.at(b, k);
            continue;
        }
        let lo = net.pool_starts.get(k) as usize;
        let m = net.pool_starts.get(k + 1) as usize - lo;
        let first = net.pool_run.get(k);
        let (basal2, raw2);
        if first >= 0 {
            basal2 = square_sum(basal, first as usize, m) / m as f64;
            raw2 = square_sum(raw, first as usize, m) / m as f64;
        } else {
            for j in 0..m {
                *values.add(j) = *basal.add(net.pool_index.get(lo + j) as usize);
            }
            basal2 = square_sum(values, 0, m) / m as f64;
            for j in 0..m {
                *values.add(j) = *raw.add(net.pool_index.get(lo + j) as usize);
            }
            raw2 = square_sum(values, 0, m) / m as f64;
        }
        *gain.add(k) = net.fraction * sqrt(basal2) / sqrt(raw2 + net.floor2 * basal2 + net.tiny);
    }
}

/// The basal input of every neuron of row `b`, as `_sweep_row_apical` and
/// `_residual_row_apical` sum it.
unsafe fn basal_input(
    net: &Net, b: usize, synaptic_input: *const f64, p: *const f64, position: *const isize,
    basal: *mut f64,
) {
    let n = net.n;
    let softmax = net.softmax();
    let s = net.s;
    for i in 0..n {
        let mut x = *synaptic_input.add(b * n + i) + net.standing.at(b, i);
        if net.has_adapt {
            x -= net.strength * net.a.at(b, i);
        }
        if net.has_nudge {
            let mut nd = 0.0f64;
            let mut pulled = false;
            if net.nmask.get(i) > 0.0 {
                if softmax {
                    nd = net.beta * (net.target.at(b, i) - *p.add(*position.add(i) as usize))
                        * net.nmask.get(i);
                } else {
                    nd = net.beta * (net.target.at(b, i) - s.at(b, i)) * net.nmask.get(i);
                }
                nd *= net.weight.get(b);
                pulled = true;
            }
            if net.anchor_gain.get(i) > 0.0 {
                nd += net.anchor_gain.get(i) * (net.anchor.at(b, i) - s.at(b, i));
                pulled = true;
            }
            if pulled {
                x += nd;
            }
        }
        *basal.add(i) = x;
    }
}

// -- one sweep of one row

/// Range `r` of row `b` with no nudge, adaptation or mask on it (`_sweep_range`). Returns the
/// running change, the largest movement, and whether the range's activation and potential
/// moved.
#[inline(always)]
unsafe fn sweep_range(
    net: &Net, b: usize, r: usize, synaptic_input: *const f64, ahead: *mut f64,
    mut change: f64, mut moved: f64,
) -> (f64, f64, bool, bool) {
    let n = net.n;
    let (v, s, standing) = (net.v.p.add(b * n), net.s.p.add(b * n), net.standing.p.add(b * n));
    let syn = synaptic_input.add(b * n);
    let ahead = ahead.add(b * n);
    let dt = net.dt;
    let mut range_moved = false;
    let mut potential_moved = false;
    let (first, stop) = (net.start(r), net.start(r + 1));
    for i in first..stop {
        let x = *v.add(i);
        *ahead.add(i) = x + dt * (*syn.add(i) + *standing.add(i) - x);
    }
    for i in first..stop {
        let vn = *ahead.add(i);
        if vn != *v.add(i) {
            potential_moved = true;
            *v.add(i) = vn;
            let sn = act_scalar(vn, net.slope, net.threshold, net.rest, net.leak);
            let d = fabs(sn - *s.add(i));
            change += d;
            if d > 0.0 {
                range_moved = true;
                if d > moved {
                    moved = d;
                }
            }
            *s.add(i) = sn;
        }
    }
    (change, moved, range_moved, potential_moved)
}

/// Every neuron of row `b` takes one step (`_sweep_row`, and `_sweep_row_plain` where nothing
/// but synapses and the drive reach a range). Returns the row's largest movement.
unsafe fn sweep_row(
    net: &Net, b: usize, synaptic_input: *const f64, ahead: *mut f64, frozen: *const u8,
    quiet: *const u8, plain: bool, p: *mut f64, zmax: *mut f64, total: *mut f64,
    group: *const usize, groups: usize, position: *const isize, row_range: *mut u8,
    row_potential: *mut u8, activity_change: View<f64>,
) -> f64 {
    let n = net.n;
    let (v, a, s) = (net.v, net.a, net.s);
    let mut moved = 0.0f64;
    let mut change = activity_change.get(b);
    let softmax = net.softmax();
    if softmax {
        softmax_groups(s.p.add(b * n), group, groups, net.gid, net.ngroups, net.softmax_t, p,
                       zmax, total);
    }
    let dt = net.dt;
    for r in 0..net.ranges {
        let mut range_moved = false;
        let mut potential_moved = false;
        let is_frozen = *frozen.add(r) != 0;
        let is_quiet = plain || *quiet.add(r) != 0;
        if is_quiet && !is_frozen {
            let got = sweep_range(net, b, r, synaptic_input, ahead, change, moved);
            change = got.0;
            moved = got.1;
            range_moved = got.2;
            potential_moved = got.3;
        }
        let stop = if is_frozen || is_quiet { net.start(r) } else { net.start(r + 1) };
        for i in net.start(r)..stop {
            let mut tot = *synaptic_input.add(b * n + i) + net.standing.at(b, i);
            if net.has_adapt {
                tot -= net.strength * a.at(b, i);
            }
            if net.has_nudge && net.nmask.get(i) > 0.0 && net.softmax_t.get(0) <= 0.0 {
                tot += net.beta * net.weight.get(b) * net.nmask.get(i)
                    * (net.target.at(b, i) - s.at(b, i));
            }
            if net.anchor_gain.get(i) > 0.0 {
                tot += net.anchor_gain.get(i) * (net.anchor.at(b, i) - s.at(b, i));
            }
            tot -= v.at(b, i);
            let mut vn = v.at(b, i) + dt * tot;
            if softmax && net.nmask.get(i) > 0.0 {
                vn += dt * net.beta * net.weight.get(b)
                    * (net.target.at(b, i) - *p.add(*position.add(i) as usize))
                    * net.nmask.get(i);
            }
            if net.masked {
                vn *= net.keep.at(b, i);
            }
            if vn != v.at(b, i) {
                potential_moved = true;
                v.put(b, i, vn);
                let mut sn = act_scalar(vn, net.slope, net.threshold, net.rest, net.leak);
                if net.masked {
                    sn *= net.keep.at(b, i);
                }
                let d = fabs(sn - s.at(b, i));
                change += d;
                if d > 0.0 {
                    range_moved = true;
                    if d > moved {
                        moved = d;
                    }
                }
                s.put(b, i, sn);
            }
            if net.has_adapt {
                a.put(b, i, a.at(b, i) + (s.at(b, i) - a.at(b, i)) / net.tau);
            }
        }
        *row_range.add(r) = range_moved as u8;
        *row_potential.add(r) = potential_moved as u8;
    }
    activity_change.set(b, change);
    moved
}

/// `_sweep_row_apical`: every neuron's basal input, the pools' gains from it, then each
/// neuron's step on its basal input plus its pool's gain times its apical input.
unsafe fn sweep_row_apical(
    net: &Net, b: usize, synaptic_input: *const f64, raw: *const f64, frozen: *const u8,
    p: *mut f64, zmax: *mut f64, total: *mut f64, group: *const usize, groups: usize,
    position: *const isize, basal: *mut f64, gain: *mut f64, values: *mut f64,
    row_range: *mut u8, row_potential: *mut u8, activity_change: View<f64>,
) -> f64 {
    let n = net.n;
    let (v, a, s) = (net.v, net.a, net.s);
    let mut moved = 0.0f64;
    let mut change = activity_change.get(b);
    if net.softmax() {
        softmax_groups(s.p.add(b * n), group, groups, net.gid, net.ngroups, net.softmax_t, p,
                       zmax, total);
    }
    basal_input(net, b, synaptic_input, p, position, basal);
    pool_gains(net, b, basal, raw.add(b * n), gain, values);
    let dt = net.dt;
    for r in 0..net.ranges {
        let mut range_moved = false;
        let mut potential_moved = false;
        let stop = if *frozen.add(r) != 0 { net.start(r) } else { net.start(r + 1) };
        for i in net.start(r)..stop {
            let mut x = *basal.add(i);
            let k = net.pool_of.get(i);
            if k >= 0 {
                x += *gain.add(k as usize) * *raw.add(b * n + i);
            }
            let mut vn = v.at(b, i) + (x - v.at(b, i)) * dt;
            if net.masked {
                vn *= net.keep.at(b, i);
            }
            if vn != v.at(b, i) {
                potential_moved = true;
                v.put(b, i, vn);
                let mut sn = act_scalar(vn, net.slope, net.threshold, net.rest, net.leak);
                if net.masked {
                    sn *= net.keep.at(b, i);
                }
                let d = fabs(sn - s.at(b, i));
                change += d;
                if d > 0.0 {
                    range_moved = true;
                    if d > moved {
                        moved = d;
                    }
                }
                s.put(b, i, sn);
            }
            if net.has_adapt {
                a.put(b, i, a.at(b, i) + (s.at(b, i) - a.at(b, i)) / net.tau);
            }
        }
        *row_range.add(r) = range_moved as u8;
        *row_potential.add(r) = potential_moved as u8;
    }
    activity_change.set(b, change);
    moved
}

/// The nudged group and each neuron's place in it.
unsafe fn nudged_group(net: &Net, scratch: &mut Scratch) -> (*mut usize, usize, *mut isize) {
    let n = net.n;
    let mut groups = 0usize;
    for i in 0..n {
        if net.nmask.get(i) > 0.0 {
            groups += 1;
        }
    }
    let group = scratch.take::<usize>(groups.max(1));
    let position = scratch.take::<isize>(n);
    let mut k = 0usize;
    for i in 0..n {
        *position.add(i) = -1;
        if net.nmask.get(i) > 0.0 {
            *group.add(k) = i;
            *position.add(i) = k as isize;
            k += 1;
        }
    }
    (group, groups, position)
}

fn group_room(net: &Net) -> usize {
    room::<usize>(net.n.max(1)) + room::<isize>(net.n)
}

/// `_keep_published`: the potentials and activations at the end of `kept`, then the record.
unsafe fn keep_published(net: &Net, kept: View<f64>) {
    let (batch, n) = (net.batch, net.n);
    let count = batch * n;
    let head = kept.len - RECORD;
    kept.set(head, if net.masked { 0.0 } else { 1.0 });
    kept.set(head + 1, count as f64);
    kept.set(head + 2, net.slope);
    kept.set(head + 3, net.threshold);
    kept.set(head + 4, net.rest);
    kept.set(head + 5, net.leak);
    if net.masked {
        return;
    }
    let mut at = head - 2 * count;
    for b in 0..batch {
        for i in 0..n {
            kept.set(at + i, net.v.at(b, i));
        }
        for i in 0..n {
            kept.set(at + count + i, net.s.at(b, i));
        }
        at += n;
    }
}

/// The settle (`_settle_rows`), its last state's products into `kept` (zeros, `kept_len`
/// long). Returns the sweeps taken.
pub(crate) unsafe fn settle(
    net: &Net, freezable: View<u8>, steps: usize, tolerance: f64, use_tolerance: bool,
    given: View<f64>, has_given: bool, activity_change: View<f64>, kept: View<f64>,
) -> i32 {
    let (batch, n, ranges) = (net.batch, net.n, net.ranges);
    let bn = batch * n;
    let (np_, nq) = (net.pairs.len, net.apical_pairs.len);
    let pools = net.pools().max(1);
    let wide = if net.has_apical { bn } else { 1 };
    let ng = net.ngroups.max(1);
    let bytes = group_room(net)
        + 2 * room::<usize>(np_.max(nq) + 1)
        + 2 * room::<u8>(ranges)
        + room::<u8>(ranges) // feeds
        + room::<usize>(np_ + nq) // todo
        + room::<u8>(ranges) // moved_range
        + 2 * room::<f64>(bn) // synaptic input, ahead
        + 3 * room::<f64>(wide) // raw, basal, values
        + room::<f64>(batch * pools)
        + room::<f64>(batch * n.max(1)) // p
        + 2 * room::<f64>(batch * ng)
        + room::<f64>(batch)
        + 2 * room::<u8>(batch * ranges)
        + 4 * room::<u8>(ranges)
        + room::<u32>(n);
    let mut scratch = match Scratch::new(bytes) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    let (group, groups, position) = nudged_group(net, &mut scratch);
    let cache_offset = scratch.take::<usize>(np_ + 1);
    let receives = scratch.take::<u8>(ranges);
    let cache_up_offset = scratch.take::<usize>(nq + 1);
    let receives_up = scratch.take::<u8>(ranges);
    product_cache(net, net.pairs, cache_offset, receives);
    product_cache(net, net.apical_pairs, cache_up_offset, receives_up);
    let size = *cache_offset.add(np_);
    let size_up = *cache_up_offset.add(nq);
    let columns = kept.p;
    let cache = kept.p.add(bn);
    let cache_up = kept.p.add(bn + size);
    let stale = kept.p.add(bn + size + size_up);
    let feeds = scratch.take::<u8>(ranges);
    for q in 0..np_ + nq {
        let k = if q < np_ { net.pairs.get(q) } else { net.apical_pairs.get(q - np_) } as usize;
        *feeds.add(net.pair_pre.get(k) as usize) = 1;
    }
    let todo = scratch.take::<usize>(np_ + nq);
    for b in 0..batch {
        copy_columns(net, b, net.s, columns, feeds);
    }
    let moved_range = scratch.take::<u8>(ranges);
    for r in 0..ranges {
        *moved_range.add(r) = 1;
    }
    if has_given {
        reusable(net, columns, bn, kept.p, given, feeds, moved_range);
    }
    let synaptic_input = scratch.take::<f64>(bn);
    let ahead = scratch.take::<f64>(bn);
    let raw = scratch.take::<f64>(wide);
    let basal = scratch.take::<f64>(wide);
    let values = scratch.take::<f64>(wide);
    let gain = scratch.take::<f64>(batch * pools);
    let p = scratch.take::<f64>(batch * n.max(1));
    let zmax = scratch.take::<f64>(batch * ng);
    let total = scratch.take::<f64>(batch * ng);
    let row_moved = scratch.take::<f64>(batch);
    let row_range = scratch.take::<u8>(batch * ranges);
    let row_potential = scratch.take::<u8>(batch * ranges);
    let frozen = scratch.take::<u8>(ranges);
    let moving = scratch.take::<u8>(ranges);
    let quiet = scratch.take::<u8>(ranges);
    let _spare = scratch.take::<u8>(ranges);
    let keep = scratch.take::<u32>(n);
    let plain = !(net.has_nudge || net.has_adapt || net.masked || net.has_apical);
    if !(net.has_adapt || net.masked || plain) {
        for r in 0..ranges {
            let mut q = 1u8;
            for i in net.start(r)..net.start(r + 1) {
                if net.nmask.get(i) > 0.0 || net.anchor_gain.get(i) > 0.0 {
                    q = 0;
                }
            }
            *quiet.add(r) = q;
        }
    }
    let mut taken = 0usize;
    for t in 0..steps {
        // the products of the blocks whose source range moved (every block at first)
        let first = t == 0 && !has_given;
        let mut due = 0usize;
        for q in 0..np_ + nq {
            let k = if q < np_ { net.pairs.get(q) } else { net.apical_pairs.get(q - np_) }
                as usize;
            if first || *moved_range.add(net.pair_pre.get(k) as usize) != 0 {
                *todo.add(due) = q;
                due += 1;
            }
        }
        for j in 0..due {
            let q = *todo.add(j);
            if q < np_ {
                block_product(net, columns, net.flat, net.pairs.get(q) as usize, cache,
                              *cache_offset.add(q), keep);
            } else {
                block_product(net, columns, net.flat_apical, net.apical_pairs.get(q - np_)
                              as usize, cache_up, *cache_up_offset.add(q - np_), keep);
            }
        }
        for r in 0..ranges {
            *moving.add(r) = (*feeds.add(r) != 0 && *frozen.add(r) == 0) as u8;
        }
        for b in 0..batch {
            gather(net, b, synaptic_input, cache, cache_offset, net.pairs, receives);
            let rr = row_range.add(b * ranges);
            let rp = row_potential.add(b * ranges);
            let moved = if net.has_apical {
                gather(net, b, raw, cache_up, cache_up_offset, net.apical_pairs, receives_up);
                sweep_row_apical(net, b, synaptic_input, raw, frozen, p.add(b * n),
                                 zmax.add(b * ng), total.add(b * ng), group, groups, position,
                                 basal.add(b * n), gain.add(b * pools), values.add(b * n), rr, rp,
                                 activity_change)
            } else {
                sweep_row(net, b, synaptic_input, ahead, frozen, quiet, plain, p.add(b * n),
                          zmax.add(b * ng), total.add(b * ng), group, groups, position, rr, rp,
                          activity_change)
            };
            *row_moved.add(b) = moved;
            copy_columns(net, b, net.s, columns, moving);
        }
        // the batch's reading of the sweep (`_close_sweep`)
        let mut moved = 0.0f64;
        for b in 0..batch {
            if *row_moved.add(b) > moved {
                moved = *row_moved.add(b);
            }
        }
        for r in 0..ranges {
            let mut range_moved = false;
            let mut potential_moved = false;
            for b in 0..batch {
                range_moved = range_moved || *row_range.add(b * ranges + r) != 0;
                potential_moved = potential_moved || *row_potential.add(b * ranges + r) != 0;
            }
            *moved_range.add(r) = range_moved as u8;
            if freezable.get(r) != 0 && !potential_moved && t > 0 {
                *frozen.add(r) = 1;
            }
        }
        taken = t + 1;
        if use_tolerance && moved < tolerance {
            break;
        }
    }
    for r in 0..ranges {
        *stale.add(r) = if *moved_range.add(r) != 0 { 1.0 } else { 0.0 };
    }
    keep_published(net, kept);
    taken as i32
}

/// The equation error of every row into `out` (`_residual_rows`): one transport at `s`, the
/// given products where they hold. With `in_place` the given array is the state's own and is
/// made current in place; otherwise `kept` (zeros, `kept_len` long) receives what the state
/// keeps.
pub(crate) unsafe fn residual(
    net: &Net, out: View<f64>, given: View<f64>, has_given: bool, in_place: bool,
    kept_out: View<f64>,
) -> i32 {
    let (batch, n, ranges) = (net.batch, net.n, net.ranges);
    let bn = batch * n;
    let (np_, nq) = (net.pairs.len, net.apical_pairs.len);
    let pools = net.pools().max(1);
    let wide = if net.has_apical { bn } else { 1 };
    let ng = net.ngroups.max(1);
    let bytes = group_room(net)
        + 2 * room::<usize>(np_.max(nq) + 1)
        + 3 * room::<u8>(ranges)
        + room::<u8>(ranges)
        + room::<f64>(bn)
        + 3 * room::<f64>(wide)
        + room::<f64>(batch * pools)
        + room::<f64>(batch * n.max(1))
        + 2 * room::<f64>(batch * ng)
        + room::<u32>(n);
    let mut scratch = match Scratch::new(bytes) {
        Some(x) => x,
        None => return NO_ROOM,
    };
    let (group, groups, position) = nudged_group(net, &mut scratch);
    let cache_offset = scratch.take::<usize>(np_ + 1);
    let receives = scratch.take::<u8>(ranges);
    let cache_up_offset = scratch.take::<usize>(nq + 1);
    let receives_up = scratch.take::<u8>(ranges);
    product_cache(net, net.pairs, cache_offset, receives);
    product_cache(net, net.apical_pairs, cache_up_offset, receives_up);
    let size = *cache_offset.add(np_);
    let size_up = *cache_up_offset.add(nq);
    let keep = scratch.take::<u32>(n);
    let feeds = scratch.take::<u8>(ranges);
    for q in 0..np_ + nq {
        let k = if q < np_ { net.pairs.get(q) } else { net.apical_pairs.get(q - np_) } as usize;
        *feeds.add(net.pair_pre.get(k) as usize) = 1;
    }
    let moved_range = scratch.take::<u8>(ranges);
    for r in 0..ranges {
        *moved_range.add(r) = 1;
    }
    let kept = if in_place { given } else { kept_out };
    let columns = kept.p;
    let cache = kept.p.add(bn);
    let cache_up = kept.p.add(bn + size);
    let stale = kept.p.add(bn + size + size_up);
    if in_place {
        for r in 0..ranges {
            *moved_range.add(r) = !(*feeds.add(r) != 0 && *stale.add(r) == 0.0) as u8;
        }
    } else {
        for b in 0..batch {
            copy_columns(net, b, net.s, columns, feeds);
        }
        if has_given {
            reusable(net, columns, bn, kept.p, given, feeds, moved_range);
        }
        keep_published(net, kept);
    }
    for r in 0..ranges {
        *stale.add(r) = 0.0;
    }
    for q in 0..np_ + nq {
        if q < np_ {
            let k = net.pairs.get(q) as usize;
            if *moved_range.add(net.pair_pre.get(k) as usize) != 0 {
                block_product(net, columns, net.flat, k, cache, *cache_offset.add(q), keep);
            }
        } else {
            let k = net.apical_pairs.get(q - np_) as usize;
            if *moved_range.add(net.pair_pre.get(k) as usize) != 0 {
                block_product(net, columns, net.flat_apical, k, cache_up,
                              *cache_up_offset.add(q - np_), keep);
            }
        }
    }
    let synaptic_input = scratch.take::<f64>(bn);
    let raw = scratch.take::<f64>(wide);
    let basal = scratch.take::<f64>(wide);
    let values = scratch.take::<f64>(wide);
    let gain = scratch.take::<f64>(batch * pools);
    let p = scratch.take::<f64>(batch * n.max(1));
    let zmax = scratch.take::<f64>(batch * ng);
    let total = scratch.take::<f64>(batch * ng);
    let (v, a, s) = (net.v, net.a, net.s);
    let softmax = net.softmax();
    for b in 0..batch {
        gather(net, b, synaptic_input, cache, cache_offset, net.pairs, receives);
        let mut worst = 0.0f64;
        let mut all_finite = true;
        let (pb, zb, tb) = (p.add(b * n), zmax.add(b * ng), total.add(b * ng));
        if softmax {
            softmax_groups(s.p.add(b * n), group, groups, net.gid, net.ngroups, net.softmax_t,
                           pb, zb, tb);
        }
        if net.has_apical {
            gather(net, b, raw, cache_up, cache_up_offset, net.apical_pairs, receives_up);
            let bas = basal.add(b * n);
            basal_input(net, b, synaptic_input, pb, position, bas);
            pool_gains(net, b, bas, raw.add(b * n), gain.add(b * pools), values.add(b * n));
            for i in 0..n {
                let mut x = *bas.add(i);
                let k = net.pool_of.get(i);
                if k >= 0 {
                    x += *gain.add(b * pools + k as usize) * *raw.add(b * n + i);
                }
                x -= v.at(b, i);
                let k_ = if net.masked { net.keep.at(b, i) } else { 1.0 };
                let mut err = fabs(k_ * x + (k_ - 1.0) * v.at(b, i) / net.dt);
                if net.has_adapt {
                    let gap = fabs(s.at(b, i) - a.at(b, i));
                    if gap > err {
                        err = gap;
                    }
                }
                if !(err <= 1e300) {
                    all_finite = false;
                } else if err > worst {
                    worst = err;
                }
            }
        } else {
            for i in 0..n {
                let mut tot = *synaptic_input.add(b * n + i) + net.standing.at(b, i) - v.at(b, i);
                if net.has_adapt {
                    tot -= net.strength * a.at(b, i);
                }
                if net.anchor_gain.get(i) > 0.0 {
                    tot += net.anchor_gain.get(i) * (net.anchor.at(b, i) - s.at(b, i));
                }
                if net.has_nudge && net.nmask.get(i) > 0.0 {
                    if net.softmax_t.get(0) > 0.0 {
                        tot += net.beta * net.weight.get(b)
                            * (net.target.at(b, i) - *pb.add(*position.add(i) as usize))
                            * net.nmask.get(i);
                    } else {
                        tot += net.beta * net.weight.get(b) * net.nmask.get(i)
                            * (net.target.at(b, i) - s.at(b, i));
                    }
                }
                let k_ = if net.masked { net.keep.at(b, i) } else { 1.0 };
                let mut err = fabs(k_ * tot + (k_ - 1.0) * v.at(b, i) / net.dt);
                if net.has_adapt {
                    let gap = fabs(s.at(b, i) - a.at(b, i));
                    if gap > err {
                        err = gap;
                    }
                }
                if !(err <= 1e300) {
                    all_finite = false;
                } else if err > worst {
                    worst = err;
                }
            }
        }
        out.set(b, if all_finite { worst } else { f64::INFINITY });
    }
    0
}

/// `_publish`: the activations of the potentials `v` into `s`, read from `kept` where they are
/// the ones it holds under the same rule (returns 1), else each neuron's (returns 0).
pub(crate) unsafe fn publish(
    v: View<f64>, s: View<f64>, kept: View<f64>, has_kept: bool, slope: f64, threshold: f64,
    rest: f64, leak: f64,
) -> i32 {
    let (batch, n) = (v.n0, v.n1);
    let count = batch * n;
    let mut same = has_kept && kept.len >= RECORD + 2 * count;
    if same {
        let head = kept.len - RECORD;
        same = kept.get(head) == 1.0
            && kept.get(head + 1) == count as f64
            && kept.get(head + 2) == slope
            && kept.get(head + 3) == threshold
            && kept.get(head + 4) == rest
            && kept.get(head + 5) == leak;
        if same {
            let at = head - 2 * count;
            let mut i = 0usize;
            while same && i < count {
                same = v.get(i) == kept.get(at + i);
                i += 1;
            }
            if same {
                let at = head - count;
                for i in 0..count {
                    s.set(i, kept.get(at + i));
                }
                return 1;
            }
        }
    }
    let mut any = false;
    for i in 0..count {
        any = any || v.get(i) != 0.0;
    }
    if !any {
        let rest_value = act_scalar(0.0, slope, threshold, rest, leak);
        for i in 0..count {
            s.set(i, rest_value);
        }
    } else {
        for i in 0..count {
            s.set(i, act_scalar(v.get(i), slope, threshold, rest, leak));
        }
    }
    0
}

/// The arguments every launch shares, from their objects' addresses; the argument numbers of
/// a refusal count from `v`.
pub(crate) unsafe fn net_of(
    ids: &[u32; 30], masked: i32, rule: [f64; 5], has_adapt: i32, strength: f64, tau: f64,
    has_nudge: i32, ngroups: i32, beta: f64, has_apical: i32, apical: [f64; 3], has_held: i32,
) -> Result<Net, i32> {
    macro_rules! get {
        ($k:expr, $t:ty, $num:expr) => {
            match view::<$t>(ids[$k], $num) {
                Some(x) => x,
                None => return Err(refused($k as i32)),
            }
        };
    }
    let v = get!(0, f64, FLOAT64);
    let net = Net {
        batch: v.n0,
        n: v.n1,
        ranges: 0,
        v,
        a: get!(1, f64, FLOAT64),
        s: get!(2, f64, FLOAT64),
        standing: get!(3, f64, FLOAT64),
        flat: get!(4, f64, FLOAT64),
        pairs: get!(5, i64, INT64),
        flat_apical: get!(6, f64, FLOAT64),
        apical_pairs: get!(7, i64, INT64),
        starts: get!(8, i64, INT64),
        pair_pre: get!(9, i64, INT64),
        pair_post: get!(10, i64, INT64),
        offset: get!(11, i64, INT64),
        keep: get!(12, f64, FLOAT64),
        masked: masked != 0,
        dt: rule[0],
        slope: rule[1],
        threshold: rule[2],
        rest: rule[3],
        leak: rule[4],
        has_adapt: has_adapt != 0,
        strength,
        tau,
        has_nudge: has_nudge != 0,
        target: get!(13, f64, FLOAT64),
        nmask: get!(14, f64, FLOAT64),
        gid: get!(15, i64, INT64),
        ngroups: ngroups.max(0) as usize,
        beta,
        softmax_t: get!(16, f64, FLOAT64),
        weight: get!(17, f64, FLOAT64),
        anchor: get!(18, f64, FLOAT64),
        anchor_gain: get!(19, f64, FLOAT64),
        has_apical: has_apical != 0,
        pool_of: get!(20, i64, INT64),
        pool_index: get!(21, i64, INT64),
        pool_starts: get!(22, i64, INT64),
        pool_run: get!(23, i64, INT64),
        fraction: apical[0],
        floor2: apical[1],
        tiny: apical[2],
        held: get!(24, f64, FLOAT64),
        has_held: has_held != 0,
        per_life: false,
    };
    let ranges = if net.starts.len > 0 { net.starts.len - 1 } else { 0 };
    let net = Net { ranges, ..net };
    // the shapes the loops index by: every state array as wide as v, every per-neuron array
    // n long, the blocks inside the flat arrays
    let (batch, n) = (net.batch, net.n);
    for (k, x) in [(1, net.a), (2, net.s), (3, net.standing)] {
        if x.n0 != batch || x.n1 != n {
            return Err(refused(k));
        }
    }
    if ranges == 0 || net.start(ranges) != n || net.offset.len != net.pair_pre.len + 1 {
        return Err(refused(8));
    }
    if net.masked && net.keep.len != batch * n {
        return Err(refused(12));
    }
    if net.has_nudge
        && (net.target.len != batch * n || net.nmask.len != n || net.gid.len != n
            || net.softmax_t.len != n || net.weight.len < batch)
    {
        return Err(refused(13));
    }
    if net.anchor_gain.len != n {
        return Err(refused(19));
    }
    for i in 0..n {
        if net.anchor_gain.get(i) > 0.0 && net.anchor.len != batch * n {
            return Err(refused(18));
        }
    }
    for (k, x) in [(4usize, net.flat), (6, net.flat_apical)] {
        let pairs = if k == 4 { net.pairs } else { net.apical_pairs };
        for q in 0..pairs.len {
            let block = pairs.get(q) as usize;
            if block >= net.pair_pre.len || net.offset.get(block + 1) as usize > x.len {
                return Err(refused(k as i32));
            }
        }
    }
    if net.has_apical && (net.pool_of.len != n || (net.has_held && net.held.len < batch)) {
        return Err(refused(20));
    }
    Ok(net)
}

#[inline(always)]
fn ids30(list: [u32; 30]) -> [u32; 30] {
    list
}

/// `_settle_rows`, then `activity_change` (batch,) and `kept` (zeros, as `kept_len` gives).
#[no_mangle]
pub unsafe extern "C" fn settle_rows(
    v: u32, a: u32, s: u32, standing: u32, flat: u32, pairs: u32, flat_apical: u32,
    apical_pairs: u32, starts: u32, pair_pre: u32, pair_post: u32, offset: u32,
    freezable: u32, keep: u32, masked: i32, dt: f64, slope: f64, threshold: f64, rest: f64,
    leak: f64, has_adapt: i32, adapt_strength: f64, adapt_tau: f64, has_nudge: i32,
    target: u32, nmask: u32, gid: u32, ngroups: i32, beta: f64, softmax_t: u32, weight: u32,
    anchor: u32, anchor_gain: u32, has_apical: i32, pool_of: u32, pool_index: u32,
    pool_starts: u32, pool_run: u32, fraction: f64, floor2: f64, tiny: f64, held: u32,
    has_held: i32, steps: i32, tolerance: f64, use_tolerance: i32, given: u32, has_given: i32,
    activity_change: u32, kept: u32,
) -> i32 {
    let ids = ids30([
        v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre, pair_post,
        offset, keep, target, nmask, gid, softmax_t, weight, anchor, anchor_gain, pool_of,
        pool_index, pool_starts, pool_run, held, 0, 0, 0, 0, 0,
    ]);
    let net = match net_of(&ids, masked, [dt, slope, threshold, rest, leak], has_adapt,
                           adapt_strength, adapt_tau, has_nudge, ngroups, beta, has_apical,
                           [fraction, floor2, tiny], has_held) {
        Ok(x) => x,
        Err(code) => return code,
    };
    let freezable = take!(freezable, u8, BOOL, 25);
    let given = take!(given, f64, FLOAT64, 26);
    let change = take!(activity_change, f64, FLOAT64, 27);
    let kept = take!(kept, f64, FLOAT64, 28);
    if freezable.len != net.ranges || change.len != net.batch || kept.len != net.kept_len() {
        return refused(28);
    }
    if has_given != 0 && given.len != kept.len {
        return refused(26);
    }
    settle(&net, freezable, steps.max(0) as usize, tolerance, use_tolerance != 0, given,
           has_given != 0, change, kept)
}

/// `settle_rows` for lives: each row a life of its own, `flat` (lives, blocks) and `flat_apical`
/// (lives, apical blocks) one row of a life's synapses each; every other argument as in
/// `settle_rows`, the bias inside each row's `standing`.
#[no_mangle]
pub unsafe extern "C" fn settle_lives(
    v: u32, a: u32, s: u32, standing: u32, flat: u32, pairs: u32, flat_apical: u32,
    apical_pairs: u32, starts: u32, pair_pre: u32, pair_post: u32, offset: u32,
    freezable: u32, keep: u32, masked: i32, dt: f64, slope: f64, threshold: f64, rest: f64,
    leak: f64, has_adapt: i32, adapt_strength: f64, adapt_tau: f64, has_nudge: i32,
    target: u32, nmask: u32, gid: u32, ngroups: i32, beta: f64, softmax_t: u32, weight: u32,
    anchor: u32, anchor_gain: u32, has_apical: i32, pool_of: u32, pool_index: u32,
    pool_starts: u32, pool_run: u32, fraction: f64, floor2: f64, tiny: f64, held: u32,
    has_held: i32, steps: i32, tolerance: f64, use_tolerance: i32, given: u32, has_given: i32,
    activity_change: u32, kept: u32,
) -> i32 {
    let ids = ids30([
        v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre, pair_post,
        offset, keep, target, nmask, gid, softmax_t, weight, anchor, anchor_gain, pool_of,
        pool_index, pool_starts, pool_run, held, 0, 0, 0, 0, 0,
    ]);
    let net = match net_of(&ids, masked, [dt, slope, threshold, rest, leak], has_adapt,
                           adapt_strength, adapt_tau, has_nudge, ngroups, beta, has_apical,
                           [fraction, floor2, tiny], has_held) {
        Ok(x) => Net { per_life: true, ..x },
        Err(code) => return code,
    };
    for (k, x) in [(4, net.flat), (6, net.flat_apical)] {
        if x.n0 != net.batch || (x.len > 0 && x.n1 < net.offset.get(net.pair_pre.len) as usize) {
            return refused(k);
        }
    }
    let freezable = take!(freezable, u8, BOOL, 25);
    let given = take!(given, f64, FLOAT64, 26);
    let change = take!(activity_change, f64, FLOAT64, 27);
    let kept = take!(kept, f64, FLOAT64, 28);
    if freezable.len != net.ranges || change.len != net.batch || kept.len != net.kept_len() {
        return refused(28);
    }
    if has_given != 0 && given.len != kept.len {
        return refused(26);
    }
    settle(&net, freezable, steps.max(0) as usize, tolerance, use_tolerance != 0, given,
           has_given != 0, change, kept)
}

/// `_residual_rows`, then `kept` (zeros, as `kept_len` gives; unread with `in_place`, where
/// the given array is the one kept).
#[no_mangle]
pub unsafe extern "C" fn residual_rows(
    v: u32, a: u32, s: u32, standing: u32, flat: u32, pairs: u32, flat_apical: u32,
    apical_pairs: u32, starts: u32, pair_pre: u32, pair_post: u32, offset: u32, keep: u32,
    masked: i32, dt: f64, slope: f64, threshold: f64, rest: f64, leak: f64, has_adapt: i32,
    adapt_strength: f64, has_nudge: i32, target: u32, nmask: u32, gid: u32, ngroups: i32,
    beta: f64, softmax_t: u32, weight: u32, anchor: u32, anchor_gain: u32, has_apical: i32,
    pool_of: u32, pool_index: u32, pool_starts: u32, pool_run: u32, fraction: f64, floor2: f64,
    tiny: f64, held: u32, has_held: i32, out: u32, given: u32, has_given: i32, in_place: i32,
    kept: u32,
) -> i32 {
    let ids = ids30([
        v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre, pair_post,
        offset, keep, target, nmask, gid, softmax_t, weight, anchor, anchor_gain, pool_of,
        pool_index, pool_starts, pool_run, held, 0, 0, 0, 0, 0,
    ]);
    let net = match net_of(&ids, masked, [dt, slope, threshold, rest, leak], has_adapt,
                           adapt_strength, 1.0, has_nudge, ngroups, beta, has_apical,
                           [fraction, floor2, tiny], has_held) {
        Ok(x) => x,
        Err(code) => return code,
    };
    let out = take!(out, f64, FLOAT64, 25);
    let given = take!(given, f64, FLOAT64, 26);
    if out.len != net.batch {
        return refused(25);
    }
    let length = net.kept_len();
    if (has_given != 0 || in_place != 0) && given.len != length {
        return refused(26);
    }
    let kept = if in_place != 0 {
        given
    } else {
        let k = take!(kept, f64, FLOAT64, 27);
        if k.len != length {
            return refused(27);
        }
        k
    };
    residual(&net, out, given, has_given != 0, in_place != 0, kept)
}

/// `_solve_rows`: the check, then chunks of sweeps and their checks while a row is over
/// `tolerance` and sweeps are left, with the apical compartment where `has_apical` says so. `given` is the state's kept array (read where `has_given`,
/// made current in place where `in_place`); `kept_a` and `kept_b` (zeros, `kept_len` long)
/// take the chunks' states in turn. `report` (int64, 5) receives the sweeps, the checks, the
/// last chunk's sweeps and which array holds what the state keeps (0 given, 1 `kept_a`, 2
/// `kept_b`).
#[no_mangle]
pub unsafe extern "C" fn solve_rows(
    v: u32, a: u32, s: u32, standing: u32, flat: u32, pairs: u32, flat_apical: u32,
    apical_pairs: u32, starts: u32, pair_pre: u32, pair_post: u32, offset: u32,
    freezable: u32, keep: u32, dt: f64, slope: f64, threshold: f64, rest: f64, leak: f64,
    strength: f64, tau: f64, target: u32, nmask: u32, gid: u32, ngroups: i32, beta: f64,
    softmax_t: u32, weight: u32, anchor: u32, anchor_gain: u32, pool_of: u32, pool_index: u32,
    pool_starts: u32, pool_run: u32, held: u32, budget: i32, chunk: i32, tolerance: f64,
    out: u32, given: u32, has_given: i32, in_place: i32, has_apical: i32, fraction: f64,
    floor2: f64, tiny: f64, change: u32, kept_a: u32, kept_b: u32, report: u32,
) -> i32 {
    let ids = ids30([
        v, a, s, standing, flat, pairs, flat_apical, apical_pairs, starts, pair_pre, pair_post,
        offset, keep, target, nmask, gid, softmax_t, weight, anchor, anchor_gain, pool_of,
        pool_index, pool_starts, pool_run, held, 0, 0, 0, 0, 0,
    ]);
    // the off flags of a free solve: no mask, adaptation or nudge, no gain held
    let net = match net_of(&ids, 0, [dt, slope, threshold, rest, leak], 0, strength, tau, 0,
                           ngroups, beta, has_apical, [fraction, floor2, tiny], 0) {
        Ok(x) => x,
        Err(code) => return code,
    };
    let freezable = take!(freezable, u8, BOOL, 25);
    let out = take!(out, f64, FLOAT64, 26);
    let given = take!(given, f64, FLOAT64, 27);
    let change = take!(change, f64, FLOAT64, 28);
    let ka = take!(kept_a, f64, FLOAT64, 29);
    let kb = take!(kept_b, f64, FLOAT64, 29);
    let report = take!(report, i64, INT64, 29);
    let length = net.kept_len();
    if freezable.len != net.ranges || out.len != net.batch || change.len != net.batch
        || ka.len != length || kb.len != length || report.len < 4
    {
        return refused(29);
    }
    if (has_given != 0 || in_place != 0) && given.len != length {
        return refused(27);
    }
    let batch = net.batch;
    let code = residual(&net, out, given, has_given != 0, in_place != 0, ka);
    if code < 0 {
        return code;
    }
    let mut kept = if in_place != 0 { 0usize } else { 1usize };
    let budget = budget.max(0) as usize;
    let chunk = chunk.max(0) as usize;
    let (mut used, mut checks, mut taken) = (0usize, 1usize, 0usize);
    let all_finite = |x: View<f64>| {
        let mut ok = true;
        for i in 0..x.len {
            ok = ok && finite(x.get(i));
        }
        ok
    };
    while used < budget {
        let mut qualified = true;
        let mut finite_out = true;
        for b in 0..batch {
            qualified = qualified && out.get(b) <= tolerance;
            finite_out = finite_out && out.get(b) <= 1.7976931348623157e308;
        }
        if qualified {
            break;
        }
        if !finite_out && !(all_finite(net.v) && all_finite(net.s) && all_finite(net.a)) {
            break;
        }
        let source = match kept {
            0 => given,
            1 => ka,
            _ => kb,
        };
        let target = if kept == 1 { kb } else { ka };
        for i in 0..target.len {
            target.set(i, 0.0);
        }
        for b in 0..batch {
            change.set(b, 0.0);
        }
        let steps = chunk.min(budget - used);
        let got = settle(&net, freezable, steps, 0.0, false, source, true, change, target);
        if got < 0 {
            return got;
        }
        taken = got as usize;
        kept = if kept == 1 { 2 } else { 1 };
        used += taken;
        let code = residual(&net, out, target, true, true, target);
        if code < 0 {
            return code;
        }
        checks += 1;
    }
    report.set(0, used as i64);
    report.set(1, checks as i64);
    report.set(2, taken as i64);
    report.set(3, kept as i64);
    0
}

/// The length of the kept array of a launch over `pairs` and `apical_pairs` at `batch` rows
/// and `n` neurons (`_kept_room`).
#[no_mangle]
pub unsafe extern "C" fn kept_length(
    starts: u32, pair_post: u32, pairs: u32, apical_pairs: u32, batch: i32, n: i32,
) -> i32 {
    let starts = take!(starts, i64, INT64, 0);
    let pair_post = take!(pair_post, i64, INT64, 1);
    let pairs = take!(pairs, i64, INT64, 2);
    let apical_pairs = take!(apical_pairs, i64, INT64, 3);
    let (batch, n) = (batch.max(0) as usize, n.max(0) as usize);
    let ranges = if starts.len > 0 { starts.len - 1 } else { 0 };
    let mut size = 0usize;
    for list in [pairs, apical_pairs] {
        for q in 0..list.len {
            let b = pair_post.get(list.get(q) as usize) as usize;
            size += batch * (starts.get(b + 1) - starts.get(b)) as usize;
        }
    }
    (3 * batch * n + size + ranges + RECORD) as i32
}

/// `_publish`; returns 1 where the kept activations were read, 0 where they were made.
#[no_mangle]
pub unsafe extern "C" fn publish_rows(
    v: u32, s: u32, kept: u32, has_kept: i32, slope: f64, threshold: f64, rest: f64,
    leak: f64,
) -> i32 {
    let v = take!(v, f64, FLOAT64, 0);
    let s = take!(s, f64, FLOAT64, 1);
    let kept = take!(kept, f64, FLOAT64, 2);
    if s.len != v.len {
        return refused(1);
    }
    publish(v, s, kept, has_kept != 0, slope, threshold, rest, leak)
}

/// `rows_finite`: 1 where every entry is finite.
#[no_mangle]
pub unsafe extern "C" fn rows_finite(x: u32) -> i32 {
    let x = take!(x, f64, FLOAT64, 0);
    let mut ok = true;
    let mut i = 0usize;
    while i + 2 <= x.len {
        let y = v128_load(x.p.add(i) as *const v128);
        let z = f64x2_sub(y, y);
        ok &= f64x2_extract_lane::<0>(z) == 0.0 && f64x2_extract_lane::<1>(z) == 0.0;
        i += 2;
    }
    if i < x.len {
        ok &= finite(x.get(i));
    }
    ok as i32
}
