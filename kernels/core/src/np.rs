//! NumPy's arithmetic where its order decides the bits: the pairwise sum of a reduction, the
//! dot product without BLAS, the first maximum, and the NaN rules of maximum, minimum and clip.
//! Every part of the core sums through these, so the core has one reduction order everywhere
//! and a moment is the same bits in the browser and natively.

/// NumPy's sum of a contiguous run (`pairwise_sum`): under eight in order from -0.0, up to 128
/// with eight running sums combined in pairs, longer runs halved at a multiple of eight.
pub fn sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    sum(&a[..half]) + sum(&a[half..])
}

/// The sum of the squares of a run, in `sum`'s order (`np.sum(x ** 2)`).
pub fn sum_squares(a: &[f64]) -> f64 {
    let squares: Vec<f64> = a.iter().map(|x| x * x).collect();
    sum(&squares)
}

/// `np.mean` of a run: its sum over its count.
pub fn mean(a: &[f64]) -> f64 {
    sum(a) / a.len() as f64
}

/// A dot product without BLAS: in order from 0.0 (NumPy's `dot`, `matmul` and a vector's
/// `linalg.norm` under Pyodide).
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// `np.linalg.norm` of a vector: the square root of its dot product with itself.
pub fn norm(a: &[f64]) -> f64 {
    libm::sqrt(dot(a, a))
}

/// `np.argmax`: the first largest entry, or the first NaN.
pub fn argmax(a: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &x) in a.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x > a[best] {
            best = i;
        }
    }
    best
}

/// `np.argmin`: the first smallest entry, or the first NaN.
pub fn argmin(a: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &x) in a.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x < a[best] {
            best = i;
        }
    }
    best
}

/// `np.maximum(a, b)`: a NaN on either side is the answer.
#[inline]
pub fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `np.minimum(a, b)`: a NaN on either side is the answer.
#[inline]
pub fn minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

/// `np.clip(x, lo, hi)`: the larger of x and lo (a NaN kept), then the smaller of that and hi.
#[inline]
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let m = if x > lo { x } else { lo };
    if m < hi {
        m
    } else {
        hi
    }
}

/// The C library's functions as NumPy calls them under Pyodide (musl's, as the `libm` crate
/// carries them; `exp` there is within one unit in the last place of Pyodide's).
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}
#[inline]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}
#[inline]
pub fn log(x: f64) -> f64 {
    libm::log(x)
}
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// NumPy's order of floats in a sort or a selection: NaN after every number.
#[inline]
fn less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// `np.argpartition(v, kth)` of one row, as NumPy's introselect leaves it (the arrangement
/// matters where values tie at the cut): the indices of `v` with the `kth` smallest at `kth`,
/// the smaller ones before it and the larger after. NumPy on x86 with AVX2 or AVX-512 selects
/// with its own SIMD routine, which arranges ties otherwise.
pub fn argpartition(v: &[f64], kth: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    if !v.is_empty() {
        introselect(v, &mut idx, kth);
    }
    idx
}

fn dumb_select(v: &[f64], t: &mut [usize], kth: usize) {
    let num = t.len();
    for i in 0..=kth {
        let mut minidx = i;
        let mut minval = v[t[i]];
        for k in i + 1..num {
            if less(v[t[k]], minval) {
                minidx = k;
                minval = v[t[k]];
            }
        }
        t.swap(i, minidx);
    }
}

fn median3_swap(v: &[f64], t: &mut [usize], low: usize, mid: usize, high: usize) {
    if less(v[t[high]], v[t[mid]]) {
        t.swap(high, mid);
    }
    if less(v[t[high]], v[t[low]]) {
        t.swap(high, low);
    }
    if less(v[t[low]], v[t[mid]]) {
        t.swap(low, mid);
    }
    t.swap(mid, low + 1);
}

fn median5(v: &[f64], t: &mut [usize]) -> usize {
    if less(v[t[1]], v[t[0]]) {
        t.swap(1, 0);
    }
    if less(v[t[4]], v[t[3]]) {
        t.swap(4, 3);
    }
    if less(v[t[3]], v[t[0]]) {
        t.swap(3, 0);
    }
    if less(v[t[4]], v[t[1]]) {
        t.swap(4, 1);
    }
    if less(v[t[2]], v[t[1]]) {
        t.swap(2, 1);
    }
    if less(v[t[3]], v[t[2]]) {
        if less(v[t[3]], v[t[1]]) {
            1
        } else {
            3
        }
    } else {
        2
    }
}

fn median_of_median5(v: &[f64], t: &mut [usize]) -> usize {
    let num = t.len();
    let nmed = num / 5;
    let mut subleft = 0;
    for i in 0..nmed {
        let m = median5(v, &mut t[subleft..subleft + 5]);
        t.swap(subleft + m, i);
        subleft += 5;
    }
    if nmed > 2 {
        introselect(v, &mut t[..nmed], nmed / 2);
    }
    nmed / 2
}

fn msb(mut n: usize) -> i64 {
    let mut depth = 0;
    while n > 1 {
        n >>= 1;
        depth += 1;
    }
    depth
}

fn introselect(v: &[f64], t: &mut [usize], kth: usize) {
    let num = t.len();
    let mut low = 0usize;
    let mut high = num - 1;
    if kth - low < 3 {
        dumb_select(v, &mut t[low..=high], kth - low);
        return;
    }
    if kth == num - 1 {
        // floats: the last of the largest goes to the end (NumPy's check for a NaN)
        let mut maxidx = low;
        let mut maxval = v[t[low]];
        for k in low + 1..num {
            if !less(v[t[k]], maxval) {
                maxidx = k;
                maxval = v[t[k]];
            }
        }
        t.swap(kth, maxidx);
        return;
    }
    let mut depth_limit = msb(num) * 2;
    while low + 1 < high {
        let mut ll = low + 1;
        let mut hh = high;
        if depth_limit > 0 || hh - ll < 5 {
            let mid = low + (high - low) / 2;
            median3_swap(v, t, low, mid, high);
        } else {
            let mid = ll + median_of_median5(v, &mut t[ll..hh]);
            t.swap(mid, low);
            ll -= 1;
            hh += 1;
        }
        depth_limit -= 1;
        let pivot = v[t[low]];
        loop {
            loop {
                ll += 1;
                if !less(v[t[ll]], pivot) {
                    break;
                }
            }
            loop {
                hh -= 1;
                if !less(pivot, v[t[hh]]) {
                    break;
                }
            }
            if hh < ll {
                break;
            }
            t.swap(ll, hh);
        }
        t.swap(low, hh);
        if hh >= kth {
            high = hh - 1;
        }
        if hh <= kth {
            low = ll;
        }
    }
    if high == low + 1 && less(v[t[high]], v[t[low]]) {
        t.swap(high, low);
    }
}

// -- the orders the recall memory reads (`cadence.stream.Recall`)

/// `a @ b` without BLAS, `a` of `n` rows by `k` and `b` of `k` rows by `m`, both row-major:
/// each entry a dot product in order from 0.0, as `dot` sums it (NumPy's `matmul`, and an
/// `einsum` of one contraction, under Pyodide). A `b` of one column is a vector.
pub fn matmul(a: &[f64], n: usize, k: usize, b: &[f64], m: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * m];
    for i in 0..n {
        let row = &mut out[i * m..(i + 1) * m];
        for l in 0..k {
            let x = a[i * k + l];
            for (o, &y) in row.iter_mut().zip(&b[l * m..(l + 1) * m]) {
                *o += x * y;
            }
        }
    }
    out
}

/// `np.max` of a run (`maximum.reduce`): a NaN anywhere is the answer, else the largest entry.
/// Between zeros of either sign it keeps the first (NumPy's sign of a zero maximum follows its
/// SIMD routine).
pub fn max(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &x in &a[1..] {
        m = maximum(m, x);
    }
    m
}

/// `np.cumsum` of a run (`add.accumulate`): each entry the running sum from the first, in order.
pub fn cumsum(a: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    let mut s = 0.0f64;
    for (i, &x) in a.iter().enumerate() {
        s = if i == 0 { x } else { s + x };
        out.push(s);
    }
    out
}

/// `np.argsort(v, kind="stable")` of one row: ascending, NaN after every number, equal entries
/// (zeros of either sign among them) in their order.
pub fn argsort_stable(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| {
        if less(v[i], v[j]) {
            std::cmp::Ordering::Less
        } else if less(v[j], v[i]) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    idx
}

/// `np.logaddexp(x, y)`, NumPy's `npy_logaddexp`: `x + log(2)` where the two are equal (the
/// infinities of one sign among them), else the larger plus `log1p(exp(-|x - y|))`, a NaN kept.
pub fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        return x + std::f64::consts::LN_2;
    }
    let tmp = x - y;
    if tmp > 0.0 {
        x + log1p(exp(-tmp))
    } else if tmp <= 0.0 {
        y + log1p(exp(tmp))
    } else {
        tmp
    }
}

#[inline]
pub fn log1p(x: f64) -> f64 {
    libm::log1p(x)
}
#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}
