//! A brain's whole moment in Rust, the state kept here between moments (cadence 0.102.0's
//! native core). The library's Python is the reference: every part of a moment here computes
//! what its Python computes, a brain's state crosses as named arrays (`state`), and a check runs
//! both side by side (`kernels/core/check`). The same source builds natively (a shared library
//! for `cadence.core`, threads across animals) and to WebAssembly with its own memory (for a
//! page's brain worker, no Pyodide in the moment).

pub mod np;
pub mod parts;
pub mod rng;
pub mod state;
pub(crate) mod ziggurat;

// The compiled kernels of `cadence.fused` and `cadence.compiled`, one source with the browser
// module (`kernels/wasm`): the settle, the solve, the lessons' passes, the map's and the bound
// records' kernels, called here on the core's own arrays.
#[path = "../../wasm/src/common.rs"]
mod common;
pub(crate) use common::*;
#[path = "../../wasm/src/bound.rs"]
pub mod bound;
#[path = "../../wasm/src/fast.rs"]
pub mod fast;
#[path = "../../wasm/src/lesson.rs"]
pub mod lesson;
#[path = "../../wasm/src/map.rs"]
pub mod map;
#[path = "../../wasm/src/settle.rs"]
pub mod settle;

thread_local! {
    /// The kernels' scratch on this thread: the rooms lent to kernels and the rooms they gave
    /// back. A room keeps the largest size it was asked for and is never freed, so a thread's
    /// moments reuse the same warm memory; a kernel zeroes what it takes of it (`Scratch::take`).
    static ROOMS: std::cell::RefCell<(Vec<Vec<u128>>, Vec<Vec<u128>>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Room of `n` bytes for a kernel's scratch, from this thread's rooms: the smallest returned
/// room that holds `n` bytes, else the largest, grown to them.
pub(crate) unsafe fn host_malloc(n: usize) -> *mut u8 {
    ROOMS.with(|rooms| {
        let mut rooms = rooms.borrow_mut();
        let (free, lent) = &mut *rooms;
        let words = n.div_ceil(16).max(1);
        let fits = free.iter().enumerate().filter(|(_, r)| r.len() >= words)
            .min_by_key(|(_, r)| r.len()).map(|(i, _)| i);
        let pick = fits.or_else(|| {
            free.iter().enumerate().max_by_key(|(_, r)| r.len()).map(|(i, _)| i)
        });
        let mut room = pick.map_or_else(Vec::new, |i| free.swap_remove(i));
        if room.len() < words {
            room.resize(words, 0);
        }
        let p = room.as_mut_ptr() as *mut u8;
        lent.push(room);
        p
    })
}

/// A room `host_malloc` lent, given back to this thread's rooms.
pub(crate) unsafe fn host_free(p: *mut u8) {
    if p.is_null() {
        return;
    }
    ROOMS.with(|rooms| {
        let mut rooms = rooms.borrow_mut();
        let (free, lent) = &mut *rooms;
        if let Some(i) = lent.iter().position(|r| std::ptr::eq(r.as_ptr() as *const u8, p)) {
            let room = lent.swap_remove(i);
            free.push(room);
        }
    })
}

pub(crate) fn trap() -> ! {
    std::process::abort()
}

/// The kernels' `exp`, `tanh` and square root: `np`'s, the platform's libm natively and musl's
/// in the browser.
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    np::exp(x)
}
#[inline(always)]
pub(crate) fn tanh(x: f64) -> f64 {
    np::tanh(x)
}
#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    np::sqrt(x)
}

use state::{Array, Named};

/// The version a caller checks before it hands anything over.
pub const VERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn core_version() -> u32 {
    VERSION
}

/// Room of `n` bytes the caller fills (a page writes its inputs there); released with
/// `core_release`.
#[no_mangle]
pub extern "C" fn core_buffer(n: usize) -> *mut u8 {
    let mut v = vec![0u8; n.max(1)];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p` and `n` are what `core_buffer`, or an output of this library, gave.
#[no_mangle]
pub unsafe extern "C" fn core_release(p: *mut u8, n: usize) {
    if !p.is_null() {
        drop(Vec::from_raw_parts(p, n.max(1), n.max(1)));
    }
}

/// Hand `out`'s bytes to the caller: their address and length, released with
/// `core_release(address, length)`.
unsafe fn give(mut out: Vec<u8>, at: *mut *mut u8, len: *mut usize) {
    let n = out.len();
    if out.is_empty() {
        out.push(0);  // a release of length 0 frees room of one byte
    }
    let mut boxed = out.into_boxed_slice();
    *at = boxed.as_mut_ptr();
    *len = n;
    std::mem::forget(boxed);
}

/// One primitive by name on named arrays, for the checks that hold the core's arithmetic
/// against NumPy's (`kernels/core/check/primitives.py`). Returns 0, or a negative code.
///
/// # Safety
/// The pointers are the caller's buffers of the given lengths.
#[no_mangle]
pub unsafe extern "C" fn core_probe(
    name: *const u8, name_len: usize, input: *const u8, input_len: usize, out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let name = match std::str::from_utf8(std::slice::from_raw_parts(name, name_len)) {
        Ok(x) => x,
        Err(_) => return -1,
    };
    let input = match Named::read(std::slice::from_raw_parts(input, input_len)) {
        Ok(x) => x,
        Err(_) => return -2,
    };
    match probe(name, &input) {
        Ok(named) => {
            give(named.write(), out, out_len);
            0
        }
        Err(_) => -3,
    }
}

fn generator(input: &Named) -> Result<rng::Pcg64, state::Error> {
    let s = input.u64s("state")?;
    let i = input.u64s("inc")?;
    Ok(rng::Pcg64 {
        state: ((s[0] as u128) << 64) | s[1] as u128,
        inc: ((i[0] as u128) << 64) | i[1] as u128,
        has_uint32: input.i64("has_uint32")? != 0,
        uinteger: input.i64("uinteger")? as u32,
    })
}

fn probe(name: &str, input: &Named) -> Result<Named, state::Error> {
    let mut out = Named::new();
    match name {
        "random" => {
            let mut g = generator(input)?;
            let n = input.i64("n")? as usize;
            let draws: Vec<f64> = (0..n).map(|_| g.random()).collect();
            out.put("draws", Array::from_f64(&[n], &draws));
            out.put("after", Array::from_u64(&[2], &[(g.state >> 64) as u64, g.state as u64]));
        }
        "integers" => {
            let mut g = generator(input)?;
            let (low, high, n) = (input.i64("low")?, input.i64("high")?, input.i64("n")? as usize);
            let draws: Vec<i64> = (0..n).map(|_| g.integers(low, high)).collect();
            out.put("draws", Array::from_i64(&[n], &draws));
        }
        "choice_p" => {
            let mut g = generator(input)?;
            let p = input.f64s("p")?;
            let n = input.i64("n")? as usize;
            let draws: Vec<i64> = (0..n).map(|_| g.choice_p(&p) as i64).collect();
            out.put("draws", Array::from_i64(&[n], &draws));
        }
        "standard_normal" => {
            let mut g = generator(input)?;
            let n = input.i64("n")? as usize;
            let draws: Vec<f64> = (0..n).map(|_| g.standard_normal()).collect();
            out.put("draws", Array::from_f64(&[n], &draws));
            out.put("after", Array::from_u64(&[2], &[(g.state >> 64) as u64, g.state as u64]));
        }
        "normal" => {
            let mut g = generator(input)?;
            let (loc, scale, n) = (input.f64("loc")?, input.f64("scale")?, input.i64("n")? as usize);
            let draws: Vec<f64> = (0..n).map(|_| g.normal(loc, scale)).collect();
            out.put("draws", Array::from_f64(&[n], &draws));
        }
        "uniform" => {
            let mut g = generator(input)?;
            let (low, high, n) = (input.f64("low")?, input.f64("high")?, input.i64("n")? as usize);
            let draws: Vec<f64> = (0..n).map(|_| g.uniform(low, high)).collect();
            out.put("draws", Array::from_f64(&[n], &draws));
        }
        "permutation" => {  // permutations of 0..m for each m of "sizes", one after another
            let mut g = generator(input)?;
            let mut draws = Vec::new();
            for m in input.i64s("sizes")? {
                draws.extend(g.permutation(m as usize));
            }
            out.put("draws", Array::from_i64(&[draws.len()], &draws));
        }
        "world_mixed" => {  // the draws a world's moment interleaves: a permutation of the biters,
            // a plant's regrowth, two normals a word heard, a uniform, an integer
            let mut g = generator(input)?;
            let n = input.i64("n")? as usize;
            let mut draws = Vec::new();
            for k in 0..n {
                draws.extend(g.permutation(k % 11).into_iter().map(|v| v as f64));
                draws.push(g.random());
                draws.push(g.normal(0.0, 0.25));
                draws.push(g.normal(0.0, 0.25));
                draws.push(g.uniform(1.0, 23.0));
                draws.push(g.integers(0, (k % 5) as i64 + 1) as f64);
                draws.push(g.standard_normal());
            }
            out.put("draws", Array::from_f64(&[draws.len()], &draws));
        }
        "seeded" => {  // `default_rng(tuple of integers)`: its state, then draws as "mixed"
            let g0 = rng::Pcg64::from_entropy(&input.u64s("entropy")?);
            out.put("state", Array::from_u64(&[2], &[(g0.state >> 64) as u64, g0.state as u64]));
            out.put("inc", Array::from_u64(&[2], &[(g0.inc >> 64) as u64, g0.inc as u64]));
            let mut g = g0;
            let n = input.i64("n")? as usize;
            let mut draws = Vec::new();
            for k in 0..n {
                draws.push(g.integers(0, (k % 9) as i64 + 1) as f64);
                draws.push(g.random());
            }
            out.put("draws", Array::from_f64(&[draws.len()], &draws));
        }
        "mixed" => {  // the draws a moment interleaves: random, integers, a 32-bit buffer
            let mut g = generator(input)?;
            let n = input.i64("n")? as usize;
            let mut draws = Vec::new();
            for k in 0..n {
                draws.push(g.random());
                draws.push(g.integers(0, (k % 7) as i64 + 2) as f64);
            }
            out.put("draws", Array::from_f64(&[draws.len()], &draws));
        }
        "sum" => {
            let a = input.f64s("a")?;
            out.put("sum", Array::scalar_f64(np::sum(&a)));
            out.put("dot", Array::scalar_f64(np::dot(&a, &a)));
            out.put("mean", Array::scalar_f64(np::mean(&a)));
        }
        "elementwise" => {
            let (a, b) = (input.f64s("a")?, input.f64s("b")?);
            let (lo, hi) = (input.f64s("lo")?, input.f64s("hi")?);
            let n = a.len();
            let pick = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..n).map(f).collect() };
            out.put("maximum", Array::from_f64(&[n], &pick(&|i| np::maximum(a[i], b[i]))));
            out.put("minimum", Array::from_f64(&[n], &pick(&|i| np::minimum(a[i], b[i]))));
            out.put("clip", Array::from_f64(&[n], &pick(&|i| np::clip(a[i], lo[i], hi[i]))));
            out.put("max", Array::scalar_f64(np::max(&a)));
        }
        "sum_rows" => {  // a.sum(axis=0) of rows of `cols`
            let a = input.f64s("a")?;
            let cols = input.i64("cols")? as usize;
            let sums = np::sum_rows(&a, cols);
            out.put("sums", Array::from_f64(&[sums.len()], &sums));
        }
        "tanh" => {  // np.tanh as NumPy's ufunc computes it, and the C library's
            let a = input.f64s("a")?;
            let ufunc: Vec<f64> = a.iter().map(|&x| np::tanh_ufunc(x)).collect();
            let libm: Vec<f64> = a.iter().map(|&x| np::tanh(x)).collect();
            out.put("ufunc", Array::from_f64(&[a.len()], &ufunc));
            out.put("libm", Array::from_f64(&[a.len()], &libm));
        }
        "einsum" => {  // each row of a against the same row of b, in both of einsum's orders
            let (rows, cols) = (input.i64("rows")? as usize, input.i64("cols")? as usize);
            let (a, b) = (input.f64s("a")?, input.f64s("b")?);
            if a.len() != rows * cols || b.len() != rows * cols {
                return Err(state::Error::Shape("einsum".into()));
            }
            let row = |x: &[f64], r: usize| x[r * cols..(r + 1) * cols].to_vec();
            let dot: Vec<f64> = (0..rows).map(|r| np::einsum_dot(&row(&a, r), &row(&b, r))).collect();
            let fused: Vec<f64> =
                (0..rows).map(|r| np::einsum_fused(&row(&a, r), &row(&b, r))).collect();
            out.put("dot", Array::from_f64(&[rows], &dot));
            out.put("fused", Array::from_f64(&[rows], &fused));
        }
        "argpartition" => {
            let (rows, cols) = (input.i64("rows")? as usize, input.i64("cols")? as usize);
            let v = input.f64s("v")?;
            let kth = input.i64("kth")? as usize;
            let mut all = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                all.extend(np::argpartition(&v[r * cols..(r + 1) * cols], kth)
                           .into_iter().map(|i| i as i64));
            }
            out.put("index", Array::from_i64(&[rows, cols], &all));
        }
        "matmul" => {
            let (a, b) = (input.get("a")?, input.get("b")?);
            let (n, k, m) = match (a.shape.as_slice(), b.shape.as_slice()) {
                (&[n, k], &[k2, m]) if k == k2 => (n, k, m),
                _ => return Err(state::Error::Shape("matmul".into())),
            };
            let product = np::matmul(&input.f64s("a")?, n, k, &input.f64s("b")?, m);
            out.put("product", Array::from_f64(&[n, m], &product));
        }
        "orders" => {  // the recall memory's: a run's maximum, running sum and stable order
            let v = input.f64s("v")?;
            let (x, y) = (input.f64s("x")?, input.f64s("y")?);
            out.put("max", Array::scalar_f64(np::max(&v)));
            out.put("cumsum", Array::from_f64(&[v.len()], &np::cumsum(&v)));
            let order: Vec<i64> = np::argsort_stable(&v).into_iter().map(|i| i as i64).collect();
            out.put("argsort", Array::from_i64(&[v.len()], &order));
            let both: Vec<f64> = x.iter().zip(&y).map(|(&a, &b)| np::logaddexp(a, b)).collect();
            out.put("logaddexp", Array::from_f64(&[both.len()], &both));
        }
        _ => return Err(state::Error::Missing(name.to_string())),
    }
    Ok(out)
}

/// One call of a part's method (`parts::call`) on its state and arguments, both named arrays.
/// Returns 0 and the state after the call with what it returned, or a negative code.
///
/// # Safety
/// The pointers are the caller's buffers of the given lengths.
#[no_mangle]
pub unsafe extern "C" fn core_call(
    part: *const u8, part_len: usize, method: *const u8, method_len: usize, state: *const u8,
    state_len: usize, args: *const u8, args_len: usize, out: *mut *mut u8, out_len: *mut usize,
) -> i32 {
    let text = |p: *const u8, n: usize| std::str::from_utf8(std::slice::from_raw_parts(p, n));
    let (part, method) = match (text(part, part_len), text(method, method_len)) {
        (Ok(a), Ok(b)) => (a, b),
        _ => return -1,
    };
    let state = match Named::read(std::slice::from_raw_parts(state, state_len)) {
        Ok(x) => x,
        Err(_) => return -2,
    };
    let args = match Named::read(std::slice::from_raw_parts(args, args_len)) {
        Ok(x) => x,
        Err(_) => return -2,
    };
    match parts::call(part, method, &state, &args) {
        Ok(named) => {
            give(named.write(), out, out_len);
            0
        }
        Err(e) if e.refused() => -5,
        Err(_) => -3,
    }
}

// -- brains held between moments: a handle each, their state in the core ---------------------

use std::sync::Mutex;

static BRAINS: Mutex<Vec<Option<parts::brain::Brain>>> = Mutex::new(Vec::new());

/// A brain from its state (named arrays, `cadence.core.parts.brain.state_of`): its handle (0 or
/// more), or a negative code.
///
/// # Safety
/// `state` is the caller's buffer of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn core_brain_new(state: *const u8, len: usize) -> i64 {
    let named = match Named::read(std::slice::from_raw_parts(state, len)) {
        Ok(x) => x,
        Err(_) => return -2,
    };
    let brain = match parts::brain::Brain::import(&named) {
        Ok(x) => x,
        Err(_) => return -3,
    };
    let mut held = BRAINS.lock().unwrap();
    if let Some(free) = held.iter().position(|b| b.is_none()) {
        held[free] = Some(brain);
        return free as i64;
    }
    held.push(Some(brain));
    held.len() as i64 - 1
}

/// One method of a held brain (`live`, ...) on named arguments; 0 and what it returned, or a
/// negative code: -5 a refusal the Python raises as `ValueError` or `RuntimeError`, -4 a handle
/// that holds no brain, -3 a fault.
///
/// # Safety
/// The pointers are the caller's buffers of the given lengths.
#[no_mangle]
pub unsafe extern "C" fn core_brain_call(
    handle: i64, method: *const u8, method_len: usize, args: *const u8, args_len: usize,
    out: *mut *mut u8, out_len: *mut usize,
) -> i32 {
    let method = match std::str::from_utf8(std::slice::from_raw_parts(method, method_len)) {
        Ok(x) => x,
        Err(_) => return -1,
    };
    let args = match Named::read(std::slice::from_raw_parts(args, args_len)) {
        Ok(x) => x,
        Err(_) => return -2,
    };
    let mut held = BRAINS.lock().unwrap();
    let brain = match held.get_mut(handle as usize).and_then(|b| b.as_mut()) {
        Some(b) if handle >= 0 => b,
        _ => return -4,
    };
    let mut named = Named::new();
    match brain.call(method, &args, &mut named) {
        Ok(()) => {
            give(named.write(), out, out_len);
            0
        }
        Err(e) if e.refused() => -5,  // the Python's ValueError or RuntimeError
        Err(_) => -3,
    }
}

/// A held brain's whole state (named arrays), for a checkpoint, an inheritance or a panel.
///
/// # Safety
/// `out` and `out_len` receive the caller's bytes, released with `core_release`.
#[no_mangle]
pub unsafe extern "C" fn core_brain_export(handle: i64, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    let held = BRAINS.lock().unwrap();
    match held.get(handle as usize).and_then(|b| b.as_ref()) {
        Some(b) if handle >= 0 => {
            let mut named = Named::new();
            b.export(&mut named);
            give(named.write(), out, out_len);
            0
        }
        _ => -4,
    }
}

/// Let a held brain go.
#[no_mangle]
pub extern "C" fn core_brain_drop(handle: i64) {
    let mut held = BRAINS.lock().unwrap();
    if handle >= 0 {
        if let Some(slot) = held.get_mut(handle as usize) {
            *slot = None;
        }
    }
}
