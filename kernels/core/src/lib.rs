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

/// The kernels' scratch, from Rust's allocator: the size kept ahead of the room.
pub(crate) unsafe fn host_malloc(n: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(n + 16, 16).unwrap();
    let p = std::alloc::alloc(layout);
    if p.is_null() {
        return p;
    }
    *(p as *mut usize) = n + 16;
    p.add(16)
}

pub(crate) unsafe fn host_free(p: *mut u8) {
    if p.is_null() {
        return;
    }
    let base = p.sub(16);
    let n = *(base as *mut usize);
    std::alloc::dealloc(base, std::alloc::Layout::from_size_align(n, 16).unwrap());
}

pub(crate) fn trap() -> ! {
    std::process::abort()
}

/// The kernels' `exp`, `tanh` and square root: musl's, as the `libm` crate carries them, in the
/// browser and natively alike.
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}
#[inline(always)]
pub(crate) fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}
#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
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
        Err(_) => -3,
    }
}
