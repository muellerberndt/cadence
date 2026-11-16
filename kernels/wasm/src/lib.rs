//! The compiled kernels of `cadence.fused` as one WebAssembly module, for the library under
//! Pyodide, where Numba is not available.
//!
//! Each export is one Numba kernel of `cadence/fused.py`, under its name, with the same
//! arguments and the same arithmetic in the same order: the Python of `fused.py` stays the
//! reference and the glue, and `cadence/_wasm.py` hands each kernel's arrays over. The module
//! runs inside Pyodide's own linear memory: it imports Pyodide's memory, its stack pointer,
//! `malloc` and `free` for scratch, and the C library's `exp`, the function NumPy calls. A
//! NumPy array is passed as its object's address (`id(array)` in CPython), and the kernel reads
//! the data pointer, shape, flags and type from NumPy's array object, so nothing is copied.
//! The module holds no data of its own and no static state: everything a kernel keeps lives in
//! the arrays it is handed.
//!
//! The same source builds natively (`cargo build --release` in `kernels/wasm`: a shared
//! library of the same exports, with the C library's `malloc` and `free`, and `exp` and `tanh`
//! from the `libm` crate), for a Python that hands it NumPy's arrays the same way.
//!
//! A block product sums each output from 0.0 in the order of its inputs, as NumPy's matrix
//! product without BLAS does (Pyodide's NumPy has none); the pairwise sums of the apical
//! compartment follow NumPy's tree. With the same `exp`, every kernel that keeps the NumPy
//! loop's bits in `fused.py` keeps them here too.

#![no_std]
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

mod bound;
mod fast;
mod lesson;
mod map;
mod settle;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

#[cfg(not(target_arch = "wasm32"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    trap()
}

/// Stop where a kernel's own room would overflow: a trap in the module, an abort natively.
#[cfg(target_arch = "wasm32")]
pub(crate) fn trap() -> ! {
    core::arch::wasm32::unreachable()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn trap() -> ! {
    #[cfg_attr(target_os = "macos", link(name = "System"))]
    #[cfg_attr(target_os = "linux", link(name = "c"))]
    extern "C" {
        fn abort() -> !;
    }
    unsafe { abort() }
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    #[link_name = "host_exp"]
    fn host_exp(x: f64) -> f64;
    #[link_name = "host_tanh"]
    fn host_tanh(x: f64) -> f64;
    #[link_name = "host_malloc"]
    fn host_malloc(n: usize) -> *mut u8;
    #[link_name = "host_free"]
    fn host_free(p: *mut u8);
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(target_os = "macos", link(name = "System"))]
#[cfg_attr(target_os = "linux", link(name = "c"))]
extern "C" {
    #[link_name = "malloc"]
    fn host_malloc(n: usize) -> *mut u8;
    #[link_name = "free"]
    fn host_free(p: *mut u8);
}

/// The C library's `exp`: in a browser the one NumPy calls under Pyodide (musl's), natively
/// musl's as the `libm` crate carries it.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    unsafe { host_exp(x) }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn tanh(x: f64) -> f64 {
    unsafe { host_tanh(x) }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(crate) fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

/// The square root, correctly rounded on every machine.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    use core::arch::wasm32::{f64x2_extract_lane, f64x2_splat, f64x2_sqrt};
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

mod common;
pub(crate) use common::*;

/// The data pointer of the array at `id`, or 0 where the module would not take it as an array
/// of type number `num`: the Python side compares it with NumPy's own before the kernels are
/// used.
#[no_mangle]
pub unsafe extern "C" fn array_data(id: usize, num: i32) -> usize {
    match object(id) {
        Some((a, type_num, _, _)) if type_num == num => a.data as usize,
        _ => 0,
    }
}

/// The two dimensions the module reads of the array at `id` (rows times 65536 plus columns,
/// for small test arrays).
#[no_mangle]
pub unsafe extern "C" fn array_shape(id: usize) -> u32 {
    match object(id) {
        Some((_, _, n0, n1)) => (n0 as u32) * 65536 + n1 as u32,
        _ => 0,
    }
}

/// The version of the kernels' argument lists, which `cadence/_wasm.py` checks.
#[no_mangle]
pub extern "C" fn kernel_version() -> i32 {
    1
}
