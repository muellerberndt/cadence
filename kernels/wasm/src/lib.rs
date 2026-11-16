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
//! A block product sums each output from 0.0 in the order of its inputs, as NumPy's matrix
//! product without BLAS does (Pyodide's NumPy has none); the pairwise sums of the apical
//! compartment follow NumPy's tree. With the same `exp`, every kernel that keeps the NumPy
//! loop's bits in `fused.py` keeps them here too.

#![no_std]
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

mod fast;
mod lesson;
mod map;
mod settle;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

#[link(wasm_import_module = "env")]
extern "C" {
    #[link_name = "host_exp"]
    fn host_exp(x: f64) -> f64;
    #[link_name = "host_malloc"]
    fn host_malloc(n: usize) -> *mut u8;
    #[link_name = "host_free"]
    fn host_free(p: *mut u8);
}

/// The C library's exponential, the one NumPy's `np.exp` calls.
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    unsafe { host_exp(x) }
}

#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    use core::arch::wasm32::{f64x2_extract_lane, f64x2_splat, f64x2_sqrt};
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

#[inline(always)]
pub(crate) fn fabs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & !(1u64 << 63))
}

/// Python's `max(a, b)`: `a` unless `b > a`.
#[inline(always)]
pub(crate) fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: `a` unless `b < a`.
#[inline(always)]
pub(crate) fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `np.isfinite(x)`.
#[inline(always)]
pub(crate) fn finite(x: f64) -> bool {
    x - x == 0.0
}

// -- NumPy's array object, as its C structs lay it out on wasm32 (NumPy 2, CPython 3)

#[repr(C)]
struct PyArrayObject {
    ob_refcnt: i32,
    ob_type: u32,
    data: *mut u8,
    nd: i32,
    dimensions: *const i32,
    strides: *const i32,
    base: u32,
    descr: *const PyArrayDescr,
    flags: i32,
}

#[repr(C)]
struct PyArrayDescr {
    ob_refcnt: i32,
    ob_type: u32,
    typeobj: u32,
    kind: u8,
    type_char: u8,
    byteorder: u8,
    former_flags: u8,
    type_num: i32,
}

const C_CONTIGUOUS: i32 = 0x0001;
const ALIGNED: i32 = 0x0100;
// NumPy's type numbers on wasm32, where `long` holds 32 bits
pub(crate) const BOOL: i32 = 0;
pub(crate) const INT32: i32 = 5;
pub(crate) const LONG32: i32 = 7;
pub(crate) const INT64: i32 = 9;
pub(crate) const FLOAT64: i32 = 12;

/// A C-contiguous, aligned array of one or two dimensions, read from its NumPy object.
#[derive(Clone, Copy)]
pub(crate) struct View<T> {
    pub p: *mut T,
    pub n0: usize,
    pub n1: usize,
    pub len: usize,
}

impl<T: Copy> View<T> {
    #[inline(always)]
    pub unsafe fn get(&self, i: usize) -> T {
        *self.p.add(i)
    }
    #[inline(always)]
    pub unsafe fn set(&self, i: usize, x: T) {
        *self.p.add(i) = x
    }
    #[inline(always)]
    pub unsafe fn at(&self, b: usize, i: usize) -> T {
        *self.p.add(b * self.n1 + i)
    }
    #[inline(always)]
    pub unsafe fn put(&self, b: usize, i: usize, x: T) {
        *self.p.add(b * self.n1 + i) = x
    }
}

/// The array object at `id` and its type number, or None where it is not a C-contiguous,
/// aligned, native-order array of at most two dimensions.
unsafe fn object(id: u32) -> Option<(&'static PyArrayObject, i32, usize, usize)> {
    if id == 0 {
        return None;
    }
    let a = &*(id as *const PyArrayObject);
    if a.flags & (C_CONTIGUOUS | ALIGNED) != (C_CONTIGUOUS | ALIGNED) || a.descr.is_null() {
        return None;
    }
    let d = &*a.descr;
    if d.byteorder == b'>' {
        return None;
    }
    let (n0, n1) = match a.nd {
        0 => (1usize, 1usize),
        1 => (*a.dimensions as usize, 1usize),
        2 => (*a.dimensions as usize, *a.dimensions.add(1) as usize),
        _ => return None,
    };
    Some((a, d.type_num, n0, n1))
}

/// The array at `id` as a view of `T`, where its type number is `num`.
pub(crate) unsafe fn view<T>(id: u32, num: i32) -> Option<View<T>> {
    let (a, type_num, n0, n1) = object(id)?;
    let same = type_num == num || (num == INT32 && type_num == LONG32);
    if !same {
        return None;
    }
    Some(View { p: a.data as *mut T, n0, n1, len: n0 * n1 })
}

/// An index array of 64-bit or 32-bit integers.
#[derive(Clone, Copy)]
pub(crate) enum Index {
    Wide(View<i64>),
    Narrow(View<i32>),
}

impl Index {
    #[inline(always)]
    pub unsafe fn get(&self, i: usize) -> i64 {
        match self {
            Index::Wide(v) => v.get(i),
            Index::Narrow(v) => v.get(i) as i64,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Index::Wide(v) => v.len,
            Index::Narrow(v) => v.len,
        }
    }
}

pub(crate) unsafe fn index(id: u32) -> Option<Index> {
    if let Some(v) = view::<i64>(id, INT64) {
        return Some(Index::Wide(v));
    }
    view::<i32>(id, INT32).map(Index::Narrow)
}

/// The return code of a kernel handed an array it does not take: argument `k` (from 0).
#[inline(always)]
pub(crate) fn refused(k: i32) -> i32 {
    -1 - k
}

/// A kernel's scratch: one block from the host's `malloc`, freed when dropped, handed out in
/// eight-byte aligned pieces.
pub(crate) struct Scratch {
    base: *mut u8,
    at: usize,
    size: usize,
}

pub(crate) const NO_ROOM: i32 = -1000;

impl Scratch {
    pub unsafe fn new(bytes: usize) -> Option<Scratch> {
        let size = (bytes + 64) & !7;
        let base = host_malloc(size);
        if base.is_null() {
            return None;
        }
        Some(Scratch { base, at: 0, size })
    }
    /// `count` entries of `T`, zeroed.
    pub unsafe fn take<T: Copy>(&mut self, count: usize) -> *mut T {
        let bytes = (count * core::mem::size_of::<T>() + 7) & !7;
        let p = self.base.add(self.at);
        if self.at + bytes > self.size {
            core::arch::wasm32::unreachable();
        }
        self.at += bytes;
        core::ptr::write_bytes(p, 0, bytes);
        p as *mut T
    }
}

/// Bytes `count` entries of `T` take in a `Scratch`.
#[inline(always)]
pub(crate) fn room<T>(count: usize) -> usize {
    (count * core::mem::size_of::<T>() + 7) & !7
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { host_free(self.base) }
    }
}

// -- the module's own checks: what the Python side reads before it hands arrays over

/// The data pointer of the array at `id`, or 0 where the module would not take it as an array
/// of type number `num`: the Python side compares it with NumPy's own before the kernels are
/// used.
#[no_mangle]
pub unsafe extern "C" fn array_data(id: u32, num: i32) -> u32 {
    match object(id) {
        Some((a, type_num, _, _)) if type_num == num => a.data as u32,
        _ => 0,
    }
}

/// The two dimensions the module reads of the array at `id` (rows times 65536 plus columns,
/// for small test arrays).
#[no_mangle]
pub unsafe extern "C" fn array_shape(id: u32) -> u32 {
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
