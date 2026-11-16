//! What the kernels share whichever library holds them: NumPy's array object read at its
//! address, views of one or two dimensions, room for a kernel's scratch, and the arithmetic
//! helpers. `kernels/wasm` (the module for Pyodide) and `kernels/core` (the native core) both
//! include this file; each defines `host_malloc`, `host_free`, `trap`, `exp`, `tanh` and
//! `sqrt` at its root.

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

// NumPy's array object and descriptor as far as the kernels read them: pointer-sized fields,
// so one layout serves the 32-bit module and a 64-bit machine alike
#[repr(C)]
pub(crate) struct PyArrayObject {
    ob_refcnt: isize,
    ob_type: *const u8,
    pub(crate) data: *mut u8,
    nd: i32,
    dimensions: *const isize,
    strides: *const isize,
    base: *const u8,
    descr: *const PyArrayDescr,
    flags: i32,
}

#[repr(C)]
struct PyArrayDescr {
    ob_refcnt: isize,
    ob_type: *const u8,
    typeobj: *const u8,
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
pub(crate) unsafe fn object(id: usize) -> Option<(&'static PyArrayObject, i32, usize, usize)> {
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
pub(crate) unsafe fn view<T>(id: usize, num: i32) -> Option<View<T>> {
    let (a, type_num, n0, n1) = object(id)?;
    // C's long is 32 bits in the module and 64 bits on a 64-bit machine
    let same = type_num == num
        || (num == if cfg!(target_pointer_width = "32") { INT32 } else { INT64 } && type_num == LONG32);
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

pub(crate) unsafe fn index(id: usize) -> Option<Index> {
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
        let base = crate::host_malloc(size);
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
            crate::trap();
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
        unsafe { crate::host_free(self.base) }
    }
}

// -- the module's own checks: what the Python side reads before it hands arrays over

