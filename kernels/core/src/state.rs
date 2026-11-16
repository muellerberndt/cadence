//! Named arrays: the one format a brain's state, a moment's inputs and its replies cross in.
//!
//! Little-endian: the magic `CAD1`, the entry count (u32), then each entry: its name's length
//! (u16) and name (UTF-8), its type (u8: 1 float64, 2 int64, 3 bool, 4 uint64), its dimension
//! count (u8) and dimensions (u64 each), and its data, padded to a multiple of eight bytes.
//! Entries keep their order, so an export read back and written again is the same bytes.

use std::collections::BTreeMap;

pub const MAGIC: &[u8; 4] = b"CAD1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F64 = 1,
    I64 = 2,
    Bool = 3,
    U64 = 4,
}

impl Kind {
    fn of(code: u8) -> Option<Kind> {
        match code {
            1 => Some(Kind::F64),
            2 => Some(Kind::I64),
            3 => Some(Kind::Bool),
            4 => Some(Kind::U64),
            _ => None,
        }
    }
    fn size(self) -> usize {
        match self {
            Kind::Bool => 1,
            _ => 8,
        }
    }
}

/// One named array: its type, shape and bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    pub kind: Kind,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Array {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn f64s(&self) -> Option<Vec<f64>> {
        (self.kind == Kind::F64).then(|| {
            self.bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
        })
    }
    pub fn i64s(&self) -> Option<Vec<i64>> {
        (self.kind == Kind::I64).then(|| {
            self.bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
        })
    }
    pub fn u64s(&self) -> Option<Vec<u64>> {
        (self.kind == Kind::U64).then(|| {
            self.bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
        })
    }
    pub fn bools(&self) -> Option<Vec<bool>> {
        (self.kind == Kind::Bool).then(|| self.bytes.iter().map(|&b| b != 0).collect())
    }
    pub fn from_f64(shape: &[usize], data: &[f64]) -> Array {
        Array { kind: Kind::F64, shape: shape.to_vec(),
                bytes: data.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn from_i64(shape: &[usize], data: &[i64]) -> Array {
        Array { kind: Kind::I64, shape: shape.to_vec(),
                bytes: data.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn from_u64(shape: &[usize], data: &[u64]) -> Array {
        Array { kind: Kind::U64, shape: shape.to_vec(),
                bytes: data.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn from_bool(shape: &[usize], data: &[bool]) -> Array {
        Array { kind: Kind::Bool, shape: shape.to_vec(), bytes: data.iter().map(|&b| b as u8).collect() }
    }
    pub fn scalar_f64(x: f64) -> Array {
        Array::from_f64(&[], &[x])
    }
    pub fn scalar_i64(x: i64) -> Array {
        Array::from_i64(&[], &[x])
    }
}

/// Named arrays in the order they were written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Named {
    pub order: Vec<String>,
    pub arrays: BTreeMap<String, Array>,
}

#[derive(Debug, PartialEq)]
pub enum Error {
    Truncated,
    Magic,
    Kind(u8),
    Name,
    Missing(String),
    Shape(String),
}

impl Named {
    pub fn new() -> Named {
        Named::default()
    }
    pub fn put(&mut self, name: &str, array: Array) {
        if !self.arrays.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.arrays.insert(name.to_string(), array);
    }
    pub fn get(&self, name: &str) -> Result<&Array, Error> {
        self.arrays.get(name).ok_or_else(|| Error::Missing(name.to_string()))
    }
    pub fn has(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }
    pub fn f64s(&self, name: &str) -> Result<Vec<f64>, Error> {
        self.get(name)?.f64s().ok_or_else(|| Error::Shape(name.to_string()))
    }
    pub fn i64s(&self, name: &str) -> Result<Vec<i64>, Error> {
        self.get(name)?.i64s().ok_or_else(|| Error::Shape(name.to_string()))
    }
    pub fn u64s(&self, name: &str) -> Result<Vec<u64>, Error> {
        self.get(name)?.u64s().ok_or_else(|| Error::Shape(name.to_string()))
    }
    pub fn bools(&self, name: &str) -> Result<Vec<bool>, Error> {
        self.get(name)?.bools().ok_or_else(|| Error::Shape(name.to_string()))
    }
    pub fn f64(&self, name: &str) -> Result<f64, Error> {
        let v = self.f64s(name)?;
        (v.len() == 1).then(|| v[0]).ok_or_else(|| Error::Shape(name.to_string()))
    }
    pub fn i64(&self, name: &str) -> Result<i64, Error> {
        let v = self.i64s(name)?;
        (v.len() == 1).then(|| v[0]).ok_or_else(|| Error::Shape(name.to_string()))
    }
    pub fn shape(&self, name: &str) -> Result<Vec<usize>, Error> {
        Ok(self.get(name)?.shape.clone())
    }

    pub fn read(bytes: &[u8]) -> Result<Named, Error> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], Error> {
            let end = at.checked_add(n).ok_or(Error::Truncated)?;
            let part = bytes.get(at..end).ok_or(Error::Truncated)?;
            at = end;
            Ok(part)
        };
        if take(4)? != MAGIC {
            return Err(Error::Magic);
        }
        let count = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut out = Named::new();
        for _ in 0..count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().unwrap()) as usize;
            let name = std::str::from_utf8(take(name_len)?).map_err(|_| Error::Name)?.to_string();
            let code = take(1)?[0];
            let kind = Kind::of(code).ok_or(Error::Kind(code))?;
            let ndim = take(1)?[0] as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize);
            }
            let n: usize = shape.iter().product::<usize>() * kind.size();
            let data = take(n)?.to_vec();
            let pad = (8 - (2 + name_len + 2 + 8 * ndim + n) % 8) % 8;
            take(pad)?;
            out.put(&name, Array { kind, shape, bytes: data });
        }
        Ok(out)
    }

    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.order.len() as u32).to_le_bytes());
        for name in &self.order {
            let a = &self.arrays[name];
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.push(a.kind as u8);
            out.push(a.shape.len() as u8);
            for &d in &a.shape {
                out.extend_from_slice(&(d as u64).to_le_bytes());
            }
            out.extend_from_slice(&a.bytes);
            let pad = (8 - (2 + name.len() + 2 + 8 * a.shape.len() + a.bytes.len()) % 8) % 8;
            out.extend(std::iter::repeat(0u8).take(pad));
        }
        out
    }
}
