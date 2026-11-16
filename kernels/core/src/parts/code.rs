//! `cadence.stream.PatternSeparator`: a key expanded through a fixed random projection (a sparse
//! fan-in, read in both signs with `cover`, or a dense matrix) and the strongest winners kept.
//! The fan-in's sums run in `einsum`'s order, the winners are chosen as NumPy's introselect
//! chooses them.

use super::trace::{flag, Rows};
use crate::np;
use crate::state::{Array, Error, Named};

#[derive(Clone, Debug, PartialEq)]
pub struct Code {
    pub inputs: i64,
    pub expansion: i64,
    pub winners: i64,
    pub fan_in: i64,
    pub cover: i64,
    pub center: f64,
    /// With a fan-in: each code neuron's channels and their weights, `(expansion, fan_in)`.
    pub index: Vec<i64>,
    pub weight: Vec<f64>,
    /// Without: the dense `(inputs, expansion)` projection.
    pub projection: Vec<f64>,
    pub mean: Vec<f64>,
}

impl Code {
    pub fn import(s: &Named) -> Result<Code, Error> {
        let fan_in = s.i64("config.fan_in")?;
        let (inputs, expansion) = (s.i64("config.inputs")?, s.i64("config.expansion")?);
        let mut held = Code {
            inputs,
            expansion,
            winners: s.i64("config.winners")?,
            fan_in,
            cover: s.i64("config.cover")?,
            center: s.f64("config.center")?,
            index: Vec::new(),
            weight: Vec::new(),
            projection: Vec::new(),
            mean: s.f64s("mean")?,
        };
        if inputs < 1 || expansion < 1 || held.winners < 1 || fan_in < 0 || held.cover < 0
            || held.mean.len() != inputs as usize
        {
            return Err(Error::Shape("a separator's sizes".into()));
        }
        let (n, e) = (inputs as usize, expansion as usize);
        if fan_in > 0 {
            held.index = s.i64s("index")?;
            held.weight = s.f64s("weight")?;
            let k = fan_in as usize;
            if s.shape("index")? != [e, k] || s.shape("weight")? != [e, k] {
                return Err(Error::Shape("index and weight are (expansion, fan_in)".into()));
            }
        } else {
            held.projection = s.f64s("projection")?;
            if s.shape("projection")? != [n, e] {
                return Err(Error::Shape("projection is (inputs, expansion)".into()));
            }
        }
        Ok(held)
    }

    pub fn export(&self, out: &mut Named) {
        for (name, value) in [
            ("config.inputs", self.inputs), ("config.expansion", self.expansion),
            ("config.winners", self.winners), ("config.fan_in", self.fan_in),
            ("config.cover", self.cover),
        ] {
            out.put(name, Array::scalar_i64(value));
        }
        out.put("config.center", Array::scalar_f64(self.center));
        let e = self.expansion as usize;
        if self.fan_in > 0 {
            let k = self.fan_in as usize;
            out.put("index", Array::from_i64(&[e, k], &self.index));
            out.put("weight", Array::from_f64(&[e, k], &self.weight));
        } else {
            out.put("projection", Array::from_f64(&[self.inputs as usize, e], &self.projection));
        }
        out.put("mean", Array::from_f64(&[self.mean.len()], &self.mean));
    }

    /// `code(key, learn)`: the sparse code of `(batch, inputs)` keys, `(batch, expansion)` with
    /// `winners` entries kept per row; `learn` moves the running mean (with `center`).
    pub fn code(&mut self, x: &Rows, learn: bool) -> Result<Rows, Error> {
        let n = self.inputs as usize;
        if x.cols != n || !x.data.iter().all(|v| v.is_finite()) {
            return Err(Error::Shape(format!("key must be a finite (batch, {n}) array")));
        }
        let (b, e) = (x.rows, self.expansion as usize);
        let mut x = x.clone();
        let mut mean = self.mean.clone();
        if self.center > 0.0 {
            if learn && b > 0 {
                // (x / len(x)).sum(axis=0): each column summed down the rows from zero
                let mut column = vec![0.0f64; n];
                for r in 0..b {
                    for (j, s) in column.iter_mut().enumerate() {
                        *s += x.at(r, j) / b as f64;
                    }
                }
                for (m, s) in mean.iter_mut().zip(&column) {
                    *m = self.center * *m + (1.0 - self.center) * s;
                }
            }
            for r in 0..b {
                for j in 0..n {
                    x.data[r * n + j] -= mean[j];
                }
            }
        }
        let mut drive = Rows::zeros(b, e);
        if self.fan_in > 0 {
            // the ON channels, then the OFF channels, with cover
            let channels = if self.cover > 0 { 2 * n } else { n };
            let k = self.fan_in as usize;
            if self.index.iter().any(|&i| i < 0 || i as usize >= channels) {
                return Err(Error::Shape("an index outside the channels".into()));
            }
            let mut read = vec![0.0f64; channels];
            let mut gathered = vec![0.0f64; k];
            for r in 0..b {
                let row = x.row(r);
                if self.cover > 0 {
                    for j in 0..n {
                        read[j] = np::maximum(row[j], 0.0);
                        read[n + j] = np::maximum(-row[j], 0.0);
                    }
                } else {
                    read.copy_from_slice(row);
                }
                for u in 0..e {
                    for (g, &i) in gathered.iter_mut().zip(&self.index[u * k..(u + 1) * k]) {
                        *g = read[i as usize];
                    }
                    let w = &self.weight[u * k..(u + 1) * k];
                    // a batch of one sums each neuron's fan-in contiguously; a larger batch's
                    // gather runs along the batch, and sums the fan-in fused in order
                    let s = if b == 1 {
                        np::einsum_dot(&gathered, w)
                    } else {
                        np::einsum_fused(&gathered, w)
                    };
                    drive.data[r * e + u] = np::maximum(s, 0.0);
                }
            }
        } else {
            // x @ projection: BLAS natively, in order here
            let mut column = vec![0.0f64; n];
            for u in 0..e {
                for (i, c) in column.iter_mut().enumerate() {
                    *c = self.projection[i * e + u];
                }
                for r in 0..b {
                    drive.data[r * e + u] = np::maximum(np::dot(x.row(r), &column), 0.0);
                }
            }
        }
        if !mean.iter().all(|v| v.is_finite()) || !drive.data.iter().all(|v| v.is_finite()) {
            return Err(Error::Shape("pattern separation overflowed; scale the input keys".into()));
        }
        if learn {
            self.mean = mean;
        }
        let winners = self.winners as usize;
        if winners >= e {
            return Ok(drive);
        }
        let mut out = Rows::zeros(b, e);
        let mut negative = vec![0.0f64; e];
        for r in 0..b {
            let row = drive.row(r);
            for (m, v) in negative.iter_mut().zip(row) {
                *m = -v;
            }
            for &u in &np::argpartition(&negative, winners - 1)[..winners] {
                out.data[r * e + u] = row[u];
            }
        }
        Ok(out)
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "code" => {
                let key = Rows::read(args, "key").map_err(|_| {
                    Error::Shape(format!("key must be a finite (batch, {}) array", self.inputs))
                })?;
                let code = self.code(&key, flag(args, "learn")?)?;
                out.put("code", code.array());
            }
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
