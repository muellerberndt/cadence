//! `cadence.memory.SynapticMemory` (with its base `cadence.stream.FastSynapses`): the
//! associative store of place records, two timescales of normalized delta plasticity and the
//! surprise gate, as the Python computes them, in its order. A product of a matrix, a contraction
//! (`einsum`) and a dot product run in order through `crate::np`. The core holds a store keyed
//! on its pre neurons (no separator) that does not fade; another is refused.

use crate::np;
use crate::state::{refused, Array, Error, Named};

const GATE_WIDTH: f64 = 2.0;
const GATE_RATE: f64 = 0.05;

#[derive(Clone, Debug, PartialEq)]
pub struct Synaptic {
    pub decay: f64,
    pub rate: f64,
    pub amplitude: f64,
    pub consolidation: f64,
    pub normalize: bool,
    pub replace: bool,
    pub delta: bool,
    pub fade: f64,
    pub fade_keep: f64,
    pub separator: bool,
    pub compiled_moment: bool,
    pub pre: Vec<i64>,
    pub post: Vec<i64>,
    /// `(batch, key_width, post)`: each stream's strengths.
    pub strength: Vec<f64>,
    pub batch: usize,
    pub mass: Vec<f64>,
    pub writes: i64,
    /// `(key_width, post)`: the records every stream shares.
    pub consolidated: Vec<f64>,
    /// The records' noise scales, with the surprise gate.
    pub gate_noise: Option<Vec<f64>>,
    /// The factor of the brain's maturity at the call, where the store has one (read only).
    pub maturity: Option<f64>,
}

/// `FastSynapses._delta_unit` of one row: scaled by its largest magnitude, then to unit length.
pub fn delta_unit(x: &[f64]) -> Vec<f64> {
    let scale = np::max(&x.iter().map(|v| v.abs()).collect::<Vec<f64>>());
    let by = if scale > 0.0 { scale } else { 1.0 };
    let scaled: Vec<f64> = x.iter().map(|v| v / by).collect();
    let norm = np::sqrt(np::sum_squares(&scaled));
    let by = if norm > 0.0 { norm } else { 1.0 };
    scaled.iter().map(|v| v / by).collect()
}

/// `rows @ matrix` of `rows` (each `k` wide) and a `k` by `w` matrix, each entry a dot product
/// in order.
fn product(rows: &[Vec<f64>], matrix: &[f64], w: usize) -> Vec<Vec<f64>> {
    let k = matrix.len() / w.max(1);
    // `np::dot` of each row with each column, the column read where it lies
    rows.iter().map(|r| (0..w).map(|j| {
        let mut s = 0.0f64;
        for (i, x) in r.iter().take(k).enumerate() {
            s += x * matrix[i * w + j];
        }
        s
    }).collect()).collect()
}

fn finite(xs: &[f64]) -> bool {
    xs.iter().all(|x| x.is_finite())
}

impl Synaptic {
    pub fn import(s: &Named) -> Result<Synaptic, Error> {
        let flag = |n: &str| -> Result<bool, Error> {
            let v = s.bools(n)?;
            (v.len() == 1).then(|| v[0]).ok_or_else(|| Error::Shape(n.to_string()))
        };
        let pre = s.i64s("pre")?;
        let post = s.i64s("post")?;
        let (kw, m) = (pre.len(), post.len());
        let shape = s.shape("strength")?;
        if shape.len() != 3 || shape[1] != kw || shape[2] != m {
            return Err(Error::Shape("strength".into()));
        }
        let batch = shape[0];
        let mass = s.f64s("mass")?;
        let consolidated = s.f64s("consolidated")?;
        if mass.len() != batch || s.shape("consolidated")? != [kw, m] {
            return Err(Error::Shape("mass or consolidated".into()));
        }
        let gate_noise = if s.has("gate_noise") {
            if s.shape("gate_noise")? != [kw, m] {
                return Err(Error::Shape("gate_noise".into()));
            }
            Some(s.f64s("gate_noise")?)
        } else {
            None
        };
        let held = Synaptic {
            decay: s.f64("config.decay")?,
            rate: s.f64("config.rate")?,
            amplitude: s.f64("config.amplitude")?,
            consolidation: s.f64("config.consolidation")?,
            normalize: flag("config.normalize")?,
            replace: flag("config.replace")?,
            delta: flag("config.delta")?,
            fade: s.f64("config.fade")?,
            fade_keep: s.f64("config.fade_keep")?,
            separator: flag("config.separator")?,
            compiled_moment: flag("config.compiled_moment")?,
            strength: s.f64s("strength")?,
            batch,
            mass,
            writes: s.i64("writes")?,
            consolidated,
            gate_noise,
            maturity: if s.has("maturity") { Some(s.f64("maturity")?) } else { None },
            pre,
            post,
        };
        if held.separator || held.fade != 0.0 || !held.delta {
            return Err(Error::Shape(
                "the core holds a delta store keyed on its pre neurons that does not fade".into()));
        }
        Ok(held)
    }

    pub fn export(&self, out: &mut Named) {
        let (kw, m) = (self.pre.len(), self.post.len());
        for (name, x) in [("config.decay", self.decay), ("config.rate", self.rate),
                          ("config.amplitude", self.amplitude),
                          ("config.consolidation", self.consolidation)] {
            out.put(name, Array::scalar_f64(x));
        }
        for (name, x) in [("config.normalize", self.normalize), ("config.replace", self.replace),
                          ("config.delta", self.delta)] {
            out.put(name, Array::from_bool(&[], &[x]));
        }
        out.put("config.fade", Array::scalar_f64(self.fade));
        out.put("config.fade_keep", Array::scalar_f64(self.fade_keep));
        out.put("config.separator", Array::from_bool(&[], &[self.separator]));
        out.put("config.compiled_moment", Array::from_bool(&[], &[self.compiled_moment]));
        out.put("pre", Array::from_i64(&[kw], &self.pre));
        out.put("post", Array::from_i64(&[m], &self.post));
        out.put("strength", Array::from_f64(&[self.batch, kw, m], &self.strength));
        out.put("mass", Array::from_f64(&[self.batch], &self.mass));
        out.put("writes", Array::scalar_i64(self.writes));
        out.put("consolidated", Array::from_f64(&[kw, m], &self.consolidated));
        if let Some(noise) = &self.gate_noise {
            out.put("gate_noise", Array::from_f64(&[kw, m], noise));
        }
        if let Some(factor) = self.maturity {
            out.put("maturity", Array::scalar_f64(factor));
        }
    }

    fn width(&self) -> (usize, usize) {
        (self.pre.len(), self.post.len())
    }

    /// `_port`: `(batch, width)` finite rows.
    fn port(values: &[f64], shape: &[usize], width: usize, name: &str)
            -> Result<Vec<Vec<f64>>, Error> {
        if shape.len() != 2 || shape[1] != width || !finite(values) {
            return Err(refused(&format!("{name} must be a finite (batch, {width}) array")));
        }
        Ok(values.chunks(width.max(1)).take(shape[0]).map(|r| r.to_vec()).collect())
    }

    /// `recall(key)`: `(batch, pre)` keys read as `(batch, post)` values; a batch of another
    /// size reads the shared records.
    pub fn recall(&self, key: &[f64], shape: &[usize]) -> Result<Vec<Vec<f64>>, Error> {
        let (kw, m) = self.width();
        let cue = Self::port(key, shape, kw, "key")?;
        if self.compiled_moment {
            return Err(Error::Shape("the compiled moment's read is the spine's".into()));
        }
        let same_batch = self.batch == cue.len();
        let mut out = Vec::with_capacity(cue.len());
        for (r, row) in cue.iter().enumerate() {
            let strength = if same_batch { &self.strength[r * kw * m..(r + 1) * kw * m] }
                           else { &self.consolidated[..] };
            let read = product(&[delta_unit(row)], strength, m).remove(0);
            out.push(read.iter().map(|x| self.amplitude * x).collect());
        }
        Ok(out)
    }

    /// `stimulate(drive, key=)`: the drive with the read added to its post columns, the read
    /// of its pre columns or of `key`.
    pub fn stimulate(&self, drive: &[f64], shape: &[usize], key: Option<(&[f64], &[usize])>)
                     -> Result<Vec<f64>, Error> {
        if shape.len() != 2 {
            return Err(Error::Shape("drive".into()));
        }
        let (rows, n) = (shape[0], shape[1]);
        if self.pre.iter().chain(&self.post).any(|&c| c < 0 || c as usize >= n) {
            return Err(Error::Shape("the drive holds the memory's neurons".into()));
        }
        let mut out = drive.to_vec();
        let read = match key {
            Some((k, s)) => self.recall(k, s)?,
            None => {
                let keys: Vec<f64> = (0..rows)
                    .flat_map(|r| self.pre.iter().map(move |&c| (r, c)))
                    .map(|(r, c)| out[r * n + c as usize]).collect();
                self.recall(&keys, &[rows, self.pre.len()])?
            }
        };
        if read.len() != rows {
            return Err(refused("a key per row of the drive"));
        }
        for (r, values) in read.iter().enumerate() {
            for (&c, v) in self.post.iter().zip(values) {
                out[r * n + c as usize] += v;
            }
        }
        Ok(out)
    }

    /// `_gate`: each writing row's gain per action and the records' noise scales moved toward
    /// the rows' squared errors.
    fn gate(&self, cue: &[Vec<f64>], value: &[Vec<f64>], previous: &[&[f64]], mask: &[Vec<bool>],
            noise: &[f64], factor: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let (kw, m) = self.width();
        let mut noise = noise.to_vec();
        let engaged = 1.0 - factor;
        let mut gains = Vec::with_capacity(cue.len());
        let mut errors = Vec::with_capacity(cue.len());
        let mut weights = Vec::with_capacity(cue.len());
        let squares: Vec<Vec<f64>> = cue.iter().map(|c| c.iter().map(|x| x * x).collect())
            .collect();
        let spreads = product(&squares, &noise, m);
        for (b, row) in cue.iter().enumerate() {
            // `einsum("bi,bij->bj", cue, previous)`: in order over i
            let read = product(&[row.clone()], previous[b], m).remove(0);
            let error: Vec<f64> = value[b].iter().zip(&read).map(|(v, r)| v - r).collect();
            let gain: Vec<f64> = (0..m).map(|j| {
                let size = error[j].abs();
                let spread = np::sqrt(spreads[b][j]);
                let ratio = GATE_WIDTH * spread / if size > 0.0 { size } else { 1.0 };
                let surprise = if size > 0.0 { np::maximum(0.0, 1.0 - ratio) } else { 0.0 };
                if mask[b][j] { 1.0 - engaged * (1.0 - surprise) } else { 1.0 }
            }).collect();
            let by = np::maximum(np::max(&squares[b]), 1e-30);
            weights.push(squares[b].iter().map(|x| x / by).collect::<Vec<f64>>());
            errors.push(error);
            gains.push(gain);
        }
        for b in 0..cue.len() {
            for j in 0..m {
                if !mask[b][j] {
                    continue;
                }
                let e2 = errors[b][j] * errors[b][j];
                for i in 0..kw {
                    let at = i * m + j;
                    noise[at] += GATE_RATE * weights[b][i] * (e2 - noise[at]);
                }
            }
        }
        (gains, noise)
    }

    /// `observe(key, value, write, salience=, value_mask=)`: the shared records consolidate
    /// the writing rows' observations, each stream's residual decays and corrects its read.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(&mut self, key: &[f64], key_shape: &[usize], value: &[f64],
                   value_shape: &[usize], write: Option<&[bool]>, salience: Option<&[f64]>,
                   value_mask: Option<(&[bool], &[usize])>) -> Result<(), Error> {
        let (kw, m) = self.width();
        let key = Self::port(key, key_shape, kw, "key")?;
        let value = Self::port(value, value_shape, m, "value")?;
        let batch = key.len();
        if value.len() != batch {
            return Err(refused("key and value batches must match"));
        }
        let salience = match salience {
            None => vec![0.0; batch],
            Some(s) => s.to_vec(),
        };
        if salience.len() != batch || !finite(&salience) || salience.iter().any(|&x| x < 0.0) {
            return Err(refused("salience must be a finite nonnegative vector matching the \
                                     batch"));
        }
        let gate = match write {
            None => vec![true; batch],
            Some(w) if w.len() == batch => w.to_vec(),
            Some(_) => return Err(refused("write must be a boolean vector with one entry \
                                                per stream")),
        };
        let observed: Vec<Vec<bool>> = match value_mask {
            None => vec![vec![true; m]; batch],
            Some((v, s)) if s == [batch, m] => v.chunks(m.max(1)).take(batch)
                .map(|r| r.to_vec()).collect(),
            Some(_) => return Err(refused("value_mask must be a boolean array matching the \
                                                values")),
        };
        let rows: Vec<usize> = (0..batch).filter(|&b| {
            gate[b] && key[b].iter().any(|&x| x != 0.0) && observed[b].iter().any(|&x| x)
        }).collect();
        let size = kw * m;
        let same_batch = self.batch == batch;
        let previous = |b: usize| -> &[f64] {
            if same_batch { &self.strength[b * size..(b + 1) * size] } else { &self.consolidated }
        };
        // W = C + F: decay F, never C
        let c = &self.consolidated;
        let mut strength = vec![0.0; batch * size];
        for b in 0..batch {
            let prev = previous(b);
            for at in 0..size {
                strength[b * size + at] = c[at] + self.decay * (prev[at] - c[at]);
            }
        }
        let mut mass: Vec<f64> = if same_batch {
            self.mass.iter().map(|x| x * self.decay).collect()
        } else {
            vec![0.0; batch]
        };
        if rows.is_empty() {
            self.strength = strength;
            self.batch = batch;
            self.mass = mass;
            return Ok(());
        }
        let cue: Vec<Vec<f64>> = rows.iter().map(|&b| delta_unit(&key[b])).collect();
        let mask: Vec<Vec<bool>> = rows.iter().map(|&b| observed[b].clone()).collect();
        let rate: Vec<f64> = rows.iter().map(|&b| {
            let r = self.consolidation + np::minimum(1.0, self.consolidation * salience[b]);
            np::minimum(1.0, r)
        }).collect();
        let read = product(&cue, c, m);
        let error: Vec<Vec<f64>> = rows.iter().enumerate().map(|(r, &b)| {
            (0..m).map(|j| (value[b][j] - read[r][j]) * if mask[r][j] { 1.0 } else { 0.0 })
                .collect()
        }).collect();
        let n = rows.len() as f64;
        // `einsum("b,bi,bj->ij", rate, cue, error)`: in order over the rows
        let contract = |error: &[Vec<f64>]| -> Vec<f64> {
            let mut out = vec![0.0; size];
            let mut lead = vec![0.0; rows.len()];
            for i in 0..kw {
                for (r, l) in lead.iter_mut().enumerate() {
                    *l = rate[r] * cue[r][i];
                }
                for j in 0..m {
                    // `np::dot(lead, error[:, j])`, the column read where it lies
                    let mut s = 0.0f64;
                    for (l, e) in lead.iter().zip(error) {
                        s += l * e[j];
                    }
                    out[i * m + j] = s;
                }
            }
            out
        };
        let change: Vec<f64> = contract(&error).iter().map(|x| x / n).collect();
        let gated = self.gate_noise.is_some() && self.maturity.is_some();
        let base = if gated { strength.clone() } else { Vec::new() };
        for b in 0..batch {
            for at in 0..size {
                strength[b * size + at] += change[at];
            }
        }
        let fast: Vec<Vec<f64>> = rows.iter().enumerate().map(|(r, &b)| {
            let prediction = product(&[cue[r].clone()], &strength[b * size..(b + 1) * size], m)
                .remove(0);
            let correction: Vec<f64> = (0..m).map(|j| {
                (value[b][j] - prediction[j]) * if mask[r][j] { 1.0 } else { 0.0 }
            }).collect();
            let mut out = vec![0.0; size];
            for i in 0..kw {
                let lead = self.rate * cue[r][i];
                for j in 0..m {
                    out[i * m + j] = lead * correction[j];
                }
            }
            out
        }).collect();
        let mut noise = None;
        let consolidated: Vec<f64>;
        if !gated {
            consolidated = c.iter().zip(&change).map(|(a, d)| a + d).collect();
            for (r, &b) in rows.iter().enumerate() {
                for at in 0..size {
                    strength[b * size + at] += fast[r][at];
                }
            }
        } else {
            let held = self.gate_noise.as_ref().unwrap();
            let before: Vec<&[f64]> = rows.iter().map(|&b| previous(b)).collect();
            let values: Vec<Vec<f64>> = rows.iter().map(|&b| value[b].clone()).collect();
            let (gain, moved_noise) = self.gate(&cue, &values, &before, &mask, held,
                                                self.maturity.unwrap());
            // one gain per action: the mean over the rows that observe it
            let shared: Vec<f64> = (0..m).map(|j| {
                let seen = mask.iter().filter(|r| r[j]).count();
                // `(gain * mask).sum(axis=0)`: the rows added in order to the sum's +0.0
                let mut total = 0.0;
                for r in 0..rows.len() {
                    total += gain[r][j] * if mask[r][j] { 1.0 } else { 0.0 };
                }
                if seen > 0 { total / seen.max(1) as f64 } else { 1.0 }
            }).collect();
            let mut moved = vec![0.0; batch * size];
            for b in 0..batch {
                moved[b * size..(b + 1) * size].copy_from_slice(&change);
            }
            for (r, &b) in rows.iter().enumerate() {
                for at in 0..size {
                    moved[b * size + at] += fast[r][at];
                }
            }
            let scaled: Vec<Vec<f64>> = error.iter()
                .map(|e| e.iter().zip(&shared).map(|(x, s)| x * s).collect()).collect();
            consolidated = c.iter().zip(contract(&scaled)).map(|(a, d)| a + d / n).collect();
            for b in 0..batch {
                for at in 0..size {
                    let k = b * size + at;
                    strength[k] = base[k] + shared[at % m] * moved[k];
                }
            }
            noise = Some(moved_noise);
        }
        if !finite(&strength) || !finite(&consolidated)
            || noise.as_ref().is_some_and(|x| !finite(x))
        {
            return Err(refused(
                "synaptic update overflowed; scale the observed values".into()));
        }
        for &b in &rows {
            mass[b] += self.rate;
        }
        self.consolidated = consolidated;
        self.strength = strength;
        self.batch = batch;
        self.mass = mass;
        if noise.is_some() {
            self.gate_noise = noise;
        }
        self.writes += rows.len() as i64;
        Ok(())
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let rows = |values: Vec<Vec<f64>>, width: usize| -> Array {
            let flat: Vec<f64> = values.concat();
            Array::from_f64(&[values.len(), width], &flat)
        };
        let m = self.post.len();
        match method {
            "observe" => {
                let write = if args.has("write") { Some(args.bools("write")?) } else { None };
                let salience = if args.has("salience") { Some(args.f64s("salience")?) }
                               else { None };
                let mask = if args.has("value_mask") {
                    Some((args.bools("value_mask")?, args.shape("value_mask")?))
                } else {
                    None
                };
                self.observe(&args.f64s("key")?, &args.shape("key")?, &args.f64s("value")?,
                             &args.shape("value")?, write.as_deref(), salience.as_deref(),
                             mask.as_ref().map(|(v, s)| (&v[..], &s[..])))?;
            }
            "recall" => {
                let read = self.recall(&args.f64s("key")?, &args.shape("key")?)?;
                out.put("read", rows(read, m));
            }
            "stimulate" | "stimulus" => {
                let drive = args.f64s("drive")?;
                let shape = args.shape("drive")?;
                let stores = !args.has("stores") || args.bools("stores")?[0];
                let given = if method == "stimulate" { "key" } else { "place" };
                let done = if !stores {
                    drive
                } else if args.has(given) {
                    let (k, s) = (args.f64s(given)?, args.shape(given)?);
                    self.stimulate(&drive, &shape, Some((&k, &s)))?
                } else {
                    self.stimulate(&drive, &shape, None)?
                };
                out.put("drive", Array::from_f64(&shape, &done));
            }
            "held_values" => {
                let (kw, m) = self.width();
                out.put("strength", Array::from_f64(&[self.batch, kw, m], &self.strength));
                out.put("mass", Array::from_f64(&[self.batch], &self.mass));
                out.put("writes", Array::scalar_i64(self.writes));
                out.put("consolidated", Array::from_f64(&[kw, m], &self.consolidated));
                if let Some(noise) = &self.gate_noise {
                    out.put("gate_noise", Array::from_f64(&[kw, m], noise));
                }
            }
            "reside" => {}  // the core holds no device: the store stays where it is
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
