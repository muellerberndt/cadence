//! `cadence.stream.Trace`, the working trace, and `cadence.stream.Efference`, the trace written
//! from the issued command: a quantity decaying across moments per stream, read into the next
//! drive as a stimulus. The arithmetic as the Python's, elementwise in its order.

use crate::state::{Array, Error, Named};

/// A row-major `(rows, cols)` array of floats, as a batch of streams holds one.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Rows {
    pub fn zeros(rows: usize, cols: usize) -> Rows {
        Rows { rows, cols, data: vec![0.0; rows * cols] }
    }
    /// The named array `name`, which must have two dimensions.
    pub fn read(named: &Named, name: &str) -> Result<Rows, Error> {
        let shape = named.shape(name)?;
        if shape.len() != 2 {
            return Err(Error::Shape(format!("{name} is one row per stream")));
        }
        Ok(Rows { rows: shape[0], cols: shape[1], data: named.f64s(name)? })
    }
    pub fn array(&self) -> Array {
        Array::from_f64(&[self.rows, self.cols], &self.data)
    }
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
}

/// A scalar flag of named arrays.
pub(crate) fn flag(named: &Named, name: &str) -> Result<bool, Error> {
    match named.bools(name)?.as_slice() {
        [x] => Ok(*x),
        _ => Err(Error::Shape(name.to_string())),
    }
}

/// NumPy's places of an index into `len` entries: an integer's (a negative one from the end),
/// or a mask's entries that hold.
pub(crate) fn chosen(named: &Named, name: &str, len: usize) -> Result<Vec<usize>, Error> {
    let array = named.get(name)?;
    if let Some(mask) = array.bools() {
        if array.shape.len() != 1 || mask.len() != len {
            return Err(Error::Shape(format!("{name}: a mask of {len} entries")));
        }
        return Ok((0..len).filter(|&i| mask[i]).collect());
    }
    named.i64s(name)?.into_iter().map(|i| place(i, len, name)).collect()
}

/// An index into `len` entries as NumPy takes it, a negative one from the end.
pub(crate) fn place(i: i64, len: usize, name: &str) -> Result<usize, Error> {
    let at = if i < 0 { i + len as i64 } else { i };
    if at < 0 || at >= len as i64 {
        return Err(Error::Shape(format!("{name}: index {i} is out of bounds for {len}")));
    }
    Ok(at as usize)
}

/// The columns `index` of every row of `x`.
pub(crate) fn columns(x: &Rows, index: &[i64], name: &str) -> Result<Rows, Error> {
    let at: Vec<usize> = index.iter().map(|&i| place(i, x.cols, name)).collect::<Result<_, _>>()?;
    let mut out = Rows::zeros(x.rows, at.len());
    for r in 0..x.rows {
        for (j, &c) in at.iter().enumerate() {
            out.data[r * at.len() + j] = x.at(r, c);
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub decay: f64,
    pub amplitude: f64,
    /// The learned multiplier on the amplitude (`Hearing`).
    pub hearing: f64,
    /// The neurons the trace stimulates, one per source neuron.
    pub glow: Vec<i64>,
    /// The source neurons traced.
    pub hidden: Vec<i64>,
    pub trace: Rows,
    pub last: Rows,
    pub cold: Vec<bool>,
    /// An efference copy: written by `issue`, never by `update`.
    pub efference: bool,
}

impl Trace {
    pub fn import(s: &Named, efference: bool) -> Result<Trace, Error> {
        let held = Trace {
            decay: s.f64("config.decay")?,
            amplitude: s.f64("config.amplitude")?,
            hearing: s.f64("hearing")?,
            glow: s.i64s("glow")?,
            hidden: s.i64s("hidden")?,
            trace: Rows::read(s, "trace")?,
            last: Rows::read(s, "last")?,
            cold: s.bools("cold")?,
            efference,
        };
        let rows = held.trace.rows;
        if s.shape("cold")?.len() != 1 || held.cold.len() != rows || held.last.rows != rows
            || held.last.cols != held.trace.cols
        {
            return Err(Error::Shape("trace, last and cold hold one row per stream".into()));
        }
        Ok(held)
    }

    pub fn export(&self, out: &mut Named) {
        out.put("config.decay", Array::scalar_f64(self.decay));
        out.put("config.amplitude", Array::scalar_f64(self.amplitude));
        out.put("hearing", Array::scalar_f64(self.hearing));
        out.put("glow", Array::from_i64(&[self.glow.len()], &self.glow));
        out.put("hidden", Array::from_i64(&[self.hidden.len()], &self.hidden));
        out.put("trace", self.trace.array());
        out.put("last", self.last.array());
        out.put("cold", Array::from_bool(&[self.cold.len()], &self.cold));
    }

    /// `reset(batch, rows=None)`: fresh for every stream, or for `rows` only.
    pub fn reset(&mut self, batch: i64, rows: Option<Vec<usize>>) -> Result<(), Error> {
        let n = self.hidden.len();
        match rows {
            Some(rows) if self.trace.rows as i64 == batch => {
                for r in rows {
                    self.trace.data[r * self.trace.cols..(r + 1) * self.trace.cols].fill(0.0);
                    self.last.data[r * self.last.cols..(r + 1) * self.last.cols].fill(0.0);
                    *self.cold.get_mut(r).ok_or(Error::Shape("cold".into()))? = true;
                }
            }
            _ => {
                if batch < 0 {
                    return Err(Error::Shape("negative dimensions are not allowed".into()));
                }
                let batch = batch as usize;
                self.trace = Rows::zeros(batch, n);
                self.last = Rows::zeros(batch, n);
                self.cold = vec![true; batch];
            }
        }
        Ok(())
    }

    /// `_source_activation(state)`: the source range's activation, from the whole activation.
    pub fn source(&self, activation: &Rows) -> Result<Rows, Error> {
        columns(activation, &self.hidden, "hidden")
    }

    /// The trace decayed toward `h` (the Python's `decay * trace + (1 - decay) * h`), `h` held
    /// as the last moment's and every stream read.
    fn toward(&mut self, h: Rows) -> Result<(), Error> {
        if self.trace.rows != h.rows {
            self.reset(h.rows as i64, None)?;
        }
        if self.trace.cols != h.cols {
            return Err(Error::Shape("the trace and its source differ in width".into()));
        }
        let (decay, keep) = (self.decay, 1.0 - self.decay);
        for (t, x) in self.trace.data.iter_mut().zip(&h.data) {
            *t = decay * *t + keep * x;
        }
        self.last = h;
        self.cold.fill(false);
        Ok(())
    }

    /// `update(state)`: after the free phase, the trace decays toward the source's activation.
    pub fn update(&mut self, activation: &Rows) -> Result<(), Error> {
        if self.efference {
            return Err(Error::Shape(
                "the efference copy is written by the issued command; use issue".into(),
            ));
        }
        let h = self.source(activation)?;
        self.toward(h)
    }

    /// `issue(command)`: after the decision, the trace decays toward the command.
    pub fn issue(&mut self, command: Rows) -> Result<(), Error> {
        if command.cols != self.hidden.len() {
            return Err(Error::Shape("a command is one row of motor values per stream".into()));
        }
        if !command.data.iter().all(|x| x.is_finite()) {
            return Err(Error::Shape("a command must be finite".into()));
        }
        self.toward(command)
    }

    /// `stimulate(drive)`: the trace read into a copy of the drive; another batch reads zero.
    pub fn stimulate(&self, mut drive: Rows) -> Result<Rows, Error> {
        let at: Vec<usize> =
            self.glow.iter().map(|&i| place(i, drive.cols, "glow")).collect::<Result<_, _>>()?;
        let same = self.trace.rows == drive.rows;
        if same && self.trace.cols != at.len() {
            return Err(Error::Shape("the trace and its target differ in width".into()));
        }
        let loud = self.amplitude * self.hearing;
        for r in 0..drive.rows {
            for (j, &c) in at.iter().enumerate() {
                drive.data[r * drive.cols + c] =
                    if same { loud * self.trace.at(r, j) } else { 0.0 };
            }
        }
        Ok(drive)
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "reset" => {
                let batch = args.i64("batch")?;
                // the rows count where the batch is the trace's own; another batch begins whole
                let rows = if args.has("rows") && self.trace.rows as i64 == batch {
                    Some(chosen(args, "rows", self.trace.rows)?)
                } else {
                    None
                };
                self.reset(batch, rows)?;
            }
            "update" => self.update(&Rows::read(args, "activation")?)?,
            "_source_activation" => {
                out.put("activation", self.source(&Rows::read(args, "activation")?)?.array())
            }
            "stimulate" | "stimulus" => {
                out.put("drive", self.stimulate(Rows::read(args, "drive")?)?.array())
            }
            "issue" if self.efference => {
                let shape = args.shape("command")?;
                if shape.len() != 2 {
                    return Err(Error::Shape(
                        "a command is one row of motor values per stream".into(),
                    ));
                }
                self.issue(Rows::read(args, "command")?)?;
            }
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
