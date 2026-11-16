//! `cadence.stream.History`: the observations as heard, one block per decay, each block moving
//! toward the observation after each act and read into the next drive at its amplitude times its
//! learned gain. The arithmetic as the Python's, elementwise in its order.

use super::trace::{chosen, place, Rows};
use crate::state::{Array, Error, Named};

#[derive(Clone, Debug, PartialEq)]
pub struct History {
    pub decays: Vec<f64>,
    pub amplitude: f64,
    /// The heard width: one block of it per decay.
    pub width: usize,
    pub glow: Vec<i64>,
    /// The learned gain per block, or per stream and block (its shape and values); None holds
    /// the working amplitude.
    pub hearing: Option<(Vec<usize>, Vec<f64>)>,
    pub trace: Rows,
}

impl History {
    pub fn import(s: &Named) -> Result<History, Error> {
        let width = s.i64("width")?;
        if width < 0 {
            return Err(Error::Shape("width".into()));
        }
        let hearing = if s.has("hearing") {
            let shape = s.shape("hearing")?;
            if shape.is_empty() || shape.len() > 2 {
                return Err(Error::Shape("hearing is one gain per block, or per stream".into()));
            }
            Some((shape, s.f64s("hearing")?))
        } else {
            None
        };
        Ok(History {
            decays: s.f64s("config.decays")?,
            amplitude: s.f64("config.amplitude")?,
            width: width as usize,
            glow: s.i64s("glow")?,
            hearing,
            trace: Rows::read(s, "trace")?,
        })
    }

    pub fn export(&self, out: &mut Named) {
        out.put("config.decays", Array::from_f64(&[self.decays.len()], &self.decays));
        out.put("config.amplitude", Array::scalar_f64(self.amplitude));
        out.put("width", Array::scalar_i64(self.width as i64));
        out.put("glow", Array::from_i64(&[self.glow.len()], &self.glow));
        if let Some((shape, gains)) = &self.hearing {
            out.put("hearing", Array::from_f64(shape, gains));
        }
        out.put("trace", self.trace.array());
    }

    /// `reset(batch, rows=None)`: silence for every stream, or for `rows` only.
    pub fn reset(&mut self, batch: i64, rows: Option<Vec<usize>>) -> Result<(), Error> {
        match rows {
            Some(rows) if self.trace.rows as i64 == batch => {
                let cols = self.trace.cols;
                for r in rows {
                    self.trace.data[r * cols..(r + 1) * cols].fill(0.0);
                }
            }
            _ => {
                if batch < 0 {
                    return Err(Error::Shape("negative dimensions are not allowed".into()));
                }
                self.trace = Rows::zeros(batch as usize, self.glow.len());
            }
        }
        Ok(())
    }

    /// `stimulate(drive)`: the history read into a copy of the drive at the amplitude times each
    /// block's gain; another batch reads silence.
    pub fn stimulate(&self, mut drive: Rows) -> Result<Rows, Error> {
        let at: Vec<usize> =
            self.glow.iter().map(|&i| place(i, drive.cols, "glow")).collect::<Result<_, _>>()?;
        let same = self.trace.rows == drive.rows;
        let n = self.trace.cols;
        if same && n != at.len() {
            return Err(Error::Shape("the history and its target differ in width".into()));
        }
        // the amplitude of each neuron: the amplitude times its block's gain (np.repeat), one
        // row of them, or one per stream
        let mut loud: Vec<f64> = vec![self.amplitude; n];
        let mut per_stream = false;
        if let Some((shape, gains)) = &self.hearing {
            let blocks = shape[shape.len() - 1];
            if same && blocks * self.width != n {
                return Err(Error::Shape("one gain per block of the history".into()));
            }
            if shape.len() == 2 && same {
                let streams = shape[0];
                if streams != self.trace.rows && streams != 1 {
                    return Err(Error::Shape("one row of gains per stream".into()));
                }
                per_stream = streams != 1;
                let rows = if per_stream { streams } else { 1 };
                loud = Vec::with_capacity(rows * n);
                for r in 0..rows {
                    for j in 0..n {
                        loud.push(self.amplitude * gains[r * blocks + j / self.width]);
                    }
                }
            } else if same {
                for (j, l) in loud.iter_mut().enumerate() {
                    *l = self.amplitude * gains[j / self.width];
                }
            }
        }
        for r in 0..drive.rows {
            let row = if per_stream { r * n } else { 0 };
            for (j, &c) in at.iter().enumerate() {
                drive.data[r * drive.cols + c] =
                    if same { loud[row + j] * self.trace.at(r, j) } else { 0.0 };
            }
        }
        Ok(drive)
    }

    /// `hear(observations)`: every block decays toward the observations the streams acted on;
    /// another batch begins in silence first.
    pub fn hear(&mut self, x: &Rows) -> Result<(), Error> {
        if x.cols != self.width {
            return Err(Error::Shape(format!(
                "observations are one row of {} values per stream", self.width)));
        }
        if !x.data.iter().all(|v| v.is_finite()) {
            return Err(Error::Shape("observations must be finite".into()));
        }
        if self.trace.rows != x.rows {
            self.reset(x.rows as i64, None)?;
        }
        let n = self.decays.len() * self.width;
        if self.trace.cols != n {
            return Err(Error::Shape("one block of the heard width per decay".into()));
        }
        for r in 0..x.rows {
            for c in 0..n {
                let d = self.decays[c / self.width];
                let t = &mut self.trace.data[r * n + c];
                *t = d * *t + (1.0 - d) * x.at(r, c % self.width);
            }
        }
        Ok(())
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "reset" => {
                let batch = args.i64("batch")?;
                let rows = if args.has("rows") && self.trace.rows as i64 == batch {
                    Some(chosen(args, "rows", self.trace.rows)?)
                } else {
                    None
                };
                self.reset(batch, rows)?;
            }
            "stimulate" | "stimulus" => {
                out.put("drive", self.stimulate(Rows::read(args, "drive")?)?.array())
            }
            "hear" => {
                let x = Rows::read(args, "observations").map_err(|_| {
                    Error::Shape(format!(
                        "observations are one row of {} values per stream", self.width))
                })?;
                self.hear(&x)?;
            }
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
