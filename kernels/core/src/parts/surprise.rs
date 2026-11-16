//! `cadence.surprise`: each unit's surprise at a moment against the size it is used to
//! (`Surprise`), and the one focus that surprise draws among the places of every sense
//! (`Orienting` and its `Focus`). The law and the competition as the Python computes them, in
//! its order.

use crate::np;
use crate::state::{Array, Error, Named};

/// The senses a focus is drawn from, by code: a sense's code is its place here, as
/// `cadence.core.parts.orienting.SENSES` names them (the default sense of `attend`, the
/// sensory neurons with places, a held map's cells, the things a heard name makes salient).
pub const SENSES: [&str; 4] = ["place", "senses", "map", "named"];

fn refused(text: &str) -> Error {
    Error::Shape(text.to_string())
}

/// `_gene`: a finite real number in `[low, high]`.
fn gene(s: &Named, name: &str, low: f64, high: f64) -> Result<f64, Error> {
    let value = s.f64(&format!("config.{name}"))?;
    if !value.is_finite() || value < low || value > high {
        return Err(refused(&format!("{name} must lie in [{low}, {high}]")));
    }
    Ok(value)
}

/// `_count`: a count of zero or more.
fn count(s: &Named, name: &str) -> Result<i64, Error> {
    let value = s.i64(name)?;
    if value < 0 {
        return Err(refused(&format!("{name} is a count")));
    }
    Ok(value)
}

/// `np.linalg.norm(p - at, axis=1)` for the `n` places `p` (rows of `frame` coordinates): each
/// row's squared differences summed in NumPy's order of a row, and its root (zero for places
/// of no coordinates).
fn distances(p: &[f64], n: usize, frame: usize, at: &[f64]) -> Vec<f64> {
    if frame == 0 {
        return vec![0.0; n];
    }
    let squares: Vec<f64> = p
        .chunks_exact(frame)
        .flat_map(|row| row.iter().zip(at).map(|(x, y)| (x - y) * (x - y)))
        .collect();
    squares.chunks_exact(frame).map(|row| np::sqrt(np::sum(row))).collect()
}

// -- Surprise ----------------------------------------------------------------------------------

/// `cadence.surprise.Surprise`: the genes, each unit's usual size, the latest surprise per
/// stream and unit, and the moments observed.
#[derive(Clone, Debug, PartialEq)]
pub struct Surprise {
    pub tolerance: f64,
    pub floor: f64,
    pub adapt: f64,
    pub recover: f64,
    /// Each unit's learned size, `(units,)`.
    pub usual: Vec<f64>,
    /// The latest surprise, `(streams, units)` row by row.
    pub last: Vec<f64>,
    pub streams: usize,
    pub moments: i64,
}

impl Surprise {
    pub const GENES: [&'static str; 4] = ["tolerance", "floor", "adapt", "recover"];

    /// `Surprise(units, tolerance=, floor=, adapt=, recover=)`: no usual yet, one stream.
    pub fn new(units: usize, tolerance: f64, floor: f64, adapt: f64, recover: f64)
               -> Result<Surprise, Error> {
        let mut s = Named::new();
        for (name, value) in [("tolerance", tolerance), ("floor", floor), ("adapt", adapt),
                              ("recover", recover)] {
            s.put(&format!("config.{name}"), Array::scalar_f64(value));
        }
        s.put("usual", Array::from_f64(&[units], &vec![0.0; units]));
        s.put("last", Array::from_f64(&[1, units], &vec![0.0; units]));
        s.put("moments", Array::scalar_i64(0));
        Surprise::import(&s)
    }

    /// The state as `cadence.core.parts.surprise.state_of` gives it, refused where the
    /// library's own loader (`Surprise.from_dict`) refuses it.
    pub fn import(s: &Named) -> Result<Surprise, Error> {
        let tolerance = gene(s, "tolerance", 0.0, f64::INFINITY)?;
        let floor = gene(s, "floor", 0.0, f64::INFINITY)?;
        let adapt = gene(s, "adapt", 0.0, 1.0)?;
        let recover = gene(s, "recover", 0.0, 1.0)?;
        if floor <= 0.0 {
            return Err(refused("a surprise needs a floor above zero"));
        }
        let usual = s.f64s("usual")?;
        if s.shape("usual")?.len() != 1 || usual.is_empty() {
            return Err(refused("a surprise's usual holds one size a unit"));
        }
        let shape = s.shape("last")?;
        let last = s.f64s("last")?;
        if shape.len() != 2 || shape[1] != usual.len() {
            return Err(refused("a surprise's last does not fit its units"));
        }
        if !usual.iter().chain(&last).all(|x| x.is_finite() && *x >= 0.0) {
            return Err(refused("a surprise's usual and last are finite and zero or more"));
        }
        Ok(Surprise { tolerance, floor, adapt, recover, usual, last, streams: shape[0],
                      moments: count(s, "moments")? })
    }

    pub fn export(&self, out: &mut Named) {
        for (name, value) in [("tolerance", self.tolerance), ("floor", self.floor),
                              ("adapt", self.adapt), ("recover", self.recover)] {
            out.put(&format!("config.{name}"), Array::scalar_f64(value));
        }
        out.put("usual", Array::from_f64(&[self.usual.len()], &self.usual));
        out.put("last", Array::from_f64(&[self.streams, self.usual.len()], &self.last));
        out.put("moments", Array::scalar_i64(self.moments));
    }

    /// The units.
    pub fn size(&self) -> usize {
        self.usual.len()
    }

    /// `moment`: the largest surprise of each stream's units, `(streams,)`.
    pub fn moment(&self) -> Vec<f64> {
        self.last
            .chunks_exact(self.size())
            .map(|row| row[1..].iter().fold(row[0], |m, &x| np::maximum(m, x)))
            .collect()
    }

    /// `observe(sizes, learn=, carried=)`: `sizes` holds `streams` rows of the units' sizes (a
    /// row of `(units,)` is one stream); the surprise per stream and unit, against the usual as
    /// it stood before the moment; the usual then learns from the mean over the streams that
    /// `carried` flags (every stream with None) unless `learn` is false.
    pub fn observe(&mut self, sizes: &[f64], streams: usize, learn: bool,
                   carried: Option<&[bool]>) -> Result<Vec<f64>, Error> {
        let n = self.size();
        if streams == 0 || streams.checked_mul(n) != Some(sizes.len()) || !sizes.iter().all(|x| x.is_finite()) {
            return Err(refused(&format!("a moment's sizes are finite, {n} a stream")));
        }
        let rows: Vec<bool> = match carried {
            Some(flags) if flags.len() == streams => flags.to_vec(),
            Some(_) => return Err(refused("carried flags one stream each")),
            None => vec![true; streams],
        };
        let r: Vec<f64> = sizes.iter().map(|x| x.abs()).collect();
        let unit: Vec<f64> = self.usual.iter().map(|&u| self.tolerance * u + self.floor).collect();
        let below: Vec<f64> = unit.iter().map(|&u| np::log(u)).collect();
        let mut out = vec![0.0; streams * n];
        for (row, flag) in rows.iter().enumerate() {
            if *flag {
                for j in 0..n {
                    out[row * n + j] = np::log(np::maximum(r[row * n + j], unit[j])) - below[j];
                }
            }
        }
        let taken: Vec<usize> = (0..streams).filter(|&row| rows[row]).collect();
        if learn && !taken.is_empty() {
            // the mean over axis 0: the rows added in order, over their count
            let mut mean = r[taken[0] * n..(taken[0] + 1) * n].to_vec();
            for &row in &taken[1..] {
                for j in 0..n {
                    mean[j] += r[row * n + j];
                }
            }
            let k = taken.len() as f64;
            for j in 0..n {
                let m = mean[j] / k;
                let rate = if m > self.usual[j] { self.adapt } else { self.recover };
                self.usual[j] += rate * (m - self.usual[j]);
            }
        }
        self.moments += 1;
        self.streams = streams;
        self.last = out.clone();
        Ok(out)
    }

    /// `reset(streams)`: the streams begin afresh, no surprise; the usual stays.
    pub fn reset(&mut self, streams: usize) {
        self.streams = streams;
        self.last = vec![0.0; streams * self.size()];
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "observe" => {
                let shape = args.shape("sizes")?;
                let sizes = args.f64s("sizes")?;
                // np.atleast_2d: a number is one stream of one unit, a row one stream
                let streams = match shape.len() {
                    0 => 1,
                    1 => 1,
                    2 => shape[0],
                    _ => return Err(refused("a moment's sizes are one stream or several")),
                };
                let carried = if args.has("carried") { Some(args.bools("carried")?) } else { None };
                if args.has("carried") && args.shape("carried")?.len() != 1 {
                    return Err(refused("carried flags one stream each"));
                }
                if shape.len() == 2 && shape[1] != self.size() {
                    return Err(refused("a moment's sizes do not fit the units"));
                }
                let surprise = self.observe(&sizes, streams, args.bools("learn")?[0],
                                            carried.as_deref())?;
                let n = self.size();
                if shape.len() == 1 {
                    out.put("surprise", Array::from_f64(&[n], &surprise));
                } else {
                    out.put("surprise", Array::from_f64(&[streams, n], &surprise));
                }
            }
            "moment" => {
                let moment = self.moment();
                out.put("moment", Array::from_f64(&[moment.len()], &moment));
            }
            "reset" => {
                let streams = args.i64("streams")?;
                if streams < 0 {
                    return Err(refused("a surprise resets to streams of zero or more"));
                }
                self.reset(streams as usize);
            }
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

// -- Orienting ---------------------------------------------------------------------------------

/// `cadence.surprise.Focus`: the place attended, the sense (its code in `SENSES`) and index of
/// the place that won, its salience, the moments held, and whether it was taken this moment.
#[derive(Clone, Debug, PartialEq)]
pub struct Focus {
    pub place: Vec<f64>,
    pub sense: i64,
    pub index: i64,
    pub salience: f64,
    pub held: i64,
    pub turned: bool,
}

impl Focus {
    /// The focus as named arrays under `prefix` (`focused` and `<prefix>.place`, ...); with no
    /// focus, `focused` false and the rest empty.
    pub fn export(focus: Option<&Focus>, prefix: &str, out: &mut Named) {
        out.put("focused", Array::from_bool(&[], &[focus.is_some()]));
        let place = focus.map_or(&[][..], |f| &f.place[..]);
        out.put(&format!("{prefix}.place"), Array::from_f64(&[place.len()], place));
        out.put(&format!("{prefix}.sense"), Array::scalar_i64(focus.map_or(-1, |f| f.sense)));
        out.put(&format!("{prefix}.index"), Array::scalar_i64(focus.map_or(-1, |f| f.index)));
        out.put(&format!("{prefix}.salience"),
                Array::scalar_f64(focus.map_or(0.0, |f| f.salience)));
        out.put(&format!("{prefix}.held"), Array::scalar_i64(focus.map_or(0, |f| f.held)));
        out.put(&format!("{prefix}.turned"),
                Array::from_bool(&[], &[focus.is_some_and(|f| f.turned)]));
    }

    fn import(s: &Named, prefix: &str) -> Result<Option<Focus>, Error> {
        if !s.bools("focused")?[0] {
            return Ok(None);
        }
        let place = s.f64s(&format!("{prefix}.place"))?;
        if s.shape(&format!("{prefix}.place"))?.len() != 1 || place.is_empty()
            || !place.iter().all(|x| x.is_finite()) {
            return Err(refused("a saved focus holds a finite place"));
        }
        let sense = s.i64(&format!("{prefix}.sense"))?;
        if sense < 0 || sense as usize >= SENSES.len() {
            return Err(refused("a focus's sense is one of the senses"));
        }
        let salience = s.f64(&format!("{prefix}.salience"))?;
        if !salience.is_finite() || salience < 0.0 {
            return Err(refused("a focus's salience must lie in [0, inf]"));
        }
        Ok(Some(Focus {
            place, sense, index: count(s, &format!("{prefix}.index"))?, salience,
            held: count(s, &format!("{prefix}.held"))?,
            turned: s.bools(&format!("{prefix}.turned"))?[0],
        }))
    }
}

/// `cadence.surprise.Orienting`: the genes, the focus held, the places inhibited (each with
/// its weight) and the moments attended.
#[derive(Clone, Debug, PartialEq)]
pub struct Orienting {
    pub hold: f64,
    pub least: f64,
    pub reach: f64,
    pub inhibit: f64,
    pub release: f64,
    pub gain: f64,
    pub focus: Option<Focus>,
    pub inhibited: Vec<(Vec<f64>, f64)>,
    pub moments: i64,
}

impl Orienting {
    pub const GENES: [&'static str; 6] = ["hold", "least", "reach", "inhibit", "release", "gain"];

    pub fn import(s: &Named) -> Result<Orienting, Error> {
        let hold = gene(s, "hold", 0.0, 1.0)?;
        let least = gene(s, "least", 0.0, f64::INFINITY)?;
        let reach = gene(s, "reach", 0.0, f64::INFINITY)?;
        let inhibit = gene(s, "inhibit", 0.0, 1.0)?;
        let release = gene(s, "release", 0.0, 1.0)?;
        let gain = gene(s, "gain", 0.0, f64::INFINITY)?;
        let flat = s.f64s("inhibited.place")?;
        let dims = s.i64s("inhibited.dim")?;
        let weights = s.f64s("inhibited.weight")?;
        let total = dims.iter().try_fold(0usize, |t, &d| t.checked_add(usize::try_from(d).ok()?));
        if dims.len() != weights.len() || dims.iter().any(|&d| d < 1) || total != Some(flat.len())
            || !flat.iter().chain(&weights).all(|x| x.is_finite()) {
            return Err(refused("saved inhibited places hold a place and a weight each"));
        }
        let mut inhibited = Vec::with_capacity(dims.len());
        let mut at = 0usize;
        for (&d, &weight) in dims.iter().zip(&weights) {
            inhibited.push((flat[at..at + d as usize].to_vec(), weight));
            at += d as usize;
        }
        Ok(Orienting { hold, least, reach, inhibit, release, gain,
                       focus: Focus::import(s, "focus")?, inhibited,
                       moments: count(s, "moments")? })
    }

    pub fn export(&self, out: &mut Named) {
        for (name, value) in [("hold", self.hold), ("least", self.least), ("reach", self.reach),
                              ("inhibit", self.inhibit), ("release", self.release),
                              ("gain", self.gain)] {
            out.put(&format!("config.{name}"), Array::scalar_f64(value));
        }
        out.put("moments", Array::scalar_i64(self.moments));
        Focus::export(self.focus.as_ref(), "focus", out);
        let flat: Vec<f64> = self.inhibited.iter().flat_map(|(at, _)| at.iter().copied()).collect();
        let dims: Vec<i64> = self.inhibited.iter().map(|(at, _)| at.len() as i64).collect();
        let weights: Vec<f64> = self.inhibited.iter().map(|(_, w)| *w).collect();
        out.put("inhibited.place", Array::from_f64(&[flat.len()], &flat));
        out.put("inhibited.dim", Array::from_i64(&[dims.len()], &dims));
        out.put("inhibited.weight", Array::from_f64(&[weights.len()], &weights));
    }

    /// `reset`: no focus and no place inhibited.
    pub fn reset(&mut self) {
        self.focus = None;
        self.inhibited = Vec::new();
        self.moments = 0;
    }

    /// `_leave`: the place of the focus held until this moment is inhibited, if there was one.
    pub fn leave(&mut self) {
        if let Some(focus) = &self.focus {
            if self.inhibit > 0.0 {
                self.inhibited.push((focus.place.clone(), self.inhibit));
            }
        }
    }

    /// `attend`: one moment of the competition among the candidates of every sense, in one
    /// frame: each one's surprise, its place (`frame` coordinates, row by row) and its sense's
    /// code, the senses in the order given (a candidate's index counts within its sense's
    /// run). The focus, or None where nothing reaches `least`. Refused before anything changes:
    /// values that are not finite, places that do not fit, a sense the core does not know.
    pub fn attend(&mut self, surprise: &[f64], places: &[f64], frame: usize, sense: &[i64])
                  -> Result<Option<Focus>, Error> {
        let n = surprise.len();
        if sense.len() != n || n.checked_mul(frame) != Some(places.len()) {
            return Err(refused("the places hold one place a unit"));
        }
        if !surprise.iter().chain(places).all(|x| x.is_finite()) {
            return Err(refused("surprise and places are finite"));
        }
        if sense.iter().any(|&c| c < 0 || c as usize >= SENSES.len()) {
            return Err(refused("a sense the core does not know"));
        }
        let mut index = vec![0i64; n];
        for i in 1..n {
            index[i] = if sense[i] == sense[i - 1] { index[i - 1] + 1 } else { 0 };
        }
        let mut s: Vec<f64> = surprise.iter().map(|&x| np::maximum(x, 0.0)).collect();
        let place = |i: usize| places[i * frame..(i + 1) * frame].to_vec();
        self.moments += 1;
        // inhibition of return: a place left loses salience for a while
        let mut kept = Vec::with_capacity(self.inhibited.len());
        for (at, weight) in &self.inhibited {
            if n > 0 && frame == at.len() {
                let keep = 1.0 - weight;
                for (k, d) in distances(places, n, frame, at).into_iter().enumerate() {
                    if d <= self.reach {
                        s[k] *= keep;
                    }
                }
            }
            if weight * self.release >= 0.01 {
                kept.push((at.clone(), weight * self.release));
            }
        }
        self.inhibited = kept;
        let mut focus = self.focus.clone();
        let mut near = vec![false; n];
        if let Some(f) = &focus {
            if n > 0 && frame == f.place.len() {
                near = distances(places, n, frame, &f.place).iter().map(|&d| d <= self.reach)
                    .collect();
            }
        }
        if let Some(f) = focus.take() {
            let held = self.hold * f.salience;
            let close: Vec<usize> = (0..n).filter(|&i| near[i]).collect();
            let values: Vec<f64> = close.iter().map(|&i| s[i]).collect();
            let won = if values.is_empty() { None } else {
                let k = np::argmax(&values);
                (values[k] > held).then(|| close[k])
            };
            focus = Some(match won {
                // renewed where it is: it follows the place
                Some(w) => Focus { place: place(w), sense: sense[w], index: index[w],
                                   salience: s[w], held: f.held + 1, turned: false },
                None => Focus { salience: held, held: f.held + 1, turned: false, ..f },
            });
        }
        if n > 0 {
            let others: Vec<f64> = (0..n).map(|i| if near[i] { 0.0 } else { s[i] }).collect();
            let best = np::argmax(&others);
            if others[best] > focus.as_ref().map_or(0.0, |f| f.salience) {
                focus = Some(Focus { place: place(best), sense: sense[best], index: index[best],
                                     salience: others[best], held: 1, turned: true });
            }
        }
        if focus.as_ref().is_some_and(|f| f.salience < self.least) {
            focus = None;
        }
        if focus.as_ref().is_none_or(|f| f.turned) {
            self.leave();  // the focus moves on or lets go: the place it leaves is inhibited
        }
        self.focus = focus;
        Ok(self.focus.clone())
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "attend" => {
                let shape = args.shape("places")?;
                if shape.len() != 2 {
                    return Err(refused("the places hold one place a unit"));
                }
                let focus = self.attend(&args.f64s("surprise")?, &args.f64s("places")?,
                                        shape[1], &args.i64s("sense")?)?;
                Focus::export(focus.as_ref(), "focus", out);
            }
            "_leave" => self.leave(),
            "reset" => self.reset(),
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
