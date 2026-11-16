//! `cadence.space.map.Map`: what is where, held in a sheet's settled activity, with its learned
//! physics (`physics`) and its cells' surprise (`surprise::Surprise`). A moment, a question by
//! place, the inventory and the forecast as the Python computes them, in its order; the map's
//! settle, its reports' pieces, claims and writes, its pieces and its moves are the compiled
//! kernels of `cadence.compiled` (`crate::map`).
//!
//! The camera (`lit` in the Python, a function of places) is a `Camera` the caller implements.

use super::physics::{self as ph, Physics};
use super::surprise::Surprise;
use crate::np;
use crate::state::{Array, Error, Named};
use crate::View;

const SENSES: usize = 3; // seen, heard, felt
const SEEN: i64 = 0;
// a thing's state in the inventory (`things`), after the senses
const HELD: i64 = 3;
const NEAR: i64 = 4;
const MOVING: i64 = 5;
const NOT_KNOWN_WHERE: i64 = 6;

/// What the camera sees: whether it sees each of a set of places.
pub trait Camera {
    /// `points` holds x, y pairs; one answer each.
    fn sees(&mut self, points: &[f64]) -> Result<Vec<bool>, Error>;
}

/// `lit`: nothing said, flags on cells, or the camera asked about places.
pub enum Lit<'a> {
    Nothing,
    Flags(Vec<bool>),
    Camera(&'a mut dyn Camera),
}

/// One report of a moment: a kind at a place by a sense (0 seen, 1 heard, 2 felt), with the
/// amounts it names on the value units (`named` flags them).
#[derive(Clone, Debug)]
pub struct Report {
    pub kind: usize,
    pub place: [f64; 2],
    pub sense: i64,
    pub values: Vec<f64>,
    pub named: Vec<bool>,
}

/// The thing at a place, read whole (`thing_at`).
#[derive(Clone, Debug)]
pub struct Thing {
    pub kind: usize,
    pub values: Vec<f64>,
    pub place: [f64; 2],
    pub bearing: f64,
    pub distance: f64,
}

/// One entry of the inventory (`things`); `place` None for a kind known with no place.
#[derive(Clone, Debug)]
pub struct Entry {
    pub kind: usize,
    pub place: Option<[f64; 2]>,
    pub pieces: Vec<Vec<usize>>,
    pub since: Option<i64>,
    pub state: i64,
    pub velocity: Option<[f64; 2]>,
    pub values: Vec<Option<i64>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Empty {
    Near,
    Clear,
    Keep,
}

/// The plain graph the map settles on: `compiled._graph_arrays` of `Map._plain`.
#[derive(Clone, Debug)]
struct Graph {
    flat: Vec<f64>,
    pairs: Vec<i64>,
    apical_pairs: Vec<i64>,
    starts: Vec<i64>,
    pair_pre: Vec<i64>,
    pair_post: Vec<i64>,
    offset: Vec<i64>,
    freezable: Vec<u8>,
    bias: Vec<f64>,
    rule: [f64; 5],
    pool_of: Vec<i64>,
    pool_index: Vec<i64>,
    pool_starts: Vec<i64>,
    pool_run: Vec<i64>,
    compartment: [f64; 3],
    runs: Vec<i64>,
}

fn vec_of<T>(v: &[T]) -> View<T> {
    View { p: v.as_ptr() as *mut T, n0: v.len(), n1: 1, len: v.len() }
}

fn mat_of<T>(v: &[T], n0: usize, n1: usize) -> View<T> {
    View { p: v.as_ptr() as *mut T, n0, n1, len: n0 * n1 }
}

fn vec_mut<T>(v: &mut [T]) -> View<T> {
    View { p: v.as_mut_ptr(), n0: v.len(), n1: 1, len: v.len() }
}

fn mat_mut<T>(v: &mut [T], n0: usize, n1: usize) -> View<T> {
    View { p: v.as_mut_ptr(), n0, n1, len: n0 * n1 }
}

fn refused(why: &str) -> Error {
    Error::Shape(why.to_string())
}

impl Graph {
    fn import(s: &Named) -> Result<Graph, Error> {
        let g = s.under("graph");
        let size = g.i64("flat.size")? as usize;
        let mut flat = vec![0.0; size];
        let (at, value) = (g.i64s("flat.at")?, g.f64s("flat.value")?);
        if at.len() != value.len() {
            return Err(refused("the graph's flat entries do not fit"));
        }
        for (&i, &x) in at.iter().zip(&value) {
            *flat.get_mut(i as usize).ok_or_else(|| refused("graph.flat.at"))? = x;
        }
        if g.bools("has_apical")?[0] {
            return Err(refused("the core settles a map's plain graph"));
        }
        let out = Graph {
            flat,
            pairs: g.i64s("pairs")?,
            apical_pairs: g.i64s("apical_pairs")?,
            starts: g.i64s("starts")?,
            pair_pre: g.i64s("pair_pre")?,
            pair_post: g.i64s("pair_post")?,
            offset: g.i64s("offset")?,
            freezable: g.bools("freezable")?.iter().map(|&b| b as u8).collect(),
            bias: g.f64s("bias")?,
            rule: [g.f64("dt")?, g.f64("slope")?, g.f64("threshold")?, g.f64("rest")?,
                   g.f64("leak")?],
            pool_of: g.i64s("pool_of")?,
            pool_index: g.i64s("pool_index")?,
            pool_starts: g.i64s("pool_starts")?,
            pool_run: g.i64s("pool_run")?,
            compartment: [g.f64("fraction")?, g.f64("floor2")?, g.f64("tiny")?],
            runs: g.i64s("runs")?,
        };
        if out.runs.len() != 16 || out.runs.iter().any(|&r| r < 0) {
            return Err(refused("graph.runs"));
        }
        Ok(out)
    }

    /// As `cadence.core.parts.map` names it: `flat` by its entries that are not +0.0.
    fn export(&self, out: &mut Named) {
        let at: Vec<i64> = (0..self.flat.len())
            .filter(|&i| self.flat[i].to_bits() != 0).map(|i| i as i64).collect();
        let value: Vec<f64> = at.iter().map(|&i| self.flat[i as usize]).collect();
        out.put("graph.flat.size", Array::scalar_i64(self.flat.len() as i64));
        out.put("graph.flat.at", Array::from_i64(&[at.len()], &at));
        out.put("graph.flat.value", Array::from_f64(&[value.len()], &value));
        for (name, v) in [("pairs", &self.pairs), ("apical_pairs", &self.apical_pairs),
                          ("starts", &self.starts), ("pair_pre", &self.pair_pre),
                          ("pair_post", &self.pair_post), ("offset", &self.offset)] {
            out.put(&format!("graph.{name}"), Array::from_i64(&[v.len()], v));
        }
        let freezable: Vec<bool> = self.freezable.iter().map(|&b| b != 0).collect();
        out.put("graph.freezable", Array::from_bool(&[freezable.len()], &freezable));
        out.put("graph.bias", Array::from_f64(&[self.bias.len()], &self.bias));
        for (name, x) in ["dt", "slope", "threshold", "rest", "leak"].iter().zip(self.rule) {
            out.put(&format!("graph.{name}"), Array::scalar_f64(x));
        }
        out.put("graph.has_apical", Array::from_bool(&[], &[false]));
        for (name, v) in [("pool_of", &self.pool_of), ("pool_index", &self.pool_index),
                          ("pool_starts", &self.pool_starts), ("pool_run", &self.pool_run)] {
            out.put(&format!("graph.{name}"), Array::from_i64(&[v.len()], v));
        }
        for (name, x) in ["fraction", "floor2", "tiny"].iter().zip(self.compartment) {
            out.put(&format!("graph.{name}"), Array::scalar_f64(x));
        }
        out.put("graph.runs", Array::from_i64(&[self.runs.len()], &self.runs));
    }

    fn neurons(&self) -> usize {
        self.starts.last().copied().unwrap_or(0).max(0) as usize
    }

    fn run(&self, k: usize) -> std::ops::Range<usize> {
        self.runs[2 * k] as usize..self.runs[2 * k + 1] as usize
    }

    fn arrays(&self) -> crate::map::MapGraph {
        crate::map::MapGraph {
            flat: vec_of(&self.flat), pairs: vec_of(&self.pairs), flat_apical: vec_of(&self.flat),
            apical_pairs: vec_of(&self.apical_pairs), starts: vec_of(&self.starts),
            pair_pre: vec_of(&self.pair_pre), pair_post: vec_of(&self.pair_post),
            offset: vec_of(&self.offset), freezable: vec_of(&self.freezable), bias: vec_of(&self.bias),
            dt: self.rule[0], slope: self.rule[1], threshold: self.rule[2], rest: self.rule[3],
            leak: self.rule[4], has_apical: 0, pool_of: vec_of(&self.pool_of),
            pool_index: vec_of(&self.pool_index), pool_starts: vec_of(&self.pool_starts),
            pool_run: vec_of(&self.pool_run), fraction: self.compartment[0],
            floor2: self.compartment[1], tiny: self.compartment[2],
        }
    }
}

/// The activity above rest of a potential: `max(tanh(v / 2), 0)`.
#[inline]
fn lit_up(v: f64) -> f64 {
    np::maximum(np::tanh(v / 2.0), 0.0)
}

/// NumPy's sort order of floats: NaN after every number.
#[inline]
fn before(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// `np.lexsort((secondary, primary))[-1]`: the last index in that order, the latest of equals.
fn lexsort_last(secondary: &[f64], primary: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..primary.len() {
        let below = before(primary[i], primary[best])
            || (!before(primary[best], primary[i]) && before(secondary[i], secondary[best]));
        if !below {
            best = i;
        }
    }
    best
}

/// A sum over the first axis of rows of `n`: the rows added in order to zeros.
fn column_sums(rows: impl Iterator<Item = Vec<f64>>, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for row in rows {
        for (o, x) in out.iter_mut().zip(row) {
            *o += x;
        }
    }
    out
}

/// The largest of values, a NaN where one is (`np.max`).
fn largest(values: impl Iterator<Item = f64>) -> f64 {
    let mut out: Option<f64> = None;
    for x in values {
        out = Some(match out {
            None => x,
            Some(m) => np::maximum(m, x),
        });
    }
    out.unwrap_or(f64::NAN)
}

fn least(values: impl Iterator<Item = f64>) -> f64 {
    let mut out: Option<f64> = None;
    for x in values {
        out = Some(match out {
            None => x,
            Some(m) => np::minimum(m, x),
        });
    }
    out.unwrap_or(f64::NAN)
}

/// The cells in `a` and not in `b`, ascending (`np.setdiff1d`).
fn setdiff(a: &[usize], b: &[bool]) -> Vec<usize> {
    let mut out: Vec<usize> = a.iter().copied().filter(|&c| !b[c]).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// One thing out of view the physics moved: its old cells, its cells now, its units, its record.
type Pending = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<f64>);

#[derive(Clone, Debug)]
pub struct Map {
    pub plane: [f64; 4],
    pub nx: usize,
    pub ny: usize,
    pub width: usize,
    pub units: usize,
    pub kinds: usize,
    /// each value group's units on the sheet: start and stop
    pub groups: Vec<(usize, usize)>,
    pub level: f64,
    pub empty: Empty,
    pub reach: f64,
    pub step: f64,
    pub stride: f64,
    pub margin: f64,
    pub floor: f64,
    pub trace: bool,
    pub codes: Vec<f64>,
    pub centers: Vec<f64>,
    pub low: [f64; 2],
    pub size: [f64; 2],
    offsets: Vec<f64>,
    density: Vec<f64>,
    pub potential: Vec<f64>,
    pub clock: Vec<f64>,
    pub lost: Vec<f64>,
    pub doubt: Vec<f64>,
    pub reading: Vec<f64>,
    pub sensed: Vec<(i64, i64)>,
    pub views: Vec<([f64; 2], Vec<bool>)>,
    pub identity: f64,
    pub work: [i64; 2],
    pub physics: Option<Physics>,
    pub motion: Vec<f64>,
    pub surprise: Option<Surprise>,
    graph: std::sync::Arc<Graph>,
}

impl Map {
    // -- the state

    pub fn import(s: &Named) -> Result<Map, Error> {
        let plane: [f64; 4] = s.f64s("plane")?.try_into().map_err(|_| refused("plane"))?;
        let cells = s.i64s("cells")?;
        if cells.len() != 2 || cells[0] < 3 || cells[1] < 3 {
            return Err(refused("a map takes 3 cells or more an axis"));
        }
        let (nx, ny) = (cells[0] as usize, cells[1] as usize);
        let count = nx * ny;
        let width = s.i64("width")? as usize;
        let codes_shape = s.shape("codes")?;
        if codes_shape.len() != 2 || codes_shape[1] > width {
            return Err(refused("codes"));
        }
        let (kinds, units) = (codes_shape[0], codes_shape[1]);
        let mut groups = Vec::new();
        let mut start = units;
        for many in s.i64s("values")? {
            groups.push((start, start + many as usize));
            start += many as usize;
        }
        if start != width {
            return Err(refused("the values' units and the kinds' fill the sheet's width"));
        }
        let empty = match s.i64("empty")? {
            0 => Empty::Near,
            1 => Empty::Clear,
            2 => Empty::Keep,
            _ => return Err(refused("empty is near, clear or keep")),
        };
        let two = |name: &str| -> Result<[f64; 2], Error> {
            s.f64s(name)?.try_into().map_err(|_| refused(name))
        };
        let looked = |m: f64| if m != 0.0 { 81 } else { 1 };
        let margin = s.f64("margin")?;
        let place = s.f64s("views.place")?;
        let seen = s.bools("views.seen")?;
        let k = place.len() / 2;
        if seen.len() != k * looked(margin) {
            return Err(refused("views.seen"));
        }
        let views = (0..k)
            .map(|i| ([place[2 * i], place[2 * i + 1]],
                      seen[i * looked(margin)..(i + 1) * looked(margin)].to_vec()))
            .collect();
        let sensed = s.i64s("sensed.kind")?.into_iter().zip(s.i64s("sensed.sense")?).collect();
        let physics = if s.holds("physics") { Some(Physics::import(s)?) } else { None };
        let motion = if physics.is_some() { s.f64s("motion")? } else { Vec::new() };
        if physics.is_some() && motion.len() != count * width * ph::FIELDS {
            return Err(refused("motion"));
        }
        let surprise = if s.holds("surprise") {
            let held = Surprise::import(&s.under("surprise"))?;
            if held.size() != count {
                return Err(refused("the map's surprise holds one unit a cell"));
            }
            Some(held)
        } else {
            None
        };
        let work = s.i64s("work")?;
        let out = Map {
            plane, nx, ny, width, units, kinds, groups,
            level: s.f64("level")?,
            empty,
            reach: s.f64("reach")?,
            step: s.f64("step")?,
            stride: s.f64("stride")?,
            margin,
            floor: s.f64("floor")?,
            trace: s.bools("trace")?[0],
            codes: s.f64s("codes")?,
            centers: s.f64s("centers")?,
            low: two("low")?,
            size: two("size")?,
            offsets: s.f64s("offsets")?,
            density: s.f64s("density")?,
            potential: s.f64s("potential")?,
            clock: s.f64s("clock")?,
            lost: s.f64s("lost")?,
            doubt: s.f64s("doubt")?,
            reading: s.f64s("reading")?,
            sensed,
            views,
            identity: s.f64("identity")?,
            work: [work[0], work[1]],
            physics,
            motion,
            surprise,
            graph: std::sync::Arc::new(Graph::import(s)?),
        };
        if out.potential.len() != count * width || out.clock.len() != count
            || out.doubt.len() != count || out.lost.len() != width
            || out.reading.len() != count + width || out.centers.len() != 2 * count
            || out.offsets.len() != 2 * out.density.len()
        {
            return Err(refused("the map's carried state does not fit its cells"));
        }
        Ok(out)
    }

    pub fn export(&self, out: &mut Named) {
        let count = self.count();
        let width = self.width;
        out.put("plane", Array::from_f64(&[4], &self.plane));
        out.put("cells", Array::from_i64(&[2], &[self.nx as i64, self.ny as i64]));
        out.put("width", Array::scalar_i64(width as i64));
        let many: Vec<i64> = self.groups.iter().map(|&(a, b)| (b - a) as i64).collect();
        out.put("values", Array::from_i64(&[many.len()], &many));
        out.put("level", Array::scalar_f64(self.level));
        let empty = match self.empty {
            Empty::Near => 0,
            Empty::Clear => 1,
            Empty::Keep => 2,
        };
        out.put("empty", Array::scalar_i64(empty));
        for (name, x) in [("reach", self.reach), ("step", self.step), ("stride", self.stride),
                          ("margin", self.margin), ("floor", self.floor)] {
            out.put(name, Array::scalar_f64(x));
        }
        out.put("trace", Array::from_bool(&[], &[self.trace]));
        out.put("codes", Array::from_f64(&[self.kinds, self.units], &self.codes));
        out.put("centers", Array::from_f64(&[count, 2], &self.centers));
        out.put("low", Array::from_f64(&[2], &self.low));
        out.put("size", Array::from_f64(&[2], &self.size));
        out.put("offsets", Array::from_f64(&[self.density.len(), 2], &self.offsets));
        out.put("density", Array::from_f64(&[self.density.len()], &self.density));
        out.put("potential", Array::from_f64(&[count, width], &self.potential));
        out.put("clock", Array::from_f64(&[count], &self.clock));
        out.put("lost", Array::from_f64(&[width], &self.lost));
        out.put("doubt", Array::from_f64(&[count], &self.doubt));
        out.put("reading", Array::from_f64(&[count + width], &self.reading));
        let k = self.sensed.len();
        let kinds: Vec<i64> = self.sensed.iter().map(|s| s.0).collect();
        let senses: Vec<i64> = self.sensed.iter().map(|s| s.1).collect();
        out.put("sensed.kind", Array::from_i64(&[k], &kinds));
        out.put("sensed.sense", Array::from_i64(&[k], &senses));
        let looked = self.looked();
        let places: Vec<f64> = self.views.iter().flat_map(|v| v.0).collect();
        let seen: Vec<bool> = self.views.iter().flat_map(|v| v.1.iter().copied()).collect();
        out.put("views.place", Array::from_f64(&[self.views.len(), 2], &places));
        out.put("views.seen", Array::from_bool(&[self.views.len(), looked], &seen));
        out.put("identity", Array::scalar_f64(self.identity));
        out.put("work", Array::from_i64(&[2], &self.work));
        if let Some(p) = &self.physics {
            p.export(out);
            out.put("motion", Array::from_f64(&[count, width, ph::FIELDS], &self.motion));
        }
        if let Some(s) = &self.surprise {
            let mut part = Named::new();
            s.export(&mut part);
            out.put_under("surprise", &part);
        }
        self.graph.export(out);
    }

    // -- what the map's rules fix

    pub fn count(&self) -> usize {
        self.nx * self.ny
    }

    fn valued(&self) -> usize {
        self.width - self.units
    }

    fn looked(&self) -> usize {
        if self.margin != 0.0 {
            81
        } else {
            1
        }
    }

    fn code(&self, kind: usize, u: usize) -> f64 {
        self.codes[kind * self.units + u]
    }

    /// A kind's units (`np.flatnonzero(codes[kind])`).
    fn kind_units(&self, kind: usize) -> Vec<usize> {
        (0..self.units).filter(|&u| self.code(kind, u) != 0.0).collect()
    }

    /// The sheet's carried activity above rest (`_activity`).
    fn activity(&self) -> Vec<f64> {
        self.potential.iter().map(|&v| lit_up(v)).collect()
    }

    /// The least of a kind's units' activity at every cell (`_of_kind`).
    fn of_kind(&self, kind: usize, activity: &[f64]) -> Vec<f64> {
        let units = self.kind_units(kind);
        (0..self.count())
            .map(|c| least(units.iter().map(|&u| activity[c * self.width + u])))
            .collect()
    }

    /// The kinds whose every code unit holds activity somewhere (`_present`).
    fn present(&self, activity: &[f64]) -> Vec<usize> {
        let held: Vec<bool> = (0..self.units)
            .map(|u| (0..self.count()).any(|c| activity[c * self.width + u] > self.floor))
            .collect();
        (0..self.kinds)
            .filter(|&k| (0..self.units).all(|u| !(self.code(k, u) > 0.0) || held[u]))
            .collect()
    }

    /// The cells where a kind other than `kind` holds a thing (`_other_kinds`), as flags.
    fn other_kinds(&self, kind: usize, activity: &[f64]) -> Vec<bool> {
        let mut out = vec![false; self.count()];
        for other in self.present(activity) {
            if other != kind {
                for (c, x) in self.of_kind(other, activity).into_iter().enumerate() {
                    out[c] |= x > self.floor;
                }
            }
        }
        out
    }

    /// The connected pieces of active cells (`_pieces`, the compiled kernel's order).
    fn pieces(&self, active: &[bool]) -> Result<Vec<Vec<usize>>, Error> {
        let count = self.count();
        let flags: Vec<u8> = active.iter().map(|&b| b as u8).collect();
        let mut cells = vec![0i64; count];
        let mut bounds = vec![0i64; count + 1];
        let found = unsafe {
            let mut scratch = crate::Scratch::new(crate::map::pieces_room(count))
                .ok_or_else(|| refused("no room for the map's pieces"))?;
            crate::map::pieces(flags.as_ptr(), self.nx, self.ny, cells.as_mut_ptr(),
                               bounds.as_mut_ptr(), &mut scratch)
        };
        Ok((0..found)
            .map(|p| cells[bounds[p] as usize..bounds[p + 1] as usize]
                 .iter().map(|&c| c as usize).collect())
            .collect())
    }

    /// The plane's corners less the camera's error (`_inside`).
    fn inside(&self) -> ([f64; 2], [f64; 2]) {
        let high = [self.low[0] + self.nx as f64 * self.size[0],
                    self.low[1] + self.ny as f64 * self.size[1]];
        ([self.low[0] + self.margin, self.low[1] + self.margin],
         [high[0] - self.margin, high[1] - self.margin])
    }

    /// Places with masses as weights on cells (`_spread`).
    fn spread(&self, points: &[f64], masses: &[f64]) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let n = masses.len();
        let mut cell = vec![0usize; n];
        let (mut ax, mut ay) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let x = np::clip((points[2 * i] - self.low[0]) / self.size[0] - 0.5, 0.0,
                             nx as f64 - 1.0);
            let y = np::clip((points[2 * i + 1] - self.low[1]) / self.size[1] - 0.5, 0.0,
                             ny as f64 - 1.0);
            let ix = (x as i64).min(nx as i64 - 2);
            let iy = (y as i64).min(ny as i64 - 2);
            ax[i] = x - ix as f64;
            ay[i] = y - iy as f64;
            cell[i] = (ix * ny as i64 + iy) as usize;
        }
        let mut out = vec![0.0; self.count()];
        for i in 0..n {
            out[cell[i]] += masses[i] * (1.0 - ax[i]) * (1.0 - ay[i]);
        }
        for i in 0..n {
            out[cell[i] + ny] += masses[i] * ax[i] * (1.0 - ay[i]);
        }
        for i in 0..n {
            out[cell[i] + 1] += masses[i] * (1.0 - ax[i]) * ay[i];
        }
        for i in 0..n {
            out[cell[i] + ny + 1] += masses[i] * ax[i] * ay[i];
        }
        out
    }

    /// A place as weights on the four cells around it (`weights`).
    pub fn weights(&self, point: [f64; 2]) -> Vec<f64> {
        self.spread(&point, &[1.0])
    }

    /// A bump's places and their shares (`_points`).
    fn points(&self, mean: [f64; 2], var: f64) -> (Vec<f64>, Vec<f64>) {
        let sd = np::sqrt(if 0.0 > var { 0.0 } else { var });
        if sd < 1e-4 {
            return (mean.to_vec(), vec![1.0]);
        }
        let places = self.offsets.chunks_exact(2)
            .flat_map(|o| [mean[0] + sd * o[0], mean[1] + sd * o[1]]).collect();
        (places, self.density.clone())
    }

    /// A bump of `mass` around `mean` with variance `var` an axis, on cells (`_bump`).
    fn bump(&self, mean: [f64; 2], var: f64, mass: f64) -> Vec<f64> {
        if !(mass > 0.0) {
            return vec![0.0; self.count()];
        }
        let (places, shares) = self.points(mean, var);
        let masses: Vec<f64> = shares.iter().map(|s| mass * s).collect();
        self.spread(&places, &masses)
    }

    /// Whether the camera sees each of `points` (`_sees`).
    fn sees(&self, lit: &mut Lit, sight: &[bool], points: &[f64]) -> Result<Vec<bool>, Error> {
        let n = points.len() / 2;
        match lit {
            Lit::Camera(camera) => {
                let out = camera.sees(points)?;
                if out.len() != n {
                    return Err(refused("the camera answers each place once"));
                }
                Ok(out)
            }
            Lit::Nothing => Ok(vec![false; n]),
            Lit::Flags(_) => Ok((0..n)
                .map(|i| {
                    let at = |k: usize, m: usize| {
                        let x = libm::floor((points[2 * i + k] - self.low[k]) / self.size[k]);
                        (x as i64).clamp(0, m as i64 - 1) as usize
                    };
                    sight[at(0, self.nx) * self.ny + at(1, self.ny)]
                })
                .collect()),
        }
    }

    /// What the camera sees of a place, or of the 81 places on a grid of half-width `within`
    /// around it (`_looks`).
    fn looks(&self, lit: &mut Lit, sight: &[bool], at: [f64; 2], within: f64)
             -> Result<Vec<bool>, Error> {
        let points: Vec<f64> = if within != 0.0 {
            (0..81)
                .flat_map(|k| {
                    let g = |i: usize| i as f64 * 0.25 + -1.0;
                    [at[0] + within * g(k / 9), at[1] + within * g(k % 9)]
                })
                .collect()
        } else {
            vec![at[0] + 0.0, at[1] + 0.0]
        };
        self.sees(lit, sight, &points)
    }

    /// What a thing of `kind` on `cells` holds on every value unit (`_amounts_of`).
    fn amounts_of(&self, kind: usize, cells: &[usize], activity: &[f64]) -> Vec<f64> {
        let own = self.of_kind(kind, activity);
        let width = self.width;
        let held = column_sums(
            cells.iter().map(|&c| {
                let mut row = vec![own[c]];
                row.extend((self.units..width).map(|u| activity[c * width + u]));
                row
            }),
            1 + self.valued(),
        );
        if !(held[0] > self.floor) {
            return vec![0.0; self.valued()];
        }
        held[1..].iter().map(|&x| np::clip(x / held[0], 0.0, 1.0)).collect()
    }

    /// A thing's values leave `cells` with it, where no other kind is (`_lift`).
    fn lift(&self, kind: usize, cells: &[usize], activity: &[f64], carried: &mut [u8])
            -> Vec<f64> {
        if self.groups.is_empty() {
            return Vec::new();
        }
        let moved = self.amounts_of(kind, cells, activity);
        for c in setdiff(cells, &self.other_kinds(kind, activity)) {
            for u in self.units..self.width {
                carried[c * self.width + u] = 0;
            }
        }
        moved
    }

    // -- the settle

    /// `Map._settle` without a cue: every neuron's potential.
    #[allow(clippy::too_many_arguments)]
    fn settle(&self, write: Option<&[f64]>, carried: Option<&[u8]>, clock: Option<&[f64]>,
              fresh: Option<&[u8]>, kept: Option<&[u8]>, doubt: Option<&[f64]>,
              sure: Option<&[u8]>) -> Result<Vec<f64>, Error> {
        if self.trace {
            return Err(refused("a traced map settles in the Python"));
        }
        let (count, width) = (self.count(), self.width);
        let none_flags = [0u8; 1];
        let none_values = [0.0f64; 1];
        let flags = |x: Option<&[u8]>| vec_of(x.unwrap_or(&none_flags));
        let values = |x: Option<&[f64]>| vec_of(x.unwrap_or(&none_values));
        let mut out = vec![0.0; self.graph.neurons()];
        let taken = unsafe {
            crate::map::settle_map(
                &self.graph.arrays(), mat_of(&self.potential, count, width), flags(carried),
                carried.is_some() as i32, values(write), write.is_some() as i32,
                vec_of(clock.unwrap_or(&self.clock)), vec_of(&self.lost),
                vec_of(doubt.unwrap_or(&self.doubt)), flags(fresh), fresh.is_some() as i32,
                flags(kept), kept.is_some() as i32, flags(sure), sure.is_some() as i32,
                vec_of(&none_values), 0, vec_of(&self.graph.runs), 1, vec_mut(&mut out))
        };
        if taken == crate::map::NOT_FINITE {
            return Err(refused("drive must be finite and the brain must contain neurons"));
        }
        if taken < 0 {
            return Err(refused(&format!("the map's settle refused its arrays ({taken})")));
        }
        Ok(out)
    }

    fn read_out(&self, v: &[f64], k: usize) -> Vec<f64> {
        v[self.graph.run(k)].iter().map(|&x| lit_up(x)).collect()
    }

    // -- one moment

    /// The sheet the map carries into the next moment (`_expected`).
    fn expected(&self) -> Result<Vec<f64>, Error> {
        match &self.physics {
            Some(p) if p.taught() => Ok(self.forecast(1)?.potential),
            _ => Ok(self.potential.clone()),
        }
    }

    /// `forecast(moments)`: the map that many moments on as its physics predicts, with nothing
    /// seen, as a copy; the map and its physics are left as they are.
    pub fn forecast(&self, moments: i64) -> Result<Map, Error> {
        if self.physics.is_none() {
            return Err(refused("a forecast is read with physics: Map(..., physics=True)"));
        }
        let mut ahead = self.clone();
        ahead.surprise = None;
        ahead.trace = false;
        if let Some(p) = ahead.physics.as_mut() {
            p.learning = false;
        }
        for _ in 0..moments.max(0) {
            ahead.moment(&[], &mut Lit::Nothing)?;
        }
        if let (Some(p), Some(mine)) = (ahead.physics.as_mut(), self.physics.as_ref()) {
            p.learning = mine.learning;
        }
        Ok(ahead)
    }

    /// `moment(seen=, heard=, felt=, lit=)`: one moment's reports, in the order seen, heard,
    /// felt, and what the camera sees.
    pub fn moment(&mut self, reports: &[Report], lit: &mut Lit) -> Result<(), Error> {
        if self.trace {
            return Err(refused("a traced map settles in the Python"));
        }
        let (count, width, valued) = (self.count(), self.width, self.valued());
        for r in reports {
            if r.kind >= self.kinds || !(0..SENSES as i64).contains(&r.sense)
                || r.values.len() != valued || r.named.len() != valued
            {
                return Err(refused("a report is a kind of the map's, a place and its values"));
            }
            if !r.place.iter().all(|x| x.is_finite()) {
                return Err(refused("a report's place is finite"));
            }
            if r.named.iter().zip(&r.values).any(|(&n, &x)| n && !(0.0..=1.0).contains(&x)) {
                return Err(refused("a value's amounts lie in 0 to 1"));
            }
        }
        let expected = if self.surprise.is_some() { Some(self.expected()?) } else { None };
        let sight: Vec<bool> = match lit {
            Lit::Camera(camera) => {
                let seen = camera.sees(&self.centers)?;
                if seen.len() != count {
                    return Err(refused("the camera answers each cell once"));
                }
                seen
            }
            Lit::Flags(flags) => {
                if flags.len() != count {
                    return Err(refused("lit flags each cell"));
                }
                flags.clone()
            }
            Lit::Nothing => vec![false; count],
        };
        let before = self.potential.clone();
        let sensed: Vec<bool> = self.clock.iter().map(|&c| c < 0.5).collect();
        let activity = self.activity();
        let mut carried = vec![1u8; count * width];
        let mut write = vec![0.0; count * width];
        let mut clock = self.clock.clone();
        let mut fresh = vec![0u8; count];
        let mut doubt = self.doubt.clone();
        let mut lost = self.lost.clone();
        let mut claimed = vec![0u8; count * width];
        let mut lines: Vec<(Vec<usize>, Vec<f64>, i64)> = Vec::new();
        if self.physics.is_some() {
            self.identity =
                largest((0..count * width).map(|i| self.motion[i * ph::FIELDS + ph::IDENT]));
        }
        // each report: its pieces, claims and writes in one compiled call, then its line
        let no_motion = [0.0f64; 1];
        let mut near_cells = vec![0i64; 2 * count];
        let mut near_bounds = vec![0i64; 2 * count + 1];
        for r in reports {
            let units = self.kind_units(r.kind);
            let named: Vec<u8> = r.named.iter().map(|&b| b as u8).collect();
            let physics = self.physics.is_some();
            let motion = if physics { mat_of(&self.motion, count, width * ph::FIELDS) }
                         else { mat_of(&no_motion, 1, 1) };
            let (hidden, ident, fields) = if physics {
                (ph::HIDDEN as i32, ph::IDENT as i32, ph::FIELDS as i32)
            } else {
                (0, 0, 1)
            };
            let taken = unsafe {
                crate::map::report(
                    r.kind as i32, r.place[0], r.place[1], mat_of(&self.codes, self.kinds, self.units),
                    mat_of(&self.centers, count, 2), mat_of(&activity, count, width), vec_of(&self.doubt),
                    motion, physics as i32, hidden, ident, fields, self.low[0], self.low[1],
                    self.size[0], self.size[1], self.floor, self.step, self.stride, self.level,
                    self.nx as i32, self.ny as i32, (!self.groups.is_empty()) as i32,
                    vec_of(&r.values), vec_of(&named), mat_mut(&mut claimed, count, width),
                    mat_mut(&mut write, count, width), vec_mut(&mut fresh),
                    vec_mut(&mut lost), vec_mut(&mut near_cells), vec_mut(&mut near_bounds))
            };
            if taken < 0 {
                return Err(refused(&format!("the map's report kernel refused ({taken})")));
            }
            if physics {
                let near: Vec<Vec<usize>> = (0..taken as usize)
                    .map(|p| near_cells[near_bounds[p] as usize..near_bounds[p + 1] as usize]
                         .iter().map(|&c| c as usize).collect())
                    .collect();
                let record = self.line(&units, &near, r.place, r.sense);
                lines.push((units, record, r.sense));
            }
        }
        for (c, k) in carried.iter_mut().zip(&claimed) {
            *c &= (*k == 0) as u8;
        }
        let mut held_by_physics = vec![false; count * width];
        let mut pending: Vec<Pending> = Vec::new();
        if self.physics.as_ref().is_some_and(|p| p.taught()) {
            held_by_physics = self.predict(&activity, &claimed, &sight, lit, &mut write,
                                           &mut clock, &mut doubt, &mut lost, &mut pending)?;
        }
        // a thing whose held place the camera sees, with no report that claims it, is not there
        let mut empty = vec![false; count];
        for kind in self.present(&activity) {
            let units = self.kind_units(kind);
            let mut own = self.of_kind(kind, &activity);
            for (c, x) in own.iter_mut().enumerate() {
                let claim = units.iter().any(|&u| claimed[c * width + u] != 0);
                *x *= if claim { 0.0 } else { 1.0 };
                let physics = units.iter().any(|&u| held_by_physics[c * width + u]);
                *x *= if physics { 0.0 } else { 1.0 };
            }
            let active: Vec<bool> = own.iter().map(|&x| x > self.floor).collect();
            for piece in self.pieces(&active)? {
                let weights: Vec<f64> = piece.iter().map(|&c| own[c]).collect();
                let xs: Vec<f64> = piece.iter().map(|&c| self.centers[2 * c]).collect();
                let ys: Vec<f64> = piece.iter().map(|&c| self.centers[2 * c + 1]).collect();
                let total = np::sum(&weights);
                let at = [np::dot(&weights, &xs) / total, np::dot(&weights, &ys) / total];
                let doubts: Vec<f64> = piece.iter().map(|&c| self.doubt[c]).collect();
                if np::mean(&doubts) > 0.5 && self.empty == Empty::Near {
                    // a thing in doubt: sight takes away the cells it sees
                    let seen_here: Vec<usize> = piece.iter().copied().filter(|&c| sight[c]).collect();
                    if !seen_here.is_empty() {
                        for &c in &seen_here {
                            for &u in &units {
                                carried[c * width + u] = 0;
                            }
                        }
                        self.lift(kind, &seen_here, &activity, &mut carried);
                        for &c in &seen_here {
                            empty[c] = true;
                        }
                        if seen_here.len() == piece.len()
                            && !(0..count).any(|c| !sight[c] && own[c] > self.floor)
                        {
                            for &u in &units {
                                lost[u] = 1.0;
                            }
                        }
                    }
                    continue;
                }
                let gap = |p: [f64; 2]| np::norm(&[p[0] - at[0], p[1] - at[1]]);
                let views: Vec<&([f64; 2], Vec<bool>)> =
                    self.views.iter().filter(|v| gap(v.0) <= self.step).collect();
                let mut seen_empty = if piece.iter().any(|&c| sensed[c]) && !views.is_empty() {
                    let mut best = 0;
                    for (i, v) in views.iter().enumerate() {
                        if gap(v.0) < gap(views[best].0) {
                            best = i;
                        }
                    }
                    let (place, then) = (views[best].0, views[best].1.clone());
                    let now = self.looks(lit, &sight, place, self.margin)?;
                    now.iter().any(|&x| x) && !then.iter().zip(&now).any(|(&t, &n)| t && !n)
                } else {
                    self.looks(lit, &sight, at, self.margin)?.iter().all(|&x| x)
                };
                seen_empty = seen_empty && self.empty != Empty::Keep;
                if !seen_empty {
                    continue; // the thing stays where the map held it
                }
                for &c in &piece {
                    for &u in &units {
                        carried[c * width + u] = 0;
                    }
                }
                let moved = self.lift(kind, &piece, &activity, &mut carried);
                for &c in &piece {
                    empty[c] = true;
                }
                if self.empty == Empty::Near {
                    let wide: Vec<f64> = (0..count)
                        .map(|c| {
                            let (dx, dy) = (self.centers[2 * c] - at[0], self.centers[2 * c + 1] - at[1]);
                            let apart = np::sqrt(-0.0 + dx * dx + dy * dy);
                            let q = apart / self.reach;
                            let near = !sight[c] && apart <= 2.0 * self.reach;
                            np::exp(-0.5 * (q * q)) * if near { 1.0 } else { 0.0 }
                        })
                        .collect();
                    if wide.iter().any(|&x| x != 0.0) {
                        let tops: Vec<f64> = piece.iter()
                            .map(|&c| largest(units.iter().map(|&u| before[c * width + u])))
                            .collect();
                        let length = np::norm(&tops);
                        let size = np::norm(&wide);
                        for c in 0..count {
                            let a = length * wide[c] / size;
                            for &u in &units {
                                write[c * width + u] += a;
                            }
                            for (v, &m) in moved.iter().enumerate() {
                                write[c * width + self.units + v] += a * m;
                            }
                        }
                        let then = largest(piece.iter().map(|&c| self.clock[c]));
                        for c in 0..count {
                            if wide[c] > 0.0 {
                                clock[c] = then;
                                doubt[c] = 1.0;
                            }
                        }
                        continue;
                    }
                }
                for &u in &units {
                    lost[u] = 1.0; // known by its kind, with no place
                }
            }
        }
        self.lost = lost;
        // a cell a sense writes, or one seen empty, holds no doubt
        let sure: Vec<u8> = (0..count)
            .map(|c| {
                let claim = (0..width).any(|u| claimed[c * width + u] != 0);
                let quiet = !(0..width).any(|u| activity[c * width + u] > self.floor);
                (fresh[c] != 0 || claim || empty[c] || (sight[c] && quiet)) as u8
            })
            .collect();
        let written = write.iter().any(|&x| x != 0.0);
        let v = self.settle(if written { Some(&write) } else { None }, Some(&carried),
                            Some(&clock), Some(&fresh), None, Some(&doubt), Some(&sure))?;
        let g = &self.graph;
        self.potential = v[g.run(2)].to_vec();
        self.clock = v[g.run(5)].to_vec();
        self.lost = v[g.run(6)].to_vec();
        self.doubt = v[g.run(7)].to_vec();
        let mut reading = self.read_out(&v, 3);
        reading.extend(self.read_out(&v, 4));
        self.reading = reading;
        if self.physics.is_some() {
            self.carry_records(&claimed, reports, &lines, &pending);
        }
        if let Some(expected) = expected {
            let moves = self.moves(&expected)?;
            if let Some(s) = self.surprise.as_mut() {
                s.observe(&moves, 1, true, None)?;
            }
        }
        self.sensed = reports.iter().map(|r| (r.kind as i64, r.sense)).collect();
        let mut views = Vec::with_capacity(reports.len());
        for r in reports {
            views.push((r.place, self.looks(lit, &sight, r.place, self.margin)?));
        }
        self.views = views;
        self.work[0] += 1;
        Ok(())
    }

    // -- learned physics

    /// The motion record a report carries on (`_line`).
    fn line(&mut self, units: &[usize], near: &[Vec<usize>], place: [f64; 2], sense: i64)
            -> Vec<f64> {
        let (width, f) = (self.width, ph::FIELDS);
        let u0 = units[0];
        let rec_at = |c: usize| (c * width + u0) * f;
        let mut record: Option<Vec<f64>> = None;
        if !near.is_empty() {
            let mut cells: Vec<usize> = near.concat();
            cells.sort_unstable();
            cells.dedup();
            let held: Vec<usize> =
                cells.into_iter().filter(|&c| self.motion[rec_at(c) + ph::IDENT] > 0.0).collect();
            if !held.is_empty() {
                let counts: Vec<f64> = held.iter().map(|&c| self.motion[rec_at(c) + ph::COUNT]).collect();
                let potentials: Vec<f64> = held.iter().map(|&c| self.potential[c * width + u0]).collect();
                let c = held[lexsort_last(&potentials, &counts)];
                record = Some(self.motion[rec_at(c)..rec_at(c) + f].to_vec());
            }
        }
        let mut ident = 0.0;
        if record.is_none() {
            self.identity += 1.0;
            ident = self.identity;
        }
        if sense != SEEN {
            let mut named = vec![0.0; f];
            named[ph::IDENT] = match &record {
                None => ident,
                Some(r) => r[ph::IDENT],
            };
            return named;
        }
        let inside = self.inside();
        self.physics.as_mut().unwrap().see(record.as_deref(), place, ident, inside)
    }

    /// Things out of view that move, one moment on (`_predict`): the cells and units the
    /// physics holds; the drive goes into `write`, the clock, doubt and lost into theirs, each
    /// thing's record into `pending`.
    #[allow(clippy::too_many_arguments)]
    fn predict(&self, activity: &[f64], claimed: &[u8], sight: &[bool], lit: &mut Lit,
               write: &mut [f64], clock: &mut [f64], doubt: &mut [f64], lost: &mut [f64],
               pending: &mut Vec<Pending>) -> Result<Vec<bool>, Error> {
        use ph::{CORE, COUNT, ERROR, EST, FIELDS, HIDDEN, IDENT, TAIL, VEL};
        let p = self.physics.as_ref().unwrap();
        let (count, width) = (self.count(), self.width);
        let mut held = vec![false; count * width];
        let ident_at = |i: usize| self.motion[i * FIELDS + IDENT];
        let alive: Vec<bool> = self.potential.iter().map(|&v| v > 0.0).collect();
        let (lo, hi) = self.inside();
        let mut idents: Vec<f64> =
            (0..count * width).filter(|&i| ident_at(i) > 0.0 && alive[i]).map(ident_at).collect();
        idents.sort_by(|a, b| a.partial_cmp(b).unwrap());
        idents.dedup();
        for ident in idents {
            let mine: Vec<bool> = (0..count * width).map(|i| ident_at(i) == ident && alive[i]).collect();
            let units: Vec<usize> =
                (0..width).filter(|&u| (0..count).any(|c| mine[c * width + u])).collect();
            let cells: Vec<usize> =
                (0..count).filter(|&c| (0..width).any(|u| mine[c * width + u])).collect();
            if cells.iter().any(|&c| units.iter().any(|&u| claimed[c * width + u] != 0)) {
                continue; // a report took it this moment
            }
            let u0 = units[0];
            let counts: Vec<f64> =
                cells.iter().map(|&c| self.motion[(c * width + u0) * FIELDS + COUNT]).collect();
            let potentials: Vec<f64> = cells.iter().map(|&c| self.potential[c * width + u0]).collect();
            let at = (cells[lexsort_last(&potentials, &counts)] * width + u0) * FIELDS;
            let mut rec = self.motion[at..at + FIELDS].to_vec();
            let mut k = rec[HIDDEN] as i64;
            if k < 0 {
                continue; // at rest out of view: the map holds it
            }
            let nu = units.len();
            let field_units: Vec<f64> = (0..count)
                .flat_map(|c| units.iter().map(move |&u| (c, u)))
                .map(|(c, u)| if mine[c * width + u] { self.potential[c * width + u] } else { 0.0 })
                .collect();
            let field: Vec<f64> =
                (0..count).map(|c| np::sum(&field_units[c * nu..(c + 1) * nu])).collect();
            let whole = np::sum(&field);
            let profile: Vec<f64> = column_sums(
                (0..count).map(|c| field_units[c * nu..(c + 1) * nu].to_vec()), nu)
                .into_iter().map(|x| x / whole).collect();
            let (mut cm, mut tm) = (rec[CORE + 7], rec[TAIL + 3]);
            let mut trail: Vec<f64>;
            if k == 0 {
                // it goes out of view now
                let share = if cells.iter().any(|&c| self.clock[c] < 0.5) {
                    p.moving([rec[VEL], rec[VEL + 1]], rec[COUNT] as i64, rec[ERROR + 2])
                } else {
                    0.0
                };
                if share < 0.5 {
                    rec[HIDDEN] = -1.0;
                    pending.push((cells.clone(), cells, units, rec));
                    continue;
                }
                let (px, pxv, pv) = (rec[ERROR], rec[ERROR + 1], rec[ERROR + 2]);
                let core = [rec[EST], rec[EST + 1], rec[VEL], rec[VEL + 1], px, pxv, pv,
                            share * whole];
                rec[CORE..CORE + 8].copy_from_slice(&core);
                trail = field.iter().map(|&f| (1.0 - share) * f).collect();
                cm = share * whole;
            } else {
                let spread = rec[CORE + 4] + (2 * k) as f64 * rec[CORE + 5]
                    + (k * k) as f64 * rec[CORE + 6];
                let core = self.bump([rec[CORE], rec[CORE + 1]], spread, cm);
                let tail = self.bump([rec[TAIL], rec[TAIL + 1]], rec[TAIL + 2], tm);
                trail = (0..count).map(|c| np::maximum(field[c] - core[c] - tail[c], 0.0)).collect();
            }
            k += 1;
            let mut cx = [rec[CORE], rec[CORE + 1]];
            let cv = [rec[CORE + 2], rec[CORE + 3]];
            let (mut px, pxv, pv) = (rec[CORE + 4], rec[CORE + 5], rec[CORE + 6]);
            let mut tx = [rec[TAIL], rec[TAIL + 1]];
            let mut tvar = rec[TAIL + 2];
            let speed = np::norm(&cv);
            let u = if speed > 0.0 { [cv[0] / speed, cv[1] / speed] } else { [0.0, 0.0] };
            let add = |trail: &mut Vec<f64>, b: Vec<f64>| {
                for (t, x) in trail.iter_mut().zip(b) {
                    *t += x;
                }
            };
            if tm > 0.0 {
                // the tail: a share comes to rest, it goes ahead and widens
                add(&mut trail, self.bump(tx, tvar, p.rest * tm));
                tm *= 1.0 - p.rest;
                let step = p.ahead * speed;
                tx = [np::clip(tx[0] + step * u[0], lo[0], hi[0]),
                      np::clip(tx[1] + step * u[1], lo[1], hi[1])];
                let wide = np::sqrt(tvar) + p.aside * speed;
                tvar = wide * wide;
            }
            if cm > 0.0 {
                // the core: a share comes to rest, a share leaves its line, it moves on
                let was = px + (2 * (k - 1)) as f64 * pxv + ((k - 1) * (k - 1)) as f64 * pv;
                add(&mut trail, self.bump(cx, was, p.rest * cm));
                let leave = p.leave * cm;
                if leave > 0.0 {
                    let total = tm + leave;
                    let middle = [(tm * tx[0] + leave * cx[0]) / total,
                                  (tm * tx[1] + leave * cx[1]) / total];
                    let d = |x: [f64; 2]| {
                        let (a, b) = (x[0] - middle[0], x[1] - middle[1]);
                        a * a + b * b
                    };
                    tvar = (tm * (tvar + d(tx) / 2.0) + leave * (was + d(cx) / 2.0)) / total;
                    tx = middle;
                    tm = total;
                }
                cm *= 1.0 - p.rest - p.leave;
                let ahead = [cx[0] + cv[0], cx[1] + cv[1]];
                if (0..2).any(|j| ahead[j] < lo[j] || ahead[j] > hi[j]) {
                    add(&mut trail, self.bump(cx, was, cm)); // the plane's edge: it rests there
                    cm = 0.0;
                }
                cx = ahead;
            }
            let var = px + (2 * k) as f64 * pxv + (k * k) as f64 * pv;
            if self.empty != Empty::Keep {
                // sight: the trail's cells it sees, the share of each bump
                for (t, &s) in trail.iter_mut().zip(sight) {
                    *t *= if s { 0.0 } else { 1.0 };
                }
                let core_at = if cm > 0.0 { self.points(cx, var).0 } else { Vec::new() };
                let tail_at = if tm > 0.0 { self.points(tx, tvar).0 } else { Vec::new() };
                let mut places = core_at.clone();
                places.extend(&tail_at);
                let dark: Vec<bool> =
                    self.sees(lit, sight, &places)?.into_iter().map(|x| !x).collect();
                let n_core = core_at.len() / 2;
                let (m2, x2, var2, cut) = self.trim(cm, cx, var, &dark[..n_core]);
                cm = m2;
                cx = x2;
                if cut && var2 < var {
                    let x = var2 - (2 * k) as f64 * pxv - (k * k) as f64 * pv;
                    px = if 0.0 > x { 0.0 } else { x };
                }
                let (m3, x3, v3, _) = self.trim(tm, tx, tvar, &dark[n_core..]);
                tm = m3;
                tx = x3;
                tvar = v3;
            }
            let left = np::sum(&trail) + cm + tm;
            if !(left > 1e-12) && k == 1 {
                // reported the moment before, and sight finds no part of it now
                rec[CORE..TAIL + 4].iter_mut().for_each(|x| *x = 0.0);
                rec[HIDDEN] = -1.0;
                pending.push((cells.clone(), cells, units, rec));
                continue;
            }
            for &c in &cells {
                for &un in &units {
                    held[c * width + un] = true;
                }
            }
            let others: Vec<bool> = (0..count)
                .map(|c| (0..width).any(|un| !units.contains(&un) && activity[c * width + un] > self.floor))
                .collect();
            if !(left > 1e-12) {
                // sight took all of it: known by its kind, with no place
                for &un in &units {
                    lost[un] = 1.0;
                }
                for &c in &cells {
                    for (j, &un) in units.iter().enumerate() {
                        write[c * width + un] -= field_units[c * nu + j];
                    }
                    if !others[c] {
                        doubt[c] = 0.0;
                    }
                }
                pending.push((cells, Vec::new(), units, rec));
                continue;
            }
            let scale = whole / left;
            trail.iter_mut().for_each(|t| *t *= scale);
            cm *= scale;
            tm *= scale;
            let core_var = px + (2 * k) as f64 * pxv + (k * k) as f64 * pv;
            if cm < 1e-3 * whole {
                add(&mut trail, self.bump(cx, core_var, cm));
                cm = 0.0;
            }
            if tm < 1e-3 * whole {
                add(&mut trail, self.bump(tx, tvar, tm));
                tm = 0.0;
            }
            let (core, tail) = (self.bump(cx, core_var, cm), self.bump(tx, tvar, tm));
            let mut new: Vec<f64> = (0..count).map(|c| trail[c] + core[c] + tail[c]).collect();
            let now: Vec<usize> = (0..count).filter(|&c| new[c] > self.floor * whole).collect();
            let mut is_now = vec![false; count];
            for &c in &now {
                is_now[c] = true;
            }
            for c in 0..count {
                if !is_now[c] {
                    new[c] = 0.0;
                }
            }
            for c in 0..count {
                for (j, &un) in units.iter().enumerate() {
                    write[c * width + un] += new[c] * profile[j] - field_units[c * nu + j];
                }
            }
            // the doubt and the clock of its cells, where no other thing holds them
            for c in setdiff(&cells, &is_now) {
                if !others[c] {
                    doubt[c] = 0.0;
                }
            }
            for &c in &now {
                if !others[c] {
                    doubt[c] = 1.0;
                    clock[c] = (k - 1) as f64; // the clock's bias counts this moment
                }
                for &un in &units {
                    held[c * width + un] = true;
                }
            }
            rec[HIDDEN] = if cm + tm > 0.0 { k as f64 } else { -1.0 };
            rec[CORE..CORE + 8].copy_from_slice(&[cx[0], cx[1], cv[0], cv[1], px, pxv, pv, cm]);
            rec[TAIL..TAIL + 4].copy_from_slice(&[tx[0], tx[1], tvar, tm]);
            pending.push((cells, now, units, rec));
        }
        Ok(held)
    }

    /// Sight takes the share of a bump it sees (`_trim`): its mass, middle and variance, and
    /// whether sight took any.
    fn trim(&self, mass: f64, mean: [f64; 2], var: f64, dark: &[bool]) -> (f64, [f64; 2], f64, bool) {
        if !(mass > 0.0) {
            return (0.0, mean, var, false);
        }
        let (places, shares) = self.points(mean, var);
        let flags: Vec<f64> = dark.iter().map(|&d| d as u8 as f64).collect();
        let share = np::dot(&shares, &flags);
        if share <= 0.0 {
            return (0.0, mean, var, true);
        }
        if share >= 1.0 {
            return (mass, mean, var, false);
        }
        let kept: Vec<f64> = shares.iter().zip(&flags).map(|(s, d)| s * d / share).collect();
        let xs: Vec<f64> = places.chunks_exact(2).map(|p| p[0]).collect();
        let ys: Vec<f64> = places.chunks_exact(2).map(|p| p[1]).collect();
        let middle = [np::dot(&kept, &xs), np::dot(&kept, &ys)];
        let apart: Vec<f64> = places.chunks_exact(2)
            .map(|p| {
                let (a, b) = (p[0] - middle[0], p[1] - middle[1]);
                a * a + b * b
            })
            .collect();
        let spread = np::dot(&apart, &kept) / 2.0;
        (mass * share, middle, if spread < var { spread } else { var }, true)
    }

    /// The motion records follow their things (`_carry_records`).
    fn carry_records(&mut self, claimed: &[u8], reports: &[Report],
                     lines: &[(Vec<usize>, Vec<f64>, i64)], pending: &[Pending]) {
        let (count, width, f) = (self.count(), self.width, ph::FIELDS);
        let held: Vec<bool> = (0..count * width).map(|i| self.motion[i * f + ph::IDENT] > 0.0).collect();
        for i in 0..count * width {
            if claimed[i] != 0 && held[i] {
                self.motion[i * f..(i + 1) * f].iter_mut().for_each(|x| *x = 0.0);
            }
        }
        for (old, now, units, rec) in pending {
            for &c in old {
                for &u in units {
                    let i = c * width + u;
                    self.motion[i * f..(i + 1) * f].iter_mut().for_each(|x| *x = 0.0);
                }
            }
            for &c in now {
                for &u in units {
                    let i = c * width + u;
                    self.motion[i * f..(i + 1) * f].copy_from_slice(rec);
                }
            }
        }
        for seen in [false, true] {
            for (r, (units, rec, sense)) in reports.iter().zip(lines) {
                if (*sense == SEEN) != seen {
                    continue;
                }
                let cells: Vec<usize> = self.weights(r.place).iter().enumerate()
                    .filter(|(_, &w)| w > 0.0).map(|(c, _)| c).collect();
                let mut all = units.clone();
                for e in self.units..width {
                    if cells.iter().any(|&c| self.potential[c * width + e] > 0.0) {
                        all.push(e);
                    }
                }
                for &c in &cells {
                    for &u in &all {
                        let i = c * width + u;
                        self.motion[i * f..(i + 1) * f].copy_from_slice(rec);
                    }
                }
            }
        }
        for i in 0..count * width {
            if self.motion[i * f + ph::IDENT] > 0.0 && self.potential[i] <= 0.0 {
                self.motion[i * f..(i + 1) * f].iter_mut().for_each(|x| *x = 0.0);
            }
        }
    }

    // -- surprise with a place

    /// `moves(expected)`: how far the moment moved each thing, in units of `step`.
    pub fn moves(&self, expected: &[f64]) -> Result<Vec<f64>, Error> {
        let (count, width) = (self.count(), self.width);
        if expected.len() != count * width {
            return Err(refused("expected holds the sheet's potentials"));
        }
        let mut out = vec![0.0; count];
        let code = unsafe {
            crate::map::moves(mat_of(&self.potential, count, width), mat_of(expected, count, width),
                              mat_of(&self.codes, self.kinds, self.units),
                              mat_of(&self.centers, count, 2), self.floor, self.stride, self.step,
                              self.nx as i32, self.ny as i32, vec_mut(&mut out))
        };
        if code < 0 {
            return Err(refused(&format!("the map's moves kernel refused its arrays ({code})")));
        }
        Ok(out)
    }

    // -- questions by place, and the inventory

    /// The kind whose code a pattern over the units matches best (`kind_of`).
    fn kind_of(&self, code: &[f64]) -> Option<usize> {
        let held: Vec<f64> = if code.len() == self.width { &code[..self.units] } else { code }
            .iter().map(|&x| np::maximum(x, 0.0)).collect();
        let top = largest(held.iter().copied());
        if !(top > 0.0) {
            return None;
        }
        let held: Vec<f64> = held.iter().map(|x| x / top).collect();
        let score: Vec<f64> = (0..self.kinds)
            .map(|k| {
                let row = &self.codes[k * self.units..(k + 1) * self.units];
                let off: Vec<f64> = row.iter().map(|x| 1.0 - x).collect();
                (np::dot(row, &held) - np::dot(&off, &held)) / (np::sum(row))
            })
            .collect();
        Some(np::argmax(&score))
    }

    /// `thing_at(place, within, cells=, body=, heading=)`: the one thing at a place, read
    /// whole; None where the cells hold no thing.
    pub fn thing_at(&mut self, place: Option<[f64; 2]>, within: Option<f64>,
                    cells: Option<Vec<bool>>, body: [f64; 2], heading: f64)
                    -> Result<Option<Thing>, Error> {
        let count = self.count();
        let near: Vec<u8> = match cells {
            Some(flags) if flags.len() == count => flags.iter().map(|&b| b as u8).collect(),
            Some(_) => return Err(refused("cells flags each cell")),
            None => {
                let reach = match within {
                    None => 1.5 * np::maximum(self.size[0], self.size[1]),
                    Some(w) => w,
                };
                let at = place.unwrap_or([f64::NAN, f64::NAN]);
                (0..count)
                    .map(|c| {
                        let (dx, dy) = (self.centers[2 * c] - at[0], self.centers[2 * c + 1] - at[1]);
                        (np::sqrt(-0.0 + dx * dx + dy * dy) <= reach) as u8
                    })
                    .collect()
            }
        };
        let v = self.settle(None, None, None, None, Some(&near), None, None)?;
        self.work[1] += 1;
        let what = self.read_out(&v, 4);
        let where_ = self.read_out(&v, 3);
        if !(largest(what[..self.units].iter().copied()) > self.floor) {
            return Ok(None);
        }
        let kind = self.kind_of(&what[..self.units]).unwrap();
        let own = least(self.kind_units(kind).into_iter().map(|u| what[u]));
        let xs: Vec<f64> = (0..count).map(|c| self.centers[2 * c]).collect();
        let ys: Vec<f64> = (0..count).map(|c| self.centers[2 * c + 1]).collect();
        let total = np::sum(&where_);
        let at = [np::dot(&where_, &xs) / total, np::dot(&where_, &ys) / total];
        let offset = [at[0] - body[0], at[1] - body[1]];
        let turn = libm::atan2(offset[1], offset[0]) - heading;
        let bearing = libm::atan2(libm::sin(turn), libm::cos(turn));
        let values = self.groups.iter()
            .flat_map(|&(a, b)| (a..b).map(|u| np::clip(what[u] / own, 0.0, 1.0)))
            .collect();
        Ok(Some(Thing { kind, values, place: at, bearing, distance: np::norm(&offset) }))
    }

    /// `sheet(kind)`: the map's activity as a picture, of one kind or of every unit summed.
    pub fn sheet(&self, kind: Option<usize>) -> Result<Vec<f64>, Error> {
        let activity = self.activity();
        match kind {
            Some(k) if k < self.kinds => Ok(self.of_kind(k, &activity)),
            Some(_) => Err(refused("no such kind on this map")),
            None => Ok(activity.chunks_exact(self.width).map(|row| np::sum(row)).collect()),
        }
    }

    /// With physics: the velocity of a thing out of view of which half or more moves
    /// (`_moving`).
    fn moving(&self, kind: usize, cells: &[usize]) -> Option<[f64; 2]> {
        use ph::{CORE, FIELDS, HIDDEN, IDENT, TAIL};
        let p = self.physics.as_ref()?;
        let width = self.width;
        let units = self.kind_units(kind);
        let u0 = units[0];
        let carried = cells.iter().position(|&c| {
            let r = (c * width + u0) * FIELDS;
            let x = if self.motion[r + HIDDEN] > 0.0 { self.motion[r + IDENT] } else { 0.0 };
            x > 0.0
        })?;
        let r = (cells[carried] * width + u0) * FIELDS;
        let record = &self.motion[r..r + FIELDS];
        let (core, tail) = (record[CORE + 7], record[TAIL + 3]);
        let parts: Vec<f64> = cells.iter()
            .flat_map(|&c| units.iter().map(move |&u| (c, u)))
            .map(|(c, u)| np::maximum(self.potential[c * width + u], 0.0))
            .collect();
        let whole = np::sum(&parts);
        if core + tail < 0.5 * whole {
            return None;
        }
        let scale = core + p.ahead * tail;
        Some([record[CORE + 2] * scale / (core + tail), record[CORE + 3] * scale / (core + tail)])
    }

    /// The value units `cells` hold: in each group the unit of most activity, above the floor
    /// (`_values_of`).
    fn values_of(&self, cells: &[usize], activity: &[f64]) -> Vec<usize> {
        let width = self.width;
        let mut out = Vec::new();
        for &(a, b) in &self.groups {
            let summed = column_sums(
                cells.iter().map(|&c| activity[c * width + a..c * width + b].to_vec()), b - a);
            if !summed.is_empty() && largest(summed.iter().copied()) > self.floor {
                out.push(a + np::argmax(&summed));
            }
        }
        out
    }

    /// `things()`: every thing the map holds, read off its carried activity.
    pub fn things(&self) -> Result<Vec<Entry>, Error> {
        let activity = self.activity();
        let count = self.count();
        let mut out = Vec::new();
        for kind in 0..self.kinds {
            let own = self.of_kind(kind, &activity);
            let active: Vec<bool> = own.iter().map(|&x| x > self.floor).collect();
            let mut groups: Vec<(i64, Vec<Vec<usize>>)> = Vec::new();
            for piece in self.pieces(&active)? {
                let heights: Vec<f64> = piece.iter().map(|&c| own[c]).collect();
                let since = self.clock[piece[np::argmax(&heights)]].round_ties_even() as i64;
                match groups.iter_mut().find(|g| g.0 == since) {
                    Some(g) => g.1.push(piece),
                    None => groups.push((since, vec![piece])),
                }
            }
            for (since, pieces) in groups {
                let cells: Vec<usize> = pieces.concat();
                let doubts: Vec<f64> = cells.iter().map(|&c| self.doubt[c]).collect();
                let doubt = np::mean(&doubts) > 0.5;
                let sets: Vec<Vec<Vec<usize>>> =
                    if doubt { vec![pieces.clone()] } else { pieces.iter().map(|p| vec![p.clone()]).collect() };
                for group in sets {
                    let held: Vec<usize> = group.concat();
                    let mut state = if doubt { NEAR } else { HELD };
                    let velocity = self.moving(kind, &held);
                    if velocity.is_some() {
                        state = MOVING;
                    }
                    if since == 0 {
                        state = self.sensed.iter().find(|s| s.0 == kind as i64).map_or(SEEN, |s| s.1);
                    }
                    let weights: Vec<f64> = held.iter().map(|&c| own[c]).collect();
                    let xs: Vec<f64> = held.iter().map(|&c| self.centers[2 * c]).collect();
                    let ys: Vec<f64> = held.iter().map(|&c| self.centers[2 * c + 1]).collect();
                    let total = np::sum(&weights);
                    let found = self.values_of(&held, &activity);
                    let values = self.groups.iter()
                        .map(|&(a, b)| found.iter().find(|&&u| a <= u && u < b).map(|&u| (u - a) as i64))
                        .collect();
                    out.push(Entry {
                        kind, place: Some([np::dot(&weights, &xs) / total, np::dot(&weights, &ys) / total]),
                        pieces: group, since: Some(since), state, velocity, values,
                    });
                }
            }
            let units = self.kind_units(kind);
            if least(units.iter().map(|&u| self.lost[u])) > 0.5 && !active.iter().any(|&x| x) {
                out.push(Entry {
                    kind, place: None, pieces: Vec::new(), since: None, state: NOT_KNOWN_WHERE,
                    velocity: None, values: vec![None; self.groups.len()],
                });
            }
        }
        let _ = count;
        Ok(out)
    }

    // -- the calls by name

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let count = self.count();
        match method {
            "state" => {} // the state as it came: its round trip
            "moment" => {
                let kinds = args.i64s("reports.kind")?;
                let places = args.f64s("reports.place")?;
                let senses = args.i64s("reports.sense")?;
                let values = args.f64s("reports.values")?;
                let named = args.bools("reports.named")?;
                let (r, valued) = (kinds.len(), self.valued());
                if places.len() != 2 * r || senses.len() != r || values.len() != r * valued
                    || named.len() != r * valued || kinds.iter().any(|&k| k < 0)
                {
                    return Err(refused("the reports do not fit the map"));
                }
                let reports: Vec<Report> = (0..r)
                    .map(|i| Report {
                        kind: kinds[i] as usize, place: [places[2 * i], places[2 * i + 1]],
                        sense: senses[i], values: values[i * valued..(i + 1) * valued].to_vec(),
                        named: named[i * valued..(i + 1) * valued].to_vec(),
                    })
                    .collect();
                match args.i64("lit.kind")? {
                    0 => self.moment(&reports, &mut Lit::Nothing),
                    1 => self.moment(&reports, &mut Lit::Flags(args.bools("lit.flags")?)),
                    2 => {
                        let mut camera = Recorded {
                            points: args.f64s("lit.points")?, bounds: args.i64s("lit.bounds")?,
                            seen: args.bools("lit.seen")?, next: 0,
                        };
                        self.moment(&reports, &mut Lit::Camera(&mut camera))?;
                        if camera.next + 1 != camera.bounds.len() {
                            return Err(refused("the camera was asked fewer questions than the \
                                                Python asked"));
                        }
                        Ok(())
                    }
                    _ => Err(refused("lit is nothing, flags or a camera")),
                }?;
            }
            "thing_at" => {
                let two = |name: &str| -> Result<[f64; 2], Error> {
                    args.f64s(name)?.try_into().map_err(|_| refused(name))
                };
                let place = if args.has("place") { Some(two("place")?) } else { None };
                let within = if args.has("within") { Some(args.f64("within")?) } else { None };
                let cells = if args.has("cells") { Some(args.bools("cells")?) } else { None };
                match self.thing_at(place, within, cells, two("body")?, args.f64("heading")?)? {
                    None => out.put("found", Array::from_bool(&[], &[false])),
                    Some(t) => {
                        out.put("found", Array::from_bool(&[], &[true]));
                        out.put("kind", Array::scalar_i64(t.kind as i64));
                        out.put("values", Array::from_f64(&[t.values.len()], &t.values));
                        out.put("place", Array::from_f64(&[2], &t.place));
                        out.put("bearing", Array::scalar_f64(t.bearing));
                        out.put("distance", Array::scalar_f64(t.distance));
                    }
                }
            }
            "things" => {
                let all = self.things()?;
                let n = all.len();
                let groups = if n > 0 { self.groups.len() } else { 0 };
                let cells: Vec<i64> =
                    all.iter().flat_map(|t| t.pieces.concat()).map(|c| c as i64).collect();
                let mut bounds = vec![0i64];
                for t in &all {
                    bounds.push(bounds.last().unwrap() + t.pieces.iter().map(|p| p.len()).sum::<usize>() as i64);
                }
                let sizes: Vec<i64> =
                    all.iter().flat_map(|t| t.pieces.iter().map(|p| p.len() as i64)).collect();
                let column = |f: &dyn Fn(&Entry) -> i64| -> Vec<i64> { all.iter().map(f).collect() };
                out.put("things.kind", Array::from_i64(&[n], &column(&|t| t.kind as i64)));
                out.put("things.state", Array::from_i64(&[n], &column(&|t| t.state)));
                out.put("things.since", Array::from_i64(&[n], &column(&|t| t.since.unwrap_or(-1))));
                let places: Vec<f64> =
                    all.iter().flat_map(|t| t.place.unwrap_or([f64::NAN, f64::NAN])).collect();
                out.put("things.place", Array::from_f64(&[n, 2], &places));
                out.put("things.cells", Array::from_i64(&[cells.len()], &cells));
                out.put("things.cell_bounds", Array::from_i64(&[n + 1], &bounds));
                out.put("things.pieces", Array::from_i64(&[n], &column(&|t| t.pieces.len() as i64)));
                out.put("things.piece_sizes", Array::from_i64(&[sizes.len()], &sizes));
                let moving: Vec<bool> = all.iter().map(|t| t.velocity.is_some()).collect();
                out.put("things.moving", Array::from_bool(&[n], &moving));
                let velocity: Vec<f64> =
                    all.iter().flat_map(|t| t.velocity.unwrap_or([0.0, 0.0])).collect();
                out.put("things.velocity", Array::from_f64(&[n, 2], &velocity));
                let values: Vec<i64> =
                    all.iter().flat_map(|t| t.values.iter().map(|v| v.unwrap_or(-1))).collect();
                out.put("things.values", Array::from_i64(&[n, groups], &values));
            }
            "sheet" => {
                let kind = if args.has("kind") {
                    let k = args.i64("kind")?;
                    Some(usize::try_from(k).map_err(|_| refused("no such kind on this map"))?)
                } else {
                    None
                };
                let sheet = self.sheet(kind)?;
                out.put("sheet", Array::from_f64(&[self.nx, self.ny], &sheet));
            }
            "forecast" => {
                let ahead = self.forecast(args.i64("moments")?)?;
                let width = self.width;
                out.put("potential", Array::from_f64(&[count, width], &ahead.potential));
                out.put("clock", Array::from_f64(&[count], &ahead.clock));
                out.put("lost", Array::from_f64(&[width], &ahead.lost));
                out.put("doubt", Array::from_f64(&[count], &ahead.doubt));
                out.put("reading", Array::from_f64(&[count + width], &ahead.reading));
                out.put("motion", Array::from_f64(&[count, width, ph::FIELDS], &ahead.motion));
            }
            "moves" => {
                let moves = self.moves(&args.f64s("expected")?)?;
                out.put("moves", Array::from_f64(&[count], &moves));
            }
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

/// The camera as the Python asked it: each call's places and answers, in order. A question the
/// Python did not ask (a place apart by more than rounding, or a call more) is refused.
struct Recorded {
    points: Vec<f64>,
    bounds: Vec<i64>,
    seen: Vec<bool>,
    next: usize,
}

impl Camera for Recorded {
    fn sees(&mut self, points: &[f64]) -> Result<Vec<bool>, Error> {
        let k = self.next;
        if k + 1 >= self.bounds.len() {
            return Err(refused("the camera was asked more questions than the Python asked"));
        }
        let (lo, hi) = (self.bounds[k] as usize, self.bounds[k + 1] as usize);
        if hi < lo || hi > self.seen.len() || 2 * hi > self.points.len() {
            return Err(refused("lit.bounds"));
        }
        let asked = &self.points[2 * lo..2 * hi];
        let same = asked.len() == points.len()
            && asked.iter().zip(points).all(|(a, b)| (a - b).abs() <= 1e-9 * a.abs().max(1.0));
        if !same {
            return Err(refused("the camera was asked another question than the Python asked"));
        }
        self.next += 1;
        Ok(self.seen[lo..hi].to_vec())
    }
}
