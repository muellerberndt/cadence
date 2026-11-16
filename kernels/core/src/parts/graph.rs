//! `cadence.brain.Brain`, the neural graph: its parameters (efficacy, log gain, bias), its
//! connectome's synapses, its layout of blocks, its neuron rule and its apical compartment, and
//! what a moment does with them: `settle_batch`, `equilibrate` (with `_equilibrate_local`,
//! `_solve_compiled` and the damping ladder), `residual` and `apical_gain`.
//!
//! The settles, the checks and the free solves are the compiled kernels of `cadence.fused`
//! (`crate::settle`, `crate::fast`) on the graph's own arrays, as the Python's compiled CPU path
//! calls them; the wrappers here are `fused.py`'s (`fused_settle`, `fused_solve`,
//! `fused_fast_solve`, `fused_residual`, `_nudge_arrays`, `_compartment`, `_fed_ranges`) in its
//! order. The per-neuron residual and the apical gain are the NumPy path of `brain.py`
//! (`BlockTransport`, `Apical.gain`, `Nudge.drive`), whose block products sum in order where
//! NumPy natively sums them in its BLAS's order.

use crate::fast;
use crate::np;
use crate::settle::{self, Net, RECORD};
use crate::state::{Array, Error, Named};
use crate::View;

/// `brain._APICAL_TINY`: under the square root, so a pool with no input at all receives none.
const APICAL_TINY: f64 = 1e-24;
/// `fused._FAST_SETTLED`: a fast solve's row stops within this.
const FAST_SETTLED: f64 = 1e-5;

/// What the Python refuses with a `ValueError`.
fn refused(text: &str) -> Error {
    Error::Refused(text.to_string())
}

/// A kernel that would not take the arrays handed to it: a fault, never the Python's refusal.
fn fault(text: &str) -> Error {
    Error::Shape(text.to_string())
}

/// `np.ldexp(x, -halvings)`: `x` halved `halvings` times, exactly.
fn halved(x: f64, halvings: i64) -> f64 {
    let mut out = x;
    for _ in 0..halvings {
        out *= 0.5;
    }
    out
}

/// A view of a run the kernels only read.
fn seen<T>(x: &[T], n0: usize, n1: usize) -> View<T> {
    View { p: x.as_ptr() as *mut T, n0, n1, len: n0 * n1 }
}

/// A view of a run the kernels write.
fn held<T>(x: &mut [T], n0: usize, n1: usize) -> View<T> {
    View { p: x.as_mut_ptr(), n0, n1, len: n0 * n1 }
}

fn column<T>(x: &[T]) -> View<T> {
    seen(x, x.len(), 1)
}


fn no_nan(x: &[f64]) -> bool {
    x.iter().all(|v| !v.is_nan())
}

fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// `NeuronModel`: the rule's constants, its rest emission as the Python made it.
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    pub dt: f64,
    pub slope: f64,
    pub threshold: f64,
    pub gain: f64,
    pub stimulus_amplitude: f64,
    pub leak: f64,
    pub rest: f64,
    /// `(tau_steps, strength)` of the adaptation, where the rule has one.
    pub adaptation: Option<(f64, f64)>,
}

impl Neuron {
    /// `NeuronModel.activation` of one potential.
    #[inline]
    pub fn act(&self, x: f64) -> f64 {
        settle::act_scalar(x, self.slope, self.threshold, self.rest, self.leak)
    }
    fn strength(&self) -> f64 {
        self.adaptation.map_or(0.0, |a| a.1)
    }
    fn tau(&self) -> f64 {
        self.adaptation.map_or(1.0, |a| a.0)
    }
    /// `_ladder_rule`: the rule at `dt` passes `NeuronModel`'s checks and `_validate_precision`
    /// in float64.
    fn check_at(&self, dt: f64) -> Result<(), Error> {
        let rest = self.rest;
        let mut values = vec![dt, self.slope, self.threshold, rest, 1.0 / (1.0 - rest),
                              self.leak / rest];
        if let Some((tau, strength)) = self.adaptation {
            values.extend([tau, strength, 1.0 / tau]);
        }
        let representable = dt > 0.0 && dt <= 1.0 && all_finite(&values) && self.slope > 0.0
            && rest > 0.0 && rest < 1.0;
        if !representable {
            return Err(refused("neuron rule is not representable at runtime precision"));
        }
        Ok(())
    }
}

/// `Apical`: the synapses that land on the apical compartment and how each pool normalizes
/// them.
#[derive(Clone, Debug, PartialEq)]
pub struct Apical {
    pub synapses: Vec<bool>,
    pub groups: Vec<i64>,
    pub fraction: f64,
    pub floor: f64,
}

/// What the parameters, the layout and the compartment determine, made once per set of them.
#[derive(Clone, Debug, PartialEq, Default)]
struct Tables {
    weights: Vec<f64>,
    /// The flat blocks the kernels read: a graph with a compartment's basal synapses alone.
    flat: Vec<f64>,
    /// The apical synapses' flat blocks; `flat` itself without a compartment.
    flat_apical: Option<Vec<f64>>,
    /// The blocks summed into the basal input (every block without a compartment) and those
    /// summed into the apical input, in block order.
    pairs: Vec<i64>,
    apical_pairs: Vec<i64>,
    pool_of: Vec<i64>,
    pool_index: Vec<i64>,
    pool_starts: Vec<i64>,
    pool_run: Vec<i64>,
    members: Vec<Vec<usize>>,
    /// `~_fed_ranges`: the ranges no block lands on.
    freezable: Vec<u8>,
}

/// A settled state (`BrainState`): `batch` rows of `n` neurons.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub batch: usize,
    pub v: Vec<f64>,
    pub activation: Vec<f64>,
    pub adaptation: Vec<f64>,
    pub steps: i64,
    pub activity_change: Option<Vec<f64>>,
    /// Whether `activation` is what a settle from the state publishes (`fused._publish`
    /// reading the record the state keeps).
    pub published: bool,
}

impl State {
    pub fn import(s: &Named, prefix: &str, n: usize) -> Result<State, Error> {
        let name = |k: &str| format!("{prefix}.{k}");
        let shape = s.shape(&name("v"))?;
        let batch = if shape.len() == 2 { shape[0] } else { 1 };
        let v = s.f64s(&name("v"))?;
        let activation = s.f64s(&name("activation"))?;
        let adaptation = s.f64s(&name("adaptation"))?;
        if shape.len() > 2 || v.len() != batch * n || activation.len() != v.len()
            || adaptation.len() != v.len()
        {
            return Err(refused("state batch does not match the drive batch"));
        }
        let change = if s.has(&name("activity_change")) {
            Some(s.f64s(&name("activity_change"))?)
        } else {
            None
        };
        let published = s.has(&name("published")) && s.bools(&name("published"))?[0];
        Ok(State { batch, v, activation, adaptation, steps: s.i64(&name("steps"))?,
                   activity_change: change, published })
    }

    pub fn export(&self, prefix: &str, n: usize, out: &mut Named) {
        let name = |k: &str| format!("{prefix}.{k}");
        let shape = [self.batch, n];
        out.put(&name("v"), Array::from_f64(&shape, &self.v));
        out.put(&name("activation"), Array::from_f64(&shape, &self.activation));
        out.put(&name("adaptation"), Array::from_f64(&shape, &self.adaptation));
        out.put(&name("steps"), Array::scalar_i64(self.steps));
        if let Some(change) = &self.activity_change {
            out.put(&name("activity_change"), Array::from_f64(&[change.len()], change));
        }
        out.put(&name("published"), Array::from_bool(&[], &[self.published]));
    }
}

/// `Equilibrium`: a bounded solve and its measured equation error per row.
#[derive(Clone, Debug, PartialEq)]
pub struct Equilibrium {
    pub state: State,
    pub residual: Vec<f64>,
    pub steps: i64,
    pub tolerance: f64,
    pub residual_checks: i64,
    pub damping_halvings: i64,
    pub stagnation_checks: i64,
    pub row_sweeps: Option<i64>,
}

impl Equilibrium {
    pub fn qualified(&self) -> bool {
        self.residual.iter().all(|&e| e <= self.tolerance)
    }
    pub fn export(&self, n: usize, out: &mut Named) {
        self.state.export("state", n, out);
        out.put("residual", Array::from_f64(&[self.residual.len()], &self.residual));
        out.put("steps", Array::scalar_i64(self.steps));
        out.put("tolerance", Array::scalar_f64(self.tolerance));
        out.put("residual_checks", Array::scalar_i64(self.residual_checks));
        out.put("damping_halvings", Array::scalar_i64(self.damping_halvings));
        out.put("stagnation_checks", Array::scalar_i64(self.stagnation_checks));
        if let Some(r) = self.row_sweeps {
            out.put("row_sweeps", Array::scalar_i64(r));
        }
    }
}

/// Rows of numbers over the neurons: a drive, a mask, a target. `rows` is 1 for a single row
/// (`ndim` 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub ndim: usize,
}

impl Rows {
    pub fn of(s: &Named, name: &str) -> Result<Rows, Error> {
        let shape = s.shape(name)?;
        let data = s.f64s(name)?;
        let (rows, cols) = match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => return Err(refused(&format!("{name} has one or two dimensions"))),
        };
        Ok(Rows { data, rows, cols, ndim: shape.len() })
    }
    pub fn row(data: Vec<f64>) -> Rows {
        let cols = data.len();
        Rows { data, rows: 1, cols, ndim: 1 }
    }
    /// The rows as `(batch, cols)`, one row broadcast.
    fn broadcast(&self, batch: usize) -> Vec<f64> {
        if self.rows == batch {
            return self.data.clone();
        }
        let mut out = Vec::with_capacity(batch * self.cols);
        for _ in 0..batch {
            out.extend_from_slice(&self.data[..self.cols]);
        }
        out
    }
    /// Whether the shape is `(n,)`, `(1, n)` or `(batch, n)`.
    fn fits(&self, batch: usize, n: usize) -> bool {
        self.cols == n && (self.ndim == 1 || (self.ndim == 2 && (self.rows == 1 || self.rows == batch)))
    }
}

/// A nudge's softmax temperature: one for every group, or one per neuron.
#[derive(Clone, Debug, PartialEq)]
pub enum Temperature {
    One(f64),
    Each(Vec<f64>),
}

/// `Nudge`: extra drive toward a target on the masked neurons, an anchor, a held apical gain.
#[derive(Clone, Debug, PartialEq)]
pub struct Nudge {
    pub target: Rows,
    pub mask: Vec<f64>,
    pub beta: f64,
    pub temperature: Option<Temperature>,
    pub weight: Option<Vec<f64>>,
    pub groups: Option<Vec<i64>>,
    pub anchor: Option<Rows>,
    pub anchor_gain: Option<Vec<f64>>,
    /// `(batch, pools)` row-major, with its rows and columns.
    pub apical_gain: Option<Rows>,
}

impl Nudge {
    pub fn import(s: &Named, prefix: &str) -> Result<Option<Nudge>, Error> {
        let name = |k: &str| format!("{prefix}.{k}");
        if !s.has(&name("target")) {
            return Ok(None);
        }
        let temperature = if s.has(&name("softmax_temperature")) {
            // one temperature crosses as a number (or a run of one, as `pack` writes it);
            // a run of one per neuron is the same nudge
            let t = s.f64s(&name("softmax_temperature"))?;
            Some(if t.len() == 1 { Temperature::One(t[0]) } else { Temperature::Each(t) })
        } else {
            None
        };
        let opt = |k: &str| -> Result<Option<Vec<f64>>, Error> {
            if s.has(&name(k)) { Ok(Some(s.f64s(&name(k))?)) } else { Ok(None) }
        };
        let rows = |k: &str| -> Result<Option<Rows>, Error> {
            if s.has(&name(k)) { Ok(Some(Rows::of(s, &name(k))?)) } else { Ok(None) }
        };
        let nudge = Nudge {
            target: Rows::of(s, &name("target"))?,
            mask: s.f64s(&name("mask"))?,
            beta: s.f64(&name("beta"))?,
            temperature,
            weight: opt("weight")?,
            groups: if s.has(&name("groups")) { Some(s.i64s(&name("groups"))?) } else { None },
            anchor: rows("anchor")?,
            anchor_gain: opt("anchor_gain")?,
            apical_gain: rows("apical_gain")?,
        };
        nudge.check()?;
        Ok(Some(nudge))
    }

    /// `Nudge.__post_init__`'s refusals.
    fn check(&self) -> Result<(), Error> {
        let n = self.mask.len();
        if self.target.cols != n || self.target.ndim == 0 {
            return Err(refused("nudge target must have one entry per neuron"));
        }
        if !self.beta.is_finite() || !all_finite(&self.target.data) || !all_finite(&self.mask)
            || self.mask.iter().any(|&m| m < 0.0)
        {
            return Err(refused("nudge target and mask must be finite; mask nonnegative"));
        }
        if let Some(t) = &self.temperature {
            let ok = match t {
                Temperature::One(x) => x.is_finite() && *x > 0.0,
                Temperature::Each(x) => x.len() == n && x.iter().all(|v| v.is_finite() && *v > 0.0),
            };
            if !ok {
                return Err(refused("softmax_temperature must be finite and positive"));
            }
        }
        if let Some(g) = &self.groups {
            if g.len() != n || g.iter().any(|&x| x < -1) {
                return Err(refused("nudge groups must be integer IDs, -1 for ungrouped"));
            }
        }
        if let Some(Temperature::Each(t)) = &self.temperature {
            for members in self.softmax_groups() {
                if members.iter().any(|&i| t[i] != t[members[0]]) {
                    return Err(refused("softmax_temperature must be constant within each group"));
                }
            }
        }
        if let Some(w) = &self.weight {
            if !all_finite(w) {
                return Err(refused("nudge weight must be finite"));
            }
        }
        if self.anchor.is_some() != self.anchor_gain.is_some() {
            return Err(refused("anchor and anchor_gain must be supplied together"));
        }
        if let (Some(anchor), Some(gain)) = (&self.anchor, &self.anchor_gain) {
            if anchor.cols != n || anchor.ndim == 0 || gain.len() != n || !all_finite(gain)
                || gain.iter().any(|&g| g < 0.0) || !all_finite(&anchor.data)
            {
                return Err(refused("anchor must be finite, anchor_gain finite and nonnegative"));
            }
        }
        if let Some(h) = &self.apical_gain {
            if h.ndim != 2 || !all_finite(&h.data) || h.data.iter().any(|&g| g < 0.0) {
                return Err(refused("apical_gain must be finite and nonnegative, one per row and pool"));
            }
        }
        Ok(())
    }

    /// `_softmax_groups`: the masked neurons as one index run per softmax group.
    fn softmax_groups(&self) -> Vec<Vec<usize>> {
        let masked: Vec<usize> = (0..self.mask.len()).filter(|&i| self.mask[i] > 0.0).collect();
        match &self.groups {
            None => {
                if masked.is_empty() { vec![] } else { vec![masked] }
            }
            Some(groups) => {
                let mut ids: Vec<i64> = masked.iter().map(|&i| groups[i]).filter(|&g| g >= 0).collect();
                ids.sort_unstable();
                ids.dedup();
                ids.iter().map(|&g| masked.iter().copied().filter(|&i| groups[i] == g).collect())
                    .collect()
            }
        }
    }

    fn group_temperature(&self, members: &[usize]) -> f64 {
        match &self.temperature {
            Some(Temperature::One(t)) => *t,
            Some(Temperature::Each(t)) => t[members[0]],
            None => 1.0,
        }
    }

    /// `Nudge.drive(s)` of `batch` rows, as NumPy computes it.
    fn drive(&self, s: &[f64], batch: usize) -> Result<Vec<f64>, Error> {
        let n = self.mask.len();
        let target = self.target.broadcast(batch);
        let mut out = vec![0.0f64; batch * n];
        match &self.temperature {
            None => {
                for k in 0..batch * n {
                    out[k] = self.beta * (target[k] - s[k]) * self.mask[k % n];
                }
            }
            Some(_) => {
                for group in self.softmax_groups() {
                    let t = self.group_temperature(&group);
                    let m = group.len();
                    let mut z = vec![0.0f64; batch * m];
                    for b in 0..batch {
                        for (j, &i) in group.iter().enumerate() {
                            z[b * m + j] = s[b * n + i] / t;
                        }
                    }
                    if !all_finite(&z) {
                        for b in 0..batch {
                            let logits: Vec<f64> = group.iter().map(|&i| s[b * n + i]).collect();
                            let top = np::max(&logits);
                            for (j, &i) in group.iter().enumerate() {
                                z[b * m + j] = (s[b * n + i] - top) / t;
                            }
                        }
                    } else {
                        for b in 0..batch {
                            let top = np::max(&z[b * m..(b + 1) * m]);
                            for x in &mut z[b * m..(b + 1) * m] {
                                *x -= top;
                            }
                        }
                    }
                    for x in z.iter_mut() {
                        *x = np::exp(*x);
                    }
                    for b in 0..batch {
                        let total = np::sum(&z[b * m..(b + 1) * m]);
                        for (j, &i) in group.iter().enumerate() {
                            let p = z[b * m + j] / total;
                            out[b * n + i] = self.beta * (target[b * n + i] - p) * self.mask[i];
                        }
                    }
                }
            }
        }
        if let Some(w) = &self.weight {
            if w.len() != batch {
                return Err(refused("nudge weight must have one entry per batch row"));
            }
            for b in 0..batch {
                for i in 0..n {
                    out[b * n + i] *= w[b];
                }
            }
        }
        if let (Some(anchor), Some(gain)) = (&self.anchor, &self.anchor_gain) {
            let anchor = anchor.broadcast(batch);
            for k in 0..batch * n {
                out[k] += gain[k % n] * (anchor[k] - s[k]);
            }
        }
        Ok(out)
    }
}

/// The nudge as the kernels read it (`fused._nudge_arrays`), or the arrays of no nudge
/// (`fused._no_nudge`).
struct NudgeArrays {
    target: Vec<f64>,
    target_shape: (usize, usize),
    nmask: Vec<f64>,
    gid: Vec<i64>,
    ngroups: usize,
    beta: f64,
    softmax_t: Vec<f64>,
    weight: Vec<f64>,
    anchor: Vec<f64>,
    anchor_shape: (usize, usize),
    anchor_gain: Vec<f64>,
}

impl NudgeArrays {
    fn of(nudge: Option<&Nudge>, batch: usize, n: usize) -> NudgeArrays {
        let Some(nudge) = nudge else {
            return NudgeArrays {
                target: vec![0.0], target_shape: (1, 1), nmask: vec![0.0; n], gid: vec![-1; n],
                ngroups: 0, beta: 0.0, softmax_t: vec![0.0; n], weight: vec![1.0; batch],
                anchor: vec![0.0], anchor_shape: (1, 1), anchor_gain: vec![0.0; n],
            };
        };
        let target = nudge.target.broadcast(batch);
        let mut nmask = nudge.mask.clone();
        let softmax_t = match &nudge.temperature {
            None => vec![0.0; n],
            Some(Temperature::One(t)) => vec![*t; n],
            Some(Temperature::Each(t)) => t.clone(),
        };
        let weight = nudge.weight.clone().unwrap_or_else(|| vec![1.0; batch]);
        let (gid, ngroups) = match &nudge.groups {
            None => (nmask.iter().map(|&m| if m > 0.0 { 0 } else { -1 }).collect(), 1),
            Some(raw) => {
                let mut ids: Vec<i64> = raw.iter().copied().filter(|&g| g >= 0).collect();
                ids.sort_unstable();
                ids.dedup();
                let mut gid = vec![-1i64; n];
                for (k, &g) in ids.iter().enumerate() {
                    for i in 0..n {
                        if raw[i] == g {
                            gid[i] = k as i64;
                        }
                    }
                }
                (gid, ids.len())
            }
        };
        if softmax_t[0] > 0.0 {
            for i in 0..n {
                if gid[i] < 0 {
                    nmask[i] = 0.0;
                }
            }
        }
        let (anchor, anchor_shape, anchor_gain) = match (&nudge.anchor, &nudge.anchor_gain) {
            (Some(a), Some(g)) => (a.broadcast(batch), (batch, n), g.clone()),
            _ => (vec![0.0], (1, 1), vec![0.0; n]),
        };
        NudgeArrays { target, target_shape: (batch, n), nmask, gid, ngroups, beta: nudge.beta,
                      softmax_t, weight, anchor, anchor_shape, anchor_gain }
    }
}

/// What a launch reads besides the state: the mask, the nudge, the held apical gain.
struct Launch {
    keep: Vec<f64>,
    keep_shape: (usize, usize),
    masked: bool,
    nudge: NudgeArrays,
    has_nudge: bool,
    held: Vec<f64>,
    held_shape: (usize, usize),
    has_held: bool,
}

/// A call's drive, state, mask and nudge.
type Arguments = (Rows, Option<State>, Option<Rows>, Option<Nudge>);

/// The neural graph (`Brain` on the CPU backend, its blocks fitting `dense_limit`).
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub efficacy: Vec<f64>,
    pub log_gain: Vec<f64>,
    pub bias: Vec<f64>,
    /// `gain * count * exp(log_gain)[pre]`, as the Python made it.
    pub gain_pre: Vec<f64>,
    pub pre: Vec<i64>,
    pub post: Vec<i64>,
    pub dense_limit: i64,
    pub starts: Vec<i64>,
    pub pair_pre: Vec<i64>,
    pub pair_post: Vec<i64>,
    pub offset: Vec<i64>,
    pub edge_index: Vec<i64>,
    /// `fused.sparse_offset(layout, apical)`: the offsets, then the blocks read by their
    /// synapses.
    pub sparse: Vec<i64>,
    /// Whether two synapses share an entry of the blocks (`Layout.parallel_synapses`): the
    /// layout's, so read once, when the graph is made.
    parallel: bool,
    pub neuron: Neuron,
    pub apical: Option<Apical>,
    tables: Tables,
    /// Which of a life's graphs this is: an update that replaces the graph counts one on
    /// (`Learner::apply`), the Python's identity of a `Brain`; not part of the state.
    pub generation: u64,
}

impl Graph {
    pub fn n(&self) -> usize {
        self.bias.len()
    }
    pub fn synapses(&self) -> usize {
        self.efficacy.len()
    }
    fn ranges(&self) -> usize {
        self.starts.len() - 1
    }
    fn size(&self) -> usize {
        *self.offset.last().unwrap_or(&0) as usize
    }
    fn start(&self, r: usize) -> usize {
        self.starts[r] as usize
    }
    /// `Layout.bounds(k)`.
    fn bounds(&self, k: usize) -> (usize, usize, usize, usize) {
        let (a, b) = (self.pair_pre[k] as usize, self.pair_post[k] as usize);
        (self.start(a), self.start(a + 1), self.start(b), self.start(b + 1))
    }
    pub fn pools(&self) -> usize {
        self.tables.members.len()
    }
    pub fn weights(&self) -> &[f64] {
        &self.tables.weights
    }
    /// `layout.parallel_synapses`: whether two synapses share an entry of a block.
    pub fn parallel_synapses(&self) -> bool {
        self.parallel
    }

    pub fn import(s: &Named) -> Result<Graph, Error> {
        let adaptation = if s.has("neuron.adaptation") {
            let a = s.f64s("neuron.adaptation")?;
            if a.len() != 2 {
                return Err(refused("adaptation holds tau_steps and strength"));
            }
            Some((a[0], a[1]))
        } else {
            None
        };
        let apical = if s.has("apical.synapses") {
            Some(Apical {
                synapses: s.bools("apical.synapses")?,
                groups: s.i64s("apical.groups")?,
                fraction: s.f64("apical.fraction")?,
                floor: s.f64("apical.floor")?,
            })
        } else {
            None
        };
        let mut graph = Graph {
            efficacy: s.f64s("efficacy")?,
            log_gain: s.f64s("log_gain")?,
            bias: s.f64s("bias")?,
            gain_pre: s.f64s("gain_pre")?,
            pre: s.i64s("pre")?,
            post: s.i64s("post")?,
            dense_limit: s.i64("dense_limit")?,
            starts: s.i64s("layout.starts")?,
            pair_pre: s.i64s("layout.pair_pre")?,
            pair_post: s.i64s("layout.pair_post")?,
            offset: s.i64s("layout.offset")?,
            edge_index: s.i64s("layout.edge_index")?,
            sparse: s.i64s("layout.sparse")?,
            parallel: false,
            neuron: Neuron {
                dt: s.f64("neuron.dt")?,
                slope: s.f64("neuron.slope")?,
                threshold: s.f64("neuron.threshold")?,
                gain: s.f64("neuron.gain")?,
                stimulus_amplitude: s.f64("neuron.stimulus_amplitude")?,
                leak: s.f64("neuron.leak")?,
                rest: s.f64("neuron.rest")?,
                adaptation,
            },
            apical,
            tables: Tables::default(),
            generation: 0,
        };
        graph.validate()?;
        let mut sorted = graph.edge_index.clone();
        sorted.sort_unstable();
        graph.parallel = sorted.windows(2).any(|w| w[0] == w[1]);
        graph.make_tables()?;
        Ok(graph)
    }

    pub fn export(&self, out: &mut Named) {
        let f = |x: &[f64]| Array::from_f64(&[x.len()], x);
        let i = |x: &[i64]| Array::from_i64(&[x.len()], x);
        out.put("efficacy", f(&self.efficacy));
        out.put("log_gain", f(&self.log_gain));
        out.put("bias", f(&self.bias));
        out.put("gain_pre", f(&self.gain_pre));
        out.put("pre", i(&self.pre));
        out.put("post", i(&self.post));
        out.put("dense_limit", Array::scalar_i64(self.dense_limit));
        out.put("layout.starts", i(&self.starts));
        out.put("layout.pair_pre", i(&self.pair_pre));
        out.put("layout.pair_post", i(&self.pair_post));
        out.put("layout.offset", i(&self.offset));
        out.put("layout.edge_index", i(&self.edge_index));
        out.put("layout.sparse", i(&self.sparse));
        let m = &self.neuron;
        for (name, value) in [
            ("neuron.dt", m.dt), ("neuron.slope", m.slope), ("neuron.threshold", m.threshold),
            ("neuron.gain", m.gain), ("neuron.stimulus_amplitude", m.stimulus_amplitude),
            ("neuron.leak", m.leak), ("neuron.rest", m.rest),
        ] {
            out.put(name, Array::scalar_f64(value));
        }
        if let Some((tau, strength)) = m.adaptation {
            out.put("neuron.adaptation", Array::from_f64(&[2], &[tau, strength]));
        }
        if let Some(ap) = &self.apical {
            out.put("apical.synapses", Array::from_bool(&[ap.synapses.len()], &ap.synapses));
            out.put("apical.groups", i(&ap.groups));
            out.put("apical.fraction", Array::scalar_f64(ap.fraction));
            out.put("apical.floor", Array::scalar_f64(ap.floor));
        }
    }

    /// The refusals of `Brain`, `Layout`, `NeuronModel` and `Apical` the core relies on.
    fn validate(&self) -> Result<(), Error> {
        let (n, e) = (self.n(), self.synapses());
        if self.log_gain.len() != n || self.gain_pre.len() != e || self.pre.len() != e
            || self.post.len() != e || self.edge_index.len() != e
        {
            return Err(refused("the graph's arrays do not fit its neurons and synapses"));
        }
        if n == 0 || self.pre.iter().chain(&self.post).any(|&x| x < 0 || x as usize >= n) {
            return Err(refused("the graph's synapses join its neurons"));
        }
        let ranges = self.starts.len().saturating_sub(1);
        if ranges == 0 || self.starts[0] != 0 || self.starts[ranges] as usize != n
            || self.starts.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(refused("the layout's ranges cover the neurons"));
        }
        let pairs = self.pair_pre.len();
        if self.pair_post.len() != pairs || self.offset.len() != pairs + 1 || self.offset[0] != 0
            || self.pair_pre.iter().chain(&self.pair_post).any(|&r| r < 0 || r as usize >= ranges)
        {
            return Err(refused("the layout's blocks join its ranges"));
        }
        for k in 0..pairs {
            let (a0, a1, b0, b1) = self.bounds(k);
            if self.offset[k + 1] - self.offset[k] != ((a1 - a0) * (b1 - b0)) as i64 {
                return Err(refused("the layout's offsets are its blocks' sizes"));
            }
        }
        let size = self.size() as i64;
        if self.edge_index.iter().any(|&x| x < 0 || x >= size) {
            return Err(refused("every synapse has its entry in a block"));
        }
        if self.sparse.len() < pairs + 1 || self.sparse[..pairs + 1] != self.offset[..] {
            return Err(refused("the sparse table begins with the layout's offsets"));
        }
        if self.dense_limit < 0 || self.size() as i128 > (self.dense_limit as i128).pow(2) {
            return Err(refused("the core holds a graph whose blocks fit dense_limit"));
        }
        let m = &self.neuron;
        let valid = [m.dt, m.slope, m.threshold, m.gain, m.stimulus_amplitude, m.leak].iter()
            .all(|x| x.is_finite())
            && m.dt > 0.0 && m.dt <= 1.0 && m.slope > 0.0 && m.gain > 0.0
            && (0.0..=1.0).contains(&m.leak) && m.rest > 0.0 && m.rest < 1.0;
        if !valid {
            return Err(refused("the neuron rule's parameters"));
        }
        if let Some((tau, strength)) = m.adaptation {
            if !(tau.is_finite() && strength.is_finite() && tau > 0.0 && strength >= 0.0) {
                return Err(refused("adaptation parameters must be finite"));
            }
        }
        if !all_finite(&self.efficacy) || !all_finite(&self.log_gain) || !all_finite(&self.bias) {
            return Err(refused("brain parameters must be finite"));
        }
        if let Some(ap) = &self.apical {
            if ap.synapses.len() != e || ap.groups.len() != n || ap.groups.iter().any(|&g| g < -1)
            {
                return Err(refused("apical synapses and groups must match the connectome"));
            }
            if !(ap.fraction.is_finite() && ap.fraction > 0.0 && ap.floor.is_finite()
                 && ap.floor > 0.0)
            {
                return Err(refused("apical fraction and floor must be finite and positive"));
            }
            if (0..e).any(|k| ap.synapses[k] && ap.groups[self.post[k] as usize] < 0) {
                return Err(refused("every apical synapse must land on a neuron of a pool"));
            }
        }
        Ok(())
    }

    /// The weights' entries of the flat blocks (`Layout.flat` of the basal weights, and of the
    /// apical ones into `flat_apical`), written in place: an entry no synapse of a compartment
    /// holds stays the zero its blocks were made with, so each synapse writes its own
    /// compartment's entry alone; parallel synapses sum at their entry from zero.
    fn fill_flats(&mut self) {
        let t = &mut self.tables;
        let up = self.apical.as_ref().map(|a| a.synapses.as_slice());
        let index = &self.edge_index;
        if !self.parallel {
            match (up, t.flat_apical.as_mut()) {
                (Some(up), Some(top)) => {
                    for ((&at, &w), &apical) in index.iter().zip(&t.weights).zip(up) {
                        if apical {
                            top[at as usize] = w;
                        } else {
                            t.flat[at as usize] = w;
                        }
                    }
                }
                _ => {
                    for (&at, &w) in index.iter().zip(&t.weights) {
                        t.flat[at as usize] = w;
                    }
                }
            }
            return;
        }
        // parallel synapses: every entry a synapse holds from zero, each synapse added in order
        for &at in index {
            t.flat[at as usize] = 0.0;
            if let Some(f) = t.flat_apical.as_mut() {
                f[at as usize] = 0.0;
            }
        }
        for (k, &at) in index.iter().enumerate() {
            let at = at as usize;
            let w = t.weights[k];
            let (basal, apical) = match up {
                Some(up) if up[k] => (0.0, w),
                _ => (w, 0.0),
            };
            t.flat[at] += basal;
            if let Some(f) = t.flat_apical.as_mut() {
                f[at] += apical;
            }
        }
    }

    /// The weights, the flat blocks, the compartment's blocks and pools: what the parameters,
    /// the layout and the compartment determine.
    fn make_tables(&mut self) -> Result<(), Error> {
        let weights: Vec<f64> =
            self.gain_pre.iter().zip(&self.efficacy).map(|(g, e)| g * e).collect();
        if !all_finite(&weights) {
            return Err(refused("effective synaptic weights must be finite; reduce gain or efficacy"));
        }
        self.make_structure();
        self.tables.weights = weights;
        self.fill_flats();
        Ok(())
    }

    /// What the layout and the compartment determine, made once: the blocks each input sums,
    /// the pools, the ranges no block lands on, and the flat blocks' room.
    fn make_structure(&mut self) {
        let n = self.n();
        let pairs = self.pair_pre.len();
        let mut t = Tables::default();
        let mut fed = vec![false; self.ranges()];
        for &r in &self.pair_post {
            fed[r as usize] = true;
        }
        t.freezable = fed.iter().map(|&f| (!f) as u8).collect();
        t.flat = vec![0.0f64; self.size()];
        match &self.apical {
            None => {
                t.flat_apical = None;
                t.pairs = (0..pairs as i64).collect();
                t.pool_of = vec![-1; n];
                t.pool_starts = vec![0];
            }
            Some(ap) => {
                t.flat_apical = Some(vec![0.0f64; self.size()]);
                // `_apical_blocks`: the blocks holding a basal synapse, and an apical one
                let mut basal_block = vec![false; pairs];
                let mut apical_block = vec![false; pairs];
                for (k, &at) in self.edge_index.iter().enumerate() {
                    let block = self.offset.partition_point(|&o| o <= at) - 1;
                    if ap.synapses[k] {
                        apical_block[block] = true;
                    } else {
                        basal_block[block] = true;
                    }
                }
                t.pairs = (0..pairs).filter(|&k| basal_block[k]).map(|k| k as i64).collect();
                t.apical_pairs = (0..pairs).filter(|&k| apical_block[k]).map(|k| k as i64).collect();
                // the pools in ascending order of their ids, each its neurons in order
                let mut ids: Vec<i64> = ap.groups.iter().copied().filter(|&g| g >= 0).collect();
                ids.sort_unstable();
                ids.dedup();
                t.pool_of = vec![-1; n];
                t.pool_starts = vec![0];
                for (k, &g) in ids.iter().enumerate() {
                    let members: Vec<usize> = (0..n).filter(|&i| ap.groups[i] == g).collect();
                    for &i in &members {
                        t.pool_of[i] = k as i64;
                        t.pool_index.push(i as i64);
                    }
                    t.pool_starts.push(t.pool_index.len() as i64);
                    let run = members[members.len() - 1] - members[0] + 1 == members.len();
                    t.pool_run.push(if run { members[0] as i64 } else { -1 });
                    t.members.push(members);
                }
            }
        }
        self.tables = t;
    }

    /// `with_parameters(efficacy=, bias=)` on the CPU backend, in place (a learner's update):
    /// the new efficacies and biases, the weights they make with the gains and the flat blocks
    /// written again; the graph counts one generation on. `efficacy` and `bias` give their
    /// values and take the graph's old ones back. A refusal changes nothing.
    pub fn set_parameters(&mut self, efficacy: Option<&mut Vec<f64>>,
                          bias: Option<&mut Vec<f64>>) -> Result<(), Error> {
        if let Some(e) = efficacy.as_deref() {
            if e.len() != self.synapses() || !all_finite(e) {
                return Err(refused("efficacy must have one finite entry per synapse"));
            }
            let finite = self.gain_pre.iter().zip(e).all(|(g, x)| (g * x).is_finite());
            if !finite {
                return Err(refused("effective synaptic weights must be finite; reduce gain or efficacy"));
            }
        }
        if let Some(b) = bias.as_deref() {
            if b.len() != self.n() || !all_finite(b) {
                return Err(refused("bias must have one finite entry per neuron"));
            }
        }
        if let Some(e) = efficacy {
            std::mem::swap(&mut self.efficacy, e);
            for ((w, g), x) in self.tables.weights.iter_mut().zip(&self.gain_pre)
                .zip(&self.efficacy)
            {
                *w = g * x;
            }
            self.fill_flats();
        }
        if let Some(b) = bias {
            std::mem::swap(&mut self.bias, b);
        }
        self.generation += 1;
        Ok(())
    }

    /// `_with_update`, in place: the efficacies, the weights the update's pass made of them and
    /// the biases, taken as they are (each buffer given takes the graph's old one back), the
    /// flat blocks written again; the graph counts one generation on. `finite` says whether the
    /// efficacies and the weights are finite, as the pass found them. A refusal changes
    /// nothing.
    pub fn update(&mut self, efficacy: &mut Vec<f64>, weights: &mut Vec<f64>,
                  bias: &mut Vec<f64>, finite: (bool, bool)) -> Result<(), Error> {
        if !finite.0 {
            return Err(refused("efficacy must have one finite entry per synapse"));
        }
        if !finite.1 {
            return Err(refused("effective synaptic weights must be finite; reduce gain or efficacy"));
        }
        if !all_finite(bias) {
            return Err(refused("bias must have one finite entry per neuron"));
        }
        if efficacy.len() != self.synapses() || weights.len() != self.synapses()
            || bias.len() != self.n()
        {
            return Err(Error::Shape("an update holds one entry per synapse and per neuron".into()));
        }
        std::mem::swap(&mut self.efficacy, efficacy);
        std::mem::swap(&mut self.tables.weights, weights);
        std::mem::swap(&mut self.bias, bias);
        self.fill_flats();
        self.generation += 1;
        Ok(())
    }

    /// `_with_dt`: the same graph at another step.
    pub fn with_dt(&self, dt: f64) -> Graph {
        let mut new = self.clone();
        new.neuron.dt = dt;
        new
    }

    // -- the kernels' arguments

    fn launch(&self, batch: usize, mask: Option<&Rows>, nudge: Option<&Nudge>)
              -> Result<Launch, Error> {
        let n = self.n();
        let (keep, keep_shape, masked) = match mask {
            None => (vec![1.0], (1, 1), false),
            Some(m) => {
                let keep = m.broadcast(batch);
                let masked = keep.iter().any(|&k| k != 1.0);
                (keep, (batch, n), masked)
            }
        };
        let pools = self.pools();
        let (held, held_shape, has_held) = match nudge.and_then(|x| x.apical_gain.as_ref()) {
            Some(h) if self.apical.is_some() => {
                if h.rows != batch || h.cols != pools {
                    return Err(refused("a nudge's apical_gain needs one entry per row and pool"));
                }
                (h.data.clone(), (batch, pools), true)
            }
            _ => (vec![0.0], (1, 1), false),
        };
        Ok(Launch { keep, keep_shape, masked, nudge: NudgeArrays::of(nudge, batch, n),
                    has_nudge: nudge.is_some(), held, held_shape, has_held })
    }

    /// The kernels' view of this graph at `batch` rows of the state `v`, `a`, `s` under the
    /// standing drive `standing` and the launch `k`.
    unsafe fn net(&self, batch: usize, v: &mut [f64], a: &mut [f64], s: &mut [f64],
                  standing: &[f64], k: &Launch) -> Net {
        let n = self.n();
        let t = &self.tables;
        let m = &self.neuron;
        let nd = &k.nudge;
        let (fraction, floor2, tiny) = match &self.apical {
            Some(ap) => (ap.fraction, ap.floor * ap.floor, APICAL_TINY),
            None => (1.0, 1.0, 0.0),
        };
        Net {
            batch,
            n,
            ranges: self.ranges(),
            v: held(v, batch, n),
            a: held(a, batch, n),
            s: held(s, batch, n),
            standing: seen(standing, batch, n),
            flat: column(&t.flat),
            pairs: column(&t.pairs),
            flat_apical: column(t.flat_apical.as_deref().unwrap_or(&t.flat)),
            apical_pairs: column(&t.apical_pairs),
            starts: column(&self.starts),
            pair_pre: column(&self.pair_pre),
            pair_post: column(&self.pair_post),
            offset: column(&self.sparse),
            keep: seen(&k.keep, k.keep_shape.0, k.keep_shape.1),
            masked: k.masked,
            dt: m.dt,
            slope: m.slope,
            threshold: m.threshold,
            rest: m.rest,
            leak: m.leak,
            has_adapt: m.adaptation.is_some(),
            strength: m.strength(),
            tau: m.tau(),
            has_nudge: k.has_nudge,
            target: seen(&nd.target, nd.target_shape.0, nd.target_shape.1),
            nmask: column(&nd.nmask),
            gid: column(&nd.gid),
            ngroups: nd.ngroups,
            beta: nd.beta,
            softmax_t: column(&nd.softmax_t),
            weight: column(&nd.weight),
            anchor: seen(&nd.anchor, nd.anchor_shape.0, nd.anchor_shape.1),
            anchor_gain: column(&nd.anchor_gain),
            has_apical: self.apical.is_some(),
            pool_of: column(&t.pool_of),
            pool_index: column(&t.pool_index),
            pool_starts: column(&t.pool_starts),
            pool_run: column(&t.pool_run),
            fraction,
            floor2,
            tiny,
            held: seen(&k.held, k.held_shape.0, k.held_shape.1),
            has_held: k.has_held,
            per_life: false,
        }
    }

    /// `_publish`: the activations of `v` (the state's own where they are its publication),
    /// under the mask.
    fn publish(&self, v: &[f64], batch: usize, state: Option<&State>, k: &Launch) -> Vec<f64> {
        let n = self.n();
        let mut s = vec![0.0f64; batch * n];
        match state {
            Some(st) if st.published => s.copy_from_slice(&st.activation),
            _ => unsafe {
                let m = &self.neuron;
                let none = [0.0f64];
                settle::publish(seen(v, batch, n), held(&mut s, batch, n), column(&none), false,
                                m.slope, m.threshold, m.rest, m.leak);
            },
        }
        if k.masked {
            for (x, keep) in s.iter_mut().zip(&k.keep) {
                *x *= keep;
            }
        }
        s
    }

    fn standing(&self, drive: &[f64], batch: usize) -> Vec<f64> {
        let n = self.n();
        let mut out = vec![0.0f64; batch * n];
        for b in 0..batch {
            for i in 0..n {
                out[b * n + i] = drive[b * n + i] + self.bias[i];
            }
        }
        out
    }

    fn range_of(&self, i: usize) -> usize {
        self.starts.partition_point(|&x| x <= i as i64) - 1
    }

    // -- settling

    /// `settle_batch(drive, steps=, state=, mask=, nudge=, tolerance=)` on the compiled path.
    pub fn settle_batch(&self, drive: &Rows, steps: i64, state: Option<&State>,
                        mask: Option<&Rows>, nudge: Option<&Nudge>, tolerance: Option<f64>)
                        -> Result<State, Error> {
        if drive.ndim == 0 || drive.ndim > 2 || drive.rows == 0 {
            return Err(refused("drive must have shape (batch, neurons) with at least one row"));
        }
        if steps < 0 {
            return Err(refused("steps must be a nonnegative integer"));
        }
        if let Some(t) = tolerance {
            if !t.is_finite() || t < 0.0 {
                return Err(refused("tolerance must be finite and nonnegative"));
            }
        }
        let tolerance = tolerance.filter(|&t| t != 0.0);
        let (batch, n) = (drive.rows, drive.cols);
        if n != self.n() {
            return Err(refused("drive must have one column per neuron"));
        }
        if !all_finite(&drive.data) {
            return Err(refused("drive must be finite and the brain must contain neurons"));
        }
        if let Some(m) = mask {
            if !m.fits(batch, n) {
                return Err(refused("mask must have one entry per neuron, optionally per batch row"));
            }
            if m.data.iter().any(|&k| !(k.is_finite() && (0.0..=1.0).contains(&k))) {
                return Err(refused("mask must contain finite fractions in [0, 1]"));
            }
        }
        if let Some(nd) = nudge {
            if nd.mask.len() != n {
                return Err(refused("nudge mask must have one entry per neuron"));
            }
            if !nd.target.fits(batch, n) {
                return Err(refused("nudge target must match the drive's neurons and batch size"));
            }
            if nd.weight.as_ref().is_some_and(|w| w.len() != batch) {
                return Err(refused("nudge weight must have one entry per batch row"));
            }
            if nd.anchor.as_ref().is_some_and(|x| !x.fits(batch, n)) {
                return Err(refused("nudge anchor must match the drive's neurons and batch size"));
            }
        }
        let (mut v, mut a) = match state {
            None => (vec![0.0f64; batch * n], vec![0.0f64; batch * n]),
            Some(st) => {
                if st.batch != batch || st.v.len() != batch * n {
                    return Err(refused("state batch does not match the drive batch"));
                }
                (st.v.clone(), st.adaptation.clone())
            }
        };
        if !(all_finite(&v) && all_finite(&a)) {
            return Err(refused("warm potentials and adaptation must be finite"));
        }
        // `fused_settle`
        let k = self.launch(batch, mask, nudge)?;
        let mut s = self.publish(&v, batch, state, &k);
        if steps == 0 {
            let published = match state {
                Some(st) if st.published => {
                    v.iter().zip(&st.v).all(|(x, y)| x == y)
                        && s.iter().zip(&st.activation).all(|(x, y)| x == y)
                }
                _ => false,
            };
            return Ok(State { batch, v, activation: s, adaptation: a, steps: 0,
                              activity_change: Some(vec![0.0; batch]), published });
        }
        let standing = self.standing(&drive.data, batch);
        let mut freezable = self.tables.freezable.clone();
        if self.neuron.adaptation.is_some() {
            freezable.iter_mut().for_each(|f| *f = 0);
        } else if nudge.is_some() {
            for i in 0..n {
                if k.nudge.nmask[i] > 0.0 || k.nudge.anchor_gain[i] > 0.0 {
                    freezable[self.range_of(i)] = 0;
                }
            }
        }
        let mut change = vec![0.0f64; batch];
        let taken = unsafe {
            let net = self.net(batch, &mut v, &mut a, &mut s, &standing, &k);
            let length = net.kept_len();
            let mut kept = vec![0.0f64; length];
            let none = [0.0f64];
            settle::settle(&net, column(&freezable), steps as usize,
                           tolerance.unwrap_or(0.0), tolerance.is_some(), column(&none), false,
                           held(&mut change, batch, 1), held(&mut kept, length, 1)) as i64
        };
        if taken < 0 {
            return Err(fault("the settle kernel refused its arrays"));
        }
        let published = !k.masked && no_nan(&v) && no_nan(&s);
        Ok(State { batch, v, activation: s, adaptation: a, steps: taken,
                   activity_change: Some(change), published })
    }

    /// `equilibrate(drive, budget=, chunk=, tolerance=, state=, mask=, nudge=, damping=,
    /// fast=)`.
    #[allow(clippy::too_many_arguments)]
    pub fn equilibrate(&self, drive: &Rows, budget: i64, chunk: i64, tolerance: f64,
                       state: Option<&State>, mask: Option<&Rows>, nudge: Option<&Nudge>,
                       damping: i64, fast: bool) -> Result<Equilibrium, Error> {
        if budget < 0 || chunk < 1 || damping < 0 {
            return Err(refused("budget, chunk and damping are integers, chunk at least 1"));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(refused("tolerance must be finite and nonnegative"));
        }
        if damping == 0 {
            return self.local(drive, state, budget, chunk, tolerance, mask, nudge, false, fast);
        }
        let attempts = (damping + 1).min(budget.max(1));
        self.neuron.check_at(halved(self.neuron.dt, attempts - 1))?;
        let (mut used, mut checks, mut stagnation) = (0i64, 0i64, 0i64);
        let mut current = state.cloned();
        let mut result = None;
        for halving in 0..attempts {
            let portion = (budget - used + attempts - halving - 1) / (attempts - halving);
            let candidate;
            let graph = if halving == 0 {
                self
            } else {
                candidate = self.with_dt(halved(self.neuron.dt, halving));
                &candidate
            };
            let phase = graph.local(drive, current.as_ref(), portion, chunk, tolerance, mask,
                                    nudge, halving < attempts - 1, false)?;
            used += phase.steps;
            checks += phase.residual_checks;
            stagnation += phase.stagnation_checks;
            let mut phase = phase;
            let mut found = phase.clone();
            if halving > 0 {
                let error = self.check(drive, &mut phase.state, nudge, mask)?;
                checks += 1;
                found = Equilibrium {
                    state: phase.state.clone(), residual: error, steps: used, tolerance,
                    residual_checks: checks, damping_halvings: halving,
                    stagnation_checks: stagnation, row_sweeps: None,
                };
            }
            if found.qualified() || used >= budget {
                return Ok(found);
            }
            let st = &phase.state;
            if !(all_finite(&st.v) && all_finite(&st.activation) && all_finite(&st.adaptation)) {
                return Ok(found);
            }
            current = Some(phase.state);
            result = Some(found);
        }
        result.ok_or_else(|| refused("a damped solve makes at least one attempt"))
    }

    /// `_equilibrate_local`: the published state, the compiled solve where it serves, else
    /// chunks of settles and their checks.
    #[allow(clippy::too_many_arguments)]
    fn local(&self, drive: &Rows, state: Option<&State>, budget: i64, chunk: i64,
             tolerance: f64, mask: Option<&Rows>, nudge: Option<&Nudge>, stop_if_stalled: bool,
             fast: bool) -> Result<Equilibrium, Error> {
        let mut current = self.settle_batch(drive, 0, state, mask, nudge, None)?;
        if mask.is_none() && nudge.is_none() && !stop_if_stalled {
            if let Some(solved) = self.solve(drive, &current, budget, chunk, tolerance, fast)? {
                return Ok(solved);
            }
        }
        let mut error = self.check(drive, &mut current, nudge, mask)?;
        let (mut used, mut checks, mut stagnation, mut repeats) = (0i64, 1i64, 0i64, 0i64);
        while used < budget && !error.iter().all(|&e| e <= tolerance) {
            let warm = all_finite(&current.v) && all_finite(&current.activation)
                && all_finite(&current.adaptation);
            if !all_finite(&error) && !warm {
                break; // a divergent state is no warm start for another chunk
            }
            let previous = current;
            current = self.settle_batch(drive, chunk.min(budget - used), Some(&previous), mask,
                                        nudge, None)?;
            used += current.steps;
            error = self.check(drive, &mut current, nudge, mask)?;
            checks += 1;
            if stop_if_stalled && used < budget && !error.iter().all(|&e| e <= tolerance) {
                stagnation += 1;
                repeats = if repeated(&previous, &current) { repeats + 1 } else { 0 };
                if repeats >= 2 {
                    break;
                }
            }
        }
        Ok(Equilibrium { state: current, residual: error, steps: used, tolerance,
                         residual_checks: checks, damping_halvings: 0,
                         stagnation_checks: stagnation, row_sweeps: None })
    }

    /// `_solve_compiled`: a free solve from the published state `current` in one compiled
    /// call (`fused_solve`, or `fused_fast_solve` for `fast` without a compartment); None
    /// under adaptation, which the launches serve.
    fn solve(&self, drive: &Rows, current: &State, budget: i64, chunk: i64, tolerance: f64,
             fast: bool) -> Result<Option<Equilibrium>, Error> {
        let (batch, n) = (current.batch, self.n());
        let m = &self.neuron;
        if m.adaptation.is_some() {
            return Ok(None);
        }
        let standing = self.standing(&drive.data, batch);
        let mut v = current.v.clone();
        let mut s = current.activation.clone();
        let a = current.adaptation.clone();
        let mut out = vec![0.0f64; batch];
        let mut report = [0i64; 4];
        let clip = |x: i64| x.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        if fast && self.apical.is_none() && m.dt == 1.0 {
            let mut size = 0usize;
            for &c in &self.pair_post {
                size += batch * (self.start(c as usize + 1) - self.start(c as usize));
            }
            let length = 3 * batch * n + size + self.ranges() + RECORD;
            let mut kept = vec![0.0f64; length];
            let code = unsafe {
                fast::fast(held(&mut v, batch, n), held(&mut s, batch, n),
                           seen(&standing, batch, n), column(&self.tables.flat),
                           column(&self.starts), column(&self.pair_pre), column(&self.pair_post),
                           column(&self.offset), m.slope, m.threshold, m.rest, m.leak,
                           clip(budget), clip(chunk.min(16)), tolerance,
                           tolerance.min(FAST_SETTLED), held(&mut out, batch, 1),
                           held(&mut report, 4, 1), held(&mut kept, length, 1))
            };
            if code < 0 {
                return Err(fault("the fast solve refused its arrays"));
            }
            let published = no_nan(&v) && no_nan(&s);
            return Ok(Some(Equilibrium {
                state: State { batch, v, activation: s, adaptation: a, steps: report[2],
                               activity_change: None, published },
                residual: out, steps: report[0], tolerance, residual_checks: report[1],
                damping_halvings: 0, stagnation_checks: 0, row_sweeps: Some(report[3]),
            }));
        }
        let k = self.launch(batch, None, None)?;
        let mut a = a;
        let mut change = vec![0.0f64; batch];
        let code = unsafe {
            let net = self.net(batch, &mut v, &mut a, &mut s, &standing, &k);
            let length = net.kept_len();
            let (mut ka, mut kb) = (vec![0.0f64; length], vec![0.0f64; length]);
            let none = [0.0f64];
            settle::solve(&net, column(&self.tables.freezable), clip(budget), clip(chunk),
                          tolerance, held(&mut out, batch, 1), column(&none), false, false,
                          held(&mut change, batch, 1), held(&mut ka, length, 1),
                          held(&mut kb, length, 1), held(&mut report, 4, 1))
        };
        if code < 0 {
            return Err(fault("the solve kernel refused its arrays"));
        }
        let published = no_nan(&v) && no_nan(&s);
        Ok(Some(Equilibrium {
            state: State { batch, v, activation: s, adaptation: a, steps: report[2],
                           activity_change: Some(change), published },
            residual: out, steps: report[0], tolerance, residual_checks: report[1],
            damping_halvings: 0, stagnation_checks: 0, row_sweeps: None,
        }))
    }

    // -- the checks

    fn check_shapes(&self, drive: &Rows, state: &State, mask: Option<&Rows>)
                    -> Result<(), Error> {
        let n = self.n();
        if drive.ndim == 0 || drive.ndim > 2 || drive.cols != n {
            return Err(refused("drive must have one column per neuron"));
        }
        if let Some(m) = mask {
            if !m.fits(drive.rows, n) {
                return Err(refused("mask must have one entry per neuron, optionally per batch row"));
            }
        }
        if state.batch != drive.rows || state.v.len() != drive.rows * n {
            return Err(refused("state batch does not match the drive batch"));
        }
        Ok(())
    }

    /// `residual(drive, state, nudge=, mask=)`: each row's largest equation error, in one
    /// compiled pass (`fused_residual`).
    pub fn residual(&self, drive: &Rows, state: &State, nudge: Option<&Nudge>,
                    mask: Option<&Rows>) -> Result<Vec<f64>, Error> {
        self.check(drive, &mut state.clone(), nudge, mask)
    }

    /// `residual`, and the state keeps the check's record as the Python's does
    /// (`_keep_products`): unmasked, its activations are then the ones it publishes.
    pub fn check(&self, drive: &Rows, state: &mut State, nudge: Option<&Nudge>,
                 mask: Option<&Rows>) -> Result<Vec<f64>, Error> {
        self.check_shapes(drive, state, mask)?;
        let batch = drive.rows;
        let k = self.launch(batch, mask, nudge)?;
        let mut v = state.v.clone();
        let mut a = state.adaptation.clone();
        let mut s = self.publish(&v, batch, Some(state), &k);
        let standing = self.standing(&drive.data, batch);
        let mut out = vec![0.0f64; batch];
        let code = unsafe {
            let net = self.net(batch, &mut v, &mut a, &mut s, &standing, &k);
            let length = net.kept_len();
            let mut kept = vec![0.0f64; length];
            let none = [0.0f64];
            settle::residual(&net, held(&mut out, batch, 1), column(&none), false, false,
                             held(&mut kept, length, 1))
        };
        if code < 0 {
            return Err(fault("the residual kernel refused its arrays"));
        }
        state.published = !k.masked && no_nan(&v)
            && s.iter().zip(&state.activation).all(|(x, y)| x == y);
        Ok(out)
    }

    /// `residual(..., per_unit=True)`: each neuron's signed error of the potential equation,
    /// `(rows, neurons)`, on the NumPy path.
    pub fn residual_per_unit(&self, drive: &Rows, state: &State, nudge: Option<&Nudge>,
                             mask: Option<&Rows>) -> Result<Vec<f64>, Error> {
        self.check_shapes(drive, state, mask)?;
        let (batch, n) = (drive.rows, self.n());
        let keep = match mask {
            Some(m) => m.broadcast(batch),
            None => vec![1.0f64; batch * n],
        };
        let v = &state.v;
        let s: Vec<f64> = v.iter().zip(&keep).map(|(&x, &k)| self.neuron.act(x) * k).collect();
        let d = &drive.data;
        let adapt = self.neuron.adaptation;
        let mut error;
        match &self.apical {
            None => {
                let syn = self.transport(&s, batch, &self.tables.flat, &self.tables.pairs);
                error = vec![0.0f64; batch * n];
                for b in 0..batch {
                    for i in 0..n {
                        let at = b * n + i;
                        error[at] = syn[at] + d[at] + self.bias[i] - v[at];
                    }
                }
                if let Some((_, strength)) = adapt {
                    for (e, x) in error.iter_mut().zip(&state.adaptation) {
                        *e -= strength * x;
                    }
                }
                if let Some(nd) = nudge {
                    for (e, x) in error.iter_mut().zip(nd.drive(&s, batch)?) {
                        *e += x;
                    }
                }
            }
            Some(_) => {
                let (basal, raw) = self.split_input(&s, batch);
                error = basal;
                for b in 0..batch {
                    for i in 0..n {
                        error[b * n + i] += d[b * n + i] + self.bias[i];
                    }
                }
                if let Some((_, strength)) = adapt {
                    for (e, x) in error.iter_mut().zip(&state.adaptation) {
                        *e -= strength * x;
                    }
                }
                if let Some(nd) = nudge {
                    for (e, x) in error.iter_mut().zip(nd.drive(&s, batch)?) {
                        *e += x;
                    }
                }
                let held_gain = nudge.and_then(|x| x.apical_gain.as_ref());
                let top = self.apical_drive(&error, &raw, batch, held_gain)?;
                for (e, x) in error.iter_mut().zip(top) {
                    *e += x;
                }
                for (e, x) in error.iter_mut().zip(v) {
                    *e -= x;
                }
            }
        }
        let dt = self.neuron.dt;
        for k in 0..batch * n {
            error[k] = keep[k] * error[k] + (keep[k] - 1.0) * v[k] / dt;
        }
        Ok(error)
    }

    /// `apical_gain(drive, state)`: the compartment's gain `(batch, pools)` at the state's
    /// activations; None without the compartment.
    pub fn apical_gain(&self, drive: &Rows, state: &State) -> Result<Option<Vec<f64>>, Error> {
        if self.apical.is_none() {
            return Ok(None);
        }
        let n = self.n();
        let batch = state.batch;
        if drive.cols != n || state.activation.len() != batch * n
            || !(drive.rows == batch || drive.rows == 1)
        {
            return Err(refused("state batch does not match the drive batch"));
        }
        let d = drive.broadcast(batch);
        let (mut basal, raw) = self.split_input(&state.activation, batch);
        for b in 0..batch {
            for i in 0..n {
                basal[b * n + i] += d[b * n + i] + self.bias[i];
            }
        }
        if let Some((_, strength)) = self.neuron.adaptation {
            for (x, a) in basal.iter_mut().zip(&state.adaptation) {
                *x -= strength * a;
            }
        }
        Ok(Some(self.pool_gain(&basal, &raw, batch)))
    }

    // -- the NumPy path's transport and compartment

    /// `BlockTransport(layout, flat, pairs).synaptic_input(s)`: every block's product summed
    /// in order, added into its neurons' input in block order.
    fn transport(&self, s: &[f64], batch: usize, flat: &[f64], pairs: &[i64]) -> Vec<f64> {
        let n = self.n();
        let mut out = vec![0.0f64; batch * n];
        for &k in pairs {
            let k = k as usize;
            let (a0, a1, b0, b1) = self.bounds(k);
            let (na, nb) = (a1 - a0, b1 - b0);
            let block = &flat[self.offset[k] as usize..self.offset[k + 1] as usize];
            for b in 0..batch {
                let product = np::matmul(&s[b * n + a0..b * n + a1], 1, na, block, nb);
                for (o, p) in out[b * n + b0..b * n + b1].iter_mut().zip(product) {
                    *o += p;
                }
            }
        }
        out
    }

    /// `_split_input`: the synaptic input of the basal synapses and of the apical synapses,
    /// apart.
    fn split_input(&self, s: &[f64], batch: usize) -> (Vec<f64>, Vec<f64>) {
        let t = &self.tables;
        let up = t.flat_apical.as_deref().unwrap_or(&t.flat);
        (self.transport(s, batch, &t.flat, &t.pairs), self.transport(s, batch, up, &t.apical_pairs))
    }

    /// `Apical.gain(basal, raw)`: `fraction * rms(b) / sqrt(rms(u)^2 + (floor * rms(b))^2)`
    /// per row and pool.
    fn pool_gain(&self, basal: &[f64], raw: &[f64], batch: usize) -> Vec<f64> {
        let ap = self.apical.as_ref().expect("a compartment");
        let (fraction, floor2) = (ap.fraction, ap.floor * ap.floor);
        let n = self.n();
        let pools = self.pools();
        let mut out = vec![0.0f64; batch * pools];
        for (k, members) in self.tables.members.iter().enumerate() {
            let m = members.len() as f64;
            for b in 0..batch {
                let bs: Vec<f64> = members.iter().map(|&i| basal[b * n + i]).collect();
                let us: Vec<f64> = members.iter().map(|&i| raw[b * n + i]).collect();
                let basal2 = np::sum_squares(&bs) / m;
                let raw2 = np::sum_squares(&us) / m;
                out[b * pools + k] =
                    fraction * np::sqrt(basal2) / np::sqrt(raw2 + floor2 * basal2 + APICAL_TINY);
            }
        }
        out
    }

    /// `Apical.drive(basal, raw, held)`: the raw apical input at the normalization's gain or
    /// the held one, zero off the pools.
    fn apical_drive(&self, basal: &[f64], raw: &[f64], batch: usize, held_gain: Option<&Rows>)
                    -> Result<Vec<f64>, Error> {
        let n = self.n();
        let pools = self.pools();
        let gain = match held_gain {
            Some(h) => {
                if h.rows != batch || h.cols != pools || h.ndim != 2 {
                    return Err(refused("a nudge's apical_gain needs one entry per row and pool"));
                }
                h.data.clone()
            }
            None => self.pool_gain(basal, raw, batch),
        };
        let mut out = vec![0.0f64; batch * n];
        for (k, members) in self.tables.members.iter().enumerate() {
            for b in 0..batch {
                for &i in members {
                    out[b * n + i] = gain[b * pools + k] * raw[b * n + i];
                }
            }
        }
        Ok(out)
    }

    // -- calls by name

    fn arguments(&self, args: &Named) -> Result<Arguments, Error> {
        let drive = Rows::of(args, "drive")?;
        let state = if args.has("state.v") { Some(State::import(args, "state", self.n())?) } else { None };
        let mask = if args.has("mask") { Some(Rows::of(args, "mask")?) } else { None };
        let nudge = Nudge::import(args, "nudge")?;
        Ok((drive, state, mask, nudge))
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let n = self.n();
        match method {
            "settle_batch" => {
                let (drive, state, mask, nudge) = self.arguments(args)?;
                if args.bools("trajectory")?[0] {
                    return Err(refused("the core keeps no trajectory"));
                }
                let tolerance = if args.has("tolerance") { Some(args.f64("tolerance")?) } else { None };
                let settled = self.settle_batch(&drive, args.i64("steps")?, state.as_ref(),
                                                mask.as_ref(), nudge.as_ref(), tolerance)?;
                settled.export("state", n, out);
            }
            "equilibrate" => {
                let (drive, state, mask, nudge) = self.arguments(args)?;
                let solved = self.equilibrate(
                    &drive, args.i64("budget")?, args.i64("chunk")?, args.f64("tolerance")?,
                    state.as_ref(), mask.as_ref(), nudge.as_ref(), args.i64("damping")?,
                    args.bools("fast")?[0])?;
                solved.export(n, out);
            }
            "residual" => {
                let (drive, state, mask, nudge) = self.arguments(args)?;
                let state = state.ok_or_else(|| refused("a check needs a state"))?;
                let drive = Rows { ndim: 2, ..drive };
                if args.bools("per_unit")?[0] {
                    let error = self.residual_per_unit(&drive, &state, nudge.as_ref(), mask.as_ref())?;
                    out.put("residual", Array::from_f64(&[drive.rows, n], &error));
                } else {
                    let error = self.residual(&drive, &state, nudge.as_ref(), mask.as_ref())?;
                    out.put("residual", Array::from_f64(&[error.len()], &error));
                }
            }
            "apical_gain" => {
                let (drive, state, _, _) = self.arguments(args)?;
                let state = state.ok_or_else(|| refused("a gain needs a state"))?;
                if let Some(gain) = self.apical_gain(&drive, &state)? {
                    out.put("gain", Array::from_f64(&[state.batch, self.pools()], &gain));
                }
            }
            "state" => {} // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

/// `_repeated_state`: every equation variable equal to the previous checkpoint's within eight
/// units of rounding.
fn repeated(previous: &State, current: &State) -> bool {
    let rounding = 8.0 * f64::EPSILON;
    [(&previous.v, &current.v), (&previous.activation, &current.activation),
     (&previous.adaptation, &current.adaptation)]
        .iter()
        .all(|(before, after)| {
            before.iter().zip(after.iter()).all(|(&x, &y)| {
                (y - x).abs() <= rounding * (1.0 + np::maximum(y.abs(), x.abs()))
            })
        })
}
