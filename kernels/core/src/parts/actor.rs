//! `cadence.plasticity.ActorCritic` and `Maturity`: the actor samples an action from the settled
//! outputs and keeps the act's two nudged phases; at the outcome the dopamine (the critic's
//! temporal-difference error, capped) steps every synapse on its eligibility trace, the critic on
//! its own, and the memory's and the carried routes' parameters on theirs. The host's path, in
//! the Python's order. The actor holds its maturity; the learner, the graph it reads and
//! replaces, the recall memory and the carried gains are the brain's, handed in by the caller.
//! `Reach` holds them all, as `cadence.core.parts.actor` crosses them.

use super::graph::{Graph, Nudge, Rows, State, Temperature};
use super::hearing::{Graph as Heard, Hearing};
use super::learner::{Labels, Learner};
use super::recall::{Recall, F};
use super::trace::Rows as Columns;
use crate::lesson;
use crate::np;
use crate::rng::Pcg64;
use crate::state::{Array, Error, Named};
use crate::{Index, View};

fn refused(text: &str) -> Error {
    Error::Refused(text.to_string())
}

/// `SATURATION_BAND`: an output within this of 0 or 1 has no slope left for a nudge.
const SATURATION_BAND: f64 = 0.02;

/// An array of any shape, row-major.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor { shape: shape.to_vec(), data: vec![0.0; shape.iter().product()] }
    }
    /// This array at `shape`, its room kept (the entries are then the caller's to write).
    fn reshape(&mut self, shape: &[usize]) {
        self.shape.clear();
        self.shape.extend_from_slice(shape);
        self.data.resize(shape.iter().product(), 0.0);
    }
    fn of(s: &Named, name: &str) -> Result<Option<Tensor>, Error> {
        if !s.has(name) {
            return Ok(None);
        }
        Ok(Some(Tensor { shape: s.shape(name)?, data: s.f64s(name)? }))
    }
    fn array(&self) -> Array {
        Array::from_f64(&self.shape, &self.data)
    }
    /// The first dimension (`len`).
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    fn width(&self) -> usize {
        if self.rows() == 0 { 0 } else { self.data.len() / self.rows() }
    }
    /// `x[done] = 0.0`: the rows of finished streams hold nothing.
    fn forget(&mut self, done: &[bool]) {
        let w = self.width();
        for (b, &gone) in done.iter().enumerate() {
            if gone {
                self.data[b * w..(b + 1) * w].iter_mut().for_each(|x| *x = 0.0);
            }
        }
    }
    /// `(weight[:, None] * x).mean(axis=0)`: each row at its weight, the rows' mean.
    fn weighted_mean(&self, weight: &[f64]) -> Vec<f64> {
        let (rows, w) = (self.rows(), self.width());
        let mut out = vec![0.0f64; w];
        for (row, &wb) in self.data.chunks(w.max(1)).zip(weight).take(rows) {
            for (o, x) in out.iter_mut().zip(row) {
                *o += wb * x;
            }
        }
        out.iter().map(|x| x / rows as f64).collect()
    }
}

/// `ActorCriticConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub gamma: f64,
    pub lam: f64,
    pub eta: f64,
    pub eta_bias: f64,
    pub eta_critic: f64,
    pub dopamine_cap: f64,
    pub entropy: f64,
    pub eligibility_steps: Option<i64>,
    pub hold: Option<f64>,
}

impl Config {
    /// The actor's eligibility decay per moment: `hold`, or `gamma * lam`.
    pub fn decay(&self) -> f64 {
        self.hold.unwrap_or(self.gamma * self.lam)
    }
}

const FLOATS: [&str; 7] = ["gamma", "lam", "eta", "eta_bias", "eta_critic", "dopamine_cap", "entropy"];

/// `Maturity`: the use of a brain's plasticity, which scales the actor's steps.
#[derive(Clone, Debug, PartialEq)]
pub struct Maturity {
    pub scale: f64,
    pub recovery: f64,
    pub use_: f64,
}

impl Maturity {
    pub const ORDER: f64 = 4.0;
    pub const LIMIT: f64 = 1e64;

    pub fn import(s: &Named) -> Result<Maturity, Error> {
        Ok(Maturity { scale: s.f64("scale")?, recovery: s.f64("recovery")?, use_: s.f64("use")? })
    }
    pub fn export(&self, out: &mut Named) {
        out.put("scale", Array::scalar_f64(self.scale));
        out.put("recovery", Array::scalar_f64(self.recovery));
        out.put("use", Array::scalar_f64(self.use_));
    }
    /// `factor()`: `1 / (1 + (use / scale) ** ORDER)`; none left where the power overflows.
    pub fn factor(&self) -> f64 {
        let ratio = self.use_ / self.scale;
        let power = np::pow(ratio, Self::ORDER);
        if power.is_infinite() && ratio.is_finite() {
            return 0.0; // Python's OverflowError
        }
        1.0 / (1.0 + power)
    }
    /// `relax()`: one moment passes.
    pub fn relax(&mut self) {
        self.use_ *= 1.0 - 1.0 / self.recovery;
    }
    pub fn call(&mut self, method: &str, out: &mut Named) -> Result<(), Error> {
        match method {
            "factor" => out.put("factor", Array::scalar_f64(self.factor())),
            "relax" => self.relax(),
            "state" => {}
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

/// The decision a sampled act keeps for its outcome: its two nudged phases and the value of
/// the free state it was chosen in.
#[derive(Clone, Debug, PartialEq)]
pub struct Pending {
    pub plus: State,
    pub minus: State,
    pub value: Vec<f64>,
}

/// A temperature a caller gives: one, or one per motor slot.
#[derive(Clone, Debug, PartialEq)]
pub enum Heat {
    One(f64),
    Each(Vec<f64>),
}

/// What an outcome did (`learn`'s report): each reading by its name.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Report(pub Vec<(&'static str, f64)>);

impl Report {
    /// The reading `name`, where the outcome made one.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
    }
    /// The mean size of the observed rows' temporal-difference errors.
    pub fn td_error(&self) -> f64 {
        self.get("td_error").unwrap_or(f64::NAN)
    }
    /// The sweeps the next state's free phase took.
    pub fn free_steps(&self) -> f64 {
        self.get("free_steps").unwrap_or(f64::NAN)
    }
}

/// `ActorCritic` on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct ActorCritic {
    pub config: Config,
    pub critic_index: Vec<i64>,
    pub w_critic: Vec<f64>,
    pub b_critic: f64,
    pub rng: Pcg64,
    pub updates: i64,
    pub trace: Option<Tensor>,
    pub trace_bias: Option<Tensor>,
    pub trace_critic: Option<Tensor>,
    pub trace_memory: Option<Tensor>,
    pub trace_hearing: Option<Tensor>,
    pub salience: Option<Tensor>,
    /// The drive of the free state, as held for the next read.
    pub drive: Option<Rows>,
    pub free: Option<State>,
    /// The learner's graph (`Learner::generation`) that settled `free` (`_free_brain`).
    pub free_generation: Option<u64>,
    pub pending: Option<Pending>,
    pub pending_memory: Option<Tensor>,
    pub pending_hearing: Option<Tensor>,
    /// Each row's `-log p(action) - H` and `H` of the act that awaits its outcome.
    pub pending_entropy: Option<(Vec<f64>, Vec<f64>)>,
    /// The brain's maturity, which scales the steps.
    pub maturity: Option<Maturity>,
    /// The carried routes whose gain is the recall memory's own (`Recall.hearing`): the
    /// Python's `Hearing.set` writes it there.
    pub hearing_routes: Vec<i64>,
    spare: Spare,
}

fn generator(s: &Named) -> Result<Pcg64, Error> {
    let (st, inc) = (s.u64s("rng.state")?, s.u64s("rng.inc")?);
    Ok(Pcg64 {
        state: ((st[0] as u128) << 64) | st[1] as u128,
        inc: ((inc[0] as u128) << 64) | inc[1] as u128,
        has_uint32: s.i64("rng.has_uint32")? != 0,
        uinteger: s.i64("rng.uinteger")? as u32,
    })
}

fn same_rows(a: &Rows, b: &Rows) -> bool {
    a.rows == b.rows && a.cols == b.cols && a.data.iter().zip(&b.data).all(|(x, y)| x == y)
}

/// `_softmax(s, temperature)` of rows of `width`.
fn softmax(s: &[f64], width: usize, temperature: f64) -> Vec<f64> {
    let mut z: Vec<f64> = s.iter().map(|x| x / temperature).collect();
    let finite = z.iter().all(|x| x.is_finite());
    for (zr, sr) in z.chunks_mut(width).zip(s.chunks(width)) {
        if finite {
            let top = np::max(zr);
            zr.iter_mut().for_each(|x| *x -= top);
        } else {
            let top = np::max(sr);
            for (x, &v) in zr.iter_mut().zip(sr) {
                *x = (v - top) / temperature;
            }
        }
    }
    let p: Vec<f64> = z.iter().map(|&x| np::exp(x)).collect();
    p.chunks(width)
        .flat_map(|row| {
            let total = np::sum(row);
            row.iter().map(move |x| x / total)
        })
        .collect()
}

impl ActorCritic {
    /// The actor of the state `s` (its own entries and `maturity.*`) on `learner` and `graph`.
    pub fn import(s: &Named, learner: &Learner, graph: &Graph) -> Result<ActorCritic, Error> {
        let opt = |name: &str| Tensor::of(s, name);
        let n = learner.n();
        let free = if s.has("free.v") { Some(State::import(s, "free", n)?) } else { None };
        let free_generation = (free.is_some() && s.bools("free.current")?[0])
            .then_some(graph.generation);
        let pending = if s.has("pending.value") {
            Some(Pending {
                plus: State::import(s, "pending.plus", n)?,
                minus: State::import(s, "pending.minus", n)?,
                value: s.f64s("pending.value")?,
            })
        } else {
            None
        };
        let entropy = if s.has("pending_entropy.surprise") {
            Some((s.f64s("pending_entropy.surprise")?, s.f64s("pending_entropy.entropy")?))
        } else {
            None
        };
        let f = |k: &str| s.f64(&format!("config.{k}"));
        let actor = ActorCritic {
            config: Config {
                gamma: f("gamma")?,
                lam: f("lam")?,
                eta: f("eta")?,
                eta_bias: f("eta_bias")?,
                eta_critic: f("eta_critic")?,
                dopamine_cap: f("dopamine_cap")?,
                entropy: f("entropy")?,
                eligibility_steps: if s.has("config.eligibility_steps") {
                    Some(s.i64("config.eligibility_steps")?)
                } else {
                    None
                },
                hold: if s.has("config.hold") { Some(f("hold")?) } else { None },
            },
            critic_index: s.i64s("critic_index")?,
            w_critic: s.f64s("w_critic")?,
            b_critic: s.f64("b_critic")?,
            rng: generator(s)?,
            updates: s.i64("updates")?,
            trace: opt("trace")?,
            trace_bias: opt("trace_bias")?,
            trace_critic: opt("trace_critic")?,
            trace_memory: opt("trace_memory")?,
            trace_hearing: opt("trace_hearing")?,
            salience: opt("salience")?,
            drive: if s.has("drive") { Some(Rows::of(s, "drive")?) } else { None },
            free,
            free_generation,
            pending,
            pending_memory: opt("pending_memory")?,
            pending_hearing: opt("pending_hearing")?,
            pending_entropy: entropy,
            maturity: if s.holds("maturity") { Some(Maturity::import(&s.under("maturity"))?) }
                      else { None },
            hearing_routes: if s.has("hearing_routes") { s.i64s("hearing_routes")? }
                            else { vec![] },
            spare: Spare::default(),
        };
        if actor.critic_index.is_empty()
            || actor.critic_index.iter().any(|&i| i < 0 || i as usize >= n)
            || actor.w_critic.len() != actor.critic_index.len()
        {
            return Err(refused("critic must contain distinct integer neuron indices"));
        }
        Ok(actor)
    }

    /// The actor's own entries and `maturity.*`, after the learner's, the memory's and the
    /// carried gains' (`hearing_routes` the first of them).
    pub fn export(&self, learner: &Learner, graph: &Graph, out: &mut Named) {
        let n = learner.n();
        if let Some(m) = &self.maturity {
            let mut part = Named::new();
            m.export(&mut part);
            out.put_under("maturity", &part);
        }
        for name in FLOATS {
            let value = match name {
                "gamma" => self.config.gamma,
                "lam" => self.config.lam,
                "eta" => self.config.eta,
                "eta_bias" => self.config.eta_bias,
                "eta_critic" => self.config.eta_critic,
                "dopamine_cap" => self.config.dopamine_cap,
                _ => self.config.entropy,
            };
            out.put(&format!("config.{name}"), Array::scalar_f64(value));
        }
        if let Some(k) = self.config.eligibility_steps {
            out.put("config.eligibility_steps", Array::scalar_i64(k));
        }
        if let Some(h) = self.config.hold {
            out.put("config.hold", Array::scalar_f64(h));
        }
        out.put("critic_index", Array::from_i64(&[self.critic_index.len()], &self.critic_index));
        out.put("w_critic", Array::from_f64(&[self.w_critic.len()], &self.w_critic));
        out.put("b_critic", Array::scalar_f64(self.b_critic));
        let g = &self.rng;
        out.put("rng.state", Array::from_u64(&[2], &[(g.state >> 64) as u64, g.state as u64]));
        out.put("rng.inc", Array::from_u64(&[2], &[(g.inc >> 64) as u64, g.inc as u64]));
        out.put("rng.has_uint32", Array::scalar_i64(g.has_uint32 as i64));
        out.put("rng.uinteger", Array::scalar_i64(g.uinteger as i64));
        out.put("updates", Array::scalar_i64(self.updates));
        for (name, value) in [("trace", &self.trace), ("trace_bias", &self.trace_bias),
                              ("trace_critic", &self.trace_critic),
                              ("trace_memory", &self.trace_memory),
                              ("trace_hearing", &self.trace_hearing),
                              ("salience", &self.salience)] {
            if let Some(t) = value {
                out.put(name, t.array());
            }
        }
        if let Some(d) = &self.drive {
            let shape: Vec<usize> = if d.ndim == 2 { vec![d.rows, d.cols] } else { vec![d.cols] };
            out.put("drive", Array::from_f64(&shape, &d.data));
        }
        if let Some(free) = &self.free {
            free.export("free", n, out);
            out.put("free.current",
                    Array::from_bool(&[], &[self.free_generation == Some(graph.generation)]));
        }
        if let Some(p) = &self.pending {
            p.plus.export("pending.plus", n, out);
            p.minus.export("pending.minus", n, out);
            out.put("pending.value", Array::from_f64(&[p.value.len()], &p.value));
        }
        for (name, value) in [("pending_memory", &self.pending_memory),
                              ("pending_hearing", &self.pending_hearing)] {
            if let Some(t) = value {
                out.put(name, t.array());
            }
        }
        if let Some((surprise, entropy)) = &self.pending_entropy {
            out.put("pending_entropy.surprise", Array::from_f64(&[surprise.len()], surprise));
            out.put("pending_entropy.entropy", Array::from_f64(&[entropy.len()], entropy));
        }
    }

    // -- readings

    /// `value(state)`: the critic's reading of each row.
    pub fn value(&self, state: &State) -> Vec<f64> {
        let n = state.v.len().checked_div(state.batch).unwrap_or(0);
        (0..state.batch)
            .map(|b| {
                let part: Vec<f64> =
                    self.critic_index.iter().map(|&i| state.activation[b * n + i as usize]).collect();
                np::dot(&part, &self.w_critic) + self.b_critic
            })
            .collect()
    }

    /// `_temperature(temperature)` of one temperature: validated.
    pub fn temperature(&self, learner: &Learner, temperature: f64) -> Result<f64, Error> {
        match self.heat(learner, Some(&Heat::One(temperature)))? {
            Heat::One(t) => Ok(t),
            Heat::Each(v) => Ok(v[0]),
        }
    }

    /// `_temperature(temperature)`: a scalar or one per motor slot, validated (the learner's
    /// where none is given).
    pub fn heat(&self, learner: &Learner, temperature: Option<&Heat>) -> Result<Heat, Error> {
        match temperature {
            None => Ok(Heat::One(learner.config.temperature)),
            Some(Heat::Each(values)) => {
                if values.len() != learner.slot_count()
                    || values.iter().any(|t| !t.is_finite() || *t <= 0.0)
                {
                    return Err(refused("temperature must be finite and positive, one per motor slot"));
                }
                Ok(Heat::Each(values.clone()))
            }
            Some(Heat::One(t)) => {
                if !t.is_finite() || *t <= 0.0 {
                    return Err(refused("temperature must be finite and positive"));
                }
                Ok(Heat::One(*t))
            }
        }
    }

    /// The outputs of a state, `(batch, outputs)`, with a vote added where one is given.
    fn outputs(state: &State, learner: &Learner, bias: Option<&Rows>) -> Result<Vec<f64>, Error> {
        let n = learner.n();
        let width = learner.output_index.len();
        let mut s: Vec<f64> = (0..state.batch)
            .flat_map(|b| learner.output_index.iter().map(move |&i| (b, i as usize)))
            .map(|(b, i)| state.activation[b * n + i])
            .collect();
        if let Some(bias) = bias {
            if bias.rows != state.batch || bias.cols != width || bias.ndim != 2
                || !bias.data.iter().all(|x| x.is_finite())
            {
                return Err(refused("bias must be a finite (batch, motor) array"));
            }
            for (x, b) in s.iter_mut().zip(&bias.data) {
                *x += b;
            }
        }
        Ok(s)
    }

    /// `probabilities(state, temperature, bias=)`: `(batch, outputs)`, or with several slots
    /// `(batch, slots, largest slot)` padded with zeros. Returns the probabilities and their
    /// shape.
    pub fn probabilities(&self, learner: &Learner, state: &State, temperature: Option<&Heat>,
                         bias: Option<&Rows>) -> Result<(Vec<f64>, Vec<usize>), Error> {
        let temperature = self.heat(learner, temperature)?;
        let s = Self::outputs(state, learner, bias)?;
        let (batch, width) = (state.batch, learner.output_index.len());
        let slots = learner.slot_count();
        if slots > 1 {
            let largest = *learner.slot_sizes.iter().max().unwrap_or(&0) as usize;
            let mut p = vec![0.0f64; batch * slots * largest];
            for j in 0..slots {
                let (start, size) = (learner.slot_offsets[j] as usize,
                                     learner.slot_sizes[j] as usize);
                let t = match &temperature {
                    Heat::Each(v) => v[j],
                    Heat::One(t) => *t,
                };
                let part: Vec<f64> = (0..batch)
                    .flat_map(|b| s[b * width + start..b * width + start + size].to_vec())
                    .collect();
                let q = softmax(&part, size, t);
                for b in 0..batch {
                    let at = (b * slots + j) * largest;
                    p[at..at + size].copy_from_slice(&q[b * size..(b + 1) * size]);
                }
            }
            return Ok((p, vec![batch, slots, largest]));
        }
        let t = match temperature {
            Heat::Each(v) => v[0],
            Heat::One(t) => t,
        };
        Ok((softmax(&s, width, t), vec![batch, width]))
    }

    /// `_validated_drive`: a nonempty finite `(batch, n)` drive.
    fn validated_drive(drive: &Rows, n: usize) -> Result<(), Error> {
        if drive.ndim != 2 || drive.cols != n || drive.rows == 0
            || !drive.data.iter().all(|x| x.is_finite())
        {
            return Err(refused("drive must be a nonempty finite (batch, n) array"));
        }
        Ok(())
    }

    /// `reset()`: no stream state and no eligibility; the learned parameters stay.
    pub fn reset(&mut self) {
        self.free = None;
        self.free_generation = None;
        self.drive = None;
        self.pending = None;
        self.pending_memory = None;
        self.pending_hearing = None;
        self.pending_entropy = None;
        self.trace = None;
        self.trace_bias = None;
        self.trace_critic = None;
        self.trace_memory = None;
        self.trace_hearing = None;
        self.salience = None;
    }

    /// `settle(drive)`: the free phase for `drive`, warm from the last one.
    pub fn settle(&mut self, learner: &Learner, graph: &Graph, drive: &Rows)
                  -> Result<State, Error> {
        self.settle_held(learner, graph, drive)?;
        Ok(self.free.clone().expect("a free state"))
    }

    /// `settle(drive)`, the free phase held as the actor's.
    fn settle_held(&mut self, learner: &Learner, graph: &Graph, drive: &Rows)
                   -> Result<(), Error> {
        Self::validated_drive(drive, learner.n())?;
        if self.free.as_ref().is_some_and(|f| f.batch != drive.rows) {
            self.reset();
        }
        let free = learner.free(graph, drive, self.free.as_ref())?;
        self.free = Some(free);
        self.free_generation = Some(graph.generation);
        self.drive = Some(drive.clone());
        self.pending = None;
        self.pending_memory = None;
        self.pending_entropy = None;
        Ok(())
    }

    /// `_nudged_groups`: from the free state, the outputs pulled toward `target` at `beta`
    /// under the sampling temperature, the apical gain the free state set held.
    #[allow(clippy::too_many_arguments)]
    fn nudged_groups(&self, learner: &Learner, graph: &Graph, drive: &Rows, free: &State,
                     target: &[f64], beta: f64, temperature: Option<&Heat>)
                     -> Result<State, Error> {
        let cfg = &learner.config;
        let n = learner.n();
        let actual = self.heat(learner, temperature)?;
        let least = match &actual {
            Heat::One(t) => *t,
            Heat::Each(v) => v[1..].iter().fold(v[0], |m, &x| np::minimum(m, x)),
        };
        let scale = if least < cfg.temperature { least } else { cfg.temperature };
        let (mask, softmax) = match &actual {
            Heat::One(t) => {
                (learner.output_mask.iter().map(|m| m * (scale / t)).collect::<Vec<f64>>(),
                 Temperature::One(*t))
            }
            Heat::Each(v) => {
                let mut expanded = vec![cfg.temperature; n];
                let mut at = 0usize;
                for (j, &size) in learner.slot_sizes.iter().enumerate() {
                    for _ in 0..size {
                        expanded[learner.output_index[at] as usize] = v[j];
                        at += 1;
                    }
                }
                (learner.output_mask.iter().zip(&expanded).map(|(m, t)| m * (scale / t)).collect(),
                 Temperature::Each(expanded))
            }
        };
        let held = graph.apical_gain(drive, free)?;
        let pools = graph.pools();
        let nudge = Nudge {
            target: Rows { data: target.to_vec(), rows: free.batch, cols: n, ndim: 2 },
            mask,
            beta,
            temperature: Some(softmax),
            weight: None,
            groups: learner.output_groups.clone(),
            anchor: None,
            anchor_gain: None,
            apical_gain: held.map(|g| Rows { data: g, rows: free.batch, cols: pools, ndim: 2 }),
        };
        let steps = self.config.eligibility_steps.unwrap_or(cfg.nudged_steps);
        graph.settle_batch(drive, steps, Some(free), None, Some(&nudge),
                           if cfg.nudged_fixed { None } else { cfg.tolerance })
    }

    /// A moment's act on the free state `free` of `drive`, as `Brain._choose` makes it: the
    /// state the act reads is set first, then `act`. An action per row.
    #[allow(clippy::too_many_arguments)]
    pub fn act(&mut self, learner: &Learner, graph: &Graph, recall: Option<&mut Recall>,
               hearing: Option<&Hearing>, drive: &Rows, free: State, greedy: bool,
               temperature: Option<f64>) -> Result<Vec<i64>, Error> {
        self.free = Some(free);
        self.free_generation = Some(graph.generation);
        self.drive = Some(drive.clone());
        let heat = temperature.map(Heat::One);
        Ok(self.act_held(learner, graph, recall, hearing, drive, greedy, heat.as_ref(), None)?.0)
    }

    /// `act(drive, greedy=, temperature=, bias=)`: an action per row (one per motor slot), and
    /// for a sampled act the decision kept for its outcome; the free state the actor holds is
    /// read where it is the current graph's at this drive, else settled. Returns the actions
    /// and their shape.
    #[allow(clippy::too_many_arguments)]
    pub fn act_held(&mut self, learner: &Learner, graph: &Graph, memory: Option<&mut Recall>,
                    hearing: Option<&Hearing>, drive: &Rows, greedy: bool,
                    temperature: Option<&Heat>, bias: Option<&Rows>)
                    -> Result<(Vec<i64>, Vec<usize>), Error> {
        let n = learner.n();
        Self::validated_drive(drive, n)?;
        let temperature = match temperature {
            Some(t) => Some(self.heat(learner, Some(t))?),
            None => None,
        };
        if let Some(b) = bias {
            if b.ndim != 2 || b.rows != drive.rows || b.cols != learner.output_index.len()
                || !b.data.iter().all(|x| x.is_finite())
            {
                return Err(refused("bias must be a finite (batch, motor) array"));
            }
        }
        let current = self.free.is_some()
            && self.free_generation == Some(graph.generation)
            && self.drive.as_ref().is_some_and(|d| same_rows(d, drive));
        if !current {
            self.settle_held(learner, graph, drive)?;
        }
        // the free state read where it is held, and held again after the act
        let free = self.free.take().expect("a free state");
        let acted = self.act_on(learner, graph, memory, hearing, drive, greedy,
                                temperature.as_ref(), bias, &free);
        self.free = Some(free);
        acted
    }

    #[allow(clippy::too_many_arguments)]
    fn act_on(&mut self, learner: &Learner, graph: &Graph, memory: Option<&mut Recall>,
              hearing: Option<&Hearing>, drive: &Rows, greedy: bool,
              temperature: Option<&Heat>, bias: Option<&Rows>, free: &State)
              -> Result<(Vec<i64>, Vec<usize>), Error> {
        let n = learner.n();
        self.pending = None;
        self.pending_memory = None;
        self.pending_hearing = None;
        self.pending_entropy = None;
        let slots = learner.slot_count();
        let batch = free.batch;
        if greedy && bias.is_none() && slots == 1 {
            let s = Self::outputs(free, learner, None)?;
            let width = learner.output_index.len();
            let mut choice = vec![0i64; batch];
            let clear = unsafe {
                lesson::choose(View { p: s.as_ptr() as *mut f64, n0: batch, n1: width,
                                      len: batch * width },
                               learner.config.temperature,
                               View { p: choice.as_mut_ptr(), n0: batch, n1: 1, len: batch })
            };
            if clear == 1 {
                return Ok((choice, vec![batch]));
            }
        }
        let (p, shape) = self.probabilities(learner, free,
                                            if greedy { None } else { temperature },
                                            bias)?;
        let width = *shape.last().expect("a shape");
        let rows = p.len() / width.max(1);
        let mut action = vec![0i64; rows];
        if greedy {
            for (r, a) in action.iter_mut().enumerate() {
                *a = np::argmax(&p[r * width..(r + 1) * width]) as i64;
            }
        } else {
            let u: Vec<f64> = (0..rows).map(|_| self.rng.random()).collect();
            for (r, a) in action.iter_mut().enumerate() {
                let cdf = np::cumsum(&p[r * width..(r + 1) * width]);
                let below = cdf.iter().filter(|&&c| c < u[r]).count() as i64;
                let limit = if shape.len() == 3 { learner.slot_sizes[r % slots] - 1 }
                            else { width as i64 - 1 };
                *a = below.min(limit);
            }
        }
        let action_shape = if shape.len() == 3 { vec![batch, slots] } else { vec![batch] };
        if !greedy {
            let labels = Labels { data: action.clone(), shape: action_shape.clone() };
            let target = learner.targets(&labels)?;
            let beta = learner.config.beta;
            let plus = self.nudged_groups(learner, graph, drive, free, &target, beta,
                                          temperature)?;
            let minus = self.nudged_groups(learner, graph, drive, free, &target, -beta,
                                           temperature)?;
            let value = self.value(free);
            if self.config.entropy != 0.0 {
                self.pending_entropy = Some(surprise(&p, &shape, &action));
            }
            self.pending_memory = match memory {
                Some(m) => {
                    let rows = m.contrast_rows(
                        graph.weights(),
                        F { shape: vec![plus.batch, n], v: plus.activation.clone() },
                        F { shape: vec![minus.batch, n], v: minus.activation.clone() })?;
                    rows.map(|f| Tensor { shape: f.shape, data: f.v })
                }
                None => None,
            };
            if let Some(h) = hearing {
                let heard = h.contrast_rows(
                    &Columns { rows: plus.batch, cols: n, data: plus.activation.clone() },
                    &Columns { rows: minus.batch, cols: n, data: minus.activation.clone() },
                    beta, &heard_graph(graph))?;
                self.pending_hearing =
                    Some(Tensor { shape: vec![heard.rows, heard.cols], data: heard.data });
            }
            self.pending = Some(Pending { plus, minus, value });
        }
        Ok((action, action_shape))
    }

    // -- learning

    /// `_validated_transition`: a well-formed outcome of the pending decision.
    fn validated_transition(&self, learner: &Learner, pending: &Pending, reward: &[f64],
                            done: &[bool], next_drive: &Rows, bootstrap: Option<&[f64]>)
                            -> Result<(), Error> {
        let batch = pending.value.len();
        if reward.len() != batch || !reward.iter().all(|x| x.is_finite()) {
            return Err(refused("reward must be a finite vector matching the action batch"));
        }
        if done.len() != batch {
            return Err(refused("done must be a boolean vector matching the action batch"));
        }
        Self::validated_drive(next_drive, learner.n())?;
        if next_drive.rows != batch {
            return Err(refused("next_drive must match the action batch"));
        }
        if let Some(b) = bootstrap {
            if b.len() != batch || !b.iter().all(|x| x.is_finite()) {
                return Err(refused("bootstrap must be a finite vector matching the action batch"));
            }
        }
        if let Some(s) = &self.salience {
            if s.shape != [batch, learner.n()] || !s.data.iter().all(|x| x.is_finite()) {
                return Err(refused("salience must be a finite (batch, neurons) array"));
            }
        }
        Ok(())
    }

    /// `learn(reward, done, next_drive, warm=)`: the outcome of the pending decision; `graph`
    /// becomes the graph the step makes. See `learn_with`.
    #[allow(clippy::too_many_arguments)]
    pub fn learn(&mut self, learner: &mut Learner, graph: &mut Graph, recall: Option<&mut Recall>,
                 hearing: Option<&mut Hearing>, reward: &[f64], done: &[bool],
                 next_drive: &Rows, warm: Option<&State>) -> Result<Report, Error> {
        self.learn_with(learner, graph, recall, hearing, reward, done, next_drive, None, None,
                        warm)
    }

    /// `learn(reward, done, next_drive, bootstrap, observed=, warm=)`: the outcome of the pending
    /// decision. A refusal leaves the traces and the count as they were (the eligibility of
    /// this outcome is made apart from the traces and taken only once the outcome stands). With
    /// a maturity its use grows by the synapse step applied. Returns the report.
    #[allow(clippy::too_many_arguments)]
    pub fn learn_with(&mut self, learner: &mut Learner, graph: &mut Graph,
                      memory: Option<&mut Recall>, hearing: Option<&mut Hearing>,
                      reward: &[f64], done: &[bool], next_drive: &Rows,
                      bootstrap: Option<&[f64]>, observed: Option<&[bool]>,
                      warm: Option<&State>) -> Result<Report, Error> {
        let factor = self.maturity.as_ref().map(|m| m.factor());
        let (pending, free) = (self.pending.take(), self.free.take());
        let found = match (&pending, &free) {
            (Some(p), Some(f)) => self.outcome(learner, graph, memory, hearing, factor, p, f,
                                               reward, done, next_drive, bootstrap, observed,
                                               warm),
            (None, _) => Err(refused("learn needs a non-greedy act first")),
            (_, None) => Err(refused("learn needs the free state")),
        };
        let report = match found {
            Ok(r) => r,
            Err(e) => {
                (self.pending, self.free) = (pending, free);
                return Err(e);
            }
        };
        let eta = self.config.eta;
        if let Some(m) = self.maturity.as_mut() {
            if eta > 0.0 {
                // the efficacies the step replaced, against the graph's
                let before = learner.previous_efficacy();
                let room = &mut self.spare.step_scale;
                room.clear();
                room.extend(graph.efficacy.iter().zip(before).map(|(a, b)| (a - b).abs()));
                let grown = m.use_ + np::sum(room) / (eta * graph.synapses() as f64);
                m.use_ = if Maturity::LIMIT < grown { Maturity::LIMIT } else { grown };
            }
        }
        Ok(Report(report))
    }

    #[allow(clippy::too_many_arguments)]
    fn outcome(&mut self, learner: &mut Learner, graph: &mut Graph, memory: Option<&mut Recall>,
               hearing: Option<&mut Hearing>, factor: Option<f64>, pending: &Pending,
               free: &State, reward: &[f64], done: &[bool], next_drive: &Rows,
               bootstrap: Option<&[f64]>, observed: Option<&[bool]>, warm: Option<&State>)
               -> Result<Vec<(&'static str, f64)>, Error> {
        self.validated_transition(learner, pending, reward, done, next_drive, bootstrap)?;
        let batch = reward.len();
        let observed: Vec<bool> = observed.map_or(vec![true; batch], |o| o.to_vec());
        if observed.len() != batch || !observed.iter().any(|&x| x) {
            return Err(refused("observed must match the batch and include a real transition"));
        }
        if warm.is_some_and(|w| w.batch != batch) {
            return Err(refused("warm must be a state of the action batch"));
        }
        let done: Vec<bool> = done.iter().zip(&observed).map(|(&d, &o)| d || !o).collect();
        let cfg = self.config.clone();
        let (first, second, value) = (&pending.plus.activation, &pending.minus.activation,
                                      &pending.value);
        let (n, e) = (learner.n(), graph.synapses());
        let mut memory = memory;
        let decay = cfg.decay();
        let k = self.critic_index.len();
        // this outcome's traces, made in the spare room from the ones held (zeros where none
        // of this batch are held)
        let fresh = self.trace.as_ref().is_none_or(|t| t.rows() != batch);
        let spare = &mut self.spare;
        spare.trace.reshape(&[batch, e]);
        spare.trace_bias.reshape(&[batch, n]);
        spare.trace_critic.reshape(&[batch, k + 1]);
        if fresh {
            spare.trace.data.iter_mut().for_each(|x| *x = 0.0);
            spare.trace_bias.data.iter_mut().for_each(|x| *x = 0.0);
            spare.trace_critic.data.iter_mut().for_each(|x| *x = 0.0);
        } else {
            let held = self.trace_critic.as_ref().expect("a critic trace");
            spare.trace_critic.data.copy_from_slice(&held.data);
        }
        let span = 2.0 * learner.config.beta;
        let fused = self.salience.is_none() && cfg.entropy == 0.0;
        let mut contrast = None;
        if !fused {
            let (pre, post) = (&graph.pre, &graph.post);
            let mut c = vec![0.0f64; batch * e];
            for b in 0..batch {
                for q in 0..e {
                    let (i, j) = (pre[q] as usize, post[q] as usize);
                    let (ap, am) = (first[b * n + i], second[b * n + i]);
                    let (bp, bm) = (first[b * n + j], second[b * n + j]);
                    c[b * e + q] = (ap * (bp - bm) + (ap - am) * bm) / span;
                }
            }
            let c_bias: Vec<f64> = first.iter().zip(second).map(|(p, m)| (p - m) / span).collect();
            if let Some(s) = &self.salience {
                for b in 0..batch {
                    for q in 0..e {
                        c[b * e + q] *= s.data[b * n + pre[q] as usize];
                    }
                }
            }
            for (held, next, add) in [(&self.trace, &mut spare.trace, &c),
                                      (&self.trace_bias, &mut spare.trace_bias, &c_bias)] {
                for (i, (t, x)) in next.data.iter_mut().zip(add.iter()).enumerate() {
                    let before = if fresh { 0.0 } else { held.as_ref().expect("a trace").data[i] };
                    *t = before * decay;
                    *t += x;
                }
            }
            contrast = Some((c, c_bias));
        }
        {
            let critic_decay = cfg.gamma * cfg.lam;
            let tc = &mut spare.trace_critic;
            tc.data.iter_mut().for_each(|x| *x *= critic_decay);
            for b in 0..batch {
                for (j, &i) in self.critic_index.iter().enumerate() {
                    tc.data[b * (k + 1) + j] += free.activation[b * n + i as usize];
                }
                tc.data[b * (k + 1) + k] += 1.0;
            }
        }
        let keeps_memory = eligibility(self.pending_memory.as_ref(), self.trace_memory.as_ref(),
                                       &mut spare.trace_memory, batch, decay);
        let keeps_hearing = eligibility(self.pending_hearing.as_ref(),
                                        self.trace_hearing.as_ref(), &mut spare.trace_hearing,
                                        batch, decay);
        // the next state, warm from this one; a finished row starts from rest
        let next_state = {
            let warm = warm.unwrap_or(free);
            if done.iter().any(|&d| d) {
                let (mut v, mut a) = (warm.v.clone(), warm.adaptation.clone());
                for (b, &d) in done.iter().enumerate() {
                    if d {
                        v[b * n..(b + 1) * n].iter_mut().for_each(|x| *x = 0.0);
                        a[b * n..(b + 1) * n].iter_mut().for_each(|x| *x = 0.0);
                    }
                }
                let rested = State { v, adaptation: a, activity_change: None, published: false,
                                     activation: warm.activation.clone(), ..*warm };
                learner.free(graph, next_drive, Some(&rested))?
            } else {
                learner.free(graph, next_drive, Some(warm))?
            }
        };
        let raw = self.value(&next_state);
        let td_error: Vec<f64> = (0..batch).map(|b| {
            let next = if done[b] { bootstrap.map_or(0.0, |x| x[b]) } else { raw[b] };
            reward[b] + cfg.gamma * next - value[b]
        }).collect();
        if !td_error.iter().all(|x| x.is_finite()) {
            return Err(refused("TD errors must remain finite"));
        }
        let observed_rows: Vec<usize> = (0..batch).filter(|&b| observed[b]).collect();
        let capped = if cfg.dopamine_cap <= 0.0 {
            0.0
        } else {
            let over = observed_rows.iter().filter(|&&b| td_error[b].abs() > cfg.dopamine_cap)
                .count();
            over as f64 / observed_rows.len() as f64
        };
        let share = observed_rows.len() as f64 / batch as f64;
        let delta: Vec<f64> = (0..batch).map(|b| {
            let mut d = td_error[b];
            if cfg.dopamine_cap > 0.0 {
                d = np::clip(d, -cfg.dopamine_cap, cfg.dopamine_cap);
            }
            if observed[b] { d / share } else { 0.0 }
        }).collect();
        let spare = &mut self.spare;
        let (mut step_scale, mut step_bias) =
            (std::mem::take(&mut spare.step_scale), std::mem::take(&mut spare.step_bias));
        if fused {
            step_scale.resize(e, 0.0);
            step_bias.resize(n, 0.0);
            let (next, next_bias) = (&mut spare.trace, &mut spare.trace_bias);
            let code = unsafe {
                let wide = |x: &[i64]| Index::Wide(View { p: x.as_ptr() as *mut i64, n0: x.len(),
                                                           n1: 1, len: x.len() });
                let seen = |x: &[f64], n0: usize, n1: usize| View { p: x.as_ptr() as *mut f64,
                                                                     n0, n1, len: n0 * n1 };
                let out = View { p: next.data.as_mut_ptr(), n0: batch, n1: e, len: batch * e };
                let out_bias = View { p: next_bias.data.as_mut_ptr(), n0: batch, n1: n,
                                      len: batch * n };
                // the held traces in, or (none of this batch held) the zeros of the room
                let (held, held_bias) = match (&self.trace, &self.trace_bias, fresh) {
                    (Some(t), Some(tb), false) => (seen(&t.data, batch, e), seen(&tb.data, batch, n)),
                    _ => (out, out_bias),
                };
                lesson::trace_pass(
                    held, held_bias, out, out_bias, decay, seen(first, batch, n),
                    seen(second, batch, n), wide(&graph.pre), wide(&graph.post), span,
                    seen(&delta, batch, 1),
                    View { p: step_scale.as_mut_ptr(), n0: e, n1: 1, len: e },
                    View { p: step_bias.as_mut_ptr(), n0: n, n1: 1, len: n })
            };
            if code < 0 {
                (spare.step_scale, spare.step_bias) = (step_scale, step_bias);
                return Err(Error::Shape("the trace pass refused its arrays".into()));
            }
        } else {
            step_scale = spare.trace.weighted_mean(&delta);
            step_bias = spare.trace_bias.weighted_mean(&delta);
        }
        let mut explored = None;
        if cfg.entropy != 0.0 {
            let Some((surprise, entropy)) =
                self.pending_entropy.clone().filter(|s| s.0.len() == batch)
            else {
                (self.spare.step_scale, self.spare.step_bias) = (step_scale, step_bias);
                return Err(refused("actor entropy needs the sampled act's policy; act again"));
            };
            let (c, c_bias) = contrast.expect("the contrast of an entropy step");
            let weight: Vec<f64> = (0..batch)
                .map(|b| if observed[b] { cfg.entropy * surprise[b] / share } else { 0.0 })
                .collect();
            let add_s = Tensor { shape: vec![batch, e], data: c }.weighted_mean(&weight);
            let add_b = Tensor { shape: vec![batch, n], data: c_bias }.weighted_mean(&weight);
            step_scale = step_scale.iter().zip(&add_s).map(|(x, y)| x + y).collect();
            step_bias = step_bias.iter().zip(&add_b).map(|(x, y)| x + y).collect();
            let seen: Vec<f64> = observed_rows.iter().map(|&b| entropy[b]).collect();
            explored = Some(np::mean(&seen));
        }
        let f = factor.unwrap_or(1.0);
        let (eta, eta_bias) = (f * cfg.eta, f * cfg.eta_bias);
        step_scale.iter_mut().for_each(|x| *x = eta * *x);
        step_bias.iter_mut().for_each(|x| *x = eta_bias * *x);
        let critic = self.critic_candidate(&self.spare.trace_critic, &delta);
        let proposal = match (&critic, keeps_memory, memory.as_deref_mut()) {
            (Ok(_), true, Some(m)) => {
                let trace = &self.spare.trace_memory;
                let mean = trace.weighted_mean(&delta);
                m.proposal(F { shape: trace.shape[1..].to_vec(), v: mean })
            }
            _ => Ok(None),
        };
        let next_generation = graph.generation; // the graph the next state settled on
        let applied = match (critic, proposal) {
            (Ok(c), Ok(p)) => learner.apply(graph, &step_scale, &step_bias).map(|r| (c, p, r)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        (self.spare.step_scale, self.spare.step_bias) = (step_scale, step_bias);
        let ((critic_weights, critic_bias), proposal, mut report) = applied?;
        self.w_critic = critic_weights;
        self.b_critic = critic_bias;
        if let (Some(p), Some(m)) = (proposal, memory.as_deref_mut()) {
            let norm = p.norm;
            m.adopt(p);
            report.push(("address_reward_step", norm));
        }
        if let (true, Some(h)) = (keeps_hearing, hearing) {
            h.step(&self.spare.trace_hearing.weighted_mean(&delta))?;
            // `Hearing.set`: the recall memory's own gain is the one its route holds
            if let Some(m) = memory {
                for &k in &self.hearing_routes {
                    if let Some(&gain) = h.gains.get(k as usize) {
                        m.hearing = gain;
                    }
                }
            }
        }
        // the outcome stands: its traces are the actor's, the ones they replace the room's
        let spare = &mut self.spare;
        for (held, next, kept) in [(&mut self.trace, &mut spare.trace, true),
                                   (&mut self.trace_bias, &mut spare.trace_bias, true),
                                   (&mut self.trace_critic, &mut spare.trace_critic, true),
                                   (&mut self.trace_memory, &mut spare.trace_memory, keeps_memory),
                                   (&mut self.trace_hearing, &mut spare.trace_hearing,
                                    keeps_hearing)] {
            if kept {
                let old = held.replace(std::mem::take(next));
                *next = old.unwrap_or_else(|| Tensor::zeros(&[]));
            } else if let Some(old) = held.take() {
                *next = old;
            }
        }
        if done.iter().any(|&d| d) {
            for t in [&mut self.trace, &mut self.trace_bias, &mut self.trace_critic,
                      &mut self.trace_memory, &mut self.trace_hearing].into_iter().flatten() {
                t.forget(&done);
            }
        }
        self.updates += 1;
        let next_steps = next_state.steps;
        self.free = Some(next_state);
        self.free_generation = Some(next_generation);
        self.drive = Some(next_drive.clone());
        self.pending_memory = None;
        self.pending_hearing = None;
        self.pending_entropy = None;
        let abs: Vec<f64> = delta.iter().map(|x| x.abs()).collect();
        report.push(("delta", np::mean(&abs)));
        report.push(("dopamine", np::mean(&delta)));
        let seen_td: Vec<f64> = observed_rows.iter().map(|&b| td_error[b].abs()).collect();
        report.push(("td_error", np::mean(&seen_td)));
        report.push(("capped", capped));
        let seen_value: Vec<f64> = observed_rows.iter().map(|&b| value[b]).collect();
        report.push(("value", np::mean(&seen_value)));
        report.push(("free_steps", next_steps as f64));
        report.push(("saturation", saturation(free, learner)));
        if let Some(x) = explored {
            report.push(("policy_entropy", x));
        }
        let trace = self.trace.as_ref().expect("a trace");
        let plastic = &learner.plastic_synapses;
        let traced = if plastic.iter().all(|&x| x) {
            mean_size(&trace.data)
        } else {
            let sizes: Vec<f64> = (0..batch)
                .flat_map(|b| (0..e).filter(|&q| plastic[q]).map(move |q| (b, q)))
                .map(|(b, q)| trace.data[b * e + q].abs()).collect();
            np::mean(&sizes)
        };
        report.push(("trace", traced));
        Ok(report)
    }

    /// `_critic_candidate(delta)`: the critic's weights and bias after the dopamine's step on
    /// its trace, divided by the trace's energy.
    fn critic_candidate(&self, tc: &Tensor, delta: &[f64]) -> Result<(Vec<f64>, f64), Error> {
        let width = tc.width();
        let mut scaled = tc.clone();
        for row in scaled.data.chunks_mut(width) {
            let energy = 1.0 + np::sum_squares(row);
            row.iter_mut().for_each(|x| *x /= energy);
        }
        let step: Vec<f64> = scaled.weighted_mean(delta).iter()
            .map(|x| self.config.eta_critic * x).collect();
        let weights: Vec<f64> = self.w_critic.iter().zip(&step).map(|(w, s)| w + s).collect();
        let bias = self.b_critic + step[width - 1];
        if !weights.iter().all(|x| x.is_finite()) || !bias.is_finite() {
            return Err(refused("critic parameters must remain finite"));
        }
        Ok((weights, bias))
    }

    /// `fade(done)`: a moment that adds no eligibility: every trace decays, the `done` rows
    /// forget theirs.
    pub fn fade(&mut self, done: &[bool]) -> Result<(), Error> {
        self.fade_rows(Some(done))
    }

    /// `fade(done=None)` or `fade(done)`.
    pub fn fade_rows(&mut self, done: Option<&[bool]>) -> Result<(), Error> {
        let decay = self.config.decay();
        let critic_decay = self.config.gamma * self.config.lam;
        if let Some(d) = done {
            let rows = [&self.trace, &self.trace_bias, &self.trace_memory, &self.trace_hearing,
                        &self.trace_critic];
            if rows.iter().any(|t| t.as_ref().is_some_and(|t| t.rows() != d.len())) {
                return Err(refused("done must match the streams of the eligibility traces"));
            }
        }
        let ended = done.is_some_and(|d| d.iter().any(|&x| x));
        for (t, rate) in [(&mut self.trace, decay), (&mut self.trace_bias, decay),
                          (&mut self.trace_memory, decay), (&mut self.trace_hearing, decay),
                          (&mut self.trace_critic, critic_decay)] {
            if let Some(t) = t {
                t.data.iter_mut().for_each(|x| *x *= rate);
                if ended {
                    t.forget(done.expect("rows that ended"));
                }
            }
        }
        Ok(())
    }
}

/// What a carried route's contrast reads of the learner's graph (`hearing._graph`).
fn heard_graph(graph: &Graph) -> Heard<'_> {
    Heard { weights: std::borrow::Cow::Borrowed(graph.weights()), efficacy: true,
            parallel: graph.parallel_synapses(), device: false }
}

/// `_memory_eligibility` and `_hearing_eligibility`: `e <- decay e + contrast`; a decision that
/// kept no contrast, or one of another batch, drops the trace.
/// the trace after this outcome into `out`, from the one `held`; false where it is dropped.
fn eligibility(contrast: Option<&Tensor>, held: Option<&Tensor>, out: &mut Tensor, batch: usize,
               decay: f64) -> bool {
    let Some(contrast) = contrast else { return false };
    if contrast.rows() != batch {
        return false;
    }
    out.reshape(&contrast.shape);
    match held {
        Some(h) if h.shape == contrast.shape => {
            for ((t, c), &x) in out.data.iter_mut().zip(&contrast.data).zip(&h.data) {
                *t = x * decay;
                *t += c;
            }
        }
        _ => {
            for (t, c) in out.data.iter_mut().zip(&contrast.data) {
                *t = 0.0 * decay;
                *t += c;
            }
        }
    }
    true
}

/// `np.abs(x).mean()` without the array of sizes: the sizes summed in the pairwise order
/// `np::sum` sums them in, a run of at most 128 at a time.
fn mean_size(x: &[f64]) -> f64 {
    fn pairwise(x: &[f64]) -> f64 {
        if x.len() <= 128 {
            let mut run = [0.0f64; 128];
            for (r, v) in run.iter_mut().zip(x) {
                *r = v.abs();
            }
            return np::pairwise(&run[..x.len()]);
        }
        let mut half = x.len() / 2;
        half -= half % 8;
        pairwise(&x[..half]) + pairwise(&x[half..])
    }
    (0.0 + pairwise(x)) / x.len() as f64
}

/// The rooms an outcome reuses, which hold no state: the traces it makes before they are the
/// actor's (and the ones they replace after), and the steps. A copy starts without them, and
/// any two compare equal.
#[derive(Default)]
pub struct Spare {
    trace: Tensor,
    trace_bias: Tensor,
    trace_critic: Tensor,
    trace_memory: Tensor,
    trace_hearing: Tensor,
    step_scale: Vec<f64>,
    step_bias: Vec<f64>,
}

impl Clone for Spare {
    fn clone(&self) -> Spare {
        Spare::default()
    }
}

impl PartialEq for Spare {
    fn eq(&self, _: &Spare) -> bool {
        true
    }
}

impl std::fmt::Debug for Spare {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Spare")
    }
}

/// `_saturation(free)`: the share of output activations within the band of 0 or 1.
fn saturation(free: &State, learner: &Learner) -> f64 {
    let n = learner.n();
    let values: Vec<f64> = (0..free.batch)
        .flat_map(|b| learner.output_index.iter()
                  .map(move |&i| free.activation[b * n + i as usize]))
        .map(|s| {
            // NumPy's `(s < band) | (s > 1 - band)`: a NaN is in neither
            let near = s < SATURATION_BAND;
            let far = s > 1.0 - SATURATION_BAND;
            if near || far { 1.0 } else { 0.0 }
        })
        .collect();
    np::mean(&values)
}

/// `_surprise(p, action)`: each row's `-log p(action) - H` and `H`, summed over the slots.
fn surprise(p: &[f64], shape: &[usize], action: &[i64]) -> (Vec<f64>, Vec<f64>) {
    let batch = shape[0];
    let (slots, width) = if shape.len() == 3 { (shape[1], shape[2]) } else { (1, shape[1]) };
    let (mut out, mut held) = (vec![0.0f64; batch], vec![0.0f64; batch]);
    for b in 0..batch {
        let (mut picked, mut entropy) = (vec![0.0f64; slots], vec![0.0f64; slots]);
        for j in 0..slots {
            let row = &p[(b * slots + j) * width..(b * slots + j + 1) * width];
            let logp: Vec<f64> = row.iter().map(|&x| if x > 0.0 { np::log(x) } else { 0.0 }).collect();
            let terms: Vec<f64> = row.iter().zip(&logp).map(|(x, l)| x * l).collect();
            entropy[j] = -np::sum(&terms);
            picked[j] = logp[action[b * slots + j] as usize];
        }
        let each: Vec<f64> = picked.iter().zip(&entropy).map(|(p, h)| -p - h).collect();
        out[b] = np::sum(&each);
        held[b] = np::sum(&entropy);
    }
    (out, held)
}

/// The actor with the parts its moments reach, as its state crosses (`cadence.core.parts.actor`):
/// the learner and its graph, the recall memory and the carried gains, each where it holds one.
#[derive(Clone, Debug, PartialEq)]
pub struct Reach {
    pub actor: ActorCritic,
    pub learner: Learner,
    pub graph: Graph,
    pub memory: Option<Recall>,
    pub hearing: Option<Hearing>,
}

fn heat(args: &Named) -> Result<Option<Heat>, Error> {
    if !args.has("temperature") {
        return Ok(None);
    }
    let values = args.f64s("temperature")?;
    let each = args.has("temperature.each") && args.bools("temperature.each")?[0];
    Ok(Some(if each { Heat::Each(values) } else { Heat::One(values[0]) }))
}

fn flags(args: &Named, name: &str) -> Result<Option<Vec<bool>>, Error> {
    if !args.has(name) {
        return Ok(None);
    }
    args.bools(name).map(Some).map_err(|_| refused(&format!("{name} must be a boolean vector")))
}

impl Reach {
    pub fn import(s: &Named) -> Result<Reach, Error> {
        let held = s.under("learner");
        let graph = Graph::import(&held.under("graph"))?;
        let learner = Learner::import(&held, &graph)?;
        let memory = if s.holds("memory") { Some(Recall::import(&s.under("memory"))?) } else { None };
        let hearing =
            if s.holds("hearing") { Some(Hearing::import(&s.under("hearing"))?) } else { None };
        let actor = ActorCritic::import(s, &learner, &graph)?;
        Ok(Reach { actor, learner, graph, memory, hearing })
    }

    pub fn export(&self, out: &mut Named) {
        let mut part = Named::new();
        let mut graph = Named::new();
        self.graph.export(&mut graph);
        part.put_under("graph", &graph);
        self.learner.export(&mut part);
        out.put_under("learner", &part);
        if let Some(m) = &self.memory {
            let mut part = Named::new();
            m.export(&mut part);
            out.put_under("memory", &part);
        }
        if let Some(h) = &self.hearing {
            let mut part = Named::new();
            h.export(&mut part);
            out.put_under("hearing", &part);
            if self.memory.is_some() {
                let routes = &self.actor.hearing_routes;
                out.put("hearing_routes", Array::from_i64(&[routes.len()], routes));
            }
        }
        self.actor.export(&self.learner, &self.graph, out);
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let n = self.learner.n();
        match method {
            "act" => {
                let drive = Rows::of(args, "drive")?;
                let bias = if args.has("bias") { Some(Rows::of(args, "bias")?) } else { None };
                let (action, shape) = self.actor.act_held(
                    &self.learner, &self.graph, self.memory.as_mut(), self.hearing.as_ref(),
                    &drive, args.bools("greedy")?[0], heat(args)?.as_ref(), bias.as_ref())?;
                out.put("action", Array::from_i64(&shape, &action));
            }
            "learn" => {
                let done = flags(args, "done")?.ok_or_else(|| refused("done is needed"))?;
                let bootstrap = if args.has("bootstrap") { Some(args.f64s("bootstrap")?) }
                                else { None };
                let warm = if args.has("warm.v") { Some(State::import(args, "warm", n)?) }
                           else { None };
                let report = self.actor.learn_with(
                    &mut self.learner, &mut self.graph, self.memory.as_mut(),
                    self.hearing.as_mut(), &args.f64s("reward")?, &done,
                    &Rows::of(args, "next_drive")?, bootstrap.as_deref(),
                    flags(args, "observed")?.as_deref(), warm.as_ref())?;
                for (name, value) in report.0 {
                    out.put(&format!("report.{name}"), Array::scalar_f64(value));
                }
            }
            "fade" => self.actor.fade_rows(flags(args, "done")?.as_deref())?,
            "probabilities" => {
                let state = State::import(args, "state", n)?;
                let bias = if args.has("bias") { Some(Rows::of(args, "bias")?) } else { None };
                let (p, shape) = self.actor.probabilities(&self.learner, &state,
                                                          heat(args)?.as_ref(), bias.as_ref())?;
                out.put("p", Array::from_f64(&shape, &p));
            }
            "value" => {
                let value = self.actor.value(&State::import(args, "state", n)?);
                out.put("value", Array::from_f64(&[value.len()], &value));
            }
            "state" => {} // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
