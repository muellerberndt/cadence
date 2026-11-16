//! A composed brain (`cadence.generic.Brain`): its parts, each under its attribute's name, the
//! senses it declares, and the moment that runs them in the Python's order (`live`, `act`,
//! `say`, with `generic.py`'s glue: `_observations`, `_code`, `_equal_senses`, `_bound_keyed`,
//! `_bound_complete`, `_stimulus` through the memories in `Memories.ORDER`, `_qualified`,
//! `_equilibrate`, `_read_surprise`, `_residual_read`, `_surprised`, `_bound_named`,
//! `_choose`, `_hear_episode`, `_bound_hear`, `_moment_surprise`, `_record`).
//!
//! A live moment is one stream: one row of the brain's whole width (`inputs`), a silent sense
//! all zeros, as the Python builds the row from the senses it is given by name. The parts are
//! the core's own (`parts::*`); the state crosses as `cadence.core.parts.brain` writes it.

use super::actor::{ActorCritic, Heat, Report};
use super::arousal::Arousal;
use super::bound::Bound;
use super::code::Code;
use super::graph::{self, Equilibrium, Graph, State};
use super::hearing::Hearing;
use super::history::History;
use super::learner::Learner;
use super::map::Map;
use super::memories::{self, ORDER};
use super::recall::{Recall, F};
use super::surprise::{Orienting, Surprise};
use super::synaptic::Synaptic;
use super::trace::{Rows, Trace};
use crate::np;
use crate::state::{refused, Array, Error, Named};

/// `arousal.MODES`: the mode a moment was answered in.
pub const MODES: [&str; 2] = ["routine", "aroused"];
/// The operations a settle's report names (`last_settlement["operation"]`).
const OPERATIONS: [&str; 4] = ["act", "wait", "predict", "answer"];
/// `Brain.equilibrate`'s chunk of sweeps between checks, as the moment's solves take it.
const CHUNK: i64 = 32;
/// The attributes a brain's parts cross under, in their order.
pub const PARTS: [&str; 15] = [
    "arousal", "sparse_code", "working_memory", "efference", "history", "recall", "hippocampus",
    "bound", "surprise", "orienting", "space", "brain", "hearing", "learner", "basal_ganglia",
];
/// `Orienting`'s senses (`surprise::SENSES`) as `_surprised` names its sources.
const FROM_MAP: i64 = 2;
const FROM_NAMED: i64 = 3;

fn fault(text: &str) -> Error {
    Error::Shape(text.to_string())
}

/// A sense by name: its columns of the whole row, and its fields' columns where it has fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Sense {
    pub name: String,
    pub start: usize,
    pub stop: usize,
    pub fields: Vec<(String, usize, usize)>,
}

impl Sense {
    /// Its columns as the moment reads them: each field's, or the sense's own.
    fn spans(&self) -> Vec<(usize, usize)> {
        if self.fields.is_empty() {
            vec![(self.start, self.stop)]
        } else {
            self.fields.iter().map(|(_, a, b)| (*a, *b)).collect()
        }
    }
}

/// The learner's settings a moment's settle reads.
struct Settle {
    free_steps: i64,
    tolerance: f64,
    damping: i64,
    qualified: bool,
    fast: bool,
}

/// The action `live` issued that awaits its outcome (`Brain._lived`, while its free state is
/// the actor's).
#[derive(Clone, Debug, PartialEq)]
pub struct Lived {
    pub x: Vec<f64>,
    pub action: Vec<i64>,
    /// The critic's value of the state it was chosen in: the forecast its outcome meets.
    pub value: f64,
    pub aroused: bool,
    /// Whether it was the brain's own best guess.
    pub own: bool,
}

/// `Brain.last_arousal`: the readings of the latest `live` moment.
#[derive(Clone, Debug, PartialEq)]
pub struct LastArousal {
    pub mode: &'static str,
    pub level: f64,
    pub error: f64,
    pub surprise: f64,
    pub want: f64,
    pub temperature: Option<f64>,
    pub temperatures: Option<Vec<f64>>,
    pub heated_slot: Option<i64>,
    pub learned: bool,
    pub recorded: bool,
    pub sweeps: i64,
    pub learning_sweeps: i64,
}

/// `Brain.last_settlement`: the latest free answer's report.
#[derive(Clone, Debug, PartialEq)]
pub struct Settlement {
    pub operation: i64,
    pub qualified: Vec<bool>,
    pub residual: Vec<f64>,
    pub steps: i64,
    pub budget: i64,
    pub residual_checks: i64,
    pub damping_halvings: i64,
    pub stagnation_checks: i64,
    pub tolerance: f64,
    pub rescue_steps: Option<i64>,
    pub row_sweeps: Option<i64>,
}

/// `Brain._surprise_read`: the observations a settle read its residual at, each unit's size and
/// whether each stream carried a state into the moment.
#[derive(Clone, Debug, PartialEq)]
pub struct SurpriseRead {
    pub x: Vec<f64>,
    pub sizes: Vec<f64>,
    pub rows: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct Brain {
    pub senses: Vec<Sense>,
    pub inputs: usize,
    /// The neurons the code (or the observations) drive, and the motor neurons.
    pub sensory: Vec<i64>,
    pub motor: Vec<i64>,
    pub coded_cortex: bool,
    pub coded_keys: bool,
    pub surprise_private: bool,
    /// Whether the place records read the maturity's factor.
    pub hippocampus_maturity: bool,
    /// The map's value groups by name, in their order.
    pub map_values: Vec<String>,

    pub arousal: Option<Arousal>,
    pub sparse_code: Option<Code>,
    pub working_memory: Option<Trace>,
    pub efference: Option<Trace>,
    pub history: Option<History>,
    pub recall: Option<Recall>,
    pub hippocampus: Option<Synaptic>,
    pub bound: Option<Bound>,
    pub surprise: Option<Surprise>,
    pub orienting: Option<Orienting>,
    pub space: Option<Map>,
    /// The neural graph (`Brain.brain`, the learner's).
    pub brain: Option<Graph>,
    pub hearing: Option<Hearing>,
    /// Each of the hearing's memories by its place in `Memories.ORDER`.
    pub hearing_memories: Vec<usize>,
    pub learner: Option<Learner>,
    /// The actor and critic, with the maturity (`basal_ganglia.maturity`).
    pub basal_ganglia: Option<ActorCritic>,

    pub lived: Option<Lived>,
    /// A sampled moment's observations and action, for the outcome's record.
    pub moment: Option<(Vec<f64>, Vec<i64>)>,
    pub prepared: Option<Vec<f64>>,
    pub surprise_read: Option<SurpriseRead>,
    pub surprise_seen: Option<Vec<f64>>,
    pub episode_told: Vec<i64>,
    pub last_arousal: Option<LastArousal>,
    pub last_settlement: Option<Settlement>,

    /// The moment's own: the observations a settle was prepared for and their code
    /// (`_coded_moment`), and the bound records' completed row with the row given
    /// (`_bound_given`).
    coded: Option<(Vec<f64>, Vec<f64>)>,
    given: Option<(Vec<f64>, Vec<f64>)>,
}

fn flag(s: &Named, name: &str) -> Result<bool, Error> {
    match s.bools(name)?.as_slice() {
        [x] => Ok(*x),
        _ => Err(fault(name)),
    }
}

/// The two columns `[start, stop]` of a span.
fn span(s: &Named, name: &str) -> Result<(usize, usize), Error> {
    match s.i64s(name)?.as_slice() {
        [a, b] if 0 <= *a && a <= b => Ok((*a as usize, *b as usize)),
        _ => Err(fault(name)),
    }
}

/// `_unit_rows` of one row: over its norm (`np.linalg.norm(axis=1)`, the root of its squares'
/// sum), where that is above zero.
fn unit_row(x: &[f64]) -> Vec<f64> {
    let norm = np::sqrt(np::sum_squares(x));
    let by = if norm > 0.0 { norm } else { 1.0 };
    x.iter().map(|v| v / by).collect()
}

fn same(held: &Option<Vec<f64>>, x: &[f64]) -> bool {
    held.as_ref().is_some_and(|h| h.len() == x.len() && h.iter().zip(x).all(|(a, b)| a == b))
}

fn one(x: &[f64]) -> Rows {
    Rows { rows: 1, cols: x.len(), data: x.to_vec() }
}

fn graph_rows(x: &[f64]) -> graph::Rows {
    graph::Rows { data: x.to_vec(), rows: 1, cols: x.len(), ndim: 2 }
}

fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

fn state_finite(s: &State) -> bool {
    all_finite(&s.v) && all_finite(&s.activation) && all_finite(&s.adaptation)
}

/// `memories.command`: the `(1, motor)` one-hot of an issued action, one per slot.
fn command(action: &[i64], motor: usize, offsets: &[i64]) -> Result<Vec<f64>, Error> {
    let mut out = vec![0.0; motor];
    for (s, &a) in action.iter().enumerate() {
        let c = a + offsets.get(s).or(offsets.first()).copied().unwrap_or(0);
        let c = if c < 0 { c + motor as i64 } else { c };
        if c < 0 || c as usize >= motor {
            return Err(fault("an action outside the motor neurons"));
        }
        out[c as usize] = 1.0;
    }
    Ok(out)
}

impl Brain {
    // -- the state

    pub fn import(s: &Named) -> Result<Brain, Error> {
        let mut senses: Vec<Sense> = Vec::new();
        for name in &s.order {
            let Some(rest) = name.strip_prefix("senses.") else { continue };
            let (a, b) = span(s, name)?;
            match rest.split_once('/') {
                Some((sense, field)) => {
                    let last = senses.last_mut().filter(|x| x.name == sense)
                        .ok_or_else(|| fault(name))?;
                    last.fields.push((field.to_string(), a, b));
                }
                None => senses.push(Sense { name: rest.to_string(), start: a, stop: b,
                                            fields: Vec::new() }),
            }
        }
        for name in &s.order {
            let head = name.split('.').next().unwrap_or("");
            if !(head == "senses" || head == "spine" || PARTS.contains(&head)) {
                return Err(Error::Missing(format!("the core holds no part {head}")));
            }
        }
        let part = |attribute: &str| -> Option<Named> {
            s.holds(attribute).then(|| s.under(attribute))
        };
        let mut map_values: Vec<(i64, String)> = Vec::new();
        for name in &s.order {
            if let Some(group) = name.strip_prefix("spine.map_value.") {
                map_values.push((s.i64(name)?, group.to_string()));
            }
        }
        map_values.sort();
        let mut brain = Brain {
            senses,
            inputs: s.i64("spine.inputs")? as usize,
            sensory: s.i64s("spine.sensory")?,
            motor: s.i64s("spine.motor")?,
            coded_cortex: flag(s, "spine.coded_cortex")?,
            coded_keys: flag(s, "spine.coded_keys")?,
            surprise_private: flag(s, "spine.surprise_private")?,
            hippocampus_maturity: flag(s, "spine.hippocampus_maturity")?,
            map_values: map_values.into_iter().map(|(_, n)| n).collect(),
            arousal: part("arousal").map(|p| Arousal::import(&p)).transpose()?,
            sparse_code: part("sparse_code").map(|p| Code::import(&p)).transpose()?,
            working_memory: part("working_memory").map(|p| Trace::import(&p, false)).transpose()?,
            efference: part("efference").map(|p| Trace::import(&p, true)).transpose()?,
            history: part("history").map(|p| History::import(&p)).transpose()?,
            recall: part("recall").map(|p| Recall::import(&p)).transpose()?,
            hippocampus: part("hippocampus").map(|p| Synaptic::import(&p)).transpose()?,
            bound: part("bound").map(|p| Bound::import(&p)).transpose()?,
            surprise: part("surprise").map(|p| Surprise::import(&p)).transpose()?,
            orienting: part("orienting").map(|p| Orienting::import(&p)).transpose()?,
            space: part("space").map(|p| Map::import(&p)).transpose()?,
            brain: part("brain").map(|p| Graph::import(&p)).transpose()?,
            hearing: None,
            hearing_memories: Vec::new(),
            learner: None,
            basal_ganglia: None,
            lived: None,
            moment: None,
            prepared: None,
            surprise_read: None,
            surprise_seen: None,
            episode_told: s.i64s("spine.episode_told")?,
            last_arousal: None,
            last_settlement: None,
            coded: None,
            given: None,
        };
        if brain.sparse_code.is_none() || brain.brain.is_none() || brain.arousal.is_none() {
            return Err(fault("the core holds a brain with a sparse code, a graph and arousal"));
        }
        if let Some(h) = part("hearing") {
            brain.import_hearing(&h)?;
        }
        let graph = brain.brain.as_ref().unwrap();
        let learner = Learner::import(&part("learner").ok_or_else(|| fault("no learner"))?, graph)?;
        brain.basal_ganglia = Some(ActorCritic::import(
            &part("basal_ganglia").ok_or_else(|| fault("no actor"))?, &learner, graph)?);
        brain.learner = Some(learner);
        if s.has("spine.lived.x") {
            brain.lived = Some(Lived {
                x: s.f64s("spine.lived.x")?,
                action: s.i64s("spine.lived.action")?,
                value: s.f64("spine.lived.value")?,
                aroused: flag(s, "spine.lived.aroused")?,
                own: flag(s, "spine.lived.own")?,
            });
        }
        if s.has("spine.moment.x") {
            brain.moment = Some((s.f64s("spine.moment.x")?, s.i64s("spine.moment.action")?));
        }
        if s.has("spine.prepared") {
            brain.prepared = Some(s.f64s("spine.prepared")?);
        }
        if s.has("spine.surprise_read.x") {
            brain.surprise_read = Some(SurpriseRead {
                x: s.f64s("spine.surprise_read.x")?,
                sizes: s.f64s("spine.surprise_read.sizes")?,
                rows: s.bools("spine.surprise_read.rows")?,
            });
        }
        if s.has("spine.surprise_seen") {
            brain.surprise_seen = Some(s.f64s("spine.surprise_seen")?);
        }
        if s.has("spine.last_arousal.mode") {
            let a = s.under("spine.last_arousal");
            let mode = a.i64("mode")?;
            brain.last_arousal = Some(LastArousal {
                mode: *MODES.get(mode as usize).ok_or_else(|| fault("mode"))?,
                level: a.f64("level")?,
                error: a.f64("error")?,
                surprise: a.f64("surprise")?,
                want: a.f64("want")?,
                temperature: if a.has("temperature") { Some(a.f64("temperature")?) } else { None },
                temperatures: if a.has("temperatures") { Some(a.f64s("temperatures")?) } else { None },
                heated_slot: if a.has("heated_slot") { Some(a.i64("heated_slot")?) } else { None },
                learned: flag(&a, "learned")?,
                recorded: flag(&a, "recorded")?,
                sweeps: a.i64("sweeps")?,
                learning_sweeps: a.i64("learning_sweeps")?,
            });
        }
        if s.has("spine.last_settlement.operation") {
            let r = s.under("spine.last_settlement");
            brain.last_settlement = Some(Settlement {
                operation: r.i64("operation")?,
                qualified: r.bools("qualified")?,
                residual: r.f64s("residual")?,
                steps: r.i64("steps")?,
                budget: r.i64("budget")?,
                residual_checks: r.i64("residual_checks")?,
                damping_halvings: r.i64("damping_halvings")?,
                stagnation_checks: r.i64("stagnation_checks")?,
                tolerance: r.f64("tolerance")?,
                rescue_steps: if r.has("rescue_steps") { Some(r.i64("rescue_steps")?) } else { None },
                row_sweeps: if r.has("row_sweeps") { Some(r.i64("row_sweeps")?) } else { None },
            });
        }
        Ok(brain)
    }

    /// The hearing from its state and its memories': the gains and the traces they hold.
    fn import_hearing(&mut self, h: &Named) -> Result<(), Error> {
        let codes = h.i64s("memory_of")?;
        let mut memories = Vec::with_capacity(codes.len());
        for &c in &codes {
            if c < 0 || c as usize >= ORDER.len() {
                return Err(fault("memory_of"));
            }
            memories.push(c as usize);
        }
        let mut whole = Named::new();
        for name in &h.order {
            if name != "memory_of" {
                whole.put(name, h.arrays[name].clone());
            }
        }
        let routes = h.i64("routes")?.max(0) as usize;
        let mut gains = Vec::with_capacity(routes);
        for k in 0..routes {
            let m = h.i64(&format!("route.{k}.memory"))?;
            let at = *memories.get(m.max(0) as usize).ok_or_else(|| fault("route memory"))?;
            gains.push(self.gain_of(at, h.i64(&format!("route.{k}.slot"))?)?);
        }
        whole.put("gains", Array::from_f64(&[gains.len()], &gains));
        for (m, &at) in memories.iter().enumerate() {
            let (trace, amplitude) = self.trace_of(at)?;
            whole.put(&format!("memory.{m}.trace"), trace.array());
            whole.put(&format!("memory.{m}.amplitude"), Array::scalar_f64(amplitude));
        }
        self.hearing = Some(Hearing::import(&whole)?);
        self.hearing_memories = memories;
        Ok(())
    }

    /// The gain a memory carries for a route (its `hearing`, or its `slot`'s).
    fn gain_of(&self, memory: usize, slot: i64) -> Result<f64, Error> {
        let missing = || fault("a route's memory is not the brain's");
        match ORDER[memory] {
            "working_memory" => Ok(self.working_memory.as_ref().ok_or_else(missing)?.hearing),
            "efference" => Ok(self.efference.as_ref().ok_or_else(missing)?.hearing),
            "recall" => Ok(self.recall.as_ref().ok_or_else(missing)?.hearing),
            "history" => {
                let held = self.history.as_ref().ok_or_else(missing)?;
                let (shape, gains) = held.hearing.as_ref().ok_or_else(missing)?;
                if shape.len() != 1 || slot < 0 || slot as usize >= gains.len() {
                    return Err(fault("the core carries a history's gain per block"));
                }
                Ok(gains[slot as usize])
            }
            _ => Err(missing()),
        }
    }

    fn set_gain(&mut self, memory: usize, slot: i64, value: f64) -> Result<(), Error> {
        let missing = || fault("a route's memory is not the brain's");
        match ORDER[memory] {
            "working_memory" => self.working_memory.as_mut().ok_or_else(missing)?.hearing = value,
            "efference" => self.efference.as_mut().ok_or_else(missing)?.hearing = value,
            "recall" => self.recall.as_mut().ok_or_else(missing)?.hearing = value,
            "history" => {
                let held = self.history.as_mut().ok_or_else(missing)?;
                let (_, gains) = held.hearing.as_mut().ok_or_else(missing)?;
                *gains.get_mut(slot.max(0) as usize).ok_or_else(missing)? = value;
            }
            _ => return Err(missing()),
        }
        Ok(())
    }

    /// A memory's trace as the hearing reads it, and its amplitude.
    fn trace_of(&self, memory: usize) -> Result<(Rows, f64), Error> {
        let missing = || fault("a route's memory is not the brain's");
        match ORDER[memory] {
            "working_memory" | "efference" => {
                let t = if memory == 0 { &self.working_memory } else { &self.efference };
                let t = t.as_ref().ok_or_else(missing)?;
                Ok((t.trace.clone(), t.amplitude))
            }
            "history" => {
                let h = self.history.as_ref().ok_or_else(missing)?;
                Ok((h.trace.clone(), h.amplitude))
            }
            "recall" => {
                let r = self.recall.as_ref().ok_or_else(missing)?;
                let (rows, cols) = match r.trace.shape.as_slice() {
                    [a, b] => (*a, *b),
                    _ => return Err(fault("the recall memory's trace")),
                };
                Ok((Rows { rows, cols, data: r.trace.v.clone() }, r.config.amplitude))
            }
            _ => Err(missing()),
        }
    }

    /// The hearing reads its memories as they stand: their gains and traces.
    pub fn hearing_in(&mut self) -> Result<(), Error> {
        let Some(hearing) = &self.hearing else { return Ok(()) };
        let routes: Vec<(usize, i64)> = hearing.routes.iter()
            .map(|r| (self.hearing_memories[r.memory], r.slot)).collect();
        let mut gains = Vec::with_capacity(routes.len());
        for (memory, slot) in routes {
            gains.push(self.gain_of(memory, slot)?);
        }
        let mut traces = Vec::new();
        for &memory in &self.hearing_memories {
            traces.push(self.trace_of(memory)?);
        }
        let hearing = self.hearing.as_mut().unwrap();
        hearing.gains = gains;
        for (held, (trace, amplitude)) in hearing.memories.iter_mut().zip(traces) {
            held.trace = trace;
            held.amplitude = amplitude;
        }
        Ok(())
    }

    /// The hearing's gains onto their memories (`Hearing.step` sets each route's on its memory).
    pub fn hearing_out(&mut self) -> Result<(), Error> {
        let Some(hearing) = &self.hearing else { return Ok(()) };
        let routes: Vec<(usize, i64, f64)> = hearing.routes.iter().zip(&hearing.gains)
            .map(|(r, &g)| (self.hearing_memories[r.memory], r.slot, g)).collect();
        for (memory, slot, gain) in routes {
            self.set_gain(memory, slot, gain)?;
        }
        Ok(())
    }

    pub fn export(&self, out: &mut Named) {
        for sense in &self.senses {
            let at = |a: usize, b: usize| Array::from_i64(&[2], &[a as i64, b as i64]);
            out.put(&format!("senses.{}", sense.name), at(sense.start, sense.stop));
            for (field, a, b) in &sense.fields {
                out.put(&format!("senses.{}/{field}", sense.name), at(*a, *b));
            }
        }
        out.put("spine.inputs", Array::scalar_i64(self.inputs as i64));
        out.put("spine.sensory", Array::from_i64(&[self.sensory.len()], &self.sensory));
        out.put("spine.motor", Array::from_i64(&[self.motor.len()], &self.motor));
        for (name, x) in [("coded_cortex", self.coded_cortex), ("coded_keys", self.coded_keys),
                          ("surprise_private", self.surprise_private),
                          ("hippocampus_maturity", self.hippocampus_maturity)] {
            out.put(&format!("spine.{name}"), Array::from_bool(&[], &[x]));
        }
        for (k, name) in self.map_values.iter().enumerate() {
            out.put(&format!("spine.map_value.{name}"), Array::scalar_i64(k as i64));
        }
        self.export_parts(out);
        let row = |x: &[f64]| Array::from_f64(&[1, x.len()], x);
        if let Some(l) = &self.lived {
            out.put("spine.lived.x", row(&l.x));
            out.put("spine.lived.action", Array::from_i64(&[l.action.len()], &l.action));
            out.put("spine.lived.value", Array::scalar_f64(l.value));
            out.put("spine.lived.aroused", Array::from_bool(&[], &[l.aroused]));
            out.put("spine.lived.own", Array::from_bool(&[], &[l.own]));
        }
        if let Some((x, action)) = &self.moment {
            out.put("spine.moment.x", row(x));
            out.put("spine.moment.action", Array::from_i64(&[action.len()], action));
        }
        if let Some(x) = &self.prepared {
            out.put("spine.prepared", row(x));
        }
        if let Some(r) = &self.surprise_read {
            out.put("spine.surprise_read.x", row(&r.x));
            out.put("spine.surprise_read.sizes", row(&r.sizes));
            out.put("spine.surprise_read.rows", Array::from_bool(&[r.rows.len()], &r.rows));
        }
        if let Some(x) = &self.surprise_seen {
            out.put("spine.surprise_seen", row(x));
        }
        out.put("spine.episode_told", Array::from_i64(&[self.episode_told.len()], &self.episode_told));
        if let Some(a) = &self.last_arousal {
            let p = |k: &str| format!("spine.last_arousal.{k}");
            out.put(&p("mode"), Array::scalar_i64((a.mode == MODES[1]) as i64));
            for (k, x) in [("level", a.level), ("error", a.error), ("surprise", a.surprise),
                           ("want", a.want)] {
                out.put(&p(k), Array::scalar_f64(x));
            }
            out.put(&p("learned"), Array::from_bool(&[], &[a.learned]));
            out.put(&p("recorded"), Array::from_bool(&[], &[a.recorded]));
            out.put(&p("sweeps"), Array::scalar_i64(a.sweeps));
            out.put(&p("learning_sweeps"), Array::scalar_i64(a.learning_sweeps));
            if let Some(t) = a.temperature {
                out.put(&p("temperature"), Array::scalar_f64(t));
            }
            if let Some(t) = &a.temperatures {
                out.put(&p("temperatures"), Array::from_f64(&[t.len()], t));
            }
            if let Some(h) = a.heated_slot {
                out.put(&p("heated_slot"), Array::scalar_i64(h));
            }
        }
        if let Some(r) = &self.last_settlement {
            let p = |k: &str| format!("spine.last_settlement.{k}");
            out.put(&p("operation"), Array::scalar_i64(r.operation));
            out.put(&p("qualified"), Array::from_bool(&[r.qualified.len()], &r.qualified));
            out.put(&p("residual"), Array::from_f64(&[r.residual.len()], &r.residual));
            for (k, x) in [("steps", r.steps), ("budget", r.budget),
                           ("residual_checks", r.residual_checks),
                           ("damping_halvings", r.damping_halvings),
                           ("stagnation_checks", r.stagnation_checks)] {
                out.put(&p(k), Array::scalar_i64(x));
            }
            out.put(&p("tolerance"), Array::scalar_f64(r.tolerance));
            if let Some(x) = r.rescue_steps {
                out.put(&p("rescue_steps"), Array::scalar_i64(x));
            }
            if let Some(x) = r.row_sweeps {
                out.put(&p("row_sweeps"), Array::scalar_i64(x));
            }
        }
    }

    fn export_parts(&self, out: &mut Named) {
        for attribute in PARTS {
            let mut part = Named::new();
            match attribute {
                "arousal" => if let Some(p) = &self.arousal { p.export(&mut part) },
                "sparse_code" => if let Some(p) = &self.sparse_code { p.export(&mut part) },
                "working_memory" => if let Some(p) = &self.working_memory { p.export(&mut part) },
                "efference" => if let Some(p) = &self.efference { p.export(&mut part) },
                "history" => if let Some(p) = &self.history { p.export(&mut part) },
                "recall" => if let Some(p) = &self.recall { p.export(&mut part) },
                "hippocampus" => if let Some(p) = &self.hippocampus {
                    let mut whole = Named::new();
                    p.export(&mut whole);
                    for name in whole.order.iter().filter(|n| *n != "maturity") {
                        part.put(name, whole.arrays[name].clone());
                    }
                },
                "bound" => if let Some(p) = &self.bound { p.export(&mut part) },
                "surprise" => if let Some(p) = &self.surprise { p.export(&mut part) },
                "orienting" => if let Some(p) = &self.orienting { p.export(&mut part) },
                "space" => if let Some(p) = &self.space { p.export(&mut part) },
                "brain" => if let Some(p) = &self.brain { p.export(&mut part) },
                "hearing" => if let Some(p) = &self.hearing {
                    let mut whole = Named::new();
                    p.export(&mut whole);
                    for name in &whole.order {
                        if name != "gains" && !name.starts_with("memory.") {
                            part.put(name, whole.arrays[name].clone());
                        }
                    }
                    let codes: Vec<i64> = self.hearing_memories.iter().map(|&m| m as i64).collect();
                    part.put("memory_of", Array::from_i64(&[codes.len()], &codes));
                },
                "learner" => if let Some(p) = &self.learner { p.export(&mut part) },
                "basal_ganglia" => if let (Some(a), Some(l), Some(g)) =
                    (&self.basal_ganglia, &self.learner, &self.brain) {
                    a.export(l, g, &mut part)
                },
                _ => {}
            }
            out.put_under(attribute, &part);
        }
    }

    // -- the API a world codes against

    /// `Brain.last_arousal`: the readings of the latest `live` moment.
    pub fn last_arousal(&self) -> Option<&LastArousal> {
        self.last_arousal.as_ref()
    }

    /// The columns of each sense, by name.
    pub fn sense(&self, name: &str) -> Option<(usize, usize)> {
        self.senses.iter().find(|s| s.name == name).map(|s| (s.start, s.stop))
    }

    // -- the moment's glue (`generic.py`)

    fn graph(&self) -> Result<&Graph, Error> {
        self.brain.as_ref().ok_or_else(|| fault("the brain holds no graph"))
    }

    fn settle(&self) -> Result<Settle, Error> {
        let c = &self.learner.as_ref().ok_or_else(|| fault("the brain holds no learner"))?.config;
        Ok(Settle {
            free_steps: c.free_steps,
            tolerance: c.tolerance
                .ok_or_else(|| refused("Brain answers require a finite residual tolerance"))?,
            damping: c.damping,
            qualified: c.qualified,
            fast: c.fast_settle,
        })
    }

    /// `_observations` of one row of the whole width: checked, and given by name (`mapping`,
    /// `_sensed_rows`) completed by the bound records where the brain holds them
    /// (`_bound_complete`).
    pub fn observations(&mut self, row: &[f64], mapping: bool) -> Result<Vec<f64>, Error> {
        if mapping && !all_finite(row) {
            return Err(refused("a sense takes a nonempty finite batch of rows; use [row] for one"));
        }
        let x = if mapping && self.bound.is_some() {
            if row.len() != self.inputs {
                return Err(refused(&format!("observations need {} values per row, got {}",
                                            self.inputs, row.len())));
            }
            let keyed = self.bound_keyed(row)?;
            let filled = self.bound.as_ref().unwrap().complete(row, &keyed, 1)?;
            self.given = Some((filled.clone(), row.to_vec()));
            filled
        } else {
            row.to_vec()
        };
        if !all_finite(&x) {
            return Err(refused("observations must be a nonempty finite batch; use [observation] for one"));
        }
        if x.len() != self.inputs {
            return Err(refused(&format!("observations need {} values per row, got {}",
                                        self.inputs, x.len())));
        }
        Ok(x)
    }

    /// `_bound_keyed`: the records' key of a row, each sense at unit length shared among its
    /// fields, through the sparse code, at unit length.
    fn bound_keyed(&mut self, row: &[f64]) -> Result<Vec<f64>, Error> {
        let mut out = row.to_vec();
        for sense in &self.senses {
            let spans = sense.spans();
            let share = 1.0 / np::sqrt(spans.len() as f64);
            for (a, b) in spans {
                let unit = unit_row(&out[a..b]);
                for (o, u) in out[a..b].iter_mut().zip(unit) {
                    *o = share * u;
                }
            }
        }
        let code = self.sparse_code.as_mut().ok_or_else(|| fault("no sparse code"))?
            .code(&one(&out), false)?;
        Ok(unit_row(&code.data))
    }

    /// `_equal_senses`: each sense's columns (each field's) at unit length.
    fn equal_senses(&self, x: &[f64]) -> Vec<f64> {
        let mut out = x.to_vec();
        for sense in &self.senses {
            for (a, b) in sense.spans() {
                let unit = unit_row(&out[a..b]);
                out[a..b].copy_from_slice(&unit);
            }
        }
        out
    }

    /// `_code`: the moment's key, the sparse code of the equalized senses at unit length.
    fn code(&mut self, x: &[f64]) -> Result<Vec<f64>, Error> {
        let equal = self.equal_senses(x);
        let code = self.sparse_code.as_mut().ok_or_else(|| fault("no sparse code"))?
            .code(&one(&equal), false)?;
        Ok(unit_row(&code.data))
    }

    /// `_prepare`: the moment's code, computed once (`_coded_moment`).
    fn prepare(&mut self, x: &[f64]) -> Result<Vec<f64>, Error> {
        let key = self.code(x)?;
        self.coded = Some((x.to_vec(), key.clone()));
        Ok(key)
    }

    /// `_coded` and `_placed`: the code of the moment's observations.
    fn coded(&mut self, x: &[f64]) -> Result<Vec<f64>, Error> {
        if let Some((held, key)) = &self.coded {
            if held.as_slice() == x {
                return Ok(key.clone());
            }
        }
        self.code(x)
    }

    /// `_stimulus` with the memories: the code (or the observations) on the sensory neurons,
    /// then each memory's read in `Memories.ORDER`.
    fn stimulus(&mut self, x: &[f64], code: &[f64]) -> Result<Vec<f64>, Error> {
        let graph = self.graph()?;
        let n = graph.n();
        let amplitude = graph.neuron.stimulus_amplitude;
        let sensed = if self.coded_cortex { code } else { x };
        if sensed.len() != self.sensory.len() {
            return Err(fault("the sensory neurons and what drives them differ in width"));
        }
        let mut drive = vec![0.0; n];
        for (&k, &v) in self.sensory.iter().zip(sensed) {
            *drive.get_mut(k as usize).ok_or_else(|| fault("sensory"))? = v * amplitude;
        }
        if let Some(h) = self.hippocampus.as_mut() {
            h.compiled_moment = false;
        }
        if let Some(h) = self.hearing.as_mut() {
            h.compiled_moment = false;
        }
        let present: Vec<&str> = ORDER.iter().copied().filter(|m| match *m {
            "working_memory" => self.working_memory.is_some(),
            "efference" => self.efference.is_some(),
            "history" => self.history.is_some(),
            "recall" => self.recall.is_some(),
            "hippocampus" => self.hippocampus.is_some(),
            _ => false,
        }).collect();
        // the place records read the moment's key where the brain keys them on its code
        let place = (self.coded_keys).then(|| code.to_vec());
        for member in memories::members(&present)? {
            drive = match member {
                "working_memory" => self.working_memory.as_ref().unwrap().stimulate(one(&drive))?.data,
                "efference" => self.efference.as_ref().unwrap().stimulate(one(&drive))?.data,
                "history" => self.history.as_ref().unwrap().stimulate(one(&drive))?.data,
                "recall" => {
                    let recall = self.recall.as_mut().unwrap();
                    let read = recall.stimulate(F { shape: vec![1, n], v: drive },
                                                F { shape: vec![1, code.len()], v: code.to_vec() })?;
                    read.v
                }
                "hippocampus" => {
                    let memory = self.hippocampus.as_ref().unwrap();
                    match &place {
                        Some(key) => memory.stimulate(&drive, &[1, n], Some((key, &[1, key.len()])))?,
                        None => memory.stimulate(&drive, &[1, n], None)?,
                    }
                }
                _ => return Err(fault("a memory the core does not hold")),
            };
        }
        Ok(drive)
    }

    /// `_moment_drive` of the moment prepared.
    fn moment_drive(&mut self, x: &[f64]) -> Result<Vec<f64>, Error> {
        let code = self.coded(x)?;
        self.stimulus(x, &code)
    }

    /// `_equilibrate`: half the budget fast, the rest at half the step where it did not
    /// qualify; with `qualified`, the damping ladder at once.
    fn equilibrate(&self, drive: &graph::Rows, state: Option<&State>, budget: i64,
                   tolerance: f64) -> Result<Equilibrium, Error> {
        let graph = self.graph()?;
        let c = self.settle()?;
        if c.qualified {
            return graph.equilibrate(drive, budget, CHUNK, tolerance, state, None, None,
                                     c.damping, false);
        }
        let first = if budget < 2 { budget } else { (budget + 1) / 2 };
        let phase = graph.equilibrate(drive, first, CHUNK, tolerance, state, None, None, 0,
                                      c.fast)?;
        if phase.qualified() || phase.steps >= budget || !state_finite(&phase.state) {
            return Ok(phase);
        }
        let damped = graph.with_dt(graph.neuron.dt / 2.0);
        let following = damped.equilibrate(drive, budget - phase.steps, CHUNK, tolerance,
                                           Some(&phase.state), None, None, 0, false)?;
        let mut state = following.state;
        let residual = graph.check(drive, &mut state, None, None)?;
        Ok(Equilibrium {
            state,
            residual,
            steps: phase.steps + following.steps,
            tolerance,
            residual_checks: phase.residual_checks + following.residual_checks + 1,
            damping_halvings: 1,
            stagnation_checks: 0,
            row_sweeps: None,
        })
    }

    /// `_qualified(drive, state, operation=)`: the free answer within the budget, with one
    /// rescue on the damping ladder; refused where it does not settle.
    fn qualified(&mut self, drive: &[f64], state: Option<&State>, operation: &str)
                 -> Result<State, Error> {
        let rows = graph_rows(drive);
        let c = self.settle()?;
        let (budget, tolerance, damping) = (c.free_steps, c.tolerance, c.damping);
        let mut phase = self.equilibrate(&rows, state, budget, tolerance)?;
        let mut rescue = None;
        if budget > 0 && !phase.qualified() && state_finite(&phase.state) {
            let before = phase;
            let graph = self.graph()?;
            phase = graph.equilibrate(&rows, budget, CHUNK, tolerance, Some(&before.state), None,
                                      None, damping.max(1), false)?;
            rescue = Some(phase.steps);
            phase.steps += before.steps;
            phase.residual_checks += before.residual_checks;
            phase.stagnation_checks += before.stagnation_checks;
        }
        let qualified: Vec<bool> = phase.residual.iter().map(|&e| e <= phase.tolerance).collect();
        self.last_settlement = Some(Settlement {
            operation: OPERATIONS.iter().position(|o| *o == operation).unwrap_or(0) as i64,
            qualified: qualified.clone(),
            residual: phase.residual.clone(),
            steps: phase.steps,
            budget,
            residual_checks: phase.residual_checks,
            damping_halvings: phase.damping_halvings,
            stagnation_checks: phase.stagnation_checks,
            tolerance: phase.tolerance,
            rescue_steps: rescue.filter(|&r| r != 0),
            row_sweeps: phase.row_sweeps,
        });
        if !qualified.iter().all(|&q| q) {
            return Err(refused("brain did not settle within its budget; no action issued"));
        }
        Ok(phase.state)
    }

    /// `_settled(x)`: the moment prepared, its drive read, the free answer from the stream's
    /// activity (`activity`, the actor's free state), and the residual surprise reads.
    pub fn settled(&mut self, x: &[f64], activity: Option<&State>)
                   -> Result<(Vec<f64>, State), Error> {
        self.prepare(x)?;
        let drive = self.moment_drive(x)?;
        let free = self.qualified(&drive, activity, "act")?;
        if self.surprise.is_some() {
            self.read_surprise(x, &drive, activity)?;
        }
        Ok((drive, free))
    }

    /// `_read_surprise`: the residual the moment's drive leaves on the carried state, read once
    /// per moment.
    fn read_surprise(&mut self, x: &[f64], drive: &[f64], carried: Option<&State>)
                     -> Result<(), Error> {
        if self.surprise.is_none() || self.surprise_private || same(&self.surprise_seen, x) {
            return Ok(());
        }
        if self.surprise_read.as_ref().is_some_and(|r| r.x.as_slice() == x) {
            return Ok(());
        }
        self.surprise_read = Some(self.residual_read(x, drive, carried, None)?);
        Ok(())
    }

    /// `_residual_read`: each unit's size of the equation a settle from `carried` under `drive`
    /// repairs; zeros, and no stream carried, where nothing was carried.
    fn residual_read(&self, x: &[f64], drive: &[f64], carried: Option<&State>,
                     rows: Option<Vec<bool>>) -> Result<SurpriseRead, Error> {
        let size = self.surprise.as_ref().map_or(0, |s| s.size());
        let carried = match carried {
            Some(c) if c.batch == 1 => c,
            _ => return Ok(SurpriseRead { x: x.to_vec(), sizes: vec![0.0; size],
                                          rows: vec![false] }),
        };
        let error = self.graph()?.residual_per_unit(&graph_rows(drive), carried, None, None)?;
        let sizes = error.iter().map(|e| {
            let a = e.abs();
            if a.is_nan() { 0.0 } else if a == f64::INFINITY { f64::MAX } else { a }
        }).collect();
        Ok(SurpriseRead { x: x.to_vec(), sizes, rows: rows.unwrap_or_else(|| vec![true]) })
    }

    /// `_surprised(x)`: the moment's surprise, once it is certain, and the focus drawn by it
    /// among the map's cells and what a heard name brings back.
    pub fn surprised(&mut self, x: &[f64]) -> Result<(), Error> {
        if self.surprise.is_none() || self.orienting.is_none() || self.surprise_private {
            return Ok(());
        }
        let read = self.surprise_read.take();
        let held = self.surprise.as_mut().unwrap();
        match read {
            Some(r) if r.x.as_slice() == x => {
                held.observe(&r.sizes, 1, true, Some(&r.rows))?;
            }
            _ if same(&self.surprise_seen, x) => return Ok(()),
            _ => {
                let size = held.size();
                held.observe(&vec![0.0; size], 1, true, Some(&[false]))?;
            }
        }
        self.surprise_seen = Some(x.to_vec());
        let (mut surprise, mut places, mut sense) = (Vec::new(), Vec::new(), Vec::new());
        if let Some(space) = &self.space {
            if let Some(mapped) = &space.surprise {
                let size = mapped.size();
                let first = mapped.last.get(..size).ok_or_else(|| fault("the map's surprise"))?;
                surprise.extend_from_slice(first);
                for c in space.centers() {
                    places.extend_from_slice(c);
                }
                sense.extend(std::iter::repeat(FROM_MAP).take(size));
            }
        }
        if let Some((salience, at)) = self.bound_named(x)? {
            sense.extend(std::iter::repeat(FROM_NAMED).take(salience.len()));
            surprise.extend(salience);
            places.extend(at);
        }
        if !sense.is_empty() {
            self.orienting.as_mut().unwrap().attend(&surprise, &places, 2, &sense)?;
        }
        Ok(())
    }

    /// The row the bound records were given at this moment, where `x` is its completion.
    fn given_row(&self, x: &[f64]) -> Vec<f64> {
        match &self.given {
            Some((filled, given)) if filled.as_slice() == x => given.clone(),
            _ => x.to_vec(),
        }
    }

    /// `_bound_named(x)`: the salience a heard name gives each thing on the map, and their
    /// places; None where it gives none.
    fn bound_named(&mut self, x: &[f64]) -> Result<Option<(Vec<f64>, Vec<f64>)>, Error> {
        let heard = match (&self.bound, &self.space) {
            (Some(b), Some(_)) => match b.heard {
                Some(h) => (b.fields[h].first, b.fields[h].last),
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        let row = self.given_row(x);
        let mut code = None;
        if row[heard.0..heard.1].iter().any(|&v| v != 0.0) {
            let mut alone = vec![0.0; row.len()];
            alone[heard.0..heard.1].copy_from_slice(&row[heard.0..heard.1]);
            code = Some(self.bound_keyed(&alone)?);
        }
        let bound = self.bound.as_ref().unwrap();
        let back = bound.named(&row, code.as_deref())?;
        if back.is_empty() {
            return Ok(None);
        }
        let template: Vec<(String, Vec<f64>)> = back.into_iter()
            .map(|(f, part)| (bound.fields[f].name.clone(), part)).collect();
        let space = self.space.as_mut().unwrap();
        let count = space.count();
        let things: Vec<_> = space.things()?.into_iter()
            .filter(|t| t.place.is_some() && t.pieces.iter().any(|p| !p.is_empty()))
            .collect();
        if things.is_empty() {
            return Ok(None);
        }
        let mut read = Vec::with_capacity(things.len());
        for thing in &things {
            let mut cells = vec![false; count];
            for piece in &thing.pieces {
                for &c in piece {
                    *cells.get_mut(c).ok_or_else(|| fault("a thing's cells"))? = true;
                }
            }
            let whole = space.thing_at(None, None, Some(&cells), [0.0, 0.0], 0.0)?;
            let mut codes = Vec::new();
            if let Some(whole) = whole {
                let mut at = 0;
                for (name, &(a, b)) in self.map_values.iter().zip(&space.groups) {
                    codes.push((name.clone(), whole.values[at..at + (b - a)].to_vec()));
                    at += b - a;
                }
            }
            read.push(codes);
        }
        let salience = Bound::salience(&template, &read);
        if !salience.iter().any(|&s| s != 0.0) {
            return Ok(None);
        }
        let places = things.iter().flat_map(|t| t.place.unwrap()).collect();
        Ok(Some((salience, places)))
    }

    /// `_moment_surprise(rows, memory)`: the moment's surprise as tellings, for a memory whose
    /// records take it.
    fn moment_surprise(&self) -> Option<Vec<f64>> {
        let told = self.surprise.as_ref()?.moment();
        if told.len() != 1 || !all_finite(&told) {
            return None;
        }
        Some(told.iter().map(|&t| np::expm1(np::clip(t, 0.0, 50.0))).collect())
    }

    /// `_bound_hear(x, acted)`: the bound records hear the row they were given.
    fn bound_hear(&mut self, x: &[f64], acted: bool) -> Result<(), Error> {
        let rows = self.given_row(x);
        let keyed = self.bound_keyed(&rows)?;
        let surprise = self.moment_surprise();
        self.bound.as_mut().unwrap().hear(&rows, &keyed, 1, surprise.as_deref(), acted)
    }

    /// `_hear_episode(x)`: the moment joins the bound records (no rehearsal is held).
    fn hear_episode(&mut self, x: &[f64], acted: bool) -> Result<(), Error> {
        if self.bound.is_some() && !self.surprise_private {
            self.bound_hear(x, acted)?;
        }
        Ok(())
    }

    /// `_choose` after the actor's choice (`action`, the actor's free state `free`): the
    /// memories that follow a decision move on.
    pub fn chosen(&mut self, x: &[f64], free: &State, action: &[i64], greedy: bool,
                  offsets: &[i64]) -> Result<(), Error> {
        let code = self.coded(x)?;
        let activation = Rows { rows: free.batch, cols: free.activation.len() / free.batch.max(1),
                                data: free.activation.clone() };
        if let Some(t) = self.working_memory.as_mut() {
            t.update(&activation)?;
        }
        if self.efference.is_some() {
            let issued = command(action, self.motor.len(), offsets)?;
            self.efference.as_mut().unwrap().issue(one(&issued))?;
        }
        let heard = if self.coded_cortex { code.clone() } else { x.to_vec() };
        if let Some(h) = self.history.as_mut() {
            h.hear(&one(&heard))?;
        }
        if let Some(r) = self.recall.as_mut() {
            r.witness(F { shape: vec![1, code.len()], v: code.clone() },
                      F { shape: vec![activation.rows, activation.cols], v: activation.data })?;
        }
        self.hear_episode(x, true)?;
        self.moment = (!greedy).then(|| (x.to_vec(), action.to_vec()));
        if !self.surprise_private {
            self.surprise_seen = None;
        }
        Ok(())
    }

    /// `_choose` before the actor's choice: the stream's state handed to the actor is the
    /// caller's; here the moment's surprise is observed before the memories write.
    pub fn choosing(&mut self, x: &[f64]) -> Result<(), Error> {
        self.prepared = None;
        if self.surprise.is_some() {
            self.surprised(x)?;
        }
        Ok(())
    }

    /// `_record(keys, action, reward, importance)`: an outcome written into the place records
    /// for the situation it was chosen in, at the maturity's factor (`factor`).
    pub fn record(&mut self, x: &[f64], action: &[i64], reward: f64, importance: f64,
                  factor: Option<f64>, offsets: &[i64]) -> Result<(), Error> {
        let keys = self.code(x)?;
        let memory = self.hippocampus.as_mut().ok_or_else(|| fault("no place records"))?;
        if self.hippocampus_maturity {
            memory.maturity = factor;
        }
        memories::record_action(memory, &keys, &[1, keys.len()], action, &[action.len()],
                                &[reward], &[importance], None, offsets)
    }

    /// `say(observations)`: what the bound records say for a row given by name, in the voice's
    /// format.
    pub fn say(&mut self, row: &[f64]) -> Result<Vec<f64>, Error> {
        if self.bound.is_none() {
            return Err(refused("say reads the brain's bound records: compose it with senses=, \
                                code= and bound=records"));
        }
        if row.len() != self.inputs || !all_finite(row) {
            return Err(refused("observations must be finite"));
        }
        if self.bound.as_ref().unwrap().voice.is_none() {
            return Err(refused("say needs a voice: compose the brain with voice= naming the \
                                sense it hears in"));
        }
        let keyed = self.bound_keyed(row)?;
        self.bound.as_mut().unwrap().say(row, &keyed, 1)
    }

    // -- the moment with the actor (`live`, `act`, `learn`, `_choose`)

    fn actor(&self) -> Result<&ActorCritic, Error> {
        self.basal_ganglia.as_ref().ok_or_else(|| fault("the brain holds no actor"))
    }

    /// The maturity's factor, where the brain holds one.
    fn factor(&self) -> Option<f64> {
        self.basal_ganglia.as_ref()?.maturity.as_ref().map(|m| m.factor())
    }

    fn offsets(&self) -> Vec<i64> {
        self.learner.as_ref().map_or(vec![0], |l| l.slot_offsets.clone())
    }

    /// `pending_feedback`: whether an issued action owns the next outcome.
    pub fn pending_feedback(&self) -> bool {
        self.basal_ganglia.as_ref().is_some_and(|a| a.pending.is_some())
            || self.lived.as_ref().is_some_and(|l| !l.aroused)
    }

    /// `decision_id`: the age of the stream when `live` issued the action that owns the next
    /// outcome, or None.
    pub fn decision_id(&self) -> Option<i64> {
        if self.lived.is_none() || !self.pending_feedback() {
            return None;
        }
        self.arousal.as_ref().map(|a| a.age)
    }

    /// `_owned(decision_id)`: refused for an outcome of an action other than the one awaiting.
    fn owned(&self, decision_id: i64) -> Result<(), Error> {
        if decision_id < 1 {
            return Err(refused("decision_id must be a positive integer read from Brain.decision_id"));
        }
        match self.decision_id() {
            None => Err(refused("the decision_id does not own an outcome: no action that live \
                                 issued awaits one")),
            Some(awaited) if awaited != decision_id => {
                Err(refused("the decision_id does not own the next outcome"))
            }
            Some(_) => Ok(()),
        }
    }

    /// `_choices(state)`: the greedy readout, one argmax per motor slot.
    fn choices(&self, state: &State) -> Result<Vec<i64>, Error> {
        let learner = self.learner.as_ref().ok_or_else(|| fault("the brain holds no learner"))?;
        let out: Vec<f64> = self.motor.iter().map(|&k| state.activation[k as usize]).collect();
        if learner.slot_count() > 1 {
            let mut at = 0usize;
            let mut choice = Vec::with_capacity(learner.slot_count());
            for &size in &learner.slot_sizes {
                let size = size as usize;
                choice.push(np::argmax(&out[at..at + size]) as i64);
                at += size;
            }
            return Ok(choice);
        }
        Ok(vec![np::argmax(&out) as i64])
    }

    /// `_choose(x, drive, free, greedy=, temperature=)`: the moment's state handed to the
    /// actor, the surprise observed, the actor's choice, then the memories that follow it.
    fn choose(&mut self, x: &[f64], drive: &[f64], free: State, greedy: bool,
              temperature: Option<Heat>) -> Result<Vec<i64>, Error> {
        let generation = self.graph()?.generation;
        {
            let actor = self.basal_ganglia.as_mut().ok_or_else(|| fault("no actor"))?;
            actor.free = Some(free);
            actor.free_generation = Some(generation);
            actor.drive = Some(graph_rows(drive));
        }
        self.choosing(x)?;
        self.hearing_in()?;
        let action = {
            let actor = self.basal_ganglia.as_mut().unwrap();
            let learner = self.learner.as_ref().ok_or_else(|| fault("no learner"))?;
            let graph = self.brain.as_ref().unwrap();
            actor.act_held(learner, graph, self.recall.as_mut(), self.hearing.as_ref(),
                           &graph_rows(drive), greedy, temperature.as_ref(), None)?.0
        };
        let state = self.actor()?.free.clone().ok_or_else(|| fault("the actor holds no state"))?;
        let offsets = self.offsets();
        self.chosen(x, &state, &action, greedy, &offsets)?;
        Ok(action)
    }

    /// `learn(reward, done, next_observations, salience=)` inside `live`: the outcome of the
    /// sampled action, its record, the actor's and critic's step, and the next moment's
    /// surprise.
    fn learn(&mut self, reward: f64, done: bool, row: &[f64], salience: Option<f64>)
             -> Result<Report, Error> {
        let following = self.observations(row, false)?;
        if self.actor()?.pending.is_none() {
            return Err(refused("learn needs a non-greedy act first"));
        }
        if done {
            return Err(Error::Missing("the end of an attempt (done) in the core's learn".into()));
        }
        let importance = match salience {
            None => reward.abs(),
            Some(s) if s.is_finite() && s >= 0.0 => s,
            Some(_) => return Err(refused(
                "salience must be a finite nonnegative vector matching the batch")),
        };
        self.prepare(&following)?;
        let kept = self.hippocampus.clone();
        let found = self.learn_outcome(reward, done, &following, importance);
        let report = match found {
            Ok(r) => r,
            Err(e) => {
                if kept.is_some() {
                    self.hippocampus = kept;
                }
                return Err(e);
            }
        };
        self.moment = None;
        self.prepared = Some(following.clone());
        Ok(report)
    }

    fn learn_outcome(&mut self, reward: f64, done: bool, following: &[f64], importance: f64)
                     -> Result<Report, Error> {
        if self.hippocampus.is_some() {
            if let Some((keys, action)) = self.moment.clone() {
                let (factor, offsets) = (self.factor(), self.offsets());
                self.record(&keys, &action, reward, importance, factor, &offsets)?;
            }
        }
        let next_drive = self.moment_drive(following)?;
        let carried = self.actor()?.free.clone();
        let read = if self.surprise.is_some() && !self.surprise_private {
            Some(self.residual_read(following, &next_drive, carried.as_ref(), Some(vec![!done]))?)
        } else {
            None
        };
        self.hearing_in()?;
        let report = {
            let actor = self.basal_ganglia.as_mut().unwrap();
            actor.learn(self.learner.as_mut().ok_or_else(|| fault("no learner"))?,
                        self.brain.as_mut().unwrap(), self.recall.as_mut(),
                        self.hearing.as_mut(), &[reward], &[done], &graph_rows(&next_drive),
                        None)?
        };
        self.hearing_out()?;
        self.moment = None;
        self.prepared = Some(following.to_vec());
        if let Some(read) = read {
            self.surprise_read = Some(read);
            self.surprise_seen = None;
            self.surprised(following)?;
        }
        Ok(report)
    }

    /// `live(observations, reward=, done=, salience=, decision_id=)`: one moment of a
    /// continuing life, the row given by name as one row of the whole width (a silent sense
    /// zeros); the action per motor slot.
    pub fn live(&mut self, observations: &[f64], reward: Option<f64>, done: Option<bool>,
                salience: Option<f64>, decision_id: Option<i64>) -> Result<Vec<i64>, Error> {
        self.live_row(observations, true, reward, done, salience, decision_id)
    }

    /// `live` with the row given whole (`mapping` false: no completion by the bound records,
    /// as the Python takes an array).
    pub fn live_row(&mut self, observations: &[f64], mapping: bool, reward: Option<f64>,
                    done: Option<bool>, salience: Option<f64>, decision_id: Option<i64>)
                    -> Result<Vec<i64>, Error> {
        let action = self.moment_of_life(observations, mapping, reward, done, salience,
                                         decision_id)?;
        // `_moment_of_life`: a moment has passed for the maturity
        if let Some(m) = self.basal_ganglia.as_mut().and_then(|a| a.maturity.as_mut()) {
            m.relax();
        }
        Ok(action)
    }

    #[allow(clippy::too_many_arguments)]
    fn moment_of_life(&mut self, observations: &[f64], mapping: bool, reward: Option<f64>,
                      done: Option<bool>, salience: Option<f64>, decision_id: Option<i64>)
                      -> Result<Vec<i64>, Error> {
        if self.arousal.is_none() {
            return Err(refused("live needs arousal genes; construct the brain with arousal=True"));
        }
        let x = self.observations(observations, mapping)?;
        if self.actor()?.free.as_ref().is_some_and(|f| f.batch != 1) {
            return Err(refused("live follows one continuing stream; reset before changing streams"));
        }
        let lived = self.lived.clone();
        let sampled = self.actor()?.pending.is_some();
        let routine = lived.as_ref().is_some_and(|l| !l.aroused) && !sampled;
        if let Some(id) = decision_id {
            self.owned(id)?;
        }
        if !sampled && !routine && (reward.is_some() || done.is_some() || salience.is_some()) {
            return Err(refused("feedback needs a preceding action; start with live(observations)"));
        }
        let r = reward.unwrap_or(0.0);
        let ended = done.unwrap_or(false);
        if !r.is_finite() {
            return Err(refused("reward must be one finite value for the stream"));
        }
        if let Some(s) = salience {
            if !s.is_finite() || s < 0.0 {
                return Err(refused("salience must be a finite nonnegative vector matching the batch"));
            }
        }
        if ended {
            return Err(Error::Missing("the end of an attempt (done) in the core's live".into()));
        }
        let mut answer: Option<(Vec<f64>, State)> = None;
        let (mut error, mut surprise, mut want) = (0.0, 0.0, 0.0);
        let (mut sweeps, mut learning_sweeps) = (0i64, 0i64);
        let (mut learned, mut recorded) = (false, false);
        if sampled {
            let report = self.learn(r, ended, &x, salience)?;
            self.lived = None;
            error = report.td_error();
            learning_sweeps += report.free_steps() as i64;
            learned = true;
        } else if routine {
            let activity = self.actor()?.free.clone();
            let settled = self.settled(&x, activity.as_ref())?;
            sweeps += self.last_settlement.as_ref().map_or(0, |s| s.steps);
            let following = if ended { 0.0 } else { self.actor()?.value(&settled.1)[0] };
            let gamma = self.actor()?.config.gamma;
            error = (r + gamma * following - lived.as_ref().unwrap().value).abs();
            answer = Some(settled);
        }
        if sampled || routine {
            let own = lived.as_ref().is_none_or(|l| l.own);
            match self.arousal.as_mut().unwrap().outcome(error, r, own, sampled) {
                Some((s, w)) => (surprise, want) = (s, w),
                None if sampled => return Err(refused(
                    "feedback was accepted but arousal could not update; retry \
                     live(observations) without reward or done or salience")),
                None => {
                    self.surprise_read = None;
                    return Err(refused("arousal moments must remain finite"));
                }
            }
            self.lived = None;
            if reward.is_some() {
                if let Some(b) = self.bound.as_mut() {
                    b.pay(&[r]);
                }
            }
            if routine && self.surprise.is_some() {
                self.surprised(&x)?;
            }
            if routine {
                // no end of an attempt here: the streams go on, the episodes' counts stay
                self.basal_ganglia.as_mut().unwrap().fade(&[ended])?;
            }
            if routine && self.arousal.as_ref().unwrap().aroused() && self.hippocampus.is_some() {
                let held = lived.as_ref().unwrap();
                let importance = salience.unwrap_or(r.abs());
                let (factor, offsets) = (self.factor(), self.offsets());
                self.record(&held.x, &held.action, r, importance, factor, &offsets)?;
                recorded = true;
                // the record can change this moment's recall: settle again under it, warm
                let (drive, free) = answer.take().unwrap();
                let generation = self.graph()?.generation;
                let actor = self.basal_ganglia.as_mut().unwrap();
                actor.free = Some(free);
                actor.free_generation = Some(generation);
                actor.drive = Some(graph_rows(&drive));
            }
        }
        let aroused = self.arousal.as_ref().unwrap().aroused();
        let base = self.learner.as_ref().ok_or_else(|| fault("no learner"))?.config.temperature;
        let temperature = aroused.then(|| base * self.arousal.as_ref().unwrap().heat());
        if let Some(t) = temperature {
            self.actor()?.temperature(self.learner.as_ref().unwrap(), t)?;
        }
        let (drive, free) = match answer {
            Some(a) => a,
            None => {
                let activity = self.actor()?.free.clone();
                let settled = self.settled(&x, activity.as_ref())?;
                sweeps += self.last_settlement.as_ref().map_or(0, |s| s.steps);
                settled
            }
        };
        let slots = self.learner.as_ref().unwrap().slot_count();
        let mut heat = temperature.map(Heat::One);
        let mut heated_slot = None;
        if let Some(t) = temperature.filter(|&t| t > base) {
            let slot = if slots > 1 {
                self.basal_ganglia.as_mut().unwrap().rng.integers(0, slots as i64)
            } else {
                0
            };
            heated_slot = Some(slot);
            if slots > 1 {
                let mut each = vec![base; slots];
                each[slot as usize] = t;
                heat = Some(Heat::Each(each));
            }
        }
        let best = self.choices(&free)?;
        let action = self.choose(&x, &drive, free, !aroused, heat.clone())?;
        let actor = self.actor()?;
        let state = actor.free.as_ref().ok_or_else(|| fault("the actor holds no state"))?;
        if let Some(p) = &actor.pending {
            learning_sweeps += p.plus.steps + p.minus.steps;
        }
        let value = actor.value(state)[0];
        self.lived = Some(Lived { x: x.clone(), action: action.clone(), value, aroused,
                                  own: best == action });
        let arousal = self.arousal.as_mut().unwrap();
        self.last_arousal = Some(LastArousal {
            mode: MODES[arousal.aroused() as usize],
            level: arousal.level,
            error,
            surprise,
            want,
            temperature,
            temperatures: heat.map(|h| match h {
                Heat::One(t) => vec![t; slots],
                Heat::Each(v) => v,
            }),
            heated_slot,
            learned,
            recorded,
            sweeps,
            learning_sweeps,
        });
        arousal.lived(sweeps, learning_sweeps);
        Ok(action)
    }

    /// `act(observations, greedy=, temperature=)`: the answer to a row given by name, with no
    /// outcome taken; an action `live` issued before no longer awaits one.
    pub fn act(&mut self, observations: &[f64], greedy: bool) -> Result<Vec<i64>, Error> {
        self.act_row(observations, true, greedy, None)
    }

    pub fn act_row(&mut self, observations: &[f64], mapping: bool, greedy: bool,
                   temperature: Option<f64>) -> Result<Vec<i64>, Error> {
        let x = self.observations(observations, mapping)?;
        if self.actor()?.free.as_ref().is_some_and(|f| f.batch != 1) {
            return Err(refused("action batch must match the live streams; reset for new streams"));
        }
        if let Some(t) = temperature {
            self.actor()?.temperature(self.learner.as_ref().ok_or_else(|| fault("no learner"))?, t)?;
        }
        let activity = self.actor()?.free.clone();
        let (drive, free) = self.settled(&x, activity.as_ref())?;
        let action = self.choose(&x, &drive, free, greedy, temperature.map(Heat::One))?;
        self.lived = None;  // the actor's state is this answer's: what live issued is passed
        Ok(action)
    }

    // -- calls by name (the replay's)

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let row = |name: &str| -> Result<Vec<f64>, Error> {
            let shape = args.shape(name)?;
            if shape.len() != 2 || shape[0] != 1 {
                return Err(refused("the core's brain lives one stream: one row"));
            }
            args.f64s(name)
        };
        let n = self.graph()?.n();
        if args.has("given.filled") {  // the bound records' row of the moment, where it crosses
            self.given = Some((args.f64s("given.filled")?, args.f64s("given.row")?));
        }
        let one_of = |name: &str| -> Result<Option<f64>, Error> {
            if args.has(name) { Ok(Some(args.f64s(name)?[0])) } else { Ok(None) }
        };
        match method {
            "live" => {
                let done = if args.has("done") { Some(args.bools("done")?[0]) } else { None };
                let decision = if args.has("decision_id") { Some(args.i64("decision_id")?) }
                               else { None };
                let action = self.live_row(&row("observations")?, flag(args, "mapping")?,
                                           one_of("reward")?, done, one_of("salience")?,
                                           decision)?;
                out.put("action", Array::from_i64(&[action.len()], &action));
            }
            "act" => {
                let action = self.act_row(&row("observations")?, flag(args, "mapping")?,
                                          flag(args, "greedy")?, one_of("temperature")?)?;
                out.put("action", Array::from_i64(&[action.len()], &action));
            }
            "_observations" => {
                let x = self.observations(&row("observations")?, flag(args, "mapping")?)?;
                out.put("x", Array::from_f64(&[1, x.len()], &x));
            }
            "say" => {
                let said = self.say(&row("observations")?)?;
                out.put("said", Array::from_f64(&[1, said.len()], &said));
            }
            "_settled" => {
                let activity = self.actor()?.free.clone();
                let x = row("x")?;
                let (drive, free) = self.settled(&x, activity.as_ref())?;
                out.put("drive", Array::from_f64(&[1, n], &drive));
                free.export("free", n, out);
            }
            "_surprised" => self.surprised(&row("x")?)?,
            "_choose" => {
                let x = row("x")?;
                let free = State::import(args, "free", n)?;
                let heat = if args.has("temperature") {
                    let t = args.f64s("temperature")?;
                    Some(if t.len() == 1 { Heat::One(t[0]) } else { Heat::Each(t) })
                } else {
                    None
                };
                let action = self.choose(&x, &row("drive")?, free, flag(args, "greedy")?, heat)?;
                out.put("action", Array::from_i64(&[action.len()], &action));
            }
            "state" => {}
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// A world's animals live on threads of their own (`rayon` across animals natively).
    #[test]
    fn a_brain_moves_between_threads() {
        fn sent<T: Send + Sync>() {}
        sent::<super::Brain>();
    }
}
