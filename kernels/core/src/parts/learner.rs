//! `cadence.learning.Learner`: the plastic efficacies and biases of a graph and the rule's last
//! half (`apply`, and `_compiled_apply` where the compiled pass serves), the free phase
//! (`free`) and a lesson's or an action's targets (`targets`, `_labels`). The configuration
//! as far as these read it, the counts and the optimizer's running moments cross whole.
//!
//! The learner reads and replaces the graph a brain holds (the Python's `learner.brain`, held
//! by its `GraphHolder`): its methods take that graph from the caller. `LearnerPart` holds the
//! two together, as `cadence.core.parts.learner` crosses them.

use super::graph::{Graph, Nudge, Rows, State, Temperature};
use crate::lesson;
use crate::np;
use crate::state::{Array, Error, Named};
use crate::{Index, View};

fn refused(text: &str) -> Error {
    Error::Refused(text.to_string())
}

/// `learning._RELATIVE_TINY`: under the projection-relative floor.
const RELATIVE_TINY: f64 = 1e-12;

/// The three phases of a lesson (`LearnedState`), every row taught.
#[derive(Clone, Debug, PartialEq)]
pub struct Learned {
    pub free: State,
    pub nudged: State,
    pub opposite: State,
}

/// A view of a run a pass only reads.
fn seen(x: &[f64]) -> View<f64> {
    View { p: x.as_ptr() as *mut f64, n0: x.len(), n1: 1, len: x.len() }
}

/// A view of a run a pass writes.
fn held(x: &mut [f64]) -> View<f64> {
    View { p: x.as_mut_ptr(), n0: x.len(), n1: 1, len: x.len() }
}

fn index(x: &[i64]) -> Index {
    Index::Wide(View { p: x.as_ptr() as *mut i64, n0: x.len(), n1: 1, len: x.len() })
}

/// `_projections`: the block each synapse's entry lies in (`layout.block_of_edge`) and the
/// range each neuron lies in.
fn projections(graph: &Graph) -> (Vec<i64>, Vec<i64>) {
    let blocks = graph.edge_index.iter()
        .map(|&x| graph.offset.partition_point(|&o| o <= x) as i64 - 1).collect();
    let ranges = (0..graph.n() as i64)
        .map(|i| graph.starts.partition_point(|&s| s <= i) as i64 - 1).collect();
    (blocks, ranges)
}

/// Each group's root mean square, `sqrt(power / count)` of `_group_power` (its squares summed
/// in index order, its members at least one).
fn group_roots(values: &[f64], group: &[i64], size: usize) -> Result<Vec<f64>, Error> {
    let (mut power, mut count) = (vec![0.0; size], vec![0.0; size]);
    let code = unsafe {
        lesson::group_power_pass(seen(values), index(group), held(&mut power), held(&mut count))
    };
    if code < 0 {
        return Err(Error::Shape("the group power's pass refused its arrays".into()));
    }
    Ok(power.iter().zip(&count).map(|(p, c)| np::sqrt(p / c)).collect())
}

/// A lesson's label for a motor slot it leaves open (`learning.OPEN`).
pub const OPEN: i64 = -1;

/// `LearnerConfig`, its fields in the dataclass's order.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub beta: f64,
    pub eta: f64,
    pub eta_bias: f64,
    pub free_steps: i64,
    pub nudged_steps: i64,
    pub tolerance: Option<f64>,
    /// `nudge == "cross_entropy"` (else `"quadratic"`).
    pub cross_entropy: bool,
    pub temperature: f64,
    pub normalize: f64,
    pub normalize_floor: f64,
    pub momentum: f64,
    pub scale_cap: f64,
    pub qualified: bool,
    pub damping: i64,
    pub surprise: bool,
    pub surprise_share: bool,
    pub normalize_relative: f64,
    pub nudged_fixed: bool,
    pub rehearsal: f64,
    pub rehearsal_capacity: i64,
    pub fast_settle: bool,
    pub compiled_moment: bool,
}

impl Config {
    fn import(s: &Named) -> Result<Config, Error> {
        let f = |k: &str| s.f64(&format!("config.{k}"));
        let i = |k: &str| s.i64(&format!("config.{k}"));
        let b = |k: &str| -> Result<bool, Error> { Ok(s.bools(&format!("config.{k}"))?[0]) };
        Ok(Config {
            beta: f("beta")?,
            eta: f("eta")?,
            eta_bias: f("eta_bias")?,
            free_steps: i("free_steps")?,
            nudged_steps: i("nudged_steps")?,
            tolerance: if s.has("config.tolerance") { Some(f("tolerance")?) } else { None },
            cross_entropy: b("cross_entropy")?,
            temperature: f("temperature")?,
            normalize: f("normalize")?,
            normalize_floor: f("normalize_floor")?,
            momentum: f("momentum")?,
            scale_cap: f("scale_cap")?,
            qualified: b("qualified")?,
            damping: i("damping")?,
            surprise: b("surprise")?,
            surprise_share: b("surprise_share")?,
            normalize_relative: f("normalize_relative")?,
            nudged_fixed: b("nudged_fixed")?,
            rehearsal: f("rehearsal")?,
            rehearsal_capacity: i("rehearsal_capacity")?,
            fast_settle: b("fast_settle")?,
            compiled_moment: b("compiled_moment")?,
        })
    }

    fn export(&self, out: &mut Named) {
        let f = |out: &mut Named, k: &str, x: f64| out.put(&format!("config.{k}"), Array::scalar_f64(x));
        let i = |out: &mut Named, k: &str, x: i64| out.put(&format!("config.{k}"), Array::scalar_i64(x));
        let b = |out: &mut Named, k: &str, x: bool| {
            out.put(&format!("config.{k}"), Array::from_bool(&[], &[x]))
        };
        f(out, "beta", self.beta);
        f(out, "eta", self.eta);
        f(out, "eta_bias", self.eta_bias);
        i(out, "free_steps", self.free_steps);
        i(out, "nudged_steps", self.nudged_steps);
        if let Some(t) = self.tolerance {
            f(out, "tolerance", t);
        }
        b(out, "cross_entropy", self.cross_entropy);
        f(out, "temperature", self.temperature);
        f(out, "normalize", self.normalize);
        f(out, "normalize_floor", self.normalize_floor);
        f(out, "momentum", self.momentum);
        f(out, "scale_cap", self.scale_cap);
        b(out, "qualified", self.qualified);
        i(out, "damping", self.damping);
        b(out, "surprise", self.surprise);
        b(out, "surprise_share", self.surprise_share);
        f(out, "normalize_relative", self.normalize_relative);
        b(out, "nudged_fixed", self.nudged_fixed);
        f(out, "rehearsal", self.rehearsal);
        i(out, "rehearsal_capacity", self.rehearsal_capacity);
        b(out, "fast_settle", self.fast_settle);
        b(out, "compiled_moment", self.compiled_moment);
    }
}

/// Labels of a lesson or an action: `(batch,)`, or `(batch, slots)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Labels {
    pub data: Vec<i64>,
    pub shape: Vec<usize>,
}

/// `np.abs(x).mean()`, the sizes written over `x`.
fn mean_size(x: &mut [f64]) -> f64 {
    x.iter_mut().for_each(|v| *v = v.abs());
    np::mean(x)
}

/// The buffers an update reuses, which hold no state: the step, and the efficacies, weights and
/// biases that go to the graph (and come back from it as its old ones). A copy starts without
/// them, and any two compare equal.
#[derive(Default)]
pub struct Rooms {
    d: Vec<f64>,
    scale: Vec<f64>,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl Clone for Rooms {
    fn clone(&self) -> Rooms {
        Rooms::default()
    }
}

impl PartialEq for Rooms {
    fn eq(&self, _: &Rooms) -> bool {
        true
    }
}

impl std::fmt::Debug for Rooms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Rooms")
    }
}

/// `Learner`: a graph's plastic parameters and the rule's step.
#[derive(Clone, Debug, PartialEq)]
pub struct Learner {
    pub config: Config,
    pub output_index: Vec<i64>,
    pub slot_sizes: Vec<i64>,
    pub plastic_synapses: Vec<bool>,
    pub plastic_neurons: Vec<bool>,
    pub reciprocal: bool,
    pub tie_groups: Option<Vec<i64>>,
    pub synapse_rate: Option<Vec<f64>>,
    pub updates: i64,
    pub contrast_updates: i64,
    pub velocity: Vec<f64>,
    pub velocity_bias: Vec<f64>,
    pub second_moment: Vec<f64>,
    pub second_moment_bias: Vec<f64>,
    /// One for an output neuron, zero elsewhere.
    pub output_mask: Vec<f64>,
    pub slot_offsets: Vec<i64>,
    /// The slot of each output neuron, -1 elsewhere, with more than one slot.
    pub output_groups: Option<Vec<i64>>,
    /// Each synapse's reciprocal partner, -1 for none.
    partner: Vec<i64>,
    paired: Vec<usize>,
    paired_reverse: Vec<usize>,
    members: Vec<usize>,
    member_groups: Vec<usize>,
    member_count: Vec<f64>,
    rooms: Rooms,
}

impl Learner {
    /// The neurons of the learner's graph.
    pub fn n(&self) -> usize {
        self.output_mask.len()
    }
    pub fn slot_count(&self) -> usize {
        self.slot_sizes.len()
    }
    /// `slot_size`: the size every slot has, or 0 where they differ.
    pub fn slot_size(&self) -> i64 {
        let first = self.slot_sizes[0];
        if self.slot_sizes.iter().all(|&k| k == first) { first } else { 0 }
    }

    /// The learner of the state `s` (its own entries; a `graph.*` there is the graph's) on
    /// `graph`.
    pub fn import(s: &Named, graph: &Graph) -> Result<Learner, Error> {
        let (n, e) = (graph.n(), graph.synapses());
        let opt_i = |k: &str| -> Result<Option<Vec<i64>>, Error> {
            if s.has(k) { Ok(Some(s.i64s(k)?)) } else { Ok(None) }
        };
        let mut learner = Learner {
            config: Config::import(s)?,
            output_index: s.i64s("outputs")?,
            slot_sizes: s.i64s("slot_sizes")?,
            plastic_synapses: s.bools("plastic_synapses")?,
            plastic_neurons: s.bools("plastic_neurons")?,
            reciprocal: s.bools("reciprocal")?[0],
            tie_groups: opt_i("tie_groups")?,
            synapse_rate: if s.has("synapse_rate") { Some(s.f64s("synapse_rate")?) } else { None },
            updates: s.i64("updates")?,
            contrast_updates: s.i64("contrast_updates")?,
            velocity: s.f64s("velocity")?,
            velocity_bias: s.f64s("velocity_bias")?,
            second_moment: s.f64s("second_moment")?,
            second_moment_bias: s.f64s("second_moment_bias")?,
            output_mask: vec![0.0; n],
            slot_offsets: vec![],
            output_groups: None,
            partner: vec![-1; e],
            paired: vec![],
            paired_reverse: vec![],
            members: vec![],
            member_groups: vec![],
            member_count: vec![],
            rooms: Rooms::default(),
        };
        learner.derive(graph)?;
        Ok(learner)
    }

    /// The learner's own entries, after its graph's (`graph.*`, which the caller writes).
    pub fn export(&self, out: &mut Named) {
        self.config.export(out);
        let i = |x: &[i64]| Array::from_i64(&[x.len()], x);
        let f = |x: &[f64]| Array::from_f64(&[x.len()], x);
        out.put("outputs", i(&self.output_index));
        out.put("slot_sizes", i(&self.slot_sizes));
        out.put("plastic_synapses", Array::from_bool(&[self.plastic_synapses.len()],
                                                     &self.plastic_synapses));
        out.put("plastic_neurons", Array::from_bool(&[self.plastic_neurons.len()],
                                                    &self.plastic_neurons));
        out.put("reciprocal", Array::from_bool(&[], &[self.reciprocal]));
        if let Some(t) = &self.tie_groups {
            out.put("tie_groups", i(t));
        }
        if let Some(r) = &self.synapse_rate {
            out.put("synapse_rate", f(r));
        }
        out.put("updates", Array::scalar_i64(self.updates));
        out.put("contrast_updates", Array::scalar_i64(self.contrast_updates));
        out.put("velocity", f(&self.velocity));
        out.put("velocity_bias", f(&self.velocity_bias));
        out.put("second_moment", f(&self.second_moment));
        out.put("second_moment_bias", f(&self.second_moment_bias));
    }

    /// `__post_init__`: the outputs' mask and slots, the reciprocal pairs and the tied groups,
    /// with its refusals.
    fn derive(&mut self, graph: &Graph) -> Result<(), Error> {
        let (n, e) = (graph.n(), graph.synapses());
        let o = &self.output_index;
        if o.is_empty() || o.iter().any(|&i| i < 0 || i as usize >= n) {
            return Err(refused("outputs must contain neuron indices of the connectome"));
        }
        let mut sorted = o.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(refused("outputs must contain distinct neuron indices"));
        }
        if self.slot_sizes.is_empty() || self.slot_sizes.iter().any(|&k| k < 1)
            || self.slot_sizes.iter().sum::<i64>() != o.len() as i64
        {
            return Err(refused("slot sizes must be positive and add up to the output neurons"));
        }
        if self.plastic_synapses.len() != e || self.plastic_neurons.len() != n
            || [&self.velocity, &self.second_moment].iter().any(|x| x.len() != e)
            || [&self.velocity_bias, &self.second_moment_bias].iter().any(|x| x.len() != n)
        {
            return Err(refused("the learner's masks and moments fit the graph"));
        }
        if let Some(r) = &self.synapse_rate {
            if r.len() != e || r.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(refused("synapse_rate must be a finite nonnegative vector"));
            }
        }
        for &i in o {
            self.output_mask[i as usize] = 1.0;
        }
        let mut offset = 0i64;
        self.slot_offsets = self.slot_sizes.iter().map(|&k| {
            let at = offset;
            offset += k;
            at
        }).collect();
        if self.slot_count() > 1 {
            let mut groups = vec![-1i64; n];
            let mut at = 0usize;
            for (slot, &size) in self.slot_sizes.iter().enumerate() {
                for _ in 0..size {
                    groups[o[at] as usize] = slot as i64;
                    at += 1;
                }
            }
            self.output_groups = Some(groups);
        }
        // the reciprocal pairs: the synapse whose key is this one's reverse key
        if self.reciprocal && e > 0 {
            let (pre, post) = (&graph.pre, &graph.post);
            let key: Vec<i64> = (0..e).map(|k| post[k] * n as i64 + pre[k]).collect();
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by_key(|&k| key[k]);
            let sorted: Vec<i64> = order.iter().map(|&k| key[k]).collect();
            for k in 0..e {
                let want = pre[k] * n as i64 + post[k];
                let hit = sorted.partition_point(|&x| x < want).min(e - 1);
                if sorted[hit] == want {
                    self.partner[k] = order[hit] as i64;
                }
            }
        }
        self.paired = (0..e).filter(|&k| self.partner[k] >= 0).collect();
        self.paired_reverse = self.paired.iter().map(|&k| self.partner[k] as usize).collect();
        if self.paired_reverse.iter().zip(&self.paired).any(|(&r, &k)| self.partner[r] != k as i64) {
            return Err(refused("reciprocal learning needs unique pairs; merge with from_synapses first"));
        }
        if let Some(groups) = &self.tie_groups {
            if groups.len() != e || groups.iter().any(|&g| g < -1) {
                return Err(refused("tie_groups must be an integer per synapse, -1 for untied"));
            }
            let mut members: Vec<usize> = (0..e).filter(|&k| groups[k] >= 0).collect();
            let partners: Vec<usize> = members.iter().filter(|&&k| self.partner[k] >= 0)
                .map(|&k| self.partner[k] as usize).collect();
            members.extend(partners);
            members.sort_unstable();
            members.dedup();
            // the members joined over their pairs and their groups (any labels of the classes)
            let mut parent: std::collections::HashMap<usize, usize> =
                members.iter().map(|&k| (k, k)).collect();
            fn root(parent: &mut std::collections::HashMap<usize, usize>, mut i: usize) -> usize {
                while parent[&i] != i {
                    let up = parent[&parent[&i]];
                    parent.insert(i, up);
                    i = up;
                }
                i
            }
            let mut representatives: std::collections::HashMap<i64, usize> = Default::default();
            for &k in &members {
                if self.partner[k] >= 0 {
                    let (a, b) = (root(&mut parent, k), root(&mut parent, self.partner[k] as usize));
                    parent.insert(a, b);
                }
                let g = groups[k];
                if g >= 0 {
                    let rep = *representatives.entry(g).or_insert(k);
                    let (a, b) = (root(&mut parent, k), root(&mut parent, rep));
                    parent.insert(a, b);
                }
            }
            let roots: Vec<usize> = members.iter().map(|&k| root(&mut parent, k)).collect();
            let mut labels = roots.clone();
            labels.sort_unstable();
            labels.dedup();
            self.member_groups = roots.iter().map(|r| labels.partition_point(|x| x < r)).collect();
            self.member_count = vec![0.0; labels.len()];
            for &g in &self.member_groups {
                self.member_count[g] += 1.0;
            }
            self.members = members;
        }
        Ok(())
    }

    // -- phases

    /// `free(drive, warm=)`: the net's own answer on `graph`, settled with the input alone.
    pub fn free(&self, graph: &Graph, drive: &Rows, warm: Option<&State>)
                -> Result<State, Error> {
        let cfg = &self.config;
        if cfg.qualified {
            let tolerance = cfg.tolerance
                .ok_or_else(|| refused("qualified learning requires a finite residual tolerance"))?;
            let phase = graph.equilibrate(drive, cfg.free_steps, 32, tolerance, warm, None, None,
                                          cfg.damping, false)?;
            if !phase.qualified() {
                return Err(refused("the free phase did not qualify"));
            }
            // `replace(phase.state, steps=phase.steps)`: a new state, without the record
            return Ok(State { steps: phase.steps, published: false, ..phase.state });
        }
        graph.settle_batch(drive, cfg.free_steps, warm, None, None, cfg.tolerance)
    }

    /// `_labels(labels, open=)`: validated class labels.
    pub fn labels(&self, labels: &Labels, open: bool) -> Result<Labels, Error> {
        let slots = self.slot_count();
        let ndim = labels.shape.len();
        if slots > 1 {
            if ndim != 2 || labels.shape[1] != slots {
                return Err(refused("a slotted learner takes labels shaped (batch, slots)"));
            }
        } else if ndim != 1 {
            return Err(refused("labels must have shape (batch,) for a single output slot"));
        }
        let width = if ndim == 2 { labels.shape[1] } else { 1 };
        for (k, &x) in labels.data.iter().enumerate() {
            let size = self.slot_sizes[if slots > 1 { k % width } else { 0 }];
            if x >= size {
                return Err(refused("a label lies outside its output slot's choices"));
            }
        }
        if labels.data.iter().any(|&x| x < 0) {
            if !open || slots == 1 || labels.data.iter().any(|&x| x < OPEN) {
                return Err(refused("a label lies outside its output slot's choices"));
            }
            if labels.data.chunks(width).any(|row| row.iter().all(|&x| x < 0)) {
                return Err(refused("a lesson's row labels at least one slot: every slot is open"));
            }
        }
        Ok(labels.clone())
    }

    /// `targets(labels)`: each row's target activation, `(batch, n)`.
    pub fn targets(&self, labels: &Labels) -> Result<Vec<f64>, Error> {
        let labels = self.labels(labels, true)?;
        let n = self.n();
        let batch = labels.shape.first().copied().unwrap_or(0);
        let mut target = vec![0.0f64; batch * n];
        let slots = self.slot_count();
        for b in 0..batch {
            for k in 0..slots {
                let x = labels.data[b * slots + k];
                if x >= 0 {
                    let neuron = self.output_index[(x + self.slot_offsets[k]) as usize] as usize;
                    target[b * n + neuron] = 1.0;
                }
            }
        }
        Ok(target)
    }

    // -- the rule's step

    /// `held()`: the synapses a rate of zero holds, and the neurons whose every inbound synapse
    /// it holds (None for each where there are none).
    fn held_neurons(&self, graph: &Graph) -> Option<Vec<bool>> {
        let rate = self.synapse_rate.as_ref()?;
        if rate.is_empty() || rate.iter().all(|&r| r > 0.0) {
            return None;
        }
        let n = self.n();
        let (mut inbound, mut stopped) = (vec![0usize; n], vec![0usize; n]);
        for (k, &j) in graph.post.iter().enumerate() {
            inbound[j as usize] += 1;
            if rate[k] == 0.0 {
                stopped[j as usize] += 1;
            }
        }
        let neurons: Vec<bool> = (0..n).map(|i| inbound[i] > 0 && stopped[i] == inbound[i]).collect();
        neurons.iter().any(|&x| x).then_some(neurons)
    }

    /// `apply(delta_scale, delta_bias)`: masks, tying, bounds; then `graph` becomes the graph
    /// of the new parameters (one generation on). Returns the report (`scale_step`,
    /// `bias_step`).
    pub fn apply(&mut self, graph: &mut Graph, delta_scale: &[f64], delta_bias: &[f64])
                 -> Result<Vec<(&'static str, f64)>, Error> {
        let (n, e) = (self.n(), graph.synapses());
        if delta_scale.len() != e || delta_bias.len() != n
            || !delta_scale.iter().chain(delta_bias).all(|x| x.is_finite())
        {
            return Err(refused("steps must be finite vectors, one per synapse and one per neuron"));
        }
        let all_trainable = self.plastic_synapses.iter().all(|&x| x);
        if all_trainable && self.synapse_rate.is_none() && self.members.is_empty() && e > 0 {
            return self.compiled_apply(graph, delta_scale, delta_bias);
        }
        let rooms = &mut self.rooms;
        let mut d = std::mem::take(&mut rooms.d);
        d.clear();
        d.extend_from_slice(delta_scale);
        if !all_trainable {
            for (x, &plastic) in d.iter_mut().zip(&self.plastic_synapses) {
                if !plastic {
                    *x = 0.0;
                }
            }
        }
        if let Some(rate) = &self.synapse_rate {
            for k in 0..e {
                d[k] *= rate[k];
            }
        }
        if !self.paired.is_empty() {
            let mean: Vec<f64> = self.paired.iter().zip(&self.paired_reverse)
                .map(|(&k, &r)| 0.5 * (d[k] + d[r])).collect();
            for (&k, m) in self.paired.iter().zip(mean) {
                d[k] = m;
            }
        }
        if !self.members.is_empty() {
            let mut total = vec![0.0f64; self.member_count.len()];
            for (&k, &g) in self.members.iter().zip(&self.member_groups) {
                total[g] += d[k];
            }
            let means: Vec<f64> = total.iter().zip(&self.member_count)
                .map(|(t, c)| t / np::maximum(*c, 1.0)).collect();
            for (&k, &g) in self.members.iter().zip(&self.member_groups) {
                d[k] = means[g];
            }
        }
        if !all_trainable {
            for (x, &plastic) in d.iter_mut().zip(&self.plastic_synapses) {
                if !plastic {
                    *x = 0.0;
                }
            }
        }
        if !d.iter().all(|x| x.is_finite()) {
            self.rooms.d = d;
            return Err(refused("scaled and tied steps must remain finite"));
        }
        let cap = self.config.scale_cap;
        let mut scale = std::mem::take(&mut self.rooms.scale);
        scale.clear();
        scale.extend((0..e).map(|k| {
            let s = graph.efficacy[k] + d[k];
            if self.plastic_synapses[k] { np::clip(s, -cap, cap) } else { s }
        }));
        let held = self.held_neurons(graph);
        let step_bias: Vec<f64> = (0..n).map(|i| {
            let moves = self.plastic_neurons[i] && !held.as_ref().is_some_and(|h| h[i]);
            if moves { delta_bias[i] } else { 0.0 }
        }).collect();
        let mut bias = std::mem::take(&mut self.rooms.bias);
        bias.clear();
        bias.extend(graph.bias.iter().zip(&step_bias).map(|(b, x)| b + x));
        let done = graph.set_parameters(Some(&mut scale), Some(&mut bias));
        // the buffers hold the graph's old efficacies and biases now (or, refused, the new)
        self.rooms.scale = scale;
        self.rooms.bias = bias;
        if let Err(e) = done {
            self.rooms.d = d;
            return Err(e);
        }
        self.updates += 1;
        let scale_step = if d.is_empty() { 0.0 } else { mean_size(&mut d) };
        self.rooms.d = d;
        Ok(vec![("scale_step", scale_step), ("bias_step", mean_size(&mut step_bias.clone()))])
    }

    // -- a lesson's step (`step` with targets)

    /// `soft_targets(p)`: a distribution over the outputs per row (`rows` of `outputs`, each
    /// slot's summing to one) as the target activation over every neuron, `(rows, n)`.
    pub fn soft_targets(&self, p: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, Error> {
        if cols != self.output_index.len() || p.len() != rows * cols {
            return Err(refused(
                "targets must have shape (batch, outputs): one probability per output neuron"));
        }
        if !p.iter().all(|x| x.is_finite() && *x >= 0.0) {
            return Err(refused("target probabilities must be finite and nonnegative"));
        }
        for row in p.chunks_exact(cols.max(1)) {
            for (k, &start) in self.slot_offsets.iter().enumerate() {
                let stop = self.slot_offsets.get(k + 1).map_or(cols, |&s| s as usize);
                let total = np::sum(&row[start as usize..stop]);
                if !((total - 1.0).abs() <= 1e-6) {
                    return Err(refused("target probabilities must sum to one over each output slot"));
                }
            }
        }
        let n = self.n();
        let mut target = vec![0.0; rows * n];
        for r in 0..rows {
            for (j, &k) in self.output_index.iter().enumerate() {
                target[r * n + k as usize] = p[r * cols + j];
            }
        }
        Ok(target)
    }

    /// `nudge_for(target, beta, weight, apical_gain)`: the outputs pulled toward `target`.
    fn nudge_for(&self, target: &[f64], batch: usize, beta: f64, weight: Option<&[f64]>,
                 held: Option<&Vec<f64>>, pools: usize) -> Nudge {
        let cfg = &self.config;
        Nudge {
            target: Rows { data: target.to_vec(), rows: batch, cols: self.n(), ndim: 2 },
            mask: self.output_mask.clone(),
            beta: beta / self.slot_count() as f64,
            temperature: cfg.cross_entropy.then_some(Temperature::One(cfg.temperature)),
            weight: weight.map(|w| w.to_vec()),
            groups: self.output_groups.clone(),
            anchor: None,
            anchor_gain: None,
            apical_gain: held.map(|g| Rows { data: g.clone(), rows: batch, cols: pools, ndim: 2 }),
        }
    }

    /// `step(drive, None, warm=, weight=, targets=)`, the lesson of `target` (`(batch, n)`, as
    /// `soft_targets` makes it) on the finite path: the free phase from `warm` and its check,
    /// the two nudged phases under the apical gain the free phase set, and the update from
    /// their contrast; `graph` becomes the update's graph. Every row is taught (targets have
    /// no answer to agree with). Qualified teaching is not held here.
    pub fn lesson(&mut self, graph: &mut Graph, drive: &Rows, warm: Option<&State>,
                  target: &[f64], weight: Option<&[f64]>) -> Result<Learned, Error> {
        let n = self.n();
        let batch = target.len() / n.max(1);
        if drive.ndim != 2 || drive.rows != batch {
            return Err(refused("drive and labels must have the same batch size"));
        }
        let cfg = self.config.clone();
        if cfg.qualified {
            return Err(Error::Missing("qualified teaching in the core's lesson".into()));
        }
        let mut free = self.free(graph, drive, warm)?;
        // `_diagnostic`: the free phase's check keeps its record on the state, as the nudged
        // phases read it
        graph.check(drive, &mut free, None, None)?;
        let held = graph.apical_gain(drive, &free)?;
        let pools = graph.pools();
        let pull = self.nudge_for(target, batch, cfg.beta, weight, held.as_ref(), pools);
        let push = self.nudge_for(target, batch, -cfg.beta, weight, held.as_ref(), pools);
        let tolerance = if cfg.nudged_fixed { None } else { cfg.tolerance };
        let nudged = graph.settle_batch(drive, cfg.nudged_steps, Some(&free), None, Some(&pull),
                                        tolerance)?;
        let opposite = graph.settle_batch(drive, cfg.nudged_steps, Some(&free), None, Some(&push),
                                          tolerance)?;
        self.update(graph, &free, &nudged, &opposite)?;
        Ok(Learned { free, nudged, opposite })
    }

    /// `contrast(free, nudged, opposite)`: each synapse's and each neuron's contrast of the two
    /// nudged phases over their span, the mean over the rows (`block_contrast` where the graph
    /// is in blocks and has more than four synapses a neuron).
    pub fn contrast(&self, graph: &Graph, plus: &State, minus: &State)
                    -> Result<(Vec<f64>, Vec<f64>), Error> {
        let n = self.n();
        let batch = plus.batch;
        if batch == 0 || minus.batch != batch || plus.activation.len() != batch * n
            || minus.activation.len() != batch * n
        {
            return Err(refused("all phases must have the same nonempty (batch, neurons) shape"));
        }
        let span = 2.0 * self.config.beta;
        let (s_plus, s_minus) = (&plus.activation, &minus.activation);
        let e = graph.synapses();
        let mut edges = vec![0.0; e];
        if e > 4 * n {
            // `block_contrast`: each block's `a+ᵀ (b+ - b-)`, plus `(a+ - a-)ᵀ b-` where the
            // block's sources moved, then each synapse's entry over `batch * span` (`gather`)
            let size = *graph.offset.last().unwrap_or(&0) as usize;
            let mut g = vec![0.0; size];
            let at = |s: &[f64], b: usize, i: usize| s[b * n + i];
            for k in 0..graph.pair_pre.len() {
                let (a, bb) = (graph.pair_pre[k] as usize, graph.pair_post[k] as usize);
                let (a0, a1) = (graph.starts[a] as usize, graph.starts[a + 1] as usize);
                let (b0, b1) = (graph.starts[bb] as usize, graph.starts[bb + 1] as usize);
                let (rows, cols) = (a1 - a0, b1 - b0);
                let place = &mut g[graph.offset[k] as usize..graph.offset[k + 1] as usize];
                let moved = (0..batch).any(|b| (a0..a1).any(|i| at(s_plus, b, i) != at(s_minus, b, i)));
                for i in 0..rows {
                    let ap: Vec<f64> = (0..batch).map(|b| at(s_plus, b, a0 + i)).collect();
                    let ad: Vec<f64> = (0..batch).map(|b| at(s_plus, b, a0 + i) - at(s_minus, b, a0 + i)).collect();
                    for j in 0..cols {
                        let bd: Vec<f64> = (0..batch).map(|b| at(s_plus, b, b0 + j) - at(s_minus, b, b0 + j)).collect();
                        let mut v = if batch == 1 { ap[0] * bd[0] } else { np::dot(&ap, &bd) };
                        if moved {
                            let bm: Vec<f64> = (0..batch).map(|b| at(s_minus, b, b0 + j)).collect();
                            v += if batch == 1 { ad[0] * bm[0] } else { np::dot(&ad, &bm) };
                        }
                        place[i * cols + j] = v;
                    }
                }
            }
            let code = unsafe {
                let index = Index::Wide(View { p: graph.edge_index.as_ptr() as *mut i64, n0: e,
                                               n1: 1, len: e });
                lesson::pick_pass(seen(&g), index, batch as f64 * span, held(&mut edges))
            };
            if code < 0 {
                return Err(Error::Shape("the gather refused its arrays".into()));
            }
        } else {
            for (k, edge) in edges.iter_mut().enumerate() {
                let (i, j) = (graph.pre[k] as usize, graph.post[k] as usize);
                // `products.mean(axis=0)`: the rows added in order to +0.0
                let mut total = 0.0;
                for b in 0..batch {
                    let (a_plus, a_minus) = (s_plus[b * n + i], s_minus[b * n + i]);
                    let (b_plus, b_minus) = (s_plus[b * n + j], s_minus[b * n + j]);
                    total += a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus;
                }
                *edge = total / batch as f64 / span;
            }
        }
        let moved: Vec<f64> = s_plus.iter().zip(s_minus).map(|(p, m)| p - m).collect();
        let neurons = np::sum_rows(&moved, n).iter().map(|s| s / batch as f64 / span).collect();
        Ok((edges, neurons))
    }

    /// `update(free, nudged, opposite)`: the contrast's step through the optimizer's moments
    /// (`_compiled_step`), then `apply`; the moments as they were where it is refused.
    pub fn update(&mut self, graph: &mut Graph, free: &State, nudged: &State, opposite: &State)
                  -> Result<Vec<(&'static str, f64)>, Error> {
        if free.batch != nudged.batch || free.activation.len() != nudged.activation.len() {
            return Err(refused("all phases must have the same nonempty (batch, neurons) shape"));
        }
        let before = (self.velocity.clone(), self.velocity_bias.clone(),
                      self.second_moment.clone(), self.second_moment_bias.clone());
        let found = self.updated(graph, nudged, opposite);
        if found.is_err() {
            (self.velocity, self.velocity_bias, self.second_moment, self.second_moment_bias) =
                before;
        }
        found
    }

    fn updated(&mut self, graph: &mut Graph, nudged: &State, opposite: &State)
               -> Result<Vec<(&'static str, f64)>, Error> {
        let (synapse_term, neuron_term) = self.contrast(graph, nudged, opposite)?;
        if !synapse_term.iter().chain(&neuron_term).all(|x| x.is_finite()) {
            return Err(refused("phase contrasts must be finite"));
        }
        if synapse_term.is_empty() {
            return Err(Error::Missing("a lesson of a graph without synapses in the core".into()));
        }
        let count = self.contrast_updates + 1;
        let held = (self.velocity.clone(), self.second_moment.clone(),
                    self.velocity_bias.clone(), self.second_moment_bias.clone());
        let (delta_scale, delta_bias) = self.compiled_step(graph, &synapse_term, &neuron_term,
                                                           count)?;
        self.hold_moments(graph, held);
        if ![&self.velocity, &self.velocity_bias, &self.second_moment, &self.second_moment_bias]
            .iter().all(|m| m.iter().all(|x| x.is_finite()))
        {
            return Err(refused("optimizer moments must remain finite"));
        }
        let report = self.apply(graph, &delta_scale, &delta_bias)?;
        self.contrast_updates += 1;
        Ok(report)
    }

    /// `_hold_moments(before)`: the moments of the synapses (and the neurons) a rate of zero
    /// holds stay as they were.
    fn hold_moments(&mut self, graph: &Graph, before: (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) {
        let Some(rate) = &self.synapse_rate else { return };
        if rate.is_empty() || rate.iter().all(|&r| r > 0.0) {
            return;
        }
        let zero: Vec<bool> = rate.iter().map(|&r| r == 0.0).collect();
        for (k, &z) in zero.iter().enumerate() {
            if z {
                self.velocity[k] = before.0[k];
                self.second_moment[k] = before.1[k];
            }
        }
        if let Some(neurons) = self.held_neurons(graph) {
            for (i, &h) in neurons.iter().enumerate() {
                if h {
                    self.velocity_bias[i] = before.2[i];
                    self.second_moment_bias[i] = before.3[i];
                }
            }
        }
    }

    /// `_compiled_step(raw_synapse, raw_neuron, count)`: the moments, the normalization and
    /// the rates of a contrast in the compiled passes (`optimizer_step`, `relative_floor`).
    fn compiled_step(&mut self, graph: &Graph, raw_synapse: &[f64], raw_neuron: &[f64],
                     count: i64) -> Result<(Vec<f64>, Vec<f64>), Error> {
        let cfg = self.config.clone();
        let (m, rho) = (cfg.momentum, cfg.normalize);
        // Python's float ** int: the C library's pow
        let momentum_correction = if m > 0.0 { 1.0 - np::pow(m, count as f64) } else { 1.0 };
        let rho_correction = if rho > 0.0 { 1.0 - np::pow(rho, count as f64) } else { 1.0 };
        let mut neuron_term = raw_neuron.to_vec();
        if m > 0.0 {
            for (i, v) in self.velocity_bias.iter_mut().enumerate() {
                *v = m * *v + (1.0 - m) * raw_neuron[i];
                neuron_term[i] = *v / momentum_correction;
            }
        }
        let (group, ranges) = projections(graph);
        let e = raw_synapse.len();
        let (has_momentum, has_normalize) = (m > 0.0, rho > 0.0);
        let mut velocity = if has_momentum { vec![0.0; e] } else { self.velocity.clone() };
        let mut second = if has_normalize { vec![0.0; e] } else { self.second_moment.clone() };
        let mut step = vec![0.0; e];
        let code = unsafe {
            if !(has_normalize && cfg.normalize_relative != 0.0) {
                lesson::plain_pass(seen(raw_synapse), seen(&self.velocity),
                                   seen(&self.second_moment), m, 1.0 - m, momentum_correction,
                                   rho, 1.0 - rho, rho_correction, has_momentum as i32,
                                   has_normalize as i32, cfg.eta, cfg.normalize_floor,
                                   held(&mut velocity), held(&mut second), held(&mut step))
            } else {
                let (mut term, mut root) = (vec![0.0; e], vec![0.0; e]);
                let mut code = lesson::moments_pass(
                    seen(raw_synapse), seen(&self.velocity), seen(&self.second_moment), m,
                    1.0 - m, momentum_correction, rho, 1.0 - rho, rho_correction,
                    has_momentum as i32, held(&mut velocity), held(&mut second), held(&mut term),
                    held(&mut root));
                if code >= 0 {
                    let groups = group.iter().max().map_or(0, |&g| g as usize + 1);
                    let group_root = group_roots(&root, &group, groups)?;
                    code = lesson::step_pass(seen(&term), seen(&root), index(&group),
                                             seen(&group_root), cfg.eta, cfg.normalize_relative,
                                             RELATIVE_TINY, held(&mut step));
                }
                code
            }
        };
        if code < 0 {
            return Err(Error::Shape("the optimizer's pass refused its arrays".into()));
        }
        if has_momentum {
            self.velocity = velocity;
        }
        if has_normalize {
            self.second_moment = second;
            for (i, v) in self.second_moment_bias.iter_mut().enumerate() {
                *v = rho * *v + (1.0 - rho) * (raw_neuron[i] * raw_neuron[i]);
            }
            let root: Vec<f64> = self.second_moment_bias.iter()
                .map(|v| np::sqrt(v / rho_correction)).collect();
            let rms_bias = if cfg.normalize_relative != 0.0 {
                // `_relative_divisor`: `relative_floor` over each neuron's range
                let size = ranges.iter().max().map_or(0, |&g| g as usize + 1);
                let group_root = group_roots(&root, &ranges, size)?;
                let mut out = vec![0.0; root.len()];
                let code = unsafe {
                    lesson::divisor_pass(seen(&root), index(&ranges), seen(&group_root),
                                         cfg.normalize_relative, RELATIVE_TINY, held(&mut out))
                };
                if code < 0 {
                    return Err(Error::Shape("the divisor's pass refused its arrays".into()));
                }
                out
            } else {
                root.iter().map(|r| r + cfg.normalize_floor).collect()
            };
            for (t, r) in neuron_term.iter_mut().zip(&rms_bias) {
                *t /= r;
            }
        }
        let delta_bias = neuron_term.iter().map(|t| cfg.eta_bias * t).collect();
        Ok((step, delta_bias))
    }

    /// The efficacies the graph held before the latest update (`apply`), which the update
    /// replaced; empty before the first.
    pub fn previous_efficacy(&self) -> &[f64] {
        &self.rooms.scale
    }

    /// `_compiled_apply`: the pair tying, the bound and the effective weights in one pass
    /// (`lesson::tied_pass`), which the graph takes as they are.
    fn compiled_apply(&mut self, graph: &mut Graph, delta_scale: &[f64], delta_bias: &[f64])
                      -> Result<Vec<(&'static str, f64)>, Error> {
        let (n, e) = (self.n(), graph.synapses());
        let rooms = &mut self.rooms;
        for room in [&mut rooms.d, &mut rooms.scale, &mut rooms.weights] {
            room.resize(e, 0.0); // every entry is written by the pass
        }
        let (d, scale, weights) = (&mut rooms.d, &mut rooms.scale, &mut rooms.weights);
        let mut report = [0i64; 3];
        let seen = |x: &[f64]| View { p: x.as_ptr() as *mut f64, n0: x.len(), n1: 1, len: x.len() };
        let held = |x: &mut [f64]| View { p: x.as_mut_ptr(), n0: x.len(), n1: 1, len: x.len() };
        let code = unsafe {
            let partner = Index::Wide(View { p: self.partner.as_ptr() as *mut i64, n0: e, n1: 1,
                                             len: e });
            lesson::tied_pass(seen(delta_scale), partner, seen(&graph.efficacy),
                              seen(&graph.gain_pre), self.config.scale_cap, held(d),
                              held(scale), held(weights),
                              View { p: report.as_mut_ptr(), n0: 3, n1: 1, len: 3 })
        };
        if code < 0 {
            return Err(Error::Shape("the tied pass refused its arrays".into()));
        }
        if report[0] != 0 {
            return Err(refused("scaled and tied steps must remain finite"));
        }

        let mut step_bias: Vec<f64> = (0..n)
            .map(|i| if self.plastic_neurons[i] { delta_bias[i] } else { 0.0 }).collect();
        let bias = &mut rooms.bias;
        bias.clear();
        bias.extend(graph.bias.iter().zip(&step_bias).map(|(b, x)| b + x));
        // the graph takes the new efficacies, weights and biases; the rooms its old ones
        graph.update(scale, weights, bias, (report[1] == 0, report[2] == 0))?;
        self.updates += 1;
        Ok(vec![("scale_step", mean_size(d)), ("bias_step", mean_size(&mut step_bias))])
    }
}

/// A learner with the graph it reads and replaces, as `cadence.core.parts.learner` crosses
/// them: the graph's entries (`graph.*`), then the learner's.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnerPart {
    pub graph: Graph,
    pub learner: Learner,
}

impl LearnerPart {
    pub fn import(s: &Named) -> Result<LearnerPart, Error> {
        let graph = Graph::import(&s.under("graph"))?;
        let learner = Learner::import(s, &graph)?;
        Ok(LearnerPart { graph, learner })
    }

    pub fn export(&self, out: &mut Named) {
        let mut graph = Named::new();
        self.graph.export(&mut graph);
        out.put_under("graph", &graph);
        self.learner.export(out);
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let n = self.learner.n();
        match method {
            "free" => {
                let drive = Rows::of(args, "drive")?;
                let warm = if args.has("warm.v") { Some(State::import(args, "warm", n)?) }
                           else { None };
                self.learner.free(&self.graph, &drive, warm.as_ref())?.export("state", n, out);
            }
            "targets" => {
                let labels = labels_of(args, "labels")?;
                let target = self.learner.targets(&labels)?;
                let batch = labels.shape.first().copied().unwrap_or(0);
                out.put("target", Array::from_f64(&[batch, n], &target));
            }
            "apply" => {
                let report = self.learner.apply(&mut self.graph, &args.f64s("delta_scale")?,
                                                &args.f64s("delta_bias")?)?;
                for (name, value) in report {
                    out.put(&format!("report.{name}"), Array::scalar_f64(value));
                }
            }
            "state" => {} // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

/// Labels as named arrays: integers, or refused as the Python refuses any other kind.
pub fn labels_of(args: &Named, name: &str) -> Result<Labels, Error> {
    let shape = args.shape(name)?;
    let data = args.i64s(name)
        .map_err(|_| refused("labels must be integer class indices"))?;
    Ok(Labels { data, shape })
}
