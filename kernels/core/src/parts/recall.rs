//! `cadence.stream.Recall`: the recall memory with its heads, as a Wild moment runs it on the
//! host. The read (`stimulate`: every head's address and attention, the later heads taking of
//! each slot what the heads before left, the position terms), the write (`witness`), the
//! reward's eligibility (`contrast_rows`) and its step (`propose`, `accept`): the Python's
//! arithmetic in its order, NumPy's reductions and products through `crate::np`.
//!
//! The memory the core holds is Wild's (`cadence.core.parts.recall`): one event per slot, no
//! command in a slot, no read of when, no gains on tokens, the cortex as it settled and no slot
//! strengths. Its shapes are its state's: the slots per stream, the heads and the map widths.

use crate::np;
use crate::state::{Array, Error, Named};

// the memory's constants (`Recall`): the present event's weight, the address lesson's
// temperature, the relative buckets either side, the reward's position rate, the slope's age
const HEARD: f64 = 1.0;
const ADDRESS_TEMPERATURE: f64 = 0.1;
const RELATIVE: i64 = 8;
const REWARD_RATE: f64 = 0.1;
const SLOPE: f64 = 8.0;

const FLOATS: [&str; 8] = ["amplitude", "temperature", "recency", "address", "address_reward",
                           "position_reward", "position_rate", "slope"];
const INTEGERS: [&str; 3] = ["capacity", "heads", "position"];
const SWITCHES: [&str; 5] = ["next", "boundary", "anchor", "relative", "drive_gain"];
const KINDS: [&str; 6] = ["address", "position", "anchor", "relative", "slope", "grid"];
const ADDRESS: usize = 0;
const POSITION: usize = 1;
const ANCHOR: usize = 2;
const RELATIVES: usize = 3;
const SLOPES: usize = 4;
const GRID: usize = 5;
const MOMENT: [&str; 5] = ["event", "cortex", "address", "key", "inputs"];

fn refused(why: &str) -> Error {
    crate::state::refused(why)
}

/// One boolean named `name`.
fn flag(s: &Named, name: &str) -> Result<bool, Error> {
    let v = s.bools(name)?;
    (v.len() == 1).then(|| v[0]).ok_or_else(|| refused(name))
}

/// A float array and its shape, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct F {
    pub shape: Vec<usize>,
    pub v: Vec<f64>,
}

impl F {
    fn read(s: &Named, name: &str) -> Result<F, Error> {
        let a = s.get(name)?;
        Ok(F { shape: a.shape.clone(), v: a.f64s().ok_or_else(|| refused(name))? })
    }
    fn zeros(shape: &[usize]) -> F {
        F { shape: shape.to_vec(), v: vec![0.0; shape.iter().product()] }
    }
    fn array(&self) -> Array {
        Array::from_f64(&self.shape, &self.v)
    }
    fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    /// A two-dimensional array of `cols` columns, or refused as `name`.
    fn matrix(self, cols: usize, name: &str) -> Result<F, Error> {
        if self.shape.len() == 2 && self.shape[1] == cols {
            Ok(self)
        } else {
            Err(refused(&format!("{name} is one row of {cols} values per stream")))
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct B {
    pub shape: Vec<usize>,
    pub v: Vec<bool>,
}

impl B {
    fn read(s: &Named, name: &str) -> Result<B, Error> {
        let a = s.get(name)?;
        Ok(B { shape: a.shape.clone(), v: a.bools().ok_or_else(|| refused(name))? })
    }
    fn array(&self) -> Array {
        Array::from_bool(&self.shape, &self.v)
    }
}

/// A head's map of the moment `(M_x, M_c, m)`: `x M_x + c M_c + m` (`_term`). The slope's
/// maps are one number each (`M_x` a vector).
#[derive(Clone, Debug, PartialEq)]
pub struct Maps {
    pub x: F,
    pub c: F,
    pub b: F,
}

impl Maps {
    fn read(s: &Named, prefix: &str, width: usize, association: usize, cols: Option<usize>)
            -> Result<Maps, Error> {
        let maps = Maps {
            x: F::read(s, &format!("{prefix}.x"))?,
            c: F::read(s, &format!("{prefix}.c"))?,
            b: F::read(s, &format!("{prefix}.b"))?,
        };
        let fits = match cols {
            Some(k) => maps.x.shape == [width, k] && maps.c.shape == [association, k]
                && maps.b.shape == [k],
            None => maps.x.shape == [width] && maps.c.shape == [association]
                && maps.b.shape == [1],
        };
        if !fits {
            return Err(refused(&format!("{prefix}: the map's shape is the memory's")));
        }
        Ok(maps)
    }
    fn put(&self, out: &mut Named, prefix: &str) {
        out.put(&format!("{prefix}.x"), self.x.array());
        out.put(&format!("{prefix}.c"), self.c.array());
        out.put(&format!("{prefix}.b"), self.b.array());
    }
    fn cols(&self) -> usize {
        if self.x.shape.len() == 1 {
            1
        } else {
            self.x.shape[1]
        }
    }
    fn vector(&self) -> bool {
        self.x.shape.len() == 1
    }
}

/// `x M_x + c M_c + m` at `n` rows of inputs `x` and cortex `c`: `n` rows of the map's columns
/// (one number a row for the slope's maps).
fn term(maps: &Maps, x: &[f64], c: &[f64], n: usize) -> Vec<f64> {
    let cols = maps.cols();
    let on_x = np::matmul(x, n, maps.x.shape[0], &maps.x.v, cols);
    let on_c = np::matmul(c, n, maps.c.shape[0], &maps.c.v, cols);
    let mut out = vec![0.0; n * cols];
    for i in 0..n {
        for j in 0..cols {
            let k = i * cols + j;
            out[k] = on_x[k] + on_c[k] + maps.b.v[if maps.vector() { 0 } else { j }];
        }
    }
    out
}

/// The address `key + x M_x + c M_c + m` (`_addressed`).
fn addressed(key: &[f64], maps: &Maps, x: &[f64], c: &[f64], n: usize) -> Vec<f64> {
    let cols = maps.cols();
    let on_x = np::matmul(x, n, maps.x.shape[0], &maps.x.v, cols);
    let on_c = np::matmul(c, n, maps.c.shape[0], &maps.c.v, cols);
    (0..n * cols).map(|k| key[k] + on_x[k] + on_c[k] + maps.b.v[k % cols]).collect()
}

/// A read port: its neurons in value order and its synapses into the association (their
/// indices, and each one's source and association coordinate).
#[derive(Clone, Debug, PartialEq)]
pub struct Port {
    pub name: String,
    pub neurons: Vec<i64>,
    pub index: Vec<i64>,
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
}

/// The present moment of every stream as the memory addressed it (`_Moment`).
#[derive(Clone, Debug, PartialEq)]
pub struct Moment {
    pub event: F,
    pub cortex: F,
    pub address: F,
    pub key: F,
    pub inputs: F,
}

impl Moment {
    fn parts(&self) -> [&F; 5] {
        [&self.event, &self.cortex, &self.address, &self.key, &self.inputs]
    }
    fn rows(&self) -> usize {
        self.event.rows()
    }
}

/// Every head's address and attention of the latest read, and the writes the bank had then.
#[derive(Clone, Debug, PartialEq)]
pub struct Reads {
    pub writes: i64,
    pub address: Vec<F>,
    pub attended: Vec<Option<F>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub amplitude: f64,
    pub temperature: f64,
    pub recency: f64,
    pub address: f64,
    pub address_reward: f64,
    pub position_reward: f64,
    pub position_rate: f64,
    pub slope: f64,
    pub capacity: i64,
    pub heads: i64,
    pub position: i64,
    pub next: bool,
    pub boundary: bool,
    pub anchor: bool,
    pub relative: bool,
    pub drive_gain: bool,
    pub grid: Vec<i64>,
}

impl Config {
    fn floats(&self) -> [f64; 8] {
        [self.amplitude, self.temperature, self.recency, self.address, self.address_reward,
         self.position_reward, self.position_rate, self.slope]
    }
    fn integers(&self) -> [i64; 3] {
        [self.capacity, self.heads, self.position]
    }
    fn switches(&self) -> [bool; 5] {
        [self.next, self.boundary, self.anchor, self.relative, self.drive_gain]
    }
}

/// Every head's attention over the slots, `n` rows of the slots each; None where it read none.
type Attended = Vec<Option<Vec<f64>>>;

/// What `_row_steps` finds: the rows that move (over the rows asked), and at those rows the
/// event, the error, the attention the lesson names and the log of the prior attention.
struct Found {
    moving: Vec<bool>,
    x: Vec<f64>,
    error: Vec<f64>,
    target: Vec<f64>,
    prior: Vec<f64>,
}

/// A reward's step on the read's maps, made and checked before it is installed
/// (`Recall::proposal`): each stepped kind of map with its new rows for the event (`x`), one per
/// head, and the step's norm.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub stepped: Vec<(usize, Vec<Vec<f64>>)>,
    pub norm: f64,
}

/// Rows a proposal replaced, kept as room for the next one; no state: a copy starts without
/// them, and any two compare equal.
#[derive(Default)]
pub struct Spare(Vec<Vec<f64>>);

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

#[derive(Clone, Debug, PartialEq)]
pub struct Recall {
    pub config: Config,
    pub hearing: f64,
    pub association: Vec<i64>,
    pub ports: Vec<Port>,
    /// every kind's maps (`KINDS`), one per head or none
    pub maps: Vec<Vec<Maps>>,
    pub keys: F,
    pub values: F,
    pub nexts: F,
    pub held: B,
    pub previous: F,
    pub ended: B,
    pub told: Vec<i64>,
    pub focus: Vec<i64>,
    pub writes: i64,
    pub trace: F,
    pub private: bool,
    pub address_held: bool,
    pub last: Option<Moment>,
    pub attended: Option<F>,
    pub reads: Option<Reads>,
    spare: Spare,
}

/// `p` along its rows: `n` rows of `width`.
fn row(a: &[f64], width: usize, b: usize) -> &[f64] {
    &a[b * width..(b + 1) * width]
}

/// `clip(1 - p, 0, 1)`: the share of each slot a head's attention leaves (`_left`).
fn left_of(p: &[f64]) -> Vec<f64> {
    p.iter().map(|&x| np::clip(1.0 - x, 0.0, 1.0)).collect()
}

/// A later head's attention among competing heads (`_share`), row by row: `logits` and `left`
/// `n` rows of `m` slots, `mass` one number a row.
fn share(logits: &[f64], left: &[f64], mass: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * m];
    for b in 0..n {
        let (l, lg) = (row(left, m, b), row(logits, m, b));
        let z: Vec<f64> = (0..m).map(|i| if l[i] > 0.0 { lg[i] } else { f64::NEG_INFINITY })
            .collect();
        let open: Vec<bool> = z.iter().map(|v| v.is_finite()).collect();
        let offered: Vec<f64> = (0..m).map(|i| if open[i] { l[i] } else { 0.0 }).collect();
        let mass_b = np::minimum(mass[b], np::sum(&offered));
        let ratio: Vec<f64> = (0..m)
            .map(|i| if open[i] { np::log(l[i]) - z[i] } else { f64::INFINITY })
            .collect();
        let order = np::argsort_stable(&ratio);
        let lefts: Vec<f64> = order.iter().map(|&i| l[i]).collect();
        let zs: Vec<f64> = order.iter().map(|&i| z[i]).collect();
        let running = np::cumsum(&lefts);
        let capped: Vec<f64> = (0..m).map(|j| running[j] - lefts[j]).collect();
        // the log attention from each slot on: logaddexp accumulated from the last slot
        let mut rest = vec![0.0; m];
        for j in (0..m).rev() {
            rest[j] = if j == m - 1 { zs[j] } else { np::logaddexp(rest[j + 1], zs[j]) };
        }
        let mut first = m;
        for j in 0..m {
            let live = rest[j].is_finite();
            let unit = if live { np::exp(zs[j] - rest[j]) } else { 0.0 };
            if live && (mass_b - capped[j]) * unit <= lefts[j] {
                first = j;
                break;
            }
        }
        let found = first < m;
        let k = first.min(m - 1);
        let spread = if found { mass_b - capped[k] } else { 0.0 };
        let log_rest = if found { rest[k] } else { 0.0 };
        for j in 0..m {
            let mut s = if j >= first { spread * np::exp(zs[j] - log_rest) } else { lefts[j] };
            s = if zs[j].is_finite() { np::clip(s, 0.0, lefts[j]) } else { 0.0 };
            out[b * m + order[j]] = s;
        }
    }
    out
}

/// Each row's sum of `a`, `n` rows of `width` (`a.sum(axis=1)`).
fn row_sums(a: &[f64], width: usize, n: usize) -> Vec<f64> {
    (0..n).map(|b| np::sum(row(a, width, b))).collect()
}

/// `einsum("bi,bi->b", x, x)`: each row's dot with itself, in order.
fn energies(x: &[f64], width: usize, n: usize) -> Vec<f64> {
    (0..n).map(|b| np::dot(row(x, width, b), row(x, width, b))).collect()
}

/// `_per_bucket`: each row's `gradient` over the slots summed into `size` buckets, slot `i`
/// into its bucket (one row of buckets for every row, or a row each), in order from zero.
fn per_bucket(gradient: &[f64], buckets: &[usize], each: bool, n: usize, m: usize,
              size: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * size];
    for b in 0..n {
        for i in 0..m {
            let k = buckets[if each { b * m + i } else { i }];
            out[b * size + k] += gradient[b * m + i];
        }
    }
    out
}

impl Recall {
    // -- the state as named arrays

    pub fn import(s: &Named) -> Result<Recall, Error> {
        let f = |name: &str| s.f64(&format!("config.{name}"));
        let i = |name: &str| s.i64(&format!("config.{name}"));
        let t = |name: &str| flag(s, &format!("config.{name}"));
        let config = Config {
            amplitude: f("amplitude")?, temperature: f("temperature")?, recency: f("recency")?,
            address: f("address")?, address_reward: f("address_reward")?,
            position_reward: f("position_reward")?, position_rate: f("position_rate")?,
            slope: f("slope")?, capacity: i("capacity")?, heads: i("heads")?,
            position: i("position")?, next: t("next")?, boundary: t("boundary")?,
            anchor: t("anchor")?, relative: t("relative")?, drive_gain: t("drive_gain")?,
            grid: s.i64s("config.grid")?,
        };
        if config.heads < 1 || config.capacity < 1 || config.position < 0
            || config.grid.iter().any(|&m| m < 1)
        {
            return Err(refused("a recall memory of heads, a capacity and a position term"));
        }
        let mut ports = Vec::new();
        for h in 0..config.heads {
            for port in if config.next { &["recall", "next"][..] } else { &["recall"][..] } {
                let name = format!("{h}.{port}");
                ports.push(Port {
                    neurons: s.i64s(&format!("ports.{name}"))?,
                    index: s.i64s(&format!("synapses.{name}.index"))?,
                    rows: s.i64s(&format!("synapses.{name}.rows"))?,
                    cols: s.i64s(&format!("synapses.{name}.cols"))?,
                    name,
                });
            }
        }
        let association = s.i64s("association")?;
        let (width, cortex) = (ports[0].neurons.len(), association.len());
        for port in &ports {
            let n = port.index.len();
            if port.neurons.len() != width || width == 0 || port.rows.len() != n
                || port.cols.len() != n || port.index.iter().any(|&k| k < 0)
                || port.rows.iter().any(|&r| r < 0 || r as usize >= width)
                || port.cols.iter().any(|&c| c < 0 || c as usize >= cortex)
            {
                return Err(refused("a port's neurons and synapses are the memory's"));
            }
        }
        let mut recall = Recall {
            config, hearing: s.f64("hearing")?, association, ports, maps: Vec::new(),
            keys: F::read(s, "keys")?, values: F::read(s, "values")?,
            nexts: F::read(s, "nexts")?, held: B::read(s, "held")?,
            previous: F::read(s, "previous")?, ended: B::read(s, "ended")?,
            told: s.i64s("told")?, focus: s.i64s("focus")?, writes: s.i64("writes")?,
            trace: F::read(s, "trace")?, private: flag(s, "private")?,
            address_held: flag(s, "address_held")?, last: None, attended: None, reads: None,
            spare: Spare::default(),
        };
        let count = recall.position_buckets();
        for (k, kind) in KINDS.iter().enumerate() {
            let cols = match k {
                ADDRESS => Some(width),
                POSITION => Some(count),
                ANCHOR => Some(count + 1),
                RELATIVES => Some(2 * RELATIVE as usize + 2),
                SLOPES => None,
                _ => Some(recall.config.grid.iter().sum::<i64>() as usize),
            };
            let mut maps = Vec::new();
            while s.has(&format!("maps.{kind}.{}.x", maps.len())) {
                let prefix = format!("maps.{kind}.{}", maps.len());
                maps.push(Maps::read(s, &prefix, width, cortex, cols)?);
            }
            if !(maps.len() == recall.config.heads as usize || (k != ADDRESS && maps.is_empty()))
                || (k != ADDRESS && !maps.is_empty() && count == 0)
            {
                return Err(refused("every head holds each map its options give"));
            }
            recall.maps.push(maps);
        }
        let (streams, slots) = match recall.keys.shape.as_slice() {
            &[a, b, _] => (a, b),
            _ => return Err(refused("the slots are keys, values and nexts per stream")),
        };
        let slots_of = |shape: &[usize]| shape == [streams, slots, width];
        if !slots_of(&recall.keys.shape) || !slots_of(&recall.values.shape)
            || !slots_of(&recall.nexts.shape) || recall.held.shape != [streams, slots]
            || recall.previous.shape.len() != 2 || recall.previous.shape[1] != cortex
            || recall.ended.shape != [streams] || recall.told.len() != streams
            || recall.focus.len() != streams || recall.trace.shape.len() != 2
        {
            return Err(refused("every stream's state is the memory's shape"));
        }
        if s.has("last.event") {
            let part = |name: &str, cols: usize| F::read(s, &format!("last.{name}"))?
                .matrix(cols, name);
            let last = Moment {
                event: part("event", width)?, cortex: part("cortex", cortex)?,
                address: part("address", width)?, key: part("key", width)?,
                inputs: part("inputs", width)?,
            };
            if last.parts().iter().any(|a| a.rows() != last.rows()) {
                return Err(refused("a moment is one row per stream"));
            }
            recall.last = Some(last);
        }
        if s.has("attended") {
            let a = F::read(s, "attended")?;
            if a.shape.len() != 2 {
                return Err(refused("an attention is one row per stream"));
            }
            recall.attended = Some(a);
        }
        if s.has("reads.writes") {
            let heads = recall.config.heads as usize;
            let mut reads = Reads { writes: s.i64("reads.writes")?, address: Vec::new(),
                                    attended: Vec::new() };
            for h in 0..heads {
                reads.address.push(F::read(s, &format!("reads.address.{h}"))?
                    .matrix(width, "an address")?);
                let name = format!("reads.attended.{h}");
                reads.attended.push(if s.has(&name) {
                    let a = F::read(s, &name)?;
                    if a.shape.len() != 2 {
                        return Err(refused("an attention is one row per stream"));
                    }
                    Some(a)
                } else {
                    None
                });
            }
            recall.reads = Some(reads);
        }
        Ok(recall)
    }

    pub fn export(&self, out: &mut Named) {
        let c = &self.config;
        for (name, value) in FLOATS.iter().zip(c.floats()) {
            out.put(&format!("config.{name}"), Array::scalar_f64(value));
        }
        for (name, value) in INTEGERS.iter().zip(c.integers()) {
            out.put(&format!("config.{name}"), Array::scalar_i64(value));
        }
        for (name, value) in SWITCHES.iter().zip(c.switches()) {
            out.put(&format!("config.{name}"), Array::from_bool(&[], &[value]));
        }
        out.put("config.grid", Array::from_i64(&[c.grid.len()], &c.grid));
        out.put("hearing", Array::scalar_f64(self.hearing));
        out.put("association", Array::from_i64(&[self.association.len()], &self.association));
        for port in &self.ports {
            out.put(&format!("ports.{}", port.name),
                    Array::from_i64(&[port.neurons.len()], &port.neurons));
            for (part, v) in [("index", &port.index), ("rows", &port.rows), ("cols", &port.cols)] {
                out.put(&format!("synapses.{}.{part}", port.name), Array::from_i64(&[v.len()], v));
            }
        }
        for (kind, maps) in KINDS.iter().zip(&self.maps) {
            for (h, held) in maps.iter().enumerate() {
                held.put(out, &format!("maps.{kind}.{h}"));
            }
        }
        out.put("keys", self.keys.array());
        out.put("values", self.values.array());
        out.put("nexts", self.nexts.array());
        out.put("held", self.held.array());
        out.put("previous", self.previous.array());
        out.put("ended", self.ended.array());
        out.put("told", Array::from_i64(&[self.told.len()], &self.told));
        out.put("focus", Array::from_i64(&[self.focus.len()], &self.focus));
        out.put("writes", Array::scalar_i64(self.writes));
        out.put("trace", self.trace.array());
        out.put("private", Array::from_bool(&[], &[self.private]));
        out.put("address_held", Array::from_bool(&[], &[self.address_held]));
        if let Some(last) = &self.last {
            for (name, part) in MOMENT.iter().zip(last.parts()) {
                out.put(&format!("last.{name}"), part.array());
            }
        }
        if let Some(a) = &self.attended {
            out.put("attended", a.array());
        }
        if let Some(reads) = &self.reads {
            out.put("reads.writes", Array::scalar_i64(reads.writes));
            for (h, a) in reads.address.iter().enumerate() {
                out.put(&format!("reads.address.{h}"), a.array());
            }
            for (h, a) in reads.attended.iter().enumerate() {
                if let Some(a) = a {
                    out.put(&format!("reads.attended.{h}"), a.array());
                }
            }
        }
    }

    // -- layout

    fn width(&self) -> usize {
        self.ports[0].neurons.len()
    }
    fn cortex(&self) -> usize {
        self.association.len()
    }
    fn heads(&self) -> usize {
        self.config.heads as usize
    }
    fn streams(&self) -> usize {
        self.keys.shape[0]
    }
    fn slots(&self) -> usize {
        self.keys.shape[1]
    }
    /// Head `h`'s recall port, and its next port with `next`.
    fn head_ports(&self, h: usize) -> (&Port, Option<&Port>) {
        if self.config.next {
            (&self.ports[2 * h], Some(&self.ports[2 * h + 1]))
        } else {
            (&self.ports[h], None)
        }
    }
    /// Every neuron the read drives (`port_neurons`): each head's recall and next neurons.
    fn port_neurons(&self) -> Vec<i64> {
        self.ports.iter().flat_map(|p| p.neurons.iter().copied()).collect()
    }
    fn bounded(&self) -> bool {
        self.config.boundary && self.config.capacity > 1
    }

    // -- the positions

    /// The age buckets of a position term (`position_count`).
    fn position_buckets(&self) -> usize {
        let (position, capacity) = (self.config.position, self.config.capacity);
        if position == 0 {
            return 0;
        }
        if capacity <= position {
            return capacity as usize;
        }
        let doublings = np::log2((capacity - 1) as f64 / position as f64).floor();
        (position as f64 + doublings + 1.0) as usize
    }
    fn ages(&self) -> Vec<f64> {
        let m = self.slots();
        (0..m).map(|i| (m - 1 - i) as f64).collect()
    }
    fn whole_ages(&self) -> Vec<i64> {
        let m = self.slots() as i64;
        (0..m).map(|i| m - 1 - i).collect()
    }
    /// `_bucket` of one age: below `position` its own, then one per doubling, the last bucket
    /// holding every older age.
    fn bucket(&self, age: f64) -> f64 {
        let exact = self.config.position as f64;
        let out = if age < exact {
            age
        } else {
            exact + np::log2(np::maximum(age, 1.0) / exact).floor()
        };
        np::minimum(out, self.position_buckets() as f64 - 1.0)
    }
    /// Every whole age's bucket up to `capacity - 1` (`_age_buckets`).
    fn age_buckets(&self) -> Vec<i64> {
        (0..self.config.capacity).map(|a| self.bucket(a as f64) as i64).collect()
    }
    /// Each slot's age bucket (`buckets`).
    fn buckets(&self) -> Result<Vec<usize>, Error> {
        if self.config.position == 0 {
            return Ok(vec![0; self.slots()]);
        }
        let table = self.age_buckets();
        self.whole_ages().iter()
            .map(|&a| table.get(a as usize).map(|&k| k as usize)
                 .ok_or_else(|| refused("a slot's age lies within the capacity")))
            .collect()
    }
    /// The told counts of the streams `rows`, or of every stream.
    fn of_rows(values: &[i64], rows: Option<&[usize]>) -> Vec<i64> {
        match rows {
            Some(r) => r.iter().map(|&k| values[k]).collect(),
            None => values.to_vec(),
        }
    }
    /// Each slot's anchored bucket for the streams `rows` (`anchored_buckets`), a row each.
    fn anchored_buckets(&self, rows: Option<&[usize]>) -> Vec<usize> {
        let told = Self::of_rows(&self.told, rows);
        let table = self.age_buckets();
        let outside = self.position_buckets();
        let ages = self.whole_ages();
        let last = table.len() as i64 - 1;
        let mut out = Vec::with_capacity(told.len() * ages.len());
        for &t in &told {
            for &a in &ages {
                let place = t - 1 - a;
                out.push(if place < 0 {
                    outside
                } else {
                    table[place.min(last) as usize] as usize
                });
            }
        }
        out
    }
    /// Each slot's relative bucket for the streams `rows` (`relative_buckets`), a row each.
    fn relative_buckets(&self, rows: Option<&[usize]>) -> Vec<usize> {
        let focus = Self::of_rows(&self.focus, rows);
        let ages = self.whole_ages();
        let mut out = Vec::with_capacity(focus.len() * ages.len());
        for &f in &focus {
            for &a in &ages {
                out.push(if f < 0 {
                    (2 * RELATIVE + 1) as usize
                } else {
                    ((a - f).clamp(-RELATIVE, RELATIVE) + RELATIVE) as usize
                });
            }
        }
        out
    }
    /// Each slot's column of the grid maps under each modulus (`grid_buckets`), a row each.
    fn grid_buckets(&self) -> Vec<Vec<usize>> {
        let ages = self.whole_ages();
        let mut start = 0i64;
        let mut out = Vec::new();
        for &modulus in &self.config.grid {
            out.push(ages.iter().map(|&a| (start + a % modulus) as usize).collect());
            start += modulus;
        }
        out
    }

    /// Head `head`'s position term at every slot for `n` moments of `event` and `cortex`
    /// (`position_term`), `n` rows of the slots; None without a position term or slots.
    /// `rows` names the streams of the moments (every stream, in order, without).
    fn position_term(&self, head: usize, event: &[f64], cortex: &[f64], n: usize,
                     rows: Option<&[usize]>) -> Result<Option<Vec<f64>>, Error> {
        let m = self.slots();
        if self.maps[POSITION].is_empty() || m == 0 {
            return Ok(None);
        }
        let maps = &self.maps[POSITION][head];
        let (table, cols) = (term(maps, event, cortex, n), maps.cols());
        let buckets = self.buckets()?;
        let mut out = vec![0.0; n * m];
        for b in 0..n {
            for i in 0..m {
                out[b * m + i] = table[b * cols + buckets[i]];
            }
        }
        if rows.is_some() || n == self.streams() {
            if !self.maps[ANCHOR].is_empty() {
                let anchored = term(&self.maps[ANCHOR][head], event, cortex, n);
                let (at, size) = (self.anchored_buckets(rows), self.position_buckets() + 1);
                for b in 0..n {
                    for i in 0..m {
                        out[b * m + i] += anchored[b * size + at[b * m + i]];
                    }
                }
            }
            if !self.maps[RELATIVES].is_empty() {
                let relative = term(&self.maps[RELATIVES][head], event, cortex, n);
                let (at, size) = (self.relative_buckets(rows), 2 * RELATIVE as usize + 2);
                for b in 0..n {
                    for i in 0..m {
                        out[b * m + i] += relative[b * size + at[b * m + i]];
                    }
                }
            }
        }
        let ages = self.ages();
        if !self.maps[SLOPES].is_empty() {
            let lean = term(&self.maps[SLOPES][head], event, cortex, n);
            for b in 0..n {
                for i in 0..m {
                    out[b * m + i] += lean[b] * (ages[i] / SLOPE);
                }
            }
        }
        if !self.maps[GRID].is_empty() {
            let maps = &self.maps[GRID][head];
            let (grid, cols) = (term(maps, event, cortex, n), maps.cols());
            let residues = self.grid_buckets();
            let position = self.config.position as f64;
            for b in 0..n {
                for i in 0..m {
                    // the residues' terms summed over the moduli, in order from zero
                    let mut s = 0.0;
                    for r in &residues {
                        s += grid[b * cols + r[i]];
                    }
                    out[b * m + i] += s * if ages[i] >= position { 1.0 } else { 0.0 };
                }
            }
        }
        Ok(Some(out))
    }

    // -- the moment's address and read

    /// The observations, one finite row of the key's width per stream (`_port`).
    fn port(&self, observations: F) -> Result<F, Error> {
        let width = self.width();
        if observations.shape.len() != 2 || observations.shape[1] != width {
            return Err(refused(&format!("observations are one row of {width} values per stream")));
        }
        if !observations.v.iter().all(|x| x.is_finite()) {
            return Err(refused("observations must be finite"));
        }
        Ok(observations)
    }

    /// The address of the present moment of each stream, from the cortex of the moment before
    /// (`_moment`).
    fn moment(&self, x: F) -> Moment {
        let (n, width, cortex) = (x.rows(), self.width(), self.cortex());
        let c = if self.previous.rows() == n {
            self.previous.clone()
        } else {
            F::zeros(&[n, cortex])
        };
        let key: Vec<f64> = x.v.iter().map(|&v| HEARD * v).collect();
        let q = addressed(&key, &self.maps[ADDRESS][0], &x.v, &c.v, n);
        Moment {
            inputs: x.clone(), cortex: c, address: F { shape: vec![n, width], v: q },
            key: F { shape: vec![n, width], v: key }, event: x,
        }
    }

    /// Head `head`'s address of a moment (`_address`).
    fn address(&self, moment: &Moment, head: usize) -> Vec<f64> {
        if head == 0 {
            return moment.address.v.clone();
        }
        addressed(&moment.key.v, &self.maps[ADDRESS][head], &moment.inputs.v, &moment.cortex.v,
                  moment.rows())
    }

    /// The content logits `address . key / temperature` of `n` rows of addresses over the
    /// slots `keys` of those rows (`_content`).
    fn content(&self, address: &[f64], keys: &[f64], n: usize, m: usize) -> Vec<f64> {
        let width = self.width();
        let mut out = Vec::with_capacity(n * m);
        for b in 0..n {
            for i in 0..m {
                let product = np::dot(row(address, width, b), row(keys, width, b * m + i));
                out.push(product / self.config.temperature);
            }
        }
        out
    }

    /// The logits of the attention over slots (`_attention`): content less `recency` times each
    /// slot's age, plus a head's position term, `-inf` at a slot its stream does not hold.
    fn attention(&self, address: &[f64], keys: &[f64], n: usize, m: usize,
                 held: Option<&[bool]>, position: Option<&[f64]>) -> Vec<f64> {
        let mut logits = self.content(address, keys, n, m);
        if self.config.recency != 0.0 {
            let ages = self.ages();
            for (k, l) in logits.iter_mut().enumerate() {
                *l -= self.config.recency * ages[k % m];
            }
        }
        if let Some(position) = position {
            for (l, p) in logits.iter_mut().zip(position) {
                *l += p;
            }
        }
        if let Some(held) = held {
            for (l, &h) in logits.iter_mut().zip(held) {
                if !h {
                    *l = f64::NEG_INFINITY;
                }
            }
        }
        logits
    }

    /// `p` over each stream's slots for `n` addresses (`weights`), or None when the streams
    /// hold no slot or are another batch.
    fn weights(&self, address: &[f64], n: usize, position: Option<&[f64]>) -> Option<Vec<f64>> {
        let m = self.slots();
        if m == 0 || self.streams() != n {
            return None;
        }
        let all = self.held.v.iter().all(|&h| h);
        let logits = self.attention(address, &self.keys.v, n, m,
                                    if all { None } else { Some(&self.held.v) }, position);
        let mut p = vec![0.0; n * m];
        for b in 0..n {
            let l = row(&logits, m, b);
            let top = if all || self.held.v[b * m..(b + 1) * m].iter().any(|&h| h) {
                np::max(l)
            } else {
                0.0
            };
            let e: Vec<f64> = l.iter().map(|&x| np::exp(x - top)).collect();
            let total = np::sum(&e);
            let total = if all || total > 0.0 { total } else { 1.0 };
            for i in 0..m {
                p[b * m + i] = e[i] / total;
            }
        }
        Some(p)
    }

    /// Every head's attention over the slots the bank holds (`_read`): the first head's own, each
    /// after it taking of every slot at most the share the heads before it left.
    fn read(&self, addresses: &[Vec<f64>], moment: &Moment) -> Result<Attended, Error> {
        let (n, m) = (moment.rows(), self.slots());
        let mut attended = Vec::new();
        let mut left: Option<Vec<f64>> = None;
        for (h, address) in addresses.iter().enumerate() {
            let position = self.position_term(h, &moment.inputs.v, &moment.cortex.v, n, None)?;
            let mut p = self.weights(address, n, position.as_deref());
            if let Some(attention) = p.as_mut() {
                if self.heads() > 1 {
                    if let Some(l) = &left {
                        let all = self.held.v.iter().all(|&x| x);
                        let logits = self.attention(address, &self.keys.v, n, m,
                                                    if all { None } else { Some(&self.held.v) },
                                                    position.as_deref());
                        *attention = share(&logits, l, &row_sums(attention, m, n), n, m);
                    }
                    let mine = left_of(attention);
                    left = Some(match left {
                        None => mine,
                        Some(l) => l.iter().zip(&mine).map(|(a, b)| a * b).collect(),
                    });
                }
            }
            attended.push(p);
        }
        Ok(attended)
    }

    /// Every head's address and attention at `moment` (`_reads`).
    fn reads_of(&self, moment: &Moment) -> Result<(Vec<Vec<f64>>, Attended), Error> {
        let addresses: Vec<Vec<f64>> = (0..self.heads()).map(|h| self.address(moment, h))
            .collect();
        let attended = self.read(&addresses, moment)?;
        Ok((addresses, attended))
    }

    /// Address the present moment of every stream and read its slots into a copy of `drive`
    /// (`stimulate`).
    pub fn stimulate(&mut self, drive: F, observations: F) -> Result<F, Error> {
        let x = self.port(observations)?;
        let moment = self.moment(x);
        let (n, m, width) = (moment.rows(), self.slots(), self.width());
        if drive.shape.len() != 2 || drive.shape[0] != n {
            return Err(refused("a drive is one row per stream"));
        }
        let columns = drive.shape[1];
        let neurons = self.port_neurons();
        if neurons.iter().any(|&k| k < 0 || k as usize >= columns) {
            return Err(refused("the drive holds every neuron the memory drives"));
        }
        let mut out = drive;
        let (addresses, attended) = self.reads_of(&moment)?;
        let amplitude = if self.config.drive_gain { 1.0 } else { self.config.amplitude };
        for (h, p) in attended.iter().enumerate() {
            let Some(p) = p else { continue };
            let (recall, following) = self.head_ports(h);
            for (port, slots) in [(Some(recall), &self.values), (following, &self.nexts)] {
                let Some(port) = port else { continue };
                for b in 0..n {
                    let read = np::matmul(row(p, m, b), 1, m, row(&slots.v, m * width, b), width);
                    for (c, &k) in port.neurons.iter().enumerate() {
                        out.v[b * columns + k as usize] = amplitude * read[c];
                    }
                }
            }
        }
        if self.config.drive_gain {
            // what the ports hold, kept for the route's lesson, at the gain
            let mut trace = Vec::with_capacity(n * neurons.len());
            for b in 0..n {
                trace.extend(neurons.iter().map(|&k| out.v[b * columns + k as usize]));
            }
            let gain = self.config.amplitude * self.hearing;
            for b in 0..n {
                for (j, &k) in neurons.iter().enumerate() {
                    out.v[b * columns + k as usize] = gain * trace[b * neurons.len() + j];
                }
            }
            self.trace = F { shape: vec![n, neurons.len()], v: trace };
        }
        let as_f = |p: &Option<Vec<f64>>| p.as_ref().map(|v| F { shape: vec![n, m], v: v.clone() });
        let first = as_f(&attended[0]);
        self.attended = first;
        self.reads = (self.heads() > 1).then(|| Reads {
            writes: self.writes,
            address: addresses.iter().map(|a| F { shape: vec![n, width], v: a.clone() }).collect(),
            attended: attended.iter().map(as_f).collect(),
        });
        self.last = Some(moment);
        Ok(out)
    }

    // -- the write after the settle

    /// No read is held (`_unread`).
    fn unread(&mut self) {
        self.last = None;
        self.attended = None;
        self.reads = None;
    }

    /// Every stream begins empty, with its boundary slot where the memory holds one (`_empty`).
    fn empty(&mut self, batch: usize) {
        let (width, cortex) = (self.width(), self.cortex());
        let slots = if self.bounded() && batch > 0 { 1 } else { 0 };
        self.keys = F::zeros(&[batch, slots, width]);
        self.values = F::zeros(&[batch, slots, width]);
        self.nexts = F::zeros(&[batch, slots, width]);
        self.held = B { shape: vec![batch, slots], v: vec![true; batch * slots] };
        self.previous = F::zeros(&[batch, cortex]);
        self.ended = B { shape: vec![batch], v: vec![false; batch] };
        self.told = vec![0; batch];
        self.focus = vec![-1; batch];
        self.unread();
    }

    /// After the settle: the observations join their slots under the moment's key, the slot
    /// before learns them as its next, and the settled cortex moves on (`witness`).
    pub fn witness(&mut self, observations: F, activation: F) -> Result<(), Error> {
        let x = self.port(observations)?;
        let moment = match &self.last {
            Some(last) if last.event.shape == x.shape && last.event.v == x.v => last.clone(),
            _ => self.moment(x.clone()),
        };
        let n = x.rows();
        let columns = if activation.shape.len() == 2 { activation.shape[1] } else { 0 };
        if self.association.iter().any(|&k| k < 0 || k as usize >= columns) {
            return Err(refused("the settled state holds the association"));
        }
        if activation.rows() != n {
            return Err(refused("the settled state needs one row per stream"));
        }
        let cortex: Vec<f64> = (0..n)
            .flat_map(|b| self.association.iter().map(move |&k| (b, k)))
            .map(|(b, k)| activation.v[b * columns + k as usize])
            .collect();
        if !self.private {
            self.observe(&moment.key, &moment.event);
            self.writes += n as i64;
        }
        self.previous = F { shape: vec![n, self.cortex()], v: cortex };
        self.last = None;
        Ok(())
    }

    /// Write one moment per stream (`_observe`): the latest slot learns what followed it, a new
    /// slot holds this moment, beyond `capacity` the oldest slot is dropped.
    fn observe(&mut self, key: &F, value: &F) {
        let n = key.rows();
        if n != self.streams() {
            self.empty(n);
        }
        let (m, width) = (self.slots(), self.width());
        // (stream, slot) pairs that learn their successor, the value of the stream's moment
        let mut successor: Vec<(usize, usize)> = Vec::new();
        if m > 0 {
            for b in 0..n {
                let held = &self.held.v[b * m..(b + 1) * m];
                if held.iter().any(|&h| h) && !self.ended.v[b] {
                    let latest = m - 1 - held.iter().rev().position(|&h| h).unwrap_or(0);
                    successor.push((b, latest));
                }
            }
        }
        if self.ended.v.iter().any(|&e| e) {
            self.ended.v = vec![false; n];
        }
        // the slot this moment's read attended most, one moment older once it is written
        let focus: Vec<i64> = match &self.attended {
            Some(a) if a.shape == [n, m] && m > 0 => (0..n).map(|b| {
                let p = row(&a.v, m, b);
                if np::max(p) > 0.0 {
                    (m - 1 - np::argmax(p)) as i64 + 1
                } else {
                    -1
                }
            }).collect(),
            _ => self.focus.iter().map(|&f| if f >= 0 { f + 1 } else { -1 }).collect(),
        };
        // the new slots: the bank less the `drop` oldest and this moment behind them
        let drop = (m + 1).saturating_sub(self.config.capacity as usize);
        let kept = m - drop + 1;
        let mut keys = Vec::with_capacity(n * kept * width);
        let mut values = Vec::with_capacity(n * kept * width);
        let mut nexts = Vec::with_capacity(n * kept * width);
        let mut held = Vec::with_capacity(n * kept);
        for b in 0..n {
            let at = |s: usize| (b * m + s) * width;
            keys.extend_from_slice(&self.keys.v[at(drop)..at(m)]);
            keys.extend_from_slice(row(&key.v, width, b));
            values.extend_from_slice(&self.values.v[at(drop)..at(m)]);
            values.extend_from_slice(row(&value.v, width, b));
            nexts.extend_from_slice(&self.nexts.v[at(drop)..at(m)]);
            nexts.extend(std::iter::repeat_n(0.0, width));
            held.extend_from_slice(&self.held.v[b * m + drop..(b + 1) * m]);
            held.push(true);
        }
        for (b, slot) in successor {
            if slot >= drop {
                let at = (b * kept + slot - drop) * width;
                nexts[at..at + width].copy_from_slice(row(&value.v, width, b));
            }
        }
        self.keys = F { shape: vec![n, kept, width], v: keys };
        self.values = F { shape: vec![n, kept, width], v: values };
        self.nexts = F { shape: vec![n, kept, width], v: nexts };
        self.held = B { shape: vec![n, kept], v: held };
        for t in self.told.iter_mut() {
            *t += 1;
        }
        self.focus = focus;
    }

    // -- the reward's eligibility

    /// `movement @ W.T` for the synapses of head `head`'s port `port` into the association
    /// (`readback`): per source neuron, the sum over its synapses of weight times the movement
    /// of its association neuron.
    fn readback(&self, port: &Port, weights: &[f64], movement: &[f64], n: usize)
                -> Result<Vec<f64>, Error> {
        let (width, cortex) = (port.neurons.len(), self.cortex());
        let mut w = vec![0.0; width * cortex];
        for s in 0..port.index.len() {
            let weight = *weights.get(port.index[s] as usize)
                .ok_or_else(|| refused("the graph's weights hold the memory's synapses"))?;
            w[port.rows[s] as usize * cortex + port.cols[s] as usize] += weight;
        }
        let mut out = Vec::with_capacity(n * width);
        for b in 0..n {
            for i in 0..width {
                out.push(np::dot(row(movement, cortex, b), row(&w, cortex, i)));
            }
        }
        Ok(out)
    }

    /// For each head's lesson, the attention it read the moment with (`_lesson_reads`): None
    /// for head 0, the kept reads while the bank holds what they read, else computed here.
    fn lesson_reads(&self, moment: &Moment) -> Result<Vec<Option<F>>, Error> {
        let heads = self.heads();
        if heads == 1 {
            return Ok(vec![None]);
        }
        let attended: Vec<Option<F>> = match &self.reads {
            Some(r) if r.writes == self.writes && r.attended.iter().all(|a| a.is_some()) => {
                r.attended.clone()
            }
            _ => {
                let (n, m) = (moment.rows(), self.slots());
                self.reads_of(moment)?.1.into_iter()
                    .map(|p| p.map(|v| F { shape: vec![n, m], v })).collect()
            }
        };
        let mut out = vec![None];
        out.extend(attended.into_iter().skip(1));
        Ok(out)
    }

    /// Head `head`'s address of the moment the latest read addressed (`_head_address`).
    fn head_address(&self, moment: &Moment, head: usize) -> Vec<f64> {
        match &self.reads {
            Some(r) if head > 0 && r.address.len() == self.heads() => r.address[head].v.clone(),
            _ => self.address(moment, head),
        }
    }

    /// Each row's step of the address lesson before the average over rows (`_row_steps`), at
    /// every stream: the rows that move and, at those, what the reward's step reads. None when
    /// no row moves. `address` is the head's address of the moment.
    #[allow(clippy::too_many_arguments)]
    fn row_steps(&self, moment: &Moment, address: &[f64], d_value: &[f64],
                 d_next: Option<&[f64]>, head: usize, read: Option<&F>)
                 -> Result<Option<Found>, Error> {
        let (n, m, width, cortex) = (self.streams(), self.slots(), self.width(), self.cortex());
        let held = &self.held.v;
        let masked = !held.iter().all(|&h| h);
        let mut score = Vec::with_capacity(n * m);
        let mut norm = Vec::with_capacity(n);
        for b in 0..n {
            let dv = row(d_value, width, b);
            for i in 0..m {
                score.push(np::dot(dv, row(&self.values.v, width, b * m + i)));
            }
            norm.push(np::sum_squares(dv));
        }
        if let Some(d_next) = d_next {
            for b in 0..n {
                let dn = row(d_next, width, b);
                for i in 0..m {
                    score[b * m + i] += np::dot(dn, row(&self.nexts.v, width, b * m + i));
                }
                norm[b] += np::sum_squares(dn);
            }
        }
        let norm: Vec<f64> = norm.iter().map(|&s| np::sqrt(s)).collect();
        let mut moving: Vec<bool> = norm.iter().map(|&s| s > 1e-12).collect();
        if masked {
            for b in 0..n {
                moving[b] &= held[b * m..(b + 1) * m].iter().any(|&h| h);
            }
        }
        if !moving.iter().any(|&v| v) {
            return Ok(None);
        }
        // the moving rows, their inputs, cortex, address, slots and logits
        let mut rows: Vec<usize> = (0..n).filter(|&b| moving[b]).collect();
        let take = |a: &[f64], w: usize, rows: &[usize]| -> Vec<f64> {
            rows.iter().flat_map(|&b| row(a, w, b).iter().copied()).collect()
        };
        let mut logits: Vec<f64> = Vec::with_capacity(rows.len() * m);
        for &b in &rows {
            for i in 0..m {
                let l = score[b * m + i] / norm[b] / ADDRESS_TEMPERATURE;
                logits.push(if masked && !held[b * m + i] { f64::NEG_INFINITY } else { l });
            }
        }
        let mut attention = match read {
            None => {
                let (x, c, q) = (take(&moment.inputs.v, width, &rows),
                                 take(&moment.cortex.v, cortex, &rows),
                                 take(address, width, &rows));
                let keys = take(&self.keys.v, m * width, &rows);
                let held_rows: Vec<bool> = rows.iter()
                    .flat_map(|&b| held[b * m..(b + 1) * m].iter().copied()).collect();
                let position = self.position_term(head, &x, &c, rows.len(), Some(&rows))?;
                self.attention(&q, &keys, rows.len(), m, masked.then_some(&held_rows[..]),
                               position.as_deref())
            }
            Some(read) => {
                if read.shape != [n, m] {
                    return Err(refused("a head's read is one row of the slots per stream"));
                }
                let attention: Vec<f64> = take(&read.v, m, &rows).iter()
                    .map(|&p| np::log(p)).collect();
                let alive: Vec<bool> = (0..rows.len())
                    .map(|r| row(&attention, m, r).iter().any(|v| v.is_finite())).collect();
                if !alive.iter().any(|&a| a) {
                    return Ok(None);
                }
                if !alive.iter().all(|&a| a) {
                    // the heads before took this row's slots whole
                    for (r, &b) in rows.iter().enumerate() {
                        if !alive[r] {
                            moving[b] = false;
                        }
                    }
                    let kept: Vec<usize> = (0..rows.len()).filter(|&r| alive[r]).collect();
                    logits = take(&logits, m, &kept);
                    rows = kept.iter().map(|&r| rows[r]).collect();
                    take(&attention, m, &kept)
                } else {
                    attention
                }
            }
        };
        let k = rows.len();
        let x = take(&moment.inputs.v, width, &rows);
        let keys = take(&self.keys.v, m * width, &rows);
        let mut p = vec![0.0; k * m];
        for r in 0..k {
            let a = &mut attention[r * m..(r + 1) * m];
            let top = np::max(a);
            for v in a.iter_mut() {
                *v -= top;
            }
            let e: Vec<f64> = a.iter().map(|&v| np::exp(v)).collect();
            let log_total = np::log(np::sum(&e));
            for v in a.iter_mut() {
                *v -= log_total;
            }
            let l = &mut logits[r * m..(r + 1) * m];
            for (v, &prior) in l.iter_mut().zip(a.iter()) {
                *v += prior;
            }
            let top = np::max(l);
            let e: Vec<f64> = l.iter().map(|&v| np::exp(v - top)).collect();
            let total = np::sum(&e);
            for i in 0..m {
                p[r * m + i] = e[i] / total;
            }
        }
        let gradient: Vec<f64> = p.iter().zip(&attention)
            .map(|(&t, &a)| t - if a.is_finite() { np::exp(a) } else { 0.0 }).collect();
        let mut error = Vec::with_capacity(k * width);
        for r in 0..k {
            error.extend(np::matmul(row(&gradient, m, r), 1, m, row(&keys, m * width, r), width));
        }
        Ok(Some(Found { moving, x, error, target: p, prior: attention }))
    }

    /// The columns of the reward eligibility that can hold anything (`reward_blocks`), as
    /// `(start, width)`.
    fn reward_blocks(&self) -> Vec<(usize, usize)> {
        let heads = self.heads();
        let columns = heads * self.width();
        let mut blocks = Vec::new();
        if self.address_held || self.config.address_reward != 0.0 {
            blocks.push((0, columns));
        }
        let count = self.position_buckets();
        let mut start = columns;
        if self.config.position_reward != 0.0 && count > 0 {
            blocks.push((start, heads * count));
            start += heads * count;
            if self.config.anchor {
                blocks.push((start, heads * (count + 1)));
            }
        }
        blocks
    }

    /// Every column of the reward eligibility (`reward_columns`).
    fn reward_columns(&self) -> usize {
        let heads = self.heads();
        let count = self.position_buckets();
        let mut columns = heads * self.width();
        if self.config.position_reward != 0.0 && count > 0 {
            columns += heads * count;
            if self.config.anchor {
                columns += heads * (count + 1);
            }
        }
        columns
    }

    /// The read's eligibility from a sampled act (`contrast_rows`): one `(inputs, kept columns)`
    /// array per row, or None where reward trains nothing of the read. `weights` are the graph's
    /// effective weights, `plus` and `minus` the activations of the act's two nudged phases.
    pub fn contrast_rows(&mut self, weights: &[f64], plus: F, minus: F)
                         -> Result<Option<F>, Error> {
        let positions = self.config.position_reward != 0.0 && !self.maps[POSITION].is_empty();
        if !(self.config.address_reward != 0.0 || positions) {
            return Ok(None);
        }
        let columns_of = |a: &F| if a.shape.len() == 2 { a.shape[1] } else { 0 };
        let (cp, cm) = (columns_of(&plus), columns_of(&minus));
        if self.association.iter().any(|&k| k < 0 || k as usize >= cp.min(cm))
            || plus.rows() != minus.rows()
        {
            return Err(refused("the phases hold the association, one row per stream each"));
        }
        let batch = plus.rows();
        let movement: Vec<f64> = (0..batch)
            .flat_map(|b| self.association.iter().map(move |&k| (b, k as usize)))
            .map(|(b, k)| plus.v[b * cp + k] - minus.v[b * cm + k])
            .collect();
        let (width, columns) = (self.width(), self.width());
        let count = self.position_buckets();
        if self.config.address_reward != 0.0 {
            self.address_held = true;
        }
        let kept: usize = self.reward_blocks().iter().map(|block| block.1).sum();
        let mut out = F::zeros(&[batch, width, kept]);
        let (m, heads) = (self.slots(), self.heads());
        let Some(moment) = self.last.as_ref() else { return Ok(Some(out)) };
        if m == 0 || self.streams() != batch || moment.key.rows() != batch {
            return Ok(Some(out));
        }
        let reads = self.lesson_reads(moment)?;
        for h in 0..heads {
            let (recall, following) = self.head_ports(h);
            let d_value = self.readback(recall, weights, &movement, batch)?;
            let d_next = match following {
                Some(port) => Some(self.readback(port, weights, &movement, batch)?),
                None => None,
            };
            let address = self.head_address(moment, h);
            let Some(found) = self.row_steps(moment, &address, &d_value, d_next.as_deref(), h,
                                             reads[h].as_ref())? else { continue };
            let index: Vec<usize> = (0..batch).filter(|&b| found.moving[b]).collect();
            let energy = energies(&found.x, width, index.len());
            let mut place = |r: usize, start: usize, size: usize, block: &[f64]| {
                for i in 0..width {
                    let at = (index[r] * width + i) * kept + start;
                    out.v[at..at + size].copy_from_slice(&block[i * size..(i + 1) * size]);
                }
            };
            if self.config.address_reward != 0.0 {
                let rate = self.config.address;
                for r in 0..index.len() {
                    let bound = np::minimum(1.0, 1.0 / (rate * energy[r]));
                    let x = row(&found.x, width, r);
                    let mut block = Vec::with_capacity(width * columns);
                    for &xi in x {
                        for &e in row(&found.error, columns, r) {
                            block.push(rate * (0.0 + xi * (e * bound)));
                        }
                    }
                    if block.iter().all(|v| v.is_finite()) {
                        place(r, h * columns, columns, &block);
                    }
                }
            }
            if !positions {
                continue;
            }
            // the position lesson's step at each row, at the founder's lesson rate
            let gradient: Vec<f64> = found.target.iter().zip(&found.prior)
                .map(|(&t, &p)| if p.is_finite() { t - np::exp(p) } else { 0.0 }).collect();
            let rate = REWARD_RATE;
            let mut blocks: Vec<(Vec<usize>, bool, usize)> = vec![(self.buckets()?, false, count)];
            if !self.maps[ANCHOR].is_empty() {
                blocks.push((self.anchored_buckets(Some(&index)), true, count + 1));
            }
            let mut start = if self.address_held { heads * columns } else { 0 };
            for (buckets, each, size) in blocks {
                let terms = per_bucket(&gradient, &buckets, each, index.len(), m, size);
                for r in 0..index.len() {
                    let bound = np::minimum(1.0, 1.0 / (rate * energy[r]));
                    let x = row(&found.x, width, r);
                    let mut block = Vec::with_capacity(width * size);
                    for &xi in x {
                        for &t in row(&terms, size, r) {
                            block.push(rate * (0.0 + xi * (t * bound)));
                        }
                    }
                    if block.iter().all(|v| v.is_finite()) {
                        place(r, start + h * size, size, &block);
                    }
                }
                start += heads * size;
            }
        }
        Ok(Some(out))
    }

    // -- the reward's step

    /// The maps after a reward's `step` (`propose`), not installed, written into `out` as
    /// `cadence.core.parts.recall` names a proposal; nothing where reward trains nothing.
    pub fn propose(&mut self, step: F, out: &mut Named) -> Result<(), Error> {
        let Some(proposal) = self.proposal(step)? else { return Ok(()) };
        let mut stepped: Vec<Option<&Vec<Vec<f64>>>> = vec![None; KINDS.len()];
        for (kind, xs) in &proposal.stepped {
            stepped[*kind] = Some(xs);
        }
        // a stepped map with its new rows for the event, as the Python's proposal holds it
        let put = |out: &mut Named, kind: usize, name: &str| {
            for (h, maps) in self.maps[kind].iter().enumerate() {
                let mut maps = maps.clone();
                if let Some(xs) = stepped[kind] {
                    maps.x.v.copy_from_slice(&xs[h]);
                }
                maps.put(out, &format!("{name}.{h}"));
            }
        };
        put(out, ADDRESS, "address");
        out.put("norm", Array::scalar_f64(proposal.norm));
        for (kind, _) in &proposal.stepped {
            if *kind != ADDRESS {
                put(out, *kind, KINDS[*kind]);
            }
        }
        self.spare.0.extend(proposal.stepped.into_iter().flat_map(|(_, xs)| xs));
        Ok(())
    }

    /// The step `propose` makes, held apart from the maps: each stepped kind's new rows for
    /// the event (`x`), one per head, and the norm of the step; None where reward trains
    /// nothing. Its new rows take their room from the rows earlier proposals replaced.
    pub fn proposal(&mut self, step: F) -> Result<Option<Proposal>, Error> {
        let positions = self.config.position_reward != 0.0 && !self.maps[POSITION].is_empty();
        if !(self.config.address_reward != 0.0 || positions) {
            return Ok(None);
        }
        let (width, columns, heads) = (self.width(), self.width(), self.heads());
        let blocks = self.reward_blocks();
        let kept: usize = blocks.iter().map(|block| block.1).sum();
        let full = [width + self.cortex() + 1, self.reward_columns()];
        let step = if step.shape == full {
            // what the actor keeps of an eligibility laid out in full (`kept_eligibility`)
            let mut v = Vec::with_capacity(width * kept);
            for i in 0..width {
                for &(start, w) in &blocks {
                    v.extend_from_slice(&step.v[i * full[1] + start..i * full[1] + start + w]);
                }
            }
            F { shape: vec![width, kept], v }
        } else if step.shape == [width, kept] {
            step
        } else {
            return Err(refused(&format!(
                "an address map's step is a ({}, {}) array", full[0], full[1])));
        };
        let split = if self.address_held || self.config.address_reward != 0.0 {
            heads * columns
        } else {
            0
        };
        // the parts stepped, in order: (kind, first column of the step, columns a head, rate)
        let mut parts: Vec<(usize, usize, usize, f64)> = Vec::new();
        if self.config.address_reward != 0.0 {
            parts.push((ADDRESS, 0, columns, self.config.address_reward));
        }
        if positions {
            let count = self.position_buckets();
            let mut start = split;
            let mut kinds = vec![(POSITION, count)];
            if !self.maps[ANCHOR].is_empty() {
                kinds.push((ANCHOR, count + 1));
            }
            for (kind, size) in kinds {
                parts.push((kind, start, size, self.config.position_reward));
                start += heads * size;
            }
        }
        // each part's step (`part`: its columns at its rate), every head's maps stepped by it,
        // and one norm over the parts in order (`np.linalg.norm` of them joined); every value
        // stepped must be finite
        let mut finite = true;
        let mut norm2 = 0.0f64;
        let mut stepped = Vec::with_capacity(parts.len());
        for &(kind, start, size, rate) in &parts {
            let wide = heads * size;
            let mut xs = Vec::with_capacity(heads);
            for h in 0..heads {
                let old = &self.maps[kind][h].x.v;
                let mut x = self.spare.0.pop().unwrap_or_default();
                x.clear();
                x.extend_from_slice(old);
                for i in 0..width {
                    for j in 0..size {
                        x[i * size + j] += rate * step.v[i * kept + start + h * size + j];
                    }
                }
                finite &= x.iter().all(|v| v.is_finite());
                xs.push(x);
            }
            for i in 0..width {
                for jj in 0..wide {
                    let moved = rate * step.v[i * kept + start + jj];
                    finite &= moved.is_finite();
                    norm2 += moved * moved;
                }
            }
            stepped.push((kind, xs));
        }
        if !finite {
            self.spare.0.extend(stepped.into_iter().flat_map(|(_, xs)| xs));
            return Err(refused("the address map's reward step must be finite"));
        }
        Ok(Some(Proposal { stepped, norm: np::sqrt(norm2) }))
    }

    /// Install a step `proposal` made: each stepped map's new rows for the event; the rows they
    /// replace are kept as room for the next proposal.
    pub fn adopt(&mut self, proposal: Proposal) {
        for (kind, xs) in proposal.stepped {
            for (h, x) in xs.into_iter().enumerate() {
                let old = std::mem::replace(&mut self.maps[kind][h].x.v, x);
                self.spare.0.push(old);
            }
        }
    }

    /// Install a step `propose` made (`accept`): the address maps, and with the position step
    /// the position maps (and the anchored ones).
    pub fn accept(&mut self, proposal: &Named) -> Result<(), Error> {
        let (width, cortex) = (self.width(), self.cortex());
        let count = self.position_buckets();
        let read = |prefix: &str, cols: usize| -> Result<Vec<Maps>, Error> {
            let mut maps = Vec::new();
            while proposal.has(&format!("{prefix}.{}.x", maps.len())) {
                let name = format!("{prefix}.{}", maps.len());
                maps.push(Maps::read(proposal, &name, width, cortex, Some(cols))?);
            }
            Ok(maps)
        };
        let address = read("address", width)?;
        let position = read("position", count)?;
        let anchor = read("anchor", count + 1)?;
        if address.len() != self.heads()
            || [&position, &anchor].iter().any(|m| !m.is_empty() && m.len() != self.heads())
        {
            return Err(refused("a proposal holds every head's maps"));
        }
        self.maps[ADDRESS] = address;
        if !position.is_empty() {
            self.maps[POSITION] = position;
        }
        if !anchor.is_empty() {
            self.maps[ANCHOR] = anchor;
        }
        Ok(())
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "stimulate" => {
                let drive = self.stimulate(F::read(args, "drive")?,
                                           F::read(args, "observations")?)?;
                out.put("drive", drive.array());
            }
            "witness" => {
                if args.has("surprise") {
                    self.port(F::read(args, "observations")?)?;
                    return Err(refused("a moment's surprise deepens its slot: it needs strength"));
                }
                self.witness(F::read(args, "observations")?, F::read(args, "activation")?)?;
            }
            "contrast_rows" => {
                let rows = self.contrast_rows(&args.f64s("weights")?, F::read(args, "plus")?,
                                              F::read(args, "minus")?)?;
                if let Some(rows) = rows {
                    out.put("rows", rows.array());
                }
            }
            "propose" => self.propose(F::read(args, "step")?, out)?,
            "accept" => self.accept(args)?,
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
