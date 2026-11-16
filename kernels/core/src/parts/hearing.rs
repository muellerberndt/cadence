//! `cadence.stream.Hearing`: one learned gain per carried route (a held memory's columns and
//! its one-way synapses into the association), the contrast that teaches each gain, and the
//! step on it over the root of its own second moment. The arithmetic as the Python's in its
//! order: the host's contrast reads each route back as a matrix product (BLAS natively, in
//! order here) and sums it in `einsum`'s order; the compiled moment's contrast reads the route
//! synapse by synapse, as its kernel does.

use super::trace::{flag, place, Rows};
use crate::np;
use crate::state::{Array, Error, Named};

const RHO: f64 = 0.99;
const FLOOR: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// The memory it holds (its place among the memories).
    pub memory: usize,
    /// The gain's place in the memory's gains, -1 where the memory holds one gain.
    pub slot: i64,
    pub start: i64,
    pub stop: i64,
    pub width: i64,
    pub bound: f64,
    pub synapses: Vec<i64>,
    /// Each synapse's entry of the route's `(width, association)` matrix.
    pub places: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub trace: Rows,
    pub amplitude: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hearing {
    pub rate: f64,
    pub compiled_moment: bool,
    pub association: Vec<i64>,
    pub slope: f64,
    pub threshold: f64,
    pub rest: f64,
    pub leak: f64,
    pub second: Vec<f64>,
    pub count: i64,
    pub gains: Vec<f64>,
    pub memories: Vec<Memory>,
    pub routes: Vec<Route>,
}

/// What a contrast reads of the graph: its weights, whether its efficacies rest on the host,
/// whether its synapses come in pairs, and whether a device holds them.
pub struct Graph {
    pub weights: Vec<f64>,
    pub efficacy: bool,
    pub parallel: bool,
    pub device: bool,
}

impl Graph {
    fn read(args: &Named) -> Result<Graph, Error> {
        Ok(Graph {
            weights: args.f64s("weights")?,
            efficacy: flag(args, "efficacy")?,
            parallel: flag(args, "parallel")?,
            device: flag(args, "device")?,
        })
    }
}

impl Hearing {
    pub fn import(s: &Named) -> Result<Hearing, Error> {
        let count = |name: &str| -> Result<usize, Error> {
            let n = s.i64(name)?;
            (n >= 0).then_some(n as usize).ok_or(Error::Shape(name.to_string()))
        };
        let (routes, memories) = (count("routes")?, count("memories")?);
        let mut held = Hearing {
            rate: s.f64("config.rate")?,
            compiled_moment: flag(s, "compiled_moment")?,
            association: s.i64s("association")?,
            slope: s.f64("model.slope")?,
            threshold: s.f64("model.threshold")?,
            rest: s.f64("model.rest")?,
            leak: s.f64("model.leak")?,
            second: s.f64s("second")?,
            count: s.i64("count")?,
            gains: s.f64s("gains")?,
            memories: Vec::with_capacity(memories),
            routes: Vec::with_capacity(routes),
        };
        if held.gains.len() != routes || s.shape("gains")?.len() != 1
            || s.shape("second")?.len() != 1
        {
            return Err(Error::Shape("one gain per route".into()));
        }
        for m in 0..memories {
            held.memories.push(Memory {
                trace: Rows::read(s, &format!("memory.{m}.trace"))?,
                amplitude: s.f64(&format!("memory.{m}.amplitude"))?,
            });
        }
        for k in 0..routes {
            let name = |part: &str| format!("route.{k}.{part}");
            let memory = s.i64(&name("memory"))?;
            if memory < 0 || memory as usize >= memories {
                return Err(Error::Shape(name("memory")));
            }
            let route = Route {
                memory: memory as usize,
                slot: s.i64(&name("slot"))?,
                start: s.i64(&name("start"))?,
                stop: s.i64(&name("stop"))?,
                width: s.i64(&name("width"))?,
                bound: s.f64(&name("bound"))?,
                synapses: s.i64s(&name("synapses"))?,
                places: s.i64s(&name("places"))?,
            };
            if route.synapses.len() != route.places.len() || route.width < 0 {
                return Err(Error::Shape(name("places")));
            }
            held.routes.push(route);
        }
        Ok(held)
    }

    pub fn export(&self, out: &mut Named) {
        out.put("config.rate", Array::scalar_f64(self.rate));
        out.put("compiled_moment", Array::from_bool(&[], &[self.compiled_moment]));
        out.put("association", Array::from_i64(&[self.association.len()], &self.association));
        out.put("model.slope", Array::scalar_f64(self.slope));
        out.put("model.threshold", Array::scalar_f64(self.threshold));
        out.put("model.rest", Array::scalar_f64(self.rest));
        out.put("model.leak", Array::scalar_f64(self.leak));
        out.put("second", Array::from_f64(&[self.second.len()], &self.second));
        out.put("count", Array::scalar_i64(self.count));
        out.put("gains", Array::from_f64(&[self.gains.len()], &self.gains));
        out.put("routes", Array::scalar_i64(self.routes.len() as i64));
        out.put("memories", Array::scalar_i64(self.memories.len() as i64));
        for (m, memory) in self.memories.iter().enumerate() {
            out.put(&format!("memory.{m}.trace"), memory.trace.array());
            out.put(&format!("memory.{m}.amplitude"), Array::scalar_f64(memory.amplitude));
        }
        for (k, r) in self.routes.iter().enumerate() {
            let name = |part: &str| format!("route.{k}.{part}");
            out.put(&name("memory"), Array::scalar_i64(r.memory as i64));
            out.put(&name("slot"), Array::scalar_i64(r.slot));
            out.put(&name("start"), Array::scalar_i64(r.start));
            out.put(&name("stop"), Array::scalar_i64(r.stop));
            out.put(&name("width"), Array::scalar_i64(r.width));
            out.put(&name("bound"), Array::scalar_f64(r.bound));
            out.put(&name("synapses"), Array::from_i64(&[r.synapses.len()], &r.synapses));
            out.put(&name("places"), Array::from_i64(&[r.places.len()], &r.places));
        }
    }

    fn route(&self, args: &Named) -> Result<usize, Error> {
        let k = args.i64("route")?;
        (k >= 0 && (k as usize) < self.routes.len())
            .then_some(k as usize)
            .ok_or(Error::Shape("no such route".into()))
    }

    /// `step(contrast)`: every route's gain on its own contrast, over the root of its own
    /// second moment, within `[0, bound]`.
    pub fn step(&mut self, contrast: &[f64]) -> Result<(), Error> {
        let n = self.routes.len();
        if self.second.len() != n || (contrast.len() != n && contrast.len() != 1) {
            return Err(Error::Shape("one contrast per route".into()));
        }
        let c = |k: usize| if contrast.len() == 1 { contrast[0] } else { contrast[k] };
        self.count += 1;
        for k in 0..n {
            self.second[k] = RHO * self.second[k] + (1.0 - RHO) * (c(k) * c(k));
        }
        // Python's float ** int: the C library's pow
        let correction = 1.0 - crate::np::pow(RHO, self.count as f64);
        for k in 0..n {
            let moved = self.rate * c(k) / (np::sqrt(self.second[k] / correction) + FLOOR);
            self.gains[k] = np::clip(self.gains[k] + moved, 0.0, self.routes[k].bound);
        }
        Ok(())
    }

    /// `_contrast(graph, movement, rows, batch)`: `c` per row and route, the association's
    /// `movement` over the phases' span at `rows` of `batch` streams; a memory of another batch
    /// reads silence.
    pub fn contrast(&self, movement: &Rows, rows: Option<&Named>, batch: i64, graph: &Graph)
                    -> Result<Rows, Error> {
        if graph.device {
            return Err(Error::Shape("the synapses rest on a device".into()));
        }
        let a = self.association.len();
        if movement.cols != a {
            return Err(Error::Shape("a movement is one row of the association per stream".into()));
        }
        let b = movement.rows;
        let mut out = Rows::zeros(b, self.routes.len());
        let compiled = self.compiled_moment && graph.efficacy && !graph.parallel;
        for (k, route) in self.routes.iter().enumerate() {
            let memory = &self.memories[route.memory];
            if memory.trace.rows as i64 != batch {
                continue;
            }
            let picked: Vec<usize> = match rows {
                None => (0..memory.trace.rows).collect(),
                Some(args) => args
                    .i64s("rows")?
                    .into_iter()
                    .map(|i| place(i, memory.trace.rows, "rows"))
                    .collect::<Result<_, _>>()?,
            };
            if route.start < 0 || route.stop < route.start
                || route.stop as usize > memory.trace.cols
            {
                return Err(Error::Shape("a route's columns lie within its memory".into()));
            }
            let (start, width) = (route.start as usize, (route.stop - route.start) as usize);
            let held = |i: usize, j: usize| memory.trace.at(picked[i], start + j);
            let h = picked.len();
            let amplitude = memory.amplitude;
            let scale = amplitude * self.gains[k];
            let mut column = vec![0.0f64; h];
            // each synapse's weight, row and column of the route's matrix
            let mut entries = Vec::with_capacity(route.synapses.len());
            for (&s, &p) in route.synapses.iter().zip(&route.places) {
                let weight = graph.weights[place(s, graph.weights.len(), "synapses")?];
                if p < 0 || p as usize >= route.width as usize * a {
                    return Err(Error::Shape("a place outside the route's matrix".into()));
                }
                entries.push((weight, p as usize / a, p as usize % a));
            }
            if compiled {
                // `_moment.route_contrast`: the movement read back synapse by synapse
                if h > b || entries.iter().any(|&(_, row, _)| row >= width) {
                    return Err(Error::Shape("the route's rows and the movement's".into()));
                }
                let mut back = vec![0.0f64; width];
                for (i, c) in column.iter_mut().enumerate() {
                    back.fill(0.0);
                    let moved = movement.row(i);
                    for &(weight, row, col) in &entries {
                        back[row] += weight * moved[col];
                    }
                    let mut total = 0.0f64;
                    for (j, &bj) in back.iter().enumerate() {
                        let x = held(i, j);
                        let v = scale * x;
                        let sig = 1.0 / (1.0 + np::exp(-self.slope * (v - self.threshold)));
                        let mut d = self.slope * sig * (1.0 - sig);
                        d = if sig > self.rest {
                            d / (1.0 - self.rest)
                        } else {
                            self.leak * d / self.rest
                        };
                        total += d * amplitude * x * bj;
                    }
                    *c = total;
                }
            } else {
                // movement @ w.T, then einsum("bj,bj->b") of the slope-weighted held values
                let rw = route.width as usize;
                if h != b || width != rw {
                    return Err(Error::Shape("the route's rows and the movement's".into()));
                }
                let mut w = vec![0.0f64; rw * a];
                for &(weight, row, col) in &entries {
                    w[row * a + col] = weight;
                }
                let (mut back, mut weighed) = (vec![0.0f64; rw], vec![0.0f64; rw]);
                for (i, c) in column.iter_mut().enumerate() {
                    for (j, bj) in back.iter_mut().enumerate() {
                        *bj = np::dot(movement.row(i), &w[j * a..(j + 1) * a]);
                    }
                    for (j, s) in weighed.iter_mut().enumerate() {
                        let x = held(i, j);
                        *s = self.slope_at(scale * x) * amplitude * x;
                    }
                    *c = np::einsum_dot(&weighed, &back);
                }
            }
            if h != b && h != 1 {
                return Err(Error::Shape("a route's contrast is one per row".into()));
            }
            for i in 0..b {
                out.data[i * self.routes.len() + k] = column[if h == 1 { 0 } else { i }];
            }
        }
        Ok(out)
    }

    /// The neuron model's `slope_at`: the derivative of the activation at `v`.
    fn slope_at(&self, v: f64) -> f64 {
        let sig = 1.0 / (1.0 + np::exp(-self.slope * (v - self.threshold)));
        let d = self.slope * sig * (1.0 - sig);
        if sig > self.rest {
            d / (1.0 - self.rest)
        } else {
            self.leak * d / self.rest
        }
    }

    /// `contrast_rows(graph, plus, minus, beta)`: a sampled act's `c` per row and route.
    pub fn contrast_rows(&self, plus: &Rows, minus: &Rows, beta: f64, graph: &Graph)
                         -> Result<Rows, Error> {
        if plus.rows != minus.rows || plus.cols != minus.cols {
            return Err(Error::Shape("the two phases differ in shape".into()));
        }
        let at: Vec<usize> = self
            .association
            .iter()
            .map(|&i| place(i, plus.cols, "association"))
            .collect::<Result<_, _>>()?;
        let span = 2.0 * beta;
        let mut movement = Rows::zeros(plus.rows, at.len());
        for r in 0..plus.rows {
            for (j, &c) in at.iter().enumerate() {
                movement.data[r * at.len() + j] = (plus.at(r, c) - minus.at(r, c)) / span;
            }
        }
        self.contrast(&movement, None, plus.rows as i64, graph)
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "gain" => {
                let k = self.route(args)?;
                out.put("gain", Array::scalar_f64(self.gains[k]));
            }
            "set" => {
                let k = self.route(args)?;
                self.gains[k] = args.f64("value")?;
            }
            "step" => {
                if args.shape("contrast")?.len() > 1 {
                    return Err(Error::Shape("one contrast per route".into()));
                }
                self.step(&args.f64s("contrast")?)?;
            }
            "_contrast" => {
                let movement = Rows::read(args, "movement")?;
                let rows = args.has("rows").then_some(args);
                let c = self.contrast(&movement, rows, args.i64("batch")?, &Graph::read(args)?)?;
                out.put("contrast", c.array());
            }
            "contrast_rows" => {
                if flag(args, "kernel")? {
                    return Err(Error::Shape("the phases rest on a device".into()));
                }
                let c = self.contrast_rows(&Rows::read(args, "plus")?, &Rows::read(args, "minus")?,
                                           args.f64("beta")?, &Graph::read(args)?)?;
                out.put("contrast", c.array());
            }
            "_movement_on_device" => {
                if flag(args, "kernel")? {
                    return Err(Error::Shape("the phases rest on a device".into()));
                }
            }
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
