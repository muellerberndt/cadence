//! `cadence.longterm.LongTermMemory`: records under a fixed budget of slots, kept by depth and
//! recent use, revised where a key is told again, as the Python computes them, in its order.
//! The core holds a store that reads every held slot (no coarse stage: Wild's stores, and every
//! store of at most `direct` slots), in float64, without a source's signature; a store of
//! another kind is refused.

use crate::np;
use crate::state::{refused, Array, Error, Named};

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub width: usize,
    pub value_width: usize,
    pub capacity: usize,
    pub temperature: f64,
    pub candidates: i64,
    pub expansion: i64,
    pub winners: i64,
    pub depth: f64,
    pub idle: f64,
    pub hold: f64,
    pub revise: f64,
    pub fade: bool,
    pub seed: i64,
    pub direct: i64,
    pub cosine: bool,
    pub pairs: i64,
    pub apart: i64,
    pub blend: bool,
    pub familiar: f64,
    pub keep: f64,
    pub forget: f64,
    pub novelty: f64,
    pub coarse: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LongTerm {
    pub config: Config,
    pub keys: Vec<f64>,
    pub values: Vec<f64>,
    pub deep: Vec<f64>,
    pub recent: Vec<f64>,
    pub faded: Vec<f64>,
    pub norms: Vec<f64>,
    pub kept: Vec<bool>,
    pub gone: Vec<bool>,
    pub count: usize,
    pub writes: i64,
    pub cursor: usize,
}

impl Config {
    /// `LongTermMemory(width, value_width, capacity)`'s settings: the founder's defaults, with
    /// `idle`, `hold`, `direct` and the coarse stage as its construction works them out. A
    /// caller sets the strength's options (`cosine`, `blend`, `revise`, `keep`, `forget`,
    /// `seed`) on what it returns.
    pub fn new(width: usize, value_width: usize, capacity: usize) -> Config {
        let quarter = capacity as f64 / 16.0;
        let (candidates, expansion) = (256i64, 8192i64);
        let direct = 2 * expansion;
        Config {
            width,
            value_width,
            capacity,
            temperature: 0.0,
            candidates,
            expansion,
            winners: 64,
            depth: 2.0,
            idle: if 2.0 > quarter { 2.0 } else { quarter },
            hold: 8.0 * capacity as f64,
            revise: 0.9,
            fade: true,
            seed: 0,
            direct,
            cosine: false,
            pairs: 0,
            apart: 0,
            blend: false,
            familiar: 0.0,
            keep: 0.0,
            forget: 0.0,
            novelty: 0.0,
            coarse: candidates != 0 && capacity as i64 > candidates.max(direct),
        }
    }
}

// the fading's constants (`LongTermMemory.ORDER` is 4: two squarings)
const FLOOR: f64 = 0.01;

/// `_norms` of one row: the root of its squares summed as NumPy sums a row.
pub fn row_norm(row: &[f64]) -> f64 {
    np::sqrt(np::sum_squares(row))
}

fn shaped(s: &Named, name: &str, shape: &[usize]) -> Result<(), Error> {
    if s.shape(name)? != shape {
        return Err(Error::Shape(name.to_string()));
    }
    Ok(())
}

impl LongTerm {
    /// An empty store of `config`'s settings (`LongTermMemory(...)`); one the core does not
    /// hold (a coarse stage, a source's signature) is a fault, settings the Python refuses a
    /// refusal.
    pub fn new(config: Config) -> Result<LongTerm, Error> {
        if config.coarse || config.apart != 0 {
            return Err(Error::Shape("the core holds stores read whole, without a source".into()));
        }
        if config.width < 1 || config.value_width < 1 || config.capacity < 1 {
            return Err(refused("width, value_width and capacity must be positive integers"));
        }
        if config.blend && !config.cosine {
            return Err(refused("blend needs cosine=True"));
        }
        if (config.keep != 0.0 || config.forget != 0.0) && !config.fade {
            return Err(refused("keep and forget read the fade: they need fade=True"));
        }
        let n = config.capacity;
        Ok(LongTerm {
            keys: vec![0.0; n * config.width],
            values: vec![0.0; n * config.value_width],
            deep: vec![0.0; n],
            recent: vec![0.0; n],
            faded: vec![0.0; n],
            norms: vec![0.0; if config.cosine { n } else { 0 }],
            kept: vec![false; if config.keep != 0.0 { n } else { 0 }],
            gone: vec![false; if config.forget != 0.0 { n } else { 0 }],
            count: 0,
            writes: 0,
            cursor: 0,
            config,
        })
    }

    /// The held slots not forgotten, in order (the records a reader such as Wild's kept words
    /// lists: each slot's `value`, `deep` and `kept`).
    pub fn live(&self) -> Vec<usize> {
        self.candidates()
    }

    /// How many records are held and not forgotten.
    pub fn live_count(&self) -> usize {
        self.live().len()
    }

    pub fn import(s: &Named) -> Result<LongTerm, Error> {
        let f = |n: &str| s.f64(&format!("config.{n}"));
        let i = |n: &str| s.i64(&format!("config.{n}"));
        let b = |n: &str| -> Result<bool, Error> {
            let v = s.bools(&format!("config.{n}"))?;
            (v.len() == 1).then(|| v[0]).ok_or_else(|| Error::Shape(n.to_string()))
        };
        let positive = |x: i64, n: &str| -> Result<usize, Error> {
            (x >= 1).then_some(x as usize).ok_or_else(|| Error::Shape(n.to_string()))
        };
        let config = Config {
            width: positive(i("width")?, "width")?,
            value_width: positive(i("value_width")?, "value_width")?,
            capacity: positive(i("capacity")?, "capacity")?,
            temperature: f("temperature")?,
            candidates: i("candidates")?,
            expansion: i("expansion")?,
            winners: i("winners")?,
            depth: f("depth")?,
            idle: f("idle")?,
            hold: f("hold")?,
            revise: f("revise")?,
            fade: b("fade")?,
            seed: i("seed")?,
            direct: i("direct")?,
            cosine: b("cosine")?,
            pairs: i("pairs")?,
            apart: i("apart")?,
            blend: b("blend")?,
            familiar: f("familiar")?,
            keep: f("keep")?,
            forget: f("forget")?,
            novelty: f("novelty")?,
            coarse: b("coarse")?,
        };
        if config.coarse || config.apart != 0 {
            return Err(Error::Shape("the core holds stores read whole, without a source".into()));
        }
        let (n, w, v) = (config.capacity, config.width, config.value_width);
        let name = |x: &str| x.to_string();
        shaped(s, &name("keys"), &[n, w])?;
        shaped(s, &name("values"), &[n, v])?;
        for x in ["deep", "recent", "faded"] {
            shaped(s, &name(x), &[n])?;
        }
        shaped(s, &name("norms"), &[if config.cosine { n } else { 0 }])?;
        shaped(s, &name("kept"), &[if config.keep != 0.0 { n } else { 0 }])?;
        shaped(s, &name("gone"), &[if config.forget != 0.0 { n } else { 0 }])?;
        let held = s.i64s(&name("state"))?;
        if held.len() != 3 || held[0] < 0 || held[0] as usize > n || held[2] < 0
            || held[2] as usize >= n
        {
            return Err(Error::Shape(name("state")));
        }
        Ok(LongTerm {
            keys: s.f64s(&name("keys"))?,
            values: s.f64s(&name("values"))?,
            deep: s.f64s(&name("deep"))?,
            recent: s.f64s(&name("recent"))?,
            faded: s.f64s(&name("faded"))?,
            norms: s.f64s(&name("norms"))?,
            kept: s.bools(&name("kept"))?,
            gone: s.bools(&name("gone"))?,
            count: held[0] as usize,
            writes: held[1],
            cursor: held[2] as usize,
            config,
        })
    }

    pub fn export(&self, out: &mut Named) {
        let c = &self.config;
        let f = |out: &mut Named, n: &str, x: f64| {
            out.put(&format!("config.{n}"), Array::scalar_f64(x))
        };
        let i = |out: &mut Named, n: &str, x: i64| {
            out.put(&format!("config.{n}"), Array::scalar_i64(x))
        };
        let b = |out: &mut Named, n: &str, x: bool| {
            out.put(&format!("config.{n}"), Array::from_bool(&[], &[x]))
        };
        i(out, "width", c.width as i64);
        i(out, "value_width", c.value_width as i64);
        i(out, "capacity", c.capacity as i64);
        f(out, "temperature", c.temperature);
        i(out, "candidates", c.candidates);
        i(out, "expansion", c.expansion);
        i(out, "winners", c.winners);
        f(out, "depth", c.depth);
        f(out, "idle", c.idle);
        f(out, "hold", c.hold);
        f(out, "revise", c.revise);
        b(out, "fade", c.fade);
        i(out, "seed", c.seed);
        i(out, "direct", c.direct);
        b(out, "cosine", c.cosine);
        i(out, "pairs", c.pairs);
        i(out, "apart", c.apart);
        b(out, "blend", c.blend);
        f(out, "familiar", c.familiar);
        f(out, "keep", c.keep);
        f(out, "forget", c.forget);
        f(out, "novelty", c.novelty);
        b(out, "coarse", c.coarse);
        let (n, w, v) = (c.capacity, c.width, c.value_width);
        out.put("keys", Array::from_f64(&[n, w], &self.keys));
        out.put("values", Array::from_f64(&[n, v], &self.values));
        out.put("deep", Array::from_f64(&[n], &self.deep));
        out.put("recent", Array::from_f64(&[n], &self.recent));
        out.put("faded", Array::from_f64(&[n], &self.faded));
        out.put("norms", Array::from_f64(&[self.norms.len()], &self.norms));
        out.put("kept", Array::from_bool(&[self.kept.len()], &self.kept));
        out.put("gone", Array::from_bool(&[self.gone.len()], &self.gone));
        out.put("state", Array::from_i64(
            &[3], &[self.count as i64, self.writes, self.cursor as i64]));
    }

    pub fn key(&self, slot: usize) -> &[f64] {
        let w = self.config.width;
        &self.keys[slot * w..(slot + 1) * w]
    }

    pub fn value(&self, slot: usize) -> &[f64] {
        let v = self.config.value_width;
        &self.values[slot * v..(slot + 1) * v]
    }

    /// `_candidates`: every held slot, a forgotten one left out (the store reads every slot).
    pub fn candidates(&self) -> Vec<usize> {
        let held = 0..self.count;
        if self.config.forget != 0.0 {
            held.filter(|&s| !self.gone[s]).collect()
        } else {
            held.collect()
        }
    }

    /// `keys[slots] @ cue`, each dot product in order.
    pub fn dots(&self, slots: &[usize], cue: &[f64]) -> Vec<f64> {
        slots.iter().map(|&s| np::dot(self.key(s), cue)).collect()
    }

    /// `_cosines`: the dot products over the keys' lengths and the cue's, 0 where either is 0.
    fn cosines(&self, slots: &[usize], dots: &[f64], cue: &[f64]) -> Vec<f64> {
        let length = row_norm(cue);
        slots.iter().zip(dots).map(|(&s, &d)| {
            let l = self.norms[s] * length;
            if l > 0.0 { d / l } else { 0.0 }
        }).collect()
    }

    /// `_novel`: 1 less the cosine of the record nearest `key`, within [0, 1].
    fn novel(&self, key: &[f64]) -> f64 {
        if self.count == 0 {
            return 1.0;
        }
        let slots = self.candidates();
        if slots.is_empty() {
            return 1.0;
        }
        let dots = self.dots(&slots, key);
        let length = np::norm(key);
        let mut near = f64::NEG_INFINITY;
        let mut first = true;
        for (&s, &d) in slots.iter().zip(&dots) {
            let l = if self.config.cosine { self.norms[s] } else { row_norm(self.key(s)) } * length;
            let x = if l > 0.0 { d / l } else { 0.0 };
            // `near.max()`: a NaN wins
            if first || x.is_nan() || (!near.is_nan() && x > near) {
                near = x;
            }
            first = false;
        }
        np::clip(1.0 - near, 0.0, 1.0)
    }

    /// `_replaced`: the most faded record not kept, else the shallowest kept one.
    fn replaced(&self) -> usize {
        if self.kept.iter().any(|&k| !k) {
            let open: Vec<f64> = self.faded.iter().zip(&self.kept)
                .map(|(&f, &k)| if k { f64::NEG_INFINITY } else { f }).collect();
            np::argmax(&open)
        } else {
            np::argmin(&self.deep)
        }
    }

    /// `write(key, value, recognised=, surprise=)`: the slot written and whether the write
    /// revised a held record. A key, value or surprise outside the finite numbers is refused.
    pub fn write(&mut self, key: &[f64], value: &[f64], recognised: Option<i64>,
                 surprise: Option<f64>) -> Result<(usize, bool), Error> {
        let c = self.config.clone();
        if key.len() != c.width || !key.iter().all(|x| x.is_finite()) {
            return Err(refused(&format!("a key is one finite row of {} values", c.width)));
        }
        if value.len() != c.value_width || !value.iter().all(|x| x.is_finite()) {
            return Err(refused(&format!("a value is one finite row of {} values",
                                            c.value_width)));
        }
        if let Some(r) = recognised {
            if r < -1 || r >= self.count as i64 {
                return Err(refused("recognised is a held slot, or -1 for none"));
            }
        }
        let surprise = match surprise {
            Some(x) if !x.is_finite() || x < 0.0 => {
                return Err(refused("surprise is a finite nonnegative real"));
            }
            Some(x) => Some(x),
            None if c.novelty != 0.0 => Some(c.novelty * self.novel(key)),
            None => None,
        };
        let (mut slot, mut revised) = (usize::MAX, false);
        if let (Some(r), true) = (recognised, c.revise != 0.0) {
            if r >= 0 {
                slot = r as usize;
                revised = true;
            }
        } else if c.revise != 0.0 && self.count > 0 {
            let slots = self.candidates();
            if !slots.is_empty() {
                let scores = self.dots(&slots, key);
                let best = if c.cosine {
                    np::argmax(&self.cosines(&slots, &scores, key))
                } else {
                    np::argmax(&scores)
                };
                let norm = np::norm(key) * np::norm(self.key(slots[best]));
                if norm > 0.0 && scores[best] / norm >= c.revise {
                    slot = slots[best];
                    revised = true;
                }
            }
        }
        let mut key = key.to_vec();
        let mut value = value.to_vec();
        if slot == usize::MAX {
            if self.count < c.capacity {
                slot = self.count;
                self.count += 1;
            } else if c.keep != 0.0 {
                slot = self.replaced();
            } else if c.fade {
                slot = np::argmax(&self.faded);
            } else {
                slot = self.cursor;
            }
            if !c.fade {
                self.cursor = (slot + 1) % c.capacity;
            }
            self.deep[slot] = 0.0;
            self.recent[slot] = 0.0;
            if c.keep != 0.0 {
                self.kept[slot] = false;
            }
            if c.forget != 0.0 {
                self.gone[slot] = false;
            }
        } else if c.blend && revised {
            // `_blended`: the record moves toward the new take by 1 / (1 + deep)
            let rate = 1.0 / (1.0 + self.deep[slot]);
            for (k, h) in key.iter_mut().zip(self.key(slot)) {
                *k = h + rate * (*k - h);
            }
            for (v, h) in value.iter_mut().zip(self.value(slot)) {
                *v = h + rate * (*v - h);
            }
        }
        let (w, vw) = (c.width, c.value_width);
        self.keys[slot * w..(slot + 1) * w].copy_from_slice(&key);
        self.values[slot * vw..(slot + 1) * vw].copy_from_slice(&value);
        if c.cosine {
            self.norms[slot] = row_norm(&key);
        }
        self.deep[slot] += match surprise {
            None => 1.0,
            Some(x) => 1.0 + x,
        };
        if c.keep != 0.0 && self.deep[slot] >= c.keep {
            self.kept[slot] = true;
        }
        self.faded[slot] = 0.0;
        self.writes += 1;
        Ok((slot, revised))
    }

    /// `moment()`: every held slot fades by its depth's and its recent use's law; a kept one
    /// does not, and one below it that fades to `forget` is forgotten; the sums decay.
    pub fn moment(&mut self) {
        let c = &self.config;
        if !c.fade || self.count == 0 {
            return;
        }
        let n = self.count;
        let (scale, unit) = (1.0 / FLOOR, 1.0 / c.depth);
        for s in 0..n {
            let mut rate = self.recent[s] * scale;
            rate = rate * rate;
            rate += 1.0;
            if c.depth.is_finite() {
                let mut law = self.deep[s] * unit;
                law = law * law;
                law = law * law;
                law += 1.0;
                rate *= law;
            }
            rate = 1.0 / rate;
            if c.keep != 0.0 && self.kept[s] {
                rate = 0.0;
            }
            self.faded[s] += rate;
            if c.forget != 0.0 {
                let mut lost = self.faded[s] >= c.forget;
                if c.keep != 0.0 {
                    lost &= !self.kept[s];
                }
                self.gone[s] |= lost;
            }
        }
        let (recent, deep) = (1.0 - 1.0 / c.idle, 1.0 - 1.0 / c.hold);
        for s in 0..n {
            self.recent[s] *= recent;
        }
        if c.hold.is_finite() {
            for s in 0..n {
                self.deep[s] *= deep;
            }
        }
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "write" => {
                let recognised = if args.has("recognised") { Some(args.i64("recognised")?) }
                                 else { None };
                let surprise = if args.has("surprise") { Some(args.f64("surprise")?) }
                               else { None };
                if args.has("source") {
                    return Err(refused("a source needs apart=its width"));
                }
                let (slot, revised) = self.write(&args.f64s("key")?, &args.f64s("value")?,
                                                 recognised, surprise)?;
                out.put("slot", Array::scalar_i64(slot as i64));
                out.put("revised", Array::from_bool(&[], &[revised]));
            }
            "moment" => self.moment(),
            "state" => {}  // the state as the core holds it: a round trip
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_store_holds_the_founders_settings() {
        // `LongTermMemory(96, 96, 13)` and `LongTermMemory(4, 3, 40000)` in the Python
        let c = Config::new(96, 96, 13);
        assert_eq!((c.idle, c.hold, c.direct, c.coarse), (2.0, 104.0, 16384, false));
        let c = Config::new(4, 3, 40000);
        assert_eq!((c.idle, c.hold, c.coarse), (2500.0, 320000.0, true));
        let mut c = Config::new(96, 96, 13);
        c.cosine = true;
        c.blend = true;
        c.revise = 0.99;
        c.keep = 8.0;
        c.forget = 400.0;
        let mut store = LongTerm::new(c).unwrap();
        let word: Vec<f64> = (0..96).map(|i| (i % 7) as f64).collect();
        assert_eq!(store.write(&word, &word, None, Some(0.0)).unwrap(), (0, false));
        assert_eq!(store.write(&word, &word, None, Some(0.0)).unwrap(), (0, true));
        store.moment();
        assert_eq!((store.live(), store.live_count()), (vec![0], 1));
        assert!(store.deep[0] < 2.0 && store.deep[0] > 1.9);
        assert!(LongTerm::new(Config::new(4, 3, 40000)).is_err());
    }
}
