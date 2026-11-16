//! `cadence.bound.BoundRecords`: a brain's records of its bound moments and its voice, as the
//! Python computes them, in its order. The read of a part (`_read`) is the compiled kernel
//! `crate::bound::recall`, the one `cadence.compiled.bound_recall` calls; the records live in a
//! `LongTerm` store.

use super::longterm::{row_norm, LongTerm};
use crate::np;
use crate::rng::Pcg64;
use crate::state::{refused, Array, Error, Named};
use crate::View;

const HEADS: i32 = 8;
const NEAR: f64 = 0.7;
const MOST: f64 = 0.3;
const ANSWER: f64 = 0.5;
const AGREE: f64 = 0.8;
const MATCH: f64 = 0.15;
const KNOWN: f64 = 0.5;
const PAY: f64 = 4.0;
const NAMED: f64 = 8.0;

/// Where each stream's latest said sound came from (`said_from`): None, said, heard, babble.
pub const FROM_NONE: i64 = -1;
pub const FROM_SAID: i64 = 0;
pub const FROM_HEARD: i64 = 1;
pub const FROM_BABBLE: i64 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    pub first: usize,
    pub last: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bound {
    pub capacity: usize,
    pub seed: i64,
    pub inputs: usize,
    pub fields: Vec<Field>,
    /// The voice's field and the heard field, by their place in `fields`.
    pub voice: Option<usize>,
    pub heard: Option<usize>,
    pub store: LongTerm,
    pub written: Vec<i64>,
    pub refused: Vec<bool>,
    pub paid: Vec<f64>,
    pub draws: i64,
    pub said_from: Vec<i64>,
}

/// `bound._unit`: a vector over its norm where the norm exceeds 1e-12.
fn unit(a: &[f64]) -> Vec<f64> {
    let norm = np::norm(a);
    if norm > 1e-12 {
        a.iter().map(|x| x / norm).collect()
    } else {
        a.to_vec()
    }
}

/// `_unit(a) @ _unit(b)`.
fn alike(a: &[f64], b: &[f64]) -> f64 {
    np::dot(&unit(a), &unit(b))
}

fn any(a: &[f64]) -> bool {
    a.iter().any(|&x| x != 0.0)
}

/// `rows.mean(axis=0)` of rows of one width: the rows added in order to the sum's +0.0 (a
/// width of one summed pairwise), over their count.
fn mean_rows(rows: &[&[f64]]) -> Vec<f64> {
    let m = rows.len() as f64;
    if rows[0].len() == 1 {
        let column: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        return vec![np::sum(&column) / m];
    }
    let mut out = vec![0.0; rows[0].len()];
    for r in rows {
        for (o, x) in out.iter_mut().zip(r.iter()) {
            *o += x;
        }
    }
    out.iter().map(|x| x / m).collect()
}

fn view<T>(data: &[T], n0: usize, n1: usize) -> View<T> {
    View { p: data.as_ptr() as *mut T, n0, n1, len: n0 * n1 }
}

impl Bound {
    pub fn import(s: &Named) -> Result<Bound, Error> {
        let mut fields = Vec::new();
        for name in &s.order {
            if let Some(field) = name.strip_prefix("field:") {
                let span = s.i64s(name)?;
                if span.len() != 2 || span[0] < 0 || span[1] < span[0] {
                    return Err(Error::Shape(name.clone()));
                }
                fields.push(Field { name: field.to_string(), first: span[0] as usize,
                                    last: span[1] as usize });
            }
        }
        let index = |name: &str| -> Result<Option<usize>, Error> {
            let at = s.i64(name)?;
            if at < -1 || at >= fields.len() as i64 {
                return Err(Error::Shape(name.to_string()));
            }
            Ok((at >= 0).then_some(at as usize))
        };
        let (voice, heard) = (index("config.voice")?, index("config.heard")?);
        let store = LongTerm::import(&s.under("store"))?;
        let capacity = s.i64("config.capacity")? as usize;
        let inputs = s.i64("config.inputs")? as usize;
        let refused = s.bools("refused")?;
        let paid = s.f64s("paid")?;
        if fields.is_empty() || capacity != store.config.capacity
            || inputs != store.config.value_width || refused.len() != capacity
            || paid.len() != capacity || fields.iter().any(|f| f.last > inputs)
        {
            return Err(Error::Shape("bound records of another shape".into()));
        }
        let written = s.i64s("written")?;
        if written.iter().any(|&w| w >= capacity as i64) {
            return Err(Error::Shape("written".into()));
        }
        Ok(Bound {
            capacity,
            seed: s.i64("config.seed")?,
            inputs,
            fields,
            voice,
            heard,
            store,
            written,
            refused,
            paid,
            draws: s.i64("draws")?,
            said_from: s.i64s("said_from")?,
        })
    }

    pub fn export(&self, out: &mut Named) {
        out.put("config.capacity", Array::scalar_i64(self.capacity as i64));
        out.put("config.seed", Array::scalar_i64(self.seed));
        out.put("config.inputs", Array::scalar_i64(self.inputs as i64));
        out.put("config.voice", Array::scalar_i64(self.voice.map_or(-1, |v| v as i64)));
        out.put("config.heard", Array::scalar_i64(self.heard.map_or(-1, |v| v as i64)));
        for f in &self.fields {
            out.put(&format!("field:{}", f.name),
                    Array::from_i64(&[2], &[f.first as i64, f.last as i64]));
        }
        let mut store = Named::new();
        self.store.export(&mut store);
        out.put_under("store", &store);
        out.put("written", Array::from_i64(&[self.written.len()], &self.written));
        out.put("refused", Array::from_bool(&[self.capacity], &self.refused));
        out.put("paid", Array::from_f64(&[self.capacity], &self.paid));
        out.put("draws", Array::scalar_i64(self.draws));
        out.put("said_from", Array::from_i64(&[self.said_from.len()], &self.said_from));
    }

    fn span(&self, f: usize) -> (usize, usize) {
        (self.fields[f].first, self.fields[f].last)
    }

    /// `holds`: whether a row holds something in a field the records keep.
    fn holds(&self, row: &[f64]) -> bool {
        self.fields.iter().any(|f| any(&row[f.first..f.last]))
    }

    fn voice_field(&self) -> Result<usize, Error> {
        self.voice.ok_or_else(|| Error::Shape("the records have no voice".into()))
    }

    /// `_known_slots` of one slot: at depth `KNOWN` or kept; a record of something said once
    /// paid or refused; none forgotten.
    fn known(&self, s: usize) -> bool {
        let store = &self.store;
        let mut known = store.deep[s] >= KNOWN;
        if store.config.keep != 0.0 {
            known |= store.kept[s];
        }
        if let Some(v) = self.voice {
            let (a, b) = self.span(v);
            if any(&store.value(s)[a..b]) {
                known = self.paid[s] > 0.0 || self.refused[s];
            }
        }
        if store.config.forget != 0.0 {
            known &= !store.gone[s];
        }
        known
    }

    /// `_taking`: the known records among the store's candidates whose keys lie at cosine
    /// `MATCH` or more to the cue, with their cosines.
    fn taking(&self, code: &[f64]) -> Vec<(usize, f64)> {
        let store = &self.store;
        if store.count == 0 || !any(code) {
            return Vec::new();
        }
        let length = np::norm(code);
        let mut out = Vec::new();
        for s in store.candidates() {
            if !self.known(s) {
                continue;
            }
            let key = store.key(s);
            let norm = row_norm(key) * length;
            let cosine = if norm > 0.0 { np::dot(key, code) / np::maximum(norm, 1e-12) }
                         else { 0.0 };
            if cosine >= MATCH {
                out.push((s, cosine));
            }
        }
        out
    }

    /// `_read`: each field the records a read of `code` reaches hold alike, as its mean, by the
    /// compiled kernel (`cadence.compiled.bound_recall`).
    fn read(&self, code: &[f64]) -> Result<Vec<(usize, Vec<f64>)>, Error> {
        let store = &self.store;
        if store.count == 0 || !any(code) {
            return Ok(Vec::new());
        }
        if code.len() != store.config.width {
            return Err(refused("a code of the store's width"));
        }
        let slots: Vec<i64> = store.candidates().into_iter().map(|s| s as i64).collect();
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let (cap, kw, inputs, nf) = (self.capacity, store.config.width, self.inputs,
                                     self.fields.len());
        let spans: Vec<i64> = self.fields.iter()
            .flat_map(|f| [f.first as i64, f.last as i64]).collect();
        let (v0, v1) = self.voice.map_or((0, 0), |v| self.span(v));
        let out = vec![0.0f64; inputs];
        let present = vec![0i64; nf];
        let kept: Vec<u8> = store.kept.iter().map(|&b| b as u8).collect();
        let gone: Vec<u8> = store.gone.iter().map(|&b| b as u8).collect();
        let refusals: Vec<u8> = self.refused.iter().map(|&b| b as u8).collect();
        let back = unsafe {
            crate::bound::recall(
                view(&slots, slots.len(), 1), view(&store.keys, cap, kw), view(code, kw, 1),
                view(&store.deep, cap, 1), view(&kept, kept.len(), 1),
                (store.config.keep != 0.0) as i32, view(&gone, gone.len(), 1),
                (store.config.forget != 0.0) as i32, view(&store.values, cap, inputs),
                self.voice.is_some() as i32, v0 as i32, v1 as i32, view(&self.paid, cap, 1),
                view(&refusals, cap, 1), view(&spans, nf, 2),
                self.voice.map_or(-1, |v| v as i32), KNOWN, MATCH, HEADS, NEAR, AGREE, MOST,
                view(&out, inputs, 1), view(&present, nf, 1),
            )
        };
        if back < 0 {
            return Err(refused(&format!("the bound records' kernel refused ({back})")));
        }
        Ok((0..nf).filter(|&f| present[f] != 0)
            .map(|f| (f, out[self.fields[f].first..self.fields[f].last].to_vec()))
            .collect())
    }

    /// `complete(rows, codes)`: each field a row leaves silent given what the records bring
    /// back for its code.
    pub fn complete(&self, rows: &[f64], codes: &[f64], batch: usize) -> Result<Vec<f64>, Error> {
        let mut out = rows.to_vec();
        let kw = self.store.config.width;
        for i in 0..batch {
            let row = &out[i * self.inputs..(i + 1) * self.inputs];
            if !self.holds(row) {
                continue;
            }
            for (f, part) in self.read(&codes[i * kw..(i + 1) * kw])? {
                let (a, b) = self.span(f);
                let at = i * self.inputs;
                if !any(&out[at + a..at + b]) {
                    out[at + a..at + b].copy_from_slice(&part);
                }
            }
        }
        Ok(out)
    }

    /// `_answer`: the said and paid sound whose moments share fields nearest the moment's.
    fn answer(&self, taking: &[(usize, f64)], row: &[f64]) -> Result<Option<Vec<f64>>, Error> {
        let vf = self.voice_field()?;
        let (v0, v1) = self.span(vf);
        let held: Vec<usize> = (0..self.fields.len())
            .filter(|&f| f != vf && any(&row[self.fields[f].first..self.fields[f].last]))
            .collect();
        if taking.is_empty() || held.is_empty() {
            return Ok(None);
        }
        let store = &self.store;
        let value = |k: usize| store.value(taking[k].0);
        let sound = |k: usize| &value(k)[v0..v1];
        let mut paid = Vec::new();
        let mut refused = Vec::new();
        for (k, &(s, _)) in taking.iter().enumerate() {
            if any(sound(k)) {
                if self.paid[s] > 0.0 && !self.refused[s] {
                    paid.push(k);
                }
                if self.refused[s] {
                    refused.push(k);
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &k in &paid {
            match groups.iter_mut().find(|g| alike(sound(k), sound(g[0])) >= AGREE) {
                Some(g) => g.push(k),
                None => groups.push(vec![k]),
            }
        }
        let (mut best, mut best_match, mut second): (Option<usize>, f64, f64) = (None, -1.0, -1.0);
        for (g, group) in groups.iter().enumerate() {
            let mut shared = Vec::new();
            let mut near = Vec::new();
            for &f in &held {
                let (a, b) = self.span(f);
                let parts: Vec<&[f64]> = group.iter().map(|&k| &value(k)[a..b]).collect();
                if !parts.iter().all(|p| any(p)) {
                    continue;
                }
                let lead = unit(parts[0]);
                if parts[1..].iter().all(|q| np::dot(&unit(q), &lead) >= AGREE) {
                    shared.push(f);
                    near.push(np::dot(&unit(&row[a..b]), &unit(&mean_rows(&parts))));
                }
            }
            if shared.is_empty() {
                continue;
            }
            let mut matched = np::mean(&near);
            for &k in &refused {
                if alike(sound(k), sound(group[0])) < AGREE {
                    continue;
                }
                let same: Vec<f64> = shared.iter().map(|&f| {
                    let (a, b) = self.span(f);
                    alike(&value(k)[a..b], &row[a..b])
                }).collect();
                if np::mean(&same) >= AGREE {
                    matched = -1.0; // said here and not paid
                    break;
                }
            }
            if matched > best_match {
                second = best_match;
                best = Some(g);
                best_match = matched;
            } else if matched > second {
                second = matched;
            }
        }
        match best {
            Some(g) if best_match >= ANSWER && best_match > second => {
                let sounds: Vec<&[f64]> = groups[g].iter().map(|&k| sound(k)).collect();
                Ok(Some(mean_rows(&sounds)))
            }
            _ => Ok(None),
        }
    }

    /// `_like`: of the fields the moment holds (the voice apart), more than half held alike.
    fn like(&self, value: &[f64], row: &[f64], vf: usize) -> bool {
        let held: Vec<usize> = (0..self.fields.len())
            .filter(|&f| f != vf && any(&row[self.fields[f].first..self.fields[f].last]))
            .collect();
        if held.is_empty() {
            return false;
        }
        let near = held.iter().filter(|&&f| {
            let (a, b) = self.span(f);
            alike(&value[a..b], &row[a..b]) >= AGREE
        }).count();
        near as f64 / held.len() as f64 > 0.5
    }

    /// `_refused_sounds`: the sounds said and not paid at moments like `row`.
    fn refused_sounds(&self, code: &[f64], row: &[f64]) -> Result<Vec<Vec<f64>>, Error> {
        let vf = self.voice_field()?;
        let (a, b) = self.span(vf);
        Ok(self.taking(code).into_iter()
            .filter(|&(s, _)| self.refused[s] && self.like(self.store.value(s), row, vf))
            .map(|(s, _)| self.store.value(s)[a..b].to_vec())
            .collect())
    }

    /// `_babble`: a sound the records hold, each distinct sound one chance, the sounds like
    /// one in `avoid` passed over while another is held; drawn from `default_rng((seed,
    /// draws))`.
    fn babble(&mut self, avoid: &[Vec<f64>]) -> Result<Option<Vec<f64>>, Error> {
        let n = self.store.count;
        if n == 0 {
            return Ok(None);
        }
        let (v0, v1) = self.span(self.voice_field()?);
        let heard = self.heard.ok_or_else(|| Error::Shape("the records hear nothing".into()))?;
        let (h0, h1) = self.span(heard);
        if v1 - v0 != h1 - h0 {
            return Err(refused("the voice and the heard sense have other widths"));
        }
        let store = &self.store;
        let mut sounds: Vec<Vec<f64>> = Vec::new();
        for s in 0..n {
            let value = store.value(s);
            let (said, heard) = (any(&value[v0..v1]), any(&value[h0..h1]));
            let alive = store.config.forget == 0.0 || !store.gone[s];
            if (said || heard) && alive {
                sounds.push(if said { value[v0..v1].to_vec() } else { value[h0..h1].to_vec() });
            }
        }
        if sounds.is_empty() {
            return Ok(None);
        }
        let units_of = |rows: &[Vec<f64>]| -> Vec<Vec<f64>> {
            rows.iter().map(|r| {
                let norm = np::maximum(row_norm(r), 1e-12);
                r.iter().map(|x| x / norm).collect()
            }).collect()
        };
        let units = units_of(&sounds);
        let mut distinct: Vec<usize> = Vec::new();
        for k in 0..units.len() {
            let dots: Vec<f64> = distinct.iter().map(|&d| np::dot(&units[d], &units[k])).collect();
            if distinct.is_empty() || np::max(&dots) < AGREE {
                distinct.push(k);
            }
        }
        let mut choices = distinct;
        if !avoid.is_empty() {
            let shunned = units_of(avoid);
            let fresh: Vec<usize> = choices.iter().copied().filter(|&c| {
                let dots: Vec<f64> = shunned.iter().map(|a| np::dot(&units[c], a)).collect();
                np::max(&dots) < AGREE
            }).collect();
            if !fresh.is_empty() {
                choices = fresh;
            }
        }
        if self.seed < 0 || self.draws < 0 {
            return Err(refused("expected non-negative integer"));
        }
        let mut rng = Pcg64::from_entropy(&[self.seed as u64, self.draws as u64]);
        self.draws += 1;
        let pick = rng.integers(0, choices.len() as i64) as usize;
        Ok(Some(sounds[choices[pick]].clone()))
    }

    /// `say(rows, codes)`: what the brain says at each row, in the voice's format, and where
    /// each came from (`said_from`).
    pub fn say(&mut self, rows: &[f64], codes: &[f64], batch: usize) -> Result<Vec<f64>, Error> {
        let vf = self.voice_field()?;
        let heard = self.heard.ok_or_else(|| Error::Shape("the records hear nothing".into()))?;
        let (v0, v1) = self.span(vf);
        let (h0, h1) = self.span(heard);
        let width = v1 - v0;
        let kw = self.store.config.width;
        let mut out = vec![0.0; batch * width];
        let mut from = Vec::with_capacity(batch);
        for i in 0..batch {
            let row = &rows[i * self.inputs..(i + 1) * self.inputs];
            let code = &codes[i * kw..(i + 1) * kw];
            let holds = self.holds(row);
            let taking = if holds { self.taking(code) } else { Vec::new() };
            // `_agreed(taking, row)`: the kernel's read where the read reaches a record
            let back = if taking.is_empty() { Vec::new() } else { self.read(code)? };
            let into = i * width..(i + 1) * width;
            if let Some(answer) = self.answer(&taking, row)? {
                out[into].copy_from_slice(&answer);
                from.push(FROM_SAID);
            } else if let (Some((_, part)), false) =
                (back.iter().find(|(f, _)| *f == heard), any(&row[h0..h1]))
            {
                out[into].copy_from_slice(part);
                from.push(FROM_HEARD);
            } else {
                let avoid = self.refused_sounds(code, row)?;
                match self.babble(&avoid)? {
                    Some(drawn) => {
                        out[into].copy_from_slice(&drawn);
                        from.push(FROM_BABBLE);
                    }
                    None => from.push(FROM_NONE),
                }
            }
        }
        self.said_from = from;
        Ok(out)
    }

    /// `hear(rows, codes, surprise=, acted=)`: each row that holds something written under its
    /// code, with the moment's surprise in tellings; then the records fade by one moment.
    pub fn hear(&mut self, rows: &[f64], codes: &[f64], batch: usize, surprise: Option<&[f64]>,
                acted: bool) -> Result<(), Error> {
        let kw = self.store.config.width;
        let mut written = vec![-1i64; batch];
        for i in 0..batch {
            let row = &rows[i * self.inputs..(i + 1) * self.inputs];
            let code = &codes[i * kw..(i + 1) * kw];
            if !self.holds(row) || !any(code) {
                continue;
            }
            let mut value = vec![0.0; self.inputs];
            for f in &self.fields {
                value[f.first..f.last].copy_from_slice(&row[f.first..f.last]);
            }
            let told = match surprise {
                None => None,
                Some(s) => {
                    let x = *s.get(i).ok_or_else(|| Error::Shape("surprise".into()))?;
                    Some(if 0.0 > x { 0.0 } else { x }) // Python's max(x, 0.0)
                }
            };
            let (slot, revised) = self.store.write(code, &value, None, told)?;
            written[i] = slot as i64;
            if !revised {
                self.refused[slot] = false;
                self.paid[slot] = 0.0;
            }
        }
        if acted {
            self.written = written;
        }
        self.store.moment();
        Ok(())
    }

    /// `pay(reward)`: each stream's record of the moment it acted at deepens by `PAY`
    /// tellings a unit of reward; a reward of zero or less refuses a record of something said.
    pub fn pay(&mut self, reward: &[f64]) {
        let said = self.voice.map(|v| self.span(v));
        for (i, &slot) in self.written.clone().iter().enumerate() {
            if slot < 0 || i >= reward.len() || !reward[i].is_finite() {
                continue;
            }
            let s = slot as usize;
            if reward[i] > 0.0 {
                let tellings = PAY * reward[i];
                let store = &mut self.store;
                store.deep[s] += tellings;
                if store.config.keep != 0.0 && store.deep[s] >= store.config.keep {
                    store.kept[s] = true;
                }
                store.faded[s] = 0.0;
                self.paid[s] += tellings;
                self.refused[s] = false;
            } else if let Some((a, b)) = said {
                if any(&self.store.value(s)[a..b]) {
                    self.refused[s] = true;
                }
            }
        }
    }

    /// `named(row, code_of)`, with `code` the code of the row's heard sense alone: the fields a
    /// heard sound alone brings back, the heard and the voice left out.
    pub fn named(&self, row: &[f64], code: Option<&[f64]>)
                 -> Result<Vec<(usize, Vec<f64>)>, Error> {
        let heard = match self.heard {
            None => return Ok(Vec::new()),
            Some(h) => h,
        };
        let (a, b) = self.span(heard);
        if !any(&row[a..b]) {
            return Ok(Vec::new());
        }
        let code = code.ok_or_else(|| Error::Missing("code".into()))?;
        Ok(self.read(code)?.into_iter()
            .filter(|(f, _)| *f != heard && Some(*f) != self.voice)
            .collect())
    }

    /// `salience(template, things)`: each thing's salience in the focus's competition.
    pub fn salience(template: &[(String, Vec<f64>)], things: &[Vec<(String, Vec<f64>)>])
                    -> Vec<f64> {
        let mut out = vec![0.0; things.len()];
        for (k, values) in things.iter().enumerate() {
            let mut near = Vec::new();
            for (field, part) in template {
                let name = field.rsplit('/').next().unwrap_or(field);
                if let Some((_, held)) = values.iter().find(|(n, _)| n == name) {
                    if held.len() == part.len() {
                        near.push(alike(held, part));
                    }
                }
            }
            if !near.is_empty() {
                let share = near.iter().filter(|&&x| x >= AGREE).count() as f64
                    / near.len() as f64;
                if share > 0.5 {
                    out[k] = NAMED * np::mean(&near);
                }
            }
        }
        out
    }

    fn put_fields(&self, out: &mut Named, back: Vec<(usize, Vec<f64>)>) {
        for (f, part) in back {
            out.put(&self.fields[f].name, Array::from_f64(&[part.len()], &part));
        }
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        let rows_of = |name: &str, width: usize| -> Result<(Vec<f64>, usize), Error> {
            let shape = args.shape(name)?;
            if shape.len() != 2 || shape[1] != width {
                return Err(Error::Shape(name.to_string()));
            }
            Ok((args.f64s(name)?, shape[0]))
        };
        let kw = self.store.config.width;
        match method {
            "hear" => {
                let (rows, batch) = rows_of("rows", self.inputs)?;
                let (codes, n) = rows_of("codes", kw)?;
                if n < batch {
                    return Err(Error::Shape("codes".into()));
                }
                let surprise = if args.has("surprise") { Some(args.f64s("surprise")?) }
                               else { None };
                let acted = args.bools("acted")?.first().copied().unwrap_or(true);
                self.hear(&rows, &codes, batch, surprise.as_deref(), acted)?;
            }
            "pay" => self.pay(&args.f64s("reward")?),
            "complete" => {
                let (rows, batch) = rows_of("rows", self.inputs)?;
                let (codes, n) = rows_of("codes", kw)?;
                if n < batch {
                    return Err(Error::Shape("codes".into()));
                }
                let done = self.complete(&rows, &codes, batch)?;
                out.put("rows", Array::from_f64(&[batch, self.inputs], &done));
            }
            "recall" => {
                let back = self.read(&args.f64s("code")?)?;
                self.put_fields(out, back);
            }
            "say" => {
                let (rows, batch) = rows_of("rows", self.inputs)?;
                let (codes, n) = rows_of("codes", kw)?;
                if n < batch {
                    return Err(Error::Shape("codes".into()));
                }
                let said = self.say(&rows, &codes, batch)?;
                let (v0, v1) = self.span(self.voice_field()?);
                out.put("said", Array::from_f64(&[batch, v1 - v0], &said));
            }
            "named" => {
                let row = args.f64s("row")?;
                if row.len() != self.inputs {
                    return Err(Error::Shape("row".into()));
                }
                let code = if args.has("code") { Some(args.f64s("code")?) } else { None };
                let back = self.named(&row, code.as_deref())?;
                self.put_fields(out, back);
            }
            "state" => {}  // the state as the core holds it: a round trip
            "salience" => {
                let mut template = Vec::new();
                let count = args.i64("things")?.max(0) as usize;
                let mut things: Vec<Vec<(String, Vec<f64>)>> = vec![Vec::new(); count];
                for name in &args.order {
                    if let Some(field) = name.strip_prefix("template:") {
                        template.push((field.to_string(), args.f64s(name)?));
                    } else if let Some(rest) = name.strip_prefix("thing:") {
                        let (k, value) = rest.split_once(':')
                            .ok_or_else(|| Error::Shape(name.clone()))?;
                        let k: usize = k.parse().map_err(|_| Error::Shape(name.clone()))?;
                        things.get_mut(k).ok_or_else(|| Error::Shape(name.clone()))?
                            .push((value.to_string(), args.f64s(name)?));
                    }
                }
                let s = Bound::salience(&template, &things);
                out.put("salience", Array::from_f64(&[s.len()], &s));
            }
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}
