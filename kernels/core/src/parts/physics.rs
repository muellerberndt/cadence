//! `cadence.space.physics.Physics`: how things move, learned by watching them (`see`, `watch`,
//! the expectation step, `fit`, `moving`), as the Python computes it, in its order.

use crate::np;
use crate::state::{Array, Error, Named};

// a thing's motion record (`physics.py`): its fields, in order
pub const EST: usize = 0;
pub const VEL: usize = 2;
pub const PREV: usize = 4;
pub const COUNT: usize = 6;
pub const LOUD: usize = 7;
pub const SURPRISE: usize = 8;
pub const FIT: usize = 9;
pub const IDENT: usize = 10;
pub const HIDDEN: usize = 11;
pub const ERROR: usize = 12;
pub const SLOTS: usize = 4;
pub const SLOT: usize = 15;
pub const WIDTH: usize = 9;
pub const CORE: usize = SLOT + WIDTH * SLOTS;
pub const TAIL: usize = CORE + 8;
pub const FIELDS: usize = TAIL + 4;

/// The line's gains on a surprise after `n` reports (`gains`).
pub fn gains(n: i64) -> (f64, f64) {
    let nn = (n * (n + 1)) as f64;
    (2.0 * (2 * n - 1) as f64 / nn, 6.0 / nn)
}

/// A line's own error one moment on and corrected by a report (`corrected`).
pub fn corrected(error: [f64; 3], a: f64, b: f64, noise: f64) -> [f64; 3] {
    let [px, pxv, pv] = error;
    let (px, pxv) = (px + 2.0 * pxv + pv, pxv + pv);
    let s2 = noise * noise;
    let one = 1.0 - a;
    [np::pow(one, 2.0) * px + a * a * s2, one * (pxv - b * px) + a * b * s2,
     pv - 2.0 * b * pxv + b * b * px + b * b * s2]
}

/// `lg(x, m, v)`: the log density of a normal, as `_expect` writes it.
#[inline]
fn lg(x: f64, m: f64, v: f64) -> f64 {
    let d = x - m;
    -0.5 * (d * d) / v - 0.5 * np::log(2.0 * std::f64::consts::PI * v)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Physics {
    pub noise: f64,
    pub line: i64,
    pub surprise: f64,
    pub settled: f64,
    pub every: i64,
    pub horizon: i64,
    pub reports: i64,
    pub prior: f64,
    pub learning: bool,
    pub stats: [f64; 10],
    pub estimate: [f64; 4],
    // the numbers acted on (`fit`)
    pub rest: f64,
    pub leave: f64,
    pub ahead: f64,
    pub aside: f64,
    pub speed: Option<f64>,
}

impl Physics {
    pub fn import(s: &Named) -> Result<Physics, Error> {
        let stats = s.f64s("physics.stats")?;
        let estimate = s.f64s("physics.estimate")?;
        if stats.len() != 10 || estimate.len() != 4 {
            return Err(Error::Shape("physics.stats".into()));
        }
        let mut out = Physics {
            noise: s.f64("physics.noise")?,
            line: s.i64("physics.line")?,
            surprise: s.f64("physics.surprise")?,
            settled: s.f64("physics.settled")?,
            every: s.i64("physics.every")?,
            horizon: s.i64("physics.horizon")?,
            reports: s.i64("physics.reports")?,
            prior: s.f64("physics.prior")?,
            learning: s.bools("physics.learning")?[0],
            stats: stats.try_into().unwrap(),
            estimate: estimate.try_into().unwrap(),
            rest: 0.0,
            leave: 0.0,
            ahead: 0.0,
            aside: 0.0,
            speed: None,
        };
        if out.line < 3 || out.every < 1 || out.horizon < 1 || out.reports < 3 {
            return Err(Error::Shape("physics takes a line of 3 reports or more".into()));
        }
        out.fit();
        Ok(out)
    }

    pub fn export(&self, out: &mut Named) {
        for (name, x) in [("noise", self.noise), ("surprise", self.surprise),
                          ("settled", self.settled), ("prior", self.prior)] {
            out.put(&format!("physics.{name}"), Array::scalar_f64(x));
        }
        for (name, x) in [("line", self.line), ("every", self.every), ("horizon", self.horizon),
                          ("reports", self.reports)] {
            out.put(&format!("physics.{name}"), Array::scalar_i64(x));
        }
        out.put("physics.learning", Array::from_bool(&[], &[self.learning]));
        out.put("physics.stats", Array::from_f64(&[10], &self.stats));
        out.put("physics.estimate", Array::from_f64(&[4], &self.estimate));
    }

    /// Whether any lesson was taken.
    pub fn taught(&self) -> bool {
        self.stats[2] > 0.0
    }

    /// The numbers acted on, from the lessons so far (`fit`).
    pub fn fit(&mut self) {
        let [rest_ev, leave_ev, exposure, num_a, den_a, num_w, den_w, _, speed2, starts] =
            self.stats;
        if exposure > 0.0 {
            self.estimate[0] = np::clip(rest_ev / exposure, 1e-5, 0.5);
            self.estimate[1] = np::clip(leave_ev / exposure, 1e-5, 0.5);
        }
        if den_a > 0.0 {
            self.estimate[2] = np::clip(num_a / den_a, 0.0, 1.0);
        }
        if den_w > 0.0 {
            self.estimate[3] = np::clip(np::sqrt(num_w / den_w), 0.02, 2.0);
        }
        let counted = exposure + self.prior;
        self.rest = if counted > 0.0 { (rest_ev + self.prior) / counted } else { 1.0 };
        self.leave = if counted > 0.0 { leave_ev / counted } else { 0.0 };
        self.ahead = self.estimate[2];
        self.aside = self.estimate[3];
        self.speed = if starts > 0.0 { Some(np::sqrt(speed2 / starts)) } else { None };
    }

    /// The share of a thing that moves (`moving`).
    pub fn moving(&self, velocity: [f64; 2], n: i64, pv: f64) -> f64 {
        let speed = match self.speed {
            Some(s) if n >= 3 => s,
            _ => return 0.0,
        };
        let pv = if 1e-12 > pv { 1e-12 } else { pv };
        let v0 = np::pow(speed, 2.0) / 2.0;
        let vv = np::dot(&velocity, &velocity);
        let lbf = 0.5 * vv / pv * (v0 / (v0 + pv)) - np::log1p(v0 / pv);
        1.0 / (1.0 + np::exp(-np::clip(lbf, -50.0, 50.0)))
    }

    /// The camera's report of a thing at `place` (`see`): its record carried on, or a new one.
    pub fn see(&mut self, record: Option<&[f64]>, place: [f64; 2], ident: f64,
               inside: ([f64; 2], [f64; 2])) -> Vec<f64> {
        let r = place;
        let mut rec = match record {
            Some(x) => x.to_vec(),
            None => vec![0.0; FIELDS],
        };
        if record.is_none() || rec[IDENT] <= 0.0 {
            rec[IDENT] = ident;
        }
        if record.is_none() || rec[HIDDEN] != 0.0 || rec[COUNT] < 1.0 {
            let keep = rec[IDENT];
            rec.iter_mut().for_each(|x| *x = 0.0);
            rec[IDENT] = keep;
            rec[COUNT] = 1.0;
            rec[EST..EST + 2].copy_from_slice(&r);
            rec[PREV..PREV + 2].copy_from_slice(&r);
            rec[FIT] = 1.0;
            rec[ERROR..ERROR + 3].copy_from_slice(&[np::pow(self.noise, 2.0), 0.0, 0.0]);
            return rec;
        }
        let mut est = [rec[EST] + rec[VEL], rec[EST + 1] + rec[VEL + 1]];
        let slot = |k: usize, f: usize| SLOT + k * WIDTH + f;
        let active: Vec<usize> = (0..SLOTS).filter(|&k| rec[slot(k, 4)] > 0.0).collect();
        if !active.is_empty() {
            for &k in &active {
                rec[slot(k, 5)] += 1.0;
            }
            let pick = |f: usize, w: usize| -> Vec<f64> {
                active.iter().flat_map(|&k| (0..w).map(move |j| (k, j)))
                    .map(|(k, j)| rec[slot(k, f + j)]).collect()
            };
            let (starts, velocities, counts) = (pick(0, 2), pick(2, 2), pick(4, 1));
            let (horizons, error) = (pick(5, 1), pick(6, 3));
            self.watch(&starts, &velocities, &counts, r, &horizons, inside, &error);
            for k in 0..SLOTS {
                if rec[slot(k, 5)] >= self.horizon as f64 {
                    rec[slot(k, 0)..slot(k, 0) + WIDTH].iter_mut().for_each(|x| *x = 0.0);
                }
            }
        }
        let mut count = rec[COUNT] as i64;
        let s2 = np::pow(self.noise, 2.0);
        let (px, pxv, pv) = (rec[ERROR], rec[ERROR + 1], rec[ERROR + 2]);
        let (d0, d1) = (r[0] - est[0], r[1] - est[1]);
        let z2 = (d0 * d0 + d1 * d1) / (2.0 * (s2 + px + 2.0 * pxv + pv));
        rec[LOUD] = if z2 > np::pow(self.surprise, 2.0) { rec[LOUD] + 1.0 } else { 0.0 };
        rec[SURPRISE] += (z2 - rec[SURPRISE]) / 4.0;
        if z2 <= 1.0 {
            rec[FIT] = count as f64;
        }
        if rec[LOUD] >= 2.0 {
            // the line ends: it begins anew at the report before and this one
            for k in 0..SLOTS {
                let launched = count as f64 - rec[slot(k, 5)];
                if rec[slot(k, 4)] > 0.0 && launched > rec[FIT] {
                    rec[slot(k, 0)..slot(k, 0) + WIDTH].iter_mut().for_each(|x| *x = 0.0);
                }
            }
            est = [rec[PREV], rec[PREV + 1]];
            rec[VEL] = 0.0;
            rec[VEL + 1] = 0.0;
            count = 1;
            rec[LOUD] = 0.0;
            rec[SURPRISE] = 0.0;
            rec[FIT] = 1.0;
        }
        count += 1;
        let (a, b) = gains(count.min(self.line));
        let e = [r[0] - est[0], r[1] - est[1]];
        rec[EST] = est[0] + a * e[0];
        rec[EST + 1] = est[1] + a * e[1];
        rec[VEL] += b * e[0];
        rec[VEL + 1] += b * e[1];
        rec[COUNT] = count as f64;
        rec[PREV..PREV + 2].copy_from_slice(&r);
        let error = if count == 2 {
            [s2, s2, 2.0 * s2]
        } else {
            corrected([rec[ERROR], rec[ERROR + 1], rec[ERROR + 2]], a, b, self.noise)
        };
        rec[ERROR..ERROR + 3].copy_from_slice(&error);
        if count >= 4 && count % self.every == 0 && rec[SURPRISE] <= self.settled {
            let free = (0..SLOTS).find(|&k| rec[slot(k, 4)] <= 0.0);
            let at = match free {
                Some(k) => k,
                None => np::argmax(&(0..SLOTS).map(|k| rec[slot(k, 5)]).collect::<Vec<_>>()),
            };
            let row = [rec[EST], rec[EST + 1], rec[VEL], rec[VEL + 1],
                       count.min(self.line) as f64, 0.0, rec[ERROR], rec[ERROR + 1],
                       rec[ERROR + 2]];
            rec[slot(at, 0)..slot(at, 0) + WIDTH].copy_from_slice(&row);
        }
        rec
    }

    /// Lessons (`watch`): lines through `counts` reports that put things at `starts` with
    /// `velocities` (rows of two), seen at `place` `horizons` moments later, `error` each line's
    /// own error then (rows of three). Gives the lessons taken.
    #[allow(clippy::too_many_arguments)]
    pub fn watch(&mut self, starts: &[f64], velocities: &[f64], counts: &[f64], place: [f64; 2],
                 horizons: &[f64], inside: ([f64; 2], [f64; 2]), error: &[f64]) -> usize {
        let rows = counts.len();
        if !self.learning || rows == 0 {
            return 0;
        }
        let (lo, hi) = inside;
        let r = place;
        let s2 = np::pow(self.noise, 2.0);
        let mut ok = vec![false; rows];
        let mut speed = vec![0.0; rows];
        for i in 0..rows {
            let (x0, v0) = (&starts[2 * i..2 * i + 2], &velocities[2 * i..2 * i + 2]);
            let (n, k, pv) = (counts[i], horizons[i], error[3 * i + 2]);
            speed[i] = np::sqrt(v0[0] * v0[0] + v0[1] * v0[1]);
            let end = [x0[0] + k * v0[0], x0[1] + k * v0[1]];
            let mut good = n >= self.reports as f64 && speed[i] >= 3.0 * np::sqrt(pv) && k >= 1.0
                && k <= self.horizon as f64;
            good &= (0..2).all(|j| end[j] >= lo[j] && end[j] <= hi[j])
                && (0..2).all(|j| x0[j] >= lo[j] && x0[j] <= hi[j]);
            good &= (0..2).all(|j| r[j] >= lo[j] + 0.03 && r[j] <= hi[j] - 0.03);
            ok[i] = good;
        }
        let taken: Vec<usize> = (0..rows).filter(|&i| ok[i]).collect();
        if taken.is_empty() {
            return 0;
        }
        let m = taken.len();
        let (mut k, mut short, mut side, mut sp, mut line) =
            (vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        for (q, &i) in taken.iter().enumerate() {
            let (x0, v0) = (&starts[2 * i..2 * i + 2], &velocities[2 * i..2 * i + 2]);
            let own = &error[3 * i..3 * i + 3];
            let ki = horizons[i];
            let u = [v0[0] / speed[i], v0[1] / speed[i]];
            let e = [r[0] - x0[0] - ki * v0[0], r[1] - x0[1] - ki * v0[1]];
            k[q] = ki;
            short[q] = -(e[0] * u[0] + e[1] * u[1]);
            side[q] = e[0] * -u[1] + e[1] * u[0];
            sp[q] = speed[i];
            line[q] = s2 + own[0] + 2.0 * ki * own[1] + ki * ki * own[2];
        }
        self.expect(&k, &short, &side, &sp, &line);
        self.fit();
        m
    }

    /// The expectation step over a batch of lessons, added to the statistics (`_expect`).
    fn expect(&mut self, k: &[f64], short: &[f64], side: &[f64], speed: &[f64], line: &[f64]) {
        let [h, g, a, w] = self.estimate;
        let stay = { let x = 1.0 - h - g; if 1e-9 > x { 1e-9 } else { x } };
        let hz = self.horizon as usize;
        let rows = k.len();
        let size = rows * hz;
        let j_of = |j: usize| (j + 1) as f64;
        let log_stay = np::log(stay);
        let (log_h, log_g) = (np::log(h), np::log(g));
        let mut left = vec![0.0; size];
        let mut lr = vec![0.0; size];
        let mut ll = vec![0.0; size];
        let mut lk = vec![0.0; rows];
        for i in 0..rows {
            lk[i] = log_stay * k[i] + lg(short[i], 0.0, line[i]) + lg(side[i], 0.0, line[i]);
            let (sp, var0) = (speed[i], line[i]);
            for j in 0..hz {
                let at = i * hz + j;
                let jj = j_of(j);
                left[at] = np::maximum(k[i] - jj + 1.0, 0.0);
                let valid = jj <= k[i];
                let before = log_stay * (jj - 1.0);
                lr[at] = if valid {
                    log_h + before + lg(short[i], left[at] * sp, var0) + lg(side[i], 0.0, var0)
                } else {
                    f64::NEG_INFINITY
                };
                let spread = w * sp * left[at];
                let vt = var0 + spread * spread;
                ll[at] = if valid {
                    log_g + before + lg(short[i], (1.0 - a) * left[at] * sp, vt)
                        + lg(side[i], 0.0, vt)
                } else {
                    f64::NEG_INFINITY
                };
            }
        }
        let row_max = |x: &[f64], i: usize| -> f64 {
            let row = &x[i * hz..(i + 1) * hz];
            let mut best = row[0];
            for &v in &row[1..] {
                best = np::maximum(best, v);
            }
            best
        };
        let mut pk = vec![0.0; rows];
        let mut pr = vec![0.0; size];
        let mut pl = vec![0.0; size];
        for i in 0..rows {
            let top = np::maximum(lk[i], np::maximum(row_max(&lr, i), row_max(&ll, i)));
            pk[i] = np::exp(lk[i] - top);
            for j in 0..hz {
                pr[i * hz + j] = np::exp(lr[i * hz + j] - top);
                pl[i * hz + j] = np::exp(ll[i * hz + j] - top);
            }
            let z = pk[i] + np::sum(&pr[i * hz..(i + 1) * hz]) + np::sum(&pl[i * hz..(i + 1) * hz]);
            pk[i] /= z;
            for j in 0..hz {
                pr[i * hz + j] /= z;
                pl[i * hz + j] /= z;
            }
        }
        let each = |f: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
            (0..size).map(|at| f(at / hz, at % hz)).collect()
        };
        let pkk: Vec<f64> = (0..rows).map(|i| pk[i] * k[i]).collect();
        let exposure = np::sum(&pkk)
            + np::sum(&each(&|i, j| (pr[i * hz + j] + pl[i * hz + j]) * j_of(j)));
        let m = each(&|i, j| left[i * hz + j] * speed[i]);
        let res = each(&|i, j| {
            let d = short[i] - (1.0 - a) * m[i * hz + j];
            side[i] * side[i] + d * d - 2.0 * line[i]
        });
        let squares: Vec<f64> = speed.iter().map(|s| s * s).collect();
        let n = rows as f64;
        let mm = np::sum(&each(&|i, j| pl[i * hz + j] * m[i * hz + j] * m[i * hz + j]));
        let add = [
            np::sum(&pr), np::sum(&pl), exposure,
            np::sum(&each(&|i, j| {
                let at = i * hz + j;
                pl[at] * m[at] * (m[at] - short[i])
            })),
            mm,
            np::sum(&each(&|i, j| pl[i * hz + j] * np::maximum(res[i * hz + j], 0.0))) / 2.0,
            mm, n, np::sum(&squares), n,
        ];
        for (s, x) in self.stats.iter_mut().zip(add) {
            *s += x;
        }
    }
}
