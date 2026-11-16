//! `cadence.arousal.Arousal`: when a continuing brain leaves routine. The law and its readings
//! as the Python computes them, in its order.

use crate::np;
use crate::state::{Array, Error, Named};

const TINY: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub threshold: f64,
    pub decay: f64,
    pub tolerance: f64,
    pub floor: f64,
    pub fast: f64,
    pub slow: f64,
    pub heat: f64,
    pub youth: i64,
    pub value_surprise: f64,
    pub need: f64,
    pub need_scale: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arousal {
    pub config: Config,
    pub level: f64,
    pub age: i64,
    pub outcomes: i64,
    pub rewards: i64,
    pub spreads: i64,
    pub square: f64,
    pub recent: f64,
    pub longrun: f64,
    pub usual: f64,
    pub learning_sweeps: i64,
    pub moments: [i64; 2],
    pub sweeps: [i64; 2],
}

/// Python's `float ** int` for a base in (0, 1] and a count: the C library's `pow`.
#[inline]
fn powi(base: f64, count: i64) -> f64 {
    libm::pow(base, count as f64)
}

fn log_ratio(error: f64, unit: f64) -> f64 {
    let ratio = error / unit;
    if ratio.is_finite() {
        np::log(ratio)
    } else {
        np::log(error) - np::log(unit)
    }
}

impl Arousal {
    pub fn import(s: &Named) -> Result<Arousal, Error> {
        let i2 = |name: &str| -> Result<[i64; 2], Error> {
            let v = s.i64s(name)?;
            Ok([v[0], v[1]])
        };
        Ok(Arousal {
            config: Config {
                threshold: s.f64("config.threshold")?,
                decay: s.f64("config.decay")?,
                tolerance: s.f64("config.tolerance")?,
                floor: s.f64("config.floor")?,
                fast: s.f64("config.fast")?,
                slow: s.f64("config.slow")?,
                heat: s.f64("config.heat")?,
                youth: s.i64("config.youth")?,
                value_surprise: s.f64("config.value_surprise")?,
                need: s.f64("config.need")?,
                need_scale: s.f64("config.need_scale")?,
            },
            level: s.f64("level")?,
            age: s.i64("age")?,
            outcomes: s.i64("outcomes")?,
            rewards: s.i64("rewards")?,
            spreads: s.i64("spreads")?,
            square: s.f64("square")?,
            recent: s.f64("recent")?,
            longrun: s.f64("longrun")?,
            usual: s.f64("usual")?,
            learning_sweeps: s.i64("learning_sweeps")?,
            moments: i2("moments")?,
            sweeps: i2("sweeps")?,
        })
    }

    pub fn export(&self, out: &mut Named) {
        let c = &self.config;
        for (name, value) in [
            ("config.threshold", c.threshold), ("config.decay", c.decay),
            ("config.tolerance", c.tolerance), ("config.floor", c.floor),
            ("config.fast", c.fast), ("config.slow", c.slow), ("config.heat", c.heat),
            ("config.value_surprise", c.value_surprise), ("config.need", c.need),
            ("config.need_scale", c.need_scale),
        ] {
            out.put(name, Array::scalar_f64(value));
        }
        out.put("config.youth", Array::scalar_i64(c.youth));
        out.put("level", Array::scalar_f64(self.level));
        out.put("age", Array::scalar_i64(self.age));
        out.put("outcomes", Array::scalar_i64(self.outcomes));
        out.put("rewards", Array::scalar_i64(self.rewards));
        out.put("spreads", Array::scalar_i64(self.spreads));
        out.put("square", Array::scalar_f64(self.square));
        out.put("recent", Array::scalar_f64(self.recent));
        out.put("longrun", Array::scalar_f64(self.longrun));
        out.put("usual", Array::scalar_f64(self.usual));
        out.put("learning_sweeps", Array::scalar_i64(self.learning_sweeps));
        out.put("moments", Array::from_i64(&[2], &self.moments));
        out.put("sweeps", Array::from_i64(&[2], &self.sweeps));
    }

    fn reading(&self, accumulated: f64, rate: f64, count: i64) -> f64 {
        if count == 0 {
            return 0.0;
        }
        let mut correction = 1.0 - powi(1.0 - rate, count);
        if correction == 0.0 {
            correction = -libm::expm1(count as f64 * libm::log1p(-rate));
        }
        accumulated / correction
    }

    pub fn scale(&self) -> f64 {
        np::sqrt(py_max(self.reading(self.square, self.config.slow, self.spreads), 0.0))
    }
    pub fn recent(&self) -> f64 {
        self.reading(self.recent, self.config.fast, self.rewards)
    }
    pub fn longrun(&self) -> f64 {
        self.reading(self.longrun, self.config.slow, self.rewards)
    }
    pub fn usual(&self) -> f64 {
        self.reading(self.usual, self.config.slow, self.outcomes)
    }

    fn unmet(recent: f64, need: f64) -> f64 {
        if recent <= 0.0 {
            1.0
        } else if recent >= need {
            0.0
        } else {
            (need - recent) / need
        }
    }

    fn want_of(&self, longrun: f64, recent: f64, scale: f64) -> f64 {
        let mut want = if scale <= 0.0 { 0.0 } else { np::clip((longrun - recent) / scale, 0.0, 1.0) };
        let need = self.config.need;
        if need > 0.0 {
            want = py_max(want, Self::unmet(recent, need));
        }
        let share = self.config.need_scale;
        if share > 0.0 && scale > 0.0 && self.age >= self.config.youth {
            want = py_max(want, Self::unmet(recent, share * scale));
        }
        want
    }

    pub fn want(&self) -> f64 {
        self.want_of(self.longrun(), self.recent(), self.scale())
    }
    pub fn aroused(&self) -> bool {
        self.age < self.config.youth || self.level >= self.config.threshold
    }
    pub fn heat(&self) -> f64 {
        1.0 + self.config.heat * self.want()
    }

    /// `outcome(error, reward, own=, learned=)`: the surprise and the want that entered the
    /// level; None where a reading leaves the finite numbers (the Python refuses it).
    pub fn outcome(&mut self, error: f64, reward: f64, own: bool, learned: bool)
                   -> Option<(f64, f64)> {
        let (error, reward) = (error.abs(), reward);
        if !error.is_finite() || !reward.is_finite() {
            return None;
        }
        let c = self.config.clone();
        let (mut outcomes, mut spreads) = (self.outcomes, self.spreads);
        let rewards = self.rewards + 1;
        let (mut recent, mut longrun, mut usual) = (self.recent, self.longrun, self.usual);
        let (mut square, scale) = (self.square, self.scale());
        recent += c.fast * (reward - recent);
        longrun += c.slow * (reward - longrun);
        let mut surprise = 0.0;
        if own {
            let first = outcomes == 0;
            let previous_usual = self.usual();
            outcomes += 1;
            usual += c.slow * (error - usual);
            let unit = c.tolerance * previous_usual + c.floor * scale + TINY;
            if error > unit && !first {
                surprise = c.value_surprise * log_ratio(error, unit);
            }
        }
        let expected = self.reading(longrun, c.slow, rewards);
        let want = self.want_of(expected, self.reading(recent, c.fast, rewards), scale);
        let level = c.decay * self.level + (1.0 - c.decay) * (surprise + want);
        if learned || self.age < c.youth || level >= c.threshold {
            spreads += 1;
            let d = reward - expected;
            square += c.slow * (d * d - square);
        }
        if ![recent, longrun, usual, square, level].iter().all(|x| x.is_finite()) {
            return None;
        }
        self.outcomes = outcomes;
        self.spreads = spreads;
        self.rewards = rewards;
        self.recent = recent;
        self.longrun = longrun;
        self.usual = usual;
        self.square = square;
        self.level = level;
        Some((surprise, want))
    }

    /// `lived(sweeps, learning_sweeps)`: one moment counted in the mode it was answered in.
    pub fn lived(&mut self, sweeps: i64, learning_sweeps: i64) {
        let mode = self.aroused() as usize;
        self.moments[mode] += 1;
        self.sweeps[mode] += sweeps;
        self.learning_sweeps += learning_sweeps;
        self.age += 1;
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "outcome" => {
                let (s, w) = self
                    .outcome(args.f64("error")?, args.f64("reward")?,
                             args.bools("own")?[0], args.bools("learned")?[0])
                    .ok_or(Error::Shape("arousal moments must remain finite".into()))?;
                out.put("surprise", Array::scalar_f64(s));
                out.put("want", Array::scalar_f64(w));
            }
            "lived" => self.lived(args.i64("sweeps")?, args.i64("learning_sweeps")?),
            _ => return Err(Error::Missing(method.to_string())),
        }
        Ok(())
    }
}

/// Python's `max(a, b)`: `b` where it is larger, else `a`.
#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}
