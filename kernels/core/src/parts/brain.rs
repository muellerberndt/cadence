//! A composed brain (`cadence.generic.Brain`): its parts, each under its attribute's name, and
//! the moment that runs them in the Python's order (`live`). A part the brain does not hold is
//! absent from its state; a part the core does not hold yet refuses the brain by name.

use super::arousal::Arousal;
use crate::state::{Error, Named};

/// The parts a brain holds, by attribute name, and whether the core holds each.
pub const SLOTS: &[&str] = &["arousal"];

#[derive(Clone, Debug, PartialEq)]
pub struct Brain {
    pub arousal: Option<Arousal>,
    /// The entries of parts the core does not hold yet, kept as they came.
    pub rest: Named,
}

impl Brain {
    pub fn import(s: &Named) -> Result<Brain, Error> {
        let arousal = if s.holds("arousal") { Some(Arousal::import(&s.under("arousal"))?) } else { None };
        let mut rest = Named::new();
        for name in &s.order {
            let slot = name.split('.').next().unwrap_or("");
            if !SLOTS.contains(&slot) {
                rest.put(name, s.arrays[name].clone());
            }
        }
        Ok(Brain { arousal, rest })
    }

    pub fn export(&self, out: &mut Named) {
        if let Some(a) = &self.arousal {
            let mut part = Named::new();
            a.export(&mut part);
            out.put_under("arousal", &part);
        }
        for name in &self.rest.order {
            out.put(name, self.rest.arrays[name].clone());
        }
    }

    /// `Brain.live(observations, reward=, done=, salience=, decision_id=)`: one moment of a
    /// continuing life, the action per motor slot. Refused by name while the brain holds a
    /// part the core does not.
    pub fn live(&mut self, observations: &[f64], reward: Option<f64>, done: Option<bool>,
                salience: Option<f64>, decision_id: Option<i64>) -> Result<Vec<i64>, Error> {
        let _ = (observations, reward, done, salience, decision_id);
        if !self.rest.order.is_empty() {
            let first = self.rest.order[0].split('.').next().unwrap_or("").to_string();
            return Err(Error::Missing(format!("the core holds no part {first}")));
        }
        Err(Error::Missing("the spine's moment".into()))
    }

    pub fn call(&mut self, method: &str, args: &Named, out: &mut Named) -> Result<(), Error> {
        match method {
            "live" => {
                let one = |name: &str| -> Result<Option<f64>, Error> {
                    if args.has(name) { Ok(Some(args.f64s(name)?[0])) } else { Ok(None) }
                };
                let done = if args.has("done") { Some(args.bools("done")?[0]) } else { None };
                let decision = if args.has("decision_id") { Some(args.i64("decision_id")?) } else { None };
                let action = self.live(&args.f64s("observations")?, one("reward")?, done,
                                       one("salience")?, decision)?;
                out.put("action", crate::state::Array::from_i64(&[action.len()], &action));
                Ok(())
            }
            _ => Err(Error::Missing(method.to_string())),
        }
    }
}
