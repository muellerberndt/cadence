//! `cadence.memories`: the order a brain reads and writes its memories in (`Memories`), and an
//! outcome written into the place records (`record_action`, with `_chosen`). The drive of a
//! moment passes through each present member's `stimulus` in `ORDER`, the spine's fold over
//! the parts that hold them.

use super::synaptic::Synaptic;
use crate::state::{refused, Error, Named};

/// `Memories.ORDER`: the members a brain reads and writes, in that order.
pub const ORDER: [&str; 8] = [
    "working_memory", "efference", "history", "recall", "message", "longterm", "hippocampus",
    "episodic",
];

/// `Memories(members)`: the members present, in `ORDER`; an unknown name is refused.
pub fn members<'a>(present: &[&'a str]) -> Result<Vec<&'a str>, Error> {
    if let Some(unknown) = present.iter().find(|n| !ORDER.contains(n)) {
        return Err(refused(&format!("unknown memory: {unknown:?}")));
    }
    Ok(ORDER.iter().filter_map(|n| present.iter().find(|p| *p == n).copied()).collect())
}

/// `_chosen`: each chosen motor neuron's row and column, one per row or one per slot (the
/// slots' neurons from `offsets`); a negative column counts from the end, as NumPy indexes.
fn chosen(action: &[i64], shape: &[usize], offsets: &[i64], motor: usize)
          -> Result<Vec<(usize, usize)>, Error> {
    let column = |c: i64| -> Result<usize, Error> {
        let c = if c < 0 { c + motor as i64 } else { c };
        (0 <= c && (c as usize) < motor).then_some(c as usize)
            .ok_or_else(|| Error::Shape("an action outside the motor neurons".into()))
    };
    match shape.len() {
        1 => action.iter().enumerate().map(|(r, &a)| Ok((r, column(a)?))).collect(),
        2 => {
            let slots = shape[1];
            if offsets.len() != slots && offsets.len() != 1 {
                return Err(refused("offsets"));
            }
            let offset = |s: usize| offsets[if offsets.len() == 1 { 0 } else { s }];
            let mut out = Vec::with_capacity(action.len());
            for r in 0..shape[0] {
                for s in 0..slots {
                    out.push((r, column(action[r * slots + s] + offset(s))?));
                }
            }
            Ok(out)
        }
        _ => Err(Error::Shape("action".into())),
    }
}

/// `record_action(memory, keys, action, reward, importance, known, offsets)`: the outcome of
/// each chosen action written into the place records for the situation it was chosen in; with
/// `known`, only for the streams where it is set.
#[allow(clippy::too_many_arguments)]
pub fn record_action(memory: &mut Synaptic, keys: &[f64], keys_shape: &[usize], action: &[i64],
                     action_shape: &[usize], reward: &[f64], importance: &[f64],
                     known: Option<&[bool]>, offsets: &[i64]) -> Result<(), Error> {
    if let Some(k) = known {
        if !k.iter().any(|&x| x) {
            return Ok(());
        }
    }
    let batch = *action_shape.first().ok_or_else(|| Error::Shape("action".into()))?;
    if reward.len() != batch && reward.len() != 1 {
        return Err(refused("a reward per row"));
    }
    let motor = memory.post.len();
    let mut target = vec![0.0; batch * motor];
    let mut observed = vec![false; batch * motor];
    for (r, c) in chosen(action, action_shape, offsets, motor)? {
        target[r * motor + c] = reward[if reward.len() == 1 { 0 } else { r }];
        observed[r * motor + c] = true;
    }
    memory.observe(keys, keys_shape, &target, &[batch, motor], known, Some(importance),
                   Some((&observed, &[batch, motor])))
}

/// One call on the place records the memories write (`record_action`), by name.
pub fn call(memory: &mut Synaptic, method: &str, args: &Named) -> Result<(), Error> {
    match method {
        "record_action" => {
            let known = if args.has("known") { Some(args.bools("known")?) } else { None };
            record_action(memory, &args.f64s("keys")?, &args.shape("keys")?,
                          &args.i64s("action")?, &args.shape("action")?, &args.f64s("reward")?,
                          &args.f64s("importance")?, known.as_deref(), &args.i64s("offsets")?)
        }
        "state" => Ok(()),  // the state as the core holds it: a round trip
        _ => Err(Error::Missing(method.to_string())),
    }
}
