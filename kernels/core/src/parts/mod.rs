//! The parts of a brain, one module each: a part's state from and to named arrays and its
//! methods by name (`call`), for the replay checks and for the moment that runs them.

pub mod arousal;
pub mod brain;
pub mod surprise;

use crate::state::{Array, Error, Named};

/// One call of a part's method on its state: the state after the call (named `state:...`) and
/// what the call returned (`out:...`).
pub fn call(part: &str, method: &str, state: &Named, args: &Named) -> Result<Named, Error> {
    let mut out = Named::new();
    let mut after = Named::new();
    match part {
        "arousal" => {
            let mut held = arousal::Arousal::import(state)?;
            held.call(method, args, &mut out)?;
            held.export(&mut after);
        }
        "surprise" => {
            let mut held = surprise::Surprise::import(state)?;
            held.call(method, args, &mut out)?;
            held.export(&mut after);
        }
        "orienting" => {
            let mut held = surprise::Orienting::import(state)?;
            held.call(method, args, &mut out)?;
            held.export(&mut after);
        }
        "brain" => {
            let mut held = brain::Brain::import(state)?;
            held.call(method, args, &mut out)?;
            held.export(&mut after);
        }
        _ => return Err(Error::Missing(part.to_string())),
    }
    let mut both = Named::new();
    for name in &after.order {
        both.put(&format!("state:{name}"), after.arrays[name].clone());
    }
    for name in &out.order {
        both.put(&format!("out:{name}"), out.arrays[name].clone());
    }
    let _ = Array::scalar_i64;
    Ok(both)
}
