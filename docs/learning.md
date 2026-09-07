# Learning: the free/nudged rule

A patch net learns the way it settles: locally. There is no backward pass. There are two
settlements and one comparison that every overlap makes on its own.

## The rule

1. **Free phase.** Clamp the input owners and settle. The state the net rests in is its
   answer; nothing about the goal has entered.
2. **Nudged phase.** From that state, settle again while the output owners feel an extra
   drive toward the target, `beta * (target - s)`, or for classification the
   cross-entropy form `beta * (target - softmax(s / T))`. The nudge travels back over the
   feedback overlaps and moves the hidden owners a little.
3. **Update.** Each overlap moves its own scale on the product of its two endpoints'
   activations, nudged minus free; each owner moves its bias on its own activation, nudged
   minus free:

       scale[e] += eta / beta * ( s+[pre] s+[post] - s0[pre] s0[post] )
       bias[i]  += eta_b / beta * ( s+[i] - s0[i] )

That is contrastive Hebbian learning, and for a symmetric wiring and a small nudge it is
the gradient of the nudge's loss with respect to the scales: the settlement descends an
energy, the nudge tilts the energy by `beta` times the loss, and the difference between
the two rest states is the loss gradient to first order. `Learner.contrast` is that
difference; the test suite checks it against finite differences of the loss.

With `centered=True` (the default) the learner runs the nudge both ways, `+beta` and
`-beta`, from the same free state and contrasts those two. The first-order error of the
one-sided estimate cancels and the update is much cleaner for the price of one more short
settlement.

## What the rule needs from the wiring and the rule

- **Feedback.** The nudge reaches a hidden owner only over an overlap from an output owner
  back to it. `layered` builds forward and feedback overlaps as one *seam*: the two
  directions share one scale (`symmetric=True`), which is what makes the settlement an
  energy descent.
- **Responsive owners.** An owner far below rest with a hard rectifier publishes exactly
  zero and has zero slope, so nothing can move it; an owner on a steep sigmoid is either
  silent or saturated. `learning_rule` therefore uses unit slope, threshold zero, and a
  small `leak` (an owner below rest publishes a small negative activation, at most
  `-leak`). Rest is still an exact zero. With the fan-scaled initial scales of `layered`
  a fresh layer sits in the responsive range.
- **Converged phases.** The contrast reads rest states. A phase that stops mid-transient
  contrasts the transient, not the nudge, and the update points nowhere. Set a
  `tolerance` and a generous step cap; the learner reports the steps each phase took.

## Using it

```python
import numpy as np
import cadence as cd

wiring = cd.layered(64, 64, 10, density=0.4, seed=0)        # sets: input, hidden, output
engine = cd.Settlement(wiring, cd.learning_rule())          # unit slope, leak 0.1, dt 0.5
learner = cd.Learner(engine, wiring.sets["output"], cd.LearnerConfig(eta=2.0, beta=0.1))

drive = engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 64))))   # pixels in [0, 1]
for epoch in range(10):
    for idx in batches:
        learner.step(drive[idx], y[idx])
learner.accuracy(drive_test, y_test)
learner.parameters()        # one number per seam plus one bias per owner
```

`learner.engine` is a plain `Settlement` at every moment: settle it, run `conformance` on
it, put its `to_dict()` in a receipt. The learner never touches the wiring; it moves
`edge_scale` and `bias`, the two parameter arrays every settlement carries.

## Reading the numbers

| knob | what it does | where to start |
|---|---|---|
| `beta` | nudge strength; smaller is closer to the gradient, larger is a stronger signal | 0.1 |
| `eta` | scale step, divided by `beta` inside the rule | 1 to 3, decayed per epoch |
| `temperature` | softmax temperature of the cross-entropy nudge | 0.1 |
| `tolerance` | when a phase is at rest | 1e-3 while learning, 1e-4 to read out |
| `leak` (rule) | sub-rest response; 0 is the connectome rule, 1 is a signed activation | 0.1 |
| `dt` (rule) | step of the owner update; larger converges in fewer steps | 0.5 to 1.0 |

The digits example in `cadence-examples` records a full table of these on a validation
split, then the one held-out test score.
