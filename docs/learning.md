# Learning: the free/nudged rule

This page is the complete account of how a patch net learns in Cadence. It has every
equation the code runs, a worked example with real numbers from a six-owner net, the
reason the rule is a gradient, what breaks it, and every knob. The runnable versions are
the tutorials in [cadence-examples](https://github.com/muellerberndt/cadence-examples).

## 1. What is being learned

A settlement has three parameter arrays, all on the `Settlement`:

- `edge_scale`, one signed number per overlap. The effective drive of overlap `e` per unit
  of presynaptic activation is `gain · count[e] · edge_scale[e] · exp(log_gain[pre[e]])`;
  in a learned net `gain`, `count`, and `log_gain` are 1, 1, and 0, so the drive is just
  `edge_scale[e]`. Call it `W[e]`.
- `bias`, one number per owner.
- `log_gain`, one per owner, which the learner leaves alone.

Learning moves `edge_scale` and `bias`. It never touches the wiring: who talks to whom is
fixed, and only how strongly changes.

## 2. What a settlement computes

Each owner `i` holds a potential `v[i]` and publishes an activation `s[i]`. One step of the
rule, for every owner at once:

    inbox[i] = sum over overlaps e with post[e] = i of  W[e] · s[pre[e]]
    total[i] = inbox[i] + clamp[i] + bias[i]            (+ nudge[i] in the nudged phase)
    v[i]    <- v[i] + dt · (total[i] - v[i])
    s[i]     = act(v[i])

`act` is the rule's activation. `learning_rule()` sets slope 1, threshold 0, and leak 0.1:

    act(v) = tanh(v / 2)          for v >= 0
    act(v) = 0.1 · tanh(v / 2)    for v <  0

so rest (`v = 0`) publishes exactly 0, positive drive saturates toward 1 around `v ≈ 4`,
and negative drive publishes a small negative number instead of a hard zero. A
settlement runs this step from rest (or from a given state) until no owner's activation
moved more than `tolerance` in a step, or until the step cap. That state is what a
readout sees.

An input is a *clamp*: a fixed drive on the input owners, one number per owner (a pixel's
level, a board cell's occupancy). Input owners have no inbound overlaps, so at rest each
sits at `v = clamp` and publishes `act(clamp)`. Nothing else is clamped.

## 3. The two phases and the update

For a batch of inputs with targets:

**Free phase.** Settle under the input clamp alone. Call the rest state `s⁰`. The output
owners' activations are the net's answer; no target has entered.

**Nudged phases.** From `s⁰`, settle again with an extra drive on the output owners only:

    nudge[i] = beta · weight · (target[i] - p[i])      for output owners i
    p        = softmax(s[outputs] / T)

`p` is a softmax over the output owners, so pushing the target's owner up pushes the
others down by their share (a cross-entropy nudge). `weight` is 1 for a label; for a
reward it is the advantage of the action (see [games](games.md)). With `centered=True`
(the default) two such settlements run from the same `s⁰`: one with `+beta`, rest state
`s⁺`, and one with `-beta`, rest state `s⁻`. The nudge travels back over the feedback
overlaps, so hidden owners rest at slightly different activations in the two phases.

**Update.** Every overlap reads its own two endpoints in the two phases, and every owner
reads itself:

    contrast[e] = mean over the batch of ( s⁺[pre[e]] · s⁺[post[e]] - s⁻[pre[e]] · s⁻[post[e]] ) / (2 beta)
    owner[i]    = mean over the batch of ( s⁺[i] - s⁻[i] ) / (2 beta)

    edge_scale[e] += eta   · contrast[e]
    bias[i]       += eta_b · owner[i]

Two more details, both in `Learner.update`: an overlap and its reverse (`i → j` and
`j → i`) form one *seam* and share one scale, so their two contrasts are averaged and both
move by the same amount; and a scale's magnitude is clipped to `[scale_floor, scale_cap]`.
With `centered=False` the second nudged phase is skipped and `s⁻` is replaced by `s⁰` with
`beta` in place of `2 beta`.

That is the whole rule. Nothing else reads the target, nothing stores activations for a
later pass, and no owner reads any number that is not its own or at one of its overlaps.

## 4. A worked example, with the numbers

Six owners: inputs 0 and 1, hidden 2 and 3, outputs 4 and 5. `cd.layered(2, 2, 2,
density=1.0, seed=3)` gives dense forward overlaps, feedback overlaps tied to them, and
two lateral overlaps between the outputs starting at 0. The rule is `learning_rule(dt=1.0)`;
the learner uses `eta=1.0`, `beta=0.1`, `T=0.2`, tolerance `1e-9` so the phases are exact.
The input is `x = (1.0, 0.2)` with label 1.

Initial seams (the reverse of each hidden↔output overlap shares its scale):

    0→2 +0.115   1→2 −0.587   0→3 +0.530   1→3 −0.196
    2→4 −0.479   3→4 +0.527   2→5 +0.633   3→5 +0.719     (feedback 4→2, 4→3, 5→2, 5→3 equal)
    4→5  0.000   5→4  0.000

Clamp: owner 0 gets 1.0, owner 1 gets 0.2, everyone else 0.

Free phase, 27 steps to rest:

    v = [1.000, 0.200, 0.011, 0.282, 0.071, 0.104]
    s = [0.462, 0.100, 0.005, 0.140, 0.036, 0.052]

Output owners publish 0.036 and 0.052; `softmax(s/T)` is (0.480, 0.520); the answer is
class 1, which happens to be right, but barely. Target: owner 5 at 1, owner 4 at 0.

Nudge drive at the first `+beta` step: `0.1 · ((0, 1) − (0.480, 0.520)) = (−0.048, +0.048)`
on owners 4 and 5. Nudged rest states:

    +beta (26 steps): s = [0.462, 0.100,  0.019, 0.143, 0.012, 0.078]
    −beta (21 steps): s = [0.462, 0.100, −0.001, 0.136, 0.064, 0.021]

Owner 5 went up and owner 4 went down under `+beta`, the reverse under `−beta`; the
hidden owners moved too, through the feedback seams, which is the only way the target
reaches them. Contrasts, `(s⁺s⁺ − s⁻s⁻) / 0.2`:

    0→2 +0.047   1→2 +0.010   0→3 +0.016   1→3 +0.003
    2→4 +0.002   3→4 −0.035   2→5 +0.008   3→5 +0.042
    4→5 −0.002   5→4 −0.002

Read the two largest. Seam 3↔5: hidden owner 3 was active (0.14) in both phases and output
5 rose under the nudge, so their product rose and the seam strengthens by +0.042. Seam
3↔4: the same hidden owner and output 4, which fell, so the seam weakens by −0.035. The
input seams into owner 3 also strengthen (0→3 by +0.016) because owner 3 itself ended a
little higher under `+beta` than `−beta`, which is the credit that flowed back. With
`eta = 1` those contrasts are the scale changes. Biases move by `eta_b · owner`:
owner 5 +0.0057, owner 4 −0.0052. After this single update a fresh free settlement gives
outputs (0.028, 0.067) instead of (0.036, 0.052): the right answer, with more room.

You can rerun this: it is `examples/worked_update.py`.

## 5. Why the contrast is a gradient

When every seam is symmetric (`W[i→j] = W[j→i]`, which `layered` and tying guarantee),
the settlement is a descent of an energy

    E(v) = Σ_i ∫₀^{v[i]} u · act'(u) du  −  ½ Σ_{i≠j} W[i→j] s[i] s[j]  −  Σ_i (clamp[i] + bias[i]) s[i]

and the rest state is a minimum of `E`. The nudge adds `beta · L(s)` to the energy, where
`L` is the loss whose gradient the nudge drive is (for the cross-entropy nudge, the
cross-entropy of `softmax(s/T)` against the target, up to the constant `T`). Scellier and
Bengio (2017, *equilibrium propagation*) showed that then

    dL/dW[i→j] at the free rest state = lim_{beta→0} ( s⁰[i] s⁰[j] − s^beta[i] s^beta[j] ) / beta

so the contrast is minus the gradient of the loss with respect to the seam, and the update
is gradient descent on `L`, computed by nothing but two settlements. The centered version
(`+beta` against `−beta`) cancels the first-order error in `beta`. The test
`test_contrast_tracks_the_loss_gradient` checks this numerically: on a random layered net
the contrast correlates above 0.9 with finite differences of the loss.

Three things break the argument, and `learning_rule` exists to avoid them:

1. **Phases that stop mid-transient.** The contrast reads rest states; read too early and
   it reads the transient instead. Always settle to a tolerance.
2. **Owners with no slope.** A hard rectifier below rest has slope zero, so nothing can
   move such an owner and no credit passes through it. The `leak` gives it a slope.
3. **Owners on a cliff.** A steep sigmoid puts every owner either silent or saturated,
   where `act'` is nearly zero and the true gradient is nearly zero; a finite nudge then
   makes moves that match no gradient. Unit slope keeps a fan-scaled layer responsive.

## 5b. Where the rule stops: wirings the nudge cannot travel back through

The contrast teaches a seam only if the nudge changes the rest state of at least one of
its two endpoints. In a layered net with tied feedback seams the nudge on the outputs
moves the hidden owners, and every seam is reachable. In a measured, *directed* wiring
the credit travels only over seams that point back toward the owners the nudge moved; a
connectome of chemical synapses mostly does not, so only the last hop before the readout
learns. The C. elegans rung of cadence-examples (at its tag v0.5.0) shows the consequence: the rule cannot
fit four textbook facts that need two sensory pathways to act differently, whatever the
gain, while a global gradient through the same settlement can. Symmetrising the wiring
as a modelling assumption (every synapse also carries its reverse, tied) was tried and
did not rescue it either, because the regime where a connectome's owners are both
responsive and sparse is narrow to nonexistent under raw synapse counts. Measured
wirings are for the protocol layer; learnable nets are built with feedback.

## 6. Using it

```python
import numpy as np
import cadence as cd

wiring = cd.layered(64, 32, 10, density=1.0, seed=0)         # sets: input, hidden, output
engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
learner = cd.Learner(engine, wiring.sets["output"],
                     cd.LearnerConfig(eta=3.0, beta=0.1, temperature=0.1, tolerance=3e-3))

drive = engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 64))))   # pixels in [0, 1]
for epoch in range(20):
    learner.config = dataclasses.replace(learner.config, eta=3.0 * 0.8**epoch)
    for idx in batches:
        learner.step(drive[idx], y[idx])                    # free, +beta, -beta, update
learner.accuracy(drive_test, y_test)
```

`learner.engine` is a plain `Settlement` at every moment: settle it, run `conformance` on
it, export `engine.dense()` for a page, put `to_dict()` in a receipt.

## 7. Every knob

| knob | where | what it does | where to start |
|---|---|---|---|
| `beta` | `LearnerConfig` | nudge strength; smaller is closer to the gradient, larger a stronger signal | 0.1 |
| `eta` | `LearnerConfig` | seam step; the contrast is already divided by `2 beta` | 2 to 3, decayed by 0.9 to 0.95 per epoch over 40 epochs; a decay of 0.8 over 15 epochs under-trains tabular tasks by two to five points |
| `eta_bias` | `LearnerConfig` | bias step | `eta / 100` |
| `temperature` | `LearnerConfig` | softmax temperature of the cross-entropy nudge; also the policy temperature when sampling actions | 0.1 (labels), 0.2 (actions) |
| `centered` | `LearnerConfig` | contrast `+beta` against `−beta` (two nudged phases) rather than against the free state | `True` |
| `tolerance`, `free_steps`, `nudged_steps` | `LearnerConfig` | when a phase is at rest, and the step caps | 3e-3 while learning, 1e-4 to read out; 100 / 12 |
| `scale_floor`, `scale_cap` | `LearnerConfig` | bounds on a seam's magnitude | 0, 8 |
| `nudge` | `LearnerConfig` | `"cross_entropy"` or `"quadratic"` (`beta · (target − s)`) | cross-entropy for classes |
| `momentum` | `LearnerConfig` | each seam steps on a running average of its own contrast (still local) | 0.9 on supervised tabular tasks, where it adds about a point; 0 elsewhere |
| `decay` | `LearnerConfig` | every update shrinks each trainable seam and bias by this fraction: a leak on the seams | 0 for a fixed training set; 0.003 on a stream that drifts, where it keeps the net plastic (see `tasks.md`, streams) |
| `normalize`, `normalize_floor` | `LearnerConfig` | each seam divides its step by the running RMS of its own contrast (still local) | 0 (off); it did not help anywhere it was tried |
| `symmetric` | `Learner` | tie an overlap and its reverse into one seam | `True` |
| `trainable_overlaps` | `Learner` | bool per overlap; freeze the rest | all |
| `leak`, `slope`, `dt` | `learning_rule` | sub-rest response, activation slope, step of the owner update | 0.1, 1.0, 0.5 to 1.0 |
| `density`, `feedback`, `lateral`, `init`, `skip` | `layered` | input→hidden density, feedback scale, output↔output scale, initial magnitude, direct input→output seams | 1.0, 1.0, 0, 1.0, `False` |

`Learner.parameters()` counts one number per seam plus one bias per owner, which is the
figure to put next to a feed-forward network's parameter count.

## 8. Warm starts, costs, and what to expect

A free phase may start from an earlier state (`learner.free(drive, warm=state)` or
`learner.step(..., warm=state)`); the fixed point is the same. In practice the gain is
small, because the seams move enough between visits that the old state is not much closer
than rest.

Each update is three settlements of tens of steps each, so on a laptop core the rule costs
ten to a hundred times the wall-clock of a forward-and-backward pass for the same accuracy,
and reaches that accuracy in fewer passes over the data. The examples' receipts record
both numbers on every rung; see [differences](differences.md) for why.
