# Credit across a prerequisite chain: development plan and the protocol to declare (2026-10-11)

Status: branch `chain-credit-20261011`, row 1 of the sequence order (issue
[#111](https://github.com/muellerberndt/cadence/issues/111)). This document is the plan for the
branch and the shape of the confirmation protocol it will declare. No reading in it is new; no
gate is passed; no default changes.

## Where 0.81.0 leaves the instrument

The [prerequisite-chain nursery](README.md) on six development seeds, 20,000 moments, chain of
four, the arena founder brain (`protocol.json`): with the state observed and pay at every
correct step the founder completes the chain at 204 per thousand moments against the tabular
learner's 250, uniform random's 42 and the stationary ceiling's 62. With pay only at the end
the founder's greedy policy reads uniform random's rate (42) while its sampled policy reads
77, with stage accuracy 0.67. The lessons arm (episodes that begin with prerequisites granted,
interleaved) reads 33, below pay at the end. With the state hidden no arm beats the stationary
ceiling. The readout's intrinsic plasticity at rate 0.1 toward 0.3 lowers every reading under
reward.

The chamber's brain point already runs the critic at 5.0, the key-door nursery's remedy for
delay five. The remaining levers are the ones below.

## Question and declared limit

Can one continuing `Brain.compose` life, the chamber's founder, complete a chain of four
actions whose pay comes at the end, with the chain's state observed, at the tabular learner's
completion rate at matched moments, using levers that exist in the released library? The
hidden-state mode is the memory requirement of
[#84](https://github.com/muellerberndt/cadence/issues/84) and is out of scope here; its
stationary ceiling stays the reading to beat there.

## Fixed

The world (`make_world`: 16 inputs, two distractor actions, chain of four), the brain point,
the controls (uniform random, the frozen newborn, the stationary memoryless ceiling, Q-learning
over the observed stage) and the readings every 2,000 moments from saved copies (greedy and
sampled completions per thousand, per-stage greedy action, probability of the right action and
margin, share of moments on the most chosen action, aroused share, learned moments, dopamine
sign) are those of `protocol.json`. Development seeds 0 to 5. Confirmation seeds are named at
the freeze, after development, and are never run before it.

## Levers, each an arm against the founder

1. **Eligibility span.** `actor_lam` 0.6 (founder) against 0.8 and 0.95, `actor_gamma` 0.95
   against 0.99, through `--brain FIELD=VALUE`. Pay four choices after the first action sits at
   the edge of what 0.6 reaches; the key-door development lives used 0.95 at delay five.
2. **Backward lessons.** Diagnose the lessons arm before changing it: per starting stage, the
   share of moments, the probability of the right action and the sign of the dopamine at the
   lesson's completion. Then the schedule: last stage first and the previous stage added once
   the later one is acquired, interleaved with complete episodes, graded by the executed
   action's outcome. A label on the right action is not a lever here.
3. **The readout gene at a width-relative target.** `learning_homeostasis_target` set to one
   over the number of actions (one sixth here) against the 0.3 of the `homeo` arm, at rates
   0.02 and 0.1. The development run showed a six-action readout's mean activation near 0.17,
   so a pull toward 0.3 flattens it; this arm asks whether the gene's harm under reward is the
   target or the mechanism.
4. **The smaller step after acquisition.** The `taper` arm (a tenth of the actor step from the
   first acquisition), read for over-drilling: acquisition at the first probe that passes, then
   the same reading at three and five times that budget. A policy that passes at 300 trips and
   fails at 1,500 is the users' report; this is the reading
   `actor-uncertainty-step-20261011` consumes.
5. **The sampled policy beside the greedy one.** Already in the readings; the gate is stated on
   both, because a frozen greedy copy can answer with one action in every situation while the
   sampled probabilities move.

## The gate to declare

On confirmation seeds, with the state observed and pay at the end: the founder brain at the
declared lever settings completes the chain, greedy and sampled, within a declared margin of
the tabular learner's rate at matched moments, with stage accuracy on every stage above the
stationary ceiling, and holds that reading at three times the acquisition budget. The margin
and the budget are fixed from the development readings before the confirmation seeds are
drawn. Beside it, the key-door ladder: the acquired delay extends from five toward ten with the
same lever settings, read on the key-door chamber's complete-life gates.

## Preservation

No library change is planned. If a lever setting is proposed as a composed default, the odour
nursery, the key-door nursery, the competing ring and the Patch World founder are the
preservation witnesses, each at the founder against the proposal.

## Deliverables

1. A development receipt with the five arms on seeds 0 to 5 (`results/`, bound to the sources).
2. `protocol.json` with confirmation seeds and gates filled; one confirmation run; its receipt.
3. The README's reading and the [sequential-tasks guide](../../docs/sequential-tasks.md)
   updated with what held and what did not.
4. A CHANGELOG entry under Unreleased. No gate, default or mechanism changes.
