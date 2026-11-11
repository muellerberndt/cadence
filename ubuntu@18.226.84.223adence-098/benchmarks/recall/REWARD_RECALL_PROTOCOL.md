# Reward-based recall and the trace write as an action (2026-10-11)

Status: row 3 of the sequence order (issue
[#84](https://github.com/muellerberndt/cadence/issues/84)), the fourth mechanism on the plan.
The instrument is `reward_recall.py` with `protocol-reward.json` and `test_reward_recall.py`;
the mechanism runs at instrument level (below) and is not part of the library. No default
changes.

## The instrument as built

`reward_recall.py` runs one continuing `Brain.live` life over the finite chamber's conditions
(`finite_horizon_inputs`: clean-0, clean-1, clean-2, distractor-1, distractor-2,
replacement-1, order-latest-1, interleaved uniformly), pays +1 or -1 at the query and delivers
the pay with the next episode's first observation and `done`, so the library's own boundary
rule resets the stream's trace. Controls: uniform random, the newborn (every arm's reading at
zero episodes), the memoryless ceiling, and a tabular learner given the asked value at the
query. Readings from frozen copies at declared probes: per condition greedy and sampled
accuracy, the erased (trace cleared) and reset lesions, the write share per event kind,
training tallies (success, income, aroused share, learned moments, dopamine sign), and the
gates below. Receipts bind the instrument, the inputs module, the protocol and every library
source.

The `gated` arm composes a second motor slot of two choices, hold or write
(`Brain.compose(..., slots=(2, 2))`). The library writes the trace after every moment; after a
hold the instrument restores the trace, its last activation and its cold flag to their values
before the moment, which equals skipping the founder's law. The choice is sampled and credited
by the same actor and the same reward as the answer, through its eligibility over the hold
span. The `fixed` arm is the founder. A library gene that skips the write inside the moment is
the promotion step, taken only on a measured gain of this arm over the founder.

Every recipe sets the body's need to 0.2 reward per moment and a youth of 300 moments, the
other arousal genes at their founders. With the founder need of 0 the brain habituates to pay
it cannot predict, stays calm on nine moments of ten and learns from few outcomes; with the
need, the unmet share keeps it aroused and learning until its recent pay meets the need.

The development recipes keep the composed trace and set the actor at the key-door nursery's
delayed-credit point (lam 0.95, gamma 0.95, critic rate 5) with the actor step at 0.1 or 0.03:
at the composed step of 1.0 the readout saturates onto one answer under this pay. An arm may
live through `step` instead of `live`, and an `exposed` control appends the asked cue's raw
payload at the query, the external-history input, to show the reward path works before any
recall is read. Two recipes read the trace loud at birth (scale 12, alone and with the recall
chamber's declared trace recipe) as declared comparisons. The development readings of these
arms are the owner's and are not part of this repository; the confirmation founders stay
unopened and no default changes.

## Demonstrated limitation

The finite continuing recall chamber's confirmations recall/4 and recall/5 on fresh founders
failed closure, horizons none, 0 and none, then none, 0 and 0, with replacement and order
failing in every founder ([README](README.md)). The measured limit: a working trace written
after every event with one fixed weight cannot both hold a cue through distractors and let a
replacement in, and a distractor differs from a write in its marker only, so the separation
needs a write that is conditional on the marker and learned. The readout's intrinsic
plasticity keeps the readout in range under teaching and delivers no horizon. Every chamber so
far teaches the answer by lesson; the brain has never been paid for remembering.

## The instrument: a reward-based recall protocol

Conditions of recall/2, pay in place of lessons. One continuing `Brain.live` life sees a cue
(one of two values), then 0, 1 or 2 neutral events, a distractor (a write-like event whose
payload shares the cue's coordinates and whose marker differs), optionally a replacement cue,
then a query at which the brain acts (one of two answers) and is paid for the right one.
Episodes are interleaved and the horizon is the largest contiguous passed prefix of
intervening events, as in `FINITE_HORIZON_PROTOCOL.md`; replacement (the later cue wins) and
order (two cues, the asked one) are the closure gates beside it. Readings from saved copies:
the probability of the right answer per condition, the greedy answer, the trace's content at
the query, the dopamine sign, aroused share and learned moments.

Controls: uniform random; the frozen newborn; the stationary policy (the memoryless ceiling,
0.5); a tabular learner given the cue explicitly at the query (the external-history control,
which must be competent for the chamber to count); and the founder with its fixed write, the
released trace law, which is the control for the mechanism below. Fresh founders are drawn at
the freeze; development founders are declared first.

## The mechanism: the write into the working trace as an action of the basal ganglia

Beside the world's actions, the motor readout gains one more choice each moment: write the
trace from the present state or hold it. The founder's law, `c <- decay * c + (1 - decay) * h`
after every event, is the control. With the gene on, a hold leaves `c` as it is and a write
applies the founder's law; the choice is made by the same actor and credited by the reward at
the query through the actor's eligibility over the hold span. Gated working memory is learned
from outcomes, in the place where recurrent networks train gates by backpropagation through
time, with no second learning law and no supervision of the gate. The write's cost (a settle is
paid either way) and the exploration of the gate under arousal are declared on the branch
before the first run; each constant is a gene with the founder as the control.

## The gate

On fresh founders: horizon 2 with the distractor, replacement and order passed, through the
trace, with the external-history control competent and the founder (fixed write) failing as
recall/5 did. Credit across the hold span is the delayed key-door contract of
[#111](https://github.com/muellerberndt/cadence/issues/111): spans up to five are within what
the critic at 5.0 reaches today; longer spans depend on the branch `chain-credit-20261011`.

## Preservation

Lives that never hold are byte-identical to the founder. The odour nursery, the key-door
nursery and the finite recall chamber under teaching are the witnesses; the chain chamber's
hidden mode, where no arm beats the stationary ceiling today, is the first consumer of a
passed horizon.

## Deliverables

1. The chamber (`reward_recall.py`, protocol, tests, receipt) with the controls, run on
   development founders at the founder's fixed write: the reading the mechanism must beat.
2. The write as an action in `src/cadence/generic.py` and `learning.py` behind a founder-off
   gene, with tests of founder identity, save and load, host and device agreement.
3. Development readings, one confirmation on fresh founders, README sections, the CHANGELOG
   entry and the issue's status. No default changes.
