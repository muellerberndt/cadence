# Sequential tasks and prerequisite chains

A continuing brain that learns a single step well, food in one move, often fails a chain of
steps: gather wood, gather stone, make fire, warm up. The usual sign is a policy that
collapses onto the one action that pays most often. This page says what the repository's
measurements establish about that failure and what to do about it today. One point is a
default since 0.82.0: the carried interpretation starts with a quiet influence on the next
moment and learning raises it where the past pays ([context](context.md)).

## What breaks a chain

**The chain's state is not in the observation.** If the brain cannot see what it carries,
the right action depends on history, and the task is a short-term memory task. The declared
recall chamber did not establish a reliable working-trace horizon on its fresh founders
([recall/5](../benchmarks/recall/README.md#recall5-on-fresh-founders-313-314-and-315-closure-failed-horizons-none-0-and-0),
[issue 84](https://github.com/muellerberndt/cadence/issues/84)). Those results do not establish
recall of a four-step inventory, or a universal upper bound on the trace under every task
and operating point. Without sufficient history, a policy cannot reliably distinguish
chain stages that have identical observations; it may settle on one action or keep sampling.
Expose inventory state when the creature can sense it: has wood, has stone, has fire.
Keep a hidden-inventory condition as a separate memory test. When history has to carry the
state, the brain carries it in its [context](context.md): give the last module enough width.

**Pay arrives several choices later.** Credit for an earlier action across intervening
choices is the delayed key-door contract of
[issue 111](https://github.com/muellerberndt/cadence/issues/111). The
[key-door nursery](../benchmarks/keydoor/README.md) acquires at delays of two and five
choices and not at ten; at five the critic's rate was the bottleneck, and the chamber runs
its critic at 5.0 in place of the composed 0.3. Pay only at the end of a four-step chain sits
at the edge of what the eligibility (`actor_lam` 0.6, `actor_gamma` 0.95) reaches.

**The readout saturates.** Near saturation, a finite nudge can produce very little
activity contrast: the arena's nursery grids record motor probabilities at 0.98 that the aroused
sampling could not escape, and the key-door chamber records the same latch under punished
exploration. Read the motor activations and the margin between the chosen and the next
action before changing anything. A per-action set point on the readout (0.81.0's intrinsic
plasticity) flattened a six-action policy under reward and was removed in 0.82.0.

**A loud context drowns the present.** Until 0.82.0 the composed brain read its carried
interpretation back through a projection of scale 12 from birth. In a chain the next action
depends on what the brain sees now; a loud carry makes it depend on the moment before instead,
and an untrained brain with it completes a fully observed chain less often than uniform random.
Since 0.82.0 the projection starts at scale 1.2: the present decides at birth, and learning
strengthens the carried context where the past predicts the outcome.

**Learning continues at the acquisition step.** A policy acquired at the nursery's actor
step erodes under its own continued learning, in a melee and in the nursery itself
([issue 169](https://github.com/muellerberndt/cadence/issues/169), the
[competing-skill ring](../benchmarks/competing/README.md#the-arena-sixty-fights-2026-10-10)).
More drilling past the point of acquisition does not consolidate, it moves the policy. A
step ten to thirty times smaller after acquisition held the skill while learning went on.
A brain that continues through `live` learns only from outcomes that surprise it. On a fully
observed chain of four actions taught by correcting executed mistakes, five fresh founders
left alone through `live` kept 42.5% of their 44.8% frozen success; left alone through `step`,
which learns from every outcome, they kept 33.2%, and two of them fell to nothing. Where most
surprises are punishments, as in the arena's melee, `live` alone does not hold the skill.

## The pattern that the measurements support

1. **Put the chain's state in the observation.** Inventory flags are senses, not a
   workaround, until a measured memory horizon exists. Keep the default context; raise
   `working_memory_scale` only for a measured reason.
2. **Grade executed actions; do not label.** Let the brain act through `Brain.live` and pay
   the outcome. The world supplies problems, never answers. Label teaching of a sequence is a
   different contract ([issue 121](https://github.com/muellerberndt/cadence/issues/121)) and
   showed its own readout instability in the recall chamber.
3. **Compare reward schedules on the complete chain.** In the six-seed four-step
   development chamber, observed-state reward at each correct step reached 204
   greedy completions per thousand moments; end-only pay reached 42 and randomly
   interleaved starting stages reached 33. Granted prerequisites can let the brain
   meet its need before learning the complete chain. Backward lessons are therefore
   an experimental control, not an established remedy. Disclose supplied shaping
   rewards and verify the full chain from stage zero under end-only pay.
4. **Track acquisition and retention at a declared probe schedule.** Longer training
   can degrade an acquired skill. Select a training budget in development, then
   freeze it before testing fresh founders; disclose any checkpoint selection.
5. **After acquisition, continue through `live` or lower the actor step.** Once the
   caregiver leaves, a life through `live` learns only from surprising outcomes and keeps
   what it was taught. Where surprises are mostly punishments, lower the step instead: the
   ring stage of the arena (`actor_eta` 0.001 to 0.003 against the nursery's 0.03) is the
   measured remedy there.
6. **Read the sampled policy beside the greedy one.** A frozen greedy copy can answer with
   one action in every situation while the sampled policy's probabilities move with the
   situation (the arena's fingerprint). Read the probability of the right action per
   situation, and the completion rate of the sampled policy, not only the argmax.
7. **Test the critic rate against its control.** The key-door nursery measured
   `actor_eta_critic=5.0` against the composed 0.3. This is a task-specific operating
   point, not evidence that raising the rate helps every delayed-credit task.

## What is not available yet

A measured memory horizon for a hidden chain state (issue 84; the research question of
issue 189 is how context persists and is revised, and the carried interpretation of
[context](context.md) is the current answer under test); a mechanism
that keeps an acquired skill at the nursery's own actor step (issue 169's next candidate is
a step that follows the brain's own uncertainty); and a passed gate for delayed credit
beyond five choices (issue 111). The
[prerequisite-chain nursery](../benchmarks/chain/README.md) is the instrument for this
page: the chain's state observed or hidden, pay at the end or in backward lessons, the
founder brain against uniform random, a frozen newborn, the stationary memoryless ceiling
and a tabular learner with the same information.
