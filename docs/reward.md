# Learning from reward

Reward learning lets the world do the teaching. Nobody tells the brain the right action; the world
scores the action it took, and the brain learns which actions pay in which situations. A food
pellet, points in a game, a penalty for a miss: each is a reward, a number per stream, positive
for good and negative for bad.

How it works, in one paragraph: when the brain samples an action, every synapse keeps an
**eligibility trace**, a fading note of how much it took part. When the reward arrives, the brain
compares it with what it expected (its **critic** forecasts the value of each situation). The
difference, one number broadcast to every synapse like dopamine, turns every trace into a weight
change: actions that did better than expected become likelier, actions that did worse less
likely. No error is sent backward through the network.

`Brain.step` and an aroused [`Brain.live`](start.md#routine-and-repair-live) take the reward of the
preceding action and return the next one:

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(2, 2, modules=(16,), seed=0, actor_eta=0.1, actor_eta_bias=0.01)
rng = np.random.default_rng(0)
cue = np.eye(2)[[rng.integers(2)]]
action = brain.step(cue)                           # the first moment has no outcome yet
for _ in range(300):
    reward = [1.0 if action[0] == cue[0].argmax() else -1.0]
    cue = np.eye(2)[[rng.integers(2)]]
    action = brain.step(cue, reward=reward)        # the reward belongs to the previous action
```

| Use | When | Learns |
| --- | --- | --- |
| `step(observations, reward=...)` | many streams, explicit control, every outcome should teach | from every outcome |
| `live(observation, reward=...)` with `arousal=True` | one creature living one life | while young, surprised or wanting ([routine and repair](start.md#routine-and-repair-live)) |
| `act` then `learn(reward, done, next_observations)` | your body times the two halves of a moment | from that outcome ([by hand](tune.md#one-experience-step-by-hand)) |

Use reward when outcomes pay actions: bandits, reversing rules, rewards a few choices later
([reversal](../benchmarks/reversal/README.md), [key-door](../benchmarks/keydoor/README.md),
[competing skills](../benchmarks/competing/README.md)). On one stream with signed, noisy
outcomes and a cost per move, set `actor_eta` to a tenth of the composed 1.0, as above: the
composed rate walks the policy to one held action, the tenth lets the cue decide
([night-replay chamber](../benchmarks/replay/README.md)). Where you know the right answer, a
lesson teaches faster ([train a brain](teach.md)).

## Eligibility traces and dopamine

`ActorCritic` wraps a `Learner`. `act` runs the free phase and draws the action from a
softmax over each motor slot. Two nudged phases, toward and away from the action taken,
give every synapse its contrast, an estimate of the score of that action under the
[equilibrium assumptions](settling.md#learning-the-freenudged-rule). The actor always uses a softmax nudge, even when
the learner imitates with a quadratic one. `learn` then applies

```
elig[e]      <- gamma * lam * elig[e] + contrast[e]
delta         = r + gamma * V(s') - V(s)
efficacy[e] += eta * delta * elig[e]
```

`delta` is the same for every synapse and sets the sign and size of each change; `V` is a
linear readout of the critic neurons, usually the hidden population. The actor's signal is
clipped at `dopamine_cap`. The critic learns from the same clipped signal, its step divided
by its trace's energy: the bounded signal stabilizes the update, and the critic need not
predict the return in reward units.

Sampling and eligibility use the same temperature. With temperature `T_i` for slot `i`, its
nudge mask is scaled by `c/T_i`, `c` the smallest of the base and supplied temperatures, so
the contrast estimates `c * grad(log policy)` summed over slots. The trace lives per stream:
it carries across moments, a routine (unaroused) moment of `live` fades it by `gamma * lam`
(`fade`), and `done` or `reset` clears it.

## Settings

`ActorCritic(learner, critic, config=None, seed=0)`. In a composed brain
each field is the named setting `actor_<field>`, set in `Brain.compose` or `brain.retune`;
the composed founders are `gamma=0.9, lam=0.8, eta=1.0, eta_bias=0.05, eta_critic=0.3`, and
`dopamine_cap=1.0` and `eligibility_steps=12` are constants of the composed brain
([defaults table](tune.md#defaults-and-expert-overrides)).

| `ActorCriticConfig` field | Default | Meaning |
| --- | --- | --- |
| `gamma` | `0.99` | Discount |
| `lam` | `0.9` | Eligibility decay; credit at a delay of `d` transitions is scaled by `(gamma * lam) ** d` |
| `eta` | `0.5` | Actor synapse step per unit dopamine per unit trace |
| `eta_bias` | `None` | Bias step; `None` derives `eta / 10` at construction |
| `eta_critic` | `0.05` | Critic step |
| `dopamine_cap` | `1.0` | Clip of the actor's signal; `0` for none |
| `eligibility_steps` | `None` | Steps of each reward-nudged phase; `None` uses the learner's `nudged_steps` |
| `hold` | `None` | The actor's eligibility decay per moment apart from the critic's; `None` uses `gamma * lam` ([a reward many moments after the decision](#a-reward-many-moments-after-the-decision)) |
| `entropy` | `0.0` | The policy's entropy in the step ([exploration](#exploration-over-long-reward-lives-opt-in)); `0`: none |

`beta`, the base `temperature` and `tolerance` come from the learner's `LearnerConfig`
(composed: `temperature`; `beta` is 0.1, a constant); the teaching point (`teaching`) sets the steps
of lessons alone.

- `act(drive, greedy=False, temperature=None, *, bias=None)`: one temperature or one per
  slot; `bias` adds a `(batch, motor)` vote before each softmax (the episodic memory votes
  this way) while eligibility scores the readout's own policy.
- `learn(reward, done, next_drive, bootstrap=None, *, observed=None, warm=None)`: `observed`
  marks real rows (at least one); padding rows are left out of plasticity and statistics
  and their traces reset. `warm` is the state to settle the next state from when the
  stream moved on ([`Brain.wait`](start.md#outcomes-that-arrive-later-wait)).
- `value(state)`, `value_of(drive)` (settled cold, cache untouched), `probabilities(state,
  temperature=None, *, bias=None)` (`(batch, slots, max_size)`, zero-padded, for several
  slots), `state` (the latest free phase), `fade(done=None)`, `reset()`. `salience`, a
  `(batch, neurons)` array set before `learn`, weights eligibility by each sender's
  salience. `memory` is a memory with learned parameters (as [`Recall`](memory.md#the-recall-memory-in-full) has)
  that the same dopamine trains beside the synapses; `trace_memory` is its eligibility.
- Report: `td_error` (mean absolute raw error), `delta` (mean absolute actor signal),
  `dopamine` (signed mean), `capped` (share beyond `dopamine_cap` before the clip),
  `saturation` (share of outputs within 0.02 of 0 or 1), `trace` (mean absolute plastic
  eligibility), `value`, `free_steps`, `scale_step`, `bias_step` (mean absolute applied
  steps), `address_reward_step` with a `memory`, and `policy_entropy` (the sampled act's
  entropy in nats, summed over the motor slots) with `entropy`.

## Actor rates

Every actor parameter steps by its rate times the dopamine times its own eligibility, with no
running average of the steps and no division by their size: the rates are steps per unit
dopamine per unit trace. Keep the bias rate below the synapse rate; a bias rate far above it
makes a policy whose greedy action ignores the observation. A named `actor_eta` in
`Brain.compose` or `retune` keeps the current bias rate; pass `actor_eta_bias=None` to derive
`eta / 10` again. Construction emits a `RuntimeWarning`, and changes nothing, when `eta_bias`
exceeds a positive `eta`; the warning does not cover `eta_critic`.

For long reward training, lower `actor_eta`. At the composed 1.0, a long life of rewards can
drive the motor neurons to the limits of their activation, where a nudge has no slope: the
greedy answer is then the same whatever the observation, and the contrast that would move it
back is near zero. Train long lives at a tenth of the composed rate or less, with the bias rate a
tenth of that, as the example at the top does. `report["saturation"]` near 1 and one greedy
`act` across different observations show the pinned readout
([troubleshooting](troubleshooting.md#learning-from-reward)).

## Maturity (opt-in)

Maturity ([#253](https://github.com/muellerberndt/cadence/issues/253)) is for a brain that goes
on learning after it has acquired a skill that this learning should not erase
([#169](https://github.com/muellerberndt/cadence/issues/169)).
`Brain.compose(..., maturity=True)` gives the brain one slow variable, the use of its
plasticity. Each actor update adds the synapse change it applied, in the brain's own step units
per synapse: `use += sum |dw| / (eta N)`, with `eta` the actor's synapse rate and `N` its
synapses. Every accepted moment of life (`live`, `step`, or `learn` called on its own) keeps
`1 - 1 / maturity_recovery` of the use. The step factor

```text
m = 1 / (1 + (use / maturity_scale) ** 4)
```

multiplies the actor's synapse and bias steps, and engages the store's surprise-gated writes
with strength `1 - m` ([memory](memory.md#maturity-and-surprise-gated-writes)). The critic,
lessons, rehearsal, a learned code and the recall memory's maps keep their steps, and their
changes add nothing to the use. A young brain learns at the founder's steps; a busy one
matures, its steps falling and its store keeping what it holds unless an outcome surprises it;
one that lives without learning returns toward its youth. No count of updates or of age is kept.

A moment refused before it learns changes neither the use nor the store's records and noise
scales. A `live` or `step` refused after its outcome was learned keeps that outcome, as each
documents, with its step in the use and without the moment's relaxation.

| Setting | Founder | Changed by | What it does |
| --- | --- | --- | --- |
| `maturity` | `False` | construction | the maturity on the actor's steps and the store's writes |
| `maturity_scale` | 10.0 | `retune` | the use at which the step factor is 0.5, in step units per synapse |
| `maturity_recovery` | 200,000 moments | `retune` | the use's time constant: each moment keeps `1 - 1 / maturity_recovery` of it |

`maturity` is a construction setting: `retune` refuses it, and `grow` and `graft` keep the
brain's maturity, its use and its store's noise scales. Without `maturity`, a
`maturity_scale` or `maturity_recovery` other than its founder is refused. A larger scale keeps
a brain young for longer. At the founder scale the reward chambers of the
[preservation panel](../benchmarks/credit/README.md#preservation) keep `m` at 0.98 or above
(the busiest use is about 3.4 step units per synapse), while an arena brain uses far more in its
100,000-moment nursery and ends it at `m` 0.07 to 0.7.

It costs one noise scale per consolidated record of the store (`brain.hippocampus.gate_noise`),
counted by `parameters()` with the use, a pass over the synapses on each actor update to
measure the step, and the gate's arithmetic on each write of the store. The variable
(`brain.basal_ganglia.maturity`, a `Maturity`) and the noise scales are saved and loaded with
the brain in checkpoint format `cadence-brain/2`, which 0.93.1 and earlier refuse
([checkpoint format](api.md#checkpoint-format); `cadence-brain/3` where the store also fades).
Off, the brain is the founder's to the bit and saves `cadence-brain/1`.

## Exploration over long reward lives (opt-in)

Over a long life of small rewards, the actor at its composed step can settle on one button that
the reward does not favour, and hold it ([#296](https://github.com/muellerberndt/cadence/issues/296)).
At birth one or two motor neurons are active and the others rest at the 0 rail, where a nudge
has little slope: an active button's contrast is .28 to .43, a resting one's .01 to .04. Each
outcome moves the active outputs, so the policy drifts toward one of them, and a button at rest
that the reward favours rises slowly. In the
[long reward life chamber](../benchmarks/actor_latch/README.md), whose three worlds keep a Doom
player's observation, the composed step (`actor_eta=1.0`) scores below the frozen newborn in the
world shaped like Doom basic, and a tenth of it does not; the bias step, the trace, the input
width and the memories leave the hold as it is.

`Brain.compose(..., actor_entropy=k)` (or `ActorCriticConfig(entropy=k)`) adds the policy's
entropy to the actor's step, the regularizer a conventional actor-critic carries. A sampled act
keeps its surprise, `-log p(action) - H` under the policy that sampled it (summed over the motor
slots), and its outcome's step adds that act's own contrast at the weight `k` times the surprise:

```
step[e] = eta * (delta * elig[e] + k * surprise * contrast[e])
```

Over the policy's own actions the surprise averages to zero, so the term is the sampled
gradient of the entropy: an act the policy takes often has a negative surprise and steps its
button down, a rare act steps its button up, and the dopamine decides where the reward pays more
than the pull toward an even policy. The term uses the act's own contrast, never the trace, so
it credits no earlier decision. `report["policy_entropy"]` reads the entropy of each act.

| Setting | Founder | Changed by | What it does |
| --- | --- | --- | --- |
| `actor_entropy` | `0.0` | `retune` | the weight of the entropy term; 0 is the composed actor |

What it gives, on six fresh founders each ([the declared run](../benchmarks/actor_latch/README.md#the-declared-run)):
`actor_entropy=0.1` removes the hold in every life of the chamber and of the Doom body (no
button above .40) and reads +0.918 in the aiming world against the default's -0.350. It reaches
its even policy by lowering the active outputs, and where the reward's preference is faint it
leaves every motor neuron at rest (`saturation` 1.00), where a nudge has little slope: in Doom's
defend the center and health gathering, whose good policies press one button far more than the
others, the brain reads below its frozen newborn. Use it where an outcome pays clearly and the
right action depends on the situation; a tenth of the actor's step slows the hold there too and
leaves it in a third of the long survival lives. A moment costs about a third more.

It learns on the host: `backend="cpu"` and `"mlx"` (whose update it takes on the host, as a
maturity does); a brain on a torch device refuses the outcome, and `Lives` refuses the setting by
name. A pending act's surprise is saved with the brain. A brain that names no entropy, or 0,
keeps every bit of the composed actor on `backend="cpu"`, and its configuration saves without
the field.

## Decision and reset contract

Drives are nonempty finite `(batch, neurons)` arrays, rewards and bootstrap values
`(batch,)`, `done` boolean; validation precedes every update. Each row keeps its own
potential, adaptation and eligibility. `done` rows restart from rest and
bootstrap from `bootstrap`, or zero without it. A time limit is not terminal: mark the
truncated row `done` and pass the value of its last observation as `bootstrap`. On a composed
brain that value is `brain.value(last)`:

```python
import numpy as np
import cadence as cd

timed = cd.Brain.compose(4, 2, seed=0)
timed.step(np.eye(4)[[0, 1]])                     # two streams; each sampled action awaits
last = np.eye(4)[[2, 3]]                          # the observations at the cut
cut = np.array([True, False])                     # the time limit cuts stream 0 here
bootstrap = timed.value(last)                     # (streams,), read before the outcome
timed.step(last, reward=np.array([0.5, 0.0]), done=cut, bootstrap=bootstrap)
```

`Brain.value` reads what `learn` settles for the next state of a stream that goes on: the drive of
the row with the working trace, the efference copy, the heard history, the recall memory's slots
and the place records as they stand, settled from the stream's activity (from rest before the
brain has acted), and the critic's value of that state. The memories enter because the episode
was cut, not ended: had it gone on, its next state would have read them. The read writes nothing
and leaves the streams, traces, pending action, records, parameters and random generators as they
were. Without the store a truncated row bootstrapped this way takes the dopamine of a row that goes
on, to the bit. With the store the two part by the outcome's own record: `learn` writes the
outcome to the place records before it settles the next state, and `value`, read before the
outcome, reads the records without it. On an `ActorCritic` built by hand the value of a drive is
`value_of(next_drive)`, settled cold. `learn`
settles the next state before the update, and the next `act` settles again from it under
the new parameters (the cached phase is reused only for an unchanged drive and
parameters): two free and two nudged phases per decision.

`act(..., greedy=True)` is an evaluation read: it keeps no eligibility for its action and
cannot be followed by `learn`. Another `act` replaces the pending action. Finish a reward
transition before changing learner parameters elsewhere, or call `reset`, which clears
stream state and traces and keeps the critic, the actor and the learner's optimizer history.
`Brain.save/load` carries the full state; `Learner.save` does not save a separately built
`ActorCritic`. A shaped reward keeps the task's optimum only in the discounted form
`gamma * Phi(next) - Phi(now)` with terminal handling.

## What delayed credit requires

A reward that follows the reading it belongs to is a record, written in one exposure by the
[associative store](memory.md#associative-store-the-hippocampus). The eligibility trace
serves the policy when the reward arrives after other decisions: it remembers which action
could have caused the reward, never an arbitrary observation. Too short a trace assigns no
useful credit, too long a one credits unrelated actions; choose `lam` from the actual
action-to-reward delays. The [delayed-credit test](../tests/test_child.py) pays a choice
three moments later: the greedy hit rate is at least 0.8 with `lam=0.9` and at most 0.65
with `lam=0` ([policy credit tests](../tests/test_policy_credit.py)). Chains of
prerequisite actions: [sequential tasks](#what-breaks-a-chain).

## Traps, with their measurements

Each is a property of the rule or the readout, with its reading in the `learn` report;
[troubleshooting](troubleshooting.md#learning-from-reward) maps symptoms to them.

- **Continued learning interferes with acquired behaviour.** Arousal gates updates and
  protects no earlier response. A smaller actor step reduces interference without retaining a
  skill; `consolidation` governs record writes and leaves actor weights alone
  ([prerequisite chains](#the-pattern-the-measurements-support), item 3).
- **The temperature is relative to the activation range [0, 1].** At 0.05, outputs 0.3
  apart choose with probability 0.998 and the nudge `beta * (target - p)` vanishes. Outputs
  near rest take 0.05, outputs at 0.7 to 1.0 take 0.3.
- **The saturation latch.** With one output saturated and the other silenced the contrast
  `a_pre * (b_plus - b_minus)` is zero whatever the reward: `saturation` near 1 with `trace`
  near 0. Bring the outputs into the sensitive band (gain, `calibrate_bias`) and keep them
  there with `LearnerConfig(scale_cap=...)` below 8.
- **A value on a code that cannot see progress.** A terminal reward on a state code that is
  the same along the approach surprises by a fixed amount every time, and the shared synapses
  drift into the latch. Make outcomes symmetric or put progress in the code.
- **A shared code, a readout on a rail, a thinned seam.**
  [`preflight`](#check-before-the-first-lesson) finds all three.
- **A critic faster than the actor.** A large `eta_critic` zeroes the dopamine before the
  actor moves; a large `eta` puts an output on its rail in one lesson. Read `delta` and
  `saturation` over repeated outcomes before raising either.
- **Eligibility mixed along one approach.** Several decisions in one episode put approach and
  avoid nudges in one trace; take one decision per episode and credit it.

## Sequential tasks and prerequisite chains

A brain that learns a single step well, food in one move, often fails a chain of steps: gather
wood, gather stone, make fire, warm up. The usual sign is a policy that settles on the one
action that pays most often. The readings below come from the
[prerequisite-chain nursery](../benchmarks/chain/README.md), the
[key-door nursery](../benchmarks/keydoor/README.md) and the
[competing-skill ring](../benchmarks/competing/README.md).

### What breaks a chain

- **Hidden state.** If the observation does not show what the creature holds, the right action
  depends on history and the chain is a short-term memory task. With the stage hidden, no
  learning arm in the chain nursery beats the memoryless ceiling (62 completions per thousand
  moments): stages with identical observations cannot be told apart. The working trace carries
  history ([memory](memory.md#working-trace)); give the last module enough width when history
  has to carry the state.
- **Late pay.** Credit across intervening choices travels through the actor's eligibility and
  the critic. With pay only at the completion of a four-step observed chain, the greedy policy
  completes as often as uniform random (42 per thousand moments) while answering 0.67 of the
  stages right: the early steps are learned and the whole chain is not. The key-door nursery
  acquires at delays of two and five choices with a raised critic rate (item 2 below) and fails
  at ten.
- **Readout saturation.** Near saturation a finite update produces little contrast between
  actions, and exploration cannot escape the latch. Read the motor activations and the margin
  between the chosen and the next action before changing a rate
  ([reward traps](#traps-with-their-measurements)).
- **Loud carried context.** In a chain the next action depends on what the brain sees in the
  present moment. A carried interpretation fed back too strongly makes it depend on the moment
  before. The founder trace scale (`scales["trace"]`) of 1.2 lets the present decide at birth,
  and learning strengthens the carry where the past predicts the outcome
  ([memory](memory.md#working-trace)).
- **Continued learning.** A policy acquired at the nursery's actor step can erode under its own
  continued learning when the later pay is signed and noisy: the noisy pay overwrites the
  associative store's one-shot record of the chosen action's reward (`memory_rate` 1.0; a new
  brain composes 0.2, which averages it, #169). A
  smaller actor step does not prevent it (item 3 below). More drilling past acquisition moves
  the policy instead of consolidating it.

### The pattern the measurements support

1. **Observe the chain's state and shape the pay.** Expose inventory as senses (has wood, has
   stone, has fire) and keep a hidden-inventory condition as a separate memory test. Pay
   executed actions through `Brain.live`; teaching a sequence by labels is a separate
   contract. With the state observed and pay at every correct step, the chain nursery's brain
   completes 204 per thousand moments against a tabular learner's 250 and answers 0.92 of the
   stages right. Backward lessons (episodes that start at a random stage with the
   prerequisites granted) reach 33: the granted completions meet the need, so the brain calms
   and learns less. Disclose supplied shaping and verify the full chain from stage zero under
   end-only pay.
2. **Raise the critic rate when credit spans about five choices.** The key-door nursery runs
   `actor_eta_critic=5.0` and acquires rule A in 18 of 20 lives, against 11 of 20 at the
   composed 0.3; at 10 the values run high and the creature latches the opposite way. Select
   the rate on the task with the composed 0.3 as control.
3. **Average or freeze the associative record where noisy pay follows acquisition.** In the
   competing ring, actor steps of 0.003 and 0.001 retain no more than the nursery's 0.03. A
   brain with an averaged record (`memory_rate=0.2`, a new brain's rate) acquires as well as
   one at `memory_rate=1.0` and takes about 0.10 off the loss over the ring. A store frozen where the noisy pay begins
   (`retune(memory_rate=0.0, consolidation=0.0)`) holds the skill there at a cost to a
   competing one.
4. **Choose `live` or `step` after the caregiver leaves.** `live` learns only while aroused,
   by a surprising outcome or by pay below what life usually brings; `step` learns from every
   outcome ([routine and repair](start.md#routine-and-repair-live)). Which path erodes a taught
   chain depends on the founder. Where most surprises are punishments, `live` alone does not
   hold the skill, an averaged record holds a little more and a frozen store holds it (item 3).
   Teach by demonstration and correction through the episodic memory
   ([episodic](memory.md#episodic-memory-what-happened-in-this-situation-before)).
5. **Read the greedy policy stage by stage.** Whole-chain success hides a stalled stage. Read
   the greedy answer, the right action's probability and its margin at every stage, and the
   sampled policy's completion rate beside the greedy one: a frozen greedy copy can answer one
   action in every situation while the sampled probabilities move with the situation.
6. **Declare the probe schedule and freeze the budget.** Longer training can degrade an
   acquired skill. Probe saved copies at a declared schedule (the chain nursery probes every
   2,000 of 20,000 moments), select the training budget in development, freeze it before
   testing fresh founders and disclose any checkpoint selection.

### Settings involved

The composed founders ([brain](tune.md#defaults-and-expert-overrides)) are `actor_gamma`
0.9, `actor_lam` 0.8, `actor_eta` 1.0 and `actor_eta_critic` 0.3
([their meaning](#eligibility-traces-and-dopamine)). The chambers run their own operating
points, which are not library defaults:

| Chamber | Operating point | Controls |
| --- | --- | --- |
| Prerequisite chain | modules (48, 24), `actor_eta` 0.03 (bias 0.003), `actor_lam` 0.6, `actor_gamma` 0.95, `actor_eta_critic` 5, temperature 0.3 | a tenth of the step; uniform random, memoryless ceiling, frozen newborn, tabular Q-learning |
| Key-door | `actor_eta_critic` 5.0, `actor_lam` 0.95, `actor_gamma` 0.95, `actor_eta` 0.1, selected on development seeds | the composed 0.3, 0.8 and 0.9 |
| Competing ring | the chain founder above; arms retune `actor_eta` to 0.003 or 0.001 at the ring's door, or compose with `memory_rate=0.2` | the nursery's 0.03 and `memory_rate` 1.0 |

Set them through `Brain.compose(..., actor_eta=0.03, ...)` or retune a running life
([retune](tune.md#retune-the-same-life)).

### Work in progress

- A measured memory horizon for a hidden chain state: [#84](https://github.com/muellerberndt/cadence/issues/84).
- How carried context persists and is revised across a task: [#189](https://github.com/muellerberndt/cadence/issues/189).
- A mechanism that keeps an acquired skill at the nursery's own step: [#169](https://github.com/muellerberndt/cadence/issues/169).
- Delayed credit beyond five choices: [#111](https://github.com/muellerberndt/cadence/issues/111).
- Conventional competitors (online Elman and GRU actor-critics) on the chain: [#184](https://github.com/muellerberndt/cadence/issues/184).

## A cue the answer does not show

When two situations end in the same observation and only an earlier cue tells which action
pays, a new brain's memory carries the cue: in the [two-cue chamber](../benchmarks/credit/README.md), every
outcome learned through `step`, six development founders reach held-out .90 on 6, 5 and 2 of 6 at delays of
0, 1 and 2 events, against 0 of 6 at each delay for `recall=False` and 5, 2 and 0 of 6 for the founder memory.
Without the memory, compose the brain with a tenth of the actor step and the episodic memory:

```python
cued = Brain.compose(12, 2, arousal=True, actor_eta=0.1, actor_eta_bias=0.01,
                     memory_choice=True, recall=False)
assert cued.memory_choice and cued.basal_ganglia.config.eta == 0.1
```

The brain without a memory does not learn this choice from the outcome. Its associative record is
keyed on the observation, which both situations share at the answer, so its vote there is the
same for both cues, and at the composed actor step the readout walks to one action for both.
The context records of `memory_choice=True` key on the stream's temporal context, so the two
cues' records differ at the shared answer; the slower step keeps the readout off a rail while
the actions the records chose teach it.

In the [two-cue chamber](../benchmarks/credit/README.md) (a cue, then an identical two-event
suffix, the answer paid +1 or -1, every outcome learned through `step`) these settings reach
held-out .90 on 11 of 12 fresh founders, against 2 of 12 at `recall=False` and the composed actor. The context
keeps half of the cue per moment: one event between the cue and the suffix gives 4 of 6
development founders, two give 0 of 6. `memory_extinction` and `memory_context_hold` leave
these readings as they are. The two-cue readings are measured through `step` only. In the other
reward chambers, read under `live`, the same settings acquire the key-door chamber on 16 of 16
lives (8 without), win 30 of 30 blocks of a bandit with reversals (27) and learn the contextual
bandit on 6 of 6 seeds (3); in the odour nursery two of six lives at the short exposure reverse
late (10 of 12 lives pass every check, 11 without).

In the chamber, that brain's own updates move the preference at the answer about 40 times
more along the association activity both cues share than along the cue, mostly through the
forward term's presynaptic activity and the motor bias. Right and wrong outcomes carry that
shared part in equal halves that cancel, so the shared preference wanders, and the outcomes
sink the common motor drive onto the activation's flat branch, where the cue part barely moves
the policy. An actor-critic with exact gradients on a small model of the same structure,
trained on the same episodes, learns the answer on 6 of 6 development founders: the structure
carries the cue, and the local update does not deliver it.

Taught by a lesson at the answer instead, the cue has to reach the answer through the events
in between. The working trace keeps about half of it per event and fails from two events. The
recall memory with no age charge, `recall_recency=0.0` as in a new brain, carries it through eight
(12 of 12 fresh founders at 4 and at 8; the founder recency of 0.25 reads 0 of 6 at 8). The
binding chambers' continuing life holds under this setting, since its
[position term](memory.md#position-how-long-ago-a-moment-happened) prefers the newer slots.
The trace, the slots and the settled activity of the moment before carry the cue to the
answer together; with all three cleared every founder answers at chance. In the prerequisite
chain taught by lessons along the teacher's path, the intermediate stages stay right on every
founder with and without the setting. In the reward-learning chain chamber (one continuing
stream through `live`), a fresh copy with no slots answers the intermediate stages right less
often (by .06 to .17 with this setting, by up to .22 at the recall memory's founder recency),
and with this setting the chain is completed from its start as often as without it.

## Reward at the end of an episode

In a game that pays only when it ends, every move's dopamine is the outcome less the critic's
value of the position the move was made in, so the critic decides how the moves of a game share
its outcome. A critic whose values move together across positions gives every move the same
baseline. What it can rank is set by what the association region it reads represents, by the
experience it gets and by its rate. Read the spread of its values, school it on demonstrated
episodes before the brain's own outcome learning starts, and teach the region the outcome with
a value lesson.

### Check the spread of the values

The mean value settles near the average return whatever the critic ranks. Read the values of a
fixed set of positions beside their mean and, where the positions' outcomes are known, how well
the values rank them. `brain.basal_ganglia.value_of(brain.stimulus(positions, memory=False))`
settles each position cold and leaves the streams as they are. `actor_eta_critic` sets the
spread more than the ranking: a raised rate spreads the values wider and ranks the positions no
better.

### A critic school

Replay demonstrated episodes through `step` before the brain's own outcome learning starts,
with the actor's rates, the associative store's write rate and its consolidation at zero. The
critic is then the only part that learns: the actions `step` issues did not run, and nothing
credits them. `memory_rate=0` alone leaves the consolidated record learning at `consolidation`
(0.05, more with salience), which would record the outcomes of the issued actions.

```python
def walk_school(rate, steps=600):
    """Demonstrated walks on a line of seven cells that pay only at the ends."""
    cells = 7
    walker = Brain.compose(cells, 2, modules=(16,), seed=0, recall=False)
    school = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "memory_rate": 0.0, "consolidation": 0.0}
    kept = {name: walker.describe()["genes"][name] for name in school}
    walker.retune(actor_eta_critic=rate, **school)
    actor = walker.learner.brain.efficacy.copy()
    rng = np.random.default_rng(1)
    place = np.full(4, cells // 2)                  # four streams, each walking from the middle
    walker.step(np.eye(cells)[place])
    for _ in range(steps):                          # the demonstrator's steps run, not the brain's
        place = place + rng.choice([-1, 1], size=4)
        reward = np.select([place == cells - 1, place == 0], [1.0, -1.0], 0.0)
        done = (place == 0) | (place == cells - 1)
        place = np.where(done, cells // 2, place)
        walker.step(np.eye(cells)[place], reward=reward, done=done)
    walker.act(np.eye(cells)[place], greedy=True)   # nothing left awaiting an outcome
    assert np.array_equal(actor, walker.learner.brain.efficacy)  # the actor did not move
    assert not walker.hippocampus.consolidated.any()  # the store recorded no issued action
    walker.retune(**kept)                           # the brain's own outcome learning starts here
    return walker.basal_ganglia.value_of(walker.stimulus(np.eye(cells)[1:-1], memory=False))

composed, raised = walk_school(0.3), walk_school(3.0)
assert composed[-1] > composed[0] and raised[-1] > raised[0]  # both rank the two ends
assert raised.std() > 2 * composed.std()            # a raised rate spreads the values wider
```

The school ranks the positions of the games it replays. The brain's own games then move the
critic toward the outcomes of its own play, and a raised critic rate lets them overwrite the
school sooner; the composed rate kept the school's ranking best.

### A value lesson

During the copy lessons a second motor slot is taught each moment's eventual outcome, so that
the association region has a reason to represent who is winning. The outcome must be known when
the lesson is given: teach a game's moments once it has ended, or replay recorded games.

```python
judge = Brain.compose(7, 4, slots=(2, 2), modules=(16,), seed=0)  # a move of two, an outcome
cells = np.eye(7)[[1, 2, 4, 5]]
moves, won = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])        # won: 1 for a won episode
judge.lesson(cells, np.stack([moves, won], axis=1))                # one label per slot
assert judge.predict(cells).shape == (4, 2)                        # one choice per slot
```

Teach the outcome slot 1 for a moment of a game that was won and 0 for one lost or drawn:

- **The taught outcomes must vary.** Taught on a demonstrator's own games, most of them won,
  the slot ranked positions near chance and the association region did not move.
- **On varied outcomes it shapes the region.** Taught at the positions of strong and weak
  players, each labelled with the demonstrator's move there and the player's own outcome, the
  region's readout ranked the eventual winner better and the slot itself ranked well. Outcomes
  shuffled between games barely moved the readout.
- **The critic gains through a school.** After a critic school the critic ranked positions
  better than without the slot, on every seed, at the end of the school and after the brain's
  own games that followed it.
- **The move slot pays a little.** Agreement with the demonstrator falls slightly, with
  shuffled outcomes too, so the cost is the second slot's. On varied outcomes the brain played
  slightly better against fixed opponents.

The composed critic reads the association region, not the slot; `probabilities` reads the
slot's choice among its two outcomes.

## A reward many moments after the decision

A reward paid when an episode ends reaches the decision that earned it through the actor's
eligibility, and the composed eligibility fades by `actor_gamma * actor_lam`, .72 a moment: ten
moments after a decision .04 of its contrast is left, fifty moments after it one part in ten
million. The critic carries credit across the gap only where the brain senses the coming outcome
on the way (a key in a pouch, a score on a screen); where the moments in between look the same
after a right and a wrong decision, the eligibility is the one path. Raising `actor_gamma` and
`actor_lam` to 1 keeps the contrast and makes the critic's value diverge, since the critic shares
that discount. `actor_hold` sets the actor's decay apart from the critic's:

```python
import numpy as np
from cadence import Brain

held = Brain.compose(9, 2, seed=0, arousal=True, actor_eta=0.1, actor_eta_bias=0.01,
                     actor_hold=1.0)
config = held.basal_ganglia.config
assert config.decay == 1.0 and config.gamma * config.lam == 0.9 * 0.8  # the critic's own
rng = np.random.default_rng(0)
owed = None
for episode in range(4):                    # a T-maze: the cue at the junction, 10 moments, the outcome
    cue = int(rng.integers(2))
    junction = np.eye(9)[[cue]]
    choice = held.step(junction) if owed is None else held.step(junction, reward=[owed], done=[True])
    for _ in range(10):                     # the corridor looks the same after either choice
        held.step(np.eye(9)[[2]], reward=[0.0])
    owed = 1.0 if int(choice[0]) == cue else -1.0  # paid with the next episode's first moment
```

With `actor_hold=1.0` each act's contrast stays in the actor's synapse, bias, recall and hearing
traces until the episode ends (`done`, or `reset`), and the step a decision gets is the
episode's dopamine summed over the moments from the decision on: the outcome less a baseline
of the critic's values, `V(s_0) + (1 - gamma) * (V(s_1) + ... + V(s_k))`. A hold below 1 gives a horizon of about `1 / (1 - hold)` moments; in a life with no
episode ends, choose one, since a hold of 1 there sums every act since the last `done`. The
critic keeps `actor_gamma` and `actor_lam`. `None`, the founder, keeps `gamma * lam` for every
trace and every bit of a brain that names no hold, and saves its checkpoint without the field; a
brain saved with a hold carries it, and 0.96.1 and earlier refuse that checkpoint.

Take a tenth of the actor step with the hold. A held trace sums the contrast of every act of the
episode into each outcome's step, and at the composed `actor_eta` of 1.0 the readout goes to a
rail within the first quarter of a life ([the saturation latch](#traps-with-their-measurements),
[issue 296](https://github.com/muellerberndt/cadence/issues/296)): in the
[delayed-credit chamber](../benchmarks/delayed_credit/README.md) the hold at the composed step
learns a 50-moment T-maze on 2 of 6 development founders, and at a tenth of the step on 6 of 6.

On six fresh founders (the chamber's declared run, 4,096 episodes through `step`), the tenth of
the step with `memory_choice=True` and `actor_hold=1.0` learns the T-maze at delays of 10 and 50
on 6 of 6, as an LSTM actor-critic with exact gradients through the episode does; without the
hold the T-maze at 50 reads 2 of 6, the default 0 of 6. Where the brain senses the coming outcome
the hold costs: on the key-door, whose pouch the critic sees, the same settings without the hold
read 5 of 6 at delay 50 and with it 3 of 6, the missing founders latched on one reach for both
keys. At 200 moments the latch takes most lives (T-maze 3 of 6, key-door 0 of 6). Hold the actor's
credit where the moments between a decision and its outcome look alike whichever way it went, and
leave it out where the brain sees the outcome coming.

## Replaying a life through `step`

`step` cannot tell a replayed observation from a live one, so replaying a saved copy's day
with the reward each choice earned is more experience at the same rates; a pass boundary is
an ordinary transition unless `done` marks it. `Brain.compose` has no night of its own. In
the [night-replay chamber](../benchmarks/replay/README.md) a night is worth about as many
fresh decisions when the day's updates are sound and harms when they are not. Before a night,
read `capped` (outcomes beyond the cap teach their sign alone), `saturation` and the
[working trace](memory.md#memory-in-the-composed-brain); a smaller actor rate makes a long
night safer. Gate a slept brain on a frozen copy's dependence on the observation; a constant
greedy choice is right only where no observation pays:

```python
copy = Brain.load(brain.save("slept.npz"))
copy.reset()
policies = []
for observation in np.eye(2)[[0, 1, 1, 0]]:
    copy.act([observation], greedy=True)              # no learning; compare with predict
    policies.append(copy.basal_ganglia.probabilities(copy.basal_ganglia.state)[0])
policies = np.asarray(policies)
dependence = float(0.5 * np.abs(policies - policies.mean(axis=0)).sum(axis=1).mean())
assert 0.0 <= dependence <= 1.0                       # near 0: the choice ignores the observation
```

`predict` reads neither trace nor memory: against the greedy `act` it shows what holds the
choice.

## Check before the first lesson

`preflight(brain, outputs, plastic, drives, *, level=0.5, saturation=0.02, shared_max=0.25,
eligibility_min=10, steps=100, tolerance=1e-4)` settles a `NeuralGraph` under `drives`, one
row per situation, and warns, naming the remedy, about:

| Finding | Remedy |
| --- | --- |
| A readout within `saturation` of 0 or 1 under every drive | [`calibrate_bias`](settling.md#calibrating-the-operating-point) |
| Plastic senders (at or above `level`) sharing over `shared_max` of their code between two drives | an upstream gain selected on the `specific` fact ([connectomes](anatomy.md#brains-from-a-connectome)) |
| Fewer than `eligibility_min` plastic synapses from active senders onto a readout under some drive | a whole seam (`seam_report`) |
| Two drives overlapping on over `shared_max` of their driven neurons | senses that tell them apart |

It returns `{"outputs", "shared", "inputs", "eligibility", "warnings"}`; an empty list
certifies neither equilibrium nor acquisition. `seam_report(connectome, pre, post)` takes
population names or indices and returns `classes`, `synapses`, `coverage_pre` (share of
`pre` with a class onto `post`), `classes_per_post` and `median_count`.
`naive_efficacy(connectome, plastic)` sets each plastic class to its sign times the mean
count over its own count, so every class has the same effective weight, and keeps the sign
elsewhere: a lesson without an inherited preference. Report it, since it changes measured
strengths.

```python
import cadence as cd

wiring = cd.layered(4, 12, 2, density=1.0, seed=0)
graph = cd.NeuralGraph(wiring, cd.learning_neuron_model())
readouts = list(wiring.populations["output"])
plastic = np.isin(wiring.pre, wiring.populations["hidden"]) & np.isin(wiring.post, readouts)
drives = np.zeros((2, wiring.n))
drives[0, wiring.populations["input"][:2]] = 1.0
drives[1, wiring.populations["input"][2:]] = 1.0
check = cd.preflight(graph, readouts, plastic, drives)
assert any("the seam is thin" in warning for warning in check["warnings"])  # senders below 0.5
seam = cd.seam_report(wiring, "hidden", "output")
assert seam["classes"] == 24 and seam["coverage_pre"] == 1.0
start = cd.NeuralGraph(wiring, cd.learning_neuron_model(),
                       efficacy=cd.naive_efficacy(wiring, plastic))
```

## A custom reward learner

Build an actor on a declared output population. `slots=[2, 3]` gives two categorical motor
slots of unequal size that settle together; `act` returns one index per slot.

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 5, density=1.0, seed=0)
learner = cd.Learner(cd.NeuralGraph(connectome, cd.learning_neuron_model()),
                     connectome.populations["output"], slots=[2, 3])
ac = cd.ActorCritic(learner, critic=connectome.populations["hidden"],
                    config=cd.ActorCriticConfig(gamma=0.9, lam=0.8, eta=0.01))
drive = np.zeros((4, connectome.n))
drive[:, list(connectome.populations["input"])] = [0.5, 0.8]
for _ in range(3):
    action = ac.act(drive, temperature=[0.2, 0.4])   # (4, 2): one draw per slot
    reward = (action[:, 0] == 1).astype(float)        # from your environment
    report = ac.learn(reward, np.zeros(4, dtype=bool), drive)
assert {"td_error", "delta", "dopamine", "capped", "saturation", "trace"} <= set(report)
```
