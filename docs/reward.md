# Learning from reward

A Cadence brain learns from reward with a three-factor rule: each synapse keeps an
eligibility trace of its recent contrasts, and one broadcast prediction error (the
dopamine) turns every trace into a weight change, with no backward computation graph.
`Brain.step` and an aroused [`Brain.live`](start.md#routine-and-repair-live) take the
reward of the preceding action and return the next one:

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

Use it when outcomes pay actions: bandits, reversing rules, rewards a few choices later
([reversal](../benchmarks/reversal/README.md), [key-door](../benchmarks/keydoor/README.md),
[competing skills](../benchmarks/competing/README.md)). With signed, noisy outcomes and a
cost per move, the composed actor rate of 1.0 on one stream walks the policy to one held
action; a tenth of it, as above, lets the cue decide
([night-replay chamber](../benchmarks/replay/README.md)). Labelled lessons belong to
[learning](settling.md#learning-the-freenudged-rule).

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

## Settings

`ActorCritic(learner, critic, config=None, seed=0)`. In a composed brain
each field is the named setting `actor_<field>`, set in `Brain.compose` or `brain.retune`;
the composed founders are `gamma=0.9, lam=0.8, eta=1.0, eta_bias=0.05, eta_critic=0.3,
eligibility_steps=12` ([defaults table](tune.md#defaults-and-expert-overrides)).

| `ActorCriticConfig` field | Default | Meaning |
| --- | --- | --- |
| `gamma` | `0.99` | Discount |
| `lam` | `0.9` | Eligibility decay; credit at a delay of `d` transitions is scaled by `(gamma * lam) ** d` |
| `eta` | `0.5` | Actor synapse step per unit dopamine per unit trace |
| `eta_bias` | `None` | Bias step; `None` derives `eta / 10` at construction |
| `eta_critic` | `0.05` | Critic step |
| `dopamine_cap` | `1.0` | Clip of the actor's signal; `0` for none |
| `eligibility_steps` | `None` | Steps of each reward-nudged phase; `None` uses the learner's `nudged_steps` |

`beta`, the base `temperature` and `tolerance` come from the learner's `LearnerConfig`
(composed: `learning_beta`, `temperature`); the teaching point (`teaching`) sets the steps
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
  steps), and `address_reward_step` with a `memory`.

## Decision and reset contract

Drives are nonempty finite `(batch, neurons)` arrays, rewards and bootstrap values
`(batch,)`, `done` boolean; validation precedes every update. Each row keeps its own
potential, adaptation and eligibility. `done` rows restart from rest and
bootstrap from `bootstrap`, or zero without it. A time limit is not terminal: mark the
truncated row `done` and pass its value, `value_of(next_drive)`, as `bootstrap`. `learn`
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

## A cue the answer does not show

When two situations end in the same observation and only an earlier cue tells which action
pays, the composed brain does not learn the choice from the outcome. Its associative record is
keyed on the observation, which both situations share at the answer, and every outcome writes
it, so its vote there is the same for both cues; at the composed actor step the readout then
walks to one action for both. In the [two-cue chamber](../benchmarks/credit/README.md) (a cue,
then an identical two-event suffix, the answer paid +1 or -1 with the next episode's first
event and `done`, every outcome learned through `step`) the composed founder reaches held-out
.90 on 0 of 6 development founders and 2 of 12 fresh ones. The slower actor step with the
episodic memory learns it:

```python
cued = Brain.compose(12, 2, arousal=True, actor_eta=0.1, actor_eta_bias=0.01,
                     memory_choice=True)
assert cued.memory_choice and cued.basal_ganglia.config.eta == 0.1
```

The context records key on the stream's temporal context, so the two cues' records differ at
the shared answer, and the slower step keeps the readout off a rail while the actions the
records chose teach it. With the suffix right after the cue it answers 1.00 on 6 of 6
development founders and on 11 of 12 fresh ones. One event between the cue and the suffix gives
4 of 6 and two give 0 of 6: the context keeps half of the cue per moment and the readout
latches on one action. `memory_extinction` and `memory_context_hold` leave these readings as
they are. Added to the other reward chambers, the same settings acquire the key-door chamber on
16 of 16 lives (8 without), win 30 of 30 blocks of a bandit with reversals (27) and learn the
contextual bandit on 6 of 6 seeds (3); in the odour nursery two of six lives at the short
exposure reverse late (10 of 12 lives pass every check, 11 without; these chambers' readings
are their `live` arms). The two-cue readings are measured through `step`; the chamber under
`live` was not read.

Taught by a lesson at the answer instead, the cue has to reach the answer through the events
in between. The working trace keeps about half of it per event and fails from two; the recall
memory with no age charge, `recall=True, recall_recency=0.0`, carries it through eight (12 of
12 fresh founders at 4 and at 8, where the founder recency of 0.25 reads 0 of 6 at 8). It keeps
the binding chambers' continuing life, where its
[position term](memory.md#position-how-long-ago-a-moment-happened) prefers the newer slots.

## Replaying a life through `step`

`step` cannot tell a replayed observation from a live one, so replaying a saved copy's day
with the reward each choice earned is more experience at the same rates; a pass boundary is
an ordinary transition unless `done` marks it. `Brain.compose` has no night of its own. In
the [night-replay chamber](../benchmarks/replay/README.md) a night is worth
about as many fresh decisions when the day's updates are sound and harms when they are not.
Read `capped` (outcomes beyond the cap teach their sign alone), `saturation` and the
[working trace](memory.md#memory-in-the-composed-brain) first; a smaller actor rate makes
a long night safer. Gate a slept brain on a frozen copy's dependence on the observation; a
constant greedy choice is right only where no observation pays:

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

## Traps, with their measurements

Each is a property of the rule or the readout, with its reading in the `learn` report;
[troubleshooting](troubleshooting.md#learning-from-reward) maps symptoms to them.

- **Continued learning interferes with acquired behaviour.** Arousal gates updates and
  protects no earlier response. A smaller actor step reduces interference without retaining a
  skill ([#169](https://github.com/muellerberndt/cadence/issues/169)); `consolidation`
  governs record writes and leaves actor weights alone.
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

## Actor rates

Every actor parameter steps by its rate times the dopamine times its own eligibility: there
is no running average of the steps and no division by their size, so the rates are steps
per unit dopamine per unit trace. A bias rate far above the synapse rate makes a policy whose
greedy action ignores the observation. A named `actor_eta` in `Brain.compose` or `retune`
keeps the current bias rate; pass `actor_eta_bias=None` to derive `eta / 10` again.
Construction emits a `RuntimeWarning`, and changes nothing, when `eta_bias` exceeds a
positive `eta`. `eta_critic` is outside it.

## Sequential tasks and prerequisite chains

A continuing brain that learns a single step well, food in one move, often fails a chain of
steps: gather wood, gather stone, make fire, warm up. The usual sign is a policy that settles
on the one action that pays most often. The measurements below come from the
[prerequisite-chain nursery](../benchmarks/chain/README.md), the
[key-door nursery](../benchmarks/keydoor/README.md) and the
[competing-skill ring](../benchmarks/competing/README.md).

### What breaks a chain

**Hidden state.** If the observation does not show what the creature holds, the right action
depends on history and the chain is a short-term memory task. In the chain nursery with the
stage hidden, no learning arm beats the memoryless ceiling (62 completions per thousand
moments, uniform over the chain's actions): stages with identical observations cannot be told
apart, and the policy settles on one action or keeps sampling. The brain carries history in
its working trace ([memory](memory.md#working-trace)); give the last module enough width
when history has to carry the state.

**Late pay.** Credit for an action across intervening choices travels through the actor's
eligibility and the critic. With pay only at the completion of a four-step observed chain,
the greedy policy completes at 42 per thousand moments, equal to uniform random, while the
sampled policy reaches 77 and the greedy answer is right at 0.67 of the stages: the early
steps are learned and the whole chain is not. The key-door nursery acquires at delays of two
and five choices and fails at ten; at five, only 4 of 10 lives acquire with the composed
critic rate, against 10 of 10 with a raised one (item 2 below).

**Readout saturation.** Near saturation a finite update produces little contrast between
actions, and exploration cannot escape the latch; the key-door chamber records it under
punished exploration. Read the motor activations and the margin between the chosen and the
next action before changing a rate ([reward traps](#traps-with-their-measurements)).

**Loud carried context.** In a chain the next action depends on what the brain sees now. A
carried interpretation fed back too strongly makes it depend on the moment before. The
founder trace scale (`scales["trace"]`) of 1.2 lets the present decide at birth, and learning
strengthens the carry where the past predicts the outcome ([memory](memory.md#working-trace)).

**Continued learning.** A policy acquired at the nursery's actor step can erode under its own
continued learning, most clearly when the later pay is signed and noisy
([#169](https://github.com/muellerberndt/cadence/issues/169)). In the
[competing ring](../benchmarks/competing/README.md), under signed noisy pay the loss reproduces
at every actor step, because the noisy pay overwrites the associative store's one-shot record
of the chosen action's reward (`memory_rate` 1.0; an averaged record at 0.2 takes about 0.10 off
the loss over the ring, and a store frozen at the ring's door holds the skill there).
More drilling past acquisition moves the policy instead of consolidating it.

### The pattern the measurements support

1. **Observe the chain's state and shape the pay.** Expose inventory as senses (has wood,
   has stone, has fire) and keep a hidden-inventory condition as a separate memory test. Pay
   executed actions through `Brain.live`; label teaching of a sequence is a separate contract
   ([#121](https://github.com/muellerberndt/cadence/issues/121)). With the state observed and
   pay at every correct step, the chamber's brain completes 204 per thousand moments against a
   tabular learner's 250 and answers 0.92 of the stages right. Backward lessons (episodes that
   start at a random stage with the prerequisites granted) reach 33: the granted completions
   meet the need, the brain calms and learns less. Disclose supplied shaping and verify the
   full chain from stage zero under end-only pay.
2. **Raise the critic rate when credit spans about five choices.** The key-door nursery runs
   `actor_eta_critic=5.0`; at 10 the values run high and the creature latches the opposite
   way. Select the rate on the task with the composed 0.3 as control.
3. **Average the associative record where noisy pay follows acquisition.** In the competing
   ring, actor steps of 0.003 and 0.001 retain no more than the nursery's 0.03 and do not
   prevent the loss under signed noisy pay; a brain composed with an averaged associative
   record (`memory_rate=0.2`) acquires as before and takes about 0.10 off the loss over the ring,
   and a store frozen where the noisy pay begins (`retune(memory_rate=0.0, consolidation=0.0)`)
   holds the skill there at a cost to a competing one
   ([the competing ring](../benchmarks/competing/README.md)).
4. **Choose `live` or `step` after the caregiver leaves.** `live` learns only while aroused,
   by a surprising outcome or by pay below what life usually brings; `step` learns from every
   outcome ([routine and repair](start.md#routine-and-repair-live)). Which path erodes a taught
   chain depends on the founder. Where most surprises are punishments, as in the melee,
   `live` alone does not hold the skill; an averaged record holds a little more, a frozen store
   holds it (item 3). Teaching by demonstration
   and correction goes through the episodic memory ([episodic](memory.md#episodic-memory-what-happened-in-this-situation-before)).
5. **Read the greedy policy stage by stage.** Whole-chain success hides a stalled stage.
   Read the greedy answer, the right action's probability and its margin at every stage, and
   the sampled policy's completion rate beside the greedy one: a frozen greedy copy can
   answer one action in every situation while the sampled probabilities move with the
   situation.
6. **Declare the probe schedule and freeze the budget.** Longer training can degrade an
   acquired skill. Probe saved copies at a declared schedule (the chain nursery probes every
   2,000 of 20,000 moments), select the training budget in development, freeze it before
   testing fresh founders and disclose any checkpoint selection.

### Settings involved

The composed founders ([brain](tune.md#defaults-and-expert-overrides)) are `actor_gamma`
0.9, `actor_lam` 0.8, `actor_eta` 1.0 and `actor_eta_critic` 0.3; their meaning is in
[reward](#eligibility-traces-and-dopamine). The instruments run their own operating
points. These are chamber settings, not library defaults:

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
