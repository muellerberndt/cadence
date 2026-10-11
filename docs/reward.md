# Learning from reward

Cadence combines free/nudged contrasts, eligibility traces and a reward prediction
error without a backward computation graph. Each synapse keeps its own eligibility
trace, and one broadcast dopamine signal gates every trace into a weight change.
This page describes those components and the tests that check them.

For one continuing stream, compose with `arousal=True` and use
[`Brain.live`](continuous.md#routine-and-repair-live); use `step` to learn from
every actual outcome or to supply a current teaching label. Both receive the
preceding action's reward and return the next action. There is no separate
training mode. The phase-level API below exposes the same mechanism for custom
architectures and measurement.

Reward learning reads `actor_eta`, `actor_eta_bias` and `actor_eta_critic`;
`learning_eta` and `learning_eta_bias` govern demonstrations and do not change
`live` reward updates. `temperature` is shared by the teaching softmax and actor
policy. Set these names in `Brain.compose` or `brain.retune`, with explicit
paired actor rates when needed. The [composed defaults table](brain.md#defaults-and-expert-overrides)
records their ownership and the remaining genes.

## Eligibility traces and dopamine

`ActorCritic` wraps a `Learner`. Acting runs the free phase; the action is a draw from the
softmax over each categorical motor slot (or, with `Bins`, one softmax draw per
continuous dimension). Unequal categorical slots use the existing `Learner(slots=[...])`
layout; every slot participates in the same settlement. Two nudged phases, toward and away from the
action taken, give every synapse its per-row contrast, an estimate of that action's score
under the [equilibrium assumptions](learning.md). The actor always uses a softmax
nudge, even when the learner uses quadratic imitation. Sampling and eligibility
use the same temperature, including explicit temperatures and per-slot exploration.
For slot temperatures `T_i`, the local mask is scaled by `c/T_i`, where `c` is the
minimum of the base temperature and the supplied temperatures. The contrast estimates
`c * grad(log policy)`; several slots estimate the sum of their log probabilities.
At the base temperature the mask remains one and the existing scale is unchanged.
The bounded mask also permits very small temperatures without an overflowing gain.
This removes the systematic score bias caused by sampling a hot policy while
differentiating a sharper one. Finite phases retain their approximation error and
do not guarantee acquisition. Each synapse keeps an eligibility trace
of those contrasts:

```
elig[e]      <- gamma * lam * elig[e] + contrast[e]
delta         = r + gamma * V(s') - V(s)
efficacy[e] += eta * delta * elig[e]
```

This is a three-factor rule. Presynaptic and postsynaptic activity enter through the
contrast and accumulate in the synapse's eligibility trace; the prediction error `delta`,
the dopamine signal, is the same for every synapse and sets the sign and size of the
weight change. `V` is a linear readout of the neurons named as the critic (usually the
hidden population). The actor signal is clipped (`dopamine_cap`) and can be centred
and scaled by its own running statistics (`dopamine_center`). The critic has two
explicit choices:

- `critic_signal="modulated"` preserves the original rule: the critic shares the
  actor signal. This can stabilize an update, but clipping or centring means it need
  not predict mean discounted return in reward units.
- `critic_signal="td"` uses the raw prediction error for the critic, independently
  of actor modulation. Use this when a calibrated return prediction is required.
  It can require retuning the critic rate and reward scale. The default
  `critic_signal="auto"` selects `"td"` when dopamine centering is enabled,
  otherwise `"modulated"`.

The critic's step can be divided by its trace's energy (`critic_normalize`). Reports
include the absolute raw `td_error`, the absolute modulated `delta`, signed
`dopamine`, and `capped`, the share of observed rows whose signal exceeded
`dopamine_cap` before the clip: near 1, every outcome moves the synapses by the
cap times the trace, whatever its size. A time limit is not a terminal state: pass
`bootstrap=` the value of the last observation for a truncated row.

The adaptive local step uses `momentum` and `normalize` to keep a running mean and RMS
of each synapse's own steps, with corrections for its short history. Each synapse reads its
own optimizer state; this does not guarantee stable learning for every task or setting.

## Centered dopamine and selective activity

With `ActorCriticConfig.dopamine_center > 0`, the actor maintains per-stream
running error statistics. `dopamine_floor` sets a band around the running centre
in which its modulation is zero, and has no effect when centering is disabled.
This reduces actor updates for familiar outcomes. It is not a correctness test:
momentum can still move parameters, the centered critic uses raw TD error, and
eligibility phases and associative writes still run, so zero modulation is not
zero computation. Each actual outcome is still supplied once.

[Life](api.md#life-cadencelife) provides a different opt-in mechanism: a governor
selects habit, imagination or learning for a `BeliefPatch`/`Steered` composition
using recorded prior surprise. The application supplies the cheap habit.
This is a separate sequential composition, not a governor automatically wired
into `Brain.compose` or a joint neural-equilibrium certificate.

## A custom reward learner

For custom wiring, build an actor on a declared output population:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 3, density=1.0, seed=0)
learner = cd.Learner(
    cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"]
)
drive = np.zeros((1, connectome.n))
drive[:, list(connectome.populations["input"])] = [[0.5, 0.8]]
ac = cd.ActorCritic(learner, critic=connectome.populations["hidden"],
                    config=cd.ActorCriticConfig(gamma=0.99, lam=0.95, eta=0.001, eta_bias=0.0001,
                                                eta_critic=0.5, normalize=0.999, momentum=0.9))
action = ac.act(drive)                       # settle, draw, keep eligibility
reward, done, next_drive = np.ones(len(drive)), np.zeros(len(drive), bool), drive  # from your environment
ac.learn(reward, done, next_drive)           # settle the next state, dopamine, every synapse moves
```

## Decision and reset contract

Drives are nonempty finite `(batch, neurons)` arrays. Rewards and optional bootstrap
values have shape `(batch,)`; `done` is boolean. Validation precedes trace and parameter
updates. Each batch row keeps its own potential, adaptation, eligibility and reward
centering. Ended rows reset before the next free phase; their neighbours get the usual
single phase. The next free phase estimates the bootstrap value before the parameter
update. The next action settles again from that warm state under the updated parameters.
A cached free phase is reused only when both the drive and the brain parameters are
unchanged. Budget for both free phases and the action’s two nudged phases.

`act(..., greedy=True)` is an evaluation read: it clears eligibility and cannot be
followed by `learn`. Calling another `act` replaces the pending action. Finish the
reward transition before changing learner parameters elsewhere, or call `reset`.
`reset` clears stream state and centering, but preserves the learned critic, actor and
optimizer history. `Brain.save/load` includes this full state; `Learner.save`
does not save a separately constructed actor-critic.

## Reward evidence

Measure actual task outcomes separately from shaped reward. An
undiscounted potential difference does not generally preserve the original
objective under discounting; that guarantee requires the discounted form
`gamma * Phi(next) - Phi(now)` with appropriate terminal handling.

The [experience guide](experience.md) connects eligibility, prediction error
and valence to the rest of the learning life.

## Budget the ongoing loop

Choose the settling budget using changing observations and the task's outcome
metric. Compare cold and warm starts, reset independent episodes, and
measure all decision work. A small step count or a cached response to one unchanged
input does not establish reliable control or lower decision cost.

## What delayed credit requires

A reward that follows the reading it belongs to is a record: written in one exposure at
the fast rate and read back at that reading ([records](memory.md#records)). The
eligibility trace is for the policy where the reward arrives after other decisions.

An eligibility trace remembers which action could have caused a later reward; it
does not remember an arbitrary observation or learn a memory address. At a delay of
`d` transitions its contribution is multiplied by `(gamma * lam) ** d`. A trace
that fades too soon cannot assign useful credit, while a long trace includes more
unrelated actions. State representation, exploration and the critic matter too.

The [delayed-credit test](../tests/test_child.py) pays a choice three moments later,
after blank moments with presses of their own; the learned greedy choice reaches a hit
rate of at least 0.8 with the trace (`lam=0.9`) and at most 0.65 without it (`lam=0`). The
[policy credit tests](../tests/test_policy_credit.py) check that the eligibility
differentiates the sampled policy. Passing these small tests does not establish
performance on any larger task.

For a real task, record raw held-out return, forgetting and every interaction used
by rehearsal or planning. Choose the eligibility horizon from actual action-to-reward
delays, and test imitation, reward practice and rehearsal separately. A biological
analogy supplies a hypothesis; the behavioral test determines whether it works.

## Replaying a life through `step`

`step` cannot tell a replayed observation from a live one. An application that lets
a saved copy of a brain re-experience its own day, calling `step(observation,
reward=...)` on the recorded observations with the reward each replayed choice would
have earned, gives that copy more experience of the same day at the same rates: every
replayed transition writes the associative memory and moves the actor and the critic
as a live one would, and the boundary between passes is an ordinary transition unless
`done` marks it. That is not a consolidation operation. The sleep of the
[record patch](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night)
is a separate model that teaches its slow weights fixed completions from its store;
`Brain.compose` has no night of its own, and its `SynapticMemory` consolidates on
each outcome.

Whether such a night helps depends on what the day's updates do, which the
[night-replay chamber](../benchmarks/replay/README.md) measures: one decision per
interval, a signed outcome in small units, a fixed cost for every non-resting
action, a world in which a cue decides the paying move and one in which nothing
does, and the same number of fresh awake decisions as the control. Three
readings decide whether a night is safe:

- **`report["capped"]`.** Outcomes of several units against `dopamine_cap=1.0`
  clip to their sign, so a gain of one unit and a loss of six move every synapse
  equally. At a high share the night repeats that distortion.
- **`report["saturation"]`.** At the composed actor rate on a single stream the
  policy walks to a held action, and replay walks it further.
- **The working trace.** At the composed amplitude the association region can
  run on its own history instead of the observation
  ([memory](memory.md#three-kinds-of-memory-in-brain)).

With the trace informing rather than leading and a smaller actor rate, replay
raised the agreement with the paying rule as the awake control did; at the larger
rate a long night lowered it. The smaller the rate, the safer a long night.
Measure a frozen copy's policy before adopting a slept brain: its dependence on
the observation is the gate, and a constant greedy choice is only correct where
no observation pays.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(4, 3, modules=(16,), seed=0)
day = np.eye(4)[[0, 1, 2, 3, 0, 1]]
brain.step(day[:1])
for observation in day[1:]:
    brain.step([observation], reward=[0.5], done=[False])
copy = Brain.load(brain.save("slept.npz"))  # the gate reads a frozen copy
copy.reset()
policies = []
for observation in day:
    copy.act([observation], greedy=True)  # no learning; the trace carried as in life
    policies.append(copy.basal_ganglia.probabilities(copy.basal_ganglia.state)[0])
policies = np.asarray(policies)
dependence = float(0.5 * np.abs(policies - policies.mean(axis=0)).sum(axis=1).mean())
assert 0.0 <= dependence <= 1.0  # near 0: the choice ignores the observation
```

Compare the greedy `act` with `predict`, which reads neither trace nor memory, to
see which of the graph, the trace and the memory holds the choice.

## Traps, with their measurements

These are properties of the rule and the readout, each with a reading in the
`learn` report. The last three are checked before any lesson by
`preflight(brain, outputs, plastic, drives)`, which names the remedy for each;
run it first.

- **Continued learning can interfere with acquired behaviour.** Arousal gates
  updates but does not protect particular earlier responses while the brain
  learns. A smaller actor step reduces interference; it is a bounded control, not
  a retention mechanism, and [#169](https://github.com/muellerberndt/cadence/issues/169)
  owns the candidates. Raising the arousal threshold only removes learning
  opportunities, and associative-memory `consolidation` controls record writes,
  not actor weights. Compare retained skills and adaptation in the same brains.
- **The temperature is relative to the activation range.** The action is a
  softmax over the output neurons' activations divided by `temperature`, and
  activations lie in [0, 1]. At 0.05, two outputs differing by 0.3 make a choice
  with probability 0.998, the nudge's push `beta * (target - p)` is nothing, and
  no synapse moves. Outputs that live near rest take 0.05; outputs that sit at
  0.7 to 1.0 under their drive take 0.3. Set it from the naive activations.
- **The saturation latch.** Once one output saturates and the other is silenced,
  the activation has no slope, the contrast `a_pre * (b_plus - b_minus)` is zero,
  and the rule cannot leave the state whatever the reward says.
  `report["saturation"]` is the fraction of output activations within 0.02 of 0
  or 1 and `report["trace"]` the mean absolute eligibility: saturation near 1
  with the trace near 0 is the latch, and it is visible long before the reward
  curve shows it. The remedies are a gain that puts the outputs in the sensitive
  band, `LearnerConfig(scale_cap=...)` below the default 8 so learning cannot
  drive them out of it, and a readout that is neither silent nor saturated under
  the task's drive before any lesson.
- **The value on a code that cannot see progress.** With a terminal reward and a
  linear critic on a state code that is the same all along the approach, the
  value stays near the discounted mean everywhere, so every outcome surprises by
  a fixed amount and the shared synapses drift until the latch. Make the outcomes
  symmetric where the assay allows it, or give the code the progress, such as a
  level that rises with distance.
- **The assay decides whether avoidance can be rewarded.** Where one avoidance
  turn leads nowhere, no reward ever arrives for it and the rule learns "approach
  everything" whatever the wiring. Where avoiding one option means taking the
  other, and every search ends at an outcome, the association can be learned.
  Build the arena so that every action the readout can take has an outcome.
- **Centring and the critic.** `dopamine_center > 0` makes the actor's dopamine
  the surprise over its running level; a critic fed the same signal chases a
  moving target and its value runs away. `critic_signal="auto"` (the default)
  gives the critic the raw error whenever the dopamine is centred.
- **A cap applied by rebuilding the brain.** Clipping efficacies with
  `brain.with_parameters` after every decision re-uploads every weight on the
  torch backend, at a factor of ten per decision. The cap is a config field,
  `LearnerConfig(scale_cap=...)`, applied inside the update.
- **A code the lesson cannot attach to.** The actor moves the synapses from the
  active cells of the state code; if that code is nearly the same for every
  stimulus, every lesson moves every stimulus, and a few outcomes at one cue drag
  both probabilities together. Measure the code with the `specific` fact before
  the lesson and select the offending population's gain on it
  ([brains from a connectome](connectomes.md)); no rule downstream repairs it.
- **A readout on a rail, and a readout with a past.** A connectome carries no
  operating point: at one global threshold a readout cell can sit at 1.00 under
  every input and its opposite at 0.01, where the nudge has little response on
  either. `calibrate_bias` searches each readout cell's bias toward a declared
  mean over the supplied situations; check the resulting means and full residual
  before installing them, since finite bisection does not guarantee the target.
  The [qualified calibration option](learning.md#calibrating-the-operating-point)
  checks every attempted state. Measured synapse counts are themselves a past: a
  naive readout can already prefer one cue, and an option that is never taken is
  never rewarded. `naive_efficacy` starts a seam with every plastic class at the
  same weight.
- **A seam thinned by custody.** A synapse floor removes a distributed memory
  along with the noise, leaving a seam that a lesson barely moves. Keep the seam
  a lesson will move at every count and read `seam_report` before designing on it.
- **The critic that learns faster than the actor.** With a large `eta_critic` the
  value reaches the outcome in a few trials and the dopamine goes to zero while
  the actor, whose contrast is small near a rail, has barely moved; with a large
  `eta` one lesson puts an output on its rail. Read `report["delta"]` across
  repeated outcomes and `report["saturation"]` before raising either.
- **Eligibility mixed along one approach.** Several decisions along one approach
  give the trace both approach and avoid nudges from the same episode, and the
  reward then credits whichever nudge was larger. Take one decision per episode
  and credit it to that decision.

## Rates under normalization

`ActorCriticConfig.normalize` divides the actor's reward-modulated eligibility
signal by its own bias-corrected running RMS plus `1e-3`. With momentum, the
numerator is its bias-corrected running mean. This applies independently to
efficacies (`eta`) and neuron biases (`eta_bias`), using the actor's optimizer
history; the supervised learner's normalization setting does not enable or
disable it. The [learning guide](learning.md#rates-under-normalization) gives the formula
and its limits.

A consistent signal above the floor gives a parameter increment near its
rate before masks, tying, decay and clipping. The rate is not a fixed increment
or an upper bound. A fresh `ActorCriticConfig` derives an unset `eta_bias` as
`eta / 10`; an explicit value is kept. Composed named overrides and `retune`
preserve the current bias rate unless it too is named: use
`actor_eta_bias=None` to derive it again, or supply an explicit paired rate.
Construction warns when the bias rate exceeds a positive `eta`. Under RMS
normalization both rates are absolute per-parameter steps: a bias step much
larger than the synapse step rewrites the policy into a bias policy whose greedy
action does not depend on the observation, with the biases far from
initialization while the synapses stay near it. The composed brain keeps
`eta_bias=0.05` at `eta=1.0`. A development sweep of `eta=0.001` to `0.003` is a
starting experiment. Measure free behavior against the task's controls; no range
guarantees learning or prevents a policy from collapsing to a held action.

Construction emits `RuntimeWarning` if `normalize > 0` and either `eta` or
`eta_bias` exceeds `0.05`. This advisory threshold does not alter settings or
certify smaller rates. `eta_critic` is excluded: `critic_normalize` divides the
critic update by trace energy, which is a separate rule from the actor's RMS
normalization. The warning introduces no new learning rule or default.
