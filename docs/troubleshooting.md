# Troubleshooting

Short answers to the questions builders ask first, each pointing at the page that has
the rest. [Common missteps](missteps.md) is the longer list of ways a good-looking
number can be wrong.

## Install and run

**`pip install cadence-net==0.81.0` and then `import cadence`.** The distribution is `cadence-net`;
the import is `cadence`. Python 3.11 or newer and NumPy are the only requirements.

**Do I need a GPU?** No. Everything runs on NumPy float64. `[fast]` adds Numba and SciPy
for the settling brain's transport, `[accel]` adds torch and `[apple]` adds MLX for the
settling brain on a device; the temporal and record patches are NumPy, and the belief
patch's slow half has a torch twin ([backends](backends.md)).

**Which interface do I want?** Start with `Brain.compose(..., arousal=True)`
and `brain.live(...)` for one continuing stream. Use `step` for explicit teaching,
batches or learning from every outcome. The
[specialist guides](quickstart.md#specialist-guides) cover other model contracts;
[build from your data](build.md) explains their shapes and encodings.

## Shapes and encodings

**`ValueError` on shapes.** Settling brains take a drive of shape `(batch, n)` with one
column per neuron of the connectome, and `Learner.step` takes integer labels of shape
`(batch,)`. Temporal, record and belief patches take paths `(batch, time, ports)` and
targets `(batch, time, outputs)`; a single moment is `(batch, 1, ports)`. The
[shapes table](orientation.md#shapes) lists every brain.

**Categorical outputs.** For a record patch declare `groups=(k,)` and give one-hot
targets; the readout ends in a softmax per group. For a settling brain the outputs are
neurons and `step` takes the index of the wanted one. A temporal patch's outputs are
continuous levels.

**An absent observation is not a zero.** Zeros mean zero drive. Add a presence port when
an observed zero must differ from no observation ([temporal learning](temporal.md#context-private-continuation-and-readback)).

## The settling brain

**Accuracy stays at chance.** Inspect free output activity and its equation residual.
Check `free_budget_exhausted` in the lesson report first: finite teaching that
runs out of free sweeps learns from states that never settled, and warns.
`learner.calibrate(training_drive)` can choose a more responsive global gain —
by default it places the top output of each row near 0.5 rather than pushing
the whole readout's mean there —
but cannot guarantee that its sampled candidates reach the target activity.
The [calibration report](learning.md#calibrating-the-operating-point)
exposes candidate residuals and refusals, and qualified learning excludes
unqualified candidates. Check that a nudge on the outputs can reach the hidden
neurons: the projection into the output region must be reciprocal
(`Projection(..., reciprocal=True)`, the default). Select `eta`, `eta_bias` and
`beta` on development data, then freeze them before confirmation
([every knob](learning.md#7-every-knob)).

**The phases do not converge.** Raise `free_steps` and `nudged_steps` in `LearnerConfig`,
or lower the gain. Check `brain.residual(drive, state)`: a small movement per step can hide
a large equation error when neurons saturate ([concepts](concepts.md#settling-and-equilibrium)).

**Qualified teaching refuses every lesson.** Read the `LearningPhaseError`: when the
failed phase is `nudged` or `opposite` and the message names a budget mismatch, the
nudged budget is below the free one. Under `qualified=True` both budgets are settle
budgets, and `Brain.compose`'s finite teaching default `nudged_steps=12` cannot settle
realistic input to tolerance. Opt in with a nudged budget comparable to the free one
([qualified teaching budgets](learning.md#qualified-teaching-budgets)); the constructor
warns about such configurations, including through named overrides. Use
`learning_qualified=True, learning_nudged_steps=1024` on a composed brain
with its default free budget; preserve other settings with `retune`.

**Is the answer an equilibrium?** `cd.certificate(brain)` says whether settling is a
contraction and bounds the remaining distance; when the row mass is above the limit,
`brain.equilibrate` with a tolerance and `brain.residual` are the checks
([certificate](certificate.md)).

**Two heads on one brain.** An update replaces `learner.brain`; hand the new brain to the
other head before its next phase, and give each head disjoint plasticity masks
([which synapses a head owns](cortex.md#which-synapses-a-head-owns)).

## The temporal patch

**`observe` returns `updated=False`.** Read `result.reason`. `contrast_asymmetric` means
the two detuned paths did not sit symmetrically around the free path even after halving
`beta`: lower `beta`, shorten the path or the hidden width. A failed phase names the
phase; `no_decreasing_parameter_step` with `backtrack=True` means no rate lowered the loss
in replay ([checking a learning step](temporal.md#checking-a-learning-step)).

**The plan reports `step_cap`.** Reaching the cap is not convergence, and control can improve
all the same: execute the first action, measure, replan. `method="bfgs"` takes fewer iterations
on a badly conditioned action path ([planning](planning.md)).

**Zero free energy, wrong predictions.** The causal recurrence has zero defect by
construction. Surprise is the observation minus the prediction; the residual says the
model agrees with itself ([temporal learning](temporal.md)).

**Learning is slow.** One detuned chain costs on the order of `N * T * H^3`; keep the hidden
width and the path length modest ([scaling](scaling.md)).

## The record patch

**The slow weights learn nothing, the records recall everything.** The loss is a mean over
time and outputs, so the useful `rate` scales with their product; try 8 and let
`backtrack=True` halve it. A rule of the present input is learned in tens of updates; a
conjunction of this input and the last is learned by the gate and takes far longer, or by
a second patch in depth ([two patches in depth](record-patch.md#two-patches-in-depth)).

**The records recall nothing.** Give every unit of the reading unit variance, and count the
cells in use and the share of activations the most active cells take
([diagnosing a record store](record-patch.md#diagnosing-a-record-store)). A store written
hundreds of times per cell holds only its last writers at `record_rate` near one; use
`record_averaging=True` where retention of a corpus matters.

**Sleep changed nothing.** Sleep restructures what the store holds; it does not correct
it. Dreams from cues the store never met teach its guesses as facts. Dream the cues of the
day, and check the slow-weights-alone score on held-out data before and after
([acquisition in two phases](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night)).

**Recall by position fails.** A record is keyed by what was heard and the context; if the
same place in a phrase must be recalled after the phase shifted, position has to be in
the reading as ports of its own ([recall is by content](record-patch.md#recall-is-by-content-not-by-position)).

**`detune` refuses categorical ports.** The quadratic detuning result applies to the linear
readout only; `observe` computes the same learning signal by its backward scan for both.

**Which checkpoint format?** Default nets save format 2; a net with `groups` or batch
writes saves format 3. `load` reads both. A structured port's layout is saved with it.

## The belief patch

**Imagination decays while the one-step read improves.** The patch leans on the next frame.
Add the imagination loss: imagine from random moments and penalise the drift of the
imagined output ([the imagination loss](belief.md#training-the-transition-the-imagination-loss)).

**A wide picture port needs a huge rate.** The loss is a mean over time and outputs. Use
a rate in the thousands, or the torch twin with an ordinary optimizer ([belief](belief.md#learning)).

## Evaluation

**Which number do I report?** Held-out data, split before tuning; the slow weights and the
records separately; a control (the majority outcome, persistence, a linear predictor, the
search alone). For categorical ports, mean log loss and calibration beside accuracy
([task design](task-design.md), [scaling](scaling.md)).

**How do I make a result reproducible?** Pin the release or the commit, fix every seed,
and bind the numbers to their sources with a [receipt](receipts.md). The examples
repository's `verify.py` scripts are the pattern.

## Learning from reward

- The reward curve is flat and `report["saturation"]` is near 1 with `report["trace"]` near 0:
  the readout latched; see [the traps](reward.md#traps-with-their-measurements) for the
  temperature, the cap and the readout choice.
- The value in the report runs away while `dopamine_center` is set: leave `critic_signal` at
  `"auto"`, which gives the critic the raw error under centring.
- A decision costs ten times what a settle costs on the torch backend: something rebuilds the
  brain per decision (`with_parameters`); use `LearnerConfig(scale_cap=...)` for the cap and
  keep parameters on the kernel.
- The policy learns "always the same action": check the assay (every action needs an
  outcome) and the symmetry of the outcomes before touching the rule.
- The greedy choice is the same whatever the observation, before and after an offline
  replay of the brain's own day: read `report["capped"]` (outcomes beyond `dopamine_cap`
  teach their sign alone) and `report["saturation"]`, compare `predict` with the greedy
  `act`, and measure the per-observation policy of a frozen copy
  ([replaying a life](reward.md#replaying-a-life-through-step)). In that chamber,
  the composed actor rate of 1.0 and the default working trace contributed to
  failure; they are not a diagnosis of every single-stream task.
  Use the [rate ownership table](brain.md#defaults-and-expert-overrides) to tune
  the affected mechanism, then measure behavior again.
- Before any of the below: `preflight(brain, outputs, plastic, drives)` reads the readouts'
  slope, the plastic senders' shared code and the seam's eligibility under the task's drives and
  names the remedy for each finding.
- Every stimulus learns the same lesson: the state code is shared. Score a `specific` row
  between two stimuli at the population the plastic synapses leave; a cosine near one there
  means a population upstream ignites at the global gain, and its gain is selected by protocol
  ([brains from a connectome](connectomes.md)).
- A readout never moves although the dopamine is large: its cells sit at 0 or 1 under the
  task's drive (`report["saturation"]`); `calibrate_bias` centres them. If the naive readout
  already prefers one stimulus, the plastic seam's measured counts are doing it; start it
  with `naive_efficacy`.
- The plastic set is a few hundred synapses on a population of thousands: `seam_report` shows a
  seam thinned by the synapse floor; rebuild the fixture keeping that seam at every count.

## Connectomes

- A predicate over a large population fails although the pathway is there: `active` asks for a
  mean over every member, which a set of 1,312 descending neurons cannot reach; use `sparse`
  for a subset and `lateralized` for a left-right pair ([protocols](protocols.md)).
- A gate passes on the whole brain and fails on the sub-net a page settles: recruit more
  hops or a lower synapse floor and keep the closure receipt; the fruit fly needed 60,000 of
  150,802 neurons for every readout below 1e-3 ([connectomes](connectomes.md)).
- The whole brain ignites or is silent at every gain you try, or one region does: one global gain
  cannot serve every circuit's excitation-inhibition balance. A threshold does not tame a
  runaway loop (the fly's local neurons stayed a third active at a bias of -6); a gain per cell
  class does, selected by protocol on the facts downstream ([connectomes](connectomes.md)).

## Where to ask

Open an issue at [github.com/muellerberndt/cadence/issues](https://github.com/muellerberndt/cadence/issues)
with the smallest script that shows the behaviour, the library version and the platform.
[Contributing](../CONTRIBUTING.md) says how to run the tests and the documentation checks.
