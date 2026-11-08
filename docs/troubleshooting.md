# Troubleshooting

Short answers to the problems builders meet first, each linking the page that owns the topic.
[Common missteps](evaluate.md#common-missteps) lists the ways a good-looking number can be wrong.

## Install and run

**`pip install cadence-net==0.95.1`, then `import cadence`.** Python 3.11 or newer and NumPy are
the only requirements.

**Do I need a GPU?** No; everything runs on NumPy float64. `[fast]` adds Numba and SciPy,
`[accel]` torch and `[apple]` MLX for the settling brain ([backends](backends.md)).

**A run reads other numbers after an upgrade.** A new brain holds the recall memory and learns at
another teaching point from 0.95.0 ([what it holds](memory.md#what-recalltrue-gives-a-new-brain)).
`recall=False` composes the brain of 0.94.0 to the bit; a saved brain keeps what it was saved with. With
a memory, 0.95.0 replaces the address lesson, so `recall_founder=True` restores the settings and not the
bits.

**A world paid by reward alone reads lower than without a memory.** That is a stated cost
([what the composition gives up](memory.md#what-the-composition-gives-up)): pass `recall=False`.

**Which interface do I want?** `Brain.compose(..., arousal=True)` and `brain.live(...)` for one
continuing stream; `step` for explicit teaching, batches or learning from every outcome. Other
contracts are in the [specialist guides](README.md#by-situation).

## Shapes and encodings

**`ValueError` on shapes.** A settling brain takes a drive `(batch, n)`, one column per neuron,
and `Learner.step` integer labels `(batch,)`; a composed `Brain` takes observations
`(batch, inputs)` ([shapes](../README.md#shapes)).

**Categorical outputs.** A settling brain's `step` takes the index of the wanted output
neuron ([build from your data](anatomy.md#build-your-own-brain)).

**An absent observation reads as zero drive.** Add a presence input when an observed zero must
differ from no observation.

## The settling brain

**Accuracy stays at chance.** Read `free_budget_exhausted` in the lesson report first
([finite free phases](settling.md#finite-free-phases-that-run-out-of-budget)).
`learner.calibrate(training_drive)` picks the sampled gain whose mean top output per row is
nearest 0.5 ([calibration](settling.md#calibrating-the-operating-point)). Keep output projections
reciprocal (`Projection(..., reciprocal=True)`, the default) so a nudge reaches the hidden
neurons; select `eta`, `eta_bias` and `beta` on development data ([knobs](settling.md#7-every-knob)).

**The phases do not converge.** Raise `free_steps` and `nudged_steps` or lower the gain. A small
movement per step can hide a large equation error under saturation: `brain.residual(drive, state)`
checks the equations at a state, `brain.equilibrate` continues to a tolerance within a budget
([concepts](settling.md#settling-and-equilibrium)), and `cd.certificate(brain)` bounds uniqueness
and distance when its hypotheses hold ([certificate](settling.md#the-settling-certificate)).

**Qualified teaching refuses every lesson.** A `LearningPhaseError` in phase `nudged` or
`opposite` naming a budget mismatch means the nudged budget is below the free one; a composed
brain teaches with 12 nudged steps (32 at `teaching="deep"`, a new brain's) against 1024 free steps. Set `learning_qualified=True,
learning_nudged_steps=1024` through `retune` ([budgets](settling.md#qualified-teaching-budgets)).

**Two learners on one brain.** Build the second with `Learner.sharing(first, outputs, ...)` so
each update reaches the other, and give them disjoint plasticity masks
([two learners](settling.md#two-learners-on-one-graph), [heads](anatomy.md#which-synapses-a-head-owns)).

## Memory

**A memory I switched on changes nothing.** Erase it in one of two loaded copies of the saved
brain and compare answers ([check](memory.md#check-that-a-memory-does-the-work)).

**`predict` ignores the cue shown a moment ago.** `predict`, `accuracy` and `fit` read only the
learned weights; test a memory through `act`, `step` or `live`
([memory](memory.md#memory-in-the-composed-brain)).

**Which memory carries this task?** See [the memory inventory](memory.md#complete-memory-inventory). No memory holds a state the observation never shows.

**What survives `reset()`?** Records, the recall address map and learned weights. The working
trace, efference copy, heard history, episodic context and recall slots are cleared
(`reset(keep=("slots", "arousal"))` keeps the slots too). The arousal state carries on as it
is by default; after `reset(keep=())` an adult brain starts calm with no income or error
references and no reward scale, so at `need=0` only a surprise at its own forecasts rouses it
and the scaled need wants nothing until outcomes enter the new scale
([reset and save](start.md#reset-and-save)).

**Stored keys do not match a brain built with another `code`.** Keys are valid only under the
code they were stored with; `retune` refuses `code` ([the observation code](senses.md#the-observation-code)).

## Evaluation

**Which number do I report?** Held-out data split before tuning, a control (majority outcome,
persistence, a linear predictor), and log loss with calibration beside accuracy
([evaluation](evaluate.md)).

**How do I make a result reproducible?** Pin the release, fix every seed and bind numbers to
sources with a [receipt](evaluate.md#receipts), as the `verify.py` scripts in [benchmarks](../benchmarks/) do.

## Learning from reward

**Before the first lesson.** Run `preflight(brain, outputs, plastic, drives)`; it names a remedy
for each finding ([check before the first lesson](reward.md#check-before-the-first-lesson)).

**The curve is flat, `report["saturation"]` near 1, `report["trace"]` near 0.** The readout
latched ([the traps](reward.md#traps-with-their-measurements)).

**A decision costs ten times a settle on torch.** Clipping efficacies with `with_parameters`
after every decision re-uploads every weight; use `LearnerConfig(scale_cap=...)` instead.

**The policy always picks the same action.** Check that every action has an outcome and that
the outcomes are symmetric before touching the rule.

**The greedy choice ignores the observation.** Read `report["capped"]` (the share of outcomes
clipped at `dopamine_cap`) and `report["saturation"]`, compare `predict` with greedy `act`
([replaying a life](reward.md#replaying-a-life-through-step)), and tune through the
[rate ownership table](tune.md#defaults-and-expert-overrides).
After a long life of rewards at the composed `actor_eta` of 1.0 the motor neurons can sit at
their limits; train long lives at a tenth of that rate or less ([actor rates](reward.md#actor-rates)).

**Reward comes only at the end of an episode and the values sit in one band.** Read their
spread across positions beside their mean, and their ranking of positions whose outcomes are
known; a critic school and a value lesson are the measured remedies
([reward at the end of an episode](reward.md#reward-at-the-end-of-an-episode)).

**Every stimulus learns the same lesson.** The state code is shared. Score a `specific` row
between two stimuli where the plastic synapses leave, and select the igniting population's gain
by protocol ([connectomes](anatomy.md#brains-from-a-connectome)).

**A readout never moves under large dopamine.** Its cells saturate; `calibrate_bias` centres
them, and `naive_efficacy` removes a naive preference.

**The plastic set is a few hundred synapses.** `seam_report` shows a seam thinned by the
synapse floor; rebuild the fixture keeping that seam.

## Connectomes

**A predicate over a large population fails.** `active` asks the mean over every member to reach
0.5; use `sparse` for a subset and `lateralized` for a left-right pair ([protocols](../benchmarks/instruments/README.md)).

**A gate passes on the whole brain, fails on the sub-net.** Recruit more hops or lower the
synapse floor, and keep the closure receipt ([connectomes](anatomy.md#brains-from-a-connectome)).

**The brain ignites or is silent at every gain.** Use a gain per cell class, selected by
protocol ([gain selection](../benchmarks/instruments/README.md#gain-selection-under-a-sparsity-cap)).

## Where to ask

Open an [issue](https://github.com/muellerberndt/cadence/issues) with the smallest script that
shows the behaviour, the library version and the platform. [Contributing](../CONTRIBUTING.md)
says how to run the tests.
