# From a task to patch ports

Cadence aims to reuse local overlap repair and equilibrium detuning across
tasks. Scaling a network is useful only when its observations, actions and
learning signal preserve the distinctions the task requires. The current
library does not establish that arbitrary tasks become solvable by increasing
width, data or compute.

Design a [continuing life](world-model.md): bootstrap, ordinary use, witnessed
disruption and local correction in the same acquired brain. Keep isolated
classification and calibration as mechanism controls. Measure retained and new
behavior after repair, alongside stable-use work and repair work; do not replace
the brain or bypass its memory between observations to stand in for this test.

Every issue and plan declares whether it tests a bounded component, integrated
continuing behavior, or infrastructure. State the applicable equilibrium
boundaries, simpler control and preservation gates. Infrastructure can explain
applicability without requiring an unrelated whole-brain demonstration. Use the
[existing capability owners and canonical principle](../CONTRIBUTING.md#scope-and-evidence-for-behavior-over-time).

## Start at the boundary

A patch has bounded local state, declared ports, readback and a repair rule.
Give each external port a meaning, unit, availability condition and source.
Distinguish an absent observation from an observed zero. Distinguish a goal,
an executed action, a measured consequence and a private prediction even when
their arrays have the same shape. The application supplies these semantics;
array dimensions alone cannot enforce them.

First supply correct action records directly to the environment or instrument.
Check what actually happens. Then compare the learner's proposed actions using
the same boundary. This separates a learning failure from an interface that
cannot realize the requested behavior.

For a sampler, an event can expose sample identity, onset time, gain and
duration. Those are playable controls. Spectral brightness and onset density
are observations of sound; mapping them directly to pitch or repeated drum
hits is an additional instrument design. It is not an acquired inverse model.
Known action demonstrations are also not a transcription of those actions
from a musical mix. That inference requires its own evidence.

The same distinction applies to robots, games and interactive software: a
forecast of an observation is not automatically an executable action.

## Check which differences survive encoding

Use two cases whose distinction matters. If the encoder maps them to the same
ports, a deterministic learner cannot recover that distinction from those
ports alone. More context or a learned prior can support a guess, but cannot
make the missing observation unique.

For example, consider one event in each sixteenth-note interval. Times
`j + 0.25` are equally spaced. Times `j + 0.25` for even `j` and `j + 0.75`
for odd `j` alternate between long and short gaps. Both have the same event
count, occupied intervals and unsigned distance to the nearest grid point.
An unsigned timing-spread summary cannot distinguish them. This is a
counterexample for that timing summary, not a claim that all audio features
of the two rendered waveforms are identical.

Use explicit signed event offsets when the task requires them. Quantized
patterns can still have musical groove; preserving microtiming alone does
not establish rhythmic quality, correct instrument choices or arrangement.

## Match the update to the behavior

`TemporalPatchNet` accepts arbitrary positive input/output dimensions. A
temporal patch can represent an event, a frame or a bar, provided the units
and time boundaries are explicit. Moving from one vector per bar to one
event per patch changes the task representation and optimization geometry;
it is not a new learning rule. Count the changed path length, output ports,
state, exposure and compute when comparing them.

Its loss is a squared error over supplied targets, optionally weighted by
`output_precision`. Common outputs can dominate less frequent but decisive
differences. Report per-behavior errors alongside the aggregate error. Supplied
unit conversions or task weights are legitimate, but do not call them learned
importance or estimated uncertainty. Fit them on declared training data and
keep them fixed during evaluation.

Test ambiguity before increasing capacity: correct versus swapped intention,
cue removal, context removal and activity reset. If individual tasks are
learnable but joint training averages them, examine surviving cue activity,
output sensitivity, competing gradients, horizon and capacity. A missing
coordination module is one hypothesis, not the only explanation.

One subtle limitation of the current bias-free residual model is exact sign
symmetry. From zero state, negating every input negates every hidden state
and output. Two otherwise identical positive behaviors cannot be learned
from opposite sign inputs alone. A declared shared context or common initial
activation breaks this fixture symmetry. It is not a clock or proof of a
learned context model. Do not silently alter a frozen task to repair it.

## Test coherent behavior over time

Sequence competence requires coherent, continuing learned context and temporal
organization to govern behavior. Sequential output and sensory feedback are
allowed. Working traces, associative memory and efference are legitimate; the
question is what behavior their acquired context supports. Next-event accuracy
and self-feeding alone are bounded component results, not evidence of phrase or
whole-sequence competence.

For an integrated sequence claim, declare the relevant context and time span,
then compare the same acquired brain with retained context, absent context and
perturbed context under matched exposure. Where relevant, test recovery after a
disturbance and counterfactual contexts requiring different continuations. Keep
the simpler control, earlier skills, pending continuation, refusals and work in
the acceptance gates. These comparisons must test the claimed behavior, rather
than only its aggregate next-event score.

Separate equation residual from musical surprise, structural coherence and task
success. State which coordinates are live in the solve and which are held:
default `Brain.act` qualifies a present state with held trace and memory input,
not a joint future trajectory. This does not require a new phrase module or
whole-song-at-once generation. Check existing state, information and experience
first; any added mechanism must use local repair in the same coupled global
equilibrium. Adapters must not conceal answers, phase or arrangement. Supplied
context or clocks belong in the declared information budget and appropriate
controls, not in a claim that the brain learned that organization itself.

## Close the action and consequence loop

For an acting system, test this full sequence:

1. Observe the current situation and retain the relevant context.
2. Form a private action proposal under a sustained intention.
3. Predict the proposal's consequences using relationships learned from
   actual action/consequence records.
4. Execute the selected action and measure the actual consequence.
5. Repair the discrepancy and test earlier skills after the update.

`imagine` supplies isolated temporal predictions. The
`TemporalPatchNet.plan` interface repairs bounded continuous input ports under that same
learned model; the [interaction guide](interaction.md) demonstrates acquisition
and execution in a small nonlinear body. A new instrument still needs
validated action/consequence acquisition and measured execution. A desired
outcome supplied during settling is not evidence that an executable action
can cause it; replay the chosen action through the actual body.

## Test coordination with the same patch rule

A possible shared workspace is a bounded patch that reads proposals and
mismatches from other patches and feeds back a common intention or constraint.
Specialized patches can still use the same overlap and detuning operations.
Recursively exposing summaries of this process is a research hypothesis about
coordination, with no claim about consciousness.

Before treating such a workspace as necessary, compare the same task with
and without useful summary feedback. Count its state, ports and work. Include
a matched recurrent model, disconnected feedback and shuffled feedback.
Require an improvement in a named behavior, such as maintaining a motif while
changing accompaniment, or revising a strategy under new constraints. Activity
in the new patch alone does not establish useful coordination.
For specialization, selectively perturb a region and measure both its distinct
contribution and the continued coordination of the whole system.

Current APIs expose detached self-readback and private predictions. They do
not yet implement or validate an automatically specialized, recursive global
workspace. The [architecture guide](architecture.md) gives the exact boundary.

## Scale after the joint test

A useful small test checks acquisition, retention after competing experience,
private prediction, successful action and the task's actual quality together.
Use independent tasks or withheld cases for generalization. Larger runs should
report quality against data, state, time and compute, with simple controls.
Formal consistency and numerical convergence are necessary checks on the
implementation; neither substitutes for that behavioral evidence.

See [common missteps](missteps.md) for failures this protocol is designed to
expose, and [temporal learning](temporal.md) for the concrete API.

For the general research goal and application-independent creativity criteria,
see [creativity and evolving self-reflection](creativity.md). Musical examples
illustrate these requirements; they do not define the learning architecture.
