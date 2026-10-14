# Cadence equilibrium world models

“Cadence equilibrium world model” names the architectural hypothesis and
intended capability. “Patch net” names the underlying computational mechanism.
This terminology does not rename existing classes or assert that every current
patch net implements a learned world model.

This guide defines an interpretation and integration contract for experiments
with Cadence. The proposed shared world-model architecture below is not an
implemented library component or a capability established by the current
examples. Existing APIs retain the scopes in the [architecture guide](architecture.md)
and [API reference](api.md).

The intended unit is an observer-like, self-reading patch: bounded local state,
ports or boundaries, readback, retained records, feedback or repair operations,
and inspectable evidence. Its active state should support predictions and
actions about the current situation. Calling that state an equilibrium does
not by itself establish accurate prediction, intelligent behavior or subjective
experience.

## A changing trajectory can be consistent

A moving world requires a changing representation. A learned path may satisfy
all its temporal constraints while its successive states differ. Following
that path cheaply is a useful interpretation of skilled behavior: expected
developments update the carried state and prompt familiar actions without
extensive reconsideration.

An equilibrium of an inference calculation instead concerns fixed evidence
and model parameters. If `z = F(z; evidence)` has converged, applying the same
map again returns the same interpretation. It does not advance the environment
or generate a later event. Predicting a future requires a transition model and
a specified action or action distribution.

The record patch's causal scan supplies the free path of its declared temporal
equations. A small seam defect checks those equations. Even zero defect can
coexist with inaccurate predictions of the external world.

## Three clocks

| Clock | What changes | What remains fixed within the operation |
| --- | --- | --- |
| Environment step | Actual observations, executed actions and carried state | Previously admitted evidence and its provenance |
| Inference iteration | Interpretation of the same evidence | Evidence, parameters and the admitted memory snapshot |
| Learning update | Parameters, memory contents or an explicitly learned address map | Which targets were observed, taught or imagined |

These are operation indices, not interchangeable physical clocks. An environment
step has the duration declared by its simulator or measured body interface;
irregular intervals need that information explicitly. Solver sweeps and the
neural `dt` relaxation fraction do not assign physical duration. Measure wall
time separately, including sensing, scheduling and qualification. The current
[`live`/`wait` contract](continuous.md#defaults-and-the-thinking-clock) specifies
which events advance trace, eligibility, arousal and learning.

An imagined future has its own step index within a private branch. Its steps
simulate environment time; they are not extra inference iterations on the
present. They also do not count as new external observations.

Applications must decide which clock a state update belongs to. Adding a
previous-state mixing term cannot add inference depth to a one-step read from
a zero initial state: that term is zero. Multiple internal steps or a deeper
port would be a separate computation with a separate cost.

## Stochastic innovation is different from teaching detuning

Even a correct probabilistic model cannot predict which random outcome will
occur. Once an outcome is observed, the current belief and appropriate action
may change without any change to the learned dynamics. Repeated systematic
errors can instead justify changing the model. State estimation and model
learning are distinct operations.

An individual random outcome can still produce a statistical learning update.
For a calibrated Bernoulli prediction, the cross-entropy logit gradient `p - y`
has zero conditional expectation, but usually is not zero on a single sample.
The distinction concerns systematic model drift, not a ban on sample-based
learning. Belief changes can be large even with correct stochastic dynamics.

For instance, a correct fair-coin model assigns one half to each outcome.
Observing heads resolves this toss; it does not justify replacing the model
with an always-heads rule. Prediction quality must be judged against the
declared uncertainty, using proper log loss and calibration where probabilities
are claimed. Increasing uncertainty indiscriminately must not make every error
appear acceptable.

An unexpected observation supplies evidence and possibly a teaching target.
It is not itself the numerical `beta` used in an equilibrium-propagation
contrast. Teaching detuning compares equilibria under a specified loss
perturbation, with the relevant boundary and parameters held fixed. Only an
accepted learning operation changes parameters. Inference damping, observation
innovation and teaching strength are different quantities.

The conditions of a detuning result depend on the component. Consult the
[record-patch guide](record-patch.md#detuning-as-the-acceptance-check),
[temporal learning guide](temporal.md) and
[graph certificate](certificate.md#equilibrium-propagation-scope); a convergence
certificate for one component does not automatically cover a different coupled
architecture.

## Evidence and branch isolation

Perception repairs an interpretation against actual evidence. Planning repairs
hypothetical actions and continuations against preferences, starting from that
factual interpretation. A preference must not remove an observed obstacle or
rewrite an earlier action to make a plan look successful.

For a reproducible decision, an application should freeze a complete relevant
snapshot: model parameters and revision, live context, record contents and
address statistics, observation preprocessing, random state, and any external
planner memory. The individual component snapshots cover only state owned by
that component. Verify branch isolation by comparing the live state before
and after private computations, including counters and normalization state.

Keep a provenance distinction between observed outcomes, supplied teaching
targets and imagined outcomes. Executing one proposed action supplies evidence
only for the action actually executed and its subsequent observations. It does
not validate every alternative branch. If an actuator clips a command, retain
the executed command as well as the proposal when needed for diagnosis.

During internal rollout, future actions may be supplied as candidate controls.
Recorded future observations or body states are unavailable unless an explicit
evaluation condition grants them. Predict all later state consumed by the
transition, or provide a declared internal body model. Represent unavailable
inputs with an explicit missing-data convention; absence is not an observation
of zero.

Records and compressed boundaries also depend on the representation that gave
them meaning. Changing a key encoder or a model can make existing memory
inconsistent. A proposed learned address mechanism needs a rule for freezing,
versioning or rebuilding its keys.

## What is available today

| Component | Implemented operation | Relevant limit |
| --- | --- | --- |
| `RecordPatchNet` | Gated causal context, residual records, adjoint learning, private reads and fixed-dream sleep | Its context does not read record outputs; no action-port planner or learned memory relevance |
| `RecordPatchNet.detune` | Centered contrast for the linear-readout slow patch | Check both solves; categorical ports are refused and records remain outside the energy |
| `RecordPatchStack` | Two contexts in depth with an adjoint through both | No joint `detune` solver; no inherited guarantee for a jointly free nonlinear energy |
| `TemporalPatchNet` | Learned temporal relationships, private imagination and bounded continuous-port planning; every contrast is checked for symmetry and beta halves when a detuned path leaves the free path's branch | Model accuracy and executed outcomes need separate validation |

A component's private `imagine` call does not by itself establish a learned
simulator. The application defines what its inputs and outputs mean and which
future variables are generated rather than supplied. A value evaluator used
inside an exact rules-based search remains a value evaluator.

## Proposed shared architecture

The candidate design has two organizing requirements:

1. A representation can be read, combined with other representations and
   revised before the next external action. Learned nonlinear ports should
   permit scene, body, proposed action and retrieved experience to interact.
2. Those representations and interactions are trained against witnessed
   consequences over time. Private planning and later consolidation reuse
   the acquired relationships, with imagined targets kept distinct from
   observations.

These requirements do not yet specify an API. Choices still needing
implementation and validation include the state representation, transition
distribution, recurrent or stacked update, memory address rule, multi-step
learning objective, stopping rule and behavior when a solve fails. A richer
nonlinear recurrence does not automatically retain the current quadratic
energy or centered-contrast guarantees.

Retrieved information would need to enter the interpretation before another
nonlinear computation if it is to affect inference beyond correcting an output.
Learning the query or address is a separate change. A table averaging output
residuals across episodes is not interchangeable with individually addressable
events from the current episode. Report what information a memory preserves,
what queries it can answer and what its full addressing operation costs.

The intended deployment has a cheap habitual response for familiar situations
and optional private continuation when additional computation is useful.
Consolidation could train an initializer or policy to reproduce validated
expensive decisions more cheaply. The current `sleep` distills fixed
completions into slow parameters; it does not implement that entire procedure.
Replay can improve generalization, but a self-consistent dream is not new
evidence of its own correctness.

## Evidence needed for the stronger claim

Skilled demonstrations can train a behavioral prior and predictive relationships
together. Synchronized action logs directly support action-conditioned learning;
video without controls needs an additional, grounded action-identification
method. Expert trajectories omit many mistakes and alternative actions, so
recovery experience and the learner's own interactions remain important. Keep
demonstrated actions, executed actions and hypothetical controls distinguishable.
Transfer across games requires whole-game holdouts and a declared demonstration
or interaction allowance; competence on trained games is a different claim.

Separate tests of perception, memory, transitions and decisions. Useful
measurements include novel combinations, delayed relevant cues, effects of
alternative actions, rollout error by horizon, calibrated uncertainty and
actual decision quality. Test record and slow outputs separately. For a
categorical prediction, define a valid normalized distribution and report mean
log loss and calibration alongside classification accuracy.

Compare added depth, internal iteration, learned addressing and changed
supervision separately before attributing an improvement to self-reading.
Compare consolidation with direct replay of the same retained evidence. A
successful habitual policy should retain outcome quality while reducing
decision computation on held-out situations, and still respond to changed
conditions.

Pin model and data revisions and separate model selection from final testing.
Match available information and tasks, and report parameters, retained memory,
addressing work, inference/rollout budgets, latency and training cost. Equal
update counts or hidden widths alone do not match resources. See
[task design](task-design.md), [scaling](scaling.md) and [receipts](receipts.md).
