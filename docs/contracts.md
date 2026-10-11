# Numerical and learning contracts

Cadence exposes several implementations of state, repair and learning. Choose
an API by its equations, stopping rule and update contract: sharing the word
"patch" does not make their solvers or learning guarantees interchangeable.

## Choose the computation

| API | How it computes | How it learns | Qualification |
| --- | --- | --- | --- |
| `NeuralGraph`, `Learner` | Rate neurons exchange activity over a directed graph. | Free/nudged local contrasts; optional reward traces. | `NeuralGraph.equilibrate` checks the full fixed-point equations. `settle` alone may run a fixed budget or stop on activity movement. Directed wiring does not inherit a reciprocal energy gradient theorem. |
| `Brain` | One recurrent neural graph, optionally extended by reciprocal state-reading observers, with held trace/record input. | The underlying contrast/reward learner plus explicit trace and synaptic-memory updates. | `act`, `predict` and `accuracy` require the complete equation residual; exhaustion refuses an answer. Opt-in `learning.qualified=True` also gates supervised free/nudged phases. Reward eligibility retains its finite-phase contract. |
| `PatchNet` | The reciprocal graph core with persistent free activity and explicit evidence ports. | Free and two nudged phases of the same network; commits only qualified phases. | All required phases must pass the equation residual. The gradient interpretation also needs compatible effective weights, a smooth stable branch and the small-nudge limit. |
| `TemporalPatchNet` | A causal free path and jointly repaired teaching paths over a finite time window. | Centered contrasts of parameter derivatives. | Whole-path residual and branch checks; the dense hidden-width solves have a different cost from sparse graph transport. |
| `RecordPatchNet` | A gated causal context scan with local record reads. | `observe` uses an adjoint backward scan; record writes use a local delta rule. | The causal path solves its declared free equations. `detune` separately checks quadratic continuous-output teaching phases; this is not the default training path or a categorical-port guarantee. |
| `BeliefPatch` | A transition followed by a fixed number of nonlinear repair iterations per observation. | An adjoint through the iterations and the time window; optional admitted parameter steps. | A finite repair budget is not a convergence certificate. Its `residual` records the last damped move, not a globally qualified fixed-point error. |
| `Steered`, `Life` | Compositions that sequence a cortex, steering and a governor. | The participating models' updates, with replay/admission. | Joint parameter-step admission is not a joint equilibrium of observer and observed activity. |

`RecordPatchStack`, `JointRecordPatches` and the experimental partitioned
temporal model have their own [port](record-patch.md#several-patches-joined-by-ports)
and [routing](partitioned.md) contracts. Finite port rounds or a causal routing
graph do not establish a general all-patch convergence theorem.

## Reading an answer and accepting a lesson

For a fixed-point claim, independently evaluate the declared equations at the
returned state. Check finite values and the largest relevant defect, including
slow states when the model has them. A small integration step can make activity
move very little while the equation defect remains large. A numerical residual
does not, by itself, prove uniqueness, stability or useful behavior.

Admission depends on the API. `PatchNet.observe` qualifies its free and nudged
equilibria before updating parameters; it does not replay the proposed update
to guarantee a loss decrease. APIs with an explicit replay-loss admission
check accept only steps meeting that check for the declared observation or
window. Numerical qualification, parameter commitment and measured learning
improvement are separate claims. None alone establishes retention on earlier
tasks, generalization or biological plausibility. Eligibility, reward and
source identity must remain attached to the action that was actually executed.

`Brain` uses `learning.free_steps` as the free-answer repair budget and
`learning.tolerance` for the full potential/adaptation equation residual
(defaults 1024 and `3e-3`). Default finite teaching uses
`dt=1.0` and 12 nudged steps. With that configuration, roughly half the free-answer sweep
budget is reserved for a fallback with half the integration step if the first
finite phase does not qualify. Both phases share the one requested budget, and
the final residual is recomputed against the original model. The equations,
parameters and teaching law remain unchanged; this numerical strategy is
independent of System 2. Cached activity must qualify again. Refused `act`
calls issue no action and preserve activity, memory, random state and pending
feedback. A `step` call may first learn an actual outcome and then refuse its
next action; that real learning remains, so retry `act` without resubmitting
the reward. Reward eligibility retains its finite nudged-phase contract and
separate `reward.eligibility_steps` budget (the Brain default is 12).
Independent `predict`/`accuracy` and `fit` epoch scores use qualified answers
without reading working trace or associative memory. `fit` teaching follows
the configured finite or qualified contract; a refused score does not undo
accepted lessons.

For supervised graph learning, `LearnerConfig(qualified=True, damping=3)`
requires the free and each required nudged phase to meet that phase's full
equation residual before the local contrast changes any parameter. The finite
configuration remains the default comparison. Numerical damping tries
successively halved integration steps within one declared budget per phase,
then checks the original equations. `LearningPhaseError` retains the failed
and completed phase diagnostics and attempted work; parameters, optimizer
history and update counts stay unchanged. `report` counts all phase sweeps,
including the opposite nudge, and residual transports separately. Stalled
damping attempts may yield unused sweeps to later halvings after two repeated
complete-state comparisons at backend rounding precision. The final attempt
retains its budget and every answer still needs the original residual. Reports
count these comparisons and attempted/accepted row presentations separately.
After `Brain.step`, `Brain.last_learning` preserves teaching diagnostics as `demonstration_*`
fields, including refused work alongside any earlier accepted reward. Direct
`contrast`/`update` and the `ActorCritic` eligibility path retain their own
contracts. Qualification does not establish a smooth stable branch, an exact
gradient at finite nudge, or acquired free behavior.

RMS-normalized learner and actor updates divide each parameter's signal by a
bias-corrected running RMS plus a floor before applying its learning rate.
The rate is neither a fixed increment nor a stability bound; masks, momentum
history, tying, decay and efficacy clipping also affect the installed change.
Config construction warns when normalization is enabled and either efficacy
or bias rate exceeds `0.05`, without changing the numerical rule or defaults.
The warning is advisory and excludes the critic's separate trace-energy rule.
See [normalized rates and evidence](learning.md#rates-under-normalization).

A refused `Brain.learn` feedback solve restores hippocampal records, terminal
working-trace resets and actor state, preserving the executed action awaiting
its outcome. Retry that same feedback after adjusting the solve. Once feedback
is accepted, it stays learned even if a later `act` or teacher lesson refuses;
do not submit that reward again.

`Brain.wait` settles observations that arrive before an awaited outcome without
taking it: only the activity and working trace advance, and the later outcome is
credited as an immediate one would be. An outcome reported under another
`decision_id` is refused before any change.

An adjoint is reverse-mode differentiation even when written explicitly in
NumPy without an autograd tape. Equivalence with an equilibrium contrast must
be established for the particular equations, parameters, phase limits and
loss; it is not inherited from another class's theorem.

## State and production integration

Keep live state, learned parameters, record tables, optimizer/step-size state,
random state and accounting separate. Save every component needed for an
exact continuation, including the boundary of a replay window. A custom
callable policy or weighing needs its own reproducible configuration; an
array snapshot cannot serialize arbitrary application code.

Validate shapes and finite values before mutation. Invalid calls and failed
transactions must not partially install parameters or records. Accounting of
attempted work may advance even when an update is rejected; it must never be
presented as accepted learning. An `imagine` call must not teach or alter live
activity, but an implementation can count its computational work.

`Connectome` arrays and its population mapping are read-only. `NeuralGraph.efficacy`,
`bias`, `log_gain` and effective `weights` expose read-only arrays. Replace complete
parameters through validated setters or `with_parameters`; rebuild topology and
use `with_populations` for population changes. This keeps transport caches and
CPU/device implementations in agreement. For other APIs, use their owning
parameter/update interface;
direct array access is not a portable backend update protocol. Pin the library
revision and backend for reproducible results, and run parity checks when
changing device or precision. See [backends](backends.md), [API](api.md) and
[receipts](receipts.md).

## Biological imports and recursive observation

A connectome provides structural evidence, not a complete dynamical model.
Transmitter predictions, receptor effects, temporal scales, sensory mappings,
plasticity and body interfaces require explicit data or assumptions. The
[connectome guide](connectomes.md) describes those choices. Importing all
retained neurons does not recover missing chemistry or innate behavior.

For an observer to belong to the same equilibrium as the system it observes,
its state and feedback must enter the same coupled equations and stopping
check. Readback changes during repair, and the observing patches repair in
response. A sequential controller over frozen diagnostics is a different
contract. Additional recursive depth is a testable architecture choice;
better capability or training efficiency must be measured against matched
shallow, feed-forward and recurrent controls.
The [recursive-settlement example](recursive-settlement.md) builds this coupled
mechanism with existing `PatchNet` APIs and verifies a causal interface cut.

`BeliefPatch` and `Steered` MAC counters are dense forward-work estimates, including
executed readbacks and replayed forward moments. They exclude adjoints, record
writes, nonlinearities, optimizer bookkeeping, application callbacks and memory
traffic. Governor steps reported as moment equivalents are a declared cost model,
not measured hardware work. Use complete profiling for a training-efficiency claim.

For a performance claim, compare total training work, elapsed time, peak
memory, inference cost, generalization and retention, and include failed
solves, nudged phases and admission replays in the cost.
