# Preparing an application for 0.60

This page describes the **development line on main**, identified as
`0.60.0.dev0` in package metadata and `cadence.__version__`. The published
baseline remains 0.50.0; installing it does not install these changes. Use the
exact reviewed development source when executing this page's examples. A
development version is not a 0.60.0 release or evidence that its gates passed.
Final release notes must identify the selected implementation and tested revision.

Merging this development version and its documentation into `main` does not
publish stable 0.60.0. The stable release still requires learned routine →
actual disturbance → useful correction → inexpensive routine, retained skill,
and a measured behavioral contribution from recursive observation.

The candidate currently adds query-local reuse of fixed sensory predictions
and explicit ownership of executed outcomes in `Reinforcement`. Its bounded
multi-transition return experiment is still under evaluation. It does not yet
provide the intended automatic System 1/System 2 attention and outcome
integration, or establish a behavioral advantage from recursive correction.

Start with [building an effective brain](BRAIN_DESIGN.md) for the supported
construction and evaluation workflow. Width, recursive depth, retained context
and repair budget solve different problems; increasing all four is not a
general recipe for a better brain.

## Keep the application boundary small

The intended architecture makes **routine cheap and changes expensive**:
specialized fast populations maintain learned skills, while deeper general
recursive populations steer when surprise or missing long-term success calls
for correction. Both should progress at their own speeds inside one brain.
For music, equilibrium means a coherent evolving performance, not a fixed note;
for a game or robot, it means competent ongoing behavior. Successful correction
should become familiar enough to need less work when encountered again.

These are design requirements, not features enabled by `0.60.0.dev0`. The current
patch types and qualification contract below are unchanged. Independent clocks
require an explicit dependency, state-ownership and qualification design;
running two callbacks does not supply that architecture.

Applications provide observations, execute qualified actions and report actual
outcomes. The intended integrated architecture manages attention and correction
inside that boundary. Users should not have to construct two brains, provide
an attention flag or connect a separate evaluator to each population. “Internal attention” does not
mean operation without sensory observations or actual outcome evidence.

**System 1** describes inexpensive learned routine behavior. **System 2**
describes additional recursive observation and correction when routine behavior
cannot maintain equilibrium, including longer-term consequences. These names
describe roles within one brain. They are not constructor names, background
worker guarantees or automatic modes enabled by the current candidate.
Ordinary layers also settle together and can supply learned nonlinear routine
competence; System 1 is not restricted to one input-only layer. Observer
contacts add current error readback to this common rule. An unresolved observer
still participates in whole-brain qualification. The public builder supports
these previously declared dependencies, not explicit recurrent state cycles.

## Change reward acknowledgments explicitly

The released 0.50 helper associates `feedback` with its pending proposal. The
candidate requires two additional keyword arguments:

| Field | Supply |
| --- | --- |
| `decision_id` | The identity returned by the successful `act` call |
| `executed_action` | The action the environment actually executed, including any actuator override |

The body must retain this association until it acknowledges the outcome. A
proposed action, an executed action and a fitted update are separate events.
For a real environment, use its execution record rather than assuming the
proposed command was applied. The following small example uses an explicitly
constructed terminal transition to exercise the acknowledgment contract; it is
not an environment-learning result.

```python
from cadence import Cortex, Reinforcement

layout = Cortex(seed=2)
senses = layout.input("senses", shape=1)
values = layout.column("values", patches=2, inputs=senses)
for index, name in enumerate(("left", "right")):
    layout.output(name, shape=(), reads=values, indices=(index,))
learner = Reinforcement(
    layout.build(), actions=2, action_input=None,
    value_output=("left", "right"), seed=2,
)

decision = learner.act({"senses": [0.25]}, explore=False)
assert decision["accepted"]
executed_action = decision["action"]  # this fixture executes the proposal
outcome = dict(
    reward=1.0, next_inputs=None, terminal=True,
    decision_id=decision["decision_id"], executed_action=executed_action,
)
record = learner.feedback(**outcome, learn=False)
assert record["stored"] and not record["accepted"]

# A transport retry acknowledges the same event without learning twice.
before_retry = learner.snapshot()
retry = learner.feedback(**outcome, learn=False)
assert retry["duplicate"] and not retry["stored"]
assert learner.snapshot() == before_retry

restored = Reinforcement.from_snapshot(learner.snapshot())
assert restored.snapshot() == learner.snapshot()
```

Only the latest identical acknowledgment is retryable. Older, canceled,
unissued or conflicting identities are rejected. The duplicate result performs
no solve and has no `qualified` field. If a recorded outcome's subsequent fit
refuses, call `replay()` to retry learning; resending the outcome does not
authorize another update. Preserve a pending decision and actual outcome
across application restarts using the complete learner and body records.

This helper still allows one pending decision. It does not supply a queue for
arbitrarily delayed or reordered body acknowledgments. A collection timeout is
not evidence of terminal zero reward. Do not place unprocessed outcomes in
`LiveController`'s replaceable latest-observation slot. See
[the reference](REFERENCE.md#reinforcement-discrete-reward-driven-choices) for
the full contract.

## Preserve the complete experiment when upgrading

The development `step` also accepts optional `targets` and `interventions`,
using the same conditional solve as `settle` while retaining qualified activity
only. Existing unclamped calls are unchanged. This enables continuation from
explicitly constrained activity without learning parameters or recording an
experience. Clamps apply only to the current call, and a desired future clamp
must not be reported as a forecast. This addition changes `brain.py`'s source
identity; preserve older source artifacts with their snapshots as described below.

Keep the original environment and source artifact with each saved model.
Snapshots bind exact implementation hashes, including repair and reinforcement
sources. This candidate changes those sources; old snapshots are not a
supported direct migration. Editing their stored hashes does not convert them.

For each comparison, retain:

1. The original source, model, observation encoder/normalizer, action decoder,
   history, timing, environment identity and evaluation seeds.
2. A fresh candidate brain constructed through its public API. Retrain using
   the declared experience when no validated migration exists. Count retraining
   and any changed information or capacity in the comparison.
3. Subsequent unclamped predictions, actual task outcomes, old-skill retention,
   refusals and complete work/latency under both implementations.
4. A candidate snapshot/resume check using that candidate's exact source.

`credit_horizon=1` retains the one-step control. Increasing the experimental
horizon changes target construction and replay cost; it does not add missing
observations, learn temporal memory or demonstrate planning. Keep this
candidate parameter out of an application recommendation until the relevant
held-out behavior and retention comparisons support it.

## Measure the reuse change correctly

The reference query path can reuse predictions whose only sources are fixed
sensory inputs, including bias-only predictions, during a single solve. Live
states, exact errors and feedback derivatives remain current. Initial and final
evaluations remain fresh, and the full graph must still qualify. Learning and
batch admission are not accelerated by this query cache; optional tensor
execution follows its own implementation.

There is no caller switch or cache carried between observations. Separate query
latency from acquisition work and report all action-selection queries, retaining
steps and learning admissions. Zero repair sweeps still have evaluation cost.
See [performance](PERFORMANCE.md) for the measured comparison and its limits.

## Reproduce the website demos before optimizing them

The first application targets are the Atari learner, Amen and Patch World.
Their current browser engines have different origins, so a Python dependency
upgrade alone does not migrate them.

| Demo | Existing implementation | Required new-architecture comparison |
| --- | --- | --- |
| Atari learner (`atari-arcade`) | A Python server and a separately implemented browser port of the 0.50 engine | Port the selected outcome/continuation contract, regenerate numerical parity fixtures from pinned Python source, then evaluate native gameplay, learning and frozen controls on the same cases. |
| Amen | An archived 0.11 record-patch brain with its own browser engine | Train a new public-API brain on the declared event representation; compare free continuations through the same instruments and retain listening assessment. Its record cells are not a current `Cortex` checkpoint. |
| Patch World | A separate JavaScript patch-law implementation with different tolerance, small budgets and no checkpoint/event custody | Validate the selected law and refusal/continuation behavior, then compare ecology and learned behavior under matched sensing, seeds and accounting of brain work. Preserve conserved mass. |

Preserve each existing baseline and its receipts. First run its original
verifier, then the candidate's port/API checks, then a short complete native
loop. A baseline verifier passing establishes only that baseline. A page
loading establishes neither acquisition nor a benefit from recursion.
For Amen, the old "one record patch" contains 128 gated context channels and
8,192 record cells. Replacing it with one input-only output population removes
those mechanisms; it is a new restricted baseline, not a reproduction of the
old layout. Count the learned record table as well as the slow parameters when
comparing capacity. Use the same sequencing rules and audio renderer: the raw
Python evaluation instrument differs from the published website instrument.
Optimization comparisons follow a working, measured reproduction and must
include the cost of learning, sensing, monitoring, correction and action
selection. Change public demo labels and results only after the replacement
has been tested.
