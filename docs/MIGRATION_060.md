# Move an application to 0.60.0

Install the matching library with `python -m pip install "cadence-net==0.60.0"`.
This release includes the tested flat, ordinary deep, recursive and mixed-layout
API, continued learning, outcome ownership and saved continuation described in
these guides. An installed 0.50.0 package does not contain every change below.

The broader System 1/System 2 goal remains routine → disturbance → useful
correction → retained inexpensive routine. Automatic internal attention,
independently progressing fast and slow populations, and a demonstrated
behavioral advantage from recursive observation remain future work. Releasing
this API does not establish those capabilities or reproduce the older demos.

| Change | Application impact |
| --- | --- |
| Explicit `Reinforcement` execution acknowledgments | Pass the issued `decision_id` and the actual `executed_action` to `feedback`. |
| Conditional activity continuation | `step` now accepts `targets` and `interventions`, retaining qualified activity without admitting experience. |
| Bounded return construction | `credit_horizon` defaults to the one-step control. Larger values change estimated targets and replay work. |
| Zero-discount replay | Immediate-reward targets make no irrelevant next-action queries, even with a larger horizon. Nonterminal next observations still must be recorded. |
| Query-local arithmetic reuse | Eligible fixed sensory and bias-only predictions may be cached inside a solve; no caller flag or cross-call cache is introduced. |

These changes preserve the common patch equation. They do not add automatic
internal attention, independently progressing populations or a demonstrated
behavioral advantage from recursive correction.

Start with [building an effective brain](BRAIN_DESIGN.md) for the supported
construction and evaluation workflow. Width, recursive depth, retained context
and repair budget solve different problems; increasing all four is not a
general recipe for a better brain.

## Preserve the brain and body boundary

The application supplies observations, executes qualified actions and records
actual outcomes. It can build flat, ordinary deep and observing branches in one
`Cortex`. All currently use the same primitive and synchronous whole-brain
qualification. Ordinary depth already provides coupled repair; observation
adds current state-and-error contacts. The public builder accepts previously
declared sources, not explicit recurrent state cycles.

System 1 and System 2 describe learned routine and useful corrective roles,
respectively. They are not public modes, patch classes or phase switches.
A capable routine can require ordinary deep composition. The intended future
runtime should manage fast and slow work behind the same body interface,
without an application wake flag or per-population evaluator. Separate workers
or a low current residual alone do not implement that contract. See
[brain design](BRAIN_DESIGN.md) for the supported construction choices.

## Change reward acknowledgments explicitly

The released 0.50 helper associates `feedback` with its pending proposal. Version 0.60
requires two additional keyword arguments:

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

## Continue from declared constraints without teaching

The 0.60 `step` accepts optional `targets` and `interventions`,
using the same conditional solve as `settle` while retaining qualified activity
only. Existing unclamped calls are unchanged. This enables continuation from
explicitly constrained activity without learning parameters or recording an
experience. Clamps apply only to the current call, and a desired future clamp
must not be reported as a forecast. A following call must supply any continuing
constraints again; otherwise it releases them and requalifies from the retained
activity. Invalid or conflicting clamps and numerical refusal leave the entire
continuation unchanged.

Use `settle` when exploring an alternative should leave live activity unchanged.
Use `observe` or `observe_batch` only when you intend to admit labeled teaching
evidence. `step` never turns a hypothetical future into a witnessed event.

## Preserve source-bound checkpoints when upgrading

Keep the original environment and source artifact with each saved model.
Brain snapshots bind exact implementation hashes, including `brain.py` and the
repair engine; complete reinforcement snapshots also bind their helper source.
Compared with 0.50, the clamped-step and reward-replay changes alter those
identities. Restore validates the bound implementation, not just the version
number: a development snapshot can load if all its bound sources still match.
For a mismatch, retain the original installation or train a new brain; editing
the stored hashes does not migrate learned state.

For each comparison, retain:

1. The original source, model, observation encoder/normalizer, action decoder,
   history, timing, environment identity and evaluation seeds.
2. A fresh 0.60 brain constructed through its public API. Retrain using
   the declared experience when no validated migration exists. Count retraining
   and any changed information or capacity in the comparison.
3. Subsequent unclamped predictions, actual task outcomes, old-skill retention,
   refusals and complete work/latency under both implementations.
4. A 0.60 snapshot/resume check using that implementation's exact source.

`credit_horizon=1` retains the one-step control. Increasing the experimental
horizon changes target construction and replay cost; it does not add missing
observations, learn temporal memory or demonstrate planning. Keep this
experimental parameter out of an application recommendation until the relevant
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
