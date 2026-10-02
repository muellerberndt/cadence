# Private demand repair: current-state certificates and owned outcomes

These private modules are an implementation candidate, not exported Cadence
APIs. They preserve the existing patch energy, analytic derivatives, boxes and
qualification tolerance. They do not establish the three requested behavioral
requirements together: useful automatically recruited recursive correction,
retained cheap routine, and an advantage over capable ordinary populations.
Current public quickstarts and the public `Brain` implementation are unchanged.

**The owner's wake rule is semantic:** expensive slow populations should run
only for surprise or an unsuccessful longer-term outcome. Familiar input
changes are not a reason to wake them. The numerical scheduler below does not
yet meet that rule. A six-founder learned diagnostic woke the observer block on
all 24 familiar changing-input queries, despite accurate predictions; only exact
repetitions needed no new jobs. Numerical nonstationarity and actual surprise
must remain separate measurements.

## Exact cached activity

[`ActivityCache`](../src/cadence/_incremental.py) retains forward predictions,
current errors, reverse adjoints and state gradients for a fixed graph and fixed
parameters. Input/state changes invalidate the dependent expressions, including
transitive error readback and returning derivatives. Changes in a relation's
slope matter even when its residual value stays unchanged. Zero coefficients
do not remove structural dependencies.

Clean arithmetic is reused only when its dependencies are unchanged. Every
current eligible coordinate contributes to the final projected-gradient maximum;
no observer or inconvenient free state is excluded. Energy still sums all
current factors. Reuse supplies the same current mathematical values as a fresh
reference evaluation, while changing the procedure from recomputing every edge.
That procedural difference is private here and needs an explicit documented
contract before public integration. It is not permission to weaken stationarity.

[`test_incremental.py`](../tests/test_incremental.py) compares against the
independent full reference evaluator, including ordinary, nested residual and
direct recurrent graph contacts, transactional failures, forks and factual
clamps. Recurrent private graph tests do not add recurrent wiring to `Cortex`.

## Automatic independently progressing activity proposals

[`ActivitySession`](../src/cadence/_attention.py) owns a copied continuation.
`from_brain` starts no jobs. `begin` atomically installs validated inputs and
clamps, optionally a new parameter generation, then recruits work. Parameters
stay fixed within the cycle; this module performs no learning admission.
`advance` collects workers and recruits current nonstationary blocks, ranked by
their projected-gradient residual. There is no application attention flag or
per-population evaluator.

One job per block proposes projected activity updates with the original Armijo
acceptance rule. Jobs whose writes would invalidate each other's gradient reads
do not run as a competing pair; unrelated blocks can progress independently.
This avoids repeatedly invalidating a slower coupled proposal. It is a changed
block update schedule, not a claim that it follows the public simultaneous
solver's identical numerical trajectory or reaches a unique stationary point.

Each proposal owns an immutable context identity, dependency-version stamps and
private arithmetic cache. Before commitment the owner checks its provenance,
all relevant versions, current slope and current whole-energy decrease. Input,
clamp and model changes invalidate affected proposals even when values later
return to their previous values. Workers never publish or mutate the canonical
public brain themselves.

`result` exposes state only with a complete **current** certificate. `is_current`
checks that owned result against the current revision; it is not an actuator
lease. The caller must serialize its final publication/execution boundary.
`run_until_complete(seconds=...)` continues a begun cycle and returns no
publishable state on refusal, changed context or soft deadline. `begin`'s job
budget counts block submissions, not public solver sweeps. A deadline does not
kill an executing worker. `close(wait=True)` drains owned work; pending or
exceptional work remains explicit rather than counted as zero.

Worker startup is also an ownership boundary. A submitted wrapper cannot start
numerical work until the owner holds its returned future. If submission raises
after enqueueing, the wrapper is denied, queued jobs are canceled and the pool
is retired. The attempted submission and unknown infrastructure work remain
counted; that session cannot publish or accept another `begin`. Reconstruct the
owner after draining it. Tests cover both thread-start failure and a wrapper
that already started before `submit` raised.

The barrier tests in [`test_attention.py`](../tests/test_attention.py) distinguish
two cases. A newly relevant unresolved slow coordinate prevents global
publication. Conversely, an obsolete old slow job can remain paused while the
current slow coordinate is valid and independent fast work produces a new full
certificate. That fixture shares a genuinely clamped observation between the
branches; it does not clamp a slow coordinate just to hide its error. Tests also
show simultaneous independent worker starts, stale/ABA rejection, corrupted
proposals, refused line searches, budget refusal and unchanged source brains.

This is genuine independent **work progress with conditional availability of
valid blocks**. It does not let a fast population issue an action while a
currently relevant coupled slow population is unresolved. Doing that would
require a different temporal boundary and qualification contract. It is not
established by these tests or by a Python worker thread.

## Costs and learned numerical comparison

The cache reports setup/evaluation counts, edge and patch visits, dependency
visits, coefficient/input/state checks, finite checks, energy terms, certificate
checks, forks and declared copied scalar slots. These are explicit logical-work
counters, not measured physical RAM traffic. The activity owner additionally
counts its dependency traversals, coordinate scans and serialized context bytes,
plus submissions, returns, proposals, backtracks, commits, stale jobs and errors.
Final energy/certificate scans, failed trials, context changes and worker copies
are included. In-flight work makes a returned cost provisional until drained;
unreturned exceptional work is marked unknown. Python allocation, scheduling and
OS overhead require wall-time measurements. Session `elapsed_seconds` is its
whole lifetime, including caller idle time, not isolated solver latency.

The bounded learned regression compares two 12-patch ordinary/observer layouts,
each at seeds 31, 37 and 43. Each uses four real batches of eight supplied
physical-transition witnesses before four changing-input, factual-clamp queries.
Both solvers qualify all 24 queries against the reference objective. The largest
state difference is about 1.47e-6; this is numerical agreement in this fixture,
not general confluence. Deliberate zero-budget refusals leave the accepted public
brain unchanged. It is not a task-acquisition or observer-benefit experiment.

Over four queries per founder, the measured activity-cache edge visits were
2,752–2,992 for ordinary layouts versus 3,620–3,712 for public repair. Observer
activity repair used 5,360–5,744 versus 4,992–5,612: it can require **more** edge
work. Owner bookkeeping and threads add further costs. These results do not
justify a general speed claim; all positive and negative cases belong in any
reported comparison.

## Factual foreground/background integration

[`ForecastSession`](../src/cadence/_experience.py), exercised by
[`test_experience.py`](../tests/test_experience.py), owns one copied brain and
combines qualified foreground activity with one private background factual batch.
It issues an immutable free forecast, accepts the matching later measured
outcome, and retains the original context/forecast if learning refuses or changes
the model. Forecast coordinates cannot be clamped, including through aliases.
Present factual measurements may condition the solve. Accepted batch parameters
can be installed only against the matching generation, preserving newer accepted
foreground activity; the next foreground query must qualify again.

An actual forecast mismatch recruits bounded factual replay without a caller's
surprise flag. A configured terminal task deficit can also leave demand despite
accurate predictions. This is not a goal-directed action policy: scalar reward
does not become an action target, and this prototype supplies no planner.
Demand is retained until a later outcome confirms recovery of the same input and
measurement context. Merely accepting a fit or seeing an unrelated quiet context
does not clear it. Exact-context deficits are bounded; overflow remains explicit
and cannot be interpreted as successful recovery. Exact matching is a finite
fixture control, not learned attention generalization to changing Doom frames.

The integrated tests exercise actual learned scalar-prediction correction,
quiet post-recovery operation, independent fit/foreground progress, preserved
activity, factual custody, terminal deficit without surprise and refusal/overflow
handling. They establish software ownership and small predictive behavior, not
recursive necessity, motor skill, independent live-population clocks or a
measured advantage over capable ordinary brains. There is no complete owner
checkpoint API: a plain `Brain` snapshot omits pending forecasts, deficit state,
evidence selection and in-flight work.

## Private temporal records: a different coupling contract

[`RecordSession`](../src/cadence/_records.py) tests a narrower candidate for
surprise-driven slow work. It has one active graph, state, fixed parameter set,
evidence ledger and publication owner. The static fast/slow boundary is a
declared design choice. Cross-boundary state/error contacts become internal
inputs holding immutable historical values; slow input reads also come from a
committed historical record. Ordinary state and residual contacts **within**
each block remain live.

This retains the local patch relation but **changes the joint objective**.
Returning influence across the boundary is now delayed through records. It is
not the existing fully live `Cortex` graph running unchanged at two speeds.
Every active fast and slow coordinate stays free, and every answer qualifies the
complete lowered graph against its declared committed record. No unresolved slow
coordinate is hidden by a clamp or omitted from the certificate. A pending slow
proposal belongs to a possible next revision, not the active state.

Only an acknowledged miss of the owned pre-issued forecast, or an acknowledged
terminal task deficit, requests a new slow record. The body supplies live inputs,
actual outcomes and the task requirement; it supplies no attention flag or
predicted task value. An optional value coordinate is read from qualified brain
activity. Ordinary fast input changes still need numerical repair, but do not
create slow requests. This prototype has no pre-terminal goal-directed planner.
`task_value` and the required goal value are ledger/recruitment metadata, not
automatically wired solver inputs or a correction objective. The missed-need
test establishes recruitment despite an accurate forecast; it does not
establish goal-directed correction.

The bounded ledger retains issued inputs, state, exact historical patch errors,
forecast and record provenance separately from the actual ACK. Historical patch
errors are errors of their recorded lowered solve; they are neither current
original-graph errors nor the later physical outcome-minus-forecast error.
The latter drives recruitment and remains separately recorded. This candidate
does not add a dedicated learned performance-error contact.

A worker owns an immutable record proposal. Qualified foreground step/ACK cycles
can continue while it is paused. Newer ACKs alone do not stale it or disappear:
remaining demands are queued. Adoption rejects a changed base record or task
specification, then requalifies **all** coordinates at the latest fast inputs
before atomically installing the record and state. The receipt exposes both
graph identities, the model identity, record origin/age and a current-publication
check. Acknowledgment or record adoption invalidates an old publication token;
the original issued forecast remains available for its one matching ACK.

Commit success alone does not clear demand. Recovery requires a later quiet
actual outcome at the same exact live input and task specification, after a
revision incorporating that demand. A terminal deficit additionally requires a
satisfactory terminal outcome. Multiple failed contexts remain distinct within
capacity. This exact-context rule is a bounded fixture control, not learned
attention generalization. A full ledger refuses a new forecast before losing
its outcome. There is no complete owner snapshot API.

Foreground transactions use [`ActivitySession.fork`](../src/cadence/_attention.py)
to copy the existing exact cache rather than reconstruct every relation. Parent
copy work is charged; the candidate has independent counters and workers. The
tests forbid cache construction after initialization and observe actual cached
prediction evaluations during changing familiar inputs: no slow prediction is
recomputed in that fixture, while complete energy/certificate scans and copy
costs remain. Startup, proposal and commit work are retained separately and in
the total. This establishes a software boundary, not a general speed advantage.

This minimal private example uses hand-set parameters and an independently
specified deterministic body. It is **not acquired skill evidence**:

```python
import math
from cadence import _repair as R
from cadence._records import RecordSession

graph = R.Graph(2, 2, (
    ("input", 0, 0), ("state", 1, 0),
    ("input", 0, 1), ("input", 1, 1), ("residual", 0, 1),
))
owner = RecordSession(
    graph, (0,), (0.0, 0.0), (0.0, 0.0),
    (0.7, 0.2, 0.2, 0.5, 0.3), (0.0, 0.0),
    forecast_indices=(0,), outcome_inputs=(1,),
)
try:
    for stimulus in (0.2, -0.4, 0.7):
        issued = owner.step((stimulus,))
        assert issued["qualified"]
        actual = (math.tanh(0.7 * stimulus) / 1.01,)
        ack = owner.acknowledge(issued["ticket"], actual)
        assert not ack["wake"]
    receipt = owner.result()
    assert receipt["qualified"]
    assert receipt["counts"]["requests"] == 0
    assert receipt["jobs"]["foreground"][1] == 0
finally:
    owner.close()
```

[`test_records.py`](../tests/test_records.py) also checks actual forecast misses,
accurately forecast missed needs, paused-worker overlap, queued evidence after
failure, atomic refusal, goal/requirement changes, historical reconstruction,
publication invalidation and exception accounting. Parameters stay fixed:
there is no temporal-feedback learning, action credit, retained learned
correction, measured recursive advantage or integrated training here. The
separate `ForecastSession` learning tests do not establish those properties for
this new record-boundary model.

The bounded [`learned_records.py`](../examples/correction_probe/learned_records.py)
diagnostic also tests ordinary and observing 12-patch brains after four batches
of eight witnesses, across three fixed seeds. All six retained held-out accuracy
under the temporal boundary (MAE .00738–.00961, versus newborn .08946–.10717).
Each completed eight familiar and four held-out input/actual-outcome cycles
without new cache construction, slow prediction evaluation or slow jobs.
Separate surprise and terminal-goal branches each recruited slow work in all
six cases. Startup and independent full-state validation costs are recorded.

This is acquired-routine timing evidence, not learned correction: parameters
stay fixed after bootstrap, and the appended actual-outcome ports have no
learned graph contacts in this fixture. The diagnostic does not establish that
the requested slow work improves behavior, teaches a retained correction, or
outperforms an ordinary brain. Its compressed receipt retains measured state,
metrics and work, with hashes for some native payloads; it is not a full
learning-trajectory replay.

Run the focused checks from this worktree with the intended interpreter:

```sh
PYTHONPATH=src python -m pytest -q tests/test_incremental.py tests/test_attention.py tests/test_experience.py tests/test_records.py
```

Before public promotion, independently audit the complete owner and certificate
contracts, exercise failures under actual body execution, and retain all work
and outcomes. Physical correction and later task skill must be measured against
ordinary and wider ordinary controls on equal information, including routine
plus factual fitting. The all-three gate remains unpassed.
