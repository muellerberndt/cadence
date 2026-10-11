# Runtime and mathematical specification

This document specifies the population DRSN engine. The equations are in
[the processing-patch description](ELEMENT.md); all arguments and result fields
are in [the API reference](REFERENCE.md). Every `0.81.0` brain is one connected
settlement: the builder refuses a population that settles with no other
population and a group of populations that settles apart from the rest.
Explicit observer wiring is experimental;
it obeys the same whole-brain numerical contract and participates in every
solve. No automatic observer sleep, internal attention or independent population
clocks are part of this specification. See [experimental scope](EXPERIMENTAL.md).

## Layout and state

- `Cortex` is a declaration builder. `build()` requires at least one output and
  populations joined into one connected system: every population reads another
  population's states or errors or is read by one, and no group settles apart
  from the rest. Only then does it resolve deterministic wiring and freeze
  declarations. After resolving wiring, all processing patches must also belong
  to one connected component under state/error contacts. Population connectivity
  alone is insufficient: sparse contacts can split populations into independent
  patch groups. Shared fixed inputs do not connect them. Disconnected compiled
  graphs raise rather than silently gaining extra edges. Every declared source
  coordinate is read by default. Positive `fan_in` requests sparse wiring;
  connection-budget overflow raises instead of dropping inputs. Connectivity is
  structural potential influence: zero coefficients, clamps or saturation can
  suppress the influence of a particular contact.
- `Input`, `Population` and `Output` are immutable identity handles owned by a
  layout. Their counts and shapes are validated before compilation.
- `inputs` connects samples or live population states. `observes` connects live
  states and derived prediction errors. Those constraints affect the same joint
  repair; observers do not query completed lower-layer answers.
- Public declarations can read only existing populations. Both state-reading
  and error-reading connections therefore follow declaration order. Returning
  energy derivatives couple their states without creating separately learned
  reverse connections. Recurrent state graphs supported by the private kernel
  are not a public builder feature.
- `Brain` owns state, relation weights and biases. Public state and parameter
  views are tuples; configuration is read-only; returned diagnostics are owned
  copies. Callers serialize access to each brain.
- Bounds apply to live state, weights, biases and clamps. Sensory samples need
  only be finite; applications choose normalization appropriate to their task.

## Batch experience objective

`observe_batch` groups `B` labeled input/target pairs under one shared
parameter vector `theta`. Each row has private activity `x_b`, its own fixed
sensory values and its own output clamps. All rows start from the same retained
pre-call live state. With `E_b` denoting the existing residual-plus-state-prior
energy for row `b`, batch repair minimizes

```text
F_batch = (1/B) * sum_b E_b(x_b, theta)
          + parameter_prior/2 * ||theta - theta_before_batch||²
```

The anchor is fixed for the complete solve and charged once. Shared parameters
couple the private experiences while every row's processing and observing
populations remain part of that joint problem. Rows have no implicit temporal
connections. Grouping witnessed transitions does not provide delayed credit
or learn missing sequence memory.

## Repair and qualification

The solver minimizes the stated nonlinear energy by projected analytic-gradient
steps with sufficient-decrease backtracking. Observed error derivatives include
all transitive dependencies. A solve qualifies only when the full projected
stationarity residual is at most `tolerance` over every eligible coordinate.
Output clamps are excluded; learned parameters are included during admission.

Batch state gradients in the mean objective have a factor `1/B`. Repair scales
their state-coordinate moves and qualification residuals by `B`; shared
parameter gradients remain the mean row gradient plus the single anchor
gradient. Thus the complete batch residual is

```text
max(max_b ||x_b - clip(x_b - grad_x E_b)||_infinity,
    ||theta - clip(theta - grad_theta F_batch)||_infinity)
```

Clamped coordinates are excluded in every row. This criterion prevents larger
batches from admitting individually unresolved rows merely through averaging.
The objective used for descent and reported energy remains `F_batch`.

Query solves omit derivatives of frozen parameters and exclude those coordinates
from line-search and secant calculations. State derivatives still include every
returning constraint through live state and error readback. An overflow confined
to an unused parameter derivative therefore cannot reject an otherwise valid
query. Learning and the mathematical `evaluate` function compute and validate
the full parameter derivatives. Finite energy and all computed derivatives
remain mandatory; input validation is never skipped.

Within a single-row reference query, predictions whose incoming contacts are
all fixed sensory inputs (including bias-only predictions) may be reused across
repair proposals. Their state-dependent errors, recursive consumers and all
returning derivatives are recomputed on every proposal. Cache setup occurs only
when a nonstationary state needs repair. The initial evaluation recomputes every prediction from the current inputs
and parameters. If no proposal was attempted, this fresh complete evaluation
also supplies final qualification. Otherwise final whole-graph qualification
recomputes every prediction without the cache. The cache is private to the solve and is neither retained
nor serialized. Learning and batch solves do not use it. Tensor proposal
arithmetic is unchanged; single-row Python reference refinement after a tensor
proposal can use the same optimization.

The first trial uses `step`. After acceptance, let `s` be the change in eligible
coordinates and `y` the change in their exact gradient under the same objective.
The next trial uses the scalar secant estimate `dot(s,s) / dot(s,y)` when its
curvature and result are positive and finite. Unchanged coordinates contribute
nothing. Unsafe arithmetic falls back to `step`; growth is capped at
`ldexp(step, min(backtracks - 1, 1023))`, with an overflowing cap also causing
fallback. Every trial projects onto the same boxes and requires a finite,
negative slope. Ordinary steps satisfy
`E_new <= E_old + 1e-4 * dot(gradient_old, displacement)`.
Near floating-point precision, a final proposal may instead qualify when its
full projected residual meets the requested `tolerance` and
`abs(E_new - E_old) <= 8 * ulp(E_old)`. This narrow finishing allowance avoids
rejecting a stationary proposal because rounded energy appears a few ulps
higher. It never admits an unqualified proposal or a larger energy increase.
After any attempted proposal the final stationarity check is freshly
recomputed in either case. A stationary call with no proposal uses its fresh
initial full evaluation; it never reuses a previous call's certificate.
Step adaptation resets on each solve; it is not additional learned memory.
The estimate is the first Barzilai–Borwein step from
[Two-Point Step Size Gradient Methods (1988)](https://doi.org/10.1093/imanum/8.1.141),
used here inside bounded projected repair with the finishing allowance above.

For a batch, the scaled state moves use the corresponding diagonal metric:
state-coordinate contributions to the secant numerator are `s_x²/B`, while
parameter-coordinate contributions remain `s_theta²`. The denominator and
Armijo slope use the actual mean-objective gradient. This preserves the same
objective while avoiding an artificial slowdown of each row's state repair as
the batch grows.

`prediction_residual` is a different quantity: the largest absolute local
prediction error. Priors, bounds and competing constraints can leave this
nonzero at a qualified point. Qualification establishes constrained numerical
stationarity to the declared tolerance. It establishes neither a unique normal
form nor a global minimum, and says nothing by itself about task accuracy.

`budget` limits accepted sweeps, with at most `backtracks` attempted steps per
sweep. Already stationary proposals may qualify with budget zero. Exhaustion
or inability to find a descent step refuses the proposal. Invalid arguments
raise `ValueError`. Numerical overflow may refuse a proposal or raise
`ValueError`; neither outcome commits continuation. Applications must not act
on diagnostic outputs from a refused solve.

## Optional tensor execution

`device="python"` selects the standard-library float64 reference solver.
Optional `"cpu"`, `"mps"` and `"cuda:N"` execution computes the same patch energy
and analytic derivatives with PyTorch tensors; `"cuda"` selects `"cuda:0"`.
MPS uses float32. CPU/CUDA default to float64 and also accept float32.
No autograd optimizer or separate learning rule is introduced. Tensor code
parallelizes arithmetic within residual-dependency levels and retains returning
influence through every level. Batch execution also vectorizes over private
row states, sharing one parameter vector. Reference qualification checks the
complete batch, including every unaveraged row-state residual. There is no
distributed multi-GPU solve or checkpoint-averaging step.

Device arithmetic is a proposal mechanism. Final diagnostics and admission
are recomputed by the reference engine with the original sensory values,
exact output/intervention clamps, original pre-experience parameter anchors,
and original frozen query parameters. The requested tolerance is never
relaxed to accommodate float32. Reference refinement, or restart after an
unacceptable float64 energy increase, consumes only the remaining total
accepted-sweep budget. With a positive budget, float32 device repair is capped
at `max(1, budget // 2)` accepted sweeps; budgets of two or more retain at least
half for reference refinement. Float64 may use the full allowance. The device
stopping hint is `max(tolerance, 64 * dtype_epsilon)`, while final qualification
always uses the original `tolerance`. A refused solve still retains nothing.

The device may take different steps or reach a different stationary point.
Its `energy_history` includes approximate device arithmetic and any subsequent
reference refinement, rather than a float64 proof of decrease at each device
step. The `execution` diagnostic identifies device, precision and reference
work. Precision and hardware comparisons must measure behavioral accuracy,
refusals and full cost, not merely device kernel timing.

Tensor dependencies and device availability are checked lazily on first solve.
Missing PyTorch raises `ImportError`; an unavailable selected device raises
`ValueError`. Neither condition silently selects another device. Reference
qualification/refinement is an explicit part of tensor execution, not an
unreported hardware substitution. Independent brains may run in separate
processes; admissions to any one brain remain serial.

## Continuation and admission

The **bootstrapping phase** prepares and checks a brain using representative
experience. The **live phase** uses the resulting persistent brain and may
continue admitting actual witnesses. These are application lifecycle terms,
not runtime modes: there is no automatic phase toggle or phase-wide parameter
freeze. The same repair law and the following per-call contracts apply in both.

| Operation | Live state | Retained parameters | External event record |
| --- | --- | --- | --- |
| `settle` / `predict` | Unchanged | Unchanged | Unchanged |
| `settle` with hypothetical clamps | Unchanged | Unchanged | Unchanged |
| Qualified `step`, with or without clamps | Commit complete solved state | Unchanged | Unchanged |
| Qualified `observe` | Commit solved state | Commit solved relations | Advance once |
| Qualified `observe_batch` | Unchanged | Commit shared solved relations | Advance once for the batch |
| Refused operation | Unchanged | Unchanged | Unchanged |

`step` accepts the same output targets and population interventions as `settle`.
It runs that frozen-parameter solve and retains its complete activity only when
all eligible free states qualify. Retained clamped values supply a starting
state for later calls; clamps are not retained as constraints. Every subsequent
call qualifies under its own inputs and explicitly supplied clamps. This does
not admit teaching evidence, change parameters or consume an event identifier.
An all-clamped state may qualify without repair because no state coordinate is
eligible; this establishes neither inference nor learning. A goal-clamped
output is not a free forecast. `settle` remains pure, including with clamps.

`observe` requires at least one output target. It fixes these values throughout
joint state/parameter repair and anchors parameters to their pre-experience
values. The entire proposal qualifies before admission. There are no partial
parameter commits and no event ID consumption on refusal.

`observe_batch` requires a finite nonempty sequence of such pairs, validated
before solving. A successful batch commits only shared parameters and event
ownership. The row states are returned, not retained: selecting the last or mean
experience as the present live state would be arbitrary. A one-row batch has
the same parameter solve as `observe` but still preserves live state. A batch
can differ from serial admissions because serial calls reanchor after each row.

Explicit nonnegative event IDs are monotonically increasing. Retrying the latest
admitted ID with identical sensory samples and physical clamps is idempotent;
it returns `duplicate=True` without solving or committing. Changed or older IDs
are rejected. This is latest-event deduplication, not a history of all events.
Omitting an ID allocates the next identity on acceptance, so applications that
need retry safety must retain their external IDs.

Both learning methods use the same ordered event sequence. Batch identity
includes its ordered input/clamp rows; reordering an otherwise identical batch
under the same event ID is a conflict. Checkpoints retain the resulting
parameters and admission identity, not the batch's private states or a pending
minibatch cursor.

Both methods accept `source="witness"` for actual observations or
`source="estimate"` for derived teaching targets. The caller declares provenance;
the engine does not authenticate it. The source label affects event identity
and appears in accepted, refused and duplicate results. Changing the label
under an admitted ID conflicts. The label changes neither the objective nor
qualification. All rows of one batch share a label.

## Learning and runtime orchestration

`Reinforcement` stores actual `(context, action, reward, next_context)` records
and constructs discrete-action Q estimates. With discount `gamma`, value scale
`v` and reward scale `r`, its teaching target is

```text
target = (1-gamma) * v * reward/r
         + gamma * clip(max_a Q(next_context, a), -v, v)
```

Terminal transitions omit the second term. At `gamma=0`, nonterminal targets
also omit all future-value work and use exactly one immediate reward, regardless
of `credit_horizon`. Next observations remain validated and retained evidence;
an irrelevant future query cannot refuse this update. Rewards must lie in
`[-r,r]`; `gamma` lies in `[0,1)`. This is a scaled discounted-return target, not a
measured future outcome. Candidate and next-action values require qualified
queries. Replay uniformly samples a bounded store while always including the
latest record; all targets use the same pre-update brain. One
`observe_batch(source="estimate")` fits them with ordinary joint repair.
There is no second optimizer or convergence guarantee for this nonlinear
Q approximation. Valid feedback records remain stored and consume the pending
decision even if fitting later refuses; retry fitting with `replay()`.

An accepted decision issues a monotonically increasing identifier. Feedback
must name that identifier and the action actually executed; the recorded action
may differ from the proposal. Validation precedes any mutation. The record and
latest outcome receipt commit before replay starts, so a refused or exceptional
fit cannot lose the outcome. The same latest identifier and normalized outcome
payload acknowledge a retry without changing records, RNG, parameters, live
state or a newer pending decision. A conflicting payload or any other identifier
without matching pending ownership is rejected. Reset discards pending ownership
without reusing identifiers. The latest receipt is bounded independently of replay
eviction. These receipts are caller-supplied execution evidence, not authenticated
measurements; they are distinct from the brain's parameter-admission events.

The experimental bounded-credit option extends the same estimated targets
to at most `credit_horizon` adjacent records. It crosses a record boundary only
when the observed next context equals the next record's context, their episode
identifiers agree, and the next recorded action attains the current pre-update
maximum action value. All queries in the return and batch use unchanged
parameters and the same retained activity. Each traversed reward is scaled by
`(1-gamma)*v/r`; rewards are discounted along the actual recorded path. A
terminal tail has zero bootstrap; a missing, reset, off-policy or horizon-cut
tail uses the clipped next value. A time limit is not automatically terminal.

Starting with a bootstrap in `[-v,v]`, each reverse return step is the convex
combination `(1-gamma)*v*reward/r + gamma*tail`, so induction bounds every target
by `[-v,v]` in real arithmetic. This target bound does not prove that the
nonlinear learner converges or improves behavior. Numerical qualification
still covers every private batch state and shared parameter. A refused query
or fit preserves learner parameters and replay RNG; actual already-recorded
outcomes remain retained. Source-bound `reinforcement/3` snapshots preserve
credit horizon, episode/decision ordering, outstanding proposal ownership and
the latest acknowledged outcome for exact retry continuation.

`History` retains a caller-fed window of samples and slot-validity masks.
It exposes temporal context but does not learn recurrent memory. Relation
parameters remain plastic; neither the proximal anchor nor replay guarantees
protected consolidation. `LearningProgress` scores positive decreases in
per-context moving predictor error. Noise can also cause a decrease; this
score is neither information gain nor solver stationarity.

`LiveController` gives a callback one serial worker with one replaceable pending
sensory sample. Its command expires from submission time, including queue and
solve latency. It does not cancel an active solve or guarantee a wall-clock
deadline. Reward/transition records must use lossless application storage,
not that replaceable sensory slot. `slew` limits actuator change toward a
caller-selected target. Neither helper selects a behavioral objective or
changes patch repair. See [live operation](LIVE.md) for ownership and timing.

## Checkpoint contract

Snapshots contain the complete layout, configuration, arrays, admission cursor
and digest. JSON must have exactly the expected fields, finite numeric values,
consistent bounds, counts and identity. Duplicate JSON fields are invalid.
Size is limited to 32 MiB. Reconstructed connections and dimensions are checked
before installing a proposed continuation.

A layout/configuration fingerprint and exact hashes of `brain.py`, `cortex.py`,
`column.py`, `ports.py`, `_repair.py`, `_tensor.py` and `_validation.py` bind
compatibility.
Previous source sets and different hashes are rejected, including across
releases. `restore` only installs a validated continuation for the same
graph/configuration; `from_snapshot` constructs one. Optional `device`/`dtype`
overrides on `from_snapshot` apply only after the saved snapshot has passed
its original complete validation. They preserve arrays and admission identity
but update resolved execution configuration and its fingerprint. Supplying a
new device with no dtype chooses that device's default. `restore` has no
overrides and requires an exact configuration match. Source identity and digest
checks are integrity checks, not cryptographic authentication of a witness or
proof of its truth. Treat caller-provided files as bounded data, never executable
code.

## Evidence boundary

Tests check numerical derivatives independently against finite differences,
analytic optima on small cases, causal feedback into observed populations,
energy descent, constrained boundaries, source coverage, witnessed acquisition,
unclamped recall, refusal rollback, event custody and continuation. Executable
documentation uses this checkout's public API. Unimplemented capabilities and
the limits of recursive recipes are identified in
[experimental scope](EXPERIMENTAL.md).

The engine supplies labeled-target learning and persistent joint activity;
the reinforcement helper adds explicit discrete action-value credit and replay,
including the bounded return described above.
These do not establish autonomous task discovery, general long-horizon credit,
learned structural growth or a biological physiology model. Performance and
advantages from recursive depth remain empirical questions. Biological
inspiration is not evidence that a numerical qualification reproduces a brain.
