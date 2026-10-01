# API reference

Cadence constructs Deep Recursive Settlement Networks from populations of
processing patches. Ordinary data connections and recursive observation share
one jointly repaired state. Public exports are `Cortex`, `Brain`, `Input`,
`Population`, `Output`, `SettlementError`, `bootstrap`, `History`,
`LearningProgress`, `Reinforcement`, `LiveController`, `slew` and `__version__`.

```python
from cadence import (
    Cortex, Brain, Input, Population, Output, SettlementError, bootstrap,
    History, LearningProgress, Reinforcement, LiveController, slew, __version__,
)
```

Use the package-level imports above. `Brain` and `SettlementError` live in
`brain.py`, `Cortex` in `cortex.py`, `Population` in `column.py`, and `Input` and
`Output` with boundary shape/value validation in `ports.py`. Numerical repair
and numeric/JSON validation remain private in `_repair.py` and `_validation.py`.
Private `_tensor.py` supplies optional device execution of the same analytic
repair law, with final qualification by the float64 reference engine.
`bootstrap.py` orchestrates example replay and unclamped checks through the
existing brain methods; it adds no solver or phase state. `memory.py` holds
explicit history and error-progress bookkeeping; `reinforcement.py` supplies
discrete-action Q-learning orchestration; `runtime.py` supplies serial live
scheduling and actuator rate limits. These helpers preserve the patch equation.
See the [quickstart](QUICKSTART.md) for a first learned relation, the
[layout quickstarts](VARIANTS.md) for flat/deep/recursive construction, and the
[architecture guide](DRSN.md) for equations. This development reference includes
the unreleased candidate's outcome-ownership interface; consult
[migration](MIGRATION_060.md) when using an installed release.

## Cortex: declare a layout

```text
Cortex(
    *,
    seed=0,
    fan_in=None,
    initial_scale=0.3,
    settle_budget=2048,
    tolerance=1e-6,
    state_prior=0.01,
    parameter_prior=0.1,
    state_bound=1.0,
    parameter_bound=4.0,
    step=1.0,
    backtracks=32,
    max_patches=10000,
    max_connections=1000000,
    max_inputs=1000000,
    device="python",
    dtype=None,
)
```

All arguments are keyword-only. Integer parameters reject booleans; `fan_in`
also accepts `None` for full connectivity. Real parameters must be finite and
strictly positive. Defaults are starting points
for small numerical and learning examples, not universal task settings.

| Parameter | Contract |
| --- | --- |
| `seed` | Nonnegative integer controlling sampled wiring and initial relation coefficients. Reproduction also requires the same layout, configuration and implementation. |
| `fan_in` | `None` (default) connects every coordinate of each declared source to every destination patch. A positive integer requests sparse sampling: a minimum count per destination, capped at source size and raised to cover the complete source across the population. Applies separately to each source and signal kind. |
| `initial_scale` | Positive initial weight scale, no greater than `parameter_bound`. Each weight is sampled uniformly in `[-initial_scale, initial_scale]` and divided by the square root of its target's total incoming connection count. Biases and live state start at zero. |
| `settle_budget` | Nonnegative maximum accepted repair sweeps per solve. Zero can qualify an already stationary state. A sweep may evaluate several rejected proposals. |
| `tolerance` | Positive maximum complete projected-gradient residual for qualification. An absolute numerical threshold, not prediction accuracy. |
| `state_prior` | Positive coefficient of the quadratic activity penalty. Changes preferred states and returning influence in free layered queries, not merely solver speed. Zero is unsupported. |
| `parameter_prior` | Positive coefficient anchoring weights and biases to their pre-call values during `observe` or `observe_batch`. The anchor stays fixed throughout the solve; a batch applies this penalty once to shared parameters. |
| `state_bound` | Positive absolute bound on processing-patch states and output/intervention clamps. Sensor values are not clipped to it. |
| `parameter_bound` | Positive absolute bound on weights and biases; repair projects eligible parameter coordinates into this box. |
| `step` | Positive initial and fallback projected-gradient step. Subsequent trials estimate a scalar step from the previous accepted displacement and gradient change. Unsafe estimates fall back to this value; growth is bounded by the available backtracking budget. |
| `backtracks` | Positive maximum line-search proposals per sweep. Rejection halves the proposed step. |
| `max_patches` | Positive integer cap on processing patches across all populations. Sensor samples and output aliases do not consume this count. |
| `max_connections` | Positive integer cap on compiled directed input, state and error-readback connections. |
| `max_inputs` | Positive integer cap on scalar samples across all sensors. |
| `device` | `"python"` (default) uses the standard-library reference engine. `"cpu"`, `"mps"`, `"cuda"` and `"cuda:N"` select optional PyTorch tensor execution. `"cuda"` resolves to `"cuda:0"`; `N` is a nonnegative device index. Availability is checked on the first solve. |
| `dtype` | `None` resolves to `"float32"` for `"mps"`, otherwise `"float64"`. Explicit `"float32"` or `"float64"` selects tensor proposal precision. `"python"` requires `"float64"`; `"mps"` requires `"float32"`. Final qualification always uses Python float64 with the original data. |

`cortex.config` is a read-only mapping of resolved settings. Construction caps
count graph elements; they do not bound actual RAM or runtime. Larger inputs
increase coverage cost even when `fan_in` is small. Tensor execution requires
`pip install "cadence-net[gpu]"`. Declaring, building or loading a brain does
not import PyTorch or allocate GPU resources. The first solve raises
`ImportError` if PyTorch is missing or `ValueError` for an unavailable device;
it never silently selects a different device. Device/precision affect numerical
trajectories and cost, not the declared patch law. See [acceleration](ACCELERATION.md).

### Layout methods

| Method | Return and behavior |
| --- | --- |
| `input(name, *, shape)` | An `Input` boundary whose supplied values remain fixed for the complete solve. |
| `column(name=None, *, patches, inputs=())` | A `Population` containing exactly `patches` processing patches, a positive integer. `inputs` accepts existing sensor or population references. |
| `observer(name=None, *, patches, inputs=(), observes)` | The same patch primitive with at least one observed population. Reads its live states and exactly recomputed prediction errors; may also receive ordinary `inputs`. |
| `output(name, *, shape, reads, indices=None)` | An `Output` exposing selected coordinates of one population, with no separate output network. |
| `build()` | A `Brain` with resolved sparse wiring. Requires at least one population and one output. A successful build freezes the layout; further construction or another build raises `ValueError`. |

Names are unique across node types, valid UTF-8 strings of 1–256 characters.
`None` generates a name using the node type and an unused positive integer.
References must be the actual handles returned by this `Cortex`; a matching
name or a handle from another layout does not establish ownership.

`inputs` and `observes` accept one handle or an iterable of distinct handles.
Sources must already exist. `inputs` reads sensor values or population states;
`observes` accepts populations only and additionally reads their errors.
Declaring a population in both fields does not duplicate its state connection.
Empty-input columns are permitted for internal-state controls.
Both kinds of population contacts follow declaration order. All eligible
states are solved jointly: ordinary state contacts already return influence
through the energy derivatives, and observation adds an exact current-error
channel. Neither construction supplies a separately scheduled critic,
external attention flag or learned recurrent state cycle.

`shape` accepts a positive integer or a tuple/list of at most eight positive
integer dimensions. `shape=()` denotes one scalar. Shape describes data layout,
not learned interpretation. Output size cannot exceed `reads.patches`.
`indices` selects distinct zero-based local coordinates from `reads`, in output
order, with length equal to output size. The default is `range(size)`.
Different outputs may alias the same patch.
Two scalar outputs reading the same population both default to index zero.
Use explicit `indices=(0,)` and `indices=(1,)` for independent named controls,
or one vector output with `shape=2`.

With default full connectivity, each destination patch reads every coordinate
of every source it declares. Explicit sparse wiring guarantees only aggregate
coverage across the population. A selected output can miss information read by
other, unconnected patches. A sensor with no connections remains unobserved.
`inspect()` reports actual edges, aggregate coverage and structural sensor
coverage for each output coordinate.
Normalization, features and motor interpretation are supplied by the application.

### Immutable layout handles

Obtain these objects from the builder instead of constructing them directly.

| Class | Public fields and properties |
| --- | --- |
| `Input` | `name`, `shape`; `size` counts scalar samples. |
| `Population` | `name`, `patches`, `inputs`, `observes`; `role` is `"observer"` when `observes` is nonempty, otherwise `"processing"`. |
| `Output` | `name`, `shape`, `reads`, `indices`. |

`role` describes wiring. Useful self-observation still needs causal tests;
assigning a name does not demonstrate metacognitive ability. Population width
is `patches`; observation nesting is defined by `observes`.

## Brain: query, continue and learn

Construct with `Cortex.build()` or `Brain.from_snapshot`; the `Brain`
constructor itself is an internal compilation interface. Use one serial owner
per brain. Methods do not provide thread synchronization. Admission and restore
are all-or-nothing under serial access, not concurrent database transactions.

| Method | Contract |
| --- | --- |
| `settle(inputs, *, targets=None, interventions=None, budget=None)` | Pure query returning a complete solve result. Optional output targets and population interventions are hypothetical clamps. Changes neither retained live state nor parameters. |
| `predict(inputs, *, budget=None)` | Pure query returning an output-name-to-flat-tuple mapping. Raises `SettlementError` if the solve does not qualify. |
| `step(inputs, *, targets=None, interventions=None, budget=None)` | Repair activity with parameters frozen, optionally conditioning on the same clamps as `settle`. Retain the complete proposed state only if qualified; parameters and event history stay unchanged. Returns the full result plus `accepted`. |
| `observe(inputs, targets, *, event_id=None, budget=None, source="witness")` | Jointly repair state, weights and biases under at least one labeled output target. A qualified solve atomically retains state, parameters and event ownership; a refusal retains none of the proposal. |
| `observe_batch(examples, *, event_id=None, budget=None, source="witness")` | Jointly repair private experience states and shared parameters under a batch of labeled targets. A qualified solve atomically retains parameters and one event identity, preserving the pre-call live state. A refusal commits nothing. |
| `inspect()` | Owned layout description, resolved graph counts, topology, observation roles and continuation metadata. |
| `snapshot()` | Complete continuation as a JSON string, limited to 32 MiB of UTF-8 text. |
| `Brain.from_snapshot(text, *, device=None, dtype=None)` | Class method validating the complete original snapshot before constructing a new brain. Optional execution overrides retain arrays and witness identity while changing configuration/fingerprint. See [checkpoints](#checkpoints). |
| `restore(text)` | Validate before replacing this brain's continuation. Layout and configuration fingerprint must match. Returns `None`. |

`budget` overrides `settle_budget` for an attempted solve and must be a
nonnegative integer; `None` uses the configured budget. An identical latest-event
retry performs no solve and ignores `budget`. Queries begin at currently retained
live state. They are read-only, not automatic resets.

### Values, clamps and events

`inputs` maps every declared sensor exactly once, using names or that brain's
original `Input` handles. Values may be correctly nested lists/tuples or flat
lists/tuples with the declared element count. Array objects with `tolist()` are
accepted without a NumPy dependency; any exposed shape must match the declared
shape or its flat vector shape. Scalar boundaries receive scalar values.
All samples must be finite real numbers; booleans are rejected. No automatic
clipping or image/audio preprocessing is applied.

`targets` maps selected output names or owned `Output` handles to values of the
output shape. `interventions` maps selected population names or owned
`Population` handles to vectors of `population.patches` values. Clamps must lie
inside `state_bound`. Aliased output/intervention clamps must agree exactly on
shared patches or the call raises `ValueError`. Returned output values are
always flat tuples, including multidimensional outputs.

In `settle` and `step`, clamps condition the solve and never become learning
admissions. `step` retains the qualified state, including clamped coordinates,
so a following query starts from that activity. The clamps themselves apply
only to that call; the next call must supply any continuing constraints again.
Qualification covers every eligible free coordinate under the declared clamps.
A goal-clamped future output equals an intention, not a prediction that the
goal will occur; leave that output free when evaluating a forecast. Invalid
arguments or a refused solve preserve the entire continuation.

`observe` requires a nonempty target mapping and has no intervention
argument. `source="witness"` labels actual observed targets;
`source="estimate"` labels derived teaching targets, including Q estimates.
These are the only accepted labels. Both use identical repair; the label is
caller-declared provenance, not authentication. It is included in every
admission result and the event identity. A batch uses one label for all rows.
Outputs returned by learning equal their supplied clamps, so measure learning
using subsequent predictions without those clamps.

`event_id` is an ordered nonnegative integer that must be serializable by the
runtime's JSON integer encoder. Excessive digit counts raise `ValueError`
before solving or admitting experience. Omission selects the latest
admitted identifier plus one; initially this is `0`. Gaps are allowed. The
latest accepted identifier may be retried with identical inputs, clamps and source,
performing no solve or second admission. Reusing it with different content, or
supplying an older identifier, raises `ValueError`. Refused proposals consume
no inferred identifier and retain no event content.

### Batch experience

`observe_batch` accepts a nonempty finite sequence of `(inputs, targets)` pairs.
Each pair follows the same boundary, shape and nonempty-target rules as
`observe`. Generators are not accepted. All rows are validated before solving;
an invalid row leaves continuation unchanged. Different rows may clamp
different outputs or supply different values for the same output: they describe
separate experiences, not conflicting clamps on one state.

A batch of size `B` has `B` private patch-state vectors, each initialized from
the **same pre-call live state**. Weights and biases are shared. Its objective
is the mean of the existing per-row residual-plus-state-prior energies, plus
one parameter-prior penalty anchored to the pre-batch parameters. Every row's
observed and observing patches participate in the joint solve. The rows
interact through shared parameters, not through temporal connections.

Qualification tests each row's **unaveraged** state gradient and the shared
mean parameter gradient plus its anchor term. Increasing batch size does not
divide a row's qualification tolerance. A qualified batch commits its shared
parameters and one admission, leaving `brain.state` unchanged. Its private
solved states remain diagnostics; no last row or average row becomes the
present live state. Even a one-row batch preserves live state, although its
parameter solve matches `observe` from the same starting continuation and
execution settings.

One ordered batch owns one `event_id`, in the same event sequence as `observe`.
Its digest includes every row's sensory inputs and physical clamps, in order,
and distinguishes witness targets from estimates.
An identical latest batch retry performs no solve or second admission. Changing
row order, values, source or batch membership under that ID conflicts. Individual and
batch calls have distinct event payloads: retry through the same method,
including for a one-row batch. Serial calls and one joint batch generally
produce different parameters: serial learning
reanchors after each admission; a batch has one fixed anchor. Batch grouping
is a learning choice, not an execution-only optimization.
Choose a bounded batch size: graph construction caps do not cap the extra
private row-state and temporary arithmetic storage needed by a batch.

### Read-only properties

| Property | Value |
| --- | --- |
| `config` | Read-only resolved configuration mapping. |
| `state` | Tuple of current patch values, concatenated in population declaration order. |
| `weights` | Tuple of retained coefficients in `graph.edges` order. |
| `biases` | Tuple of retained prediction offsets in patch order. |
| `graph` | Immutable topology with `n_inputs`, `n_patches`, `edges`, `incoming` and `residual_order`. |

An edge is `(kind, source, target)`, where `kind` is `"input"`, `"state"` or
`"residual"`. Input sources index the flattened sensor vector; other sources
and all targets index the combined patch vector. `incoming` contains edge-index
tuples per target. `residual_order` orders recomputation of derived errors.
Edges expose read dependencies. Feedback uses derivatives of the same energy,
not a separately stored reverse edge.

### Solve results

`settle`, `step` and newly attempted `observe` calls return a dictionary:

| Key | Meaning |
| --- | --- |
| `outputs` | Names mapped to selected state tuples. Refused outputs are diagnostic proposals and must not drive actions. |
| `state`, `weights`, `biases` | Proposed final coordinate tuples. A pure query leaves the brain's properties unchanged. |
| `predictions` | Exactly recomputed `tanh` predictions, one per patch. |
| `errors` | Exact `state - prediction` values, one per patch. |
| `energy` | Final objective, including the fixed parameter anchor penalty during learning. |
| `stationarity` | Maximum absolute projected-gradient component over eligible coordinates. |
| `prediction_residual` | Largest absolute prediction error; may remain nonzero at qualified stationarity. |
| `qualified` | Whether a fresh final evaluation satisfies the stationarity tolerance. |
| `reason` | `"qualified"`, `"budget"` or `"line_search"`. |
| `sweeps` | Accepted repair sweeps. |
| `energy_history` | With the Python engine, initial energy followed by each accepted proposal's energy. Tensor execution records its approximate device trajectory and any reference refinement; do not treat that combined trace as a float64 monotonicity certificate. |
| `work` | Algorithmic work counts, including attempted work on refused solves. Tensor and reference qualification/refinement work are both counted. |
| `execution` | Present for tensor solves: selected device/precision, tensor library version and work split; see below. |

`step` adds `accepted`, equal to `qualified`. A newly attempted `observe` adds
`accepted`, `duplicate=False`, `event_id` and `source`. An identical latest-event
retry returns only `accepted=False`, `qualified=True`, `duplicate=True`,
`event_id` and `source`.
Its qualification refers to the prior admission; it is not a fresh prediction
or qualification of live state after intervening calls.

A newly attempted `observe_batch` returns the same common diagnostics and
admission fields, with `batch_size` and these per-example fields:

| Key | Batch meaning |
| --- | --- |
| `states` | Tuple of proposed patch-state tuples, in example order. There is no singular `state` field. These states are not committed to `brain.state`. |
| `predictions`, `errors` | Tuples of per-row patch vectors, in example order. |
| `outputs` | Tuple of output-name-to-state-tuple mappings, in example order. Supplied targets are clamped, so these are not acquisition scores. |
| `weights`, `biases` | One shared proposed parameter vector of each kind. |
| `energy` | Mean row energy plus one shared parameter-anchor penalty. |
| `stationarity` | Maximum of all unaveraged row-state projected residuals and the shared parameter projected residual. |
| `prediction_residual` | Maximum absolute patch prediction error across every row. |
| `batch_size` | Number of examples in this atomic admission. |

An identical latest batch retry returns `accepted=False`, `qualified=True`,
`duplicate=True`, `event_id`, `source` and `batch_size`, without new solve diagnostics.
Returned batch arrays belong to the result, not the retained continuation.

`work` contains `evaluations`, `patch_visits`, `edge_visits`, `proposals` and
`backtracks`. Evaluations count attempted energy/gradient computations,
including the final check. Visits count prediction/error and derivative
traversals, including partial work before numeric failure. Proposals count
line-search attempts; backtracks count rejected proposals, including the last
rejection when line search fails. These are not CPU instruction counts or
complete memory/latency measurements.
Query-local reuse of invariant sensory predictions reduces the forward edge
visits actually performed, while live error and reverse derivative traversals
remain counted. Cache construction and copying are not edge visits; measure
complete elapsed time separately.

Tensor `execution` contains `device`, `dtype`, `torch` (library version),
`tensor_sweeps`, `reference_sweeps`, `reference_evaluations` and
`reference_restart`. The last flag reports whether an unacceptable float64
energy increase discarded the device candidate and restarted from the original
coordinates. Reference checking may refine a device candidate, using only the
remaining total sweep allowance. For a positive budget, float32 device repair
uses at most `max(1, budget // 2)` accepted sweeps, reserving at least half of
budgets of two or more for reference refinement. Float64 may use the full
allowance. The device stopping hint `max(tolerance, 64 * dtype_epsilon)` never
relaxes final admission tolerance. `sweeps` includes both stages. Returned
`energy`, `predictions`, `errors`, `stationarity` and `qualified` come from the
reference check using original inputs, exact clamps, original learning anchors
and original frozen query parameters. Device proposals may follow a different
trajectory; matching device results bit for bit is not promised.

### Inspector result

`inspect()` returns layout `inputs`, `populations` and `outputs`, plus `config`,
`patches`, `input_samples`, `connections`, `edges`, `observed_fields`,
`sensor_coverage`, `output_connected_patches`, `fingerprint`, `implementation`, `admissions` and
`last_event_id`. Population records add `role` and global patch `indices`.
`observed_fields` is `("state", "prediction_error")`. `sensor_coverage` counts
unique input coordinates attached anywhere; it does not certify their influence
on a selected output. Each output record adds `sensor_coverage_by_coordinate`,
a flat tuple of distinct reachable sensor-coordinate counts in output order.
`output_connected_patches` counts processing patches in any output's coupled
component. State and residual connections join components in both directions;
shared fixed inputs do not join otherwise independent patches. These are
structural possibilities, not guarantees of causal influence: zero weights,
saturation or learned cancellation can suppress a path. `last_event_id` is `-1`
before any admission. Editing inspector copies does not modify the brain.

## Numerical and learning contract

Each patch computes `p = tanh(b + sum(weight * signal))` and error `e = x - p`.
Query energy is `sum(e²)/2 + state_prior * sum(x²)/2`. Learning adds
`parameter_prior * sum((parameter - pre_experience_parameter)²)/2`, making
weights and biases eligible alongside unclamped live states.
For `observe_batch`, the residual and state-prior energy is averaged over
private row states; the parameter penalty is applied once. Its state projection
diagnostic uses the gradient of each row's energy before averaging, while the
parameter diagnostic uses the gradient of the complete batch objective.

Repair is synchronized projected-gradient descent with a scalar secant step
and Armijo backtracking. A fully stationary final proposal may finish within
eight energy ulps when rounding prevents sufficient decrease; the requested
stationarity tolerance is unchanged. The energy and analytic gradient do
not change when the step adapts. See the specification for the update formula.
The residual is `z - clip(z - gradient, -bound, bound)` for each eligible
coordinate. Clamped states and frozen query parameters are excluded.
Derivatives include transitive error-readback influence. Observers and observed
populations participate in the same objective and final qualification, not a
sequence of independently settled answers.

Qualification establishes bounded projected stationarity, not zero disagreement,
a unique normal form, a global optimum or asynchronous confluence. Nonconvex
energy can retain different stationary points. Budget or line-search exhaustion
is a numerical refusal. Invalid arguments and nonrepresentable initial numeric
quantities raise `ValueError`. `SettlementError` is a `RuntimeError` used by
`predict` for a valid but unqualified solve. Its message includes the refusal
reason, stationarity, tolerance and sweep count; use `settle` for full diagnostics.

The learning primitive fits labeled target constraints. `Reinforcement` adds
one-step discrete-action Q targets and bounded replay through that primitive;
it is not an additional parameter updater. Automatic episodic retrieval,
protected consolidation, planning policies and learned structural growth are
not provided by `observe`. Useful perception, behavior and benefits from
observation depth require separate application tests.

## bootstrap

```text
bootstrap(
    brain,
    examples,
    *,
    checks,
    max_error,
    epochs=20,
    seed=0,
    budget=None,
    batch_size=1,
)
```

Convenience orchestration for the **bootstrapping phase**, returning a plain
report and admitting examples to the supplied `Brain`. The **live phase** uses
that same brain, with continued `observe` calls when actual witnesses arrive.
These are application phases; no solver mode changes at the boundary.

| Argument | Contract |
| --- | --- |
| `brain` | Existing `Brain` to bootstrap. Earlier admitted experience is preserved. |
| `examples` | Nonempty finite sequence of `(inputs, targets)` pairs, with the same named/owned-handle mappings and shapes as `observe`. Each pair needs at least one target. Rows may repeat; generators are not accepted. |
| `checks` | Nonempty sequence in the same format, used only for unclamped evaluation. These cases influence stopping and are development data; reserve a separate final test. |
| `max_error` | Required finite nonnegative maximum absolute error in encoded output units. This is an application error limit, separate from solver `tolerance`. |
| `epochs` | Nonnegative maximum complete replay passes, default 20. Zero evaluates the current brain without learning. |
| `seed` | Nonnegative integer for a private RNG that shuffles example indices each epoch. Does not alter Python's global random state. |
| `budget` | Nonnegative solve-budget override applied to every admission and check; `None` uses the brain's configured ceiling. |
| `batch_size` | Positive integer, default 1. One preserves ordered `observe` admissions. Larger values group the shuffled examples into `observe_batch` calls; the last group may be smaller. |

All options, examples and checks are validated and samples copied before any
solve or admission. Inputs and targets are not normalized automatically.
Malformed examples fail with their collection/index and leave the brain
unchanged. Numerical exceptions during subsequent solves propagate as in the
underlying brain methods; earlier accepted experiences remain committed.

Before the first epoch and after each complete epoch, pure `settle` calls score
the examples (recall), then the checks, with **no targets supplied to the solve**.
Only actual targeted state coordinates are compared; agreeing output aliases
count once within a pair. Every query must qualify and both maximum errors must
meet `max_error` to pass. An already satisfactory brain returns without replay.
A refusal stops immediately; unqualified outputs never contribute a score.

With `batch_size=1`, each presentation uses `observe` and a fresh automatic
event ID. With a larger size, the helper shuffles the same example indices,
groups them into minibatches and gives each group one fresh ID. Every batch
starts from the retained live state, and successful batches preserve that state.
The final short group still uses its own mean objective and one shared anchor;
its size can affect the learning trajectory.

Replay is repeated supervised experience, not new environmental data. Only
each admission is atomic: a later refusal keeps earlier committed examples or
batches. Starting a new helper call starts a new shuffle sequence and report;
it does not resume an interrupted helper cursor. For external retry identities,
streaming data or custom environment metrics, use `observe`, `observe_batch`
and `settle` directly.

| Report field | Meaning |
| --- | --- |
| `options` | Requested `max_error`, epoch allowance, shuffle `seed` and `batch_size`, plus the resolved integer solve `budget`. Reuse these with the same starting checkpoint and data to repeat the call. |
| `passed` | All current recall/check queries qualified and both errors met the declared limit. Applies only to the supplied cases. |
| `reason` | `"passed"`, `"epochs"` (allowance exhausted), or `"refused"`. |
| `epochs` | Number of fully admitted replay passes; excludes a partly completed pass. |
| `presentations`, `accepted` | Attempted and admitted example presentations in this call, including replay. |
| `updates` | Successful atomic admissions. A committed batch counts once here and by its number of rows in `accepted`; with `batch_size=1`, `updates == accepted`. |
| `examples`, `checks` | Counts of supplied rows, not deduplicated experiences. |
| `history` | Epoch-zero assessment and an assessment after each complete epoch. Each entry has `epoch`, `recall` and `checks`. |
| `work` | Summed solver work counters over admissions and all checks, including refused solves. Does not count Python preprocessing/bookkeeping. |
| `failure` | `None`, or `stage` (`"observe"`, `"observe_batch"`, `"recall"`, `"checks"`), original row `index`, solver `reason`, and `stationarity`. For a refused batch, `index` is its first original example index and `indices` is the tuple of all original indices in batch order. |

Each assessment metric contains `evaluated`, `qualified` and `max_error`.
A refused query sets that collection's `max_error` to `None`, never a score
over only the successful subset. A collection not reached after refusal has
metric `None`. Checkpoint the brain and retain the report plus preprocessing
alongside it; the report is not stored in the brain's snapshot. Exact replay
also needs the **starting** checkpoint and the original examples/checks. The
final checkpoint is the continuation used in the live phase.

## History: explicit temporal context

```text
History(size, *, steps=4)
```

`size` is the positive integer number of raw scalars per sample; `steps` is a
positive integer window length. Encodings contain oldest-to-newest blocks of
`[sample values..., present]`, with `present=1` for a supplied sample and zero
for initial padding. Missing sensor data within a supplied sample needs its own
caller encoding. The oldest sample is discarded when the window fills.

| Property or method | Contract |
| --- | --- |
| `input_size`, `steps` | Read-only raw sample width and retained window length. |
| `size`, `shape` | Read-only encoded width `steps * (input_size + 1)` and `(size,)`, suitable for an input port. Encoded width is limited to one million coordinates. |
| `push(values)` | Validate a finite flat sample, retain it, and return the padded encoding as a tuple. Invalid samples leave history unchanged. |
| `preview(values)` | Return the same hypothetical encoding without retaining the sample. |
| `reset()` | Clear retained samples. |
| `snapshot()` | JSON of dimensions and retained samples. |
| `History.from_snapshot(text)` | Validate exact schema, finite rows, dimensions and window bounds before restoring. Text limit: 32 MiB. |

Samples follow flat boundary validation, including array objects with a valid
`tolist()`/shape; booleans and nonfinite values are rejected. This helper stores
caller-provided observations, not learned neural or episodic memory. Save its
snapshot separately from the brain when resuming an application.

## LearningProgress: predictor-error reduction

```text
LearningProgress(*, rate=0.1, capacity=128)
```

`rate` is finite in `(0,1]`; integer `capacity` is in `[1,4096]`. Both are
read-only properties. `update(key, error)` accepts a nonempty string key of at
most 256 UTF-8 bytes and a finite nonnegative error. It retains one moving mean
per key, evicting the least recently updated context when full. New contexts
score zero; subsequent updates use

```text
new_mean = old_mean + rate * (error - old_mean)
progress = max(0, old_mean - new_mean) / max(1, old_mean)
```

The score lies in `[0,1]` and depends on error units. A constant stream from the
first observation scores zero, as does an error at or above its current mean.
Random downward fluctuations can score positively.
Use actual predictor errors with a consistent definition. This heuristic is
neither information gain nor solver stationarity and does not update a brain.
`snapshot()` and `LearningProgress.from_snapshot(text)` preserve means and
eviction order; loading validates exact fields, bounded capacity, unique keys
and finite errors within 32 MiB. Invalid updates change no records or ordering.

## Reinforcement: discrete reward-driven choices

```text
Reinforcement(
    brain,
    *,
    actions,
    action_input="action",
    value_output="value",
    discount=0.95,
    exploration=0.1,
    reward_scale=1.0,
    value_scale=0.9,
    capacity=1024,
    batch_size=16,
    credit_horizon=1,
    seed=0,
)
```

The helper originated in 0.50.0; `credit_horizon` and explicit executed-outcome
acknowledgments below belong to this development candidate. With the default
action-conditioned form, the compiled
`brain` needs an `action_input` sensor with exactly `actions` coordinates and a `value_output`
selecting one scalar patch state. With `action_input=None`, `value_output`
must be a tuple/list of exactly `actions` scalar output names exposing distinct
physical patches. These action values come from one jointly settled query;
only the selected output is clamped by its teaching estimate. Other
inputs describe the situation, explicit history and any drives. Values are
settled outputs, not a separate policy head. `brain` remains accessible and
`config` is a read-only mapping; one serial owner must control both objects.

| Argument | Contract |
| --- | --- |
| `actions` | Integer at least two. Number of discrete choices. |
| `action_input`, `value_output` | Sensor name plus scalar output name for one-hot action-conditioned queries; or `None` plus a tuple/list of distinct scalar output names for one joint vector-value query. Lists are normalized to tuples in configuration. |
| `discount` | Finite discount in `[0,1)`. |
| `exploration` | Finite probability in `[0,1]` of a uniform action during exploratory selection. |
| `reward_scale` | Positive finite bound for supplied reward magnitude. Rewards beyond it are rejected, not silently clipped. |
| `value_scale` | Positive finite target scale, strictly below both `state_bound` and `1 / (1 + state_prior)`. This check does not establish that a chosen architecture learns the value function. |
| `capacity` | Positive integer bound on stored transition records; oldest records are discarded when full. |
| `batch_size` | Positive integer no greater than capacity. Replay samples up to this many records, always including the latest. |
| `credit_horizon` | Experimental 0.60 candidate: positive integer no greater than capacity; maximum actual consecutive transitions used per return. One preserves the original one-step target. |
| `seed` | Nonnegative integer for private exploration, tie-breaking and replay sampling. |

For reward `r`, the target is a normalized discounted-return estimate:

```text
y = (1-discount) * value_scale * r/reward_scale
    + discount * clip(max_a Q(next_inputs, a), -value_scale, value_scale)
```

Terminal records omit the second term. With `discount=0`, every target uses
only its immediate reward: replay makes no next-action queries or later-record
traversal, even when `credit_horizon` is larger than one. Nonterminal feedback
still records and validates the actual next observation. All targets use
unchanged pre-update parameters. `observe_batch(source="estimate")` fits them with ordinary joint
repair, preserving current live activity. Actual rewards and observations are
records; fitted future-return targets are estimates. Replay is not a guarantee
of protected retention or convergence of nonlinear Q-learning.

In the unreleased temporal-credit candidate, `credit_horizon > 1` follows the
next recorded transition only if its observation exactly matches the preceding
next observation, it belongs to the same episode, and its executed action is
greedy under the same frozen pre-update values. Exact ties count as greedy.
At an off-policy action, missing future, context gap or horizon bound, bootstrap
from the preceding next observation instead. At a terminal record use zero
future value. For a path of length `n`, the target is

```text
sum(k=0..n-1, discount**k * (1-discount) * value_scale * reward[k]/reward_scale)
    + discount**n * bootstrap
```

This is a finite greedy-cut return in the spirit of Watkins traces, not a second
learning rule. See [Munos et al. (2016)](https://arxiv.org/abs/1606.02647) for
off-policy return operators; its tabular guarantees are not guarantees for
Cadence's approximate nonlinear repair. No future outcome is invented when
collection stops. `reset()` cuts temporal links without clearing past replay.

| Method | Contract |
| --- | --- |
| `act(inputs, *, explore=True, budget=None)` | Supply every sensor except a declared action input. Query each action in conditioned mode, or all value outputs jointly in vector mode; require qualification, choose epsilon exploration or a maximum value, breaking ties uniformly; retain qualified activity with `step`. A pending action must receive feedback or be reset before another `act`. |
| `feedback(reward, next_inputs=None, *, decision_id, executed_action, terminal=False, learn=True, budget=None)` | Acknowledge an issued decision and record the action actually executed, which may differ from the proposal. Nonterminal outcomes require next inputs; terminal outcomes require `None`. A first valid acknowledgment consumes the pending decision even if subsequent fitting refuses. An identical retry of the latest outcome does not record or learn again. `learn=False` records without fitting. Invalid or conflicting arguments change nothing. |
| `replay(*, budget=None)` | Attempt one sampled batch update from retained records. Other than the mandatory latest record, sample uniformly without replacement. Next-action query or fit refusal commits no parameters; retry this method rather than resubmitting feedback. |
| `reset()` | End the current episode/credit segment and discard a pending action without inventing a reward or clearing retained records, RNG, parameters or live activity. |
| `inspect()` | Copy `config`, current `records`, cumulative `transitions`, successful helper `updates`, boolean `pending`, current `episode`, `issued_decisions`, `pending_decision_id`, `pending_action` and `last_feedback_id`. Absent pending/receipt identities are `None`. |
| `snapshot()` | Save brain, replay records with episode/decision identifiers, pending proposal, issued-decision counter, latest outcome receipt, RNG and counters in bounded JSON. The unreleased candidate uses `reinforcement/3`; old source-bound checkpoints are not silently migrated. |
| `Reinforcement.from_snapshot(text)` | Validate the complete continuation, including the reinforcement source hash and the brain's own source identity. Text limit: 32 MiB. |

Successful `act` returns `accepted=True`, integer `action`, positive integer
`decision_id`, candidate `values`,
boolean `exploratory` and the selected `settlement`. Values are scaled return
estimates, not probabilities or confidence scores. Query or step refusal
returns `accepted=False`, `action=None`, `decision_id=None`, `values` and reason `"query_refused"`
or `"step_refused"`; pending state and RNG stay unchanged. `explore=False`
disables epsilon moves but still breaks exact ties randomly.

A completed replay attempt returns the batch result plus sampled `indices`,
derived `targets`, `credit_horizons`, `credit_stops` and cumulative `updates`.
Stop reasons are `terminal`, `zero_discount`, `horizon`, `pending_future`,
`discontinuity` and `off_policy`. `zero_discount` reports an immediate-reward
nonterminal target with credit horizon one; terminal records retain `terminal`.
Earlier refusals return
`accepted=False` with `"empty_replay"` or `"bootstrap_refused"`. `feedback`
on a first acknowledgment adds `stored=True`, `duplicate=False`, the `decision_id` and
cumulative `transitions`; with learning disabled its
reason is `"learning_disabled"` and `accepted=False`. A stored transition and
an accepted parameter update are different events. Each `budget` applies to
each underlying query/solve, not an aggregate interaction deadline. An
action-conditioned `act` performs `actions` value queries plus one retaining `step`; vector mode
performs one value query plus one `step`. All successful decisions consume RNG,
even with `explore=False`, because tie-breaking uses the private generator.
Measure every candidate/next-action query and the batch fit, not only the returned fit.

`act` stores a proposal, but cannot observe whether a body executed it. Pass its
`decision_id` and the actual `executed_action` to `feedback`; a discarded command
needs `reset`. IDs increase only after accepted decisions and are never reused
after reset. `decision_id` must be a positive integer and `executed_action` an
integer in `[0, actions)`; booleans are rejected. Feedback stores the executed
action for credit assignment, including
an actuator override. Only the latest acknowledged outcome can be retried:
identical action, reward, next context and terminal status return `stored=False`
and `duplicate=True`, leaving even a newer pending decision intact. Changed
payloads and older, canceled or unissued IDs are rejected atomically. `learn`
and `budget` are execution options, so changing them on a duplicate never
triggers learning. Retry a refused fit with `replay()`.

A duplicate returns `accepted=False`, `stored=False`, `duplicate=True`,
`decision_id`, cumulative `transitions`, `reason="duplicate_feedback"` and
`work={}`. It has no settlement or `qualified` field because it performs no solve.

The latest outcome receipt survives replay eviction and saved continuation.
Decision identity is separate from a brain's learning-event identity: several
replay admissions can use the same actual experience, and direct observations
can interleave with replay. These are acknowledgment guarantees, not proof that
caller-supplied actions or rewards actually occurred in the environment.

`explore=False` changes selection only. `feedback(..., learn=False)` freezes
parameters for that feedback, but still records a transition and can evict the
oldest record. Extra `replay` calls still learn. Snapshot a separate learner for
an evaluation that must not change the live replay, RNG or retained activity.
`Reinforcement.from_snapshot` has no device/dtype override, unlike `Brain`.

## LiveController and actuator limits

```text
LiveController(callback, *, fallback, max_age=0.25, clock=time.monotonic)
slew(current, target, *, rate, dt)
```

`callback` is called as `callback(observation)` on one daemon worker. It must return a mapping
with exact boolean `qualified`. A qualified result needs a finite nonempty
`command` sequence matching the length of `fallback`; a refused result may
omit it. Extra fields are ignored, including `accepted`. Translate a policy's
result into this protocol explicitly. Exceptions and malformed results become
errors; later work can still run. The worker must exclusively own callback
access to any brain, history or reinforcement helper it uses.

`fallback` is a finite nonempty command vector. `max_age` is positive finite
seconds. `clock` is a fast callable yielding finite nondecreasing seconds.
This is best-effort scheduling: the Python GIL, locks, data copying and
validation still consume time, and an active callback cannot be cancelled.

| Method | Contract |
| --- | --- |
| `submit(observation)` | Copy finite JSON-like data and return a positive submission ID without waiting for a solve. At most one observation waits behind the active callback; a newer submission replaces that pending observation. Submission after closure raises `ValueError`. |
| `read()` | Return a command tuple from the latest fresh qualified completion, or fallback before any completion, after refusal/error, after expiry or after closure. A fresh prior command remains available while newer work waits. |
| `inspect()` | Copy counters, worker/queue state, decision status, result age and latest timing/error diagnostics; see below. |
| `close(*, wait=True, timeout=1.0)` | Stop submissions, drop pending work and allow active work to finish. Wait at most finite nonnegative `timeout` seconds if requested; it must not exceed `threading.TIMEOUT_MAX`. Return whether the worker has exited. No solve cancellation occurs. |

Age is measured from submission, not completion: queue and solve time count.
The accepted observation data are string-keyed dictionaries, lists, tuples,
strings, booleans, finite numbers and `None`; cycles and other types are rejected.
The replaceable pending slot is for sensory refreshes. **Do not put reward or
transition records that must be retained in this slot.** Store those losslessly
and deliver them in order to the serial owner. Expiry or closing does not undo
state changes made by a callback that is already running.

Inspection counters are `submitted`, `dropped`, `started`, `completed`,
`qualified`, `refused` and `errors`. Other fields are `closed`, `busy`, `pending`,
`worker_alive`, `status`, `last_result_id`, `result_age`, `last_solve_seconds`,
`last_latency_seconds` and `last_error`. Status is `"waiting"`, `"qualified"`,
`"refused"`, `"error"`, `"stale"` or `"closed"`. Callback/result validation
time is included in solve time; latency additionally includes queue time.
Timing and result identity are `None` until available. Dropped counts pending
replacements and pending work discarded by closing, not cancelled active work.

Unlike brain inputs, runtime observations must already be ordinary JSON-like
Python values: convert array objects to lists before submission. Command and
fallback vectors must be nonempty Python sequences; no array conversion is
performed. There is no runtime checkpoint. A false `close` result means the
callback still owns its brain; do not transfer ownership until it exits.

`slew` returns a tuple moving each `current` coordinate toward `target` by at
most `rate * dt`. Vectors must be finite, nonempty and equal length; `rate` is
a nonnegative scalar or equal-length vector, and `dt` is finite nonnegative
seconds. Nonfinite products are rejected. It limits actuator motion without
choosing the target. See [live-system examples](LIVE.md).

## Checkpoints

Checkpoints contain schema, full configuration and layout, graph fingerprint,
live state, weights, biases, admission count and latest event identity. They
bind the exact source hashes of `brain.py`, `cortex.py`, `column.py`, `ports.py`,
`_repair.py`, `_tensor.py` and `_validation.py`. Loading rejects previous source sets or
different hashes even if package version labels match; compatibility is
therefore stricter than version compatibility.

Preserve the exact compatible source artifact with saved models; even a
comment-only edit to a hashed module changes checkpoint identity. Do not
rewrite stored hashes to make a different implementation load. Application
continuation also needs its observation encoding/normalizer, action decoder,
external history, per-call query-budget overrides, action timing and environment
identity. Those are outside a `Brain` snapshot (the configured default
`settle_budget` is saved): the same weights with changed adapter settings
need not implement the same policy. See [whole-life saves](LIVE.md#save-a-whole-life-and-run-the-small-gates).

Loading checks structure, configuration, deterministic topology, array lengths,
finite values, bounds and event-ownership consistency. Restoration builds and
validates a complete proposal before replacing continuation. Newly loaded brains
own different handles; address them by names when using reconstructed layouts.
`restore` retains the receiving brain's existing handles and requires the
complete configuration, including device and dtype, to match.

`Brain.from_snapshot(text, device=..., dtype=...)` first validates the complete
original snapshot against its saved configuration and current implementation.
An override cannot bypass an invalid fingerprint, source mismatch or malformed
array. Only then are explicit execution settings applied. With both overrides
omitted, saved settings are retained. Supplying `device` without `dtype` uses
the new device's default precision; supplying only `dtype` preserves the saved
device. Arrays, bounds, topology and event ownership are preserved exactly;
execution changes alter the configuration fingerprint. Choosing float32 does
not round stored checkpoint arrays: conversion occurs for device proposals
on the next solve. Hardware availability remains a first-solve check.

JSON is data, not executable deserialization. A structurally valid checkpoint
does not authenticate its maker, prove its witness history or certify current
stationarity. Check external provenance separately and solve before acting.
The 32 MiB text limit does not bound peak decoding or construction memory.
