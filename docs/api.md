# API reference

Start with [Brain.compose](#brain-cadence) for a continuing brain with memory
and optional observers, then use `brain.live(...)` for one continuing stream
with `arousal=True`. [NeuralGraph](#neuralgraph-cadence) is the lower-level
graph API. The [quickstart](quickstart.md) runs the main interaction loop;
sections below describe specialist operations. Pass optional arguments by keyword.

This reference describes Cadence 0.80.0. Install it with
`python -m pip install cadence-net==0.80.0`. `Brain.last_settlement` is an optional
diagnostic report, independent of the interaction loop you use.

The temporal patch: [TemporalPatchNet](#temporalpatchnet-cadencetemporal),
[TemporalPlan](#temporalplan-cadenceplanning), [TemporalMemory](#temporalmemory-cadencetemporal_memory),
[fixed connectivity](#experimental-fixed-connectivity-cadenceexperimental).
The record and belief patches: [RecordPatchNet](#recordpatchnet-cadencerecord_patch),
[Ports](#ports-cadenceports), [BeliefPatch](#beliefpatch-cadencebelief).
A brain that reads itself: [Steered](#steered-cadencesteering), [Life](#life-cadencelife),
[Instruments](#instruments-cadenceinstruments).
The settling brain: [Connectome](#connectome-cadenceconnectome), [Neuron model](#neuron-model-cadenceneuron),
[NeuralGraph](#neuralgraph-cadence), [Blocks](#blocks-cadenceblocks), [Streams](#streams-cadencestream),
[Records](#records-cadencerecords), [Regions](#regions-cadenceregions),
[Genome](#genome-cadencegenome), [Learning](#learning-cadencelearning), [The agent and the valence](#the-agent-and-the-valence-cadenceplasticity),
[Certificate](#certificate-cadencecertificate).
Instruments: [the quickstart demos](#the-quickstart-demos), [Timing](#timing-cadencetiming),
[the reference](#neuron-by-neuron-reference-cadencereference), [Protocols](#protocols-cadenceprotocol),
[Checkpoints](#checkpoints-cadencecheckpoint), [Atlas](#atlas), [Receipts](#receipts-cadencereceipts),
[recording](#record-every-settling-step). Other compositions: [PatchNet](#patchnet-cadencepatch)
and [task compositions](#optional-task-compositions).

## TemporalPatchNet (`cadence.temporal`)

- `TemporalPatchNet(inputs, hidden, outputs, *, seed=0, output_precision=None, ...)`
  creates the residual temporal model with shared A/B/C maps. Paths have shape
  `(batch, time, ports)`; omitted positive output precision means all ones.
- `observe(inputs, target, *, beta=0.01, rate=0.1, backtrack=False,
  symmetry_tolerance=0.15, max_halvings=8)` repairs observed teaching
  paths and returns `TemporalObservation`. Only valid free activity becomes live;
  detuned activity does not become an observed record. Beta is halved, up to
  `max_halvings` times, until the two detuned paths sit within
  `symmetry_tolerance` of center around the free path; a pair that stays off
  center is refused as `contrast_asymmetric`. See
  [temporal learning](temporal.md#checking-that-the-contrast-is-centered).
  The optional `backtrack` argument:
  accept parameters only after a decreasing causal replay from the original
  boundary. See [temporal learning](temporal.md#checking-a-learning-step).
- `contrast_asymmetry(free, plus, minus)` in `cadence.temporal` is the ratio the
  check reads: `||plus + minus - 2 free|| / ||plus - minus||` over hidden paths.
- `advance(inputs)` carries a free path into live context. `imagine(inputs, *,
  state=None)` predicts privately. `settle(inputs, *, target=None, beta=0.0,
  state=None)` exposes a private phase directly.
- `plan(inputs, *, goal, controls, bounds=None, state=None, beta=0.01,
  rate=1.0, max_steps=32, max_backtracks=16, tolerance=1e-6,
  goal_tolerance=1e-6, symmetry_tolerance=0.15, max_halvings=8,
  method="steepest")` repairs
  selected continuous input ports. Boolean controls
  and bounds broadcast to the input shape. It executes no action and changes
  no live state. Each contrast passes the same symmetry check as `observe`.
  `method="bfgs"` steps along a quasi-Newton direction built from the accepted
  steps, with the same line search and replay.
  See [planning](planning.md) for precise cost and failure semantics.
- `readback()`, `state`, `parameters()` and `snapshot()` expose detached values.
  `reset()` clears activity, not learned parameters. `save(path)`, `load(path)`
  and `restore(snapshot)` preserve continuation state and configuration.
- `set_output_precision(precision)` replaces the supplied loss geometry
  atomically. Free energy and recurrence do not use teaching precision, so
  parameter revisions, free diagnostics and protected constraints stay bound to
  the unchanged maps. `set_parameters(mapping)` validates and replaces all
  learned arrays in one transaction.
- `TemporalPhase` exposes solved hidden/output paths, residuals, curvature and
  work. `TemporalObservation` exposes `updated`, `reason`, the phases, raw
  `delta`, the `beta` of the last contrast and its `contrast_halvings`. With
  backtracking it also reports initial/final loss, accepted rate,
  trial losses and replay count. `TemporalReadback` binds current activity to
  its parameter revision.

## TemporalPlan (`cadence.planning`)

The detached result contains proposed `inputs`, a target-free `prediction`,
`initial_prediction`, fixed `boundary`, `losses`, `step_sizes` and model revision.
`cost`, `initial_cost`, `improved` and `iterations` summarize accepted work.
`converged` concerns the finite-beta projected residual; `predicted_goal_met`
checks modeled cost. Neither certifies an actual outcome. `beta` and
`contrast_halvings` report the detuning of the last contrast and how often it
was halved to center the detuned paths; `method` names the search direction.
Work and failure
fields are described in the [planning guide](planning.md).

## The quickstart demos

The [quickstart](quickstart.md) runs `Brain.compose`; application demos live in
[cadence-demos](https://github.com/muellerberndt/cadence-demos).

## RecordPatchNet (`cadence.record_patch`)

See the [record patch guide](record-patch.md).

- `RecordPatchNet(inputs, hidden, outputs, *, seed=0, output_precision=None,
  cells=4096, active=32, record_rate=0.5, habituation=1e-5, record_bias=0.3,
  slowest=128.0, record_averaging=False, record_homeostasis=0.0, groups=None,
  record_writes="sequential", record_width=None)` creates a
  gated linear context with a `Records` store over the reading
  `[u * sqrt(n) / s, r * h]`, both blocks of unit variance per unit (`s` is the
  running rms norm of witnessed inputs). `record_averaging` and
  `record_homeostasis` pass to `Records` as `averaging` and `homeostasis`. Paths have shape `(batch, time, ports)`.
  `groups=(n1, n2, ...)` makes the ports categorical: one softmax per group,
  cross-entropy for the slow readout, records of `onehot - softmax`.
  `record_writes="batch"` writes a call's moments at once through
  `Records.write_batch(codes, targets)`, each cell moving by the mean of its
  writers' moves. `record_width=w` makes the store hold a fixed random
  `w`-column sign code of the residual instead of one column per output.
- `RecordPatchStack(inputs, hidden, outputs, *, lower=None, seed=0,
  slowest=128.0, groups=None, **upper)` puts a context patch of width `lower`
  below a `RecordPatchNet` that reads `[u, r1 * h1]`; `observe`, `imagine`
  (`state` is the pair of contexts), `advance`, `reset`, `parameters`,
  `snapshot` and `restore` as for one patch. `observe` returns
  `StackObservation`: `updated`, `reason`, the upper patch's `prediction`,
  `delta`, `initial_loss`, `final_loss`, `accepted_rate` and `writes`.
- `JointRecordPatches(cortices, own_inputs, ports, *, rounds=1, damping=1.0,
  cross_adjoint=True)` couples several `RecordPatchNet`s through finite Jacobi rounds;
  each `Port(source, target, start, width)` carries a band of the source's scaled
  context into the target's inputs in the same moment, over `rounds` Jacobi
  rounds. `observe(xs, ys, rate=, backtrack=, write=)` returns a
  `JointObservation` (`updated`, `reason`, the `settled` path with `seam` and
  `settle` per moment and round, `delta` per cortex, the losses, `writes`);
  `imagine(xs, states=)`, `advance(xs)`, `reset`, `parameters`, `snapshot`,
  `restore`, `clone`, `save`, `load`; `cut = True` zeroes every port.
  Snapshots preserve the cut flag; saves use atomic replacement. Inputs, targets
  and state lists must match the cortex count, and nonfinite paths are refused.
  A failed joint record write restores earlier writes and does not advance live
  contexts. Fixed rounds do not certify a joint equilibrium.
  `cadence.record_ports.build(hidden, own_inputs, outputs, ports, *, seed, ...)`
  grows the patches at the widths the ports need.
- `observe(inputs, target, *, rate=1.0, backtrack=False, write=True)`
  predicts with the records at the start of the call, moves the slow
  parameters against the adjoint gradient of the precision-weighted half
  mean squared error of the slow readout `C h + c` (admitted by causal
  replay when `backtrack=True`) and writes the residual `target - C h - c`
  into each reading's records. Returns `RecordObservation`: `updated`,
  `reason`, the `prediction` (`RecordPath` with `hidden`, `output`, `gate`,
  `read`, `loss` of the prediction and `slow_loss` of the slow readout),
  `delta`, the slow readout's admission losses and rates, replay count and
  write count.
  At `rate=0`, single and joint record observations return `updated=False`,
  `reason="no_step"`; gradients, valid activity and requested record writes remain
  available, but slow-update counters do not advance.
- `imagine(inputs, *, state=None)` is private; `advance(inputs)` carries
  context; `reset()` clears context and keeps parameters and records.
- `dream(inputs)` is the store's completion of a cue from rest, as a target (the chosen
  category per group for categorical ports); `sleep(cues, *, passes=1, rate=1.0,
  backtrack=False, dawn_passes=2)` dreams every cue once, teaches the slow weights the
  fixed dreams by `observe(write=False)`, and at dawn writes the dreams back so the store
  holds only what the slow weights did not take. Returns admitted updates, the mean slow
  loss on the dreams before and after, and the dawn writes.
- `detune(inputs, target, *, beta=1e-3, state=None, tolerance=1e-14,
  max_iterations=10000)` solves both detuned equilibria of the quadratic
  energy by conjugate gradients and returns `RecordContrast`: the centered
  `contrast`, both hidden paths, energies, iterations, residuals and
  `converged`. It changes nothing.
- `parameters()` and `set_parameters(mapping)` cover `G`, `g`, `B`, `b`, `C`
  and `c`; `records` is the `Records` store. `readback()` returns
  `RecordReadback`: `state`, `updates`, `writes`, `parameter_revision`,
  `state_parameter_revision` and `record_entries`, read-only and never
  admitted as teaching data by itself. `snapshot()`, `restore(snapshot)`,
  `save(path)` and `load(path)` carry parameters, records, counts and live
  context.

## Ports (`cadence.ports`)

- `MapBlock(start, channels_in, height, width, channels_out, kernel, stride=1)`: a tied local
  kernel over a grid of the inputs; `DenseBlock(start, inputs, outputs)`: a matrix over a slice.
- `StructuredPort(inputs, blocks, broadcast=None, mask=None)`: `apply(u, weights)`,
  `transpose(v, weights)`, `gradient(v, u)`, `initial(rng, scale)`, `weight_shape(block)`,
  `dense_matrix(weights)`, `to_dict()`, `from_dict(d)`. `broadcast=(start, count)` tiles that
  slice into every map block as constant channels. `mask` `(inputs,)` names which inputs the
  port hears; a masked input is zero to every block in all three maps and the mask travels
  with `to_dict`.
- `RecordPatchNet(..., port=StructuredPort)` and `RecordPatchStack(..., lower_port=StructuredPort)`
  read their inputs through the port; `hidden` (or `lower`) equals the port's outputs.

## BeliefPatch (`cadence.belief`)

See the [belief patch guide](belief.md).

- `BeliefPatch(observation: StructuredPort, actions, belief, outputs, *, iterations=2, damping=0.5,
  cells=4096, active=32, record_rate=0.5, record_width=64, habituation=1e-5, record_bias=0.3,
  output_precision=None, seed=0)`. `block_count` is the number of the port's blocks.
- `assimilate(observations, actions, observed=None, *, state=None, gains=None, probe=False,
  keep_live=False) -> BeliefPath`: advance the belief through observed moments; nothing
  learned or written. `imagine(actions, *, state=None, gains=None) -> BeliefPath`: the
  transition alone under declared actions, private. `observe(observations, actions,
  target=None, *, observed=None, rate=1.0, write=False, state=None, gains=None,
  loss_weight=None, output_gradient=None, backtrack=None, probe=False, keep_live=False)
  -> BeliefObservation`: one backward scan and, with `write=True`, the store's writes. `state` starts the
  moments from a given boundary instead of the live belief; the final belief becomes the
  live state unless `keep_live=True`. `observed` masks moments `(time,)` or rows
  `(batch, time)`; a row that observes nothing keeps its expectation. `gains` `(blocks,)`,
  `(batch, blocks)` or `(batch, time, blocks)` multiplies each block's encoded evidence before
  the repair map and the store read see it. `loss_weight` `(time,)` or `(batch, time)` weighs
  each moment's error, normalized by its sum; a moment of weight zero is neither taught nor
  written. `output_gradient` `(batch, time, outputs)` replaces `target`: the adjoint of an
  external loss on the outputs; nothing is written and no loss is reported. A step on a
  target is admitted (`backtrack` None or True): the largest halving of the start whose
  replay of the chunk from the same boundary, with the store as it stands, lowers the loss
  by the Armijo margin (sixteen halvings at most), the start being twice the last admitted
  step and at most `rate`; `backtrack=False` is the plain step at `rate`, and a step on an
  external gradient is plain. `probe=True` computes the residual-alone probe per block.
- `readback(observations, actions, *, state=None, probe=True) -> BeliefReadback`: one moment
  `(batch, inputs)`, `(batch, actions)` before its repair, from the live belief or `state`:
  `expectation` `(batch, belief)`, `residual_alone` and `surprise` `(batch, blocks)`,
  `evidence` `(batch, encoded)` before any gain. `encode(observations) -> (..., encoded)`:
  the encoded evidence of any reading before any gain. Both change nothing.
- `step_size`: the last admitted step, None before one; the next admission starts from
  twice it. `reset_step()` drops it; `reset()` keeps it; it travels with the snapshot.
- `macs_per_moment(*, probes=False, surprise=True) -> int`: a dense forward-MAC estimate
  for one moment; includes spatial kernel reuse. `readback_macs(*, probe=True)` estimates
  a separate readback. Neither counts the backward adjoint or complete training work.
  `cost`: `{"moments", "macs", "replays"}` counted since `reset_cost()`, every moment
  assimilated, observed, imagined or replayed in an admission.
- `set_implied_reading(implied, units=None)`: declares the map from the outputs
  `(batch, outputs)` to the reading each block should give `(batch, inputs)`, channels left
  `NaN` not compared, and the persistence error of each block's compared channels `(blocks,)`;
  paths then carry `surprise`. `None` withdraws it. Not part of a snapshot.
- `BeliefPath`: `belief`, `expectation`, `residual`, `step` (the last repair move per unit, whose
  norm is `residual`), `output`, `read`, `loss`, `slow_output`, `final_state`; `evidence`
  `(batch, time, encoded)`, the encoded evidence after the gains; `code` `(batch, time, cells)`,
  the store's plain code at the final reading; `gains` `(batch, time, blocks)`, the gains used;
  `residual_alone` `(batch, time, blocks)`, the repair map's move with one block heard and the
  store read at zero (with `probe=True`); `surprise` `(batch, time, blocks)`, each block's
  reading against the reading the previous belief's slow readout implies, in persistence
  units (with an implied reading declared).
- `BeliefObservation`: `updated`, `reason`, `path`, `delta`, `initial_loss`, `writes`,
  `final_loss` (the replayed loss under the admitted parameters), `accepted_rate` (the rate of
  the step taken, None without a step), `replay_calls`, `gain_gradient` `(batch, time, blocks)`.
- `reset()` (the live state; the step size and the counters stay), `state`, `parameters()`,
  `set_parameters()`, `set_output_precision()`, `records`, `snapshot()`, `restore()`,
  `save()`, `load()`.
- `cadence.belief_torch.TorchBelief(port, actions, belief, outputs, *, iterations, damping, record_width)`:
  the slow half on torch; `forward(observations | None, actions, state=None, reads=None,
  gains=None, observed=None)`, `export()`, `load(params)`. A gains tensor that requires grad
  receives the gradient into the gains.

## Steered (`cadence.steering`)

See [a brain that reads itself](steering.md).

- `Steered(cortex, steering=None, weighing=None, *, reads_output=False, evidence=(), extra=None,
  extra_channels=0, rate_scale=1.0, lagged=False, relative=False, reads_age=False, age_core=0.5)`:
  a `BeliefPatch` cortex whose gains a steering `BeliefPatch` sets through a weighing, or a
  `Rule`, or nothing (gains of one). The readback per moment, in order: the residual-alone probe
  per block, the surprise per block, the previous residual, the previous outputs
  (`reads_output`), the age of each block (`reads_age`: revolutions since its gain last reached
  `age_core`), the encoded evidence of the blocks in `evidence`, the channels
  `extra(readback, previous_output, previous_residual)` returns. `lagged=True` fills the probe,
  surprise and evidence channels from the previous moment's path, masked where that moment's
  gains were zero (what a window heard, under a sensing cost); `relative=True` rolls every
  per-block group so that index zero is the block under a `Gaze`'s centre. The steering patch's
  port reads exactly `channels` inputs; its mask says which it hears. `readback_names`,
  `layout`, `channels`, `probes_on`.
- `run(observations, actions, target=None, *, rate=0.0, state=None, keep_live=False,
  learn_cortex=True, backtrack=True) -> SteeredPath`: the moments of a chunk in order; with a
  target and a rate, one joint step admitted by a replay of the chunk (the steering patch on the
  recorded readbacks, taken as given, the cortex under the gains it returns), halved while the
  objective does not fall by the Armijo margin, from twice the last admitted step, at most
  `rate`; `learn_cortex=False` lets the cortex sleep; under a rule or fixed gains the cortex
  learns by its own admitted step. `SteeredPath`: `output`, `gains`, `readback`, `residual`,
  `steering_output`, `loss`, `price`, `objective`, `updated`, `steering_updated`, `step`,
  `halvings`, `replays`, `reason`, `last`, `last_steering`.
- `boundary() -> Boundary | None`: a detached copy of where the life is (`cortex`, `steering`, `output`, `residual`,
  `weighing`, `moments`, `heard`: what the previous moment heard and the ages); `run(state=boundary)`
  starts there. `reset()` forgets it.
- `ablation` (`None`, `"cut"` for gains of one, or a callable over the gains `(batch, blocks)`,
  the steering patch still run and counted), `deaf` (a mask over the readback channels zeroed
  at run time).
- `step_size`, `reset_step()`, `moments_per_decision()`, `macs_per_moment()`, `cost`,
  `reset_cost()`, `parameter_count()`, `parameters()`, `set_parameters()`, `snapshot()`,
  `restore(snapshot, *, rule=None, extra=None, ablation=None)`, `save()`, `load()`.
  Snapshots preserve the live boundary, gaze/heard/age state, deaf mask and ablation;
  custom rule, extra-channel or ablation callables must be supplied again. The cortex's
  implied reading is declared again after a restore. Old snapshots without a live
  boundary restore parameters but cannot resume the missing activity state. Record
  writes through `Steered.observe(write=True)` are unsupported and explicitly rejected.
- Weighings: `Softmax(blocks, *, span=1.0)` (gains summing to the blocks); `Gaze(blocks, *,
  sigma, cut=2.5, lamp=0.2, span=0.5, price=0.0, start=0.0)` (a window whose centre the
  steering output turns; `profile(centre)`, `turn(y)`; the centre is the weighing's state);
  `Rule(fn, *, macs=0)` (the hand-written control). A weighing is `begin(n)`, `gains(y, state)
  -> (gains, state)`, `pull(ys, gains, dgains, states) -> dy`, `price(ys, states)`, `to_dict()`.

## Life (`cadence.life`)

See [a life with a governor](steering.md#a-life-with-a-governor).

- `Life(patch, governor, *, habit, propose, advance, cost, target, baseline=1.0, residual0=1.0,
  config=None, refit=None, keep_records=True)`: the loop over a `BeliefPatch` or a `Steered`.
  `habit(reading) -> action`; `propose(reading) -> (candidates, actions)`; `advance(reading,
  output) -> reading`; `cost(readings, actions) -> (candidates,)`; `target(reading, next) ->
  (outputs,)`. `LifeConfig`: `window`, `passes`, `learn_rate`, `rollback`, `validity`,
  `min_window`, `min_cooldown`, `cooldown`, `keep`, `horizon`, `hold`, `baseline_rate`, `floor`,
  `habituate`, `slow_rate`, `recent`, `write`.
- `decide(reading) -> action` and `outcome(next_reading) -> Decision`, or `step(reading, world)
  -> (next_reading, Decision)` with `world(action) -> next_reading`. `readback()` is the seven
  channels (`READBACK`); `signals()` what a governor reads. `records`, `learns`, `totals`,
  `compute()` (moments, macs, replays, the governor's steps as moments, per decision), `reset()`.
- Governors: `PatchGovernor(genome=None, *, cortex=4, model=GOVERNOR_MODEL, budget=200, chunk=10,
  tolerance=1e-3)` with `hand_set(cortex)` and `space(cortex)`; `ThresholdGovernor(genome)` with
  `HAND_SET` and `SPACE`; `AlwaysAwake(genome)`; `NeverWakes()`. A governor is `settle(signals)
  -> (mode, steps)`, `after(signals)`, `reset()`, `synapses`. `PatchGovernor.converged`
  and `.residual` expose solve qualification; a capped solve selects habit. The
  threshold governor's cooldown limits repeat learning requests; `LifeConfig`'s
  eligibility/cooldown still applies independently. `Life.reset()` drops unfinished
  decisions and clears all replay-window boundaries. Failed learning restores
  parameters, records, update counts and step-size memory; attempted work stays counted.

## Instruments (`cadence.instruments`)

- `orienting(gain, events, *, pre=4, post=12, quiet=None, bins=10) -> {"rows", "kinds", "pre",
  "post"}`: per event the baseline, peak, capture, latency and return; per kind the count, the
  capture of the first and last `bins` events, the curve, the mean capture, the latency shares
  and the mean return. `return` is None when recovery was not observed or capture
  was nonpositive; `return_censored` distinguishes an unobserved positive capture's
  recovery. Kind summaries count censored events and omit them from `return_mean`.
  A partial last habituation bin is retained.
- `dishabituation(rows, *, consequential, kind, window=60, count=2) -> {"before", "after",
  "events"}`.

## TemporalMemory (`cadence.temporal_memory`)

- `TemporalMemory(*, relative_tolerance=1e-12)` creates explicit local response
  constraints. `protect(net, inputs, *, state=None)` admits the current free
  response of a caller-selected query, without targets or network mutation,
  and returns `ConstraintReport`: `ranks`, `bytes` and `maximum_residual`.
- `observe(net, inputs, target, *, beta=0.01, rate=0.1,
  readout_damping=None, symmetry_tolerance=0.15, max_halvings=8)` stages
  protected learning atomically. Positive finite
  readout damping selects the local metric and causal acceptance check described
  in the [memory guide](temporal-memory.md); `None` retains ordinary projection.
  The symmetry settings pass through to `TemporalPatchNet.observe`.
- `report()` returns the same `ConstraintReport` for the current bases. `snapshot()` and
  `restore(snapshot)` preserve bases and parameter binding. Save the net and
  its memory together. Lower-level `project(before, proposed)` requires the
  caller's explicit transaction and is documented in the source.

## Experimental fixed connectivity (`cadence.experimental`)

This namespace is not exported at
the top level. `PartitionedTemporalPatchNet(inputs, hidden, outputs, *,
masks=None, **options)` accepts the temporal constructor options and boolean
`A`/`B`/`C` masks; omitted masks allow all entries. It preserves the existing
solver and masks parameter gradients before candidate admission.

`two_group_masks(context_hidden, motor_hidden, inputs, outputs, *,
context_inputs, motor_inputs, cross_coupling=True)` builds one supplied routing
pattern. The model's `masks` property returns copies and
`trainable_parameter_count` counts permitted entries, not allocated storage.
Restore with the subclass's `restore`/`load` to preserve mask enforcement.
**`TemporalMemory.observe` rejects this subclass before mutation.** See the
[experimental guide](partitioned.md) for checkpoint, planning and evidence scope.

## PatchNet (`cadence.patch`)

`PatchNet.recursive(inputs, layers, outputs, *, seed=0,
coupling=1.0, config=None, backend="cpu", device=None, **runtime_options)` builds
one reciprocal graph. `layers[0]` is the base hidden population; later widths
add observers connected in both directions to all previous neurons. The output
neurons also settle. `coupling` sets the initial maximum absolute weight row
sum, not a permanent bound after learning. The factory uses the smooth
`learning_neuron_model(leak=1.0)` and returns the ordinary `PatchNet`; it does
not impose executive priority. See the runnable
[recursive-settlement guide](recursive-settlement.md).

`PatchNet` composes the existing `NeuralGraph` and `Learner` for ongoing continuous
observations. The [guide](patchnet.md) explains the equations, memory boundaries
and a complete runnable example.

- `PatchNet.create(inputs, hidden, outputs, *, seed=0, ...)` creates a reciprocal
  graph with declared input, hidden and output ports. Configure learning with
  `config=LearnerConfig(...)`, phase budgets with `steps` and `tolerance`, and
  optional temporal overlap with `context_strength` and `context_mask`.
- `solver="hybrid"` optionally follows the local budget with up to
  `refinement_steps=64` accepted Newton steps per unresolved row. The default
  `solver="local"` preserves the existing local-only contract. Hybrid requires
  the CPU smooth `tanh(v/2)` rule, reciprocal effective weights, no adaptation,
  and an independent input population with no output or context penalty there.
  It eliminates those inputs exactly, retaining their feedback, and checks the
  complete equations and positive reduced energy curvature for every row.
  Unsupported configurations raise; there is no silent backend downgrade.
  Dense curvature checks are global numerical work, not local neural repairs.
  See the [training guide](recursive-training.md) for cost and stability limits.
- `stimulus(inputs, *, amplitude=1.0)` converts a batch of continuous input
  values into full neural drives. `settle(drive)` updates live free activity;
  `read(phase)` reads the output ports of an `Equilibrium` or `BrainState`.
- `observe(drive, target, *, observed=None, weight=None, source_id=None)` returns
  a `PatchObservation` with the free and nudged phases, `updated` and `reason`.
  Targets only enter the nudged phases. A required phase that misses the
  residual tolerance prevents the learning commit. Hybrid phases must also
  meet the curvature check; an otherwise converged but unqualified phase returns
  `free_unqualified` or `nudge_unqualified`. Use `phase.qualified` before acting
  and `observation.updated` to count committed learning. The observation mask is
  shared across batch rows; nonnegative teaching weights are per row.
- `imagine(drives, *, state=None)` returns consecutive free equilibria on a
  private branch without modifying live activity, parameters or evidence IDs.
- `reset()` clears activity while preserving learned parameters, optimizer
  history and the optional bounded evidence-ID window. `state` and `snapshot()`
  expose detached state for inspection.
- `save(path, *, compressed=True)` and `PatchNet.load(path, ...)` preserve
  continuation parameters, optimizer history, current activity and configuration.
  Format `cadence-patch/3` includes the solver and refinement budget. Formats 1
  and 2 remain readable with their original local-only solver policy.
  Checkpoint correctness does not establish retention during new learning.

## Connectome (`cadence.connectome`)

- `Connectome.from_synapses(n, *, pre, post, count=None, sign=None, populations=None, label="connectome", min_count=0.0)`:
  build from synapse lists. Parallel synapses merge (counts add), autapses drop, synapses
  below `min_count` drop, and the arrays are sorted by `(post, pre)`. `count` (synaptic
  contacts per synapse) defaults to 1 and `sign` to +1.
- Fields: `n`, `pre`, `post`, `count`, `sign` (arrays), `populations` (name → tuple of
  neurons), `label`. Arrays and the population mapping are read-only; reconstruct
  topology or use `with_populations` to change groups. Property `synapses` counts synapses.
- `members(*names)`, `with_populations(**populations)`, `in_degree()`, `out_degree()`,
  `summary()`, `digest()` (SHA-256 of the sorted arrays).

## Neuron model (`cadence.neuron`)

- `NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0, adaptation=None, leak=0.0)`:
  the graded (rate) neuron. `activation(v)` is the rectified, re-based sigmoid with optional leak;
  `slope_at(v)` its derivative; `rest_emission` the raw sigmoid's value at rest, which is
  subtracted so rest publishes zero. `replace(**changes)`, `to_dict()`.
- `Adaptation(tau_steps=50.0, strength=1.0)`: a slow per-neuron variable that can
  produce rhythm in suitable circuits.
- `learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0)` (in
  `cadence.learning`): responsive starting settings for local learning, to validate for your task.

## NeuralGraph (`cadence`)

- `NeuralGraph(connectome, neuron_model, *, backend="cpu" | "torch" | "mlx", efficacy=None, log_gain=None, bias=None, device=None, dense_limit=2048, layout=None, precision=None)`:
  the runnable brain. `efficacy` (the learned synaptic efficacy) defaults to the connectome's
  signs; `log_gain` and `bias` to zero. Attributes include `connectome`, `neuron_model`,
  `efficacy`, `log_gain` and `bias`. When the connectome's dense blocks fit in `dense_limit`
  squared entries the transport is the block transport (see `cadence.blocks`); `layout`
  passes a precomputed cut, as `with_parameters` does. `brain.layout` is the cut in use.
  `precision` (torch only) is `"float32"` or `"float64"`; the default is float64 except on
  MPS. Float32 is the speed of a consumer GPU; compare it with float64 and record the
  measured precision error. Parameter arrays and effective `weights` are read-only;
  assign complete `efficacy`, `bias` or `log_gain` arrays through validated setters,
  or use `with_parameters` to construct a new brain. Invalid/overflowing values are
  rejected before changing host parameters or device transport.
- `settle(stimulus=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> BrainState`:
  one stimulus; `stimulus` is a list of neurons at full amplitude, a `{neuron: level}` map, or a
  dense vector. Map values are levels multiplied by `stimulus_amplitude`; dense vectors
  are drives directly. `settle_batch(drive, ...)` takes `(batch, n)` drives.
  `mask` accepts `(n,)`, `(1, n)`, or `(batch, n)`, with finite values in `[0, 1]`; zero suppresses a neuron.
  With a positive `tolerance`, both methods stop once every activation moves less than
  that amount in a step. `None` or zero uses the full step cap. Nonfinite drives and warm
  potentials/adaptation are rejected. Use floating arrays for dense drives and maps for
  selected indices: the legacy integer vector of length `n` containing only 0/1 is a drive.
- `equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None, nudge=None, damping=0) -> Equilibrium`:
  seek a joint state whose equation residual is below tolerance, checking after each chunk.
  `budget` caps additional settling steps exactly, including a short final chunk;
  zero checks the starting state. Each check uses one transport; unread float64 Torch
  states stay on their device and return one scalar per row.
  Optional `damping` divides the same budget among successively halved integration
  steps, with every result checked against the original equations. Two consecutive
  repeated complete-state checkpoints can end a stalled attempt early when another
  halving remains; unused sweeps stay available to the later attempts. Repetition
  never qualifies an answer, and the final attempt does not stop for stagnation.
  `Equilibrium` has `state`, per-row `residual`, total `steps`, `tolerance`,
  `residual_checks`, `damping_halvings`, `stagnation_checks`, and a boolean
  per-row `converged` property. `residual_checks` counts additional transport
  evaluations outside the sweeps; `stagnation_checks` counts complete-state comparisons
  without synaptic transport. `state.steps` is the last chunk's count. Convergence here
  does not prove stability, uniqueness or task quality. `qualified` equals
  `converged` for these local solves. Hybrid `PatchNet` solves additionally
  require positive local energy curvature and a successful refinement status;
  they do not change the meaning of `converged`.
  Their optional `refinement: RefinementReport` records the original local
  residual, accepted Newton `steps` per row, `min_curvature` and a per-row
  `status` tuple (`local` or `refined` on success, a failure reason otherwise).
  `Equilibrium.steps` always counts local sweeps, not Newton work; curvature
  evaluations and line-search trials are additional work. These checks do not
  establish global uniqueness or that free/nudged phases occupy the same branch.
- `residual(drive, state, *, nudge=None, mask=None, on_device=True)`: per-row maximum remaining
  fixed-point equation discrepancy, including adaptation when enabled. One transport
  evaluation, no state change. Unread float64 Torch states use the resident kernel;
  float32 states and other backends use the float64 CPU reference. `on_device=False`
  selects that reference explicitly. It is a diagnostic and does not prove stability or uniqueness.
- `ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12) -> EPStructure`: reports the
  maximum asymmetry of effective free/free weights, incoming mass on excluded source
  neurons, and adaptation. Its `compatible` flag covers structure only: inspect phase
  residuals, unchanged unnudged inputs, smoothness, stability, finite-beta bias, and
  parameter/loss units separately. See [the certificate guide](certificate.md#equilibrium-propagation-scope).
- `stimulus_vector(stimulus)`, `stimulus_levels(levels)` (finite levels times the stimulus
  amplitude, with signed values allowed), `readings(state, names, i=0, level=0.5)`, `with_parameters(*, efficacy=None, log_gain=None, bias=None)`,
  `weights` (effective drive per synapse), `dense()` (the `W[pre, post]` matrix), `to_dict()`.
- `NeuralGraph.contrast_on_device(plus, minus)`: the learning rule's per-synapse and per-neuron
  contrast computed on the device when both states carry its handle; `None` otherwise.
- `BrainState`: `v`, `activation`, `adaptation`, `steps`, `trajectory`, `activity_change`
  (the total movement of the activations while settling, per row), `device` (the same
  state on the accelerator that produced it, or `None`). State arrays have shape `(n,)`
  from `settle` or `(batch, n)` from `settle_batch`; trajectories add a leading step axis.
  `row(i)`, `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`,
  `batched`.
- `Nudge(target, mask, beta, softmax_temperature=None, weight=None, groups=None)`: extra drive
  `beta · mask · (target − s)`, or `beta · mask · (target − softmax(s/T))` over each
  masked group with a temperature; `weight` scales rows. `groups` assigns a separate
  softmax group per neuron (`-1` excludes a neuron). The temperature is a finite positive
  scalar or one value per neuron, constant within each active group. Fractional masks
  scale the local drive in both modes. `drive(s)`.
- `available_backends()`: `{"cpu": "numpy float64", "torch": "mps float32" | "cuda float64" | "cpu float64", "mlx": "gpu float32"}`, for what is installed.

## Blocks (`cadence.blocks`)

- `layout(connectome, *, max_pairs=256) -> Layout`: cut the neurons into contiguous ranges at
  the boundaries of the connectome's contiguous populations and pair the ranges that carry
  synapses. A connectome with no contiguous populations, or one that fragments into more
  than `max_pairs` blocks, gets one block, the full matrix.
- `Layout`: `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`; `ranges`, `pairs`,
  `size`, `bounds(k)`, `sources()` (ranges that receive no synapses), `flat(weights)`,
  `blocks(flat)`, `to_dict()`.
- `BlockTransport(layout, flat)`: `synaptic_input(s)` returns the synaptic input for the
  activations `s`, reusing the product of every source range that did not move since the
  previous call.
- `block_contrast(layout, s_plus, s_minus)`: the learning rule's per-synapse contrast as one
  Gram product per block.

## Streams (`cadence.stream`)

- `stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0, context_init=1.0) -> (Connectome, tie_groups)`:
  `embedded` plus a `context` range of `hidden` neurons that receive no synapses and reach
  every hidden neuron. Populations `input`, `context`, `embedding`, `hidden`, `output`.
- `Trace(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")`:
  the memory of the moment before, per stream. It keeps a trace of the `source` range's
  activation and adds it to the next settling as a stimulus on the paired `target` range
  (one neuron per source neuron). `focus` above zero weights each neuron by its movement
  since the last moment (its share of the row's mean movement, to that power), so the trace
  is brightest where the moment changed. `reset(batch, rows=None)`, `stimulate(drive)`,
  `update(state)`, `keep(rows)`, `ringing(floor=0.1)` (each source neuron's share of what is
  still ringing, one elsewhere; a salience for `ActorCritic.salience`), `to_dict()`.
- `Echo(connectome, decay=0.5, amplitude=1.0, ...)`: a `Trace` with the same defaults, used at
  focus 0 into the `context` range (the carried state of the earlier moments).
- `Afterglow(connectome, decay=0.5, amplitude=1.0, focus=1.0, source="hidden", target="afterglow")`:
  the focused `Trace`. With `source="input"` it is an afterimage of the picture itself, the
  memory that reads a cue against a static background (`tests/test_child.py`: 1.00 where the
  Echo reads chance).
- `Efference(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="motor", target="efference")`:
  the corollary discharge, a `Trace` written from the command the brain issued instead of a
  settled activation: `issue(command)` decays the trace toward the `(batch, motor)` one-hot of
  the executed action (`update(state)` raises). One `target` neuron per motor neuron; the
  association region reads it through a plastic projection. `Brain.compose(...,
  efference_amplitude=...)` adds it; `Brain.efference` holds it.
- `FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False, rule="hebb", separator=None, writes=0)`:
  one mutable `(pre, post)` matrix per stream; `writes` counts the rows written. `separator`
  is a `PatternSeparator` applied to keys and queries, which makes the matrix
  `(expansion, post)` per stream. `rule="delta"`
  uses unit keys and writes `rate * outer(key, value - prediction)`; it rejects
  `normalize`/`replace` and rates outside `[0, 1]`. The default `rule="hebb"` is additive
  Hebbian memory, with optional count-averaged reads (`normalize`) and one-hot row
  replacement (`replace`).
  `observe(key, value, write=None)` uses `(batch, width)` ports and writes all rows unless
  given a boolean mask; `recall(key)` reads without decay. `update(state, write=None, post=None)`
  reads named neuron activations instead and defaults to decay-only; `read(drive)` reads
  key neurons' drive columns. `stimulate(drive, inplace=False)` adds the read into post columns.
  `reset(batch, rows=None)`, `keep(rows)`, `to_dict()`. See [memory](memory.md) for stream
  identity, representation alignment, key interference, and checkpoint boundaries.
- `cadence.stream.columns(index)`: a slice when the neurons are one contiguous range,
  else the index array. Slices can avoid the copies required by advanced indexing.
- `PatternSeparator(inputs, expansion, winners, seed=0, center=0.0)`: pattern separation of
  the keys of a `FastSynapses` or `SynapticMemory`. `projection` is a fixed
  `(inputs, expansion)` Gaussian matrix from `numpy.random.default_rng(seed)`, divided by
  `sqrt(inputs)`; `mean` is the running key mean. `code(key, learn=False) -> (batch, expansion)`:
  with `center` above zero, `mean` is subtracted first, and `learn=True` first moves `mean`
  by the forgetting factor `center` toward the batch mean of the keys (writes learn, reads
  do not); the code keeps the `winners` largest entries of the rectified `key @ projection`
  and sets the rest to zero. `habituate(keys)` sets `mean` to the mean of a nonempty sample
  of keys. `to_dict()`; attributes `inputs`, `expansion`, `winners`, `seed`, `center`,
  `projection` and `mean`. Raises `ValueError` for nonpositive sizes, `winners` above
  `expansion`, `center` outside `[0, 1)`, and keys whose code overflows.
- `SynapticMemory(pre, post, decay=0.9, rate=1.0, amplitude=1.0, normalize=False, replace=False, rule="delta", separator=None, writes=0, consolidation=0.05)`:
  normalized delta synapses with a shared persistent `consolidated` matrix and per-stream
  effective `strength` matrices. `observe(key, value, write=None, *, salience=None,
  value_mask=None)` consolidates only observed values. Salience is a finite nonnegative
  `(batch,)` vector; the observed-value mask is boolean with the values' shape.
  `reset(batch, rows=None)` clears transient residuals and retains persistent synapses,
  including across batch changes. `clear()` erases both. Inherits `recall`, `read`,
  `stimulate`, `update` and `keep`; its rule is always delta. Reads never learn. A
  `separator` must have `center=0`.
  See [the equations and lifecycle](continuous.md#repetition-and-salience-become-lasting-synaptic-changes).

## Records (`cadence.records`)

- `Records(inputs, fields, *, cells=8000, active=40, rate=0.2, valued=(), valued_rate=1.0, habituation=1e-5, bias=0.3, pathways=(), pathway_rate=0.002, tasks=(), fan_in=0, seed=0, averaging=False, homeostasis=0.0)`:
  the records cortex over readings of `inputs` units. `fields` maps each predicted field to
  its width; of `cells` code cells, `active` stay per reading. `rate` is the write rate of
  the consequence fields (with `averaging`, each cell's rate is the larger of `rate` and one
  over the code mass written into it, so a fresh cell takes its first outcome whole);
  `homeostasis` is the rate at which each cell's activation share is tracked and its offset
  moved toward `active / cells` (0 leaves the offsets fixed); `valued` names the fields that read and write through the valued
  code, at `valued_rate`. `habituation` is the slowest rate of each unit's running mean (0
  subtracts nothing). `pathways` are one-dimensional index arrays into the reading whose
  running norms, moved at `pathway_rate`, equalise their say in the valued code; empty
  pathways are dropped. `tasks` indexes the reading's task units: the cells are divided into
  one group per task unit, and the valued code of a reading draws its winners from the group
  of the unit with the largest value, from every cell when no unit is positive. `fan_in`
  restricts each cell to that many pathways, drawn for the cell from the generator; the
  cell reads those and every input outside the pathways, and its projection column is
  rescaled by the square root of `inputs` over the inputs it reads (0 reads every input).
  `bias` scales the cells' fixed offsets. `seed` starts the `Mulberry32` generator that
  draws the projection, then the offsets, then the division into groups (a Fisher-Yates
  shuffle of the cells from `cells - 1` uniform draws), then the pathways of each cell in
  turn (a shuffle of the pathways from `len(pathways) - 1` draws per cell). Raises
  `ValueError` for a nonpositive `inputs`, `cells`, `active` or field width, `active` above
  `cells`, no fields, a valued name outside `fields`, `rate` or `valued_rate` outside
  `[0, 2]`, `habituation` or `pathway_rate` outside `[0, 1]`, a negative or nonfinite `bias`,
  a pathway or task index outside the reading, and fewer than `active` cells per task
  group. See [records](memory.md#records).
  - `code(readings, *, adapt=False, valued=True) -> ndarray`: the codes of `(batch, inputs)` readings, or
    of one `(inputs,)` reading, as `(2, batch, cells)`: the plain code, then the valued code.
    With `habituation` above zero, `adapt=True` first counts each reading in `seen` and moves
    `mean` toward it at the rate `max(habituation, 1 / seen)`, and every reading is coded
    with `mean` subtracted. A code keeps the `active` cells with the largest drive
    `reading @ projection + offset`, rectifies them and scales the row to unit length (a row
    with no positive drive stays zero). With `pathways`, the valued code divides each
    pathway of the mean-free reading by its running norm plus `1e-3`, and `adapt=True` first
    moves `pathway_norm` toward the norms of the readings' pathways; without pathways the
    valued code is the plain code. Witnessed readings adapt; imagined readings do not.
    Raises `ValueError` for a nonfinite reading or a wrong width.
  - `read(code) -> dict[str, ndarray]`: each field's read, the field's code times its table:
    `(width,)` for a `(2, cells)` code, `(batch, width)` for a `(2, batch, cells)` code.
    Valued fields read the valued code.
  - `write(code, targets, known=None) -> int`: the witnessed outcome of one reading, `code`
    of shape `(2, cells)`. Each field named in `targets`, a finite vector of its width, moves
    by `rate` (`valued_rate` for a valued field) times `outer(code, target - code @ table)`,
    through the active cells only; a field absent from `targets` is not written. `known`
    maps a field to a boolean mask of the observed target entries; the others have zero
    error. Returns the number of fields written and adds it to `writes`.
  - `parameters() -> int`: the record entries, `cells` times each field's width, summed.
  - `to_dict() -> dict`: the configuration (`inputs`, `fields`, `cells`, `active`, `rate`,
    `valued` as a sorted list, `valued_rate`, `habituation`, `bias`, `pathways` as lists,
    `pathway_rate`, `tasks` as a list, `fan_in`, `seed`); `Records(**records.to_dict())` rebuilds the
    same `projection`, `offset` and `task_of_cell`.
  - Attributes: `projection`, `(inputs, cells)` standard normal draws divided by
    `sqrt(inputs)`; `offset`, `(cells,)` standard normal draws times `bias`; `mean`, the
    `(inputs,)` running mean, zero at construction; `seen`, the witnessed readings counted
    into the mean; `pathway_norm`, one running norm per pathway, one at construction;
    `tables`, a dict from field name to its `(cells, width)` records, zero at construction;
    `writes`, the number of field writes; `tasks`, the task units, and `task_of_cell`, the
    `(cells,)` group of every cell (all zero without tasks). `mean`, `seen`, `pathway_norm`,
    `tables` and `writes` are the learned state. The configuration is readable as `inputs`, `fields`,
    `cells`, `active`, `rate`, `valued` (a frozenset), `valued_rate`, `habituation`, `bias`,
    `pathways`, `pathway_rate` and `seed` (reduced to 32 bits).
- `Mulberry32(seed)`: the 32-bit generator `mulberry` of the examples' viewer `brain_scan.js`, so a page draws
  the same numbers from the same seed; `state` holds the 32-bit state, the seed reduced to
  32 bits at construction.
  - `random() -> float`: one uniform draw in `[0, 1)`.
  - `batch(n) -> ndarray`: `n` draws at once, equal to `n` calls of `random` and leaving the
    same `state`; raises `ValueError` for a negative `n`.
  - `normals(n) -> ndarray`: `n` standard normal draws by the Box-Muller transform of
    `batch(2 * ceil(n / 2))`: the first half of the uniform draws gives the radii and the
    second half the angles, and the cosine values precede the sine values before the cut
    to `n`.

## Regions (`cadence.regions`)

- `Region(name, size=0, circuit=None, inputs=None, outputs=None)`: a named group of neurons.
  A blank region has a `size` and no synapses of its own. A designed region has a `circuit`
  (a `Connectome`) and takes its size from it; `inputs` and `outputs` name the circuit
  populations that receive and send projections, and default to the whole region.
  `designed`, `neurons(population=None)` (region-local indices), `to_dict()`.
- `visual_cortex(height, width, *, channels=1, features=8, field=3, stride=1, init=1.0, seed=0, name="visual")`:
  population `input` with one neuron per pixel and channel, in the order of an image array
  `(height, width, channels)` flattened row by row, and population `output` with `features`
  feature maps. Each feature neuron receives synapses from one `field` by `field` window of
  the input; windows step by `stride`. Efficacies start random and fan-scaled.
- `cortex(size, *, lateral=0.0, name="association")`: a blank region, or with a negative
  `lateral` a designed one whose neurons inhibit each other pairwise.
- `motor_cortex(actions, *, lateral=0.0, name="motor")`: population `actions` with one neuron
  per action and optional pairwise lateral inhibition.
- `prefrontal_cortex(holds, *, name="prefrontal")`: a blank region with one neuron per neuron
  of `holds` (a `Region` or a size), for a `Trace` from the held region.

See [write a cortex](cortex.md) for regions, projections, ports and learning heads.

## Brain (`cadence`)

`Brain.compose` is the default System 1 entry. Its animal-like foundation
includes continuing perception/action, plasticity and memory. Set `observers`
to positive region widths to add optional System 2 state feedback within the
same neural-graph settlement. This is an implemented interface, not a claim
that recursive benefit or automatic reflective behavior has been learned.

- `Brain.compose(inputs, actions, *, modules=(64,), observers=(), lateral=None, sensory_scale=1.0, seed=0, slots=1, **options) -> Brain`:
  the direct vector-input constructor. `slots` groups the motor neurons into several
  softmax readouts that settle together, a count of equal groups or one size per group
  covering `actions`: `act` and `step` return one index per slot, lateral inhibition
  stays within a slot, the unset `lateral` follows the largest slot, and `save`/`load`
  carry the grouping (issue 142). Positive `modules` widths form a reciprocal
  processing chain; the final module is the association cortex. Optional positive
  `observers` widths add regions with reciprocal state-reading and returning connections
  to every processing and motor region and earlier observers. The sensory, prefrontal
  and motor regions belong to that same graph; observers do not settle
  as separate controllers. Trace and consolidating SynapticMemory are enabled by
  default. `sensory_scale` is a finite nonnegative initial sensory-projection
  scale; it preserves the existing wiring at `1.0` and is construction-only.
  Constructor `options` accept memory/backend settings and every config field
  through `learning_<field>`, `actor_<field>` or `arousal_<field>`.
  `temperature` aliases `learning_temperature`, shared by teaching and the
  actor policy. Named overrides change only those fields; complete config
  objects remain full replacements. Unknown names, ambiguous bare rates such
  as `eta`, and duplicate aliases are rejected. A rate-only override keeps the
  resolved bias rate; pass `actor_eta_bias=None` or `learning_eta_bias=None`
  explicitly to derive one tenth of the respective rate. See the
  [composed defaults and rate ownership](brain.md#defaults-and-expert-overrides).
  This state feedback is distinct from exact error readback in
  `cadence.experimental.equilibrium`.
  The `lateral` option is the finite signed weight between each pair
  of distinct motor neurons; zero removes those connections while preserving
  reciprocal association/motor feedback. Left unset it resolves to -0.5 up to
  8 actions and 0.0 above: each action adds another inhibitory input per motor
  neuron, and measured composed brains with -0.5 stop settling undamped from
  about 12 actions (issue 124). An explicit value is used as given; compare
  observed operating points rather than assuming the same activity at every
  vocabulary size.
  `efference_amplitude` above zero (default 0.0, the released composition, byte-identical)
  appends an `efference` population of one neuron per motor neuron and a plastic
  `efference` to `association` projection at the working trace's scale 12; `efference_decay`
  (default 0.2) sets how the copy of the issued command fades. Every earlier region and
  projection is developed from the same random draws, so the brain with the copy is the
  brain without it plus those neurons and that projection. It is a gene; zero is its control.
- `Brain.build(inputs, actions, *, hidden=64, density=1.0, lateral=None, working_memory=False, memory_scale=12.0, episodic=True, features=8, field=3, seed=0, slots=1, **options)`:
  develops `Brain.genome(...)` and wraps it. `inputs` is a vector length, or an image
  shape `(height, width)` or `(height, width, channels)` for a `visual_cortex`. `options` go
  to the constructor.
- `Brain.genome(inputs, actions, ...) -> Genome`: regions `sensory` (or `visual`),
  `association` (`hidden` neurons), `motor` (`motor_cortex(actions, lateral=lateral)`) and,
  with `working_memory`, `prefrontal`; projections sensory to association (reciprocal for a
  visual cortex), association to motor (reciprocal), and prefrontal to association at
  `memory_scale`.
- `Brain(connectome, *, episodic=True, consolidation=0.05, memory_decay=0.9, memory_rate=1.0, memory_amplitude=1.0, working_memory_decay=0.2, working_memory_amplitude=3.0, working_memory_focus=0.0, efference_decay=0.2, efference_amplitude=0.0, learning=None, reward=None, resting_bias=0.0, slots=1, arousal=None, seed=0, backend="cpu", device=None, **genes)`:
  `arousal=True` enables the founder `ArousalConfig`; a mapping patches its
  fields, and an `ArousalConfig` supplies an explicit configuration. Omitting
  arousal preserves the brain without that controller, and `live` raises.
  `compose` and `build` pass these options through. Memory settings configure
  the existing trace/associative components; they do not add a missing population.
  At construction, settings for disabled or absent components have no runtime
  effect and are omitted from `describe()["genes"]`. `retune` rejects those
  settings when the component is absent.
  `resting_bias` is a finite nonnegative real scalar; booleans and arrays are rejected.
  It initializes named populations outside the `sensory`, `visual`, `prefrontal`,
  `efference` and `motor` families (the part of the name before `/`). Excluded family membership
  takes precedence over overlapping aliases; unnamed neurons also start at zero.
  In `compose`, modules, association and observers receive the value. In an image
  builder the whole visual region stays at zero. Biases remain plastic, and a
  positive initialization does not guarantee responsiveness or successful learning.
  `resting_bias` records the initialization choice separately from `brain.bias`;
  save/load preserves both without reapplying the initializer. Files without this
  metadata retain their stored bias vector and use `resting_bias=0.0`.
  The connectome needs populations `sensory` or `visual/input`, `association` and `motor`, and uses
  `prefrontal` for a working memory and `efference` (one neuron per motor neuron) for an
  efference copy when present. `efference_amplitude` is the read gain of that copy, a finite
  nonnegative real scalar; `efference_decay` its decay across moments. `learning` defaults to
  `LearnerConfig(beta=0.1, eta=0.5, temperature=0.2, tolerance=3e-3, free_steps=1024, nudged_steps=12, momentum=0.9)` (whose unset `eta_bias` derives `eta / 10` = 0.05),
  `reward` to `ActorCriticConfig(gamma=0.9, lam=0.8, eta=1.0, eta_bias=0.05, eta_critic=0.3, eligibility_steps=12)`,
  the measured composed bias rate; a bare `ActorCriticConfig` derives `eta / 10`.
  The live model and finite teaching phases retain `dt=1.0`. Qualified free
  settlement reserves roughly half its sweep budget for a numerical fallback:
  if the initial finite state does not qualify, continue with half the integration
  step and the remaining budget. Both phases together stay within the requested
  budget, and the final residual is recomputed against the original model. This
  preserves its equations and parameters; it is independent of System 2 wiring.
  With `learning.qualified=True`, independent recall and supervised learning
  instead use the configured `learning.damping` allowance. The separate reward
  eligibility mechanism retains its finite-phase contract.
  Attributes `connectome`, `brain`, `learner`, `basal_ganglia` (`ActorCritic` reading the
  association cortex), `working_memory` (`Trace` or `None`), `efference` (`Efference` or
  `None`), `hippocampus` (`SynapticMemory`
  from sensory to motor neurons, or `None`), `sensory_index`, `association_index`,
  `motor_index`.
  - `retune(*, reset_arousal=False, **genes) -> Brain`: change selected
    mutable genes in the same life. Accepts `learning_*`, `actor_*`, `arousal_*`,
    `temperature`, working-trace/efference settings and
    `memory_decay`, `memory_rate`, `memory_amplitude`, `consolidation`.
    `arousal={...}` changes only its named fields. Full `learning`, `reward`
    and `arousal` configuration objects replace that config before named
    overrides. Duplicate alias/mapping fields are rejected. Validation is
    atomic: an invalid request changes nothing. Successful calls preserve
    acquired parameters, optimizer history, activity, traces, records, random
    state and pending feedback. `reset_arousal=True` resets arousal's level
    and reward references, preserving its age and work counters. Retuning cannot
    alter topology, `sensory_scale`, backend or `resting_bias`, nor add a missing
    component. Actor rates apply to
    the next learned outcome, including one already pending; its eligibility
    and actual sampling temperature are retained. Changing `learning_beta`
    with pending sampled feedback is refused until the outcome is consumed.
    See [retuning a life](brain.md#retune-the-same-life).
  - `describe()`: detached, JSON-safe `layout`, effective flat `genes`, resolved
    `learning`/`actor`/`arousal` configurations, memory settings, `initialization`
    provenance and `pending_feedback`. Save/load preserves the metadata without
    reapplying initialization to learned weights. Older files with no composed
    layout provenance report `initialization["composition"]` as `None`.
  - `pending_feedback`: read-only boolean; the current issued action awaits
    its actual outcome, including a routine `live` choice. Use this when
    handling a refusal to avoid submitting an already consumed outcome twice.
    `wait` keeps it.
  - `decision_id`: read-only; the identity of the `live` action that owns the next
    outcome, which is the `arousal.age` once `live` issued it (its first action is 1),
    or `None` when no `live` action awaits one (an action `step` or `act` issued has
    none). `wait` keeps it, `reset` keeps the age so numbers are not reused, and
    save/load restores it. `live(..., decision_id=...)` refuses an outcome reported
    under any other value. Unlike the experimental population engine's
    [`Reinforcement.feedback`](equilibrium/LIVE.md#preserve-execution-and-feedback-ownership),
    naming the decision is optional, a repeated outcome raises instead of being
    acknowledged again, and the executed action is not reported.
  - `stimulus(observations, *, memory=True)`: the drive of a batch; with `memory`, the
    working memory, the efference copy and the hippocampal recall are added.
  - `step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None)`:
    the ongoing interaction API, returning the next sampled actions. Reward/done concern
    the preceding action; teacher labels concern the current observation. Omitted reward
    means a real zero-reward transition. Wait for the outcome before calling again.
    There is one operating mode; no training/inference toggle is needed.
    The first call cannot receive past-action feedback.
    `last_learning` exposes the previous transition's report, accepted `demonstrations`
    count and the complete teaching report under `demonstration_*` keys. A qualified
    teaching refusal retains its attempted work with zero accepted presentations,
    alongside any preceding actual feedback already accepted by this call.
    Supplied salience controls memory consolidation; by default it is absolute reward.
    If real feedback is learned but the next action refuses, that learning remains.
    Retry `act(observations)`; do not submit the same reward again.
  - `fit(observations, labels, *, epochs=30, batch=32) -> list[float]` (training accuracy per
    epoch), `predict(observations)`, `accuracy(observations, labels)`: independent samples,
    without trace or associative recall. All three score through qualified independent
    predictions; `fit` teaches with the configured finite or qualified phase contract.
    Predictions carry one choice per output slot, shaped `(batch,)` for one slot and
    `(batch, slots)` when the motor neurons split into slots, matching `act` and `step`.
    A refused epoch score leaves its already accepted teaching updates in place.
    These operations do not switch modes. `fit` resets pending stream
    state before its updates; use `live` for one continuing life, or `step` for
    explicit teaching, batches or learning from every outcome.
  - `imagine(observations, *, budget=1024, tolerance=1e-6) -> tuple[Equilibrium, ...]`:
    `observations` is a sequence of finite, nonempty batches with the same stream
    identities. Each possible observation uses its own bounded free solve, including
    the same numerical damping fallback, with a private trace
    carried through the branch and durable memory read without writes. Inspect each
    phase's `qualified` field; a refused phase ends the branch and remains in the
    tuple. The empty sequence returns `()`. Live activity, parameters, records, random
    state and pending actual outcomes remain unchanged. This predicts brain responses
    to supplied observations; use the separate `TemporalPatchNet.plan` interface for
    a learned external-world action/consequence model.
  - `live(observations, *, reward=None, done=None, decision_id=None) -> actions`: one moment of a
    continuing life on one stream, for a brain constructed with `arousal`. Reward and
    done concern the preceding action, as in `step`; omitted reward consumes a
    pending action as a zero-reward transition, not a missing outcome. A calm brain answers with the
    greedy choice of one qualified settle and changes no parameter, memory record or
    optimizer state; the eligibility of earlier sampled actions fades by one step
    (`ActorCritic.fade`), as it does between two outcomes that are learned from. Its
    forecasts for the preceding action are the critic's value when it acted and, when it
    has an associative memory, the record it held for that action in that situation
    (`last_arousal["record_error"]`); the outcome is measured against the value with the
    value of the present state and against the record as it is. Surprise or want raises the arousal (see
    [Arousal](#arousal-cadencearousal)); only the outcome of the brain's own greedy
    choice can surprise it or change its usual forecast error. Every actual reward
    updates its recent and long-run income, including sampled choices. An aroused brain
    samples its base policy, with one uniformly selected motor slot at
    `learning.temperature * arousal.heat`; other slots keep `learning.temperature`.
    When extra heat is zero no slot is selected. It keeps eligibility and learns from the outcome
    as `step` does; the outcome that woke a calm brain is written to its memory for the
    situation it was chosen in. An action sampled by `step` or `act` is adopted, so a
    bootstrapped brain continues without `reset`; this first adopted action is
    treated as an own choice for arousal, even if it was sampled away from the greedy
    choice. Another operation that acts on the
    stream between two `live` calls takes it over: feedback then needs a preceding
    action again. A refused forecast settle leaves everything unchanged, and the same
    call can be retried; if the answer refuses after the outcome was taken, the outcome
    stays learned and counted, and the retry is `live(observations)` without it.
    An unrepresentable arousal update raises `ValueError`. For a routine action
    its feedback remains pending; after a sampled action's feedback was learned,
    the error says that feedback was accepted and must not be submitted again.
    `decision_id` names the action the outcome belongs to (`Brain.decision_id`); a
    malformed value, or any number other than the awaited action's, raises
    `ValueError` before anything changes, so an outcome reported twice, or late for
    an action that was replaced, is not credited to the action now awaiting one.
    When the next observations come before the outcome, sense them with `wait`.
    See [routine and repair](continuous.md#routine-and-repair-live).
  - `wait(observations) -> None`: one moment of a `live` stream whose issued action
    still awaits its outcome. It settles the observation from the current activity,
    reading the working trace, the efference copy and associative recall, and
    advances the working trace to that state. No action is issued and no outcome is
    taken: the awaited action keeps its forecasts, eligibility and situation, and the
    outcome later given to `live` is credited to it as an immediate outcome would be,
    with the next state settled from the state sensed last; a routine forecast, `act`,
    `step` and `imagine` also start from that state, and a finished episode from rest.
    Parameters, the critic, eligibility traces, associative memory, random state, the
    efference copy, `last_arousal` and the arousal state are unchanged: arousal, its
    age, youth and `need` advance with live moments, and one outcome is one
    temporal-difference step however many moments were waited: an event-time rule,
    with a per-moment decay as a candidate gene against it. `last_settlement` reports
    the settle with `operation="wait"`; like the work of a refused attempt it is not
    part of arousal's counts, so add each wait's `steps` to them to total a stream's
    work. The caller decides which moments are waited and supplies the link between
    the late outcome and its action; a result that uses `wait` declares that link as
    supplied, and a delay whose credit the brain is to learn is reported through
    `live` moment by moment. Give the awaited outcome with an explicit `reward`: an
    omitted reward is a zero outcome. Without arousal or with more than one stream
    it raises `ValueError`, without an action awaiting its outcome `RuntimeError`;
    invalid observations raise `ValueError`, and a settle that cannot qualify raises
    `RuntimeError` and changes nothing but `last_settlement`. There is no deadline:
    `act` replaces the awaited action without learning from it, while `step`, like
    `live`, takes a sampled action's omitted reward as a zero outcome.
    See [outcomes that arrive later](continuous.md#outcomes-that-arrive-later-wait).
  - `last_arousal: Mapping[str, Any] | None`: an immutable snapshot of the latest `live`
    moment: `mode` (`"routine"` or `"aroused"`), `level`, `error` (the unsigned
    temporal-difference error of the preceding action against its forecast),
    `record_error` (the unsigned error of the record held for that action, `None`
    without a memory or an outcome), `surprise`
    and `want` (what that outcome added to the level), `temperature` (the exploration
    temperature), `temperatures` (an immutable tuple of actual temperatures per motor
    slot; both temperature readings are `None` in routine), `heated_slot` (the slot
    receiving extra heat, or `None` when no extra heat is applied),
    `learned` (a feedback update ran), `recorded` (the
    waking outcome was written to memory), `sweeps` (free-solve sweeps of the forecast
    and the answer) and `learning_sweeps` (eligibility and feedback sweeps). It is
    `None` after construction, reset or load and is excluded from save files.
    `arousal` holds the stream's `Arousal`, or `None`.
  - `act(observations, *, greedy=False, temperature=None) -> actions`: one row per continuing stream.
    `temperature` samples the settled motor state at another softmax temperature than
    `learning.temperature`, or takes one finite positive temperature per motor slot.
    Eligibility credits the policy that actually sampled the action; a greedy read
    ignores the supplied temperatures.
    The bounded free solve checks complete potential/adaptation equations, including
    observers, under `learning.free_steps` and `learning.tolerance`. Its numerical
    fallback changes neither the live model nor teaching phases. Cached states are
    freshly checked. Exhaustion raises `RuntimeError` before changing live activity,
    memory, random state or pending feedback; `tolerance=None` raises `ValueError`.
    Qualified actions advance the working trace. Keep batch row identities fixed;
    call `reset()` before introducing a different set of streams. Reward eligibility
    retains its separate finite nudged-phase contract.
  - `last_settlement: Mapping[str, Any] | None`: an immutable snapshot of the latest
    completed free-answer solve, including a refused one. Fields are `operation`
    (`"act"`, `"wait"` or `"predict"`), `scope="free_answer"`, `qualified` (whole-batch boolean),
    `row_qualified` and `residual` (per-row tuples), `max_residual`, `steps`, `budget`,
    `tolerance`, `residual_checks`, `damping_halvings` and `stagnation_checks`.
    `steps` includes numerical fallback and stays within the shared budget.
    `step` exposes its final `act`; batched `accuracy` and `fit` scoring expose only
    their last prediction batch. Invalid arguments and failures before a free-answer
    solve retain the prior report. `learn` and `imagine` do not replace it.
    The report holds no mutable state arrays, is excluded from save files and is
    `None` after construction, reset or load. It excludes reward eligibility,
    feedback, teaching and memory work. Inspect `last_learning` and opt-in
    [settlement records](#record-every-settling-step) for additional work; neither
    residual nor sweep count is physical energy or environmental prediction error.
  - `learn(reward, done, next_observations, *, bootstrap=None, salience=None) -> report`: the hippocampus records the
    reward of the chosen action for its situation, `done` rows reset their working memory,
    and the basal ganglia learn from dopamine. For truncated episodes `bootstrap` supplies
    the value of the old episode's final observation; `next_observations` holds the reset
    observation for ended rows. Reward and bootstrap are finite batch vectors; done is boolean.
    If the feedback solve refuses, hippocampal writes, terminal trace resets and actor
    changes are rolled back; the action remains pending. Adjust the solve and retry the
    same outcome. This differs from an accepted outcome followed by a refused next action.
    After `wait`, the next state settles from the state the stream sensed last; the
    critic's eligibility still uses the state in which the action was chosen.
  - `reset()` clears working state, action cache, eligibility and reward centering; hippocampal
    records and slow parameters are kept. A brain with arousal begins the next stream calm,
    with what it was used to cleared and its age and work counts kept.
    `parameters()` counts actor/critic parameters
    and the shared consolidated memory matrix; per-stream state is additional storage.
  - `save(path) -> Path`, `Brain.load(path, *, backend="cpu", device=None, precision=None)`:
    complete composition checkpoints, including both optimizers, critic, random state,
    stream traces, prepared state, both memory timescales and any action awaiting feedback.
    A brain with arousal saves it, with the action `live` issued and its forecast, under
    the format name `cadence-generic/3`; brains without arousal keep `cadence-generic/2`.
    A brain with an efference copy saves as `cadence-generic/4`, with or without arousal;
    `load` refuses a `cadence-generic/4` file without the copy's state and a copy under an
    earlier format name. A stream saved during `wait` stores the state it sensed under
    `cadence-generic/5`, with or without an efference copy; a stream that is not waiting
    keeps the earlier format names, and `load` refuses sensed state under another name.
    The archive is replaced atomically. A learner-only checkpoint
    is rejected by `Brain.load`; `Learner.load` can extract a learner from either.
  Observations must be a nonempty finite batch, with image dimensions flattened per row.
  `fit` rejects noninteger labels and mismatched batches before updating. It resets current
  action/working state but keeps episodic records. `brain` always returns the current
  `learner.brain`, including after learning. See [compose a brain](brain.md#brain).

## Arousal (`cadence.arousal`)

The arousal of one continuing stream: when a composed brain leaves routine and how it
returns. `Brain.live` runs it; the classes can also be used alone.

- `ArousalConfig(threshold=0.2, decay=0.9, tolerance=2.0, floor=0.1, fast=0.05, slow=0.005, heat=2.0, youth=100, value_surprise=1.0, record_surprise=0.0, need=0.0)`:
  the genes of the law, with the hand-set founders as defaults. `to_dict()` returns them;
  `ArousalConfig.space()` declares their space for [`genes`](evolution.md#any-genome).
  `fast` at `slow` removes the long-run want, a large `tolerance` removes surprise, `heat`
  at zero removes the wider exploration, each surprise weight at zero removes its
  channel, and `need` at zero removes the need's want. Construction rejects values
  outside their ranges, a negative `need` and `fast < slow`.
- `Arousal(config=None)`: the state.
  `outcome(error, reward, *, own=True, learned=True, record_error=None) -> (surprise, want)`
  takes the unsigned temporal-difference error of one outcome and its reward and applies

  ```text
  surprise = log(error / (tolerance * usual + floor * scale))   when positive, else 0
  want     = max(clip((longrun - recent) / scale, 0, 1), clip((need - recent) / need, 0, 1))
  level    = decay * level + (1 - decay) * (surprise + want)
  ```

  Each want term is zero when its denominator is zero; `need=0` preserves the
  previous law exactly. Here `usual` is the running size of the error; `recent` and
  `longrun` are the running reward at the `fast` and `slow` rates, corrected for their
  short history. `need` is the
  reward per moment the body requires; the share of it the recent reward leaves unmet is
  a want of its own, measured against the need and not against the spread, because a
  reward that comes once in `L` moments has a spread near `1 / sqrt(L)` while its mean is
  `1 / L`: a brain that loses such a reward falls short of its long-run reward by only
  `1 / sqrt(L)` spreads, and is short of its whole need. A need never habituates; a life
  that never paid wants from its first moment. `own` says
  the action was the brain's own best guess: only such an outcome can surprise it and
  enters `usual`. Every actual outcome updates `recent` and `longrun`, including
  explored actions, so want follows the income actually received while sampling.
  `outcomes` counts own outcomes used for the usual TD error; `rewards` separately
  counts all outcomes used for the two income averages. `scale` is the spread of the
  outcomes the brain has learned from, the running RMS distance of their reward from
  `longrun`: `learned` outcomes and the outcome that wakes the brain enter it, routine
  outcomes do not, so a long calm does not shrink it. Scaling reward, error and need by
  the same positive factor preserves the law within numerical precision away
  from its absolute `1e-12` surprise guard. With `need=0`, shifting
  reward by a constant also preserves it: the first outcome establishes the reward
  reference even when it comes from an explored action.
  These are properties of supplied scalar outcomes, not a reward-transformation
  guarantee for the complete learning brain. A positive need is a level of reward, so a
  shift changes what is unmet. A stream's first own outcome is no surprise.
  `record_error` is the unsigned error of the record the brain held for its chosen
  action against the outcome, when it held one; it has its own running size,
  `usual_record`, and its own count, `records`, and the moment's surprise is the larger
  of `value_surprise` times the value channel's and `record_surprise` times the record
  channel's. With the founders the record channel is measured and weighs nothing: on
  the [odour nursery](../benchmarks/reversal/README.md)'s development seeds it woke
  the brain sooner and left more lives searching too briefly.
  `aroused` is true while `level >= threshold` and for the first `youth` moments of the
  brain's life; `mode` names it. `heat` is `1 + config.heat * want`, the factor on the
  policy temperature for the one motor slot selected for extra exploration.
  Other slots keep their base policy temperature; every slot remains available to
  the uniform selection on subsequent moments. `lived(sweeps, learning_sweeps=0)`
  counts one moment: `moments` and `sweeps` per mode, `learning_sweeps` and `age`.
  `Brain.live` counts a moment when its action is issued; the work of a refused
  attempt is reported by `Brain.last_settlement` and `Brain.last_learning` and is
  absent from these counts.
  `reset()` begins another stream calm and keeps the age and work counts; it clears
  the income/error averages and their observation counts.
  `to_dict()` and `Arousal.from_dict(values)` carry the complete state.
  An outcome whose running statistics cannot remain finite raises `ValueError`
  without changing those statistics. New saves carry a `cadence-arousal/2` format
  marker and loading requires every saved gene and continuation field. Unversioned
  checkpoints with the complete pre-need gene set load with `need=0`, preserving
  their zero-need configuration; other missing genes are rejected. The previous
  `cadence-arousal/1` and unversioned saves admitted only own outcomes to income:
  loading preserves their accumulated readings and initializes `rewards` from
  `outcomes`. Future sampled rewards enter income under the repaired law, without
  resetting acquired parameters, memory, age or pending feedback.

The default law (`need=0`) is relative: it responds to outcomes that differ from what
the stream is used to. A brain whose life has always paid poorly, and whose youth has
ended, is not roused by it. A positive need adds a persistent want when recent reward
falls below it. In a world that does not change, noise in the reward rate still rouses the
founder genes for a small share of moments; the
[odour nursery](../benchmarks/reversal/README.md) reports it.

## Genome (`cadence.genome`)

- `Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)`:
  synapses from `pre` to a fraction `density` of `post`. An end is a region name, meaning
  the region's `outputs` (for `pre`) or `inputs` (for `post`), or `region/population`.
  `sign` is the mean sign (−1 all inhibitory, +1 all excitatory, 0 mixed); `scale`
  multiplies the fan-scaled magnitudes; `reciprocal` adds the reverse synapses with the
  same weights.
  `Genome(regions, projections, label="genome")`: a connectome before development; it
  checks that region names are unique and that every projection end names a region or one of
  its populations. `region(name)`, `to_dict()`, `Genome.from_dict(d, designed=None)` (a record keeps a
  designed region's circuit label and digest, and `designed` supplies the region by name).
- `develop(genome, seed=0) -> Connectome`: development, deterministic in the seed. Regions
  are laid out in order as contiguous populations named after them; a designed region adds
  its circuit's synapses and its populations as `region/population`; each projection is
  drawn between its ends.
- `cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=())`: one offspring
  (module level; `evolve` uses it). Designed regions and regions in `fixed` keep their size;
  each `(leader, follower)` pair in `tied` keeps the follower the size of the leader.
- `genes(space, *, rate=1.0)` returns a mutation over a dict genome for a declared space of
  `log`, `linear`, `int` and `choice` genes; see [any genome](evolution.md#any-genome).
- `evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map, report=None, mutate=None, grow=None, **mutation) -> Lineage`:
  selection under `fitness(connectome, seed) -> float`. Generation 0 scores the genome and
  `population - 1` offspring; each later generation scores `population` offspring of the
  `keep` best genomes of the generation before. Each life develops its genome at the seed
  `seed + 1000 * generation + index` and passes that seed to `fitness`; scores must be
  finite. `mapper` runs a generation's lives (a process pool's `map` needs a picklable
  `fitness`); `report` is called with the lineage after every generation; the remaining
  keyword arguments go to `mutate`. See [evolve a brain](evolution.md).
- `cadence.genome.Lineage`: `generations` (one record per generation with `generation`,
  `best_fitness`, `mean_fitness`, and `best` as `Genome.to_dict()`), `best` (the best genome
  over all generations, the earliest on a tie) and `best_fitness`.

## Timing (`cadence.timing`)

- `latency(decide, *, repeats=1000, warmup=20)`: time `decide()` `repeats` times; the median,
  90th and 99th percentiles and maximum in microseconds, the mean, `jitter` (p99 over p50
  minus one), and the voluntary and involuntary context switches during the measurement.
- `environment()`: machine, cores, Python, thread limits, pinned cores (Linux), load average,
  library versions.

## Neuron-by-neuron reference (`cadence.reference`)

- `cadence.reference.settle_neuron_by_neuron(connectome, neuron_model, stimulus, *, steps, log_gain=None, bias=None, efficacy=None) -> (trajectory, Ledger)` (module level; `conformance` uses it):
  one neuron at a time, each reading only its own state and its synaptic input, counting one
  transmission per declared synapse per step.
- `conformance(brain, stimulus, *, steps=60) -> dict`: the brain against the reference on
  the same stimulus; `max_abs_deviation`, the ledger, the backend.
- `cadence.reference.Ledger`: `declared_synapses`, `steps`, `transmissions`, `undeclared`;
  property `clean`, also reported by `to_dict()`.

## Protocols (`cadence.protocol`)

- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment")`.
- `Protocol(stimuli, rows, training=(), levels=Levels(), steps=60)`: `score(brain)`,
  `neurons_for(connectome, stimulus)`, `to_dict()`. `stimuli` maps each stimulus name to the
  populations it drives at full amplitude.
- `cadence.protocol.Levels(active=0.5, inactive=0.2, margin=0.15, sparse_min=0.005, sparse_max=0.2, densify_margin=0.05, specific_max=0.25, code_level=0.5)` (module level).
- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment", versus="")`: `versus` names the second stimulus a `specific` row holds the readout's code apart from; a training fact with a fourth element does the same.
- `cadence.protocol.shared_code(activation, other, members, level) -> float`: what the two codes (the `members` at or above `level` under each activation) share, as a share of their union.
- `evaluate_predicate(predicate, value, reference, levels=None) -> bool`; `PREDICATES`
  maps each name to its definition.
- `shuffled(connectome, seed, *, keep=None) -> Connectome`: the control.
- `select_gain(make_brain, protocol, grid, *, sparsity_cap=0.05) -> (gain, table)`:
  among admissible candidates, maximize training facts passed and break ties by smallest
  value. `make_brain` builds the brain for a candidate, so the candidate can be the global gain
  or any number a dictionary declares, such as one population's gain through
  `NeuralGraph(log_gain=...)`. `sparsity_cap=None` admits every candidate. An empty grid or no
  admissible candidate raises `ValueError`.

## Checkpoints (`cadence.checkpoint`)

- `save(learner, path, *, compressed=True) -> Path` and `load(path, *, backend=None, device=None, config=None, precision=None) -> Learner`,
  also as `Learner.save(path, *, compressed=True)` and `Learner.load(path, ...)`: one `.npz` file holding the
  connectome, every synapse's efficacy, every neuron's gain and bias, the neuron model, the
  configuration, the outputs and slots, the plasticity masks, tie groups, synapse rates,
  momentum and normalisation state, and the update count. `compressed=False` writes an
  uncompressed archive, faster for a very large brain. `backend` and `device` may differ from the
  saved ones; `precision` overrides saved precision and `config` replaces the saved
  configuration. `predict` and `free` read the parameters and update nothing.
  Separate `FastSynapses`, `Trace` and `ActorCritic` objects are not saved by this API.
  Use `Brain.save/load` for the full standard composition. Archive replacement is
  atomic, so a failed write leaves the previous checkpoint intact.

## Learning (`cadence.learning`)

- `LearnerConfig(beta=0.1, eta=0.2, eta_bias=None, centered=True, free_steps=100, nudged_steps=50, tolerance=1e-4, nudge="cross_entropy", temperature=0.2, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0, scale_cap=8.0, qualified=False, damping=3)`:
  `eta_bias` left as `None` derives `eta / 10` at construction; an explicit
  value is kept, and construction warns when a positive `eta` is below
  `eta_bias`, where the bias step dominates (issue 126).
  `scale_cap` is the magnitude a plastic synapse's efficacy may not exceed (every update clips
  to it); a smaller cap keeps a readout neuron out of saturation, where a nudge has no slope
  ([the latch](reward.md#traps-with-their-measurements)).
  `momentum` steps each synapse on a running average of its own contrast; `decay` shrinks every
  plastic synapse's efficacy and every plastic neuron's bias by that fraction on each update
  (a leak on the synapses, for streams).
  `normalize > 0` divides both efficacy and bias signals by their own
  bias-corrected running RMS of the raw contrast plus `normalize_floor`.
  Construction emits `RuntimeWarning` if normalization is enabled and either
  `eta` or `eta_bias` exceeds `0.05`; rates and defaults remain unchanged. The
  threshold is advisory, not a stability bound. See [normalized rates](learning.md#rates-under-normalization).
  `qualified=True` requires each free and nudged phase to meet the full equation
  residual at `tolerance` before `step` applies an update. It makes both step
  limits settle budgets, so construction also emits `RuntimeWarning` when
  `qualified=True` and `nudged_steps < free_steps`; budgets remain unchanged.
  See [qualified teaching budgets](learning.md#qualified-teaching-budgets). Direct `update` remains
  unchecked. `damping` is the maximum number of
  numerical integration-step halvings within each phase's existing sweep budget;
  the original model and its fixed-point equations are preserved. Qualified
  learning requires a finite residual tolerance.
- `Learner(brain, outputs, config=LearnerConfig(), plastic_synapses=None, plastic_neurons=None, reciprocal=True, tie_groups=None, synapse_rate=None, slots=1, updates=0, contrast_updates=0)`:
  `plastic_synapses` and `plastic_neurons` are bool masks over synapses and neurons; only those
  move and decay, so two learners can share one brain without one's decay eroding the other's
  synapses. With `reciprocal`, each reciprocal synapse pair shares one efficacy.
  `tie_groups` is an int per synapse (−1 for none); synapses in a group share one efficacy and
  move by the mean of their contrasts, which is how an embedding is shared across positions.
  `synapse_rate` is a nonnegative float per synapse that multiplies its step before tying
  (`None`: every synapse at one).
  Overlapping reciprocal/explicit ties form one group; arbitrary group IDs are compacted.
  Ties constrain increments, so initialize tied values equally to keep them equal. Frozen
  members keep their values, including under decay and clipping, and contribute zero to
  the group's mean increment. Reciprocal learning rejects ambiguous parallel pairs;
  merge those with `from_synapses`, or use `reciprocal=False`. Masks remain mutable;
  changing the original boolean array affects later updates. Rebuild the learner to
  change tie topology.
  `slots` splits the outputs into softmax groups (a count of equal groups, or one size per
  group); `updates` counts all applied updates, while `contrast_updates` counts only
  this learner's own optimizer history, excluding external reward/direct updates.
  - `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`,
    `targets(labels)`, `nudge_for(target, beta, weight=None)`;
  - `contrast(free, nudged, opposite=None) -> (per_synapse, per_neuron)`,
    `contrast_rows(free, nudged, opposite=None)` (the same differences per batch row),
    `update(free, nudged, opposite=None) -> {"scale_step", "bias_step"}`,
    `apply(delta_scale, delta_bias) -> {"scale_step", "bias_step"}` (a computed step through
    the masks, synapse rates, tying, decay and clipping),
    `step(drive, labels, warm=None, weight=None) -> (LearnedState, report)`;
    labels are integer indices within each output group: `(batch,)` for one group,
    `(batch, slots)` for several; `accuracy` averages all row/slot choices;
    Reports include attempted/accepted row presentations, all required phases' steps,
    residual checks, damping halvings and stagnation comparisons; finite lessons
    also report `free_budget_exhausted` (1.0 when the free phase used its entire
    budget under a movement tolerance, with a `RuntimeWarning`). `total_row_sweeps`
    and `total_row_residual_checks` multiply each phase's work by its batch size.
    These are computation counters, not a complete hardware-operation estimate.
    `LearningPhaseError` carries `phase`, `phases`, `report` and `hint` on refusal
    (`hint` names a known configuration cause, such as a qualified nudged budget
    below the free budget, and is otherwise `None`);
    parameters, optimizer history and update counts remain unchanged.
  - `calibrate(drive, *, level=None, grid=None) -> float`: choose the tested global
    synaptic gain by its free output operating point. Left unset, the target is
    the mean over rows and output slots of the highest output activation, at
    0.5; an explicit `level` targets the mean over every output neuron instead,
    which on a wide readout pushes all outputs up together (issue 125). Inputs
    must be a finite, nonempty drive batch. The default grid tries the
    current gain first, then its multiples `2**k` for `k=-8, …, 8` excluding zero,
    retaining representable positive candidates. Explicit grids retain their
    supplied values and order; all entries must be positive and finite.
    `qualified=True` uses full-equation solves with `free_steps`, `tolerance` and
    `damping`, excluding failed or nonfinite candidates; finite mode keeps finite
    settling and reports that qualification is not required. With no usable
    candidate, raise `RuntimeError` without changing the graph or optimizer.
    `last_calibration` reports every candidate and total attempted work, including
    refusals: sweeps, residual transports, stagnation comparisons and activation-
    cache checks, with batch-row totals, each candidate's `mean_output` and
    `top_output`, and the `target` in use. Calibration does not update learned efficacy/bias or optimizer
    history, and does not guarantee reaching the target or useful acquisition.
  - `predict(drive)`, `accuracy(drive, labels, batch=256)`,
    `parameters()`, `to_dict()`; attributes `brain`, `reverse` (index of each synapse's
    reverse, or −1), `second_moment` (when normalising).
- `calibrate_bias(brain, drives, targets, *, per_neuron=False, rounds=3, span=(-6.0, 6.0), iterations=16, steps=100, tolerance=1e-4, qualified=False, damping=3, report=None) -> np.ndarray`:
  candidate biases found by coordinate bisection under a finite, nonempty drive
  batch. `targets` maps population names or neuron indices to desired mean
  activations over their members and input rows. Use one shared bias per
  population, or one per member with `per_neuron=True`. Every population is
  visited in turn, `rounds` times, within `span`; targets outside the attainable
  range can leave an endpoint bias, and coupled populations can miss their targets.
  The original graph is unchanged; inspect the final means before installing
  the returned array through `with_parameters(bias=...)`.
  The `qualified=True` option requires every midpoint and the final
  candidate to meet the full original equations for every row, using each solve's
  `steps` budget, `tolerance` and bounded `damping`. Refusal raises `RuntimeError`
  without changing the original graph. This option does not inherit
  `LearnerConfig.qualified`. The default retains finite settling as a heuristic.
  An optional `report` dictionary receives attempted solve work, final observed
  means and target gaps, including refused work. Qualification certifies the
  states, not monotonicity, target attainment or useful learning.
  Finite searches also reject nonfinite or
  inconsistent midpoint/final states with `RuntimeError`, including when
  `report=None`. This validation does not require finite phases to equilibrate.
- `naive_efficacy(connectome, plastic) -> np.ndarray`: efficacies that give every plastic
  synapse class the same weight (sign times mean count over the class's count); the other
  synapses keep their sign. For a lesson that should start naive at a memory site whose counts
  are a specimen's memories.
- `preflight(brain, outputs, plastic, drives, *, level=0.5, saturation=0.02, shared_max=0.25, eligibility_min=10, steps=100, tolerance=1e-4) -> dict`:
  the checks a lesson needs before its first decision, under the drives the brain will decide
  in: a readout with no slope (within `saturation` of 0 or 1 under every drive), the plastic
  senders' code shared between two drives beyond `shared_max`, fewer than `eligibility_min`
  plastic synapses from active senders onto a readout, two drives that drive the same neurons
  beyond `shared_max` (the senses, not the wiring, are to be told apart). Returns the readings and `warnings`, one
  sentence per finding naming the block that repairs it; an empty list is what a receipt shows.
  These diagnostics use finite settling; an empty warning list is not a full
  equation certificate or evidence that a lesson will learn.
- `seam_report(connectome, pre, post) -> dict`: `classes`, `synapses`, `coverage_pre` (the
  fraction of pre neurons with a class onto the post population), `classes_per_post`,
  `median_count`; what custody left of a plastic seam.
- `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, skip_init=None, excitatory_forward=False) -> Connectome`
  with populations `input`, `hidden`, `output`. `skip_init=None` uses `init` for
  direct input-to-output projections. A finite nonnegative value overrides that
  scale; `skip=True, skip_init=0.0` adds trainable zero-efficacy projections while
  preserving the other effective parameters and initial predictions.
- `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Connectome, tie_groups)`:
  a window of one-hot tokens through one embedding table shared across positions, then a
  dense hidden layer and the outputs, feedback synapses tied in pairs; populations `input`,
  `embedding`, `hidden`, `output`.
- `LearnedState(free, nudged, opposite=None)`.

## The agent and the valence (`cadence.plasticity`)

- `ActorCritic(learner, critic, config=None, seed=0, population=None)`: the agent of a stream
  of moments, the composition of the elements (`reward.md`). `learner` is a `Learner` whose
  outputs are the action neurons; `critic` the neurons whose settled activation, read through a
  learned linear readout, is the expectation of the reward to come; `population` a `Bins` for
  a continuous action.
  - `act(drive, greedy=False, temperature=None) -> action`: one free phase, then a draw from the softmax
    over the output neurons (with `Bins`, one draw per dimension), or the most probable.
    `temperature` draws from the same settled outputs at another softmax temperature,
    or supplies one temperature per motor slot (per dimension with `Bins`). Values
    must be finite and positive and are checked before any state changes. Eligibility
    uses the same temperatures: its boundary drive is `c/T` times the categorical
    nudge, where `c` is the minimum of the base temperature and all supplied
    temperatures. The contrast therefore estimates `c * grad(log policy)` for the
    actual joint draw under the equilibrium assumptions. This keeps the original
    scale at the base temperature and bounds each mask gain by one, even for tiny
    temperatures. The supplied temperature is held fixed during the local phases.
    Cache reuse requires the same drive and unchanged brain parameters; caller buffers
    are copied. After learning, the next action refreshes its warm state. A greedy action clears
    pending eligibility and cannot be followed by `learn`. Repeated `act` replaces the
    pending decision. `Bins` requires at least two levels per dimension. Without
    `Bins`, `Learner(slots=[2, 3])` returns two categorical action indices per row;
    padding is never sampled. Actor nudges differentiate the softmax policy,
    independently of the learner's imitation loss;
  - `learn(reward, done, next_drive, bootstrap=None, *, observed=None, warm=None) -> report`: the prediction error
    `reward + gamma * V(next) - V(now)` made into the dopamine by the valence and written
    through every synapse's eligibility, the trace of the last act's contrast decaying by
    `gamma * lam` a moment. The critic uses its own trace and `critic_signal`: raw
    prediction error (`"td"`) or modulated error (`"modulated"`); the default `"auto"`
    is `"td"` whenever the dopamine is centred and `"modulated"` otherwise
    (`ActorCriticConfig.critic_target` is the resolved choice).
    Reports include absolute raw `td_error`, absolute modulated `delta`, signed
    `dopamine`, `capped` (the share of observed rows whose signal exceeded
    `dopamine_cap` before the clip; 0 without a cap), `saturation` (the fraction of
    output activations within 0.02 of 0 or 1,
    where a nudge has no slope) and `trace` (the mean absolute eligibility over the plastic
    synapses); a `saturation` near 1 with a `trace` near 0 is the latch of
    [learning from reward](reward.md#traps-with-their-measurements), and a `capped`
    near 1 means the actor learns from the sign of each outcome alone
    ([replaying a life](reward.md#replaying-a-life-through-step)).
    `done` rows start their next life from rest; a truncated row passes `value_of` its last
    observation as `bootstrap`;
    `observed` is a boolean batch vector for real transitions. Padding rows do not
    teach the actor or critic or enter reward statistics; their eligibility resets.
    Updates average over observed rows. At least one row must be observed.
    `warm` is the state the next state settles from when the streams' activity moved on
    after the action (`Brain.wait`); by default it is the free phase the action was
    chosen in, which always carries the critic's eligibility.
  - `fade(done=None)`: one moment passed that added no eligibility, its action having
    been answered greedily. Every eligibility trace decays by `gamma * lam`, the step
    `learn` applies between two sampled actions, and `done` rows forget their traces.
    Parameters, the critic and optimizer history are unchanged. `Brain.live` calls it
    for each routine outcome.
  - `reset()` (cached input/state, eligibility, salience and centering cleared; learned
    parameters and optimizer history retained), `probabilities(state, temperature=None)` (shape `(batch, actions)`
    or `(batch, slots, max_size)` for categorical slots, with exact zero padding;
    `(batch, dims, size)` with `Bins`; `temperature` reads the settled outputs at
    another softmax temperature than the learner's, or one per motor slot), `settle(drive)`,
    `value(state)`, `value_of(drive)`, `parameters()`, `to_dict()`; the attributes `valence`,
    `salience`, `delta_mean`, `delta_var`.
- `ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=None, eta_critic=0.05, normalize=0.0, momentum=0.0, dopamine_cap=1.0, dopamine_center=0.0, dopamine_floor=0.0, center_scale=True, critic_normalize=True, critic_signal="auto", eligibility_steps=None)`:
  `gamma` the discount and `lam` the trace's decay; `eta` and `eta_bias` the actor's rates,
  where `eta_bias` left as `None` derives `eta / 10` at construction, an explicit value
  is used as given, and construction warns when the bias rate exceeds a positive `eta`
  (issue 143);
  `eta_critic` the critic's; `normalize` and `momentum` the adaptive local step, as the
  learner's, with the actor's own history and a fixed RMS floor of `1e-3`.
  Construction emits `RuntimeWarning` if `normalize > 0` and either actor rate
  (`eta` or `eta_bias`) exceeds `0.05`, without changing the settings. This
  advisory check excludes `eta_critic`, whose normalization uses trace energy.
  See [normalized actor rates](reward.md#rates-under-normalization).
  `dopamine_center` the rate at which the reward's running level and scale follow
  it (0 for no centring), `dopamine_floor` the band around the level, in scales, within
  which the dopamine is zero, `dopamine_cap` its cap, `center_scale` whether the surprise is
  measured in scales of the usual (`True`) or in the reward's own units; `critic_normalize`
  divides the critic's step by its trace's energy. `critic_signal="td"` keeps the
  critic target in reward units; `"modulated"` may change its fixed point through
  clipping or centring; `"auto"` (the default) is `"td"` when `dopamine_center > 0` and
  `"modulated"` otherwise, because a critic fed the centred signal chases a moving target
  (measured: a value running to -15 within 300 decisions on the fruit fly's T-maze). See
  [the choice and its measured tradeoff](reward.md).
  `eligibility_steps` independently caps each finite reward-nudged phase. `None`
  inherits `learner.config.nudged_steps`; an explicit nonnegative integer, including
  zero, overrides it. The built-in `Brain` reward configuration uses 12 independently
  of supervised phase budgets. This does not qualify reward eligibility equilibria.
- `Bins(dims, size=9)`: the population code for `dims` continuous dimensions, each a softmax
  over `size` bins (`centres`, `groups`, `read`, `size`).
- `Valence(level=0.0, floor=0.0, cap=1.0, units=True, per_stream=True, mean=0.0, var=1.0)`: the
  reward less its expectation, made into the dopamine. Called on `delta` (a prediction error,
  or the reward alone), it subtracts the running level (`level` is the forgetting factor; 0 for
  none), kept per stream with `per_stream` or shared across streams otherwise; `mean` and
  `var` hold that running level and variance. The result is in the reward's own units or over
  its running scale (`units`), zero within `floor` scales of the level (quiet while the reward
  is what it usually is), and capped at `cap` (0 for no cap). `reset()`. `ActorCritic.valence`
  is the agent's, built per stream from `dopamine_center`, `dopamine_floor` and `center_scale`
  of its config; the cap is `dopamine_cap`, applied by `learn`.
  Calling `valence(delta, observed=mask)` excludes unobserved rows from the
  running statistics and returns zero for them. An all-false mask leaves it unchanged.
- `ActorCritic.state`: the free phase of the latest moment, the state `act` read or `learn`
  settled; `None` after `reset`.
- `ActorCritic.salience`: `(batch, neurons)`, set before `learn`; each synapse's eligibility is
  weighted by its pre neuron's entry (a `Trace.ringing`), so that what is still ringing is
  what a signal writes through. None by default.

## Certificate (`cadence.certificate`)

- `certificate(brain) -> Certificate`: the settling certificate of the free phase, from
  `row_mass(brain)`, `lipschitz_constant(brain.neuron_model)`, the model's `dt` and whether
  it has adaptation. See [the certificate guide](certificate.md).
- `Certificate(row_mass, lipschitz, dt, adaptation)` (frozen): properties `rate`
  (`1 - dt * (1 - lipschitz * row_mass)`), `certified` (`rate < 1` and no adaptation) and
  `mass_limit` (`1 / lipschitz`). `error_bound(movement)` is the remaining sup-norm distance
  `movement / (1 - rate)` from the last step's potential movement, per row, and
  `apriori_bound(first_movement, steps)` is `first_movement * rate ** steps / (1 - rate)`;
  both return infinity for an uncertified brain. `steps_for(change, tolerance)` returns the
  warm-start steps after a stimulus change of sup-norm size `change` (zero for no change)
  and raises `ValueError` for an uncertified brain. `to_dict()`. Construction requires
  finite values, a nonnegative `row_mass`, a positive `lipschitz` and `dt` in `(0, 1]`.
- `row_mass(brain) -> float`: the largest absolute incoming effective weight sum over all
  neurons.
- `lipschitz_constant(model) -> float`: the supremum of the activation's slope,
  `(slope / 4) * max(1 / (1 - rest), leak / rest)` with `rest` the rest emission.
- `EPStructure` (frozen), returned by `ep_structure`: `fixed_inputs`, `free_neurons`,
  `free_asymmetry` (the largest absolute difference between a free/free effective weight and
  its reverse, parallel synapses summed), `fixed_incoming_mass` (the largest absolute
  incoming effective weight of an excluded input), `adaptation` and `tolerance`; property
  `compatible` (both measures within `tolerance` and no adaptation); `to_dict()`.

## Atlas

The library supplies `record_settlements` for capturing actual graph iterations;
the application owns their display ([pages](pages.md)). Application demos live in
[cadence-demos](https://github.com/muellerberndt/cadence-demos).

## Receipts (`cadence.receipts`)

- `Receipt.build(kind, body, sources=()) -> Receipt`; `write(path)`; `Receipt.read(path)`;
  `Receipt.verify(path, *, sources=None, check=None) -> (ok, message)`; `to_dict()`.
- `canonical_json(value)`; `cadence.receipts.canonical_sha256(value)` and `cadence.receipts.source_manifest(files)` at module level.

## Optional task compositions

`from cadence.circuits import assemble, reflex_arc, imagine, Deliberator, ActivityMonitor` imports small
sensorimotor, counterfactual-search and self-reading compositions. See
[defaults and the thinking clock](continuous.md#defaults-and-the-thinking-clock) for budgets
and scheduling. These optional architectural
helpers compose ordinary neuron dynamics with explicit host-side orchestration.

`assemble(regions, synapses=()) -> Connectome` merges an insertion-ordered mapping of
region names to connectomes into one connectome. Each entry of `synapses` is
`(source_region, local_neuron, target_region, local_neuron, weight)`, a directed synapse
with count 1 and sign `weight`. Each region remains addressable as a population, and each
of its populations as `region/population`. Use the result with one `NeuralGraph` and concatenate
drives in region insertion order. Unknown regions, neurons outside their region, autapses
and nonfinite weights are rejected. The helper adds topology only; convergence depends on
the combined system.


- `Deliberator(actions, transition, evaluate, terminal, *, depth=6, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy)`:
  resumable iterative deepening over isolated futures. `start(live)` snapshots new input
  and replaces old work; `tick(nodes=128)` visits at most that many new positions and
  returns an isolated copy of the last completed `Deliberation`, or `None`.
  `pending` reports unfinished work; `pause()`/`resume()` preserve it; `cancel()` drops
  the continuation and obsolete result. `nodes` counts total work for this observation,
  `budget_exhausted` reports the hard limit, and `result` is the last completed depth.
  No work starts until the caller supplies input and ticks. Callbacks obey `imagine`'s
  isolation contract and remain fixed for one search; node limits do not preempt callbacks.
  This object stores no learned weights and writes no real-action feedback.
  See [defaults and scheduling](continuous.md#defaults-and-the-thinking-clock).
- `imagine(live, actions, transition, evaluate, terminal, *, depth=2, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy) -> Deliberation`:
  compare copied futures. Scores use the root actor's perspective; adversarial layers
  alternate min/max. Pruning uses alpha-beta bounds with fresh bounds per root action.
  `clone` must isolate mutable branch state; other callbacks must not mutate external
  objects. Invalid budgets/nonfinite scores raise. Exhaustion raises before the next
  transition and returns no partial ranking. `Deliberation.futures` are sorted by score;
  each `Future` holds `action`, `score`, `sequence`, `state`. `nodes` counts visited
  successor states, and `depth` records the requested horizon.
- `ActivityMonitor().read(activity, scores, *, pressure=0) -> Readback`: finite 1D vectors
  and pressure in `[0, 1]`. Returns `activity_change`, `ambiguity`, `pressure`, `uncertainty`,
  `request_more`, `state`. `reset()` clears its previous reading and state. This heuristic
  does not learn by itself and does not establish consciousness.
- `reflex_arc(axes=2)`: sensory error ports and opposing motor pairs; the application
  supplies body dynamics and interprets the motor readout.

## Record every settling step

`record_settlements(callback, *, label="")` captures calls made inside its context,
including calls inside `Learner`, `ActorCritic` and supplied imagination routines.
The callback receives a `SettlementRecord` after each call. It contains the
connectome, neuron model, effective weights, bias, drive, mask, nudge and full
potential, activation and adaptation histories. Histories have shape
`(steps + 1, batch, neurons)`: the first row is the initial state, then one row
for every actual iteration. Differences of successive potentials are the signed
local repairs. Even a zero-step call has its initial row.

```python
import numpy as np
import cadence as cd

brain = cd.NeuralGraph(cd.layered(4, 16, 2, seed=0), cd.learning_neuron_model())
records = []
with cd.record_settlements(records.append, label="observe and act"):
    result = brain.settle(stimulus={0: 1.0}, steps=32)
repairs = np.diff(records[0].potential, axis=0)
```

Recording is opt-in. It copies every neuron's state after every step and can be
expensive for large brains. Write bounded chunks from the callback rather than
keeping an entire long task in RAM. Callback diagnostics do not recursively
record themselves; nested recording contexts restore the previous callback on
exit. Callback failures propagate. State and parameter arrays own their storage;
subsequent learning cannot change those saved arrays. Treat the shared connectome
as read-only.

CPU recording uses the inspectable NumPy kernel rather than the fused path, so
round-off and wall time may differ. These are measurements of the recorded run.
Iteration traces are simulated neural activity.

`ActorCritic.learn` reports signed mean `dopamine` alongside the existing mean
absolute `delta`. For a one-stream agent it is that transition's signed,
centered and capped learning signal. It is a global modulation signal; spatial
neurotransmitter diffusion is not part of this model.


## `cadence.population`

- `PopulationPatch(inputs, hidden, outputs, *, instances, streams, seed=0, cells=4096, active=32, record_rate=0.5, habituation=1e-5, record_bias=0.3, slowest=2.0, groups=None, device=None, dtype=torch.float32)`: record patches in lockstep on a device; `from_patch(net, *, instances, streams, device=None, dtype=torch.float32)`; `imagine(inputs) -> {output, slow, read, hidden, code}`; `observe(inputs, target, *, rate=1.0, write=True) -> {loss, slow, hidden, residual, code}`; `code(readings)`, `read(code)`, `clear_records()`, `inherit(parents, *, sigma=0.0, generator=None)`, `parameters()`, `state()`, `memory_bytes()`.
- `device_of(name=None)`: cuda, then mps, then cpu.
