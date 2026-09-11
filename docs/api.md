# API reference

Everything below is importable from `cadence`. Signatures are the current ones; the
docstrings in the source carry the details.

## Wiring (`cadence.wiring`)

- `Wiring.from_edges(n, *, pre, post, count=None, sign=None, sets=None, label="wiring", min_count=0.0)`:
  build from edge lists. Parallel overlaps merge (counts add), autapses drop, overlaps
  below `min_count` drop, and the arrays are sorted by `(post, pre)`. `count` defaults to
  1 and `sign` to +1.
- Fields: `n`, `pre`, `post`, `count`, `sign` (arrays), `sets` (name → tuple of owners),
  `label`. Property `edges`.
- `members(*names)`, `with_sets(**sets)`, `in_degree()`, `out_degree()`, `summary()`,
  `digest()` (SHA-256 of the sorted arrays).

## Rules (`cadence.rules`)

- `GradedRule(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, clamp_amplitude=3.0, adaptation=None, leak=0.0)`:
  the owner rule. `activation(v)` is the rectified, re-based sigmoid with optional leak;
  `slope_at(v)` its derivative; `rest_emission` the raw sigmoid's value at rest, which is
  subtracted so rest publishes zero. `replace(**changes)`, `to_dict()`.
- `Adaptation(tau_steps=50.0, strength=1.0)`: the slow per-owner variable that turns fixed
  points into rhythm.
- `learning_rule(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, clamp_amplitude=1.0)` (in
  `cadence.learning`): the rule settings a learnable net needs.

## Settlement (`cadence.settle`)

- `Settlement(wiring, rule, *, backend="cpu", edge_scale=None, log_gain=None, bias=None, device=None, dense_limit=2048, layout=None, precision=None)`:
  the engine. `edge_scale` defaults to the wiring's signs; `log_gain` and `bias` to zero.
  When the wiring's dense blocks fit in `dense_limit` squared entries the transport is the
  block transport (see `cadence.blocks`); `layout` passes a precomputed cut, as
  `with_parameters` does. `engine.layout` is the cut in use. `precision` (torch only) is
  `"float32"` or `"float64"`; the default is float64 except on MPS. Float32 is the speed of a
  consumer GPU; read out on the cpu backend for a receipt.
- `settle(clamp=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> SettledState`:
  one clamp; `clamp` is a list of owners at full amplitude, a `{owner: level}` map, or a
  dense vector. `settle_batch(drive, ...)` takes `(batch, n)` drives. Both stop early at
  `tolerance` and report the steps taken.
- `clamp_vector(clamp)`, `clamp_levels(levels)` (levels in [0, 1] times the clamp amplitude),
  `readings(state, names)`, `with_parameters(*, edge_scale, log_gain, bias)`, `weights`
  (effective drive per overlap), `dense()` (the `W[pre, post]` matrix), `to_dict()`.
- `SettledState`: `v`, `activation`, `adaptation`, `steps`, `trajectory`, `repair` (the total
  movement of the activations, per row); `row(i)`,
  `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`, `batched`.
- `Nudge(target, mask, beta, softmax_temperature=None, weight=None)`: extra drive
  `beta · (target − s)` on the masked owners, or `beta · (target − softmax(s/T))` over the
  masked group with a temperature; `weight` scales rows. `drive(s)`.
- `available_backends()`: `{"cpu": "numpy float64", "torch": "mps float32" | "cuda float64" | "cpu float64"}`.

## Blocks (`cadence.blocks`)

- `layout(wiring, *, max_pairs=256) -> Layout`: cut the owners into contiguous ranges at the
  boundaries of the wiring's contiguous named sets and pair the ranges that carry overlaps.
  A wiring with no contiguous sets, or one that fragments into more than `max_pairs` blocks,
  gets one block: the full matrix.
- `Layout`: `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`; `ranges`, `pairs`,
  `size`, `bounds(k)`, `sources()` (ranges that hear nothing), `flat(weights)`,
  `blocks(flat)`, `to_dict()`.
- `BlockTransport(layout, flat)`: `inbox(s)` for one settlement, reusing the product of every
  source range that did not move since the previous call.
- `block_contrast(layout, s_plus, s_minus)`: the learning rule's per-overlap contrast as one
  Gram product per block.

## Streams (`cadence.stream`)

- `stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0, context_init=1.0) -> (Wiring, tie_groups)`:
  `embedded` plus a `context` range of `hidden` owners that hear nothing and reach every
  hidden owner. Sets `input`, `context`, `embedding`, `hidden`, `output`.
- `Echo(wiring, decay=0.5, amplitude=1.0)`: the carried state of a batch of streams.
  `reset(batch)`, `clamp(drive)` (writes `amplitude * trace` into the context columns),
  `update(state)` (`trace <- decay * trace + (1 - decay) * hidden activation`), `keep(rows)`,
  `to_dict()`.

## Timing (`cadence.timing`)

- `latency(decide, *, repeats=1000, warmup=20)`: time `decide()` `repeats` times; the median,
  90th and 99th percentiles and maximum in microseconds, the mean, `jitter` (p99 over p50
  minus one), and the voluntary and involuntary context switches during the measurement.
- `environment()`: machine, cores, Python, thread limits, pinned cores (Linux), load average,
  library versions.

## Reference engine (`cadence.reference`)

- `settle_owner_by_owner(wiring, rule, clamp, *, steps, log_gain=None, bias=None, edge_scale=None) -> (trajectory, Ledger)`:
  one owner at a time, reading only its own row and its inbox slice, counting one delivery
  per declared overlap per step.
- `conformance(engine, clamp, *, steps=60) -> dict`: the engine against the reference on
  the same clamp; `max_abs_deviation`, the ledger, the backend.
- `Ledger`: `declared_overlaps`, `steps`, `deliveries`, `undeclared`; `clean` in `to_dict()`.

## Protocols (`cadence.protocol`)

- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment")`.
- `Protocol(stimuli, rows, training=(), levels=Levels(), steps=60)`: `score(engine)`,
  `clamp_for(wiring, stimulus)`, `to_dict()`.
- `Levels(active=0.5, inactive=0.2, margin=0.15, sparse_min=0.005, sparse_max=0.2, densify_margin=0.05)`.
- `evaluate_predicate(predicate, value, reference, levels=None) -> bool`; `PREDICATES`
  maps each name to its definition.
- `shuffled(wiring, seed, *, keep=None) -> Wiring`: the control.
- `select_gain(make_engine, protocol, grid, *, sparsity_cap=0.05) -> (gain, table)`.

## Checkpoints (`cadence.checkpoint`)

- `save(learner, path) -> Path` and `load(path, *, backend=None, device=None, config=None) -> Learner`,
  also as `Learner.save(path)` and `Learner.load(path, ...)`: one `.npz` file holding the wiring,
  every seam's scale, every owner's gain and bias, the rule, the configuration, the masks, tie
  groups, momentum and normalisation state, and the update count. `backend` and `device` may
  differ from the saved ones; `config` replaces the saved configuration (a frozen deployment
  passes `LearnerConfig(eta=0.0, eta_bias=0.0)`).

## Learning (`cadence.learning`)

- `LearnerConfig(beta=0.1, eta=0.2, eta_bias=0.02, centered=True, free_steps=100, nudged_steps=50, tolerance=1e-4, scale_floor=0.0, scale_cap=8.0, target_level=1.0, off_level=0.0, nudge="cross_entropy", temperature=0.2, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0)`:
  `momentum` steps each seam on a running average of its own contrast; `decay` shrinks every
  trainable seam and bias by that fraction on each update (a leak on the seams, for streams).
- `Learner(engine, outputs, config=LearnerConfig(), trainable_overlaps=None, symmetric=True, tie_groups=None, trainable_owners=None)`:
  `trainable_overlaps` and `trainable_owners` are bool masks over overlaps and owners; only those
  move and decay, so two learners can share one net without one's decay eroding the other's seams;
  `tie_groups` is an int per overlap (−1 for none); overlaps in a group share one scale and
  move by the mean of their contrasts, which is how an embedding is shared across positions;
  - `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`,
    `targets(labels)`, `nudge_for(target, beta, weight=None)`;
  - `contrast(free, nudged, opposite=None) -> (per_overlap, per_owner)`,
    `update(free, nudged, opposite=None) -> {"scale_step", "bias_step"}`,
    `step(drive, labels, warm=None, weight=None) -> (LearnedState, report)`;
  - `calibrate(drive, *, level=0.5, grid=None)`, `predict(drive)`, `accuracy(drive, labels, batch=256)`,
    `parameters()`, `to_dict()`; attributes `engine`, `reverse` (index of each overlap's
    reverse, or −1), `second_moment` (when normalising).
- `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, excitatory_forward=False) -> Wiring`
  with sets `input`, `hidden`, `output`.
- `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Wiring, tie_groups)`:
  a window of one-hot tokens through one embedding table shared across positions, then a
  dense hidden layer and the outputs, feedback seams tied in pairs; sets `input`,
  `embedding`, `hidden`, `output`.
- `LearnedState(free, nudged, opposite)`.

## Receipts and custody (`cadence.receipts`, `cadence.custody`)

- `Receipt.build(kind, body, sources=()) -> Receipt`; `write(path)`;
  `Receipt.verify(path, *, sources=None, check=None) -> (ok, message)`; `to_dict()`.
- `canonical_json(value)`, `canonical_sha256(value)`, `source_manifest(files)`.
- `Source(key, file, url, sha256, citation="")`, `fetch(sources, root, *, allow_download=False)`,
  `manifest(sources, extra=None)`, `sha256_of(path)`, `CustodyError`.
