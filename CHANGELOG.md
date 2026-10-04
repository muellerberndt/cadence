# Changelog

## Unreleased

- Accumulate CUDA block transport directly into its destination to avoid a
  temporary product and a separate addition kernel per block. Add actual-device
  System 1 equation, gradient, refusal, memory and continuation checks, plus a
  source-bound CPU/CUDA runtime and memory comparison for issue 98.
  Preserve POSIX source keys in credit diagnostics and LF bytes in the frozen
  phrase fixture so the existing provenance checks also pass on Windows.

- Record the measured decline of a normalized composed default
  ([issue 131](https://github.com/muellerberndt/cadence/issues/131)): at the
  proposed `eta=0.003, normalize=0.99, momentum=0.9` for both composed
  learners, supervised acquisition contracts pass but the reward stream fails
  its re-adaptation contract (0.486 against 0.9 after a contingency change).
  Composed defaults remain unnormalized; normalized rates stay per-application
  settings behind the 0.72.1 construction warning. Documentation only; no
  default or equation changes.

## 0.73.0 — 2026-10-04

- Resolve the unset motor `lateral` of `Brain.compose`, `build` and `genome` by
  readout width: -0.5 up to 8 actions, 0.0 above
  ([issue 124](https://github.com/muellerberndt/cadence/issues/124)). Measured on
  composed brains, -0.5 settles a small action menu in the same few dozen sweeps
  as 0.0, while from 12 actions the undamped free solve stops settling and damped
  answers take about nine times the sweeps. An explicit `lateral` is used as
  given, and small-menu brains are unchanged.
- Calibrate to the competitive operating point by default
  ([issue 125](https://github.com/muellerberndt/cadence/issues/125)):
  `Learner.calibrate` left without `level` now places the mean top output per
  row and slot near 0.5 instead of the whole readout's mean, which on a 36-way
  readout had selected saturating gains (12, then 128). An explicit `level`
  keeps the mean target; reports carry `target`, `mean_output` and `top_output`
  per candidate. Single-output calibration selects as before.
- Derive an unset `LearnerConfig.eta_bias` as `eta / 10` at construction
  ([issue 126](https://github.com/muellerberndt/cadence/issues/126)); the
  standalone default stays 0.02 and the composed default becomes 0.05. An
  explicit value is kept, a resolved value rides through `dataclasses.replace`
  unless re-derived with `eta_bias=None`, and a bias rate above a positive
  synapse rate warns, since the bias step then dominates.
- Count and warn when a finite teaching free phase uses its entire `free_steps`
  budget under a movement tolerance
  ([issue 127](https://github.com/muellerberndt/cadence/issues/127)): the lesson
  was learned from a state that may not have settled. `Learner.step` reports
  `free_budget_exhausted` (`demonstration_free_budget_exhausted` through
  `Brain.step`); phases with `tolerance=None` remain declared fixed-length and
  silent. The finite update law is unchanged.
- Warn at `LearnerConfig` construction when `qualified=True` and `nudged_steps`
  is below `free_steps`, including through `dataclasses.replace` on a composed
  configuration: qualified phases are settle budgets, and the composed finite
  teaching default of 12 nudged sweeps refuses every realistic lesson
  ([issue 123](https://github.com/muellerberndt/cadence/issues/123)). A refused
  nudged or opposite phase under such a configuration names the budget mismatch
  in its `LearningPhaseError` message and new `hint` attribute. Document the
  budget semantics in the learning, composition and troubleshooting guides.
  Finite teaching, the composed defaults, the reward-eligibility contract and
  refusal transactions are unchanged; deliberately small qualified budgets
  remain allowed.

## 0.72.1 — 2026-10-03

- Warn at learner or actor-critic config construction when `normalize > 0` and
  either `eta` or `eta_bias` exceeds `0.05`; include the actor's independent bias
  rate and point the warning to the constructor caller. The diagnostic excludes
  the critic's separate rate and does not change optimizer equations or defaults.
- Document the RMS update, floor and momentum effects, independent bias rates
  and the limits of the reported Atari, Transcribe and Patch World pilots from
  [issue 131](https://github.com/muellerberndt/cadence/issues/131). The warning
  threshold and suggested development sweeps are not stability guarantees.

## 0.72.0 — 2026-10-03

- Add optional `resting_bias` to `Brain` and `Brain.compose`, with finite scalar
  validation, protected sensory/visual, working-memory and motor boundaries,
  and saved initialization metadata separate from learned biases. The default
  remains zero; responsiveness is not an acquisition or retention guarantee.
- Add the vanished-cue recall chamber (`benchmarks/recall/vanished_cue.py`, issue 84): a
  continuing brain's free recall of a cue across blank or distracting delays, against
  erased and shuffled trace controls and a separately trained history comparator.
  Freeze paired episodes, fork complete probe checkpoints, charge attempted
  teaching and action solves, and retain refusal and source records.
- Add the `lateral0-local-rms` and `lateral0-resting` candidate genes to the
  acquisition microscope, with explicit effective settings and unchanged
  first-refusal stopping rules.
- Correct acquisition evidence descriptions for effective rates, nudged budgets
  and accepted versus attempted updates. Document supervised-only continuing
  interaction separately from zero-reward transitions. Broader acquisition and
  recall acceptance remains open; no memory or learning default is changed.

- Expose immutable `Brain.last_settlement` diagnostics for successful and refused
  action/prediction solves: per-row residuals, qualification, sweeps, checks and
  damping. Keep diagnostic state outside checkpoints and preserve action and
  feedback transactions. The report explicitly excludes learning and memory work.
- Add a guided documentation entry and link previously orphaned guides. Clarify
  reciprocal composition, the equilibrium world-model hypothesis, separate
  transition predictors, concrete memory/readback mechanisms, configuration
  defaults and the scope of centered dopamine and `Life`.
- Give contributors and agents a task-to-guide map, continuing-brain construction
  recipe and explicit sleep/dream preservation guidance. Include `AGENTS.md` in
  source distributions and validate its documentation links.
- Extend the continuing example with witnessed corrective teaching, repair and
  continued use, reporting actual outcomes alongside free-answer and other
  settling work. Preserve saved pending-feedback continuation.

- Center introductory and contributor guidance on one continuing equilibrium
  brain across bootstrap, use, witnessed disruption and local correction. Add
  an executable world-model guide with explicit current integration boundaries;
  keep independent learning controls and distinct model families labeled.

## 0.71.1 — 2026-10-03

- Reject nonfinite or inconsistent states during finite bias calibration, even
  without a report. Check the final candidate before returning biases; an invalid
  solve raises `RuntimeError` without using its output to advance the search or
  changing the source graph. Finite searches still permit unsettled states.

## 0.71.0 — 2026-10-03

- Add opt-in qualified graph learning through `LearnerConfig.qualified`. The
  free, positive and required negative teaching phases must satisfy the original
  full equations before one local update. `LearningPhaseError` retains phase
  states and attempted work; refusal preserves parameters and optimizer history.
- Extend `NeuralGraph.equilibrate` with bounded numerical step halvings. Repeated
  complete-state checkpoints can move an unqualified attempt to a smaller step
  sooner, while every accepted state still meets the original residual and all
  attempts share the declared sweep budget.
- Report all teaching phases, original residuals, residual transports,
  stagnation comparisons and attempted/accepted row presentations. Preserve
  accepted and refused demonstration costs in `Brain.last_learning`.
- Make gain calibration honor qualified learning and reject invalid states.
  Its default grid spans the current gain's representable powers-of-two multiples
  from 1/256 to 256, trying the current gain first. Explicit grids retain their
  supplied order. `Learner.last_calibration` records every attempted candidate;
  an all-refused search preserves the graph and optimizer.
- Add opt-in qualification and reports to `calibrate_bias`, checking every
  midpoint and the final candidate before returning biases. Report observed means,
  target gaps and all solve work. Finite calibration remains available; a
  qualified operating point does not guarantee a requested target or acquisition.
- Expose motor competition through `Brain.compose(lateral=...)`, retaining the
  default per-pair weight of -0.5. Zero removes those lateral connections while
  preserving reciprocal processing/motor feedback.
- Separate reward eligibility duration with `ActorCriticConfig.eligibility_steps`.
  The Brain default stays at 12 finite nudged steps independently of supervised
  teaching budgets; standalone `None` retains learner-budget inheritance.
- Restore associative memories, separator state and terminal working traces when
  reward bootstrap qualification fails, preserving the actual outcome for retry.
  Accepted real feedback remains learned if a subsequent lesson or action refuses.
- Score `Brain.fit` epochs through qualified, memory-free public predictions.
  A refused score preserves lessons already accepted in that epoch.
- Add a runnable `Brain.compose` example and NumPy-only CI coverage for actual
  feedback, demonstrations, free recall and saved pending-feedback continuation.
- Add repository acquisition and retention instruments with source-frozen
  protocols, independent residual/contrast checks, charged rehearsal, preserved
  refusal/case censuses and saved continuation. Separate graph acquisition,
  working traces and consolidated associative storage.
- Clarify conditional gradient assumptions, independent learning-rate
  hyperparameters, standalone versus composed defaults, calibration limits and
  device execution. The adaptive `ActorCritic` optimizer uses host arrays;
  supported blocked PyTorch `Learner` updates remain on the device. Document
  the independent `eta_bias=0.02` and `temperature=0.2` defaults accurately.
- Add numerical and behavioral regressions for phase refusal, bounded damping,
  calibration admission, motor wiring, memory rollback and saved continuation
  across supported backends. Gradient checks retain their symmetry, smooth-branch,
  nudge-limit and loss-scaling hypotheses.
- Distinguish current application demos from archived research examples. Restore
  the recorded 0.61.0 release and 0.62.0 development history without inventing
  releases for unpublished version numbers.
- Update consumer guidance for explicit qualified protocols, calibration and
  refused-lesson retry. Correct Atari settings and pooled processing-time labels;
  retain historical finite-probe measurements and their source identity.

- Document focused local contract checks, separate foundation/population suites
  and a NumPy-only environment for shorter iteration. Keep full CI coverage,
  fixtures and assertions; optional backend skips remain explicit.

Finite supervised teaching remains the default. System 1 memory, plasticity,
private imagination and action remain available; optional System 2 continues
to join the same neural graph. Numerical qualification alone does not establish
general acquisition, lifelong retention or an efficiency advantage.

## 0.70.0 — 2026-10-02

- Make System 1 the default continuing brain, with working trace, fast and
  persistent associative memory, plasticity, action and private imagination.
- Provide `Brain.compose` for reciprocal base modules and optional
  System 2 observer regions within the same neural graph.
- Add `Brain.imagine` for private responses to supplied hypothetical
  observations. Learned environmental consequences and action planning use
  the temporal-model API.
- Qualify actions and independent predictions against the full state equations.
  Refusal preserves action state and pending feedback; real outcomes learned
  before a subsequent refusal remain learned.
- Use bounded numerical damping when a free solve needs it, then check the
  original model's residual. The total budget and finite teaching rule stay fixed.
- Provide event records and consolidation, learned temporal paths, continuous
  action planning and finite response protection through advanced APIs.
- Keep exact state-and-error feedback available in the advanced population solver.
- Simplify guides around current usage. This is experimental software; backward
  compatibility is not a design requirement. NumPy is required, with optional
  acceleration backends.

## 0.62.0 development revision — 2026-10-02

This entry records the development sources at
[`1f9daac`](https://github.com/muellerberndt/cadence/commit/1f9daac).
The recovered work became release 0.70.0; 0.62.0 was not published as a
GitHub release or on PyPI. Names below describe that development revision.

- Restore the capable pre-reset foundation from 930ee807: continuing
  `GenericBrain` interaction, `Trace`/`Afterglow`, consolidating
  `SynapticMemory`, record patches and sleep, temporal learning, private
  imagination, action planning and response protection. Preserve subsequent
  numerical, continuation and recursive-wiring hardening.
- Add `GenericBrain.compose` as a direct modular entry with working trace and
  consolidating memory, optional reciprocal observer regions, and the existing
  continuing interaction interface. Add private `GenericBrain.imagine` over
  supplied hypothetical observations; environment prediction remains the
  separate learned temporal-model contract.
- Qualify `GenericBrain.act`, `predict` and `accuracy` against the full state
  equations. Exhausted action repair preserves live state and pending feedback;
  consumed real outcomes stay learned if a following action refuses. Keep finite
  eligibility/training phases distinct from this free-answer qualification.
- Keep cortical observation optional. The foundation can already be deep and
  modular; observer feedback extends the shared graph rather than replacing
  working memory and learning with a narrower model.
- Preserve the newer state-and-error solver under
  `cadence.experimental.equilibrium`, with its own guides, examples and tests.
  Its sparse patch-connectivity checks and same-call stationary-evaluation
  optimization remain available there, without changing the restored APIs.
- Rewrite the entry guides around the biological-brain objective, working
  mechanisms and actual application source identities. The default package
  requires NumPy. Keep current GPL-3.0 licensing and historical attribution.
- Preserve original Amen, Connect Four and Atari checkpoints and browser
  engines. Library recovery, checkpoint parity and native application behavior
  require separate verification; no old receipt is silently promoted.
- Recover the capable foundation before releasing the narrower candidate as the
  default. Its separate numerical, CI and package evidence stays source-bound;
  the restored package requires its own verification.

## 0.61.0 — 2026-10-02

- Restore the principle as an enforced default: patches repair local
  disagreement to reach a coherent brain state, and further repair is driven by
  that state's mismatch with reality. `Cortex.build()` now refuses a layout in
  which a population settles with no other population, and a layout in which a
  group of populations settles apart from the rest. Every population must read
  another population's states or errors, or be read by one, and those reads
  must join all populations into one connected system; an unread sensors-only
  population or a disconnected group raises `ValueError` naming it. The
  smallest brain is two populations.
- Remove the input-only "flat" layout from the README, quickstart, layout and
  design guides, agent guides, examples and test fixtures. The layout example
  defaults to a two-population brain (`small`), with `deep` and the explicit
  `recursive` experiment. The query-cost and temporal-credit examples no longer
  build an input-only arm; their recorded receipts stay as recorded.
- Lead the README and the contributor guide with the main hypothesis and the
  simplicity premise, and contrast settlement with feed-forward backpropagation.
- The repair law, energy, qualification tolerance and admission contract are
  unchanged. `cortex.py` changed, so snapshots bind to this release's sources;
  snapshots saved by `0.60.0` load only in `0.60.0`.
- Error-reading observers remain experimental; this release still claims no
  automatic System 2, retained useful recursive correction or reproduced
  musical quality.

Earlier entries remain in the [source changelog before the guide simplification](https://github.com/muellerberndt/cadence/blob/1f9daac/CHANGELOG.md).
Version numbers 0.63.0–0.69.0 were not published; they are not missing release entries.
