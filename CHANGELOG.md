# Changelog

## Unreleased

- Add `Brain.wait` for an outcome that arrives after the stream has sensed more: an
  awaited action's custody for one `live` stream
  ([#111](https://github.com/muellerberndt/cadence/issues/111),
  [#122](https://github.com/muellerberndt/cadence/issues/122)). A waiting stream settles
  each observation and advances its working trace without taking an outcome; the
  awaited action keeps its forecasts, eligibility and situation, so the outcome later
  given to `live` is credited as an immediate outcome would be, while the next state
  settles from where the stream is. Wait settles are reported by `last_settlement`,
  outside arousal's counts. Add `Brain.decision_id`, the arousal age once `live` issued
  the awaited action, and `live(..., decision_id=...)`, which refuses an outcome
  reported under another identity before any change. A stream saved while waiting uses
  checkpoint format `cadence-generic/5`; lives that do not wait keep their formats and
  contents. `ActorCritic.learn` takes the state to settle the next state from (`warm`).
  The guide adds that the caller supplies a late outcome's link to its action, so a
  delay whose credit the brain is to learn goes through `live`; that a wait follows
  event time ([#116](https://github.com/muellerberndt/cadence/issues/116)); and how to
  total a waiting stream's work.
  Defaults, learning laws and existing calls are unchanged; no behavioral gain is claimed.
- Add the readout's intrinsic plasticity, two genes of `LearnerConfig`:
  `homeostasis_rate` (founder 0) and `homeostasis_target` (0.3). At every teaching or
  reward update, on the host and on the device, each output neuron's bias moves by the rate
  times the shortfall of its free activation below the target, so a readout neither
  saturates, where a finite nudge has no slope, nor falls silent, where every contrast is
  tiny. Reachable as `learning_homeostasis_rate` and `learning_homeostasis_target` in
  `compose` and `retune`, reported by `describe()`, saved with the learner configuration;
  `apply` reports `homeostasis_step`. The founder rate leaves every composed brain's
  behavior and continuation as they were; learning reports gain `homeostasis_step` (0 at the
  founder) and saved configurations the two fields, which earlier releases ignore. A learner
  with a positive rate is saved as `cadence-checkpoint/3`, which earlier releases refuse
  instead of loading it without the gene. Padding rows that `ActorCritic.learn(...,
  observed=...)` excludes do not set the readout's operating point, and a frozen output bias
  takes and reports no step. Demonstrated limitation: the recall
  chamber's readout saturates under continued teaching in two of five development founders
  and a third drifts to chance ([#84](https://github.com/muellerberndt/cadence/issues/84));
  with the rate at 0.1 and raw local steps, all four development founders tested end 64
  repeats at clean recall of 0.95 or better (`benchmarks/recall/credit_landing.py`
  receipts). The fresh confirmation recall/4 (`benchmarks/recall/protocol-finite-4.json`,
  founders 310 to 312) failed closure: two founders recall the cue at every delay of the
  prefix and one reads the latest of two cues, while the external-history control ends at
  chance in two founders under the raw rate and the paired and nuisance floors are missed;
  on top of the normalized recipe the gene does not hold the readout against the fixed-size
  pushes in three of four development founders. With the chamber's `surprise` rule, a lesson only on the
  rows answered wrong, the control reads 1.00 in every development founder and the
  vanished-cue recall comes and goes with more lessons (development receipts). The fresh
  confirmation recall/5 (`protocol-finite-5.json`, founders 313 to 315: the surprise rule,
  the gene, raw steps at eta 0.2 on a brain composed at sensory scale 2, selected by a scale
  by rate sweep on the development founders) failed closure as well, horizons none, 0 and
  0, with replacement and order failing in every founder: the measured limit of a trace
  written after every event with one fixed weight (`benchmarks/recall/README.md`). The
  chamber's `make_brain` takes `brain.sensory_scale` (founder 1). In the odour nursery
  (development founders 0 to 5, `benchmarks/reversal` receipts of 2026-10-11) the gene at
  0.02 keeps acquisition, reversal and return from an exposure of 300 trials and impairs
  reversal after 100: one life of six does not reverse and three of six keep a wrong greedy
  choice. No default changes; #84 stays open.
- Add the competing-skill ring, `benchmarks/competing`, a bounded table-world instrument for
  [#169](https://github.com/muellerberndt/cadence/issues/169): a policy acquired by the
  arena's founder brain in a nursery, then a ring that brings a competing skill, unavoidable
  burn, both or neither, with the issue's controls applied through `retune` on one saved
  nursery brain per seed, readings from saved copies with the working trace carried and
  reset, per-projection drift and per-synapse step consistency. Development readings on six
  seeds: the acquired skill is not lost at the gene step for a certain or a weak payoff; the
  smaller steps retain no more and hold the competing skill less; for a weak payoff a frozen
  greedy test loses answers through the context its trace carries. The ring may pay as the
  arena's melee does (`--ring-pay`, a signed contingency of its own; `--brain FIELD=VALUE`
  overrides the brain point; the ledger tallies the dopamine per skill): under that pay the
  loss of #169 reproduces at every actor step while the actor's efficacies barely move, the
  dopamine on the old skill's moments is zero-mean, and what the melee overwrites is the
  episodic memory's one-shot record of the chosen action's reward (`memory_rate` 1.0);
  without the memory the nursery acquires little, and with the memory writing an average
  (`memory_rate` 0.2 or 0.05) the loss halves. A slow set point under every parameter and a
  sign-consistency stiffness, both screened on unshipped library variants, do not hold the
  skill and were removed. No gate, default or mechanism changes.
- Add `docs/sequential-tasks.md`, what the repository's measurements establish about a chain
  of actions that collapses onto one action and the pattern they support (the chain's state
  in the observation, grading over labels, backward one-step lessons, the smallest passing
  budget, a smaller actor step after acquisition, the sampled policy read beside the greedy
  one, the critic's rate for delayed pay), and `benchmarks/chain`, the prerequisite-chain
  nursery ([#111](https://github.com/muellerberndt/cadence/issues/111),
  [#113](https://github.com/muellerberndt/cadence/issues/113),
  [#84](https://github.com/muellerberndt/cadence/issues/84)): a chain of four actions with its
  state observed or hidden, pay at the end, shaped or in backward lessons, the arena founder
  brain at its gene step, a tenth of it, tapered and with the readout gene, against uniform
  random, a frozen newborn, the stationary memoryless ceiling and a tabular learner with the
  same information; readings from saved copies every two thousand moments. Development
  receipt on six seeds: with the state observed and pay at every step the founder brain
  completes the chain at 204 per thousand moments against the tabular learner's 250; with
  pay at the end its greedy policy reads uniform random's rate (42) while its sampled
  policy reads 77; with the state hidden no arm beats the stationary ceiling; the readout's
  intrinsic plasticity at its development point lowers every reading under reward, so the
  gene as declared is a teaching-time mechanism. No gate, default or mechanism changes.
- Add `benchmarks/recall/credit_landing.py`, development diagnostics for
  [#84](https://github.com/muellerberndt/cadence/issues/84) on the finite recall chamber's
  recipe: what an untrained founder's trace carries, and where each lesson's credit lands.
  Every founder tried reaches clean recall of 0.95 or better on a saved copy at some
  evaluation; in two of five the motor readout then saturates at full activation, where the
  finite nudge moves nothing and the efficacy steps fall a hundredfold, and a third drifts
  to chance: the founder dependence is consistent with a readout operating point. Chamber-
  level probes of a fixed and a homeostatic motor bias are diagnostics, not library
  mechanisms; the library gene above is the mechanism they motivated.

## 0.80.0 — 2026-10-09

- Simplify the continuing-brain interface for
  [#170](https://github.com/muellerberndt/cadence/issues/170) and
  [#176](https://github.com/muellerberndt/cadence/issues/176): `arousal=True`
  selects the founder controller, `sensory_scale` exposes the initial sensory
  projection, and `learning_*`, `actor_*`, `arousal_*` plus named memory settings
  override individual genes without replacing unrelated configuration values.
  Preserve all numerical founders and the composition when no options are passed.
- Add atomic `Brain.retune(...)` for changes within an acquired life, preserving
  parameters, optimizer history, traces, records, randomness and pending
  feedback. A reset of arousal is explicit; topology and initialization remain
  construction choices. Refuse a nudge-scale change while sampled feedback is
  pending. Add read-only `pending_feedback` for outcome-safe retries and
  JSON-safe `describe()` for layout, effective settings and saved initialization
  provenance. These are interface changes, with no new learning law.
- Normalize NumPy integer and floating configuration scalars so arithmetic stays
  consistent across save/load, and serialize scalar checkpoint metadata. Validate
  the boolean `centered` teaching setting, including when loading a checkpoint.
  Preserve accepted settings and continuation while rejecting malformed values.
- Lead guides with `compose` and `live`, centralize composed defaults and rate
  ownership, and use public overrides for expert recipes. Address
  [#171](https://github.com/muellerberndt/cadence/issues/171) through the documented
  founder/control distinction: the current efference default is zero and the
  actor rate is 1.0; historical arena settings do not define those defaults.
  Keep arena, C64, Atari and Amen model/experience choices distinct. Smaller arena
  rates remain bounded application measurements, not general default improvements.

- Add the native sensory-history rhythm instrument with matched controls, continuation
  checks and source-bound receipts. The period-four component passes; the C64 transfer
  remains a failure under [#140](https://github.com/muellerberndt/cadence/issues/140).
  Runtime laws and composed defaults are unchanged.
- Clarify that arousal gates learning without guaranteeing retention of earlier skills.
  Keep the bounded arena measurements and candidate mechanisms in
  [#169](https://github.com/muellerberndt/cadence/issues/169), with smaller actor steps as a
  measured control. Distinguish learner checkpoints from a complete composed life and
  require workload parity checks when changing machine, backend or precision.
- Add `MANIFESTO.md`, the design points in one page, with the dedicated language goal:
  whole consistent sentences formulated from the settled state, tracked in
  [#173](https://github.com/muellerberndt/cadence/issues/173).
  Include it in source distributions and the documentation link checks.
- Require every change to meet the manifesto through `CONTRIBUTING.md`, agent
  guidance and the PR template. Keep design requirements and review checks in sync,
  distinguish goals from implemented capabilities, and check contribution-guide links.

## 0.79.0 — 2026-10-09

- Apply extra arousal heat to one uniformly chosen motor slot per moment
  ([#159](https://github.com/muellerberndt/cadence/issues/159)); other slots sample
  the learned base policy. Preserve the one-slot sampling law and existing genes.
  Report actual per-slot temperatures and the heated slot, and preserve random state
  across refused answers and saved continuation.
- Credit the policy that actually sampled a reward-bearing action
  ([#160](https://github.com/muellerberndt/cadence/issues/160)). Scalar or
  per-slot behavior temperatures now reach the local actor nudges, with one common
  score scale and bounded gains. Preserve the base-temperature update, actual-action
  custody and pending saved continuation; explicit hot exploration no longer
  differentiates a different, sharper policy.
- Fix income freezing during exploration in multi-slot `Brain.live` lives
  ([#158](https://github.com/muellerberndt/cadence/issues/158)). Every actual reward
  updates recent and long-run income; surprise and its usual error remain limited
  to the brain's own greedy choices. Keep configuration defaults and action credit unchanged.
  Save the separate income count and migrate earlier checkpoints without resetting
  their acquired state.
- Retain bounded arena and nursery preservation evidence, including failed
  comparisons. Five nursery founder pairs retain their original gates, with slower
  median reacquisition. In one 80,000-moment arena pair both brains retain
  observation-dependent behavior; the corrected policy has weaker closing and
  combat scores than the earlier runtime. Continuation of six acquired fighters
  also has weaker closing and damage. These checks establish neither general
  combat improvement nor recovery of every previously acquired policy. Publish
  the receipts, source identities and limitations with the mechanisms they test.
- Preserve the completed nine-life arena confirmation: seven lives improve
  progress against eight required, seven pass the attack requirement, and six
  pass both. All nine preserve actual-outcome custody and exact saved continuation.
  The original physics and failed readiness result remain identified in the
  [confirmation archive](benchmarks/arena_confirmation/README.md).
- Retain the later [memory](benchmarks/arena_memory/README.md) and
  [reward-phase](benchmarks/arena_phase_budget/README.md) comparisons, including
  their losses of acquired attack performance. Those experimental changes are
  excluded from this release. Longer arena training remains an experiment;
  the fixes do not establish reliable fighting across founders.

## 0.78.0 — 2026-10-08

- Add the period-four loop chamber for #140, retaining the declared protocol and
  original receipts. Its passing `copy` arm uses an external sensory-history
  adapter; the native own-command and trace-only controls fail in the original
  freeze. Label selective teaching as an application policy. Require every prime,
  matched untaught founders, complete controls, no refusals and saved continuation;
  independently verify raw answers, scores and work. This bounded adapter result
  does not close native closed-loop generation. A separate trace-only development
  comparison finds learned prime-dependent playback on one founder at two existing
  trace settings; the simplest setting passes only 1/6 development founders after
  a predeclared extension, so fresh confirmation is not attempted.
- Correct the finite recall worker to honor its declared lesson count. Historical
  freezes 2 and 3 actually ran 192 episodes per founder, not the declared 384;
  preserve those authentic receipts as protocol deviations. Add strict census,
  score and work checks and retain the original `every` teaching behavior. Report
  corrected measurements separately. Full 384-episode correction audits complete
  on all six reused founders: protocol 2 horizons are none/0/0, protocol 3 has no
  passing prefix, and every founder fails nuisance acceptance. Both receipts verify;
  #84 remains open. Library behavior, interfaces and defaults are unchanged.
- Retain and replay raw recall-trace transitions, measure separation and motor
  margins, compare equal-time histories with different event counts, and enforce
  founder time limits while preserving unfinished work. Losslessly pack the evidence
  after retaining an oversized incomplete attempt; all corrected audits fit the
  unchanged 160-MiB cap. Add the missing shuffled-time
  and untaught controls to the native period-two rhythm instrument, with explicit
  actuator timing, recovery and continuation bounds declared before confirmation.
  Charge saved-copy work and remove a redundant, previously uncounted replay.
  Complete the missing paced recurrent-control comparison: all 16 cases meet
  the unchanged bounds, while the brain's two development attempts still fail
  physical timing. Verify rounded display fields at their declared precision;
  all acceptance bounds continue to use raw timestamps.
- Add key-door/3's third, returning rule, longer corridor, variable delays and
  recurrent control for [#111](https://github.com/muellerberndt/cadence/issues/111).
  The original 300 lives failed the declared acceptance gate. Return performance
  includes renewed learning, so it measures reacquisition rather than preservation
  without learning. The pouch-blind arm is a development lead, not evidence that
  the working trace retained the key. Original protocols and receipts remain intact.
- Correct the recurrent control's feedback/gradient ordering and clip its complete
  actor and critic gradients. Independently verify recorded trip statistics, probe
  probabilities and work, and require a complete census and the same lives to meet
  all acceptance criteria. Distinguish historical results from corrected audits.
  Add a separate development assay of the stable door skill before and after
  competing experience, on saved copies with learning disabled. Library behavior,
  public interfaces and composed defaults are unchanged.

## 0.77.0 — 2026-10-08

- Correct the reward-rhythm instrument's saved world boundaries, refusal retries
  and work accounting. Require the same founders to satisfy every acceptance
  criterion, retain missing controls and unfinished runs in the denominator, and
  verify receipt arithmetic against recorded events. Preserve the original
  protocols and receipts; corrected reruns have separate source-bound receipts.
  Align the runnable example's income and alternation windows. Library behavior,
  public interfaces and composed defaults remain unchanged.
- Add the reward-rhythm chamber (`benchmarks/rhythm/reward_rhythm.py`, protocols
  `protocol-reward.json`, `protocol-reward-2.json` and `protocol-reward-3.json`):
  the beat paid by the world. One
  continuing `Brain.live` life, the same drive every moment, one unit for a step on the
  other foot than the last, nothing for a repeat, paid at the next moment; no teacher. The
  walker carries the efference copy; arms without it, on the composed reward defaults, always
  learning, at the control eligibility decay, frozen, a tabular learner given the same one
  bit, and uniform random. reward/1 (key-door operating point, need 0.5) failed its gate on
  fresh seeds 301 to 305: 3/5 acquired, one of them from birth, two limping at two changed
  steps in three. reward/2 (eligibility decay 0, a gate on learned beats against the frozen
  founder) passed on fresh seeds 401 to 405: 4/5 learned, none from birth, all five calm in
  the window and continuing identically from a mid-life checkpoint; the walker without the
  copy earns occasional rewards while exploring but its greedy probe holds one foot.
  The limp, a period-three attractor
  paid above the need, is the residual failure; reward/3 (need 0.9, fresh seeds 501 to 505)
  removes it on all five founders but fails its learned-credit gate on two with the beat from
  birth. The receipts are taken on the released 0.76.0. `examples/walking_for_reward.py` shows
  one founder with and without the copy. No default changes.

## 0.76.0 — 2026-10-08

- Give the introductory guides one recommended continuing-life loop:
  `Brain.compose(..., arousal=ArousalConfig())` followed by `brain.live(...)`.
  Use the composed defaults in the starting example, explain diagnostic reports
  separately, and keep explicit teaching/batching in the `step` reference.
  Executable examples check arousal and saved continuation. Runtime defaults
  and supported operations are unchanged.
- Harden the opt-in efference copy at episode and checkpoint boundaries: routine
  terminal forecasts clear both traces before choosing the next action and restore
  them on refusal; `resting_bias` leaves command neurons silent until a command;
  loading rejects invalid command ports, read gains and missing continuation state.
- Add the efference copy, `Brain.compose(..., efference_amplitude=..., efference_decay=...)`
  and `cadence.Efference`: a trace written from the command the brain issued, one
  `efference` neuron per motor neuron, read by the association region through a plastic
  projection at the working trace's scale. The working trace copies the settled state
  before the decision, so a brain whose observations do not change could not learn what
  to do after what it did; the steady-rhythm chamber of
  [#116](https://github.com/muellerberndt/cadence/issues/116) recorded that limit at
  0.37 to 0.44 alternation against 1.00 for a control given its own previous action. The
  default `efference_amplitude=0.0` builds the released composition byte-identically; the
  copy is a gene with zero as its control. `act`, `step`, `live`, `learn`, `reset`,
  `imagine` (a private copy issued the imagined response's own best guess), `save` and
  `load` carry it; a checkpoint with the copy is `cadence-generic/4`, and checkpoints
  without one keep their formats. `predict` and `accuracy` ignore it as they ignore the
  working trace. The steady-rhythm chamber gains `protocol-2.json` (rhythm/2): the copy at
  amplitude 3.0 and decay 0.0 on the rhythm/1 selected recipe, against that recipe as its
  control, on fresh seeds 201 to 205, with the erased, shuffled, static and reset controls
  acting on the trace and the copy. Confirmation (`results/confirmation-2-2026-10-08.json.gz`,
  verified): mean intact alternation 0.97 (`every`) and 0.93 (`mismatch`) against 0.50 and
  0.40 for the control, 1.00 for the flip-flop and 0.50 for uniform random; four of five
  founders at 1.00 in the `every` arm, recovery after disturbances in 16 of 20 rows
  against 0 to 1; per-event actions identical across cadences, budgets of 64 and above,
  tolerances and host load in 20 of 20 founders; 128 refusals at a 16-sweep budget and
  more teaching sweeps with the copy. The development grid (seeds 0 to 5, every cell
  kept in `results/development-efference.json`) gives 1.00 on all six founders in the
  `every` arm against 0.78 without the copy. `examples/walking_rhythm.py` shows one
  founder with and without it. The copy is a gene with zero as its control, not a default.
- `Brain.predict` and `Brain.accuracy` now honor slotted motor readouts (`slots > 1`):
  predictions are one choice per slot, shaped `(batch, slots)` like `act` and `step`,
  instead of a single argmax over the whole motor menu, and slotted `accuracy`/`fit`
  no longer fail with a broadcast error after teaching. One-slot brains are unchanged.
- `ArousalConfig` gains `need`, a nonnegative required reward rate per moment. Its
  unmet fraction supplies an additional arousal signal even when poor rewards are
  expected. The default is zero, preserving the previous reward law. This is an
  opt-in scalar need in the existing System 1 lifecycle, not a general goal detector,
  learned goal representation or default promotion.
- Keep the need calculation finite for extreme finite rewards and need values by
  clipping before subtraction or division. New saved arousal state carries the
  `cadence-arousal/1` marker; complete legacy configurations without `need` load
  with `need=0`, while an incomplete marked configuration is rejected. This
  preserves continuation of existing zero-need lives without treating missing
  fields in new checkpoints as valid defaults.
- Add the key-door nursery (`benchmarks/keydoor`) for
  [#111](https://github.com/muellerberndt/cadence/issues/111), roadmap row 07. One
  continuing life encounters delayed door rewards and a moved key source, with
  always-learning, zero-eligibility, retimed-own-reward, frozen, no-pouch, tabular
  Q(lambda) and random controls. Two historical `key-door/1` freezes failed their
  declared gates. The second records acquisition in 20/20 gated lives, re-adaptation
  in 19/20, frugality in 19/20 and calm in 17/20 against 18 required; delay 10 is
  ungated. These are exploratory results at selected settings: shared random draws
  made cut and food schedules differ between arms. The no-pouch result does not
  identify trace memory; a scarcity variant's unchanged 90%-fed gate is unsuitable
  when successful door openings pay with probability 0.5. Moving the key does not
  test changed goals, and varying lever counts does not validate physical-time
  semantics. Broad delayed-credit and reuse acceptance remains open.
- Correct the key-door instrument under `key-door/2`: separate world and yoked
  random streams, draw food availability independently of actions, include every
  pre-door cut position, measure door openings within the last 50 completed trips,
  probe the actual phase end, record undelivered final feedback and yoked rewards,
  count probe memory reads separately, and retain completed forecast work when the
  following answer refuses. Preserve all historical receipt and
  protocol bytes; corrected runs are not frozen confirmation of that old protocol.
  Verification labels legacy limitations and reporting verifies before rendering.
  No corrected confirmation campaign or passing capability gate accompanies these
  fixes. Historical routine latency is a within-life laptop observation, not
  matched-state efficiency, reduced settlement work or electrical-energy evidence.

## 0.75.0 — 2026-10-06

- Add `Brain.live` and arousal for one continuing stream
  ([issue 88](https://github.com/muellerberndt/cadence/issues/88),
  [issue 122](https://github.com/muellerberndt/cadence/issues/122)). A brain
  constructed with `arousal=ArousalConfig()` answers a routine moment with the
  greedy choice of one qualified settle and changes no parameter, record or
  optimizer state, while the eligibility of earlier sampled actions fades by one
  step (`ActorCritic.fade`). An outcome that contradicts the forecast made before it
  (surprise) or a reward below what the stream usually pays (want) rouses it; an
  aroused brain samples at a temperature its want raises, keeps eligibility,
  learns from every outcome and writes memory, and the outcome that woke it is
  written to its memory. `Arousal` and `ArousalConfig` are exported; every
  constant of the law is a gene with hand-set founders and a declared space for
  `genes`. `Brain.act`, `ActorCritic.act` and `ActorCritic.probabilities` take
  an optional sampling `temperature`; `Brain.last_arousal` reports each moment;
  a brain with arousal saves it under the format name `cadence-generic/3`.
  `step`, the composed defaults, the settling and learning equations and the
  checkpoints of brains without arousal are unchanged.
- `ArousalConfig` gains `value_surprise` and `record_surprise`: a brain with an
  associative memory also forecasts the outcome of its chosen action from the record
  it holds, and the error of that record is a second surprise channel with its own
  usual size (`Arousal.usual_record`, `Brain.last_arousal["record_error"]`). The
  founders weigh it at zero, so the law is unchanged by default: on the nursery's
  development seeds the record channel woke the brain sooner and left more lives
  searching too briefly. Checkpoints of brains with arousal carry the record forecast
  of the awaited action.
- Add the odour nursery (`benchmarks/reversal`): one continuing life through
  acquisition, reversal and return with an unrelated stable skill, at pre-switch
  exposures from 100 to 10,000 trials, on a frozen protocol with ten fresh
  confirmation seeds, against always-learning, released-default, memory-only,
  graph-only, frozen, replay, reset, tabular and uniform-random arms, with
  source-bound receipts and a verifier. Its gates passed on fresh seeds at each of its
  three freezes; at the third, of the 40 gated lives 39 reversed, 39 returned and 40
  kept the stable pair, with median reversal lags of 15 to 30 trials, and one life
  never searched for the moved reward. It reproduces the historical finding that the
  always-learning brain stops sampling the choice that must change, counts the
  witnessed approaches until the greedy choice turns (one), and measures the
  routine share and the settling work of each mode. The associative memory
  carries the adaptation, the graph's reward learning alone does not acquire the
  task, a routine moment pays one full settle, and arousal on the released
  composition stays at chance: the operating point of one continuing stream it
  declares (working trace amplitude 0.3, consolidation 0.25, a tenth of the
  composed actor rate) is a development setting of that chamber; no default
  changes.
- Odour nursery, third freeze (`benchmarks/reversal`), for the readings
  [issue 88](https://github.com/muellerberndt/cadence/issues/88) still required:
  the probability of approaching each odour under the behaviour that acted and under
  the base policy, and the probability of each executed action, read from the living
  brain at every trial; a `replay` control that presents the brain's own witnessed
  records of the first rule to its memory again, one per trial, in place of the
  earlier counterfactual payoff; and the complete work of a life, with probes,
  checkpoint files, memory reads and writes, presentations, brains built, the sweeps
  of a refused attempt and the wall time of a moment per mode. Receipts of the earlier
  freezes verify by their own kind. The confirmation ran once on fresh seeds.
- Harden `live` continuation and its audit instruments: reject contradictory
  pending-action checkpoints and incomplete arousal genes; preserve routine
  feedback on a refused arousal update and identify already accepted sampled
  feedback. Keep arousal statistics atomic on numerical overflow, and support
  tiny positive averaging rates and sampling temperatures. Validate nursery
  receipt plans, sources and readings, and count discarded reset brains and
  frozen actions. Historical receipts retain their original sources and work
  limitations; the old-rule control is not equal-work witnessed replay. Issues
  88 and 122 retain their remaining acceptance requirements.
- Add the bounded steady-rhythm chamber (`benchmarks/rhythm`) for
  [issue 116](https://github.com/muellerberndt/cadence/issues/116): one
  continuing `Brain.compose` life taught to alternate two actions under constant
  drive, with frozen inputs, a frozen protocol, a declared physical event
  cadence, checkpoint-forked erased/shuffled/reset/static controls, a matched
  flip-flop control, a uniform-random baseline, pause/distractor disturbances,
  paced runs under solver-budget and host-load variation, and checkpoint
  continuation checks between actions and during a pause. Results are measured
  limits of the current System 1 on five fresh seeds; no core source, default,
  mechanism or gene changes.

## 0.74.0 — 2026-10-04

- `ActorCriticConfig.eta_bias` left unset derives `eta / 10` at construction, the
  rule of issue 126 applied to the actor ([issue 143](https://github.com/muellerberndt/cadence/issues/143)).
  An explicit value is used as given; construction warns when the bias rate exceeds a
  positive `eta`. The bare default (`eta=0.5`) resolves to the former 0.05, and the
  composed brain keeps its measured `eta_bias=0.05` at `eta=1.0`, so composed and
  default brains are unchanged; an actor with a lowered `eta` and no explicit bias
  rate learns with a bias step a tenth of its synapse step instead of 25 times it.
- Grouped motor slots on the composed brain ([issue 142](https://github.com/muellerberndt/cadence/issues/142)):
  `Brain.compose`, `Brain.build`, `Brain.genome` and `Brain(...)` take `slots`, a
  count of equal groups or one size per group covering the actions. Each slot is one
  softmax that settles with the others; `act` and `step` return one index per slot;
  lateral inhibition stays within a slot and the unset `lateral` follows the largest
  slot; episodic memory writes the chosen neuron of every slot; `Brain.load` rebuilds
  the actor on the saved grouping. One slot is the unchanged default.

- Report `capped`, the share of observed rows whose dopamine exceeded `dopamine_cap`
  before the clip, in every `learn` report and through `Brain.last_learning`
  ([issue 139](https://github.com/muellerberndt/cadence/issues/139)). Add the
  night-replay chamber (`benchmarks/replay/`): a day of decisions through `step`,
  a night in which a saved copy re-experiences that day, an equal-experience awake
  control, and frozen policy readings as the adoption gate. Document that a replay
  through `step` is more experience at the same rates, not consolidation, with
  the measured causes of a policy that ignores its observation (sign-only
  dopamine at the composed actor rate on one stream, the default working trace)
  and a setting under which the night helped as much as fresh experience, the
  equal-experience comparison of
  [issue 112](https://github.com/muellerberndt/cadence/issues/112). Defaults,
  equations and saved-state semantics are unchanged.

- Accumulate CUDA block transport directly into its destination to avoid a
  temporary product and a separate addition kernel per block. Add actual-device
  System 1 equation, gradient, refusal, memory and continuation checks, plus a
  source-bound CPU/CUDA runtime and memory comparison for issue 98.
  Preserve POSIX source keys in credit diagnostics and LF bytes in the frozen
  phrase fixture so the existing provenance checks also pass on Windows.

## 0.73.1 — 2026-10-04

- Distinguish signed activity below rest from silence (issue 106). Preserve the
  historical strict activity-fraction expected failure and add functional
  response, qualified acquisition/retention and full saved next-update guards
  for the existing zero and optional 0.5 processing-bias settings. Core equations,
  defaults and saved-state semantics are unchanged.

- Record the measured decline of a normalized composed default
  ([issue 131](https://github.com/muellerberndt/cadence/issues/131)): at the
  proposed `eta=0.003, normalize=0.99, momentum=0.9` for both composed
  learners, supervised acquisition contracts pass but the reward stream fails
  its re-adaptation contract (0.486 against 0.9 after a contingency change).
  Composed defaults remain unnormalized; normalized rates stay per-application
  settings behind the 0.72.1 construction warning. Documentation only; no
  default or equation changes.

- Harden the acquisition and retention instruments: admit source and input
  identities before execution, distinguish executed outcomes from accepted
  feedback, count memory writes only after commit, and charge final receipt
  writes against declared time and storage limits. Preserve refused work and
  complete case censuses, with independent verification and adversarial guards.
- Make retention preparation work in a standalone checkout or source
  distribution. Formal-source snapshots are explicitly requested inputs;
  omitting them does not claim a formal verification result.
- Add actual sampled-action association, partial-cue continuation and saved
  feedback guards, and bounded finite-horizon input/trace checks. Broaden the
  default test inventory to include the acquisition instrument guards.
- Add a portable, read-only acquisition/retention results demo and document the
  research wrap-up. The 360/360 partial-cue and 24/24 order-sensitive results are
  bounded development screens from pinned 0.73.0 workers; fresh retention
  confirmation, integrated replay and native transfer remain open in issues
  [85](https://github.com/muellerberndt/cadence/issues/85) and
  [110](https://github.com/muellerberndt/cadence/issues/110). Failed controls and
  source/custody limits remain visible.
- Require proposals to use local agreement repair into the same global
  equilibrium, test the simplest existing System 1 first, and demonstrate
  benefit while preserving acquired capabilities. Animal and human brains
  remain the reference, including their finite capacity and possible rigidity.
  Update contributor/agent instructions, review template and current guides.

The numerical runtime, public defaults, learning and memory equations, and
checkpoint contracts match 0.73.0. Experimental centering and a normalized
composed default are not promoted by this release.

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
