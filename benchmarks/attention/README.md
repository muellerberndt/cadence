# Memory relevance and attention

## Retrieved content becomes the answer: the copy pathway (`recall_copy`, `recall_tokens`)

The [copy-tokens benchmark](copy_tokens.py) asks two things of eight tokens. A token is heard,
two other tokens follow, and a cue asks for the heard token back: the cue's key names the heard
slot, so the read is right before any lesson, and tokens 5 to 7 are never taught as answers.
And a cue names a token of a sequence of five heard once and asks for the token that followed
it, which the read holds on its next port while its recall port holds the cue's own token.
Lessons at the cue only, 300 and 600 per founder, `teaching="normalized"` with
`learning_surprise=True`, `episodic=False`. Its [receipt](results/copy-tokens.json), six founders
of one module of 24 neurons:

| Arm | Untaught produced | Taught | Succession | Untaught successor |
| --- | --- | --- | --- | --- |
| founder memory | 0.00 | 0.72 to 0.98 | 0.95 to 0.99 | 0.00 |
| `recall_copy=2.0`, `recall_tokens=range(8)` | 1.00 on every founder | 1.00 | 0.94 to 0.99 | 0.00 |
| `recall_copy=1.0` with tokens | 0.17 to 1.00 (mean 0.78) | 0.86 to 1.00 | 0.95 to 1.00 | 0.00 |
| `recall_copy=0.5`, no tokens | 0.00 | 0.57 to 0.99 | 0.93 to 1.00 | 0.00 |
| `recall_copy=1.2`, no tokens | 0.00 | 0.22 to 0.87 | 0.86 to 0.97 | 0.00 |

Uniform random answers 0.125. The founder's read names the untaught token at 0.32 to 1.00 (the
founder of seed 2 moved its own read with its lessons) and never produces it. Two founders of a
deep apical brain read the same: untaught 1.00, succession 0.94 and 0.98 against 0.97 and 0.99.
Every declared gate passes; the result is recomputed by `test_copy_tokens.py`.

The design was chosen on these tasks, and the alternatives read as follows (development
readings, outside the receipt unless stated):

- An identity on both ports at once (founder 0, identity 1.0 to 4.0): the next port of the heard
  token's slot holds the distractor that followed it, and the newborn produced the heard token on
  0.53 to 0.67 of untaught questions; 300 lessons took it to 0.00 to 0.13. The identity starts on
  the recall port only.
- No echo suppression (the echo synapses zeroed at birth, six founders): untaught 1.00, but
  succession 0.79 to 0.95, mean 0.87, against 0.94 to 0.99 with it: the recall port answers a cue
  with its own token.
- Each port's identity tied into one learned gain (`Learner(tie_groups=...)`): on the receipt's
  six founders it produced the untaught tokens at 1.00 as the untied pathway does, with succession
  0.91 to 0.98 against 0.94 to 0.99, so the ties were left out.
- The apical compartment as a bound was not needed: on the deep apical brain the pathway, which
  lands basally, keeps succession, and a bound relative to the motor region's bottom-up drive would
  cap the copy of a token whose action that drive never raised. It was not measured.

The first run of this receipt declared a gate that the founder's read of the untaught tokens is
0.80 or better on every founder; founder 2 read them at 0.32, and the gate was replaced by its
production alone (0.00 on every founder), with the reads reported beside it.

The receipt was recorded on the NumPy solver (`CADENCE_FUSED=0`); the compiled dense kernel gave
the same answers, with the reported gains differing in their last digits.

```sh
CADENCE_FUSED=0 PYTHONPATH=src python benchmarks/attention/copy_tokens.py --workers 4 \
  --out /tmp/copy-tokens.json
```

## Its own actions in the slots (`recall_action`)

The [own-action benchmark](own_action.py) gives a brain the same observation at every moment, so
only what it did tells its moments apart. It is taught to alternate its two actions, and to press
every third moment (press when its own last two actions were waits); each moment's answer is the
one its own history implies, so a constant policy scores near zero. 150 bouts of twelve moments,
one module of 24 neurons, `teaching="normalized"` with `learning_surprise=True`. Its
[receipt](results/own-action.json), six founders, means:

| Arm | Alternate | Press every third |
| --- | --- | --- |
| no recall memory | 0.47 | 0.36 |
| founder recall memory | 0.78 | 0.65 |
| `recall_action=True` | 1.00 (every founder) | 0.97 (0.83 to 1.00) |
| the same, commands erased before every test moment | 0.26 | 0.33 |
| efference copy (`efference_amplitude=3.0`, `efference_decay=0.5`) | 1.00 | 1.00 |
| uniform random | 0.48 | 0.46 |

The commands do the work: erased, the brain falls below uniform random. The efference copy is
the simpler mechanism where only the last commands matter. Every declared gate passes; the result
is recomputed by `test_own_action.py`.

```sh
CADENCE_FUSED=0 PYTHONPATH=src python benchmarks/attention/own_action.py --workers 3 \
  --out /tmp/own-action.json
```

## Looking back by position (`recall_position`, 0.89.0)

The [look-back benchmark](look_back.py) streams five random symbols, then a cue, `first` or
`last`, and asks for the symbol at the cued place. The cue matches no stored key and recency
is off, so only the position term can tell the slots apart. Both memories hold the episode's
boundary (`recall_boundary=True`), so the first symbol, like every later one, can be read
through its own slot or its predecessor's next. Only the cue moment is taught, 1200 episodes
per founder at `teaching="normalized"`; 160 test episodes follow without lessons. Its
four-founder [receipt](results/look-back.json):

| Founder | With the term: `first`, `last` | Without it: `first`, `last` | Uniform random |
| --- | --- | --- | --- |
| 0 | 1.00, 1.00 | 0.38, 0.76 | 0.25 |
| 1 | 1.00, 1.00 | 0.45, 0.67 | 0.25 |
| 2 | 1.00, 1.00 | 0.44, 0.80 | 0.25 |
| 3 | 1.00, 1.00 | 0.29, 0.76 | 0.25 |

Every founder passes the declared gate of 0.9 on both cues, and every founder's term beats the
memory without it. Without the boundary, founder 3 answered `first` at 0.83: it had learned to
answer through the next port, and no slot's next held the first symbol. The result is
recomputed by `test_look_back.py`.

```sh
PYTHONPATH=src python benchmarks/attention/look_back.py --out /tmp/look-back.json
```

## Several facts in one settle: held heads (`recall_heads`)

`Brain.compose(..., learned_code=units, attention=True, recall_heads=k)` reads one witnessed
bank through k copies of the held read, each with its own port and learned address. The
[held-heads benchmark](held_heads.py) asks for two of four witnessed facts in one settle,
reported in two slots (six values) or combined into one action (ten positions), with both
orders of some value pairs never taught. Its four-founder receipt with competing heads
([0.91.3](results/held-heads-competing.json), `CADENCE_FUSED=0`, one thread; the retained
[receipt](results/held-heads.json) is the run before heads competed):

| Task | Two heads, taught pairs | Two heads, untaught pairs | One head, taught pairs | Uniform random |
| --- | --- | --- | --- | --- |
| Report two facts | 1.00 on every founder | 0.85, 0.77, 0.98, 0.82 | 0.21 to 0.24 | 0.03 |
| One action from two facts | 1.00 on every founder | 0.98, 0.99, 0.98, 0.98 | 0.59 to 0.76 | 0.10 |

The heads compete for the slots (0.91.3, `recall_compete`). Under 0.91.2, with the same
settings (`CADENCE_FUSED=0`, one thread), the untaught pairs read 0.79, 0.77, 0.98, 0.82 and
0.96, 0.96, 0.97, 0.98, the one-head arm identically to the row above, every taught pair 1.00.
The heads prefer different role records on every trial, with no refusal. One head learns its
address at the same settings. Lessons switch to the brain's wrong rows once its running share
of wrong answers falls below 0.02 (lesson 176 to 626; under 0.91.2 176 to 415, the one action
later on two founders and sooner on two). Misses of the reporting slots on
untaught pairs all give the other asked value (`other_asked_value` equals `wrong_slots`); a
single action generalizes. The result is
recomputed at its original source and metric scope by `test_held_heads.py`. Current runs
also record global maxima, mass on relevant records and distinct relevant facts. Two slots
from the same fact, or two heads peaking on a shared neutral record, do not pass the current
distinct-facts gate. Head selection and correct acquisition retain separate gates.

```sh
PYTHONPATH=src python benchmarks/attention/held_heads.py --out /tmp/held-heads.json
```

### Current and historical evidence

The current attention runners bind every library Python module, package metadata, the
protocol script and `benchmarks/provenance.py`. They record execution mode, numerical
dependencies and thread settings; worker outputs retain their own identities before a
merge. Checkers validate the declared configuration and founder/arm census as well as the
arithmetic. Receipt verification checks those records; reproducing acquisition requires
running the declared training protocol.

The original positive receipts remain unchanged, with the source bytes and metric coverage
they actually recorded. Verify those explicitly as historical evidence:

```sh
python benchmarks/provenance.py --historical benchmarks/attention/results/held-heads.json
```

This verifies the registered result, archived source hashes and recorded gate arithmetic.
It does not label old scores as measurements of a newer implementation. Keep new measured
results in separate files.

## Public single-head contextual recipe (0.87.0)

`Brain.compose(..., learned_code=units, attention=True)` now exposes the simplest positive
result here: the existing held Recall reader with a two-moment ordered learned-code key,
successor evidence, zero recency bias and a sharp read. The
[contextual-choice assay](contextual_choice.py) keeps the final ASK observation identical,
cold-clears other context routes and disables episodic memory. Its retained four-founder
[receipt](results/contextual-choice-v0870.json) records 100% intact accuracy, 50%
present-only and memory-off controls, relevant and irrelevant record erasures, explicit
uniform-random comparison and zero refusals. The source-bound result is recomputed by
`test_contextual_choice.py` in the regression suite.

This establishes bounded, action-relevant selection by one learned-code head. It does not
show an advantage from revising a query during its settle, a continuing-world advantage or
useful multiple heads. The option is therefore not a default. The code width of eight is the
assay's measured capacity and remains explicit in user compositions.

## Coupled memory relevance: admission before integration

The new [direct-route candidate](DIRECT_ROUTE.md) is the first single-head
component here to pass acquisition and causal relevance together. It uses one
live 64-coordinate route and one locally learned map; there is no separate
query state or anchor. Its retained development founder reaches 100% allowed
and held accuracy and passes `W=0`, relation/relevant erasure, unrelated
erasure, payload-donor and material-selection checks. This is positive
component evidence, not yet a supported `Brain.compose` option or default;
independent founders, lifecycle preservation and larger continuing worlds are
still required. The negative experiments below remain controls and are not
rewritten by this result.

The coupled-query and direct-route mechanisms below are **component experiments**. They do
not implement the supported `attention=True` option, which deliberately uses the simpler
existing held-query reader described above.
The original query selects both an event and its witnessed successor; these
are not two attention heads. A benchmark-only [multi-head instrument](heads.py)
now supplies independent live query/read ports over the same witnessed bank.
Its [two-binding protocol](HEADS_PROTOCOL.md) requires behavioral comparisons
before claiming a useful second head.
The [extended capacity control](results/founder-capacity-extended.json) now
passes acquisition on both exposed and both supplied-read founders. The first
[learned-reader comparison](results/heads-relevance-founder-extended.json)
fails all ten declared cells: H2 hits its training time limit; the one-head
variants refuse a numerical solve. A separately declared
[exact H2 continuation](results/heads-continuation-development.json) finishes
both original 512-round schedules with no refusals, but each founder ends at
8/48 trained pairs and 0/16 held-out pairs correct. Extra training does not
repair this acquisition failure. Useful parallel heads remain unproved.
See the [parallel-head assessment](PARALLEL_HEADS_ASSESSMENT.md) for the
predeclared distinction between retrieval success and a useful second head.
The [ordinary-query initialization control](QUERY_FOUNDER.md) tests one
different starting condition for the same live query connections. Its staged
runner requires new matched capacity before H2, and both H2 founders before
matched one-head/resource comparisons. The old positive capacity receipt is
baseline evidence for the original initializer.

That [matched capacity run](results/query-founder-capacity.json) passes all
four 512-round acquisition controls without refusals. The changed
[learned-reader run](results/query-founder-h2.json) nevertheless fails: its
two founders stop on query refusals after 297/182 rounds. Trained and all-case
query panels refuse; held-out queries complete but score 0/16 in both founders.
Ordinary query initialization is not an established repair. Retention,
resource comparisons, fresh confirmation and longer-world evaluation remain
closed for this point.

The separate [minimal two-binding prerequisite](MINIMAL_BINDING.md) keeps the
same reader and local learner, but uses two active values and a fixed A/B
answer order. All eight cases are trained and evaluated, with no held-out
claim. Its [staged protocol](minimal-binding-protocol.json) requires new matched
exposed/held-read capacity before either learned H2 founder. Actual one-fact
donor swaps, erasures, shams and head lesions distinguish content use from
useful parallel selection. It cannot pass or explain the original task, and
opens no longer-world comparison by itself.

The [minimal capacity controls](results/minimal-binding-capacity.json) now
pass, and both [learned H2 founders](results/minimal-binding-h2.json) acquire
all eight pairs without refusals. They follow both actual donor facts and
fail appropriately under record erasure, but changing the neutral record
changes 4/8 and 2/8 answers. Both also fail the predeclared named-record
specialization gate. The full primary gate remains negative; retention and
resource comparisons stay closed. The separate [direct single-head test](SINGLE_RELEVANCE.md)
checks both questions on copies of the same witnessed history, with new matched
capacity and causal selection gates. Its four capacity controls pass, and both
learned founders answer all 16 questions correctly. Both nevertheless depend
on irrelevant context and fail question-specific record selection. The
[retained H1 result](results/single-relevance-h1.json) does not admit learned
relevance or longer-world attention claims.

The separately declared [neutral-variation control](NEUTRAL_VARIATION.md) changes
only the actually witnessed neutral strength during training. It preserves
these failures and all original minimal-task criteria, and requires fresh
matched capacity before either learned H2 founder. Those controls pass, and both
[learned founders](results/neutral-variation-h2.json) now pass content use,
neutral-context robustness and 256-round same-task continuation. Both still
fail distinct-head specialization; no matched H1 resource comparison has run.

The new [continuous-payload single-head component](CONTINUOUS_RELEVANCE.md)
tests both questions with fresh scalar values in a shared payload coordinate.
It requires its own exposed and supplied-address capacity controls before
learned H1 evaluation. All four controls pass at 100% on fresh values, but the
two learned H1 founders reach only 69.5% and 66.6% and fail causal relevance.
The [aggregate result](results/continuous-relevance-h1-summary.json) preserves
that negative outcome. This reset-round component establishes no learned
meaning codes, continuing-life preservation or Stone Age advantage.

The demonstrated limitation of held-query retrieval is that the current
interpretation cannot revise its selection during a settle. The earlier
competitive reader described in [the binding instrument](../binding/index_read.py)
had a different failure: content strengthened its own selection, sometimes
completing a nearly random stored moment. Reinstalling it would not finish the
relevance mechanism.

The [protocol](protocol.json) fixes two candidate equations, two development
founders, a local budget and unused confirmation founders before behavioral
results. It keeps the original named-fact instrument and its acquisition floor
unchanged. If supplying both facts still fails to teach the query relation,
behavioral attention comparisons stop: that failure has not been localized to
memory. The original admissions below failed; the separately declared
[teaching-policy repair](repair-protocol.json) now passes on both development
and all twenty fresh confirmation founders. Numerical checks and that acquisition
control alone cannot admit a library attention option.

## Equations and boundaries

With witnessed keys `K`, contents `V` and a zero-content null slot, let
`p(q) = softmax(q K / T - recency * age, null / T)` and `R(q) = sum p_i V_i`.
Event and successor coordinates are concatenated in `V`; the selector is shared.
The candidate is an agreement energy on the live read activation `r`:

```
E_memory(q, r) = alpha / 2 * |r - R(q)|²
drive_read = alpha * (R(q) - r)
drive_query = alpha * J_R(q).T * (r - R(q))
```

Historically the instrument multiplies this energy and both drives by a
per-stream `any_held` mask. The opt-in `empty_anchor=True` retains the displayed
equations on empty streams too: their null mixture is zero, so the read gets
`-alpha*r` and the query gets zero memory force. The option is confined to the
agreement prototype; the old masked control remains the default.

The implementation evaluates the actual softmax Jacobian, including the mean
subtraction and null probability. A transpose of the normalized recall is not
that derivative. Reciprocal read/association and query/association projections
let the present interpretation and memory selection influence each other.
Query projections start at zero. The existing graph's local free/nudged contrast
teaches them; there is no separate address lesson or backpropagated answer head.
Derivative checks use finite differences as diagnostics only.

The second, simpler control is `E = -alpha q M r`, where
`M = K.T V / number_of_held_slots`. It supplies paired linear drives and avoids
claiming that a linear association is a normalized content selector. The old
competitive energy remains a negative control. A test retains its unsupported
positive and negative completions from tiny different starts, even with a null
slot. Agreement removes those completions in that example; that is not a proof
against all spurious equilibria.

The agreement candidate also has explicit negative checks: without query
evidence, biased stored contents can still drive a nonzero read, and duplicating
records changes their aggregate probability against the null. Neither a null
slot nor restoring terms establish warranted recall or learned abstention.

Observation codes, the working trace, witnessed keys, values, successors, ages,
slot masks and parameters are held across each free/positive/negative phase.
Query, read, association and motor activations are live. `p(q)` is an algebraic
relation recomputed in **every sweep and residual check**. No unresolved query
or read is frozen to publish an answer. A slot is written only after an actual
accepted observation, never from a teacher label or an imagined event.

The prototype uses the CPU float64 reference path. Its bounded step-halving
fallback now retains the same immutable memory boundary in every numerical
candidate and checks the original full equations. Historical recipes retain
damping zero; the [founder numerical control](heads-founder-protocol.json)
tests the composed brain's existing step and damping policy.
The lifecycle prototype now reuses the existing reward/eligibility law with
immutable action boundaries, private coupled imagination, and an explicit
attention checkpoint format. Its mechanical custody and continuation tests
do not establish a useful reader or a supported library attention option.
The retained behavioral receipts below precede this lifecycle prototype;
their archived source hashes identify the measured implementation. They do
not measure acquisition or preservation under the modified lifecycle code.
Raw controls retain their historical `.1` step and disabled damping unless
overridden; learned-code compositions retain the native outer step `1.0` and
default damping `3`, including the native checkpoint validation.
Saved-drive teaching now takes corresponding rows of the immutable memory
snapshot through all phases, restoring the complete boundary on success or
refusal. These lifecycle gates, backend conformance, causal
relevance, learned semantic codes and retention must pass before integration.
Raw coordinates in the admission chamber are explicitly a control. Neither its
finite acquisition nor a small joint residual establishes learned meaning,
two-link binding, sentence generation or coherent behavior over time.

## Run

From the repository root:

```sh
PYTHONPATH=src python -m pytest benchmarks/attention -q
PYTHONPATH=src python benchmarks/attention/admission.py --out /tmp/attention-admission.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/admission-256.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/admission-1024.json
```

The admission receipt records source hashes, the declared recipe, resolved
settings, training completion, decisions, refusals and solver work. Its wall
budget is checked between bounded trials; bounded evaluation follows. Host
seconds and solver work are not physical energy. A budget-truncated run is a
failure to establish acquisition, not evidence that the learner cannot acquire
the task with more work. Confirmation seeds stay unused on admission failure.

## Earlier 22 October 2026 admissions: retained failures

Both recipes used the same normalized teaching, development seeds and task;
only the free/nudged sweep caps rose from 256 to 1,024 in the declared
[numerical retry](protocol-longer.json). Failed or unrun answers remain in the
denominator. These are final acquired readings, not best checkpoints selected
along the training path.

| Sweep cap per phase | Founder | Completed rounds / 256 | Final overall accuracy | Unequal-fact accuracy | Teaching refusals | Training action refusals |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 0 | 256 | 87.5% | 75% | 204 | 0 |
| 256 | 1 | 60 | 100% | 100% | 0 | 1 |
| 1,024 | 0 | 66 | 37.5% | 25% | 0 | 1 |
| 1,024 | 1 | 71 | 50% | 50% | 0 | 1 |

In the longer-budget final evaluations, half the queries refused for each
founder. A 100% answer reading after an incomplete training run does not pass
the declared completion gate. The two admissions took 15.14 and 5.76 host
seconds; their [receipts](results/) retain decisions, completion and all recorded
teaching/action solver work. No behavioral candidate round or confirmation seed
was used. Historical source bytes changed during CLI and guard work are retained
by content hash in `results/sources/`; the verifier checks those exact bytes,
receipt digests and recomputed scores/gates. It does not claim independent
behavioral reproduction.

The [diagnostic replay](diagnose.py) repeats the same failed 1,024-sweep recipe
and tries the existing three-halving damping fallback at each refused query.
Its [readback](results/damping-readback.json) retains the extra training and solve
work separately. Both queries still fail the original residual tolerance. This
does not establish that all solvers or teaching recipes fail; it establishes
that enabling that existing fallback at these boundaries does not close this
admission. The local Jacobian is measured at an **unqualified attempted state**,
not at an equilibrium, and is not a stability theorem.

These failures establish no retrieval deficit. The separately declared repair
below addresses exposed-fact acquisition, preserving these results and the 0.95
floor. Supplied-content use, held LMS versus `toward`, causal learned relevance
and lifecycle integration remain separate gates. This experiment must not be
merged as completed attention or used to close a relevance capability issue.

## Acquisition repair: the existing wrong-answer teaching control

The failed admission omitted an existing successful control in
[`acquisition_controls.py`](../relevance/acquisition_controls.py): teach only
incorrect answers from the **complete pre-query stimulus**. The new
[protocol](repair-protocol.json) fixes the task, two development founders,
twenty unused confirmation founders, a 600-host-second total budget and the
original 0.95 acquisition floor before running. It keeps the same 32-unit
reciprocal System 1, lateral 0, normalized teaching, `dt=0.1` and qualified
1,024-sweep free/nudged phases. Both matched arms now act before teaching;
one teaches every row, the other only wrong rows. The old admission's different
teach-before-act schedule remains explicit in its retained receipts.

Capture the complete drive before the query act advances the working trace,
then restrict that saved drive to the rows actually answered incorrectly.
Rebuilding it from a smaller observation batch would lose the original context;
rebuilding after acting would teach from the query-advanced trace. Teachers
still supply only final answers, with no slot index or relevance mask. An
all-correct round performs no lesson. Every phase of each accepted lesson
must qualify under the existing full-equation rule.

| Policy | Development founder | Completed rounds / 512 | Final overall accuracy | Lessons / taught rows | Training refusals |
| --- | --- | --- | --- | --- | --- |
| Every row | 0 | 67 | 37.5% | 67 / 1,072 | 1 query |
| Every row | 1 | 72 | 50% | 72 / 1,152 | 1 query |
| Wrong rows | 0 | 512 | 100% | 36 / 157 | 0 |
| Wrong rows | 1 | 512 | 100% | 31 / 103 | 0 |

Both wrong-row founders also reach 100% on unequal facts and each query. The
first development run was retained, then repeated after tightening only the
verifier. The replay reproduces **every complete per-run record**, including
parameter/checkpoint digests, decisions and work counts. The first passing
recipe was frozen before opening confirmation; the historical finite-teaching
fallback with lateral -0.5 was never run or restored as a default.

**Confirmation: all 20 fresh founders, seeds 100–119, pass.** Each completes
512 rounds and reaches 100% overall, unequal-fact and both-query accuracy,
with zero teaching or action refusals. Final acquired readings are used,
without selecting the best checkpoint along training. Initial mean accuracy
is 51.25%, final accuracy 100%, and the separate uniform-random control averages
50%. Confirmation uses 564 lessons on 2,579 taught rows. The original
development, verifier-only replay and confirmation take 29.47, 21.76 and
180.33 host seconds respectively, all charged within the declared 600-second
budget. Each evaluation forks saved state and checks that the acquired brain
is unchanged. Checkpoint contents are hashed in the receipts; binary checkpoints
remain local. This is not an efficiency comparison: the failed every-row arms
stop early and therefore do less work.

This repairs the acquisition **instrument's teaching recipe**, not a runtime
gradient or retrieval defect. Continuing to teach an already correct decision
can still move its parameters; wrong-row teaching removes that pressure. The
matched comparison measures this policy's effect here; it does not identify
the precise dynamical cause of the every-row refusals or propose a new default.

All sixteen fact/order/query cases are taught, using raw input coordinates as
an explicit control. Fresh founders do not establish unseen-binding or semantic
generalization. Trace-erased queries also qualify, but their mean accuracy is
77.5%; the intact acquisition result uses the continuing trial's prefix context.
This acquisition panel alone does not test content delivered through recall
ports; the separate development control below does. The coupled reader still
needs learned causal relevance, abstention, retention and lifecycle integration.
No behavioral coupled candidate was run in this acquisition panel. Seeds
100–119 are now opened for acquisition and must not be described as untouched
confirmation seeds for a later relevance candidate.

Run or verify the repair from the repository root:

```sh
PYTHONPATH=src python benchmarks/attention/repair_admission.py --stage development \
  --checkpoints /tmp/attention-dev-checkpoints --out /tmp/attention-development.json
PYTHONPATH=src python benchmarks/attention/repair_admission.py --stage confirmation \
  --development /tmp/attention-development.json --checkpoints /tmp/attention-confirm-checkpoints \
  --out /tmp/attention-confirmation.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/repair-development.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/repair-confirmation.json
```

The verifier binds the frozen protocol, complete ordered founder/arm sets,
matched development settings and schedules, actual answered cases, and original
checkpoint preservation. Optimistic flags cannot hide refused or unrun queries.
The earlier development source is retained by content hash alongside its first
receipt; a stricter verifier does not silently rewrite historical evidence.

## Supplied recall content: development passes

The [content protocol](content-protocol.json) keeps the repaired 32-unit,
qualified teaching recipe. Two actual fact events and one neutral event are
witnessed, then the working trace and warm neural state are cleared at the
query. Both unselected facts are supplied together through ordinary recall
ports. The matched sensory arm receives the same content in exposed sensory
coordinates. This explicit query boundary isolates content use; it is not a
continuing-life result or learned selection.

All four development lives (founders 0/1, recall-port/sensory arms) complete
512 rounds and answer every case, unequal fact and query correctly, with zero
action or teaching refusals. Erasing content reduces accuracy to 50%. Relevant
flips and donor content are followed perfectly; irrelevant flips and identical
shams change no answer. Only actual wrong decisions are taught from their saved
pre-query drives. Total host time is 59.52 seconds. The
[receipt](results/content-results-development.json) and original source snapshot are
retained; final binary checkpoints remain local. Founders 300–319 are unopened.

```sh
PYTHONPATH=src python benchmarks/attention/content_control.py \
  --out /tmp/attention-content-development.json
PYTHONPATH=src python benchmarks/attention/verify.py \
  benchmarks/attention/results/content-results-development.json
```

## Minimal two-head and contextual-representation task

Each trial gives `A is value` and `B is value`, using four **shared** value
coordinates, in either order. The same ambiguous cue asks for `AB` or `BA` in
two four-choice motor slots during one settle. Four value pairs are withheld
from training. Swapping the bindings preserves the query and unordered token
multiset but changes both correct answers; an even mixture loses the bindings.
`recall_next=False` prevents one selected event and its successor from already
supplying the two facts.

The [protocol](HEADS_PROTOCOL.md) declares a plain exposed acquisition arm, a
two-port supplied-content capacity control, one/two independent selectors, a
wider single-selector port, wider single-head cortex and a larger sweep cap.
Total memory coupling stays fixed. Receipts count actual graph transports,
memory scans, softmax elements, residual checks and refused work. More heads
or a larger sweep cap alone do not establish improved accuracy or efficiency.

The contextual representation is the **settled association activity**. The
same cue may acquire a different interpretation as recalled bindings change.
After retrieval and causal gates pass, a fixed diagnostic readout must transfer
that task context to held-out pairs and fail appropriately under memory lesions.
Distances or different attention pictures alone are insufficient. The current
`LearnedCode` boundary is context-free; this test does not claim that its
observation code shifts inside the settle, or that general language meaning
has been learned.

```sh
PYTHONPATH=src python benchmarks/attention/heads_admission.py \
  --out /tmp/attention-heads-capacity.json
```

## Dual-binding capacity result: still blocked

Both exposed founders complete 512 rounds at 100% on the 48 training cases.
The supplied-read controls do not complete the declared acquisition run:

| Read control | Sweep cap | Completed rounds, founders 0 / 1 | Stop |
| --- | --- | --- | --- |
| Original external drive, erroneous prefix coupling | 1,024 | 30 / 24 | Prefix refuses |
| Corrected external drive, coupling disabled throughout | 1,024 | 30 / 13 | Prefix refuses |
| Same corrected control, declared numerical retry | 2,048 | 83 / 42 | Prefix refuses |
| Held-address agreement, matching the candidate's restoring law | 2,048 | 42 / 21 | Prefix refuses |

Each supplied-control life has one refused action and zero teaching refusals.
The [protocol amendments](HEADS_PROTOCOL.md#declared-numerical-admission-retry)
retain every failed run and the exact single-boundary numerical diagnosis.
The [final control receipt](results/heads-capacity-fixed-read.json) qualifies
neither learned selection nor useful head specialization. Total capacity and
diagnostic work is 304.11 host seconds, including conservative charges for an
interrupted diagnostic helper, within the shared 600-second cap.

**No behavioral candidate equation round was run; confirmation remains
unopened.** The original 95% acquisition and later relevance gates stay fixed.
The multi-head anatomy, matched causal runner and contextual-state diagnostic
are implemented, but their behavioral and representation claims remain unrun.
The next failure to resolve is reliable acquisition/settling with supplied
content in this reciprocal multi-output composition, before attributing a
deficit to learned selection or adding another head mechanism. The earlier
single-query content and acquisition passes retain their separate scope.

```sh
PYTHONPATH=src python benchmarks/attention/verify.py \
  benchmarks/attention/results/heads-capacity-fixed-read.json
```
