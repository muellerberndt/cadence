# Two required bindings in one settle

Declared 22 October 2026 before this benchmark's behavioral runs. This is a
bounded component experiment, using the existing reciprocal System 1 and local
free/nudged teaching. It does not change a library default or establish general
semantic attention, sentence production or continuing sequence competence.
The machine-readable declaration is heads-protocol.json.

## Task, split and supplied information

There are two names, A and B, and four **shared** value tokens, 0 to 3. A trial
presents `IS + A + value_A` and `IS + B + value_B`, in either order, one neutral
event, then `AMBIGUOUS_CUE + ASK_AB` or `AMBIGUOUS_CUE + ASK_BA`. The two motor
slots each have four choices. Their final answer is respectively `(value_A,
value_B)` or `(value_B, value_A)`. The query supplies an order convention; it
supplies neither value, a retained answer nor an intermediate address. Both
independently sampled values are needed, including trials where they are equal.
Both answers must come from **one final settle**, without another environment
event, a hidden second read or an event/successor pair counted as two heads.

The 19 observation coordinates are five markers (`IS`, `NEUTRAL`,
`AMBIGUOUS_CUE`, `ASK_AB`, `ASK_BA`), two names, four shared values and eight
normally dormant exposed-control coordinates: four A values and four B values.
The exposed control's query carries **both** named values in the last eight
coordinates. All retrieval arms leave them zero. No name or value has a
private slot coordinate in a witnessed fact.

Hold out value pairs `(0,1)`, `(1,0)`, `(2,3)`, `(3,2)` from training. The other
12 ordered pairs include every value under each name, all four equal pairs,
both query orders and both fact orders: 48 training cases and 16 held-out
cases. This split is fixed before the first run, including the swapped
counterparts. Each training round samples 16 of the 48 cases, with replacement,
from `SeedSequence([622022, founder])`; every arm uses the same schedule.
Evaluation enumerates every case once. Report exact pair and each-slot accuracy,
unequal-value cases and each query order separately; score refusals as wrong.

On a matched swapped-values panel, exchange A's and B's stored values while
keeping query, fact order and the fact-token multiset fixed. Each motor slot
must follow its own requested binding. The two equal-value facts stay as sham
swaps. Aggregate correctness on a convenient subset cannot substitute for the
held-out pair score or this unequal-value swap test.

## Boundaries, teaching and comparison arms

Use `recall_next=False`, no associative/episodic store and one bank of the
actual accepted observations. Each head reads the **same** immutable keys,
values, masks and ages; no replicated bank creates additional experience.
Source-only retrieval is the primary diagnostic: after the prefix and before
the query, clear the working trace and warm neural state identically in
training and evaluation, retaining the witnessed bank. Capture the complete
query drive and immutable bank before the query act. The query, each read,
association and motor state then settle jointly under a full residual check.
An intact trace is a separately labeled later comparison, never silently
substituted for this boundary. This supplied reset is a diagnostic intervention,
not learned context management or a continuing-life result.

Use 32 association units, normalized teaching (`eta=.003`, bias `.0003`, RMS
decay `.99`, floor `.0001`, momentum `.9`), beta `.1`, temperature `.2`, motor
lateral `0`, graph dt `.1`, tolerance `.003`, qualified free/positive/negative
phases, 1,024 sweeps per phase and damping disabled. Teach only wrongly answered
rows from the saved **pre-query** drive and the corresponding bank rows. A
correct pair receives no lesson; for a wrong pair supply both final motor
labels. No teacher supplies head roles, selected records, query targets,
relevance masks or intermediate values to a candidate. Records commit only at
accepted witnessed event boundaries. Run the independent equation, row-custody
and local-credit tests before any behavioral run.

| Arm | Extra resource and purpose |
| --- | --- |
| Exposed | Plain reciprocal `Brain`, 32 units; both unselected facts supplied at the query. The original .95 acquisition floor is retained. |
| Supplied reads | H2 anatomy with memory energy disabled. Its two read ports receive `3 * fact_A` and `3 * fact_B`, independent of query order. This supplied-content control tests read/cortex use; it is not a candidate selector. |
| H1 | One live query/read, association width 32, total memory coupling 3. |
| H2 | Two independent live queries/reads, association width 32, coupling 1.5 each; total coupling remains 3. |
| H1-width | One query/read, association width 64; tests a wider ordinary cortex. Report actual neuron, synapse and parameter counts rather than call this exact size matching. |
| H1-port-copies | One selector, two query/read port copies sharing that selector; same additional port count as H2. Keys, values and cues are duplicated with a `1/sqrt(2)` isometry, preserving match/read norm. Tests added coordinates and routes without independent selections. Total coupling remains 3. |
| H1-work | The H1 anatomy with up to 2,048 sweeps per phase. Compare at the measured H2 total neuron/transport work, including all teaching and refused work; an unmatched run supports no efficiency or head-specific advantage. |
| Frozen and uniform random | Acquired-interface freeze/shuffle on copies, frozen newborn controls and an independently seeded uniform-random pair policy. |

All heads initially receive the same complete query code; query-to-cortex
relations start at zero and ordinary reciprocal read/cortex synapses are drawn
independently. No initial head is assigned A, B, a query position or a record.
Duplicated-head and head-shuffle controls distinguish useful specialization
from redundant selectors. Multiple passes over the live equations are solver
work; they are not additional environment decisions.

## Admission, causal relevance and preservation gates

1. **Exposed acquisition:** complete all training rounds and evaluation; reach
   .95 exact pair, each slot, unequal-value exact pair and each query's exact
   pair score on the 48 trained cases, on both development founders. Report the
   16 held-out cases separately. Their result does not weaken this .95 floor.
2. **Content use:** train the supplied-reads H2 control, with memory energy
   disabled and the two actual fact events exposed at its two read ports by
   fixed A/B identity, not by which answer slot the query asks for. Keep the
   witnessed snapshot for custody, but do not use it to supply a learned or
   ask-dependent address. Teach only the final pair and keep the answer free.
   Require .95 exact pair, each slot, unequal-value pair and each query on the
   training panel, on both founders, before coupled selection work. Report the
   held-out panel separately. This oracle/content-exposure result measures
   reader/cortex capacity, not learned selection. The H1 one-mixture arm cannot
   be given two separate complete facts without changing its representation;
   report that information limit rather than impose an impossible oracle gate
   or invent an extra read for it.
3. **Relevance:** require .90 exact pair on all 16 held-out cases and .90 exact
   pair on the unequal-value swapped panel, on both development founders.
   Every named slot and query order is reported. With 16 cases, .90 requires
   at least 15 correct; preserve the full denominator.
4. **Causality:** erase one queried record on acquired copies and require at
   least .20 loss in its corresponding answer slot; report the other slot's
   loss, head erasures and individual selector distributions. A matched bank
   transplant whose two values change must make both outputs follow the donor
   on at least .80 of cases. Sham/reordered-record interventions change at
   most .05 of answers. Include irrelevant neutral-record changes. Freeze or
   shuffle the learned selection interface without removing access to facts.
   Report whether useful specialization persists under head duplication and
   head permutation with its downstream connections kept or permuted explicitly.
   Merely different attention weights do not establish a useful second head.
5. **Retention:** after 256 further balanced training rounds using the same
   twelve training pairs and alternating query order, held-out exact pair
   accuracy loses at most .05 against the acquired copy, while the trained
   panel keeps .95. Count this continued work; this bounded task retention is
   not a competing-task or P7 persistence claim.

Admission and content-use failure stop selection attribution, not the retained
failure record. A candidate may not enter confirmation or be promoted because
its neural residual is small or a supplied read gives the answer. The original
acquisition failures, earlier coupled-reader failures, repair receipts and
existing binding/chain gates keep their scope.

## Ambiguity and contextual representation: separate second phase

The same ambiguous cue must support different acquired answers when the facts
change. After the retrieval gates pass, fit two diagnostic ridge readouts from
the **free settled association activation**, one for each requested value,
using the training pairs only, one-hot targets, centered features and fixed
ridge penalty `.01`. Freeze them before reading the held-out pairs and the
matched swapped panel. Record exact pair transfer, not only cosine distances
or a visual separation of activations. Require .90 pair transfer, at least
.20 above matched raw-query and frozen-newborn association controls, and at
least .20 loss after erasing the relevant records. The diagnostic readouts
never participate in the brain's answer or its local learning; success means
the learned cortical interpretation exposes task-relevant context to a fixed
held-out readout. A failed readout is retained and does not erase a behavioral
retrieval pass.

This first assay uses raw coordinates, with a fixed random observation-code
arm reported separately if run. It cannot establish learned semantic codes.
`LearnedCode` currently publishes a context-free held boundary: a changed
association state is not evidence that its code changed within the settle.
A later learned-code comparison must separately freeze its width, training,
source and budget; compare raw, fixed-random, learned and no-top-down controls;
and test held-out alias/group members with no raw overlap, misleading raw
overlap, distinct member facts and old facts after competing learning. A
representation-only geometric shift cannot pass this meaning-transfer gate.

## Source freeze, selection and budget

Freeze this protocol, all runner/anatomy/teaching source bytes and their hashes
in a commit before development begins. Record the source commit and complete
resolved settings in every receipt; retain the exact bytes of any later
corrected source. Development uses founders 0 and 1 only, 512 training rounds,
batch 16, at most 60 host seconds of training per arm and 600 host seconds
total, one local process at a time. Evaluation, oracle controls, retention,
representation probes and refusals are charged to the total. Check deadlines
between bounded trials; a capped or unrun cell fails to establish its gate and
remains in the declared denominator. If the next stage will not fit, stop and
declare its budget before running; do not truncate it into a positive result.

The candidate order is agreement, then the linear reciprocal special case only
if the first fails. These are at most **two candidate equation rounds**, not
two opportunities per arm, seed or parameter grid. H1/H2 and their resource
controls share each round's law and frozen settings. A failed first round is
retained before the second is selected. If the simpler one-query control
passes, keep it; H2 earns a separate benefit only against the pertinent width,
port and measured-work controls. No tuning on held-out cases or confirmation.

Fresh founders 200 to 219 remain unopened until development acquisition,
content use, relevance and causal gates pass. Confirmation needs a new fixed
time/work budget declared before the first fresh run; at least 18 of 20 must
meet every selected behavioral gate. A new recipe after a failed confirmation
must use a different unused founder block. No paid run, library integration,
reward-learning, private-imagination or exact saved-continuation claim follows
from this protocol; those remain separate preservation requirements before
integration, along with the existing sequence and sentence-training gates.

## Declared numerical admission retry

The [initial admission](results/heads-capacity-initial.json) and
[corrected control](results/heads-capacity-corrected.json) retain both failures.
The first mistakenly left memory coupling active during the supplied-read
prefix; the correction disables it throughout that control, as declared.
Both exposed founders learn every training pair. Supplied-read founders still
refuse a prefix after 30/13 completed rounds in the corrected instrument.

An [exact boundary diagnostic](results/heads-capacity-diagnostic.json) reproduces
founder 0's parameters and decisions. Its first refusal starts from rest with
an empty bank and zero memory drive: residual `.009084` at 1,024 sweeps falls
below `.003` after 160 more sweeps at the same `.1` step. Damping within the
original 1,024 budget still fails. This locates one numerical budget failure;
it does not establish that later boundaries or any selector pass.

The separate longer protocol declares **one**
numerical retry at 2,048 free/nudged sweeps for every arm, with 4,096 for the
larger-work single-head control. The step, tolerance, teaching, anatomy,
candidate equations and gates stay fixed. The previous 128.228748 host seconds
and conservative 30.519933-second diagnostic charge consume 158.748681 of the
same 600-second budget. Another failure stops this numerical retry; it does
not authorize another sweep-cap search. No confirmation founder is opened.

## Held-address agreement capacity control

The 2,048-sweep retry also fails: supplied-read founders stop on a prefix after
83/42 completed rounds, despite both exposed controls acquiring all trained
pairs. Keep the [failed retry](results/heads-capacity-longer.json); do not raise
the sweep cap again. No behavioral selector comparison follows from it.

The fixed-read declaration changes the
**oracle capacity control only**. The external `3*fact` source with memory
energy disabled omits the candidate's restoring term. Instead hold its two
oracle addresses on the latest actually witnessed A/B facts, independent of
query order, and use `sum alpha_h/2 * |r_h - V_named|² * held.any`. Its query
derivative is zero and its read derivative is `alpha_h*(V_named-r_h)*held.any`.
An absent name in a nonempty bank restores toward zero; an entirely empty bank
has zero force. No future fact is supplied, and no external read drive is added.

Per-head coupling stays 1.5, total coupling 3, with the same ordinary reciprocal
seams, 2,048 sweep caps, step, tolerance and final-answer-only teacher. Fixed
name assignments are explicitly an oracle, never learned-head evidence or
candidate address lessons. This isolates downstream content use under the
same restoring law. The learned candidates remain the original agreement and
linear equations; this control is not a third candidate equation. Its single
declared revision shares the remaining 348.877009 host seconds after the
251.122991 seconds already charged. Earlier failures and confirmation holds
remain intact.

**Result of this control revision:** both exposed lives again complete 512
rounds at 100% accuracy on training pairs. Held-address supplied-read founders
stop after 42/21 rounds on a refused prefix, with one action refusal and zero
teaching refusals each. The [receipt](results/heads-capacity-fixed-read.json)
retains the complete failed readings. Cumulative capacity/diagnostic work is
304.109854 host seconds. The numerical retry and declared control revision
are consumed; no candidate equation round or fresh confirmation is run.
Contextual readout transfer remains unrun. Resolve this shared acquisition/
qualified-settling failure with supplied facts before a selector claim.

## Matched empty-read boundary repair

The separately frozen [empty-read experiment](EMPTY_BOUNDARY.md) preserves the
old masked control and tests agreement with zero content when no record is held.
Its [receipt](results/empty-boundary-development.json) reproduces both old
failures and verifies identical occupied-query learning trajectories. Restoring
founders reach 140/279 rounds, then refuse a populated-memory prefix and query
respectively. Both still fail acquisition. The 2,048-sweep cap, tolerance and
teacher are unchanged. Learned relevance, head benefit and representation
transfer remain unrun.

## Founder numerical control

The founder control restores the numerical
settings already used by the composed brain: initial step `1.0` and up to
three halvings within one total qualified-sweep budget. The earlier prototype
had explicitly reduced the step to `.1` and disabled damping. Its rejection
was necessary while the library fallback constructed a plain graph and lost
the experimental memory force.

The benchmark now clones its own graph for the same bounded fallback, keeping
the identical witnessed boundary and checking the original complete equations.
No memory equation, teacher, rate, tolerance, head anatomy or initialization
policy changes in this control. Empty-read restoring remains enabled. All four
exposed/supplied-read founder cells must complete the unchanged acquisition
gates before any learned-head comparison. Capacity has a separately declared
360-host-second budget; a subsequent relevance stage, if admitted, has 600
additional host seconds and retains all previous charges. The fallback shares
the 2,048-sweep cap across attempts; it does not raise it.

Local spectra motivate this comparison but do not establish convergence: a
coarse step can be unstable, numerical attempts can enter different basins, and
later learned parameters remain untested until the run. Earlier failed runs,
including the smaller-step and initial-guess diagnostics, are not discarded.

**Result:** both exposed founders finish 512 rounds with 100% trained accuracy.
Supplied-read founders reach 419/370 rounds with zero action or teaching
refusals, then hit the 120-host-second training guard. Their trained pair
accuracy is 100%/87.5%; all four held-out pair readings are 0%. The
[capped receipt](results/founder-capacity-development.json) fails admission:
the original 512-round requirement is retained. It costs 254.714154 host
seconds, bringing retained work to 756.051724 seconds.

The host-time extension keeps every
task, neural, teacher and qualified-sweep setting unchanged. It declares 240
host seconds per capacity arm and 600 additional seconds total before rerunning
the same four controls. Its later relevance stage, only after complete positive
capacity, declares 240 seconds per training call and 3,000 additional seconds
for all ten cells and conditional retention/probes. Earlier capped and
held-out failures remain recorded.

**Extension result:** all four controls complete 512 rounds with 100% trained
pair/slot/unequal/query accuracy and zero action or teaching refusals. All
intact panels are answered, but all held-out pair accuracies remain 0%.
The [positive capacity receipt](results/founder-capacity-extended.json) admits
learned-reader development; it establishes neither transfer nor learned
selection. This stage costs 294.088845 host seconds, bringing retained work
to 1,050.140569 seconds. No confirmation founder is opened.

Before the first learned-reader cell, the
[parallel-head assessment](PARALLEL_HEADS_ASSESSMENT.md) fixes separate
requirements for distinct evidence, both-head usefulness and improvement
against completed one-head/resource controls. The runner's generic pass or
selected-arm flag does not establish those claims.

## Learned-reader development: no admitted candidate

The [first agreement-only comparison](results/heads-relevance-founder-extended.json)
retains all ten arm/founder cells under the extended protocol. None passes.

| Arm | Founder 0: completed rounds / stop | Founder 1: completed rounds / stop |
| --- | --- | --- |
| H1 | 135 / prefix refusal | 113 / query refusal |
| H2 | 321 / training host cap | 289 / training host cap |
| H1-width | 58 / prefix refusal | 99 / query refusal |
| H1-port-copies | 137 / prefix refusal | 68 / query refusal |
| H1-work | 127 / query refusal | 141 / prefix refusal |

H2 has zero training action or lesson refusals, but incomplete acquisition:
trained exact-pair accuracy is 16/48 and 26/48, with 0/16 held-out correct in
both founders. It does not meet the unchanged trained or transfer floors.
The other arms' zero final scores reflect refused/unanswered panels; they
are not evidence of zero accuracy among qualified answers. More width,
duplicated ports and the larger sweep ceiling do not qualify those points.

No arm enters retention or the representation probe; no fresh confirmation
founder opens. This consumes the one declared agreement candidate round.
The stage takes 1,287.076165 host seconds, bringing cumulative retained work
to 2,337.216734 seconds. No parallel-head advantage, continuing-life or
Stone Age result follows. Further repair must first distinguish numerical
refusal from capped acquisition; a generic negative pass flag does not
localize a head-count defect.

## Exact boundary diagnostic and H2 recipe completion

The [H1 boundary diagnostic](results/heads-h1-boundary-diagnostic.json)
reconstructs only the two original first refusals, retaining the actual field
before action rollback. Parameters and maximum residuals match the original
failures exactly. Each endpoint has a locally stable continuous linearization
and a locally stable final `.125` Euler step. This is pointwise evidence;
it proves neither global convergence nor that another schedule would rescue
the original 2,048-sweep run. No alternative step, teacher or training run was
tried. Its measured cost is 0.859652 host seconds.

The separately frozen continuation protocol
finishes only H2's missing 191/223 draws from its original schedules. It restores
all three parameter arrays, all four optimizer arrays and both update counters
from source-bound research snapshots. A 2+2-round identity check compares this
restoration with uninterrupted training. This is valid at the declared reset
round boundary; it is not a saved continuing-life checkpoint. Equations, local
teacher, rates, qualified-sweep caps, task and gates remain unchanged. The
original negative ten-cell receipt and all eight failed comparators remain
immutable; the new receipt explicitly identifies its derived combined reading.

**Result:** both H2 founders complete 512 rounds with zero training action or
lesson refusals. Each ends at 8/48 trained exact pairs and 0/16 held-out pairs
correct, below its earlier capped trained score of 16/48 or 26/48. All intact
panels are answered. Neither acquisition nor causal relevance passes; retention,
representation and confirmation remain unopened. No failed H1 comparator grants
a head advantage. This stage costs 396.728604 host seconds and brings retained
work, including the H1 diagnostic once, to 2,734.804991 seconds.

The [new receipt](results/heads-continuation-development.json) has its own strict
offline verifier, leaving historical execution dependencies unchanged:

```sh
PYTHONPATH=src python benchmarks/attention/heads_continuation.py \
  --verify benchmarks/attention/results/heads-continuation-development.json
```

The completed failure distinguishes limited training time from this recipe's
acquisition problem. Check actual local lesson directions on the same captured
memory boundary before changing the learning rule or anatomy. This result does
not open a longer-world comparison or establish useful parallel attention.

## Lesson checks and an ordinary query initializer control

The [prior-work ledger](results/query-founder-prior.json) retains three further
bounded diagnostic records and their exact charges. An initial work-counter
failure returns no phase; its corrected one-case lesson audit qualifies every
phase. On that trained unequal case, both original and refined local contrasts
point toward lower loss in both answer slots. The actual momentum/RMS update
reduces total loss and corrects one slot. This identifies no simple update-sign
or precision repair.

A separate full-panel endpoint check supplies one lesson on the 40 wrong rows
and evaluates all 48 trained rows. Both the native and zero-momentum proposals
reduce mean cross-entropy but lose the same two previously correct pairs:
8/48 becomes 6/48, with no new correct pairs. All three paired precision
references qualify and preserve these decisions. This synthetic aggregate
lesson is not an original sixteen-row minibatch or evidence of a training
remedy. The JSON ledger verifies retained records and accounting; it does not
independently replay binary phase states. Cumulative model work is
2,748.518539 host seconds, including the failed instrument attempt once.

The new initializer declaration tests one
ordinary reciprocal projection draw on each existing query/cortex interface,
once before a fresh brain's first event. The [construction guide](QUERY_FOUNDER.md)
states its independent seed streams and limits. No runtime equation, neuron,
contact, teacher or head role changes. Fresh matched exposed/supplied-read
capacity must pass before both H2 founders run; the eight matched one-head and
resource comparisons open only if both H2 primary and retention gates pass.
Original failed results and all behavioral floors remain in force.

### Ordinary-query result

The [new matched capacity receipt](results/query-founder-capacity.json)
passes all four cells: 512 rounds, zero action/lesson refusals and 100% trained
exact pairs. Held-out scores are 0/16, 0/16, 0/16 and 4/16 for exposed0/1 and
supplied-read0/1; capacity imposes no held-out accuracy floor. This stage uses
305.571460 host seconds.

The [learned H2 receipt](results/query-founder-h2.json) is negative in both
founders. Query refusals stop training at 297/182 rounds, with no teaching
refusals. Trained and all-case panels also refuse their query after successful
prefixes; their unanswered scores are not accuracy measurements of qualified
answers. Both held-out panels complete and answer 0/16 pairs correctly.
The changed initialization has not repaired acquisition or numerical
qualification. No retention, representation probe, matched resource control,
fresh confirmation or longer-world run opens. Its 464.384521 host seconds bring
cumulative retained model work to 3,518.474520 seconds. The original q0 results
and all acceptance floors remain unchanged.

```sh
PYTHONPATH=src python benchmarks/attention/query_founder_campaign.py \
  --verify benchmarks/attention/results/query-founder-capacity.json
PYTHONPATH=src python benchmarks/attention/query_founder_campaign.py \
  --verify benchmarks/attention/results/query-founder-h2.json
```

## Smaller simultaneous-binding prerequisite

The separately declared [eight-case chamber](MINIMAL_BINDING.md) removes
requested output permutation, two values and held-pair transfer. It changes
no reader equation or learner. New matched capacity passes in all four cells.
Both learned H2 founders then complete 512 rounds, answer 8/8 pairs and
qualify every causal panel. Donor changes are followed perfectly; erasing
either actual fact loses .50 corresponding-slot accuracy.

Both primary gates nevertheless fail: half-amplitude neutral key/value
changes alter 4/8 and 2/8 answers. Both fail the existing majority/margin
specialization requirement. Retention, resource comparisons and larger-world
runs remain closed; this is not a pass or diagnosis of the original task.
The two stages add 195.990749 and 438.514331 host seconds, respectively,
bringing retained model work to 4,152.979600 seconds. The exact typed receipts
and all prior negatives remain unchanged.
