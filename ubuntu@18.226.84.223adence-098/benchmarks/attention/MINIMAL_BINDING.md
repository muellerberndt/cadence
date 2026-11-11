# Minimal simultaneous two-binding prerequisite

The standalone component controller is implemented. All 48 focused checks
pass: 29 pure task tests, nine immutable-bank checks and ten controller
checks including fresh constructors and synthetic capacity/H2/retention
stages. Neural equilibration is forbidden in those tests. The subsequent
scientific capacity and learned-reader runs are retained below. The measured
ordinary-query outcomes remain separate evidence.

The task reduces the earlier four-value, query-dependent output problem to
eight cases: `A,B ∈ {0,1}`, each with the two fact orders. Both actual facts
and a neutral event precede the same `AMBIGUOUS_CUE + ASK_AB` query. The two
final labels are always `(value_A, value_B)`. All four pairs are trained and
evaluated. Both facts matter: changing either value while holding the other
fixed changes exactly its corresponding final answer.

The interface retains 19 sensory coordinates and two four-choice output
slots. Only value coordinates 7/8 are active; the unused value coordinates
and `ASK_BA` remain dormant. Motor alternatives 2/3 remain possible errors.
Only the separately labeled exposed capacity control receives both values
at query time in its existing control coordinates.

This is a smaller raw-input component assay. It has no held-out split,
permutation transfer, semantic-code claim or continuing-life result. It
cannot satisfy the original benchmark or identify which original difficulty
caused failure. It removes several difficulties together to ask the prior
question: can the unchanged coupled learner acquire and use both facts at all?

`minimal_binding_task.py` imports NumPy only and constructs no brain. Its functions are:

| Function | Contract |
| --- | --- |
| `events(rows, exposed=False)` | Three actual prefix events and one fixed query; no candidate value/address in query |
| `targets(rows)` | Both final values in fixed A/B order |
| `panel('train'/'all')` | The same eight full-factorial cases; returned copies |
| `draw(seed, rounds, batch=16)` | Existing `SeedSequence([622022, founder])` namespace, eight-case draws with replacement |
| `scores(answers, rows)` | All rows retained; counts, pair/slot/prefix scores; one fixed-query score; `-1` refusal/unrun and motor choices2/3 count wrong |
| `swapped(rows)` | Actual counterpart with the named values exchanged |
| `flipped(rows, name)` | Actual donor case changing one named binary value only |
| `random_references(seed)` | Actual binary-aware and full-motor uniform answers, scores and analytic expectations |

Rows retain the old four-column shape `(A, B, prefix_order, query_code)`, but
query code must be zero. There is no `HELD` mask or `HELD_PAIRS` constant.
`panel('held_out')` raises explicitly; aliasing a training panel as held
would manufacture a transfer claim. The one-element `per_query_accuracy`
list represents the single observed query, rather than an absent second one.

The task-aware binary uniform reference has pair chance 1/4 and slot chance
1/2. The unchanged four-choice motor uniform reference has pair chance 1/16
and slot chance 1/4. Both retain independently sampled actual answers and
scores. Their RNG namespaces are 622024 and 622023 respectively; neither
consumes the training or action RNG. A constant `(0,0)` answer also scores
2/8 pairs. Knowing just one actual named fact, with no information about
the independent other fact, permits at most 4/8 correct pairs. The 8/8
acquisition gate and independent donor flips require both facts to matter.

The proposal fixes the current q1 ordinary initializer, before the first
event, with no q0 search. It retains the same agreement law, coupling,
32-unit association cortex, memory data, numerical qualification, damping,
optimizer, wrong-pair/full-label teacher and 512×16 schedule length. H2 has
no name, output or address assigned to either head. The teacher supplies
the final actions only. The trace and warm-state clearing at query is an
explicit component boundary, not a policy for integrated life.

New matched exposed and named held-read capacity must precede both H2
founders. Complete .95 floors on eight cases require 8/8 pairs; the four
unequal cases require 4/4. Refused, capped and unrun readings fail. The
conditional actual-donor flips, queried-record erasures and shams test
both-fact use on acquired copies with the host query fixed. Only after
acquisition/content-use may same-task 256-round retention run. The proposal
states separate evidence for useful heads; a successful H2 alone establishes
no head advantage. Any later H1/resource comparison needs its own declaration;
the capacity/H2 controller opens no comparator stage. A future head-benefit
comparison requires both H2 founders to pass primary, retention and useful-head
gates, then all eight qualified matched controls to finish. H2 must exceed
each comparator by at least .10 on each founder (one of eight net pairs),
with actual work disclosed. Missing controls establish no advantage.

Head usefulness uses each row's preferred named fact before lesion, so the
heads may exchange roles across rows. Distinct confident preferences are
required on all four unequal cases. Favored-slot lesion loss is measured
over all eight cases, where .20 requires at least two lost correct slots;
the unequal subset is also reported. On equal-value rows the preferred name
is ambiguous; deterministic A-on-tie reporting supplies no confidence claim.

`minimal_binding_campaign.py` owns the new panels, score recomputation,
source-bound typed receipts, 600-second capacity stage and 1800-second H2
stage. It reuses the unchanged low-level constructor, prefix preparation and
training helpers on the eight valid fixed-query rows. It does not reuse the
old whole-trial held-panel gates or treat an old positive receipt as new
capacity. The negative q1 receipt is verified before new construction. Its
verification cache is bound to the exact receipt, complete source manifest
and prior declaration; changes invalidate it. Actual parameters, all four
optimizer arrays and counters are retained as research snapshots, with no
supported continuing-checkpoint claim.

The 29 NumPy-only tests in `test_minimal_binding_task.py` pass. They
cover exact event/target custody, independent necessity of each fact,
full coverage without held cases, inactive/refused answers, reproducible
schedules, both actual random policies and invalid inputs. No model or
constructor test is included. Nine tests in `test_minimal_binding_bank.py`
also pass using the actual immutable bank types with synthetic witnessed
facts. They verify donor key/value custody, shared heads, unchanged host
query and atomic refusal of missing/ambiguous records. They explicitly
forbid brain construction and equilibration. The controller checks construct
fresh brains with neural equilibration forbidden, exercise synthetic stage
boundaries and verify their receipts. These mechanical checks establish no
learned behavior.

After freezing source, run capacity and verify its receipt:

```bash
python benchmarks/attention/minimal_binding_campaign.py --runtime . \
  --stage capacity --out /tmp/minimal-binding-capacity.json \
  --parameters /tmp/minimal-binding-capacity-parameters --commit <frozenSHA>
python benchmarks/attention/minimal_binding_campaign.py --runtime . \
  --verify /tmp/minimal-binding-capacity.json
```

Only a positive new matched capacity receipt admits the separate H2 stage:

```bash
python benchmarks/attention/minimal_binding_campaign.py --runtime . \
  --capacity /tmp/minimal-binding-capacity.json --stage h2 \
  --out /tmp/minimal-binding-h2.json --parameters /tmp/minimal-binding-h2-parameters \
  --commit <frozenSHA>
python benchmarks/attention/minimal_binding_campaign.py --runtime . \
  --capacity /tmp/minimal-binding-capacity.json --verify /tmp/minimal-binding-h2.json
```

Stages are serial. Each has its own additional deadline, includes attempted
and refused work, and retains every capped or unrun cell as a failure. No
extra seed, initialization search, comparator stage or larger-world run
opens automatically.

## Measured result

All four [capacity cells](results/minimal-binding-capacity.json) complete
512 rounds, answer 8/8 pairs and have zero action or lesson refusals.
Both [learned H2 founders](results/minimal-binding-h2.json) also complete
512 rounds and answer 8/8 pairs without refusals. Every causal panel completes
and qualifies. Both follow the actual A-only and B-only donor changes
perfectly, preserve the other answer, and lose .50 relevant-slot accuracy
when either fact is erased.

The full primary gate still fails in both founders:

| Founder | Trained pairs | Neutral-record changes | Named-record specialization | Retention |
| --- | --- | --- | --- | --- |
| 0 | 8/8 | 4/8 answers change | Fails | Not opened |
| 1 | 8/8 | 2/8 answers change | Fails | Not opened |

Reordering records or head descriptors changes no answers. Each head lesion
and the clone intervention loses 6/8 pairs in both founders, but those
interventions also alter restoring forces or cortical transport. They do not
isolate a head-count advantage. The heads use broad mixtures; they fail the
predeclared majority/margin test for distinct named-record selection.
That test is sufficient for this specific specialization claim, not necessary
for every possible useful soft representation. No matched H1 resource
comparison has run, so this result establishes no parallel-head benefit.

Training used a constant unit-amplitude neutral event. The failed neutral
intervention scales its key and value to .5 while leaving both facts and the
query fixed. This exposes nuisance dependence; it identifies no missing query
force or update-sign defect. A future data-variation control must retain this
failure and declare its own matched capacity, source and cost before running.

Capacity uses 195.990749 host seconds and H2 uses 438.514331, bringing cumulative
retained model work to 4,152.979600 seconds. Retention, resource comparisons,
fresh confirmation and longer-world runs remain closed for this point.
Learned coupled single-head relevance also needs its own direct test; exposed
or supplied-content acquisition is not that evidence.
