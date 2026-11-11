# Direct learned single-head relevance

This component assay asks whether the coupled reader learns to select the
fact requested by the current question. It uses the existing raw binary
named-fact task: witness A, B and one neutral event, then answer either
"what was A?" or "what was B?". Both questions start from copies of the same
eight witnessed histories. All 16 cases are trained; there is no held-out
split, learned meaning code or continuing-creature claim.

The candidate has one query/read pair, one binary action slot and a 32-unit
association cortex. It retains the live agreement energy, total coupling 3,
ordinary q1 initialization, qualified phases and optimizer of the earlier
reader experiments. Query and read remain live throughout settlement; the
witnessed memory bank stays fixed. The teacher supplies only the final
binary answer on wrong rows. It supplies no address, head role or relevance
label. Trace and warm-state clearing before the question are explicit
component boundaries.

The protocol first requires four new matched capacity cells: two founders
with both unselected facts exposed at the query, and two with the actual
asked-name record supplied through a fixed-address restoring read. The
second arm is an oracle control. Its prefix read is null, and at the question
it selects an actual witnessed record by name and age; it never invents a
target value. Old exposed or supplied-read results cannot admit this task.

Only positive current capacity opens both learned H1 founders. The 95%
acquisition floor requires all 16 answers. Acquisition alone is insufficient:

- Changing the requested witnessed fact must change the answer appropriately;
  changing the other fact must preserve it.
- Erasing each requested fact must reduce its corresponding answer accuracy.
- Erasing the irrelevant neutral record and reordering whole records must
  preserve answers. Neutral erasure changes only its held flag, preserving
  its actual key, value and age.
- On unequal-value cases, each question must give its requested record at
  least .5 probability and .1 margin over every alternative, including null.
  The same acquired bank is used for both questions.
- Restoring the newborn interface or shuffling the learned interface must
  reduce accuracy on complete, qualified panels. These interventions also
  change cortical transport, so their losses do not isolate selector causality.

The all-erased panel is an unsupported-input diagnostic, with refusals and
work retained. Two binary actions provide no abstention label. That panel
cannot contribute a lesion loss or establish learned abstention. Conditional
256-round same-task retention opens only after the full primary gate passes;
it is not competing-learning retention.

The standalone controller binds the preceding failed minimal H2 receipt,
the complete source inventory, native one-slot teaching configuration and
all attempted work. It retains actual parameters, four optimizer arrays and
update counters as research snapshots, not supported continuing checkpoints.
Capacity has a 600-second stage budget; learned H1 has 1800 seconds. Both
founders are attempted even when the first fails. Neither stage launches a
world, a fresh-seed search or AWS work.

After freezing source, run and independently verify capacity:

```bash
python benchmarks/attention/single_relevance_campaign.py --runtime . \
  --stage capacity --out /tmp/single-relevance-capacity.json \
  --parameters /tmp/single-relevance-capacity-parameters --commit <frozenSHA>
python benchmarks/attention/single_relevance_campaign.py --runtime . \
  --verify /tmp/single-relevance-capacity.json
```

Only a positive new receipt admits learned relevance:

```bash
python benchmarks/attention/single_relevance_campaign.py --runtime . \
  --capacity /tmp/single-relevance-capacity.json --stage learned_h1 \
  --out /tmp/single-relevance-h1.json \
  --parameters /tmp/single-relevance-h1-parameters --commit <frozenSHA>
python benchmarks/attention/single_relevance_campaign.py --runtime . \
  --capacity /tmp/single-relevance-capacity.json \
  --verify /tmp/single-relevance-h1.json
```

Passing this assay would establish bounded learned raw-memory relevance.
It would establish neither a parallel-head advantage nor the learned-code,
outcome, imagination and continuation preservation required before an
integrated creature comparison.

## Measured result

All four [matched capacity cells](results/single-relevance-capacity.json)
complete 512 rounds, answer all 16 questions correctly and have zero action
or teaching refusals. Both [learned H1 founders](results/single-relevance-h1.json)
also complete 512 rounds without refusals and answer 16/16 correctly.

Both learned founders follow relevant donor changes perfectly, preserve
answers under irrelevant donor changes and lose .50 accuracy when the asked
fact is erased. Their full primary gates nevertheless fail:

| Founder | Learned answers | Neutral-erasure changes | Irrelevant-fact erasure | Requested-record selection |
| --- | --- | --- | --- | --- |
| 0 | 16/16 | 2/16 | No answer changes | Fails |
| 1 | 16/16 | 4/16 | Erasing B changes 4/8 answers about A | Fails |

Both readers retain broad mixtures. Neither meets the declared mass/margin
gate on any unequal row, so correct answers do not establish question-specific
memory selection. Record reordering changes no answers. Frozen and shuffled
interfaces lose .375 accuracy for founder 0 and .50 for founder 1, with the
transport caveat above. These losses cannot rescue the failed causal gate.
Retention is not opened for either founder.

Capacity costs 96.664401 host seconds and learned H1 costs 179.175963,
bringing cumulative retained model work to 4,428.819964 seconds. This resolves
basic acquisition on this small task, while learned selective relevance,
nuisance robustness and longer-world admission remain unfinished.
