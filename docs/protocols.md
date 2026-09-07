# Protocols: held-out tests with preconditions

A protocol is a declared list of what a settled net will be asked, and what it may be
shown before it is asked. It is how the connectome lanes are scored, and the same
discipline (a validation split, a test read once, a control) is what the learning
examples follow in their scripts.

## Stimuli, rows, training facts

```python
protocol = cd.Protocol(
    stimuli={"rest": (), "touch": ("sensors",)},              # name -> sets clamped at full amplitude
    training=[("touch", "motors", "active")],                  # what the model may be shown
    rows=[
        cd.Row("R1", "rest", "motors", "inactive", "no input, no output"),
        cd.Row("R2", "touch", "motors", "reduced", "cutting the sensors", ablate=("sensors",)),
    ],
    levels=cd.protocol.Levels(active=0.5, inactive=0.2, margin=0.15),
    steps=60,
)
report = protocol.score(engine)       # {"rows": [...], "passed": k, "total": m, ...}
```

A `Row` names a stimulus, a readout set, a predicate, a citation-style reference, and
optionally sets to ablate (owners zeroed for that row) and a second readout to compare
against (`relative_to`). `tier` is free text for grouping.

## Predicates and their preconditions

| predicate | passes when | precondition |
|---|---|---|
| `active` | mean(readout) ≥ `active` | |
| `inactive` | mean(readout) ≤ `inactive` | |
| `reduced` | mean(readout) ≤ mean(reference) − `margin` | the reference (intact) net was active |
| `retained` | mean(readout) ≥ `active` | the reference was active |
| `released` | mean(readout) ≥ mean(reference) + `margin` | the reference was inactive |
| `exceeds` | mean(readout) ≥ `active` and ≥ mean(other) + `margin` | |
| `lateralized` | one side exceeds the other by `margin` | |
| `sparse` | `sparse_min` ≤ fraction active ≤ `sparse_max` | |
| `densified` | fraction(readout) ≥ fraction(reference) + `densify_margin` | the reference was sparse |

A precondition that fails makes the row fail. That is what stops a dead net from passing
`reduced` or a saturated one from passing `released`. `evaluate_predicate(predicate,
value, reference, levels)` is the pure function, and a receipt's verifier reruns it on the
stored readings.

## The shuffled control

`cd.shuffled(wiring, seed)` permutes the postsynaptic endpoints of every overlap and keeps
everything else: every count, every sign, every owner's out-degree, every named set, and
no autapses. Score the protocol on it with the same rule and gain. What the wiring passes
and the control does not is what the wiring predicted.

## Gain selection under a sparsity cap

```python
gain, table = cd.select_gain(lambda g: cd.Settlement(w, rule.replace(gain=g)), protocol,
                             grid=(0.01, 0.02, 0.03, 0.05), sparsity_cap=0.05)
```

Every gain on the grid is tried; a gain is admissible only if the training facts pass and
no more than the cap's fraction of owners is active under the training stimuli. The table
records every gain and why it was or was not admissible, and belongs in the receipt.
Pass `sparsity_cap=None` for toy nets meant to light entirely.

## For learned nets

The learning examples do not use `Protocol`; their held-out facts are a test set. The
same three habits carry over: selection on a validation split of the training data only,
the test set read once after selection, and a control that should fail (a net trained on
shuffled labels scoring at chance).
