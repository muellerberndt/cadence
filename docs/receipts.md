# Receipts

A receipt stores outcomes, an embedded digest, and a manifest of selected source
files. The caller decides which outcomes to record and what the verifier checks.

## Write and verify a small receipt

Save this as `receipt_demo.py` and run `python receipt_demo.py`. The example binds
a small arithmetic result to its script and the installed receipt implementation:

```python
from pathlib import Path
import cadence as cd

sources = [
    ("receipt_demo.py", Path(__file__)),
    ("cadence/receipts.py", Path(cd.__file__).with_name("receipts.py")),
]
values = [1, 2, 3]
receipt = cd.Receipt.build(
    "sum-demo/v1", {"values": values, "total": sum(values)}, sources=sources,
)
path = receipt.write(Path("sum_receipt.json"))

def check(body):
    if body["values"] != [1, 2, 3] or body["total"] != sum(body["values"]):
        return "stored sum differs from the declared inputs"
    return None

ok, message = cd.Receipt.verify(path, sources=sources, check=check)
assert ok, message
print(message)
```

Expected output: `canonical form, digest, sources, arithmetic agree`.
For an experiment, bind every consumed source and dataset, then supply checks for
its schedule, outcomes, and metrics. This small sum example verifies only its
declared arithmetic; it does not reproduce a model or a training run.

The examples' run receipts use a second format, `cadence-experience-run/v1`, written by
their stage runners: `stage`, `status` (`incomplete` until the schedule completes), `seeds`,
`sources` (the library's commit and a hash of its source files, the hashes of the stage
files), `artifacts` (the event logs with their sha256, and the checkpoints), `acceptance`
(each predicate with its value, threshold, aggregation and pass flag) and
`receipt_sha256`. The stage verifier recomputes the metrics from the event logs.

## Shape

```json
{
  "kind": "experiment/v1",
  "body": { "...": "recorded readings, rows, gain tables, custody" },
  "source": { "files": [{"path": "experiment.py", "sha256": "..."}], "manifest_sha256": "..." },
  "digest": "sha256 of the canonical JSON of kind, body, and source"
}
```

Written as newline-terminated canonical JSON: sorted keys, no whitespace, no NaN.

## Verification

`Receipt.verify(path, sources=..., check=...)` checks the following and fails on the first disagreement.
Source and arithmetic checks are optional and must be supplied by the caller:

1. the file is byte-for-byte the canonical form of its own content;
2. the embedded digest matches;
3. when `sources` is supplied, that complete source manifest matches the receipt;
4. when supplied, the caller's `check(body)` finds no arithmetic problem, which for a protocol receipt
   means every pass flag follows from the stored readings and every tally follows from
   the rows.

## What belongs in the body

- the connectome summary and digest, and the custody block for measured data;
- the neuron model and the brain description (`NeuralGraph.to_dict()`);
- the gain selection table, every gain tried, with its admissibility;
- the protocol as data, including the reference for every row;
- the score, with readings and reference readings on every row;
- the control's score;
- the conformance report for the backend used;
- an explicit boundary block: what the experiment supplies rather than derives, and what it does
  not claim.

## The cost and the orienting fields

A demo's receipt carries the same accounting for every arm, so that arms compare: per arm,
`macs_per_moment` (from `BeliefPatch.macs_per_moment` or `Steered.macs_per_moment`),
`moments_per_decision`, and the `cost` counters (`moments`, `macs`, `replays`) over the
life, from which a moments-per-decision and a multiply-accumulates-per-decision follow; a
life adds the governor's steps as moments and the decisions by mode (`Life.compute`). A
demo that reads a gain's response to events carries `orienting`'s output per kind (the
count, the capture of the first and last events, the curve, the latency shares, the mean
return) and the `dishabituation` pair, so that two demos' captures mean the same thing.
Unobserved returns are `null`, with `return_censored=true` after a positive capture;
they are not counted as observed recovery at the end of the window. Partial final
habituation bins remain in the curve.

Label MAC counts as dense forward-work estimates. Include their exclusions
(adjoints, writes, nonlinearities, callbacks and memory traffic) and any governor
step-to-moment conversion. They are not total training FLOPs. Report total elapsed
time, peak memory, learning/search work and all failed attempts for an efficiency
comparison. Test targets must never calibrate the governor or choose a baseline;
record the disjoint calibration/validation split and its seeds.

## Editing invalidates, on purpose

Every file named in the source manifest is bound to the receipt. If such a file
changes, a source check against those edited files fails. Record source hashes
as well as a library version, since a version string can cover several source
revisions. Verification does not launch an experiment automatically.

A historical receipt remains evidence for its original source snapshot. Preserve its
bytes and verify against that snapshot; do not re-sign an old result against later code.
Each run should record library source hashes, data generation/seeds, every scheduled
condition, predictions and targets, and validation choices. A verifier should reject
missing outcomes instead of recomputing averages over whichever rows remain.
