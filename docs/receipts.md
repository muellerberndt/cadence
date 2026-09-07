# Receipts

A receipt is a result that can be checked by someone who was not there.

## Shape

```json
{
  "kind": "my-lane/v1",
  "body": { "...": "whatever the lane recorded: readings, rows, gain tables, custody" },
  "source": { "files": [{"path": "lane.py", "sha256": "..."}], "manifest_sha256": "..." },
  "digest": "sha256 of the canonical JSON of kind, body, and source"
}
```

Written as newline-terminated canonical JSON: sorted keys, no whitespace, no NaN.

## Verification

`Receipt.verify(path, sources=..., check=...)` recomputes four things and fails on the first
that disagrees:

1. the file is byte-for-byte the canonical form of its own content;
2. the embedded digest matches;
3. the source files named in the manifest still hash to what the receipt says;
4. the caller's `check(body)` finds no arithmetic problem, which for a protocol receipt
   means every pass flag follows from the stored readings and every tally follows from
   the rows.

## What belongs in the body

- the wiring summary and digest, and the custody block for measured data;
- the rule and the engine description;
- the gain selection table, every gain tried, with its admissibility;
- the protocol as data, including the reference for every row;
- the score, with readings and reference readings on every row;
- the control's score;
- the conformance report for the backend used;
- an explicit boundary block: what the lane declares rather than derives, and what it does
  not claim.

## Editing invalidates, on purpose

Every file named in the source manifest is bound to the receipt. Change one line in the
lane and the receipt no longer verifies until the lane is re-run. Batch edits, then re-run
once. When several lanes share a module, an edit to that module re-runs all of them; plan
it as a versioned break and record the library version in the body.
