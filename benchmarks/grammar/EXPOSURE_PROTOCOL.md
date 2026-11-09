# Exposure parity with rehearsal: declared protocol (cadence#270)

Declared on 2026-11-09, committed to the lane branch before the first life. The founders, arms,
readings and gates below are fixed; a failed gate is reported as failed.

## The question

The transformer's offline control replays the grammar chamber's 4,000 training batches in shuffled
epochs until its loss stops falling, and reads 1.00 on the trained panels. A brain hears each batch
once. Does the default composition with its rehearsal (`learning_rehearsal=1.0`: after each lesson as
many rows again, drawn with their episodes from a reservoir of earlier lesson rows), taught the same
batches in passes, reach every panel within .05 of the transformer's offline reading at the
transformer's compute? If not, which mechanism keeps a panel apart?

## Instrument

`benchmarks/grammar/exposure.py` (`plan`, `run`, `read`) and `benchmarks/grammar/controls.py`. A
brain life is `Brain.compose(35, 18, modules=(64,), seed=founder, backend="cpu")` with the arm's
settings, nothing else named. The first grammar's 4,000 batches of the founder's stream, taught in
passes: the first in the stream's order, each later one in the order of the transformer's next
offline epoch. After each pass the eight readings below are read on 256-sentence panels; after the
last, `located` for the recall read. A life stops after 4 passes, or after the first pass that raises
no reading's mean by more than .01 over the pass before it. Compute is the module's analytic count.

The control: `controls.py`'s transformer, offline, its width matched to the brain's `parameters()`,
on the same batches and panels, to convergence (patience 3, at most 30 epochs).

## Founders and arms

Founders 9627000 to 9627005 (no life in any lane). Arms `rehearsal` (`learning_rehearsal=1.0`) and
`default` (no rehearsal), twelve brain lives and six control lives.

## Readings

Clauses, held-out clauses, clauses at three and four distractors, roles (both right), brackets,
brackets at depths four and five, paths, three hops: the exact sentences of `clause`, `clause_held`,
`clause_long`, `roles`, `dyck`, `dyck_deep`, `path` and `path_long`.

## Gates

| Gate | Reading | Pass |
| --- | --- | --- |
| E1 | `rehearsal` at the transformer's offline compute: each founder's last pass whose cumulative compute is at most that founder's control's (the first pass where none is) | every reading's mean at least the control's mean less .05 |
| E2 | `rehearsal` at its last pass (saturation or the fourth) | the same |
| E3 | `default` at E1's rule and at its last pass | reported beside E1 and E2, no bar |

**Verdict.** The acceptance's parity line is met when E1 passes. Where E1 fails, the mechanism that
keeps each apart panel apart is read from three things recorded before any reading: the panel's mean
per pass (a panel that rises with passes and stops short at the control's compute is apart by
compute; one that stays flat across passes is apart by mechanism), `rehearsal` against `default` at
equal passes (what the rehearsal adds), and `located` on that panel at the last pass (a word not
delivered by the read is lost at the read; one delivered and not said is lost at the readout). The
reading is recorded on the issue whichever way it falls.

## Compute

One AWS c7i.16xlarge (or this Mac where the box is unavailable, named with the results), one process
per life, one BLAS thread per process.
