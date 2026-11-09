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

## Result

Read on one c7i.16xlarge, founders 9627000 to 9627005, 256-sentence panels. The transformer offline
(width 40, 28,658 parameters against the brain's 28,716) converged in 13 to 30 epochs at a mean of
1.30e12 multiply-accumulates.

| Arm, pass | Compute | Clauses | Held-out | Long clauses | Roles | Brackets | Deep brackets | Paths | Three hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer offline | 1.30e12 | 1.000 | .999 | .000 | .995 | .966 | .003 | .997 | .000 |
| Default, pass 1 | 0.95e12 | .995 | .952 | .001 | .966 | .840 | .042 | .965 | .519 |
| Default, pass 2 | | .999 | .971 | .003 | .980 | .896 | .044 | .979 | .439 |
| Default, pass 3 | | .995 | .957 | .002 | .969 | .961 | .057 | .990 | .355 |
| Default, pass 4 | | .980 | .936 | .004 | .941 | .988 | .062 | .987 | .347 |
| Rehearsal, pass 1 | 5.44e12 | .953 | .914 | .003 | .909 | .852 | .051 | .822 | .247 |
| Rehearsal, pass 2 | | .992 | .980 | .003 | .975 | .831 | .014 | .835 | .197 |
| Rehearsal, pass 3 | | .997 | .985 | .002 | .989 | .901 | .020 | .927 | .177 |
| Rehearsal, pass 4 | | .987 | .922 | .002 | .961 | .898 | .008 | .891 | .109 |

| Gate | Verdict | Readings apart by more than .05 |
| --- | --- | --- |
| E1, rehearsal at the transformer's compute | fail | no pass fits; at pass 1 held-out, roles, brackets, paths |
| E2, rehearsal at its last pass | fail | held-out, brackets, paths |
| E3, default at the transformer's compute | reported | brackets (.862 against .966) |
| E3, default at its last pass | reported | held-out (.934 against .999), brackets (.898) |

What keeps each panel apart:

- **The rehearsal's cost.** A rehearsal replays each drawn row's whole episode on fresh streams, about
  3.7 million replayed moments a pass against 50,000 acts, so a pass with it costs 5.7 times a pass
  without; at the transformer's compute it has less than one pass.
- **Paths under rehearsal: the read.** The recall read delivers a hop's fact on .78 to .89 of hops
  with the rehearsal against .98 to 1.00 without it, and says a delivered hop on .90 to .96; paths
  fall from .97 to .82 at pass 1 and stay under .93.
- **Brackets: exposure.** Without rehearsal brackets rise with every pass, .84, .90, .96, .99, and
  pass by the control at pass 3 (about twice its compute). A closing bracket is said right on .96 to
  .99 while the read delivers its open bracket on .32 to .75, so brackets are carried by the trace
  and the cortex more than by the read.
- **Held-out clauses: no exposure lever.** Without rehearsal they hold at .94 to .97 over four passes
  against the control's .999.
- Long clauses and deep brackets read near zero for every learner, and the brain leads on three hops
  (.48 against .00).
