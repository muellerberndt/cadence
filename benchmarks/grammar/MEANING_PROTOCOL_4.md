# Held-out combinations said from the state: declared protocol 4 (cadence#286)

Declared on 2026-11-10, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed. It follows
[the third protocol](MEANING_PROTOCOL_3.md).

## The question

#286's acceptance asks for sentences said from a held state. With the brain's own message
(`message=280`) at 256 association neurons and the recall read barred from the meaning, clauses
and role twins reach .90 on most founders and held-out combinations fail on two of six (.61 to .83
in the third run, with clauses and roles intact on those founders): the readout of a superposed
state on combinations no lesson showed. Three additions are tested against the message alone.

## Founders and arms

Founders 10028600 to 10028611 (twelve, no life in any lane). Every arm: 256 association neurons,
`message=280`, every row taught (`learning_surprise=False`), the recall read barred from the
meaning at `GO` in lessons and readings, 4,000 batches of the first grammar.

| Arm | Beside the message | Parameters |
| --- | --- | --- |
| `own_256` | nothing: the reference | 432,528 |
| `read_256` | the read of one block (`message_read=True`), untested at this width | 587,416 |
| `heard_own_256` | the previous word (`history_decays=(0.0,)`), which lifted held-out combinations at 64 association neurons | 441,524 |
| `both_256` | both | 596,412 |

## Readings

Whole sentences on `clause`, `clause_held`, `roles` (both twins), `dyck`, `path` and `path_long`
(three hops); each arm less `own_256` paired by founder; multiply-accumulates and seconds per life.
Where held-out combinations are lost, on every life: each clause word kind said right on
held-out combinations with the previous words given (8 batches of 16 per trained shape), and
linear probes of every clause component from the message, the carried trace and the settled
association at `GO` and at the last word, fitted on trained combinations and scored on held-out
ones (8 batches of 16 of each per shape).

## Gate H1c

An arm holds where `clause_held` reads .90 on 10 of 12 founders, `clause` and `roles` read .90 on
10 of 12, and on `dyck`, `path` and three hops its paired mean difference against `own_256` is at
or above -.03 and `own_256` is not above it by two standard errors.

**Expectations, stated before the run.** `own_256` reads held-out combinations at .90 on 8 to 9 of
12 and fails. `heard_own_256` lifts held-out combinations by .02 to .05 and holds or misses by one
founder. `read_256` reads within .03 of `own_256`. `both_256` follows `heard_own_256`.

**What a failure names.** On the founders whose held-out combinations fail: where the transfer
probe reads the failing component from the message at .95 and over at `GO` and from the settled
association under it, the meaning is held and the association does not carry it to new
combinations; where the message's own probe falls, the code does not hold the combination apart;
the word kinds said name which word is lost.

## Part 2

Run only where H1c holds for at least one arm: longer clauses, deeper brackets and longer paths
(`clause_long`, `dyck_deep`, `path_long`) from the state on the arm that holds with the highest
`clause_held` mean, against the chamber's LSTM and transformer at its parameter count, online and
offline, with the read located as in the first protocol. Its founders and gate are declared in an
amendment before its lives.

## Compute

One AWS c7i.8xlarge spot box (Name=cadence-097-meaning), one process per life, one BLAS thread
per process.
