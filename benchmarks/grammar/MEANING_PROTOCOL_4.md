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

## Result of part 1

Read on one c7i.16xlarge, founders 10028600 to 10028611, 48 lives on commit 14aa827f. A first
spot box was reclaimed after thirty minutes with no life read; the same plan ran again on the
same commit.

| Arm | Parameters | Multiply-accumulates | clause | clause_held | roles | dyck | path | three hops | at .90 of 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `own_256` | 432,528 | 2.42e13 | 1.00 | .94 | .93 | .93 | .51 | .00 | 12, 9, 9 |
| `read_256` | 587,416 | 2.54e13 | .98 | .91 | .93 | .92 | .48 | .00 | 12, 8, 10 |
| `heard_own_256` | 441,524 | 2.40e13 | 1.00 | .95 | .90 | .91 | .66 | .00 | 12, 11, 8 |
| `both_256` | 596,412 | 2.54e13 | .99 | .94 | .96 | .91 | .51 | .00 | 12, 11, 11 |

**H1c holds on `both_256`**: clauses 12, held-out combinations 11 and roles 11 of 12 at .90, and
against `own_256`, paired, brackets -.02 (.01), paths -.00 (.05), three hops +.00. It fails on
`own_256` (9 and 9), on `read_256` (held-out 8, and clauses -.02 beyond two standard errors) and
on `heard_own_256` (roles 8).

The gate holds through the weakest founders and not at the mean: `both_256` less `own_256`,
paired, reads held-out combinations +.00 (.02) and roles +.02 (.04), with its least founders at
.73 and .84 where `own_256`'s are at .67 and .72. The previous word alone lifts paths said from
the state by .15 (.07), beyond two standard errors.

**Against the expectations.** `own_256` read 9 of 12 as stated. `heard_own_256` reached held-out
combinations on 11 of 12 as stated and lost roles (8 of 12, -.03), which was not expected.
`read_256` read within .03 of `own_256` as stated. `both_256` was expected to follow
`heard_own_256` and kept roles as well.

**Where the remaining held-out sentences are lost.** With the previous words given, every word
kind of a held-out clause is said at .96 to 1.00 but the verb: .93 to .98 at the mean and .74 to
.79 on the least founder for `own_256`, .85 to .88 for `both_256`. On every life under .90 the
probe fitted on trained combinations reads the verb and the agent's number of a held-out clause
from the message at .99 to 1.00. The settled association holds the agent's number at .97 to 1.00
at `GO`, and the verb at .96 to 1.00 at `GO` without the read of one block (.72 to .95 with it)
and at .96 to 1.00 at the last word. The panel's verb stem is right (.99 to 1.00) and its
agreement is not (.94 to .97): the meaning is held, in the message and in the association, and
the readout says the wrong agreeing form of the verb for an agent, number and verb no lesson
showed together. Founder 10028609 is under .90 in every arm (.67, .73, .81, .73).

With the read of one block the settled association at `GO` holds the patient's noun at .28 to .43
by the same probe, against .55 to .77 without it, and the patient is said at .99: the brain with
the read looks the patient up as it says it.

## Amendment 1: lengths and depths from the state

Written after part 1 was read and before any life of these arms. H1c holds on `both_256`, so the
protocol's second part runs on it.

**Founders.** 10128600 to 10128605 (six, no life in any lane).

| Learner | What it is | Parameters |
| --- | --- | --- |
| `both_256` | part 1's arm: `message=280`, `message_read=True`, `history_decays=(0.0,)`, 256 association neurons, every row taught, the recall read barred | 596,412 |
| `default` | the default composition with its recall read free: where the read loses the long panels, as in the first protocol | 28,716 |
| LSTM | the chamber's control at `both_256`'s count, online (one pass) and offline (to the stopping rule) | 596,598 |
| transformer | the same | 592,058 |

**Readings.** Whole sentences on `clause_long` (three and four distractors), `dyck_deep` (depths
four and five) and `path_long` (three hops), beside the trained shapes; multiply-accumulates per
life; on `both_256`, each word kind of the long panels said with the previous words given, and
probes of the clause components fitted on trained shapes and scored on longer clauses, from the
message and the settled association at `GO`; on `default`, the read's delivery per term
(`located`).

**Gate A2s.** `clause_long`, `dyck_deep` and `path_long` at .80 each on 5 of 6 founders, said
from the state. A failure carries its failing term.

**Expectations.** A2s fails on all three at .00 to .02: part 1's twelve `both_256` lives read
.00, .00 and .00 on these panels. Every control is under .80 on all three. The failing terms: in
a clause of three or four distractors the message's blocks (`t mod 4`) hold a distractor or the
patient where no lesson put one, so the words from the third distractor on are lost and the
agent and the verb are said; in deeper brackets the closes past depth three; in a longer path
the third hop.

## Result of Amendment 1

Read on the same box, founders 10128600 to 10128605, 12 brain lives and 24 control lives on
commit 2575a6ec.

| Learner | Parameters | Multiply-accumulates | clause | clause_held | roles | clause_long | dyck_deep | three hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `both_256`, from the state | 596,412 | 2.49e13 | .98 | .94 | .94 | .00 | .00 | .00 |
| `default`, the read free | 28,716 | 9.70e11 | .95 | .87 | .87 | .00 | .03 | .22 |
| LSTM, offline | 596,598 | 3.24e13 | 1.00 | .99 | .92 | .00 | .02 | .00 |
| LSTM, online | 596,598 | 1.42e12 | 1.00 | .96 | .84 | .00 | .00 | .00 |
| transformer, offline | 592,058 | 3.77e13 | 1.00 | .98 | .99 | .00 | .00 | .00 |
| transformer, online | 592,058 | 1.42e12 | .88 | .85 | .77 | .00 | .00 | .00 |

**A2s fails** on all three long panels, 0 of 6 founders at .80 on each, as stated; every control
reads .00 to .02 on them. On the trained shapes the six `both_256` lives read clauses .98,
held-out combinations .94 and roles .94, as part 1's twelve did.

**Where the long panels are lost, from the state.** With the previous words given:

- **Longer clauses.** The verb is said at .84 and .68 (three and four distractors), the agent at
  .79 and .67, and the words from the second distractor on at .14 to .51; the patient at .57 and
  .37. Probes fitted on trained shapes and scored on longer clauses read, at `GO`, the verb from
  the message at 1.00, the patient's noun at .92 to 1.00, the agent's at .89 to 1.00, and the
  first and second distractors' nouns at .74 to .78 and .56 to .65; from the settled association
  the verb at 1.00 and the patient's noun at .58 to .70. A clause of three or four distractors
  puts a fifth and sixth moment in blocks 1 and 2 (`t mod 4`), beside the agent's and the first
  distractor's: the patient and the verb are held, and the distractors share their blocks with
  the moments a lesson put there.
- **Deeper brackets.** Opens at depths four and five are said at .63 and .56 and closes at .48
  and .55: the brackets past depth three are the term.
- **Three hops.** The start is said at .99 and the hops at .45, .37 and .42: a path's hops are
  said from the read, which is barred.

**The default composition with its read free** loses the same panels at the read: on longer
clauses the read delivers the verb's moment at .13 and .22, and the distractors from the second on
at .30 to .59; on deeper brackets the closes at .07 to .13; on three hops it delivers the hops at
.88, .65 and .53 and says them at .90 to .97 when delivered.
