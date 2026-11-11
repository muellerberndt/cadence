# Longer clauses said from the state: declared protocol 5 (cadence#286)

Declared on 2026-11-10, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed. It follows
[the fourth protocol](MEANING_PROTOCOL_4.md), whose arm `both_256` says clauses, held-out
combinations and role twins from the brain's held state at .90 on 12, 11 and 11 founders of
twelve, and longer clauses, deeper brackets and three hops at .00.

## The question

The fourth run located the longer clauses: the `t`-th moment of an episode joins block
`t mod 4`, so in a clause of three or four distractors the moments from the fifth on land in
blocks that hold the verb, the agent and the first distractor. The message holds the verb (1.00
by a probe fitted on trained shapes) and the patient (.92 to 1.00), and the distractors that
share a block with an earlier moment at .56 to .78. Does a message with a block for every moment
say longer clauses from the state?

## Founders and arms

Founders 10328600 to 10328605 (six, no life in any lane). Every arm: `both_256`'s settings
(`message=280`, `message_read=True`, `history_decays=(0.0,)`), 256 association neurons, every
row taught, the recall read barred from the meaning at `GO`, 4,000 batches of the first grammar.
The longest declared clause (`clause_long`, four distractors) holds 7 heard moments and 16 with
`GO` and the words said.

| Arm | Blocks (`message_phases`) | Parameters |
| --- | --- | --- |
| `both_256` | 4: the reference | 596,412 |
| `both_p7` | 7: a block for every heard moment of the longest clause | 813,171 |
| `both_p8` | 8: one past it, `GO` in a block of its own | 885,424 |
| `both_p16` | 16: a block for every moment of the longest clause's episode, the words said included | 1,463,448 |

## Readings

Whole sentences on `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `path_long`
(three hops), `clause_long` and `dyck_deep`; each arm less `both_256` paired by founder;
multiply-accumulates and seconds per life; the learned gains. Where the long panels are lost:
each word kind of `clause_long`, `dyck_deep` and `path_long` said with the previous words given,
and probes of the clause components fitted on trained shapes and scored on longer clauses, from
the message and the settled association at `GO`.

## Gate A2c

An arm holds where `clause_long` reads .80 on 5 of 6 founders, `clause`, `clause_held` and
`roles` read .90 on 5 of 6, and on `dyck`, `path` and three hops its paired mean against
`both_256` is at or above -.03 and `both_256` is not above it by two standard errors.

Deeper brackets are read and located in this run, with no mechanism and no gate. Three hops from
the state are a read problem by the fourth run's location (the start is said at .99 and the hops
at .37 to .45, a path's hops coming from the read, which is barred): they are recorded and left.

**Expectations, stated before the run.** A2c fails on every arm. A block holds a position in the
episode, and a longer clause puts its later distractors and its patient at positions no lesson
filled with them: with 7 or 8 blocks the third and fourth distractors and the patient land in
blocks 4 to 6, where a lesson's clause put its patient, `GO` or the first word said; with 16
blocks a lesson writes blocks 0 to 12 and a long clause's words said land in blocks 13 to 15,
which no lesson wrote. `clause_long` reads under .20 on every arm. The trained panels hold within
.03 of `both_256` on `both_p7` and `both_p8`; `both_p16` loses some of them.

**What a failure names.** Where the probe reads the later distractors and the patient from the
message at .95 and over and the words are lost, the meaning is held in blocks the readout never
read them from: the code holds position and the readout does not carry a role from one position
to another. Where the probe falls too, the message does not hold them.

## Compute

One AWS c7i.16xlarge spot box (Name=cadence-097-meaning), one process per life, one BLAS thread
per process.

## Result

Read on one c7i.16xlarge, founders 10328600 to 10328605, 24 lives on commit d3977b61.

| Arm | Blocks | Parameters | Multiply-accumulates | clause | clause_held | roles | dyck | path | three hops | clause_long | dyck_deep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `both_256` | 4 | 596,412 | 2.55e13 | .97 | .90 | .89 | .91 | .48 | .00 | .00 | .01 |
| `both_p7` | 7 | 813,171 | 3.75e13 | .99 | .90 | .87 | .89 | .38 | .00 | .00 | .01 |
| `both_p8` | 8 | 885,424 | 4.17e13 | .99 | .97 | .97 | .93 | .43 | .00 | .01 | .00 |
| `both_p16` | 16 | 1,463,448 | 7.31e13 | .99 | .91 | .92 | .91 | .36 | .00 | .00 | .00 |

**A2c fails on every arm**: `clause_long` reads .00 to .02 on every life, as stated. Against
`both_256`, paired, paths fall .10 (.03), .05 (.02) and .12 (.05) with 7, 8 and 16 blocks, each
beyond two standard errors.

**Where longer clauses are lost.** With a block for every heard moment the message holds the
first distractor of a long clause apart (.97 by a probe fitted on trained shapes, .78 with four
blocks) and the patient falls (.63, .97 with four blocks): a long clause puts its patient in a
block where no lesson put a patient, and the reading set up on trained shapes does not find it
there. With the previous words given, the verb of a four-distractor clause is said at .00 to .07
with 7, 8 or 16 blocks (.69 with four), the fourth distractor at .08 to .22 and the patient at
.00 to .38. A block holds a position in the episode, and the readout does not carry a word's role
from one position to another: more blocks keep the moments apart and leave them at positions the
readout never read.

**Deeper brackets**, read with no mechanism: opens at depths four and five are said at .56 to .70
and closes at .45 to .61 with the previous words given, in every arm; the number of blocks does
not move them. **Three hops** from the state: the start is said at .98 to .99 and the hops at .14
to .46, recorded and left.

**Against the expectations.** A2c failed on every arm and `clause_long` read under .20, as stated.
The trained panels held within .03 on `both_p7` and `both_p16` and over it on `both_p8`, which
was not stated: with 8 blocks, `GO` in a block of its own, held-out combinations and role twins
read .97 and .97 (6 and 6 of 6 at .90; +.08 and +.09 against `both_256`, paired, beyond two
standard errors), and paths fall .05. Paths fell on every arm, which was not stated.
