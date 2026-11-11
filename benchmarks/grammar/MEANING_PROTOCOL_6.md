# A message keyed by role: declared protocol 6 (cadence#301)

Declared on 2026-11-11, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed. It follows
[the fifth protocol](MEANING_PROTOCOL_5.md) and holds the first runs of #301.

## The questions

1. **`GO` in a block of its own.** With 8 blocks the fifth run read held-out combinations and role
   twins at .97 and .97 on 6 of 6 founders (+.08 and +.09 against `both_256`, paired), with paths
   .05 under. Does it hold on twelve fresh founders?
2. **A message keyed by role.** The fifth run located longer clauses: a block holds a position, a
   long clause puts its patient in a block where no lesson put a patient, and the readout set up
   on trained shapes does not find it there. If a moment's role chooses its block, every patient
   joins the patient's blocks wherever in the meaning it was heard. Does the brain say held-out
   combinations (20 of 24) and longer clauses (.80 on 5 of 6) from such a message?

## The opt-in

`Brain.compose(message=units, message_roles=groups)` (and `cadence.Message(..., roles=groups)`),
off by default. `groups` is a tuple of groups of input channels. A heard moment with a channel of
group `r` on (the first such group) joins block `count_r mod message_phases` of that group's
`message_phases` blocks, `count_r` the moments of role `r` the episode heard before it; a moment
of no group joins the last group's blocks by the same rule. The region holds
`(len(groups) + 1) * message_phases * units` neurons, one learned gain per block under
`carried_gain`. The write is fixed, holds no learned parameter, and nothing outside the brain
writes onto a neuron. Without `message_roles` every bit is as on 0.97.0.

In the chamber the groups are the agent (`AGENT`), the patient (`PATIENT`), a distractor (`MOD`),
the verb (`SEE`, `CHASE`), `GO`, and the brain's own words (`SPEAK`); bracket and path moments join
the last group. With 4 blocks a group, a clause's agent, patient, verb and `GO` each have blocks of
their own, its first to fourth distractors lie in the distractor blocks 0 to 3, and its words
said lie in the words' blocks by the order said.

## Founders and arms

Founders 10428600 to 10428623 (24, no life in any lane). Every arm: 256 association neurons,
`message=280`, `message_read=True`, `history_decays=(0.0,)` (the 0.97.0 default), every row taught
(`learning_surprise=False`), the recall read barred from the meaning at `GO` in lessons and
readings, 4,000 batches of the first grammar on 0.97.0.

| Arm | Founders | The message | Parameters |
| --- | --- | --- | --- |
| `both_256` | all 24 | 4 blocks by position: the reference | 596,412 |
| `both_p8` | the first 12 | 8 blocks by position: `GO` in a block of its own | 885,424 |
| `role_256` | all 24 | blocks by role, 4 a group, 7 groups (28 blocks) | 2,330,484 |

**Controls.** The chamber's LSTM and transformer at `both_p8`'s count (884,718 and 878,834
parameters) and at `role_256`'s (2,325,446 and 2,339,298), online (one pass) and offline (to the
stopping rule), on the first six founders. `both_256`'s controls at 596,412 parameters were read in
protocol 4's Amendment 1.

## Readings

Whole sentences on `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `path_long`
(three hops), `clause_long` and `dyck_deep`; each arm less `both_256` paired by founder;
multiply-accumulates and seconds per life. Where sentences are lost: each word kind of held-out
clauses and of the long panels said with the previous words given, and probes of every clause
component from the message and the settled association at `GO`, fitted on trained combinations
and shapes and scored on held-out combinations and on longer clauses.

## Gates

- **G8**, `GO` in a block of its own: `both_p8` reads `clause`, `clause_held` and `roles` at .90 on
  10 of 12, and on `dyck`, `path` and three hops its paired mean against `both_256` is at or above
  -.03 and `both_256` is not above it by two standard errors.
- **R1**, held-out combinations from a message keyed by role: `role_256` reads `clause_held`,
  `clause` and `roles` at .90 on 20 of 24, with the same rule on `dyck`, `path` and three hops.
- **R2**, longer clauses: `role_256` reads `clause_long` at .80 on 5 of the first six founders
  (10428600 to 10428605), beside the controls.

**Expectations, stated before the run.** G8 fails on held-out combinations (8 to 10 of 12) with
role twins at 10 to 12 and paths .05 under: the fifth run's 6 of 6 is one sample of six. R1 fails
on held-out combinations at 17 to 20 of 24: the loss of held-out combinations sits at the verb's
form (#302), which the message's layout does not reach. R2 fails: with the distractors in their
own blocks by order, a four-distractor clause puts its third and fourth distractors in blocks 2 and
3 of the distractor group, which no lesson writes (lessons hold at most two); the patient, the
agent and the verb are said at .90 and over with the previous words given and the third and fourth
distractors under .30, and `clause_long` reads under .20. The controls read `clause_long` at .00 to
.05.

**What a failure names.** Where R2 fails with the patient and the verb said and the later
distractors lost, the term is the blocks no lesson wrote; where the patient is lost too, the role's
block does not carry the patient to a longer clause. Where R1 fails, the held-out word kinds say
whether the verb's form or a role's block is lost.

## Compute

One AWS c7i.16xlarge spot box (Name=cadence-098-meaning), the 60 brain lives in one pass and the
48 control lives after them, one process per life, one BLAS thread per process.
