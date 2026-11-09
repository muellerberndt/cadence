# Sentences said from a held meaning: declared protocol (cadence#286)

Declared on 2026-11-09, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

In the grammar chamber a meaning arrives as concept moments and the brain says the sentence at
`GO`. The default composition looks each word's content up in its recall memory. Can the whole
meaning be held in the state the brain carries, and the sentence be said from that state? And does
a held meaning beside the read lift what the read alone does?

## What development found (founders 0 to 5, design input)

- **Where the meaning lives.** With every recall slot emptied at `GO` the default composition says
  .00 of every panel, and with the working trace emptied nothing moves. The carried state holds a
  partial copy (the agent's noun .73, a path's start .99 by a linear probe) that the readout does
  not use. The matched LSTM holds every component at .94 to 1.00 and says the panels from its state.
- **Existing parts as a message.** With the read barred from the meaning in every lesson, the
  working trace says clauses at .02 (decay 0 and decay .8) and the heard history at two decays .31.
- **A message of fixed sparse codes.** A region that holds the episode as sums of sparse codes of
  the heard moments is used by the brain: clauses .69 from the state alone. The order of two
  moments of the same kind is not read from magnitudes (primacy or recency weights: the first of two
  distractors .33 to .65). Phased unions, where a block's content is present or absent, carry it
  (.94 to 1.00 at 128 association neurons with every row taught). A buffer of the raw observations
  reads lower than the codes (.59 against .69).
- **Held-out combinations need a wide code.** At 70 units a block the state alone reads held-out
  clauses .63 to .74; at 280 units a block .89 (.81 to .98), with 128 association neurons and every
  row taught. At 560 units a block two lives read .94 and .64.
- **A message the brain restates in motor slots** (one slot per role, carried by the efference
  copy) says trained clauses at .50 to .92 and held-out clauses at .03 to .18: slots settled
  together bind what was taught together.
- **Paraphrase by the read.** The default composition, taught paraphrases with its read free, says
  the second grammar's sentence of a heard first-grammar sentence at .99, and at .96 on held-out
  combinations (three founders).
- **Beside a free read** the message changes no panel beyond the founders' spread on six founders,
  and three hops fall (.40 to .07). The present moment's code alone on the same neurons, a sham
  with no held meaning, read three hops at .71.

## Instrument

`benchmarks/grammar/meaning.py` (`plan`, `lives`, `read`) and `benchmarks/grammar/controls.py`. A
brain life is `Brain.compose(35, 18, modules=(64,), seed=founder, backend="cpu", recall=True)`, the
default composition, with the arm's settings. Lessons, panels and scores are the chamber's
(`grammar.py`): the first grammar's 4,000 batches from `SeedSequence([founder, 11])`, 64-sentence
panels from `SeedSequence([founder, 33])`, whole sentences said closed loop and greedy.

**The message.** `cadence.Message` (a class with no neurons, added on this branch): each heard
moment's observation is given a fixed sparse code (`units` code units, each reading 16 of the
observation's ON and OFF channels, the 5 largest sums set to one), and the `t`-th moment of the
episode joins the union of block `t mod 4`. Its state is 4 blocks of `units`; its key is the union
over the blocks. The instrument writes the state after every act onto neurons the association
reads through a plastic projection (heard-history blocks, driven at 3 times `scale`), an emulation
with no change to any brain: the write is fixed and nothing of it is learned. The brain's own words
join the message as it speaks.

**The bar.** `bar`: at `GO`, in lessons and in readings, the recall slots of the meaning's moments
are emptied (keys, values and nexts at zero). A clause's and a bracket sentence's meaning is every
moment heard; a path's is its start, and the world's facts stay.

## Founders and arms

Founders 9628600 to 9628605 (no life in any lane).

| Arm | The brain |
| --- | --- |
| `default` | the default composition, the read free: the reference |
| `barred` | `default` with the read barred: the floor |
| `message` | `default` with the message at 70 units a block, `scale` .15, the read free |
| `message_wide` | `default` with the message at 280 units a block, `scale` .05, the read free |
| `sham` | `message` holding the present moment's code alone (no held meaning) |
| `state` | the message at 280 units a block, `scale` .5, the read barred, 128 association neurons, every row taught (`learning_surprise=False`) |
| `state_default` | `state` at the default composition's width and lessons |
| `restated` | one motor slot per role beside the word's (agent, verb, patient, two distractors), every moment taught the notes so far, the efference copy carrying them; the read barred from a clause's meaning; 128 association neurons |
| `paraphrase` | `state` with 8 blocks of 140 units, taught 6,000 batches: the first grammar's mix (one half), the second grammar's clauses from concepts (one quarter) and paraphrases (one quarter: the first grammar's sentence heard as words, then `DIALECT`, the second grammar's sentence said) |
| `paraphrase_read` | `default`, the read free, taught the `paraphrase` arm's 6,000 batches: the reference that looks the heard sentence up |
| `capacity_u70`, `_u140`, `_u280` | `state` with that many units a block, taught clauses of zero to four distractors alone |
| `capacity_m64`, `_m256` | `state` at 280 units a block with that many association neurons, the same lessons |

`default`, `message` and `state` also live the persistence schedule (the second grammar for 1,500
batches, then the first for 500) and read the context twins and the two orders of a meaning.

**Controls.** The chamber's LSTM and transformer at the parameter counts of `default`, `message`
and `state`, online (one pass in the brain's order through the persistence schedule) and offline
(the first grammar to the stopping rule), scored independently on the same panels.

**Other chambers.** The sequence battery's look back, copying (unique and repeated), induction,
n-back, two hops and a period of 40, 1,500 lessons each, on the default composition with the
`message` arm's message and without it; and the goal-route chamber where its file is present.

## Readings

Whole sentences on `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `clause_long`,
`dyck_deep`, `path_long`; the context twins `ctx_agent`, `ctx_verb`, `ctx_dyck` intact, removed,
and erased from the recall memory, the working trace or the message; each word said right by kind
with every previous word given (`said`, the readout, on fresh sentences); linear probes of every
component of the meaning from the message as written, the carried trace and the settled association
at `GO` and at the last word (`held`; 5,760 clauses per `state` life, 1,920 per `default` and
`message` life, 70% fitted and 30% scored); seconds and analytic multiply-accumulates.

## Gates

A gate on lives holds where at least 5 of the 6 founders reach its bar.

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| H1, said from the state | `state` | `clause`, `clause_held`, `roles` | .90 each |
| H2, held | `state` | every clause component's probe from the message at `GO` and at the last word | .95 each |
| H2u, used | `state` | every clause word kind said right with the previous words given | .95 each |
| H3, one meaning in two orders | `state` | the key of the two orders closer than any other meaning's (`closer`), and both orders said right | .95, .90 |
| H4, paraphrase | `paraphrase` | `para` and `para_held` | .80 each; `paraphrase_read` beside it |
| H5, capacity | `capacity_*` | clauses said exactly per number of distractors, per width | reported: the most distractors said at .90 |
| R, retrieval by the code | no brain | 256 records (128 clauses, each beside its role-swapped twin): a reordered cue finds its record, a role-swapped cue finds the twin's | .90 each; 1,024 records, partial cues, words and crossed forms reported |
| A1, #286 | `default`, `message` | `clause_held`, `roles`, `ctx_agent`, `ctx_verb`, `ctx_dyck` (both twins) | .90 each; the twins removed or erased reported |
| A2, #286 | every arm, controls | `clause_long`, `dyck_deep`, `path_long` | .80 each, or a measured negative with the failing term from `located` |
| A3, #286 | `default`, `message`, `state`, controls | after the second grammar, `clause`, `dyck` and `path` within .05 of their reading before | per arm |
| M1, the message beside the read | `message` less `default`, paired by founder | each panel's mean difference | at least .05 and beyond two standard errors; the same for `message_wide` and for `sham` |
| K, other chambers | message less default, paired | each battery task's accuracy, the goal route's first hop | reported; a fall beyond .03 and two standard errors is named |

**Expectations, stated before the run.** H1 holds on `clause` and fails on `clause_held` or `roles`
(development: .98, .89, .91 on three founders, one founder of three under the bar on each). H2
holds from the message. H3 holds on the key; the phased state itself differs between the two
orders. `restated` fails held-out clauses. M1 fails: no panel lifts by .05, and three hops fall.
`sham` lifts three hops. A2 and A3 fail for every learner, the controls included.

**What a failure names.** Where H1 fails with H2 passing, the meaning is held and the sentence is
lost at the readout: `said` names the word kind, and the panel (trained, held-out combination,
roles) names whether the readout fails on what it was taught or on new combinations of it.
`capacity_*` separates the code's width from the association's. Where M1 fails, the twins erased
from the message say whether the brain used the message at all.

## Compute

One AWS c7i.16xlarge, one process per life, one BLAS thread per process.

## Amendment 1: the moment before, as heard

Written after the first block's sentences were read and before any life of this arm on the declared
founders.

In the first block the `sham` arm, the code of the moment before alone on the message's neurons,
read roles and three hops above `default` on the declared founders, and `default` read A1's
held-out lines under their bar. The library holds the same carrier as an opt-in of its own, the
heard history at decay 0 (`history_decays=(0.0,)`: the observation of the moment before, as heard,
on a region the association reads). Development founders 0 to 5 read it at clauses 1.00, held-out
clauses .97, roles .96, brackets .89, paths .98 and three hops .64, against the default's .98, .91,
.96, .88, .95 and .40.

**Arm `heard`:** `default` with `history_decays=(0.0,)`, founders 9628600 to 9628605, with the
context twins. **Gates:** A1's five lines at .90 on 5 of 6 founders, and M1's paired difference
against `default` per panel. **Expectation:** held-out clauses and three hops lift, and A1 holds.

## Amendment 2: the state alone at 256 association neurons

Written after the first block's capacity arms were read and before any life of this arm.

The capacity arms read the association's width as a term of its own: at 280 units a block, clauses
of four distractors are said from the state at .30 with 64 association neurons, .66 with 128 and
.94 with 256, and clauses of three at .83, .97 and 1.00. H1 failed at 128 neurons on combinations no
lesson showed (`clause_held` 2 of 6, `roles` 0 of 6).

**Arm `state_256`:** `state` with 256 association neurons (the message at 280 units a block,
`scale` .5, the read barred, every row taught), founders 9628600 to 9628605, with the context twins
and the two orders of a meaning. **Controls:** the chamber's LSTM and transformer online at its
parameter count. **Gates:** H1's three panels at .90 and H2u at .95, on 5 of 6 founders.
**Expectation:** `clause_held` and `roles` rise over `state`; whether they reach the bar is open.
