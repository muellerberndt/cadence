# The whole sentence held and said: the second declared run (cadence#308)

Declared on 2026-11-13, committed before the first life. The first declared run
([SENTENCE_PROTOCOL.md](SENTENCE_PROTOCOL.md#result-of-the-declared-run)) located two terms: a
heard word carried no role, so heard words joined the last group by position, and the carried
state kept the latest moments and lost the earlier ones while the sentence was said.

## What is built

- **A store the brain works itself** (`message_settled=True`, `message_operations`; docs/memory.md,
  "A store the brain works itself"). The message region holds one block per group, the group's
  head; a third motor slot, the role slot, names at every moment what the moment is to the
  sentence, and that choice runs fixed operations on the store (`set`, `write`, `shift`, `pop`,
  `move`, `clear`). A word's place in the store is the brain's own settled choice from the words
  heard.
- **A block takes the moment alone** (`message_alone=True`): the association's settle of the
  moment's observation with every memory silent, the same activity for one word wherever it was
  heard.
- **The instrument** (`sentence.py`): the grammar chamber's brain saying each word by its parts
  (a stem slot and a form slot, #302's protocol 7) beside the role slot; lessons name each heard
  moment's word and role (an echo) and each said word's role; the full mix holds the first
  grammar's sentences heard as concepts, clauses heard as words and said back, the second
  grammar from concepts, one wording said from the other, and questions about a clause heard as
  concepts or as words. The recall slots of the heard sentence are emptied at `GO` in every
  lesson and reading (a path keeps its facts and loses its start).

## Development

Founders 0 to 5, the full mix, 6,000 batches (the first row 4,000), lessons where the answer
differs (`learning_surprise`), the design `alone` (a block takes the moment alone; brackets are
heard into a sequence and worked onto a stack as they are said). Said from the state with the
recall read barred; `roles` counts a twin pair said right only where both twins are.

| Arm | Parameters | `clause` | `clause_held` | `roles` | `path` | `clause_long` | `back_held` | `para_12_held` | `para_21_held` | `ask_concepts_held` | `ask_words_held` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `alone`, 4,000 batches | 683,528 | .99 | .93 | .80 | .13 | .84 | .92 | .82 | .91 | .86 | .83 |
| `alone` | 683,528 | .98 | .87 | .70 | .16 | .76 | .87 | .85 | .87 | .92 | .89 |
| `alone`, the echo's read lesson held (`quiet`) | 683,528 | 1.00 | .94 | .91 | .12 | .90 | .97 | .90 | .96 | .96 | .95 |
| `alone` with the read of one head (`message_read`) | 826,408 | .93 | .90 | .83 | .11 | .72 | .91 | .78 | .92 | .94 | .93 |
| the read of one head with `quiet` | 826,408 | .99 | .93 | .81 | .14 | .79 | .88 | .83 | .83 | .92 | .92 |
| the facts echoed, the start said out | 683,820 | 1.00 | .98 | .89 | .01 | .88 | .98 | .94 | .98 | .93 | .91 |
| the facts echoed, `quiet` | 683,528 | 1.00 | .99 | .97 | .01 | .96 | .87 | .85 | .87 | .95 | .88 |
| a path's start clears the store, `quiet` | 683,528 | 1.00 | .98 | .92 | .07 | .84 | .94 | .94 | .95 | .93 | .98 |
| the same, the read's lesson also held where the bar empties the slots (`barred`) | 683,528 | .96 | .89 | .83 | .31 | .93 | .90 | .90 | .88 | .92 | .95 |
| LSTM offline (founders 0 to 2) | 686,366 | 1.00 | 1.00 | .99 | .81 | .00 | 1.00 | 1.00 | .99 | 1.00 | 1.00 |
| transformer offline (founders 0 to 2) | 681,378 | 1.00 | 1.00 | .99 | .83 | .05 | .99 | 1.00 | 1.00 | 1.00 | .99 |

What development located:

- **An echo's lesson stepped the recall read's address.** At a heard moment the moment an echo
  names is not yet in the recall slots, so the read's address lesson there steps toward another
  moment. Holding the read's address and position rates at zero for the echo (`quiet`) lifts role
  twins from .70 to .91, held-out combinations from .87 to .94 and longer clauses from .76 to .90
  on the same founders.
- **The read of one head at a time costs** (`message_read` over the worked store, beside the heads
  read at once): held-out combinations .90 against .94, longer clauses .72 against .90, and with
  `quiet` .93 and .79. Read through the port alone (`message_amplitude=0.0`) the brain says
  clauses at .11 to .30 after 1,000 to 4,000 batches (checkpoints of 32 sentences).
- **A path's facts are heard as a clause's words.** The facts `X near Y end` begin as a clause
  heard as words does (`agent near distractor`); where no lesson names their role the brain names
  a clause's (per life `start` 208, `distractor` 122 and `agent` 64 of 896 fact moments), and the
  store holds them as a clause's parts. Echoing the facts with the role `none` costs the say-back
  of held-out clauses (.87 against .97): the same words get two roles. A path's start that clears
  every other group (`alone_clean`) keeps the clause panels and leaves paths at .07.
- **The recall read never learned a path's hops.** Read from saved brains of three arms, at the
  moment that hears the start said the read attends the start's emptied slot (.78 to .99), and
  at a hop its own said words (.31 to .48) and other facts (.21 to .44), the fact's next word .00
  to .17; paths read the same with the start's slot emptied and kept (.11 and .11, .00 and .00,
  .14 and .13). Said with the previous words given, the first hop is right on .27 of paths. The
  read is taught mostly on clauses, whose slots the bar empties at `GO`, so it learns to attend
  its own said words.
- **The read's lesson held where the bar empties the slots** (`barred`): the read is taught on
  paths alone, and on four of six founders it learns part of the chain (paths .36 to .48; on one
  founder the hop reads the fact's `near`, whose next is the hop, at .54). On two it attends its
  own words (paths .03 and .11), and one founder of six fails on clauses (.73, held-out .50, role
  twins .19). A faster address (`recall_address=0.01`) and a path's every row taught leave paths
  at .08 to .26.

## Founders

10828600 to 10828611 (twelve, no life in any lane).

## The arm

Every life: 8,000 batches of the full mix (the controls taught the same batches), panels of four
batches of 16 sentences, probes on 16 batches per shape.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `quiet` | 256 association neurons; the worked store of design `alone_clean` (`message=256`, `message_settled`, `message_operations` of 14 roles over 8 groups of 8 blocks, `message_stacks`, `message_alone`); three motor slots (stem, form, role); `history_decays=(0.0,)`; lessons where the answer differs; echoes at the heard moments that leave the recall read's address as it is | 683,528 |
| `barred` | `quiet`, and a said word's lesson in a sentence whose meaning the bar empties (every family but a path) leaves the recall read's address as it is | 683,528 |

## Controls

The chamber's LSTM and transformer taught the same lesson batches at the arm's parameter count
(within 2%), online (one pass) and offline (to the controls' stopping rule), and offline with the
echoes and roles taught beside each word (a linear head of their own reading the role from the
hidden state), on all twelve founders, saying the same panels and probed the same way.

## Readings

- The chamber's panels heard as concepts (`clause`, `clause_held`, `roles`, `dyck`, `path`,
  `clause_long`), the say-back panels heard as words (`back`, `back_held`, `back_long`), and the
  full mix's panels (`para_12`, `para_21`, `para_22`, `g2`, `ask_concepts`, `ask_words`, each on
  trained and held-out combinations), every one said from the state with the recall read barred.
- Probes of every clause component after the last heard word of a clause heard as words, fitted
  on trained combinations and scored on held-out ones, from the store read with the association;
  and across wordings, fitted on one grammar's trained combinations and scored on the other's
  held-out ones.
- Each word kind said with the previous words and roles given; the roles the brain names.
- Parameters, multiply-accumulates and seconds per life; each arm less each matched control
  offline, paired by founder.

## Gates (twelve founders)

- **H1, held:** after the last heard word of a clause heard as words, the least clause
  component's probe at .95 on 10 of 12, from the store read with the association.
- **H2, said:** from that state, `clause` and `clause_held` at .90 and `clause_long` at .80 on 10
  of 12.
- **H3, the same meaning:** one wording said from the other (`para_12` and `para_21`, trained and
  held-out combinations) at .90 on 10 of 12, and the least component's probe across wordings at
  .95 in both directions on 10 of 12.
- **H4, used:** the questions about a clause heard as concepts and as words (`ask_concepts`,
  `ask_words`, trained and held-out) at .90 on 10 of 12.
- **P, parity:** on every panel read above, the arm less the matched transformer offline and less
  the matched LSTM offline, paired by founder: the mean at or above -.03 and not under zero by two
  standard errors.

## Expectations, stated before the run

- `quiet`: H1 holds (12 of 12). H2: clauses 12 of 12, held-out combinations 10 to 12, longer
  clauses 9 to 11 of 12. H3: one wording from the other held out (`para_12_held`) under .90 on two
  to four founders, the probes across wordings at .95 on 12 of 12. H4: 9 to 11 of 12 on each
  panel. P fails on paths (-.70 against both controls) and on held-out panels by .03 to .07
  (`back_held`, `para_12_held`, `g2_held`, `ask_concepts_held`).
- `barred`: paths .25 to .45, P fails on them; the clause panels as `quiet` on ten or eleven
  founders and failing on one or two, so H2 holds or fails at the bar.
- The controls offline read every panel at .97 or above but longer clauses (.00 to .05) and
  paths (.79 to .88).

## Result
