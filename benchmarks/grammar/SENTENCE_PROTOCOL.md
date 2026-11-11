# The whole sentence in the hidden layers: declared protocol (cadence#308)

The runs of the build for cadence#308, each declared here and on the issue before its first life.
A failed gate is reported as failed.

## Step 1: the instrument and the floors

Declared on 2026-11-11, committed before the first life. No gate: these are the floors and the
ceilings the later steps are read against.

**The instrument** (`benchmarks/grammar/meaning.py`, `plan --floors`, `lives`, `read --floors`).
- **Lessons** (`"train": "sayback"`): 4,000 batches, one half the first grammar's mix heard as
  concept moments, one half clauses of trained combinations heard as their words and said back at
  `GO` (`sayback`). Every lesson and every reading has the recall slots of the heard sentence
  emptied at `GO` (the bar of the earlier protocols), and no arm but the reference holds a
  `message`.
- **Said:** the chamber's panels heard as concepts (`clause`, `clause_held`, `roles`, `dyck`,
  `path`, `clause_long`) and the say-back panels heard as words (`back` of trained combinations,
  `back_held` of held-out ones, `back_long` of three and four distractors), whole sentences in a
  closed loop.
- **Held:** linear probes of every clause component (agent's noun and number, verb, the verb's
  form, patient's noun and number, each distractor) fitted on trained combinations and scored on
  held-out ones, after the last heard moment, at `GO` and at the last word, from the settled
  association, the carried trace, and where an arm holds a store, the store and the store read
  with the association (`joint`). Heard as concepts and heard as words.
- **Controls:** the chamber's LSTM and transformer taught the same lessons, online (one pass) and
  offline (to the stopping rule), saying the same panels, their hidden state probed at the same
  moments.

**Founders** 10628600 to 10628605 (six, no life in any lane).

| Arm | The brain | Parameters |
| --- | --- | --- |
| `default_64` | the 0.97.0 default (64 association neurons), words said by their parts, the recall read barred | 30,692 |
| `plain_256` | 256 association neurons, every row taught, the recall read barred: the working trace carries the association's last state | 152,804 |
| `slow_256` | `plain_256` with `working_memory_decay=0.8`: the carried state at a leak | 152,804 |
| `both_256` | the reference with a fixed code: `message=280`, the read of one block, also read with the message silent | 595,536 |

Controls at `plain_256`'s count (LSTM 153,138, transformer 154,770 parameters) and at `both_256`'s
(596,598 and 592,058).

**Expectations.** `default_64`, `plain_256` and `slow_256` say .00 to .10 of every clause panel
heard either way, and their settled association holds the agent's noun under .80 after the last
heard moment. `both_256` says clauses heard as concepts at .97 and over and clauses heard as words
under them; its store read with the association holds every component at .95 and over, its
association alone under .60 for the patient. The LSTM and the transformer offline say every
trained and held-out panel at .95 and over, hold every component at .95 and over, and say .00 of
the longer clauses.

## The declared run (H1, H2 and parity)

Declared on 2026-11-11, committed before the first life. Steps 3 to 6 of the build are compressed
into this one run under the release schedule; the kill tests are read from development and
reported beside it.

**What development found** (founders 0 to 5, design input, the recall read barred, said back from
the state heard as concept moments; clause, held-out combinations, role twins):
- the 0.97.0 default .05, .00, .00, and at 256 association neurons .08, .00, .00;
- the carried state at a leak (`working_memory_decay=0.8`) .42, .13, .06; with lessons at the
  heard moments that name each heard word (echo lessons) .59, .45, .46; at a leak of .9 and .95
  .51 and .58 on clauses, role twins .15 and .33;
- the settled store (`message_settled`, one block a role, read one block at a time through the
  port, the association reading no block directly): .48, .24, .17, and with echo lessons .67,
  .51, .36; with the carried state at a leak beside it .67, .53, .24. After the last heard moment
  the store read with the association holds the least clause component at .89 to .91 against .35
  to .37 for the association alone;
- **kill tests, read on development lives:** the store read through the shared port against each
  block read through its own synapses (`slots_untied`, 1,995,008 parameters): .48, .24, .17
  against .23, .16, .06; longer clauses .00 in both. The tied read carries a role across
  positions where a block holds a fixed code (`tied_fixed`: the patient of a longer clause said
  at .88 and .91) and not where it holds settled activity (.27 and .21);
- the fixed code (`both_256`, 595,536 parameters) .99, .98, .98.

**Founders** 10728600 to 10728611 (twelve, no life in any lane). Every life: the say-back mix of
step 1 (4,000 batches), the recall slots of the heard sentence emptied at `GO` in every lesson
and reading, words said by their parts.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `settled` | 256 association neurons, `message=256`, `message_settled=True`, `message_roles` (agent, patient, distractor, verb, `GO`, the brain's own words, the rest), `message_read=True`, `message_amplitude=0.0`, `history_decays=(0.0,)`, every row taught, echo lessons | 302,560 |
| `carried` | 256 association neurons, `working_memory_decay=0.8`, `history_decays=(0.0,)`, every row taught, echo lessons | 152,804 |
| `both_256` | the fixed-code reference: `message=280` with the read of one block | 595,536 |
| `default_64` | the 0.97.0 default with the recall read barred | 30,692 |

**Controls.** The chamber's LSTM and transformer taught the same lessons, at `settled`'s count
(304,658 and 306,290 parameters) and at `carried`'s (153,138 and 154,770), online and offline,
on all twelve founders, saying the same panels and probed the same way.

**Readings.** As step 1: the chamber's panels heard as concepts and the say-back panels heard as
words; probes of every clause component after the last heard moment, at `GO` and at the last
word, fitted on trained combinations and scored on held-out ones; each word kind of the long
panels said with the previous words given; each arm less the matched controls offline, paired by
founder; the store's width and fill; parameters, seconds and multiply-accumulates per life.

**Gates** (on twelve founders):
- **H1, held:** after the last heard word of a clause heard as words, the least clause component's
  probe at .95 on 10 of 12; from the settled association, and for `settled` from its store read
  with the association.
- **H2, said:** from that state, `clause` and `clause_held` at .90 and `clause_long` at .80 on 10 of
  12.
- **P, parity:** on every panel of the chamber, the arm's paired mean against the matched
  transformer offline and the matched LSTM offline at or above -.03 and not under it by two
  standard errors.

H3 (one meaning in two wordings) and H4 (a question answered from the state) do not fit this run
and are the open remainder.

**Expectations, stated before the run.** H1 fails on every arm but `both_256`'s store: the
`settled` store with the association holds the least component at .85 to .92, the carried state
at .45 to .55. H2 fails on `settled` (clauses .60 to .70, held-out .45 to .55) and `carried`
(.55 to .65, .40 to .50), and longer clauses read .00 on every arm and control. P fails for both
candidates on clauses, held-out combinations and role twins; on paths the controls read .9 and the
arms under .6. `both_256` reads .99, .98, .98 and fails P on brackets and paths.
