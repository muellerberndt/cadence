# A word said by its parts: declared protocol 7 (cadence#302)

Declared on 2026-11-11, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

Said from the held state (`message=280`, the read of one block, the previous word, 256
association neurons, the recall read barred), held-out combinations are lost at the verb's form.
A development run on founders 0 to 5 located the term. At the settle that says the verb of a
held-out clause, probes fitted on trained combinations read the stem from the settled association
at .97 to 1.00 and the agent's number at .95 to 1.00. The same probe reads the form (stem and
number together) at .84 to .98, and the brain says the form at the same rate founder by founder
(.86 to 1.00). The association holds the two parts each on its own, and a readout of the whole
word set up on trained combinations reads their conjunction only as well as the association
carries it for an agent, number and verb no lesson showed together. Does a motor that says a word
by its parts, each part read on its own, say the held-out form?

## The arm

The chamber's motor is two slots that settle together (`Brain.compose(..., slots=(12, 3))`, the
library's motor slots, no new setting): the stem (the four nouns, the two verbs, `near`, the four
brackets, `end`) and the form (singular, plural, none). A noun's form is its number and a verb's
the number its agent calls for. Every lesson that says a word teaches its stem and its form, and the
chamber says the word its two slots choose. The instrument is `parts_brain` in
`benchmarks/grammar/meaning.py`.

## Founders and arms

Founders 10528600 to 10528611 (twelve, no life in any lane). 4,000 batches of the first grammar on
0.97.0.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `words_256` | `message=280`, `message_read=True`, `history_decays=(0.0,)`, 256 association neurons, every row taught, the recall read barred: the reference | 596,412 |
| `parts_256` | `words_256` saying each word by its parts | 595,536 |
| `w10_256` | `words_256` with `message_winners=10`: #302's reference arm | 596,412 |
| `words_free` | the 0.97.0 default, the read free, 64 association neurons | 30,992 |
| `parts_free` | `words_free` saying each word by its parts | 30,692 |

**Controls.** The chamber's LSTM (596,598 parameters) and transformer (592,058) at `parts_256`'s
count, online and offline, on the first six founders.

## Readings

The panels `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `path_long` (three hops)
and `clause_long`; each arm less its reference, paired by founder; multiply-accumulates and
seconds per life. On held-out clauses (4 batches of 16 per trained shape), every word kind said
with the previous words given, and probes of every clause component (the verb's form included) at
`GO`, at the settle that says the verb and at the last word, from the message, the carried trace
and the settled association, fitted on trained combinations and scored on held-out ones. The
panel's agreement and the share of wrong forms a distractor's number drew.

**With the message silent** (cadence#308's reading), on the arms with a message: `clause`,
`clause_held` and `roles` said, the held-out word kinds said and the same probes, with the
message's drive and its read held at zero through the reading: what the association's own state
carries and says.

## Gates

- **P1, the form**: `parts_256` says the verb's form on held-out clauses at .95 with the previous
  words given on 10 of 12 founders, and its mean is within .03 of the mean of the probe's reading
  at that settle (the lesser of the stem's and the agent's number's).
- **P2, the sentence**: `parts_256` reads `clause_held` at .90 on 10 of 12, said from the held
  state.
- **P3, no cost**: every panel other than `clause_held`, paired by founder, no more than .03 under
  the reference and not under it by two standard errors: `parts_256` against `words_256`, and with
  the read free `parts_free` against `words_free`.

The scaled lexicon (16 nouns and 8 verbs, and 512 and 128) is left to a later protocol: the chamber
holds four nouns and two verbs.

**Expectations, stated before the run.** P1 holds: the form slot reads the agent's number, which
the association holds at .95 to 1.00 at that settle. P2 holds on 10 or 11 of 12. P3 holds from the
held state and with the read free, within .03 on every panel. `w10_256` lifts held-out
combinations by .03 to .05 against `words_256`. With the message silent every arm says .00 to .05
of clauses and held-out combinations.

**What a failure names.** Where P1 fails with the probe of the agent's number at .95 and over, the
form slot does not read the number the association holds; where the probe falls too, the
association does not hold the number at that settle on new combinations.

## Compute

One AWS c7i.16xlarge spot box (Name=cadence-098-meaning), the 60 brain lives at once and the 24
controls after them, one process per life, one BLAS thread per process.

## Result

Read on one c7i.16xlarge, founders 10528600 to 10528611, 60 brain lives and 24 control lives on
commit eb16a20f.

| Learner | Parameters | Multiply-accumulates | Seconds | clause | clause_held | roles | dyck | path | three hops | clause_long |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `words_256` (12) | 596,412 | 2.56e13 | 1,917 | .99 | .93 | .95 | .87 | .47 | .00 | .00 |
| `parts_256` (12) | 595,536 | 2.50e13 | 1,907 | .99 | .99 | .96 | .85 | .40 | .00 | .00 |
| `w10_256` (12) | 596,412 | 2.41e13 | 2,009 | .98 | .97 | .95 | .83 | .47 | .00 | .00 |
| `words_free` (12) | 30,992 | 1.03e12 | 244 | .99 | .95 | .98 | .84 | .97 | .65 | .01 |
| `parts_free` (12) | 30,692 | 9.61e11 | 235 | .98 | .96 | .96 | .83 | .94 | .52 | .00 |
| LSTM, offline / online (6) | 596,598 | 3.37e13 / 1.42e12 | 1,209 / 71 | 1.00 / .99 | .99 / .97 | .99 / .87 | .98 / .85 | .96 / .31 | .00 / .00 | .00 / .00 |
| transformer, offline / online (6) | 592,058 | 3.21e13 / 1.42e12 | 1,586 / 83 | 1.00 / .78 | 1.00 / .82 | .99 / .79 | .94 / .14 | .92 / .10 | .00 / .00 | .00 / .00 |

- **P1 holds.** `parts_256` says the verb's form on held-out clauses with the previous words given
  at .98 to 1.00 on 12 of 12 founders (mean 1.00), against the probe's .99 at that settle;
  `words_256` says it at .79 to 1.00 (mean .94).
- **P2 holds.** `parts_256` reads `clause_held` at .90 on 12 of 12 (.97 to 1.00), +.06 (.02) against
  `words_256`, beyond two standard errors; agreement on held-out clauses 1.00, no wrong form drawn by
  a distractor (`words_256` .97 and .07).
- **P3 fails on paths.** From the held state, `parts_256` less `words_256`: clause +.01 (.00),
  roles +.01 (.02), brackets -.02 (.03), paths -.07 (.04), three hops and longer clauses .00. Paths
  are .07 under at the mean (founders -.34 to +.19), inside two standard errors. With the previous
  words given a path's second hop is said at .50 and .53 against .67 and .58: a path's hops come from
  the recall read, which is barred. With the read free, `parts_free` less `words_free`: clauses -.01,
  held-out +.02, roles -.02, brackets -.00, paths -.03 (.03), three hops -.13 (.10) (founders -.70 to
  +.36). The read delivers a three-hop path's start, second and third hops at .93, .86 and .76
  against 1.00, .91 and .87 for words, and the words delivered are said at .92 to .99 in both: the
  term is the read's delivery, whose address lesson reads the motor's movement, two slots in place
  of one.
- **Ten winners** (`w10_256`) lift held-out combinations +.04 (.02) against `words_256`; agreement
  .99.
- **With the message silent** every arm says .00 of clauses, held-out combinations and role twins,
  and the verb at .09 to .12. At the settle that says the verb, the settled association reads the
  verb's form at .95 to .99 with the message and .31 to .40 without it.

**Against the expectations.** P1 and P2 held as stated, P2 on 12 of 12. P3 was expected to hold
and failed on paths from the held state (-.07) and on three hops with the read free (-.13), each
inside two standard errors and outside .03. `w10_256` lifted held-out combinations as stated. The
message-silent readings read .00 as stated.
