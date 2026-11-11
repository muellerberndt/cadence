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
