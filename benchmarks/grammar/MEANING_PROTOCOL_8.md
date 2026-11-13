# The form a word does not have, left open: declared protocol 8 (cadence#302)

A draft, not declared: the founders, arms, readings and gates below are fixed by the commit that
declares it on cadence#302, before the first declared life.

## The question

[Protocol 7](MEANING_PROTOCOL_7.md) said each word by its parts, a stem and a form, and held its
gates for the form and the sentence. Its third gate, no cost on any other panel, failed on paths:
from the held state `parts_256` read paths .07 under `words_256`, and with the read free
`parts_free` read three hops .13 under `words_free`. Its reading named the read's delivery. A
development run (the default composition with the read free; founders 0 to 5, 100 to 123 and
protocol 7's twelve, 42 lives an arm) locates the term elsewhere.

- **The address lesson names the same slots with either motor.** On the hops of a path its
  target puts .95 to .97 on the slots that hold the taught word with parts and .97 to .98 with
  words, whichever part the brain had wrong.
- **Three hops are lost where the sentence goes on.** After the second hop a three-hop sentence
  says `near` at a place where every lesson says `end`. Given the previous words the next port
  holds `near` at the same share with either motor (.51 with parts, .50 with words). On six of
  protocol 7's founders, four of them its lowest on three hops, the parts brain says `near` there
  on .53 of the sentences against .83; over all 42 it says it on .83 against .86.
- **Parts lives end with louder carried routes.** The learned gain of the working trace ends at
  .18 against .10 and the heard history's at 2.18 against 1.55. On six lives, halving the trace's
  gain at the reading lifts three hops from .32 to .43 with parts (.63 to .67 with words) and
  doubling it reads .17 (.39).
- **The third form class raises the gains.** Protocol 7 teaches the form `none` on every word
  without a number (`near`, the brackets, `end`). The form slot then has to tell at every word
  whether the word has a number, and it reads that from the state the brain carries. With that
  form left untaught the gains end at .11 and 1.16.

Over the 42 development founders protocol 7's parts read three hops .054 (.045) and brackets
.029 (.027) under words; with the form left open they read +.007 (.043) and +.050 (.026).

Does a motor of parts whose lessons leave the form of a word without one open say paths, three
hops and every other panel as a motor of whole words does, and keep protocol 7's form and
sentence?

## The arm

The motor is protocol 7's (`slots=(12, 3)`). A word with a number teaches its stem and its form.
A word without one teaches its stem and leaves the form slot open: its label there is `-1`
(`cadence.learning.OPEN`, on this lane's branch), which a lesson neither judges nor pulls. In the
nudged phases the open slot's target is what the slot says in the lesson's own free phase, so the
pull on it is zero there. No lesson names the third form neuron. The instrument is
`"parts": "open"` in `benchmarks/grammar/meaning.py`; a lesson that labels every slot keeps its
bits.

Two development arms are not run here, each a located negative:

- **Each motor slot's own movement scoring the records** (two more nudged phases a slot, the
  scores of the slots added): paths .35 against .90 on six founders. A verb's stem and its form
  are read from two different records, and a record that every slot names does not exist.
- **A form slot of two neurons** (no third class at all): held-out combinations .019 (.008) and
  longer clauses .012 (.005) under the slot of three, 42 founders.

## Founders and arms

Founders 11028600 to 11028695 (96, no life in any lane). 4,000 batches of the first grammar.

| Arm | The brain | Parameters | Founders |
| --- | --- | --- | --- |
| `words_free` | the default composition, the read free, 64 association neurons: the reference | 30,992 | 96 |
| `open_free` | `words_free` saying each word by its parts, the form of a formless word left open | 30,692 | 96 |
| `parts_free` | protocol 7's `parts_free`: the form `none` taught | 30,692 | the first 24 |
| `words_256` | `message=280`, `message_read=True`, `history_decays=(0.0,)`, 256 association neurons, every row taught, the recall read barred: the reference | 596,412 | the first 24 |
| `open_256` | `words_256` by parts, the form of a formless word left open | 595,536 | the first 24 |
| `parts_256` | protocol 7's `parts_256` | 595,536 | the first 12 |

The founders differ by arm for one reason, stated here before any life: three hops read between
.00 and .92 from founder to founder, so the paired standard error of that panel is about .08 on
twelve founders, .06 on 24 and .03 on 96, and the gate's bound is .03. The arms with the read
free cost about two minutes a life and are read on all 96; the held arms cost half an hour a life
and are read on 24.

**Controls.** The chamber's LSTM (596,598 parameters) and transformer (592,058) at `open_256`'s
count, online and offline, on the first six founders.

## Readings

The panels `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `path_long` (three hops)
and `clause_long`; each arm less its reference, paired by founder, with its standard error;
multiply-accumulates and seconds per life. On held-out clauses (4 batches of 16 per trained
shape), every word kind said with the previous words given, and probes of every clause component
(the verb's form included) from the message, the carried trace and the settled association,
fitted on trained combinations and scored on held-out ones. The panel's agreement and the share
of wrong forms a distractor's number drew. The learned gains of the carried routes at the life's
end, by arm. With the read free, where the read lands on a three-hop path (`located`).

## Gates

- **P1, the form**: `open_256` says the verb's form on held-out clauses at .95 with the previous
  words given on 20 of 24 founders, and its mean is within .03 of the mean of the probe's reading
  at that settle (the lesser of the stem's and the agent's number's).
- **P2, the sentence**: `open_256` reads `clause_held` at .90 on 20 of 24, said from the held
  state.
- **P3, no cost**: every panel other than `clause_held`, paired by founder, no more than .03 under
  the reference and not under it by two standard errors: `open_256` against `words_256` on 24
  founders, and with the read free `open_free` against `words_free` on 96.

`parts_256` and `parts_free` are read against the same references and against the open form,
unjudged: protocol 7's cost on fresh founders, and what leaving the form open changes. The scaled
lexicon is the next protocol's.

**Expectations, stated before the run.** P1 holds on 23 or 24 of 24 and P2 on 22 to 24. P3 holds
with the read free: three hops between -.03 and +.05, brackets between .00 and +.08, paths, roles
and clauses within .02. P3 from the held state is the open reading: no held life of the open form
was read when this was declared beyond six development founders, on which
the open form read paths .19 (.03) under the motor of whole words. `parts_free` reads three hops .03 to .10 under `words_free`. The trace's gain
ends at .09 to .13 in `words_free` and `open_free` and at .16 to .20 in `parts_free`.

**What a failure names.** Where P3 fails on a panel with the open arm's gains at the reference's,
the cost is not the carried routes' loudness and the motor's two slots cost that panel by
another term, to be located on that panel's words (the decision after a hop for paths, the
matching close for brackets). Where the gains differ, a lesson other than the form `none` raises
them.

## Compute

One AWS c7i.4xlarge (Name=cadence-0100-paths), one process per life, one BLAS thread per process:
the 216 lives with the read free, then the 60 held lives, then the 24 controls.
