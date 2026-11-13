# Words in time: declared protocol T1 (cadence#326, W8)

Declared on 2026-11-13, committed to the lane branch `lane-0100-words-time` before the first
declared life. The founders, arms, readings and gates below are fixed; a failed gate is reported
as failed.

## The question

A parrot holds a map of its room and learns words from naming (protocol 1). The teacher carries
things to new places while the parrot looks on, and asks "where was the X?" and "where is the X?".
The map holds a thing's latest place alone. Does the parrot answer where a thing was from its own
records, written while it lived, where it is from its map, and does it learn which words ask for
which, from answered questions alone? With its records barred, is the past answered at chance or
refused?

## The brain

`cadence.words.episodes.Times` with its defaults, on the map of protocol 1 (`Map(values={"colour":
6})`, 32 by 32 cells over 4 m by 4 m, one unit a kind, no physics):

- **The records**: a `cadence.longterm.LongTermMemory` of 256 slots, read by the cosine with the
  first choice whole (`temperature=0`, `cosine=True`), every write its own record (`revise=0`).
  A key is the thing's code on the map's sheet (its kind's unit and its colour's, 18 units) at
  length 2 beside a drifting context of 64 units at unit length (`Clock`, drift .99 a moment). A
  value is the place the thing left and its code. Each moment the brain compares where its map
  holds each thing with where it rested: a thing that left its resting place by more than .3 m is
  written as a record of that place, and rests again where it moves less than .02 m in a moment.
  A recall is cued by the thing's code and the context now; the record found is the thing's where
  its code holds every unit of the cue's.
- **The lexicon** of protocol 1 with three more concepts, one a source (`when`: the records, the
  map, the forecast): 256 word neurons and 29 concept neurons, reciprocal synapses from zero.
  Learned parameters: 7,424 (one efficacy a reciprocal pair; the map's graph has no taught synapse
  and the records are memory).
- **An answer**: each source finds the thing a question names and reads it as protocol 1 reads
  the map (the forecast of a map with no physics is the map itself). The settled `when`
  activities, over the most active one, are gains: a thing's drive on the concepts is each
  source's concept at its gain, and the lexicon's motor says the most active word of that drive,
  or nothing below its threshold; where no word is said, the parrot says the word for the read
  that finds nothing. A question that names no source above .05 asks the map alone.
- **Practice**: protocol 1's lesson of the read, and the lesson of time: where the teacher's
  answer is one word naming a concept and the sources disagree on the thing asked about, the
  question's words are taught the sources that gave the answer, at equal shares.

## The chamber

`benchmarks/words/times.py` with its defaults. Every word is a nonce word drawn for the founder;
the words for "was" and "will" are drawn apart from the founder's other words, and "is" is the
founder's word of protocol 1. One life:

1. **Naming**: 40 rooms of 4 to 6 things of the 8 trained kinds, 15 namings a room, as in
   protocol 1.
2. **Practice**: 40 rooms of 4 to 6 trained things. The parrot looks around (9 moments); the
   teacher carries 2 things, each to a place at least .8 m from where it was and in another of
   the four places around the parrot; the parrot looks around twice. The teacher asks, in a drawn
   order, where each carried thing was and where it is (`where <was|is> the <kind>`), where one
   thing that stayed is and its colour (protocol 1's forms), and says each answer.
3. **The test**, nothing taught: 30 fresh rooms carried as in practice; where each carried thing
   was and where it is; each past question again with the records barred; where one thing that
   stayed is and its colour.

An answer is right where the words said are the answer's words as a set.

## Founders and arms

Founders 32610000 to 32610011 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| the brain | `Times`, the defaults above | 7,424 |
| LSTM, transformer, online | one pass over the script in the brain's order, one Adam step per naming (its two sequences) or practice question (`times.control`, the training of `controls.py`) | 7,288 and 6,824 (the widths nearest 7,424) |
| LSTM, transformer, offline | shuffled epochs with early stopping on a tenth held aside | as above |

The controls are given the same script as tokens: each naming as in protocol 1 (the room as the
parrot faces, then the two sequences); each question about a carried room prompted with the room
as first seen (each thing's kind, colour and place around the parrot), a `MOVE` marker and the
room as seen after the carries, then the question's words; the answer's words after `SAY`.

## Readings

For every arm and founder:

- `W8_past`: where each carried thing was, right; `W8_present`: where it is, right;
- `W8_map_rate` (the brain): the share of carried things whose place the map holds, its own
  answer: the map holds the kind within .15 m of its place and nowhere farther than .3 m from it;
- `W8_barred` (the brain): the past questions asked again with the records barred, right;
  `W8_barred_refused`: the share answered with the word for the read that finds nothing, or
  nothing;
- `W10_was`, `W10_is`, `W10` (the brain): on the questions where the records and the map hold the
  thing in two different places around the parrot, the share whose answer is the asked time's
  source's (the records for "was", the map for "is");
- `still_where`, `still_colour`: protocol 1's questions about a thing that stayed;
- parameters, multiply-accumulates (the brain: every lexicon settle at its sweeps times its
  synapses, every map moment at one sweep and every map question at two over the map's synapses,
  every records read at its records times the key's width, a clock moment at its units; the
  controls: analytic, a backward pass at twice the forward) and seconds per life.

## Gates

A gate over founders holds where the mean over the twelve reaches its threshold and at least ten
of twelve do.

- **W8, the past from memory**:
  - `W8_past` at .90;
  - `W8_present` within .02 of `W8_map_rate` on every founder;
  - `W8_barred` at most .25 on every founder (chance among the four places).
- **Toward W10** (a reading of this protocol; W10's gate reads "will" as well, in the protocol
  with learned physics): `W10` at .95.
- The table of every arm with parameters and multiply-accumulates (W7's form).

**Expectations, stated before the run.** From development founders 900 to 905 (no gated life):
the brain reads 1.000 on `W8_past`, `W8_present` (the map's rate 1.000), `W10`, `still_where` and
`still_colour`; with its records barred 0.000, every past question refused. The matched controls
read `W8_past` at .25 to .32 and `W8_present` at .23 to .29 (chance .25).

**What a failure names.** A past miss with the records holding the thing names the gains (the
time words' lesson); a past miss with no record names the writing (the leave rule on the map's
places); a present gap from the map's rate names the gains' leak from the records; a barred reading
above chance names a source other than the records that holds the past.

## Compute

One AWS c7a.4xlarge spot box (Name=cadence-0100-words-time), one process a life, one BLAS thread
a process, results synced to s3://cadence-0100-words-time-20261113.
