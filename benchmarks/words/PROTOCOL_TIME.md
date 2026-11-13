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

# Words in time: declared protocol T2 (cadence#326, W8 to W10)

Declared on 2026-11-13, committed to the lane branch `lane-0100-words-time` before the first
declared life of T2. T1's readings stand as read; T2 reads W8 again with things that roll, and
reads W9 and W10. The founders, arms, readings and gates below are fixed.

## The question

The parrot watches things roll and learns how things move (the map's learned physics, #324).
Then the teacher carries things, and rolls a thing while the parrot watches and turns its head,
and asks "where was the X?", "where is the X?" and "where will the X be?". Does the parrot answer
the future from its own forecast, within the forecast's own error, leaving its map as it was?
Does it tell "was", "is" and "will" apart, words it learned from answered questions alone?

## The brain, and what changed after T1

`Times` with its defaults as in T1, with these differences, each found in development readings
after T1 (founders 900 to 905) and stated here before T2's first life:

- **The map has learned physics**: `Map(physics=True)`, the physics' own defaults (camera error
  .01 m, a line of up to 30 reports, lessons every 8 reports over a horizon of 30, a prior of 500
  moments of rest). The code of the physics is lane-0100-physics at d8e96b59 merged with the
  values hook, and one fix: two things first seen in one moment take two identities in the
  motion records (with one identity, a thing that rolled out of view beside one in view was never
  carried on; reported to the physics lane, regression test in `tests/test_words_episodes.py`).
- **The forecast** is `Map.forecast(10)`, read on a copy (`Times(ahead=10)`).
- **The records are read by the dot product** (T1: the cosine). With the cosine, a record whose
  code held the kind alone (a thing whose colour the map did not hold at that moment) outscored
  the latest record that held kind and colour; a development founder read the past at .725 with
  the cosine and 1.000 with the dot product.

Learned parameters: 7,424 (the lexicon), and the physics' four numbers (rest, leave, ahead,
aside), learned from its lessons.

## The chamber

`benchmarks/words/times.py` with `PROTOCOLS["T2"]`:

1. **Naming** as in T1 (40 rooms, 15 namings).
2. **Watching**: 300 rooms of 4 to 6 things. The parrot faces a drawn thing (within .5 radians);
   up to two things in view roll at .03 to .06 m a moment for 40 moments while it watches. No
   word is said. The physics learns from these rolls.
3. **Practice**: 20 carried rooms as in T1 (was and is of each carried thing; where and colour
   of a thing that stayed), and 40 rolled rooms. In a rolled room one thing rolls for 16 to 28
   moments while the parrot watches (it sees the thing for at least the last 6 of them), then
   the parrot turns its head to a drawn heading for one more moment. Where the thing began,
   where it is then and where it will be 10 moments later are three different places around the
   parrot as it faces, each at least 10 degrees inside its place. The teacher asks where it
   was, is and will be, where a thing that stayed is and its colour, and says each answer.
4. **The test**, nothing taught: 15 carried rooms as in T1 (was and is; the past again with the
   records barred; where and colour of a thing that stayed) and 50 rolled rooms (was, is and will
   of the thing that rolled; the past again with the records barred; where a thing that stayed is
   and where it will be).

The truth of "will" is where the thing is 10 moments later in the world: it keeps its line.

## Founders and arms

Founders 32610100 to 32610111 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| the brain | `Times`, as above | 7,424 and the physics' 4 |
| LSTM, transformer, online and offline | as in T1 | 7,213 and 6,953 (the widths nearest 7,424 with this chamber's tokens) |

The controls are given T1's tokens, and each roll as a track: `TRACK`, the thing's kind and its
place at five moments evenly over the roll, each place a token of 48 (16 sectors around the
parrot as it faces, four to each place around it, by 3 rings of distance). Each watched room
gives each of its tracks as a lesson (the first two places, then the three after); a question
about a rolled room is prompted with the room as the parrot faces after its turn and the track.

## Readings

As in T1, pooled over carried and rolled things where both are asked, and:

- `W9_future`: where the rolled thing will be, right;
- `W9_forecast_rate` (the brain): the forecast's own answer is right: every place the forecast
  holds of the thing lies in the right place around the parrot (and it holds one);
- `W9_error` (the brain): the distance in metres from the forecast's nearest place of the thing
  to where it is 10 moments later;
- `W9_now_rate`: the share where the place now is the place it will be (answering "is" to a
  question of "will" reads this; 0 by the chamber's design);
- `W9_unforecast` (the brain): the future questions asked again with the forecast looking 0
  moments ahead (the forecast barred: it is the map now), right;
- `W9_map_kept` (the brain): the share of the test's questions after which the map's arrays are
  the same, bit for bit, as before the question;
- `W10_three_was`, `W10_three_is`, `W10_three_will`, `W10_three` (the brain): on the questions
  where the records, the map and the forecast hold the thing in three different places around
  the parrot, the share whose answer is the asked time's source's;
- `still_will`: "where will the X be?" of a thing that stayed, right.

## Gates

A gate over founders holds where the mean over the twelve reaches its threshold and at least ten
of twelve do.

- **W8, the past from memory** (again, with rolled things): `W8_past` at .90; `W8_present`
  within .02 of `W8_map_rate` on every founder; `W8_barred` at most .25 on every founder.
- **W9, the future from the forecast**: `W9_future` within .02 of `W9_forecast_rate` on every
  founder, and `W9_map_kept` 1.000 on every founder.
- **W10, a sense of time learned**: `W10_three` at .95.
- The table of every arm with parameters and multiply-accumulates.

**Expectations, stated before the run.** From development founders 900 to 905 (no gated life,
the chamber and brain above): the brain reads 1.000 on `W8_past`, `W8_present` (the map's rate
1.000), `W9_future` (the forecast's own rate 1.000, its error 1.9 cm), `W9_map_kept`, `W10_three`
and the questions about things that stayed; with the records barred the past reads 0.000, every
past question refused. The matched controls read .24 to .29 on the past, the present and the
future (chance .25). `W9_unforecast` was added after those readings: with the forecast barred a
question of "will" finds the thing where it is now, and `W9_now_rate` is 0 by design, so it is
expected near 0.

**What a failure names.** A future miss the forecast holds right names the gains (the words'
lesson of "will"); a gap between the forecast's rate and the truth names the physics (#324); a
map not kept names `Map.forecast`; a past miss on a rolled thing names the writing (the rest
and leave rules on a moving thing's places).

## Compute

The box of T1, results synced to s3://cadence-0100-words-time-20261113 (`time326/gateT2`).
