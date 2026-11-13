# Words for things with concepts the parrot learns: declared protocol C1 (cadence#326)

Declared on 2026-11-13, committed to the branch `lane-0100-words-concepts` before the first declared
life. The founders, arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

Protocol 1's parrot answers about its room with words bound to concepts it was given: twelve kind
units, six colour units, the four quarters around its body. Here it is given none. Its eyes report
each thing as raw readings (three shape and size features, a hue, a place), and its body knows the
direction of a place from where it faces. Do the concepts its words name form from those readings
and from naming, so that:

- the concepts match the names its teacher uses (C1);
- readings never shown are named by the nearest learned concept (C2);
- where each founder's teacher draws its own boundaries (red ends at a different hue, behind
  begins at a different angle), the parrot's boundaries follow its own teacher's (C3);
- and every question of protocol 1 is answered as well as with the given concepts (W1 to W7)?

## The brain

`cadence.words.concepts.LearnedWords` on `cadence.words.concepts.Concepts` with their defaults
(guide: the module's docstring, `src/cadence/words/concepts.py`):

- **Born senses**: the hue and the direction around the body, each over 48 tuned inputs around the
  circle (`Circle`); the three shape features over 8 tuned inputs each, an input a combination
  (`Features`, 512 inputs).
- **Regions of concept neurons**: kinds (24 neurons, the map's kind units), colours (12, the map's
  colour values), places around the body (10). A region's neurons read its sense through one-way
  synapses and settle with the library's graph; the most active is the concept. A kind enters
  where no kind concept settles at .5 of its level (new from the senses); a colour or a place
  concept enters by naming.
- **A word's image of each sense**: the mean of what the sense gave at its namings (word to sense
  synapses). The sense where the image stands apart from what the senses give over every thing
  held (by 1.5 times the next, at .1 at least, from 5 readings) is the word's sense.
- **A naming's lesson**: in the word's sense, the word's own concept (the one it names most that
  no other word of that sense names more) is pulled to its level on the reading and every other
  concept of the region to rest, by the library's free and nudged phases (a lesson closes .3 of
  its error); a word with no concept of its own there takes a new one at the reading. A word heard
  three times or fewer on a thing whose kind, colour and place each carry another word's
  name takes a new kind concept (mutual exclusivity and the whole thing).
- **The lexicon** (protocol 1's `Lexicon`, rate .2): a word whose image names a sense is taught that
  sense's concept alone; a concept another word of its sense names more is left at rest.
- **The map** (`Map(units=24, values={"colour": 12})`) holds each thing's kind and colour concepts.
  "What is behind you" reads the cells whose directions settle on the place concept.

Learned parameters: 56,864 (the lexicon's 12,800 reciprocal pairs, the regions' 13,344 synapses,
the images' 30,720 synapses).

## The chamber

`benchmarks/words/concepts.py`. Protocol 1's script (`words.py`), with these differences:

- **Raw readings.** Each thing has a hue drawn uniformly in its colour's band and three shape
  features drawn around its kind's shape (spread .04 a feature).
- **The founder's teacher.** Six colour bands whose boundaries lie 60 degrees apart, each moved by
  a draw within 15 degrees; twelve kind shapes drawn at least .3 apart in the feature cube; the
  four places' turns each moved by a draw within 15 degrees.
- **Held out of every room until the held-out test rooms.** A 6-degree gap inside each colour's
  band (at least 6 degrees from its edges), and for each kind one side of one feature.
- **Naming**: 160 rooms of 15 namings (protocol 1: 40). The teacher names a kind, a colour or a place
  around the body at a third each. For a kind or a colour, the parrot turns its head to the thing;
  its body does not turn, so its front and behind stay where they were.
- **New words**: the parrot turns to each new thing before its three namings.
- **Test**: protocol 1's 30 test rooms and 16 rooms with a thing in its kind's unseen colour, and
  16 held-out rooms whose things have held-out hues and held-out shapes (C2_room: their colour and
  where they are).
- **Probes** after the life, nothing taught:
  - C1: the parrot's word for the hue at every degree outside the gaps, for 20 fresh things of
    each trained kind, and for the direction at every degree, against the teacher's;
  - C2: 10 held-out hues a colour and 10 held-out shapes a trained kind;
  - C3: the hue and the direction where the parrot's word changes, read half a degree at a time
    within 30 degrees of each of the teacher's boundaries.

## Founders and arms

Founders 32650000 to 32650011 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| learned | `LearnedWords` on `Concepts`, the defaults above | 56,864 |
| given | protocol 1's brain on the given concepts: the thing's kind (the teacher's own shape nearest its features), its hue's 60-degree band at the unmoved boundaries, the unmoved quarters | 6,656 |
| senses | the same regions taught by the senses alone: every region takes concepts from its readings (vigilance .5 kinds, .15 colours and places), no naming lesson | 56,864 |
| LSTM, transformer, online and offline | `concepts_controls.py`: each thing of the room as one position with its seven raw readings, the words as tokens, protocol 1's training (`controls.py`) | at 56,864 (nearest width) |

## Readings and gates

A gate on a reading holds where the mean over the twelve founders reaches its threshold and at
least ten of twelve founders do.

- **C1, the concepts match the naming**: C1_colour at .90, C1_kind at .95, C1_place at .90.
- **C2, the nearest concept for readings never shown**: C2_hue at .90, C2_shape at .90, C2_room at
  .90.
- **C3, each founder's own teacher's boundaries**: over the 72 colour boundaries of the twelve
  founders, the slope of where the parrot's word changes on where the teacher's does (both
  measured from the unmoved place) at .8 or more and the mean distance between the two at 4 degrees
  or less, and on the boundaries the teacher moved by 5 degrees or more, the parrot's lies nearer
  its own teacher's than the unmoved place on .8 of them or more; over the 48 turns around the body,
  the slope at .8 or more and the mean distance at 4 degrees or less.
- **W1 to W7, protocol 1's gates on this chamber**: W1 at .95 and its named-only part at .95; W2 at
  .90; W3 at .90; W4 out of view within .02 of the map's own rate on every founder, never seen at
  .95, never taught at .95; W5 by colour at .90 and by place at .90; W6, with the lexicon's
  synapses at zero W1 at .05 at most on every founder; W7, the table of every arm with parameters
  and multiply-accumulates.

`python concepts.py run <out> 32650000 12` runs every arm; `python concepts.py read <out>` gives the
table and each verdict.

## Expectations, stated before the run

From development founders 900 to 911 (no gated life), at the defaults:

- **The learned brain** holds every gate on 12 of 12 founders:

  | Reading | Mean | Lowest founder |
  | --- | --- | --- |
  | W1 | .980 | .968 |
  | W1, named only | .983 | |
  | W2 | .972 | |
  | W3 | .969 | .91 |
  | W5 by colour | .967 | .90 |
  | W5 by place | .960 | |
  | C1 colour | .971 | |
  | C1 kind | 1.000 | |
  | C1 place | .985 | |
  | C2 | .995 to 1.000 | |

  Each founder forms 6 colour concepts and 4 place concepts, and 12 to 15 kind concepts for its 12
  kinds. C3: the colour boundaries lie 1.6 degrees from the teacher's on the mean (slope .87, the
  largest 4.8 degrees), the turns 1.2 degrees (slope 1.03), and 43 of 43 moved boundaries lie
  nearer the founder's own teacher.
- **Given concepts** (founders 900 to 903) read W1 .89, C1 colour .71, W5 by colour .79, and C3 a
  slope of -.29: their boundaries stay where they were drawn.
- **The senses alone** (founders 900 to 903) read W1 .21 and C1 colour .21.
- **The controls**: the LSTM online (900 and 901, 57,299 parameters) reads W1 .00 and .07. At 60
  naming rooms (founder 900), the LSTM offline reads .08 and the transformer online .00.

With fewer namings the boundaries are less exact. At 60 naming rooms W1 reads .95 to .99, but W3
and W5 by colour reach .90 on 8 and 9 of 12 founders; at 120 rooms, on 11 and 10. Every miss there
is a thing within a few degrees of a boundary.

## What a failure names

- A C1 or C3 miss names the region's lessons: too few namings near a boundary, or a concept two
  words share.
- A C2 miss names a region's reach beyond the readings it was taught.
- A W3 or W5 by-colour miss on a thing within a few degrees of a teacher's boundary names the
  boundary's precision. These answers name every thing of a colour, so one thing misnamed is a
  miss.
- A W2 miss names a new kind that shares a kind concept with another.

## Compute

One AWS c7i.4xlarge spot box (Name=cadence-0100-words-concepts), one process a life, one BLAS
thread a process, results synced to s3://cadence-0100-words-concepts-20261113.
