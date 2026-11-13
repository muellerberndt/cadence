# Several things, how many, is there: declared protocol 1 (cadence#328)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

The parrot of the words chamber holds a map of its room (`cadence.space.map.Map` with each
thing's colour on the map's own sheet) and learns words from naming. Its rooms hold two or three
things of one kind in different colours. In rooms it has never seen, does it answer:

- "where is the red ball?" with two or three balls in the room (things of one kind told apart by
  their colours);
- "where is the ball?", "what colour is the ball?", "what is red?", "what is behind you?" by
  naming every thing the answer holds;
- "how many balls are there?", "how many red things are there?", "how many red balls are there?"
  with a number word;
- "is there a ball?", "is there a red ball?" with yes or no;
- "I do not know" for a word never taught,

where the number words and yes and no are nonce words heard as the teacher's answers alone (no
naming shows them), and for kinds that are named and never asked about in practice?

## The brain

`cadence.words.several.Several` with its defaults (`src/cadence/words/several.py`; nothing else
of the library changes):

- **The map** as in protocol 1 of #326: a plane of 4 m by 4 m, 32 by 32 cells, one unit a kind
  (12 kinds) and 6 colour units (`Map(values={"colour": 6})`), the identity rules unchanged.
- **The lexicon**: 256 word neurons and 36 concept neurons (12 kinds, 6 colours, 4 places around
  the parrot, 5 reads, the read that finds nothing, 6 counts, 2 truths), reciprocal synapses from
  zero, as in protocol 1 of #326. Learned parameters: 9,216 (one efficacy a reciprocal pair).
- **The number sense** is born, as the map is: one neuron a count of the places a settle of the
  map finds, from none to five or more. No lesson changes it; the words for the counts are
  learned.
- **What a question names**: each of its words heard alone names its most active concept of a
  thing (a kind, a colour, a place around the parrot) above .05; the words heard together give
  the read (the kind, the colour or the place of each thing found, their count, whether there is
  one).
- **The things it names**: the map settles on one named concept, and a place is kept where the
  map's settle at it gives every other named concept.
- **Practice**: the question's words are taught each read whose concepts the motor would say as
  the teacher's words. An answer of one word no naming has shown is named with what was found
  (the count, whether there is one, the read that finds nothing for a read of a thing that found
  none), at the gain of the most active read of the question among the reads whose concepts hold
  no other firm word, or alike over them where none is heard.

## The chamber

`benchmarks/words/several.py`, on the world of `benchmarks/words/world.py` (the parrot at the
room's middle, its camera 60 degrees either side of where it faces). Every word is a nonce word
drawn for the founder: the chamber's words, and the six question words, four number words (none
to three) and the answers no and yes of this rung. One life (`several.script`):

1. **Naming**: 40 rooms of 4 to 6 things of the 8 trained kinds, none to two kinds held two or
   three times in distinct colours, no colour on more than three things, things 0.6 to 1.8 m from
   the parrot and 0.45 m apart or more. The parrot looks around (8 moments); 15 namings a room
   (its kind's word at .5, its colour's at .25, its place's around the parrot at .25).
2. **Practice**: 40 rooms of 3 to 5 things of kinds 0 to 3, one or two kinds held two or three
   times; 14 questions a room with the teacher's answer, drawn from the forms below. The
   teacher's order: the first 12 rooms ask every form but "is there?", the other 28 every form.
3. **The test**: 30 fresh rooms of 4 to 6 things of the 8 trained kinds, one or two kinds held
   two or three times; every form of every thing; nothing taught.

| Form | Question (English for the reader) | Answer |
| --- | --- | --- |
| told apart | where is the red ball? (each thing of a kind held several times) | its place around the parrot |
| where, colour | where is the ball? what colour is the ball? | every ball's place, or colour |
| by colour, by place | what is red? what is behind you? | every thing's kind |
| how many | how many balls / red things / red balls (and a colour the kind has not) are there? | the number word |
| is there | is there a ball (and a kind not held)? is there a red ball (and a colour the kind has not)? | no or yes |
| never taught | where is / how many / is there (a foil)? | i do not know |
| never seen | where is a trained kind not in the room? | i do not know |

An answer is right where the words said are the answer's words, each as many times.

## Founders and arms

Founders 32800000 to 32800011 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| the brain | `Several`, the defaults above | 9,216 |
| the brain, no order | the same, every form asked from the first practice room (a reading, no gate) | 9,216 |
| LSTM, transformer, online | one pass over the script in the brain's order, one Adam step per naming (its two sequences) or practice question (`controls.py`'s training) | nearest width to 9,216 |
| LSTM, transformer, offline | shuffled epochs with early stopping on a tenth held aside | nearest width to 9,216 |
| LSTM, transformer, offline, large | the same at nine times the brain's parameters (a reading, no gate) | nearest width to 82,944 |

The controls are given the same script as tokens, as in protocol 1 of #326: the room's things as
concept tokens (kind, colour, place around the parrot), each naming as two sequences, each
practice and test question with its answer.

## Readings

For every arm and founder: S1 practised (kinds 0 to 3) and named-only (kinds 4 to 7); S2; S3 by
kind practised and named-only, and by colour; S4 practised and named-only; S5 never taught,
never seen, and out of view (every question whose answer holds a thing out of view) against the
map's own rates (each thing held within 0.15 m with its colour; each kind's places as many as its
things); `one` (the forms of #326 that name one thing, a reading); S6's ablation (the brain's map
and body with its lexicon's synapses at zero, answering every S1, S3 and S4 question);
parameters, multiply-accumulates (the brain: every lexicon settle at its sweeps times its
synapses, every map moment at one sweep and every map question at two over the map's synapses;
the controls: analytic, a backward pass at twice the forward) and seconds per life.

## Gates

A gate holds where the mean over the twelve founders reaches its threshold and at least ten of
twelve founders do.

- **S1, told apart by colour**: S1 practised at .90 and S1 named-only at .90.
- **S2, answers that name several things**: S2 at .90.
- **S3, how many**: S3 practised at .90, S3 named-only at .90 and S3 by colour at .90.
- **S4, is there**: S4 practised at .90 and S4 named-only at .90.
- **S5, from the state**: never taught at .95 and never seen at .95; out of view within .02 of
  the map's own rate on every founder.
- **S6, said by the brain**: every answer is the lexicon motor's letters (`Several.ask`, read by
  `tests/test_words_several.py`), and with the lexicon's synapses at zero S1, S3 and S4 read .05
  at most on every founder.
- **S7, beside the controls**: the table above for every arm, with parameters and
  multiply-accumulates.

**Expectations, stated before the run.** Development founders 2800 to 2811 (no gated life): the
brain reads 1.000 on every reading and .000 with its lexicon at zero. With no order (founders
2800 to 2807), four of eight founders read S3 and S4 at .50 to .66 (the words for none and no
take each other's concept), and one reads never taught and never seen at .00. Development
founders 2800 and 2801: the six control arms read S1 at .00 to .33, S2 at .01 at most, S3 at .23
to .51 and S4 at .15 to .59 (a coin reads .50 on S4).

**What a failure names.** An S1 or S4 miss on a thing the map holds names what a question names
(a word naming a second concept) or the map's settle at a place; an S3 or S4 miss on things not
held names the words for none and no; an out-of-view shortfall names the map's carry.

## The map's ground

No change to the map. Two or three things of one kind told apart by `values=` were measured
first: seen together, two balls 0.45 m apart or more are two things, each with its colour; a ball
first seen while another ball within `stride` (0.5 m) is held out of view is taken as that ball
moved unseen. The chamber's look-around (8 moments 45 degrees apart, a field of 60 degrees either
side, things 0.45 m apart or more) sees every pair within 0.5 m together before either enters
alone: 40 rooms with two balls and 40 with three held each ball with its colour at 1.000.

## Compute

One AWS c7i.4xlarge spot box (Name=cadence-0100-words-several), one process a life, one BLAS
thread a process, results synced to s3://cadence-0100-words-several-20261113.
