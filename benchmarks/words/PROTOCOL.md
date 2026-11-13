# Words for things: declared protocol 1 (cadence#326)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

A robot holds a map of its room (`cadence.space.map.Map`, each thing's colour on the map's own
sheet with `values=`), knows where things are around its body, and learns words from naming: while
it attends to a thing it hears one word, and its lexicon (`cadence.words.Lexicon`) associates the
word with the concepts active at that moment. Does it answer typed questions about rooms it has
never seen, in both directions ("where is the ball?", "what colour is the ball?", "what is red?",
"what is behind you?"), with words learned from naming alone, a word taught live in three
namings, colours a kind never had, things out of view, and "I do not know" where the answer is
not held?

## The brain

`cadence.words.Words` with its defaults: a map of the chamber's plane of 4 m by 4 m, 32 by 32
cells, one unit a kind (12 kinds) and 6 colour units (`Map(values={"colour": 6})`); the body at the
room's middle; a lexicon of 256 word neurons and 26 concept neurons (12 kinds, 6 colours, 4 places
around the body, 3 reads, the read that finds nothing), reciprocal synapses from zero, a word's
n-th lesson closing 1/n of its error and no less than .05, a word threshold of .15; naming shares
.4 (the kind), .3 (the colour), .3 (the place around the body); a concept counts as named above
.05. Learned parameters: 6,656 (one efficacy a reciprocal pair of the lexicon; the map's graph
has no taught synapse).

## The chamber

`benchmarks/words` (`world.py`, `words.py`). Every word is a nonce word drawn for the founder from
consonant-vowel syllables. One life, one script shared with the controls (`words.script`):

1. **Naming**: 40 rooms of 4 to 6 things of the 8 trained kinds (no kind in its unseen colour,
   drawn for the founder); the robot looks around (8 moments), then 15 namings a room: the robot
   turns to a drawn heading, attends to a drawn thing (in view or out of it) and hears its kind's
   word (rate .5), its colour's (.25) or its place's around the body (.25).
2. **Practice**: 24 rooms of 3 things of kinds 0 to 3; 10 questions a room with the teacher's
   answer, drawn from: where a thing is, its colour, what has a colour held in the room, what is at
   each place around the robot, a kind not in the room, a foil (foils 0 to 5). Kinds 4 to 7 are
   never asked about in practice.
3. **The test**, nothing taught: 30 fresh rooms of 4 to 6 trained things, every question form
   (where and what colour of every thing, what has one colour held there, what is at each of the
   four places, a trained kind not in the room, a foil of 6 to 11); 16 rooms with one trained
   thing in its kind's unseen colour (its colour, where it is, what has that colour).
4. **New words**: each of the 4 new kinds is shown in a room and named three times (its kind's
   word, the robot attending to it); then 16 fresh rooms holding new kinds, each asked where it is
   and its colour.

An answer is right where the words said are the answer's words as a set, one a thing.

## Founders and arms

Founders 32600000 to 32600011 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| the brain | `cadence.words.Words`, the defaults above | 6,656 |
| LSTM, transformer, online | one pass over the script in the brain's order, one Adam step per naming (its two sequences) or practice question (`controls.py`) | at 6,656 (nearest width) |
| LSTM, transformer, offline | shuffled epochs with early stopping on a tenth held aside | at 6,656 |
| LSTM, transformer, offline, large | the same at nine times the brain's parameters (a reading, no gate) | at 59,904 |

The controls are given the same script as tokens: the room's things as concept tokens (kind,
colour, place around the robot as it faces), each naming as two sequences (concepts then word,
word then concepts), each practice and test question with its answer. The new words' three
namings are one Adam step each.

## Readings

For every arm and founder: W1 (where and what colour of trained things, pooled), its practised
(kinds 0 to 3) and named-only (kinds 4 to 7) parts, W2, W3 (the colour and what has the colour of
a thing in its unseen colour; where it is as a reading), W4's out-of-view share against the map's
own rate (the share of out-of-view things the map holds within 0.15 m of their place with their
colour), never seen and never taught, W5 by colour and by place; W6's ablation (the brain's map and
body with its lexicon's synapses at zero, answering every W1 question); parameters,
multiply-accumulates (the brain: every lexicon settle at its sweeps times its synapses, every map
moment at one sweep and every map question at two over the map's synapses; the controls: analytic,
a backward pass at twice the forward) and seconds per life.

## Gates

A gate holds where the mean over the twelve founders reaches its threshold and at least ten of
twelve founders do.

- **W1, words from naming**: W1 at .95, and its named-only part at .95.
- **W2, a new word in three namings**: W2 at .90.
- **W3, combinations never seen**: W3 at .90.
- **W4, from the state**: the out-of-view share within .02 of the map's own rate on every
  founder; never seen at .95; never taught at .95.
- **W5, both directions**: by colour at .90 and by place at .90.
- **W6, said by the brain**: every answer is the lexicon motor's letters (`Words.ask`, read by the
  test `tests/test_words.py`), and with the lexicon's synapses at zero W1 reads .05 at most on
  every founder.
- **W7, beside the controls**: the table above for every arm, with parameters and
  multiply-accumulates.

**Expectations, stated before the run.** From development founders 900 to 905 (no gated life):
the brain reads 1.0 on W1 to W4 and .99 to 1.00 on W5; the matched controls read W1 at .05 to .21
offline and .00 online; the transformer at nine times the parameters fits its own practice
questions at .83 after 60 epochs.

**What a failure names.** A W1 or W2 miss with the map holding the thing names the lexicon (the
question's concept or read); a W5 miss names the map's read of a colour or a place around the
body; a W4 out-of-view gap names the map's carry.

## Compute

One AWS c7i.8xlarge spot box (Name=cadence-0100-words), one process a life, one BLAS thread a
process, results synced to s3://cadence-0100-words-20261113.
