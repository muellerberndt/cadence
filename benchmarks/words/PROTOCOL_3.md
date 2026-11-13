# Words for things: declared protocol 3, the training game (cadence#326)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

The parrot learns a word for a thing as an African Grey does: no naming lesson is given. Hearing
a word while it focuses on a thing leaves a mark on their links (the library's free and nudged
phases and their contrast, held as an eligibility trace, `Lexicon.mark`); the mark fades every
moment (`Lexicon.fade`, a half a moment); a pistachio makes the marks of that moment lasting links
in both directions (`Lexicon.confirm`, through `Learner.apply`). Does the training game teach
every word, so that the parrot then answers protocol 1's questions? Does naming with no pistachio
leave it at chance? Does withholding the pistachio for some words stop those links alone?

## The game

`benchmarks/words/game.py`, 60 rooms of 4 to 6 trained things, 8 rounds a room; a round about a
drawn thing and one of its words (its kind's at .5, its colour's at .25, its place's around the
parrot at .25), with six moments between steps:

1. **Look and listen**: the parrot's focus moves to the thing; the trainer says the word twice.
2. **The parrot's turn**: "what is this?" ("what colour is this?", "where is this?"); the parrot
   tries a read and says its best guess, the word it heard last where no link names the thing;
   a right answer earns a pistachio; a wrong one none, and the trainer says the word again. A
   pistachio for an answer the parrot imitated makes links and teaches no read.
3. **Understanding by acting**: the trainer says a word of a thing in the room; the parrot turns
   to the thing its links name most (one at random where they name none); the right one earns a
   pistachio.
4. **Two trainers** (rooms 21 to 60): the parrot watches one trainer answer another about a thing
   (right at .8); a right answer's pistachio, seen given, makes the parrot's marks links.
5. **Thinning out** (rooms 41 to 60): a right answer earns a pistachio at .5.

Then protocol 1's test (30 fresh rooms, 16 rooms of unseen colours; nothing taught), and four new
kinds, each taught by three rounds of steps 1 and 2, asked about in 16 fresh rooms.

## Founders and arms

Founders 32600200 to 32600211 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| `brain_game` | `cadence.words.Words`, every step paid | 6,656 |
| `brain_naming` | the same game, no pistachio ever | 6,656 |
| `brain_withheld` | no pistachio at steps 2 to 4 for a colour's word; every other word paid | 6,656 |
| `table_game` | a count table in the lexicon's place (`arms.Table`): the same program, marks and pistachios (a control) | 6,656 |
| `delta_game` | a delta rule in the lexicon's place (`arms.Delta`; a control) | 6,656 |
| LSTM, transformer, online and offline | the same game as tokens (`controls.run_game`): at the parrot's turn a sampled word, at the pointing a sampled position, each by REINFORCE from the pistachio; a paid answer watched is a supervised step; one pass online, ten offline | at 6,656 (nearest width) |

## Readings

For every arm and founder: each trained word heard alone names its concept above the floor
(`understood_kind`, `_colour`, `_place`) and each concept said is its word (`said_*`); W1 and its
two forms (`W1_where`, `W1_colour`), W2, W3, W5 by colour and by place where things are; the game's
rates of right turns and right pointing by stage; parameters, multiply-accumulates and seconds.

## Gates

A gate holds where, for each of its readings, the mean over the twelve founders is on the stated
side of the threshold and at least ten of twelve founders are.

- **G1, the game teaches every word**: in `brain_game`, understood and said, each group, at least
  .95; W1 at least .95; W5 by colour and by place where things are at least .90; W2 at least .90.
- **G2, naming alone, no pistachio**: in `brain_naming`, understood and said, each group, at most
  .05; W1 at most .34.
- **G3, a pistachio withheld stops that link**: in `brain_withheld`, the colours understood and said
  at most .05 and W1's colour form at most .34; the kinds and places understood at least .95 and
  W1's where form at least .95.
- **W7**: the table of every arm with parameters and multiply-accumulates.

**Expectations, stated before the run.** From development founders 900 to 902 (no gated life):
`brain_game` reads 1.0 on every reading; `brain_naming` 0.0; `brain_withheld` colours 0.0, kinds and
places 1.0; the delta rule passes most of G1 (W1 .83, W5 1.0, every word 1.0) and the table fails W5
(.0); the LSTM and transformer read W1 at .04 to .21. The game's gates do not separate the brain
from a lookup in the lexicon's place: the program around it carries the reads, and the audit's
gates (word order, a wrong value refused, two things in a question, noisy words, a form never
practised) are the next protocol's.

**What a failure names.** G1 failing with words understood names the reads (what a question
asks for, learned from paid answers); G2 failing names a mark that outlasts its moment; G3 failing
names a mark confirmed by a pistachio for another word.

## Compute

One AWS c7i.8xlarge spot box (Name=cadence-0100-words), one process a life, one BLAS thread a
process, results synced to s3://cadence-0100-words-20261113.
