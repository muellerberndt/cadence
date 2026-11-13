# Words for things: declared protocol 2, pistachios (cadence#326, W11)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

The parrot learns its words from naming, as in protocol 1. Then the answers to the practice
questions are never told: the parrot answers, and a right answer earns a pistachio (a reward of
1, `Words.reward`). Does it learn what each question word asks for (a place, a colour, a thing)
from the pistachios alone? Does every word taught by naming keep its meaning while the pistachios
teach the questions? With the pistachios withheld, is what they teach at chance?

## The brain

`cadence.words.Words` with its defaults, at the declared commit: the map with colours, the body,
the lexicon learned from naming. A practice question by reward is `ask(question, explore=True)`:
the brain tries a read drawn from the reads' settled activity (a softmax at `temperature` .1),
and `reward(1.0)` for a right answer, nothing for a wrong one. The reward teaches the question's
words of asking (the words that name no concept of the senses) the read the answer took, at the
reward's surprise: the reward less the chance the brain gave that read. Content words keep no
read, and the lessons of reward change no synapse between a word and the concepts of the senses.
Learned parameters: 6,656.

## The chamber

Protocol 1's script (`benchmarks/words/words.py`), with the practice that teaches:

| Arm | Practice |
| --- | --- |
| `told` | 24 rooms of 10 questions, the teacher says each answer (protocol 1, the reference) |
| `reward` | 60 rooms of 10 questions, answers never told, a pistachio for each right one |
| `withheld` | 60 rooms of 10 questions, answers never told, no pistachio |

The naming, the test and the new words are protocol 1's.

## Founders and arms

Founders 32600100 to 32600111 (twelve, no life in any lane). The brain in the three arms above.
The controls: the chamber's LSTM and transformer at the brain's parameter count, online and
offline, given the same script as tokens, the namings as sequences and the practice questions
by reward: each control samples its answer from its own distribution and a right answer earns
1 (REINFORCE with a running baseline; one pass online, 30 passes after the namings' epochs
offline).

## Readings

For every arm and founder: W1 and its named-only part, W2, W3, W4 never seen and never taught,
W5 by colour and by place where things are (`W5_by_place_held`: an empty place's answer is
"I do not know", a word the pistachios cannot teach), W11_kept (the share of the words that name
a concept after the namings whose concept is the same after the practice), the pistachios earned,
W6's ablation; parameters, multiply-accumulates and seconds.

## Gates

A gate holds where, for each of its readings, the mean over the twelve founders is on the stated
side of the threshold and at least ten of twelve founders are.

- **W11a, what a question asks for, from pistachios**: in `reward`, W1 at least .95, W5 by
  colour at least .90 and W5 by place where things are at least .90.
- **W11b, the words of naming kept**: in `reward`, W1's named-only part at least .95 and
  W11_kept at 1.0.
- **W11c, withheld**: in `withheld`, W1, W5 by colour and W5 by place where things are at most
  .34 (a read drawn at random among three).
- **W7**: the table of every arm with parameters and multiply-accumulates.

W2, W3 and W4 are read and not gated: "I do not know" is a word, learned when the teacher says
it, and no pistachio teaches it.

**Expectations, stated before the run.** From development founders 900 to 905 (no gated life):
`reward` at 60 rooms reads W1, W2, W3, W5 by colour and by place where things are at 1.0 (at 24
rooms one founder reads "what is <colour>?" at .0: the kind read was tried too seldom); `withheld`
reads .0 everywhere (the parrot says nothing it was not taught); `told` reads 1.0. The controls
by reward read W1 at .00 to .05 (founders 900 to 902).

**What a failure names.** W11a failing with the reads of "what" and "is" leaning to the colour
names exploration (the right read tried too seldom); W11b failing names a reward lesson that
reaches a content word.

## Compute

One AWS c7i.8xlarge spot box (Name=cadence-0100-words), one process a life, one BLAS thread a
process, results synced to s3://cadence-0100-words-20261113.
