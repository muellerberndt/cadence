# The scaled lexicon: declared protocol 9 (cadence#302)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gate below are fixed; a failed gate is reported as failed.

## The question

[Protocol 7](MEANING_PROTOCOL_7.md) read a word said by its parts in the chamber's own lexicon,
four nouns and two verbs, where two of the sixteen agent-number-verb triples are held out and
fourteen are taught. cadence#302 asks for the same reading at a lexicon where held-out
combinations are most of the language: does a brain that says each word by a stem and a form say a
clause whose agent, number and verb no lesson showed together, where three of four such triples
are new?

## The lexicon

Sixteen nouns and eight verbs (`CADENCE_GRAMMAR_LEXICON=16x8`, read once when the chamber is
imported): 54 words and 89 inputs, against 18 and 35. The grammar, the three families, their mix,
their shapes and the 4,000 batches stay. With no variable set the chamber is the one protocols 1
to 8 read, episode for episode (`test_the_default_lexicon_keeps_its_episodes`).

**The split** (`Split.scaled`). Each agent in each number is taught with two of the eight verbs,
64 triples of 256, and the other 192 (three quarters) are held out. The taught verbs go round the
verbs in a founder's own random order, so every verb form is taught with four agents and every
agent form with two verbs: each stem and each form is said in lessons, and most of their
combinations are not. One pair of nouns is never agent and patient of one trained clause, as
before. `clause` holds trained clauses and `clause_held` held-out ones.

**Development** (founders 0 and 1, 16 nouns and 8 verbs, the arms below with the held message
emptied at a path's start, a development variant that changes path episodes alone and is not
run here): `open_256` read held-out clauses at .97 and .97 and said their verb's form at .99 and
1.00 with the previous words given, against the probe's .99 and 1.00 at that settle; `words_256`
read them at .75 and .84 and said the form at .69 and .80 against the probe's .94 and .98. Role
twins .95 against .69.

## Founders and arms

Founders 11128600 to 11128611 (twelve, no life in any lane). 4,000 batches of the first grammar at
16 nouns and 8 verbs.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `words_256` | `message=280` with ten winners a moment, `message_read=True`, `history_decays=(0.0,)`, 256 association neurons, every row taught, the recall read barred: the reference | 707,904 |
| `open_256` | `words_256` saying each word by its parts (`slots=(30, 3)`: sixteen nouns, eight verbs, `near`, four brackets, `end`; singular, plural, none), the form of a word without a number left open (`"parts": "open"`, [protocol 8](MEANING_PROTOCOL_8.md)) | 700,638 |

Ten winners a moment (protocol 4's Amendment 2, protocol 7's `w10_256`): at 89 inputs five winners
give two forms of one role a shared code more often.

**Controls.** The chamber's LSTM (702,314 parameters) and transformer (699,774) at `open_256`'s
count, online and offline, on the first six founders.

**At 512 nouns and 128 verbs** (`CADENCE_GRAMMAR_LEXICON=512x128`: 1,286 words, 1,937 inputs,
32,768 of 131,072 agent-number-verb triples taught), reported beside the gate and not judged: the
same two arms on founders 11128600 and 11128601 (`words_256` 10,381,568 parameters, `open_256`
8,983,990, `slots=(646, 3)`), and the LSTM (8,972,682) and transformer (8,967,626) at
`open_256`'s count, online, on the same two founders. An offline control replays up to 30 epochs,
more than a day a life at that size, and is not run there.

## Readings

The panels `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `path_long` and
`clause_long` by arm and control, with multiply-accumulates and seconds per life; `clause_held`
by founder; on held-out clauses (8 batches of 16 per trained shape) the verb's form said with the
previous words given beside the probe's reading of the stem and the agent's number at that
settle; agreement and the share of wrong forms a distractor's number drew; `open_256` less
`words_256` on every panel, paired by founder.

## Gate

- **S1, the scaled lexicon**: `open_256` reads `clause_held` at .90 on 10 of 12 founders, said
  from the held state.

**Expectations, stated before the run.** S1 holds on 11 or 12 of 12, with the verb's form said at
.97 and over against the probe's reading. `words_256` reads held-out clauses at .70 to .90 and
`open_256` reads them .10 to .20 over it, beyond two standard errors; role twins .15 to .30 over.
Paths are unjudged here: from the held state they are said at the level of a guess among the
facts' objects with either motor ([protocol 8](MEANING_PROTOCOL_8.md)'s development), and the
motor of parts reads them under the motor of words. At 512 nouns and 128 verbs no life was read
before the declaration; the expectation is the arms' order on held-out clauses, `open_256` over
`words_256`.

## Compute

One AWS c7i.4xlarge (Name=cadence-0100-paths), one process per life, one BLAS thread per process:
the four lives at 512 nouns and 128 verbs first (about five hours each), beside them the 24 brain
lives and 24 controls at 16 nouns and 8 verbs, then the four controls at 512 and 128.
