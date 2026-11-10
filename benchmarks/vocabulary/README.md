# Vocabulary chamber: the read's rule on thousands of words

This chamber reads [#266](https://github.com/muellerberndt/cadence/issues/266): the
[grammar chamber](../grammar/README.md) holds 18 words and every reading of the read's rule stands
there. Here the same sentences, panels and scorer run at 200, 1,000 and 5,000 words, with raw codes,
a fixed code and a learned code; a word heard once is asked in another role beside the matched
networks; and a word of several pieces is said. It establishes no default.

## The language

`language.py` holds `Language(nouns, verbs)`, the chamber's three families and two grammars at any
size. Sizes go by class, two noun stems to a verb, each in two forms, with the chamber's six function
words unchanged.

| Words | Noun stems | Verbs | Function words | Inputs | Actions |
| --- | --- | --- | --- | --- | --- |
| 18 | 4 | 2 | 6 | 35 | 18 |
| 200 | 65 | 32 | 6 | 308 | 200 |
| 1,000 | 331 | 166 | 6 | 1,508 | 1,000 |
| 5,000 | 1,665 | 832 | 6 | 7,508 | 5,000 |

`Language(4, 2)` is the grammar chamber to the bit: panels, training batches and a composed life
(`test_vocabulary.py`). At any larger size the split holds the chamber's shares: about one
agent-number-verb triple in eight and one noun pair in six, by a hash of the founder's salt. A
word's own coordinate, a concept per stem and the chamber's eleven flags make the input.

## The composition

Every brain is the read's rule, `Fd`: `Brain.compose(inputs, actions, modules=(64,), episodic=False,
recall=True, recall_recency=0.0, learning_surprise=True, recall_boundary=True, recall_anchor=True,
recall_slope=0.1, recall_position_rate=0.01, recall_relative=True, teaching="deep",
backend="cpu")`, with the memory `Brain.compose` gives a new brain beside those settings (its grid,
capacity of 128 and carried read) and the address lesson as a contrast, the library's one law. The
readings of 6 November below were taken before the composed memory and the contrast law were the
library's, with a capacity of 64 and no grid. The working trace is built in every arm. Its block is
`w^2` synapses of `w^2 + (3 I + 2 A) w`, `I` the key width and `A` the motor neurons.

| Arm | Key width | Setting |
| --- | --- | --- |
| raw | the inputs | none |
| fixed code | 128, 256, 512 at 200, 1,000, 5,000 words | `code=units` |
| learned code | 64, 128, 256 at 200, 1,000, 5,000 words | `learned_code=units`; an observation wider than the code keys the memories on the code alone |
| coded maps | the inputs; the maps read `units` | an emulation (`mapcode.py`): the read's maps read a fixed sparse code of the event, keys and values stay raw, a map holds `units * inputs` in place of `inputs ** 2` |

## Words by their pieces

`pieces.py` holds `PieceLanguage`: the chamber's clauses, brackets and paths with one motor neuron
per piece in place of one per word. A word is a stem of pieces and a number piece (`ref ##act ##or
##0`, `dog ##s`); a verb's last piece agrees with its agent. `##0` is a silent piece, so that every
form of a stem has one length and a batch's streams keep in step. `lexicon.py` makes the forms from
a coder's words and a ranked English list by byte-pair merges, words of at most four pieces. The
sentences carry no English meaning: only the forms and how they share pieces are real.

| Lexicon | Stems | Pieces | Inputs (heard) | Actions | Brain's parameters |
| --- | --- | --- | --- | --- | --- |
| 200 words | 200 | 143 | 162 | 150 | 101,925 |
| 1,000 words | 1,000 | 391 | 410 | 398 | 345,709 |
| 5,000 words | 5,000 | 994 | 1,013 | 1,001 | 1,451,611 |

A meaning names a word in one of two ways (`route`). `heard`: the word's pieces, one moment each,
every moment with its role flag, so a word the brain has never been taught can be asked as long as
its pieces are known. `concept`: one moment, the word's own concept coordinate and flags, as in the
grammar chamber, so only taught words can be asked and the brain has to know their pieces.

A tenth of the stems are novel per founder: they stand in no training sentence, and every piece of
them does. `swap` adds a family: the world says `A verb B end` with no flags, the brain says `B verb
A end`, each noun in the role it was not heard in and the verb agreeing with its new agent.

| Panel | Asks |
| --- | --- |
| `taught` | stems from training, zero to two distractors |
| `long` | three and four distractors |
| `novel_agent`, `novel_patient` | one novel stem in that role (`heard` only) |
| `novel_all` | every noun and the verb novel (`heard` only) |
| `swap`, `swap_novel` | heard in one role and said in another; taught stems, novel stems |
| `dyck` | the chamber's brackets |
| `path`, `path_novel`, `path_long` | chains over taught stems, chains over novel stems, three hops |

## One hearing across episodes

`hearing.py` reads the word language with `fresh` noun stems out of training. On a fork of a taught
life sixteen fresh stems are heard, one batch of sixteen taught sentences, eight as the agent and
eight as the patient. Then they are asked in new sentences in the same role and in the other one,
after 1, 2, 4, ... 512 hearings, with zero hearings as the baseline. `said` is the fresh word in its
place, closed loop; `exact` the whole sentence; `rank` the fresh word's rank among all outputs,
teacher-forced. `mention` asks the word in the episode that first says it, a path `X near wug end`.
A network hears the same batches: one Adam step per batch (equal exposure), or sixteen (nearer
equal compute). The tied form shares its embedding with its output, and the pretrained form, a
labelled control, is handed a table in which a stem's words and concept share a vector from the start.

## Controls

`controls.py` holds the chamber's LSTM and causal transformer over any language, their size the one
whose parameter count is nearest the brain's. One pass is one Adam step per batch at rate 0.003, the
brain's exposure. Replay runs the same stored batches in shuffled epochs at rate 0.001 until the
brain's seconds are spent, equal compute. They are independently scored controls and never part of
any brain's answer path.

## Run

```bash
python benchmarks/vocabulary/life.py --words 200 --founder 0 --checkpoints 2000 8000 --out DIR/raw.jsonl
python benchmarks/vocabulary/life.py --summary DIR/raw.jsonl
python benchmarks/vocabulary/lexicon.py --ranked RANKED.txt --words 5000 --pieces 400 --longest 6 --out LEX.json
python benchmarks/vocabulary/life.py --lexicon LEX.json --swap --pseudo 0.5 --founder 0 \
    --checkpoints 2000 8000 --settings '{"recall_capacity": 128}' --save DIR --out DIR/pieces.jsonl
python benchmarks/vocabulary/controls.py --model transformer --lexicon LEX.json --swap --pseudo 0.5 \
    --founder 0 --parameters 357837 --checkpoints 2000 8000 --seconds 7000 --out DIR/controls.jsonl
python benchmarks/vocabulary/hearing.py --speaker brain --words 200 --founder 0 --checkpoints 8000 --out DIR/h.jsonl
python benchmarks/vocabulary/diagnose.py --brain DIR/BRAIN.npz --lexicon LEX.json --swap --pseudo 0.5 --out DIR/d.json
python -m pytest benchmarks/vocabulary/test_vocabulary.py
```

`recall_capacity=128` holds a path of long stems: at the founder's 64 slots a path of six-piece words
drops its first fact before the sentence ends.

`mapcode.py`, `chamber_gain`, `address_scale` and `cortex` (`install_cortex`) in `life.py` are
emulations on one process per life: they change no library file and a brain with them is not saved.
`terms` in a life's spec splits the first head's logits at every word whose content a heard moment
holds by source (`logit_terms`): the event against a slot's key, the address map's event rows, cortex
columns and bias, and the same three of the position maps, each at the right slot less the best
other slot, per kind of word and over the words the read misses.

## Readings (6 November 2026)

Development unless declared: founders 0 to 2, one process per life on a shared 64-thread box at a
load of about 80, so seconds are of that load. A row of one life is a screen. On three lives a
difference under about .1 is noise, on six lives under about .07. `Read` is the share where the
first head attends most to the word's own slot or the one before it, teacher-forced; `said` is the
share of those words said, teacher-forced. Brackets and paths are bimodal, and the tables give them
per life where they spread.

**The read's rule on whole words.** It holds at 200 words and does not hold at 1,000. At 200 raw
words the read lands right at 1.00, 1.00, .98 and .98 by 8,000 batches and the nouns are said at .97
to .99. What is left is the verb's number: its identity reads .98 and its agreement .65, and clauses
read .62 against the matched transformer's .96. At 1,000 raw words the read is right at 500 batches
and then falls and stays down, with words said at .00 at 8,000 batches, while the matched transformer
reads clauses 1.00 at the same exposure. At 5,000 raw words a life costs 11.4 s per batch and reads
nothing at 250 batches; the matched transformer reads .86 at 8,000 batches.

**Words by their pieces.** With one motor neuron per piece the read holds on later and number pieces
at every size read here (.98 to 1.00), on first pieces at 400 tokens and fewer, and not at 994 (.00
to .69 at 2,000 batches), and a word is said piece by piece in one composition. On six fresh founders, with half the training stems
made up on the spot, later pieces of a word never taught are said at .97 to .98 and its first pieces
at .93 to .98, teacher-forced. Whole sentences stand below the transformer on every panel but
the ones both read at .00: .32 against .87 on taught words, .21 against .58 on words never taught, .15
against .84 on words heard in one role and said in another, .68 against .80 on brackets. On a saved
development brain the largest single cause is the verb's agreeing piece, which reads at chance.

**One hearing.** Neither speaker says a word after one hearing at equal exposure. The brain takes
16 to 32 times the hearings of the transformer in the role heard, and more in another role.

**The learned code.** At 64, 128 and 256 units every noun concept takes one code (cosine .9997 to
.9999): it keeps the class and loses the word, so the read and the lookup fail. With 308 units the
key joins the identity, and the code then shows no class (.008 within nouns) at 6.3 s per batch.

### Whole words, by size

| Words | Arm | Params | Batches | Lives | s/batch | Sweeps/act | Clauses | Held | Roles | Agree | Brackets | Paths | Read: agent, distractor, patient, verb | Said |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | raw | 24,384 | 2,000 | 1 | .12 | 61 | .98 | .89 | .94 | 1.00 | .41 | 1.00 | 1.00 1.00 1.00 .66 | 1.00 .98 1.00 1.00 |
| 200 | raw | 224,311 | 2,000 | 3 | .83 | 102 | .01 | .00 | .00 | .07 | .00 | .00 | .99 .96 .90 .17 | .85 .65 .59 .02 |
| 200 | raw | 224,311 | 8,000 | 3 | .73 | 91 | .62 | .68 | .57 | .65 | .00 .16 .64 | .00 .30 .05 | 1.00 1.00 .98 .98 | .99 .98 .97 .66 |
| 200 | raw | 224,311 | 16,000 | 1 | .61 | 79 | .56 | .50 | .50 | .66 | .48 | .00 | 1.00 1.00 .91 .92 | 1.00 .98 .88 .66 |
| 200 | first synapses at their 18-word magnitude | 224,311 | 2,000 | 3 | .59 | 135 | .10 | .11 | .14 | .16 | .33 .00 .09 | .01 | 1.00 .99 1.00 .94 | .97 .87 .90 .15 |
| 200 | first synapses at their 18-word magnitude | 224,311 | 4,000 | 3 | .53 | 127 | .51 | .58 | .52 | .53 | .39 .11 .42 | .02 | 1.00 1.00 1.00 1.00 | 1.00 .99 .99 .52 |
| 200 | fixed `code=128` | 83,551 | 8,000 | 1 | .58 | 82 | .36 | .19 | .23 | .75 | .25 | .00 | 1.00 .98 .88 1.00 | .89 .90 .52 .77 |
| 200 | coded maps, 64 units | 128,175 | 8,000 | 1 | .75 | 93 | .53 | .50 | .44 | .66 | .25 | .00 | 1.00 1.00 1.00 1.00 | .88 .88 .97 .66 |
| 200 | two heads | 411,313 | 8,000 | 1 | .56 | 90 | .59 | .50 | .66 | .61 | .12 | .00 | 1.00 1.00 1.00 1.00 | 1.00 1.00 .98 .61 |
| 200 | learned code, 64 | 73,235 | 2,000 | 2 | 1.1 | 81 | .00 | .00 | .00 | .01 | .00 | .00 | .66 .67 .52 .00; .25 .00 .00 .95 | .02 .00 .00 .00; .00 |
| 200 | learned code, 308, identity joined | 710,335 | 2,000 | 1 | 6.3 | 103 | .00 | .00 | .00 | .12 | .03 | .00 | .98 .98 .94 .42 | .80 .58 .33 .00 |
| 1,000 | raw | 2,870,711 | 8,000 | 1 | 1.68 | 39 | .00 | .00 | .00 | .00 | .03 | .00 | .11 .08 .17 .81 | .00 .00 .02 .00 |
| 1,000 | first synapses at their 18-word magnitude | 2,870,711 | 2,000 | 1 | 1.40 | 43 | .00 | .00 | .00 | .00 | .00 | .00 | .00 .00 .00 .55 | .00 .02 .00 .00 |
| 1,000 | fixed `code=256` | 228,991 | 2,000 | 1 | 1.17 | 40 | .00 | .00 | .00 | .00 | .00 | .00 | .00 .25 .28 .50 | .00 .00 .02 .00 |
| 1,000 | learned code, 128 | 338,275 | 1,000 | 1 | 1.87 | 61 | .00 | .00 | .00 | .00 | .00 | .00 | .14 .00 .41 .50 | .00 |
| 5,000 | raw | 59,302,711 | 250 | 1 | 11.4 | 32 | .00 | .00 | .00 | .00 | .00 | .00 | .30 .00 .00 .75 | .00 |
| 5,000 | fixed `code=512` | 774,175 | 1,000 | 1 | 2.91 | 32 | .00 | .00 | .00 | .00 | .00 | .00 | .00 .00 .00 1.00 | .00 |
| 5,000 | coded maps, 256 units | 4,231,023 | 250 | 1 | 4.05 | 32 | .00 | .00 | .00 | .00 | .00 | .00 | .30 .00 .00 .77 | .00 |
| 5,000 | learned code, 256 | 2,434,931 | 250 | 1 | 6.77 | 32 | .00 | .00 | .00 | .00 | .00 | .00 | .00 .00 .00 1.00 | .00 |

The trace block is built in every row: it is `w^2` = 4,096 of the synapses, and at raw 5,000 words
the address map alone holds 56.4 M of the 59.3 M parameters. Memory is .3 GB at 200 and 1,000 words
(.42 GB raw at 1,000) and 1.9 GB raw at 5,000. Raw 5,000 words stop being practical on one process: a
batch of 16 takes 11.4 s, 2,000 batches 6.3 h.

| Words | Control, one pass | Params | s/batch | Batches | Clauses | Held | Roles | Agree | Brackets | Paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | transformer, 3 lives | 220,826 | .014 | 2,000 | .90 | .92 | .88 | .95 | .09 | .04 |
| 200 | transformer, 3 lives | 220,826 | .015 | 8,000 | .96 | .97 | .96 | .97 | .61 .00 .73 | .13 |
| 200 | transformer, 3 lives | 220,826 | .018 | 32,000 | .99 | .99 | .99 | 1.00 | 1.00 .23 .92 | .19 |
| 200 | LSTM, 3 lives | 225,816 | .011 | 8,000 | .97 | .95 | .94 | 1.00 | 1.00 | .03 |
| 1,000 | transformer, 1 life | 668,940 | .025 | 8,000 | 1.00 | 1.00 | .98 | 1.00 | .56 | .16 |
| 1,000 | LSTM, 1 life | 667,000 | .020 | 8,000 | .58 | .48 | .45 | .98 | .77 | .00 |
| 5,000 | transformer, 1 life | 785,622 | .034 | 8,000 | .86 | .84 | .80 | .94 | .75 | .02 |

The controls at 1,000 and 5,000 words are matched to the coded-map brain and the fixed-code brain,
far smaller than the raw brains. Equal compute: for its 8,000 batches a raw 200-word brain takes about
5,800 s and the transformer's one pass 75 to 120 s. Replaying the same stored batches for 5,000 s on
founder 0, the transformer reads clauses 1.00, roles 1.00 and brackets 1.00 from the second epoch
(143 s) and paths .78 from the sixteenth (1,467 s).

**Where raw 1,000 words go.** The read on the agent reads 1.00 at 500 batches, .08 at 2,000 and .11
at 8,000. Three causes were read, one life each, and none holds it:

| Arm at raw 1,000 | Read: agent, distractor, patient, verb, at 2,000 batches |
| --- | --- |
| plain | .08 .15 .34 .48 |
| the chamber's gain on the first synapses (`chamber_gain`) | .00 .00 .00 .55 |
| a vocabulary that grows, 200 words for 2,000 batches, all by 5,000 | at 5,000: .00 .00 .00 .00 (.83 .73 .78 .00 at 2,000) |
| the address lesson's temperature by the key width (`address_scale`) | .95 .35 .77 .00 (.39 .10 .75 .00 at 1,000; .14 .00 .73 .11 at 1,500) |

Three more cuts, one life each: `teaching="normalized"`, no relative floor, holds the read fixed at
1.00 .67 .20 .00 from 500 to 2,000 batches with words said at .00; a module of 128 (3.34 M) reads .53
.02 .78 .00 at 1,000 with words said at .00; words named by their own coordinate, with no concept
coordinates and 1,008 motor neurons, read .00 on the nouns at 1,000 with words said at .00, where the
same form at 200 words reads 1.00 .90 .91 .00 and says .75 .31 .58 .00 at 2,000. None of the seven
brings a word said at 1,000 tokens above .02, where at 400 tokens or fewer words form by 2,000 batches,
so the failure follows the number of tokens, and it needs no concept coordinates. Whole-word brains at
1,000 words stay at 33 to 44 sweeps per act, a newborn's, where brains at 400 tokens or fewer reach 90
to 150 and the 994-piece brain 80. The cause is open.

**Steps and gain at 200 words.** Three times the deep point's steps is much worse (said .03 .04 .00
against .85 .65 .59 at 2,000 batches, one life). A projection's first synapses scale by its fan,
`sqrt(6 / (pre + post))`, and at a one-hot word only one presynaptic neuron is active, so the drive
through it falls with the square root of the vocabulary. Restored to its magnitude at 18 words
(`chamber_gain`), on the same three founders at 2,000 batches, the read lands right at 1.00 .99 1.00 .94
against .99 .96 .90 .17 and the words are said at .97 .87 .90 .15 against .85 .65 .59 .02. At 4,000
batches clauses read .51 and the nouns are said at 1.00 .99 .99, where raw takes 8,000 for .62 and
.99 .98 .97. The verb stays at .52, so agreement is what the gain leaves. It adds no parameter and reads 127
sweeps per act against 91.

### Words by their pieces

| Lexicon | Training | Arm | Params | s/batch | Batches | Lives | Taught | Never taught: agent, patient, all | Swap, never-taught swap | Brackets | Paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 stems, 143 pieces | plain | brain | 102,379 | 1.24 | 8,000 | 1 | .23 | .02 .05 .00 | .20 .06 | .02 | .16 |
| | plain | transformer | 102,000 | .022 | 8,000 | 2 | .81 | .02 .04 .00 | .95 .00 | .66 | .23 |
| | half made up | brain, declared | 102,379 | 1.08 | 8,000 | 6 | .32 | .28 .28 .21 | .24 .15 | .68 | .08 |
| | half made up | transformer, declared | 102,000 | .016 | 8,000 | 6 | .86 | .75 .76 .58 | .88 .84 | .80 | .32 |
| | half made up | tied transformer, declared | 101,456 | .018 | 8,000 | 6 | .79 | .70 .68 .60 | .90 .87 | .62 | .25 |
| | half made up | LSTM, declared | 102,054 | .009 | 8,000 | 6 | .00 | .00 .00 .00 | .01 .00 | .91 | .01 |
| | half made up | brain, two heads | 180,189 | 1.18 | 8,000 | 1 | .30 | .39 .27 .42 | .44 .42 | .67 | .16 |
| 1,000 stems, 391 pieces | plain | brain | 346,659 | 3.43 | 2,000 | 1 | .03 | .00 .00 .00 | .00 .00 | .33 | .05 |
| | plain | transformer | 349,458 | .029 | 8,000 | 1 | .67 | .56 .52 .23 | .86 .22 | .64 | .83 |
| | half made up | brain | 346,659 | 2.09 | 3,000 | 1 | .00 | .00 .00 .00 | .00 .00 | .00 | .02 |
| 5,000 stems, 994 pieces | plain | brain | 1,453,767 | 5.17 | 2,000 | 1 | .00 | .00 .00 .00 | .00 .00 | .00 | .00 |
| | plain | transformer | 1,450,605 | .067 | 4,000 | 1 | .27 | .23 .12 .12 | .45 .22 | .25 | .00 |
| 5,000 stems, 400 pieces, up to six | plain | brain | 357,837 | 2.52 | 3,000 | 1 | .00 | .00 .02 .00 | .00 .00 | .23 | .06 |
| | plain | transformer | 361,397 | .028 | 3,000 | 1 | .20 | .00 .06 .00 | .42 .47 | .56 | .11 |
| | plain | transformer | 361,397 | .030 | 8,000 | 1 | .75 | .50 .61 .36 | .95 .95 | .86 | .20 |

`Taught` at 5,000 stems over 400 pieces reads .00 with the agent's word said right on .70 of sentences
at 3,000 batches (the transformer .91). Five thousand words, `kubernetes` as `k ##ub ##er ##n ##et
##es`, fit in 407 motor neurons and 358 k parameters, against 5,000 motor neurons and 59.3 M at raw
codes. The read holds there on later and number pieces at 1.00 and on the nouns' first pieces at .81 to
.92, on taught words and much the same on words never taught.

Teacher-forced, read then said:

| Lexicon, batches | Agent first | Agent later | Number | Distractor first | Patient first | Patient later | Verb first | Verb later |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200, plain, 8,000, taught | 1.00 1.00 | .99 1.00 | 1.00 1.00 | .98 .96 | 1.00 1.00 | 1.00 .99 | .89 .84 | 1.00 1.00 |
| 200, plain, 8,000, never taught | .95 .91 | .96 .61 | .95 .97 | .92 .85 | .91 .80 | .95 .57 | .95 .06 | .99 .41 |
| 200, half made up, 8,000, never taught, 6 lives | .97 .98 | .99 .98 | .98 .98 | .91 .90 | .89 .93 | .99 .98 | .63 .54 | .95 .93 |
| 1,000, plain, 2,000, taught | .97 .86 | 1.00 .96 | 1.00 1.00 | .88 .73 | .94 .64 | 1.00 1.00 | .70 .09 | .99 .92 |
| 1,000, half made up, 3,000, never taught | .97 .28 | 1.00 .99 | 1.00 1.00 | .67 .19 | .64 .13 | 1.00 .98 | .23 .02 | 1.00 .98 |
| 5,000 over 994, plain, 2,000, taught | .31 .00 | .99 .01 | 1.00 .14 | .69 .00 | .17 .00 | 1.00 .01 | .00 .00 | .98 .02 |
| 5,000 over 400, plain, 3,000, taught | .92 .70 | 1.00 1.00 | 1.00 1.00 | .92 .56 | .81 .59 | 1.00 .99 | .41 .27 | .99 .99 |
| 5,000 over 400, plain, 3,000, never taught | .95 .56 | 1.00 .94 | 1.00 1.00 | 1.00 .88 | .58 .38 | .99 .92 | .41 .19 | .97 .91 |

**Made-up stems.** With plain training the read lands right on a word never taught and the later
pieces are said at .41 to .65 at 8,000 batches: the cortex answers with the piece sequences it has
memorized. With half
the stems made up on the spot they are said at .97 to .98, and the transformer learns the same from
them (never-taught sentences .00 plain, .58 made up). The change is to training and not to the brain.

**The value path's gain.** With `recall_gain` the later pieces are said at .82 to .99 at 2,000 batches
and first pieces at .02 to .36, with taught sentences .00 (one life): at a word's first piece the read
lands on that piece's own slot, and the next port's token, the second piece, outvotes the value port's.

**Words by concept.** A meaning that names a word by its own concept asks the brain to know the word's
pieces. At 200 stems, 4,000 batches, one life (276,779 parameters): the agent's word reads .42 with its
first piece said at 1.00 and later pieces at .99 teacher-forced; the patient's first piece at .03 and
the verb's at .02; taught sentences .00. The transformer on the same route reads taught .62 and the
agent's word 1.00 at 8,000, and .50 at 4,000.

**Where whole sentences go wrong.** On a saved brain (founder 0, made-up stems, 4,000 batches, 128
sentences a panel, `diagnose.py`):

| Panel | Right | First wrong word: agreeing piece | `near` | Verb first | Distractor first | Patient first | Teacher-forced: agreeing piece |
| --- | --- | --- | --- | --- | --- | --- | --- |
| taught | 31 | 39 | 28 | 15 | 11 | 0 | .53 |
| never taught | 62 | 44 | 7 | 6 | 5 | 1 | .61 |
| never-taught swap | 27 | 53 | 0 | 21 | 0 | 16 | .48 |

Every other kind reads .71 to 1.00 teacher-forced. A verb that agrees with a singular agent says `##s`,
which the slots of plural nouns hold, and the read's lesson names a slot by what it holds, so the
shared token could mislead the lesson. On one life it does not: with tokens of its own (`##vs`, `##v0`)
agreement on taught sentences reads .22 at 3,000 batches against .22 with the shared piece at 4,000.
With two heads as well it reads .50 at 2,000 and .36 at 3,000, taught .42 and .34, never-taught .27 and
.22, never-taught swap .34 and .34, one life, against the declared transformer's taught .15 at 2,000
and .73 at 4,000. On raw words a second head brings the read and the nouns to 1.00 and leaves agreement
at .61. Agreement is open.

### The declared confirmation

Declared before any run on them: founders 266000 to 266005, one box, 200 stems over 143 pieces, half
the training stems made up. A failed gate is reported as failed.

| Gate, at 8,000 batches | Requirement | Reading | Verdict |
| --- | --- | --- | --- |
| 1. The read on a word never taught | each of four at least .90; their mean at least .90 on 5 of 6 | agent first .966, agent later .988, patient first .888, patient later .989; the mean .92 to .98 on 6 of 6 | failed, by .012 on the patient's first piece |
| 2. Said, teacher-forced | later pieces at least .90, first at least .80 | later .981 .969 .984; first .982 .932 | passed |
| 3. Never-taught whole sentences | within .05 of the transformer | brain .21 (.27 .19 .17 .27 .28 .11), transformer .58 (.53 .19 .64 .62 .81 .66) | failed |
| 4. Heard in one role, said in another, never taught | within .05 | brain .15 (.02 .06 .28 .08 .16 .28), transformer .84 (.69 .80 .84 .92 .86 .94) | failed |
| 5. Taught words | within .05 | brain .32 (.22 .36 .23 .36 .39 .33), transformer .87 (.75 .86 .91 .88 .95 .84) | failed |

A declared brain took about 8,600 s for its 8,000 batches, at 113 sweeps per act and .034 s per
moment, with no refusal and no exhausted act. Agreement on taught sentences reads .28 to .41 against
the transformer's .89 to .97.

At equal compute the gap widens. Replaying the same stored batches on two of the founders, the
transformer reads taught .98 and .98, never-taught .95 and .77, never-taught swap .92 and 1.00 at its
second epoch (239 s and 269 s), and 1.00 on all three, with paths at .98, at 7,000 s (62 and 60 epochs),
under the brain's 8,600 s. These replays are read beside the gates and carry none.

### One hearing across episodes

Sixteen fresh stems, on a fork of a taught life, founder 0 for the brain and founders 0 and 1 for the
networks, at a base of 8,000 batches (clauses: brain .75, transformer .99). Rank is the fresh word's
place among 200 outputs after one hearing, from its place before any.

| Speaker | Rank at 1 hearing | Same role: 1, 16, 64, 256, 512 hearings | Other role: 1, 16, 64, 256, 512 hearings |
| --- | --- | --- | --- |
| brain | 164 to 166 | .00 .00 .09 .78 .94 | .00 .00 .03 .06 .19 |
| transformer | 130 to 98 | .00 1.00 .98 1.00 1.00 | .00 .12 .36 .59 .66 |
| tied transformer | 133 to 115 | .00 .62 .97 1.00 1.00 | .00 .00 .25 .36 .41 |
| pretrained embedding, a control | 18 to 17 | .03 .97 1.00 1.00 1.00 | .00 .28 .59 .70 .70 |
| LSTM | 184 to 182 | .00 .17 .98 1.00 1.00 | .00 .00 .00 .00 .00 |
| transformer, 16 steps a hearing | 130 to 6 | .64 1.00 .98 1.00 .98 | .14 .38 .53 .86 .97 |

Sixteen Adam steps a hearing is about the brain's compute for that hearing. A word mentioned in the
episode that first says it, a path `X near wug end` with no lesson on the word, reads .00 for every
speaker at no hearing. A pretrained table places a fresh word near its class (rank 18 against 130) and
does not say it at one hearing either, so the embedding alone is not what is missing at this size.

### The learned code

| Units, words | Noun concepts: within-class cosine, pairs the key cannot tell apart | Noun words: within-class cosine | Active units, concept |
| --- | --- | --- | --- |
| 64, 200 | .9998, all | .81 | 13 |
| 128, 1,000 | .9999, all | .97 | 37 |
| 256, 5,000 | .9997, all | .35 | 86 |
| 308, 200, identity joined | .54 (the flag), none | .008 | 152 |

A code narrower than the input keys the memories on the code alone, and there every noun concept reads
as one: the read cannot find a noun and the cortex cannot tell which one it read. That is the group
kept and the member lost, at every size read. Keeping the member takes a code at least as wide as the
input, where the key joins the identity. It then reads about as raw does at 2,000 batches at eight
times the seconds, and shows no class, so it gives a fresh word nothing. At this size and exposure the
learned code adds cost and no gain, on words and on similar words.

## Declared reading at 1,000 words (9 November 2026)

Six fresh founders (2660901 to 2660906), one 32-vCPU box, 1,000 raw words by class, the first grammar,
4,000 batches, the protocol written before any life. The brain is `Fd` on the composed memory, with
and without `recall_cortex_norm`; the matched transformer (2,929,812 parameters against the brain's
2,928,914) takes one Adam step per batch on the same batches. The gate, taught whole sentences within
.05 of the transformer on 5 of 6, is not met:

| Learner | Taught clauses at 4,000, per founder | Within .05 of the transformer |
| --- | --- | --- |
| Transformer | .80 .92 .83 .97 .61 .38 | |
| `Fd` | .00 on 6 | 0 of 6 |
| `Fd`, cortex at unit norm | .00 on 6 | 0 of 6 |

`locate` and `terms` read every word whose content a heard moment holds, teacher-forced. The ranges are
over the six founders; a term's gap is its logit at the right slot less at the best other slot, the
mean over the words the read misses.

| Arm | Batches | The read lands right: agent, mod, patient, verb | Said | Misses | The two terms that carry the misses | Sweeps per act |
| --- | --- | --- | --- | --- | --- | --- |
| `Fd` | 500 | .30-1.00, .10-.54, .02-.73, .00 | .00 | .43-.75 | the position maps' cortex columns -1.74, the address map's cortex columns -1.57 | 21-26 |
| | 1,000 | .22-1.00, .25-.48, .64-.81, .00-.05 | .00 | .42-.68 | position cortex -2.46, cortex -.85 | 23-27 |
| | 2,000 | .09-1.00, .00-.48, .55-.81, .00-.42 | .00-.02 | .43-.71 | position cortex -1.28, cortex -1.05 | 26-30 |
| | 4,000 | .00-.80, .00-.40, .09-.70, .00-.81 | .00-.02 | .47-.83 | the raw key -1.40, cortex -1.09 | 28-38 |
| `Fd`, unit norm | 500 | .88-1.00, .25-.48, .56-.81, .00 | .00 | .43-.49 | the position maps' bias -.99, position cortex -.80 | 22-29 |
| | 1,000 | .06-.66, .00-.48, .44-.81, .00 | .00-.02 | .50-.85 | the position maps' event rows -.65, the raw key -.54 | 24-29 |
| | 2,000 | .00-.03, .00-.38, .02-.70, .00-.50 | .00-.02 | .71-.85 | the raw key -1.39, position event -.58 | 28-31 |
| | 4,000 | .00, .00, .00-.72, .00-.53 | .00 | .78-.87 | the raw key -1.42, position event -.50 | 30-33 |

**The fall above 1,000 tokens, located.**

- **No word is said.** Through 4,000 batches the brain says .00 to .02 of the words, on every founder
  and both arms, where the read lands right on the agent's word on up to 1.00 of rows. An act settles in
  21 to 38 sweeps, near a newborn's, against 116 to 128 at 200 words from 2,000 batches (development),
  so the motor
  readout of a one-hot word over 1,000 actions is not formed at this exposure.
- **The read's selection falls to shared terms.** Each word's own rows of the address and position
  maps learn only from that word's lessons, five times fewer at 1,000 words than at 200; the shared
  columns learn from every lesson. The misses are carried first by the cortex columns of the position
  and address maps (-1.6 to -2.5), and from 2,000 batches by the raw key itself: the event heard matches
  an earlier slot of the same word (-1.4), which the word's own rows would have to outweigh. The cortex
  at unit norm moves the early misses to the position maps' bias and event rows and the late ones to the
  raw key, and the read falls faster: the agent's word lands right at .88 to 1.00 at 500 batches and at
  .00 to .03 at 2,000.

At 200 words (development, one life) the cortex at unit norm brings the verb's read to 1.00 by 2,000
batches, where the read at raw codes reads .00 there, and clauses to .61 at 4,000 batches, against
.62 at raw codes at 8,000 in the readings of 6 November.

## What this does and does not establish

It establishes, on six fresh founders, that a word never taught is said piece by piece with its
previous pieces given, at .93 to .98, in one composition with one motor neuron per piece. In development
it reads that 5,000 words fit in about 400 motor neurons with the read holding on later and number
pieces at 1.00, that the read's rule at raw codes holds at 200 words and fails at 1,000, that the learned
code loses the word when it is narrower than the input, and that a pretrained table does not by itself
give a word in one hearing.

It establishes no parity with the transformer: whole sentences stand below it wherever it reads above
.00, and at equal compute the gap widens. It leaves open the cause of the collapse at raw 1,000 words,
the verb's number at any size, and whether lookups form at 994 tokens, where they read .00 to .14 at
2,000 batches. It holds no reading of real English: the sentences are the chamber's and only the forms
and how they share pieces are real. Most rows are one life. Nothing here reads a spoken word: the pieces
are given as input coordinates, and hearing them from audio is another question.
