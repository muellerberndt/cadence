# Letters chamber

This chamber asks whether one continuing `Brain.compose` life, hearing letters as sounds one frame
per moment, says each letter once per sound and says whole words, including words no lesson
said ([issue 304](https://github.com/muellerberndt/cadence/issues/304)). A listener who writes
down what they hear writes each letter once for its sound; a brain that says a letter on every
frame of its sound, or says it again after a pause, writes letter strings where a word is. Read
[teaching inside a stream](../../docs/teach.md#teaching-inside-a-stream-stepteacher) before
interpreting results.

## The world

Sixteen letters, each a fixed vector of 16 values in [0, 1], heard for one to four frames with
Gaussian noise (standard deviation 0.1); silence is zero. Forty words of three to five letters,
none with a letter twice back to back, so every sound boundary inside a word is a change of
letter. Eight words are never said in training, and the other 32 hold every letter. One to three
silent frames follow each word. A seed fixes the lexicon, the training stream, a held-out stream
of 300 words drawn from all 40 words, the founder brain and the controls' initial weights. Tests
need no download.

At every frame the brain says one of 18 symbols: blank, a letter, or space. Its lesson at a frame
(`Brain.step(frame, teacher=symbol)`) follows the arm's timing:

| Lessons | The symbol taught |
| --- | --- |
| `onset` | a letter at its sound's first frame, space at the first silent frame after a word, blank elsewhere (the supplied alignment) |
| `even` | a word's letters spread evenly over the word's frames, space at the middle of its silence: the timing a speech instrument supplies when it knows where each word lies |
| `frames` | a letter on every frame of its sound, space on every silent frame after a word |
| `own` | a letter from its sound's first frame until the brain's act says it, blank after |

`delay` moves each due symbol later. `motor="split"` gives the brain two motor slots that settle
together: the symbol heard (taught as `frames`) and whether to say it (taught 1 where the arm's
lessons put a symbol). It says the first slot's symbol where the second says 1, blank elsewhere.

## Readings

A saved copy is loaded, reset, and answers the held-out stream greedily without learning. Every
learner is read the same way:

- **As said:** blanks dropped, nothing merged, so a letter said on two steps is a letter said
  twice. Characters (one minus the character error rate over letters and spaces) and inserted,
  deleted and substituted characters per reference character.
- **The CTC reading:** adjacent repeats read once, blanks dropped; the same counts, words (one
  minus the word error rate) and the share of trained and of held-out words said whole.
- **Held steps:** for each due symbol, the steps that say it from its due frame to the next
  symbol's: once, held (two or more), missed (none).
- **Stages,** each with the stage before supplied: the letter from the sound (the letter the motor
  prefers among the letters at its due frame), its emission given the letter, and the word given
  its letters.
- **By place:** misses by sound length, first letter against later letters, and after a letter
  pair the trained words hold against a pair they do not hold.
- **Probes:** a linear read of the letter, its sound's first frame and the letter before, fitted on
  the first 70% of frames and scored on the rest, from each population's settled activation.

## Controls

[`controls.py`](controls.py) trains learners by gradient at the brain's parameter count on the
same streams, cut into chunks of eight words: an LSTM and a causal transformer by CTC (they choose
their own timing), an LSTM on the default's per-frame `onset` targets, and an LSTM with two heads
on the split brain's two targets. `online` is one pass at the brain's exposure; `offline` replays
the chunks in shuffled epochs to the stopping rule. Each records seconds and multiply-accumulates.

## Run

```sh
PYTHONPATH=src python benchmarks/letters/letters.py --protocol benchmarks/letters/protocol-step1.json --arm split --seed 304100 --out <dir>
PYTHONPATH=src python benchmarks/letters/controls.py --model lstm --loss ctc --mode online --seed 304100 --protocol benchmarks/letters/protocol-step1.json --brain-parameters 21331 --out <dir>
PYTHONPATH=src python benchmarks/letters/letters.py --report <dir>
python -m pytest -q benchmarks/letters
```

One life runs per process and writes a gzipped JSON record. Results are kept outside this
repository.

## Step 0: where letters are held and inserted

[`protocol-step0.json`](protocol-step0.json), development founders 0 to 5. The 0.97.0 default
(21,169 parameters) taught `onset` lessons:

| Learner | Parameters | Words heard | Characters | Inserted per character | Said once | Held | Missed | Held-out words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default, `onset` | 21,169 | 6,000 | .943 | .008 | .93 | .006 | .062 | .35 |
| Default, `onset` | 21,169 | 18,000 | .968 | .003 | .96 | .003 | .037 | .46 |
| Default, `frames` | 21,169 | 6,000 | .996 | .003 | .24 | .757 | .001 | .99 |
| Default, `even` | 21,169 | 6,000 | .606 | .003 | .26 | .201 | .537 | .10 |
| LSTM, CTC, one pass | 20,974 | 6,000 | .993 | .004 | .24 | .754 | .002 | .87 |
| LSTM, `onset` targets, one pass | 20,974 | 6,000 | .936 | .001 | .92 | .000 | .081 | .37 |

- Under the supplied alignment the default says a letter on one step and holds almost none.
- Taught at a step the sound does not mark (`even`), it holds letters and misses half of them
  (`ohha___pppp` for "ohap"): the lesson's timing is the term behind held letters.
- Words are lost at the emission of a letter after a new pair: after a pair the trained words
  hold the default misses .006 of letters, after a pair they do not hold .286 (18,000 words). Its
  settled state holds the letter there at .99, and the motor says blank on a quarter of them.

## Step 1: the symbol heard and, apart, whether to say it

Declared in [`PROTOCOL_1.md`](PROTOCOL_1.md) before its first life: two motor slots that settle
together, the symbol heard (taught on every frame of its sound) and whether to say it (taught 1 at
the sound's first frame), against the 0.97.0 default; six fresh founders (304100 to 304105),
18,000 training words each. The gates read the stream as said, nothing merged.

| Learner | Parameters | Training | Characters as said | Inserted per character | Emission gate (5 of 6) | Held-out words | Words gate (.90 on 5 of 6) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Two slots (`split`) | 21,331 | one life | .906 (.76-.97) | .020 | 3 of 6, fails | .40 (.08-.65) | 0 of 6, fails |
| Default, `onset` | 21,169 | one life | .967 (.96-.98) | .004 | 6 of 6 | .47 (.33-.62) | 0 of 6 |
| LSTM, CTC | 21,564 | one pass / offline | CTC reading .997 / .996 | | | .94 / .94 | 5 of 6 / 5 of 6 |
| Transformer, CTC | 20,452 | one pass / offline | CTC reading .765 / .940 | | | .67 / .85 | 4 of 6 / 5 of 6 |
| LSTM, the default's targets | 21,564 | one pass / offline | .971 / .978 | .000 / .000 | | .42 / .53 | 0 of 6 / 0 of 6 |
| LSTM, the split brain's targets | 21,100 | one pass / offline | .984 / .985 | .001 / .000 | | .64 / .67 | 0 of 6 / 0 of 6 |

The CTC learners say each letter on every frame of its sound (.25 of sounds once), so their
stream as said inserts more than one character per character; their words come from the
CTC reading.

- **A measured negative.** The two slots fail both gates. Paired by founder against the default,
  held-out words read -.07 (two standard errors .24) and characters as said -.06 (.07).
- **The term located: the emission decision, learned from the lesson's step.** The split brain's
  first slot names the letter at its due frame at .94 to 1.00 on every life; its second slot says
  it at .80 of due frames on the founder that falls furthest (304102). After a letter pair the
  trained words do not hold, the split brain misses .30 of letters and the default .32, the LSTM
  on the default's targets .36 (one pass) and .29 (offline), the LSTM on the split targets .19
  (offline), and the CTC LSTM .01. Every
  learner taught when to say a letter by a per-frame lesson at the sound's first frame, the brains
  and the LSTMs alike, reads held-out words at .40 to .67; the learners that choose their own
  timing by CTC read .85 to .94.

## Step 2: a teacher's forced alignment on the brain's own beliefs

Declared in [`PROTOCOL_2.md`](PROTOCOL_2.md) before its first life. The teacher knows each word's
frames and letters, not where each letter's sound begins. The first 500 words are taught on the
even split of each word's frames among its letters. Then, every 500 words, a frozen copy of the
brain answers the next 500 words greedily, and each word's frames go to its letters by the
monotonic best path through the copy's log beliefs (`forced_path`). Each letter is taught on every
frame of its path. The path is computed outside the brain. Six fresh founders (304200 to 304205),
18,000 words, gates on the CTC reading.

| Learner | Parameters | Training | Characters | Inserted per character | Emission gate | Held-out words | Words gate | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `forced` | 21,169 | one life | .999 (.997-1.000) | .001 | 6 of 6 | .994 (.97-1.00) | 6 of 6 | 675 |
| `frames` (the supplied alignment) | 21,169 | one life | .997 | .002 | 6 of 6 | .983 (.95-1.00) | 6 of 6 | 583 |
| LSTM, CTC | 20,974 | one pass / offline | 1.000 / .999 | | | .99 / .99 | 6 of 6 / 6 of 6 | 6 / 123 |
| Transformer, CTC | 20,452 | one pass / offline | .884 / .896 | | | .83 / .83 | 5 of 6 / 5 of 6 | 8 / 179 |

- Both gates hold on 6 of 6. The alignment puts .92 to .96 of letter frames on the letter heard at
  its first pass and .999 to 1.000 at its last; paired against the supplied alignment, held-out
  words read +.011 (two standard errors .007).
- After a letter pair no trained word holds, the brain misses .002 of letters, where the lesson
  at a sound's first frame missed .32 (Step 1).
- Every arm here says a letter on every frame of its sound (.23 to .27 of sounds once), so the
  stream as said, nothing merged, inserts more than one character per character. The words come
  from the CTC reading, as they do for the CTC learners.

## Step 3: read speech, a measured target

Declared in [`protocol-speech.json`](protocol-speech.json) before its first life; adapter
[`speech.py`](speech.py). The data is LibriSpeech (CC BY 4.0): 4,000 utterances of
train-clean-100 (765,752 steps of 40 ms, about 8.5 hours) and 300 held-out utterances of dev-clean.
A step is 40 ms of 40 log-mel bands. The teacher knows each utterance's text and not its timing.
It starts on the even split of an utterance's steps among its characters, and every 250
utterances it re-aligns the next 250 by CTC's forced alignment through a frozen copy's beliefs.
Three fresh founders (304300 to 304302).

| Learner | Parameters | Training | Characters | Deleted per character | Words | Letter from the sound | Emission given the letter | Milliseconds per training step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brain, the forced alignment | 116,092 | one life | .209 (.19-.22) | .671 | -.04 | .18 | 1.00 | 6.19 |
| LSTM, CTC | 116,735 | one pass / offline (30 epochs) | .401 / .491 | .282 / .194 | -.04 / .02 | .41 / .50 | .86 / .93 | .03 |
| Transformer, CTC | 113,909 | one pass / offline (30 epochs) | .376 / .500 | .304 / .214 | -.07 / .01 | .36 / .49 | .87 / .93 | .03 |

- No learner forms words at this size and exposure; the CTC learners say about half the
  characters offline.
- **The term located: the letter from the sound.** On the reference's forced path through its
  own beliefs, the brain prefers the right letter on .18 of characters, where the CTC learners
  prefer it on .36 to .50. Once it prefers the letter, it says it (1.00).
- With beliefs that do not tell letters apart, the teacher's path has nothing to follow. It
  moves .83 to .85 of steps from the even split and the lessons drift toward blank. The brain
  says one or two letters with spaces (`h a a a a`) and deletes .67 of characters.
- The brain's milliseconds per step include the teacher's alignment passes and the readings, on
  one core; the gradient learners process a whole utterance in one call.

## Round 3: read speech on the 2,000-word corpus, through learned fields

Declared in [`protocol-speech-fields256.json`](protocol-speech-fields256.json),
[`protocol-speech-fields512.json`](protocol-speech-fields512.json) and, as an amendment,
[`protocol-speech-fields256-p.json`](protocol-speech-fields256-p.json).

- **The list** ([`words-2000.txt`](words-2000.txt), with counts) holds the 2,000 most frequent
  words of the train-clean-100 and train-clean-360 transcripts.
- **The corpus:** the utterances of those splits whose every word is on the list, at most 12
  seconds: 2,100 utterances, 2.5 hours. Held out: the 274 such utterances of dev-clean.
- **Hearing:** a 40 ms step hears its 240 ms of log-mel as a 40 by 24 image through
  `learned_fields=64`.
- **Exposure:** 32 utterances heard side by side as streams, two passes, the forced alignment
  re-aligned every 8 batches. Three fresh founders (304400 to 304402).

| Learner | Parameters | Characters | Letter from the sound | Emission given the letter | Words, list-matched | Training seconds per audio second |
| --- | --- | --- | --- | --- | --- | --- |
| `modules=(256,)` | 1,269,020 | .259 | .347 | .418 | -.01 | .35 |
| `modules=(512,)` | 2,242,588 | .177 (.03-.28) | .349 | .326 | -.01 | .61 |
| `modules=(256,)`, every row taught, `fields_motor=1.0` | 1,285,724 | .329 | .334 | .780 | -.04 | .32 |
| LSTM, CTC, two passes / offline | 1,266,551 | .413 / .437 | .410 / .464 | .880 / .938 | .03 / .09 | .012 |
| Transformer, CTC, two passes / offline | 1,271,117 | .334 / .438 | .331 / .459 | .840 / .947 | .00 / .08 | .011 |

- Learned fields lift the letter from the sound from .18 (Step 3) to .35, at either width.
- Teaching every row with the pools' pathway to the motor region lifts the emission to .78:
  paired by founder, characters +.070 (two standard errors .009).
- No learner forms words at this exposure. The term that loses the brain's words is the letter
  from the sound.
- A letter said once per sound is open; the CTC reading is the decoding.

## What this does and does not establish

The chamber measures composed brains on one synthetic world with supplied sound timing and one
sound per letter; Step 3 reads speech at one size and exposure, where no learner forms words.
