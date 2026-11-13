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

## On a device (issue 314, R3)

For [issue 314](https://github.com/muellerberndt/cadence/issues/314)'s R3, [`device.py`](device.py) runs the round 3 amendment's arm, `Brain.compose((40, 24), 29,
learned_fields=64, modules=(256,), fields_motor=1.0, learning_surprise=False)` with the actor's
rates at zero (`letters`, 1,285,724 parameters), and the same arm on the spoken chamber's 40 by
98 window over 37 symbols (`spoken`, 1,296,996 parameters). A training step is the
forced-alignment lesson's form, `step(x, teacher=labels)`, the utterances heard side by side as
streams; frames are random windows at the chamber's scale.

**The lessons on each backend against the CPU's.** Six steps from one saved CPU brain loaded on
each backend: the efficacy differences from the CPU's as a share of the CPU's own lesson steps
(the largest difference over the largest step, and the median over synapses that moved), the
motor's settled activity and greedy choices on one more act. The fields read and learn on the
host on every backend, and their tables match to the bit.

| Backend | Shape, streams | Largest difference over the largest step | Median relative difference | Motor activity, largest difference | Greedy choices agree |
| --- | --- | --- | --- | --- | --- |
| CUDA float32 (NVIDIA A10G) | letters, 32 | .0065 | 4.1e-5 | 1.9e-5 | 100% |
| CUDA float32 | letters, 128 (4 steps) | .0013 | 2.4e-5 | 1.3e-5 | 100% |
| CUDA float32 | spoken, 32 | .0027 | 2.6e-5 | 5.9e-6 | 100% |
| CUDA float64 | letters, 32 | .0161 | 5.6e-5 | 5.0e-5 | 100% |
| MLX (Apple M4) | letters, 32 | .0136 | 1.0e-4 | 7.0e-5 | 100% |
| MLX | spoken, 32 | .0025 | 2.3e-5 | 6.1e-6 | 100% |
| torch on the CPU, float32 | letters, 32 | .0132 | 4.8e-5 | 3.0e-5 | 100% |
| torch on the CPU, float64 | letters, 32 | .0161 | 5.6e-5 | 5.0e-5 | 100% |

The float64 backends sit at the same distance from the CPU's lessons as the float32 ones: the
difference is where each backend's solve stops within the settle's tolerance, and float32 adds
nothing above it. A brain saved on each backend loads on the CPU and on that backend to the same
parameters, to the bit; a brain saved on CUDA loads on MLX and the CPU, and one saved on MLX on
CUDA and the CPU, to the saved parameters, and four further steps on the second backend stay
within 1.1e-4 of the CPU's largest step. A save mid-episode holds every stream's eligibility
(0.73 GB at 128 streams against 47 MB at 4): save at rest, after `reset()`, as `save_slot` does.

**Seconds per hour of audio.** A training step is a lesson and, for the teacher's re-alignment,
one greedy act of the frozen copy; a step hears 40 ms per stream. Medians after warm-up; the
learned fields keep the codes of frames read again (the same arrays to the bit).

| Where | letters, 32 streams | letters, 128 | spoken, 32 | spoken, 128 |
| --- | --- | --- | --- | --- |
| NVIDIA A10G (g5.xlarge), CUDA float32 | 187 | 126 | 565 | 630 |
| one CPU core of the same machine (AMD EPYC 7R32) | 770 | 691 | 1,198 | 1,190 |
| Apple M4, MLX, shared (load 20 to 37) | 456 to 1,083 | 251 to 501 | 1,981 | 2,181 |

Over a life of 300 steps at 128 streams (12 seconds of each utterance) the A10G reads 140
seconds per hour of audio and holds at most 1.2 GB. Before the kept codes it read 210, 149, 692
and 808. With four host threads in place of one it reads within 4% of these. The learned fields
read and learn on the host on every backend: at 128 streams on the A10G they are about 60% of a
lesson step (a profile of 12 steps), and the device waits for them.

```sh
PYTHONPATH=src python benchmarks/letters/device.py match --rows 32 --places cpu torch:cuda:float32 mlx
PYTHONPATH=src python benchmarks/letters/device.py time --places torch:cuda:float32 cpu --batches 32 128
```

## Round 4: exposure, more speech and the word list

Declared in [`protocol-speech-r4.json`](protocol-speech-r4.json) and
[`protocol-speech-r4-more.json`](protocol-speech-r4-more.json), with amendments on
[issue 304](https://github.com/muellerberndt/cadence/issues/304) that read the gates at pass 6,
the last pass all twelve lives reached by the reading time. The arm is `fields256_p` (1,285,724
parameters), on twelve fresh founders (304500 to 304511), the 2.5-hour on-list set, and a reading
after every pass. A life saves its brain at rest at every re-alignment, and a life stopped on the
way continues from there (ten did once).

| Characters | Pass 1 | 2 | 3 | 4 | 5 | 6 | 10 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brain, `fields256_p` | .306 | .343 | .349 | .363 | .376 | .384 | | |
| LSTM, CTC, online (1,284,410) | .385 | .409 | .419 | .426 | .433 | .433 | .444 | .439 |
| Transformer, CTC, online (1,289,333) | .257 | .318 | .355 | .363 | .383 | .388 | .405 | .420 |
| LSTM, CTC, offline | .402 | .448 | .461 | .468 | .478 | .481 | .477 | .440 |
| Transformer, CTC, offline | .371 | .417 | .439 | .449 | .460 | .469 | .476 | .447 |

- **Both gates fail.** At pass 6, paired against the offline CTC transformer at epoch 6:
  characters -.085 (two standard errors .010) and the letter from the sound -.081 (.011), on 0 of
  12 founders within the bars (-.03 and -.05).
- **The term is the letter from the sound.** The brain climbs at the controls' rate from a lower
  start: its gap to the offline transformer holds at -.08 from pass 3 to pass 6, and no term of
  the brain has stopped climbing at pass 6. Emission given the letter sits at .81 to .83 from
  pass 2, against .91 for the transformer.
- **More speech** (6,023 utterances, 7.1 hours, six founders, read at pass 1): at equal training
  steps the brain reads characters .348, as on 2.5 hours heard three times (.349). The offline
  transformer gains from the grown set (.513 at epoch 10 against .476).
- **The word list.** `cadence.speech.decode.WordBeam` over the 2,000-word list raises the
  brain's words from -.037 to .003 at pass 6 (+.040, two standard errors .010). The CTC controls
  gain more (+.06 to +.09). No learner forms words at this exposure (the best is .147, the offline
  transformer with the list's counts as a prior).
- **Where word knowledge would enter.** On the reference's path through the brain's beliefs,
  letters are heard as space on a fifth to two fifths of their steps (t .36, d .36, e .22), and
  the reference word is among the five list words nearest the said word for 5.8% of words. A word
  prior acts on letters before words form, not on a choice among near words.

## Round 5: the teacher and the lesson's rate

### Step 0: which link of the chain holds the letters

Round 4 located the letter from the sound and left open where in the brain's chain it is lost.
[`ceiling.py`](ceiling.py) trains gradient learners on single links of that chain and reads them
as every learner of the chamber is read: the window the fields read, the pooled code of a saved
brain's fields held still, the steps before beside the step, the brain's settled activity. Each
is a control, never part of a brain's answer path. Development readings, ungated: round 4's
brains at pass 6 (founders 304500 to 304502), trained as the offline controls train (shuffled
epochs, one Adam step per utterance at rate 0.001), read after each of twelve epochs on the 274
held-out utterances. Three founders where no count is given.

| What the learner hears | Learner | Lesson | Parameters | Characters after epoch 1 / 2 / 6 / 12 | Highest |
| --- | --- | --- | --- | --- | --- |
| The brain's pools (576) | linear | CTC | 16,733 | .377 / .390 / .399 / .400 | .406 |
| The brain's pools | one hidden layer of 256 | CTC | 155,165 | .406 / .432 / .455 / .452 | .459 |
| The brain's pools | one hidden layer of 256 | the brain's teacher: the forced path from an even split | 155,165 | .201 / .274 / .331 / .362 | .363 |
| The pools and the pools of the step before | one hidden layer of 256 | CTC | 302,621 | .411 / .435 / .454 / .448 | .457 |
| The pools and the pools of the step before | one hidden layer of 256 | the brain's teacher | 302,621 | .199 / .273 / .368 / .394 | .394 |
| The brain's pools | LSTM of 396 (one founder) | CTC | 1,554,329 | .434 / .465 / .487 / .477 | .487 |
| The window (960) | linear | CTC | 27,869 | .298 / .303 / .291 / .297 | .309 |
| The window | one hidden layer of 256 | CTC | 253,469 | .364 / .404 / .420 / .432 | .432 |
| The window | one hidden layer of 256 | the brain's teacher | 253,469 | .170 / .222 / .287 / .342 | .342 |
| The window | LSTM of 396 (two founders) | CTC | 2,162,585 | .422 / .462 / .486 / .473 | .491 |
| The brain's motor neurons (29) | linear (two founders) | CTC | 870 | .304 / .327 / .394 / .394 | .397 |
| The brain's association (256) | linear (one founder) | CTC | 7,453 | .348 / .380 / .385 / .387 | .392 |
| The brain itself, round 4 (passes 1 / 2 / 6) | | the forced path | 1,285,724 | .306 / .343 / .384 | |

- **The teacher is the first term.** On the same pools and the same hidden layer, the forced
  path from an even split reads .20 under CTC after one epoch, .12 under after six and .09
  under after twelve. Round 4's brain (.306, .343, .384) lies along the curve of a gradient
  learner taught as the brain is taught.
- **The pools hold the letters.** An LSTM reads the pools as it reads the window (.49), and a
  learner without memory reads more from the pools than from the window (.459 against .432;
  a linear readout .406 against .309).
- **Round 4's brain reads what a linear readout of its pools reads:** .397 from its motor
  neurons and .392 from its association, against .406 from the pools.
- **More of the past beside the step adds nothing for a learner without memory:** .457 with the
  step before, .450 with the four steps before (two founders) and .446 to .452 with the heard
  history at decays 0 to .99 (one founder), against .459; .419 through a window of 480 ms and
  .392 through 960 ms (two founders), against .432 through the 240 ms. An LSTM's memory adds
  .03.
- **A copy of the learner's own answers,** read beside the pools as a brain's efference copy
  holds its commands, lifts the hidden layer of 256 from .459 to .473, and to .476 where blank
  issues no command and the copy holds the letters said (one founder).
- A soft path whose sharpness rises over the epochs, and one that weighs blank down, read at
  or under CTC's (.453 to .461, and .37 after two epochs against .43; one founder).

### The soft teacher, development

Each step is taught the soft path of its text through the brain's own beliefs
(`cadence.speech.align.soft_path`, [the guide](../../docs/speech.md#the-soft-lesson-every-path-of-the-transcript)),
read from the brain itself before every batch (`log_beliefs(..., hold=True)`). Development
founders 0 and 1 on the GPU, the round 4 arm otherwise unchanged (1,285,724 parameters).
Characters / the letter from the sound:

| Arm | Pass 1 | Pass 2 | Pass 3 |
| --- | --- | --- | --- |
| The forced path, round 4 (twelve founders) | .306 / .307 | .343 / .342 | .349 / .352 |
| The soft path, lesson rates as composed (one founder at pass 3) | .360, .372 / .393, .398 | .340, .385 / .402, .408 | .394 / .397 |
| The soft path, rates at 0.3 | .347, .375 / .391, .398 | .343, .375 / .410, .423 | .414, .410 / .426, .423 |
| The soft path, rates at 0.1 | .223, .318 / .382, .382 | .268, .335 / .415, .410 | .394, .389 / .430, .417 |

- **The soft path lifts the start.** At pass 1 the letter from the sound reads .39 to .40
  against .31 under the forced path, and at pass 3 .42 to .43 against .35.
- **Beliefs read several batches earlier teach blank.** Read every eighth batch, the brain says
  only blank after half a pass, where read before every batch it reads .27 to .33 characters.
- **The rates.** At 0.3 the letter reads highest at pass 3. Emission given the letter swings
  between readings half a pass apart at every rate (.59 to .86 as composed, .59 to .88 at 0.3).
  Rates falling from 0.3 to .04 or .01 over four passes read .405 and .379 at pass 4 (one
  founder).
- **Where the soft brain's letters are** (founder 0, pass 3, read link by link as in Step 0):
  its pools .395 through a linear readout and .450 through a hidden layer of 256; its
  association .421 through a linear readout after one epoch and .440 at most; its motor neurons
  .419; the brain itself .414. The association holds more than the brain says.
- **The heard history is the association's route to the sound.** Without it the brain reads
  characters .03 to .10 at pass 1.5; with its learned gain held at the founder's 1, .30 and .34
  at pass 1 against .35 and .38. The gain climbs to its bound (7.9 of 8), where the sensory
  neurons, hearing the pools at unit length per row, sit at the floor of their activation (mean
  .009). The pools heard at 4, 8 or 16 times their scale in the sensory drive read the letter as
  without (.413 to .425 at pass 1.5 against .415 and .419) and say less of it; every synapse at
  the composed step (no step at 64 over the association's width) reads within .01; the history
  heard at twice its amplitude reads lower at pass 1 (.33 and .35).

### Round 5, declared: the teacher

Declared on [issue 304](https://github.com/muellerberndt/cadence/issues/304) before its first
life: [`protocol-speech-r5.json`](protocol-speech-r5.json). The round 4 arm, its lessons the soft
path through its own beliefs read before every batch, its lesson rates at 0.3 of the composed;
twelve fresh founders (304600 to 304611), six passes, read after every pass, beside CTC controls
at its parameter count (online six passes, offline to the stopping rule or 12 epochs; on the
GPU). G1 and G2 as in round 4, at pass 6 against each founder's offline CTC transformer at its
epoch 6.

## What this does and does not establish

The chamber measures composed brains on one synthetic world with supplied sound timing and one
sound per letter; Step 3 reads speech at one size and exposure, where no learner forms words.
