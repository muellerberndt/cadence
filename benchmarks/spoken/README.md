# Spoken-phrase chamber: a new speaker and a dialect on a closed vocabulary

A person who hears a new speaker, or a dialect they have not met, understands them after a few
minutes and keeps understanding everyone else. This chamber reads that on a closed vocabulary:
one continuing brain learns phrases of 35 spoken words, then adapts within its life to a new
speaker and to a new dialect from minutes of their speech, against small networks trained and
then fine-tuned online on the same audio, with the earlier speakers' accuracy measured. It is the
instrument of [issue 303](https://github.com/muellerberndt/cadence/issues/303), the spoken-phrase
test of the comparisons epic ([issue 183](https://github.com/muellerberndt/cadence/issues/183)).

## The corpora

| Corpus | What it holds | Licence | Role |
| --- | --- | --- | --- |
| Speech Commands v0.02 (Warden 2018) | 105,829 one-second clips of 35 words by 2,618 speakers, an official speaker-disjoint split, six background recordings | CC BY 4.0 | the base vocabulary; new speakers from its test split; every phrase's noise bed |
| AudioMNIST (Becker and colleagues 2018) | 30,000 recordings of the digits zero to nine by 60 speakers at 48 kHz, each speaker's reported accent | MIT | the dialect: speakers who report a German accent (41), ten words shared with Speech Commands |
| Common Voice single-word target segment (Mozilla) | the digits, `yes` and `no`, each speaker's reported accent | CC0 | an adapter (`corpora.CommonVoiceSingleWord`); its download asks for an account |
| synthetic speech (`corpora.Synthetic`) | each word three voiced tones whose pitch and formants a speaker shifts | | the tests, which need no download |

The corpora live outside the repository. `phrases.py build` records each download's checksum,
version and counts in the built set's `manifest.json`.

## The task

A phrase is two to four word clips of one speaker over a bed of background noise (`phrases.py`):
each clip keeps a quarter second around its extent (its frames above a tenth of its peak energy),
the first word starts 0.25 to 0.45 s into the stream, 0.5 to 0.85 s of silence separate two
words, and the bed lies at least 3 dB below the weakest word. This is the construction of the
second run of issue 183.

The brain hears the stream in steps of 0.49 s: at step `k` the latest second, 98 log-mel frames
of 40 bands, as an image of 40 rows (bands) by 98 columns (frames), centred on its own mean,
standardized per band, clipped at three standard deviations and scaled by 0.3
(`chamber.Hearing`). It acts one of 37 actions per step: a word, silence or `end`. A word is due
at the first step whose window holds its last frame, `end` two steps after the last word and at
the step that closes the stream, silence elsewhere. A phrase's transcript is its words up to the
first `end`; word accuracy is one minus the word error rate.

| Set | Speakers | Phrases |
| --- | --- | --- |
| `base_train` | Speech Commands training speakers | 24,000 of 2 and 3 words |
| `base_validation`, `base_test` | its validation and test speakers | 1,000 each of 2, 3 and 4 words (4 never taught) |
| `speaker/<id>/adapt`, `/evaluate` | 24 test speakers with the most clips, left out of `base_test` | 90 labelled phrases (5.0 to 5.5 minutes) and 16 phrases of that speaker's other recordings |
| `dialect/<id>/adapt`, `/evaluate` | AudioMNIST's 41 German-accent speakers | the same, of digits |

Of each word's clips of a new speaker, two in three go to adaptation and one to evaluation, so
no recording is on both sides; a Speech Commands speaker's adaptation phrases deal their clips
again once they run out (74 to 173 distinct recordings over the 90 phrases).

## Learners

| Arm | What it is |
| --- | --- |
| `fields` | `Brain.compose(inputs=(40, 98), actions=37, learned_fields=64)`: the 0.97.0 default (the recall memory, the heard moment before, the store) reading the window through [learned receptive fields](../../docs/senses.md#learned-receptive-fields) |
| `fields_no_history`, `fields_no_recall` | the same with `history_decays=()`, or `recall=False` |
| `fields_drawn` | the same with its fields held at their draw from birth |
| `keyword` | `Brain.compose(3920, 37, recall=False)`: the window as one vector, the front end of issue 183's runs |
| `transformer_ctc`, `transformer_step` | two convolutions over the window, a causal transformer over the phrase's steps (two layers, four heads), trained with CTC over the words and `end`, or with the brain's per-step targets (`controls.py`) |
| `lstm_ctc`, `lstm_step` | the same front end and one LSTM layer |

A baseline's width matches its parameter count to the `fields` brain's. Every learner of a
founder hears the same batches of 32 phrases of equal steps in the same order; a brain takes a
lesson on the step's targets before each act, a baseline one Adam step per batch (online, one
pass at 24,000 phrases).

**Adaptation** (`chamber.adapt`): from a copy of the life, a new speaker's adaptation phrases one
at a time, in order, read at 0, 1, 2 and 5 minutes of their speech on the speaker's evaluation
phrases and on 300 phrases of `base_test` (retention). `plain` teaches the phrase alone;
`replay` teaches it beside three `base_train` phrases of equal steps. A rate scales every rate
of the lesson's learner (a brain) or Adam's step (a baseline). Readings run on a copy whose fields
take no step, so a reading never changes a life.

```sh
PYTHONPATH=src python benchmarks/spoken/phrases.py build \
    --speech-commands /data/speech_commands_v0.02 --audiomnist /data/AudioMNIST --out /data/phrases
PYTHONPATH=src python benchmarks/spoken/spoken.py --data /data/phrases \
    --learners fields transformer_ctc --founders 0 1 --budgets 6000 24000 \
    --speakers speaker/<id> dialect/<id> --probe --out results/life.jsonl --workers 4
python benchmarks/spoken/summary.py results/*.jsonl
PYTHONPATH=src python benchmarks/spoken/timing.py --data /data/phrases --out results/timing.jsonl
```

`--probe` adds ridge readouts at the word steps of what the window, the pools (the fields'
output the cortex reads), the pools of fields at their draw and the association hold: readings
that locate a failure, never part of a life.

## Readings

### Step 0: where the 0.97.0 default with `learned_fields` fails (development founders 0 and 1)

24,000 phrases (one pass); word accuracy on 300 test phrases of held-out speakers.

| Arm | Parameters | Word accuracy (founders 0, 1) | Exact | At word steps: right / other word / silence |
| --- | --- | --- | --- | --- |
| `transformer_ctc`, matched to `fields_wide` | 1,281,498 | .743 (.728, .759) | .482 | |
| `transformer_ctc`, matched to `fields` | 640,638 | .731 (.702, .759) | .465 | |
| `transformer_step`, matched to `fields` | 640,549 | .654 (.620, .688) | .373 | .644 / .258 / .097 |
| `fields_wide` (`modules=(256,)`) | 1,275,684 | .425 (.409, .441) | .137 | .421 / .319 / .259 |
| `keyword` (founder teaching) | 406,534 | .368 (.353, .382) | .093 | .357 / .456 / .185 |
| `fields_normalized` (`recall=False`, `teaching="normalized"`) | 69,606 | .333 (.306, .361) | .058 | .326 / .437 / .235 |
| `transformer_ctc`, matched to `fields_normalized` | 69,606 | .568 (.528, .607) | | |
| `fields`, the 0.97.0 default | 629,988 | .286 (.281, .291) | .048 | .272 / .278 / .449 |
| `fields_no_history` | 592,547 | .284 (.268, .300) | .050 | .266 / .282 / .451 |
| `fields_no_recall` | 107,048 | .258 (.246, .271) | .040 | .251 / .269 / .478 |
| `keyword_taught` | 661,336 | .253 (.243, .262) | .038 | .243 / .414 / .342 |
| `lstm_step`, `lstm_ctc` | 617,109, 617,214 | .218, .212 | .015, .027 | |
| `fields_drawn` | 629,988 | .129 (.112, .146) | .005 | .111 / .391 / .493 |
| `fields_founder` (`recall=False`, founder teaching) | 69,606 | .056 (.052, .059) | .000 | .027 / .311 / .651 |

What holds the word at a word step (ridge readouts fitted on 3,000 training word steps; test
speakers / the two development speakers):

| Arm | Parameters | The window | The pools | Pools, fields at their draw | Association, from rest | The brain says it |
| --- | --- | --- | --- | --- | --- | --- |
| `fields` | 629,988 | .309 / .375 | .626 / .581 | .553 / .556 | .431 / .394 | .272 |
| `fields_wide` | 1,275,684 | .309 / .375 | .626 / .581 | .553 / .556 | .575 / .544 | .421 |

- The fields hold the words: their pools read .626, above the same fields at their draw, and the
  brain with its fields held at their draw reads .129. Whether a word is due at this step or the
  next reads .85 from the pools, as from the window; 6 or 12 columns of pools add nothing.
- The brain's read of its pools trails: the association of 64 holds .431 and the brain says the
  right word at .272 of the word steps and silence at .449. An association of 256 holds .575 and
  says .421, silence .259; 512 reads .349 and .424 overall, no higher than 256.
- In development, `learning_surprise_share` (.26 to .28 at 64, .32 to .34 at 256), a word held due
  over the step after it (`--hold 2`: .102 and .154 at 12,000 phrases against .160 and .218) and
  more exposure for the default (.276 and .396 at 48,000 phrases, where the transformer with CTC
  reads .796 and .821) leave the gap.

Adaptation on one development speaker and one dialect speaker (40 words per reading, a standard
error near .07), at a tenth of the life's rates:

| Learner | Parameters | Mode | New speaker at 0, 2 and 5 minutes | Retention change at 2 | Dialect speaker at 0, 2 and 5 minutes | Retention change at 2 |
| --- | --- | --- | --- | --- | --- | --- |
| `fields` | 629,988 | plain | .375, .538, .475 | +.043 | .450, .500, .525 | +.040 |
| `fields` | 629,988 | replay | .375, .487, .450 | +.066 | .450, .500, .500 | +.055 |
| `fields_wide` | 1,275,684 | plain | .488, .550, .475 | -.008 | .575, .613, .650 | -.014 |
| `transformer_ctc` | 640,638 | plain | .562, .725, .850 | -.067 | .725, .875, .988 | -.033 |
| `transformer_ctc` | 640,638 | replay | .562, .637, .750 | +.030 | .725, .812, .875 | +.042 |
| `transformer_ctc` | 1,281,498 | plain | .613, .738, .887 | -.073 | .713, .863, .963 | -.052 |

At the life's full rate every learner's retention falls, the transformers' by .51 to .62.

### The declared run (cadence#303: fresh founders 3030 to 3035, 18 new speakers, 18 dialect speakers)

The [protocol](https://github.com/muellerberndt/cadence/issues/303#issuecomment-6786230453) put
the brain at `modules=(256,)` under test, the width Step 0 located, against the transformers at
its parameter count; adaptation at a tenth of the life's rates. No gate reads for Cadence.

| Learner | Parameters | Word accuracy, mean (range) | Exact | At word steps: right / other word / silence | Host seconds of training |
| --- | --- | --- | --- | --- | --- |
| `fields_wide` | 1,275,684 | .357 (.268 to .426) | .097 | .344 / .269 / .387 | 1,287 |
| `fields` | 629,988 | .265 (.181 to .286) | .044 | .252 / .317 / .429 | 1,119 |
| `transformer_ctc` | 1,281,498 | .739 (.716 to .768) | .491 | | 104 |
| `transformer_step` | 1,281,349 | .658 (.644 to .673) | .376 | .649 / .241 / .110 | 95 |

| Learner | Parameters | Mode | New speakers at 0, 1, 2, 5 min | Retention at 0, 2, 5 | Dialect at 0, 1, 2, 5 min | Retention at 0, 2, 5 |
| --- | --- | --- | --- | --- | --- | --- |
| `fields_wide` | 1,275,684 | plain | .363, .464, .495, .519 | .354, .422, .399 | .502, .570, .611, .663 | .354, .398, .369 |
| `fields_wide` | 1,275,684 | replay | .363, .459, .469, .484 | .354, .436, .443 | .502, .575, .597, .631 | .354, .436, .439 |
| `transformer_ctc` | 1,281,498 | plain | .717, .814, .858, .890 | .749, .688, .674 | .727, .905, .928, .960 | .749, .681, .656 |
| `transformer_ctc` | 1,281,498 | replay | .717, .783, .820, .863 | .749, .757, .762 | .727, .874, .926, .953 | .749, .755, .763 |
| `transformer_step` | 1,281,349 | plain | .644, .741, .784, .825 | .674, .609, .593 | .654, .837, .864, .909 | .674, .591, .569 |
| `transformer_step` | 1,281,349 | replay | .644, .708, .750, .784 | .674, .681, .683 | .654, .811, .856, .894 | .674, .680, .686 |

| Gate | Reading | Founders |
| --- | --- | --- |
| G1, the base life within .05 of the transformer with CTC | -.290 to -.459 | 0 of 6 |
| G2, a new speaker at 2 minutes, accuracy and retention against the fine-tuned transformer with CTC | accuracy -.304 to -.397, retention -.211 to -.301 | 0 of 6 |
| G3, the dialect, the same | | 0 of 6 |

- The gain itself, on new speakers, is level: the brain's gain over two minutes minus the
  transformer with CTC's reads -.009 (5 of 6 founders not below zero by two standard errors), and
  its retention change is higher by +.128 (each founder beyond two standard errors). On the
  dialect the gain trails by .093 (2 of 6), with retention change higher by +.112.
- The term is the base life's: the pools hold the word at .627 for a linear readout, the
  association of 256 holds .574, and the brain says the right word at .344 of the word steps and
  silence at .387.

Seconds per step and per correction, one stream on one Intel Xeon 8488C core (`timing.py`):

| Learner | Parameters | Step of 0.49 s, median | Times real time | A correction, median |
| --- | --- | --- | --- | --- |
| `fields_wide` | 1,275,684 | 2.43 ms | 201 | 0.132 s |
| `fields` | 629,988 | 2.13 ms | 230 | 0.043 s |
| `transformer_ctc` | 1,281,498 | 1.12 ms | 438 | 0.0085 s |
| `transformer_step` | 1,281,349 | 1.10 ms | 447 | 0.0086 s |

### Round 2: the brain's read of its pools (development founders 0 and 1, then a declared run)

The bar: what the brain says at a word step against the pools' linear readout, the first declared
run's gap (.627 against .344) halved, .142 or less. `--silence-weight` weighs the rows taught
silence (`Brain.lesson(weights=)`); `--soft` teaches the step after a word step the word at that
share and silence at the rest (`Brain.lesson(targets=)`, every row taught); `{"direct": 1.0}`
composes `fields_motor=1.0`, a plastic pathway from the pools to the motor region;
`{"learning_surprise": false}` teaches every row its label.

| Arm (association of 256 unless named) | Parameters | Word accuracy (founders 0, 1) | At word steps: right / other / silence | Pools minus right |
| --- | --- | --- | --- | --- |
| Step 0's arm | 1,275,684 | .425 (.441, .409) | .421 / .319 / .259 | .205 |
| silence weight .25 | 1,275,684 | .426 (.454, .397) | .446 / .362 / .118 | .180 |
| silence weight .1 | 1,275,684 | .439 (.451, .428) | .539 / .397 / .053 | .086 |
| soft .5 | 1,275,684 | .514 (.518, .511) | .501 / .391 / .108 | .125 |
| every row taught | 1,275,684 | .524 (.536, .512) | .504 / .329 / .164 | .122 |
| `fields_motor=1.0` | 1,296,996 | .431 (.449, .413) | .423 / .273 / .303 | .202 |
| `fields_motor=1.0`, silence weight .25 | 1,296,996 | .477 (.484, .469) | .506 / .382 / .112 | .120 |
| `fields_motor=1.0`, every row taught | 1,296,996 | .551 (.556, .547) | .528 / .317 / .153 | .098 |
| `fields_motor=1.0`, soft .5 (declared) | 1,296,996 | .556 (.561, .551) | .537 / .358 / .104 | .088 |
| association of 512 | 2,251,300 | .387 (.349, .424) | .381 / .300 / .226 | .245 |
| association of 512, soft .5 | 2,251,300 | .553 (.549, .558) | .533 / .357 / .110 | .093 |

Weights against silence end the silence majority and turn deletions into wrong words; teaching
every row lifts the reading, soft targets add nothing beyond it, and the pathway adds .027 to .042.

The declared run (founders 3036 to 3041, the first run's speakers, `fields_motor=1.0` with soft
targets .5): no gate reads for Cadence.

| Learner | Parameters | Word accuracy, mean (range) | At word steps: right / other / silence | New speakers at 0, 2, 5 min (plain) | Retention at 0, 2 | Dialect at 0, 2, 5 min |
| --- | --- | --- | --- | --- | --- | --- |
| the arm | 1,296,996 | .539 (.502 to .552) | .510 / .322 / .168 | .514, .610, .617 | .544, .530 | .634, .739, .788 |
| `transformer_ctc` | 1,281,498 | .747 (.736 to .764) | | .732, .868, .898 | .751, .692 | .742, .933, .966 |
| `transformer_step` | 1,281,349 | .661 (.641 to .687) | .659 / .226 / .115 | .652, .793, .823 | .668, .604 | .616, .864, .910 |

G1 0 of 6 (-.183 to -.244), G2 and G3 0 of 6. The pools hold .621, the association .605, the brain
says the right word at .510 (the bar met). Paired over speakers, the arm's two-minute gain trails
the transformer with CTC's by .039 on new speakers and .085 on the dialect, with retention change
higher by .045 and .019. One stream on one c7i core: 2.37 ms per 0.49 s step, 4.84 ms per second of
audio; 0.249 s per correction.
