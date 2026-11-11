# Speech: a recogniser from recordings and their transcripts

This guide trains a brain to write what it hears. You give it recorded speech and the text of
each recording; it learns to say one letter (or a space, or nothing) every 40 ms, and the
letters it says are read back as text. Afterwards it learns a new speaker from a few corrected
recordings and runs on live audio as the samples arrive.

The pieces are in `cadence.speech`, an opt-in package: a program that does not import it gets a
library without it.

| Step | Piece | Section |
| --- | --- | --- |
| Turn audio into what a brain hears | `frontend.log_mel`, `frontend.Hearing` | [hearing](#hearing-audio-as-an-image) |
| Compose the brain | `Brain.compose(..., learned_fields=64, fields_motor=1.0, ...)` | [the composition](#the-composition) |
| Teach it from transcripts | `align.Alignments`, `align.log_beliefs` | [training from transcripts](#training-from-transcripts) |
| Read text from it | `decode.collapse`, `decode.WordBeam` | [reading text](#reading-text-with-and-without-a-word-list) |
| Adapt it to a speaker | `correct.Rehearsal` | [adapting to a speaker](#adapting-to-a-speaker) |
| Run it on live audio | `stream.StreamReader` | [streaming](#streaming) |

The whole recipe is one script, `benchmarks/letters/recipe.py` ([the recipe](#the-recipe-script)).

```python
import numpy as np
from cadence import Brain
from cadence.speech.frontend import Hearing, log_mel
from cadence.speech.decode import Alphabet, WordBeam, collapse
from cadence.speech.align import Alignments, log_beliefs
from cadence.speech.correct import Rehearsal
from cadence.speech.stream import StreamReader

rng = np.random.default_rng(0)
second = rng.standard_normal(16000) * np.abs(np.sin(np.linspace(0, 20, 16000)))  # 1 s of "audio"
```

## Hearing: audio as an image

Audio at 16,000 samples a second becomes log-mel energies: 40 frequency bands, one frame every
10 ms (`log_mel`). The brain hears in steps of 40 ms. At each step it reads the latest 24 frames
(240 ms) as a small image of 40 rows (the bands) by 24 columns (the frames). Each window is
centred on its own mean, standardized per band with statistics from the training recordings,
clipped and scaled (`Hearing`).

```python
frames = log_mel(second)                    # (98, 40): 10 ms frames, 40 bands
hearing = Hearing.fit([frames])             # per-band statistics; fit on your training audio
steps = hearing.steps(frames)               # (24, 960): one row per 40 ms step
assert frames.shape == (98, 40) and steps.shape == (24, 40 * 24)
assert hearing.shape == (40, 24)            # the image a brain reads
```

`Hearing.to_dict()` and `Hearing.from_dict()` save and restore the statistics with the brain.

## The composition

```python
alphabet = Alphabet()                       # blank, a-z, the apostrophe, space: 29 symbols
ears = Brain.compose(hearing.shape, alphabet.size, seed=0,
                     learned_fields=64, modules=(256,), fields_motor=1.0,
                     learning_surprise=False,
                     actor_eta=0.0, actor_eta_bias=0.0, actor_eta_critic=0.0)
assert ears.parameters() == 1_285_724
```

Each setting is the one the letters chamber (`benchmarks/letters`,
[issue 304](https://github.com/muellerberndt/cadence/issues/304)) measured best on read speech
(LibriSpeech, the 2,100 utterances of 2.5 hours whose every word is among the 2,000 most common):

- **`learned_fields=64`** reads the image through receptive fields a local rule shapes
  ([learned receptive fields](senses.md#learned-receptive-fields)). It doubles the letter the
  brain reads from the sound, from .18 to .35.
- **`fields_motor=1.0`** adds a plastic pathway from the fields straight to the motor neurons,
  beside the association's. With **`learning_surprise=False`** (every row of a lesson taught,
  where surprise lessons teach only the rows the brain answers wrong) the letter the brain says,
  given the letter it holds, rises from .42 to .78, and characters from .26 to .33.
- **`modules=(256,)`**: an association of 256 neurons. On spoken words (issue 303) the
  association of 64 holds the word at .431 of what its fields hold at .626; at 256 it holds .575.
- The actor's rates at zero: the brain learns from lessons alone.

**Utterances side by side.** Training hears a batch of utterances at once, each a stream of the
same brain: row `r` of every step is utterance `r`. Each batch starts from rest
(`brain.reset()`). An utterance shorter than the batch's longest holds its last step after its
end and is taught blank there. The chamber trains 32 utterances side by side.

## Training from transcripts

A transcript says which letters an utterance holds, not when. Each step's lesson needs a
letter, so a teacher chooses one: the best path of the transcript through the brain's own
beliefs at each step, blank allowed on any step and required between a letter and the same
letter after it (the forced alignment of CTC, `align.forced_path`). Before the brain believes
anything, the steps are split evenly among the letters (`Alignments.even`). The path is the
teacher's, computed outside the brain; the brain learns from it by its own rule.

```python
utterances = [hearing.steps(log_mel(second)), hearing.steps(log_mel(second[:12000]))]
texts = [alphabet.encode("go on"), alphabet.encode("no")]
paths = Alignments()
for k, (rows, text) in enumerate(zip(utterances, texts)):
    paths.even(k, len(rows), text)                  # the first batches: an even split
x = np.stack([np.concatenate([u, np.repeat(u[-1:], 24 - len(u), axis=0)]) for u in utterances],
             axis=1)                                 # (steps, utterances, inputs)
labels = paths.lesson([0, 1], len(x))               # (steps, utterances): blank past an end
ears.reset()
for t in range(len(x)):
    ears.step(x[t], teacher=labels[t])              # one lesson and one act per step
```

Every few batches (8 in the chamber) the teacher saves the brain, loads a frozen copy, reads its
beliefs on the next batches' utterances (`log_beliefs(copy, rows)`) and aligns each transcript
through them (`Alignments.realign(key, beliefs, text)`). A key names one hearing of an
utterance, so an utterance heard in two passes holds a path for each.

## Reading text, with and without a word list

The brain's beliefs at each step are the softmax of its motor neurons. Reading the most believed
symbol at each step and collapsing repeats and blanks gives the text (`collapse`, the reading of
CTC). A word list reads the same beliefs through a beam over a trie of the list, so every word
written is a list word (`WordBeam`).

```python
said = [0, 7, 7, 0, 15, 28, 28, 15, 14, 0]          # blank g g blank o space space o n blank
assert alphabet.decode(collapse(said)) == "go on"
beliefs = np.log(np.full((len(said), alphabet.size), 0.01))
beliefs[np.arange(len(said)), said] = np.log(0.72)  # a confident brain
listed = WordBeam(["go", "on", "no"])
assert listed.decode(beliefs) == "go on"
```

On the 2.5-hour set the brain reads .33 characters after two passes, level with a CTC transformer
of its size at two passes; trained offline to their stopping rule, CTC controls read .44. At that
letter level no learner forms words, and a word list has little to choose from (issue 304).

## Adapting to a speaker

A correction teaches the brain one recording it got wrong, labelled by the forced alignment of
its transcript. Taught alone it pulls the brain toward the new speaker and away from the
speakers it knew. `Rehearsal` hears each correction as one stream beside three earlier
utterances, taught at a quarter of the correction's weight (`Brain.lesson(weights=)`), and
keeps the earlier utterances in a bank under a byte budget.

```python
bank = Rehearsal(rows=3, weight=0.25, budget=64 * 2**20)
for rows, text in zip(utterances, texts):
    bank.remember(rows, Alignments().even("u", len(rows), text))   # earlier rows, labelled
correction = utterances[0]
report = bank.correct(ears, correction, labels[: len(correction), 0])
assert report["streams"] == 3 and report["refused"] == 0           # the correction and two banked
```

`Rehearsal` was read on the spoken-phrase chamber (`benchmarks/spoken`, issue 303, mode
`rehearsal`): six lives of the chamber's declared run, each corrected on 18 new speakers and 18
speakers of a dialect for two minutes of their speech, with its bank holding 256 of the lives'
training phrases.

| Correction (two minutes) | New speakers: gain / earlier speakers | Dialect: gain / earlier speakers |
| --- | --- | --- |
| alone | +.080 / -.018 | +.096 / -.056 |
| `Rehearsal(rows=3, weight=0.25)` | +.072 / +.005 | +.080 / -.002 |
| beside three earlier phrases at full weight | +.062 / +.012 | +.062 / +.012 |

Correct a brain with the lessons it learned from. These lives learned soft targets; corrected with
plain labels instead, `Rehearsal` gains +.027 and +.049 and the earlier speakers lose .021 and
.030. `correct(brain, rows, targets=...)` and `remember(rows, targets=...)` take soft targets; a
recogniser trained by the forced alignment, as in this guide, is corrected by labels.

## Streaming

`StreamReader` runs one stream of a trained brain on live audio. `push(samples)` takes the next
samples in blocks of any size; each completed 40 ms step is heard and answered, and `push`
returns the text it adds. `finish()` returns the transcript, read against a word list where one
is given. The reader hears what `Hearing.steps(log_mel(audio))` makes of the whole recording.

```python
ears.fields.learning = False                        # read without moving the fields
reader = StreamReader(ears, hearing, words=listed)
text = "".join(reader.push(second[k : k + 1600]) for k in range(0, 16000, 1600))  # 100 ms blocks
assert len(reader.actions) == 24 and isinstance(reader.finish(), str)
assert reader.latency()["step_median_s"] > 0
```

`latency()` reports the seconds each step took on the machine it ran on: the median, the 95th
percentile and the seconds per second of audio. On one idle core of an AMD EPYC 9R14 (AWS c7a)
the recipe's composition answers a 40 ms step in 1.05 ms at the median (1.10 ms at the 95th
percentile): 0.026 seconds per second of audio. The beam over the 2,000-word list reads 30 seconds
of beliefs in 0.08 seconds (`recipe.py --timing`).

## The recipe script

`benchmarks/letters/recipe.py` runs every step above on LibriSpeech (CC BY 4.0, from
https://www.openslr.org/12; it reads the `train-clean-100` and `dev-clean` folders with
`soundfile`):

```sh
PYTHONPATH=src python benchmarks/letters/recipe.py --librispeech /data/libri \
    --words benchmarks/letters/words-2000.txt --utterances 2100 --passes 2
```

It prints, after every pass, the characters and words on held-out speakers (with the word list
read beside the plain collapse). Then it corrects a new speaker (the `dev-clean` speaker with the
most recordings, kept apart from the held-out ones) on half of their recordings and reads their
other half and the held-out speakers before and after. Last, it hears one held-out recording
100 ms at a time and prints the seconds per step. `--synthetic` runs it on tone words with no
download, as its test does.

On LibriSpeech's `train-clean-100`, whose 468 utterances with every word among the 2,000 most
common hold 0.55 hours, two passes read characters .21 on 53 held-out `dev-clean` recordings
(the brain says only blanks after the first pass). Eleven corrections of a new speaker raise that
speaker's other recordings from .20 to .27 and the held-out speakers from .21 to .27: at this
level each correction also teaches the language. Training takes about 1,150 seconds per hour of
audio on one Intel Xeon core (AWS c7i) with 32 utterances side by side.

## Settings, refusals and cost

| Piece | Settings | Refused with `ValueError` |
| --- | --- | --- |
| `log_mel` | `rate` 16,000, `mels` 40, `frame` 400, `hop` 160, `fft` 512 | audio not finite and one-dimensional |
| `Hearing` | `window` 24 frames, `stack` 4 frames, `clip` 3, `scale` .3 | statistics not finite or not positive |
| `Alignments`, `forced_path` | `blank` 0 | a transcript longer than its steps allow |
| `WordBeam` | `beam` 16, `top` 8, `counts`, `prior_weight`, `word_bonus` | a list word outside the alphabet |
| `Rehearsal` | `rows` 3, `weight` .25, `budget` 64 MiB, `seed` | negative rows or weight, a budget of no bytes, one utterance larger than the budget |
| `StreamReader` | `alphabet`, `words`, `temperature` | a brain whose motor neurons do not match the alphabet |

A correction resets the brain and leaves it at rest. At weight zero the banked utterances teach
nothing at the normalized teaching points a composed brain takes. Under the founder teaching
(`recall=False` without a teaching point) a weight-zero row still shrinks the step.
