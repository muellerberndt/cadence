# Streaming spoken-word chamber

This bounded instrument asks whether one continuing `Brain.compose` life, hearing a stream one
frame per moment, names each word at its last frame, and how it compares with the same brain
given a window of frames pressed into one observation. Naming a word from a one-second window
is classification; transcribing a stream needs history carried from frame to frame, the same
carried history that playing music needs
([issue 200](https://github.com/muellerberndt/cadence/issues/200), part of
[issue 121](https://github.com/muellerberndt/cadence/issues/121)). The synthetic world comes
first; real audio is a later step, planned below and declared separately before it runs.
Read [continuous interaction](../../docs/continuous.md) and
[lessons only where the answer differs](../../docs/learning.md#lessons-only-where-the-answer-differs)
before interpreting results.

## The world

A lexicon of twelve words over sixteen phones. A phone is a fixed vector of sixteen values in
[0, 1]; silence is zero; every frame adds Gaussian noise (standard deviation 0.1). Words come in
six pairs whose two words share every phone but the last, so a word is ambiguous until its last
frame. Every word's penultimate phone is the same joint phone, and each final phone ends two
words of neighbouring pairs, whose last prefix phones differ: the last two phones leave two
candidates, and the phone before them decides. A prefix has two or three phones. Every phone
before the final one lasts one frame, or two with probability 0.5; the final phone lasts one
frame, so the end of a word is a frame its history can identify. Words are 4 to 9 frames long,
drawn uniformly, each after one to three silent frames.

Each frame is one observation. Its label is blank on every frame except a word's last, which
carries the word: one output slot of thirteen choices. A seed fixes the lexicon, the training
stream (6,000 words, about 50,000 frames), a held-out evaluation stream of 300 words with the
same lexicon and fresh order, durations and noise, the founder brain and the learners' initial
weights.

## Arms and controls

| Arm | Life |
| --- | --- |
| `surprise` | the composed founder (`Brain.compose(16, 13)`, its working trace and episodic store), every frame through `Brain.step(frame, teacher=label)` with `learning_surprise=True`: a lesson where the answer its act reads differs from the label |
| `every_frame` | the same with a lesson on every frame, the paired control |
| `window` | the same composition and rule given the last nine frames (the longest word) concatenated into one observation of 144 values: the pressed-in window, as a control |
| `gru` | a supervised online gated recurrent unit, the cell of [`benchmarks/competitors.py`](../competitors.py), hidden 32, with a softmax readout; at every frame the cross-entropy of that frame through the last 16 frames (truncated backpropagation), its norm clipped at 5, one Adam step at rate 0.003 |
| `majority` | blank on every frame |
| `uniform` | a uniform choice among the thirteen on every frame |
| `phone` | the most frequent label of each noise-free phone on the evaluation stream itself: a bound for any answer read from the current frame alone |

The world pays nothing. The brain arms set the reward learner's rates to zero, so the
zero-reward feedback that `step` applies to its previous sampled act changes nothing; the
remaining genes are the composed founder's. The GRU's settings were chosen before any run,
none selected on this chamber's seeds. [`protocol-history.json`](protocol-history.json), an
amendment declared before its run, adds the arm `history`: the `surprise` arm composed with the
opt-in heard history (`history_decays` 0.2, 0.8 and 0.95, amplitude 3, scale 1.2, the settings of
the option's first declared protocol), a 48-neuron region of the frames as heard at three decays
read by the association region. Run it with `--protocol
benchmarks/stream_words/protocol-history.json --arm history`.

## Readings and gate

Every 1,000 training words a saved copy is loaded, reset and answers the evaluation stream
greedily without learning; the GRU answers from a zero hidden state. Word accuracy is the share
of word ends answered with that word, the recall of emitted words. Emissions elsewhere are frames
that end no word answered with a word, per word said. Precision is the share of emissions that
name a word at its end. The confusions at word ends separate the pair partner (same prefix), the
rhyme partner (same last two phones), another word and blank. The receipts record the lessons,
the sweeps of the acts and lessons, and for the GRU its updates and an analytic count of
multiply-accumulates.

The gate, declared in [`protocol.json`](protocol.json) before the first run, reads the probe
after the 6,000th word on the means over five seeds: the `surprise` arm's word accuracy is at or
above the `window` arm's and above 0.5 (two words end in every final phone, so an answer read
from the final phone alone is right at most half the time), with at most 0.2 emissions elsewhere
per word. The GRU and the `every_frame` arm stand beside it.

```sh
PYTHONPATH=src python benchmarks/stream_words/stream_words.py --arm surprise --seed 0 --out <dir>
PYTHONPATH=src python benchmarks/stream_words/stream_words.py --arm gru --seed 0 --out <dir> --smoke
PYTHONPATH=src python benchmarks/stream_words/stream_words.py --report <dir>
PYTHONPATH=src python benchmarks/stream_words/stream_words.py --verify <dir>/surprise-0.json.gz --current
python -m pytest -q benchmarks/stream_words
```

One life runs per process and writes a receipt that binds the chamber, the protocol, the
competitor module and every library source. Results are kept outside this repository.

## What this does and does not establish

The chamber measures one composition on one synthetic lexicon with supplied frame timing and
supplied word-end labels. A pass shows that the composed brain's carried history names words in
a stream at the level of a pressed-in window on this lexicon; it shows nothing about real
speech, larger vocabularies, speakers, or phrases. A failure names the arm and probe where the
history did not carry.

## Plan for real audio (Speech Commands)

The next step uses the Speech Commands corpus (version 0.02: 105,829 one-second clips of 35
words at 16 kHz from 2,618 speakers, CC BY 4.0, about 2.3 GB). Nothing is downloaded until that
step's protocol is declared. The plan:

- **Frames.** Sixteen log-mel bands per 25 ms frame at a 20 ms hop (50 frames per second),
  normalized per band on the training speakers; a frame is one observation, sixteen values, as
  here.
- **Words and streams.** Twelve words in six confusable pairs from the corpus: shared
  beginnings ("on" and "off", "bed" and "bird", "down" and "dog", "four" and "forward") and
  shared endings ("tree" and "three", "no" and "go"). A stream
  concatenates clips trimmed to their voiced frames (an energy threshold on the training
  speakers), separated by 2 to 6 frames of the corpus's background noise at a declared level.
- **Labels.** The word at the last voiced frame of each clip, blank elsewhere; a declared
  tolerance of two frames around that frame for scoring, since a voiced end is not sharp.
- **Speakers.** The corpus's validation and test lists hold out speakers; the training stream
  draws only training speakers.
- **Arms and gate.** The same arms: the composed brain with lessons where its answer differs,
  the every-frame control, the window classifier with the window set to the longest trimmed
  clip in the training stream, the online GRU, and the three controls. The gate keeps this
  chamber's form, with its bound recomputed from the chosen pairs and declared before the run.
