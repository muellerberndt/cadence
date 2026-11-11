# Letters, Step 2: a teacher's forced alignment on the brain's own beliefs

Declared on [#304](https://github.com/muellerberndt/cadence/issues/304) before the first life.
Settings: [`protocol-step2.json`](protocol-step2.json). Instrument: [`letters.py`](letters.py)
(`forced_path`, `forced_labels`), [`controls.py`](controls.py).

## Why this arm

Step 1 located the loss of words on the lesson's step: every learner taught when to say a letter
by a label at its sound's first frame reads held-out words at .40 to .67, and the learners that
choose their own timing by CTC at .85 to .94. Taught each letter on every frame of its sound and
read by the CTC reading, the 0.97.0 default reads held-out words at .99. A speech teacher knows
where each word lies and which letters it holds, and not where each letter's sound begins. This
step supplies the frame timing from the brain's own beliefs.

**Candidate (b), located (development founders 0 to 5, 6,000 words).** Taught a letter from its
sound's first frame until its act says it, the brain first says a letter at the sound's first
frame on .47 of training sounds, one frame later on .19, never on .26, the same from 1,000 words
on. A frame after the letter's first saying is taught blank where the sampled act said it and the
letter where it did not, so the same sound frame takes both lessons; the brain then says a letter
on two or more steps for .42 of sounds and in two separate runs for .12 (inserted letters .08 per
character, held-out words .28).

**Development readings of this arm** (same founders and exposure): even split held fixed reads
held-out words .06 (the split puts .74 of letter frames on the letter heard); re-aligned every
500 words the alignment puts .90 of letter frames right at the first pass and .974 by 5,500 words,
and held-out words rise from .12 to .81 (3 of 6 at .90), characters .975; re-aligned every 2,000
words, .33. The supplied alignment reads .99.

## Arms

| Arm | Lessons | Parameters |
| --- | --- | --- |
| `forced` | the 0.97.0 default, `Brain.compose(16, 18)`; the first 500 words on the even split of each word's frames among its letters; then every 500 words a frozen copy of the brain answers the next 500 words greedily and each word's frames go to its known letters by the monotonic best path through the copy's log beliefs, each letter at least one frame, in order; each letter taught on every frame of its path, space on every silent frame after the word | 21,169 |
| `frames` | each letter taught on every frame of its sound (the supplied alignment), the reference | 21,169 |

The path is computed by the teacher outside the brain, from the copy's settled motor activations,
and used only as the lessons' labels. Every frame is one `Brain.step(frame, teacher=...)`; no
backward pass in the brain. The world pays nothing.

## Founders, exposure, controls

Six fresh founders, 304200 to 304205, 18,000 training words each, a reading every 3,000 words
on 300 held-out words. Controls at 21,169 parameters on the same streams: an LSTM and a causal
transformer by CTC, one pass and offline to the stopping rule, with seconds and
multiply-accumulates.

## Gates

Read on the CTC reading (adjacent repeats once, blanks dropped), the reading the every-frame
lesson is built for. An every-frame lesson says a letter on each frame of its sound, so the
stream as said with nothing merged is reported beside the gates and not gated; "a letter said
once per sound" is not what this arm delivers.

- **Emission:** `forced`, inserted characters under .02 per reference character and characters at
  .95, on 5 of 6 founders.
- **Words:** `forced`, held-out words said whole at .90 on 5 of 6 founders, beside both CTC
  baselines.
- Beside the gates: `forced` against `frames` paired by founder, and the share of letter frames
  the alignment puts on the letter heard at each re-alignment.

If the gates hold, the read-speech reading follows under its own protocol.
