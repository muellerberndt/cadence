# Letters, Step 1: a letter said once, decided apart from the letter heard

Declared on [#304](https://github.com/muellerberndt/cadence/issues/304) before the first life.
Settings: [`protocol-step1.json`](protocol-step1.json). Instrument: [`letters.py`](letters.py),
[`controls.py`](controls.py).

## Why this arm

Step 0 (development founders 0 to 5) located the loss of words in the letters world on the
emission: the 0.97.0 default taught a letter at its sound's first frame misses .006 of letters
after a letter pair the trained words hold and .286 after a pair they do not hold, while its
settled state holds the letter at .99. Taught the letter on every frame of its sound, the same
brain names the letter at .997 and says it on 2.5 steps. Development arms (founders 0 to 5, 6,000
words): two motor slots, the symbol heard and whether to say it, read held-out words .75 against
.41 and miss .091 of letters after an unseen pair against .389; the efference copy and a copy that
holds the last letter said read as the default.

## Arms

| Arm | Brain | Lessons | Parameters |
| --- | --- | --- | --- |
| `split` | `Brain.compose(16, 20, slots=[18, 2])`, the 0.97.0 default composition | first slot: the symbol heard on every frame of its sound; second slot: 1 at the sound's first frame (space at the first silent frame after a word), 0 elsewhere | 21,331 |
| `default` | `Brain.compose(16, 18)` | the symbol at its sound's first frame, blank elsewhere | 21,169 |

The world pays nothing (the reward learner's rates are zero). Every frame is one
`Brain.step(frame, teacher=...)` with the default's surprise lessons; no backward pass. The split
brain says the first slot's symbol where the second slot says 1 and blank elsewhere.

## Founders, exposure and readings

Six fresh founders, 304100 to 304105. Each life hears 18,000 training words (about 218,000
frames); a saved copy answers 300 held-out words (all 40 words of the lexicon) greedily every
3,000 words. Readings as in the chamber: the stream as said with nothing merged, the CTC reading,
held-out and trained words, each letter's said steps, the stages, and the misses after a letter
pair the trained words do not hold.

## Controls

At 21,331 parameters (the split brain's), on the same streams: an LSTM and a causal transformer
trained by CTC, an LSTM trained on the default's per-frame targets and an LSTM with two heads
trained on the split brain's two targets; one pass at the brain's exposure (online) and shuffled
epochs to the stopping rule (offline). Seconds and multiply-accumulates beside each.

## Gates

- **Emission:** `split`, read on the stream as said (nothing merged): inserted characters under
  .02 per reference character and characters at .95, on 5 of 6 founders.
- **Words:** `split`, held-out words said whole at .90 on 5 of 6 founders, beside both CTC
  baselines.
- Beside the gates, not a gate: `split` against `default` paired by founder.

A failed gate is reported with its term located: the letter from the sound, its emission, or the
word from the letters. Read speech is a later step under its own protocol.
