# Words for things (cadence#326)

A robot holds a map of its room and learns words from naming: while it attends to a thing it
hears one word. It then answers typed questions about rooms it has never seen, in both directions:
"where is the ball?", "what colour is the ball?", "what is red?", "what is behind you?", and "I do
not know" where it holds no answer. The brain is `cadence.words.Words` ([the guide](../../docs/words.md)).

| File | What it holds |
| --- | --- |
| `world.py` | Rooms of things (kind, colour, place), the robot's camera, the places around the robot, the nonce words drawn for each founder, the teacher's question forms and the scoring |
| `words.py` | A founder's life as one script of events (rooms, turns, namings, practice, test questions), the brain's life on it and its readings |
| `controls.py` | An LSTM and a causal transformer given the same script as tokens, online and offline, at the brain's parameter count and at nine times it |
| `gates.py` | The declared run: every arm's lives on the founders, and the gates read from their files |
| `PROTOCOL.md` | The declared protocol 1: founders, arms, readings and gates |
| `test_words.py` | The chamber's contracts |

Every word is a nonce word of consonant-vowel syllables drawn for the founder, so no word can be
built in; the English names in `world.py` serve the reader alone. Kinds 4 to 7 are named and never
asked about in practice, so their questions read words learned from naming alone; four new kinds
are each named three times and then asked about.

```
python words.py 900                       # one life of the brain, its readings as JSON
python controls.py 900 transformer offline
python gates.py run <out> 32600000 12     # the declared run
python gates.py read <out>
```

## Development readings

Founders 900 to 905, no gated life, on the lane branch: the brain reads 1.0 on W1 (practised and
named-only kinds), W2, W3 and W4 (out of view at the map's own rate of 1.0, never seen, never
taught) and .99 to 1.0 on W5; with its lexicon's synapses at zero it reads W1 at .00. Founders 900
and 901, the matched controls offline read W1 at .06 (LSTM) and .14 (transformer), online .00;
at nine times the parameters .11 and .09. The declared run's readings follow on cadence#326.
