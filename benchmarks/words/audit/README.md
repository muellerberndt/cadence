# The words chamber's audit (cadence#326)

Tests that try to break the claims of `benchmarks/words`: that the parrot's brain answers
questions about its room in words learned from naming, from what it holds, with no table from
word to concept and no table from question word to read in its path. Each script lives the
founder's script as `words.life` does and asks every test question again with one thing
changed. Nothing here is a gate: these are readings.

| Script | What it asks |
| --- | --- |
| `ablations.py` | Where does each answer come from? The lexicon, the map, one word's synapses, the reads, "I do not know", each question word taken away; the body turned, the colours reported one on, a room seen in part, the map never emptied |
| `adversarial.py` | Questions shuffled, with a word never heard, with a colour the thing does not have, with two kinds, without their question words or without their content word; scenes with two things close together, two things of one kind, a crowded room |
| `lookup.py` | The lexicon's place taken by a count table (a control by the lane's rules) and by a hand-written table of the world's words: how much of the readings is the program around the lexicon |
| `path.py` | The steps of the question's path that the program chooses; the motor's letters against the heard form; the lexicon's synapses against a count table fed the same lessons |
| `genes.py` | The brain's hand-set values (a naming's shares, the lesson's rate, the word threshold), each beside the value changed |
| `reward.py` | The pistachio (W11): practice by reward beside a pistachio withheld, given at random at the same rate, given for wrong answers, given one answer late, and the told answers |
| `heldout.py` | What is held out (unseen colours, new kinds, named-only kinds, the test's foils), and whether the nonce words are each founder's own |
| `battery.py`, `read.py` | Every script on a range of founders, and the readings as tables |

```
python battery.py <out> 32600000 12 15      # every script on the declared founders, 15 at a time
python read.py <out>                        # ablations, adversarial, lookup
python genes.py read <out>
python ablations.py 900 <out> short         # one script, one founder, a short life
```

`common.py` is the harness (the life replayed with probes beside the brain), `truth.py` the
answer a question has over a room as the brain could hold it.
