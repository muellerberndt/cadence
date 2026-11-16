# Symbols chamber: the parrot that starts with no words

A brain with eyes, an ear and a voice (`Brain.compose(senses={"eyes": ..., "ear": ...},
code=2048, bound=4096, voice="ear")`) lives through what a trainer does with a young parrot:
the trainer says words, names things, asks a question and pays a reward for a right answer. The
chamber reads whether a sound comes to stand for a thing and a phrase for what it asks, through
the brain's records of its bound moments ([`cadence.bound`](../../src/cadence/bound.py), the
guide's [sounds as symbols](../../docs/memory.md#records-of-bound-moments-opt-in)), for
[#342](https://github.com/muellerberndt/cadence/issues/342) and
[#344](https://github.com/muellerberndt/cadence/issues/344).

`world.py` holds the world and the trainer: eight kinds of things in two colours each, at places
around the body, seen as tuned cells (`cadence.senses.Things`); seventeen words (the kinds, five
colours, two questions), each a pattern of log-mel energies that a speaker says faster or slower,
higher or lower, louder or softer and with the room's noise, heard as one code
(`cadence.senses.Ear`). The trainer hears what the parrot says as the nearest word. `chamber.py`
holds the life and the readings; its docstring gives each step.

```sh
PYTHONPATH=src python benchmarks/symbols/chamber.py --founders 0 1 --out dev.json
python -m pytest -q benchmarks/symbols
```

## The life

1. **Listen:** every word said three times, nothing in view.
2. **Naming:** a thing held up and its kind's name said, eight times a kind, each time at another
   place and in either colour.
3. **Reading Y1h, Y2** on that brain.
4. **Reward** (a fresh brain of the founder after Listen alone): four things, one after the other,
   each until the parrot names it right three times running or 80 exchanges, a reward of 1.0 for
   the right name and 0.0 otherwise; then the thing at places no exchange used (**Y1**); then
   "what colour is this" with four things, the same way. The same exchanges unpaid on another
   fresh brain.
5. **Controls on the same exchanges:** a lookup table keyed by (the thing, the word heard), and a
   delta-rule readout of the brain's code taught by the reward.
6. **A heard name draws the focus (N):** a brain with a map of things whose values are the eyes'
   fields and `surprise=True`, after Listen and Naming; three things appear one by one; the
   trainer says the name of one of the first two. Twelve rooms, each with the named thing in view
   and behind a screen; the same after Listen alone.

## Declared protocol

Declared on [#342](https://github.com/muellerberndt/cadence/issues/342) before the first life of
its founders. Founders **1010100 to 1010111** (twelve, fresh). Development used founders 0 to 3.
One life a founder on the CPU backend (`backend="cpu"`), one thread.

| Gate | Reading | Bar |
| --- | --- | --- |
| S1 | Y2, records: the kind read from the shape and surface a kind's name heard alone brings back (40 takes) | .90, on 10 of 12 |
| S2 | Y2, state: the kind read by a linear probe from the association's settled state after a name heard alone (the probe fitted on things seen alone, 128 views) | .90, on 10 of 12 |
| S3 | Y2 controls: the same probe with the records barred, and with the records' values dealt to other keys | each at most .25 (chance .125), on 10 of 12 |
| S4 | Y3, names: each of the four things named right three times running within 80 exchanges | all four, on 10 of 12; unpaid: the share right at most .15 (chance about .06) |
| S5 | Y3, the phrase "what colour is this": each of four things answered right three times running within 80 exchanges | all four, on 10 of 12; unpaid as S4 |
| S6 | Y1: after S4, the four things at places no exchange used (20 views), named right | .90, on 10 of 12 |
| S7 | N: the focus moves to within 0.25 of the named thing, in view and behind a screen (12 rooms each) | each .90, on 10 of 12; after Listen alone at most .10 |

Reported without a bar: Y1h (a thing named from the naming moments alone, no reward), the
exchanges each learner took (the brain, the table, the delta rule), the records held, the
parameters and the seconds a life.

**What is designed, by name.** The world, the trainer and its ear for words; the records' rule
and its constants (`BoundRecords`: the read by the nearest records that hold a field, the
heaviest group, a refused record weighing against its sound, a reward's tellings, the babble);
the name's pull on the focus (`NAMED`). **Learned:** every record, its strength, what each sound
brings back, what the parrot says, which thing a name draws the focus to.

**Expectations, from development (founders 0 to 3).** S1 1.00 on every founder; S2 .68 to 1.00
(one founder of four under .90 in the first development pass, 1.00 and .98 in the second);
S3 .03 to .28 barred and .00 to .10 dealt; S4 and S5 each thing learned in 4 to 71 exchanges
with one phrase item past the budget on one founder, unpaid .05 to .15; S6 .60 to 1.00 (one
founder of two at 1.00, one at .80 in the second pass); S7 .83 to 1.00 taught and .00 untaught.
S2, S5 and S6 may miss on more than two founders.

**The located limit (Y1h).** A thing seen alone is named from the naming moments at .82 to .90:
the read weighs the thing's place and colour as it weighs its form, and a thing's form decides
its name. Records alone do not learn which field a name follows; the reward of S4 does it for
the things it pays.

## Readings

### The first declared run: S5 and S6 fail

Founders 1010100 to 1010111 at `08465632`, as declared above.

| founder | S1 | S2 | barred | dealt | names (exchanges) | colours (exchanges) | unpaid names | unpaid colours | S6 | S7 in view | S7 screen | untaught | Y1h | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1010100 | 1.0 | 0.875 | 0.225 | 0.0 | [9, 18, 46, 17] | [36, 16, 79, 42] | 0.059 | 0.053 | 0.8 | .92 | 1.0 | 0.0 | 0.975 | 119.7 |
| 1010101 | 1.0 | 0.975 | 0.3 | 0.0 | [7, 14, 27, 32] | [49, 38, 79, 42] | 0.091 | 0.084 | 0.6 | 1.0 | 1.0 | 0.0 | 0.95 | 128.3 |
| 1010102 | 1.0 | 1.0 | 0.05 | 0.0 | [17, 9, 38, 6] | [34, 14, 4, 40] | 0.059 | 0.053 | 1.0 | .92 | 1.0 | 0.0 | 0.925 | 93.9 |
| 1010103 | 1.0 | 0.975 | 0.175 | 0.0 | [5, 20, 33, 14] | [20, 18, 37, 26] | 0.05 | 0.059 | 0.85 | 1.0 | 1.0 | 0.0 | 0.825 | 100.1 |
| 1010104 | 1.0 | 0.9 | 0.325 | 0.0 | [5, 9, 31, 19] | [10, 7, None, 45] | 0.053 | 0.041 | 0.65 | .92 | 1.0 | 0.0 | 0.85 | 111.0 |
| 1010105 | 1.0 | 1.0 | 0.25 | 0.025 | [3, 9, 10, 15] | [38, 21, 79, 38] | 0.05 | 0.069 | 0.9 | .92 | 1.0 | 0.0 | 0.95 | 117.2 |
| 1010106 | 1.0 | 0.975 | 0.125 | 0.0 | [14, 29, 12, 37] | [19, 59, 24, 30] | 0.081 | 0.059 | 0.85 | .92 | 1.0 | 0.0 | 0.825 | 171.7 |
| 1010107 | 1.0 | 1.0 | 0.05 | 0.125 | [7, 9, 8, 11] | [None, 12, 45, 35] | 0.05 | 0.081 | 0.95 | 1.0 | 1.0 | 0.0 | 0.95 | 177.4 |
| 1010108 | 1.0 | 0.9 | 0.125 | 0.15 | [6, 5, 7, 31] | [22, 13, 43, 26] | 0.05 | 0.053 | 0.95 | 1.0 | 1.0 | 0.0 | 0.825 | 140.0 |
| 1010109 | 1.0 | 0.9 | 0.075 | 0.05 | [13, 5, 7, 27] | [57, 7, None, 27] | 0.059 | 0.066 | 0.7 | 1.0 | 1.0 | 0.0 | 0.875 | 134.0 |
| 1010110 | 1.0 | 0.9 | 0.2 | 0.025 | [11, 22, 11, 16] | [20, 29, None, 71] | 0.075 | 0.041 | 0.75 | .92 | 1.0 | 0.0 | 0.95 | 118.6 |
| 1010111 | 1.0 | 1.0 | 0.125 | 0.0 | [15, 6, 7, 43] | [6, 79, 55, 78] | 0.081 | 0.087 | 1.0 | 1.0 | 1.0 | 0.0 | 0.925 | 161.8 |


| Gate | Founders that hold it | |
| --- | --- | --- |
| S1 | 12 of 12 | holds |
| S2 | 11 of 12 | holds |
| S3 | 10 of 12 | holds |
| S4 | 12 of 12 | holds |
| S5 | 8 of 12 | fails |
| S6 | 5 of 12 | fails |
| S7 | 12 of 12 | holds |

**Located.** The voice said the sound of the paid records nearest the moment by the whole code,
where a thing's place weighs as much as its form: at a new place the paid records of another
thing at a near place won as often as the thing's own (S6), and a phrase's answer paid once was
lost at the next place (S5). A trace of one phrase item past the budget (development founder 1)
showed a second term: the sounds passed over in a babble were those refused at every moment the
read reached, the same question asked of other things included, so the right colour, refused for
another thing, was passed over for this one; and a babble drew each record once, so the names
paid many times in S4 took most draws.

### The second declared run

**What changed** (`cadence.bound`, before the run): the voice says the paid sound whose moments
share fields nearest the moment's (a group of paid records with one sound stands for the fields
all of them hold alike; a field where they differ is left out; `ANSWER` .5); a refused record
weighs against its sound only at moments like it (more than half of the moment's fields held
alike), for the answer and for the babble; a babble draws each distinct sound once. Development,
founders 0 to 5: S4 3 to 17 exchanges a thing, S5 5 to 16, S6 1.00 on 6 of 6, unpaid .04 to
.09; on founders 0 to 2 the whole life: S1 1.00, S2 .975 to 1.00, barred .075 to .275, S7 .83 to
1.00 in view and 1.00 behind a screen.

**Declared** on [#342](https://github.com/muellerberndt/cadence/issues/342) before the first life:
founders **1010200 to 1010211**, twelve and fresh, the same chamber,
gates and bars as the first run (S1 to S7 above), one life a founder on the CPU.

**Expectations.** S1, S4, S6 and S7 hold on 11 or 12 founders; S2 and S3 as in the first run
(11 and 10 of 12); S5 on 11 or 12. A founder whose barred probe reads above .25 (two of the first
run's twelve) misses S3, and S3 may fail on three.
