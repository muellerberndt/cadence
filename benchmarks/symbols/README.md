# Symbols chamber: a brain that starts with no words

A brain with eyes, an ear and a voice (`Brain.compose(senses={"eyes": ..., "ear": ...},
code=2048, bound=4096, voice="ear")`) lives through what a trainer does with a young animal:
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
(`cadence.senses.Ear`). The trainer hears what the brain says as the nearest word. `chamber.py`
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
   each until the brain names it right three times running or 80 exchanges, a reward of 1.0 for
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
brings back, what the brain says, which thing a name draws the focus to.

**Expectations, from development (founders 0 to 3).** S1 1.00 on every founder; S2 .68 to 1.00
(one founder of four under .90 in the first development pass, 1.00 and .98 in the second);
S3 .03 to .28 barred and .00 to .10 dealt; S4 and S5 each thing learned in 4 to 71 exchanges
with one phrase item past the budget on one founder, unpaid .05 to .15; S6 .60 to 1.00 (one
founder of two at 1.00, one at .80 in the second pass); S7 .83 to 1.00 taught and .00 untaught.
S2, S5 and S6 may miss on more than two founders.

**The located limit (Y1h).** A thing seen alone is named from the naming moments at .82 to .90
in development (.725 to .975 in the third declared run):
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

**Read: S2 fails (8 of 12); S1, S3, S4, S5, S6 and S7 hold** (12, 12, 12, 12, 12 and 11 of 12).

| founder | S1 | S2 | barred | dealt | names (exchanges) | colours (exchanges) | unpaid names | unpaid colours | S6 | S7 in view | S7 screen | untaught | Y1h | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1010200 | 1.0 | 1.0 | 0.1 | 0.05 | [12, 8, 5, 6] | [11, 8, 17, 15] | 0.072 | 0.075 | 1.0 | 1.0 | 1.0 | 0.0 | 0.925 | 62.0 |
| 1010201 | 1.0 | 1.0 | 0.025 | 0.025 | [8, 5, 4, 4] | [9, 21, 7, 8] | 0.062 | 0.059 | 1.0 | 1.0 | 1.0 | 0.0 | 0.95 | 63.2 |
| 1010202 | 1.0 | 0.875 | 0.075 | 0.025 | [3, 6, 8, 3] | [17, 7, 11, 11] | 0.05 | 0.084 | 1.0 | .83 | 1.0 | 0.0 | 0.9 | 60.3 |
| 1010203 | 1.0 | 1.0 | 0.15 | 0.0 | [11, 7, 8, 11] | [7, 4, 10, 18] | 0.072 | 0.066 | 1.0 | 1.0 | 1.0 | 0.0 | 0.9 | 57.6 |
| 1010204 | 1.0 | 1.0 | 0.125 | 0.0 | [14, 7, 15, 12] | [16, 14, 19, 12] | 0.081 | 0.078 | 1.0 | .92 | 1.0 | 0.0 | 0.925 | 61.4 |
| 1010205 | 1.0 | 0.975 | 0.1 | 0.0 | [9, 13, 7, 16] | [8, 17, 15, 14] | 0.066 | 0.069 | 1.0 | 1.0 | 1.0 | 0.0 | 0.875 | 57.5 |
| 1010206 | 1.0 | 0.875 | 0.15 | 0.0 | [12, 13, 13, 4] | [17, 11, 7, 7] | 0.056 | 0.062 | 1.0 | 1.0 | .92 | 0.0 | 0.85 | 57.8 |
| 1010207 | 1.0 | 1.0 | 0.125 | 0.075 | [13, 13, 4, 15] | [10, 7, 14, 18] | 0.087 | 0.056 | 1.0 | .92 | 1.0 | 0.0 | 0.85 | 58.1 |
| 1010208 | 1.0 | 0.975 | 0.125 | 0.0 | [16, 16, 11, 4] | [15, 13, 12, 18] | 0.059 | 0.05 | 1.0 | 1.0 | 1.0 | 0.0 | 0.875 | 57.8 |
| 1010209 | 1.0 | 0.875 | 0.05 | 0.025 | [4, 4, 12, 13] | [6, 18, 15, 5] | 0.107 | 0.047 | 1.0 | 1.0 | 1.0 | 0.0 | 0.8 | 50.4 |
| 1010210 | 1.0 | 0.875 | 0.125 | 0.0 | [8, 6, 12, 10] | [18, 11, 6, 16] | 0.089 | 0.047 | 1.0 | .92 | 1.0 | 0.0 | 0.8 | 48.3 |
| 1010211 | 1.0 | 1.0 | 0.25 | 0.1 | [15, 8, 5, 10] | [16, 16, 12, 15] | 0.037 | 0.084 | 1.0 | .92 | 1.0 | 0.0 | 0.925 | 54.0 |


**Located (S2).** On every founder that misses S2 the records bring the right kind back (the
records' own read, S1, is 1.00 on every founder) and the state's probe reads one kind as another
on all five takes: an apple as a ball (three founders), an apple as a banana or a pear, a ball as
an apple. The probe misreads the same kinds when they are seen (.925 to .975 on things seen
alone, four founders): the 64 neurons of the association settle an apple and a ball, alike in
form and both red, to states the probe does not tell apart. With 256 association neurons the
probe reads things seen at 1.00 and a name heard alone at 1.00 (development founders 0 to 2).

### The third declared run

**What changed:** the chamber's brain has 256 association neurons (`modules=(256,)`, 7,180,417
parameters in place of 5,128,321); the library is the second run's. Development, founders 0 to 2,
the whole life: S1 1.00, S2 1.00, barred .025 to .15, dealt .00, S4 5 to 17 exchanges, S5 5 to 16,
S6 1.00, S7 .83 to 1.00 in view and 1.00 behind a screen, 96 to 98 seconds a life.

**Declared** on [#342](https://github.com/muellerberndt/cadence/issues/342) before the first life:
founders **1010300 to 1010311**, twelve and fresh, the same gates and bars (S1 to S7 above).

**Expectations.** Every gate on 11 or 12 founders; S3 and S7 may miss on one or two, as in the
first two runs.

**Read: every gate holds.**

| founder | S1 | S2 | barred | dealt | names (exchanges) | colours (exchanges) | unpaid names | unpaid colours | S6 | S7 in view | S7 screen | untaught | Y1h | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1010300 | 1.0 | 1.0 | 0.05 | 0.0 | [7, 7, 19, 10] | [10, 7, 16, 15] | 0.072 | 0.091 | 1.0 | 1.0 | 1.0 | 0.0 | 0.85 | 86.7 |
| 1010301 | 1.0 | 1.0 | 0.05 | 0.075 | [16, 4, 16, 16] | [9, 10, 11, 16] | 0.084 | 0.074 | 1.0 | 1.0 | 1.0 | 0.0 | 0.825 | 85.5 |
| 1010302 | 1.0 | 1.0 | 0.05 | 0.0 | [5, 8, 13, 9] | [7, 14, 19, 9] | 0.072 | 0.066 | 1.0 | 1.0 | 1.0 | 0.0 | 0.9 | 88.6 |
| 1010303 | 1.0 | 1.0 | 0.3 | 0.0 | [5, 5, 12, 17] | [6, 17, 15, 5] | 0.072 | 0.05 | 1.0 | .92 | 1.0 | 0.0 | 0.975 | 99.3 |
| 1010304 | 1.0 | 1.0 | 0.1 | 0.0 | [18, 4, 13, 6] | [7, 16, 10, 11] | 0.075 | 0.062 | 1.0 | .92 | 1.0 | 0.0 | 0.95 | 98.0 |
| 1010305 | 1.0 | 1.0 | 0.05 | 0.05 | [4, 11, 16, 4] | [15, 6, 16, 17] | 0.075 | 0.047 | 1.0 | .92 | .92 | 0.0 | 0.875 | 101.3 |
| 1010306 | 1.0 | 1.0 | 0.1 | 0.0 | [5, 15, 16, 8] | [9, 8, 7, 5] | 0.056 | 0.05 | 1.0 | .92 | 1.0 | 0.0 | 0.9 | 117.2 |
| 1010307 | 1.0 | 1.0 | 0.1 | 0.0 | [10, 9, 4, 10] | [5, 11, 6, 16] | 0.081 | 0.072 | 1.0 | .83 | 1.0 | 0.0 | 0.9 | 108.0 |
| 1010308 | 1.0 | 1.0 | 0.0 | 0.0 | [7, 8, 12, 4] | [5, 6, 17, 5] | 0.078 | 0.066 | 1.0 | 1.0 | 1.0 | 0.0 | 0.9 | 106.0 |
| 1010309 | 1.0 | 1.0 | 0.025 | 0.0 | [15, 6, 13, 18] | [12, 10, 11, 16] | 0.056 | 0.069 | 1.0 | 1.0 | 1.0 | 0.0 | 0.725 | 102.9 |
| 1010310 | 1.0 | 1.0 | 0.225 | 0.0 | [9, 10, 13, 16] | [9, 16, 13, 16] | 0.107 | 0.062 | 1.0 | .92 | 1.0 | 0.0 | 0.85 | 102.3 |
| 1010311 | 1.0 | 1.0 | 0.1 | 0.0 | [12, 6, 15, 5] | [16, 16, 12, 12] | 0.059 | 0.066 | 1.0 | .92 | 1.0 | 0.0 | 0.875 | 105.6 |


| Gate | Founders that hold it | |
| --- | --- | --- |
| S1, a name brings its thing back from the records | 12 of 12 (1.00 each) | holds |
| S2, the thing in the settled state | 12 of 12 (1.00 each) | holds |
| S3, controls: barred .00 to .30, dealt .00 to .075 | 11 of 12 | holds |
| S4, names learned by reward | 12 of 12 (4 to 19 exchanges a thing; unpaid .056 to .107) | holds |
| S5, "what colour is this" learned by reward | 12 of 12 (5 to 19; unpaid .047 to .091) | holds |
| S6, the named things at new places | 12 of 12 (1.00 each) | holds |
| S7, a heard name draws the focus | 11 of 12 (.83 to 1.00 in view, .92 to 1.00 behind a screen, .00 after listening alone) | holds |

Beside them, without a bar: a thing named from the naming moments alone (Y1h) .725 to .975; the
lookup table and the delta-rule readout given the same exchanges are in each life's record.

### The fourth declared run: the code that ships

**What changed after the third run** (an independent review of the release; `cadence.bound`): a
record of something said answers only once a reward paid it (its pay counted apart from its
depth, which telling a moment again and the brain's surprise also raise); a reward after `wait`
pays the moment the brain acted at; records kept for good stay readable; a refused record weighs
only on what the voice says, and completion reads every known record; a voice of a sense with
fields is refused. The chamber and its brain are the third run's.

**Declared** on [#342](https://github.com/muellerberndt/cadence/issues/342) before the first life:
founders **1010400 to 1010411**, twelve and fresh, the same gates and bars.

**Expectations.** As the third run: every gate on 11 or 12 founders, S3 and S7 possibly missing
on one or two.

**Read: every gate holds** (S1 to S6 on 12 of 12, S7 on 10 of 12).

| founder | S1 | S2 | barred | dealt | names (exchanges) | colours (exchanges) | unpaid names | unpaid colours | S6 | S7 in view | S7 screen | untaught | Y1h | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1010400 | 1.0 | 1.0 | 0.175 | 0.075 | [11, 12, 17, 8] | [7, 7, 14, 8] | 0.062 | 0.066 | 1.0 | .92 | 1.0 | 0.0 | 0.95 | 169.5 |
| 1010401 | 1.0 | 1.0 | 0.1 | 0.0 | [15, 4, 9, 5] | [6, 12, 7, 16] | 0.059 | 0.075 | 1.0 | 1.0 | 1.0 | 0.0 | 0.9 | 167.2 |
| 1010402 | 1.0 | 1.0 | 0.175 | 0.0 | [17, 6, 11, 5] | [9, 15, 18, 17] | 0.081 | 0.05 | 1.0 | 1.0 | 1.0 | 0.0 | 0.85 | 167.9 |
| 1010403 | 1.0 | 1.0 | 0.125 | 0.0 | [13, 14, 6, 12] | [13, 19, 12, 21] | 0.075 | 0.066 | 1.0 | .92 | .92 | 0.0 | 0.925 | 137.1 |
| 1010404 | 1.0 | 1.0 | 0.05 | 0.0 | [9, 7, 4, 4] | [4, 5, 8, 17] | 0.091 | 0.053 | 1.0 | .92 | 1.0 | 0.0 | 0.95 | 133.5 |
| 1010405 | 1.0 | 0.975 | 0.05 | 0.0 | [12, 5, 9, 3] | [11, 10, 8, 17] | 0.056 | 0.072 | 1.0 | .83 | .92 | 0.0 | 0.925 | 133.2 |
| 1010406 | 1.0 | 1.0 | 0.15 | 0.125 | [6, 15, 9, 15] | [16, 12, 12, 10] | 0.044 | 0.069 | 1.0 | .92 | 1.0 | 0.0 | 0.925 | 120.5 |
| 1010407 | 1.0 | 1.0 | 0.175 | 0.0 | [14, 10, 6, 17] | [16, 6, 8, 14] | 0.072 | 0.053 | 1.0 | 1.0 | 1.0 | 0.0 | 0.825 | 124.9 |
| 1010408 | 1.0 | 1.0 | 0.075 | 0.0 | [12, 10, 18, 12] | [10, 9, 12, 17] | 0.094 | 0.069 | 1.0 | .83 | 1.0 | 0.0 | 0.825 | 122.5 |
| 1010409 | 1.0 | 1.0 | 0.05 | 0.0 | [15, 3, 17, 12] | [9, 5, 5, 5] | 0.084 | 0.066 | 1.0 | .92 | 1.0 | 0.0 | 0.825 | 199.4 |
| 1010410 | 1.0 | 1.0 | 0.175 | 0.0 | [11, 13, 3, 6] | [7, 11, 19, 5] | 0.05 | 0.053 | 1.0 | 1.0 | 1.0 | 0.0 | 0.925 | 195.2 |
| 1010411 | 1.0 | 1.0 | 0.175 | 0.0 | [4, 7, 9, 4] | [5, 6, 13, 11] | 0.069 | 0.072 | 1.0 | .92 | .92 | 0.0 | 0.8 | 195.5 |


| Gate | Founders that hold it | |
| --- | --- | --- |
| S1 | 12 of 12 (1.00 each) | holds |
| S2 | 12 of 12 (.975 to 1.00) | holds |
| S3 | 12 of 12 (barred .05 to .175, dealt .00 to .125) | holds |
| S4 | 12 of 12 (3 to 18 exchanges a thing; unpaid .044 to .094) | holds |
| S5 | 12 of 12 (4 to 21; unpaid .05 to .075) | holds |
| S6 | 12 of 12 (1.00 each) | holds |
| S7 | 10 of 12 (.83 to 1.00 in view, .92 to 1.00 behind a screen, .00 after listening alone) | holds |

Without a bar: Y1h .80 to .95.

### The senses beside the code (0.102.0)

`--cortex-reads both` composes every brain of the life with `cortex_reads="both"`: the cortex
reads the senses beside the code, and every memory keys on the code as before. A quick reading of
the gates on four fresh founders, **1011200 to 1011203**, against the same founders with the code
alone, the chamber and the bars of the fourth run, at `e8718323`
([receipt](results/cortex_reads.json)).

| founder | arm | S1 | S2 | barred | dealt | names (exchanges) | colours (exchanges) | unpaid names | unpaid colours | S6 | S7 in view | S7 screen | untaught | Y1h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1011200 | code | 1.0 | 1.0 | .125 | .0 | [9, 13, 15, 10] | [9, 15, 10, 8] | .062 | .078 | 1.0 | 1.0 | 1.0 | .0 | .925 |
| 1011200 | both | 1.0 | 1.0 | .175 | .0 | [9, 13, 15, 10] | [9, 15, 10, 8] | .062 | .078 | 1.0 | 1.0 | 1.0 | .0 | .925 |
| 1011201 | code | 1.0 | 1.0 | .15 | .0 | [14, 15, 7, 9] | [9, 10, 13, 8] | .062 | .062 | 1.0 | 1.0 | 1.0 | .0 | .875 |
| 1011201 | both | 1.0 | 1.0 | .125 | .0 | [14, 15, 7, 9] | [9, 10, 13, 8] | .062 | .062 | 1.0 | 1.0 | 1.0 | .0 | .875 |
| 1011202 | code | 1.0 | 1.0 | .125 | .1 | [17, 5, 4, 8] | [7, 16, 8, 7] | .072 | .087 | 1.0 | .92 | 1.0 | .0 | .925 |
| 1011202 | both | 1.0 | 1.0 | .125 | .0 | [17, 5, 4, 8] | [7, 16, 8, 7] | .072 | .087 | 1.0 | .92 | 1.0 | .0 | .925 |
| 1011203 | code | 1.0 | 1.0 | .05 | .0 | [5, 6, 15, 3] | [15, 10, 12, 17] | .069 | .056 | 1.0 | .92 | 1.0 | .0 | .825 |
| 1011203 | both | 1.0 | 1.0 | .125 | .0 | [5, 6, 15, 3] | [15, 10, 12, 17] | .069 | .056 | 1.0 | .92 | 1.0 | .0 | .825 |

Every gate holds on every founder in both arms, and the two arms read alike at every gate: a
name heard alone brings its thing back from the records and in the settled state at 1.00, the
exchanges a reward takes are the same, and so are the named things at new places and the focus a
heard name draws. The controls of S3 move within their bar (.25): with the records barred
the state reads .125 to .175 against .05 to .15, and with their values dealt .00 against .00 to
.10. The brain holds 7,523,769 parameters against 7,180,417.
