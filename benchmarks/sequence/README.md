# Sequence battery: reads at lengths no lesson reached

These instruments measure the recall read's 0.93 options: where a read looks in a story longer
than any it was taught ([#232](https://github.com/muellerberndt/cadence/issues/232),
[#237](https://github.com/muellerberndt/cadence/issues/237)), a read placed where an earlier
read pointed ([#234](https://github.com/muellerberndt/cadence/issues/234)), when a read item
happened (#237), which write a read finds when only reward teaches it
([#236](https://github.com/muellerberndt/cadence/issues/236)), what earlier lessons keep when
types are taught one after another ([#238](https://github.com/muellerberndt/cadence/issues/238)),
and whether a wider module learns as a narrow one does
([#239](https://github.com/muellerberndt/cadence/issues/239)). They measure settings of the
released library against the founder; they establish no default.

## The battery (`battery.py`)

Each task is a continuing stream of one-hot moments, 16 independent streams in lockstep. An
answer moment's lesson comes before the act on the same observation (`Brain.lesson`, then
`Brain.act`); every other moment is heard without a lesson, and every act is greedy. After the
lessons a frozen test on fresh episodes follows, without lessons.

| Task | A story, then | The answer |
| --- | --- | --- |
| `lookback` | symbols from 4, then a cue: `first`, `second` or `last` | the symbol at the cued place |
| `repeat` | `START`, tokens from 16, `GO` | each token in turn; after `GO` the brain hears the token just answered, its own answer in the test (closed loop) |
| `induction` | tokens from 8, among them `A B`, then `A` | `B` |
| `nback` | tokens from 8, then a cue `k` from 1 to 4 | the token `k` moments back |
| `periodic` | a motif of period `p` repeated four times, a tenth of the symbols varied | from the second period on, the motif's next symbol |
| `twohop` | four pairs, among them `a b` and `b c`, then `a` | `c` (with `think`, `b` at the query and `c` at a `THINK` moment after it) |
| `ago` | 8 to 16 distinct tokens from 24, then one of them with a question flag | how long ago it was told: 0-1, 2-3, 4-7 or 8 and more moments before |
| `before` | 8 to 16 distinct tokens, then `A` with one flag, then `B` with another | whether `A` was told before `B`, answered at `B` |

`lookback` and `repeat` take their training lengths (`lengths`) and their test lengths
(`test_lengths`) apart, so a test can read lengths no lesson reached.

**The life.** `Brain.compose(inputs, actions, modules=(64,), seed=founder, episodic=False,
teaching="normalized")` with the arm's settings beside it. 1,500 lessons of one batch each, then
100 test batches. The training episodes come from `SeedSequence([founder, 11])` and the test
episodes from `SeedSequence([founder, 33])`, so every arm of a founder is taught and read on
the same episodes. `repeat` is read in closed loop and once more teacher-forced.

**Two readings beside the test.** `--erase` sets every head's anchored maps to zero after the
test and reads the same test again: does the anchor carry the read? `--phases first last
second` teaches look back's cues one after another, the lessons split evenly, instead of mixed:
what do the earlier cues keep?

| Arm | Settings |
| --- | --- |
| `table` | `recall=True, recall_recency=0.0, learning_surprise=True`: the position term at its founder rate, no age charge, lessons on the rows answered wrong |
| `boundary` | `table` with `recall_boundary=True` |
| `slow` | `boundary` with `recall_position_rate=0.01` |
| `slope` | `boundary` with `recall_slope=0.1` |
| `slope_slow` | `boundary` with `recall_slope=0.1, recall_position_rate=0.01` |
| `anchor` | `boundary` with `recall_anchor=True` |
| `anchor_relative` | `anchor` with `recall_relative=True` |
| `recipe` | `boundary` with `recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01` |
| `recipe_relative` | `recipe` with `recall_relative=True` |
| `heads` | `table` with `recall_heads=2` |
| `chain` | `heads` with `recall_chain=True` |
| `heads_copy`, `chain_copy` | `heads`, `chain` with the copy pathway as the tokens' identity (`recall_copy=2.0, recall_tokens=range(16)`) |
| `every_row` | `recall=True, recall_recency=0.0`: every row taught |
| `when` | `every_row` with `recall_when=True` |
| `when_anchor` | `when` with `recall_anchor=True` |
| `rehearsal` | `anchor` with `learning_rehearsal=1.0` |

## Ordered recall (`order.py`)

Each episode writes two values and asks for one: `START`, a write, a write, `NEUTRAL`, then
`QUERY` for the latest value written or `QUERY_FIRST` for the first; in the `-gap` conditions a
`NEUTRAL` event falls between the writes. A write carries its value on four payload inputs and
nothing after it does, so the answer needs the read to find the right write by when it happened.
Two actions; the query's action is paid +1 when it names the asked value and -1 otherwise, every
other action nothing. This is the ordered recall world of the Stone Age sequence instrument
(`benchmarks/sequence_v1/worlds.py` in
[cadence-stone-age](https://github.com/muellerberndt/cadence-stone-age)).

Two ways to teach. `--teach reward`: one continuing stream through `Brain.live`, the outcome of
each act arriving with the next event, the query's with the next episode's first event and
`done`; nothing teaches but the outcomes. `--teach lesson`: every episode on a fresh stream,
every event a greedy act, and at the query a lesson naming the asked value before the act. The
brain is the instrument's: `Brain.compose(10, 2, modules=(32,), lateral=0.0, episodic=False,
arousal=True, learning_qualified=True, learning_free_steps=1024, learning_nudged_steps=1024,
learning_tolerance=0.003, working_memory_decay=0.0, working_memory_amplitude=0.3, recall=True)`
with the arm's settings: `founder` (none), `position` (`recall_position_reward=4.0`), `address`
(`recall_address_reward=1.0`), `position_address` (both), `position_address_anchor` (both, with
`recall_anchor=True`), `slow` (`recall_position_rate=0.01`) and `recipe` (`recall_anchor=True,
recall_slope=0.1, recall_position_rate=0.01, recall_recency=0.0, recall_boundary=True`).

Readings at 1,536 and 3,072 episodes: under reward the share of the life's answers paid in the
last 256 episodes (`live` answers greedily while calm and samples while aroused), and on forks
(`copy.deepcopy`) that never change the life, 224 balanced episodes each on a fresh stream with
every event an `act`, greedy on one fork and sampled on another.

## The width sweep (`width.py`)

One life per task, arm, module width and founder. The brain is taught for 1,500 lessons drawn in
turn from a fixed pool of 128 training batches, so the error on the pool and on fresh episodes
can be told apart, and is read at checkpoints: accuracy on the pool and on 24 fresh batches, the
share of motor and association units at a bound of the activation (where no learning signal
passes), the share of synapses at the efficacy cap, settle sweeps per act, refused acts (counted
as misses) and rescued answers. Tasks: `induction` over 128 tokens, `associative` recall of 64
keys and 16 values, and `lookup`, one noisy view of one of 256 fixed prototypes labelled with
its class. `--memory` adds the recall memory with surprise lessons. Arms: `founder`, and
`relative`, every synapse's step scaled to its neuron's fan-in in the same brain composed at
width 64 (`fan_in_rates` of [docs/tune.md](../../docs/tune.md#wide-brains)). `--settings` adds
options, such as `{"learning_surprise_share": true}`.

## Run

From the repository root, with one BLAS thread per process:

```sh
PYTHONPATH=src python benchmarks/sequence/battery.py --task lookback \
    --task-args '{"lengths": [5, 12], "test_lengths": [24, 60]}' \
    --arms boundary recipe --founders 0 1 2 --erase --out DIR/lookback.jsonl --workers 3
PYTHONPATH=src python benchmarks/sequence/battery.py --task repeat \
    --task-args '{"unique": false, "lengths": [3, 6], "test_lengths": [8, 12]}' \
    --arms anchor_relative recipe_relative --out DIR/copy.jsonl --workers 3
PYTHONPATH=src python benchmarks/sequence/battery.py --task lookback --arms anchor rehearsal \
    --lessons 1800 --phases first last second --out DIR/phases.jsonl --workers 3
python benchmarks/sequence/battery.py --summary DIR/lookback.jsonl
PYTHONPATH=src python benchmarks/sequence/order.py --arms founder position_address \
    --out DIR/order.jsonl --workers 5
PYTHONPATH=src python benchmarks/sequence/width.py --task induction --memory --arms relative \
    --widths 1024 --settings '{"learning_surprise_share": true}' --out DIR/width.jsonl
python -m pytest -q benchmarks/sequence
```

A battery life takes one to six minutes on one core, a `periodic` life at a period of 40 about
an hour; an ordered recall life to 3,072 episodes under a minute; a width life at 1,024 neurons
about an hour.

## Readings on 0.93

Read on the 0.93.0 release. The development and confirmation readings were first taken with the
development probes these instruments were ported from; the port gives the probes' numbers
exactly on the same library, and every reading below was taken again on the release. Every
battery life whose protocol the probes shared (147 of them) repeats its earlier reading exactly;
`ago`, `before` and the cues taught in sequence ran other protocols before and are read here
for the first time with this one.

### Positions at lengths no lesson reached (#232, #237)

Development founders 0 to 2. Look back taught at 5 to 12 and read at 24 to 60; copying unique
tokens taught at 3 to 6 and read at 8 to 12 in closed loop:

| Arm | Look back `first` | `second` | `last` | Copying unique tokens |
| --- | --- | --- | --- | --- |
| `boundary` (the founder table) | 0.251 to 0.271 | 0.250 to 0.325 | 0.707 to 0.850 | 0.301 to 0.356 |
| `slow` | 0.313 to 0.347 | 0.300 to 0.328 | 0.662 to 0.954 | 0.863 to 0.993 |
| `slope` | 0.624 to 0.927 | 0.458 to 0.689 | 0.899 to 0.998 | 0.473 to 0.882 |
| `slope_slow` | 0.697 to 0.815 | 0.528 to 0.553 | 0.905 to 0.976 | 0.992 to 0.999 |
| `anchor` | 0.719 to 0.748 | 0.808 to 0.849 | 0.678 to 0.921 | |
| `recipe` | 0.998 to 1.000 | 0.996 to 1.000 | 0.970 to 1.000 | |

**The recipe's confirmation**, declared before any run on founders 232100 to 232105 (gates: at
least 5 of 6 founders at G1 `first` .90, G2 copying unique tokens .95, G3 copying repeated
tokens with `recipe_relative` .90, G4 `second` .90, and G5 `second` falling by at least .30
with the anchored maps set to zero):

| Founder | G1 `first` | G4 `second` | `last` | G5 `second` erased (fall) | G2 copying unique | G3 copying repeats |
| --- | --- | --- | --- | --- | --- | --- |
| 232100 | 1.000 | 0.988 | 0.993 | 0.539 (0.45) | 0.999 | 0.941 |
| 232101 | 0.998 | 0.996 | 1.000 | 0.497 (0.50) | 0.995 | 0.941 |
| 232102 | 0.986 | 0.996 | 0.992 | 0.484 (0.51) | 0.999 | 0.918 |
| 232103 | 0.992 | 0.991 | 0.989 | 0.484 (0.51) | 1.000 | 0.918 |
| 232104 | 1.000 | 1.000 | 0.996 | 0.514 (0.49) | 0.992 | 0.932 |
| 232105 | 1.000 | 0.990 | 0.973 | 0.628 (0.36) | 1.000 | 0.932 |

Every gate passes on 6 of 6. Its costs on founders 232100 to 232102, `table` against `recipe`:
induction 0.999 to 1.000 against 0.998 to 1.000, n-back 1.000 against 0.986 to 0.993, periodic
continuation at a period of 40 0.345 to 0.369 against 0.353 to 0.390: a period of 40 needs the
table exact to 64 ages.

**The anchor's confirmation**, declared before any run on founders 300 to 305 (look back
taught and read at 24 to 60, the cues mixed): `first` 0.998 to 1.000 and `second` 0.990 to
1.000 with the anchor, 0.310 to 0.451 and 0.312 to 0.342 without (`boundary`); `last` 0.989 to
1.000 in both. The battery at its defaults, mean over the six founders, without and with the
anchor: induction 1.000 and 1.000, n-back 1.000 and 0.999, periodic continuation 0.988 and
1.000, copying 1.000 and 1.000.

### A read placed where an earlier read pointed (#234)

Copying repeated tokens (`repeat` with `unique` off) taught at 3 to 6 and read at 8 to 12:

| Arm | Founders | Closed loop | The first token after `GO` | Teacher-forced |
| --- | --- | --- | --- | --- |
| `anchor` | 0 to 2 | 0.099 to 0.138 | 0.102 to 0.236 | 0.241 to 0.361 |
| `anchor_relative` | 0 to 2 | 0.544 to 0.741 | 0.625 to 0.803 | 0.870 to 0.932 |
| `recipe_relative` | 232100 to 232105 | 0.918 to 0.941 | 0.967 to 0.992 | 0.963 to 0.976 |

Two hops asked in one moment (`twohop` without `think`; founders 0 to 2): `heads` 0.436 to
0.469 and `chain` 0.439 to 0.458 after 1,500 lessons; with the copy pathway, after 3,000,
`heads_copy` 0.473 to 0.502 and `chain_copy` 0.506 to 0.533 (0.03 higher on each founder).
With a `THINK` moment the two heads answer the second hop at 0.983 to 0.990.

### When a read item happened (#237)

Founders 0 to 2, every row taught:

| Arm | "How long ago was X told?" | "Was A told before B?" |
| --- | --- | --- |
| `every_row` | 0.509 to 0.796 | 0.594 to 0.718 |
| `when` | 0.809 to 0.907 | 0.594 to 0.796 |
| `when_anchor` | 0.911 to 0.917 | 0.590 to 0.803 |

### Cues taught in sequence (#238)

Look back at its defaults, founders 0 to 2, 1,800 lessons: taught mixed (`anchor` without
phases) every cue reads 1.000. Taught `first`, then `last`, then `second`, 600 lessons each:

| Arm | `first` | `last` | `second` |
| --- | --- | --- | --- |
| `anchor` | 0.324 to 0.739 | 0.647 to 0.949 | 1.000 |
| `rehearsal` | 0.285 to 0.437 | 0.432 to 0.589 | 1.000 |

The rehearsal teaches stored drives again, as the read gave them when they were stored; the
cues held by where the read looks lose more with it than without it.

### Ordered recall (#236, #232)

Five development founders (1001 to 1005), 3,072 episodes. Under reward the share of the life's
answers paid over its last 256 episodes, with the forks' greedy and sampled readings:

| Arm, taught by reward through `live` | The life's last 256 answers | Greedy | Sampled |
| --- | --- | --- | --- |
| `founder` | 0.46 to 0.50 | 0.50 on every founder | 0.47 to 0.51 |
| `position` (`recall_position_reward=4`) | 0.73 to 1.00; three founders 1.00 | 0.75 to 1.00 | 0.73 to 0.88 |
| `position` at 16 | 0.45 to 1.00; four founders 1.00, one at chance | 0.50 to 1.00 | 0.49 to 0.91 |
| `address` (`recall_address_reward=1`) | 0.47 to 1.00 | 0.50 to 1.00 | 0.50 to 0.97 |
| `position_address` | 0.48 to 1.00; one founder at chance | 0.50 to 1.00 | 0.50 to 0.79 |
| `position_address_anchor` | 0.50 to 1.00 | 0.50 to 1.00 | 0.50 to 0.86 |

#236 asks for 0.90 under `live`. The position reward reaches it on three to four founders of
five, not on every one, and adding the address reward does not lift it.

Taught by a lesson at the query (1,536 episodes), the founder table answers every query
greedily (sampled 0.97 to 0.99). The table at 0.01 alone (`slow`) answers 0.50 to 1.00 greedily
(mean 0.80; 0.875 at 3,072 episodes), and the recipe of the battery (`recipe`) 0.875 to 1.00
(mean 0.975): the slower table costs reads that need exact ages, and the anchor and the slope
beside it give most of it back.

### The width sweep (#239)

Induction over 128 tokens at 1,024 neurons with the recall memory and surprise lessons, each
synapse's step scaled to its neuron's fan-in at width 64 (`relative`), 1,500 lessons, held-out
accuracy every 100 lessons:

| Founder | Without the share | With `learning_surprise_share=True` |
| --- | --- | --- |
| 0 | | 0.12, 0.65, 0.76, 0.85, then 0.97 to 0.99 from 500 lessons; 0.99 at the end |
| 1 | swings: 0.42 at 500, 0.65 at 900, 1.00 from 1,200 | 0.96 to 0.99 from 500 (0.90 at 800); 0.99 at the end |
| 2 | swings: 0.01 at 400 and 500 and at 1,000, 0.18 at 1,300, 0.98 at the end | 0.85 to 0.95 from 500; 0.95 at the end |

No act was refused in these lives. The life that refused an act at lesson 343 before the settle
rescue existed (induction at 64 neurons on the composed steps, founder 0) runs its 1,500
lessons: two answers are rescued, none is refused, and it reads 0.64, 0.82, 0.93 and 0.97
held-out at 250, 500, 1,000 and 1,500 lessons. Lookup and associative recall at 1,024 neurons,
founder and `relative` arms, repeat their 0.92.0 lives exactly (checked at 250 lessons), so the
sweep's readings in [docs/tune.md](../../docs/tune.md#wide-brains) stand on 0.93.
