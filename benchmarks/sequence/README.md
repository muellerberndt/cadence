# Sequence battery: reads at lengths no lesson reached

Readings on this page taken before 0.95 took the address lesson as the distance, which left the library in 0.95, so a memory now reads with the contrast and those tables are historical.

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
| `recipe64` | `recipe` with `recall_position=64`: the table exact to 64 ages |
| `recipe_relative` | `recipe` with `recall_relative=True` |
| `heads` | `table` with `recall_heads=2` |
| `every_row` | `recall=True, recall_recency=0.0`: every row taught |
| `when` | `every_row` with `recall_when=True` |
| `when_anchor` | `when` with `recall_anchor=True` |
| `when_sharp` | `when` with `recall_position_rate=0.0, recall_temperature=0.1`: the position term not taught and a sharp softmax, so the read lands on the item heard and stays there |
| `rehearsal` | `anchor` with `learning_rehearsal=1.0`: the lessons' episodes replayed |

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
with the arm's settings: `founder` (none), `position` (`recall_position_reward=4.0`),
`reward_recipe` (`recall_position_reward=4.0, recall_address_reward=1.0, recall_anchor=True,
recall_next=False`, which left with `recall_next=False` in 0.96.0; its readings below are
historical), `address` (`recall_address_reward=1.0`), `position_address` (both),
`position_address_anchor` (both, with `recall_anchor=True`), `slow`
(`recall_position_rate=0.01`) and `recipe` (`recall_anchor=True, recall_slope=0.1,
recall_position_rate=0.01, recall_recency=0.0, recall_boundary=True`).

Readings at 1,536 and 3,072 episodes: under reward the share of the life's answers paid in the
last 256 episodes (`live` answers greedily while calm and samples while aroused), and on forks
(`copy.deepcopy`) that never change the life, 224 balanced episodes each on a fresh stream with
every event an `act`, greedy on one fork and sampled on another.

`--world recall` reads the instrument's recall world instead, where `QUERY` always asks the
latest value written (nine inputs, no `QUERY_FIRST`), in seven conditions: one write followed
by 0, 1 or 2 `NEUTRAL` events (`clean-0` to `clean-2`) or by 1 or 2 `DISTRACTOR` events whose
payload is drawn from [-0.25, 0.25] (`distractor-1`, `distractor-2`); the other value, a
`NEUTRAL`, the asked value and a `NEUTRAL` (`replacement`); the other value, the asked value
and a `NEUTRAL` (`order-latest`). Its events and its schedule (balanced blocks of 14, each
episode with a seed of its own for its distractors) are the Stone Age world's, checked equal on
every condition; the brain, the arms, the life and the readings are the same as above.

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
width 64 (`fan_in_rates` in `width.py`). `--settings` adds
options, such as `{"learning_rehearsal": 1.0}`.

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
PYTHONPATH=src python benchmarks/sequence/battery.py --task periodic \
    --task-args '{"periods": [40]}' --arms recipe recipe64 --out DIR/period40.jsonl --workers 3
python benchmarks/sequence/battery.py --summary DIR/lookback.jsonl
PYTHONPATH=src python benchmarks/sequence/order.py --arms founder position_address \
    --out DIR/order.jsonl --workers 5
PYTHONPATH=src python benchmarks/sequence/width.py --task induction --memory --arms relative \
    --widths 1024 --out DIR/width.jsonl
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

**A period of 40** (#232 asks at least .85). At the founder's `recall_position` of 32 the ages
from 32 share one bucket per doubling, so no term singles out the slot one period back. With the
table exact to 64 ages (`recipe64`) development founders 0 to 2 answer 0.931, 0.941 and 0.912. A
confirmation declared before any run on founders 232200 to 232205 (gates, each on at least 5 of 6
founders: P = 40 at least .85, look back taught at 5 to 12 and read at 24 to 60 `first` at least
.90, copying unique tokens taught at 3 to 6 and read at 8 to 12 at least .95, closed loop):

| Founder | P = 40 | Look back `first` | `second` | `last` | Copying unique tokens | Teacher-forced |
| --- | --- | --- | --- | --- | --- | --- |
| 232200 | 0.937 | 0.998 | 0.998 | 0.998 | 0.999 | 1.000 |
| 232201 | 0.941 | 1.000 | 0.998 | 0.985 | 0.999 | 1.000 |
| 232202 | 0.907 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| 232203 | 0.941 | 1.000 | 0.994 | 0.994 | 0.986 | 0.998 |
| 232204 | 0.858 | 0.981 | 0.947 | 0.968 | 0.999 | 1.000 |
| 232205 | 0.943 | 0.998 | 0.998 | 1.000 | 0.999 | 1.000 |

Every gate passes on 6 of 6.

**The anchor's confirmation**, declared before any run on founders 300 to 305 (look back
taught and read at 24 to 60, the cues mixed): `first` 0.998 to 1.000 and `second` 0.990 to
1.000 with the anchor, 0.310 to 0.451 and 0.312 to 0.342 without (`boundary`); `last` 0.989 to
1.000 in both. The battery at its defaults, mean over the six founders, without and with the
anchor: induction 1.000 and 1.000, n-back 1.000 and 0.999, periodic continuation 0.988 and
1.000, copying 1.000 and 1.000.

### The panel of position changes (#232)

The 0.90 default search declared a panel for any change of the position term. It is read here
against the founder memory (`recall=True` at its founders: 32 exact ages taught at 0.1, recency
0.25), each instrument with the lessons that search declared for it: surprise lessons for this
battery and for the two module-32 benchmarks, every row for the Stone Age worlds and the
composer. The parts:

- this battery, founders 0 to 3, 2,000 lessons (1,000 for the period of 24): look back taught at
  5 to 12 and read at 13 to 24, copying unique tokens taught at 3 to 6 and read at 8 to 10 in
  closed loop, n-back with k from 1 to 4, periodic continuation at a period of 24, two hops with
  a think moment;
- look back with the cues `first` and `last`, and copying unique tokens of 16, as the held 0.90
  benchmarks taught them (module 32, 600 lessons, read at 13 to 24 and at 8 to 10), founders 0
  to 3;
- the ordered recall world taught (1,536 episodes) and the hidden route cue guided (480
  episodes) of the Stone Age sequence instrument, founders 1001 to 1010, frozen sampled success;
- a composer of C64 tunes outside the library (128 neurons, a window of 4, 24,576 moments, seeds
  0 and 1): the held-out read's accuracy and the answer's nats per voice-row.

A change keeps the panel when look back `first` and copying at unseen lengths reach 0.90, no
probe of this battery falls more than 0.03 below the founder memory (means over founders), the
Stone Age worlds stay within 0.03 (the founder-paired mean), and the composer's read stays
within 0.005 and its answer within 0.03 nats (means over seeds).

| Reading | Founder memory | `recipe` | `recipe64` |
| --- | --- | --- | --- |
| Look back `first` at 13 to 24 | 0.226 to 0.281 | 0.998 to 1.000 | 0.993 to 1.000 |
| Copying at 8 to 10, closed loop | 0.056 to 0.074 | 0.999 to 1.000 | 0.999 to 1.000 |
| N-back, k from 1 to 4 | 1.000 | 0.996 to 0.999 | 0.996 to 0.999 |
| A period of 24 | 0.910 to 0.930 (mean 0.920) | 0.792 to 0.883 (0.834) | 0.728 to 0.946 (0.837) |
| Two hops with a think moment | 0.975 to 0.993 | 0.996 to 0.999 | 0.995 to 0.998 |
| Module 32: look back `first` at 13 to 24 | 0.205 to 0.269 | 0.967 to 0.994 | 0.967 to 0.994 |
| Module 32: copying at 8 to 10 | 0.080 to 0.119 | 0.992 to 0.999 | 0.992 to 0.999 |
| Stone Age ordered recall, taught | 0.996 | 0.972 (paired -0.025) | 0.972 (paired -0.025) |
| Stone Age hidden route, guided | 0.428 | 0.246 (paired -0.181) | 0.246 (paired -0.181) |
| Composer read accuracy, seeds 0 and 1 | 0.819, 0.823 | 0.798, 0.775 | 0.794, 0.810 |
| Composer answer nats, seeds 0 and 1 | 1.443, 1.492 | 1.542, 1.685 | 1.561, 1.561 |

Neither keeps the panel. Both read at unseen lengths, and n-back, two hops and ordered recall
stay within their bars (ordered recall loses one founder of ten, 0.750 against 0.996); both cost
the period of 24 (0.08 on the mean), the hidden route (0.18; the route drops on six founders of
ten and rises on four) and the composer (read 0.035 and 0.019 lower, answer 0.15 and 0.09 nats
higher). No slot of the module-32 benchmarks or the Stone Age worlds reaches an age of 32, and
there the two arms answer alike.

Where the costs come from, one part of the recipe at a time (development readings after the
panel; no gate):

| Change | Reading | Founder memory | `recipe` | With the change |
| --- | --- | --- | --- | --- |
| The table at 0.1 | hidden route, founders 1001 to 1010 | 0.428 | 0.246 | 0.490 |
| Recency 0.25 | hidden route | 0.428 | 0.246 | 0.296 |
| No anchor | hidden route | 0.428 | 0.246 | 0.338 |
| No slope | composer read, answer nats (seeds 0 and 1) | 0.821, 1.468 | 0.786, 1.613 | 0.815, 1.464 |
| The slope alone, on the founder memory | composer read, answer nats | 0.821, 1.468 | | 0.809, 1.522 |
| No slope | look back `first` at 24 to 60 (founders 0 to 2) | | 0.998 to 1.000 | 0.568 to 0.662 |
| No anchor (`slope_slow`) | a period of 24, founders 0 and 1 | 0.923, 0.917 | 0.804, 0.883 | 0.444, 0.788 |
| The anchor with the table at 0.1, no slope (`anchor`) | a period of 24, founder 0 | 0.923 | 0.804 | 0.858 |
| Recency 0.25 | a period of 24, founder 0 | 0.923 | 0.804 | 0.837 |
| No boundary | a period of 24, founder 0 | 0.923 | 0.804 | 0.745 |
| 3,000 lessons | a period of 24, founders 0 and 1 | 0.923, 0.917 | 0.804, 0.883 | 0.888, 0.849 |

The route's cost is the slower table, the composer's the slope; the period of 24 has no single
cause. Each part pays elsewhere: the slope carries look back at unseen lengths, and the table at
0.01 keeps copying at unseen lengths (0.99 to 1.00 with the slope beside it, 0.47 to 0.88 with
the table at 0.1).

### A read placed where an earlier read pointed (#234)

Copying repeated tokens (`repeat` with `unique` off) taught at 3 to 6 and read at 8 to 12:

| Arm | Founders | Closed loop | The first token after `GO` | Teacher-forced |
| --- | --- | --- | --- | --- |
| `anchor` | 0 to 2 | 0.099 to 0.138 | 0.102 to 0.236 | 0.241 to 0.361 |
| `anchor_relative` | 0 to 2 | 0.544 to 0.741 | 0.625 to 0.803 | 0.870 to 0.932 |
| `recipe_relative` | 232100 to 232105 | 0.918 to 0.941 | 0.967 to 0.992 | 0.963 to 0.976 |

Two hops asked in one moment (`twohop` without `think`; founders 0 to 2): `heads` 0.436 to
0.469 after 1,500 lessons and, with the copy pathway (`heads_copy`, which left with the pathway in
0.96.0), 0.473 to 0.502 after 3,000.
A chained second head (its address taking what the first head read), an option tried on the
release branch and withdrawn before release, read 0.439 to 0.458 beside `heads` and 0.506 to
0.533 beside `heads_copy` (0.03 higher on each founder). With a `THINK` moment the two heads
answer the second hop at 0.983 to 0.990.

### When a read item happened (#237)

Founders 0 to 2, every row taught:

| Arm | "How long ago was X told?" | "Was A told before B?" |
| --- | --- | --- |
| `every_row` | 0.509 to 0.796 | 0.594 to 0.718 |
| `when` | 0.809 to 0.907 | 0.594 to 0.796 |
| `when_anchor` | 0.911 to 0.917 | 0.590 to 0.803 |
| `when_sharp` | 0.861 to 0.951 | 0.944 to 0.958 |

**The read that stays on its items** (`when_sharp`). The position lesson, and to a lesser degree
the address lesson, take their target from the slots whose stored contents the lesson's
movement points to; "was A told before B?" is answered yes or no, which no slot holds, and the
lessons pull the read off the items. At the founder settings the read gives A's slot a median
0.008 of its attention at A and B's slot 0.0005 at B (founder 0; with the address and position
lessons off, 0.83 and 0.82). With the position term not taught and a sharp softmax the read
lands on the item (median attention 0.999 on A's slot at A and on B's at B), and the working
trace carries A's report to B's moment. Freezing the address map as well (`when_sharp` with
`recall_address=0.0`) reads 0.927 to 0.957 on founders 0 to 2. Development founders 0 to 5 and
fresh founders 237050 to 237055, every row taught, 1,500 lessons:

| Founder | "Before" `when_sharp` | "How long ago" `when_sharp` | "How long ago" `when_anchor` |
| --- | --- | --- | --- |
| 0 | 0.944 | 0.951 | 0.911 |
| 1 | 0.958 | 0.861 | 0.917 |
| 2 | 0.949 | 0.898 | 0.911 |
| 3 | 0.947 | 0.898 | 0.817 |
| 4 | 0.942 | 0.876 | 0.933 |
| 5 | 0.953 | 0.904 | 0.854 |
| 237050 | 0.938 | 0.953 | 0.685 |
| 237051 | 0.966 | 0.953 | 0.783 |
| 237052 | 0.949 | 0.880 | 0.899 |
| 237053 | 0.963 | 0.866 | 0.849 |
| 237054 | 0.958 | 0.894 | 0.853 |
| 237055 | 0.946 | 0.928 | 0.696 |

`when_anchor` reads "before" 0.587 to 0.793 on the fresh founders. "How long ago" with
`when_sharp` averages 0.898 on the development founders and 0.912 on the fresh ones, against
0.891 and 0.794 with `when_anchor`; its misses lie in the middle buckets, 2 to 3 and 4 to 7
moments back.

Founders 237050 to 237055 were declared before any run on them, as the simpler control of a
protocol whose mechanism was withdrawn: a region per head carrying its previous read of when to
the next moment read "before" 0.969 to 0.983 on them, 0.006 to 0.040 above `when_sharp`, and
0.599 to 0.843 at the founder read (founders 0 to 2), where the read leaves the items; it is not
in the library.

**A boundary of the when regions.** Beside the recipe for unseen lengths (`recipe` with
`recall_when=True`), look back taught at 5 to 12 and read at 24 to 60 reads 0.996, 0.999 and
0.992 on founders 0 to 2 and 0.926, 0.996 and 0.996 on 237050 to 237052; on 237050 `first`
falls from 0.998 to 0.899 and `second` from 0.988 to 0.899 (`recipe` alone reads 0.984 there).
The regions' age features at ages no lesson reached (`log(1 + age) / log 9` reaches 1.87 at age
60 and 1.17 at age 12, the oldest a lesson taught) enter the association through synapses
taught at short ages.

### Cues taught in sequence (#238)

Look back at its defaults, 1,800 lessons: taught mixed (`anchor` without phases) every cue reads
0.998 to 1.000 on founders 0 to 5. Taught `first`, then `last`, then `second`, 600 lessons each:

| Arm | Founders | `first` | `last` | `second` |
| --- | --- | --- | --- | --- |
| `anchor` | 0 to 5 | 0.324 to 0.739 | 0.583 to 0.949 | 1.000 |
| `rehearsal` | 0 to 5 | 0.994 to 1.000 | 0.968 to 1.000 | 1.000 |
| `anchor`, 3,600 lessons (1,200 each) | 0 to 5 | 0.400 to 0.791 | 0.491 to 0.900 | 1.000 |

The rehearsal hears each kept lesson's episode again on fresh streams and teaches its last
moment, so the read is computed and its maps taught under the present parameters: every earlier
cue keeps at least 0.968, against at most 0.949 in sequence without it. The last arm gives the
lessons and the replayed rows (16 a lesson) as ordinary lessons in sequence and keeps no more than
1,800 lessons in sequence do. In a paired run on one host (founder 0, 900 lessons) a rehearsed
life took 3.1 times as long. The `anchor` lives of founders 0 to 2 repeat the release re-reading
exactly; the `rehearsal` lives were read on the development commit of the replay, the mechanism
as released here.

**The rehearsal's confirmation**, declared before any run on founders 238400 to 238405 (gates: G1
`first` and `last` under `rehearsal` each at 0.9 of the founder's interleaved reading, G3 `second`
the same, each on at least 5 of 6 founders; G2 the matched lessons, 3,600 in sequence, meeting
G1's condition on at most 1 of 6):

| Founder | Interleaved | `rehearsal` `first` | `last` | `second` | 3,600 lessons `first` | `last` | `anchor` `first` | `last` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 238400 | 1.000 | 0.998 | 1.000 | 1.000 | 0.760 | 0.892 | 0.580 | 0.868 |
| 238401 | 1.000 | 1.000 | 1.000 | 1.000 | 0.571 | 0.891 | 0.605 | 0.843 |
| 238402 | 1.000 | 1.000 | 0.993 | 1.000 | 0.845 | 0.758 | 0.777 | 0.890 |
| 238403 | 1.000 | 1.000 | 1.000 | 1.000 | 0.585 | 0.780 | 0.735 | 0.890 |
| 238404 | 1.000 | 0.998 | 1.000 | 1.000 | 0.558 | 0.726 | 0.474 | 0.656 |
| 238405 | 1.000 | 0.994 | 0.984 | 1.000 | 0.420 | 0.571 | 0.433 | 0.398 |

Interleaved every cue reads 1.000 on every founder. G1 and G3 pass on 6 of 6, and G2 holds: no
founder keeps both earlier cues at 0.9 with the matched lessons. The newest cue, `second`, reads
0.996 to 1.000 in every arm. A form of the rehearsal that kept each row's drive instead of its
episode, tried on the release branch and withdrawn before release, kept first 0.286 to 0.458 and
last 0.458 to 0.700 on these founders (0.285 to 0.437 and 0.432 to 0.589 on founders 0 to 2):
it taught stored drives again, as the read gave them when they were stored, and the cues held by
where the read looks lost more with it than without it.

### Ordered recall (#236, #232)

3,072 episodes under reward through `live`. The share of the life's answers paid over its last
256 episodes, and the founders at .90 or above, with the forks' greedy and sampled readings.
Development founders 1001 to 1012. The reward step moves the rows of the event heard; stepping
every row, the form tried on the release branch and withdrawn before release, is kept here as
the receipt of the withdrawn form (the rows marked "every row"; the `address` arms were read
with it, on the first five founders):

| Arm | Founders | At .90 | The life's last 256 answers | Greedy | Sampled |
| --- | --- | --- | --- | --- | --- |
| `founder` | 1001 to 1005 | 0 of 5 | 0.46 to 0.50 | 0.50 | 0.47 to 0.51 |
| `position` (`recall_position_reward=4`) | 1001 to 1012 | 8 of 12 | 0.46 to 1.00 | 0.50 to 1.00 | 0.50 to 0.92 |
| `position`, every row (withdrawn) | 1001 to 1010, 1012 | 5 of 11 | 0.47 to 1.00 | 0.50 to 1.00 | 0.50 to 0.91 |
| `address`, `position_address`, `position_address_anchor`, every row | 1001 to 1005 | 1, 3, 2 of 5 | 0.47 to 1.00 | 0.50 to 1.00 | 0.50 to 0.97 |
| `position` with `recall_anchor=True` | 1001 to 1012 | 7 of 12 | 0.46 to 1.00 | 0.50 to 1.00 | 0.50 to 0.87 |
| `position` with the anchor and `recall_next=False` | 1001 to 1012 | 12 of 12 | 1.00 | 1.00 | 0.50 to 0.92 |
| `reward_recipe` | 1001 to 1012 | 11 of 12 | 0.48 to 1.00 | 0.50 to 1.00 | 0.50 to 0.89 |

Stepping every row, in the lives short of .90 one query kind answered every query with one
value: the position term reads the cortex of the moment before and a bias, which both queries
share (they follow the same `NEUTRAL`), and the step one query's reward took there moved the
other's read with it. The event's rows alone end that; then one condition of four can stay
unanswered, where the gap moves the first write from two moments old to three, or where the
write is read through the next port of the slot before it in one condition and through its own
value in the other. The anchor reads the first write at the same place in either condition, and
without the next port each payload is held in one place. In the recall world (below) that read
answers the latest of two writes at 0.21 (the founder 0.67): the query learns the episode's
first write, the only one where one value is written. The address reward gives the read the
writes by what they hold, and the recency the latest of them (`reward_recipe`).

**Confirmation**, declared before any run on founders 936001 to 936012 (gates: G1
`reward_recipe` at .90 or above on at least 10 of 12; G2 its recall world reading no lower than
the founder's by .05; G3 the preservation panel):

| Arm | At .90 | The life's last 256 answers | Greedy | Sampled |
| --- | --- | --- | --- | --- |
| `founder` | 0 of 12 | 0.45 to 0.52 | 0.50 | 0.47 to 0.55 |
| `position`, every row (withdrawn) | 6 of 12 | 0.52 to 1.00 | 0.50 to 1.00 | 0.53 to 0.96 |
| `reward_recipe` | 11 of 12 (G1 passes) | 1.00 on 11, 0.46 on one | 0.50 to 1.00 | 0.50 to 0.93 |

The life at chance (as development founder 1012) answers action 0 on nine queries in ten while
it samples, and every greedy answer is 0: the read's eligibility names the slot that holds the
sampled action's value, the first write or the latest as the episode's values fall, so it has
no direction.

The recall world (`--world recall`) on the same founders, the life's last 256 answers (mean and
range) and the greedy reading per condition, the mean over founders:

| Arm | Last 256 | One write: clean 0, 1, 2 | Distractors 1, 2 | Replacement | Order-latest |
| --- | --- | --- | --- | --- | --- |
| `founder` | 0.864 (0.57 to 1.00) | 0.96, 1.00, 0.92 | 0.95, 0.90 | 0.83 | 0.58 |
| `position`, every row (withdrawn) | 0.916 (0.82 to 1.00) | 0.92, 1.00, 0.83 | 1.00, 0.89 | 0.96 | 0.83 |
| `reward_recipe` | 0.848 (0.66 to 1.00; G2 passes, paired -0.015) | 0.96, 1.00, 0.96 | 1.00, 0.96 | 0.75 | 0.38 |

The reward recipe answers the ordered world and the recall world's single writes; of two writes
in the recall world it answers the latest less often than the founder. G3 fails: in the reward
chambers of [the preservation panel](../credit/README.md#preservation), with the recipe and the
memory it needs against the chambers as they are on the same seeds, the odour nursery passes 9
lives of 12 against 11 through `live` and 4 against 7 through `step`, and the key-door nursery
acquires its first rule in 5 lives of 16 against 8 (the panel breaks a chamber at a fall of
two). The memory alone, at its founder genes and read without reward, acquires that rule in 3
of 16, so the key-door's fall comes with the memory; the odour nursery keeps its lives with the
memory alone (10 and 7) and with the reward read in the memory as composed (10 and 8), and
loses them where the anchor and the missing next port meet the reward read.

Taught by a lesson at the query (1,536 episodes), the founder table answers every query
greedily (sampled 0.97 to 0.99). The table at 0.01 alone (`slow`) answers 0.50 to 1.00 greedily
(mean 0.80; 0.875 at 3,072 episodes), and the recipe of the battery (`recipe`) 0.875 to 1.00
(mean 0.975): the slower table costs reads that need exact ages, and the anchor and the slope
beside it give most of it back.

### The width sweep (#239)

Induction over 128 tokens at 1,024 neurons with the recall memory and surprise lessons, each
synapse's step scaled to its neuron's fan-in at width 64 (`relative`), 1,500 lessons, held-out
accuracy every 100 lessons:

`learning_surprise_share` left the library in 0.96.0 (#281): a new brain steps at its width
([wide brains](../../docs/tune.md#wide-brains)), and on the composition's battery the share cost
copying, induction and n-back (#258). Its readings at 1,024 neurons:

| Founder | Without the share | With `learning_surprise_share=True` |
| --- | --- | --- |
| 0 | | 0.12, 0.65, 0.76, 0.85, then 0.97 to 0.99 from 500 lessons; 0.99 at the end |
| 1 | swings: 0.42 at 500, 0.65 at 900, 1.00 from 1,200 | 0.96 to 0.99 from 500 (0.90 at 800); 0.99 at the end |
| 2 | swings: 0.01 at 400 and 500 and at 1,000, 0.18 at 1,300, 0.98 at the end | 0.85 to 0.95 from 500; 0.95 at the end |

A confirmation declared before the run on six fresh founders (239300 to 239305): with the share
every life ends at 0.966 to 0.992 (the gate, .90 on at least 5 of 6: passed), and four of the six
keep every reading from 500 lessons on above 0.80 (the gate, 5 of 6: failed), founder 239300
dipping once to 0.18 and 239304 to 0.77 before they recover. Without the share, on the same
founders, every life ends at 0.935 to 0.992 and every one collapses at least once after 500
lessons, to 0.013 to 0.294.

No act was refused in these lives. The life that refused an act at lesson 343 before the settle
rescue existed (induction at 64 neurons on the composed steps, founder 0) runs its 1,500
lessons: two answers are rescued, none is refused, and it reads 0.64, 0.82, 0.93 and 0.97
held-out at 250, 500, 1,000 and 1,500 lessons. The 0.92.0 sweep of the
fan-in steps (lookup 0.77 at 64 and 0.92 at 1,024
neurons on the founder steps, 0.82 to 0.87 with the fan-in steps; associative recall 0.73 to 0.84
at 1,024 on the founder steps, 1.00 with the fan-in steps; two founders) is `--task lookup` and
`--task associative --memory` with `--arms founder relative --widths 64 128 256 512 1024`; its
lives at 1,024 neurons repeat on 0.93 exactly (checked at 250 lessons).
