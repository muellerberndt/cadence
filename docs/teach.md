# Teach by demonstration

You want to teach by demonstration: show the brain the act for a situation and let it learn
it. `brain.lesson(observations, labels)` teaches the shown act on the present observation, and
`brain.step(observations, teacher=labels)` folds a lesson into a moment of the stream. With few
lessons the founder teaching point acquires fastest; with many lessons, as in the imitation of a
demonstrator, `teaching="normalized"` holds more of what was shown
([many lessons](#many-lessons-the-normalized-point-for-imitation)). Teach on the brain's own
path when its mistakes take it where the demonstrator never went
([below](#teach-on-the-brains-own-path)). A lesson changes the graph's weights; the episodic
memory (`memory_choice=True`) also records the shown act for its situation
([memory](memory.md#episodic-memory)).

## Explicit learning with `step`

`step(observations, *, reward=None, done=None, teacher=None, salience=None,
bootstrap=None)` learns from the preceding outcome, teaches the label of the present
observation, then samples and returns the next action.

```python
import numpy as np
import cadence as cd
from cadence import Brain

cues = np.eye(4)  # four one-hot observations
observation, following = cues[[0]], cues[[1]]

stepped = Brain.compose(4, 2, modules=(16, 8), seed=7)
action = stepped.step(observation)
reward = (action == 0).astype(float)   # the body pays action 0
action = stepped.step(following, reward=reward, done=np.array([False]))
stepped.act(following, greedy=True)              # a teacher acted instead; nothing to credit
shown = stepped.lesson(following, np.array([1]))  # teach the shown action
assert stepped.last_learning["demonstrations"] == 1.0
```

| Argument | Describes | Notes |
| --- | --- | --- |
| `reward` | the preceding action | omitted is a real zero-reward transition, never an unknown outcome |
| `done` | the preceding action | for an ended row pass the next episode's reset observation |
| `bootstrap` | an ended row | per-row value of a truncated episode's last observation; omitted, an ended row bootstraps from zero |
| `salience` | the preceding outcome | nonnegative per row; defaults to `abs(reward)` for consolidation |
| `teacher` | the present observation | changes graph parameters |

Call `step` once per real transition. `act(observations, *, greedy=False,
temperature=None)` and `learn(reward, done, next_observations, *, bootstrap=None,
salience=None)` split the moment when the body needs that timing
([by hand](tune.md#one-experience-step-by-hand)). Teach from actual labels; the
brain's own guesses can reinforce mistakes. A `teacher` label writes no associative
record, except under `memory_choice=True`, where it is recorded and zero outcomes in
the middle of an attempt write nothing ([episodic memory](memory.md#how-it-works)).

When a teacher's action ran in place of the issued one, as at the end of the example,
replace the issued action with a greedy `act`, which learns nothing from it, and teach
the shown action with `lesson(observations, labels)`; the next `step` takes no reward.
A reward reported for the shown action would credit an action that did not run.
`lesson` reads every memory the act reads, keeps its report in `last_learning` under
`demonstration_*` keys and refuses while a sampled action awaits its outcome. In a brain with
a recall memory an act writes the moment to it, so a lesson after an act on the same
observations reads the moment itself, its successor still empty, and warns: teach first,
then act, where the order is yours to choose.

## A supervised-only stream

`step(teacher=...)` issues a reward-eligible action, so the next `step` without a
reward teaches the actor and critic a zero outcome. For demonstrations with no reward
protocol, read context with greedy actions and teach the learner on the memory-bearing
drive of `stimulus(observations, *, memory=True)`:

```python
supervised = Brain.compose(4, 2, modules=(16, 8), seed=7)
supervised.act(observation, greedy=True)   # advance context without reward eligibility
_, lesson_work = supervised.learner.step(supervised.stimulus(following), np.array([1]))
assert supervised.act(following, greedy=True).shape == (1,)
```

Direct learner calls leave `Brain.last_learning` untouched; keep their report. Finish
any outstanding real outcome before a reward-driven life switches to greedy actions.

## Lessons only where the answer differs

`LearnerConfig(surprise=True)` (`learning_surprise=True` when composing) teaches only the rows
whose answer differs from the label. A standalone learner judges by its free phase, which runs
on every row; agreeing rows get no nudged phase and no update. A caller holding the answer passes
`step(drive, labels, answer=...)`, and agreeing rows are dropped before the free phase. The
composed brain passes the act it would issue, warm from the stream; with an
[episodic vote](memory.md#one-choice-for-acting-teaching-and-imagining) that act includes the
vote, so rows the memory answers are never taught to the cortex. The report adds
`agreed_presentations`, `LearnedState.taught` marks the taught rows, and an episodic memory
records only those demonstrations.

Select it where a readout saturates under lessons it answers correctly, or where streams are
shown row by row for many passes. The founder is `False`;
[benchmarks/stream_words](../benchmarks/stream_words/README.md) uses it with every-row lessons as
the control.

Under the [normalized point](#many-lessons-the-normalized-point-for-imitation) select it from the
first lesson of a long curriculum. Normalized steps do not shrink as an answer becomes right, so
lessons on rows the brain already answers keep driving the readout until the nudge no longer
moves it; a lapse then leaves the answers on one action with no contrast left to undo it, and the
reports show `scale_step` falling toward zero while answers stay wrong. On the two-fact
conjunction task of the Stone Age attention series (V0.5B: two heads, lessons of 16 rows, every
row taught until the running wrong share falls below .02, then only the wrong rows, as the
[several-heads guide](memory.md#several-heads-opt-in) advised), 3 of 5 founders answered most
taught pairs and then collapsed onto one action within 4,096 lessons, two of them without ever
reaching the switch; on one founder the nudge's movement of the motor population fell a
hundredfold. With surprise from the first lesson all five answered every taught pair from about
1,024 lessons on and .72 to .96 of the held-out pairs (mean .85 against .53), and erasing either
relevant fact still dropped the held-out answers to chance
([EXP-72](https://github.com/WGlynn/cadence-teachers/blob/main/experiments/EXP-72.md), Will
Glynn). Founder 59201 reads 1.00 / .88 against .39 / .31 on the series' runner.

**In proportion to the surprise (opt-in, `learning_surprise_share=True`).** A surprise lesson's
contrast is read over the rows it teaches, so a lesson with one wrong row of sixteen takes as
large a step as a lesson with sixteen. Under the normalized point that step does not shrink as
the answers become right, and a few wrong rows can undo what the others hold. With
`surprise_share` the step is scaled by the share of the lesson's rows it taught: three wrong rows
of eight move the net three eighths as far as eight would. The contrast and the optimizer's
running moments still read the taught rows alone. On induction over a vocabulary of 128 at
width 1,024 in the [width sweep](../benchmarks/sequence/README.md#the-width-sweep-widthpy) (the
recall memory, the normalized point, each synapse's step scaled to its neuron's fan-in at width
64 through `Learner.synapse_rate`, three seeds, 1,500 lessons) the held-out
answers end at 0.95 to 0.99 with it and stay at or above 0.85 at every reading from 500 lessons
on. Without it the readings swing: one seed reads 0.01 after 1,000 lessons, 0.18 after 1,300 and
0.98 at the end. On six fresh founders declared before the run, every life with the share ends
at 0.97 to 0.99; four of the six keep every reading from 500 lessons on above 0.80, and two dip
once (to 0.18 and to 0.77) before they recover; without the share every one of the six collapses
at least once after 500 lessons (to 0.01 to 0.29) and still ends at 0.93 to 0.99. The share
removes the collapse at the end everywhere and the collapses on the way on four of six. Teaching every row (the sweep with `learning_surprise` off) holds 0.92 to 0.99
from 800 lessons on at about a third more time per lesson. Scaling the contrast instead, so
that the running moments read it as
well, swings as the plain rule does: the normalization divides the scale out
([#239](https://github.com/muellerberndt/cadence/issues/239)).

## Acquisition, working memory and durable retention

Acquisition, working-memory recall and durable retention are separate measurements: `predict`
and `accuracy` test the graph, `act` the graph with its memories
([check that a memory does the work](memory.md#check-that-a-memory-does-the-work),
[evaluate a brain](evaluate.md)).

### Teach several types in sequence: rehearsal

Lessons on a new type move the same synapses the earlier types' answers rest on, so teaching types
one after another can erase the earlier ones. `learning_rehearsal=r` (opt-in, founder 0) keeps a
reservoir of the lessons' rows, a uniform sample of every lesson row
(`learning_rehearsal_capacity`, 512 rows), each row with its episode: the moments its stream heard
since the episode began (at a `reset`, or where a `done` ended the stream's attempt), at most as
many as the recall memory keeps slots (its `capacity`, 64 at the founder; 64 without a recall
memory), then the moment it was taught on. After each lesson `r` times the lesson's rows (at least
one) are drawn and replayed: the rows of equally long episodes together on fresh streams, as after
`reset`, which hear every moment before the last as a greedy `act` hears it (a learned code takes
no step on them) and are taught at the last as a lesson teaches, under the same rule, so with
`learning_surprise=True` only the rows the brain now answers wrong move it. The memories are read
again under the present parameters, and a recall read's maps take their lesson with the cortex.
Every stream then returns to the state it was in, the recall memory's slots, `told` and `focus`
among it, so the live streams continue as they would have without the replay, under the parameters
it taught. A replayed lesson writes no demonstration to the episodic records, and a refused act or
lesson ends the replay (`rehearsal_refused` in `last_learning`) while the lesson that called it
stands. An episode begins on fresh streams, so slots that `reset(keep=("slots",))` carried over
from an earlier episode are not replayed with it
([#238](https://github.com/muellerberndt/cadence/issues/238)).

The reservoir (`Brain.rehearsal_moments` and `rehearsal_labels`, with `rehearsal_seen` and
`rehearsal_draws`) and the streams' episodes are saved with the brain, the reservoir also while
rehearsal is retuned off. A rate retuned above zero keeps the streams' moments from then on, so an
episode in progress is replayed from the moments heard after the retune; a lowered
`learning_rehearsal_capacity` cuts the reservoir at once; a grown brain starts the reservoir
afresh, a grafted one keeps this brain's. The capacity counts lesson rows: at 512, up to 512
episodes of up to 64 moments each.

```python
brain = Brain.compose(7, 4, seed=0, recall=True, learning_surprise=True, learning_rehearsal=1.0)
assert brain.learner.config.rehearsal == 1.0 and not brain.rehearsal_moments
```

Compare it with the types taught interleaved (the upper control) and in sequence without rehearsal
(the lower control) at matched lesson counts. On look back's three cues taught in sequence (first,
then last, then second, 600 lessons each; the [sequence
battery](../benchmarks/sequence/README.md#cues-taught-in-sequence-238), six development founders)
the rehearsal keeps the earlier cues at 0.994 to 1.000 (first) and 0.968 to 1.000 (last), and the
newest reads 1.000; taught interleaved every cue reads 0.998 to 1.000. In sequence without it first
reads 0.32 to 0.74 and last 0.58 to 0.95, and 3,600 lessons in sequence, the lessons and the
replayed rows given as ordinary lessons, leave 0.40 to 0.79 and 0.49 to 0.90: more lessons alone do
not keep the earlier cues. On six fresh founders declared before the run the rehearsal keeps first
at 0.994 to 1.000 and last at 0.984 to 1.000, every cue reading 1.000 interleaved; 3,600 lessons in
sequence keep 0.42 to 0.85 and 0.57 to 0.89. In a paired run on one host a life with the rehearsal
took 3.1 times as long as without it: the replayed rows settle in small batches, one per episode
length. A form that kept each row's drive instead of its episode, as the read gave it when the row
was kept, tried on the release branch and withdrawn before release, kept types the cortex holds but
cost the cues the read holds (first 0.286 to 0.458 and last 0.458 to 0.700 on the same fresh
founders): the stale reads taught the cortex, and the read's own maps were never taught again.

## Rates under normalization

With `normalize > 0` each plastic efficacy and bias keeps a running second moment of its raw
signal. Before masks, synapse rates, tying and clipping, the increment is

```text
increment = rate * signal / (sqrt(bias_corrected_second_moment) + floor)
```

with `rate` = `eta` or `eta_bias`, `signal` the raw contrast (its bias-corrected mean under
`momentum`; the second moment always reads the raw contrast) and `floor` = `normalize_floor`. A
first update is `rate * g / (abs(g) + floor)`, so a consistent signal well above the floor steps
by about the rate whatever its size; changing signals can step above it. Momentum with RMS is an
Adam-style update.

A constant floor is no reference for every synapse at once. Credit that reaches a region several
synapses from the output is orders of magnitude fainter than the readout's, so the floor that keeps
the readout's quiet synapses still swallows the deep region's whole signal. `normalize_relative`
above zero replaces it by a floor of that share of each synapse's projection (the block of synapses
between its two neuron ranges; a bias reads its range): every projection steps at about the rate on
its own scale, the faint one as the strong one, and a synapse quieter than its projection still
steps less. It needs `normalize` above zero; the teaching point `"deep"` sets it at 0.1 for
brains with the [apical compartment](anatomy.md#the-apical-compartment), the other points keep
`normalize_floor`, and `LearnerConfig.normalize_relative` sets it for an expert configuration.

The composed rates (`eta=0.5` teaching, 1.0 actor) are unnormalized; normalized at those rates,
steps can dwarf the raw signals and saturate the outputs. Retune `eta` and `eta_bias` together;
a development sweep might start at `eta` 0.001 to 0.003, checked against acquisition, retention
and actual step sizes. `LearnerConfig` warns when `normalize > 0` and `eta` or `eta_bias` exceeds
0.05; smaller rates can also fail. The actor steps on its raw contrast
([actor rates](reward.md#actor-rates)).

### Many lessons: the normalized point for imitation

For a brain that learns a demonstrator's policy from many lessons, compose it with
`teaching="normalized"` (the teaching learner's steps 0.003 and 0.0003 under a running RMS of
forgetting 0.99 and floor 1e-4, momentum 0.9) or move a living brain there with
`retune(teaching="normalized")`, and teach about 32 rows per lesson. In a copy measurement a composed
brain with one region of 256 neurons (`Brain.compose(260, 3, modules=(256,))`) learned the moves
of a three-move look-ahead demonstrator in a light-cycle game on a 24 x 24 grid: 260 inputs of an
egocentric view, three actions, about 124,000 labelled moments from 1,400 games, and 100 further
games held out. The moments came in random order through
`brain.learner.step(brain.stimulus(x, memory=False), labels)` and were scored cold with
`predict`. The reference is a network with one hidden layer of 256 trained by backpropagation
with Adam (rate 1e-3) on the same samples in the same order and batch size. Each arm ran two
seeds; the table gives the agreement with the demonstrator on the held-out moments after 240,000
samples.

| Learner | All held-out moves | Moves where the demonstrator turned |
| --- | --- | --- |
| backpropagation with Adam, batch 8 | 0.874 / 0.888 | 0.690 / 0.804 |
| composed founder (`eta=0.5`, unnormalized), batch 8 | 0.81 at 80,000 samples | 0.55 at 80,000 samples |
| `normalize=0.99`, `eta=0.003`, batch 32 | 0.868 / 0.866 | 0.740 / 0.722 |
| `normalize=0.99`, `eta=0.003`, 48 nudged steps, batch 32 | 0.869 / 0.863 | 0.760 / 0.678 |
| `normalize=0.99`, `eta=0.001`, batch 8 | 0.865 / 0.869 | 0.701 / 0.701 |

At 720,000 samples backpropagation reached 0.900 / 0.903 on all moves and the normalized brain
0.886. Training agreement equalled held-out agreement in every arm. With lessons of 8 rows the
rate 0.001 reached the same agreement; normalization at `eta=0.01` overshot.

The point is a recommendation for that regime and the composed founder keeps its rates. Past the
point where the answers are right, teach it with `learning_surprise=True`
([lessons only where the answer differs](#lessons-only-where-the-answer-differs)). With few
lessons the founder's larger steps acquire faster: in an instrument that teaches a chain of steps
from a few hundred corrections, the normalized point lowered the taught chain's success, and in
the token-binding chamber it lowered the held-out answers of the content-memory arms at long
budgets ([issue 215](https://github.com/muellerberndt/cadence/issues/215),
[issue 218](https://github.com/muellerberndt/cadence/issues/218)). Measure both on the task.

The working trace lowers teaching fidelity at its composed amplitude. In a continuing stream
school of the copy task (32 streams watching the demonstrator, a greedy `act` and then a `lesson`
at each moment, the working trace on), the normalized point agreed with the demonstrator on 0.86
of all moves at 240,000 samples with a `working_memory_amplitude` of 0 or 0.3, and on 0.80 at the
composed amplitude of 3.0. Applications that bootstrap by imitation may lower
`working_memory_amplitude`.

A normalized founder for both composed learners was measured and declined in 0.73.1
([issue 131](https://github.com/muellerberndt/cadence/issues/131)): at `eta=0.003`,
`normalize=0.99` and `momentum=0.9` for teaching and the actor together, the reward stream's
re-adaptation after a contingency change reached 0.486 against its 0.9 contract. The point above
sets the teaching learner alone; the actor keeps its rates.

## Teach on the brain's own path

A brain that only watches a demonstrator learns the states the demonstrator visits. Once it
acts, its own small errors lead it to states no lesson covered, and the errors compound over an
episode. Labels on the states the brain itself reaches cover exactly those. The recipe is DAgger
(Ross, Gordon and Bagnell 2011). After a bootstrap by watching, the brain acts and its own move
is executed with `brain.act(observation, greedy=True)`, the demonstrator labels that same
observation, and `brain.lesson(observation, label)` teaches it. A greedy act holds nothing for
reward, so the lesson is legal; `lesson` refuses while a sampled action awaits its outcome. Each
batch row below is one stream:

```python
def observe(position):  # a one-hot position on a line of five cells
    return np.eye(5)[position]

def demonstrator(position):  # the policy the brain watches: 1 steps right, 0 steps left
    return (position < 2).astype(np.int64)

def move(position, action):
    return np.clip(position + np.where(action == 1, 1, -1), 0, 4)

pupil = cd.Brain.compose(inputs=5, actions=2, seed=0, teaching="normalized")  # the point above
position = np.full(32, 2)  # 32 streams, one per batch row, starting in the middle
for moment in range(32):  # bootstrap by watching: the demonstrator's moves are executed
    label = demonstrator(position)
    pupil.lesson(observe(position), label)
    position = move(position, label)
cells = np.arange(5)
assert (pupil.predict(observe(cells)) != demonstrator(cells)).any()  # cells it never saw

pupil.reset()  # new streams, starting anywhere
position = np.random.default_rng(0).integers(0, 5, 32)
for moment in range(32):  # then the brain's own path
    shown = observe(position)
    action = pupil.act(shown, greedy=True)  # its own move is executed; nothing awaits reward
    pupil.lesson(shown, demonstrator(position))  # the demonstrator labels the state reached
    position = move(position, action)
assert (pupil.predict(observe(cells)) == demonstrator(cells)).all()
```

Watching alone, the pupil copies the two cells the demonstrator visits and answers the others
wrongly; on its own path it reaches those cells and learns them. The labels must come from the
demonstrator the brain watched: in the measurement of
[issue 215](https://github.com/muellerberndt/cadence/issues/215), corrections from a different
policy after a bootstrap by watching lowered the agreement with the watched demonstrator. Teach
many rows at once at the normalized point [above](#many-lessons-the-normalized-point-for-imitation).
