# Teach by demonstration

Show the brain the act for a situation and it learns it. `brain.lesson(observations, labels)`
teaches the shown act on the present observation, and `brain.step(observations, teacher=labels)`
folds a lesson into a moment of the stream. A lesson changes the graph's weights; with
`memory_choice=True` the episodic memory also records the shown act for its situation
([episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before)).

A new brain with the memory teaches at `teaching="deep"` and only where its answer differs; a brain
without one (`recall=False`) at the founder point on every row. Name another point at `compose` or
with `retune`. Choose it by the number of lessons:

- With few lessons `teaching="founder"` acquires fastest.
- With many lessons, as in the imitation of a demonstrator, `teaching="normalized"` holds more
  of what was shown ([many lessons](#many-lessons-the-normalized-point-for-imitation)).
- When the brain's mistakes take it where the demonstrator never went, teach on its own path
  ([below](#teach-on-the-brains-own-path)).

Acquisition, working-memory recall and durable retention are separate measurements: `predict`
and `accuracy` test the graph, `act` the graph with its memories
([check that a memory does the work](memory.md#check-that-a-memory-does-the-work),
[evaluate a brain](evaluate.md)).

## Explicit learning with `step`

`step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None)`
learns from the preceding outcome, teaches the label of the present observation, then samples
and returns the next action.

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

Call `step` once per real transition. `act(observations, *, greedy=False, temperature=None)` and
`learn(reward, done, next_observations, *, bootstrap=None, salience=None)` split the moment when
the body needs that timing ([by hand](tune.md#one-experience-step-by-hand)). Teach from actual
labels; the brain's own guesses can reinforce mistakes. A `teacher` label writes no associative
record, except under `memory_choice=True`, where it is recorded and zero outcomes in the middle
of an attempt write nothing ([episodic memory](memory.md#how-it-works)).

When a teacher's action ran in place of the issued one, as at the end of the example, replace the
issued action with a greedy `act`, which learns nothing from it, and teach the shown action with
`lesson(observations, labels)`; the next `step` takes no reward. A reward reported for the shown
action would credit an action that did not run.

`lesson` reads every memory the act reads, keeps its report in `last_learning` under
`demonstration_*` keys and refuses while a sampled action awaits its outcome. In a brain with a
recall memory an act writes the moment to it, so a lesson after an act on the same observations
reads the moment itself, its successor empty, and warns. Teach first, then act, where the order
is yours to choose.

## Teach on the brain's own path

A brain that only watches a demonstrator learns the states the demonstrator visits. Once it acts,
its own small errors lead it to states no lesson covered, and the errors compound over an
episode. Labels on the states the brain itself reaches cover exactly those. The recipe is DAgger
(Ross, Gordon and Bagnell 2011):

1. Bootstrap by watching: teach the demonstrator's moves with `lesson` and execute them.
2. Let the brain act with `brain.act(observation, greedy=True)` and execute its own move.
3. Have the demonstrator label that same observation and teach it with
   `brain.lesson(observation, label)`.

A greedy act holds nothing for reward, so the lesson is legal; `lesson` refuses while a sampled
action awaits its outcome. Each batch row below is one stream:

```python
def observe(position):  # a one-hot position on a line of five cells
    return np.eye(5)[position]

def demonstrator(position):  # the policy the brain watches: 1 steps right, 0 steps left
    return (position < 2).astype(np.int64)

def move(position, action):
    return np.clip(position + np.where(action == 1, 1, -1), 0, 4)

pupil = cd.Brain.compose(inputs=5, actions=2, seed=0, teaching="normalized", recall=False)  # the point above
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
wrongly; on its own path it reaches those cells and learns them. Take the labels from the
demonstrator the brain watched: corrections from a different policy after a bootstrap by watching
lower the agreement with the watched demonstrator
([issue 215](https://github.com/muellerberndt/cadence/issues/215)). Teach many rows at once at the
normalized point [above](#many-lessons-the-normalized-point-for-imitation).

## Lessons only where the answer differs

`learning_surprise=True` at `compose` (`LearnerConfig(surprise=True)` for a learner) teaches only
the rows whose answer differs from the label. The founder is `False`, and a new brain with the memory
holds `True`.

- A standalone learner judges by its free phase, which runs on every row; agreeing rows get no
  nudged phase and no update.
- A caller holding the answer passes `step(drive, labels, answer=...)`, and agreeing rows are
  dropped before the free phase.
- The composed brain passes the act it would issue, warm from the stream. With an
  [episodic vote](memory.md#one-choice-for-acting-teaching-and-imagining) that act includes the
  vote, so rows the memory answers are never taught to the cortex.

The report adds `agreed_presentations`, `LearnedState.taught` marks the taught rows, and an
episodic memory records only those demonstrations.

Select it where a readout saturates under lessons it answers correctly, or where streams are
shown row by row for many passes; [benchmarks/stream_words](../benchmarks/stream_words/README.md)
uses it with every-row lessons as the control.

Under the [normalized point](#many-lessons-the-normalized-point-for-imitation), select it from the
first lesson of a long curriculum. Normalized steps do not shrink as an answer becomes right, so
lessons on rows the brain answers correctly keep driving the readout until the nudge stops
moving it. A lapse then leaves the answers on one action with no contrast left to undo it; the
reports show `scale_step` falling toward zero while answers stay wrong. Surprise from the first
lesson prevents that collapse (measured on a two-fact conjunction task,
[EXP-72](https://github.com/WGlynn/cadence-teachers/blob/main/experiments/EXP-72.md), Will Glynn).

### In proportion to the surprise

**Opt-in, `learning_surprise_share=True`.** A surprise lesson's contrast is read over the rows it
teaches, so a lesson with one wrong row of sixteen takes as large a step as a lesson with sixteen.
Under the normalized point that step does not shrink as the answers become right, and a few wrong
rows can undo what the others hold. With the share the step is scaled by the share of the
lesson's rows it taught: three wrong rows of eight move the net three eighths as far as eight
would. The contrast and the optimizer's running moments read the taught rows alone; the scale
applies to the normalized step, since the normalization would divide a scaled contrast out.

The share scales surprise lessons only; set it with `learning_surprise=True` at the normalized
point. On induction over a vocabulary of 128 at
width 1,024 in the [width sweep](../benchmarks/sequence/README.md#the-width-sweep-widthpy) (the
recall memory, the normalized point, each synapse's step scaled to its neuron's fan-in at width
64 through `Learner.synapse_rate`, 1,500 lessons, three development and six fresh founders):

- With the share every life ends at 0.95 to 0.99 held-out, and seven of nine keep every reading
  from 500 lessons on above 0.80; two dip once (to 0.18 and 0.77) and recover.
- Without it every fresh life collapses at least once after 500 lessons (to 0.01 to 0.29) and
  ends at 0.93 to 0.99.

## Teach several types in sequence: rehearsal

Lessons on a new type move the same synapses the earlier types' answers rest on, so teaching
types one after another can erase the earlier ones. `learning_rehearsal=r` (opt-in, founder 0)
replays earlier lessons alongside the new ones.

- **The reservoir.** A uniform sample of every lesson row (512 rows, a constant), each row with
  its episode: the moments its stream heard since the episode began (at a
  `reset`, or where a `done` ended the stream's attempt), at most as many as the recall memory
  keeps slots (its `capacity`, 128 in a new brain; 64 without a recall memory), then the moment it
  was taught on. The reservoir holds up to 512 episodes of up to that many moments each.
- **The replay.** After each lesson `r` times the lesson's rows (at least one) are drawn. The rows
  of equally long episodes replay together on fresh streams, as after `reset`. They hear every
  moment before the last as a greedy `act` hears it (a learned code takes no step on them) and are
  taught at the last as a lesson teaches, under the same rule, so with `learning_surprise=True`
  only the rows the brain answers wrong move it. The memories are read under the present
  parameters, and a recall read's maps take their lesson with the cortex.
- **The live streams.** Every stream then returns to the state it was in, the recall memory's
  slots, `told` and `focus` among it, and continues under the parameters the replay taught.
- **Limits.** A replayed lesson writes no demonstration to the episodic records. A refused act or
  lesson ends the replay (`rehearsal_refused` in `last_learning`) while the lesson that called it
  stands. An episode begins on fresh streams, so slots that `reset(keep=("slots",))` carried over
  from an earlier episode are not replayed with it.

The reservoir (`Brain.rehearsal_moments` and `rehearsal_labels`, with `rehearsal_seen` and
`rehearsal_draws`) and the streams' episodes are saved with the brain, the reservoir also while
rehearsal is retuned off. A rate retuned above zero keeps the streams' moments from then on, so an
episode in progress is replayed from the moments heard after the retune. A grown brain starts the reservoir
afresh, a grafted one keeps this brain's.

```python
brain = Brain.compose(7, 4, seed=0, recall=True, learning_surprise=True, learning_rehearsal=1.0)
assert brain.learner.config.rehearsal == 1.0 and not brain.rehearsal_moments
```

Compare it with the types taught interleaved (the upper control) and in sequence without
rehearsal (the lower control) at matched lesson counts. On look back's three cues taught in
sequence (first, then last, then second, 600 lessons each; the
[sequence battery](../benchmarks/sequence/README.md#cues-taught-in-sequence-238), six development
and six fresh founders):

| Arm | First cue | Last cue |
| --- | --- | --- |
| rehearsal, `r=1` | 0.994 to 1.000 | 0.968 to 1.000 |
| interleaved | 0.998 to 1.000 | 0.998 to 1.000 |
| in sequence without rehearsal | 0.32 to 0.74 | 0.58 to 0.95 |
| in sequence, 3,600 lessons | 0.40 to 0.85 | 0.49 to 0.90 |

The row without rehearsal reads the development founders; the others read both sets. With
rehearsal the newest cue reads 1.000 on the development founders. More lessons alone do not keep the earlier cues. A life with
rehearsal takes about 3.1 times as long, because the replayed rows settle in small batches, one
per episode length.

## Many lessons: the normalized point for imitation

For a brain that learns a demonstrator's policy from many lessons, compose it with
`teaching="normalized"` or move a living brain there with `retune(teaching="normalized")`, and
teach about 32 rows per lesson. The point sets the teaching learner's steps to 0.003 and 0.0003
under a running RMS of forgetting 0.99 and floor 1e-4, momentum 0.9. With lessons of 8 rows a
rate of 0.001 does as well; a rate of 0.01 overshoots.

- **Few lessons.** The founder's larger steps acquire faster. In an instrument that teaches a
  chain of steps from a few hundred corrections, the normalized point lowered the taught chain's
  success, and in the token-binding chamber it lowered the held-out answers of the content-memory
  arms at long budgets ([issue 215](https://github.com/muellerberndt/cadence/issues/215),
  [issue 218](https://github.com/muellerberndt/cadence/issues/218)). Measure both on the task.
- **Answers that are right.** Past that point, teach with `learning_surprise=True`
  ([lessons only where the answer differs](#lessons-only-where-the-answer-differs)).
- **The working trace.** At its composed amplitude of 3.0 it lowers teaching fidelity in a
  continuing stream school. Applications that bootstrap by imitation may lower
  `working_memory_amplitude` (0 or 0.3).
- **The actor.** The point sets the teaching learner alone; the actor keeps its rates. Normalizing
  the actor as well kept the reward stream's re-adaptation after a contingency change below its
  contract ([issue 131](https://github.com/muellerberndt/cadence/issues/131)).

### Rates under normalization

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
synapses from the output is orders of magnitude fainter than the readout's, so the floor that
keeps the readout's quiet synapses swallows the deep region's whole signal. `normalize_relative`
above zero replaces it by a floor of that share of each synapse's projection (the block of
synapses between its two neuron ranges; a bias reads its range): every projection steps at about
the rate on its own scale, the faint one as the strong one, and a synapse quieter than its
projection steps less. It needs `normalize` above zero. The teaching point `"deep"` sets it at
0.1, with or without the [apical compartment](anatomy.md#the-apical-compartment), the other points
keep `normalize_floor`, and `LearnerConfig.normalize_relative` sets it for an expert
configuration.

The founder point's rates (`eta=0.5` teaching) and the actor's (1.0) are unnormalized; normalized at those rates,
steps can dwarf the raw signals and saturate the outputs. Retune `eta` and `eta_bias` together; a
development sweep might start at `eta` 0.001 to 0.003, checked against acquisition, retention and
actual step sizes. `LearnerConfig` warns when `normalize > 0` and `eta` or `eta_bias` exceeds
0.05; smaller rates can also fail. The actor steps on its raw contrast
([actor rates](reward.md#actor-rates)).

## A supervised-only stream

`step(teacher=...)` issues a reward-eligible action, so the next `step` without a reward teaches
the actor and critic a zero outcome. For demonstrations with no reward protocol, read context with
greedy actions and teach the learner on the memory-bearing drive of
`stimulus(observations, *, memory=True)`:

```python
supervised = Brain.compose(4, 2, modules=(16, 8), seed=7)
supervised.act(observation, greedy=True)   # advance context without reward eligibility
_, lesson_work = supervised.learner.step(supervised.stimulus(following), np.array([1]))
assert supervised.act(following, greedy=True).shape == (1,)
```

Direct learner calls leave `Brain.last_learning` untouched; keep their report. Finish any
outstanding real outcome before a reward-driven life switches to greedy actions.
