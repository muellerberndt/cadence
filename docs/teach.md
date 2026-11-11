# Train a brain

A Cadence brain has no separate training mode. It learns while it lives: every lesson you give
it and every outcome the world reports changes its connections a little, and the next answer
comes from the changed brain. Animal and human brains learn from experience and not by
backpropagation with gradient descent. Cadence is designed the same way: its connections change
locally while the brain keeps running.

This guide shows every way to train a brain, how to choose rates and lesson sizes, how to save
and resume, how to check a result without fooling yourself, and what to do when training goes
wrong. Read [start here](start.md) first: it introduces observations, streams, `act`, `lesson`
and `step`.

## Two signals: lessons and outcomes

A brain learns from two kinds of signal.

- **A lesson** says what the right answer was. You know the answer for an observation and show
  it: `brain.lesson(observations, labels)`. This is supervised learning, like a teacher who
  points at the right move.
- **An outcome** says how good the brain's own action was. The world pays a reward for what the
  brain did, and the brain learns which actions pay: `brain.step(observations, reward=...)` or
  `brain.live(...)`. This is reinforcement learning, like an animal that finds food.

A lesson, in its simplest form:

```python
import numpy as np
from cadence import Brain

cues = np.eye(4)                      # four observations, one row each
labels = np.array([0, 1, 1, 0])       # the right action for each

brain = Brain.compose(inputs=4, actions=2, seed=0)
for lesson in range(100):
    brain.lesson(cues, labels)        # "for this observation, the answer is this action"
    if (brain.predict(cues) == labels).all():
        break
assert (brain.predict(cues) == labels).all()
```

An outcome, in its simplest form. The reward passed to `step` always belongs to the action the
brain returned on the call before:

```python
player = Brain.compose(inputs=4, actions=2, seed=0, actor_eta=0.1, actor_eta_bias=0.01)
rng = np.random.default_rng(0)
cue = int(rng.integers(4))
action = player.step(cues[[cue]])                      # the first moment has no outcome yet
paid = []
for moment in range(200):
    reward = 1.0 if action[0] == labels[cue] else 0.0  # the world scores the action taken
    paid.append(reward)
    cue = int(rng.integers(4))
    action = player.step(cues[[cue]], reward=[reward])
assert np.mean(paid[-100:]) > 0.8                      # it learned which actions pay
```

Lessons learn faster, because every lesson names the answer. Outcomes need no teacher, because
the world itself does the scoring. Many applications use both: lessons to bootstrap, outcomes
to keep improving.

## Which way to train?

| You have | Use | Section |
| --- | --- | --- |
| The right answer for each observation, as a dataset | `lesson(observations, labels)` | [supervised lessons](#supervised-lessons) |
| How good each action is, as a probability per action (a search's visit shares, another brain's policy) | `lesson(observations, targets=p)` | [distill another brain or a search](#distill-another-brain-or-a-search) |
| Some rows that matter more than others (rare classes, critical positions) | `lesson(observations, labels, weights=w)` | [teach a distribution, weigh the rows](#teach-a-distribution-weigh-the-rows) |
| A teacher who answers while the brain lives in a stream | `step(observations, teacher=labels)` | [teaching inside a stream](#explicit-learning-with-step) |
| A world that pays rewards for actions | `step(observations, reward=r)`, or `live` for one creature | [learning from outcomes](#learning-from-outcomes) |
| A demonstrator the brain can ask while it acts on its own | `act(greedy=True)`, then `lesson` on the state it reached | [teach on the brain's own path](#teach-on-the-brains-own-path) |
| A way to improve the brain's own choices: a lookahead, a game result, a stronger copy | `lesson(targets=...)` on the improved choices | [self-play and expert iteration](#self-play-and-expert-iteration) |
| Several tasks taught one after another | `learning_rehearsal=1.0` | [curriculum and rehearsal](#curriculum-and-rehearsal) |

One minimal example per row, on a brain of its own:

```python
demo = Brain.compose(inputs=4, actions=2, seed=3)
x = cues[:2]
demo.lesson(x, labels[:2])                                   # labels
demo.lesson(x, targets=np.array([[0.7, 0.3], [0.2, 0.8]]))   # a distribution per row
demo.lesson(x, labels[:2], weights=np.array([1.0, 3.0]))     # the second row counts three times
action = demo.step(x, teacher=labels[:2])                    # teach inside a stream, then act
action = demo.step(x, reward=np.array([1.0, 0.0]))           # the world scored that act
demo.act(x, greedy=True)                                     # a greedy act: nothing awaits a reward
```

## Supervised lessons

`brain.lesson(observations, labels)` teaches the shown action for each row of observations. It
returns a report and keeps it in `brain.last_learning` under `demonstration_*` keys.

What a lesson does, in plain words: the brain settles on the observation and reads its own
answer (the free phase), then settles twice more with its motor neurons pulled gently toward
the shown answer and away from it (the nudged phases). Every synapse compares its two neurons'
activity across those settles and moves a small step. No error is sent backward through the
network ([the free/nudged rule](settling.md#learning-the-freenudged-rule)).

Things to know:

- **Lessons only where the answer is wrong.** A new brain holds `learning_surprise=True`: rows the
  brain answers right are skipped, and the report counts them in `agreed_presentations`
  ([lessons only where the answer differs](#lessons-only-where-the-answer-differs)). With
  several motor slots a row is right only where every slot is; `agreed_slots` counts the slots.
- **Rows are streams.** Each row is one stream. Once the brain has acted on a batch, a lesson must
  have the same number of rows (`ValueError` "lesson batch must match the live streams") until
  `reset()`. Lessons on their own, before any act, can take any number of rows.
- **The lesson teaches the state the act starts from.** A lesson's free phase starts from the
  stream's activity, the state the next act settles from. After `reset`, for streams that have not
  acted, and in a curriculum of lessons alone, it starts from rest. A deep brain can hold two
  stable states for one observation, one reached from the stream and one from rest; the lesson
  corrects the one the act answers from
  ([a deep brain whose lessons do not reach its act](#a-deep-brain-whose-lessons-do-not-reach-its-act)).
- **Teach first, then act.** With a recall memory, every act writes its moment into the memory. A
  lesson on the observations the brain just acted on reads that moment back and warns with a
  `RuntimeWarning`. Where you choose the order, teach and then act, as `step(teacher=...)` does.
- **No lesson while a reward is pending.** A sampled action waits for its outcome, and `lesson`
  refuses with `RuntimeError` until the outcome arrives or a greedy `act` replaces the action.
- **What a lesson changes.** The graph's weights and, with a recall memory, its address map.
  With `memory_choice=True` the episodic memory also records the shown action for its situation
  ([episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before)).

`predict(observations)` answers each row on its own, from rest and without memories, so it
tests what the weights hold. `act` answers inside the stream with every memory
([evaluate without fooling yourself](#evaluate-without-fooling-yourself)).

<a id="explicit-learning-with-step"></a>

## Teaching inside a stream: `step(teacher=...)`

`step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None)`
does a whole moment: it learns from the preceding action's outcome, teaches the label of the
present observation, then samples and returns the next action.

```python
cue_now, cue_next = cues[[0]], cues[[1]]

stepped = Brain.compose(4, 2, modules=(16, 8), seed=7)
action = stepped.step(cue_now)
reward = (action == 0).astype(float)                # the body pays action 0
action = stepped.step(cue_next, reward=reward, done=np.array([False]))
stepped.act(cue_next, greedy=True)                  # a teacher acted instead; nothing to credit
shown = stepped.lesson(cue_next, np.array([1]))     # teach the shown action
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

## Learning from outcomes

`step(observations, reward=...)` learns from every outcome it is given. `live(observation,
reward=...)` runs one creature (a brain composed with `arousal=True`) that learns while it is
young or aroused and answers calmly otherwise
([routine and repair](start.md#routine-and-repair-live)). Both learn by a three-factor rule:
each synapse keeps an eligibility trace of what it did recently, and one broadcast prediction
error turns the traces into weight changes ([learning from reward](reward.md)).

Two settings matter most:

- **`actor_eta`, the actor's step.** The composed 1.0 learns fast and suits short lives. For long
  reward training use a tenth of it or less, with the bias rate a tenth of that, as the example at
  the top does: a long life of rewards at 1.0 can drive the motor neurons to the limits of their
  activation, where the greedy answer is the same whatever the observation
  ([actor rates](reward.md#actor-rates)).
- **`actor_lam` and `actor_gamma`, how far back credit reaches.** A reward a few moments after the
  action that earned it needs eligibility that lasts that long
  ([what delayed credit requires](reward.md#what-delayed-credit-requires)).

## Distill another brain or a search

Distillation teaches a student the answers of a teacher: another brain, a search or any program
that plays well. Two forms:

- **Hard labels.** Ask the teacher for its answer and teach it as a label. This is the simplest
  and works whenever the teacher has one best answer.
- **Soft targets.** Ask the teacher for a probability per action and teach it with
  `lesson(observations, targets=p)`. The student learns the teacher's preferences between the
  other actions too, which helps where several actions are nearly as good.

```python
teacher = brain                                     # the brain trained at the top of this page
student = Brain.compose(inputs=4, actions=2, seed=1)
answers = teacher.predict(cues)                     # hard labels: the teacher's answers
for lesson in range(100):
    student.lesson(cues, answers)
    if (student.predict(cues) == answers).all():
        break
assert (student.predict(cues) == answers).all()

visits = np.array([[30, 10], [5, 35], [12, 28], [33, 7]])   # a search's visit counts per action
shares = visits / visits.sum(axis=1, keepdims=True)         # each row sums to one
soft_student = Brain.compose(inputs=4, actions=2, seed=1)
soft_student.lesson(cues, targets=shares)
```

A brain's own policy is a distribution too: after a greedy `act`,
`teacher.basal_ganglia.probabilities(teacher.basal_ganglia.state, temperature=t)` gives one row
per stream, and a smaller `t` makes the targets sharper. Read the teacher where it acts: an
`act` reads its memories, `predict` does not.

## Teach a distribution, weigh the rows

Both are opt-in; a lesson without them is the plain lesson above to the bit.

**`targets=`, a distribution per row.** `lesson(observations, targets=p)` takes `p` of shape
`(rows, actions)` in place of labels: nonnegative, each row summing to one (with several motor
slots, each slot's columns side by side, each slot's summing to one). The nudged phases pull the
motor neurons toward `p` as they pull toward a label's one-hot row, and a one-hot `p` teaches as
its label does.

- A distribution has no single answer to agree with: under `learning_surprise` every row of
  targets is taught.
- An episodic memory records no demonstration from targets.
- A brain with `recall_tokens` or `learning_rehearsal` refuses them: both read a single label.

**`weights=`, one per row.** A nonnegative number per row scales that row's pull, and with it the
row's share of the update: weight one is the lesson without weights, weight zero teaches the row
nothing. Use it to balance a curriculum, giving rare classes or critical positions more weight.
Under the [normalized point](#many-lessons-the-normalized-point-for-imitation) a weight common to
every row divides out, so the weights act through their ratios. Under `learning_surprise` the
weights of the taught rows apply. The rehearsal replays its rows at weight one, so a brain with
`learning_rehearsal` refuses weights other than one.

A common choice for classes of unequal size is the inverse of each class's share, so every
class pulls with the same total weight:

```python
student = Brain.compose(4, 3, modules=(16,), seed=2)
student.lesson(cues[[2]], targets=np.array([[0.6, 0.4, 0.0]]))   # 0.6 on action 0, 0.4 on 1
classes = np.array([0, 0, 0, 2])                                  # class 2 is rare
counts = np.bincount(classes, minlength=3)
weights = len(classes) / (3 * counts[classes])                    # rare rows weigh more
student.lesson(cues, classes, weights=weights)
assert student.last_learning["demonstrations"] == 4.0
```

Weights shift where the brain draws its boundaries, and too large a weight costs the common
classes. Choose them on held-out data with per-class accuracy
([evaluate without fooling yourself](#evaluate-without-fooling-yourself)).

## Curriculum and rehearsal

A curriculum is the order in which lessons come. Three rules of thumb:

- **Interleave what must be kept together.** Lessons on a new type move the same synapses the
  earlier types' answers rest on, so teaching types one after another can erase the earlier ones.
  Mixed batches avoid that.
- **Start simple where the task allows.** Short episodes, few distractors and clear cues first,
  then the full task. Check the earlier stages again after the later ones.
- **Rehearse when the order is fixed.** Where the types arrive one after another and cannot be
  mixed, `learning_rehearsal` replays earlier lessons beside the new ones (next section).

<a id="teach-several-types-in-sequence-rehearsal"></a>

### Teach several types in sequence: rehearsal

`learning_rehearsal=r` (opt-in, founder 0) replays earlier lessons alongside the new ones.

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
episode in progress is replayed from the moments heard after the retune. A grown brain starts the
reservoir afresh, a grafted one keeps this brain's.

```python
rehearsing = Brain.compose(7, 4, seed=0, learning_rehearsal=1.0)
assert rehearsing.learner.config.rehearsal == 1.0 and not rehearsing.rehearsal_moments
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
rehearsal the newest cue reads 1.000 on the development founders. More lessons alone do not keep
the earlier cues. A life with rehearsal takes about 3.1 times as long, because the replayed rows
settle in small batches, one per episode length.

## Teach on the brain's own path

A brain that only watches a demonstrator learns the states the demonstrator visits. Once it acts,
its own small errors lead it to states no lesson covered, and the errors compound over an
episode. Labels on the states the brain itself reaches cover exactly those. The recipe is DAgger
(Ross, Gordon and Bagnell 2011):

1. Bootstrap by watching: teach the demonstrator's moves with `lesson` and execute them.
2. Let the brain act with `brain.act(observation, greedy=True)` and execute its own move.
3. Have the demonstrator label that same observation and teach it with
   `brain.lesson(observation, label)`.

A greedy act holds nothing for reward, so the lesson is legal. Each batch row below is one
stream:

```python
def observe(position):  # a one-hot position on a line of five cells
    return np.eye(5)[position]

def demonstrator(position):  # the policy the brain watches: 1 steps right, 0 steps left
    return (position < 2).astype(np.int64)

def move(position, action):
    return np.clip(position + np.where(action == 1, 1, -1), 0, 4)

pupil = Brain.compose(inputs=5, actions=2, seed=0, teaching="normalized", recall=False)
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
wrongly; on its own path it reaches those cells and learns them.

Teach before acting on the same observation, as `step(..., teacher=labels)` orders them: an act
writes its moment to the recall memory, and a lesson after it reads that moment before its
successor is written. With surprise lessons (a new brain's default) a row the brain answers right
is judged and skipped, so the lesson corrects only where the brain would go wrong. The pupil
above has no recall memory, so its order does not matter.

**The first round on the brain's own path.** On the [cart-pole chamber](../benchmarks/coaching/README.md)
(three founders, 32 held-out starts, 16 episodes a round) the first round on the brain's own path
drops every founder to 0 starts of 32, 10 to 76 steps on average, from 1 to 22 starts after
watching; two of the three recover to 32 of 32 by the fourth round. Following each of its own
episodes with a watched one reads 7 to 14 starts and 385 to 459 steps after the first round, and no
founder below 259 steps in any round. A candidate term, which no arm isolates: a brain that fails
early takes its lessons from the moments before a failure, and the ordinary moments it learned by
watching go untaught. On a brain of modules `(64, 32)` the episodic memory, the efference copy and
the compiled moment change the pace (arousal, maturity and fading records change no lesson there):
with all six opt-ins it balanced 32 of 32 on every founder by the third round, where the same brain
without them balanced 14, 15 and 24 of 32
([the opt-in ablation](../benchmarks/coaching/README.md#which-opt-ins-a-coached-brain-needs)).
Without the efference copy that brain balanced 32, 13 and 21 starts after the first round against
23, 4 and 0 with it; without the episodic memory the rounds swung (one founder fell from 27 to 2).
Taught after each act instead of before it, the whole set also balanced 32 of 32 by the third
round on this chamber.

**Read agreement per answer.** A watch that stops once the brain agrees with the demonstrator on
a share of moments can stop on the commonest answer alone: a brain that scans everywhere agrees
with a demonstrator that mostly scans. Count each answer's agreement apart, and let the watch run
whole episodes ([every answer is the majority class](#every-answer-is-the-majority-class)). Take the labels from the
demonstrator the brain watched: corrections from a different policy after a bootstrap by watching
lower the agreement with the watched demonstrator
([issue 215](https://github.com/muellerberndt/cadence/issues/215)). Teach many rows at once at the
normalized point ([many lessons](#many-lessons-the-normalized-point-for-imitation)).

## Self-play and expert iteration

When no teacher knows the answer, the brain can improve on its own play. The pattern has three
parts:

1. **Play.** The brain acts in the task, against the world or against a copy of itself.
2. **Improve.** Something turns its choices into better ones: a lookahead search over a model of
   the task, the final result of the game credited to the moves that led to it, or a stronger
   frozen copy that answers the same positions.
3. **Distill.** Teach the improved choices with `lesson(observations, targets=...)`, a
   distribution over actions per position, or with labels where the improvement names one move.

Repeat. The brain's play reaches the positions it will meet, and the improvement makes each
lesson a little better than the brain was. Here a one-step lookahead over a known model of a
line world improves the moves of a brain that should walk to the right end:

```python
def improve(position):  # a lookahead: score each move by how close it lands to cell 4
    scores = np.stack([-np.abs(4 - move(position, a)) for a in (0, 1)], axis=1).astype(float)
    shares = np.exp(scores / 0.5)
    return shares / shares.sum(axis=1, keepdims=True)   # a distribution over the two moves

walker = Brain.compose(inputs=5, actions=2, seed=0)
rng = np.random.default_rng(0)
position = rng.integers(0, 4, 16)                       # 16 games at once, one per row
for moment in range(40):
    seen = observe(position)
    walker.lesson(seen, targets=improve(position))      # distill the improved choice first
    action = walker.act(seen, greedy=True)              # then play the brain's own move
    position = move(position, action)
    position = np.where(position == 4, rng.integers(0, 4, 16), position)  # a finished game restarts
assert (walker.predict(observe(np.arange(4))) == 1).all()
```

For two-player games, keep the opponent fixed while the learner trains: a saved copy
(`Brain.load`) or `copy.deepcopy(brain)`. Promote the learner to be the next opponent only when it
beats the fixed one over enough games, and keep earlier opponents in the pool so the brain does
not forget how to beat them.

## Choosing rates and steps

Start with the defaults. A new brain sets its teaching point, its rates and its width scaling
for you. Change one thing at a time, and compare against the default on held-out data.

### Which teaching point?

The teaching point sets how big a lesson's steps are. Name it at `compose` or move a living brain
with `retune(teaching=...)`.

| Point | Steps | Use it for |
| --- | --- | --- |
| `"deep"` | the normalized steps, a floor relative to each projection, nudged phases that run their whole budget (32 sweeps) | a new brain with the recall memory (its default), and every brain with several `modules` |
| `"normalized"` | 0.003 and 0.0003 under a running RMS of each synapse's contrast | many lessons, as in imitating a demonstrator from thousands of rows |
| `"founder"` | 0.5 and 0.05 on the raw contrast | few lessons; the default of a brain composed with `recall=False` |

```python
few = Brain.compose(4, 2, seed=0, recall=False)                 # "founder": big steps
many = Brain.compose(4, 2, seed=0, teaching="normalized")       # small, normalized steps
assert few.describe()["genes"]["teaching"] == "founder"
assert Brain.compose(4, 2, seed=0).describe()["genes"]["teaching"] == "deep"
many.retune(teaching="deep")                                     # move a living brain
```

### Lesson size and count

- **Rows per lesson.** At the normalized point about 32 rows per lesson; with lessons of 8 rows a
  rate of 0.001 does as well, and a rate of 0.01 overshoots
  ([many lessons](#many-lessons-the-normalized-point-for-imitation)).
- **How many lessons.** Stop when held-out accuracy stops rising. More lessons on rows the brain
  answers right add nothing under surprise lessons and can harm without them.
- **Wide brains.** Above an association of 64 neurons a new brain scales its steps to its width by
  itself ([wide brains](tune.md#wide-brains)). With surprise lessons in a wide brain add
  `learning_surprise_share=True` ([in proportion to the surprise](#in-proportion-to-the-surprise)).
- **Reward rates.** `actor_eta` and `actor_eta_bias` belong to outcomes, the teaching point to
  lessons, and `temperature` to both ([actor rates](reward.md#actor-rates)).

### Lessons only where the answer differs

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

A row of several motor slots agrees only where every slot does, so a brain of twelve slots that
answers ten of each row's as shown reports no agreed row and skips none. The composed brain's
report also holds `agreed_slots`: the slots its answer gave as shown, counted over every row of
the lesson (with one slot, the agreed rows; with `recall_tokens`, a row whose label names no token is
taught at every lesson and counts no agreed slot). `agreed_slots / (attempted_presentations * slots)`
is the share of the answer that is right. It reads zero where the lesson judges no row: under
`learning_surprise=False` and for `targets`.

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

### Many lessons: the normalized point for imitation

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

The founder point's rates (`eta=0.5` teaching) and the actor's (1.0) are unnormalized; normalized
at those rates, steps can dwarf the raw signals and saturate the outputs. Retune `eta` and
`eta_bias` together through an expert `learning=LearnerConfig(...)`; a development sweep might
start at `eta` 0.001 to 0.003, checked against acquisition, retention and actual step sizes.
`LearnerConfig` warns when `normalize > 0` and `eta` or `eta_bias` exceeds 0.05; smaller rates can
also fail. The actor steps on its raw contrast ([actor rates](reward.md#actor-rates)).

## Save, load and resume

`brain.save(path)` writes the whole brain: weights, the optimizer's history, every memory, the
streams' state, the random state and an action that awaits its outcome. `Brain.load(path)`
continues the same life, so training can stop and resume at any moment.

```python
brain.save("training.npz")
resumed = Brain.load("training.npz")
assert np.array_equal(resumed.predict(cues), brain.predict(cues))
resumed.lesson(cues, labels)                     # training continues where it stopped
```

- **Pending outcomes.** A brain saved while a sampled action waits for its reward loads with that
  action pending. Report the outcome once, to the loaded brain
  ([reset and save](start.md#reset-and-save)).
- **The environment is yours to save.** The brain does not save the world it lives in. Save the
  environment's state, and the rows' order, beside the brain.
- **Keep checkpoints to choose from.** Save a copy at a declared schedule, score each copy on the
  same held-out data, and report which copy you chose and why. A later copy is not always the
  better one.
- **Another machine.** `Brain.load(path, backend="cpu")` brings a brain trained on a device back to
  the CPU in float64 ([backends](backends.md#what-runs-where)).

## Evaluate without fooling yourself

A number is only as good as the test behind it.

- **Hold data out.** Split a held-out set before you choose any setting, and never teach on it.
  Tune on a separate development set.
- **Test the right thing.** `predict` and `accuracy` answer each row from rest, without memories:
  they test the weights. A task that needs memory is tested through `act`, `step` or `live` on a
  continuing stream. Compare a greedy `act` with `predict` to see what the memories add.
- **Use enough samples.** An accuracy over `n` held-out answers has a standard error of about
  `sqrt(p * (1 - p) / n)`: at 50 answers and 0.8 that is 0.06. Two runs that differ by less than
  two standard errors have not been told apart.
- **Use several seeds.** A brain's founder seed changes its result. Report the mean over seeds,
  with the least and the largest.
- **Add noise.** Test on observations a little different from the ones taught, so a brain that
  memorised exact inputs shows it.
- **Compare with a control.** Always answering the commoner class, a linear model, the task's
  simplest solver ([evaluate a brain](evaluate.md)).
- **Look per class.** Overall accuracy hides a rare class answered wrong every time.

```python
rng = np.random.default_rng(1)
which = rng.integers(0, 4, 200)                           # 200 held-out answers
held_out = cues[which] + 0.2 * rng.standard_normal((200, 4))   # noisy versions of the cues
truth = labels[which]
correct = brain.predict(held_out) == truth
accuracy = correct.mean()
error = np.sqrt(accuracy * (1 - accuracy) / len(correct))     # its standard error
per_class = {c: round(float(correct[truth == c].mean()), 2) for c in (0, 1)}
majority = max(np.mean(truth == 0), np.mean(truth == 1))       # always the commoner answer
print(f"held out {accuracy:.2f} ± {2 * error:.2f}, per class {per_class}, control {majority:.2f}")
```

## When training goes wrong

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Every answer is the commoner class | class imbalance | [weights; compare every-row and surprise lessons](#every-answer-is-the-majority-class) |
| A brain that answered well gets worse as training continues | steps too large for a trained brain | [smaller steps, surprise lessons](#a-trained-brain-gets-worse-with-more-training) |
| After long reward training the greedy answer is the same for every observation | the motor neurons saturated at a high `actor_eta` | [a tenth of the actor rate](#reward-training-pins-the-answer) |
| `predict` answers right, `act` answers wrong, and lessons do not change the act | a deep brain with two stable states for one observation | [teach inside the stream](#a-deep-brain-whose-lessons-do-not-reach-its-act) |
| Accuracy stays at chance and the report shows `free_budget_exhausted` | the free phase did not settle | [finite free phases](settling.md#finite-free-phases-that-run-out-of-budget) |
| The actor learns a zero reward from lessons given through `step` | `step(teacher=...)` issues a reward-eligible action | [a supervised-only stream](#a-supervised-only-stream) |

### Every answer is the majority class

When one class is much commoner than another, most of each lesson's pull goes to the common
class, and a brain can settle on answering it everywhere. Overall accuracy then looks good while
the rare class is never answered. Read per-class accuracy, then:

- weigh the rows by the inverse of their class's share (`weights=`,
  [above](#teach-a-distribution-weigh-the-rows)). The weight moves accuracy from the common class
  to the rare one and does not add to both: on the
  [scan-and-fire chamber](../benchmarks/coaching/README.md#the-scan-and-fire-chamber) (the
  commonest answer 73% of the moments, six founders) inverse weights raised the rarest answer on
  the two weakest founders from 0.74 and 0.83 to 0.92 and 0.93 and lowered the commonest from 0.91
  to 1.00 to 0.83 to 0.96, and unweighted lessons agreed best overall (0.90 to 0.98);
- compare `learning_surprise=False`, which teaches every row, the right ones too: on the
  same chamber it agreed best overall (0.97 to 0.99 against 0.90 to 0.98) and raised the weakest
  founder's rare answer from 0.74 to 0.87. Over a long curriculum at the normalized point,
  teaching rows the brain answers right can collapse it
  ([lessons only where the answer differs](#lessons-only-where-the-answer-differs)), so read
  both on the task;
- or sample the lessons so every class appears about equally often.

### A trained brain gets worse with more training

Large steps that suit a newborn can move a trained brain past what it knew. Signs: held-out
accuracy rises, then falls or swings; `scale_step` in the lesson report stays large while answers
are right.

- Keep surprise lessons on, so right answers are not taught again.
- Move to smaller steps: `retune(teaching="normalized")`, or `"deep"` for a deep brain.
- At the normalized point in a wide brain, add `learning_surprise_share=True`.
- Fine-tune from rewards at a smaller `actor_eta` than the brain was raised with.
- Keep the last good checkpoint and compare against it.

### Reward training pins the answer

At the composed `actor_eta` of 1.0 a long life of rewards can drive the motor neurons to the
limits of their activation. There a nudge has no slope: the greedy answer is the same whatever
the observation, and the contrast that would move it back is near zero. The reward report shows
`saturation` near 1 and `trace` near 0. Train long lives at a tenth of the composed rate or less,
with the bias rate a tenth of that ([actor rates](reward.md#actor-rates)).

```python
calm = Brain.compose(inputs=4, actions=2, seed=0, actor_eta=0.1, actor_eta_bias=0.01)
assert calm.describe()["genes"]["actor_eta"] == 0.1
```

### A deep brain whose lessons do not reach its act

A brain of several regions can hold two stable states for one observation: one it reaches from the
stream it lives in, one it reaches from rest. A lesson teaches the state the next act settles from,
so teach inside the same stream the brain acts in: the same rows, without a `reset` between the
acts and the lessons. A lesson after `reset` or on streams that have not acted starts from rest.
For a deep brain use `teaching="deep"` (a new brain's default): its nudged phases run their whole
budget, and the contrast stays a good estimate of the gradient down to the lowest region. A
standalone `Learner` takes the stream's state as `warm=`.

To see whether a brain answers differently in its stream than from rest, compare a greedy `act`
with `predict` on the same rows:

```python
in_stream = brain.act(cues, greedy=True)       # inside the stream, with memories
from_rest = brain.predict(cues)                # each row from rest, without memories
print("stream and rest agree on", float((in_stream == from_rest).mean()))
```

### A supervised-only stream

`step(teacher=...)` issues a reward-eligible action, so the next `step` without a reward teaches
the actor and critic a zero outcome. For demonstrations with no reward protocol, read context with
greedy actions and teach the learner on the memory-bearing drive of
`stimulus(observations, *, memory=True)`:

```python
supervised = Brain.compose(4, 2, modules=(16, 8), seed=7)
supervised.act(cue_now, greedy=True)       # advance context without reward eligibility
_, lesson_work = supervised.learner.step(supervised.stimulus(cue_next), np.array([1]))
assert supervised.act(cue_next, greedy=True).shape == (1,)
```

Direct learner calls leave `Brain.last_learning` untouched; keep their report. Finish any
outstanding real outcome before a reward-driven life switches to greedy actions.
