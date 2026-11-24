# Start here: your first brain

This page builds a brain and explains, one idea at a time, what it does: observations and actions,
settling, acting, the two ways it learns, the state it carries from moment to moment, resetting
and saving. Every example runs as written. Install Cadence first
([install](README.md#install)).

## Your first brain

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)  # 4 senses, 2 possible actions
observation = np.array([[1.0, 0.0, 0.0, 0.0]])            # what it senses at this moment
action = brain.live(observation)                          # it settles and chooses an action
reward = (action == 0).astype(float)                      # your world scores that action
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.live(next_observation, reward=reward)      # it learns from the score, then acts again
assert brain.learner.updates > 0
```

That is a whole life in miniature: the brain senses, acts, hears how the action went and acts
again, a little changed. The sections below take each step apart.

## 1. A brain has senses and actions

`Brain.compose(inputs, actions)` builds a brain with `inputs` sensory neurons and one motor
neuron per action. Between them sits an **association region** of 64 neurons, where the
brain forms its interpretation of what it senses. Around that core the brain holds four
memories, all on by default: a working trace of what it was just thinking, a heard history of
what it heard a moment ago, a recall memory of the stream's recent moments and an associative
store of what paid where ([the default brain](modules.md#the-default-brain) lists each part).

All of these settle together into one state, and that state is the brain's answer. This default
brain is called **System 1**. Optional **System 2** observer regions can be added to the same
brain ([observers](life.md#add-optional-observers)). `arousal=True` gives the brain the arousal that
`live` needs: it decides from moment to moment whether the brain learns.

| Setting | What it sets |
| --- | --- |
| `inputs` | the observation's width, or an image shape ([wide inputs](senses.md#image-inputs)) |
| `senses` | in place of `inputs`: several senses by name, each a width, given by name at every moment ([senses by name](senses.md#senses-by-name)) |
| `actions` | the number of actions to choose from |
| `arousal` | `True` for a brain that `live` drives |
| `seed` | the random draw of the newborn brain; the same seed gives the same brain |
| `modules` | the sizes of the processing regions, `(64,)` by default ([larger brains](tune.md#larger-brains)) |

`brain.describe()` lists every setting a brain holds, and
[tune and read a living brain](tune.md#defaults-and-expert-overrides) explains them. To build a
brain for a purpose of your own, [parts of a brain](modules.md) lists every part you can add
(senses, regions, memories, ways to learn, modules that arrive as files) and composes
whole brains from them.

## 2. Observations, actions and streams

An **observation** is one row of numbers, one per input: what the brain senses at one moment.
An **action** is an integer from `0` to `actions - 1`. Calls take a batch of rows,
shape `(rows, inputs)`, and return one action per row.

Each row is a **stream**: one continuing life, such as one game or one conversation. Row 0 at
this moment and row 0 at the next moment are the same stream, so keep every row's identity fixed.
A brain that has acted on a batch of a given size answers batches of that size until `reset()`
starts new streams. `live` drives one stream; `act`, `step` and `lesson` take any number.

```python
streams = Brain.compose(inputs=4, actions=2, seed=0)
both = streams.act(np.eye(4)[:2])           # two streams, one observation each
assert both.shape == (2,)
again = streams.act(np.eye(4)[2:])          # the same two streams, one moment later
streams.reset()                             # new streams may have another count
assert streams.act(np.eye(4)[:3]).shape == (3,)
```

<a id="inspect-the-work-of-answering"></a>

## 3. Settling: how a brain answers

A Cadence brain answers by settling. Its neurons drive each other back and forth, senses to
association to motor and back, until their activity stops changing: the brain **settles** into an
equilibrium, and the action is read from the motor neurons of that settled state. Each
repetition of the update is a **sweep**. The settled state also shapes the next moment, since
the brain starts its next settle from it.

`brain.last_settlement` reports the latest settle:

```python
report = brain.last_settlement
assert report is not None and report["qualified"]
assert report["max_residual"] <= report["tolerance"]
print("sweeps:", report["steps"], "residual:", report["max_residual"])
```

The **residual** measures how far the state is from satisfying the brain's equations. A settle
**qualifies** when the residual is below the tolerance; the brain only acts from a qualified
state, and refuses an answer that does not qualify within its budget
([qualification and refusal](life.md#qualification-and-refusal)). A small residual means the network
agrees with itself; whether the action was good is for the task's outcomes to say. Reward
learning, lessons and memory work are outside this report. The
[continuing example](../examples/continuing_brain.py) tracks these counters through a whole life.

## 4. Two ways to learn: lessons and outcomes

A brain changes its connections from two signals:

- **A lesson** shows the right answer: `brain.lesson(observations, labels)`. Use it when you know
  the answer.
- **An outcome** scores the brain's own action: the `reward` passed to `live` or `step` belongs to
  the action returned on the call before. Use it when the world, and no teacher, says what is good.

```python
pupil = Brain.compose(inputs=4, actions=2, seed=0)
cues, answers = np.eye(4), np.array([1, 0, 0, 1])
for lesson in range(100):
    pupil.lesson(cues, answers)                # here is the right answer for each cue
    if (pupil.predict(cues) == answers).all():
        break
assert (pupil.predict(cues) == answers).all()
```

Both kinds of learning are local: each synapse changes from the activity of the two neurons it
joins, while the brain keeps running. Animal and human brains learn from experience and not by
backpropagation with gradient descent. Cadence is designed the same way: its connections change
locally while the brain keeps running. [Train a brain](teach.md) covers every way to train,
including distillation, rare classes, curricula and what to do when training goes wrong.

## 5. Which call should I use?

| Call | What it does | Learns | Use it for |
| --- | --- | --- | --- |
| `live(observation, reward=...)` | one moment of one creature: takes the last outcome, answers | while young or aroused | a single agent in a world ([routine and repair](life.md#routine-and-repair-live)) |
| `step(observations, reward=..., teacher=...)` | one moment of many streams: takes the last outcomes and labels, answers | from every outcome and label | batches, teaching inside a stream, learning from every outcome ([train](teach.md#explicit-learning-with-step)) |
| `act(observations, greedy=...)` | answers, without learning | nothing | evaluation, or the first half of a moment you time yourself |
| `learn(reward, done, next_observations)` | the outcome of the last sampled `act` | from that outcome | the second half of a moment you time yourself ([by hand](tune.md#one-experience-step-by-hand)); it pays no [record of a bound moment](memory.md#records-of-bound-moments-opt-in), which `step` and `live` do |
| `lesson(observations, labels)` | teaches the right answers, without acting | from the labels | supervised training ([train](teach.md#supervised-lessons)) |
| `predict(observations)` | each row's answer from rest, without memories | nothing | testing what the weights hold ([evaluate](teach.md#evaluate-without-fooling-yourself)) |

One minimal example of each:

```python
creature = Brain.compose(4, 2, seed=1, arousal=True)
a = creature.live(cues[[0]])                         # live: one creature
a = creature.live(cues[[1]], reward=[1.0])           # ... the reward of the action before

school = Brain.compose(4, 2, seed=1)
a = school.step(cues[:2], teacher=answers[:2])       # step: teach the present rows, then act
a = school.step(cues[2:], reward=[1.0, 0.0])         # ... the rewards of the actions before

body = Brain.compose(4, 2, seed=1)
a = body.act(cues[[0]])                              # act: a sampled answer
body.learn([1.0], [False], cues[[1]])                # learn: its outcome and the next observation
greedy = body.act(cues[[1]], greedy=True)            # a greedy answer learns nothing

school.act(cues[:2], greedy=True)                    # a greedy answer replaces the waiting action
school.lesson(cues[2:], answers[2:])                 # lesson: answers shown, no act
guesses = school.predict(cues)                       # predict: from rest, no memories
```

A sampled action (from `live`, `step` or `act` without `greedy`) waits for its outcome. The next
`live` or `step` call delivers it as `reward`; a reward you leave out counts as a real zero. A
greedy `act` takes nothing and replaces any action that was waiting.

## 6. The stream state: what carries from moment to moment

Between two calls a brain keeps, for every stream:

- its **neural activity**, the settled state the next settle starts from;
- the **working trace** of its last interpretation;
- the **recall slots**, the moments the recall memory holds;
- an **action awaiting its outcome**, if it sampled one.

This is why the same observation can get different answers at different moments: what came
before is part of the present settle. `predict` reads none of this state; `act`, `step` and
`live` read all of it.

```python
before = brain.recall.slots
brain.live(observation, reward=[0.0])
assert brain.recall.slots > before          # the moment was written to the recall memory
assert brain.working_memory.trace.any()     # the last interpretation is carried
assert brain.pending_feedback               # this action waits for its outcome
```

## 7. Reset: start a new stream

`brain.reset()` ends every stream and starts fresh ones: activity, traces, recall slots and any
waiting action are cleared. What the brain has learned stays: its weights, the associative
records, the recall memory's learned addresses and, by default, its arousal. Reset between
independent episodes, such as two separate games, and when the number of streams changes.

```python
brain.reset()
assert not brain.pending_feedback and brain.recall.slots == 0
```

[Reset and save](life.md#reset-and-save) gives the details, including `keep=("slots",)` and
resetting the arousal.

## 8. Save and load

`brain.save(path)` writes the whole brain, streams and all; `Brain.load(path)` continues the
same life, an action awaiting its outcome included.

```python
brain.live(observation)
brain.save("first-brain.npz")
same = Brain.load("first-brain.npz")
assert same.pending_feedback                 # the loaded brain awaits the same outcome
```

## Where to go next

- [Assemble a brain](assembly.md): senses by name, a map of things, memories, attention, an ear
  and a voice, put together into four whole brains.
- [A continuing life](life.md): the whole contract of `live`: when it learns, refusal and retry,
  outcomes that arrive later, surprise, imagination, resetting and saving.
- [Parts of a brain](modules.md): the catalogue of every part, and brains composed for a purpose.
- [Train a brain](teach.md): every way to train, rates, evaluation and failure modes.
- [Memory](memory.md): which memory to use for which task.
- [Learning from reward](reward.md): eligibility traces, the actor's rates, delayed credit.
- [Tune and read a living brain](tune.md): every setting, and changing them in a living brain.
- [Backends](backends.md): where a brain runs, CPU or GPU.

## The reference for a continuing life

<a id="continuous-interaction"></a>
<a id="routine-and-repair-live"></a>
<a id="qualification-and-refusal"></a>
<a id="outcomes-that-arrive-later-wait"></a>
<a id="learning-during-a-continuing-life"></a>
<a id="surprise-and-the-focus-it-draws-opt-in"></a>
<a id="private-imagination"></a>
<a id="reset-and-save"></a>
<a id="add-optional-observers"></a>
<a id="defaults-and-the-thinking-clock"></a>
<a id="one-continuing-equilibrium-brain"></a>

The whole contract of a continuing life is a page of its own, [a continuing life](life.md):
[routine and repair: `live`](life.md#routine-and-repair-live),
[qualification and refusal](life.md#qualification-and-refusal),
[outcomes that arrive later: `wait`](life.md#outcomes-that-arrive-later-wait),
[learning during a continuing life](life.md#learning-during-a-continuing-life),
[surprise, and the focus it draws](life.md#surprise-and-the-focus-it-draws-opt-in),
[private imagination](life.md#private-imagination),
[reset and save](life.md#reset-and-save),
[observers](life.md#add-optional-observers) and
[defaults and the thinking clock](life.md#defaults-and-the-thinking-clock). The design direction
and the contract of each mechanism are in
[one continuing equilibrium brain](world-model.md#mechanisms-and-their-boundaries).
