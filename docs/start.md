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
brain forms its interpretation of what it senses. Around that core the brain holds three
memories, all on by default:

- a **working trace**, which carries the last moment's interpretation into the next one;
- a **recall memory**, a notebook of the stream's recent moments that gives one back when the
  present asks for it;
- an **associative store**, which remembers which action paid in which situation.

All of these settle together into one state, and that state is the brain's answer. This default
brain is called **System 1**. Optional **System 2** observer regions can be added to the same
brain ([observers](#add-optional-observers)). `arousal=True` gives the brain the arousal that
`live` needs: it decides from moment to moment whether the brain learns.

| Setting | What it sets |
| --- | --- |
| `inputs` | the observation's width, or an image shape ([wide inputs](senses.md#image-inputs)) |
| `actions` | the number of actions to choose from |
| `arousal` | `True` for a brain that `live` drives |
| `seed` | the random draw of the newborn brain; the same seed gives the same brain |
| `modules` | the sizes of the processing regions, `(64,)` by default ([larger brains](tune.md#larger-brains)) |

`brain.describe()` lists every setting a brain holds, and
[tune and read a living brain](tune.md#defaults-and-expert-overrides) explains them. To build a
brain for a purpose of your own, [parts of a brain](modules.md) lists every part you can add
(senses, regions, memories, ways to learn, modules that arrive as files) and composes five
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
([qualification and refusal](#qualification-and-refusal)). A small residual means the network
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
| `live(observation, reward=...)` | one moment of one creature: takes the last outcome, answers | while young or aroused | a single agent in a world ([routine and repair](#routine-and-repair-live)) |
| `step(observations, reward=..., teacher=...)` | one moment of many streams: takes the last outcomes and labels, answers | from every outcome and label | batches, teaching inside a stream, learning from every outcome ([train](teach.md#explicit-learning-with-step)) |
| `act(observations, greedy=...)` | answers, without learning | nothing | evaluation, or the first half of a moment you time yourself |
| `learn(reward, done, next_observations)` | the outcome of the last sampled `act` | from that outcome | the second half of a moment you time yourself ([by hand](tune.md#one-experience-step-by-hand)) |
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

[Reset and save](#reset-and-save) gives the details, including `keep=("slots",)` and resetting
the arousal.

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

- [Parts of a brain](modules.md): the catalogue of every part, and brains composed for a purpose.
- [Train a brain](teach.md): every way to train, rates, evaluation and failure modes.
- [Memory](memory.md): which memory to use for which task.
- [Learning from reward](reward.md): eligibility traces, the actor's rates, delayed credit.
- [Tune and read a living brain](tune.md): every setting, and changing them in a living brain.
- [Backends](backends.md): where a brain runs, CPU or GPU.

The rest of this page is the reference for a continuing life: how `live` decides when to learn,
qualification and refusal, outcomes that arrive later, resetting and saving in full.

## Continuous interaction

Each call is one moment: the body reports the outcome of the preceding action, and the brain
settles on the present observation and issues the next action
([which call](#5-which-call-should-i-use)). This part of the page gives the full contracts.

### Routine and repair: `live`

`live(observations, *, reward=None, done=None, salience=None, decision_id=None)` follows one
stream of a brain composed with `arousal=True`. A calm brain answers with the greedy choice of
one qualified settle and learns nothing: no eligibility phases, no parameter change, no
associative write, and earlier eligibility fades by one step. Two things rouse it:

- **Surprise**: the outcome contradicts the forecast made before acting, the critic's value of
  the situation. Only outcomes of the brain's own greedy choices can surprise it or move its
  usual error.
- **Want**: the recent reward falls below what this life usually pays, below the body's `need`,
  or below `need_scale` times the spread of the rewards the brain has learned from. The need's
  want is the unmet share of the need; it never habituates, and a rare reward's small mean cannot
  dilute it. The scaled need applies after youth, once an outcome has entered the spread.

An aroused brain samples its policy, learns from every outcome as `step` does and writes the
outcome that woke it to memory. Want heats one uniformly chosen motor slot to
`learning.temperature * (1 + heat * want)`. Every actual reward updates the income averages, so
a brain habituates to a poorer life unless a positive `need` keeps it wanting.

`salience=[importance]` marks the preceding action's actual outcome, as in `step`, with one
finite nonnegative value. When that outcome is learned, a positive mark makes a zero outcome
known to the episodic records without changing its reward or stopping replay. It also sets
consolidation's importance for the outcome that wakes a calm brain. A mark does not wake the
brain or permit a calm write, and it needs an awaited action (after `wait` or a saved
continuation too). The caller decides what to mark; marking every outcome can harm a reward
chain ([episodic limits](memory.md#episodic-limits)).

```python
import numpy as np
from cadence import Brain

cues = np.eye(4)
observation, following = cues[[0]], cues[[1]]
brain = Brain.compose(
    4, 2, modules=(16,), seed=0,
    consolidation=0.25,                  # lasting memory takes half of a witnessed unit outcome
    arousal=True, arousal_youth=30,      # a short youth for a short life
    actor_eta=0.1, actor_eta_bias=0.01,  # one stream's operating point on the odour nursery
)
rng = np.random.default_rng(0)
cue = 0
action = brain.live(cues[[cue]])
modes, hits = [], []
for moment in range(600):
    rewarded = cue % 2 if moment < 300 else 1 - cue % 2   # the rule turns over halfway
    hits.append(action[0] == rewarded)
    cue = int(rng.integers(4))
    action = brain.live(cues[[cue]], reward=[1.0 if hits[-1] else -1.0])
    modes.append(brain.last_arousal["mode"])

assert modes[250:300].count("routine") >= 45 and sum(hits[250:300]) >= 45  # calm and right
assert modes[300:340].count("aroused") >= 20                               # the change wakes it
assert modes[-50:].count("routine") >= 45 and sum(hits[-50:]) >= 45        # repaired, calm again
assert brain.arousal.moments["routine"] > 4 * brain.arousal.moments["aroused"]
```

`consolidation` and the actor rates above are one stream's operating point on the
[odour nursery](../benchmarks/reversal/README.md), where the composed actor rate locks a single
stream's policy. Select them per task; [`retune`](tune.md#retune-the-same-life) changes them in
the same life.

The law, per outcome with unsigned temporal-difference error `error`:

```text
surprise = log(error / (tolerance * usual + floor * scale))   when positive, else 0
want     = max(clip((longrun - recent) / scale, 0, 1), clip((need - recent) / need, 0, 1),
               clip((need_scale * scale - recent) / (need_scale * scale), 0, 1))
level    = decay * level + (1 - decay) * (surprise + want)
```

`usual` is the running own-choice error, `recent` and `longrun` the reward at the `fast` and
`slow` rates, and `scale` the spread of learned outcomes around `longrun`. A want term with a
zero denominator is zero. The `need_scale` term reads the scale as it stood before the outcome
entered it, applies after `youth` and is zero while no outcome has entered the scale; the want
and heat read afterwards use the scale as it stands. The surprise is weighted by
`value_surprise`. The brain is aroused while `level >= threshold` and for its first `youth`
moments. Set genes with `arousal_<field>` at `compose` or `retune`
([composition](tune.md#defaults-and-expert-overrides)); construction refuses values outside
their ranges, a negative `need` or `need_scale`, and `fast < slow`.

| Gene | Founder | Effect |
| --- | --- | --- |
| `threshold` | 0.2 | level at or above which the brain leaves routine |
| `decay` | 0.9 | share of the level one moment hands to the next, in `[0, 1)` |
| `tolerance` | 2.0 | errors within this many usual errors are no surprise; large removes surprise |
| `floor` | 0.1 | nor are errors within this many reward scales |
| `fast` | 0.05 | rate of the recent reward; at `slow` it removes the long-run want |
| `slow` | 0.005 | rate of the long-run reward, the usual error and the scale |
| `heat` | 2.0 | temperature gain per unit of want on the heated slot; 0 removes it |
| `youth` | 100 | first moments of a life spent aroused |
| `value_surprise` | 1.0 | weight of surprise at the critic's forecast |
| `need` | 0.0 | reward per moment the body requires; 0 removes the need's want |
| `need_scale` | 0.1 | the need as a share of the reward scale, after youth; a constant of the composed brain (an expert `ArousalConfig(need_scale=...)` passed whole moves it) |

`brain.last_arousal` snapshots the latest moment: `mode`, `level`, `error`, `surprise`, `want`,
`temperature`, per-slot `temperatures`, `heated_slot`, `learned`, `recorded`, `sweeps` and
`learning_sweeps`. It is `None` after construction, reset or load. `brain.arousal` counts
`moments` and `sweeps` per mode, `learning_sweeps` and `age`
([class methods](api.md#arousal-cadencearousal)). An action that `step` or `act` sampled is
adopted, so a bootstrapped brain continues through `live` without a reset. Under
`memory_choice=True` the greedy choice includes the
[episodic vote](memory.md#one-choice-for-acting-teaching-and-imagining).

What to expect:

- At `need=0` an adult whose life always paid poorly is not roused; bootstrap it with `step` or a
  longer `youth`.
- The scaled need rouses an adult whose income falls below a tenth of its learned reward spread,
  whatever the unit of its rewards, and leaves a life that never paid at all calm (its scale is
  zero). In a world whose income stays below that share, the want stays at the unmet share and
  the brain stays aroused, paying learning sweeps at every moment. It is outside
  `ArousalConfig.space()`, and through `step` it is inert.
- In the nursery the associative memory carries the adaptation, and a minority of lives never
  find the moved reward.
- In an unchanging world, reward noise rouses the founders for a small share of moments.
- A routine moment pays one full settle.

### Qualification and refusal

Every action, wait and independent prediction must satisfy the full potential/adaptation
equations, observers included, to `learning.tolerance` (`3e-3`) within `learning.free_steps`
(1024). The final residual is always checked against the original model, and the teaching rule
is unchanged by any of the following:

- A hard solve may fall back to half-step numerical damping inside the same budget (a
  `LearnerConfig(qualified=True)` learner uses its `damping` instead).
- A solve that does not qualify within the budget continues once from where it stopped, for
  another `free_steps` sweeps in equal shares at the full step and at successively halved steps
  (the learner's `damping` halvings, at least one). Only then is the answer refused.
  `last_settlement["rescue_steps"]` counts that continuation, and `steps`, `residual_checks` and
  `stagnation_checks` sum both solves.
- A solve that ended on a value that is not finite, and a budget of no sweeps, are refused
  without the continuation. `tolerance=None` is refused with `ValueError`.

`imagine` settles its phases to the same tolerance.

A refused answer raises `RuntimeError` before changing activity, memory, random state or pending
feedback. The retry depends on what was accepted first:

| What refused | State afterwards | Retry |
| --- | --- | --- |
| the forecast settle of `live`, before feedback | unchanged | the same call |
| the answer, after the outcome was accepted (`live`, `step`) | outcome learned and counted | `live(observations)` or `act(observations)` without the outcome |
| learning from the outcome (`learn` feedback solve) | memory writes, trace resets and actor changes rolled back; action pending | the same outcome |

`brain.pending_feedback` says whether the current action awaits its outcome, including a routine
choice; read it before resubmitting.

```python
answer_work = brain.last_settlement
assert answer_work["qualified"] and answer_work["steps"] <= 2 * answer_work["budget"]
assert answer_work["max_residual"] <= answer_work["tolerance"]
```

`last_settlement` holds the latest completed free-answer solve, refused ones included:

- `operation`: `"act"`, `"wait"` or `"predict"` (`step` reports its `act`, `accuracy` and `fit`
  their last `predict` batch);
- `qualified`, per-row `row_qualified` and `residual`, `max_residual`;
- `steps`, `budget`, `tolerance`, `residual_checks`, `damping_halvings` and `stagnation_checks`;
- `rescue_steps` where the solve was continued (`steps` then exceeds `budget`).
- `row_sweeps` under `learning_fast_settle`, the sweeps summed over rows that stop apart.

Validation errors, `learn` and `imagine` leave it as it was; reset and load clear it. It counts
answer work only. Teaching and feedback work, refused teaching included, is in `last_learning`,
and [settlement records](api.md#record-every-settling-step) capture every solver call. Reward
eligibility has its own budget of 12 sweeps (a constant); qualified teaching is opt-in
([budgets](settling.md#qualified-teaching-budgets)).

### Outcomes that arrive later: `wait`

When a body reports an outcome only after more frames, sense those frames with
`wait(observations)`. It settles the frame, advancing activity, working trace, heard history and
recall slots, and issues no action and takes no outcome. The awaited action keeps its forecasts,
eligibility and situation, so its outcome is credited as an immediate one would be.

```python
creature = Brain.compose(4, 2, modules=(16,), seed=0, arousal=True)
creature.live(cues[[0]])
owner = creature.decision_id                 # the action that owns the next outcome
waited = 0
for frame in (cues[[1]], cues[[2]]):         # the reach is under way
    creature.wait(frame)
    waited += creature.last_settlement["steps"]
assert creature.pending_feedback and creature.decision_id == owner
creature.live(cues[[3]], reward=[1.0], decision_id=owner)
assert creature.decision_id == owner + 1     # the outcome was taken once
work = sum(creature.arousal.sweeps.values()) + creature.arousal.learning_sweeps + waited
```

- Waits are outside arousal's counts; add their `steps` to total a stream's work.
- One outcome is one temporal-difference step however many frames were waited.
- `decision_id` is the arousal age of the awaiting `live` action (the first is 1), or `None`. An
  outcome under another identity, or one reported twice, raises `ValueError` and changes nothing.
- Give the awaited outcome an explicit `reward`; an omitted one is a zero outcome.
- There is no deadline. For an outcome that will never come, `act` replaces the action without
  learning and `reset` starts a new stream.
- `wait` refuses without arousal or with several streams (`ValueError`), and with no awaited
  action (`RuntimeError`).

The caller decides which frames are waited and links the late outcome to its action. Where the
brain is to learn that link, as in a delayed key-door reward, report the moments in between
through `live` and leave the credit to eligibility.

### Learning during a continuing life

A brain that keeps living learns from outcomes through `live` and `step`, and from lessons where
a teacher answers. Where the founder's learning does not reach a behaviour, these opt-in settings
do, each read in the chamber that measures it (under `live` with reward only where it learns
from reward):

| Situation | Settings | Chamber and mechanism |
| --- | --- | --- |
| A reversal whose failures pay nothing: the paid arm's record keeps voting after the pay moves | `memory_choice=True, memory_extinction=0.5` | [extinction](../benchmarks/extinction/README.md); [a paid action that stops paying](memory.md#a-paid-action-that-stops-paying-extinction) |
| A chain whose stages look alike and whose misses leave the observation as it was | `memory_choice=True, memory_context_hold="unchanged"` | [context key](../benchmarks/context_key/README.md); [stages that look alike](memory.md#stages-that-look-alike-the-context-hold) |
| A choice that depends on a cue the answer does not show, from the terminal reward alone | `actor_eta=0.1, actor_eta_bias=0.01, memory_choice=True` | [credit](../benchmarks/credit/README.md); [a cue the answer does not show](reward.md#a-cue-the-answer-does-not-show) |
| The same cue carried through distractors, taught by a lesson at the answer | nothing: a new brain's memory has no age charge (`recall_recency=0.0`) | the same chamber and section |
| Which of two writes a read finds, from reward alone | nothing: a new brain holds `recall_position_reward=4.0` and the anchor; the chamber's recipe adds `recall_address_reward=1.0, recall_next=False` | [ordered recall](../benchmarks/sequence/README.md#ordered-recall-orderpy); [position](memory.md#position-how-long-ago-a-moment-happened) |

The chambers' READMEs give each setting's readings and costs.

```python
import numpy as np
from cadence import Brain

bandit = Brain.compose(1, 2, modules=(8,), seed=0, arousal=True, memory_choice=True,
                       memory_extinction=0.5)
cue = np.ones((1, 1))
arm = int(bandit.live(cue)[0])
for trial in range(24):  # arm 0 pays for twelve trials, then arm 1; failures pay nothing
    reward = 1.0 if arm == (0 if trial < 12 else 1) else 0.0
    arm = int(bandit.live(cue, reward=[reward])[0])
assert bandit.last_arousal is not None
```

### Private imagination

```python
phases = brain.imagine([observation, following], budget=1024)
assert all(phase.qualified for phase in phases)
```

`imagine(observations, *, budget=1024)` settles supplied observations (each a batch of the live
streams) on a private trace and reads durable memory without writing it, each phase to the
learner's tolerance within `budget` sweeps. The branch's efference copy takes in the act the
brain would issue at each phase. A refused phase ends the branch and is returned, so check each
phase's `qualified` flags before using a response. Live state, parameters, records, random state
and pending outcomes are untouched. The composed brain integrates no learned model of the
world's consequences.

### Reset and save

`reset(*, keep=("arousal",))` starts every stream afresh. It clears activity, eligibility, any
pending action, the working trace, the efference copy, the heard history and the episodic
context, along with `last_learning`, `last_settlement` and `last_arousal`. It keeps parameters
and associative records; recall slots survive only with `"slots"` in `keep`
([recall](memory.md#the-recall-memory-in-full)). For the arousal:

- **Kept (the default).** Its level, recent and long-run reward, usual errors, reward scale (and
  with it the scaled need), their counts, its age and its work counts carry into the new stream,
  as if `brain.arousal.to_dict()` saved before the reset were restored with `Arousal.from_dict`
  after it. A brain aroused when it was reset begins the new stream aroused.
- **Not kept** (`keep=()` or `keep=("slots",)`). The arousal starts calm with its references
  cleared and its age kept, as after `brain.arousal.reset()`. An adult at `need=0` is then roused
  only by surprise and can stay calm in a world that stopped paying, and the scaled need wants
  nothing until outcomes enter the new stream's scale.

Keep the arousal when the new stream continues the same life. In a two-armed bandit with one
constant observation whose paying arm moved after 300 moments
(`Brain.compose(1, 2, modules=(16,), arousal=True, recall=False)`, seeds 0 to 4):

- Reset with `keep=()`, adults chose the newly paying arm at none of the next 120 moments.
- Reset keeping the arousal, they fell short of the income they were used to, woke and chose as
  they did living on without the reset at every one of the 120 moments, the newly paying arm at
  each of the last 40. Uniform random chooses it at half the moments.
- Reset into a stream that paid as the old one did, they kept their arm, calm, at all 120 moments.
- Reset into one whose arm paid 2.0 in place of 1.0, they were aroused at 9 of 600 moments and at
  none of the last 40 of any seed; adults living on without the reset were aroused at 6 of 600.

Either reset starts the neural state afresh: in the stream that paid as the old one did, the first 5 to 12
moments after a reset settled from rest at 32 sweeps each, where the continuing stream needed no
sweep.

A save carries parameters, critic, the learner's optimizer, random state, every trace and
memory, arousal, an action awaiting feedback and what a waiting stream has sensed; load
validates them ([formats](api.md#brain-cadence)). On the CPU the resumed brain continues to
the same bits; on the NumPy path that holds at one BLAS thread count, so a reloaded life
compared to the bit runs with the thread variables at 1
([a reloaded brain and BLAS threads](backends.md#a-reloaded-brain-and-blas-threads)).
Resume the same rows, supply the pending outcome once and save the environment separately. If `live` refuses an answer, the
[qualification rules](#qualification-and-refusal) say how to retry without reporting an outcome
twice.

```python
brain.save("brain.npz")
resumed = Brain.load("brain.npz")

# Resume the same pending action with the same measured outcome.
reward = (action == 0).astype(float)
continued = brain.live(observation, reward=reward)
replayed = resumed.live(observation, reward=reward)
assert np.array_equal(continued, replayed)
assert brain.arousal.to_dict() == resumed.arousal.to_dict()
assert brain.last_arousal == resumed.last_arousal
```

```python
state = brain.arousal.to_dict()
brain.reset()                                    # the arousal state carries on as it is
assert brain.arousal.to_dict() == state
brain.save("continuing-brain.npz")
resumed = Brain.load("continuing-brain.npz")
```

### Add optional observers

**System 2** adds observer regions to the same graph:

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8,), seed=7,
    arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Observers read from and return influence to the processing regions, the motor region and
earlier observers, inside the same settlement and the same `live` and `step` calls. They attach
to any module count; this project adds them to a developed two-module System 1 by convention.
Any task advantage they bring has to be learned and measured. [Compose a brain](tune.md#brain)
covers modules and observers.

### Defaults and the thinking clock

| Mechanism | Default in `Brain.compose` | Advances on |
| --- | --- | --- |
| Neural activity | retained | each accepted action and each `wait` |
| Memories | working trace, recall memory and associative store on, the rest opt-in | as listed in [memory](memory.md#memory-in-the-composed-brain); `wait` advances the working trace, heard history and recall slots |
| Reward plasticity and demonstrations | `step`, `learn`, `lesson`, aroused `live` | actual outcomes and supplied labels |
| Arousal | off unless `arousal=True` | each outcome `live` receives |
| Observers | none unless requested | the same neural solve |
| Imagination | explicit call | supplied hypothetical observations |
| Backend | `backend="auto"`: the fastest the machine has for the brain's size, chosen at its first moment; float32 on a device | once, at the first moment ([the default backend](backends.md#the-default-backend)) |

A run that must repeat to the bit names `backend="cpu"`. No hidden thread drives the brain; the application owns scheduling. A `step` consumes a real
transition, so never call it per UI frame; frames before a `live` outcome go to `wait`.
[Memory](memory.md) describes each store.

## One continuing equilibrium brain

Cadence builds one brain that acquires a reusable interpretation of its world, carries it
through experience and repairs it when experience disagrees. Perception, memory and action
constrain one another through reciprocal connections, and the answer is read from the settled
state of the brain's motor population.

The learned relations and retained memories encode a family of possible equilibria, one for each
combination of evidence and context. Learning changes the family; ordinary use moves between its
members as observations, goals and context change.

`Brain.compose` builds the default **System 1** brain: reciprocal processing regions, local
plasticity, a working trace and associative memory. Optional **System 2** observers add regions
with returning feedback inside the same settlement. Biological names describe software roles.
The composed brain integrates no learned model of environmental transitions
([issue #93](https://github.com/muellerberndt/cadence/issues/93)).

| Mechanism | Contract and boundary |
| --- | --- |
| Present interpretation and action | `Brain.act` qualifies the whole neural graph's residual with the trace and memory reads held as drive. With `memory_choice=True` the episodic vote joins the motor readout after the settle. The memory stores do not equilibrate jointly with the graph. |
| Settling work | `Brain.last_settlement` reports free-answer residuals, sweeps, checks and qualification, refused attempts included. Eligibility, teaching, feedback, imagination and memory work are excluded; [`record_settlements`](api.md#record-every-settling-step) records every settlement, learning and imagination included, at a memory and copy cost. |
| Memories | Working trace, associative store and recall memory on; efference copy, heard history and episodic memory opt-in. Each is read as held drive (the episodic vote joins after the settle) and written after the settle. [Memory](memory.md#memory-in-the-composed-brain). |
| Local correction | Teacher labels change graph parameters, the address map of a recall memory and, with `memory_choice=True`, the episodic records. Chosen-action outcomes drive reward plasticity and associative writes. A label credits no earlier action. |
| Routine and repair | `Brain.live`, for a brain built with `arousal`, answers greedily from one qualified settle while calm and changes no parameter or associative record; the trace, heard history and recall slots advance. Surprise, a reward shortfall or an unmet `need` (founder 0) or `need_scale` (0.1, a constant) raises arousal, and the outcome that wakes the brain is written to its associative store. The first `youth` moments (100) and aroused moments sample, keep eligibility and learn, from success too. One stream; the law's constants are genes. [Routine and repair](#routine-and-repair-live). |
| Private imagination | `Brain.imagine` settles **supplied** observation sequences with a private trace and read-only durable memory. It learns nothing and generates no environmental transition. [Private imagination](#private-imagination). |

`step` learns from every outcome it is given. `live` gates learning by arousal; its evidence is
the [odour nursery](../benchmarks/reversal/README.md), where the associative memory carries the
repair. Reuse of a settled state across moments and repair confined to declared dependencies are
open work in [issue #122](https://github.com/muellerberndt/cadence/issues/122).

Components for compositions beyond the default brain:

| Function | Build with | Learned or supplied explicitly |
| --- | --- | --- |
| Perception and grounding | Input regions, associations and local updates (`Brain.compose`, `Genome`) | Features, entities, word/referent bindings |
| Episodes and consolidation | `FastSynapses`, `SynapticMemory` | Addressing, relevant writes, retention, correction |
| Workspace and goals | Recurrent state, `Trace`, goal and readback ports | Persistent context, goals and subgoals |
| Action and value | `ActorCritic`, eligibility, dopamine | Exploration, delayed credit |

[One continuing equilibrium brain](world-model.md) states the direction in full: bootstrap, use,
repair and continue; coherent behavior over time; how to build and measure the whole life.
