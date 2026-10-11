# Continuous interaction with Brain

`Brain.compose` creates a continuing **System 1** brain with working trace,
plastic connections and fast/persistent associative memory; optional observer
regions add **System 2** feedback in the same neural graph. For one creature,
construct it with `arousal=True` and use `brain.live(...)`: observe, report the
preceding action's actual outcome, then act again. There is no
training/inference mode switch, and bootstrap, unchanged conditions and
witnessed disruption belong to the same life.

[`live`](#routine-and-repair-live) lets arousal decide when to explore and learn
and when to answer greedily without learning. The explicit `step` loop below is
for batched streams, teacher labels or learning from every outcome. Keep a task
error separate from a numerical failure to settle. The
[quickstart](quickstart.md) is the small starting example, and
[one continuing equilibrium brain](world-model.md) the design.

<a id="observations-actions-and-reward"></a>

## Explicit learning with `step`

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(4, 2, modules=(16, 8), seed=7)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(observation)

# A tiny body rewards action 0, then reports its next observation.
reward = (action == 0).astype(float)
following = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.step(following, reward=reward, done=np.array([False]))
assert action.shape == (1,)
```

Reward and `done` describe the **previous action**. `teacher=` labels the
**current observation**. Every vector has one entry per stream; observations
have shape `(streams, inputs)`. Keep the same stream in each batch row until
`reset()`. The first call has no preceding action to reward.

For an ended row, pass its next episode's reset observation with `done=True`.
A truncation can supply a final value through `bootstrap`. Omitted reward means
zero reward on an actual transition, not an unknown outcome or permission to
advance before the body acts. The critic and eligibility still advance.

```python
# A separate life receives a demonstration for its first observation.
student = Brain.compose(4, 2, modules=(16, 8), seed=7)
student.step(observation, teacher=np.array([0]))
```

Use actual labels and consequences. Teaching from the brain's own guesses can
reinforce mistakes. `teacher` changes graph parameters; it does not write its
label into associative memory. The reward loop records the chosen action's
observed reward, with unobserved action entries masked. For frozen measurements,
use a separate instance. Its `predict` and `accuracy` ignore both the working
trace and associative memory; greedy `act` reads both and advances the trace.
Lower-level `act`/`learn` separates action and feedback timing.
The [continuing brain example](../examples/continuing_brain.py) runs this loop
through bootstrap, unchanged conditions and changed conditions, then checks a
saved pending action's continuation. It reports actual task outcomes without
assuming that settlement guarantees recovery.

The example supplies a corrective label only after an executed mistake and
repeats that cue for the next action. Thus its teacher labels the **current**
observation while reward still describes the previous action. Successes still
reach reward learning and associative memory. This is an application teaching
policy, not automatic suppression of all successful-outcome updates. See
[centered dopamine and Life](reward.md#centered-dopamine-and-selective-activity)
for the narrower selective mechanisms already available.

## A supervised-only stream

`step(teacher=...)` still issues a reward-eligible action. On the next `step`,
omitting `reward` supplies zero on that transition; the actor and critic can
update as well as the teacher. It does not mean "teach without reward learning,"
and it need not preserve the effect of an earlier lesson.

For a stream that has demonstrations but no reward protocol, start a separate
continuing brain and use the lower-level learner with memory-bearing drives:

```python
supervised = Brain.compose(4, 2, modules=(16, 8), seed=7)
supervised.act(observation, greedy=True)  # Read and advance context, without reward eligibility.
_, lesson_work = supervised.learner.step(
    supervised.stimulus(following), np.array([1])
)
answer = supervised.act(following, greedy=True)
assert answer.shape == (1,)
```

The label belongs to `following`, whose drive includes the carried context.
Keep the returned teaching report alongside `last_settlement` when counting
work. Direct learner calls do not populate `Brain.last_learning`; they retain
the learner's configured finite or qualified contract. Greedy actions advance
the working trace without reward learning or associative outcome writes. Do
not use a greedy action to discard an outstanding real outcome from an existing
reward-driven life; finish that outcome first.

## Qualification and refusal

Every `Brain` action and independent prediction must satisfy the full
neural equation residual, including optional observers. Default free answers
have a 1024-step budget and tolerance `3e-3`. The solver may use half-step
numerical damping within that total budget, then checks the original model's
residual. This numerical fallback does not alter the finite teaching rule or add
a second controller.

A refused `act` raises `RuntimeError` before changing activity, memory, random
state or pending feedback. If `step` has learned a real outcome and its following
action refuses, that learning remains. Retry `act` after adjusting the solve;
do not submit that outcome twice. `tolerance=None` cannot disable action
qualification. See [contracts](contracts.md) for the separate finite
free/nudged learning and eligibility rules.

Inspect the latest free-answer attempt without changing the action return type:

```python
answer_work = brain.last_settlement
assert answer_work is not None
assert answer_work["qualified"]
assert answer_work["steps"] <= answer_work["budget"]
assert answer_work["max_residual"] <= answer_work["tolerance"]
```

`last_settlement` is a read-only snapshot with per-row residuals and admission,
sweeps, residual checks, damping halvings and stagnation comparisons. A refused
answer leaves its report available after the exception. It describes the last
completed free-answer solve: validation errors and failures before that solve
do not replace it. `operation` distinguishes `act` from independent `predict`;
`step` delegates to `act`. `reset` and loading a checkpoint clear this diagnostic.

This is not a total cost report. It excludes feedback, eligibility, teaching and
memory work; `last_learning` reports learning separately. Optional
[settlement recording](api.md#record-every-settling-step) captures the other
solver calls at additional cost. The continuing example records those sweeps
too. No numerical counter substitutes for measured task success or hardware
energy, and a surprise need not increase settling work.

Teaching diagnostics, eligibility duration and feedback rollback have
separate contracts:
`LearnerConfig(qualified=True, ...)` requires full-equation qualification
of every supervised free and teaching phase before changing parameters.
`LearningPhaseError` reports an attempted lesson that could not qualify.
Qualified solves use configurable damping; stagnation detection can
admit a smaller numerical step sooner within the same budget and original
equation check. The default finite configuration keeps its half-step fallback.
Accepted and refused teaching diagnostics appear in `brain.last_learning` with
`demonstration_` prefixes, alongside any preceding accepted reward report. They
count attempted presentations and solver work. See [building a brain](brain.md)
for a runnable configuration. Reward eligibility is configured independently:
`ActorCriticConfig.eligibility_steps=12` is the Brain default. For a standalone
`ActorCritic`, `None` uses its learner's `nudged_steps`.

If learning cannot qualify the next state needed for a reward bootstrap, it
preserves the preceding action, memories, optimizer state and random state.
Adjust the solve and retry that same outcome. Once the outcome has been accepted,
a later action refusal has the different retry rule above: call `act` alone.

## Private imagination

```python
phases = brain.imagine([observation, following], budget=1024, tolerance=1e-6)
assert phases
```

Each phase reports its state and `converged` flags. A refused phase is returned
and ends the branch. The private trace can carry earlier hypothetical observations
forward, while live activity, durable memory, random state and pending outcomes
stay unchanged. These are responses to supplied observations, not predictions of
what the environment will do. [Temporal learning and planning](interaction.md)
provide the separate learned action-consequence interface.

## Repetition and salience become lasting synaptic changes

`compose` includes a working `Trace` and `SynapticMemory`; `episodic=False` omits
the associative pathway. The trace retains earlier activity as input to later
settlement. Durable graph plasticity retains learned weights and biases, while
the associative pathway retains cue-to-outcome records. These are separate
mechanisms with separate acquisition and retention tests. Losing an old response
after new teaching can be parameter interference even when no working trace is
involved. A warm numerical starting state alone does not guarantee recall.

`SynapticMemory` has persistent matrix `C`, shared across streams, and fast
residual `F` for each stream. With a normalized key `k` and an observed value `v`:

```text
F *= decay
alpha = min(1, consolidation * (1 + salience))
C += alpha * outer(k, v - k @ C)
F += rate * outer(k, v - k @ (C + F))
```

Defaults are `decay=0.9`, `consolidation=0.05` and `rate=1`. In the reward loop,
salience defaults to `abs(reward)`; explicit `salience=` supplies a nonnegative
vector. This is a supplied importance signal. The signed value determines what
is remembered. Batch writes average persistent updates over observed rows,
and a value mask excludes unobserved actions.

```python
from cadence import SynapticMemory

memory = SynapticMemory(np.arange(3), np.arange(3, 5))
cue = np.array([[1.0, 0.0, 0.0]])
for _ in range(40):
    memory.observe(cue, np.array([[1.0, 0.0]]))
memory.reset(1)  # Clear fast residuals, retaining persistent associations.
assert memory.recall(cue)[0, 0] > 0.85
```

Orthogonal keys preserve one another under the stated rule; correlated keys can
interfere. Storage is fixed and new observations can revise associations.
Rehearsing actual earlier observations can protect recall, but consumes extra
writes and does not guarantee retention. Test old and new responses in the same
brain after intervening experience, with answers free and the relevant transient
state cleared. A successful record-store test does not establish acquisition by
the graph's contrast rule.
Persistent memory costs `key_width × value_width` numbers, plus that amount per
stream for fast weights. Reads neither consolidate nor decay memory. Learning
weights does not require growing new anatomical connections.

## Routine and repair: `live`

`step` samples an action, keeps its eligibility and learns from its outcome at
every moment. `live` lets the brain's arousal decide. A calm brain answers with
the greedy choice of one qualified settle and learns nothing: no eligibility
phases, no parameter change, no memory write, and the eligibility of its earlier
sampled actions fades with each moment.

Two things rouse it:

- **Surprise**: an outcome that contradicts the forecast it made before acting.
  The forecast is the critic's value of the situation and, through a gene the
  founders weigh at zero, the record it holds for the action it chose. Only
  outcomes of its own greedy choices can surprise it or move its usual forecast
  error.
- **Want**: a reward below what its life usually pays, which also covers a
  failure it predicts correctly, or a reward below the body's `need` — a gene
  the founders set at zero, measured as the share of the need left unmet, which
  a rare reward's small mean cannot dilute and which never habituates.

An aroused brain samples its policy, keeps eligibility, learns from every
outcome and writes the outcome that woke it to memory at once. Want raises the
temperature of one uniformly chosen motor slot per moment, so extra exploration
does not flatten every motor choice at once; a one-slot brain keeps its existing
sampling law. Eligibility credits the actual temperatures used, and
`last_arousal["temperatures"]` records them beside `heated_slot` (`None` when no
extra heat was applied). Every actual reward updates the recent and long-run
income, including sampled choices, so a brain exploring with many slots does not
stay aroused by an old shortfall. It can habituate to a poorer life while
sampling; a positive unmet `need` still keeps it wanting.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(
    4, 2, modules=(16,), seed=0,
    working_memory_amplitude=0.3,   # the trace informs; the present input leads
    consolidation=0.25,             # lasting memory takes half of a witnessed unit outcome
    arousal=True, arousal_youth=30,
    actor_eta=0.1, actor_eta_bias=0.01,  # this chamber's measured rates
)

cues = np.eye(4)
rng = np.random.default_rng(0)
cue = 0
action = brain.live(cues[[cue]])
modes, hits = [], []
for moment in range(600):
    rewarded = cue % 2 if moment < 300 else 1 - cue % 2   # the rule turns over halfway
    hits.append(action[0] == rewarded)
    reward = 1.0 if hits[-1] else -1.0
    cue = int(rng.integers(4))
    action = brain.live(cues[[cue]], reward=[reward])
    modes.append(brain.last_arousal["mode"])

assert modes[250:300].count("routine") >= 45 and sum(hits[250:300]) >= 45  # calm and right
assert modes[300:340].count("aroused") >= 20                               # the change wakes it
assert modes[-50:].count("routine") >= 45 and sum(hits[-50:]) >= 45        # repaired, calm again
assert brain.arousal.moments["routine"] > 4 * brain.arousal.moments["aroused"]
```

Reward and `done` describe the preceding action, as in `step`, and `live`
follows one stream. `brain.last_arousal` holds the readings of the moment and
`brain.arousal` the stream's state with the moments and settling sweeps of each
mode; [the API](api.md#arousal-cadencearousal) gives the law. A brain taught
through `step` continues through `live` without a reset: the action `step`
sampled is adopted. If the forecast settle refuses, nothing has changed and the
same call can be retried. If the answer refuses after the outcome was taken, the
outcome stays learned: retry with `live(observations)` alone. The read-only
`brain.pending_feedback` reports whether the current action still awaits its
outcome, including a routine choice.

When an action awaits feedback, omitting `reward` supplies zero, as in `step`;
it does not represent a missing or delayed outcome. Do not advance this stream
with `live` before the body's actual outcome; observations that arrive first go
to [`wait`](#outcomes-that-arrive-later-wait). Youth and sustained arousal permit learning
from successful outcomes too. The arousal statistics control sampling and
eligibility outside the neural solve; they are not another settled patch or a
certificate of task failure. Every answer still comes from the qualified graph,
and learning uses its existing local updates and associative write rule.

The constants of the law are genes, `ArousalConfig`, and the values above are
hand-set founders. The three other settings are the operating point of one
continuing stream measured on the
[odour nursery](../benchmarks/reversal/README.md): at the composed defaults the
working trace outweighs the present input of a continuing life, and the actor
rate, selected on batches of streams, locks one stream's policy. They are that
chamber's development settings, to be selected again for another task.

What this establishes is bounded. With the founder `need=0`, arousal responds to
change, so a brain whose life has always paid poorly and whose youth has ended is
not roused by it: a long bootstrap belongs to `step` or to a longer `youth`. A
positive `need` keeps an unmet want active, with its benefit to be measured. In
the nursery the associative memory carries the adaptation; the graph's reward
learning alone does not acquire the task in one stream, and a minority of gated
lives never find the moved reward. A routine moment still pays one full settle,
and a settled routine answer satisfies the neural equations while it can still
be wrong about the world; the next outcome is what tells.

Change a measured operating point with [`brain.retune(...)`](brain.md#retune-the-same-life),
using `actor_*` for reward rates and `learning_*` for teaching rates. It preserves
acquired state and pending outcomes; arousal resets are explicit. Inspect effective
settings with `brain.describe()` and keep the environment's stage in its own save.

### Outcomes that arrive later: `wait`

A body may report an action's outcome only after the stream has sensed more: an
arm that needs several frames to finish a reach, or a reply that comes back over
a network. Not calling the brain until the outcome arrives keeps that action's
custody, but nothing sensed in between reaches the brain. Calling `live` per
frame senses every frame, but takes each one as the outcome, zero when the reward
is omitted, and takes the real outcome as the outcome of the last action issued.
`wait` settles such a frame without taking an outcome or issuing an action:

```python
creature = Brain.compose(4, 2, modules=(16,), seed=0, arousal=True)
action = creature.live(cues[[0]])
owner = creature.decision_id                   # the action that owns the next outcome
writes = creature.hippocampus.writes
waited = 0                                     # settling sweeps spent while waiting
for frame in (cues[[1]], cues[[2]]):           # the reach is still under way
    creature.wait(frame)
    waited += creature.last_settlement["steps"]
assert creature.pending_feedback and creature.decision_id == owner
assert creature.hippocampus.writes == writes   # no frame was taken as an outcome
action = creature.live(cues[[3]], reward=[1.0], decision_id=owner)  # an explicit outcome
assert creature.decision_id == owner + 1       # the outcome was taken once
counted = sum(creature.arousal.sweeps.values()) + creature.arousal.learning_sweeps
work = counted + waited                        # waits are outside arousal's counts
assert work > counted > 0 and waited > 0
```

Waiting moves the stream's activity and working trace with what it senses;
`last_settlement` reports each settle with `operation` `wait`, and like the work
of a refused attempt it is not part of `arousal`'s counts, which still add up the
readings of the live moments. A stream's settling work is those counts plus the
`steps` of every wait, as the example adds them. The awaited action keeps the
forecasts made before it, its eligibility and the situation it was chosen in, so
its outcome is credited as an immediate outcome would be: the same eligibility,
forecast and associative record. What settles next starts from where the stream
now is: the outcome's next state, a routine forecast, an answer that replaces the
action and `imagine`; a finished episode still starts from rest.
Parameters, eligibility traces, associative memory, random state, the copy of
the issued command and the arousal state do not change while waiting.
Eligibility, arousal, its age, youth and `need` advance with live moments, and
one outcome is one temporal-difference step however many frames were waited.

This is an event-time rule: a waited frame does not discount the outcome, fade
eligibility or add to a want. In animals the eligibility of an action fades with
the time that passes before its outcome; a per-frame decay is a candidate gene,
with this rule as its control. Event time belongs to
[#116](https://github.com/muellerberndt/cadence/issues/116), and irregular
physical time is among the open delay items of
[#111](https://github.com/muellerberndt/cadence/issues/111).

The caller decides which frames are waited, as the environment decides when an
outcome is reported; the brain does not choose to wait, and this is not the
learned behavior of waiting through a pause. `decision_id` is the arousal age
once `live` issued the awaited action (its first action is 1), or `None` when no
`live` action awaits an outcome. An outcome reported under any other identity
raises `ValueError` and changes nothing, including the same outcome reported
again once it was taken; pass it when a body or a network may deliver an outcome
twice or late. Give the awaited outcome with an explicit `reward`: an omitted
reward after a wait is a zero outcome, which `live` takes as that action's
outcome. Waiting has no deadline. An outcome that will never come is not a
zero reward: `act` replaces the action without learning from it (`step`, like
`live`, would take a sampled action's omitted reward as a zero outcome), and
`reset` begins a new stream. A save during the wait resumes it.

The caller also supplies the link between a late outcome and the action it
belongs to; the brain does not infer it. That fits an action still under way in
the body or in a channel. Where the brain is to learn the link between a choice
and its delayed consequence, as in the delayed key-door reward of
[#111](https://github.com/muellerberndt/cadence/issues/111) or a cache dug up
later, the moments in between are actions with outcomes of their own: report them
through `live` and leave the credit to eligibility. A result obtained with `wait`
declares the link as supplied by its adapter.

This is the custody of one awaited action in one stream. It does not establish
better behavior from the additional sensing, the timing of a real body, several
actions in flight, or that the body executed the action it was given.

## Reset and save

`brain.reset()` clears live neural/eligibility state and the working trace,
retaining associative memories. `brain.hippocampus.reset(batch)` clears fast
residuals but retains persistent synapses; `clear()` erases both. Changing memory
batch size resets fast residuals on a write; reads use the persistent baseline
without changing the live memory.

```python
brain.save("continuing-brain.npz")
resumed = Brain.load("continuing-brain.npz")
```

Save/load includes parameters, critic, optimizers, traces, fast and persistent
memory, random state, arousal, an action awaiting feedback and what a waiting
stream has sensed since. Resume the same rows and supply that action's actual
outcome once. Save the environment separately.
If a pattern separator is used, its actual projection and running mean are
saved too. Shapes, finite values and continuation state are validated on load.

## Defaults and the thinking clock

| Mechanism | Default in `Brain.compose` | Advances on |
| --- | --- | --- |
| Neural activity | Retained | Actual interaction, including frames sensed by `wait` |
| Working trace | Included | Each admitted action's free state and each `wait` |
| Reward plasticity and demonstrations | Available through `step` | Actual outcomes and supplied current labels |
| Arousal | Absent unless enabled, e.g. `arousal=True` | Each outcome `live` receives |
| Fast/persistent associations | Included | Observed chosen-action outcomes |
| Recursive observers | Empty unless requested | The same neural solve when included |
| Private imagination | Explicit call | Supplied hypothetical observations |

`Brain.build` is a separate configurable builder; its working trace is
opt-in. No hidden thread drives either interface. Do not call `step` for every UI
frame: it consumes a real transition. Frames that a `live` stream senses before its
outcome go to `wait`. The application owns scheduling.

Advanced `Deliberator` search retains unfinished work across bounded `tick`
calls using supplied actions, transition and evaluator. It does not automatically
override the brain's action. Keep the model fixed during a search and restart
after learning; imagined outcomes must not train the live brain. Its node budget
does not bound a callback's wall time. See [the API](api.md) and
[deliberation tests](../tests/test_deliberator.py).

For partially filled environment pools, lower-level `ActorCritic.learn(...,
observed=active_rows)` masks padding transitions. Stop when no real rows remain.
For visualizations, record actual iterations and their action/outcome identity
with [record_settlements](api.md#record-every-settling-step); do not invent
oscillations or a reward signal from an absolute update statistic.
