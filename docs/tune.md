# Tune and read a living brain

`describe()` says what a brain is: its layout, its genes, its memories and whether an action
awaits its outcome. `retune(...)` changes named settings within the same life.
`last_settlement`, `last_arousal` and `last_learning` report the latest moment's work. The
arousal law that decides when a `live` brain learns is in
[routine and repair](start.md#routine-and-repair-live).

`Brain.compose` builds **System 1**: one continuing neural graph with senses, reciprocal
processing regions, a motor choice, a working trace and an associative memory, all settling
together. Optional observer regions add **System 2** feedback inside the same graph. Keep one
composed brain through acquisition, use and correction
([one continuing equilibrium brain](start.md#one-continuing-equilibrium-brain)).

```python
import cadence as cd
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.live(observation)
assert action.shape == (1,)
```

`live` takes one stream's observation of shape `(1, inputs)` and returns one action index
([continuous interaction](start.md#continuous-interaction) covers `live`, `step` and saving).

## What to use

| You want | Use | Section |
| --- | --- | --- |
| To know what a brain is | `describe()`: its layout, its genes, its memories, the pending outcome | [defaults](#defaults-and-expert-overrides) |
| To know what the last moment did and cost | `last_settlement`, `last_arousal`, `last_learning` | [the API](api.md#attributes) |
| Another value from a brain's first moment | the setting's name at `compose` | [defaults](#defaults-and-expert-overrides) |
| Another value in a life under way | `retune(...)` | [retune the same life](#retune-the-same-life) |
| A new part on a trained brain | `grow(...)`, `graft(...)` | [grow and combine brains](grow.md) |
| An action with parts (kind, pitch, octave) | `slots=(3, 2, ...)` | [motor competition](#operating-point-and-motor-competition) |
| The next `k` actions settled together | `actions=k * menu, slots=k` | [a planning window](#a-planning-window) |
| Vector senses and several motors, a brain in a body | one motor slot per motor, `live` and `retune` across stages, as in the [Robot Arena demo](https://github.com/muellerberndt/cadence-demos/tree/main/robot-arena) | [one experience step by hand](#one-experience-step-by-hand) |
| Pixels and a teacher | image `inputs`, teaching through `step`, as in the [Atari demo](https://github.com/muellerberndt/cadence-demos/tree/main/atari-arcade) | [wide inputs](senses.md#image-inputs) |
| Features of features | several `modules`, `apical`, `teaching="deep"` | [deep brains](#deep-brains) |
| Modules of several hundred neurons | a new brain's rates follow its width; surprise lessons with `learning_surprise_share=True` | [wide brains](#wide-brains) |
| To time a body's own moments | `act` and `learn` | [one experience step by hand](#one-experience-step-by-hand) |
| To draw what a brain does | `record_settlements` | [the brain in a browser page](#the-brain-in-a-browser-page) |
| Which memory to use | the table of the memory guide | [memory](memory.md#start-with-the-need) |

## Brain

The composed graph, in the order its regions are developed:

| Region | Neurons | Wiring | Home |
| --- | --- | --- | --- |
| `sensory`, or `visual` for an image | `inputs`, the `code=units` code, or the visual cortex | Projects to the first module (to `code` with a learned code); initial scale `scales["sensory"]` | [image inputs](senses.md#image-inputs) |
| `code` | `learned_code` units | With `learned_code=units`: the dictionary from `sensory`, lateral inhibition through its Gram matrix, the predictor from `prefrontal`; projects to the first module at `scales["sensory"]` | [the learned code](senses.md#the-learned-code) |
| `module_0` ... | `modules[:-1]` | A chain of reciprocal projections | this page |
| `association` | `modules[-1]` | Reciprocal with `motor` | this page |
| `prefrontal` | one per association neuron | The working trace of the settled association, held at `working_memory_amplitude` 3.0, read through a plastic projection of initial scale `scales["trace"]` 1.2 | [memory](memory.md#working-trace) |
| `motor` | `actions`, in `slots` | Lateral inhibition within a slot | [motor competition](#operating-point-and-motor-competition) |
| `observer_0` ... | `observers` | Reciprocal with every module, `motor` and earlier observers | below |
| `efference` | `actions` | With `efference_amplitude` above 0 | [memory](memory.md#efference-copy) |
| `history` | sensory width per decay | With a nonempty `history_decays` | [memory](memory.md#heard-history) |
| `recall`, `next` | observation-code width | With `recall=True`; `next` unless `recall_next=False`; with `recall_copy` also into `motor` | [recall memory](memory.md#the-recall-memory-in-full) |
| `recall_action`, `next_action` | `actions` each | With `recall_action=True`, one per read port region; synapses into the association start at zero | [its own actions](memory.md#its-own-actions-in-the-slots-opt-in) |
| `recall_when`, `recall_1_when` ... | 2 each, 4 with `recall_anchor` | With `recall_when=True`, one per head; synapses into the association start at zero | [when a read item happened](memory.md#when-a-read-item-happened-opt-in) |

The hippocampus is a `SynapticMemory` from the observation code to the motor drive; with
`memory_choice=True` it is the
[episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before), and with
`episodic=False` it is absent. The default `modules=(64,)` is one region settling recurrently
with the others; in `modules=(16, 8)` the last entry is the association region and sets the
working trace's width. An action is issued only from a state, observers included, that meets
the full-equation tolerance ([qualification and refusal](start.md#qualification-and-refusal)).

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8, 4), seed=7, arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Observers settle with the base and never override a finished answer. The wiring permits
recursive correction without teaching it.

## Defaults and expert overrides

Set only the values you intend to change; the rest keep their composed founders, which are the
control for any change you measure.

```python
tuned = Brain.compose(
    inputs=4, actions=2, arousal=True,
    scales={"sensory": 2.0}, actor_eta=0.1, actor_eta_bias=0.01, arousal_youth=30,
)
settings = tuned.describe()
assert settings["genes"]["actor_eta"] == 0.1
assert settings["layout"]["composition"]["sensory_scale"] == 2.0
```

`describe()` returns four JSON-safe sections: `layout` (the regions; `code`, what the memories
key on; `composition`, the construction choices, not current weights), `genes` (every setting
by the name `compose` and `retune` take, `teaching` among them), `memories` (each member of
`brain.memories` with its settings and counts) and `pending` (whether an issued action owns the
next outcome). For an expert `LearnerConfig` outside the named teaching points,
`genes["teaching"]` is `"custom"` and `genes["learning"]` gives its effective `eta`, `eta_bias`,
`normalize` and `normalize_floor`.

| Setting | Composed founder | Changed at | Home |
| --- | --- | --- | --- |
| `modules`, `observers` | `(64,)`, `()` | construction | [map](#brain) |
| `apical` | `0.0`: every synapse summed alike | construction | [the apical compartment](anatomy.md#the-apical-compartment) |
| `lateral`, `slots` | `0.0`, `1` | construction | [motor competition](#operating-point-and-motor-competition) |
| `scales`: `"sensory"`, `"trace"` | `1.0`, `1.2` | construction | [image inputs](senses.md#image-inputs), [memory](memory.md#working-trace) |
| `features`, `field`, `seed` | `8`, `3`, `0` | construction | [image inputs](senses.md#image-inputs) |
| `attention` | `False`; `True` requires `learned_code=units` and supplies the contextual Recall recipe | construction | [single-head learned-code attention](memory.md#single-head-learned-code-attention-opt-in) |
| `teaching` | `"founder"`: steps 0.5 and 0.05 on the raw contrast; `"normalized"`: 0.003 and 0.0003 under a running RMS (0.99, floor 1e-4); `"deep"`: the normalized point with the projection-relative floor 0.1 and nudged phases that run their whole budget, set to 32 sweeps on the way in (a brain keeps the budget it has) | retune | [learning](teach.md#many-lessons-the-normalized-point-for-imitation), [deep brains](#deep-brains) |
| `temperature` | `0.2` | retune | [learning](settling.md#7-every-knob) |
| `learning_free_steps`, `learning_nudged_steps`, `learning_tolerance`, `learning_qualified`, `learning_damping` | `1024`, `12`, `0.003`, `False`, `3` | retune | [learning](settling.md#qualified-teaching-budgets) |
| `learning_scale_cap`, `learning_surprise`, `learning_surprise_share` | `8.0`, `False`, `False` | retune | [learning](settling.md#7-every-knob) |
| `learning_rehearsal` | `0.0` (off) | retune | [rehearsal](teach.md#teach-several-types-in-sequence-rehearsal) |
| `learning_fast_settle` | `False`, `True` in a new brain with the memory | retune | [settling](settling.md#settling-and-equilibrium) |
| `learning_rehearsal`, `learning_rehearsal_capacity` | `0.0` (off), `512` lesson rows, each with its episode | retune | [rehearsal](teach.md#teach-several-types-in-sequence-rehearsal) |
| `actor_gamma`, `actor_lam`, `actor_eta`, `actor_eta_bias`, `actor_eta_critic` | `0.9`, `0.8`, `1.0`, `0.05`, `0.3` | retune | [reward](reward.md#eligibility-traces-and-dopamine) |
| `maturity`, `maturity_scale`, `maturity_recovery` | `False`, `10.0`, `200000.0` moments | `maturity` at construction; the scale and the recovery retune | [maturity](reward.md#maturity-opt-in) |
| `arousal` | off; `True` is `ArousalConfig()` | construction (switch), retune (fields) | [continuous](start.md#routine-and-repair-live) |
| `arousal_threshold`, `arousal_decay`, `arousal_tolerance`, `arousal_floor`, `arousal_fast`, `arousal_slow` | `0.2`, `0.9`, `2.0`, `0.1`, `0.05`, `0.005` | retune | [continuous](start.md#routine-and-repair-live) |
| `arousal_heat`, `arousal_youth`, `arousal_value_surprise`, `arousal_need` | `2.0`, `100`, `1.0`, `0.0` | retune | [continuous](start.md#routine-and-repair-live) |
| `working_memory_*` | amplitude `3.0`, decay `0.0` | retune | [memory](memory.md#working-trace) |
| `carried_gain` | `0.0` (off), `0.01` in a new brain with the memory | construction | [memory](memory.md#how-strongly-a-carried-region-is-heard) |
| `recall_drive_gain` | `False`, `True` in a new brain with the memory | construction | [memory](memory.md#the-reads-drive-as-a-route) |
| `efference_*` | amplitude `0.0` (absent), decay `0.2` | a positive amplitude at construction adds it | [memory](memory.md#efference-copy) |
| `history_*` | decays `()` (absent), amplitude `3.0` | decays at construction | [memory](memory.md#heard-history) |
| `episodic`, `memory_rate`, `memory_amplitude`, `consolidation` | `True`, `1.0` (`0.5` with `memory_choice`), `1.0`, `0.05` | `episodic` at construction | [memory](memory.md#associative-store-the-hippocampus) |
| `code` | unset: the observation itself | construction | [memory](senses.md#the-observation-code) |
| `memory_choice` and the other `memory_*` episodic settings | `False` | construction | [episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before) |
| `recall` and every `recall_*` | `True` up to 4,096 code coordinates; `recall=False` builds no memory | see the page | [recall memory](memory.md#the-recall-memory-in-full) |
| `backend`, `device`, `dense_limit` | `"cpu"`, `None`, `2048` (with `recall=True` as wide as dense blocks) | construction | [backends](backends.md) |

Constants of the composed brain, not settings: the nudge `beta` 0.1, the learner's `momentum`
0.9, the rehearsal's reservoir of 512 lesson rows, the dopamine cap 1.0, reward eligibility phases
of 12 sweeps, the arousal `need_scale` 0.1,
the episodic memory's decay 0.9, replay of 8 decisions, demonstration strength 1.0 and 24
winners per key, and the `history` and `recall` projection scales 1.2. Only an expert
configuration passed whole moves them.

Override rules:

- Every field of `LearnerConfig`, `ActorCriticConfig` and `ArousalConfig` is a
  `learning_<field>`, `actor_<field>` or `arousal_<field>` name, except the fields a teaching
  point sets (`eta`, `eta_bias`, `normalize`, `normalize_floor`, `normalize_relative`,
  `nudged_fixed`) and the constants above: their names raise `ValueError`, as does a retired
  setting, with a message naming what replaced it ([CHANGELOG](../CHANGELOG.md)).
- `temperature` sets both the teaching softmax and the sampled policy.
- Unknown names and the ambiguous bare `eta` or `eta_bias` raise `TypeError`.
- `arousal=True` uses the founders, a mapping such as `arousal={"heat": 1.0}` patches them, an
  `ArousalConfig` replaces them, and `arousal_*` names need one of the three. Without
  `arousal`, `live` raises `ValueError`.
- A changed actor rate keeps the existing bias rate; `actor_eta_bias=None` derives one tenth of
  the new rate.
- A setting for an absent component, such as `memory_rate` with `episodic=False`, has no
  effect; `describe()` omits it and `retune` refuses it.
- `learning=LearnerConfig(...)`, `reward=ActorCriticConfig(...)` and
  `arousal=ArousalConfig(...)` replace the whole configuration, constants included, and a fresh
  config replaces every founder: `LearnerConfig()` has `eta=0.2`, `momentum=0`, `normalize=0`
  (raw steps), budgets `100/50` and tolerance `1e-4`; `ActorCriticConfig()` has `gamma=0.99`,
  `lam=0.9` and rates `0.5/0.05/0.05`. Use named overrides for a one-field edit.

## Retune the same life

`retune` changes named settings on a living brain, for example from practice to ordinary
interaction:

```python
tuned.retune(
    arousal={"need": 0.0, "heat": 0.0},
    temperature=0.2,
    actor_eta=0.03, actor_eta_bias=0.003,
)
```

It takes the same names, validates the whole request before installing anything, runs no
moment and returns the same brain. Parameters, optimizer history, activity, traces, records,
random state and pending feedback are kept, and new rates apply from the next learned outcome.
It raises `ValueError` for a construction setting and for a missing component (arousal
included); with `recall_address_reward` above zero, also for `recall_address` and
`recall_address_reward` ([recall memory](memory.md#the-recall-memory-in-full)). To add a part
that a construction setting adds (the recall memory or more of its heads, a heard history, an
efference copy, new output slots) to a trained brain, use `grow`, which returns a new brain;
`graft` puts trained regions of another brain into it ([grow and combine brains](grow.md)).

`retune` resets nothing. `brain.arousal.reset()` resets the arousal level and reward references
and keeps its age and work counters; an adult brain at the founder `need=0` is then roused only
by surprise at its own forecasts ([reset and save](start.md#reset-and-save)).

## Operating point and motor competition

`lateral` is the signed weight between each pair of motor neurons in a slot; the default `0.0`
removes those synapses and keeps the association and motor feedback. A negative weight makes the
actions of a slot compete, but at -0.5 the undamped free solve oscillates from 4 actions on:
acts settle only under the damping fallback, at many times the sweeps, and lessons run out
their budget. Choose a competing readout against `0.0` on development tasks.

`slots` groups the motor neurons into readouts that settle together, one softmax each: a count
of equal groups or one size per group. An action with parts (kind, pitch, octave) is one slot
per part. `act` and `step` return one index per slot, a nonzero `lateral` is wired within a
slot, and `act(temperature=...)` takes one value or one per slot. `save` and `load` carry the
grouping.

```python
chord = Brain.compose(inputs=4, actions=5, slots=(3, 2), seed=3)
chosen = chord.act(observation, temperature=[0.5, 0.1])
assert chosen.shape == (1, 2)
```

### A planning window

A brain that speaks, plays or moves one action per moment also plans one action per moment:
whatever the next action must agree with has to survive in the trace or be read back from
memory. A planning window ([#257](https://github.com/muellerberndt/cadence/issues/257)) settles
the next `k` actions jointly under the meaning the brain holds, emits them and moves on. It is
`slots` with the same menu in every slot: compose with `actions=k * menu, slots=k`. One settle
answers the `k` slots together, so a constraint between two of them (a verb and its agent, a
bracket and its closing) is met inside that settle; a lesson teaches the `k` slots at once with
labels shaped `(batch, k)`; `act`, `step` and `live` return `k` indices. The window advances
when the emitted actions join the next moment's context: the world echoes them (the caller
hears the window back, each action at 1.0 with a turn flag), or the
[efference copy](memory.md#efference-copy) holds the `k` commands issued. What lies beyond the
window is the memories' to carry.

```python
import numpy as np
from cadence import Brain

k, menu = 4, 18
speaker = Brain.compose(inputs=35, actions=k * menu, slots=k, seed=0, efference_amplitude=1.0)
message = np.eye(35)[[20]]                              # a meaning, as the chamber cues one
window = speaker.act(message, greedy=True)              # (1, 4): four words in one settle
assert window.shape == (1, k)
speaker.lesson(message, np.array([[3, 12, 7, 17]]))     # the four words taught together
copied = speaker.efference.trace[0].reshape(k, menu).argmax(axis=1)
assert copied.tolist() == window[0].tolist()            # the window issued, in the copy
```

Each slot has motor neurons and synapses of its own: a position in the window is a coordinate
the brain learns word by word, not a role the same synapses fill at every place, so a window
taught on sentences of up to `k` words says nothing about the words after the `k`th. Weight
sharing across positions is a second form, read only if this one holds. The founder is one slot
and is unchanged. The [grammar chamber](../benchmarks/grammar/README.md#a-planning-window)
reads the window on its sentences at `k` of 1, 3, 4 and 8.

Every bias starts at zero and is plastic: `brain.brain.bias` holds the learned values, and
loading restores them. To start the processing neurons at a bias of your own, set
`brain.learner.brain = brain.learner.brain.with_parameters(bias=bias)` before the first lesson.
A signed neuron below rest responds to input and contributes to teaching, so the share of
nonpositive emissions is an activity diagnostic (`tests/test_operating_range_function.py`).

Gain, bias and solver steps are distinct levers: gain scales synaptic drive in the whole graph,
a population bias moves selected neurons' operating points, and steps and damping change only
how the equations are solved. Check candidate biases with the
[calibration guide](settling.md#calibrating-the-operating-point), choose on development inputs
and compare against the founders on your own task ([evaluation](evaluate.md)).

## One experience step by hand

`step` learns from the previous outcome and chooses the next action. Use `act` and `learn` when
the body manages that timing:

```python
body = Brain.compose(4, 2, modules=(16, 8), seed=7)
chosen = body.act(observation)
reward = (chosen == 0).astype(float)  # action 0 earns one unit
following = np.array([[0.0, 1.0, 0.0, 0.0]])
body.learn(reward, np.array([False]), following)
next_action = body.act(following)
```

Reward and termination describe the preceding executed action; a demonstration labels the
current observation. Keep batch-row identities fixed until `reset()`. The
[continuing brain example](../examples/continuing_brain.py) runs the full reward, teacher and
checkpoint loop.

## What a moment costs

One thread of the CPU backend with Numba does a moment's work: the memory's read, the free solve,
the choice and the memory's write. A lesson adds a solve that judges the answer, which the act after
a lesson that taught no row reads instead of settling again, and at the rows it teaches two nudged
phases and the update. The kernels compile at their first use under a source version, about 10
seconds on an Apple M4, and load from Numba's cache after that.

A pass of the [grammar chamber](../benchmarks/grammar/README.md) is 4,000 batches of 16 rows, about
50,000 acts and 23,000 lessons. On one core of a Xeon 8488C kept apart from other work:

| Learner | Parameters | Seconds | Multiply-adds | Against the transformer: seconds, multiply-adds |
| --- | ---: | ---: | ---: | ---: |
| The chamber's recipe with `teaching="deep"` | 24,384 | 131 | 1.6e11 | 8.0, 2.8 |
| The same with `learning_fast_settle=True` | 24,384 | 74 | 1.3e11 | 4.5, 2.3 |
| The recipe with `history_decays=(0.5, 0.9)` and no prefrontal region | 24,774 | 162 | 2.2e11 | 10, 3.8 |
| The same with `learning_fast_settle=True` | 24,774 | 73 | 1.6e11 | 4.5, 2.7 |
| The chamber's transformer | 23,490 | 16 | 5.7e10 | 1, 1 |
| The chamber's LSTM | 24,378 | 8 | 5.7e10 | 0.5, 1 |

Two things remain between the brain and these networks at equal compute. The first is arithmetic: a
moment settles, 65 to 90 sweeps between the association and the motor regions, where a network runs
each word once forward and once back, so a pass costs 2.3 to 3.8 times their multiply-adds. The
second is rate: at equal multiply-adds the brain takes 1.7 to 2.9 times the transformer's seconds
and 3.5 to 6 times the LSTM's. In the recipe's pass the free solves take more than half of the time
and the exponential in the activation about a third of that. Under `learning_fast_settle` the solves
take about three tenths and the memory's read a sixth, and most of the rest is Python and NumPy,
many small steps around the kernels that a network's compiled batch does not make. What a pass reads
at these costs is in the [chamber's tables](../benchmarks/grammar/README.md).

| Work | Brain | One thread, Apple M4 |
| --- | --- | ---: |
| One stream of speech: 3,920 inputs, association 512, one act per 500 ms of audio | founder settings | 0.9 ms an act, 25 ms a taught lesson |
| The same | the recipe | 10 ms, 190 ms |
| The same | the recipe with `history_decays=(0.5, 0.9)` | 14 ms, 270 ms |
| Long context: 16 streams, 256 inputs, 64, 1,024 and 4,096 slots | the chamber's recipe | 1, 6 and 21 ms an act, 7, 27 and 99 ms a moment with its lesson |

At the speech size the read's address map, 3,920 by 3,920 entries, is most of an act, and with
thousands of slots the read is: at 4,096 slots it is 20 of the act's 21 ms, and the write 0.3 ms. A
realtime loop's limit is the act: at the speech size a lesson that teaches costs about twenty times
an act, and only the act has to keep up with the stream.

| To spend less | Setting | What changes |
| --- | --- | --- |
| Sweeps per solve | `learning_fast_settle=True`, a new brain's | Another solve of the same equations, in which rows stop apart and the activation vectorizes: about half of a chamber pass on the Xeon above, and .74 of one on an Apple M4 (229 to 169 CPU seconds, four pairs at the default brain on one thread of a shared machine), whose exponential costs less. `learning_fast_settle=False` names the ordinary solve, with other bits. |
| Sweeps per solve | a larger `learning_tolerance` | At 0.01 in place of 0.003, about a fifth fewer sweeps per act and an eighth of a chamber pass. |
| Slots | a smaller `recall_capacity` | A read grows with the slots a stream holds; a write does not. |
| Address map | `code` at construction | At the speech size `code=1024` takes about a sixth of an act and a quarter of a taught lesson, with 3.6 million parameters in place of 24 million. The memory then addresses the code, so read your own panels. |
| Blocks | `dense_limit` at construction | A brain over the limit settles on the sparse transport in Python, about 30 times slower at the speech size. A brain with the memory takes a limit as wide as its blocks unless it names one. |
| Device | `backend="mlx"` from the speech size | At the speech size MLX on the same M4 takes a fifth of an act and a twelfth of a taught lesson. At the chamber's size, 16 rows over 82 settling neurons, it takes about 1.2 times as long per batch. [Which hardware](backends.md#which-hardware) gives the sizes where a device pays. |

## Larger brains

Deeper and wider brains need settings the founder does not set; each recipe below names the
benchmark that measured it.

### Deep brains

Several processing regions learn features of features only with the
[apical compartment](anatomy.md#the-apical-compartment) and `teaching="deep"`. The recipe of the
[depth benchmark](../benchmarks/depth/README.md):

```python
deep = Brain.compose(
    inputs=48, actions=7, modules=(128, 64, 64, 64), apical=0.25, teaching="deep", seed=0,
)
```

- **What it buys.** On a task that needs two layers of thresholds, one module reads 0.58 of
  held-out rows after 4,000 lessons, two 0.66, three 0.81 and four 0.86, against 0.89 for a
  network of two hidden layers trained by backpropagation.
- **Why the teaching point.** Credit shrinks about fourfold per region below the motor region;
  the projection-relative floor lets each projection step on its own scale, and the fixed
  nudged phases carry the nudge down to the lower regions.
- **Cost.** An apical brain settles at half the founder's step, in fewer sweeps than a flat
  deep brain.
- **Lessons.** A deeper brain acquires more slowly at first; compare depths at matched lessons.
- **Observers.** With `apical` set, an observer's return lands on the same compartment: System 2
  adds at most `apical` times each region's bottom-up drive, enough to bias the settle and never
  enough to take it over.

A deep brain with every memory and attention feature is one call:

```python
full = Brain.compose(
    inputs=16, actions=4, seed=0, arousal=True,
    modules=(64, 32, 32), apical=0.25, teaching="deep",  # three regions that learn in depth
    learned_code=16, attention=True,                     # a learned code, read by attention
    recall_heads=2,                                      # two facts in one settle
    recall_boundary=True,                                # the start of an episode has a slot
    recall_action=True,                                  # the slots keep what it did
    recall_copy=2.0, recall_tokens=range(4),             # observations 0-3 name the 4 actions
    observers=(16,),                                     # optional System 2
)
assert full.live(np.eye(16)[[3]]).shape == (1,)
```

- `learned_code=16, attention=True`: the recall memory attends over recent moments by a learned
  code of the observations ([single-head attention](memory.md#single-head-learned-code-attention-opt-in)).
- `recall_heads=2`: two heads read with addresses of their own, so one settle can combine two
  remembered facts ([several heads](memory.md#several-heads-opt-in)).
- `recall_boundary=True`: a question about the start of an episode has a slot to read
  ([the episode's boundary](memory.md#episodes-keys-and-windows)).
- `recall_action=True`: every slot keeps the command executed at its moment
  ([its own actions](memory.md#its-own-actions-in-the-slots-opt-in)).
- `recall_copy=2.0, recall_tokens=range(4)`: a token the read delivers drives the action of the
  same token, including tokens never taught as answers
  ([the copy pathway](memory.md#the-copy-pathway-opt-in)). Declare only identities the task
  has; without shared tokens leave it out.
- `observers=(16,)`: a System 2 region inside the same settle, bounded by the compartment.

The recall memory reads into the top region;
[memory on deep brains](../benchmarks/depth/README.md#memory-on-deep-brains) measures what depth
keeps and costs.

### Wide brains

For modules of several hundred neurons and more
([width sweep](../benchmarks/sequence/README.md#the-width-sweep-widthpy)):

- **Surprise lessons.** With `learning_surprise`, set `learning_surprise_share=True` so a lesson's
  step scales with the share of rows it taught
  ([in proportion to the surprise](teach.md#in-proportion-to-the-surprise)); otherwise one wrong
  row takes a whole lesson's step and a wide brain's answers swing.
- **The trace's amplitude.** Beside a recall memory at `teaching="deep"` the founder's 3.0 falls with
  width, and `carried_gain=0.01` or `working_memory_amplitude=0.3` takes most of that back at whole steps
  ([the amplitude beside a read](memory.md#the-amplitude-beside-a-read)).
- **Projection rates and the read map.** A neuron's drive sums over its fan-in, so a projection that grows with the
  width moves that drive further per lesson, and the read's address map, stepped on the association's settled
  activity, moves further as well. A brain composed with the memory takes the answer from its association's width
  `w`, with no setting: above 64 the projection from prefrontal to association and the association-motor pair step at
  `64 / w` (`Learner.synapse_rate`), and `recall_address` is .01 times `64 / w`. At 64 and under every step is whole.
  Every other projection keeps its step: the sensory one, a chain between modules, an observer's and the memory's reads.
  The rate multiplies every step on those synapses, a lesson's and a reward's.

A rate you name wins over its part:

| You name | The read map | The two projections |
| --- | --- | --- |
| Nothing | .01 times `64 / w` | `64 / w` |
| `recall_address=x`, at `compose` or in `retune` | `x` | `64 / w` |
| `teaching="founder"`, `teaching="normalized"` or `learning=LearnerConfig(...)` at `compose` | .01 times `64 / w` | The named steps, whole |
| `brain.learner.synapse_rate = rates` at any time, `None` for whole steps | As it was | `rates` |
| `recall_founder=True` | The founder's .01 | Whole |
| `recall=False` | None | Whole |

```python
wide = Brain.compose(8, 4, modules=(256,), seed=0)
assert wide.recall.address == 0.0025
assert wide.describe()["layout"]["projection_rates"] == {
    "association->motor": 0.25, "motor->association": 0.25, "prefrontal->association": 0.25,
}

whole = Brain.compose(8, 4, modules=(256,), seed=0, recall_address=0.01)
whole.learner.synapse_rate = None
assert "projection_rates" not in whole.describe()["layout"]
```

`describe()` states the steps under `layout["projection_rates"]` and the map under `genes["recall_address"]`. A checkpoint
keeps both, and a brain saved under other rates keeps those. `retune` keeps the steps through a change of teaching point.
The width is fixed at birth, so `grow` keeps these rates: a new slot's synapses with the association take the pair's step,
every other new synapse steps at 1, and a memory grown on a brain without one takes the read map of this width while the
brain keeps its steps. `graft` leaves every synapse at the host's rates, the grafted ones too. `grow` and `graft` refuse
rates set by hand; set those on the new brain.

The grammar chamber, one online pass, twelve lives per row on six fresh founders, each on its own newborn and on one
further seed, the mean with the least life on paths. Whole steps are the same brain with the earlier rates named. Seconds are
one thread of a shared machine. The transformer is scored independently at the brain's parameter count, on the six founders:

| Width | Steps | Clauses | Two distractors | Held-out | Roles | Brackets | Paths | Least paths | Two hops | Three hops | Sweeps per act | Seconds | Multiply-accumulates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Whole | .98 | .95 | .95 | .97 | .90 | .94 | .72 | .89 | .34 | 78 | 366 | 2.2e12 |
| 128 | At its width | .99 | .99 | .93 | .93 | .89 | .97 | .84 | .94 | .39 | 58 | 306 | 1.7e12 |
| 128 | Transformer, one pass | 1.00 | 1.00 | 1.00 | 1.00 | .56 | .29 | | .07 | .00 | | 32 | 1.4e11 |
| 128 | Transformer, offline | 1.00 | 1.00 | .98 | 1.00 | 1.00 | .94 | | .91 | .00 | | 776 | 3.0e12 |
| 256 | Whole | .95 | .88 | .94 | .93 | .94 | .94 | .67 | .91 | .41 | 78 | 658 | 6.6e12 |
| 256 | At its width | .99 | .96 | .96 | .96 | .82 | .96 | .81 | .94 | .33 | 44 | 462 | 3.7e12 |
| 256 | Transformer, one pass | 1.00 | 1.00 | .99 | .99 | .39 | .36 | | .19 | .00 | | 50 | 3.4e11 |
| 256 | Transformer, offline | 1.00 | 1.00 | .97 | .97 | 1.00 | .87 | | .86 | .00 | | 876 | 7.6e12 |

At its width no life reads paths under .80, against one of twelve at whole steps at each width. Paired, with two standard
errors, no panel moves past them at 128. At 256 clauses rise .04 (±.04) and two distractors .09 (±.10), and brackets fall .12
(±.09): that is the rates' cost. The acts settle in 25 percent fewer sweeps at 128 and 44 percent fewer at 256, and a life
takes 24 and 43 percent fewer multiply-accumulates. This block holds paths at whole steps as well; over 42 development lives at
256 whole steps read paths .84, nine under .80, and the rates .94, three under .80. On that form at 512, six development lives
read clauses .88 and paths .66 at the rates and clauses .90 and paths .56 at whole steps.

These lives took the ordinary solve. With the fast solve, on a further block of six fresh founders, twelve lives per width, a new
brain reads clauses .99 and .99, brackets .89 and .93, paths .96 (least .91) and .97 (least .88), no life under .80, at 128 and 256,
in 111 and 147 seconds a life ([one composition](memory.md#one-composition-for-sequences)).

## The brain in a browser page

The library ships no browser page, environment or body. For an application display,
`record_settlements(callback, label=)` sends every settlement inside the context to the
callback, including those inside learning and imagination, with the potential, activation and
adaptation at every iteration:

```python
records = []
with cd.record_settlements(records.append, label="display"):
    brain.live(observation)
assert records and records[0].label == "display"
```

Recording copies every iteration and costs time, so use it for a display or an audit
([record every settling step](api.md#record-every-settling-step)). The public
[cadence-demos](https://github.com/muellerberndt/cadence-demos) repository holds application
pages that run or draw a brain.
