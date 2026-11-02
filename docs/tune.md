# Tune and read a living brain

You want to read or change a brain that is living. `describe()` says what it is, by its
layout, its genes, its memories and whether an action awaits its outcome; `retune(...)`
changes named settings within the same life; `last_settlement`, `last_arousal` and
`last_learning` report the latest moment's work. The tables below give every founder and the
moment each can change; the arousal law that decides when a `live` brain learns is in
[routine and repair](start.md#routine-and-repair-live).

`Brain.compose` builds **System 1**: one continuing neural graph with senses,
reciprocal processing regions, a motor choice, a working trace and an associative
memory, all settling together. Optional observer regions add **System 2** feedback
inside the same graph. Keep one composed brain through acquisition, use and
correction; [one continuing equilibrium brain](start.md#one-continuing-equilibrium-brain) explains that life.

```python
import cadence as cd
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.live(observation)
assert action.shape == (1,)
```

`live` takes one stream's observation of shape `(1, inputs)` and returns one action
index ([continuous interaction](start.md#continuous-interaction) covers `live`, `step` and saving).

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
| `recall_when`, `recall_1_when` ... | 2 each, 4 with `recall_anchor` | With `recall_when=True`, one per head; synapses into the association start at zero | [when a read item happened](memory.md#position-how-long-ago-a-moment-happened) |

The hippocampus is a `SynapticMemory` from the observation code to the motor drive,
with `memory_choice=True` the [episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before), and with
`episodic=False` absent. The default `modules=(64,)` is one region settling
recurrently with the others; in `modules=(16, 8)` the last entry is the association
region and sets the working trace's width. An action is issued only from a state,
observers included, that meets the full-equation tolerance
([qualification and refusal](start.md#qualification-and-refusal)).

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8, 4), seed=7, arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Observers settle with the base and never override a finished answer. The wiring
permits recursive correction without teaching it.

### Deep brains

Several processing regions learn features of features only with the
[apical compartment](anatomy.md#the-apical-compartment). Without it each region hears the
region above it through a loop that grows with the weights, and a second module carries an echo
of the answer. The recipe of the [depth benchmark](../benchmarks/depth/README.md):

```python
deep = Brain.compose(
    inputs=48, actions=7, modules=(128, 64, 64, 64), apical=0.25,
    teaching="normalized", learning_normalize_relative=0.1,
    learning_nudged_steps=32, learning_nudged_fixed=True, seed=0,
)
```

- **What it buys.** On a task that needs two layers of thresholds, one module reads 0.58 of
  held-out rows after 4,000 lessons, two 0.66, three 0.81 and four 0.86, against 0.89 for a
  network of two hidden layers trained by backpropagation.
- **Why the learning settings.** Credit reaching a region k levels below the motor region
  shrinks about fourfold per level. The projection-relative floor lets each projection step on
  its own scale, and fixed nudged phases carry the nudge down to the lower regions.
- **Cost.** An apical brain settles at half the founder's step, yet its bounded top-down loop
  settles in fewer sweeps than a flat brain whose loop has grown.
- **Lessons.** A deeper brain acquires more slowly at first; compare depths at matched lessons
  and give it a longer budget before reading it.
- **Memories.** The recall memory reads into the top region, and its benchmarks run on brains
  of two and three modules ([memory on deep brains](../benchmarks/depth/README.md#memory-on-deep-brains)).
  With `recall_boundary=True` a question about the start of an episode has a slot to read.
- **Observers.** An observer's return to the modules, the association and the motor region is a
  top-down half too, so with `apical` set it lands on the same compartment: System 2 adds at most
  `apical` times each region's bottom-up drive, enough to bias the settle and never enough to
  take it over.

A deep brain with every memory and attention feature is one call. Each line adds one feature to
the recipe above:

```python
full = Brain.compose(
    inputs=16, actions=4, seed=0, arousal=True,
    modules=(64, 32, 32), apical=0.25,                  # three regions that learn in depth
    teaching="normalized", learning_normalize_relative=0.1,
    learning_nudged_steps=32, learning_nudged_fixed=True,
    learned_code=16, attention=True,                     # a learned code, read by attention
    recall_heads=2,                                      # two facts in one settle
    recall_boundary=True,                                # the start of an episode has a slot
    recall_action=True,                                  # the slots keep what it did
    recall_copy=2.0, recall_tokens=range(4),             # observations 0-3 name the 4 actions
    observers=(16,),                                     # optional System 2
)
assert full.live(np.eye(16)[[3]]).shape == (1,)
```

- `learned_code=16, attention=True`: the brain learns a code of its observations and its recall
  memory attends over its own recent moments by that code
  ([single-head attention](memory.md#single-head-learned-code-attention-opt-in)).
- `recall_heads=2`: two heads read the memory with addresses of their own, so one settle can
  combine two remembered facts ([several heads](memory.md#several-heads-opt-in)). The position
  term, on by default, lets a cue choose how far back a head looks.
- `recall_boundary=True`: a question about the start of an episode has a slot to read
  ([the episode's boundary](memory.md#episodes-keys-and-windows)).
- `recall_action=True`: every slot keeps the command executed at its moment, so a read can find
  what the brain itself did ([its own actions](memory.md#its-own-actions-in-the-slots-opt-in);
  measured on a brain of one module).
- `recall_copy=2.0, recall_tokens=range(4)`: where observation coordinates and actions name the
  same tokens, a token the read delivers drives its own action, tokens never taught as answers
  included ([the copy pathway](memory.md#the-copy-pathway-opt-in)). Declare only identities the
  task has: here observations 0 to 3 are the four actions' tokens. Without shared tokens leave it
  out.
- `observers=(16,)`: a System 2 region inside the same settle, bounded by the compartment.

Every option is checked by the tests beside every other (`tests/test_observers_compat.py`), and
the [memory benchmarks on deep brains](../benchmarks/depth/README.md#memory-on-deep-brains)
measure what depth keeps and costs.

### Wide brains

On 0.89 more neurons in a module did not lower held-out error
([#239](https://github.com/muellerberndt/cadence/issues/239)). A width sweep (64 to 1,024 neurons;
induction over a vocabulary of 128, associative recall and a lookup task; the recall memory under
the normalized point) located three causes at the widest modules:

- **Refused answers.** A trained brain's settle could run out of budget: the finite learner spends
  half its sweeps at the full step and the rest at half the step, and a state that needs more
  full-step sweeps was refused, which stops a continuing life. A solve that does not qualify now
  continues once with halved steps before the answer is refused
  ([qualification and refusal](start.md#qualification-and-refusal)).
- **Surprise lessons out of proportion.** A surprise lesson gives one wrong row a whole lesson's
  step, and at 1,024 neurons the answers swing between collapse and recovery.
  `learning_surprise_share=True` scales the step by the share of rows taught
  ([in proportion to the surprise](teach.md#lessons-only-where-the-answer-differs)).
- **A saturated readout.** A neuron's drive sums over its fan-in, so with every synapse stepped as
  in a module of 64 the motor neurons of a wide module saturate: associative recall collapses at
  1,024 neurons on the composed steps. Scaling each synapse's step to its neuron's fan-in in the
  same brain composed at 64 (`Learner.synapse_rate`) keeps associative recall at 1.00 at every
  width; it costs the lookup task its gain from width (0.82 to 0.87 at every width, against 0.77
  at 64 and 0.92 at 1,024 on the composed steps), so it is a recipe, not a default:

```python
def fan_in_rates(wide: Brain, narrow: Brain) -> np.ndarray:
    """Each synapse's step scaled to the largest fan-in of its neuron's population in
    ``narrow``, the same composition at a smaller width; capped at 1."""
    c, n = wide.connectome, narrow.connectome
    fan = np.bincount(c.post, minlength=c.n).astype(float)
    fan_narrow = np.bincount(n.post, minlength=n.n).astype(float)
    reference = fan.copy()
    for name, neurons in c.populations.items():
        index = np.asarray(neurons, dtype=np.int64)
        if name in n.populations and len(index):
            top = fan_narrow[np.asarray(n.populations[name], dtype=np.int64)].max(initial=0.0)
            if top > 0:
                reference[index] = np.minimum(reference[index], top)
    return np.minimum(1.0, reference[c.post] / np.maximum(fan[c.post], 1.0))


wide = Brain.compose(8, 4, modules=(256,), seed=0, recall=True)
narrow = Brain.compose(8, 4, modules=(64,), seed=0, recall=True)
wide.learner.synapse_rate = fan_in_rates(wide, narrow)
assert wide.learner.synapse_rate.max() <= 1.0
```

A brain with per-synapse rates cannot be grown (`grow` refuses it); set the rates on the grown
brain instead. Whether a wider module then lowers held-out error on the composer and on Connect
Four, the cases #239 opened with, is not yet measured.

## Defaults and expert overrides

Set only the values you intend to change; the rest keep their composed founders, which
are the control for any change you measure.

```python
tuned = Brain.compose(
    inputs=4, actions=2, arousal=True,
    scales={"sensory": 2.0}, actor_eta=0.1, actor_eta_bias=0.01, arousal_youth=30,
)
settings = tuned.describe()
assert settings["genes"]["actor_eta"] == 0.1
assert settings["layout"]["composition"]["sensory_scale"] == 2.0
```

`describe()` returns four JSON-safe sections: `layout` (the regions; `code`, what the
memories key on; `composition`, the construction choices, not current weights), `genes`
(every setting by the name `compose` and `retune` take, `teaching` among them), `memories`
(each member of `brain.memories` with its settings and counts) and `pending` (whether an
issued action owns the next outcome).

For an expert `LearnerConfig` outside both teaching points, `genes["teaching"]` is
`"custom"` and `genes["learning"]` gives its effective `eta`, `eta_bias`, `normalize` and
`normalize_floor`. Named teaching points need only their name. These four expert values
are a diagnostic mapping; change them with `learning=LearnerConfig(...)` at construction
or `retune`, as before.

| Setting | Composed founder | Changed at | Home |
| --- | --- | --- | --- |
| `modules`, `observers` | `(64,)`, `()` | construction | [map](#brain) |
| `apical` | `0.0`: every synapse summed alike | construction | [the apical compartment](anatomy.md#the-apical-compartment) |
| `lateral`, `slots` | `0.0`, `1` | construction | [motor competition](#operating-point-and-motor-competition) |
| `scales`: `"sensory"`, `"trace"`, `"history"`, `"recall"` | `1.0`, `1.2`, `1.2`, `1.2` | construction | [image inputs](senses.md#image-inputs), [memory](memory.md#working-trace) |
| `features`, `field`, `seed` | `8`, `3`, `0` | construction | [image inputs](senses.md#image-inputs) |
| `attention` | `False`; `True` requires `learned_code=units` and supplies the contextual Recall recipe | construction | [single-head learned-code attention](memory.md#single-head-learned-code-attention-opt-in) |
| `teaching` | `"founder"`: steps 0.5 and 0.05 on the raw contrast; `"normalized"`: 0.003 and 0.0003 under a running RMS (0.99, floor 1e-4) | retune | [learning](teach.md#many-lessons-the-normalized-point-for-imitation) |
| `learning_beta`, `temperature` | `0.1`, `0.2` | retune | [learning](settling.md#7-every-knob) |
| `learning_free_steps`, `learning_nudged_steps`, `learning_tolerance`, `learning_qualified`, `learning_damping` | `1024`, `12`, `0.003`, `False`, `3` | retune | [learning](settling.md#qualified-teaching-budgets) |
| `learning_initial_damping` | `0` | retune, qualified CPU float64 teaching only | [initial numerical step](settling.md#qualified-teaching-budgets) |
| `learning_scale_cap`, `learning_surprise`, `learning_surprise_share` | `8.0`, `False`, `False` | retune | [learning](settling.md#7-every-knob) |
| `learning_momentum` | `0.9` | retune | [learning](teach.md#rates-under-normalization) |
| `learning_normalize_relative` | `0.0`: the teaching point's `normalize_floor` | retune | [learning](teach.md#rates-under-normalization) |
| `actor_gamma`, `actor_lam`, `actor_eligibility_steps`, `actor_eta`, `actor_eta_bias`, `actor_eta_critic` | `0.9`, `0.8`, `12`, `1.0`, `0.05`, `0.3` | retune | [reward](reward.md#eligibility-traces-and-dopamine) |
| `actor_dopamine_cap` | `1.0` | retune | [reward](reward.md#eligibility-traces-and-dopamine) |
| `arousal` | off; `True` is `ArousalConfig()` | construction (switch), retune (fields) | [continuous](start.md#routine-and-repair-live) |
| `arousal_threshold`, `arousal_decay`, `arousal_tolerance`, `arousal_floor`, `arousal_fast`, `arousal_slow` | `0.2`, `0.9`, `2.0`, `0.1`, `0.05`, `0.005` | retune | [continuous](start.md#routine-and-repair-live) |
| `arousal_heat`, `arousal_youth`, `arousal_value_surprise`, `arousal_need`, `arousal_need_scale` | `2.0`, `100`, `1.0`, `0.0`, `0.1` | retune | [continuous](start.md#routine-and-repair-live) |
| `arousal_hindsight` | `0` | retune | [continuous](start.md#calm-moments-in-hindsight) |
| `working_memory_*` | amplitude `3.0`, decay `0.0` | retune | [memory](memory.md#working-trace) |
| `efference_*` | amplitude `0.0` (absent), decay `0.2` | a positive amplitude at construction adds it | [memory](memory.md#efference-copy) |
| `history_*` | decays `()` (absent), amplitude `3.0` | decays at construction | [memory](memory.md#heard-history) |
| `episodic`, `memory_rate`, `memory_decay`, `memory_amplitude`, `consolidation` | `True`, `1.0` (`0.5` with `memory_choice`), `0.9`, `1.0`, `0.05` | `episodic` at construction | [memory](memory.md#associative-store-the-hippocampus) |
| `code` | unset: the observation itself | construction | [memory](senses.md#the-observation-code) |
| `memory_choice` and the other `memory_*` episodic settings | `False` | construction | [episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before) |
| `recall` and every `recall_*` | `False` | see the page | [recall memory](memory.md#the-recall-memory-in-full) |
| `backend`, `device` | `"cpu"`, `None` | construction | [backends](backends.md) |

Override rules:

- Every field of `LearnerConfig`, `ActorCriticConfig` and `ArousalConfig` is a
  `learning_<field>`, `actor_<field>` or `arousal_<field>` name, but the teaching steps
  and their normalization (`eta`, `eta_bias`, `normalize`, `normalize_floor`), which
  `teaching` sets: their names raise `ValueError`. `temperature` aliases
  `learning_temperature` and sets both the teaching softmax and the sampled policy.
- Unknown names and the ambiguous bare `eta` or `eta_bias` raise `TypeError`.
- `arousal=True` uses the founders, a mapping such as `arousal={"heat": 1.0}` patches
  them, an `ArousalConfig` replaces them, and `arousal_*` names need one of the three.
  Without `arousal`, `live` raises `ValueError`.
- A changed actor rate keeps the existing bias rate; `actor_eta_bias=None` derives one
  tenth of the new rate.
- A setting for an absent component, such as `memory_rate` with `episodic=False`, has
  no effect; `describe()` omits it and `retune` refuses it.
- `learning=LearnerConfig(...)` and `reward=ActorCriticConfig(...)` replace the whole
  configuration, and a fresh config replaces every founder: `LearnerConfig()` has
  `eta=0.2`, `momentum=0`, `normalize=0` (raw steps), budgets `100/50` and tolerance
  `1e-4`; `ActorCriticConfig()` has `gamma=0.99`, `lam=0.9` and rates `0.5/0.05/0.05`.
  Use named overrides for a one-field edit.

## Retune the same life

`retune` changes named settings on a living brain, for example from practice to
ordinary interaction:

```python
tuned.retune(
    arousal={"need": 0.0, "heat": 0.0},
    temperature=0.2,
    actor_eta=0.03, actor_eta_bias=0.003,
)
```

It takes the same names, validates the whole request before installing anything,
runs no moment and returns the same brain. Parameters, optimizer history, activity,
traces, records, random state and pending feedback are kept, and new rates apply from
the next learned outcome. It raises `ValueError` for a construction setting (those
marked construction above), for a missing component (arousal included), and for a
change of `learning_beta` while a sampled action awaits feedback, whose contrast
belongs to the old scale; learn that outcome first. With `recall_address_reward`
above zero the same holds for `recall_address`, `recall_address_step` and
`recall_address_reward` ([recall memory](memory.md#the-recall-memory-in-full)). The recall settings of 0.84.3 that
left in this version are refused by name, at `compose` and `retune` alike
([removed in this version](memory.md#removed-in-this-version)). A part that a
construction setting adds (the recall memory or more of its heads, a heard history, an
efference copy, new output slots) is added to a trained brain by `grow`, which returns a new
brain; `graft` puts trained regions of another brain into it ([grow and combine
brains](grow.md)).

`retune` resets nothing. `brain.arousal.reset()` resets the arousal level and reward
references and keeps its age and work counters; an adult brain at the founder `need=0` is
then roused only by surprise at its own forecasts, and a positive `need_scale` wants nothing
until outcomes enter the new stream's reward scale ([reset and save](start.md#reset-and-save)).

## Choose settings for the application

| Application | Interface |
| --- | --- |
| Vector senses, several motors | One motor slot per motor, `live` and `retune`, each real action paired with its outcome across stage changes, as in the [Robot Arena demo](https://github.com/muellerberndt/cadence-demos/tree/main/robot-arena). |
| Pixels and a teacher | Image `inputs`, teaching through `step`; teaching rates are `learning_*`, reward rates `actor_*`. See the [Atari demo](https://github.com/muellerberndt/cadence-demos/tree/main/atari-arcade). |

Gain, bias and solver steps are distinct levers: gain scales synaptic drive in the
whole graph, a population bias moves selected neurons' operating points, and steps and
damping change only how the equations are solved. Check candidate biases with the
[calibration guide](settling.md#calibrating-the-operating-point), choose on development
inputs and compare against the founders on your own task ([evaluation](evaluate.md)).

## Operating point and motor competition

`lateral` is the signed weight between each pair of motor neurons in a slot; `0.0`
removes those synapses and keeps the association and motor feedback. Left unset it is
`0.0` at every menu size, as in the `cortex`, `motor_cortex` and `layered` factories. A
negative weight makes the actions of a slot compete, and at -0.5 the undamped free solve
oscillates from 4 actions on: on 8-action composed brains a fifth of one-row greedy acts and
every act of 16 rows settled only under the damping fallback (544 sweeps against 32 at
`0.0`), and a lesson of 16 rows, whose free phase has no fallback, ran out its 1,024-sweep
budget on 28 of 30 lessons at 8 actions and 7 of 30 at 4. Choose a competing readout
against `0.0` on development tasks.

`slots` groups the motor neurons into readouts that settle together, one softmax each:
a count of equal groups or one size per group. An action with parts (kind, pitch,
octave) is one slot per part. `act` and `step` return one index per slot, a nonzero
`lateral` is wired within a slot, and `act(temperature=...)` takes one value or one per
slot. `save` and `load` carry the grouping.

```python
chord = Brain.compose(inputs=4, actions=5, slots=(3, 2), seed=3)
chosen = chord.act(observation, temperature=[0.5, 0.1])
assert chosen.shape == (1, 2)
```

Every bias starts at zero and is plastic: `brain.brain.bias` holds the learned values,
and loading restores them. An initial bias of your own on the processing neurons is
`brain.learner.brain = brain.learner.brain.with_parameters(bias=bias)` before the first
lesson (the acquisition chamber's `resting` sets 0.5 on the modules, the observers and the
association region). A signed neuron below rest responds to input and contributes to
teaching, so the share of nonpositive emissions is an activity diagnostic
(`tests/test_operating_range_function.py`).

## One experience step by hand

`step` learns from the previous outcome and chooses the next action. Use `act` and
`learn` when the body manages that timing:

```python
body = Brain.compose(4, 2, modules=(16, 8), seed=7)
chosen = body.act(observation)
reward = (chosen == 0).astype(float)  # action 0 earns one unit
following = np.array([[0.0, 1.0, 0.0, 0.0]])
body.learn(reward, np.array([False]), following)
next_action = body.act(following)
```

Reward and termination describe the preceding executed action; a demonstration labels
the current observation. Keep batch-row identities fixed until `reset()`. The
[continuing brain example](../examples/continuing_brain.py) runs the full reward,
teacher and checkpoint loop.

## The brain in a browser page

The library ships no browser page, environment or body. For an application display,
`record_settlements(callback, label=)` sends every actual settlement inside the
context to the callback, including those inside learning and imagination, with the
potential, activation and adaptation at every iteration:

```python
records = []
with cd.record_settlements(records.append, label="display"):
    brain.live(observation)
assert records and records[0].label == "display"
```

Recording copies every iteration and costs time, so use it for a display or an audit
([record every settling step](api.md#record-every-settling-step)). The public
[cadence-demos](https://github.com/muellerberndt/cadence-demos) repository holds
application pages that run or draw a brain.
