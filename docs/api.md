# API reference

One entry per public name: signature, defaults, behaviour in a few lines, refusals, and
the guide that explains it. `Brain.compose` is the entry point; pass optional arguments by
keyword.

| Section | Names |
| --- | --- |
| [Brain](#brain-cadence) | `Brain`, its methods, attributes and checkpoint formats |
| [Arousal](#arousal-cadencearousal) | `ArousalConfig`, `Arousal` |
| [Streams](#streams-cadencestream) | `Trace`, `Efference`, `History`, `Recall`, `FastSynapses`, `SynapticMemory`, `PatternSeparator` |
| [Learning](#learning-cadencelearning), [checkpoints](#checkpoints-cadencecheckpoint), [reward](#the-agent-cadenceplasticity) | `LearnerConfig`, `Learner`, `GraphHolder`, `LearningPhaseError`, graph factories and diagnostics, `save`, `load`, `ActorCritic`, `ActorCriticConfig` |
| [The graph](#neuralgraph-cadence) | `NeuralGraph`, `BrainState`, `Equilibrium`, `Nudge`, `Connectome`, `NeuronModel`, `Adaptation`, blocks, `Certificate`, the neuron-by-neuron reference |
| [Wiring and instruments](#regions-and-genome-cadenceregions-cadencegenome) | `Region`, `regions`, `Projection`, `Genome`, `develop`, `evolve`, `genes`, `Receipt` |
| [Recording](#record-every-settling-step) | `record_settlements` |

`cadence.__version__` is the installed release string.

## Brain (`cadence`)

### Construction

```text
Brain.compose(inputs, actions, *, connectome=None, modules=(64,), observers=(),
              lateral=None, teaching=None, scales=None, features=8, field=3, code=None,
              learned_code=None, attention=False, seed=0, slots=1, **options) -> Brain
```

The composed System 1 brain, with optional System 2 `observers` in the same graph.

- `inputs`: a vector length or an image shape `(height, width[, channels])`; `features` and
  `field` size an image's visual cortex.
- `modules`: the widths of the reciprocal processing chain, the last the association region.
  Each `observers` width adds a region connected both ways to every module, the motor region
  and earlier observers.
- `lateral`: the weight between motor neurons of one slot (unset: 0.0). `slots`: a count of
  equal motor groups or one size per group, each read as its own softmax.
- `code=units`: keys the memories on a fixed sparse code of that many units and, for a vector
  brain, makes the sensory region that code
  ([the observation code](senses.md#the-observation-code)).
- `learned_code=units` (a vector brain, without `code`): adds a `code` region of that many
  neurons that settles with the brain to a learned sparse code of the observation, read by the
  cortex and keyed on by the memories. Its one setting is `learned_code_fields`
  ([the learned code](senses.md#the-learned-code)).
- `scales`: maps `"sensory"` and `"trace"` to the initial magnitudes of the projections from
  those regions (founders 1.0 and 1.2; a name left out keeps its founder), fixed at
  construction. The history's and the recall memory's projections start at 1.2, a constant.
- `teaching`: the teaching point of the lesson's learner, `"founder"` (unset), `"normalized"`
  or `"deep"` (the normalized point with the projection-relative floor 0.1 and nudged phases
  of a whole 32 sweeps, for a brain with an apical compartment). It sets `eta`, `eta_bias`,
  `normalize`, `normalize_floor`, `normalize_relative` and `nudged_fixed`, whose `learning_`
  names it replaces ([many lessons](teach.md#many-lessons-the-normalized-point-for-imitation),
  [deep brains](tune.md#deep-brains)). `retune(teaching=...)` moves a living brain between the
  points; `learning=LearnerConfig(...)` replaces the whole configuration instead.
- `options`: every constructor keyword below and every `learning_<field>`, `actor_<field>`
  and `arousal_<field>` name; with `memory_choice=True` it fills in the episodic founders.

The region map and the founder table are in
[compose a brain](tune.md#defaults-and-expert-overrides); the memory settings are owned by
[memory](memory.md),
[episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before) and
[recall memory](memory.md#the-recall-memory-in-full).

`attention=True` is the opt-in contextual-memory recipe: it requires a vector
`learned_code=units`, enables the recall memory and supplies its learned-code settings where
the caller did not, through k heads with `recall_heads=k`
([single-head learned-code attention](memory.md#single-head-learned-code-attention-opt-in),
[several heads](memory.md#several-heads-opt-in)). It changes no default and does not enable
episodic action voting.

Refusals: unknown names, the bare `eta` or `eta_bias`, and duplicate aliases raise
`TypeError`; a `recall_` name without `recall=True`, a setting outside its range, and
`arousal_*` names without `arousal` raise `ValueError`. A retired setting is refused by name
with a message naming its replacement ([CHANGELOG](../CHANGELOG.md)). `attention` must be boolean;
`attention=True` without `learned_code`, with `recall=False`, with `recall_window < 2` or
with `recall_next=False` raises `ValueError`.

`connectome` wraps a custom anatomy with the same founders, the episodic founders filled in
under `memory_choice=True`: its `sensory` (or `visual/input`) population holds the inputs (the
code's units with `code=units`) and its `motor` population the actions. It brings its own
layout, so `modules`, `observers`, `lateral`, `scales`, `features` and `field` stay at their
founders (`ValueError` otherwise). `Genome.composed` returns the layout
`compose` develops, to edit or evolve ([genome, development,
brain](anatomy.md#genome-development-brain)).

```text
Brain(connectome, *, episodic=True, consolidation=0.05,
      memory_rate=1.0, memory_amplitude=1.0, memory_choice=False,
      memory_context=0.0, memory_expansion=0, memory_fan_in=None, memory_extinction=0.0,
      memory_context_hold=None, maturity=False, maturity_scale=10.0,
      maturity_recovery=200000.0,
      working_memory_decay=0.0, working_memory_amplitude=3.0,
      efference_decay=0.2, efference_amplitude=0.0,
      history_decays=(), history_amplitude=3.0, recall=False, learning=None,
      reward=None, slots=1, arousal=None, seed=0,
      backend="cpu", device=None, inputs=None, code=None, learned_code=None, apical=0.0,
      **genes)
```

Wraps any connectome with populations `association`, `motor` and `sensory` (or
`visual/input`). It uses `prefrontal` for a working trace, `efference` (one neuron per motor
neuron) for an efference copy, `history` (one sensory-width block per entry of
`history_decays`; one block of `learned_code` units with a learned code) for a heard
history, with `recall=True` the `recall` and `next` regions, and with `learned_code=units` a
`code` population of that many neurons with its complete dictionary, lateral and predictor
synapses ([the learned code](senses.md#the-learned-code)).
`inputs` names the observation width when the sensory region is a `code` of that width.
`genes` carries the `recall_*` settings and `learned_code_fields`. `learning` and `reward` default to the composed
`LearnerConfig` and `ActorCriticConfig`; `arousal` is `True`, a field mapping or an
`ArousalConfig`. With `memory_choice=True`, `compose` fills in `memory_context=0.5`,
`memory_rate=0.5` and `memory_expansion=1024` (0 with a fixed `code`); the constructor keeps
its own defaults above, so pass these here or use `compose` ([episodic
settings](memory.md#episodic-settings)). The stores' constants (a record's decay 0.9, a replay
of 8 decisions, demonstrations recorded at 1.0, 24 active separation cells) are no arguments.

| Rule | Raised |
| --- | --- |
| `memory_choice` needs `episodic=True` | `ValueError` |
| `memory_context`, `memory_expansion`, `memory_extinction` need `memory_choice`; `memory_context_hold` (`"unchanged"` or `"punished"`) needs `memory_choice` and `memory_context` | `ValueError` |
| `memory_fan_in` needs `memory_expansion` (so `memory_choice` too) and at most the key width; a brain with a fixed `code` refuses both | `ValueError` |
| `learned_code` at least 2; `learned_code_fields` without it, a removed `learned_code_*` name (`LearnedCode.REMOVED`), `code` beside it, an image brain, a backend other than `cpu`, or a connectome without the region's populations and complete synapses | `ValueError` |
| `memory_context` in `[0, 1)`; `memory_extinction` in `[0, 1]`; `working_memory_decay`, `efference_decay` in `[0, 1)` | `ValueError` |
| `maturity` boolean; `maturity_scale` and `maturity_recovery` other than their founders (10.0 and 200,000) need `maturity` | `ValueError` |
| `maturity_scale` finite and at least `1e-12`; `maturity_recovery` finite and at least 1 | `ValueError` |
| a missing population that a switched-on component needs | `ValueError` |

The efference projection into `association` has the fixed initial scale 12.0; the working
trace's (`scales["trace"]`) founder is 1.2, and the history's and the recall memory's
projections start at 1.2, a constant (`scales["history"]` and `scales["recall"]` are taken at
1.2 alone).

### Methods

| Method | Behaviour | Guide |
| --- | --- | --- |
| `live(observations, *, reward=None, done=None, salience=None, decision_id=None) -> actions` | One moment of one stream; needs `arousal`. Calm: one greedy qualified settle, no learning. Aroused: samples, learns from the outcome, writes memory. Omitted `reward` with a pending action is a zero outcome. Optional `salience` is one finite nonnegative importance for that outcome; it preserves reward and the arousal gate. A wrong `decision_id` or invalid salience raises before any change; a mark without an awaited action raises `RuntimeError`. | [routine and repair](start.md#routine-and-repair-live) |
| `wait(observations) -> None` | Senses a moment while a `live` action awaits its outcome: settles, advances the traces, writes the recall slot; issues nothing and learns nothing. Raises `ValueError` without arousal or with several streams, `RuntimeError` without an awaited action or when the settle cannot qualify. | [outcomes that arrive later](start.md#outcomes-that-arrive-later-wait) |
| `step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None) -> actions` | Learns from the preceding outcome, teaches `teacher` labels of the present observation, samples the next action. Omitted `reward` is a real zero. | [explicit learning](teach.md#explicit-learning-with-step) |
| `lesson(observations, labels) -> dict` | The teaching part of `step` without the act; records the demonstration in an episodic memory and runs the recall address lesson. With a recall memory, teach a moment before acting on it, as `step` does: a lesson on the observations the brain just acted on reads the recall memory as that act left it (the moment itself, its successor empty) and warns with `RuntimeWarning`. Refused with `RuntimeError` while a sampled action awaits its outcome; a refused phase raises `LearningPhaseError` and changes nothing. | [explicit learning](teach.md#explicit-learning-with-step) |
| `act(observations, *, greedy=False, temperature=None) -> actions` | One qualified free settle and a sampled (or greedy) choice, one index per slot; advances the working trace. `temperature` is one value or one per slot. Exhaustion raises `RuntimeError` before any change. | [one experience step by hand](tune.md#one-experience-step-by-hand) |
| `learn(reward, done, next_observations, *, bootstrap=None, salience=None) -> dict` | Feedback for the pending sampled action: hippocampal record, actor and critic update; `done` rows reset their traces. A refused feedback solve rolls everything back and keeps the action pending. | [reward](reward.md#decision-and-reset-contract) |
| `imagine(observations, *, budget=1024) -> tuple[Equilibrium, ...]` | A private branch over a sequence of batches: memories read, nothing written, live state unchanged; each phase settles to the learner's tolerance within `budget`. A refused phase ends the branch and is the last entry. | [private imagination](start.md#private-imagination) |
| `fit(observations, labels, *, epochs=30, batch=32) -> list[float]`, `predict(observations)`, `accuracy(observations, labels)` | Independent samples, without traces or recall; `fit` resets the streams first, keeping the arousal, and returns training accuracy per epoch. Noninteger labels and mismatched batches raise before any update. | [independent-sample controls](evaluate.md#independent-sample-controls) |
| `retune(**genes) -> Brain` | Changes mutable settings in the same life, atomically, and resets nothing (`brain.arousal.reset()` resets the arousal). Refuses construction settings, absent components, and the recall address genes under `recall_address_reward` while a sampled action awaits feedback. | [retune the same life](tune.md#retune-the-same-life) |
| `grow(**parts) -> Brain` | A new brain: this trained brain with new parts (`recall=True` with any `recall_*` setting but `recall_tokens`, `recall_heads`, `history_decays`, `efference_amplitude` above zero, `slots` extending the brain's own), everything learned, recorded and set carried, the streams afresh; the new synapses start at zero, so it settles and chooses as this brain does until learning gives them weight. Refuses a brain that records no construction (a custom anatomy), a part inside the loop (`modules`, `observers`, a code), arousal and the episodic memory, settings that `retune` changes, recall tokens, and slots where a memory is laid out per action or would join an apical pool. This brain is unchanged. | [grow a brain](grow.md#grow-a-trained-brain) |
| `graft(donor, regions=None, *, into=None, joints="zero", freeze=False) -> Brain` | A new brain: this brain with trained regions of `donor`; `regions` names or maps donor populations onto this brain's, `into` grafts the whole donor into one population. The grafted synapses keep the donor's weights, the joints start at zero (`joints="kept"` keeps this brain's), and `freeze=True` stops the graft learning. Refuses populations of other sizes or either brain lacks, regions that share no synapse, donor wiring this brain lacks among the grafted neurons or a one-way donor synapse on half of a reciprocal pair, another neuron model, apical compartment or apical role, a learned code's region or a joint landing on it, tied or rated synapses. Both brains are unchanged. | [graft trained regions](grow.md#graft-trained-regions-into-a-larger-brain) |
| `reset(*, keep=("arousal",))` | Clears per-stream state and eligibility; keeps records and weights. `keep` names what carries over: `"arousal"` (the default) the whole arousal state, `"slots"` the recall slots, ending each stream's episode; a reset that does not keep `"arousal"` begins it calm with its references cleared and its age kept. Other names raise `ValueError`, checked before anything changes; an unknown keyword raises `TypeError`. | [reset and save](start.md#reset-and-save) |
| `describe() -> dict` | JSON-safe `layout` (regions, `code`, `composition`), `genes` (every setting by its `compose` name; at `teaching="custom"`, `learning` gives the six effective expert teaching settings), `memories` (each member's `to_dict()`) and `pending`. | [defaults](tune.md#defaults-and-expert-overrides) |
| `observation_code(observations) -> (batch, code_width)` | The unit-length key every memory reads and writes; with a learned code, the code the observations settle to without a context. | [the observation code](senses.md#the-observation-code), [the learned code](senses.md#the-learned-code) |
| `stimulus(observations, *, memory=True) -> drive` | The drive of a batch; with `memory` the trace, copy, history, recall and hippocampal reads are added, and the recall address is fixed for the moment. | [memory](memory.md) |
| `parameters() -> int` | Actor and critic parameters, the consolidated matrices of the place and context records and of the direct payments, the recall memory's address and position maps and its options' maps (anchor, relative, slope), and a learned code's tables. | |
| `copy.deepcopy(brain)` | An independent copy: every array, store, state and random generator copied, the immutable connectomes shared. |
| `save(path, *, compressed=True) -> Path`, `Brain.load(path, *, backend="cpu", device=None, precision=None)` | The whole continuation, written atomically ([the format](#checkpoint-format)); `compressed=False` writes a plain archive, faster for a large brain. `Brain.load` refuses a learner-only file; `Learner.load` reads either. | [reset and save](start.md#reset-and-save) |

Observations are a nonempty finite batch `(streams, inputs)`, images flattened per row or in
their shape. Keep row identities fixed until `reset()`.

### Attributes

| Attribute | Holds |
| --- | --- |
| `connectome`, `brain`, `learner` | the developed graph, the current `NeuralGraph` (`learner.brain`) and the `Learner` |
| `basal_ganglia` | the `ActorCritic` reading the association region |
| `memories` | the present memories by name, in the order the brain reads them ([`Memories`](#the-memories-cadencememories)) |
| `working_memory`, `efference`, `history`, `recall` | the `Trace`, `Efference`, `History` and `Recall`, each `None` when absent |
| `hippocampus` | the associative `SynapticMemory` (with `memory_choice` the episodic place records), or `None` with `episodic=False` |
| `context_memory` | the episodic context records with `memory_choice` and a positive `memory_context`, else `None` |
| `sparse_code`, `code_width`, `coded_cortex`, `inputs` | the `PatternSeparator` of `code=units` or `None`; the key width; whether the sensory region is the code; the observation width |
| `learned_code`, `code_index`, `prefrontal_index` | the `LearnedCode` of `learned_code=units` or `None`; neuron indices of the `code` and `prefrontal` populations (empty without them) |
| `attention` | whether `compose` applied the opt-in learned-code contextual-memory recipe; its effective mechanism is recorded by the learned-code and Recall settings |
| `sensory_index`, `association_index`, `motor_index` | neuron indices of those populations |
| `memory_choice`, `memory_context`, `memory_extinction`, `memory_context_hold` | the resolved construction settings |
| `arousal` | the stream's `Arousal`, or `None` |
| `pending_feedback` | read-only: the issued action awaits its outcome (`wait` keeps it) |
| `decision_id` | read-only: the `arousal.age` at which `live` issued the awaited action, or `None`; saved and restored |
| `last_arousal` | immutable snapshot of the latest `live` moment; `None` after construction, reset or load ([fields](start.md#routine-and-repair-live)) |
| `last_settlement` | immutable report of the latest free-answer solve, refused ones included: `operation`, `qualified`, `row_qualified`, `residual`, `max_residual`, `steps`, `budget`, `tolerance`, `residual_checks`, `damping_halvings`, `stagnation_checks`, `rescue_steps` when a solve that did not qualify within `free_steps` continued once on the damping ladder before it qualified or was refused (`steps` and the checks then sum both solves), and with a learned code `code_steps` and `code_residual` and, for a brain with a place store, `context_free_steps` ([inspect the work](start.md#inspect-the-work-of-answering)) |
| `last_learning` | the latest transition's learning report, teaching under `demonstration_*` keys and, with `learning_rehearsal`, the replayed lesson's report under `rehearsal_*` keys (`rehearsal_rows`; `rehearsal_refused` when its phase refused) |
| `rng` | the brain's `numpy.random.Generator` |

### Checkpoint format

`Brain.save` writes `cadence-brain/1`, one format whatever the brain holds, except that a brain
with maturity ([reward](reward.md#maturity-opt-in)) writes `cadence-brain/2`, which 0.93.1 and
earlier refuse: the same members with the maturity (`scale`, `recovery`, `use`) under `maturity`
in the metadata and the store's noise scales (`memories/hippocampus/gate_noise`). A file of
either format carrying the other's maturity is refused.

| Members | What they hold |
| --- | --- |
| `meta` and the unprefixed arrays | the learner's checkpoint ([checkpoints](#checkpoints-cadencecheckpoint)) |
| `rehearsal/` | with `learning_rehearsal`: the reservoir's rows with their episodes and the streams' episodes in progress (`heard`, `told`, `lengths`, `moments`, `labels`), the counts `seen` and `draws` under `rehearsal` in the metadata; written whenever the brain holds either |
| `brain` | the metadata: the actor's config, both random states, the free phase and a pending action, the arousal and the action `live` issued with its forecast, the state a waiting stream sensed, a `code=units` code or a learned code, the composition, and under `memories` the settings of each memory the brain holds |
| `actor/`, `free/`, `pending/`, `moment/`, `prepared`, `lived/`, `awaiting/` | the critic and eligibility, the free phase, a pending action's phases and decision, the action `live` issued, the sensed state |
| `code/`, `learned_code/` | a `code=units` code's tables; a learned code's `dictionary`, `weights` and `thresholds` |
| `memories/working_memory/`, `memories/efference/` | each trace's `trace`, `last` and `cold` |
| `memories/history/` | the heard history's `trace` |
| `memories/recall/` | the slots, `held`, `previous`, the address map, and where set the actor's `trace` and `pending` contrast of the map, `ended` and the window's `past`; the per-stream `told` and `focus` and the anchored, relative and slope maps of the options that have them |
| `memories/hippocampus/` | the associative store's ports, strengths, masses, consolidated records and separator |
| `memories/episodic/` | the episodic choice (`memory_choice`): the context store (`context_memory/`), the stream's `context` and `decision`, the replay `path/`, with the hold the context and code of the last decision (`before`, `seen`), and with extinction the direct payments (`paid_memory/`, sharing the place records' separator) |

`Brain.load` also reads the earlier formats `cadence-generic/1` to `/10`; a retired setting
loads at the one value this version keeps for it and is refused by name at any other
([CHANGELOG](../CHANGELOG.md)).

Loading refuses a file that does not match the brain it describes: record and recall key
widths that differ from the code, code tables that do not read every ON and OFF channel twice
with positive weights, a code whose metadata lacks the `cover` field, a recall window of
another length, and learned-code tables that disagree with the saved synapses and biases of
the code region. A pending sampled action needs its saved observation, action and, with a
learned code, its decision-time code. The learner's output order must match the motor
population, so action indices keep their meaning after loading. A saved maturity holds exactly
`scale`, `recovery` and `use`, each a finite real in its range (`use` at most `scale * 1e76`,
`Maturity.LIMIT`), with the store's noise scales as nonnegative finite numbers under a boolean
`surprise_gate`; anything else is refused with `ValueError` naming the field.

## Arousal (`cadence.arousal`)

```text
ArousalConfig(threshold=0.2, decay=0.9, tolerance=2.0, floor=0.1, fast=0.05,
              slow=0.005, heat=2.0, youth=100, value_surprise=1.0, need=0.0,
              need_scale=0.1)
Arousal(config=None)
```

`ArousalConfig` holds the genes of the law; `to_dict()` returns them and
`ArousalConfig.space()` declares their mutation space for [`genes`](anatomy.md#any-genome),
without `need_scale`, which a space of your own can add
(`{**ArousalConfig.space(), "need_scale": ("linear", 0.02, 0.0, 1.0)}`). The composed brain's
need scale is 0.1, a constant (`arousal_need_scale` is refused by name, in a mapping too); an
`ArousalConfig` given as `arousal=` keeps it as given. Construction
refuses negative values, `decay` outside `[0, 1)`, `fast` or `slow` outside `(0, 1]`,
`fast < slow` and a `youth` that is not a nonnegative integer. The law and the
meaning of each gene are in [routine and repair](start.md#routine-and-repair-live).

`Arousal` is one stream's state; `Brain.live` runs it, and it can be used alone.

| Member | Behaviour |
| --- | --- |
| `outcome(error, reward, *, own=True, learned=True) -> (surprise, want)` | Applies one outcome. Only `own` outcomes can surprise and enter `usual`; every outcome moves `recent` and `longrun`; `learned` outcomes, and any in youth or that leave the stream aroused, enter `scale`. An outcome whose statistics cannot stay finite raises `ValueError` and changes nothing. |
| `lived(sweeps, learning_sweeps=0)` | Counts one moment in `moments`, `sweeps` (per mode), `learning_sweeps` and `age`. |
| `reset()` | Begins another stream calm; clears the averages and their counts, the reward scale among them (so a scaled need is zero until outcomes enter it again), keeps `age` and the work counts. `Brain.reset()` leaves the state as it is; `Brain.reset(keep=())` calls this. |
| `to_dict()`, `Arousal.from_dict(values)` | The complete state with every gene, marked `cadence-arousal/4`. |
| `level`, `age`, `moments`, `sweeps`, `learning_sweeps`, `outcomes`, `rewards`, `spreads`, `config` | State and counters: `outcomes` counts own outcomes, `rewards` all outcomes. |
| `aroused`, `mode`, `want`, `heat`, `usual`, `recent`, `longrun`, `scale` | Read-only: `aroused` while `level >= threshold` or within `youth`; `heat` is `1 + config.heat * want`. |

`from_dict` also reads the states saved by earlier releases; a retired gene loads at the one
value this version keeps for it and is refused by name at any other
([CHANGELOG](../CHANGELOG.md)).

## The memories (`cadence.memories`)

```text
Memories(members)
Moment(observations, code, place, stores=True)
EpisodicRecords(place, temporal, *, width, offsets, share, paid=None, extinction=0.0, hold=None)
```

`brain.memories` is a `Memories`: a read-only mapping from a memory's name to the memory, the
present ones only, in the order `working_memory`, `efference`, `history`, `recall`,
`hippocampus`, `episodic`. Every member keeps the contract `Memory`:

| Member | Behaviour |
| --- | --- |
| `stimulus(drive, moment) -> drive` | the drive of a moment before the settle with what the memory adds: the trace or history read, the recall memory's address of `moment.code`, the store's read under `moment.place` (of the drive without one; nothing when `moment.stores` is false, as under an episodic vote) |
| `branch()` | a private copy for an imagined branch, which reads the live state and writes none |
| `reset(batch, rows=None)`, `keep(rows)` | fresh streams, all or `rows`; the state of `rows` kept. `Brain.reset` begins every stream afresh so, the stores keeping their records and the recall memory its slots with `keep=("slots",)` (`Recall.end`) |
| `to_dict()`, `arrays()` | the settings and counts, and the arrays a checkpoint keeps under `memories/<name>/` |

`Memories.stimulus(drive, moment)` applies every member in order.

With `memory_choice=True` the member `episodic` is an `EpisodicRecords`. `place` is the
associative store (`brain.hippocampus`) and `temporal` the context records
(`brain.context_memory`, None at `memory_context=0`); `share` and `extinction` are
`memory_context` and `memory_extinction`, and `steps` and `demonstrations` the constants
`REPLAY` (8) and `DEMONSTRATIONS` (1.0); `paid`, the store of direct payments that
`memory_extinction` reads, is
None without it; `hold` is `memory_context_hold`, and with it the streams also keep
`before` (the context before the last decision) and `seen` (that decision's observation
code). Per stream it holds
`context` (the temporal context after the last decision), `decision` (that decision's key)
and `paths` (the recent decisions, one per situation left). `vote(code, key)` is the mean of
the place read and the context read that joins the motor readout; `key(code)` is the key of a
decision taken now, in the context `held_context(code)` gives (the hold applied), and
`issue(key, action, code)` the bookkeeping of a decision the stream or an imagined branch takes; `outcome(...)` records a known outcome and replays a reward along the
path; `remember(...)` writes both kinds of record; `reset(batch, rows)` begins the next
attempt of `rows` with no context and no path. Its `stimulus` returns the drive unchanged:
the vote joins after the settle.

## Streams (`cadence.stream`)

The per-stream memories of the composed brain, usable alone on any graph. The composed
brain's memories are explained in [memory](memory.md); the standalone stores in
[standalone memory stores](memory.md#standalone-memory-stores).

```text
Trace(connectome, decay=0.5, amplitude=1.0, source="hidden", target="context")
Efference(connectome, decay=0.5, amplitude=1.0, source="motor", target="efference")
```

Methods: `reset(batch, rows=None)`, `keep(rows)`, `stimulate(drive)`, `update(state)`,
`ringing(floor=0.1)`, `to_dict()`; `Efference.issue(command)` (its `update` raises).
Behaviour: [traces](memory.md#traces); in the brain: [working trace](memory.md#working-trace),
[efference copy](memory.md#efference-copy).

```text
History(connectome, decays, amplitude=3.0, source="sensory", target="history")
```

One block of `source` width per decay in `target`; `hear(observations)` moves block `d` by
`h <- d h + (1 - d) x`, and `stimulate(drive)` writes `amplitude * h` into the `target`
columns of a copy (another batch size reads silence). `reset(batch, rows=None)`, `keep(rows)`, `trace` `(batch, blocks * source)`,
`to_dict()`. In the brain: [heard history](memory.md#heard-history).

<a id="recall-memory-genes"></a>

```text
Recall(connectome, amplitude=3.0, temperature=0.25, recency=0.25,
       capacity=64, next=True, address=0.01,
       null=None, address_reward=0.0, position_reward=0.0,
       window=1, heads=1,
       source="sensory", target="recall", inputs=0, writes=0, position=32, position_rate=0.1,
       boundary=False, copy=0.0, tokens=None, action=False, compete=True, anchor=False,
       relative=False, slope=0.0, when=False, actions=0)
```

The held-input index memory of each stream's recent moments. Each constructor argument is the
brain's `recall_<name>` gene; their meanings, founders and retune rules are the
[recall settings table](memory.md#recall-settings), the mechanism is
[how a moment is addressed](memory.md#how-a-moment-is-addressed). `inputs` is the key width
when it differs from the `source` population (a coded brain passes the code width), `actions`
the motor width a slot's command covers with `action` (a brain passes its motor neurons). A
retired setting is refused by name with a sentence saying what the memory does instead
([CHANGELOG](../CHANGELOG.md)).

| Member | Behaviour |
| --- | --- |
| `stimulate(drive, observations)` | Addresses the moment, fixes the softmax over the slots (`attended_heads`) and writes the read into `recall` and, with `next`, `next`; with `heads` above one each head reads with its own address into its own port. |
| `witness(observations, state, command=None)` | Writes the moment as a slot after its settle; with `action` the `(batch, actions)` command executed at it joins the slot's key and value (zero when None). |
| `copy_synapses()` | The copy pathway's synapses into the motor region, with the value coordinate and the motor neuron of each. |
| `learn(graph, learned) -> dict` | The address lesson after a teaching step; returns `address_rows`, `address_error`, `address_refused`, and with a position term `position_rows` and `position_refused` (zero when nothing moved). A nonfinite step is refused per row (per head for the position maps). With several heads each head steps its own maps: rows and refusals are summed over the heads, the error averaged over the heads that moved. The position term learns whatever `address` is. |
| `contrast_rows(graph, plus, minus)`, `propose(step)`, `accept(proposal)` | Reward training of the address map at `address_reward > 0` and of the position maps (and the anchored ones) at `position_reward > 0`, on the rows of the event heard; `ActorCritic` calls them. `contrast_rows` is `None` when neither trains the read; `reward_columns` is its width. |
| `end(batch, rows=None)` | Ends episodes and keeps the slots, as `Brain.reset(keep=("slots",))` does; the latest slot learns no successor. With `boundary` an end of every stream opens one boundary slot. |
| `weights(address, position=None)`, `position_term(head, event, cortex, rows=None)`, `ages()`, `buckets()`, `readback(graph, region, movement, head=0)`, `activity(state)` | The read's slot shares (with a head's position term from `position_term`, the shares the read used; without it the content's and recency's alone), one head's position term at every slot, slot ages and their position buckets, and the association movement read back through one head's recall synapses. |
| `reset(batch, rows=None)`, `keep(rows)`, `snapshot()`, `restore(snapshot)`, `reading()`, `hold(reading)`, `branch()` | Stream bookkeeping; `branch()` is a copy that reads the live slots and writes none. |
| `Recall.founders()`, `Recall.options()`, `Recall.setting(name, value)`, `Recall.null_logit(null, temperature)`, `Recall.removed(name)`, `Recall.ports(heads, next=True)`, `Recall.action_ports(heads, next=True)`, `Recall.position_count(position, capacity)`, `Recall.map_names(kind, head)` | The always-described genes, the sixteen options saved only where they differ, validation of one gene, the null slot's logit, the sentence that refuses a retired setting, the regions of each head's port and of its command, the number of age buckets, and the checkpoint names of one head's three maps of a kind (`"address"`, `"position"`, `"anchor"`, `"relative"`, `"slope"`; head 0's bare, head h's with `_h`). |
| `told`, `focus`, `anchor_maps`, `relative_maps`, `slope_maps`, `anchored_buckets(rows=None)`, `relative_buckets(rows=None)`, `when_regions(heads)`, `when_features`, `when_width`, `RELATIVE`, `SLOPE` | The options' state: each stream's moments told in its episode and the age of the slot its previous read attended most (`-1` before a read); one `(E_x, E_c, e)`, `(Q_x, Q_c, q)`, `(S_x, S_c, s)` map per head for the anchor, the relative read and the slope (empty lists without the option); each slot's anchored and relative bucket per stream; the when regions' names, the features a read delivers per slot (two of its age, with the anchor two more of its place) and their count; the relative offsets kept apart (8) and the slope's age scale (8). |
| `attended_heads`, `head_map(head)`, `head_maps`, `position_map(head)`, `position_maps`, `position_buckets` | Every head's attention of the latest read, head 0's first (None before a read, when the read found no slots, and once the streams changed); one head's address map `(w_x, w_c, b)` and every head's; one head's position maps `(P_x, P_c, p)` (refused at `position=0`), every head's (empty at `position=0`), and their number of age buckets. |
| `RUNTIME`, `CONSTRUCTION`, `OPTIONS`, `CONSTRUCTION_OPTIONS`, `SAVED`, `RETIRED`, `FORMAT` | Class tables: retunable genes, construction genes, the options and those fixed at construction, the keys a memory always describes and saves (the founders and the constants `heard` 1.0 and `address_temperature` 0.1; the third constant, `window_decay` 1.0, is not saved), the retired settings (each with the release that retired it, the one saved value a file loads at and what the memory does instead), and the keys an earlier checkpoint format wrote for three of them with the one value each loads at. |
| `keys`, `values`, `nexts`, `held`, `previous`, `ended`, `past`, `heads`, `address_width`, `neurons`, `streams`, `slots`, `to_dict()` | State arrays and sizes. |

Construction and `Brain.retune` refuse a `null` whose logit is not finite.

```text
FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False,
             replace=False, rule="hebb", separator=None, writes=0)
SynapticMemory(pre, post, decay=0.9, rate=1.0, amplitude=1.0, normalize=False,
               replace=False, rule="delta", separator=None, writes=0,
               consolidation=0.05)
PatternSeparator(inputs, expansion, winners, seed=0, center=0.0, fan_in=0, cover=0)
```

`FastSynapses` is one mutable `(pre, post)` matrix per stream: `observe(key, value,
write=None)`, `recall(key)`, `update(state, write=None, post=None)`, `read(drive)`,
`stimulate(drive, inplace=False, key=None)`, `reset(batch, rows=None)`, `keep(rows)`,
`key_width`, `to_dict()`. `rule="delta"` refuses `normalize`, `replace` and rates outside
`[0, 1]`. `SynapticMemory` (in `cadence.memory`) adds a persistent `consolidated` matrix
shared across streams: `observe(..., *, salience=None, value_mask=None)`, `reset` keeps the
consolidated synapses, `clear()` erases both, `salience_vector(salience, batch)` validates a
salience; it refuses any `rule` but `"delta"`, a `consolidation` outside `[0, 1]` and a
`separator` with nonzero `center`.
`PatternSeparator` codes keys by a fixed random projection keeping `winners` of `expansion`:
`code(key, learn=False)`, `habituate(keys)`, `arrays()`, `channels`, `to_dict()`; it raises
`ValueError` for nonpositive sizes, `winners > expansion`, a `fan_in` or `cover` without room,
and `center` outside `[0, 1)`. All three: [standalone memory stores](memory.md#standalone-memory-stores); the
associative store: [memory](memory.md#associative-store-the-hippocampus).

```text
cadence.stream.columns(index) -> slice | ndarray
```

`columns` returns a slice for a contiguous index.

## The learned code (`cadence.code`)

```text
LearnedCode(inputs, units, context, model, seed=0, fields=None)
```

The learned code region's tables and rules, built by `Brain.compose(..., learned_code=units)`
and held as `brain.learned_code`: `dictionary` `(inputs, units)` with unit-length columns,
`weights` `(units, context)` the predictor, `thresholds` `(units,)`, `read` (the gain the
sensory activation is read at, one over the activation one input at amplitude 1 gives, from
the model), the lateral `connectome`
and `graph` of the units (the sub-network a moment's code phase settles). `lateral()`,
`lateral_efficacy()`, `gram()`; `stimulus(sensory, context)`, `prediction(context)`,
`settle(stimulus, warm, *, budget, tolerance, damping=3) -> Equilibrium`,
`code_of(activation)` (the rectified code at unit length, what the memories key on and the
first module reads), `residuals(sensory, activation, context)`; `standing() -> Nudge | None` (the region's quadratic boundary toward zero at gain
`top_down` over its units); `stability(v) -> int` (the halvings of the step the settled
code's competition is stable under), `rest_stability() -> int` (the same over every state
the region can reach, from the dictionary alone) and `restep(halvings)`, `halvings`; `learn(sensory,
activation, context, free=None) -> bool` (one step of every table from a settled moment,
each row an event; `free` the context-free code, whose activity the thresholds' share reads
and whose residual the repair reads), `install()`,
`take(dictionary, weights, thresholds)`;
`arrays()`, `genes()`, `to_dict()`, `parameters()`, `nbytes()`, `work` (settles, rows, sweeps,
capped settles, updates, and `repairs`, the unit-rows the repair lowered), `last` (the latest phase's sensory activation, context, activation,
prediction and, at a learned moment, the context-free code `free`); the constants
`TOP_DOWN`, `RATE`, `PREDICTOR`, `THRESHOLD`, `THRESHOLD_RATE`, `TARGET`, `REPAIR`,
`SILENCE`, `ACTIVE` and `IDENTITY`, held as the instance's attributes of the same names in
lower case, `learning` (False holds every table still) and `top_down` (0 is the control
without the top-down boundary) the two a reading or a control changes on a copy.
Both step bounds include the supplied neuron model's synaptic `gain`; the standing
`top_down` anchor has its own gain. They bound negative modes only. A custom model
needs its largest slope-weighted coupling eigenvalue below one. A refused enclosing
action or outcome restores the previous code step; attempted-work diagnostics may advance.
`LearnedCode.founders()` and `LearnedCode.setting(name, value)` give and validate the one
setting, `fields`; `GENES`, `RUNTIME` and `CONSTRUCTION` name it, and `REMOVED` the
retired settings with what the region does instead. Guide:
[the learned code](senses.md#the-learned-code).

## Learning (`cadence.learning`)

```text
LearnerConfig(beta=0.1, eta=0.2, eta_bias=None, free_steps=100, nudged_steps=50,
              tolerance=1e-4, nudge="cross_entropy", temperature=0.2, normalize=0.0,
              normalize_floor=1e-3, momentum=0.0, scale_cap=8.0, qualified=False,
              damping=3, surprise=False, normalize_relative=0.0,
              nudged_fixed=False, rehearsal=0.0, rehearsal_capacity=512,
              surprise_share=False)
Learner(brain, outputs, config=LearnerConfig(), plastic_synapses=None,
        plastic_neurons=None, reciprocal=True, tie_groups=None,
        synapse_rate=None, slots=1, updates=0, contrast_updates=0)
```

Every field is explained in [every knob](settling.md#7-every-knob). `eta_bias=None` derives
`eta / 10`. Construction warns (`RuntimeWarning`) when `eta_bias` exceeds a positive `eta`,
when `normalize > 0` with a rate above 0.05
([rates under normalization](teach.md#rates-under-normalization)), and when
`qualified=True` with `nudged_steps < free_steps`
([qualified budgets](settling.md#qualified-teaching-budgets)). `beta` and `momentum` are
constants of a composed brain (0.1 and 0.9; `learning_beta` and `learning_momentum`
are refused by name); a `LearnerConfig` given to `Brain.compose` or
`retune` is the expert route and keeps them as given.
`rehearsal` above zero: a composed brain replays, after each lesson, that many times the lesson's
rows again (at least one row), each with its episode, drawn from a reservoir of
`rehearsal_capacity` rows, a uniform sample of every lesson row (`Brain.rehearsal_moments`,
`rehearsal_labels`, `rehearsal_seen`, `rehearsal_draws`); a learner alone replays nothing
([teach several types in sequence](teach.md#teach-several-types-in-sequence-rehearsal)).

| `Learner` member | Behaviour |
| --- | --- |
| `step(drive, labels, warm=None, weight=None, *, answer=None) -> (LearnedState, report)` | Free phase, nudged phase(s), one update. A refused qualified phase raises `LearningPhaseError` and changes nothing. |
| `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`, `targets(labels)`, `nudge_for(target, beta, weight=None)` | One phase; the per-neuron target; the `Nudge`. |
| `contrast(free, nudged, opposite=None)`, `contrast_rows(...)`, `update(free, nudged, opposite=None)`, `apply(delta_scale, delta_bias)` | The contrast, per row, and an update through masks, rates, tying and the cap. `update` does not check qualification. |
| `calibrate(drive, *, level=None, grid=None) -> float` | Picks the global gain by the free output operating point; reports in `last_calibration`; raises `RuntimeError` without a usable candidate and changes nothing ([calibration](settling.md#calibrating-the-operating-point)). |
| `predict(drive)`, `accuracy(drive, labels, batch=256)`, `parameters()`, `to_dict()` | Readout, score, plastic parameter count. |
| `save(path, *, compressed=True)`, `Learner.load(path, ...)` | [Checkpoints](#checkpoints-cadencecheckpoint). |
| `Learner.sharing(other, outputs, **options)`, `holder`, `brain` | A second learner on the same `GraphHolder` ([two learners](settling.md#two-learners-on-one-graph)). |
| `reverse`, `velocity`, `velocity_bias`, `second_moment`, `second_moment_bias`, `output_index`, `updates`, `contrast_updates` | Reverse-synapse index (-1 for none), optimizer state, counters. |

Construction refuses duplicate reciprocal pairs (`ValueError`; merge with
`Connectome.from_synapses` or pass `reciprocal=False`).

`GraphHolder(brain)` holds the graph that learners on it share. `LearningPhaseError(phase,
phases, hint=None)` is a `RuntimeError` carrying `phase`, `phases`, `report` and `hint`.
`cadence.learning.LearnedState(free, nudged, opposite=None, taught=None)` is what `step`
returns; under `surprise`, `taught` marks the taught rows.

| Function | Behaviour |
| --- | --- |
| `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, skip_init=None, excitatory_forward=False) -> Connectome` | Populations `input`, `hidden`, `output` ([every knob](settling.md#7-every-knob)). |
| `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Connectome, tie_groups)` | A one-hot token window through one shared embedding. |
| `learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0) -> NeuronModel` | Responsive starting settings for local learning. |
| `calibrate_bias(brain, drives, targets, *, per_neuron=False, rounds=3, span=(-6.0, 6.0), iterations=16, steps=100, tolerance=1e-4, qualified=False, damping=3, report=None) -> ndarray` | Candidate biases by coordinate bisection toward target mean activations; the graph is unchanged. `qualified=True` raises `RuntimeError` on a refused solve; nonfinite states always raise. |
| `naive_efficacy(connectome, plastic) -> ndarray` | Efficacies giving every plastic synapse class the same weight. |
| `preflight(brain, outputs, plastic, drives, *, level=0.5, saturation=0.02, shared_max=0.25, eligibility_min=10, steps=100, tolerance=1e-4) -> dict` | Pre-lesson checks: saturated readouts, shared sender codes, too few eligible synapses, indistinct drives; `warnings` names the repair ([check before the first lesson](reward.md#check-before-the-first-lesson)). |
| `seam_report(connectome, pre, post) -> dict` | `classes`, `synapses`, `coverage_pre`, `classes_per_post`, `median_count` of a plastic seam ([connectomes](anatomy.md#brains-from-a-connectome)). |

## Checkpoints (`cadence.checkpoint`)

```text
save(learner, path, *, compressed=True) -> Path
load(path, *, backend=None, device=None, config=None, precision=None) -> Learner
```

Also `Learner.save` and `Learner.load`. One `.npz` with the connectome, efficacies, gains,
biases, neuron model, configuration, outputs and slots, plasticity masks, ties, synapse rates,
optimizer state and update counts, replaced atomically. `config` replaces the saved
configuration; `backend`, `device` and `precision` may differ from the saved ones. Separate
`FastSynapses`, `Trace` and `ActorCritic` objects are not saved; use `Brain.save` for a
composed brain. The format is `cadence-checkpoint/2`, and `cadence-checkpoint/5` for a graph
with an apical compartment (which adds the compartment's synapses, pools, fraction and floor)
or a learner at the projection-relative floor or fixed nudged phases. `/1` loads; the formats
`/3` and `/4` of earlier releases are refused with a message naming the release
([CHANGELOG](../CHANGELOG.md)).

## The agent (`cadence.plasticity`)

```text
ActorCritic(learner, critic, config=None, seed=0)
ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=None, eta_critic=0.05,
                  dopamine_cap=1.0, eligibility_steps=None)
```

The reward learner of a stream: a `Learner` whose outputs are the actions, a learned linear
critic on the `critic` neurons, eligibility traces and a dopamine signal. Fields, the report
and the decision contract are in [learning from reward](reward.md#settings). `dopamine_cap`
and `eligibility_steps` are constants of a composed brain (1.0 and 12; `actor_dopamine_cap` and
`actor_eligibility_steps` are refused by name); an `ActorCriticConfig`
given as `reward=` is the expert route and keeps them as given.

| Member | Behaviour |
| --- | --- |
| `act(drive, greedy=False, temperature=None, *, bias=None) -> action` | One free phase and a draw per slot. `bias` adds a finite `(batch, motor)` vote before each softmax; eligibility scores the readout's own policy. A greedy act drops the pending action, so `learn` refuses until a sampled act; the decaying traces are kept. |
| `learn(reward, done, next_drive, bootstrap=None, *, observed=None, warm=None) -> dict` | The TD error made into dopamine and written through the eligibility. At least one row must be `observed`. `bootstrap` applies only to `done` rows: a truncated row passes `done=True` with the value of its last observation (`value_of`) as `bootstrap`; other `done` rows bootstrap from zero. |
| `fade(done=None)` | One moment without a new eligibility: every trace decays by `gamma * lam`. |
| `probabilities(state, temperature=None, *, bias=None)`, `value(state)`, `value_of(drive)`, `settle(drive)` | Policy, critic value, a cold settle. |
| `reset()`, `parameters()`, `to_dict()` | Clears cache, eligibility and salience; keeps learned parameters. |
| `state`, `salience`, `memory`, `trace`, `trace_memory` | The latest free phase; per-sender salience weights; a memory with learned parameters (`Recall`) and its eligibility. |

Construction warns when `eta_bias` exceeds a positive `eta`.

## NeuralGraph (`cadence`)

```text
NeuralGraph(connectome, neuron_model, *, backend="cpu", efficacy=None,
            log_gain=None, bias=None, device=None, dense_limit=2048,
            layout=None, precision=None, apical=None)
```

The runnable graph of graded neurons ([how the graph settles](settling.md)). `efficacy`
defaults to the connectome's signs, `log_gain` and `bias` to zero. Graphs whose dense blocks
fit in `dense_limit` squared entries use the [block transport](#blocks-cadenceblocks);
`layout` passes a precomputed cut. `precision` (torch only) is `"float32"` or `"float64"`
([backends](backends.md)). `apical` (an `Apical`) lands some synapses on an apical
compartment ([the apical compartment](anatomy.md#the-apical-compartment)); the mlx backend
refuses it. Parameter arrays are read-only; assign whole validated arrays to
`efficacy`, `bias` or `log_gain`, or use `with_parameters`. Nonfinite values are refused
before any change.

| Member | Behaviour |
| --- | --- |
| `settle(stimulus=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> BrainState` | A list of neurons at full amplitude, a `{neuron: level}` map or a dense drive vector. A positive `tolerance` stops once no activation moves more. |
| `settle_batch(drive, ...) -> BrainState` | The same for `(batch, n)` drives. `mask` is `(n,)`, `(1, n)` or `(batch, n)` in `[0, 1]`. |
| `equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None, nudge=None, damping=0) -> Equilibrium` | Settles until the equation residual is below `tolerance`, checked per chunk; up to `damping` halvings of the step share the budget. |
| `residual(drive, state, *, nudge=None, mask=None, on_device=True) -> ndarray` | Per-row fixed-point discrepancy; no state change. |
| `apical_gain(drive, state) -> ndarray or None` | The apical compartment's gain per row and pool at `state`, what a nudged settle from it holds; `None` without the compartment. |
| `stimulus_vector(stimulus)`, `stimulus_levels(levels)`, `readings(state, names, i=0, level=0.5)` | Drive construction ([pattern targets](settling.md#pattern-targets)); population readings. |
| `with_parameters(*, efficacy=None, log_gain=None, bias=None)`, `weights`, `dense()`, `to_dict()` | A new graph; effective drive per synapse; the `W[pre, post]` matrix. |
| `contrast_on_device(plus, minus)` | The learning contrast on the accelerator, or `None`. |
| `connectome`, `neuron_model`, `layout` | The topology, the model and the block cut in use. |

Nonfinite drives and warm states are refused. Convergence of a solve says nothing about
stability, uniqueness or task quality ([the certificate](settling.md#the-settling-certificate)).

| Type | Fields |
| --- | --- |
| `BrainState(v, activation, adaptation, steps, trajectory=None, activity_change=None, device=None, discarded_steps=0)` | `(n,)` or `(batch, n)` arrays; `row(i)`, `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`, `batched`. `attempted_steps = steps + discarded_steps` includes CUDA tolerance lookahead beyond the returned state. |
| `Equilibrium(state, residual, steps, tolerance, residual_checks=0, damping_halvings=0, stagnation_checks=0)` | Per-row `converged` and `qualified` (the residual within `tolerance`); `steps` counts local sweeps; `damping_halvings` is the final halving index. |
| `Apical(synapses, groups, fraction, floor=1.0)` | `synapses` a bool per synapse, `groups` an int64-representable pool per neuron (`-1` none); `drive(basal, raw)` the normalized apical drive, `members`, `check(synapses, neurons, post)`, `to_dict()`. Checkpoint loading applies the same strict types and rejects coerced or incomplete topology. |
| `Nudge(target, mask, beta, softmax_temperature=None, weight=None, groups=None, anchor=None, anchor_gain=None, apical_gain=None)` | Extra drive `beta * mask * (target - s)`, or against `softmax(s / T)` per group; `weight` scales rows. `anchor` and `anchor_gain` (supplied together, finite, gain nonnegative per neuron) add `anchor_gain * (anchor - s)`, unscaled by `beta`, mask or weight. `apical_gain` (`(batch, pools)`, nonnegative) is the apical compartment's gain the settle holds. `drive(s)`. |

`available_backends() -> dict` names the backends that import here, with the device and
default precision each would use, such as `{"cpu": "numpy float64", "torch": "mps float32"}`;
MLX answers from a child interpreter ([backends](backends.md#choose-a-backend)). `ep_structure` and the settling
certificate are in [Certificate](#certificate-cadencecertificate).

## Connectome (`cadence.connectome`)

```text
Connectome.from_synapses(n, *, pre, post, count=None, sign=None, populations=None,
                         label="connectome", min_count=0.0) -> Connectome
```

Parallel synapses merge (counts add), autapses and synapses below `min_count` drop, and the
arrays are sorted by `(post, pre)`. Fields `n`, `pre`, `post`, `count`, `sign`, `populations`
and `label` are read-only. `members(*names)`, `with_populations(**populations)`,
`in_degree()`, `out_degree()`, `synapses`, `summary()`, `digest()` (SHA-256 of the arrays).
Guide: [brains from a connectome](anatomy.md#brains-from-a-connectome).

## Neuron model (`cadence.neuron`)

```text
NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0,
            adaptation=None, leak=0.0)
Adaptation(tau_steps=50.0, strength=1.0)
```

The graded neuron: `activation(v)` is a rectified re-based sigmoid that publishes zero at
rest, `slope_at(v)` its derivative, `rest_emission` the raw value subtracted at rest;
`replace(**changes)`, `to_dict()`. `Adaptation` is a slow per-neuron variable
([activation and adaptation](settling.md#activation-and-adaptation)).

## Blocks (`cadence.blocks`)

```text
layout(connectome, *, max_pairs=256) -> Layout
BlockTransport(layout, flat, pairs=None)
block_contrast(layout, s_plus, s_minus) -> ndarray
```

`layout` cuts the neurons at the boundaries of contiguous populations and pairs the ranges
that carry synapses; with no contiguous populations or more than `max_pairs` blocks it is one
full block. `Layout` has `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`, `ranges`,
`pairs`, `size`, `bounds(k)`, `sources()` (ranges receiving no synapses),
`parallel_synapses` (distinct synapses sharing a matrix entry), `flat(weights)`,
`blocks(flat)`, `block_of_edge` (the block of each synapse) and `to_dict()`.
`BlockTransport.synaptic_input(s)` sums the blocks `pairs` names (every block by default) and
reuses the product of every source range that did not move since the previous call. `block_contrast` is the
learning contrast as one Gram product per block.

## Certificate (`cadence.certificate`)

```text
certificate(brain) -> Certificate
Certificate(row_mass, lipschitz, dt, adaptation)
row_mass(brain) -> float
lipschitz_constant(model) -> float
ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12) -> EPStructure
```

`Certificate` (frozen) has `rate` (`1 - dt * (1 - lipschitz * row_mass)`), `certified`
(`rate < 1` and no adaptation), `mass_limit`, `error_bound(movement)`,
`apriori_bound(first_movement, steps)`, `steps_for(change, tolerance)` (raises `ValueError`
when uncertified) and `to_dict()`; construction requires finite values, `row_mass >= 0`,
`lipschitz > 0` and `dt` in `(0, 1]`. `EPStructure` (frozen) reports `fixed_inputs`,
`free_neurons`, `free_asymmetry`, `fixed_incoming_mass`, `adaptation`, `tolerance` and
`compatible`, which covers structure only. `certificate` and `ep_structure` raise `ValueError`
for apical graphs; their plain-sum arguments do not cover state-dependent compartment gains.
Guide: [the settling certificate](settling.md#the-settling-certificate),
[equilibrium-propagation scope](settling.md#equilibrium-propagation-scope).

## Neuron-by-neuron reference (`cadence.reference`)

```text
cadence.reference.settle_neuron_by_neuron(connectome, neuron_model, stimulus, *, steps,
    log_gain=None, bias=None, efficacy=None) -> (trajectory, Ledger)
conformance(brain, stimulus, *, steps=60) -> dict
```

The reference settles one neuron at a time, each reading only its own state and synaptic
input, and counts one transmission per declared synapse per step. `conformance` runs a brain
against it on the same stimulus and returns `backend`, `steps`, `max_abs_deviation`, `ledger`
and `final_active` (neurons at or above 0.5).
`Ledger(declared_synapses, steps=0, transmissions=0, undeclared=0)` has `clean` and
`to_dict()` ([conformance](backends.md#residual-checks-masks-and-conformance)).

## Regions and genome (`cadence.regions`, `cadence.genome`)

```text
Region(name, size=0, circuit=None, inputs=None, outputs=None)
cadence.regions.visual_cortex(height, width, *, channels=1, features=8, field=3,
    stride=1, init=1.0, seed=0, name="visual")
cadence.regions.cortex(size, *, lateral=0.0, name="association")
cadence.regions.motor_cortex(actions, *, lateral=0.0, name="motor", slots=1)
cadence.regions.prefrontal_cortex(holds, *, name="prefrontal")
cadence.regions.slot_sizes(actions, slots) -> list[int]
```

A blank region has a `size`; a designed one a `circuit` whose `inputs` and `outputs`
populations receive and send projections. `designed`, `neurons(population=None)`,
`to_dict()`. `visual_cortex` has populations `input` (pixels, row by row) and `output`
(`features` maps of `field`-square windows). A nonzero `lateral` connects every pair of
neurons with that weight (negative inhibits): all of a `cortex`, fixing its size, or each
slot of a `motor_cortex`, whose population is `actions`. `slot_sizes` refuses slots that do
not divide or add up to `actions`. `prefrontal_cortex` has one neuron per neuron it holds. Guide:
[write a cortex](anatomy.md#write-a-cortex).

```text
Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)
Genome(regions, projections, label="genome")
Genome.composed(inputs, actions, **settings) -> Genome
Genome.observation_width(inputs, code=None, learned_code=None) -> int
develop(genome, seed=0) -> Connectome
cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=()) -> Genome
genes(space, *, rate=1.0) -> mutation
evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map,
       report=None, mutate=None, grow=None, **mutation) -> Lineage
```

A projection end is a region name or `region/population`; `sign` is the mean sign and
`reciprocal` adds the reverse synapses with the same weights. `Genome` checks unique names and
valid ends; `region(name)`, `to_dict()`, `Genome.from_dict(d, designed=None)`.
`Genome.composed` is the layout `Brain.compose(inputs, actions, **settings)` develops; the
settings that do not shape it change nothing. `Genome.observation_width` is the key width
the memories of such a brain will use, the size of a custom anatomy's `recall` and `next`
regions. `develop` is
deterministic in the seed. `evolve` scores `fitness(grown, seed)` per life, requires finite
scores and returns `cadence.genome.Lineage` (`generations`, `best`, `best_fitness`). Guide:
[evolve a brain](anatomy.md#evolve-a-brain).

## Receipts (`cadence.receipts`)

```text
Receipt.build(kind, body, sources=()) -> Receipt
Receipt.read(path) -> Receipt
Receipt.verify(path, *, sources=None, check=None) -> (ok, message)
canonical_json(value) -> str
cadence.receipts.canonical_sha256(value) -> str
cadence.receipts.source_manifest(files) -> dict
```

`Receipt(kind, body, source, digest)` with `write(path)` and `to_dict()`. Guide:
[receipts](evaluate.md#receipts).

## Record every settling step

```text
record_settlements(callback, *, label="")
```

A context manager: every settle inside it, including those inside `Learner`, `ActorCritic`
and imagination, calls `callback` with a `SettlementRecord(label, connectome, neuron_model,
drive, mask, weights, bias, nudge, potential, activation, adaptation, steps)`. Histories have
shape `(steps + 1, batch, neurons)`: the initial state, then one row per actual iteration, so
successive differences of `potential` are the signed local repairs.

```python
import numpy as np
import cadence as cd

brain = cd.NeuralGraph(cd.layered(4, 16, 2, seed=0), cd.learning_neuron_model())
records = []
with cd.record_settlements(records.append, label="observe and act"):
    result = brain.settle(stimulus={0: 1.0}, steps=32)
repairs = np.diff(records[0].potential, axis=0)
assert records[0].potential.shape == (33, 1, brain.connectome.n)
```

Recording copies every neuron's state after every step; write bounded chunks from the
callback for long runs. CPU recording uses the NumPy kernel instead of the fused path, so
round-off and wall time can differ. A callback that is not callable raises `TypeError`; it
runs synchronously with recording suspended. Nested contexts restore the previous callback on
exit, callback failures propagate, and the recorded arrays own their storage. Treat the shared
connectome as read-only. For a display see
[the brain in a browser page](tune.md#the-brain-in-a-browser-page).
