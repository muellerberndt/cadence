# Compose a brain

Start with `Brain.compose`. It builds **System 1**: one continuing neural
graph with sensory input, reciprocal processing regions, motor choices, a working
trace and fast/persistent associative memory. Optional observer regions add
**System 2** feedback within that graph.

Keep this brain through acquisition, ordinary use and correction. The
[world-model guide](world-model.md) explains why its learned relations and memory
support changing equilibria, and distinguishes that design from the mechanisms
implemented today. The calibration and isolated response examples below are
controls for those mechanisms.

## Brain

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.live(observation)
assert action.shape == (1,)
```

`live` follows one stream: inputs have shape `(1, inputs)` and the output is one
action index. The default has one 64-neuron processing region. For more regions,
`modules=(16, 8)` creates two; the last is the association region.
Sensory input reaches the first, the association
region exchanges signals with motor neurons, and its working trace enters through
context neurons. `SynapticMemory` records the actually chosen action's reward;
its fast and persistent associations influence later choices.

`Brain.compose(..., efference_amplitude=1.0)` adds the efference copy, the
corollary discharge of the issued command. The working trace is written from
the settled state before the decision, so under identical observations it
carries which action was executed only through the margin the motor
competition left, and nothing of a sampled choice. The copy is one
`efference` neuron per motor neuron, driven by the fading one-hot of the action
each stream issued (`efference_decay`, default 0.2) and read by the association
region through a plastic projection of the working trace's scale. The write is
fixed, like the working trace's; what to do after what it did is learned. The
founder value `efference_amplitude=0.0` builds the released composition,
byte-identical; the copy is a gene selected against that control. The
[steady-rhythm chamber](../benchmarks/rhythm/README.md) measures it on the
task the working trace alone did not carry: alternate two actions under
identical drive.

The base can already be deep. Add recursive readback separately:

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8, 4), seed=7,
    arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Each observer exchanges activity with the base, motor regions and earlier
observers. All regions participate in the same settlement. This provides the
connections for recursive correction; it does not automatically learn useful
reflection. These observers read neural state. The advanced
[population solver](equilibrium/index.md) separately implements exact state-and-error
readback under its own equations.

## Defaults and expert overrides

The composed founders are controls, not values proven optimal for every body.
Version 0.80.0 keeps their numerical values. `arousal=True` is shorthand for
`ArousalConfig()`; omit `arousal` for the existing composition without a `live`
controller. No setting is silently inferred from a demo, input count or reward.

Set only the values you intend to change:

```python
tuned = Brain.compose(
    inputs=4, actions=2, arousal=True,
    sensory_scale=2.0,
    actor_eta=0.1, actor_eta_bias=0.01,
    arousal_youth=30,
)
settings = tuned.describe()
assert settings["genes"]["actor_eta"] == 0.1
assert settings["initialization"]["composition"]["sensory_scale"] == 2.0
```

This is an override example, not a performance recommendation. `sensory_scale`
sets the initial sensory-to-first-module projection, so custom sensory strength
no longer requires rebuilding the composed genome. The projection stays plastic.
`describe()` returns JSON-safe `layout`, effective flat `genes`, resolved
`learning`/`actor`/`arousal` configurations and `initialization` provenance.
The initialization records choices such as sensory scale, not the current
plastic weights. Save/load carries those settings alongside acquired state;
older files without composition metadata report that provenance as unknown.
Constructor settings for an absent component have no runtime effect: for
example, `memory_rate` with `episodic=False`, or `efference_decay` with no
efference population. `describe()["genes"]` omits those inactive settings;
`retune` refuses them instead of adding a component to an acquired brain.

Each field of `LearnerConfig`, `ActorCriticConfig` and `ArousalConfig` is reachable
as `learning_<field>`, `actor_<field>` and `arousal_<field>`. `temperature` is a
short alias for `learning_temperature`; it controls both teaching and the
actor's base policy. Unknown names and ambiguous bare rates such as `eta` are
rejected. This is the composed defaults table; the separate config constructors
retain their own defaults, listed in the [API reference](api.md).

| Named setting | Composed founder | Used by |
| --- | --- | --- |
| `sensory_scale` | `1.0` | Initial sensory projection; construction only |
| `learning_eta`, `learning_eta_bias` | `0.5`, `0.05` | Teaching with `step(teacher=...)`, `fit` or the learner; `live` does not read these rates |
| `learning_homeostasis_rate`, `learning_homeostasis_target` | `0.0`, `0.3` | The readout's intrinsic plasticity at every teaching or reward update, including aroused `live`; the founder rate leaves the readout as composed |
| `actor_eta`, `actor_eta_bias` | `1.0`, `0.05` | Reward plasticity in `step`/`learn`, and aroused `live` |
| `actor_eta_critic` | `0.3` | Value updates from those same learned outcomes |
| `temperature` | `0.2` | Teaching softmax and sampled action policy; arousal can add exploration heat |
| `learning_beta` | `0.1` | Nudge scale for teaching and reward eligibility |
| `learning_centered` | `True` | Teaching contrasts; reward eligibility always uses both nudges |
| `learning_free_steps`, `learning_tolerance` | `1024`, `0.003` | Free answers and teaching; free answers require equation qualification |
| `learning_nudged_steps` | `12` | Teaching budget; reward eligibility uses its own budget below |
| `learning_qualified`, `learning_damping` | `False`, `3` | Opt-in qualified teaching and its numerical fallback |
| `learning_nudge` | `"cross_entropy"` | Teaching target law; the actor uses its categorical policy law |
| `learning_normalize`, `learning_normalize_floor`, `learning_momentum` | `0.0`, `0.001`, `0.9` | Teaching optimizer; independent of actor normalization and momentum |
| `learning_decay`, `learning_scale_cap` | `0.0`, `8.0` | Parameter decay and efficacy cap in the shared learner update machinery |
| `actor_gamma`, `actor_lam` | `0.9`, `0.8` | Reward discount and eligibility decay; routine `live` also fades eligibility |
| `actor_eligibility_steps` | `12` | Each finite reward-nudged phase, independent of teaching qualification |
| `actor_normalize`, `actor_momentum` | `0.0`, `0.0` | Reward optimizer |
| `actor_dopamine_cap`, `actor_dopamine_center`, `actor_dopamine_floor` | `1.0`, `0.0`, `0.0` | Reward modulation |
| `actor_center_scale`, `actor_critic_normalize`, `actor_critic_signal` | `True`, `True`, `"auto"` | Dopamine units and critic update law |
| `working_memory_decay`, `working_memory_amplitude`, `working_memory_focus` | `0.2`, `3.0`, `0.0` | Continuing working trace |
| `efference_decay`, `efference_amplitude` | `0.2`, `0.0` | Issued-command trace; a positive amplitude at construction adds its population |
| `memory_decay`, `memory_rate`, `memory_amplitude`, `consolidation` | `0.9`, `1.0`, `1.0`, `0.05` | Associative retention, actual-outcome writes, recall and consolidation |
| `arousal_threshold`, `arousal_decay`, `arousal_tolerance`, `arousal_floor` | `0.2`, `0.9`, `2.0`, `0.1` | Routine/aroused decision in `live`, when arousal is enabled |
| `arousal_fast`, `arousal_slow` | `0.05`, `0.005` | Recent and long-run outcome statistics in `live` |
| `arousal_heat`, `arousal_youth` | `2.0`, `100` | Exploration and initial aroused moments in `live` |
| `arousal_value_surprise`, `arousal_record_surprise`, `arousal_need` | `1.0`, `0.0`, `0.0` | Critic/record surprise and unmet need in `live` |

Changing one rate preserves the existing bias rate. Pass both rates explicitly,
or use `actor_eta_bias=None` (likewise `learning_eta_bias=None`) to derive one
tenth of the newly resolved rate. For example,
`brain.retune(actor_eta=0.1, actor_eta_bias=None)` resolves the bias rate to
`0.01`. Enabling RMS normalization changes the effective scale of updates;
review [teaching rates](learning.md#rates-under-normalization) and
[actor rates](reward.md#rates-under-normalization) before choosing them.

Experts can still supply complete `learning=LearnerConfig(...)` and
`reward=ActorCriticConfig(...)` objects. Those replace whole configurations;
named overrides then change only selected fields. A fresh learner config has
`eta=0.2`, `eta_bias=0.02`, `momentum=0`, free/nudged budgets `100/50` and
tolerance `1e-4`; a fresh actor config has discount/trace `0.99/0.9`,
actor/bias/critic rates `0.5/0.05/0.05`, and inherits the learner's nudged budget.
Do not use a fresh config to express a one-field edit to a composed brain.

## Retune the same life

Use `retune` when measured task needs change, such as moving from practice to
ordinary interaction:

```python
tuned.retune(
    arousal={"need": 0.0, "heat": 0.0},
    temperature=0.2,
    actor_eta=0.03, actor_eta_bias=0.003,
)
```

`retune` accepts the same mutable named genes and partial arousal settings;
expert configuration objects retain their complete-replacement meaning.
It validates the whole request against the existing configuration domains before
installing anything. Evolutionary search spaces choose a sampling range within
those domains; they are not universal limits on expert settings. Retuning keeps
learned parameters, optimizer history, neural activity, traces, records, random state
and pending feedback. It does not run a moment or replay an outcome.
`reset_arousal=True` explicitly resets the arousal level and reward references,
retaining its age and work counters; use it only when that reset is part of the
intended stage.

A pending sampled action retains its actual eligibility and behavior temperature.
Rate changes apply when its next real outcome is learned. A change of
`learning_beta` is refused while that action awaits feedback, because the pending
contrast was formed at the earlier nudge scale; consume its outcome before
changing the scale. Construction choices such as `sensory_scale`, topology,
backend and `resting_bias` cannot be retuned, nor can a retune add a missing
memory, efference or arousal component. Create those components when composing.

## Choose settings for the application

The same interface serves different observation shapes, action groupings and
experience protocols. Preserve each application's demonstrated model and recipe
when simplifying its wrapper.

| Application | Interface and decisions |
| --- | --- |
| Robot arena | `compose` with vector senses and multiple motor `slots`, then `live` and `retune`; preserve real action/outcome custody through stage changes. |
| C64 musical control | The composed groove lane uses `step(teacher=...)` for witnessed demonstrations and explicit grouped motor choices where appropriate. Supplied timing and teacher labels remain application information; next-event learning alone does not establish a coherent phrase. |
| Atari | The [Atari demo](https://github.com/muellerberndt/cadence-demos/tree/main/atari-arcade) uses pixel observations, teaching and `step`; select teaching rates with `learning_*` and reward rates with `actor_*`. Its historical library pin remains part of its evidence. |
| Amen | The [Amen demo](https://github.com/muellerberndt/cadence-demos/tree/main/amen) uses the separate `RecordPatchNet` model and its retained-record recipe. Its rates and guarantees are those of [record patches](record-patch.md), not the composed actor. |

The [source-pinned arena report](https://github.com/muellerberndt/cadence-demos/blob/2a32c3e/robot-arena/docs/brain.md)
describes a 22-input body using sensory scale `4`, efference amplitude `0`,
nursery actor rates `0.03/0.003`, and later stage experiments with smaller rates.
Its earlier `0.003` ring experiment and later `0.001` choice are bounded
measurements, not generic recommendations. The report's historical comparison
to an efference founder of `3` is not the current library default: both 0.79.0
and 0.80.0 compose with `efference_amplitude=0` and actor `eta=1.0`.
The [odour nursery](continuous.md#routine-and-repair-live) uses another measured
operating point. Keep these recipes distinct and measure fresh acquisition,
retention, interference and recovery against their controls before changing a
founder. No arena setting establishes an optimal default for music, games or
another body.

## Operating point and motor competition

`Brain.compose(..., lateral=...)` configures motor competition. This is
the signed weight between each distinct pair of motor neurons; zero removes
those lateral connections while keeping reciprocal processing/motor feedback.
Left unset, the default is -0.5 up to 8 actions and 0.0 above. Each additional
action adds another inhibitory input per motor neuron: measured on composed
brains, -0.5 settles a small action menu in the same few dozen sweeps as 0.0,
while from about 12 actions the undamped free solve stops settling at all and
damped answers take about nine times the sweeps (issue 124). Inspect free
activity and its full residual on the actual task before selecting a
different value.

`Brain.compose(..., slots=...)` groups the motor neurons into several readouts that
settle together, one softmax each: a count of equal groups, or one size per group
covering `actions`. An action with several parts, such as kind, pitch class and
octave for a row of music, is one slot per part instead of one softmax over every
combination. `act` and `step` return one index per slot, lateral inhibition stays
within a slot, the unset `lateral` follows the largest slot, and `save`/`load` carry
the grouping (issue 142).

`Brain.compose(..., resting_bias=0.5)` initializes the modules, association region
and any observers with that bias. Sensory, working-memory, efference and motor biases start
at zero. The value must be a finite nonnegative real scalar; booleans and arrays
are rejected. The default remains zero. This is a selectable operating-point
candidate: it can reduce silence under some random drives, but positive bias
does not guarantee responsive activity, acquisition, retention or convergence.
Compare those outcomes against the zero-bias control before selecting it.

With the signed leaky neuron, an emission below rest can still respond to input
and contribute to local teaching. The fraction of nonpositive emissions is an
activity diagnostic, rather than a count of dead neurons. The functional
regression in `tests/test_operating_range_function.py` checks an independent
signed response derivative, qualified acquisition and old/new retention on one
declared continuing graph task, and complete saved next-update continuation
for both zero and 0.5 bias. This bounded evidence supports the existing option;
it does not select a better default or establish behavior on every task.

All these biases remain plastic. `brain.resting_bias` records the initialization
choice; `brain.brain.bias` holds the current learned values. Checkpoints preserve
both, and loading restores the learned vector without reapplying initialization.
For custom connectomes, named populations outside the `sensory`, `visual`,
`prefrontal`, `efference` and `motor` families are eligible. A neuron in any excluded family
keeps zero initial bias even if another population aliases it. Image builders
therefore leave the entire visual region at zero, including its processing cells.

Global gain changes synaptic drive throughout the graph; population bias
changes selected neurons' operating points. Integration steps and numerical
damping affect how the equations are solved. A faster qualified solve does not
establish that its motor state responds usefully to teaching. The
[calibration guide](learning.md#calibrating-the-operating-point) shows how to
check candidate biases before installing them. Choose gain, bias targets and
lateral wiring on development inputs, retain their hand-set controls, and
freeze them before confirmation.

## One experience step by hand

`step` combines learning from the previous outcome and choosing the next action.
Use `act` and `learn` separately when the body needs to manage that timing:

```python
body_brain = Brain.compose(4, 2, modules=(16, 8), seed=7)
chosen = body_brain.act(observation)

# Execute the choice in a tiny environment: action 0 earns one unit.
reward = (chosen == 0).astype(float)
following = np.array([[0.0, 1.0, 0.0, 0.0]])
body_brain.learn(reward, np.array([False]), following)
next_action = body_brain.act(following)
```

Reward and termination describe the preceding executed action. A demonstration
labels the current observation instead. Keep batch-row identities fixed until
`reset()`. [Continuous interaction](continuous.md) covers episodes, teaching,
private imagination and retries. The
[continuing brain example](../examples/continuing_brain.py) shows the complete
reward, teacher and checkpoint loop using the current composition API.
Independent [Records](memory.md#records) can
store declared observation/action/outcome fields; their reads and writes have
an explicit record rule rather than a neural-settlement certificate.

## Settle and check

`Brain.act`, `predict` and `accuracy` check the full potential/adaptation
equations before returning answers, including observer state. The defaults allow
1024 free steps at residual tolerance `3e-3`. A difficult free solve may use
half-step numerical damping within that same total budget; its final residual
is checked against the original model. Default teaching uses finite nudged phases,
where `tolerance` checks activity movement rather than the full equation residual.

The [defaults table](#defaults-and-expert-overrides) names the teaching and
reward budgets separately. Both optimizers leave RMS normalization disabled
unless explicitly selected. Numerical settings are task controls, not evidence
that an answer is useful or that its computation is cheap.

Other operations have independent defaults: `Brain.imagine` uses residual
tolerance `1e-6`, `NeuralGraph.equilibrate` uses `1e-5`, and `calibrate_bias`
uses `1e-4` (finite movement by default, full residual with `qualified=True`).
They do not inherit the action tolerance. Record the actual tolerance and phase
contract when comparing answers or work.

Set named overrides to preserve the composition's other defaults. Here qualified
teaching receives a longer nudged budget and checks the original full equations
in every phase:

```python
qualified = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), seed=7,
    learning_qualified=True, learning_nudged_steps=1024,
)
qualified.step(observation, teacher=np.array([0]))
assert qualified.last_learning["demonstration_qualified"] == 1.0
```

This requires every free and teaching phase to satisfy the original full
equations before a supervised update. The longer nudged budget is part of the
opt-in: keeping the composed `nudged_steps=12` while setting `qualified=True`
refuses every lesson on realistic input, so constructing a qualified
configuration whose nudged budget is below its free budget warns, and the
refusal names the mismatch
([qualified teaching budgets](learning.md#qualified-teaching-budgets)).
`LearningPhaseError` retains the attempted
phases and their cost report; a refusal preserves parameters and optimizer
history. `last_learning` records accepted and refused teaching diagnostics under
`demonstration_` keys, including attempted presentations and row work.
These qualified solves use configurable damping: a stalled attempt
can move to a smaller numerical step sooner, within the same total budget and
without relaxing the original equation check.
Qualified teaching does not change the reward eligibility contract: the
Brain default allows up to 12 finite nudged steps, configured separately through
`ActorCriticConfig.eligibility_steps`.

An exhausted `act` raises `RuntimeError` without changing live activity, memory,
randomness or pending feedback. If `step` learned an outcome before the next action
refused, keep the learning and retry `act`; do not send the same reward again.
A qualified state satisfies the equations to tolerance. Accuracy, uniqueness
and stability require their own evidence.

`predict` and `accuracy` start independent cold graph solves without reading the
working trace or associative memory. Use greedy `act` on a separately loaded
brain to measure the complete memory-aware response. Reset live state and the
trace for each independent query; clear fast associative residuals too when the
question is recall from consolidated associations alone. Keep those resets
separate from erasing durable parameters or records. [Memory](memory.md) describes
the three stores and their limits.

The lower-level `NeuralGraph.equilibrate` returns an `Equilibrium` with per-row
`residual` and `converged`, the state and total steps. `NeuralGraph.residual` checks an
existing state without settling. See [contracts](contracts.md) and
[certificates](certificate.md).

## Imagination and checkpoints

<a id="checkpoints"></a>

```python
phases = brain.imagine([observation, following], budget=1024, tolerance=1e-6)
assert phases
brain.save("brain.npz")
resumed = Brain.load("brain.npz")
```

Inspect each phase's `converged` flags before using it. Imagination stops at the
first refused phase and includes that phase in its result. It privately advances
a copied trace; it does not change live memory, parameters, randomness or pending
outcomes. It evaluates observations you supply. [Temporal planning](planning.md)
uses a learned environmental model to consider action consequences.

`Brain.save/load` includes neural parameters, critic, optimizers, random
state, traces, both associative-memory timescales and a pending action's feedback
state. Save the body separately and resume the same stream identities. Memory
shapes and numerical values are validated before use.

For advanced compositions, `Learner.save/load` saves the learner rather than an
entire body loop. Independently owned `Records` need their configuration, `tables`,
`mean`, `pathway_norm`, `seen` and `writes`; independently owned traces and critics
also belong to the caller's saved state. [API details](api.md) define each contract.

## Genome, development, brain

Use `Genome` for named regions, custom projections or evolved wiring:

```python
import cadence as cd
from cadence.regions import cortex, motor_cortex

genome = cd.Genome(
    regions=(cd.Region("senses", 5), cortex(16), motor_cortex(3)),
    projections=(
        cd.Projection("senses", "association", reciprocal=False),
        cd.Projection("association", "motor"),
    ),
)
connectome = cd.develop(genome, seed=0)
network = cd.NeuralGraph(connectome, cd.learning_neuron_model())
assert network.connectome.n == connectome.n
```

[Write a cortex](cortex.md) describes ports and projections;
[evolution](evolution.md) describes genomes and selection.
The lower-level `motor_cortex`, `cortex` and `layered` factories default to
`lateral=0.0`; the custom example above therefore has no motor competition.
Pass the intended value explicitly when comparing it with `Brain.compose`,
which left unset keeps `lateral=-0.5` up to 8 actions and drops to `0.0`
above ([operating point and motor competition](#operating-point-and-motor-competition)).
`Brain.build` offers the image/vector builder, and
`Brain(connectome, ...)` accepts the named populations required by its
interaction loop. These are advanced construction options for specific wiring
needs. A custom `NeuralGraph` alone does not install the complete Brain loop.

## The brain in a browser page

The archived [viewer](https://github.com/muellerberndt/cadence-examples/tree/main/viewer)
visualizes a connectome and recorded settlement in its declared environment.
Current `record_settlements` captures actual graph iterations for an
application-owned display. [cadence-demos](https://github.com/muellerberndt/cadence-demos)
contains current application examples; the library does not provide a browser
environment or body.
